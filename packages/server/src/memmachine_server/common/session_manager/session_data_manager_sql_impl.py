"""Manages database for session config and short term data."""

import os
from pathlib import Path
from typing import Annotated, Any

from pydantic import JsonValue
from sqlalchemy import (
    JSON,
    ForeignKeyConstraint,
    Integer,
    PrimaryKeyConstraint,
    String,
    and_,
    func,
    insert,
    select,
    text,
    update,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.ext.asyncio import AsyncEngine, async_sessionmaker
from sqlalchemy.orm import (
    DeclarativeBase,
    Mapped,
    mapped_column,
    relationship,
)
from sqlalchemy.sql.elements import ColumnElement

from memmachine_server.common.configuration.episodic_config import EpisodicMemoryConf
from memmachine_server.common.data_types import PropertyValue
from memmachine_server.common.errors import (
    SessionAlreadyExistsError,
    SessionNotFoundError,
)
from memmachine_server.common.metrics_factory import (
    MetricsFactory,
    OperationTracker,
    timed,
)
from memmachine_server.common.schema_migration import (
    Lineage,
    check_lineage,
    upgrade_lineage,
)
from memmachine_server.common.session_manager.session_data_manager import (
    SessionDataManager,
)
from memmachine_server.common.sql import write_session

# Base class for declarative class definitions


class Base(DeclarativeBase):  # pylint: disable=too-few-public-methods
    """Base class for declarative class definitions."""


JSON_AUTO = JSON().with_variant(JSONB, "postgresql")

IntColumn = Annotated[int, mapped_column(Integer)]
StringKeyColumn = Annotated[str, mapped_column(String, primary_key=True)]
StringColumn = Annotated[str, mapped_column(String)]
JSONColumn = Annotated[dict, mapped_column(JSON_AUTO)]


SESSION_LINEAGE = Lineage(
    name="session",
    versions=Path(__file__).parent / "migrations",
    metadata=Base.metadata,
)


class SessionDataManagerSQL(SessionDataManager):
    """Handle the session-related data persistency."""

    class SessionConfig(Base):  # pylint: disable=too-few-public-methods
        """ORM model for a session configuration (session_key is the primary key)."""

        __tablename__ = "sessions"
        session_key: Mapped[StringKeyColumn]
        timestamp: Mapped[IntColumn]
        configuration: Mapped[JSONColumn]
        param_data: Mapped[JSONColumn]
        description: Mapped[StringColumn]
        user_metadata: Mapped[JSONColumn]
        status: Mapped[str] = mapped_column(
            String, default="active", server_default=text("'active'")
        )
        __table_args__ = (PrimaryKeyConstraint("session_key"),)
        short_term_memory_data = relationship(
            "ShortTermMemoryData",
            cascade="all, delete-orphan",
        )

    class ShortTermMemoryData(Base):  # pylint: disable=too-few-public-methods
        """ORM model for short term memory data (session_key is the primary key)."""

        __tablename__ = "short_term_memory_data"
        session_key: Mapped[StringKeyColumn]
        summary: Mapped[StringColumn]
        last_seq: Mapped[IntColumn]
        episode_num: Mapped[IntColumn]
        timestamp: Mapped[IntColumn]
        __table_args__ = (
            PrimaryKeyConstraint("session_key"),
            ForeignKeyConstraint(["session_key"], ["sessions.session_key"]),
        )

    def __init__(
        self,
        engine: AsyncEngine,
        schema: str | None = None,
        metrics_factory: MetricsFactory | None = None,
    ) -> None:
        """Initialize with an async engine and optional schema."""
        self._engine = engine
        self._tracker = OperationTracker(
            metrics_factory,
            prefix="session_store_sqlalchemy",
        )
        self._async_session = async_sessionmaker(
            bind=self._engine,
            expire_on_commit=False,
        )
        if schema:
            for table in Base.metadata.tables.values():
                table.schema = schema

    async def migrate(self) -> None:
        """Create the session tables, or bring them up to date."""
        await upgrade_lineage(self._engine, SESSION_LINEAGE)

    async def startup(self) -> None:
        """Check that the session tables are migrated."""
        await check_lineage(self._engine, SESSION_LINEAGE)

    async def drop_tables(self) -> None:
        """Drop all tables from the database."""
        async with self._engine.begin() as conn:
            await conn.run_sync(Base.metadata.drop_all)

    async def close(self) -> None:
        """Close any underlying connections."""

    async def create_or_validate_session(
        self,
        session_key: str,
        configuration: dict[str, JsonValue],
        param: EpisodicMemoryConf,
        description: str,
        metadata: dict[str, JsonValue],
    ) -> None:
        """Create a session, or accept an existing one with matching data."""
        param_data = param.model_dump(mode="json")

        for _ in range(3):
            async with write_session(self._async_session) as dbsession:
                query = select(self.SessionConfig).where(
                    self.SessionConfig.session_key == session_key,
                )
                session = (await dbsession.execute(query)).scalars().first()
                if session is None:
                    values = {
                        "session_key": session_key,
                        "timestamp": int(os.times()[4]),
                        "configuration": configuration,
                        "param_data": param_data,
                        "description": description,
                        "user_metadata": metadata,
                    }
                    if self._engine.dialect.name == "sqlite":
                        statement = sqlite_insert(self.SessionConfig).values(**values)
                    else:
                        statement = pg_insert(self.SessionConfig).values(**values)
                    inserted_key = (
                        await dbsession.execute(
                            statement.on_conflict_do_nothing(
                                index_elements=["session_key"]
                            ).returning(self.SessionConfig.session_key)
                        )
                    ).scalar_one_or_none()
                    if inserted_key is not None:
                        return
                    session = (await dbsession.execute(query)).scalars().first()
                    if session is None:
                        # The conflicting row was deleted before our re-read.
                        continue

                if (
                    session.configuration == configuration
                    and session.param_data == param_data
                    and session.user_metadata == metadata
                ):
                    return
                raise SessionAlreadyExistsError(session_key)
        raise RuntimeError(f"Session '{session_key}' changed during creation")

    @timed("update_session_status")
    async def update_session_status(
        self,
        session_key: str,
        status: str,
    ) -> None:
        """Update the status of a session."""
        async with write_session(self._async_session) as dbsession:
            row = await dbsession.get(self.SessionConfig, session_key)
            if row is None:
                raise SessionNotFoundError(session_key)
            row.status = status

    async def get_sessions_by_status(
        self,
        status: str,
    ) -> list[str]:
        """Get a list of session keys by status."""
        async with self._async_session() as dbsession:
            stmt = select(self.SessionConfig.session_key).where(
                self.SessionConfig.status == status
            )
            sessions = await dbsession.execute(stmt)
            return list(sessions.scalars().all())

    async def delete_session(self, session_key: str) -> None:
        """Delete a session and its related data from the database."""
        async with write_session(self._async_session) as dbsession:
            # Query for an existing session with the same ID
            row = await dbsession.get(self.SessionConfig, session_key)
            if row is None:
                raise SessionNotFoundError(session_key)
            await dbsession.delete(row)

    @timed("get_session_info")
    async def get_session_info(
        self,
        session_key: str,
        status: SessionDataManager.SessionStatus | None = None,
    ) -> SessionDataManager.SessionInfo | None:
        """Retrieve a session's configuration, metadata, and params."""
        async with self._async_session() as dbsession:
            sessions = await dbsession.execute(
                select(self.SessionConfig).where(
                    self.SessionConfig.session_key == session_key,
                ),
            )
            session = sessions.scalars().first()
            if session is None:
                return None
            if status is not None and session.status != status:
                return None
            param = EpisodicMemoryConf(**session.param_data)

            return SessionDataManager.SessionInfo(
                configuration=session.configuration,
                description=session.description,
                user_metadata=session.user_metadata,
                episode_memory_conf=param,
                status=session.status,
            )

    def _json_contains(
        self,
        column: ColumnElement[object],
        filters: dict[str, PropertyValue | None],
    ) -> ColumnElement[Any]:
        if self._engine.dialect.name == "mysql":
            return func.json_contains(column, func.json_quote(func.json(filters)))

        if self._engine.dialect.name == "postgresql":
            return column.op("@>")(filters)

        if self._engine.dialect.name == "sqlite":
            # SQLite has no JSON_CONTAINS; emulate using json_extract
            if not isinstance(filters, dict):
                raise ValueError("SQLite emulation only supports dict values")
            conditions = [
                func.json_extract(column, f"$.{k}") == v for k, v in filters.items()
            ]
            return and_(*conditions)

        raise NotImplementedError(
            f"json_contains not supported for dialect '{self._engine.dialect.name}'",
        )

    @timed("get_sessions")
    async def get_sessions(
        self,
        filters: dict[str, PropertyValue | None] | None = None,
        status: SessionDataManager.SessionStatus | None = None,
    ) -> list[str]:
        """Retrieve session keys, optionally filtered by metadata and status."""
        stmt = select(self.SessionConfig.session_key)
        if filters is not None:
            stmt = stmt.where(
                self._json_contains(
                    self.SessionConfig.user_metadata.property.columns[0], filters
                ),
            )
        if status is not None:
            stmt = stmt.where(self.SessionConfig.status == status)
        async with self._async_session() as dbsession:
            sessions = await dbsession.execute(stmt)
            return list(sessions.scalars().all())

    @timed("save_short_term_memory")
    async def save_short_term_memory(
        self,
        session_key: str,
        summary: str,
        last_seq: int,
        episode_num: int,
    ) -> None:
        """Save or update short-term memory data for a session."""
        async with write_session(self._async_session) as dbsession:
            # Query for an existing session with the same ID
            sessions = await dbsession.execute(
                select(self.SessionConfig).where(
                    self.SessionConfig.session_key == session_key,
                ),
            )
            session = sessions.first()
            if session is None:
                raise SessionNotFoundError(session_key)
            short_term_datas = await dbsession.execute(
                select(self.ShortTermMemoryData).where(
                    self.ShortTermMemoryData.session_key == session_key,
                ),
            )
            short_term_data = short_term_datas.scalars().first()
            if short_term_data is not None:
                update_stmt = (
                    update(self.ShortTermMemoryData)
                    .where(self.ShortTermMemoryData.session_key == session_key)
                    .values(
                        summary=summary,
                        last_seq=last_seq,
                        episode_num=episode_num,
                        timestamp=int(os.times()[4]),
                    )
                )
                await dbsession.execute(update_stmt)
            else:
                insert_stmt = insert(self.ShortTermMemoryData).values(
                    session_key=session_key,
                    summary=summary,
                    last_seq=last_seq,
                    episode_num=episode_num,
                    timestamp=int(os.times()[4]),
                )
                await dbsession.execute(insert_stmt)

    @timed("get_short_term_memory")
    async def get_short_term_memory(self, session_key: str) -> tuple[str, int, int]:
        """Retrieve the short-term memory data for a session."""
        async with self._async_session() as dbsession:
            short_term_data = (
                (
                    await dbsession.execute(
                        select(self.ShortTermMemoryData).where(
                            self.ShortTermMemoryData.session_key == session_key,
                        ),
                    )
                )
                .scalars()
                .first()
            )
            if short_term_data is None:
                raise ValueError(
                    f"""session {session_key} does not have short term memory""",
                )
            return (
                short_term_data.summary,
                short_term_data.episode_num,
                short_term_data.last_seq,
            )

    async def update_session_episodic_config(
        self,
        session_key: str,
        enabled: bool | None = None,
        long_term_memory_enabled: bool | None = None,
        short_term_memory_enabled: bool | None = None,
    ) -> None:
        """Update episodic memory configuration flags for a session."""
        async with write_session(self._async_session) as dbsession:
            # Read under the row lock (on SQLite, under the write lock the
            # transaction holds), so a concurrent update of other flags waits
            # instead of being overwritten with the flags read here.
            sessions = await dbsession.execute(
                select(self.SessionConfig)
                .where(self.SessionConfig.session_key == session_key)
                .with_for_update(),
            )
            session = sessions.scalars().first()
            if session is None:
                raise SessionNotFoundError(session_key)

            # Get the current param_data and update it
            param_data = dict(session.param_data)

            if enabled is not None:
                param_data["enabled"] = enabled
            if long_term_memory_enabled is not None:
                param_data["long_term_memory_enabled"] = long_term_memory_enabled
            if short_term_memory_enabled is not None:
                param_data["short_term_memory_enabled"] = short_term_memory_enabled

            # Update the session
            update_stmt = (
                update(self.SessionConfig)
                .where(self.SessionConfig.session_key == session_key)
                .values(param_data=param_data)
            )
            await dbsession.execute(update_stmt)
