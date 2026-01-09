"""Manages database for session config and short term data."""

import json
import os
import pickle
from typing import Annotated, Any

from sqlalchemy import (
    JSON,
    ForeignKeyConstraint,
    Integer,
    PrimaryKeyConstraint,
    String,
    and_,
    func,
    insert,
    inspect,
    select,
    text,
    update,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.ext.asyncio import AsyncConnection, AsyncEngine, async_sessionmaker
from sqlalchemy.orm import (
    DeclarativeBase,
    Mapped,
    mapped_column,
    relationship,
)
from sqlalchemy.sql.elements import ColumnElement

from memmachine.common.configuration.episodic_config import EpisodicMemoryConf
from memmachine.common.errors import SessionAlreadyExistsError, SessionNotFoundError
from memmachine.common.session_manager.session_data_manager import SessionDataManager


# Base class for declarative class definitions
class Base(DeclarativeBase):  # pylint: disable=too-few-public-methods
    """Base class for declarative class definitions."""


JSON_AUTO = JSON().with_variant(JSONB, "postgresql")

IntColumn = Annotated[int, mapped_column(Integer)]
StringKeyColumn = Annotated[str, mapped_column(String, primary_key=True)]
StringColumn = Annotated[str, mapped_column(String)]
JSONColumn = Annotated[dict, mapped_column(JSON_AUTO)]


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

    def __init__(self, engine: AsyncEngine, schema: str | None = None) -> None:
        """Initialize with an async engine and optional schema."""
        self._engine = engine
        self._async_session = async_sessionmaker(
            bind=self._engine,
            expire_on_commit=False,
        )
        if schema:
            for table in Base.metadata.tables.values():
                table.schema = schema

    async def create_tables(self) -> None:
        """Create the necessary tables in the database."""

        def _check_migration_needed(conn: AsyncConnection) -> bool:
            inspector = inspect(conn)
            schema = self.SessionConfig.__table__.schema
            table_name = "sessions"
            table_names = inspector.get_table_names(schema=schema)
            if table_name not in table_names:
                return False
            columns = inspector.get_columns(table_name, schema=schema)
            for column in columns:
                if column["name"] == "param_data":
                    return "JSON" not in str(column["type"]).upper()
            return False

        async with self._engine.begin() as conn:
            if await conn.run_sync(_check_migration_needed):
                await self._migrate_pickle_to_json()
                return
            await conn.run_sync(Base.metadata.create_all)

    async def drop_tables(self) -> None:
        """Drop all tables from the database."""
        async with self._engine.begin() as conn:
            await conn.run_sync(Base.metadata.drop_all)

    async def close(self) -> None:
        """Close any underlying connections."""

    async def _migrate_pickle_to_json(self) -> None:
        """Migrate param_data from pickle to JSON."""
        schema = self.SessionConfig.__table__.schema
        table_name = f"{schema}.sessions" if schema else "sessions"
        json_type = "JSON"

        async with self._engine.begin() as conn:
            await conn.execute(
                text(
                    f"ALTER TABLE {table_name} RENAME COLUMN param_data TO param_data_blob"
                )
            )
            await conn.execute(
                text(f"ALTER TABLE {table_name} ADD COLUMN param_data {json_type}")
            )

            result = await conn.execute(
                text(f"SELECT session_key, param_data_blob FROM {table_name}")
            )

            for session_key, blob_data in result:
                if not blob_data:
                    continue
                try:
                    obj = pickle.loads(blob_data)
                    if hasattr(obj, "model_dump"):
                        json_data = obj.model_dump(mode="json")
                    elif hasattr(obj, "dict"):
                        json_data = obj.dict()
                    else:
                        json_data = obj.__dict__

                    val = json.dumps(json_data, default=str)

                    await conn.execute(
                        text(
                            f"UPDATE {table_name} SET param_data = :val WHERE session_key = :key"
                        ),
                        {"val": val, "key": session_key},
                    )
                except Exception as exc:
                    raise RuntimeError(
                        f"Error migrating session {session_key}"
                    ) from exc

            await conn.execute(
                text(f"ALTER TABLE {table_name} DROP COLUMN param_data_blob")
            )

    async def create_new_session(
        self,
        session_key: str,
        configuration: dict[str, object],
        param: EpisodicMemoryConf,
        description: str,
        metadata: dict[str, object],
    ) -> None:
        """Create a new session entry in the database."""
        if hasattr(param, "model_dump"):
            param_data = param.model_dump(mode="json")
        elif hasattr(param, "dict"):
            param_data = param.dict()
        else:
            param_data = param.__dict__

        async with self._async_session() as dbsession:
            # Query for an existing session with the same ID
            sessions = await dbsession.execute(
                select(self.SessionConfig).where(
                    self.SessionConfig.session_key == session_key,
                ),
            )
            session = sessions.first()
            if session is not None:
                raise SessionAlreadyExistsError(session_key)
            # create a new entry
            new_session = self.SessionConfig(
                session_key=session_key,
                timestamp=int(os.times()[4]),
                configuration=configuration,
                param_data=param_data,
                description=description,
                user_metadata=metadata,
            )
            dbsession.add(new_session)
            await dbsession.commit()

    async def delete_session(self, session_key: str) -> None:
        """Delete a session and its related data from the database."""
        async with self._async_session() as dbsession:
            # Query for an existing session with the same ID
            row = await dbsession.get(self.SessionConfig, session_key)
            if row is None:
                raise SessionNotFoundError(session_key)
            await dbsession.delete(row)
            await dbsession.commit()
            return

    async def get_session_info(
        self,
        session_key: str,
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
            param = EpisodicMemoryConf(**session.param_data)

            return SessionDataManager.SessionInfo(
                configuration=session.configuration,
                description=session.description,
                user_metadata=session.user_metadata,
                episode_memory_conf=param,
            )

    def _json_contains(
        self,
        column: ColumnElement[object],
        filters: dict[str, object],
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

    async def get_sessions(
        self,
        filters: dict[str, object] | None = None,
    ) -> list[str]:
        """Retrieve session keys, optionally filtered by metadata."""
        if filters is None:
            stmt = select(self.SessionConfig.session_key)
        else:
            stmt = select(self.SessionConfig.session_key).where(
                self._json_contains(
                    self.SessionConfig.user_metadata.property.columns[0], filters
                ),
            )
        async with self._async_session() as dbsession:
            sessions = await dbsession.execute(stmt)
            return list(sessions.scalars().all())

    async def save_short_term_memory(
        self,
        session_key: str,
        summary: str,
        last_seq: int,
        episode_num: int,
    ) -> None:
        """Save or update short-term memory data for a session."""
        async with self._async_session() as dbsession:
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
            await dbsession.commit()

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
