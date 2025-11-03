"""Manages database for session config and short term data"""

import io
import os
import pickle
from typing import Annotated

from sqlalchemy import (
    JSON,
    ForeignKeyConstraint,
    Integer,
    LargeBinary,
    PrimaryKeyConstraint,
    String,
    and_,
    func,
    insert,
    select,
    update,
)
from sqlalchemy.ext.asyncio import AsyncEngine, async_sessionmaker
from sqlalchemy.orm import (
    DeclarativeBase,
    Mapped,
    mapped_column,
    relationship,
)

from memmachine.configuration.episodic_config import EpisodicMemoryParams

from .session_manager_interface import SessionDataManager


# Base class for declarative class definitions
class Base(DeclarativeBase):  # pylint: disable=too-few-public-methods
    """
    Base class for declarative class definitions.
    """


IntColumn = Annotated[int, mapped_column(Integer)]
StringKeyColumn = Annotated[str, mapped_column(String, primary_key=True)]
StringColumn = Annotated[str, mapped_column(String)]
JSONColumen = Annotated[dict, mapped_column(JSON)]
BinaryColumn = Annotated[bytes, mapped_column(LargeBinary)]



class SessionDataManagerImpl(SessionDataManager):
    """
    Handle's the session related data persistency.
    """

    class SessionConfig(Base):  # pylint: disable=too-few-public-methods
        """ORM model for a session configuration.
        session_key is the primary key
        """

        __tablename__ = "sessions"
        session_key: Mapped[StringKeyColumn]
        timestamp: Mapped[IntColumn]
        configuration: Mapped[JSONColumen]
        param_data: Mapped[BinaryColumn]
        description: Mapped[StringColumn]
        user_metadata: Mapped[JSONColumen]
        __table_args__ = (PrimaryKeyConstraint("session_key"),)
        shorttem_memory_data = relationship(
            "ShortTermMemoryData", cascade="all, delete-orphan"
        )

    class ShortTermMemoryData(Base):  # pylint: disable=too-few-public-methods
        """ORM model for short term memory data.
        session_key is the primary key
        """

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

    def __init__(self, engine: AsyncEngine, schema: str | None = None):
        """Initializes the SessionDataManagerImpl.

        Args:
            engine: The SQLAlchemy async engine to use for database connections.
            schema: The database schema to use for the tables.
        """
        self._engine = engine
        self._async_session = async_sessionmaker(bind=self._engine, expire_on_commit=False)
        if schema:
            for table in Base.metadata.tables.values():
                table.schema = schema

    async def create_tables(self) -> None:
        """Creates the necessary tables in the database.

        This method connects to the database and creates all tables defined in the
        Base metadata if they don't already exist.
        """
        async with self._engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

    async def close(self):
        """Closes the database connection engine."""
        if hasattr(self, "_engine"):
            await self._engine.dispose()

    async def create_new_session(
        self, session_key: str, configuration: dict, param: EpisodicMemoryParams, description: str, metadata: dict
    ):
        """
        Creates a new session entry in the database.

        Args:
            session_key: The unique identifier for the session.
            configuration: A dictionary containing the session's configuration.
            description: A brief description of the session.
            metadata: A dictionary for user-defined metadata.
            param: The episodic memory parameters.

        Raises:
            ValueError: If a session with the given session_key already exists.
        """

        buffer = io.BytesIO()
        pickle.dump(param, buffer)
        buffer.seek(0)
        param_data = buffer.getvalue()
        async with self._async_session() as dbsession:
            # Query for an existing session with the same ID
            sessions = (
                await dbsession.execute(select(self.SessionConfig)
                .where(self.SessionConfig.session_key == session_key))
            )
            session = sessions.first()
            if session is not None:
                raise ValueError(f"""Session {session_key} already exists""")
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

    async def delete_session(self, session_key: str):
        """Deletes a session and its related data from the database.

        Args:
            session_key: The unique identifier of the session to delete.
        """
        async with self._async_session() as dbsession:
            # Query for an existing session with the same ID
            row = await dbsession.get(self.SessionConfig, session_key)
            if row is None:
                raise ValueError(f"""Session {session_key} does not exists""")
            await dbsession.delete(row)
            await dbsession.commit()
            return


    async def get_session_info(self, session_key: str) -> tuple[dict, str, dict, EpisodicMemoryParams]:
        """Retrieves a session's data from the database.

        Args:
            session_key: The unique identifier of the session to retrieve.

        Returns:
            A tuple containing the configuration dictionary, description string,
            user metadata dictionary, and the EpisodicMemoryParams object.

        Raises:
            ValueError: If the session with the given session_key does not exist.
        """
        async with self._async_session() as dbsession:
            sessions = await dbsession.execute(select(self.SessionConfig).where(
                self.SessionConfig.session_key == session_key))
            session = sessions.scalars().first()
            if session is None:
                raise ValueError(f"""Session {session_key} does not exists""")
            binary_buffer = io.BytesIO(session.param_data)
            binary_buffer.seek(0)
            param: EpisodicMemoryParams = pickle.load(binary_buffer)
            return session.configuration, session.description, session.user_metadata, param

    def _json_contains(self, column, filter):
        if self._engine.dialect.name == "mysql":
            return func.json_contains(column, func.json_quote(func.json(filter)))

        elif self._engine.dialect.name == "postgresql":
            return column.op("@>")(filter)

        elif self._engine.dialect.name == "sqlite":
            # SQLite has no JSON_CONTAINS; emulate using json_extract
            if not isinstance(filter, dict):
                raise ValueError("SQLite emulation only supports dict values")
            conditions = [
                func.json_extract(column, f'$.{k}') == v
                for k, v in filter.items()
            ]
            return and_(*conditions)

        else:
            raise NotImplementedError(f"json_contains not supported for dialect '{self._engine.dialect.name}'")

    async def get_sessions(self, filter: dict[str, str] | None = None) -> list[str]:
        """Retrieves a list of all session keys from the database.

        Returns:
            A list of session keys.
        """
        if filter is None:
            stmt = select(self.SessionConfig.session_key)
        else:
            stmt = select(self.SessionConfig.session_key).where(self._json_contains(self.SessionConfig.user_metadata, filter))
        async with self._async_session() as dbsession:
            sessions = await dbsession.execute(stmt)
            return list(sessions.scalars().all())


    async def save_short_term_memory(self, session_key: str, summary: str, last_seq: int, episode_num: int):
        """Saves or updates the short-term memory data for a session.

        Args:
            session_key: The unique identifier for the session.
            summary: The summary of the short-term memory.
            last_seq: The last sequence number of the episodes in the short-term memory.
            episode_num: The number of episodes in the short-term memory.
        """
        async with self._async_session() as dbsession:
            # Query for an existing session with the same ID
            sessions = (
                await dbsession.execute(select(self.SessionConfig)
                .where(self.SessionConfig.session_key == session_key)
                )
            )
            session = sessions.first()
            if session is None:
                raise ValueError(f"""Session {session_key} does not exists""")
            short_term_datas = (
                await dbsession.execute(select(self.ShortTermMemoryData)
                .where(self.ShortTermMemoryData.session_key == session_key))
            )
            short_term_data = short_term_datas.scalars().first()
            if short_term_data is not None:
                update_stmt = update(self.ShortTermMemoryData).where(
                    self.ShortTermMemoryData.session_key == session_key
                ).values(
                    summary=summary,
                    last_seq=last_seq,
                    episode_num=episode_num,
                    timestamp=int(os.times()[4]),
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
        """Retrieves the short-term memory data for a session.

        Args:
            session_key: The unique identifier for the session.

        Returns:
            A tuple containing the summary string, the number of episodes, and the last sequence number.

        Raises:
            ValueError: If no short-term memory data exists for the given session_key.
        """
        async with self._async_session() as dbsession:
            short_term_data = (
                await dbsession.execute(select(self.ShortTermMemoryData)
                .where(self.ShortTermMemoryData.session_key == session_key))
            ).scalars().first()
            if short_term_data is None:
                raise ValueError(
                    f"""session {session_key} does not have short term memory"""
                )
            return short_term_data.summary, short_term_data.episode_num, short_term_data.last_seq
