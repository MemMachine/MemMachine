"""Tests for session data manager migration."""

import pickle

import pytest
from sqlalchemy import (
    JSON,
    Column,
    Integer,
    LargeBinary,
    MetaData,
    String,
    Table,
    insert,
    inspect,
)
from sqlalchemy.ext.asyncio import AsyncEngine

from memmachine.common.configuration.episodic_config import EpisodicMemoryConf
from memmachine.common.session_manager.session_data_manager_sql_impl import (
    SessionDataManagerSQL,
)


@pytest.mark.asyncio
async def test_migrate_pickle_to_json(sqlalchemy_engine: AsyncEngine) -> None:
    """Test that the database migrates from pickle to JSON correctly."""

    # Define the old schema (before migration)
    metadata = MetaData()
    sessions_table = Table(
        "sessions",
        metadata,
        Column("session_key", String, primary_key=True),
        Column("timestamp", Integer),
        Column("configuration", JSON),
        Column("param_data", LargeBinary),  # Old type: LargeBinary (BLOB)
        Column("description", String),
        Column("user_metadata", JSON),
    )

    # Create the table with the old schema
    async with sqlalchemy_engine.begin() as conn:
        await conn.run_sync(metadata.create_all)

    # Create a dummy EpisodicMemoryConf and pickle it
    # We assume EpisodicMemoryConf can be instantiated with defaults.
    session_key = "test_session_migration"
    param = EpisodicMemoryConf(session_key=session_key)
    pickled_data = pickle.dumps(param)

    # Insert a record with the pickled data
    async with sqlalchemy_engine.begin() as conn:
        await conn.execute(
            insert(sessions_table).values(
                session_key=session_key,
                timestamp=1234567890,
                configuration={"test": "config"},
                param_data=pickled_data,
                description="Test Session",
                user_metadata={"meta": "data"},
            )
        )

    # Initialize the SessionDataManagerSQL
    # This should trigger the migration in create_tables
    manager = SessionDataManagerSQL(sqlalchemy_engine)
    await manager.create_tables()

    # Verify the migration
    # 1. Check that we can retrieve the session info (which implies successful JSON deserialization)
    session_info = await manager.get_session_info(session_key)
    assert session_info is not None
    assert session_info.episode_memory_conf.session_key == session_key

    assert isinstance(session_info.episode_memory_conf, EpisodicMemoryConf)

    # 2. Verify the column type in the database is now JSON (or TEXT in SQLite)
    def check_column(conn):
        inspector = inspect(conn)
        columns = inspector.get_columns("sessions")
        for col in columns:
            if col["name"] == "param_data":
                assert "JSON" in str(col["type"]).upper()

    async with sqlalchemy_engine.connect() as conn:
        await conn.run_sync(check_column)

    await manager.drop_tables()
    await manager.close()
    await sqlalchemy_engine.dispose()
