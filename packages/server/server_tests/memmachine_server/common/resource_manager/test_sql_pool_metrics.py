"""Tests for the connection pool metrics each SQL engine reports."""

import asyncio
from pathlib import Path
from uuid import uuid4

import pytest
from prometheus_client import REGISTRY
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine

from memmachine_server.common.configuration.database_conf import (
    DatabasesConf,
    SqlAlchemyConf,
)
from memmachine_server.common.resource_manager.database_manager import DatabaseManager


def _database_manager(conf: SqlAlchemyConf) -> tuple[DatabaseManager, str]:
    # A random pool name keeps each test's samples apart in the shared registry.
    pool_name = f"pool-{uuid4()}"
    manager = DatabaseManager(DatabasesConf(relational_db_confs={pool_name: conf}))
    return manager, pool_name


def _sqlite_file_conf(tmp_path: Path, *, pool_size: int) -> SqlAlchemyConf:
    return SqlAlchemyConf(
        dialect="sqlite",
        driver="aiosqlite",
        path=str(tmp_path / "pool.db"),
        pool_size=pool_size,
        max_overflow=0,
    )


def _sample(name: str, labels: dict[str, str]) -> float:
    value = REGISTRY.get_sample_value(name, labels)
    assert value is not None
    return value


async def _select_one(engine: AsyncEngine) -> None:
    async with engine.connect() as connection:
        await connection.execute(text("SELECT 1"))


@pytest.mark.asyncio
async def test_the_pool_reports_its_checked_out_connections(tmp_path):
    manager, pool_name = _database_manager(_sqlite_file_conf(tmp_path, pool_size=2))
    labels = {"pool_name": pool_name, "state": "used"}
    engine = await manager.async_get_sql_engine(pool_name)
    try:
        assert _sample("db_client_connection_count", labels) == 0
        async with engine.connect(), engine.connect():
            assert _sample("db_client_connection_count", labels) == 2
        assert _sample("db_client_connection_count", labels) == 0
    finally:
        await manager.close()


@pytest.mark.asyncio
async def test_a_checkout_reports_how_long_it_waited_for_a_connection(tmp_path):
    manager, pool_name = _database_manager(_sqlite_file_conf(tmp_path, pool_size=1))
    labels = {"pool_name": pool_name}
    engine = await manager.async_get_sql_engine(pool_name)
    try:
        held = await engine.connect()
        waiting = asyncio.create_task(_select_one(engine))
        await asyncio.sleep(0.2)
        assert not waiting.done()
        await held.close()
        await waiting

        assert _sample("db_client_connection_wait_time_seconds_count", labels) == 2
        assert _sample("db_client_connection_wait_time_seconds_sum", labels) >= 0.2
    finally:
        await manager.close()


@pytest.mark.asyncio
async def test_every_connection_to_an_in_memory_sqlite_database_shares_it():
    """The pool that times checkouts would give each connection its own database."""
    manager, pool_name = _database_manager(
        SqlAlchemyConf(dialect="sqlite", driver="aiosqlite", path=":memory:")
    )
    engine = await manager.async_get_sql_engine(pool_name)
    try:
        async with engine.connect() as first:
            await first.execute(text("CREATE TABLE shared (id INTEGER)"))
            async with engine.connect() as second:
                result = await second.execute(text("SELECT count(*) FROM shared"))
                assert result.scalar_one() == 0
    finally:
        await manager.close()
