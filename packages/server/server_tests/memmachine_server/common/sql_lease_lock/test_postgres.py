"""PostgreSQL parity tests for the SQL lease lock service."""

import asyncio
from datetime import timedelta

import pytest
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine

from memmachine_server.common.sql_lease_lock import (
    LeaseLostError,
    SQLLeaseLockService,
)
from memmachine_server.common.sql_lease_lock._store import _metadata

pytestmark = pytest.mark.integration


@pytest.mark.asyncio
async def test_postgres_exclusive_expiry_and_fencing(
    sqlalchemy_pg_engine: AsyncEngine,
) -> None:
    second_engine = create_async_engine(sqlalchemy_pg_engine.url)
    first = SQLLeaseLockService(sqlalchemy_pg_engine)
    second = SQLLeaseLockService(second_engine)
    try:
        await first.startup()
        first_lease = await first.try_acquire(
            "pg-parity", lease_duration=timedelta(seconds=2)
        )
        assert first_lease is not None
        assert (
            await second.try_acquire("pg-parity", lease_duration=timedelta(seconds=2))
            is None
        )
        await first_lease.release()

        writer = await first.try_acquire(
            "pg-parity", lease_duration=timedelta(seconds=2)
        )
        assert writer is not None
        assert writer.fencing_token > first_lease.fencing_token
        assert (
            await second.try_acquire("pg-parity", lease_duration=timedelta(seconds=2))
            is None
        )
        await writer.release()

        short = await first.try_acquire(
            "pg-parity", lease_duration=timedelta(milliseconds=100)
        )
        assert short is not None
        await asyncio.sleep(0.2)
        replacement = await second.try_acquire(
            "pg-parity", lease_duration=timedelta(seconds=2)
        )
        assert replacement is not None
        assert replacement.fencing_token > short.fencing_token
        with pytest.raises(LeaseLostError):
            await short.renew()
    finally:
        async with sqlalchemy_pg_engine.begin() as conn:
            await conn.run_sync(_metadata.drop_all)
        await second_engine.dispose()


@pytest.mark.asyncio
async def test_postgres_simultaneous_writers_are_exclusive(
    sqlalchemy_pg_engine: AsyncEngine,
) -> None:
    second_engine = create_async_engine(sqlalchemy_pg_engine.url)
    first = SQLLeaseLockService(sqlalchemy_pg_engine)
    second = SQLLeaseLockService(second_engine)
    try:
        await first.startup()
        grants = await asyncio.gather(
            first.try_acquire("pg-writers", lease_duration=timedelta(seconds=2)),
            second.try_acquire("pg-writers", lease_duration=timedelta(seconds=2)),
        )
        assert sum(grant is not None for grant in grants) == 1
    finally:
        async with sqlalchemy_pg_engine.begin() as conn:
            await conn.run_sync(_metadata.drop_all)
        await second_engine.dispose()
