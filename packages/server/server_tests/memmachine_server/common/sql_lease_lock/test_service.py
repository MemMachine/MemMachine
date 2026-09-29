"""Cross-instance behavior for SQL lease locks."""

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager, suppress
from datetime import timedelta
from pathlib import Path

import pytest
from sqlalchemy import text
from sqlalchemy.exc import OperationalError
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine

from memmachine_server.common.sql_lease_lock import (
    LeaseLostError,
    LockAcquireTimeout,
    SqlLeaseRWLockService,
)
from memmachine_server.common.sql_lease_lock._store import SqlLeaseStore


@asynccontextmanager
async def two_stores(path: Path) -> AsyncIterator[tuple[SqlLeaseStore, SqlLeaseStore]]:
    uri = f"sqlite+aiosqlite:///{path}"
    first_engine = create_async_engine(uri)
    second_engine = create_async_engine(uri)
    try:
        first = SqlLeaseStore(first_engine)
        second = SqlLeaseStore(second_engine)
        await first.startup()
        yield first, second
    finally:
        await first_engine.dispose()
        await second_engine.dispose()


@pytest.mark.asyncio
async def test_readers_share_a_key_and_block_writers(tmp_path: Path) -> None:
    async with two_stores(tmp_path / "locks.db") as (first, second):
        read_a = await first.try_acquire("resource", "read", 2_000)
        read_b = await second.try_acquire("resource", "read", 2_000)
        assert read_a is not None
        assert read_b is not None
        assert read_a.lease_id != read_b.lease_id
        assert read_b.fencing_token > read_a.fencing_token
        assert await first.try_acquire("resource", "write", 2_000) is None
        assert await first.release("resource", read_a.lease_id)
        assert await second.try_acquire("resource", "write", 2_000) is None
        assert await second.release("resource", read_b.lease_id)
        writer = await first.try_acquire("resource", "write", 2_000)
        assert writer is not None
        assert writer.fencing_token > read_b.fencing_token
        assert await second.try_acquire("resource", "read", 2_000) is None


@pytest.mark.asyncio
async def test_simultaneous_writers_cannot_both_acquire(tmp_path: Path) -> None:
    async with two_stores(tmp_path / "locks.db") as (first, second):
        grants = await asyncio.gather(
            first.try_acquire("resource", "write", 2_000),
            second.try_acquire("resource", "write", 2_000),
        )
        assert sum(grant is not None for grant in grants) == 1


@pytest.mark.asyncio
async def test_expired_lease_is_replaced_and_old_holder_loses_it(
    tmp_path: Path,
) -> None:
    async with two_stores(tmp_path / "locks.db") as (first, second):
        old = await first.try_acquire("resource", "write", 100)
        other_key = await second.try_acquire("other", "write", 2_000)
        assert old is not None
        assert other_key is not None
        await asyncio.sleep(0.2)
        replacement = await second.try_acquire("resource", "write", 2_000)
        assert replacement is not None
        assert replacement.fencing_token > old.fencing_token
        assert await first.renew("resource", old.lease_id, 2_000) is None
        assert not await first.release("resource", old.lease_id)


@pytest.mark.asyncio
async def test_database_error_is_not_reported_as_contention(tmp_path: Path) -> None:
    async with two_stores(tmp_path / "locks.db") as (first, _second):
        async with first._engine.begin() as conn:
            await conn.execute(text("DROP TABLE lease_lock_holder"))
        with pytest.raises(OperationalError):
            await first.try_acquire("resource", "read", 2_000)


@pytest.mark.asyncio
async def test_public_service_validates_key_and_duration(tmp_path: Path) -> None:
    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'locks.db'}")
    try:
        service = SqlLeaseRWLockService(engine)
        await service.startup()
        with pytest.raises(ValueError, match="resource key"):
            await service.try_acquire_read("", lease_duration=timedelta(seconds=1))
        with pytest.raises(ValueError, match="lease duration"):
            await service.try_acquire_write("resource", lease_duration=timedelta(0))
        with pytest.raises(ValueError, match="lease duration"):
            await service.try_acquire_read(
                "resource", lease_duration=timedelta(milliseconds=-1)
            )
        async with engine.connect() as conn:
            count = (
                await conn.execute(text("SELECT COUNT(*) FROM lease_lock_resource"))
            ).scalar_one()
        assert count == 0
    finally:
        await engine.dispose()


@pytest.mark.asyncio
async def test_public_lease_renewal_and_release(tmp_path: Path) -> None:
    uri = f"sqlite+aiosqlite:///{tmp_path / 'locks.db'}"
    first_engine = create_async_engine(uri)
    second_engine = create_async_engine(uri)
    try:
        first = SqlLeaseRWLockService(first_engine)
        second = SqlLeaseRWLockService(second_engine)
        await first.startup()
        read = await first.try_acquire_read(
            "resource", lease_duration=timedelta(milliseconds=300)
        )
        assert read is not None
        assert read.mode == "read"
        original_expiry = read.expires_at
        await asyncio.sleep(0.15)
        await read.renew()
        assert read.expires_at > original_expiry
        await asyncio.sleep(0.2)
        assert (
            await second.try_acquire_write(
                "resource", lease_duration=timedelta(seconds=1)
            )
            is None
        )
        await read.release()
        with pytest.raises(LeaseLostError):
            await read.release()
        writer = await second.try_acquire_write(
            "resource", lease_duration=timedelta(seconds=1)
        )
        assert writer is not None
        assert writer.fencing_token > read.fencing_token
    finally:
        await first_engine.dispose()
        await second_engine.dispose()


@pytest.mark.asyncio
async def test_expired_public_handle_cannot_touch_replacement(tmp_path: Path) -> None:
    uri = f"sqlite+aiosqlite:///{tmp_path / 'locks.db'}"
    first_engine = create_async_engine(uri)
    second_engine = create_async_engine(uri)
    try:
        first = SqlLeaseRWLockService(first_engine)
        second = SqlLeaseRWLockService(second_engine)
        await first.startup()
        old = await first.try_acquire_write(
            "resource", lease_duration=timedelta(milliseconds=100)
        )
        assert old is not None
        await asyncio.sleep(0.2)
        replacement = await second.try_acquire_write(
            "resource", lease_duration=timedelta(seconds=1)
        )
        assert replacement is not None
        with pytest.raises(LeaseLostError):
            await old.renew()
        with pytest.raises(LeaseLostError):
            await old.release()
        assert (
            await first.try_acquire_read(
                "resource", lease_duration=timedelta(seconds=1)
            )
            is None
        )
    finally:
        await first_engine.dispose()
        await second_engine.dispose()


@asynccontextmanager
async def one_service(
    path: Path,
) -> AsyncIterator[tuple[SqlLeaseRWLockService, AsyncEngine]]:
    engine = create_async_engine(f"sqlite+aiosqlite:///{path}")
    try:
        service = SqlLeaseRWLockService(engine)
        await service.startup()
        yield service, engine
    finally:
        await engine.dispose()


@pytest.mark.asyncio
async def test_waiting_acquire_gets_lock_after_release(tmp_path: Path) -> None:
    async with one_service(tmp_path / "locks.db") as (service, _engine):
        holder = await service.try_acquire_write(
            "resource", lease_duration=timedelta(seconds=1)
        )
        assert holder is not None
        waiter = asyncio.create_task(
            service.acquire_read(
                "resource",
                lease_duration=timedelta(seconds=1),
                wait_timeout=timedelta(seconds=1),
            )
        )
        await asyncio.sleep(0.05)
        assert not waiter.done()
        await holder.release()
        granted = await waiter
        assert granted.mode == "read"
        await granted.release()


@pytest.mark.asyncio
async def test_wait_timeout_and_waiter_cancellation(tmp_path: Path) -> None:
    async with one_service(tmp_path / "locks.db") as (service, _engine):
        holder = await service.try_acquire_write(
            "resource", lease_duration=timedelta(seconds=1)
        )
        assert holder is not None
        with pytest.raises(LockAcquireTimeout):
            await service.acquire_read(
                "resource",
                lease_duration=timedelta(seconds=1),
                wait_timeout=timedelta(0),
            )
        with pytest.raises(LockAcquireTimeout):
            await service.acquire_read(
                "resource",
                lease_duration=timedelta(seconds=1),
                wait_timeout=timedelta(milliseconds=50),
            )
        with pytest.raises(ValueError, match="wait timeout"):
            await service.acquire_read(
                "resource",
                lease_duration=timedelta(seconds=1),
                wait_timeout=timedelta(milliseconds=-1),
            )
        waiter = asyncio.create_task(
            service.acquire_read("resource", lease_duration=timedelta(seconds=1))
        )
        await asyncio.sleep(0.05)
        waiter.cancel()
        with pytest.raises(asyncio.CancelledError):
            await waiter
        await holder.release()
        replacement = await service.try_acquire_write(
            "resource", lease_duration=timedelta(seconds=1)
        )
        assert replacement is not None


@pytest.mark.asyncio
async def test_context_renews_and_releases(tmp_path: Path) -> None:
    async with one_service(tmp_path / "locks.db") as (service, _engine):
        async with service.read_lock(
            "resource", lease_duration=timedelta(milliseconds=120)
        ) as lease:
            await asyncio.sleep(0.3)
            assert lease.expires_at is not None
            assert (
                await service.try_acquire_write(
                    "resource", lease_duration=timedelta(seconds=1)
                )
                is None
            )
        writer = await service.try_acquire_write(
            "resource", lease_duration=timedelta(seconds=1)
        )
        assert writer is not None


@pytest.mark.asyncio
async def test_context_preserves_body_error_and_cleans_up(tmp_path: Path) -> None:
    async with one_service(tmp_path / "locks.db") as (service, _engine):
        with pytest.raises(RuntimeError, match="body failed"):
            async with service.write_lock(
                "resource", lease_duration=timedelta(seconds=1)
            ) as _lease:
                raise RuntimeError("body failed")
        next_holder = await service.try_acquire_write(
            "resource", lease_duration=timedelta(seconds=1)
        )
        assert next_holder is not None


@pytest.mark.asyncio
async def test_context_cancellation_releases_lock(tmp_path: Path) -> None:
    async with one_service(tmp_path / "locks.db") as (service, _engine):
        acquired = asyncio.Event()

        async def hold_forever() -> None:
            async with service.write_lock(
                "resource", lease_duration=timedelta(seconds=1)
            ) as _lease:
                acquired.set()
                await asyncio.Event().wait()

        task = asyncio.create_task(hold_forever())
        await acquired.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        next_holder = await service.try_acquire_write(
            "resource", lease_duration=timedelta(seconds=1)
        )
        assert next_holder is not None


@pytest.mark.asyncio
async def test_context_interrupts_body_when_lease_is_lost(tmp_path: Path) -> None:
    async with one_service(tmp_path / "locks.db") as (service, engine):

        async def lose_lease() -> None:
            async with service.write_lock(
                "resource", lease_duration=timedelta(milliseconds=120)
            ) as lease:
                async with engine.begin() as conn:
                    await conn.execute(
                        text("DELETE FROM lease_lock_holder WHERE lease_id = :id"),
                        {"id": lease.lease_id},
                    )
                await asyncio.sleep(0.3)

        with pytest.raises(LeaseLostError):
            await lose_lease()


@pytest.mark.asyncio
async def test_cancellation_after_sql_grant_releases_lease(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    async with one_service(tmp_path / "locks.db") as (service, _engine):
        real_try_acquire = service._store.try_acquire
        granted_in_store = asyncio.Event()
        resume_store = asyncio.Event()

        async def delayed_grant(key, mode, duration_ms):
            result = await real_try_acquire(key, mode, duration_ms)
            granted_in_store.set()
            await resume_store.wait()
            return result

        monkeypatch.setattr(service._store, "try_acquire", delayed_grant)
        waiter = asyncio.create_task(
            service.acquire_read("resource", lease_duration=timedelta(seconds=1))
        )
        await granted_in_store.wait()
        waiter.cancel()
        resume_store.set()
        with pytest.raises(asyncio.CancelledError):
            await waiter
        monkeypatch.setattr(service._store, "try_acquire", real_try_acquire)
        writer = await service.try_acquire_write(
            "resource", lease_duration=timedelta(seconds=1)
        )
        assert writer is not None


@pytest.mark.asyncio
async def test_context_waits_for_in_flight_renewal_before_release(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    async with one_service(tmp_path / "locks.db") as (service, _engine):
        renewing = asyncio.Event()
        finish_renewal = asyncio.Event()

        async def use_context() -> None:
            async with service.write_lock(
                "resource", lease_duration=timedelta(milliseconds=120)
            ) as lease:
                real_renew = lease.renew

                async def delayed_renew() -> None:
                    renewing.set()
                    await finish_renewal.wait()
                    await real_renew()

                monkeypatch.setattr(lease, "renew", delayed_renew)
                await renewing.wait()

        task = asyncio.create_task(use_context())
        await renewing.wait()
        await asyncio.sleep(0.02)
        assert not task.done()
        finish_renewal.set()
        await task
        writer = await service.try_acquire_write(
            "resource", lease_duration=timedelta(seconds=1)
        )
        assert writer is not None


@pytest.mark.asyncio
async def test_context_reports_renewal_failure_even_if_body_swallows_cancel(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    async with one_service(tmp_path / "locks.db") as (service, _engine):

        async def work() -> None:
            async with service.write_lock(
                "resource", lease_duration=timedelta(milliseconds=120)
            ) as lease:

                async def fail_renewal() -> None:
                    raise RuntimeError("renewal failed")

                monkeypatch.setattr(lease, "renew", fail_renewal)
                with suppress(asyncio.CancelledError):
                    await asyncio.sleep(0.3)

        with pytest.raises(RuntimeError, match="renewal failed"):
            await work()
