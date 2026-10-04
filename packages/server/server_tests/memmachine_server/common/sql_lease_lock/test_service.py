"""Cross-instance behavior for SQL lease locks."""

import asyncio
import logging
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
    SQLLeaseLockService,
)
from memmachine_server.common.sql_lease_lock import service as lease_service_module
from memmachine_server.common.sql_lease_lock._store import SQLLeaseStore, StoredLease


@asynccontextmanager
async def two_stores(path: Path) -> AsyncIterator[tuple[SQLLeaseStore, SQLLeaseStore]]:
    uri = f"sqlite+aiosqlite:///{path}"
    first_engine = create_async_engine(uri)
    second_engine = create_async_engine(uri)
    try:
        first = SQLLeaseStore(first_engine)
        second = SQLLeaseStore(second_engine)
        await first.startup()
        yield first, second
    finally:
        await first_engine.dispose()
        await second_engine.dispose()


@pytest.mark.asyncio
async def test_exclusive_grant_blocks_same_key_but_not_other_key(
    tmp_path: Path,
) -> None:
    async with two_stores(tmp_path / "locks.db") as (first, second):
        first_grant = await first.try_acquire("resource", 2_000)
        assert first_grant is not None
        assert await second.try_acquire("resource", 2_000) is None
        assert await second.try_acquire("other", 2_000) is not None
        assert await first.release("resource", first_grant.lease_id)
        replacement = await second.try_acquire("resource", 2_000)
        assert replacement is not None
        assert replacement.fencing_token > first_grant.fencing_token


@pytest.mark.asyncio
async def test_release_deletes_key_row_and_new_grant_keeps_increasing_token(
    tmp_path: Path,
) -> None:
    async with two_stores(tmp_path / "locks.db") as (first, second):
        first_grant = await first.try_acquire("ephemeral", 2_000)
        assert first_grant is not None
        assert await first.release("ephemeral", first_grant.lease_id)
        async with first._engine.connect() as conn:
            rows = (
                await conn.execute(
                    text("SELECT COUNT(*) FROM lease_lock WHERE key = 'ephemeral'")
                )
            ).scalar_one()
        assert rows == 0
        replacement = await second.try_acquire("ephemeral", 2_000)
        assert replacement is not None
        assert replacement.fencing_token > first_grant.fencing_token


@pytest.mark.asyncio
async def test_stale_renew_and_release_do_not_recreate_deleted_key(
    tmp_path: Path,
) -> None:
    async with two_stores(tmp_path / "locks.db") as (first, _second):
        lease = await first.try_acquire("ephemeral", 2_000)
        assert lease is not None
        assert await first.release("ephemeral", lease.lease_id)
        assert await first.renew("ephemeral", lease.lease_id, 2_000) is None
        assert not await first.release("ephemeral", lease.lease_id)
        async with first._engine.connect() as conn:
            rows = (
                await conn.execute(
                    text("SELECT COUNT(*) FROM lease_lock WHERE key = 'ephemeral'")
                )
            ).scalar_one()
        assert rows == 0


@pytest.mark.asyncio
async def test_simultaneous_acquires_cannot_both_grant_missing_key(
    tmp_path: Path,
) -> None:
    async with two_stores(tmp_path / "locks.db") as (first, second):
        grants = await asyncio.gather(
            first.try_acquire("resource", 2_000),
            second.try_acquire("resource", 2_000),
        )
        assert sum(grant is not None for grant in grants) == 1


@pytest.mark.asyncio
async def test_expired_lease_is_replaced_and_old_holder_loses_it(
    tmp_path: Path,
) -> None:
    async with two_stores(tmp_path / "locks.db") as (first, second):
        old = await first.try_acquire("resource", 100)
        other_key = await second.try_acquire("other", 2_000)
        assert old is not None
        assert other_key is not None
        await asyncio.sleep(0.2)
        replacement = await second.try_acquire("resource", 2_000)
        assert replacement is not None
        assert replacement.fencing_token > old.fencing_token
        assert await first.renew("resource", old.lease_id, 2_000) is None
        assert not await first.release("resource", old.lease_id)


@pytest.mark.asyncio
async def test_database_error_is_not_reported_as_contention(tmp_path: Path) -> None:
    async with two_stores(tmp_path / "locks.db") as (first, _second):
        async with first._engine.begin() as conn:
            await conn.execute(text("DROP TABLE lease_lock"))
        with pytest.raises(OperationalError):
            await first.try_acquire("resource", 2_000)


@pytest.mark.asyncio
async def test_public_service_validates_key_and_duration(tmp_path: Path) -> None:
    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'locks.db'}")
    try:
        service = SQLLeaseLockService(engine)
        await service.startup()
        with pytest.raises(ValueError, match="resource key"):
            await service.try_acquire("", lease_duration=timedelta(seconds=1))
        with pytest.raises(ValueError, match="lease duration"):
            await service.try_acquire("resource", lease_duration=timedelta(0))
        with pytest.raises(ValueError, match="at least 100 milliseconds"):
            await service.try_acquire(
                "resource", lease_duration=timedelta(microseconds=500)
            )
        with pytest.raises(ValueError, match="lease duration"):
            await service.try_acquire(
                "resource", lease_duration=timedelta(milliseconds=-1)
            )
        async with engine.connect() as conn:
            count = (
                await conn.execute(text("SELECT COUNT(*) FROM lease_lock"))
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
        first = SQLLeaseLockService(first_engine)
        second = SQLLeaseLockService(second_engine)
        await first.startup()
        lease = await first.try_acquire(
            "resource", lease_duration=timedelta(milliseconds=300)
        )
        assert lease is not None
        original_expiry = lease.expires_at
        await asyncio.sleep(0.15)
        await lease.renew()
        assert lease.expires_at > original_expiry
        await asyncio.sleep(0.2)
        assert (
            await second.try_acquire("resource", lease_duration=timedelta(seconds=1))
            is None
        )
        await lease.release()
        with pytest.raises(LeaseLostError):
            await lease.release()
        replacement = await second.try_acquire(
            "resource", lease_duration=timedelta(seconds=1)
        )
        assert replacement is not None
        assert replacement.fencing_token > lease.fencing_token
    finally:
        await first_engine.dispose()
        await second_engine.dispose()


@pytest.mark.asyncio
async def test_expired_public_handle_cannot_touch_replacement(tmp_path: Path) -> None:
    uri = f"sqlite+aiosqlite:///{tmp_path / 'locks.db'}"
    first_engine = create_async_engine(uri)
    second_engine = create_async_engine(uri)
    try:
        first = SQLLeaseLockService(first_engine)
        second = SQLLeaseLockService(second_engine)
        await first.startup()
        old = await first.try_acquire(
            "resource", lease_duration=timedelta(milliseconds=100)
        )
        assert old is not None
        await asyncio.sleep(0.2)
        replacement = await second.try_acquire(
            "resource", lease_duration=timedelta(seconds=1)
        )
        assert replacement is not None
        with pytest.raises(LeaseLostError):
            await old.renew()
        with pytest.raises(LeaseLostError):
            await old.release()
        assert (
            await first.try_acquire("resource", lease_duration=timedelta(seconds=1))
            is None
        )
    finally:
        await first_engine.dispose()
        await second_engine.dispose()


@asynccontextmanager
async def one_service(
    path: Path,
) -> AsyncIterator[tuple[SQLLeaseLockService, AsyncEngine]]:
    engine = create_async_engine(f"sqlite+aiosqlite:///{path}")
    try:
        service = SQLLeaseLockService(engine)
        await service.startup()
        yield service, engine
    finally:
        await engine.dispose()


@pytest.mark.asyncio
async def test_waiting_acquire_gets_lock_after_release(tmp_path: Path) -> None:
    async with one_service(tmp_path / "locks.db") as (service, _engine):
        holder = await service.try_acquire(
            "resource", lease_duration=timedelta(seconds=1)
        )
        assert holder is not None
        waiter = asyncio.create_task(
            service.acquire(
                "resource",
                lease_duration=timedelta(seconds=1),
                wait_timeout=timedelta(seconds=1),
            )
        )
        await asyncio.sleep(0.05)
        assert not waiter.done()
        await holder.release()
        granted = await waiter
        assert granted.lease_id != holder.lease_id
        await granted.release()


@pytest.mark.asyncio
async def test_waiting_acquire_caps_backoff_at_lease_duration(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    async with one_service(tmp_path / "locks.db") as (service, _engine):
        attempts = 0
        delays: list[float] = []
        granted = object()

        async def contended_then_granted(
            key: str, duration: timedelta
        ) -> object | None:
            nonlocal attempts
            assert key == "resource"
            assert duration == timedelta(seconds=1)
            attempts += 1
            return granted if attempts == 13 else None

        async def record_sleep(delay: float) -> None:
            delays.append(delay)

        monkeypatch.setattr(service, "_try_acquire", contended_then_granted)
        monkeypatch.setattr(lease_service_module, "uniform", lambda _low, high: high)
        monkeypatch.setattr(lease_service_module.asyncio, "sleep", record_sleep)

        assert (
            await service.acquire("resource", lease_duration=timedelta(seconds=1))
            is granted
        )
        assert delays == [0.05, 0.1, 0.2, 0.4, 0.8, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]


@pytest.mark.asyncio
async def test_retry_window_respects_short_wait_timeout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    async with one_service(tmp_path / "locks.db") as (service, _engine):
        attempts = 0
        delays: list[float] = []
        granted = object()

        async def contended_then_granted(
            _key: str, _duration: timedelta
        ) -> object | None:
            nonlocal attempts
            attempts += 1
            return granted if attempts == 2 else None

        async def record_sleep(delay: float) -> None:
            delays.append(delay)

        monkeypatch.setattr(service, "_try_acquire", contended_then_granted)
        monkeypatch.setattr(lease_service_module, "uniform", lambda low, _high: low)
        monkeypatch.setattr(lease_service_module.asyncio, "sleep", record_sleep)

        assert (
            await service.acquire(
                "resource",
                lease_duration=timedelta(seconds=1),
                wait_timeout=timedelta(milliseconds=10),
            )
            is granted
        )
        assert delays == [0.005]


@pytest.mark.asyncio
async def test_acquire_returns_grant_after_sleep_crosses_wait_timeout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    async with one_service(tmp_path / "locks.db") as (service, _engine):
        attempts = 0
        real_sleep = asyncio.sleep

        async def contended_then_granted(
            _key: str, _duration: timedelta
        ) -> object | None:
            nonlocal attempts
            attempts += 1
            return object() if attempts == 2 else None

        async def sleep_past_deadline(_delay: float) -> None:
            await real_sleep(0.02)

        monkeypatch.setattr(service, "_try_acquire", contended_then_granted)
        monkeypatch.setattr(lease_service_module.asyncio, "sleep", sleep_past_deadline)

        assert (
            await service.acquire(
                "resource",
                lease_duration=timedelta(seconds=1),
                wait_timeout=timedelta(milliseconds=10),
            )
            is not None
        )
        assert attempts == 2


@pytest.mark.asyncio
async def test_slow_database_grant_is_returned_after_wait_timeout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    async with one_service(tmp_path / "locks.db") as (service, _engine):
        finish_attempt = asyncio.Event()
        grant = StoredLease("resource", "late-grant", 1_000_000, 1)

        async def slow_grant(key: str, duration_ms: int) -> StoredLease:
            assert key == "resource"
            assert duration_ms == 1_000
            await finish_attempt.wait()
            return grant

        monkeypatch.setattr(service._store, "try_acquire", slow_grant)
        waiter = asyncio.create_task(
            service.acquire(
                "resource",
                lease_duration=timedelta(seconds=1),
                wait_timeout=timedelta(milliseconds=10),
            )
        )
        await asyncio.sleep(0.05)
        finish_attempt.set()
        lease = await waiter
        assert lease.lease_id == "late-grant"


@pytest.mark.asyncio
async def test_zero_wait_timeout_still_attempts_uncontended_lock(
    tmp_path: Path,
) -> None:
    async with one_service(tmp_path / "locks.db") as (service, _engine):
        lease = await service.acquire(
            "resource",
            lease_duration=timedelta(seconds=1),
            wait_timeout=timedelta(0),
        )
        await lease.release()


@pytest.mark.asyncio
async def test_wait_timeout_and_waiter_cancellation(tmp_path: Path) -> None:
    async with one_service(tmp_path / "locks.db") as (service, _engine):
        holder = await service.try_acquire(
            "resource", lease_duration=timedelta(seconds=1)
        )
        assert holder is not None
        with pytest.raises(LockAcquireTimeout):
            await service.acquire(
                "resource",
                lease_duration=timedelta(seconds=1),
                wait_timeout=timedelta(0),
            )
        with pytest.raises(LockAcquireTimeout):
            await service.acquire(
                "resource",
                lease_duration=timedelta(seconds=1),
                wait_timeout=timedelta(milliseconds=50),
            )
        with pytest.raises(ValueError, match="wait timeout"):
            await service.acquire(
                "resource",
                lease_duration=timedelta(seconds=1),
                wait_timeout=timedelta(milliseconds=-1),
            )
        waiter = asyncio.create_task(
            service.acquire("resource", lease_duration=timedelta(seconds=1))
        )
        await asyncio.sleep(0.05)
        waiter.cancel()
        with pytest.raises(asyncio.CancelledError):
            await waiter
        await holder.release()
        replacement = await service.try_acquire(
            "resource", lease_duration=timedelta(seconds=1)
        )
        assert replacement is not None


@pytest.mark.asyncio
async def test_context_renews_and_releases(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    async with one_service(tmp_path / "locks.db") as (service, _engine):
        async with service.lock(
            "resource", lease_duration=timedelta(seconds=1)
        ) as lease:
            initial_expiry = lease.expires_at
            renewed = asyncio.Event()
            real_renew = lease.renew

            async def observe_renewal() -> None:
                await real_renew()
                renewed.set()

            monkeypatch.setattr(lease, "renew", observe_renewal)
            await asyncio.wait_for(renewed.wait(), timeout=2)
            assert lease.expires_at > initial_expiry
            assert (
                await service.try_acquire(
                    "resource", lease_duration=timedelta(seconds=1)
                )
                is None
            )
        writer = await service.try_acquire(
            "resource", lease_duration=timedelta(seconds=1)
        )
        assert writer is not None


@pytest.mark.asyncio
async def test_context_preserves_body_error_and_cleans_up(tmp_path: Path) -> None:
    async with one_service(tmp_path / "locks.db") as (service, _engine):
        with pytest.raises(RuntimeError, match="body failed"):
            async with service.lock(
                "resource", lease_duration=timedelta(seconds=1)
            ) as _lease:
                raise RuntimeError("body failed")
        next_holder = await service.try_acquire(
            "resource", lease_duration=timedelta(seconds=1)
        )
        assert next_holder is not None


@pytest.mark.asyncio
async def test_context_cancellation_releases_lock(tmp_path: Path) -> None:
    async with one_service(tmp_path / "locks.db") as (service, _engine):
        acquired = asyncio.Event()

        async def hold_forever() -> None:
            async with service.lock(
                "resource", lease_duration=timedelta(seconds=1)
            ) as _lease:
                acquired.set()
                await asyncio.Event().wait()

        task = asyncio.create_task(hold_forever())
        await acquired.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        next_holder = await service.try_acquire(
            "resource", lease_duration=timedelta(seconds=1)
        )
        assert next_holder is not None


@pytest.mark.asyncio
async def test_context_interrupts_body_when_lease_is_lost(tmp_path: Path) -> None:
    async with one_service(tmp_path / "locks.db") as (service, engine):

        async def lose_lease() -> None:
            async with service.lock(
                "resource", lease_duration=timedelta(milliseconds=120)
            ) as lease:
                async with engine.begin() as conn:
                    await conn.execute(
                        text(
                            "UPDATE lease_lock SET lease_id = NULL, expires_at_ms = NULL WHERE lease_id = :id"
                        ),
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

        async def delayed_grant(key, duration_ms):
            result = await real_try_acquire(key, duration_ms)
            granted_in_store.set()
            await resume_store.wait()
            return result

        monkeypatch.setattr(service._store, "try_acquire", delayed_grant)
        waiter = asyncio.create_task(
            service.acquire("resource", lease_duration=timedelta(seconds=1))
        )
        await granted_in_store.wait()
        waiter.cancel()
        resume_store.set()
        with pytest.raises(asyncio.CancelledError):
            await waiter
        monkeypatch.setattr(service._store, "try_acquire", real_try_acquire)
        writer = await service.try_acquire(
            "resource", lease_duration=timedelta(seconds=1)
        )
        assert writer is not None


@pytest.mark.asyncio
async def test_repeated_cancellation_after_sql_grant_releases_lease(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    async with one_service(tmp_path / "locks.db") as (service, _engine):
        real_try_acquire = service._store.try_acquire
        granted_in_store = asyncio.Event()
        resume_store = asyncio.Event()

        async def delayed_grant(key, duration_ms):
            result = await real_try_acquire(key, duration_ms)
            granted_in_store.set()
            await resume_store.wait()
            return result

        monkeypatch.setattr(service._store, "try_acquire", delayed_grant)
        waiter = asyncio.create_task(
            service.acquire("resource", lease_duration=timedelta(seconds=1))
        )
        await granted_in_store.wait()
        waiter.cancel()
        await asyncio.sleep(0)
        waiter.cancel()
        resume_store.set()
        with pytest.raises(asyncio.CancelledError):
            await waiter
        monkeypatch.setattr(service._store, "try_acquire", real_try_acquire)
        replacement = await service.try_acquire(
            "resource", lease_duration=timedelta(seconds=1)
        )
        assert replacement is not None


@pytest.mark.asyncio
async def test_context_waits_for_in_flight_renewal_before_release(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    async with one_service(tmp_path / "locks.db") as (service, _engine):
        renewing = asyncio.Event()
        finish_renewal = asyncio.Event()

        async def use_context() -> None:
            async with service.lock(
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
        writer = await service.try_acquire(
            "resource", lease_duration=timedelta(seconds=1)
        )
        assert writer is not None


@pytest.mark.asyncio
async def test_cancellation_during_context_exit_releases_lease(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    async with one_service(tmp_path / "locks.db") as (service, _engine):
        renewing = asyncio.Event()
        finish_renewal = asyncio.Event()
        body_finished = asyncio.Event()

        async def work() -> None:
            async with service.lock(
                "resource", lease_duration=timedelta(milliseconds=500)
            ) as lease:
                real_renew = lease.renew

                async def delayed_renew() -> None:
                    renewing.set()
                    await finish_renewal.wait()
                    await real_renew()

                monkeypatch.setattr(lease, "renew", delayed_renew)
                await renewing.wait()
                body_finished.set()

        task = asyncio.create_task(work())
        await body_finished.wait()
        task.cancel()
        await asyncio.sleep(0)
        finish_renewal.set()
        with pytest.raises(asyncio.CancelledError):
            await task
        replacement = await service.try_acquire(
            "resource", lease_duration=timedelta(seconds=1)
        )
        assert replacement is not None


@pytest.mark.asyncio
async def test_context_reports_renewal_failure_even_if_body_swallows_cancel(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    async with one_service(tmp_path / "locks.db") as (service, _engine):

        async def work() -> None:
            async with service.lock(
                "resource", lease_duration=timedelta(milliseconds=120)
            ) as lease:

                async def fail_renewal() -> None:
                    raise RuntimeError("renewal failed")

                monkeypatch.setattr(lease, "renew", fail_renewal)
                with suppress(asyncio.CancelledError):
                    await asyncio.sleep(0.3)

        with pytest.raises(LeaseLostError, match="can no longer be trusted"):
            await work()


@pytest.mark.asyncio
async def test_renewal_failure_during_exit_reports_error_and_releases(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    async with one_service(tmp_path / "locks.db") as (service, _engine):
        renew_started = asyncio.Event()
        finish_renewal = asyncio.Event()
        body_finished = asyncio.Event()

        async def work() -> None:
            async with service.lock(
                "resource", lease_duration=timedelta(milliseconds=500)
            ) as lease:

                async def failing_renew() -> None:
                    renew_started.set()
                    await finish_renewal.wait()
                    raise LeaseLostError("renew lost lease during exit")

                monkeypatch.setattr(lease, "renew", failing_renew)
                await renew_started.wait()
                body_finished.set()

        task = asyncio.create_task(work())
        await asyncio.wait_for(body_finished.wait(), timeout=2)
        assert not task.done()
        finish_renewal.set()
        with pytest.raises(LeaseLostError, match="renew lost lease during exit"):
            await task
        replacement = await service.try_acquire(
            "resource", lease_duration=timedelta(seconds=1)
        )
        assert replacement is not None


@pytest.mark.asyncio
@pytest.mark.parametrize("outside_cancel", [False, True])
async def test_renewal_failure_restores_only_its_cancellation_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, outside_cancel: bool
) -> None:
    async with one_service(tmp_path / "locks.db") as (service, _engine):

        async def work() -> int:
            owner = asyncio.current_task()
            assert owner is not None
            try:
                async with service.lock(
                    "resource", lease_duration=timedelta(milliseconds=120)
                ) as lease:

                    async def failing_renew() -> None:
                        if outside_cancel:
                            owner.cancel()
                        raise LeaseLostError("renewal lost the lease")  # noqa: TRY301

                    monkeypatch.setattr(lease, "renew", failing_renew)
                    await asyncio.sleep(0.3)
            except LeaseLostError:
                return owner.cancelling()
            pytest.fail("renewal failure was not reported")

        task = asyncio.create_task(work())
        if outside_cancel:
            with pytest.raises(asyncio.CancelledError):
                await task
        else:
            assert await task == 0


@pytest.mark.asyncio
async def test_transient_renewal_error_does_not_cancel_body(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    async with one_service(tmp_path / "locks.db") as (service, _engine):
        attempts = 0
        renewed = asyncio.Event()

        async def work() -> None:
            nonlocal attempts
            async with service.lock(
                "resource", lease_duration=timedelta(seconds=1)
            ) as lease:
                real_renew = lease.renew

                async def flaky_renew() -> None:
                    nonlocal attempts
                    attempts += 1
                    if attempts == 1:
                        raise RuntimeError("transient renewal error")
                    await real_renew()
                    renewed.set()

                monkeypatch.setattr(lease, "renew", flaky_renew)
                await asyncio.wait_for(renewed.wait(), timeout=2)

        await work()
        assert attempts >= 2


@pytest.mark.asyncio
async def test_cancelled_acquire_preserves_cancellation_on_store_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    async with one_service(tmp_path / "locks.db") as (service, _engine):
        started = asyncio.Event()
        finish = asyncio.Event()

        async def failed_acquire(_key: str, _duration_ms: int) -> None:
            started.set()
            await finish.wait()
            raise RuntimeError("database failed")

        monkeypatch.setattr(service._store, "try_acquire", failed_acquire)
        task = asyncio.create_task(
            service.try_acquire("resource", lease_duration=timedelta(seconds=1))
        )
        await started.wait()
        task.cancel()
        finish.set()
        with pytest.raises(asyncio.CancelledError):
            await task


@pytest.mark.asyncio
async def test_cleanup_wait_has_deadline(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(lease_service_module, "_CLEANUP_TIMEOUT_SECONDS", 0.02)
    finish = asyncio.Event()
    task = asyncio.create_task(finish.wait())
    loop = asyncio.get_running_loop()
    started = loop.time()
    loop.call_later(0.12, finish.set)
    try:
        async with asyncio.timeout(0.1):
            with pytest.raises(TimeoutError):
                await lease_service_module._await_cleanup(task, [])
        assert loop.time() - started < 0.08
    finally:
        finish.set()
        await task


@pytest.mark.asyncio
async def test_repeated_cancellation_does_not_extend_cleanup_deadline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(lease_service_module, "_CLEANUP_TIMEOUT_SECONDS", 0.03)
    finish = asyncio.Event()
    cleanup = asyncio.create_task(finish.wait())
    cancellations: list[bool] = []
    owner = asyncio.create_task(
        lease_service_module._await_cleanup(cleanup, cancellations)
    )
    try:
        for _ in range(3):
            await asyncio.sleep(0.005)
            owner.cancel()
        with pytest.raises(TimeoutError):
            await asyncio.wait_for(owner, timeout=0.1)
        assert len(cancellations) == 3
    finally:
        finish.set()
        await cleanup


@pytest.mark.asyncio
async def test_release_error_is_logged_while_body_error_propagates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    async with one_service(tmp_path / "locks.db") as (service, _engine):

        async def failed_release(_key: str, _lease_id: str) -> bool:
            raise RuntimeError("release database failed")

        monkeypatch.setattr(service._store, "release", failed_release)
        with (
            caplog.at_level(logging.ERROR),
            pytest.raises(ValueError, match="body failed"),
        ):
            async with service.lock("resource", lease_duration=timedelta(seconds=1)):
                raise ValueError("body failed")
        assert "release database failed" in caplog.text
