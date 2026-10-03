"""Exclusive SQL leases for distributed tasks across server instances."""

import asyncio
import logging
from collections.abc import AsyncGenerator
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from datetime import UTC, datetime, timedelta
from random import uniform

from sqlalchemy.ext.asyncio import AsyncEngine

from memmachine_server.common.sql_lease_lock._store import (
    SQLLeaseStore,
    StoredLease,
)

logger = logging.getLogger(__name__)
_CLEANUP_TIMEOUT_SECONDS = 5.0


def _observe_late_cleanup(task: asyncio.Task[object]) -> None:
    if not task.cancelled() and (error := task.exception()) is not None:
        logger.error("SQL lease cleanup failed after its deadline", exc_info=error)


class LeaseLostError(RuntimeError):
    """The caller no longer owns its lease."""


class LockAcquireTimeoutError(TimeoutError):
    """A lock remained unavailable until the wait deadline."""


LockAcquireTimeout = LockAcquireTimeoutError


async def _await_cleanup[T](task: asyncio.Task[T], cancellations: list[bool]) -> T:
    """Allow cleanup a bounded time despite cancellation of its caller."""
    loop = asyncio.get_running_loop()
    deadline = loop.time() + _CLEANUP_TIMEOUT_SECONDS
    while True:
        remaining = deadline - loop.time()
        if remaining <= 0:
            task.add_done_callback(_observe_late_cleanup)
            raise TimeoutError("SQL lease cleanup timed out")
        try:
            return await asyncio.wait_for(asyncio.shield(task), remaining)
        except asyncio.CancelledError:
            if task.cancelled():
                raise
            cancellations.append(True)
        except TimeoutError:
            task.add_done_callback(_observe_late_cleanup)
            raise


def _duration_ms(duration: timedelta) -> int:
    microseconds = (
        duration.days * 86_400 + duration.seconds
    ) * 1_000_000 + duration.microseconds
    if microseconds <= 0:
        raise ValueError("lease duration must be positive")
    if microseconds < 100_000:
        raise ValueError("lease duration must be at least 100 milliseconds")
    return (microseconds + 999) // 1_000


class Lease:
    """A handle that can renew or release one specific SQL lease."""

    def __init__(
        self,
        store: SQLLeaseStore,
        stored: StoredLease,
        duration_ms: int,
        granted_after: float,
    ) -> None:
        """Bind this handle to one granted lease."""
        self._store = store
        self._stored = stored
        self._duration_ms = duration_ms
        self._expires_at_ms = stored.expires_at_ms
        self._safe_until = granted_after + duration_ms / 1_000

    @property
    def key(self) -> str:
        """The locked resource key."""
        return self._stored.key

    @property
    def lease_id(self) -> str:
        """An opaque identifier unique to this grant."""
        return self._stored.lease_id

    @property
    def fencing_token(self) -> int:
        """A global token that grows with each successful grant."""
        return self._stored.fencing_token

    @property
    def expires_at(self) -> datetime:
        """The last known expiry, in UTC."""
        return datetime.fromtimestamp(self._expires_at_ms / 1_000, tz=UTC)

    def trusted_for(self) -> float:
        """Seconds until this handle's locally conservative lease deadline."""
        return self._safe_until - asyncio.get_running_loop().time()

    async def renew(self) -> None:
        """Extend this lease from database time or raise if it was lost."""
        started = asyncio.get_running_loop().time()
        expiry = await self._store.renew(self.key, self.lease_id, self._duration_ms)
        if expiry is None:
            raise LeaseLostError(f"Lease {self.lease_id} is no longer held")
        self._expires_at_ms = expiry
        self._safe_until = started + self._duration_ms / 1_000

    async def release(self) -> None:
        """Remove this lease or raise if it was already lost."""
        if not await self._store.release(self.key, self.lease_id):
            raise LeaseLostError(f"Lease {self.lease_id} is no longer held")


async def _renew_before_deadline(lease: Lease) -> None:
    remaining = lease.trusted_for()
    if remaining <= 0:
        raise LeaseLostError(f"Lease {lease.lease_id} can no longer be trusted")
    async with asyncio.timeout(remaining):
        await lease.renew()


async def _finish_and_release(renewal_task: asyncio.Task[None], lease: Lease) -> None:
    try:
        await renewal_task
    finally:
        await lease.release()


async def _finish_lock_context(
    renewal_task: asyncio.Task[None],
    lease: Lease,
    body_error: BaseException | None,
    renewal_errors: list[Exception],
) -> None:
    cleanup = asyncio.create_task(_finish_and_release(renewal_task, lease))
    cancellations: list[bool] = []
    try:
        await _await_cleanup(cleanup, cancellations)
    except Exception:
        if body_error is None and not renewal_errors and not cancellations:
            raise
        logger.exception("Failed to release SQL lease during context cleanup")
    if cancellations:
        if renewal_errors:
            logger.error("SQL lease renewal also failed", exc_info=renewal_errors[0])
        raise asyncio.CancelledError
    if body_error is None and renewal_errors:
        raise renewal_errors[0]


class SQLLeaseLockService:
    """Grant exclusive leases for distributed tasks that outlive SQL transactions.

    A key has one live holder, but waiting callers have no FIFO guarantee.
    Successful release deletes the key row. A durable global counter keeps
    fencing tokens monotonic when a key is later reacquired. Expired leases
    whose workers never return leave rows until that key is acquired again;
    workloads with abandoned one-time keys need periodic expiry cleanup.
    Consumers must check fencing tokens at external side-effect boundaries.
    """

    def __init__(self, engine: AsyncEngine) -> None:
        """Use an existing asynchronous SQLAlchemy engine."""
        self._store = SQLLeaseStore(engine)

    async def startup(self) -> None:
        """Create the service's SQL tables."""
        await self._store.startup()

    async def try_acquire(self, key: str, *, lease_duration: timedelta) -> Lease | None:
        """Try once to acquire an exclusive lease; return None on contention."""
        return await self._try_acquire(key, lease_duration)

    async def _try_acquire(self, key: str, duration: timedelta) -> Lease | None:
        if not key:
            raise ValueError("resource key must be nonempty")
        duration_ms = _duration_ms(duration)
        started = asyncio.get_running_loop().time()
        operation = asyncio.create_task(self._store.try_acquire(key, duration_ms))
        try:
            stored = await asyncio.shield(operation)
        except asyncio.CancelledError:
            # Let the database transaction finish before returning cancellation.
            # If it granted a lease, release that grant before the caller exits.
            async def discard_grant() -> None:
                stored = await operation
                if stored is not None:
                    await self._store.release(key, stored.lease_id)

            cleanup = asyncio.create_task(discard_grant())
            try:
                await _await_cleanup(cleanup, [])
            except Exception:
                logger.exception("Failed to clean up cancelled lock acquisition")
            raise
        if stored is None:
            return None
        return Lease(self._store, stored, duration_ms, started)

    async def acquire(
        self,
        key: str,
        *,
        lease_duration: timedelta,
        wait_timeout: timedelta | None = None,
    ) -> Lease:
        """Wait for an exclusive lease or raise on timeout."""
        return await self._acquire(key, lease_duration, wait_timeout)

    async def _acquire(
        self,
        key: str,
        lease_duration: timedelta,
        wait_timeout: timedelta | None,
    ) -> Lease:
        if wait_timeout is not None and wait_timeout < timedelta(0):
            raise ValueError("wait timeout cannot be negative")
        loop = asyncio.get_running_loop()
        deadline = (
            None if wait_timeout is None else loop.time() + wait_timeout.total_seconds()
        )
        retry_cap = (
            30.0 if wait_timeout is None else min(30.0, wait_timeout.total_seconds())
        )
        retry_cap = min(retry_cap, lease_duration.total_seconds())
        retry_max_delay = min(0.05, retry_cap)
        while True:
            lease = await self._try_acquire(key, lease_duration)
            if lease is not None:
                return lease
            remaining = None if deadline is None else deadline - loop.time()
            if remaining is not None and remaining <= 0:
                raise LockAcquireTimeoutError(f"Timed out acquiring lock for {key!r}")
            delay = uniform(retry_max_delay / 2, retry_max_delay)
            await asyncio.sleep(delay if remaining is None else min(delay, remaining))
            retry_max_delay = min(retry_max_delay * 2, retry_cap)

    def lock(
        self,
        key: str,
        *,
        lease_duration: timedelta,
        wait_timeout: timedelta | None = None,
    ) -> AbstractAsyncContextManager[Lease]:
        """Hold and renew a lease until exit.

        Releasing the handle inside the block or allowing the lease to expire
        raises LeaseLostError, possibly after cancelling the body.
        """
        return self._lock_context(key, lease_duration, wait_timeout)

    @staticmethod
    async def _renew_loop(
        lease: Lease,
        stop_renewal: asyncio.Event,
        owner: asyncio.Task[object],
        interval: float,
        errors: list[Exception],
        owner_cancellations: list[bool],
    ) -> None:
        retry_delay = interval
        while not stop_renewal.is_set():
            try:
                await asyncio.wait_for(stop_renewal.wait(), timeout=retry_delay)
            except TimeoutError:
                pass
            else:
                return
            if stop_renewal.is_set():
                return
            try:
                await _renew_before_deadline(lease)
            except Exception as err:
                if not isinstance(err, LeaseLostError):
                    remaining = lease.trusted_for()
                    if remaining > 0:
                        logger.debug("Retrying failed SQL lease renewal", exc_info=err)
                        retry_delay = min(0.05, max(0.005, remaining / 2))
                        continue
                    err = LeaseLostError(
                        f"Lease {lease.lease_id} can no longer be trusted"
                    )
                    logger.exception("SQL lease renewal failed until expiry")
                errors.append(err)
                if stop_renewal.is_set():
                    return
                owner_cancellations.append(True)
                owner.cancel()
                return
            retry_delay = interval

    @asynccontextmanager
    async def _lock_context(
        self,
        key: str,
        lease_duration: timedelta,
        wait_timeout: timedelta | None,
    ) -> AsyncGenerator[Lease, None]:
        lease = await self._acquire(key, lease_duration, wait_timeout)
        owner = asyncio.current_task()
        if owner is None:
            raise RuntimeError("A lock context requires an asyncio task")
        stop_renewal = asyncio.Event()
        renewal_errors: list[Exception] = []
        owner_cancellations: list[bool] = []
        initial_cancellations = owner.cancelling()
        renewal_task = asyncio.create_task(
            self._renew_loop(
                lease,
                stop_renewal,
                owner,
                lease_duration.total_seconds() / 3,
                renewal_errors,
                owner_cancellations,
            )
        )
        body_error: BaseException | None = None
        try:
            yield lease
        except asyncio.CancelledError as err:
            if renewal_errors:
                if owner.cancelling() > initial_cancellations + 1:
                    body_error = err
                    raise
                body_error = renewal_errors[0]
                raise renewal_errors[0] from err
            body_error = err
            raise
        except BaseException as err:
            body_error = err
            raise
        finally:
            stop_renewal.set()
            if owner_cancellations:
                # The renewal loop issued one cancel request. Remove only that
                # request when this context substitutes its renewal error.
                owner.uncancel()
            await _finish_lock_context(renewal_task, lease, body_error, renewal_errors)
