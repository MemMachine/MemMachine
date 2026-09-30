"""Exclusive SQL leases for distributed tasks across server instances."""

import asyncio
from collections.abc import AsyncGenerator
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from datetime import UTC, datetime, timedelta
from random import uniform

from sqlalchemy.ext.asyncio import AsyncEngine

from memmachine_server.common.sql_lease_lock._store import (
    SQLLeaseStore,
    StoredLease,
)


class LeaseLostError(RuntimeError):
    """The caller no longer owns its lease."""


class LockAcquireTimeout(TimeoutError):  # noqa: N818 - name fixed by public API
    """A lock remained unavailable until the wait deadline."""


def _duration_ms(duration: timedelta) -> int:
    microseconds = (
        duration.days * 86_400 + duration.seconds
    ) * 1_000_000 + duration.microseconds
    if microseconds <= 0:
        raise ValueError("lease duration must be positive")
    return (microseconds + 999) // 1_000


class Lease:
    """A handle that can renew or release one specific SQL lease."""

    def __init__(
        self, store: SQLLeaseStore, stored: StoredLease, duration_ms: int
    ) -> None:
        """Bind this handle to one granted lease."""
        self._store = store
        self._stored = stored
        self._duration_ms = duration_ms
        self._expires_at_ms = stored.expires_at_ms

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
        """A resource-local token that grows with each grant."""
        return self._stored.fencing_token

    @property
    def expires_at(self) -> datetime:
        """The last known expiry, in UTC."""
        return datetime.fromtimestamp(self._expires_at_ms / 1_000, tz=UTC)

    async def renew(self) -> None:
        """Extend this lease from database time or raise if it was lost."""
        expiry = await self._store.renew(self.key, self.lease_id, self._duration_ms)
        if expiry is None:
            raise LeaseLostError(f"Lease {self.lease_id} is no longer held")
        self._expires_at_ms = expiry

    async def release(self) -> None:
        """Remove this lease or raise if it was already lost."""
        if not await self._store.release(self.key, self.lease_id):
            raise LeaseLostError(f"Lease {self.lease_id} is no longer held")


class SQLLeaseLockService:
    """Grant exclusive leases for distributed tasks that outlive SQL transactions.

    A key has one live holder, but waiting callers have no FIFO guarantee.
    Per-key rows remain to keep fencing tokens monotonic. Use bounded or
    stable key sets; ephemeral keys need a separate retention strategy.
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
        operation = asyncio.create_task(self._store.try_acquire(key, duration_ms))
        try:
            stored = await asyncio.shield(operation)
        except asyncio.CancelledError:
            # Let the database transaction finish before returning cancellation.
            # If it granted a lease, release that grant before the caller exits.
            stored = await operation
            if stored is not None:
                await self._store.release(key, stored.lease_id)
            raise
        if stored is None:
            return None
        return Lease(self._store, stored, duration_ms)

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
        while True:
            lease = await self._try_acquire(key, lease_duration)
            if lease is not None:
                return lease
            remaining = None if deadline is None else deadline - loop.time()
            if remaining is not None and remaining <= 0:
                raise LockAcquireTimeout(f"Timed out acquiring lock for {key!r}")
            delay = uniform(0.025, 0.05)
            await asyncio.sleep(delay if remaining is None else min(delay, remaining))

    def lock(
        self,
        key: str,
        *,
        lease_duration: timedelta,
        wait_timeout: timedelta | None = None,
    ) -> AbstractAsyncContextManager[Lease]:
        """Hold an exclusive lease, renewing it until the block exits."""
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
        while not stop_renewal.is_set():
            try:
                await asyncio.wait_for(stop_renewal.wait(), timeout=interval)
            except TimeoutError:
                pass
            else:
                return
            if stop_renewal.is_set():
                return
            try:
                await lease.renew()
            except Exception as err:
                errors.append(err)
                if stop_renewal.is_set():
                    return
                owner_cancellations.append(True)
                owner.cancel()
                return

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
            await renewal_task
            try:
                await lease.release()
            except Exception:
                if body_error is None and not renewal_errors:
                    raise
            if body_error is None and renewal_errors:
                raise renewal_errors[0]
