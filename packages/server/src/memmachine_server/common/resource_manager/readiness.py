"""Bounded, fresh storage checks which do not own serving clients."""

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from enum import StrEnum
from time import monotonic
from typing import Literal

from pydantic import BaseModel, Field


class ComponentStatus(StrEnum):
    """Public readiness states; backend exception messages are never returned."""

    READY = "ready"
    UNAVAILABLE = "unavailable"
    TIMEOUT = "timeout"
    UNSUPPORTED = "unsupported"
    STARTING = "starting"
    STOPPING = "stopping"


class ReadinessResult(BaseModel):
    """Storage availability for an initialized application process."""

    status: Literal["ready", "not_ready"] = "not_ready"
    service: str = "memmachine"
    components: dict[str, ComponentStatus] = Field(default_factory=dict)


class UnsupportedProbeError(Exception):
    """The configured backend cannot be checked with a bounded read-only probe."""


@dataclass(frozen=True)
class StorageProbe:
    """A private resource key, a safe public label, and a read-only operation."""

    key: tuple[str, str]
    label: str
    check: Callable[[], Awaitable[None]]


class ReadinessService:
    """Coalesce requests without cancelling or duplicating underlying work.

    Results are reused for at most one second from the refresh's start. Each
    check has a two-second observation budget and the HTTP-facing call has a
    three-second budget. A timed-out operation retains its resource reservation
    until it actually finishes, including synchronous SDK work. At most four
    operations run concurrently, regardless of probe request volume.
    """

    def __init__(
        self,
        probes: Callable[[], list[StorageProbe]],
        *,
        check_timeout: float = 2.0,
        cache_seconds: float = 1.0,
        response_timeout: float = 3.0,
        max_concurrency: int = 4,
    ) -> None:
        """Initialize a per-process coordinator without starting background work."""
        self._probes = probes
        self._check_timeout = check_timeout
        self._cache_seconds = cache_seconds
        self._response_timeout = response_timeout
        self._slots = asyncio.Semaphore(max_concurrency)
        self._lifecycle = ComponentStatus.STARTING
        self._operations: dict[tuple[str, str], asyncio.Task[None]] = {}
        self._refresh: asyncio.Task[ReadinessResult] | None = None
        self._cached: tuple[float, ReadinessResult] | None = None

    def started(self) -> None:
        """Allow readiness only after all application startup steps succeed."""
        if self._lifecycle == ComponentStatus.STARTING:
            self._lifecycle = ComponentStatus.READY
            self._cached = None

    def stopping(self) -> None:
        """Withdraw readiness synchronously before resource teardown starts."""
        self._lifecycle = ComponentStatus.STOPPING
        self._cached = None

    def _lifecycle_result(self) -> ReadinessResult:
        return ReadinessResult(components={"lifecycle": self._lifecycle})

    async def check(self) -> ReadinessResult:
        """Return fresh availability, bounded even if a backend ignores cancellation."""
        if self._lifecycle != ComponentStatus.READY:
            return self._lifecycle_result()
        if self._cached is not None:
            checked_at, result = self._cached
            if monotonic() - checked_at < self._cache_seconds:
                return result
        if self._refresh is None or self._refresh.done():
            self._refresh = asyncio.create_task(self._check_all())
            self._refresh.add_done_callback(self._observe_exception)
        try:
            async with asyncio.timeout(self._response_timeout):
                result = await asyncio.shield(self._refresh)
        except TimeoutError:
            result = ReadinessResult(
                components={"storage": ComponentStatus.TIMEOUT},
            )
        except asyncio.CancelledError:
            # Closing the coordinator cancels the shared observation task.
            # Requests already waiting on it must also see shutdown as 503.
            if self._lifecycle == ComponentStatus.STOPPING:
                return self._lifecycle_result()
            raise
        if self._lifecycle != ComponentStatus.READY:
            return self._lifecycle_result()
        return result

    async def _check_all(self) -> ReadinessResult:
        started_at = monotonic()
        try:
            probes = self._probes()
            selected = {probe.key for probe in probes}
            self._operations = {
                key: operation
                for key, operation in self._operations.items()
                if key in selected or not operation.done()
            }
            statuses = await asyncio.gather(*(self._check_one(p) for p in probes))
        except Exception:
            # Configuration errors are an availability failure, not a response
            # containing an endpoint, credential, resource name, or traceback.
            result = ReadinessResult(
                components={"configuration": ComponentStatus.UNAVAILABLE},
            )
        else:
            components = dict(zip((p.label for p in probes), statuses, strict=True))
            result = ReadinessResult(
                status="ready"
                if all(s == ComponentStatus.READY for s in statuses)
                else "not_ready",
                components=components,
            )
        if self._lifecycle == ComponentStatus.READY:
            self._cached = (started_at, result)
        return result

    async def _run(self, probe: StorageProbe) -> None:
        async with self._slots:
            await probe.check()

    async def _check_one(self, probe: StorageProbe) -> ComponentStatus:
        operation = self._operations.get(probe.key)
        if operation is None or operation.done():
            operation = asyncio.create_task(self._run(probe))
            operation.add_done_callback(self._observe_exception)
            self._operations[probe.key] = operation
        # wait() does not cancel the actual task on timeout or caller
        # cancellation. In particular, a worker thread still owns its slot.
        done, _ = await asyncio.wait({operation}, timeout=self._check_timeout)
        if not done:
            return ComponentStatus.TIMEOUT
        try:
            operation.result()
        except UnsupportedProbeError:
            return ComponentStatus.UNSUPPORTED
        except TimeoutError:
            return ComponentStatus.TIMEOUT
        except Exception:
            return ComponentStatus.UNAVAILABLE
        return ComponentStatus.READY

    @staticmethod
    def _observe_exception(task: asyncio.Task[object]) -> None:
        if not task.cancelled():
            task.exception()

    async def close(self) -> None:
        """Stop admission and bound teardown without closing borrowed clients."""
        self.stopping()
        if self._refresh is not None:
            self._refresh.cancel()
        operations = set(self._operations.values())
        if operations:
            _, pending = await asyncio.wait(operations, timeout=self._response_timeout)
            for operation in pending:
                operation.cancel()
        self._cached = None
