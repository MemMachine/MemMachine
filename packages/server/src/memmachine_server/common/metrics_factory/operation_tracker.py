"""OperationTracker: async context manager for timing operations."""

import time
from collections.abc import AsyncIterator
from contextlib import AbstractAsyncContextManager, asynccontextmanager

from .metrics_factory import MetricsFactory


class OperationTracker:
    """Tracks operation latency via async context manager.

    Emits a single histogram ``{prefix}_latency_seconds`` with ``operation``
    and ``status`` ("ok" / "error") labels.

    Usage::

        tracker = OperationTracker(factory, prefix="my_service")
        async with tracker("read"):
            result = await client.read(...)
    """

    def __init__(
        self,
        factory: MetricsFactory | None,
        prefix: str,
    ) -> None:
        """Register a single latency histogram for all operations."""
        self._histogram: MetricsFactory.Histogram | None = None
        if factory is not None:
            self._histogram = factory.get_histogram(
                f"{prefix}_latency_seconds",
                f"Latency in seconds for {prefix} operations",
                label_names=["operation", "status"],
            )

    def __call__(self, operation: str) -> AbstractAsyncContextManager[None]:
        return self._track(operation)

    @asynccontextmanager
    async def _track(self, operation: str) -> AsyncIterator[None]:
        start = time.monotonic()
        status = "ok"
        try:
            yield
        except Exception:
            status = "error"
            raise
        finally:
            self.emit(operation, time.monotonic() - start, status)

    def emit(self, operation: str, elapsed: float, status: str = "ok") -> None:
        """Manually emit a latency observation."""
        if self._histogram is not None:
            self._histogram.observe(
                value=elapsed,
                labels={"operation": operation, "status": status},
            )
