"""Tests for the event-loop diagnostics."""

import asyncio
import contextlib
import time
from unittest.mock import MagicMock

import pytest

from memmachine_server.common.metrics_factory import MetricsFactory
from memmachine_server.server.diagnostics import report_event_loop_lag


def _keep_the_loop_busy(seconds: float) -> None:
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        pass


@pytest.mark.asyncio
async def test_a_busy_loop_is_reported_as_lag():
    lag = MagicMock()
    metrics_factory = MagicMock(spec=MetricsFactory)
    metrics_factory.get_histogram.return_value = lag

    reporting = asyncio.create_task(report_event_loop_lag(metrics_factory))
    try:
        # Let the reporter schedule its first wakeup, then hold the loop past it.
        await asyncio.sleep(0)
        _keep_the_loop_busy(0.5)
        await asyncio.sleep(0.05)
    finally:
        reporting.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await reporting

    observed = [observation.args[0] for observation in lag.observe.call_args_list]
    assert max(observed) >= 0.35
    assert min(observed) >= 0
