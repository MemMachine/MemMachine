"""Tests for the HTTP app's lifespan."""

import asyncio
import time
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from unittest.mock import MagicMock

import pytest
from fastapi import FastAPI

from memmachine_server.common.metrics_factory import MetricsFactory
from memmachine_server.server.api_v2 import mcp


def _keep_the_loop_busy(seconds: float) -> None:
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        pass


@asynccontextmanager
async def _blocking_startup() -> AsyncGenerator[None, None]:
    # Startup awaits I/O, letting other tasks run, before it blocks the loop.
    await asyncio.sleep(0.05)
    _keep_the_loop_busy(0.3)
    yield


@asynccontextmanager
async def _no_op_lifespan(_application: FastAPI) -> AsyncGenerator[None, None]:
    yield


@pytest.mark.asyncio
async def test_blocking_startup_work_is_not_reported_as_lag(monkeypatch):
    lag = MagicMock()
    metrics_factory = MagicMock(spec=MetricsFactory)
    metrics_factory.get_histogram.return_value = lag

    async def get_metrics_factory(_name: str) -> MetricsFactory:
        return metrics_factory

    monkeypatch.setattr(
        mcp.ResourceManagerImpl, "get_metrics_factory", get_metrics_factory
    )
    monkeypatch.setattr(mcp, "global_memory_lifespan", _blocking_startup)
    monkeypatch.setattr(mcp, "mcp_app", MagicMock(lifespan=_no_op_lifespan))
    monkeypatch.setattr(mcp, "install_sigusr1_handler", lambda: None)
    monkeypatch.setattr(mcp, "dump_traceback", lambda: None)

    async with mcp.mcp_http_lifespan(FastAPI()):
        await asyncio.sleep(0.25)

    observed = [observation.args[0] for observation in lag.observe.call_args_list]
    assert observed
    assert max(observed) < 0.1
