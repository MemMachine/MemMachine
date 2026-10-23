"""Tests for the metrics the server's worker processes share."""

import os
import socket
import subprocess
import sys
import textwrap
import threading
import time
import urllib.request
from pathlib import Path

import pytest

from memmachine_server.server.metrics import (
    metrics_registry,
    remove_exited_process_gauges,
    start_metrics_server,
)

ENV_VAR = "PROMETHEUS_MULTIPROC_DIR"

# A worker process: sets its value of a summed gauge, reports that it has, and
# keeps running until its stdin closes.
SET_SUMMED_GAUGE = textwrap.dedent(
    """
    import sys

    from memmachine_server.common.metrics_factory import PrometheusMetricsFactory

    gauge = PrometheusMetricsFactory().get_gauge(
        "test_summed_gauge",
        "A gauge summed across processes",
        label_names=("pool_name",),
        sum_across_processes=True,
    )
    gauge.set(float(sys.argv[1]), labels={"pool_name": "pool"})
    print("set", flush=True)
    sys.stdin.read()
    """
)


def _start_worker(directory: Path, value: float) -> subprocess.Popen[str]:
    worker = subprocess.Popen(
        [sys.executable, "-c", SET_SUMMED_GAUGE, str(value)],
        env={**os.environ, ENV_VAR: str(directory)},
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        text=True,
    )
    assert worker.stdout is not None
    assert worker.stdout.readline() == "set\n"
    return worker


def _summed_gauge_value() -> float | None:
    return metrics_registry().get_sample_value(
        "test_summed_gauge", {"pool_name": "pool"}
    )


def test_a_summed_gauge_stops_counting_an_exited_worker(monkeypatch, tmp_path):
    monkeypatch.setenv(ENV_VAR, str(tmp_path))
    running = _start_worker(tmp_path, 2)
    try:
        exited = _start_worker(tmp_path, 3)
        exited.communicate()
        assert _summed_gauge_value() == 5

        remove_exited_process_gauges()

        assert _summed_gauge_value() == 2
    finally:
        running.communicate()


def test_removing_exited_gauges_without_a_directory_is_a_no_op(monkeypatch):
    monkeypatch.delenv(ENV_VAR, raising=False)
    remove_exited_process_gauges()


@pytest.mark.asyncio
async def test_the_metrics_server_answers_while_the_event_loop_is_busy(monkeypatch):
    monkeypatch.delenv(ENV_VAR, raising=False)
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    start_metrics_server(host="127.0.0.1", port=port)

    scrapes: list[bytes] = []
    scraper = threading.Thread(target=_scrape, args=(port, scrapes))
    scraper.start()
    # Hold the event loop, as a saturated worker would.
    busy_until = time.monotonic() + 1
    while time.monotonic() < busy_until:
        pass
    answered_while_busy = not scraper.is_alive()
    scraper.join()

    assert answered_while_busy
    assert b"python_info" in scrapes[0]


def _scrape(port: int, scrapes: list[bytes]) -> None:
    with urllib.request.urlopen(f"http://127.0.0.1:{port}/", timeout=5) as response:
        scrapes.append(response.read())
