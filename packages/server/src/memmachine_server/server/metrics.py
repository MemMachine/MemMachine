"""Prometheus metrics output shared by the server's worker processes."""

import contextlib
import os
from pathlib import Path

import psutil
from prometheus_client import (
    REGISTRY,
    CollectorRegistry,
    multiprocess,
    start_http_server,
)


def start_metrics_server(*, host: str, port: int) -> None:
    """Serve scrapes of `metrics_registry()` from a daemon thread.

    The thread answers while the event loop is busy, unless native code holds
    the GIL. With PROMETHEUS_MULTIPROC_DIR set, it serves every worker's
    metrics from any process.
    """
    start_http_server(port, addr=host, registry=metrics_registry())


def metrics_registry() -> CollectorRegistry:
    """Return the registry a scrape reads.

    Each worker process keeps its own registry, so a scrape answered from one
    of them would report that worker alone. With PROMETHEUS_MULTIPROC_DIR set,
    every process writes its metrics to that directory, and the returned
    registry merges them.
    """
    if not os.environ.get("PROMETHEUS_MULTIPROC_DIR"):
        return REGISTRY
    registry = CollectorRegistry()
    multiprocess.MultiProcessCollector(registry)
    return registry


def remove_exited_process_gauges() -> None:
    """Stop summing the gauge values of processes that have exited.

    A gauge summed across processes keeps counting a process's last values
    until they are removed from PROMETHEUS_MULTIPROC_DIR. Counters and
    histograms keep an exited process's totals, which their sums need.
    No-op when the variable is unset.
    """
    directory = os.environ.get("PROMETHEUS_MULTIPROC_DIR")
    if not directory:
        return
    # Every metric file is named for the process that writes it: <kind>_<pid>.db.
    pids = {int(path.stem.rpartition("_")[2]) for path in Path(directory).glob("*.db")}
    for pid in pids:
        if not psutil.pid_exists(pid):
            # Two processes starting together can remove the same file.
            with contextlib.suppress(FileNotFoundError):
                multiprocess.mark_process_dead(pid, directory)
