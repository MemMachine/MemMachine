"""Prometheus metrics output shared by the server's worker processes."""

import os

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
