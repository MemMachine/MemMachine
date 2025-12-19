"""Implement the middlewares for the API server."""

import logging
import time

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.types import ASGIApp

from memmachine.common.configuration.mixin_confs import (
    WithMetricsFactory,
)

logger = logging.getLogger(__name__)


class AccessLogMiddleware(BaseHTTPMiddleware):
    """Middleware to log access details for each HTTP request."""

    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        """Log access details for the HTTP request."""
        start_time = time.perf_counter()

        response = await call_next(request)

        duration_ms = (time.perf_counter() - start_time) * 1000
        size = response.headers.get("content-length", "-")
        client = request.client.host if request.client else "-"

        logger.info(
            'client=%s method=%s path="%s" status=%d duration_ms=%.2f size=%s',
            client,
            request.method,
            request.url.path,
            response.status_code,
            duration_ms,
            size,
        )

        return response


class RequestMetricsMiddleware(BaseHTTPMiddleware, WithMetricsFactory):
    """Middleware to record metrics for each HTTP request."""

    def __init__(self, app: ASGIApp) -> None:
        """Initialize the middleware and metrics."""
        self.metrics_factory_id: str | None = None
        super().__init__(app)
        metrics_factory = self.get_metrics_factory()

        self._request_counter = metrics_factory.get_counter(
            "http_requests_total",
            "Total number of HTTP requests",
            label_names=("method", "path", "status"),
        )

        self._request_duration = metrics_factory.get_histogram(
            "http_request_duration_seconds",
            "Duration of HTTP requests in seconds",
            label_names=("method", "path", "status"),
        )

    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        """Record metrics for the HTTP request."""
        method = request.method
        start = time.perf_counter()
        status = "500"

        try:
            response = await call_next(request)
            status = str(response.status_code)
            return response
        finally:
            duration = time.perf_counter() - start

            route = request.scope.get("route")
            path = "unknown" if route is None else getattr(route, "path", "unknown")

            labels = {
                "method": method,
                "path": path,
                "status": status,
            }

            self._request_counter.increment(1, labels=labels)
            self._request_duration.observe(duration, labels=labels)
