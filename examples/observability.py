import os
import time
import functools
import inspect
from typing import Callable, Any, Optional, Awaitable

from starlette.middleware.base import BaseHTTPMiddleware
from fastapi import Request, Response
from fastapi.responses import JSONResponse

from jaeger_client import Config
import opentracing
from opentracing import Format, tags

from prometheus_client import Counter, Histogram, make_asgi_app

__all__ = [
    "init_tracer",
    "TracingMiddleware",
    "traced_endpoint",
    "metrics_app",
]

# ---------- Prometheus metrics ----------
REQUEST_COUNT = Counter(
    "http_requests_total",
    "Total number of HTTP requests",
    ["method", "endpoint", "status"],
)

REQUEST_EXCEPTIONS = Counter(
    "http_request_exceptions_total",
    "Total number of exceptions raised while handling requests",
    ["endpoint", "exception_type"],
)

REQUEST_LATENCY = Histogram(
    "http_request_duration_seconds",
    "Request latency in seconds",
    ["method", "endpoint"],
)

# Exposed ASGI app you can mount at /metrics
metrics_app = make_asgi_app()


# ---------- Jaeger tracer ----------
def init_tracer(
    service_name: str = "example-fastapi",
    jaeger_agent_host: Optional[str] = None,
    jaeger_agent_port: Optional[int] = None,
    sampler_type: str = "const",
    sampler_param: float = 1.0,
    log_spans: bool = True,
):
    """
    Initialize a Jaeger tracer and set it as the global OpenTracing tracer.
    Reads defaults from env vars if not provided:
      JAEGER_HOST, JAEGER_PORT
    """
    jaeger_agent_host = jaeger_agent_host or os.getenv("JAEGER_HOST", "jaeger")
    jaeger_agent_port = int(jaeger_agent_port or os.getenv("JAEGER_PORT", "6831"))
    print(f"Initializing Jaeger tracer with agent at {jaeger_agent_host}:{jaeger_agent_port}")
    config = Config(
        config={
            "sampler": {"type": "const", "param": 1},
            "logging": True,
            "local_agent": {
                "reporting_host": jaeger_agent_host,
                "reporting_port": jaeger_agent_port,
            },
        },
        service_name=service_name,
        validate=True,
    )
    tracer = config.initialize_tracer()  # sets opentracing.global_tracer()
    print(f"Global tracer set: {opentracing.global_tracer()}")
    with tracer.start_active_span("tracer_initialized") as scope:
        span = scope.span
        span.log_kv({"event": "tracer initialized", "service_name": service_name})
        span.finish()
        scope.close()
    return tracer


# ---------- Tracing middleware ----------
class TracingMiddleware(BaseHTTPMiddleware):
    """
    - Extracts parent span context from incoming headers
    - Starts a server span covering the whole request
    - Records standard OpenTracing HTTP tags
    """

    def __init__(self, app, tracer: opentracing.Tracer):
        super().__init__(app)
        self.tracer = tracer

    async def dispatch(self, request: Request, call_next: Callable[[Request], Awaitable[Response]]):
        route = request.scope.get("route")
        endpoint = getattr(route, "path", None) or request.url.path
        method = request.method

        # Extract parent context (if headers present)
        parent_ctx = None
        try:
            headers = {k: v for k, v in request.headers.items()}
            parent_ctx = self.tracer.extract(Format.HTTP_HEADERS, headers)
        except Exception:
            parent_ctx = None

        scope = self.tracer.start_active_span(
            operation_name=f"{method} {endpoint}",
            child_of=parent_ctx,
            finish_on_close=True,
        )
        span = scope.span
        span.set_tag(tags.SPAN_KIND, tags.SPAN_KIND_RPC_SERVER)
        span.set_tag(tags.COMPONENT, "fastapi")
        span.set_tag(tags.HTTP_METHOD, method)
        span.set_tag(tags.HTTP_URL, endpoint)

        start_time = time.perf_counter()

        try:
            response = await call_next(request)
            span.set_tag(tags.HTTP_STATUS_CODE, response.status_code)
            REQUEST_COUNT.labels(method=method, endpoint=endpoint, status=str(response.status_code)).inc()
            return response
        except Exception as exc:
            span.set_tag(tags.ERROR, True)
            span.log_kv({"event": "error", "error.object": exc})
            REQUEST_EXCEPTIONS.labels(endpoint=endpoint, exception_type=exc.__class__.__name__).inc()
            # Return a 500 JSON; change to "raise" if you want upstream handling
            return JSONResponse({"detail": "Internal Server Error"}, status_code=500)
        finally:
            duration = time.perf_counter() - start_time
            REQUEST_LATENCY.labels(method=method, endpoint=endpoint).observe(duration)
            scope.close()


# ---------- Decorator for endpoint spans ----------
def traced_endpoint(operation: Optional[str] = None):
    """
    Decorator that creates a child span around the endpoint handler.
    Works for both async and sync endpoints.

    Usage:
        @app.get("/hello")
        @traced_endpoint("say-hello")
        async def hello(): ...
    """
    def decorator(fn: Callable[..., Any]):
        is_async = inspect.iscoroutinefunction(fn)
        print(f"Decorating {'async' if is_async else 'sync'} function {fn.__name__} with tracing")
        @functools.wraps(fn)
        async def async_wrapper(*args, **kwargs):
            tracer = opentracing.global_tracer()
            print(f"Global tracer inside wrapper: {tracer}")
            op_name = operation or fn.__name__
            with tracer.start_active_span(op_name) as scope:
                span = scope.span
                span.set_tag(tags.SPAN_KIND, "internal")
                try:
                    span.log_kv({"event": "handler.start"})
                    result = await fn(*args, **kwargs)
                    span.log_kv({"event": "handler.ok"})
                    return result
                except Exception as exc:
                    span.set_tag(tags.ERROR, True)
                    span.log_kv({"event": "error", "error.object": exc})
                    raise

        @functools.wraps(fn)
        def sync_wrapper(*args, **kwargs):
            tracer = opentracing.global_tracer()
            op_name = operation or fn.__name__
            with tracer.start_active_span(op_name) as scope:
                span = scope.span
                span.set_tag(tags.SPAN_KIND, "internal")
                try:
                    span.log_kv({"event": "handler.start"})
                    result = fn(*args, **kwargs)
                    span.log_kv({"event": "handler.ok"})
                    return result
                except Exception as exc:
                    span.set_tag(tags.ERROR, True)
                    span.log_kv({"event": "error", "error.object": exc})
                    raise

        return async_wrapper if is_async else sync_wrapper

    return decorator