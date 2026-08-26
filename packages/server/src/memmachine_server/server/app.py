"""
FastAPI application for the MemMachine memory system.

This module sets up and runs a FastAPI web server that provides endpoints for
interacting with the Profile Memory and Episodic Memory components.
It includes:
- API endpoints for adding and searching memories.
- Integration with FastMCP for exposing memory functions as tools to LLMs.
- Pydantic models for request and response validation.
- Lifespan management for initializing and cleaning up resources like database
  connections and memory managers.
"""

import argparse
import asyncio
import logging
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, cast

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.exceptions import RequestValidationError
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.types import ExceptionHandler, Lifespan

from memmachine_server.common.api.version import get_version
from memmachine_server.server.api_v2.mcp import (
    initialize_resource,
    load_configuration,
    mcp,
    mcp_app,
    mcp_http_lifespan,
)
from memmachine_server.server.api_v2.router import RestError, load_v2_api_router
from memmachine_server.server.diagnostics import dump_traceback, install_sigusr1_handler
from memmachine_server.server.middleware import (
    AccessLogMiddleware,
    RequestMetricsMiddleware,
)

logger = logging.getLogger(__name__)


class MemMachineAPI(FastAPI):
    """MemMachine API wrapper."""

    def __init__(
        self, lifespan: Lifespan[Any] | None = None, with_config_api: bool = False
    ) -> None:
        """Init the MemMachine API wrapper."""
        title = "MemMachine Server"
        description = "REST API server for MemMachine memory system"
        super().__init__(
            title=title,
            description=description,
            lifespan=cast(Any, lifespan),
        )
        self._with_config_api = with_config_api
        self._configure()

    def _configure(self) -> None:
        """Configure the exception handler and routers."""
        self.add_exception_handler(
            RequestValidationError,
            self._validation_error_handler_factory(422),
        )
        self.mount("/mcp", mcp_app)
        load_v2_api_router(self, with_config_api=self._with_config_api)

    @staticmethod
    def _validation_error_handler_factory(error_code: int) -> ExceptionHandler:
        """Create an error handler factory for the validation error."""

        async def handler(_: Request, exc: Exception) -> JSONResponse:
            err = RestError(
                code=error_code,
                message="Invalid request payload",
                ex=cast(RequestValidationError, exc),
            )
            content = None
            if err.payload is not None:
                content = {"detail": err.payload.model_dump()}
            return JSONResponse(status_code=error_code, content=content)

        return cast(ExceptionHandler, handler)


app = MemMachineAPI(
    lifespan=mcp_http_lifespan,
    with_config_api=bool(os.getenv("MEMMACHINE_CONFIG_API")),
)
app.add_middleware(cast(type, AccessLogMiddleware))
app.add_middleware(cast(type, RequestMetricsMiddleware))


def start_http() -> None:
    """Run the FastAPI HTTP application using the uvicorn server."""
    # For the single-worker case, the module-level `app` was created before
    # main() set the env var. Include the config router explicitly here.
    if os.getenv("MEMMACHINE_CONFIG_API"):
        from memmachine_server.server.api_v2.config_router import config_router

        app.include_router(config_router, prefix="/api/v2")

    config = load_configuration()

    workers = _worker_count()

    if workers == 1:
        logger.info("Starting server with 1 worker")
    else:
        logger.info("Starting server with %d workers", workers)

    # Use uvicorn.run() to correctly handle multiprocessing with workers
    uvicorn.run(
        "memmachine_server.server.app:app",
        host=config.server.host,
        port=config.server.port,
        workers=workers,
        access_log=True,
        log_level=str(config.logging.level).lower(),
        ws="websockets-sansio",
    )


def _worker_count() -> int:
    """Resolve MEMMACHINE_WORKERS, defaulting to 1.

    Note: We do not use (os.cpu_count() - 1) as this is often inaccurate in
    container environments (reporting host CPUs vs container limits). We leave
    it to the user to configure MEMMACHINE_WORKERS based on their allocated
    vCPUs to avoid creating excessive worker processes at startup.
    """
    workers_env = os.getenv("MEMMACHINE_WORKERS")
    if not workers_env:
        return 1
    try:
        return int(workers_env)
    except ValueError:
        logger.warning(
            "Invalid MEMMACHINE_WORKERS value '%s'. Defaulting to 1.", workers_env
        )
        return 1


def _prepare_multiproc_dir() -> None:
    """Make PROMETHEUS_MULTIPROC_DIR usable before any metric is created.

    prometheus_client picks its value class at import time and each worker
    mmaps a file into this directory as soon as it registers a metric, so the
    directory has to exist by then - after that it is too late, and the failure
    surfaces as an unrelated error deep in a worker.

    Stale files are cleared for the same reason: they belong to workers from a
    previous run of this process, and MultiProcessCollector would otherwise add
    their dead counters to the live ones on every scrape.

    Deliberately best-effort. A deployment may mount the directory read-only or
    pre-seed it, and refusing to start over metrics would be the wrong trade.

    With more than one worker and no directory configured, one is chosen here
    rather than left unset: without it every worker keeps its own unaggregated
    registry and a scrape returns whichever worker answered, which reads as a
    plausible number rather than an error. The path is fixed so all workers
    share it; two servers on one host must set the variable themselves.
    """
    path = os.environ.get("PROMETHEUS_MULTIPROC_DIR")
    chosen = False
    if not path:
        if _worker_count() <= 1:
            return
        path = str(Path(tempfile.gettempdir()) / "memmachine-prometheus-multiproc")
        os.environ["PROMETHEUS_MULTIPROC_DIR"] = path
        chosen = True
        logger.info(
            "MEMMACHINE_WORKERS is above 1 and PROMETHEUS_MULTIPROC_DIR is "
            "unset; using %s so per-worker metrics are aggregated",
            path,
        )
    try:
        directory = Path(path)
        directory.mkdir(parents=True, exist_ok=True)
        for stale in directory.glob("*.db"):
            stale.unlink()
    except OSError:
        if chosen:
            # prometheus_client raises in every worker if the directory it is
            # pointed at cannot be opened, so withdraw a choice we made rather
            # than take the process down with it.
            del os.environ["PROMETHEUS_MULTIPROC_DIR"]
        logger.warning(
            "PROMETHEUS_MULTIPROC_DIR=%s is not writable; per-worker metrics "
            "will not be aggregated",
            path,
            exc_info=True,
        )


def main() -> None:
    """Execute the CLI entry point for the application."""
    # Load environment variables from .env file
    conf_env_file = str(Path("~/.config/memmachine/.env").expanduser())
    if Path(conf_env_file).is_file():
        load_dotenv(conf_env_file)
    else:
        load_dotenv()

    # Configure basic logging to ensure we see startup messages
    logging.basicConfig(level=logging.INFO)
    logger.debug("memmachine-server entrypoint called")

    _prepare_multiproc_dir()

    # Parse command line arguments
    parser = argparse.ArgumentParser(description="MemMachine server")
    parser.add_argument(
        "--stdio",
        action="store_true",
        help="Run in MCP stdio mode",
    )
    parser.add_argument(
        "--version",
        action="store_true",
        help="Show the version and exit",
    )
    parser.add_argument(
        "--with-config-api",
        action="store_true",
        help="Enable the configuration management API endpoints",
    )
    args = parser.parse_args()

    # Handle --version early
    if args.version:
        sys.stdout.write(f"{get_version()}\n")
        sys.exit(0)

    if args.with_config_api:
        os.environ["MEMMACHINE_CONFIG_API"] = "1"

    try:
        if args.stdio:
            # MCP stdio mode
            async def run_mcp_server() -> None:
                """Initialize resources and run MCP server in the same event loop."""
                install_sigusr1_handler()
                try:
                    await initialize_resource()
                    await mcp.run_stdio_async()
                finally:
                    dump_traceback()

            asyncio.run(run_mcp_server())
        else:
            # HTTP mode for REST API
            start_http()
    except KeyboardInterrupt:
        logger.warning("Application cancelled by user.")
        sys.exit(130)  # Standard exit code for Ctrl+C


if __name__ == "__main__":
    main()
