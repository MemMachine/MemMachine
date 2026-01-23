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
from pathlib import Path
from typing import Any, cast

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.exceptions import RequestValidationError
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.types import ExceptionHandler, Lifespan

from memmachine.common.api.version import get_version
from memmachine.server.api_v2.mcp import (
    initialize_resource,
    load_configuration,
    mcp,
    mcp_app,
    mcp_http_lifespan,
)
from memmachine.server.api_v2.router import RestError, load_v2_api_router
from memmachine.server.middleware import AccessLogMiddleware, RequestMetricsMiddleware

logger = logging.getLogger(__name__)


class MemMachineAPI(FastAPI):
    """MemMachine API wrapper."""

    def __init__(self, lifespan: Lifespan[Any] | None = None) -> None:
        """Init the MemMachine API wrapper."""
        title = "MemMachine Server"
        description = "REST API server for MemMachine memory system"
        super().__init__(
            title=title,
            description=description,
            lifespan=cast(Any, lifespan),
        )
        self._configure()

    def _configure(self) -> None:
        """Configure the exception handler and routers."""
        self.add_exception_handler(
            RequestValidationError,
            self._validation_error_handler_factory(422),
        )
        self.mount("/mcp", mcp_app)
        load_v2_api_router(self)

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


app = MemMachineAPI(lifespan=mcp_http_lifespan)
app.add_middleware(cast(type, AccessLogMiddleware))
app.add_middleware(cast(type, RequestMetricsMiddleware))


def start_http() -> None:
    """Run the FastAPI HTTP application using the uvicorn server."""
    config = load_configuration()

    # Determine number of workers.
    # Note: We do not use (os.cpu_count() - 1) as this is often inaccurate in container
    # environments (reporting host CPUs vs container limits). We leave it to the
    # user to configure MEMMACHINE_WORKERS based on their allocated vCPUs to
    # avoid creating excessive worker processes at startup.
    workers_env = os.getenv("MEMMACHINE_WORKERS")
    if workers_env:
        # Verify workers_env is a valid integer, default to 1 if invalid
        try:
            workers = int(workers_env)
        except ValueError:
            logger.warning(
                "Invalid MEMMACHINE_WORKERS value '%s'. Defaulting to 1.", workers_env
            )
            workers = 1
    else:
        # Default to 1 for predictable resource usage in containers
        workers = 1

    if workers == 1:
        logger.info("Starting server with 1 worker")
    else:
        logger.info("Starting server with %d workers", workers)

    # Use uvicorn.run() to correctly handle multiprocessing with workers
    uvicorn.run(
        "memmachine.server.app:app",
        host=config.server.host,
        port=config.server.port,
        workers=workers,
        access_log=True,
        log_level=str(config.logging.level).lower(),
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
    args = parser.parse_args()

    # Handle --version early
    if args.version:
        sys.stdout.write(f"{get_version()}\n")
        sys.exit(0)

    try:
        if args.stdio:
            # MCP stdio mode
            async def run_mcp_server() -> None:
                """Initialize resources and run MCP server in the same event loop."""
                await initialize_resource()
                await mcp.run_stdio_async()

            asyncio.run(run_mcp_server())
        else:
            # HTTP mode for REST API
            start_http()
    except KeyboardInterrupt:
        logger.warning("Application cancelled by user.")
        sys.exit(130)  # Standard exit code for Ctrl+C


if __name__ == "__main__":
    main()
