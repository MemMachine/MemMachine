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
import sys
import time
from collections.abc import Callable
from pathlib import Path

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.exceptions import RequestValidationError
from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from starlette.types import AppType, ExceptionHandler, Lifespan

from memmachine.server.api_v2.mcp import (
    initialize_resource,
    load_configuration,
    mcp,
    mcp_app,
    mcp_http_lifespan,
)
from memmachine.server.api_v2.router import RestError, load_v2_api_router

logger = logging.getLogger(__name__)


class MemMachineAPI(FastAPI):
    """MemMachine API wrapper."""

    def __init__(self, lifespan: Lifespan[AppType] | None = None) -> None:
        """Init the MemMachine API wrapper."""
        title = "MemMachine Server"
        description = "REST API server for MemMachine memory system"
        super().__init__(title=title, description=description, lifespan=lifespan)
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

        async def handler(_: Request, exc: RequestValidationError) -> JSONResponse:
            err = RestError(
                code=error_code,
                message="Invalid request payload",
                ex=exc,
            )
            content = None
            if err.payload is not None:
                content = {"detail": err.payload.model_dump()}
            return JSONResponse(status_code=error_code, content=content)

        return handler


app = MemMachineAPI(
    lifespan=mcp_http_lifespan,
)
app.mount("/mcp", mcp_app)


@app.middleware("http")
async def access_log_middleware(request: Request, call_next: Callable) -> Response:
    """Middleware to log access details for each HTTP request."""
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


async def start() -> None:
    """Run the FastAPI application using uvicorn server."""
    config = load_configuration()
    await uvicorn.Server(
        uvicorn.Config(
            app,
            host=config.server.host,
            port=config.server.port,
            access_log=True,
            log_level=str(config.logging.level).lower(),
        ),
    ).serve()


def main() -> None:
    """Execute the CLI entry point for the application."""
    # Load environment variables from .env file
    conf_env_file = str(Path("~/.config/memmachine/.env").expanduser())
    if Path(conf_env_file).is_file():
        load_dotenv(conf_env_file)
    else:
        load_dotenv()

    # Parse command line arguments
    parser = argparse.ArgumentParser(description="MemMachine server")
    parser.add_argument(
        "--stdio",
        action="store_true",
        help="Run in MCP stdio mode",
    )
    args = parser.parse_args()

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
            asyncio.run(start())
    except KeyboardInterrupt:
        logger.warning("Application cancelled by user.")
        sys.exit(130)  # Standard exit code for Ctrl+C


if __name__ == "__main__":
    main()
