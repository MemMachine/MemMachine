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

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI

from memmachine.server.api_v2.mcp import (
    initialize_resource,
    mcp,
    mcp_app,
    mcp_http_lifespan,
)
from memmachine.server.api_v2.router import load_v2_api_router

logger = logging.getLogger(__name__)


app = FastAPI(lifespan=mcp_http_lifespan)
app.mount("/mcp", mcp_app)


async def start() -> None:
    """Run the FastAPI application using uvicorn server."""
    port_num = os.getenv("PORT", "8080")
    host_name = os.getenv("HOST", "0.0.0.0")

    load_v2_api_router(app)

    await uvicorn.Server(
        uvicorn.Config(app, host=host_name, port=int(port_num)),
    ).serve()


def main() -> None:
    """Execute the CLI entry point for the application."""
    log_level = os.getenv("LOG_LEVEL", "INFO").upper()
    log_format = os.getenv("LOG_FORMAT", "%(levelname)-7s %(message)s")
    logging.basicConfig(
        level=log_level,
        format=log_format,
    )
    # Load environment variables from .env file
    load_dotenv()

    # Parse command line arguments
    parser = argparse.ArgumentParser(description="MemMachine server")
    parser.add_argument(
        "--stdio",
        action="store_true",
        help="Run in MCP stdio mode",
    )
    args = parser.parse_args()

    if args.stdio:
        # MCP stdio mode
        config_file = os.getenv("MEMORY_CONFIG", "configuration.yml")

        async def run_mcp_server() -> None:
            """Initialize resources and run MCP server in the same event loop."""
            await initialize_resource(config_file)
            await mcp.run_stdio_async()

        asyncio.run(run_mcp_server())
    else:
        # HTTP mode for REST API
        asyncio.run(start())


if __name__ == "__main__":
    main()
