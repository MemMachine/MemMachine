"""HTTP entrypoint for exposing the MCP server over FastAPI/uvicorn."""

import argparse
import asyncio
import ipaddress
import logging

import uvicorn

from memmachine.server.api_v2.mcp import global_memory_lifespan
from memmachine.server.app import mcp

logger = logging.getLogger(__name__)


async def run_mcp_http(host: str, port: int) -> None:
    """Run MCP server in HTTP mode."""
    try:
        logger.info(
            "Starting MemMachine MCP HTTP server at http://%s:%s",
            host,
            port,
        )
        async with global_memory_lifespan():
            await uvicorn.Server(
                uvicorn.Config(mcp.get_app(), host=host, port=int(port)),
            ).serve()
    except Exception:
        logger.exception("MemMachine MCP HTTP server crashed")
    finally:
        logger.info("MemMachine MCP HTTP server stopped")


def parse_args() -> argparse.Namespace:
    """Parse and validate command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Start MemMachine MCP server in HTTP mode.",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="localhost",
        help="Host address to bind the server to (default: localhost).",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8080,
        help="Port number to listen on (default: 8080).",
    )

    args = parser.parse_args()

    # Validate host
    try:
        # Accept both IP and hostname
        if args.host not in ("localhost",):
            ipaddress.ip_address(args.host)
    except ValueError:
        parser.error(f"Invalid host: {args.host!r}")

    # Validate port
    if not (1 <= args.port <= 65535):
        parser.error("Port must be between 1 and 65535")

    return args


def main() -> None:
    """Entry point for launching the MCP HTTP server."""
    args = parse_args()
    try:
        asyncio.run(run_mcp_http(args.host, args.port))
    except KeyboardInterrupt:
        logger.info("MemMachine MCP HTTP server stopped by user")


if __name__ == "__main__":
    main()
