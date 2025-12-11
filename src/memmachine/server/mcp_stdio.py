"""STDIO entrypoint for running the MCP server via FastMCP."""

import asyncio
import logging

from memmachine.server.api_v2.mcp import global_memory_lifespan
from memmachine.server.app import mcp

logger = logging.getLogger(__name__)


def main() -> None:
    """Start the MCP server using asyncio."""
    try:
        asyncio.run(run_mcp_stdio())
    except KeyboardInterrupt:
        logger.info("MemMachine MCP server stopped by user")
    except Exception:
        logger.exception("MemMachine MCP server crashed")


async def run_mcp_stdio() -> None:
    """Run the MCP server over stdio, ensuring resources are cleaned up."""
    try:
        logger.info("starting the MemMachine MCP server")
        async with global_memory_lifespan():
            await mcp.run_async()
    except Exception:
        logger.exception("MemMachine MCP server crashed")
    finally:
        logger.info("MemMachine MCP server stopped")


if __name__ == "__main__":
    main()
