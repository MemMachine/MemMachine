"""
STDIO entrypoint for running the MCP server via FastMCP.

This module configures logging and environment to ensure MCP STDIO protocol
compliance (stdout must contain only JSON-RPC messages).

"""

import asyncio
import logging
import os
import sys

# Signal MCP stdio mode for other modules
os.environ["MCP_STDIO_MODE"] = "1"

# Disable Rich console output to prevent stdout pollution
os.environ["NO_COLOR"] = "1"
os.environ["TERM"] = "dumb"

# Configure logging to stderr to keep stdout clean for JSON-RPC only
logging.basicConfig(
    stream=sys.stderr,
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    force=True,
)

from memmachine.server.api_v2.mcp import global_memory_lifespan  # noqa: E402
from memmachine.server.app import mcp  # noqa: E402

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
            # Disable banner to prevent stdout pollution in MCP STDIO mode
            await mcp.run_async(show_banner=False)
    except Exception:
        logger.exception("MemMachine MCP server crashed")
    finally:
        logger.info("MemMachine MCP server stopped")


if __name__ == "__main__":
    main()
