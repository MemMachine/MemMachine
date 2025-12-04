"""
STDIO entrypoint for running the MCP server via FastMCP.

This module performs critical stdout redirection before any imports to ensure
MCP STDIO protocol compliance (stdout must contain only JSON-RPC messages).

"""

import asyncio
import logging
import os
import sys

# MCP STDIO protocol requires stdout to contain only JSON-RPC messages.
# Redirect stdout to stderr at the OS file descriptor level to prevent
# FastMCP's Rich Console and other libraries from polluting stdout.
# This must be done before any imports that might write to stdout.
os.dup2(2, 1)  # Redirect FD 1 (stdout) → 2 (stderr)

# Configure logging to use stderr before importing modules that may create loggers
logging.basicConfig(
    stream=sys.stderr,
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    force=True,  # Override any existing configuration
)

# Disable Rich console output to prevent stdout pollution
os.environ["NO_COLOR"] = "1"
os.environ["TERM"] = "dumb"

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
