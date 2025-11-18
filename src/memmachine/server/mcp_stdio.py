import asyncio
import logging

from memmachine.server.app import global_memory_lifespan, mcp

logger = logging.getLogger(__name__)


def main():
    try:
        asyncio.run(run_mcp_stdio())
    except KeyboardInterrupt:
        logger.info("MemMachine MCP server stopped by user")
    except Exception as e:
        logger.exception(f"MemMachine MCP server crashed: {e}")


async def run_mcp_stdio():
    try:
        logger.info("starting the MemMachine MCP server")
        async with global_memory_lifespan():
            await mcp.run_async()
    except Exception as e:
        logger.exception(f"MemMachine MCP server crashed: {e}")
    finally:
        logger.info("MemMachine MCP server stopped")


if __name__ == "__main__":
    main()
