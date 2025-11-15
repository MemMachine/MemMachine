import asyncio

from fastmcp import FastMCP

from server.middleware import MCPAuthMiddleware

from .memory import memory_mcp

mcp = FastMCP(name="MemMachineExtensionMCP")

mcp.mount(memory_mcp)

mcp_app = mcp.http_app(path="/mcp")

mcp_app.add_middleware(MCPAuthMiddleware)

