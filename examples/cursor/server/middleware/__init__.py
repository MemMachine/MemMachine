"""
Middlewares for the MemMachine Extension MCP Server.
"""

from .logging import LoggingMiddleware
from .mcp_auth import MCPAuthMiddleware

__all__ = ["MCPAuthMiddleware", "LoggingMiddleware"]
