"""
Strands-compatible tool functions for MemMachine.

This module provides tool functions that can be loaded by AWS Strands Agent SDK
using module path strings like: "strands_tools:add_memory"
"""

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from tool import MemMachineTools

# Global tool instance (will be initialized by integration code)
_tools_instance: "MemMachineTools | None" = None

# Explicitly export all tool functions for Strands SDK
__all__ = ["add_memory", "get_context", "search_memory", "set_tools_instance"]


def set_tools_instance(tools: "MemMachineTools") -> None:
    """Set the global MemMachine tools instance."""
    global _tools_instance
    _tools_instance = tools


def add_memory(
    content: str,
    user_id: str | None = None,
    episode_type: str = "text",
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Store important information about the user or conversation into memory.

    Args:
        content: The content to store in memory
        user_id: Optional user ID override
        episode_type: Type of episode (default: "text")
        metadata: Additional metadata dictionary

    Returns:
        Dictionary with status, message, and content fields

    """
    if _tools_instance is None:
        return {
            "status": "error",
            "message": "Tools not initialized. Call set_tools_instance() first.",
        }
    return _tools_instance.add_memory(
        content=content,
        user_id=user_id,
        episode_type=episode_type,
        metadata=metadata,
    )


def search_memory(
    query: str,
    user_id: str | None = None,
    limit: int = 5,
) -> dict[str, Any]:
    """
    Retrieve relevant context, memories, or profile for a user.

    Args:
        query: Search query string
        user_id: Optional user ID override
        limit: Maximum number of results (default: 5, max: 20)

    Returns:
        Dictionary with status, results, and summary fields

    """
    if _tools_instance is None:
        return {
            "status": "error",
            "message": "Tools not initialized. Call set_tools_instance() first.",
        }
    return _tools_instance.search_memory(
        query=query,
        user_id=user_id,
        limit=limit,
    )


def get_context(user_id: str | None = None) -> dict[str, Any]:
    """
    Get the current memory context configuration.

    Args:
        user_id: Optional user ID override

    Returns:
        Dictionary containing group_id, agent_id, user_id, and session_id

    """
    if _tools_instance is None:
        return {
            "status": "error",
            "message": "Tools not initialized. Call set_tools_instance() first.",
        }
    return _tools_instance.get_context(user_id=user_id)
