import logging

import requests
from fastapi import Request
from fastmcp import FastMCP
from fastmcp.server.dependencies import get_http_request
from pydantic import Field

from ..core.formatter import (
    _convert_episodic_memories_to_markdown,
    _convert_profile_memories_to_markdown,
    _format_search_results,
)
from ..core.handlers import (
    _fetch_episodic_memory,
    _fetch_profile_memory,
    _handle_add_memory,
    _handle_delete_episodic_memory,
    _handle_search_memory,
)
from ..core.mm_client import memmachine_client

logger = logging.getLogger(__name__)

memory_mcp = FastMCP(name="MemoryMCP")


@memory_mcp.tool()
async def add_memory(
    content: str = Field(..., description="The content to store in memory"),
) -> str:
    """
    Add a new memory. This method is called everytime the user informs anything about themselves,
    their preferences, or anything that has any relevant information which can be useful in the
    future conversation. This can also be called when the user asks you to remember something.

    Args:
        content: The content to store in memory

    Returns:
        Simple confirmation message string
    """
    try:
        request: Request = get_http_request()
        # Get current user from request state
        current_user = getattr(request.state, "current_user", None)
        if not current_user:
            return "Error: No authenticated user found"

        logger.info(f"Adding memory - Content: {content[:100]}...")

        # Create session and episode via unified handler
        await _handle_add_memory(
            user=current_user,
            content=content,
            memmachine_client=memmachine_client,
        )
        logger.info("Successfully added memory")
        return "Memory saved successfully"

    except requests.exceptions.RequestException as e:
        logger.error(f"HTTP error adding memory: {e}")
        return f"Failed to add memory: {str(e)}"
    except Exception as e:
        logger.error(f"Error adding memory: {e}")
        return f"Internal error: {str(e)}"


@memory_mcp.tool()
async def search_memory(
    query: str = Field(..., description="The raw content that user input"),
    limit: int = Field(
        ..., description="Maximum number of results to return (recommended: 10)"
    ),
) -> str:
    """
    Search for memories in MemMachine. This function should be invoked to find
    relevant context, previous conversations, user preferences, or important
    information stored in memory that can inform the response to the current
    user query.

    Args:
        query: The raw content that user input
        limit: Maximum number of results to return (recommended: 10)

    Returns:
        Formatted text string in simple markdown format:

        Example:
        # Memory Search Results

        ## Episodic Memories
        - User prefers functional components in React
        - User is working with Next.js 14 and Tailwind CSS

        ## Profile Memories
        - User prefers TypeScript over JavaScript
    """
    try:
        # Get current user from request state
        request: Request = get_http_request()
        # Get current user from request state
        current_user = getattr(request.state, "current_user", None)
        if not current_user:
            return "Error: No authenticated user found"

        logger.info(f"Searching memory - Query: {query[:100]}...")
        logger.info(f"User: {current_user.username}, Limit: {limit}")

        # Unified search handler
        result = await _handle_search_memory(
            user=current_user,
            query=query,
            limit=limit,
            memmachine_client=memmachine_client,
        )

        # Return only formatted results for MCP tool (LLM-friendly format)
        results = result.get("results", {})
        episodic_memory = results.get("episodic_memory", [])
        profile_memory = results.get("profile_memory", [])
        formatted_results = _format_search_results(
            episodic_memory=episodic_memory, profile_memory=profile_memory
        )

        logger.info("Search completed via unified handler")
        return formatted_results

    except requests.exceptions.RequestException as e:
        logger.error(f"HTTP error searching memory: {e}")
        return f"Error searching memory: {str(e)}"
    except Exception as e:
        logger.error(f"Error searching memory: {e}")
        return f"Internal error: {str(e)}"


@memory_mcp.tool()
async def delete_episodic_memory() -> str:
    """
    Delete all memories for the current episodic.

    Returns:
        Simple confirmation message string
    """
    try:
        # Get current user from request state
        request: Request = get_http_request()
        # Get current user from request state
        current_user = getattr(request.state, "current_user", None)
        if not current_user:
            return "Error: No authenticated user found"

        logger.info(f"Deleting episodic memory - User: {current_user.username}")

        await _handle_delete_episodic_memory(
            user=current_user,
            memmachine_client=memmachine_client,
        )
        logger.info("Successfully deleted episodic memory")
        return "Successfully deleted all memories for episodic memory"

    except requests.exceptions.RequestException as e:
        logger.error(f"HTTP error deleting episodic memory: {e}")
        return f"Failed to delete episodic memory: {str(e)}"
    except Exception as e:
        logger.error(f"Error deleting episodic memory: {e}")
        return f"Internal error: {str(e)}"


@memory_mcp.tool()
async def get_profile_memory(
    limit: int = Field(..., description="Maximum number of profile memories to return"),
) -> str:
    """
    Get the profile memory for the current user.

    This function retrieves user profile information stored in MemMachine,
    including user preferences, facts, and personalized data that persists
    across multiple sessions and interactions.

    Args:
        limit: Maximum number of profile memories to return (default: 10)

    Returns:
        Formatted text string with profile memories in simple markdown format:

        Example:
        # Profile Memories

        - User prefers TypeScript over JavaScript
        - Uses pnpm for package management
    """
    try:
        # Get current user from request state
        request: Request = get_http_request()
        # Get current user from request state
        current_user = getattr(request.state, "current_user", None)
        if not current_user:
            return "Error: No authenticated user found"

        logger.info(
            f"Retrieving profile memory - User: {current_user.username}, Limit: {limit}"
        )

        result = await _fetch_profile_memory(
            user=current_user,
            limit=limit,
            memmachine_client=memmachine_client,
        )

        # Extract and format profile memories
        profile_memory = result.get("profile_memory", [])
        formatted_result = _convert_profile_memories_to_markdown(profile_memory)

        logger.info("Profile memory retrieval completed")
        return formatted_result

    except requests.exceptions.RequestException as e:
        logger.error(f"HTTP error retrieving profile memory: {e}")
        return f"Failed to retrieve profile memory: {str(e)}"
    except Exception as e:
        logger.error(f"Error retrieving profile memory: {e}")
        return f"Internal error: {str(e)}"


@memory_mcp.tool()
async def get_episodic_memory(
    limit: int = Field(
        ..., description="Maximum number of episodic memories to return"
    ),
) -> str:
    """
    Get the episodic memory for the current user.

    This function retrieves conversation episodes and contextual memories
    stored in MemMachine, including recent interactions, conversation history,
    and session-specific context that helps maintain continuity across
    multiple interactions.

    Args:
        limit: Maximum number of episodic memories to return (default: 10)

    Returns:
        Formatted text string with episodic memories in simple markdown format:

        Example:
        # Episodic Memories

        - User asked about authentication implementation
        - Discussed React component patterns
    """
    try:
        request: Request = get_http_request()
        # Get current user from request state
        current_user = getattr(request.state, "current_user", None)
        if not current_user:
            return "Error: No authenticated user found"

        logger.info(
            f"Retrieving episodic memory - User: {current_user.username}, Limit: {limit}"
        )

        result = await _fetch_episodic_memory(
            user=current_user,
            limit=limit,
            memmachine_client=memmachine_client,
        )

        # Extract and format episodic memories
        episodic_memory = result.get("episodic_memory", [])
        formatted_result = _convert_episodic_memories_to_markdown(episodic_memory)

        logger.info("Episodic memory retrieval completed")
        return formatted_result

    except requests.exceptions.RequestException as e:
        logger.error(f"HTTP error retrieving episodic memory: {e}")
        return f"Failed to retrieve episodic memory: {str(e)}"
    except Exception as e:
        logger.error(f"Error retrieving episodic memory: {e}")
        return f"Internal error: {str(e)}"
