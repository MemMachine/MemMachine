"""
MCP tool functions for the Cursor MCP Server.
"""

import logging

import requests
from fastapi import Request
from fastmcp.server.dependencies import get_http_request
from pydantic import Field

from server.auth import TokenStore

from .client import MemMachineClient
from .handlers import (
    _fetch_episodic_memory,
    _fetch_profile_memory,
    _format_episodic_memories,
    _format_profile_memories,
    _format_search_results,
    _handle_add_memory,
    _handle_delete_session_memory,
    _handle_search_memory,
)

logger = logging.getLogger(__name__)


def register_mcp_tools(
    mcp, memmachine_client: MemMachineClient, token_store: TokenStore
):
    """Register all MCP tools with the FastMCP instance."""

    @mcp.tool()
    async def mcp_add_memory(
        content: str = Field(..., description="The content to store in memory"),
        session_id: str = Field(
            ...,
            description="Memory session identifier. In IDE environment: use 'PROJECT-${workspacename}' format for current workspace, or specific project name when recalling memories from other projects.",
        ),
    ) -> str:
        """
        Add a new memory. This method is called everytime the user informs anything about themselves, their preferences, or anything that has any relevant information which can be useful in the future conversation. This can also be called when the user asks you to remember something.

        Args:
            content: The content to store in memory
            session_id: Memory session identifier in the format 'PROJECT-${workspacename}'

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
                session_id=session_id,
                content=content,
                memmachine_client=memmachine_client,
            )
            logger.info(f"Successfully added memory for user {current_user.username}")
            return f"Memory saved successfully for session {session_id}"

        except requests.exceptions.RequestException as e:
            logger.error(f"HTTP error adding memory: {e}")
            return f"Failed to add memory: {str(e)}"
        except Exception as e:
            logger.error(f"Error adding memory: {e}")
            return f"Internal error: {str(e)}"

    @mcp.tool()
    async def mcp_search_memory(
        query: str = Field(..., description="The raw content that user input"),
        limit: int = Field(
            ..., description="Maximum number of results to return (recommended: 10)"
        ),
        session_id: str = Field(
            ...,
            description="Memory session identifier. In IDE environment: use 'PROJECT-${workspacename}' format for current workspace, or specific project name when recalling memories from other projects.",
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
            session_id: Memory session identifier in the format 'PROJECT-${workspacename}'

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
            logger.info(f"User: {current_user.username}, Session ID: {session_id}, Limit: {limit}")

            # Unified search handler
            result = await _handle_search_memory(
                user=current_user,
                session_id=session_id,
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

    @mcp.tool()
    async def mcp_delete_session_memory(
        session_id: str = Field(
            ...,
            description="Memory session identifier. In IDE environment: use 'PROJECT-${workspacename}' format for current workspace, or specific project name when recalling memories from other projects.",
        ),
    ) -> str:
        """
        Delete all memories for the current session.

        Args:
            session_id: Memory session identifier in the format 'PROJECT-${workspacename}'

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

            logger.info(
                f"Deleting session memory - User: {current_user.username}, Session ID: {session_id}"
            )

            await _handle_delete_session_memory(
                user=current_user,
                session_id=session_id,
                memmachine_client=memmachine_client,
            )
            logger.info(f"Successfully deleted session {session_id} for user {current_user.username}")
            return f"Successfully deleted all memories for session {session_id}"

        except requests.exceptions.RequestException as e:
            logger.error(f"HTTP error deleting session: {e}")
            return f"Failed to delete session: {str(e)}"
        except Exception as e:
            logger.error(f"Error deleting session: {e}")
            return f"Internal error: {str(e)}"

    @mcp.tool()
    async def mcp_get_profile_memory(
        limit: int = Field(
            ..., description="Maximum number of profile memories to return"
        ),
        session_id: str = Field(
            ...,
            description="Memory session identifier. In IDE environment: use 'PROJECT-${workspacename}' format for current workspace, or specific project name when recalling memories from other projects.",
        ),
    ) -> str:
        """
        Get the profile memory for the current session.

        This function retrieves user profile information stored in MemMachine,
        including user preferences, facts, and personalized data that persists
        across multiple sessions and interactions.

        Args:
            limit: Maximum number of profile memories to return (default: 10)
            session_id: Memory session identifier in the format 'PROJECT-${workspacename}'

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
                f"Retrieving profile memory - User: {current_user.username}, Session ID: {session_id}, Limit: {limit}"
            )

            result = await _fetch_profile_memory(
                user=current_user,
                session_id=session_id,
                limit=limit,
                memmachine_client=memmachine_client,
            )

            # Extract and format profile memories
            profile_memory = result.get("profile_memory", [])
            formatted_result = _format_profile_memories(profile_memory)

            logger.info("Profile memory retrieval completed")
            return formatted_result

        except requests.exceptions.RequestException as e:
            logger.error(f"HTTP error retrieving profile memory: {e}")
            return f"Failed to retrieve profile memory: {str(e)}"
        except Exception as e:
            logger.error(f"Error retrieving profile memory: {e}")
            return f"Internal error: {str(e)}"

    @mcp.tool()
    async def mcp_get_episodic_memory(
        limit: int = Field(
            ..., description="Maximum number of episodic memories to return"
        ),
        session_id: str = Field(
            ...,
            description="Memory session identifier. In IDE environment: use 'PROJECT-${workspacename}' format for current workspace, or specific project name when recalling memories from other projects.",
        ),
    ) -> str:
        """
        Get the episodic memory for the current session.

        This function retrieves conversation episodes and contextual memories
        stored in MemMachine, including recent interactions, conversation history,
        and session-specific context that helps maintain continuity across
        multiple interactions.

        Args:
            limit: Maximum number of episodic memories to return (default: 10)
            session_id: Memory session identifier in the format 'PROJECT-${workspacename}'

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
                f"Retrieving episodic memory - User: {current_user.username}, Session ID: {session_id}, Limit: {limit}"
            )

            result = await _fetch_episodic_memory(
                user=current_user,
                session_id=session_id,
                limit=limit,
                memmachine_client=memmachine_client,
            )

            # Extract and format episodic memories
            episodic_memory = result.get("episodic_memory", [])
            formatted_result = _format_episodic_memories(episodic_memory)

            logger.info("Episodic memory retrieval completed")
            return formatted_result

        except requests.exceptions.RequestException as e:
            logger.error(f"HTTP error retrieving episodic memory: {e}")
            return f"Failed to retrieve episodic memory: {str(e)}"
        except Exception as e:
            logger.error(f"Error retrieving episodic memory: {e}")
            return f"Internal error: {str(e)}"
