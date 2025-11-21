"""
Memory operation handlers for the  MemMachine Extension Server.
"""

import logging
from datetime import datetime
from typing import Any, Dict, Optional

from server.models import User

from ..schemas import DeleteRequest, MemoryEpisode, SearchQuery, SessionData
from .constants import DEFAULT_EPISODE_TYPE, DEFAULT_PRODUCED_FOR
from .formatter import (
    _flatten_memory_list,
    _format_episodic_memory,
    _format_profile_memory,
)
from .mm_client import MemMachineClient

logger = logging.getLogger(__name__)


def _is_valid_episodic_memory_item(item: Any) -> bool:
    """Check if an episodic memory item is valid.

    Returns True only if the item is a dictionary (valid memory structure).
    """
    # Only dictionaries are considered valid memory items
    return isinstance(item, dict)


def create_session_data(user_id: str) -> SessionData:
    """Create session data for MemMachine requests.

    Args:
        user_id: User identifier

    Returns:
        SessionData object
    """
    return SessionData(
        group_id=user_id,
        agent_id=[DEFAULT_PRODUCED_FOR],
        user_id=[user_id],
        session_id=user_id,
    )


async def _fetch_profile_memory(
    user: User, limit: int, memmachine_client: MemMachineClient
) -> Dict[str, Any]:
    """Fetch profile memory items for a given user and session.

    This internal helper consolidates the logic to construct the session and
    search requests and extracts the profile memory content from the result.
    """
    session_data = create_session_data(user_id=user.username)
    search_data = SearchQuery(session=session_data, query="", limit=limit, filter=None)
    result = memmachine_client.search_memory(search_data)
    content = result.get("content", {})
    profile_memory = content.get("profile_memory", [])

    logger.info(
        f"Profile memory retrieval completed - Found {len(profile_memory)} profile memories"
    )

    # Format profile memory data for better readability
    formatted_profile_memory = []
    for memory in profile_memory:
        formatted_memory = _format_profile_memory(memory)
        formatted_profile_memory.append(formatted_memory)

    return {
        "profile_memory": formatted_profile_memory,
        "total_profile_memories": len(formatted_profile_memory),
        "limit_requested": limit,
    }


async def _fetch_episodic_memory(
    user: User, limit: int, memmachine_client: MemMachineClient
) -> Dict[str, Any]:
    """Fetch episodic memory items for a given user and session.

    This internal helper consolidates the logic to construct the session and
    search requests and extracts the episodic memory content from the result.
    """
    session_data = create_session_data(user_id=user.username)
    search_data = SearchQuery(session=session_data, query="", limit=limit, filter=None)
    result = memmachine_client.search_memory(search_data)

    logger.info(
        f"Episodic memory retrieval completed - Found {len(result)} episodic memories, data: {result}"
    )

    # Extract episodic memory from the search results
    content = result.get("content", {})
    episodic_memory = content.get("episodic_memory", [])

    # First flatten the memory list to handle any nested structures
    flattened_episodic_memory = _flatten_memory_list(episodic_memory)

    # Filter out invalid episodic memory items
    filtered_episodic_memory = [
        item
        for item in flattened_episodic_memory
        if _is_valid_episodic_memory_item(item)
    ]

    # Format episodic memory data for better readability
    formatted_episodic_memory = []
    for memory in filtered_episodic_memory:
        logger.info(f"Episodic memory: {memory}")
        formatted_episodic_memory.append(_format_episodic_memory(memory))

    return {
        "episodic_memory": formatted_episodic_memory,
        "total_episodic_memories": len(formatted_episodic_memory),
        "limit_requested": limit,
    }


async def _handle_add_memory(
    user: User,
    content: str,
    memmachine_client: MemMachineClient,
    metadata: Optional[Dict[str, Any]] = None,
) -> None:
    """Handle adding a memory episode using resolved user and session ids.

    This helper centralizes the core logic so both MCP tools and REST endpoints
    can reuse it without duplicating implementation details.
    """
    session_data = create_session_data(user_id=user.username)

    metadata = metadata or {
        "speaker": user.username,
        "timestamp": datetime.now().isoformat(),
        "type": "message",
    }
    episode_data = MemoryEpisode(
        session=session_data,
        episode_content=content,
        producer=user.username,
        produced_for=DEFAULT_PRODUCED_FOR,
        episode_type=DEFAULT_EPISODE_TYPE,
        metadata=metadata,
    )
    memmachine_client.add_memory(episode_data)


async def _handle_search_memory(
    user: User,
    query: str,
    limit: int,
    memmachine_client: MemMachineClient,
) -> Dict[str, Any]:
    """Handle memory search with unified logic for both MCP and REST callers."""
    session_data = create_session_data(user_id=user.username)
    search_data = SearchQuery(
        session=session_data, query=query, limit=limit, filter=None
    )
    result = memmachine_client.search_memory(search_data)
    content = result.get("content", {})
    episodic_memory = content.get("episodic_memory", [])
    profile_memory = content.get("profile_memory", [])

    # First flatten the memory list to handle any nested structures
    flattened_episodic_memory = _flatten_memory_list(episodic_memory)

    # Filter out invalid episodic memory items
    filtered_episodic_memory = [
        item
        for item in flattened_episodic_memory
        if _is_valid_episodic_memory_item(item)
    ]

    total_results = len(filtered_episodic_memory) + len(profile_memory)

    # Format episodic memory data for better readability
    formatted_episodic_memory = []
    for memory in filtered_episodic_memory:
        formatted_episodic_memory.append(_format_episodic_memory(memory))

    formatted_profile_memory = []
    for memory in profile_memory:
        formatted_profile_memory.append(_format_profile_memory(memory))

    return {
        "results": {
            "episodic_memory": formatted_episodic_memory,
            "profile_memory": profile_memory,
        },
        "total_results": total_results,
    }


async def _handle_delete_episodic_memory(
    user: User, memmachine_client: MemMachineClient
) -> None:
    """Handle episodic memory deletion with unified logic for both MCP and REST callers."""
    session_data = create_session_data(user_id=user.username)
    delete_request = DeleteRequest(session=session_data)
    memmachine_client.delete_session_memory(delete_request)
