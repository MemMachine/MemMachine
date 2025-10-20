"""
Memory operation handlers for the Cursor MCP Server.
"""

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

from server.models import User

from .client import MemMachineClient
from .config import (
    DEFAULT_EPISODE_TYPE,
    DEFAULT_PRODUCED_FOR,
)
from .schemas import DeleteRequest, MemoryEpisode, SearchQuery, SessionData

logger = logging.getLogger(__name__)

def create_session_data(session_id: str, user_id: str) -> SessionData:
    """Create session data for MemMachine requests.

    Args:
        user_id: User identifier
        session_id: Optional session identifier

    Returns:
        SessionData object
    """
    return SessionData(
        group_id=user_id,
        agent_id=[DEFAULT_PRODUCED_FOR],
        user_id=[user_id],
        session_id=session_id,
    )

def _real_format_episodic_memory(memory: Dict[str, Any]) -> Dict[str, Any]:
    """Format episodic memory data for better readability."""
    logger.info(f"Formatting episodic memory: {memory}")
    formatted_memory = {
        "uuid": memory.get("uuid", "unknown"),
        "episode_type": memory.get("episode_type", "message"),
        "content_type": memory.get("content_type", "string"),
        "content": memory.get("content", ""),
        "timestamp": memory.get("timestamp", ""),
        "group_id": memory.get("group_id", "unknown"),
        "session_id": memory.get("session_id", "unknown"),
        "producer_id": memory.get("producer_id", "unknown"),
        "produced_for_id": memory.get("produced_for_id", "unknown"),
        "user_metadata": memory.get("user_metadata", {}),
    }
    return formatted_memory


def _format_episodic_memory(memory) -> Optional[List[Dict[str, Any]]]:
    formatted_episodic_memory = []
    if isinstance(memory, list):
        for m in memory:
            formatted_memory = _format_episodic_memory(m)
            if formatted_memory:
                formatted_episodic_memory.extend(formatted_memory)
    elif isinstance(memory, dict):
        formatted_memory = _real_format_episodic_memory(memory)
        if formatted_memory:
            formatted_episodic_memory.append(formatted_memory)
    else:
        logger.warning(f"Unknown episodic memory type: {type(memory)}")
    return formatted_episodic_memory


async def _fetch_profile_memory(
    user: User, session_id: str, limit: int, memmachine_client: MemMachineClient
) -> Dict[str, Any]:
    """Fetch profile memory items for a given user and session.

    This internal helper consolidates the logic to construct the session and
    search requests and extracts the profile memory content from the result.
    """
    session_data = create_session_data(session_id=session_id, user_id=user.username)
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
        formatted_memory = {
            "id": memory.get("metadata", {}).get("id", "unknown"),
            "similarity_score": memory.get("metadata", {}).get("similarity_score", 0.0),
            "tag": memory.get("tag", ""),
            "feature": memory.get("feature", ""),
            "value": memory.get("value", ""),
        }
        formatted_profile_memory.append(formatted_memory)

    return {
        "session_id": session_id,
        "user_id": user.username,
        "profile_memory": formatted_profile_memory,
        "total_profile_memories": len(formatted_profile_memory),
        "limit_requested": limit,
    }


async def _fetch_episodic_memory(
    user: User, session_id: str, limit: int, memmachine_client: MemMachineClient
) -> Dict[str, Any]:
    """Fetch episodic memory items for a given user and session.

    This internal helper consolidates the logic to construct the session and
    search requests and extracts the episodic memory content from the result.
    """
    session_data = create_session_data(session_id=session_id, user_id=user.username)
    search_data = SearchQuery(session=session_data, query="", limit=limit, filter=None)
    result = memmachine_client.search_memory(search_data)

    logger.info(
        f"Episodic memory retrieval completed - Found {len(result)} episodic memories, data: {result}"
    )

    # Extract episodic memory from the search results
    content = result.get("content", {})
    episodic_memory = content.get("episodic_memory", [])

    # Filter out invalid episodic memory items
    filtered_episodic_memory = [
        item for item in episodic_memory if _is_valid_episodic_memory_item(item)
    ]

    logger.info(
        f"Episodic memory retrieval completed - Found {len(filtered_episodic_memory)} valid episodic memories"
    )

    # Format episodic memory data for better readability
    formatted_episodic_memory = []
    for memory in filtered_episodic_memory:
        logger.info(f"Episodic memory: {memory}")
        formatted_memory = _format_episodic_memory(memory)
        if formatted_memory:
            formatted_episodic_memory.extend(formatted_memory)

    return {
        "session_id": session_id,
        "user_id": user.username,
        "episodic_memory": formatted_episodic_memory,
        "total_episodic_memories": len(formatted_episodic_memory),
        "limit_requested": limit,
    }


async def _handle_add_memory(
    user: User, session_id: str, content: str, memmachine_client: MemMachineClient, metadata: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Handle adding a memory episode using resolved user and session ids.

    This helper centralizes the core logic so both MCP tools and REST endpoints
    can reuse it without duplicating implementation details.
    """
    session_data = create_session_data(session_id=session_id, user_id=user.username)

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
    return {
        "session_id": session_id,
        "user_id": user.username,
    }


def _is_valid_episodic_memory_item(item: Any) -> bool:
    """Check if an episodic memory item is valid.

    Filters out:
    - Empty arrays
    - Arrays containing only empty strings or None values
    - Empty strings
    - None values
    """
    if item is None:
        return False

    # If it's a string, check if it's not empty
    if isinstance(item, str):
        return bool(item.strip())

    # If it's a list/array, check if it's not empty and doesn't contain only empty strings
    if isinstance(item, list):
        if not item:
            return False
        # Filter out arrays that contain only empty strings or None
        valid_items = [
            i for i in item if i is not None and (not isinstance(i, str) or i.strip())
        ]
        return len(valid_items) > 0

    # For dictionaries or other types, consider them valid
    return True


def _flatten_memory_list(memory_list: List[Any]) -> List[Any]:
    """Flatten nested memory list structure.

    Args:
        memory_list: List of memory items (may be nested)

    Returns:
        Flattened list of memory items
    """
    flattened = []
    for item in memory_list:
        if isinstance(item, list):
            # Recursively flatten nested lists
            flattened.extend(_flatten_memory_list(item))
        else:
            flattened.append(item)
    return flattened


def _extract_memory_content(item: Any) -> str:
    """Extract the actual content from a memory item.

    Args:
        item: Memory item (should be a dict or string, not a list)

    Returns:
        The content string
    """
    # If it's a string, return it directly
    if isinstance(item, str):
        return item.strip()

    # If it's a dictionary, try to extract the 'content' field
    if isinstance(item, dict):
        # Try common content field names
        if "content" in item:
            content_value = item["content"]
            # Handle case where content itself might be a dict or other complex type
            if isinstance(content_value, str):
                return content_value.strip()
            else:
                return str(content_value).strip()
        if "text" in item:
            return str(item["text"]).strip()
        if "message" in item:
            return str(item["message"]).strip()
        # If no content field found, log and return empty string instead of whole dict
        logger.warning(f"No content field found in memory item: {item.keys()}")
        return ""

    # If it's a list, this shouldn't happen after flattening, but handle it
    if isinstance(item, list):
        logger.warning(
            "Unexpected list in _extract_memory_content - should be flattened first"
        )
        return ""

    # For other types, convert to string
    return str(item)


def _format_search_results(
    episodic_memory: List[Any],
    profile_memory: List[Any],
) -> str:
    """Format search results into a simple LLM-friendly text format.

    Args:
        episodic_memory: List of episodic memory items (may be nested)
        profile_memory: List of profile memory items (may be nested)

    Returns:
        Formatted string containing all memory results in simple markdown format
    """
    lines = []

    # Header
    lines.append("# Memory Search Results")
    lines.append("")

    # Flatten and format Episodic Memories Section
    lines.append("## Episodic Memories")
    if episodic_memory:
        flattened_episodic = _flatten_memory_list(episodic_memory)
        for item in flattened_episodic:
            content = _extract_memory_content(item)
            if content:  # Only add non-empty content
                lines.append(f"- {content}")
    else:
        lines.append("- No episodic memories found")
    lines.append("")

    # Flatten and format Profile Memories Section
    lines.append("## Profile Memories")
    if profile_memory:
        flattened_profile = _flatten_memory_list(profile_memory)
        for item in flattened_profile:
            content = _extract_memory_content(item)
            if content:  # Only add non-empty content
                lines.append(f"- {content}")
    else:
        lines.append("- No profile memories found")

    return "\n".join(lines)


def _format_profile_memories(profile_memory: List[Any]) -> str:
    """Format profile memories into a simple LLM-friendly text format.

    Args:
        profile_memory: List of profile memory items (may be nested)

    Returns:
        Formatted string containing profile memories in simple markdown format
    """
    lines = []
    lines.append("# Profile Memories")
    lines.append("")

    if profile_memory:
        flattened_profile = _flatten_memory_list(profile_memory)
        for item in flattened_profile:
            content = _extract_memory_content(item)
            if content:  # Only add non-empty content
                lines.append(f"- {content}")
    else:
        lines.append("- No profile memories found")

    return "\n".join(lines)


def _format_episodic_memories(episodic_memory: List[Any]) -> str:
    """Format episodic memories into a simple LLM-friendly text format.

    Args:
        episodic_memory: List of episodic memory items (may be nested)

    Returns:
        Formatted string containing episodic memories in simple markdown format
    """
    lines = []
    lines.append("# Episodic Memories")
    lines.append("")

    if episodic_memory:
        flattened_episodic = _flatten_memory_list(episodic_memory)
        for item in flattened_episodic:
            content = _extract_memory_content(item)
            if content:  # Only add non-empty content
                lines.append(f"- {content}")
    else:
        lines.append("- No episodic memories found")

    return "\n".join(lines)


async def _handle_search_memory(
    user: User,
    session_id: str,
    query: str,
    limit: int,
    memmachine_client: MemMachineClient,
) -> Dict[str, Any]:
    """Handle memory search with unified logic for both MCP and REST callers."""
    session_data = create_session_data(session_id=session_id, user_id=user.username)
    search_data = SearchQuery(
        session=session_data, query=query, limit=limit, filter=None
    )
    result = memmachine_client.search_memory(search_data)
    content = result.get("content", {})
    episodic_memory = content.get("episodic_memory", [])
    profile_memory = content.get("profile_memory", [])

    # Filter out invalid episodic memory items (empty arrays, arrays with only empty strings, etc.)
    filtered_episodic_memory = [
        item for item in episodic_memory if _is_valid_episodic_memory_item(item)
    ]

    total_results = len(filtered_episodic_memory) + len(profile_memory)

    return {
        "session_id": session_id,
        "user_id": user.username,
        "query": query,
        "results": {
            "episodic_memory": filtered_episodic_memory,
            "profile_memory": profile_memory,
        },
        "total_results": total_results,
    }


async def _handle_delete_session_memory(
    user: User, session_id: str, memmachine_client: MemMachineClient
) -> Dict[str, Any]:
    """Handle session memory deletion with unified logic for both MCP and REST callers."""
    if not session_id:
        raise ValueError("Session ID is required for delete operation")
    session_data = create_session_data(session_id=session_id, user_id=user.username)
    delete_request = DeleteRequest(session=session_data)
    memmachine_client.delete_session_memory(delete_request)
    return {
        "session_id": session_id,
        "user_id": user.username,
    }
