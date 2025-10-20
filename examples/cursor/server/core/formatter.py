"""
Memory formatting utilities for the  MemMachine Extension Server.
"""

import logging
from typing import Any, Dict, List

logger = logging.getLogger(__name__)


def _format_episodic_memory(memory: Dict[str, Any]) -> Dict[str, Any]:
    """Format episodic memory data for better readability."""
    formatted_memory = {
        "uuid": memory.get("uuid", "unknown"),
        "episode_type": memory.get("episode_type", "message"),
        "content_type": memory.get("content_type", "string"),
        "content": memory.get("content", ""),
        "timestamp": memory.get("timestamp", ""),
        "group_id": memory.get("group_id", "unknown"),
        "producer_id": memory.get("producer_id", "unknown"),
        "produced_for_id": memory.get("produced_for_id", "unknown"),
        "user_metadata": memory.get("user_metadata", {}),
    }
    return formatted_memory


def _format_profile_memory(memory: Dict[str, Any]) -> Dict[str, Any]:
    """Format profile memory data for better readability."""
    logger.info(f"Formatting profile memory: {memory}")
    formatted_memory = {
        "id": memory.get("metadata", {}).get("id", "unknown"),
        "similarity_score": memory.get("metadata", {}).get("similarity_score", 0.0),
        "tag": memory.get("tag", ""),
        "feature": memory.get("feature", ""),
        "value": memory.get("value", ""),
    }
    return formatted_memory


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
            content = item.get("content", "")
            if content:  # Only add non-empty content
                lines.append(f"- {content}")
    else:
        lines.append("- No episodic memories found")
    lines.append("")

    # Flatten and format Profile Memories Section
    lines.append("## Profile Memories")
    if profile_memory:
        flattened_profile = profile_memory
        for item in flattened_profile:
            content = item.get("value", "")
            if content:  # Only add non-empty content
                lines.append(f"- {content}")
    else:
        lines.append("- No profile memories found")

    return "\n".join(lines)


def _convert_profile_memories_to_markdown(profile_memory: List[Any]) -> str:
    """Format profile memories into a simple LLM-friendly text format.

    Args:
        profile_memory: List of profile memory items

    Returns:
        Formatted string containing profile memories in simple markdown format
    """
    lines = []
    lines.append("# Profile Memories")
    lines.append("")

    if profile_memory:
        flattened_profile = profile_memory
        for item in flattened_profile:
            content = item.get("value", "")
            if content:  # Only add non-empty content
                lines.append(f"- {content}")
    else:
        lines.append("- No profile memories found")

    return "\n".join(lines)


def _convert_episodic_memories_to_markdown(episodic_memory: List[Any]) -> str:
    """Format episodic memories into a simple LLM-friendly text format.

    Args:
        episodic_memory: List of episodic memory items

    Returns:
        Formatted string containing episodic memories in simple markdown format
    """
    lines = []
    lines.append("# Episodic Memories")
    lines.append("")

    if episodic_memory:
        flattened_episodic = _flatten_memory_list(episodic_memory)
        for item in flattened_episodic:
            content = item.get("content", "")
            if content:  # Only add non-empty content
                lines.append(f"- {content}")
    else:
        lines.append("- No episodic memories found")

    return "\n".join(lines)
