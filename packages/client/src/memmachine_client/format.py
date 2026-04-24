"""LLM-friendly formatting for memory search and list results.

These functions mirror the server's internal formatting logic so that
client-side consumers get the same compact representation the server
uses when feeding memories into language models.

Episodic format matches ``string_from_episode_context`` in the server.
Semantic format matches ``_features_to_llm_format`` in the server.
"""

from __future__ import annotations

import json
from collections.abc import Iterable

from memmachine_common.api.spec import (
    Episode,
    EpisodeResponse,
    SearchResult,
    SemanticFeature,
)


def format_episodes(episodes: Iterable[EpisodeResponse | Episode]) -> str:
    """Format episodic memories as an LLM-friendly string.

    Each episode is rendered as::

        [Monday, January 01, 2024 at 01:30 PM] producer_id: "content"

    This mirrors the server's ``string_from_episode_context`` output.

    Args:
        episodes: Episodic memory entries from a search or list result.

    Returns:
        Newline-terminated formatted string (empty string when *episodes*
        is empty).

    """
    result = ""
    for episode in episodes:
        if episode.created_at is not None:
            date_str = episode.created_at.strftime("%A, %B %d, %Y")
            time_str = episode.created_at.strftime("%I:%M %p")
            result += f"[{date_str} at {time_str}] {episode.producer_id}: {json.dumps(episode.content)}\n"
        else:
            result += f"{episode.producer_id}: {json.dumps(episode.content)}\n"
    return result


def format_semantic_memories(features: Iterable[SemanticFeature]) -> str:
    """Format semantic memories as a compact JSON string.

    Produces a ``{tag: {feature_name: value}}`` structure, omitting all
    metadata for context efficiency.  This mirrors the server's
    ``_features_to_llm_format`` output.

    Args:
        features: Semantic memory entries from a search or list result.

    Returns:
        JSON string of the grouped features.

    """
    structured: dict[str, dict[str, str]] = {}
    for feature in features:
        structured.setdefault(feature.tag, {})[feature.feature_name] = feature.value
    return json.dumps(structured)


def format_search_result(result: SearchResult) -> str:
    """Format a search result as an LLM-friendly string.

    Combines episodic and semantic memories from a
    :class:`~memmachine_common.api.spec.SearchResult` returned by
    ``Memory.search()``.

    Args:
        result: A ``SearchResult`` object.

    Returns:
        Formatted string combining both memory types.

    """
    sections: list[str] = []

    if result.content.episodic_memory is not None:
        episodes = (
            result.content.episodic_memory.long_term_memory.episodes
            + result.content.episodic_memory.short_term_memory.episodes
        )
        if episodes:
            sections.append(f"[Episodic Memory]\n{format_episodes(episodes)}")

    if result.content.semantic_memory:
        sections.append(
            f"[Semantic Memory]\n{format_semantic_memories(result.content.semantic_memory)}"
        )

    return "\n".join(sections)
