"""Shared direct MemMachine search helpers for retrieval-agent runtimes."""

from __future__ import annotations

import datetime
import time
from typing import Any

from memmachine_server.common.episode_store import Episode, EpisodeType
from memmachine_server.episodic_memory import EpisodicMemory
from memmachine_server.retrieval_agent.common.agent_api import QueryParam

DIRECT_MEMORY_SELECTED_AGENT = "direct_memory"
DIRECT_MEMORY_SELECTED_AGENT_NAME = "DirectMemorySearch"


async def run_direct_memory_search(
    query: QueryParam,
) -> tuple[list[Episode], dict[str, Any]]:
    """Query long-term episodic memory directly for retrieval-agent tool calls."""
    perf_metrics: dict[str, Any] = {
        "memory_search_called": 0,
        "memory_retrieval_time": 0.0,
        "memory_search_latency_seconds": [],
        "selected_agent": DIRECT_MEMORY_SELECTED_AGENT,
        "selected_agent_name": DIRECT_MEMORY_SELECTED_AGENT_NAME,
    }
    mem_retrieval_start = time.perf_counter()
    query_response = await query.memory.query_memory(
        query=query.query,
        limit=query.limit,
        expand_context=query.expand_context,
        score_threshold=query.score_threshold,
        property_filter=query.property_filter,
        mode=EpisodicMemory.QueryMode.LONG_TERM_ONLY,
    )
    if query_response is None:
        episodes = []
    else:
        episodes = [
            Episode(
                uid=episode.uid,
                content=episode.content,
                session_key=query.memory.session_key,
                created_at=episode.created_at
                or datetime.datetime.now(tz=datetime.UTC),
                producer_id=episode.producer_id,
                producer_role=episode.producer_role,
                produced_for_id=episode.produced_for_id,
                episode_type=episode.episode_type or EpisodeType.MESSAGE,
                metadata=episode.metadata,
            )
            for episode in query_response.long_term_memory.episodes
        ]

    elapsed_seconds = time.perf_counter() - mem_retrieval_start
    perf_metrics["memory_search_called"] += 1
    perf_metrics["memory_retrieval_time"] += elapsed_seconds
    perf_metrics["memory_search_latency_seconds"].append(elapsed_seconds)
    return episodes, perf_metrics
