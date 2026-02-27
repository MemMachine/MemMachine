"""Memory-retrieval agent that queries episodic memory long-term context."""

import datetime
import logging
import time
from typing import Any

from memmachine.common.api import EpisodeType
from memmachine.common.episode_store import Episode
from memmachine.episodic_memory import EpisodicMemory
from memmachine.retrieval_agent.common.agent_api import (
    AgentToolBase,
    AgentToolBaseParam,
    QueryParam,
    QueryPolicy,
)

logger = logging.getLogger(__name__)


class MemMachineAgent(AgentToolBase):
    """Agent that runs long-term episodic memory search without rewriting."""

    def __init__(self, param: AgentToolBaseParam) -> None:
        """Initialize retrieval behavior and shared dependencies."""
        super().__init__(param)

    @property
    def agent_name(self) -> str:
        return "MemMachineAgent"

    @property
    def agent_description(self) -> str:
        return "This agent retrieve data from MemMachine memory directly"

    @property
    def accuracy_score(self) -> int:
        return 0

    @property
    def token_cost(self) -> int:
        return 0

    @property
    def time_cost(self) -> int:
        return 0

    async def do_query(
        self,
        policy: QueryPolicy,
        query: QueryParam,
    ) -> tuple[list[Episode], dict[str, Any]]:
        _ = policy
        logger.info("CALLING %s with query: %s", self.agent_name, query.query)

        perf_metrics: dict[str, Any] = {
            "memory_search_called": 0,
            "memory_retrieval_time": 0.0,
            "agent": self.agent_name,
        }
        mem_retrieval_start = time.time()
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

        perf_metrics["memory_search_called"] += 1
        perf_metrics["memory_retrieval_time"] += time.time() - mem_retrieval_start

        return episodes, perf_metrics
