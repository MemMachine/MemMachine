"""Memory-retrieval agent that queries declarative memory directly."""

import logging
import time
from typing import Any

from memmachine.episodic_memory.declarative_memory import (
    DeclarativeMemory,
    Episode,
)
from memmachine.retrieval_agent.common.agent_api import (
    AgentToolBase,
    AgentToolBaseParam,
    QueryParam,
    QueryPolicy,
)

logger = logging.getLogger(__name__)


class MemMachineAgent(AgentToolBase):
    """Agent that uses declarative memory search without query rewriting."""

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

        perf_matrics: dict[str, Any] = {
            "memory_search_called": 0,
            "memory_retrieval_time": 0.0,
            "agent": self.agent_name,
        }
        memory = query.memory
        if memory is None:
            raise ValueError("QueryParam.memory must be provided for MemMachineAgent")
        if not isinstance(memory, DeclarativeMemory):
            raise TypeError(
                "QueryParam.memory must be DeclarativeMemory: "
                f"{type(memory).__name__}"
            )
        mem_retrieval_start = time.time()
        scored_episodes = await memory.search_scored(
            query=query.query,
            max_num_episodes=query.limit,
            expand_context=query.expand_context,
            property_filter=query.property_filter,
        )
        perf_matrics["memory_search_called"] += 1
        perf_matrics["memory_retrieval_time"] += time.time() - mem_retrieval_start

        episodes = [episode for _, episode in scored_episodes]
        return episodes, perf_matrics
