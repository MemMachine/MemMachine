import logging
import time
from memmachine.episodic_memory.declarative_memory import (
    Episode,
    DeclarativeMemory,
    DeclarativeMemoryParams,
)
from memmachine.retrieval_agent.common.agent_api import (
    AgentToolBase,
    AgentToolBaseParam,
    QueryParam,
    QueryPolicy,
)
from typing import Any

logger = logging.getLogger(__name__)

class MemMachineAgent(AgentToolBase):
    def __init__(self, param: AgentToolBaseParam):
        super().__init__(param)
        if param.extra_params is None:
            raise ValueError("Did not find extra params")
        self._memory: DeclarativeMemory = param.extra_params.get("memory")
        if not self._memory:
            raise ValueError("Did not find memory instance")
        if not isinstance(self._memory, DeclarativeMemory):
            raise ValueError("The memory type is not DeclarativeMemory: %s", type(self._memory).__name__)
    
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
        
    async def do_query(self, policy: QueryPolicy, query: QueryParam) -> tuple[list[Episode], dict[str, Any]]:
        logger.info(f"CALLING {self.agent_name} with query: {query.query}")

        perf_matrics = {"memory_retrieval_time": 0.0}
        mem_retrieval_start = time.time()
        scored_episodes = await self._memory.search_scored(
            query=query.query,
            max_num_episodes=query.limit,
            expand_context=query.expand_context,
            property_filter=query.property_filter,
        )
        perf_matrics["memory_retrieval_time"] += time.time() - mem_retrieval_start

        episodes = [episode for _, episode in scored_episodes]
        return episodes, perf_matrics
