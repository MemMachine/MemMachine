
import time
from abc import abstractmethod
import asyncio
from pydantic import BaseModel, InstanceOf, ConfigDict, JsonValue
from datetime import datetime
from typing import Any
from collections.abc import Iterable
from memmachine.common.language_model.language_model import LanguageModel
from memmachine.episodic_memory.declarative_memory import (
    ContentType,
    Episode,
    DeclarativeMemory,
)
from memmachine.common.reranker.reranker import Reranker
from memmachine.common.filter.filter_parser import (
    FilterExpr,
)

class QueryPolicy(BaseModel):
    token_cost: int
    time_cost: int
    accuracy_score: float
    confidence_score: float
    max_attempts: int = 5
    max_return_len: int = 100000

class QueryParam(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    query: str
    limit: int | None = None
    expand_context: int = 0
    property_filter: FilterExpr | None = None


class AgentToolBaseParam(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    model: InstanceOf[LanguageModel] | None= None
    children_tools: list[InstanceOf['AgentToolBase']] | None = None
    extra_params: dict[str, Any] | None = None
    reranker: InstanceOf[Reranker] | None = None

class AgentToolBase:
    def __init__(self, param: AgentToolBaseParam):
        super().__init__()
        self._model = param.model
        self._children_tools = param.children_tools or []
        self._reranker = param.reranker
        self._child_token_cost = 0
        self._child_time_cost = 0
        for tool in self._children_tools:
            self._child_token_cost += tool.token_cost
            self._child_time_cost += tool.time_cost

    @property
    @abstractmethod
    def agent_name(self) -> str:
        pass

    @property
    @abstractmethod
    def agent_description(self) -> str:
        pass

    def _update_perf_matrics(self, source: dict[str, Any], target: dict[str, Any]) -> dict[str, Any]:
        for key, value in source.items():
            if key not in target:
                target[key] = value
            else:
                if isinstance(value, (int, float)):
                    target[key] += value
                elif isinstance(value, list):
                    target[key].extend(value)
        return target

    async def _do_rerank(self, query: QueryParam, episodes: list[Episode]) -> list[Episode]:
        if len(episodes) <= query.limit or self._reranker is None:
            if len(episodes) == 0:
                return episodes
            return sorted(episodes, key=lambda x: x.timestamp)

        contents = []
        for e in episodes:
            contents.append(DeclarativeMemory.string_from_episode_context([e]))
        success = False
        max_retry = 60
        scores = []
        while not success:
            try:
                scores = await self._reranker.score(query.query, contents)
                success = True
            except Exception as e:
                max_retry -= 1
                if max_retry == 0:
                    print(f"ERROR: Reranker failed after maximum retries.")
                    raise e
                if "ThrottlingException" in str(e):
                    print(f"WARNING: Reranker throttling exception, retrying after 5 second...")
                    time.sleep(5)
                else:
                    raise e

        result = sorted(
            zip(episodes, scores),
            key=lambda x: x[1],   # sort by score
            reverse=True          # highest score first
        )

        res = [r[0] for r in result[:query.limit]]
        return sorted(res, key=lambda x: x.timestamp)

    async def do_query(self, policy: QueryPolicy, query: QueryParam) -> tuple[list[Episode], dict[str, Any]]:
        if len(self._children_tools) == 0:
            raise RuntimeError("No child tool to call")
        tasks = []
        for tool in self._children_tools:
            task = tool.do_query(policy, query)
            tasks.append(task)
        results = await asyncio.gather(*tasks)
        data: list[Episode] = []
        perf_matrics: dict[str, Any] = {}
        for res, p_matric in results:
            if res is None:
                continue
            data.extend(res)
            perf_matrics = self._update_perf_matrics(perf_matrics, p_matric)
        return data, perf_matrics

    @property
    @abstractmethod
    def accuracy_score(self) -> int:
        pass

    @property
    @abstractmethod
    def token_cost(self) -> int:
        pass

    @property
    @abstractmethod
    def time_cost(self) -> int:
        pass

    def agent_tools(self) -> list['AgentToolBase']:
        return self._children_tools
