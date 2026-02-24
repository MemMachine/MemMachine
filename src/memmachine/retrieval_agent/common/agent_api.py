"""Shared interfaces and base implementation for retrieval-agent tools."""

import asyncio
import logging
from abc import abstractmethod
from typing import Any

from pydantic import BaseModel, ConfigDict, InstanceOf

from memmachine.common.filter.filter_parser import (
    FilterExpr,
)
from memmachine.common.language_model.language_model import LanguageModel
from memmachine.common.reranker.reranker import Reranker
from memmachine.episodic_memory.declarative_memory import (
    DeclarativeMemory,
    Episode,
)

logger = logging.getLogger(__name__)


class QueryPolicy(BaseModel):
    """Scoring and budget policy used by retrieval-agent tools."""

    token_cost: int
    time_cost: int
    accuracy_score: float
    confidence_score: float
    max_attempts: int = 5
    max_return_len: int = 100000


class QueryParam(BaseModel):
    """Input parameters for a retrieval-agent query."""

    model_config = ConfigDict(arbitrary_types_allowed=True)
    query: str
    limit: int = 0
    expand_context: int = 0
    property_filter: FilterExpr | None = None
    memory: InstanceOf[DeclarativeMemory] | None = None


class AgentToolBaseParam(BaseModel):
    """Dependency bundle used to construct an agent tool."""

    model_config = ConfigDict(arbitrary_types_allowed=True)
    model: InstanceOf[LanguageModel] | None = None
    children_tools: list[InstanceOf["AgentToolBase"]] | None = None
    extra_params: dict[str, Any] | None = None
    reranker: InstanceOf[Reranker] | None = None


class AgentToolBase:
    """Base class for retrieval-agent tool implementations."""

    def __init__(self, param: AgentToolBaseParam) -> None:
        """Initialize tool dependencies and aggregate child costs."""
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

    def _update_perf_matrics(
        self,
        source: dict[str, Any],
        target: dict[str, Any],
    ) -> dict[str, Any]:
        for key, value in source.items():
            if key not in target:
                target[key] = value
            else:
                if isinstance(value, (int, float)):
                    target[key] += value
                elif isinstance(value, list):
                    target[key].extend(value)
        return target

    async def _do_rerank(
        self, query: QueryParam, episodes: list[Episode]
    ) -> list[Episode]:
        if query.limit <= 0:
            return sorted(episodes, key=lambda x: x.timestamp)

        if len(episodes) <= query.limit or self._reranker is None:
            if len(episodes) == 0:
                return episodes
            return sorted(episodes, key=lambda x: x.timestamp)

        contents = [
            DeclarativeMemory.string_from_episode_context([episode])
            for episode in episodes
        ]
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
                    logger.exception("Reranker failed after maximum retries.")
                    raise
                if "ThrottlingException" in str(e):
                    logger.warning(
                        "Reranker throttling exception, retrying after 5 seconds..."
                    )
                    await asyncio.sleep(5)
                else:
                    raise

        result = sorted(
            zip(episodes, scores, strict=True),
            key=lambda x: x[1],  # sort by score
            reverse=True,  # highest score first
        )

        result = result[: query.limit] if query.limit > 0 else result
        res = [r[0] for r in result]
        return sorted(res, key=lambda x: x.timestamp)

    async def do_query(
        self, policy: QueryPolicy, query: QueryParam
    ) -> tuple[list[Episode], dict[str, Any]]:
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

    def agent_tools(self) -> list["AgentToolBase"]:
        return self._children_tools
