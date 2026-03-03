"""Shared interfaces and base implementation for retrieval-skill tools."""

import asyncio
import logging
from typing import Any

from pydantic import BaseModel, ConfigDict, InstanceOf

from memmachine_server.common.episode_store import Episode
from memmachine_server.common.episode_store.episode_model import episodes_to_string
from memmachine_server.common.filter.filter_parser import FilterExpr
from memmachine_server.common.language_model.language_model import LanguageModel
from memmachine_server.common.reranker.reranker import Reranker
from memmachine_server.episodic_memory import EpisodicMemory

logger = logging.getLogger(__name__)


class QueryPolicy(BaseModel):
    """Scoring and budget policy used by retrieval-skill tools."""

    token_cost: int
    time_cost: int
    accuracy_score: float
    confidence_score: float
    max_attempts: int = 5
    max_return_len: int = 100000


class QueryParam(BaseModel):
    """Input parameters for a retrieval-skill query."""

    model_config = ConfigDict(arbitrary_types_allowed=True)
    query: str
    limit: int = 0
    expand_context: int = 0
    score_threshold: float = -float("inf")
    property_filter: FilterExpr | None = None
    memory: InstanceOf[EpisodicMemory]


class SkillToolBaseParam(BaseModel):
    """Dependency bundle used to construct a retrieval skill tool."""

    model_config = ConfigDict(arbitrary_types_allowed=True)
    model: InstanceOf[LanguageModel] | None = None
    children_tools: list[InstanceOf["SkillToolBase"]] | None = None
    extra_params: dict[str, Any] | None = None
    reranker: InstanceOf[Reranker] | None = None


class SkillToolBase:
    """Base class for retrieval skill implementations."""

    def __init__(self, param: SkillToolBaseParam) -> None:
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
    def skill_name(self) -> str:
        """Canonical retrieval skill identifier."""
        raise NotImplementedError("Subclasses must implement skill_name.")

    @property
    def skill_description(self) -> str:
        """Canonical retrieval skill description."""
        raise NotImplementedError("Subclasses must implement skill_description.")

    def _update_perf_metrics(
        self,
        source: dict[str, Any],
        target: dict[str, Any],
    ) -> dict[str, Any]:
        for key, value in source.items():
            if key not in target:
                target[key] = value
            else:
                if isinstance(value, int | float):
                    target[key] += value
                elif isinstance(value, list):
                    target[key].extend(value)
        return target

    async def _do_rerank(
        self, query: QueryParam, episodes: list[Episode]
    ) -> list[Episode]:
        if query.limit <= 0:
            return sorted(episodes, key=lambda x: x.created_at)

        if len(episodes) <= query.limit or self._reranker is None:
            if len(episodes) == 0:
                return episodes
            return sorted(episodes, key=lambda x: x.created_at)

        contents = [episodes_to_string([episode]) for episode in episodes]
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
            key=lambda x: x[1],
            reverse=True,
        )

        result = result[: query.limit] if query.limit > 0 else result
        reranked = [r[0] for r in result]
        return sorted(reranked, key=lambda x: x.created_at)

    async def do_query(
        self, policy: QueryPolicy, query: QueryParam
    ) -> tuple[list[Episode], dict[str, Any]]:
        if len(self._children_tools) == 0:
            raise RuntimeError("No child tool to call")
        tasks = [tool.do_query(policy, query) for tool in self._children_tools]
        results = await asyncio.gather(*tasks)
        data: list[Episode] = []
        perf_metrics: dict[str, Any] = {}
        for res, p_metric in results:
            if res is None:
                continue
            data.extend(res)
            perf_metrics = self._update_perf_metrics(p_metric, perf_metrics)
        return data, perf_metrics

    @property
    def accuracy_score(self) -> int:
        raise NotImplementedError

    @property
    def token_cost(self) -> int:
        raise NotImplementedError

    @property
    def time_cost(self) -> int:
        raise NotImplementedError

    def skill_tools(self) -> list["SkillToolBase"]:
        return self._children_tools
