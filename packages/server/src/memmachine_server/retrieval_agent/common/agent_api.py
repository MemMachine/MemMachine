"""Shared interfaces and helper types for retrieval agents."""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Protocol, runtime_checkable

from memmachine_common.api import MemoryType
from pydantic import BaseModel, ConfigDict, InstanceOf

from memmachine_server.common.episode_store import Episode
from memmachine_server.common.episode_store.episode_model import episodes_to_string
from memmachine_server.common.filter.filter_parser import FilterExpr
from memmachine_server.common.language_model.language_model import LanguageModel
from memmachine_server.common.reranker.reranker import Reranker
from memmachine_server.episodic_memory import EpisodicMemory
from memmachine_server.semantic_memory.semantic_model import SemanticFeature

logger = logging.getLogger(__name__)


class QueryPolicy(BaseModel):
    """Scoring and budget policy used by retrieval agents."""

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
    score_threshold: float = -float("inf")
    property_filter: FilterExpr | None = None
    session_key: str
    rest_memory: SearchToolMemoryProtocol
    target_memories: list[MemoryType]


class RetrievalAgentResult(BaseModel):
    """Combined search results returned by the top-level retrieval agent."""

    model_config = ConfigDict(arbitrary_types_allowed=True)
    episodic_memory: EpisodicMemory.QueryResponse | None = None
    semantic_memory: list[SemanticFeature] | None = None


@runtime_checkable
class SearchToolMemoryProtocol(Protocol):
    """Minimal REST-style search dependency consumed by SkillRunner."""

    async def search(
        self,
        query: str,
        *,
        limit: int | None = None,
        expand_context: int = 0,
        score_threshold: float | None = None,
        agent_mode: bool = False,
    ) -> object: ...


class RetrievalAgentParams(BaseModel):
    """Dependency bundle used to construct a retrieval agent."""

    model_config = ConfigDict(arbitrary_types_allowed=True)
    model: InstanceOf[LanguageModel]
    extra_params: dict[str, Any] | None = None
    reranker: InstanceOf[Reranker] | None = None


@runtime_checkable
class RetrievalAgentProtocol(Protocol):
    """Protocol implemented by server-side retrieval agents."""

    @property
    def agent_name(self) -> str: ...

    @property
    def agent_description(self) -> str: ...

    @property
    def accuracy_score(self) -> int: ...

    @property
    def token_cost(self) -> int: ...

    @property
    def time_cost(self) -> int: ...

    async def do_query(
        self,
        policy: QueryPolicy,
        query: QueryParam,
    ) -> tuple[RetrievalAgentResult, dict[str, Any]]: ...


async def rerank_episodes(
    *,
    query: QueryParam,
    episodes: list[Episode],
    reranker: Reranker | None,
) -> list[Episode]:
    """Rerank retrieved episodes when the caller configured a reranker."""
    if query.limit <= 0:
        return sorted(episodes, key=lambda item: item.created_at)

    if len(episodes) <= query.limit or reranker is None:
        if not episodes:
            return episodes
        return sorted(episodes, key=lambda item: item.created_at)

    contents = [episodes_to_string([episode]) for episode in episodes]
    success = False
    retries_remaining = 60
    scores: list[float] = []
    while not success:
        try:
            scores = await reranker.score(query.query, contents)
            success = True
        except Exception as err:
            retries_remaining -= 1
            if retries_remaining == 0:
                logger.exception("Reranker failed after maximum retries.")
                raise
            if "ThrottlingException" in str(err):
                logger.warning(
                    "Reranker throttling exception, retrying after 5 seconds..."
                )
                await asyncio.sleep(5)
            else:
                raise

    ranked = sorted(
        zip(episodes, scores, strict=True),
        key=lambda item: item[1],
        reverse=True,
    )
    limited = ranked[: query.limit] if query.limit > 0 else ranked
    reranked = [episode for episode, _score in limited]
    return sorted(reranked, key=lambda item: item.created_at)
