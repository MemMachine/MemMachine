from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import pytest

from memmachine_server.common.configuration.retrieval_config import (
    RetrievalAgentConf,
    RetrievalAgentSessionProvider,
)
from memmachine_server.common.episode_store import Episode, EpisodeResponse
from memmachine_server.common.language_model.language_model import LanguageModel
from memmachine_server.common.reranker.reranker import Reranker
from memmachine_server.episodic_memory import EpisodicMemory
from memmachine_server.retrieval_agent.agents.memory_search import (
    DIRECT_MEMORY_SELECTED_AGENT_NAME,
    run_direct_memory_search,
)
from memmachine_server.retrieval_agent.common.agent_api import (
    QueryParam,
    QueryPolicy,
)
from memmachine_server.retrieval_agent.service_locator import create_retrieval_agent


class DummyLanguageModel(LanguageModel):
    """Lightweight language model stub for unit tests."""

    def __init__(self, responses: list[str] | str) -> None:
        if isinstance(responses, str):
            responses = [responses]
        self._responses = responses
        self.call_count = 0

    async def generate_parsed_response(
        self,
        output_format: type[Any],
        system_prompt: str | None = None,
        user_prompt: str | None = None,
        max_attempts: int = 1,
    ) -> Any | None:
        return None

    async def generate_response(
        self,
        system_prompt: str | None = None,
        user_prompt: str | None = None,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, str] | None = None,
        max_attempts: int = 1,
    ) -> tuple[str, Any]:
        return "", None

    async def generate_response_with_token_usage(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> tuple[str, Any, int, int]:
        idx = min(self.call_count, len(self._responses) - 1)
        response = self._responses[idx]
        self.call_count += 1
        return response, None, 1, 1


class DummyReranker(Reranker):
    """Reranker stub returning predefined scores."""

    def __init__(self, scores: list[float] | None = None) -> None:
        self._scores = scores or []
        self.call_count = 0

    async def score(self, query: str, candidates: list[str]) -> list[float]:
        self.call_count += 1
        if self._scores and len(self._scores) == len(candidates):
            return list(self._scores)
        return [float(len(candidates) - idx) for idx in range(len(candidates))]


class FakeEpisodicMemory(EpisodicMemory):
    """EpisodicMemory stub that returns preset long-term episodes by query."""

    def __init__(self, episodes_by_query: dict[str, list[Episode]]) -> None:
        self._episodes_by_query = episodes_by_query
        self._session_key = "test-session"
        self.queries: list[str] = []
        self.calls: list[dict[str, Any]] = []

    async def query_memory(
        self,
        query: str,
        *,
        limit: int | None = None,
        expand_context: int = 0,
        score_threshold: float = -float("inf"),
        property_filter: Any | None = None,
        mode: EpisodicMemory.QueryMode = EpisodicMemory.QueryMode.BOTH,
    ) -> EpisodicMemory.QueryResponse | None:
        self.queries.append(query)
        self.calls.append(
            {
                "query": query,
                "limit": limit,
                "expand_context": expand_context,
                "score_threshold": score_threshold,
                "property_filter": property_filter,
                "mode": mode,
            }
        )
        episodes = self._episodes_by_query.get(query, [])
        search_limit = limit if limit is not None else len(episodes)
        return EpisodicMemory.QueryResponse(
            long_term_memory=EpisodicMemory.QueryResponse.LongTermMemoryResponse(
                episodes=[
                    EpisodeResponse(score=1.0, **episode.model_dump())
                    for episode in episodes[:search_limit]
                ]
            ),
            short_term_memory=EpisodicMemory.QueryResponse.ShortTermMemoryResponse(
                episodes=[],
                episode_summary=[],
            ),
        )


@pytest.fixture
def query_policy() -> QueryPolicy:
    return QueryPolicy(
        token_cost=0,
        time_cost=0,
        accuracy_score=0.0,
        confidence_score=0.0,
    )


def _build_episode(*, uid: str, content: str, created_at: datetime) -> Episode:
    return Episode(
        uid=uid,
        content=content,
        session_key="test-session",
        created_at=created_at,
        producer_id="unit-test",
        producer_role="assistant",
    )


@pytest.mark.asyncio
async def test_direct_memory_search_returns_episodes(
    query_policy: QueryPolicy,
) -> None:
    _ = query_policy
    now = datetime.now(tz=UTC)
    episode = _build_episode(uid="e1", content="hello", created_at=now)
    memory = FakeEpisodicMemory({"hello": [episode]})

    result, metrics = await run_direct_memory_search(
        QueryParam(query="hello", limit=5, memory=memory),
    )

    assert result == [episode]
    assert metrics["memory_search_called"] == 1
    assert metrics["selected_agent"] == "direct_memory"
    assert metrics["selected_agent_name"] == DIRECT_MEMORY_SELECTED_AGENT_NAME
    assert isinstance(metrics.get("memory_search_latency_seconds"), list)
    assert len(metrics["memory_search_latency_seconds"]) == 1
    assert float(metrics["memory_retrieval_time"]) >= float(
        metrics["memory_search_latency_seconds"][0]
    )


@pytest.mark.asyncio
async def test_direct_memory_search_queries_long_term_memory_only(
    query_policy: QueryPolicy,
) -> None:
    _ = query_policy
    now = datetime.now(tz=UTC)
    episode = _build_episode(uid="callback-e1", content="from-memory", created_at=now)
    memory = FakeEpisodicMemory({"callback-query": [episode]})

    result, metrics = await run_direct_memory_search(
        QueryParam(
            query="callback-query",
            limit=3,
            expand_context=2,
            score_threshold=0.55,
            memory=memory,
        ),
    )

    assert result == [episode]
    assert metrics["memory_search_called"] == 1
    assert isinstance(metrics.get("memory_search_latency_seconds"), list)
    assert len(metrics["memory_search_latency_seconds"]) == 1
    assert memory.calls == [
        {
            "query": "callback-query",
            "limit": 3,
            "expand_context": 2,
            "score_threshold": 0.55,
            "property_filter": None,
            "mode": EpisodicMemory.QueryMode.LONG_TERM_ONLY,
        }
    ]


def test_service_locator_defaults_to_retrieve_agent() -> None:
    model = DummyLanguageModel("direct-memory")
    agent = create_retrieval_agent(
        model=model,
        reranker=DummyReranker(),
    )
    assert agent.agent_name == "RetrievalAgent"


def test_service_locator_accepts_generic_language_model_without_session_model() -> None:
    agent = create_retrieval_agent(
        model=DummyLanguageModel("direct-memory"),
        reranker=DummyReranker(),
    )
    assert agent.agent_name == "RetrievalAgent"


def test_service_locator_rejects_legacy_agent_routes() -> None:
    model = DummyLanguageModel("direct-memory")
    reranker = DummyReranker()
    for agent_name in [
        "MemMachineSkill",
        "ChainOfQueryAgent",
        "ToolSelectSkill",
    ]:
        with pytest.raises(
            ValueError, match="only supports agent_name='RetrievalAgent'"
        ):
            _ = create_retrieval_agent(
                model=model,
                reranker=reranker,
                agent_name=agent_name,
            )

    agent = create_retrieval_agent(
        model=model,
        reranker=reranker,
        agent_name="RetrievalAgent",
    )
    assert agent.agent_name == "RetrievalAgent"


def test_service_locator_applies_retrieval_conf_budgets() -> None:
    model = DummyLanguageModel("direct-memory")
    reranker = DummyReranker()
    conf = RetrievalAgentConf(
        agent_session_provider=RetrievalAgentSessionProvider.ANTHROPIC,
        anthropic_api_key="anthropic-key",
        agent_session_timeout_seconds=180,
        agent_session_max_combined_calls=10,
    )

    agent = create_retrieval_agent(
        model=model,
        reranker=reranker,
        retrieval_conf=conf,
    )

    assert agent.agent_name == "RetrievalAgent"
    assert agent._global_timeout_seconds == 180
    assert agent._max_combined_calls == 10
    assert agent._available_sub_agents == ["coq"]
