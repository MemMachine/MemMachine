from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import pytest

from memmachine_server.common.episode_store import Episode, EpisodeResponse
from memmachine_server.common.reranker.reranker import Reranker
from memmachine_server.episodic_memory import EpisodicMemory
from memmachine_server.retrieval_agent.agents.memory_search import (
    DIRECT_MEMORY_SELECTED_AGENT_NAME,
)
from memmachine_server.retrieval_agent.agents.retrieve_agent import RetrievalAgent
from memmachine_server.retrieval_agent.common.agent_api import (
    QueryParam,
    QueryPolicy,
    RetrievalAgentParams,
)
from server_tests.memmachine_server.retrieval_agent.provider_runner_stub import (
    FakeOpenAIInstalledAgentModel,
    openai_function_call,
    openai_multi_tool_call_response,
    openai_text_response,
    openai_tool_call_response,
)


class DummyReranker(Reranker):
    async def score(self, query: str, candidates: list[str]) -> list[float]:
        _ = query
        return [float(len(candidates) - idx) for idx in range(len(candidates))]


class FakeEpisodicMemory(EpisodicMemory):
    def __init__(self, episodes_by_query: dict[str, list[Episode]]) -> None:
        self._episodes_by_query = episodes_by_query
        self._session_key = "test-session"

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
        _ = expand_context, score_threshold, property_filter, mode
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


def _build_skill(
    model: FakeOpenAIInstalledAgentModel,
    *,
    tmp_path,
    **extra_params: object,
) -> RetrievalAgent:
    reranker = DummyReranker()
    return RetrievalAgent(
        RetrievalAgentParams(
            model=model,
            extra_params={
                "agent_install_cache_path": tmp_path / ".agent-cache.json",
                **extra_params,
            },
            reranker=reranker,
        )
    )


def _build_episode(uid: str) -> Episode:
    return Episode(
        uid=uid,
        content="hello",
        session_key="test-session",
        created_at=datetime.now(tz=UTC),
        producer_id="unit-test",
        producer_role="assistant",
    )


@pytest.mark.asyncio
async def test_selected_agent_defaults_to_direct_memory_label(
    query_policy: QueryPolicy,
    tmp_path,
) -> None:
    episode = _build_episode("route-direct")
    memory = FakeEpisodicMemory({"hello": [episode]})
    model = FakeOpenAIInstalledAgentModel(
        [
            openai_tool_call_response(query="hello"),
            openai_text_response("final answer"),
        ]
    )

    skill = _build_skill(model, tmp_path=tmp_path)
    _, metrics = await skill.do_query(
        query_policy,
        QueryParam(query="hello", limit=5, memory=memory),
    )

    assert metrics["selected_agent"] == "direct_memory"
    assert metrics["selected_agent_name"] == DIRECT_MEMORY_SELECTED_AGENT_NAME
    assert metrics["orchestrator_sub_agent_runs"] == []


@pytest.mark.asyncio
async def test_plain_text_completion_keeps_raw_memory_episodes(
    query_policy: QueryPolicy,
    tmp_path,
) -> None:
    episode_a = _build_episode("route-a")
    episode_b = _build_episode("route-b")
    memory = FakeEpisodicMemory({"branch a": [episode_a], "branch b": [episode_b]})
    model = FakeOpenAIInstalledAgentModel(
        [
            openai_multi_tool_call_response(
                calls=[
                    openai_function_call(query="branch a", call_id="call-1"),
                    openai_function_call(query="branch b", call_id="call-2"),
                ]
            ),
            openai_text_response("combined answer"),
        ]
    )

    skill = _build_skill(model, tmp_path=tmp_path)
    episodes, metrics = await skill.do_query(
        query_policy,
        QueryParam(query="hello", limit=5, memory=memory),
    )

    assert [item.uid for item in episodes] == ["route-a", "route-b"]
    assert metrics["stage_result_memory_returned"] is False
    assert metrics["memory_search_called"] == 2
