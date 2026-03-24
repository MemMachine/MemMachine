from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import pytest

from memmachine_server.common.episode_store import Episode, EpisodeResponse
from memmachine_server.common.reranker.reranker import Reranker
from memmachine_server.episodic_memory import EpisodicMemory
from memmachine_server.retrieval_agent.agents.retrieve_agent import RetrievalAgent
from memmachine_server.retrieval_agent.common.agent_api import (
    QueryParam,
    QueryPolicy,
    RetrievalAgentParams,
)
from server_tests.memmachine_server.retrieval_agent.provider_runner_stub import (
    FakeOpenAIInstalledAgentModel,
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
        self.queries: list[str] = []

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
        self.queries.append(query)
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


def _build_episode(uid: str, content: str) -> Episode:
    return Episode(
        uid=uid,
        content=content,
        session_key="test-session",
        created_at=datetime.now(tz=UTC),
        producer_id="unit-test",
        producer_role="assistant",
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


@pytest.fixture
def query_policy() -> QueryPolicy:
    return QueryPolicy(
        token_cost=0,
        time_cost=0,
        accuracy_score=0.0,
        confidence_score=0.0,
    )


@pytest.mark.asyncio
async def test_memmachine_search_collects_episodes_and_final_response_metrics(
    query_policy: QueryPolicy,
    tmp_path,
) -> None:
    episode = _build_episode("ep-direct", "direct")
    memory = FakeEpisodicMemory({"hello": [episode]})
    model = FakeOpenAIInstalledAgentModel(
        [
            openai_tool_call_response(
                query="hello",
                arguments={"query": "hello", "rationale": "fetch evidence"},
            ),
            openai_text_response("finalized"),
        ]
    )
    retrieve_agent = _build_skill(model, tmp_path=tmp_path)

    episodes, metrics = await retrieve_agent.do_query(
        query_policy,
        QueryParam(query="hello", limit=5, memory=memory),
    )

    assert [item.uid for item in episodes] == ["ep-direct"]
    assert metrics["route"] == "RetrievalAgent"
    assert metrics["orchestrator_completed"] is True
    assert metrics["orchestrator_tool_call_count"] == 1
    assert metrics["orchestrator_sub_agent_count"] == 0
    assert metrics["orchestrator_final_response"] == "finalized"
    assert metrics["top_level_is_sufficient"] is True
    assert metrics["answer_candidate"] == "finalized"
    trace = metrics["orchestrator_trace"]
    assert isinstance(trace, dict)
    search_call = trace["tool_calls"][0]
    assert search_call["tool_name"] == "memmachine_search"
    assert search_call["raw_result"]["episodes_returned"] == 1
    assert search_call["raw_result"]["query"] == "hello"
    assert "episodes_human_readable" not in search_call["raw_result"]


@pytest.mark.asyncio
async def test_missing_tool_call_defaults_to_direct_search(
    query_policy: QueryPolicy,
    tmp_path,
) -> None:
    episode = _build_episode("ep-fallback-search", "fallback")
    memory = FakeEpisodicMemory({"hello": [episode]})
    model = FakeOpenAIInstalledAgentModel(
        [openai_text_response("plain response without tool call")]
    )
    retrieve_agent = _build_skill(model, tmp_path=tmp_path)

    episodes, metrics = await retrieve_agent.do_query(
        query_policy,
        QueryParam(query="hello", limit=5, memory=memory),
    )

    assert [item.uid for item in episodes] == ["ep-fallback-search"]
    assert metrics["orchestrator_tool_call_count"] == 1
    trace = metrics["orchestrator_trace"]
    assert trace["tool_calls"][0]["tool_name"] == "memmachine_search"
    assert trace["events"][-1]["event_type"] == "orchestration_completed"


@pytest.mark.asyncio
async def test_inconclusive_plain_text_response_does_not_mark_sufficient(
    query_policy: QueryPolicy,
    tmp_path,
) -> None:
    episode = _build_episode("ep-unclear", "unclear evidence")
    memory = FakeEpisodicMemory({"hello": [episode]})
    model = FakeOpenAIInstalledAgentModel(
        [
            openai_tool_call_response(query="hello"),
            openai_text_response("I don't know based on the retrieved evidence."),
        ]
    )
    retrieve_agent = _build_skill(model, tmp_path=tmp_path)

    _, metrics = await retrieve_agent.do_query(
        query_policy,
        QueryParam(query="hello", limit=5, memory=memory),
    )

    assert metrics["top_level_is_sufficient"] is False
    assert "answer_candidate" not in metrics


@pytest.mark.asyncio
async def test_legacy_tool_name_triggers_fallback(
    query_policy: QueryPolicy,
    tmp_path,
) -> None:
    fallback_episode = _build_episode("ep-fallback", "fallback")
    memory = FakeEpisodicMemory({"hello": [fallback_episode]})
    model = FakeOpenAIInstalledAgentModel(
        [
            openai_tool_call_response(
                query="hello",
                tool_name="spawn_sub_agent",
                arguments={"agent_name": "coq", "query": "hello"},
            )
        ]
    )
    retrieve_agent = _build_skill(model, tmp_path=tmp_path)

    episodes, metrics = await retrieve_agent.do_query(
        query_policy,
        QueryParam(query="hello", limit=5, memory=memory),
    )

    assert [item.uid for item in episodes] == ["ep-fallback"]
    assert metrics["fallback_trigger_reason"] == "invalid_tool_call"
    assert metrics["orchestrator_trace"]["tool_calls"][0]["tool_name"] == (
        "memmachine_search"
    )
