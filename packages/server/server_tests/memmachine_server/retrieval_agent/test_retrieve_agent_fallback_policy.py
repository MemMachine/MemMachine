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
    openai_function_call,
    openai_multi_tool_call_response,
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


@pytest.mark.asyncio
async def test_max_steps_exceeded_returns_partial_result(
    query_policy: QueryPolicy,
    tmp_path,
) -> None:
    episode = _build_episode("fb-steps", "fallback-steps")
    memory = FakeEpisodicMemory({"hello": [episode]})
    model = FakeOpenAIInstalledAgentModel(
        [openai_tool_call_response(query="hello", response_id=f"resp-{index}") for index in range(8)]
    )

    skill = _build_skill(model, tmp_path=tmp_path)
    episodes, metrics = await skill.do_query(
        query_policy,
        QueryParam(query="hello", limit=5, memory=memory),
    )

    assert [item.uid for item in episodes] == ["fb-steps"]
    assert metrics["orchestrator_final_response"] == (
        "Partial result: memmachine_search loop reached max_turns."
    )
    assert metrics["top_level_is_sufficient"] is False


@pytest.mark.asyncio
async def test_combined_call_budget_exceeded_triggers_fallback(
    query_policy: QueryPolicy,
    tmp_path,
) -> None:
    fallback_episode = _build_episode("budget-fallback", "budget-fallback")
    memory = FakeEpisodicMemory(
        {
            "hello": [fallback_episode],
            "hello again": [fallback_episode],
        }
    )
    model = FakeOpenAIInstalledAgentModel(
        [
            openai_multi_tool_call_response(
                calls=[
                    openai_function_call(query="hello", call_id="call-1"),
                    openai_function_call(query="hello again", call_id="call-2"),
                ],
                output_text="top-level",
            )
        ]
    )

    skill = _build_skill(model, tmp_path=tmp_path, max_combined_calls=1)
    episodes, metrics = await skill.do_query(
        query_policy,
        QueryParam(query="hello", limit=5, memory=memory),
    )

    assert [item.uid for item in episodes] == ["budget-fallback"]
    assert metrics["fallback_trigger_reason"] == "session_call_budget_exceeded"
    assert metrics["session_call_budget_limit"] == 1


@pytest.mark.asyncio
async def test_invalid_tool_name_falls_back_with_raw_error_details(
    query_policy: QueryPolicy,
    tmp_path,
) -> None:
    fallback_episode = _build_episode("fb-invalid", "fallback-invalid")
    memory = FakeEpisodicMemory({"hello": [fallback_episode]})
    model = FakeOpenAIInstalledAgentModel(
        [
            openai_tool_call_response(
                query="hello",
                tool_name="return_final",
                arguments={"final_response": "done"},
                output_text="bad tool",
            )
        ]
    )

    skill = _build_skill(model, tmp_path=tmp_path)
    episodes, metrics = await skill.do_query(
        query_policy,
        QueryParam(query="hello", limit=5, memory=memory),
    )

    assert [item.uid for item in episodes] == ["fb-invalid"]
    assert metrics["fallback_trigger_reason"] == "invalid_tool_call"
    error_diagnostics = metrics.get("error_diagnostics")
    assert isinstance(error_diagnostics, dict)
    assert error_diagnostics.get("context") == "top_level_tool_not_found"
