from __future__ import annotations

from datetime import UTC, datetime

import pytest

from memmachine_server.common.episode_store import Episode
from memmachine_server.common.reranker.reranker import Reranker
from memmachine_server.retrieval_agent.agents.retrieve_agent import RetrievalAgent
from memmachine_server.retrieval_agent.common.agent_api import (
    QueryPolicy,
    RetrievalAgentParams,
)
from server_tests.memmachine_server.retrieval_agent.provider_runner_stub import (
    FakeOpenAIInstalledAgentModel,
    FakeRestMemory,
    build_query_param,
    openai_function_call,
    openai_multi_tool_call_response,
    openai_text_response,
    openai_tool_call_response,
)


class DummyReranker(Reranker):
    async def score(self, query: str, candidates: list[str]) -> list[float]:
        _ = query
        return [float(len(candidates) - idx) for idx in range(len(candidates))]


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
async def test_selected_agent_defaults_to_memmachine_search_label(
    query_policy: QueryPolicy,
    tmp_path,
) -> None:
    episode = _build_episode("route-direct")
    memory = FakeRestMemory({"hello": [episode]})
    model = FakeOpenAIInstalledAgentModel(
        [
            openai_tool_call_response(query="hello"),
            openai_text_response("final answer"),
        ]
    )

    skill = _build_skill(model, tmp_path=tmp_path)
    _, metrics = await skill.do_query(
        query_policy,
        build_query_param(query="hello", rest_memory=memory),
    )

    assert metrics["selected_agent"] == "memmachine_search"
    assert metrics["selected_agent_name"] == "MemMachineSearch"
    assert metrics["orchestrator_sub_agent_runs"] == []


@pytest.mark.asyncio
async def test_plain_text_completion_keeps_raw_memory_episodes(
    query_policy: QueryPolicy,
    tmp_path,
) -> None:
    episode_a = _build_episode("route-a")
    episode_b = _build_episode("route-b")
    memory = FakeRestMemory({"branch a": [episode_a], "branch b": [episode_b]})
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
    result, metrics = await skill.do_query(
        query_policy,
        build_query_param(query="hello", rest_memory=memory),
    )

    assert result.episodic_memory is not None
    assert [item.uid for item in result.episodic_memory.long_term_memory.episodes] == [
        "route-a",
        "route-b",
    ]
    assert metrics["stage_result_memory_returned"] is False
    assert metrics["memory_search_called"] == 2
