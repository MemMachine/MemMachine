from __future__ import annotations

from datetime import UTC, datetime

import pytest

from memmachine_server.common.episode_store import Episode
from memmachine_server.common.reranker.reranker import Reranker
from memmachine_server.retrieval_agent.agents.retrieve_agent import RetrievalAgent
from memmachine_server.retrieval_agent.agents.session_state import (
    TopLevelAgentSessionState,
)
from memmachine_server.retrieval_agent.common.agent_api import (
    QueryPolicy,
    RetrievalAgentParams,
)
from server_tests.memmachine_server.retrieval_agent.provider_runner_stub import (
    FakeOpenAIInstalledAgentModel,
    FakeRestMemory,
    build_query_param,
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


def test_top_level_session_state_merges_unique_episodes() -> None:
    now = datetime.now(tz=UTC)
    episode_a = Episode(
        uid="a",
        content="alpha",
        session_key="test-session",
        created_at=now,
        producer_id="unit-test",
        producer_role="assistant",
    )
    episode_b = Episode(
        uid="b",
        content="beta",
        session_key="test-session",
        created_at=now,
        producer_id="unit-test",
        producer_role="assistant",
    )
    state = TopLevelAgentSessionState.new(
        route_name="retrieve-agent",
        policy_name="retrieve-agent",
        query="hello",
    )
    state.merge_episodes([episode_a, episode_a])
    state.merge_episodes([episode_b])

    assert [episode.uid for episode in state.merged_episodes] == ["a", "b"]


@pytest.mark.asyncio
async def test_retrieve_agent_emits_session_state_metrics(
    query_policy: QueryPolicy,
    tmp_path,
) -> None:
    now = datetime.now(tz=UTC)
    episode = Episode(
        uid="state-episode",
        content="from-memory",
        session_key="test-session",
        created_at=now,
        producer_id="unit-test",
        producer_role="assistant",
    )
    retrieve_agent = RetrievalAgent(
        RetrievalAgentParams(
            model=FakeOpenAIInstalledAgentModel(
                [
                    openai_tool_call_response(query="hello"),
                    openai_text_response("retrieved"),
                ]
            ),
            extra_params={
                "agent_install_cache_path": tmp_path / ".agent-cache.json",
            },
            reranker=DummyReranker(),
        )
    )

    result, metrics = await retrieve_agent.do_query(
        query_policy,
        build_query_param(
            query="hello",
            rest_memory=FakeRestMemory({"hello": [episode]}),
        ),
    )

    assert result.episodic_memory is not None
    assert len(result.episodic_memory.long_term_memory.episodes) == 1
    assert metrics["route"] == "RetrievalAgent"
    assert metrics["orchestrator_step_count"] == 1
    assert metrics["orchestrator_sub_agent_count"] == 0
    assert metrics["orchestrator_event_count"] >= 2
    assert metrics["orchestrator_episode_count"] == 1
    assert metrics["orchestrator_completed"] is True
    assert metrics["branch_total"] == 0
    assert "rerank_applied" in metrics
