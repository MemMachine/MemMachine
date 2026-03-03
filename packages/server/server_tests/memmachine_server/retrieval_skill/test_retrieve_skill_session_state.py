from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import pytest

from memmachine_server.common.episode_store import Episode, EpisodeResponse
from memmachine_server.common.language_model.language_model import LanguageModel
from memmachine_server.common.reranker.reranker import Reranker
from memmachine_server.episodic_memory import EpisodicMemory
from memmachine_server.retrieval_skill.common.skill_api import (
    QueryParam,
    QueryPolicy,
    SkillToolBaseParam,
)
from memmachine_server.retrieval_skill.skills.retrieve_skill import RetrieveSkill
from memmachine_server.retrieval_skill.skills.session_state import (
    TopLevelSkillSessionState,
)
from memmachine_server.retrieval_skill.subskills.direct_memory_skill import (
    MemMachineSkill,
)
from server_tests.memmachine_server.retrieval_skill.skill_session_stub import (
    ScriptedSkillSessionModel,
)


class DummyLanguageModel(LanguageModel):
    async def generate_parsed_response(
        self,
        output_format: type[Any],
        system_prompt: str | None = None,
        user_prompt: str | None = None,
        max_attempts: int = 1,
    ) -> Any | None:
        _ = output_format, system_prompt, user_prompt, max_attempts
        return None

    async def generate_response(
        self,
        system_prompt: str | None = None,
        user_prompt: str | None = None,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, str] | None = None,
        max_attempts: int = 1,
    ) -> tuple[str, Any]:
        _ = system_prompt, user_prompt, tools, tool_choice, max_attempts
        return "", []

    async def generate_response_with_token_usage(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> tuple[str, Any, int, int]:
        _ = args, kwargs
        return (
            '{"selected_route":"direct_memory","confidence_score":0.92,"reason_code":"default"}',
            None,
            0,
            0,
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
    state = TopLevelSkillSessionState.new(
        route_name="retrieve-skill",
        policy_name="retrieve-skill",
        query="hello",
    )
    state.merge_episodes([episode_a, episode_a])
    state.merge_episodes([episode_b])

    assert [episode.uid for episode in state.merged_episodes] == ["a", "b"]


@pytest.mark.asyncio
async def test_retrieve_skill_emits_session_state_metrics(
    query_policy: QueryPolicy,
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
    memory_tool = MemMachineSkill(
        SkillToolBaseParam(
            model=None,
            children_tools=[],
            extra_params={},
            reranker=DummyReranker(),
        )
    )
    model = DummyLanguageModel()
    retrieve_skill = RetrieveSkill(
        SkillToolBaseParam(
            model=model,
            children_tools=[memory_tool],
            extra_params={
                "fallback_tool_name": "MemMachineSkill",
                "skill_session_model": ScriptedSkillSessionModel(model),
            },
            reranker=DummyReranker(),
        )
    )

    episodes, metrics = await retrieve_skill.do_query(
        query_policy,
        QueryParam(
            query="hello",
            limit=5,
            memory=FakeEpisodicMemory({"hello": [episode]}),
        ),
    )

    assert len(episodes) == 1
    assert metrics["route"] == "RetrieveSkill"
    assert metrics["orchestrator_step_count"] == 1
    assert metrics["orchestrator_sub_skill_count"] == 0
    assert metrics["orchestrator_event_count"] >= 2
    assert metrics["orchestrator_episode_count"] == 1
    assert metrics["orchestrator_completed"] is True
    assert metrics["branch_total"] == 0
    assert "rerank_applied" in metrics
