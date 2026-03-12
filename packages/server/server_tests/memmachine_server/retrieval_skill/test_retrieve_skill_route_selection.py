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
from memmachine_server.retrieval_skill.subskills.direct_memory_skill import (
    MemMachineSkill,
)
from server_tests.memmachine_server.retrieval_skill.skill_session_stub import (
    ScriptedSkillSessionModel,
)


class ScriptedLanguageModel(LanguageModel):
    def __init__(
        self,
        *,
        outputs: list[tuple[str, list[dict[str, Any]] | None]],
    ) -> None:
        self._outputs = outputs

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
        if not self._outputs:
            return "", []
        return self._outputs.pop(0)

    async def generate_response_with_token_usage(
        self,
        system_prompt: str | None = None,
        user_prompt: str | None = None,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, str] | None = None,
        max_attempts: int = 1,
    ) -> tuple[str, Any, int, int]:
        _ = system_prompt, user_prompt, tools, tool_choice, max_attempts
        return "", None, 0, 0


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


def _build_skill(model: LanguageModel) -> RetrieveSkill:
    reranker = DummyReranker()
    memory_tool = MemMachineSkill(
        SkillToolBaseParam(
            model=None,
            children_tools=[],
            extra_params={},
            reranker=reranker,
        )
    )
    return RetrieveSkill(
        SkillToolBaseParam(
            model=model,
            children_tools=[memory_tool],
            extra_params={
                "fallback_tool_name": "MemMachineSkill",
                "skill_session_model": ScriptedSkillSessionModel(model),
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
async def test_selected_skill_defaults_to_direct_memory_label(
    query_policy: QueryPolicy,
) -> None:
    episode = _build_episode("route-direct")
    memory = FakeEpisodicMemory({"hello": [episode]})
    model = ScriptedLanguageModel(
        outputs=[
            (
                "final answer",
                [
                    {
                        "function": {
                            "name": "memmachine_search",
                            "arguments": {"query": "hello"},
                        }
                    }
                ],
            )
        ]
    )

    skill = _build_skill(model)
    _, metrics = await skill.do_query(
        query_policy,
        QueryParam(query="hello", limit=5, memory=memory),
    )

    assert metrics["selected_skill"] == "direct_memory"
    assert metrics["selected_skill_name"] == "MemMachineSkill"
    assert metrics["orchestrator_sub_skill_runs"] == []


@pytest.mark.asyncio
async def test_plain_text_completion_keeps_raw_memory_episodes(
    query_policy: QueryPolicy,
) -> None:
    episode_a = _build_episode("route-a")
    episode_b = _build_episode("route-b")
    memory = FakeEpisodicMemory({"branch a": [episode_a], "branch b": [episode_b]})
    model = ScriptedLanguageModel(
        outputs=[
            (
                "combined answer",
                [
                    {
                        "function": {
                            "name": "memmachine_search",
                            "arguments": {"query": "branch a"},
                        }
                    },
                    {
                        "function": {
                            "name": "memmachine_search",
                            "arguments": {"query": "branch b"},
                        }
                    },
                ],
            )
        ]
    )

    skill = _build_skill(model)
    episodes, metrics = await skill.do_query(
        query_policy,
        QueryParam(query="hello", limit=5, memory=memory),
    )

    assert [item.uid for item in episodes] == ["route-a", "route-b"]
    assert metrics["stage_result_memory_returned"] is False
    assert metrics["memory_search_called"] == 2
