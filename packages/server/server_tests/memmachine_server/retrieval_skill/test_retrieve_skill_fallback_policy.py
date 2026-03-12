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


class PolicyLanguageModel(LanguageModel):
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


def _build_episode(uid: str, content: str) -> Episode:
    return Episode(
        uid=uid,
        content=content,
        session_key="test-session",
        created_at=datetime.now(tz=UTC),
        producer_id="unit-test",
        producer_role="assistant",
    )


def _build_skill(model: LanguageModel, **extra_params: object) -> RetrieveSkill:
    reranker = DummyReranker()
    memory_tool = MemMachineSkill(
        SkillToolBaseParam(
            model=None,
            children_tools=[],
            extra_params={},
            reranker=reranker,
        )
    )
    params = {
        "fallback_tool_name": "MemMachineSkill",
        "skill_session_model": ScriptedSkillSessionModel(model),
        **extra_params,
    }
    return RetrieveSkill(
        SkillToolBaseParam(
            model=model,
            children_tools=[memory_tool],
            extra_params=params,
            reranker=reranker,
        )
    )


@pytest.mark.asyncio
async def test_max_steps_exceeded_retries_to_fallback(
    query_policy: QueryPolicy,
) -> None:
    fallback_episode = _build_episode("fb-steps", "fallback-steps")
    memory = FakeEpisodicMemory({"hello": [fallback_episode]})
    too_many_calls = [
        {
            "function": {
                "name": "memmachine_search",
                "arguments": {"query": "hello"},
            }
        }
        for _ in range(9)
    ]
    model = PolicyLanguageModel(outputs=[("attempt-1", too_many_calls)])

    skill = _build_skill(model)
    episodes, metrics = await skill.do_query(
        query_policy,
        QueryParam(query="hello", limit=5, memory=memory),
    )

    assert [item.uid for item in episodes] == ["fb-steps"]
    assert metrics["fallback_trigger_reason"] == "max_steps_exceeded"


@pytest.mark.asyncio
async def test_combined_call_budget_exceeded_triggers_fallback(
    query_policy: QueryPolicy,
) -> None:
    fallback_episode = _build_episode("budget-fallback", "budget-fallback")
    memory = FakeEpisodicMemory({"hello": [fallback_episode]})
    model = PolicyLanguageModel(
        outputs=[
            (
                "top-level",
                [
                    {
                        "function": {
                            "name": "memmachine_search",
                            "arguments": {"query": "hello"},
                        }
                    },
                    {
                        "function": {
                            "name": "memmachine_search",
                            "arguments": {"query": "hello again"},
                        }
                    },
                ],
            )
        ],
    )

    skill = _build_skill(model, max_combined_calls=1)
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
) -> None:
    fallback_episode = _build_episode("fb-invalid", "fallback-invalid")
    memory = FakeEpisodicMemory({"hello": [fallback_episode]})
    model = PolicyLanguageModel(
        outputs=[
            (
                "bad tool",
                [
                    {
                        "function": {
                            "name": "return_final",
                            "arguments": {"final_response": "done"},
                        }
                    }
                ],
            )
        ]
    )

    skill = _build_skill(model)
    episodes, metrics = await skill.do_query(
        query_policy,
        QueryParam(query="hello", limit=5, memory=memory),
    )

    assert [item.uid for item in episodes] == ["fb-invalid"]
    assert metrics["fallback_trigger_reason"] == "invalid_tool_call"
    error_diagnostics = metrics.get("error_diagnostics")
    assert isinstance(error_diagnostics, dict)
    assert error_diagnostics.get("context") == "top_level_tool_not_found"
