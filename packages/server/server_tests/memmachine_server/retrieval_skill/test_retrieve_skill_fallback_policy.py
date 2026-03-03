from __future__ import annotations

import asyncio
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
        sub_skill_delay_seconds: float = 0.0,
    ) -> None:
        self._outputs = outputs
        self._sub_skill_delay_seconds = sub_skill_delay_seconds

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
        _ = system_prompt, user_prompt, tool_choice, max_attempts
        tool_names = {
            str(item.get("function", {}).get("name", ""))
            for item in (tools or [])
            if isinstance(item, dict)
        }
        if "memmachine_search" in tool_names and self._sub_skill_delay_seconds > 0:
            await asyncio.sleep(self._sub_skill_delay_seconds)
            return "", []

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


def _spawn_direct_memory_call(query: str = "hello") -> dict[str, Any]:
    return {
        "function": {
            "name": "spawn_sub_skill",
            "arguments": {
                "skill_name": "direct_memory",
                "query": query,
                "rationale": "branch",
            },
        }
    }


def _spawn_split_call(query: str = "hello") -> dict[str, Any]:
    return {
        "function": {
            "name": "spawn_sub_skill",
            "arguments": {
                "skill_name": "split",
                "query": query,
                "rationale": "split",
            },
        }
    }


def _return_final_call() -> dict[str, Any]:
    return {
        "function": {
            "name": "return_final",
            "arguments": {"final_response": "done"},
        }
    }


@pytest.mark.asyncio
async def test_sub_skill_timeout_retries_once_then_fallback(
    query_policy: QueryPolicy,
) -> None:
    fallback_episode = _build_episode("fb-timeout", "fallback-timeout")
    memory = FakeEpisodicMemory({"hello": [fallback_episode]})
    model = PolicyLanguageModel(
        outputs=[
            ("attempt-1", [_spawn_direct_memory_call("hello"), _return_final_call()]),
            ("attempt-2", [_spawn_direct_memory_call("hello"), _return_final_call()]),
        ],
        sub_skill_delay_seconds=0.01,
    )

    skill = _build_skill(model, sub_skill_timeout_seconds=0)
    episodes, metrics = await skill.do_query(
        query_policy,
        QueryParam(query="hello", limit=5, memory=memory),
    )

    assert [item.uid for item in episodes] == ["fb-timeout"]
    assert metrics["fallback_trigger_reason"] == "sub_skill_timeout"
    trace = metrics["orchestrator_trace"]
    assert isinstance(trace, dict)
    assert any(
        event.get("event_type") == "guardrail_retry"
        for event in trace.get("events", [])
    )


@pytest.mark.asyncio
async def test_max_steps_exceeded_retries_once_then_fallback(
    query_policy: QueryPolicy,
) -> None:
    fallback_episode = _build_episode("fb-steps", "fallback-steps")
    memory = FakeEpisodicMemory({"hello": [fallback_episode]})
    too_many_calls = [
        {
            "function": {
                "name": "direct_memory_search",
                "arguments": {"query": "hello"},
            }
        }
        for _ in range(9)
    ]
    model = PolicyLanguageModel(
        outputs=[("attempt-1", too_many_calls), ("attempt-2", too_many_calls)],
    )

    skill = _build_skill(model)
    episodes, metrics = await skill.do_query(
        query_policy,
        QueryParam(query="hello", limit=5, memory=memory),
    )

    assert [item.uid for item in episodes] == ["fb-steps"]
    assert metrics["fallback_trigger_reason"] == "max_steps_exceeded"


@pytest.mark.asyncio
async def test_fallback_preserves_partial_evidence_before_timeout(
    query_policy: QueryPolicy,
) -> None:
    partial_episode = _build_episode("partial", "partial-evidence")
    fallback_episode = _build_episode("final", "fallback-evidence")
    memory = FakeEpisodicMemory({"hello": [partial_episode, fallback_episode]})
    model = PolicyLanguageModel(
        outputs=[
            (
                "attempt-1",
                [
                    {
                        "function": {
                            "name": "direct_memory_search",
                            "arguments": {"query": "hello"},
                        }
                    },
                    _spawn_direct_memory_call("hello"),
                    _return_final_call(),
                ],
            ),
            ("attempt-2", [_spawn_direct_memory_call("hello"), _return_final_call()]),
        ],
        sub_skill_delay_seconds=0.01,
    )

    skill = _build_skill(model, sub_skill_timeout_seconds=0)
    episodes, metrics = await skill.do_query(
        query_policy,
        QueryParam(query="hello", limit=5, memory=memory),
    )

    assert metrics["fallback_trigger_reason"] == "sub_skill_timeout"
    uids = [item.uid for item in episodes]
    assert "partial" in uids
    assert "final" in uids


@pytest.mark.asyncio
async def test_split_branch_failure_triggers_fallback(
    query_policy: QueryPolicy,
) -> None:
    fallback_episode = _build_episode("split-fallback", "split-fallback")
    memory = FakeEpisodicMemory({"hello": [fallback_episode]})
    model = PolicyLanguageModel(
        outputs=[
            (
                "top-level",
                [
                    _spawn_split_call("hello"),
                    {
                        "function": {
                            "name": "spawn_sub_skill",
                            "arguments": {
                                "skill_name": "coq",
                                "query": "branch then detail",
                            },
                        }
                    },
                    _return_final_call(),
                ],
            ),
            (
                "split-planner",
                [
                    {
                        "function": {
                            "name": "return_sub_skill_result",
                            "arguments": {
                                "summary": '{"sub_queries":["branch then detail"]}'
                            },
                        }
                    }
                ],
            ),
            (
                "coq-attempt-1",
                [{"function": {"name": "unknown_tool", "arguments": {}}}],
            ),
        ],
    )

    skill = _build_skill(model)
    episodes, metrics = await skill.do_query(
        query_policy,
        QueryParam(query="hello", limit=5, memory=memory),
    )

    assert [item.uid for item in episodes] == ["split-fallback"]
    assert metrics["fallback_trigger_reason"] == "sub_skill_exception"
