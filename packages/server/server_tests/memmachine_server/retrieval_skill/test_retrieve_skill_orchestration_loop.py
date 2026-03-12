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
        scripted_responses: list[tuple[str, list[dict[str, Any]] | None]],
    ) -> None:
        self._scripted_responses = scripted_responses

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
        if not self._scripted_responses:
            return "", []
        return self._scripted_responses.pop(0)

    async def generate_response_with_token_usage(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> tuple[str, Any, int, int]:
        _ = args, kwargs
        return "", None, 0, 0


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
) -> None:
    episode = _build_episode("ep-direct", "direct")
    memory = FakeEpisodicMemory({"hello": [episode]})
    model = ScriptedLanguageModel(
        [
            (
                "finalized",
                [
                    {
                        "function": {
                            "name": "memmachine_search",
                            "arguments": {
                                "query": "hello",
                                "rationale": "fetch evidence",
                            },
                        }
                    }
                ],
            )
        ]
    )
    retrieve_skill = _build_skill(model)

    episodes, metrics = await retrieve_skill.do_query(
        query_policy,
        QueryParam(query="hello", limit=5, memory=memory),
    )

    assert [item.uid for item in episodes] == ["ep-direct"]
    assert metrics["route"] == "RetrieveSkill"
    assert metrics["orchestrator_completed"] is True
    assert metrics["orchestrator_tool_call_count"] == 1
    assert metrics["orchestrator_sub_skill_count"] == 0
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
) -> None:
    episode = _build_episode("ep-fallback-search", "fallback")
    memory = FakeEpisodicMemory({"hello": [episode]})
    model = ScriptedLanguageModel([("plain response without tool call", [])])
    retrieve_skill = _build_skill(model)

    episodes, metrics = await retrieve_skill.do_query(
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
) -> None:
    episode = _build_episode("ep-unclear", "unclear evidence")
    memory = FakeEpisodicMemory({"hello": [episode]})
    model = ScriptedLanguageModel(
        [
            (
                "I don't know based on the retrieved evidence.",
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
    retrieve_skill = _build_skill(model)

    _, metrics = await retrieve_skill.do_query(
        query_policy,
        QueryParam(query="hello", limit=5, memory=memory),
    )

    assert metrics["top_level_is_sufficient"] is False
    assert "answer_candidate" not in metrics


@pytest.mark.asyncio
async def test_legacy_tool_name_triggers_fallback(
    query_policy: QueryPolicy,
) -> None:
    fallback_episode = _build_episode("ep-fallback", "fallback")
    memory = FakeEpisodicMemory({"hello": [fallback_episode]})
    model = ScriptedLanguageModel(
        [
            (
                "bad tool",
                [
                    {
                        "function": {
                            "name": "spawn_sub_skill",
                            "arguments": {"skill_name": "coq", "query": "hello"},
                        }
                    }
                ],
            )
        ]
    )
    retrieve_skill = _build_skill(model)

    episodes, metrics = await retrieve_skill.do_query(
        query_policy,
        QueryParam(query="hello", limit=5, memory=memory),
    )

    assert [item.uid for item in episodes] == ["ep-fallback"]
    assert metrics["fallback_trigger_reason"] == "invalid_tool_call"
    assert metrics["orchestrator_trace"]["tool_calls"][0]["tool_name"] == (
        "memmachine_search"
    )
