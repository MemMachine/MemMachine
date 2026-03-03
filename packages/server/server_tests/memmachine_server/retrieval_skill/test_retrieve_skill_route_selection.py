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
async def test_coq_sub_skill_summary_sets_sufficiency_metrics(
    query_policy: QueryPolicy,
) -> None:
    episode = _build_episode("route-coq")
    memory = FakeEpisodicMemory({"hello": [episode]})
    model = ScriptedLanguageModel(
        outputs=[
            (
                "top-level",
                [
                    {
                        "function": {
                            "name": "spawn_sub_skill",
                            "arguments": {
                                "skill_name": "coq",
                                "query": "hello",
                                "rationale": "dependent multi-hop",
                            },
                        }
                    },
                    {
                        "function": {
                            "name": "return_final",
                            "arguments": {"final_response": "ok"},
                        }
                    },
                ],
            ),
            (
                "coq",
                [
                    {
                        "function": {
                            "name": "return_sub_skill_result",
                            "arguments": {
                                "summary": (
                                    '{"is_sufficient":true,'
                                    '"confidence_score":0.91,'
                                    '"reason_code":"coq_sufficient",'
                                    '"answer_candidate":"alice"}'
                                )
                            },
                        }
                    }
                ],
            ),
        ]
    )

    skill = _build_skill(model)
    episodes, metrics = await skill.do_query(
        query_policy,
        QueryParam(query="hello", limit=5, memory=memory),
    )

    assert [item.uid for item in episodes] == ["route-coq"]
    assert metrics["latest_sufficiency_signal_skill"] == "coq"
    assert metrics["evidence_sufficient"] is True
    assert metrics["top_level_is_sufficient"] is True
    assert metrics["selected_skill"] == "coq"
    assert metrics["selected_skill_name"] == "ChainOfQuerySkill"


@pytest.mark.asyncio
async def test_split_sub_skill_is_reflected_in_selected_skill_name(
    query_policy: QueryPolicy,
) -> None:
    episode = _build_episode("route-split")
    memory = FakeEpisodicMemory({"hello": [episode]})
    model = ScriptedLanguageModel(
        outputs=[
            (
                "top-level",
                [
                    {
                        "function": {
                            "name": "spawn_sub_skill",
                            "arguments": {
                                "skill_name": "split",
                                "query": "hello",
                                "rationale": "independent branches",
                            },
                        }
                    },
                    {
                        "function": {
                            "name": "return_final",
                            "arguments": {
                                "final_response": "ok",
                                "is_sufficient": False,
                            },
                        }
                    },
                ],
            ),
            (
                "split",
                [
                    {
                        "function": {
                            "name": "return_sub_skill_result",
                            "arguments": {
                                "summary": (
                                    '{"sub_queries":["a?","b?"],'
                                    '"reason_code":"split_needed",'
                                    '"reason_note":"parallel lookups"}'
                                )
                            },
                        }
                    }
                ],
            ),
        ]
    )

    skill = _build_skill(model)
    episodes, metrics = await skill.do_query(
        query_policy,
        QueryParam(query="hello", limit=5, memory=memory),
    )

    assert episodes == []
    assert metrics["selected_skill"] == "split"
    assert metrics["selected_skill_name"] == "SplitSkill"


@pytest.mark.asyncio
async def test_removed_tool_select_sub_skill_is_rejected_by_runtime(
    query_policy: QueryPolicy,
) -> None:
    fallback_episode = _build_episode("fallback-removed")
    memory = FakeEpisodicMemory({"hello": [fallback_episode]})
    model = ScriptedLanguageModel(
        outputs=[
            (
                "top-level",
                [
                    {
                        "function": {
                            "name": "spawn_sub_skill",
                            "arguments": {
                                "skill_name": "tool_select",
                                "query": "hello",
                                "rationale": "legacy route selection",
                            },
                        }
                    },
                    {
                        "function": {
                            "name": "return_final",
                            "arguments": {"final_response": "ok"},
                        }
                    },
                ],
            ),
        ]
    )

    skill = _build_skill(model)
    episodes, metrics = await skill.do_query(
        query_policy,
        QueryParam(query="hello", limit=5, memory=memory),
    )

    assert [item.uid for item in episodes] == ["fallback-removed"]
    assert metrics["fallback_trigger_reason"] == "invalid_tool_call"
