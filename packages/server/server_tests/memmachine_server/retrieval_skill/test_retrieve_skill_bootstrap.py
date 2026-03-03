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
from memmachine_server.retrieval_skill.service_locator import create_retrieval_skill
from memmachine_server.retrieval_skill.skills.retrieve_skill import RetrieveSkill
from memmachine_server.retrieval_skill.subskills.direct_memory_skill import (
    MemMachineSkill,
)
from server_tests.memmachine_server.retrieval_skill.skill_session_stub import (
    ScriptedSkillSessionModel,
)


class DummyLanguageModel(LanguageModel):
    def __init__(self, response: str) -> None:
        self._response = response

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
        return self._response, None

    async def generate_response_with_token_usage(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> tuple[str, Any, int, int]:
        _ = args, kwargs
        return self._response, None, 1, 1


class FailingLanguageModel(DummyLanguageModel):
    async def generate_response(
        self,
        system_prompt: str | None = None,
        user_prompt: str | None = None,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, str] | None = None,
        max_attempts: int = 1,
    ) -> tuple[str, Any]:
        _ = system_prompt, user_prompt, tools, tool_choice, max_attempts
        raise RuntimeError("forced top-level model failure")

    async def generate_response_with_token_usage(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> tuple[str, Any, int, int]:
        _ = args, kwargs
        return (
            '{"selected_route":"direct_memory","confidence_score":0.93,"reason_code":"default"}',
            None,
            1,
            1,
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


def _build_episode(uid: str = "ep-1") -> Episode:
    return Episode(
        uid=uid,
        content="hello",
        session_key="test-session",
        created_at=datetime.now(tz=UTC),
        producer_id="test",
        producer_role="assistant",
    )


@pytest.mark.asyncio
async def test_retrieve_skill_bootstrap_entry_path(query_policy: QueryPolicy) -> None:
    episode = _build_episode()
    memory = FakeEpisodicMemory({"hello": [episode]})
    model = DummyLanguageModel("MemMachineSkill")
    skill = create_retrieval_skill(
        model=model,
        reranker=DummyReranker(),
        skill_session_model=ScriptedSkillSessionModel(model),
    )

    result, metrics = await skill.do_query(
        query_policy,
        QueryParam(query="hello", limit=5, memory=memory),
    )

    assert result == [episode]
    assert metrics["route"] == "RetrieveSkill"
    assert metrics["orchestrator_tool_call_count"] >= 1


@pytest.mark.asyncio
async def test_retrieve_skill_bootstrap_fallback_reason_for_errors(
    query_policy: QueryPolicy,
) -> None:
    episode = _build_episode(uid="fallback")
    memory = FakeEpisodicMemory({"hello": [episode]})
    fallback_tool = MemMachineSkill(
        SkillToolBaseParam(
            model=None,
            children_tools=[],
            extra_params={},
            reranker=DummyReranker(),
        ),
    )
    model = FailingLanguageModel("unused")
    retrieve_skill = RetrieveSkill(
        SkillToolBaseParam(
            model=model,
            children_tools=[fallback_tool],
            extra_params={
                "fallback_tool_name": "MemMachineSkill",
                "skill_session_model": ScriptedSkillSessionModel(model),
            },
            reranker=DummyReranker(),
        ),
    )

    result, metrics = await retrieve_skill.do_query(
        query_policy,
        QueryParam(query="hello", limit=5, memory=memory),
    )

    assert result == [episode]
    assert metrics["route"] == "RetrieveSkill"
    assert metrics["fallback_trigger_reason"] == "downstream_tool_failure"
    assert metrics["skill_contract_error_code"] == "SKILL_CONTRACT_DOWNSTREAM_FAILURE"


@pytest.mark.asyncio
async def test_retrieve_skill_bootstrap_invalid_entry_uses_fallback(
    query_policy: QueryPolicy,
) -> None:
    episode = _build_episode(uid="empty-query")
    memory = FakeEpisodicMemory({"": [episode]})
    model = DummyLanguageModel("MemMachineSkill")
    skill = create_retrieval_skill(
        model=model,
        reranker=DummyReranker(),
        skill_session_model=ScriptedSkillSessionModel(model),
    )

    result, metrics = await skill.do_query(
        query_policy,
        QueryParam(query="", limit=5, memory=memory),
    )

    assert result == [episode]
    assert metrics["route"] == "RetrieveSkill"
    assert metrics["fallback_trigger_reason"] == "invalid_skill_request"
    assert metrics["skill_contract_error_code"] == "SKILL_CONTRACT_INVALID_REQUEST"
