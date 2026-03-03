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
from memmachine_server.retrieval_skill.subskills import MemMachineSkill
from server_tests.memmachine_server.retrieval_skill.skill_session_stub import (
    ScriptedSkillSessionModel,
)


class DummyLanguageModel(LanguageModel):
    """Lightweight language model stub for unit tests."""

    def __init__(self, responses: list[str] | str) -> None:
        if isinstance(responses, str):
            responses = [responses]
        self._responses = responses
        self.call_count = 0

    async def generate_parsed_response(
        self,
        output_format: type[Any],
        system_prompt: str | None = None,
        user_prompt: str | None = None,
        max_attempts: int = 1,
    ) -> Any | None:
        return None

    async def generate_response(
        self,
        system_prompt: str | None = None,
        user_prompt: str | None = None,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, str] | None = None,
        max_attempts: int = 1,
    ) -> tuple[str, Any]:
        return "", None

    async def generate_response_with_token_usage(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> tuple[str, Any, int, int]:
        idx = min(self.call_count, len(self._responses) - 1)
        response = self._responses[idx]
        self.call_count += 1
        return response, None, 1, 1


class DummyReranker(Reranker):
    """Reranker stub returning predefined scores."""

    def __init__(self, scores: list[float] | None = None) -> None:
        self._scores = scores or []
        self.call_count = 0

    async def score(self, query: str, candidates: list[str]) -> list[float]:
        self.call_count += 1
        if self._scores and len(self._scores) == len(candidates):
            return list(self._scores)
        return [float(len(candidates) - idx) for idx in range(len(candidates))]


class FakeEpisodicMemory(EpisodicMemory):
    """EpisodicMemory stub that returns preset long-term episodes by query."""

    def __init__(self, episodes_by_query: dict[str, list[Episode]]) -> None:
        self._episodes_by_query = episodes_by_query
        self._session_key = "test-session"
        self.queries: list[str] = []
        self.calls: list[dict[str, Any]] = []

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
        self.queries.append(query)
        self.calls.append(
            {
                "query": query,
                "limit": limit,
                "expand_context": expand_context,
                "score_threshold": score_threshold,
                "property_filter": property_filter,
                "mode": mode,
            }
        )
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


def _build_episode(*, uid: str, content: str, created_at: datetime) -> Episode:
    return Episode(
        uid=uid,
        content=content,
        session_key="test-session",
        created_at=created_at,
        producer_id="unit-test",
        producer_role="assistant",
    )


@pytest.mark.asyncio
async def test_memmachine_skill_returns_episodes(query_policy: QueryPolicy) -> None:
    now = datetime.now(tz=UTC)
    episode = _build_episode(uid="e1", content="hello", created_at=now)
    memory = FakeEpisodicMemory({"hello": [episode]})
    reranker = DummyReranker()
    skill = MemMachineSkill(
        SkillToolBaseParam(
            model=None,
            children_tools=[],
            extra_params={},
            reranker=reranker,
        ),
    )

    result, metrics = await skill.do_query(
        query_policy,
        QueryParam(query="hello", limit=5, memory=memory),
    )

    assert result == [episode]
    assert metrics["memory_search_called"] == 1


@pytest.mark.asyncio
async def test_memmachine_skill_queries_long_term_memory_only(
    query_policy: QueryPolicy,
) -> None:
    now = datetime.now(tz=UTC)
    episode = _build_episode(uid="callback-e1", content="from-memory", created_at=now)
    memory = FakeEpisodicMemory({"callback-query": [episode]})
    skill = MemMachineSkill(
        SkillToolBaseParam(
            model=None,
            children_tools=[],
            extra_params={},
            reranker=DummyReranker(),
        ),
    )

    result, metrics = await skill.do_query(
        query_policy,
        QueryParam(
            query="callback-query",
            limit=3,
            expand_context=2,
            score_threshold=0.55,
            memory=memory,
        ),
    )

    assert result == [episode]
    assert metrics["memory_search_called"] == 1
    assert memory.calls == [
        {
            "query": "callback-query",
            "limit": 3,
            "expand_context": 2,
            "score_threshold": 0.55,
            "property_filter": None,
            "mode": EpisodicMemory.QueryMode.LONG_TERM_ONLY,
        }
    ]


def test_service_locator_defaults_to_retrieve_skill() -> None:
    model = DummyLanguageModel("MemMachineSkill")
    skill = create_retrieval_skill(
        model=model,
        reranker=DummyReranker(),
        skill_session_model=ScriptedSkillSessionModel(model),
    )
    assert skill.skill_name == "RetrieveSkill"


def test_service_locator_requires_openai_or_explicit_session_model() -> None:
    with pytest.raises(TypeError, match="requires OpenAIResponsesLanguageModel"):
        _ = create_retrieval_skill(
            model=DummyLanguageModel("MemMachineSkill"),
            reranker=DummyReranker(),
        )


def test_service_locator_ignores_legacy_skill_routes() -> None:
    model = DummyLanguageModel("MemMachineSkill")
    reranker = DummyReranker()
    session_model = ScriptedSkillSessionModel(model)
    for skill_name in [
        "MemMachineSkill",
        "SplitSkill",
        "ChainOfQuerySkill",
        "ToolSelectSkill",
        "RetrieveSkill",
    ]:
        skill = create_retrieval_skill(
            model=model,
            reranker=reranker,
            skill_name=skill_name,
            skill_session_model=session_model,
        )
        assert skill.skill_name == "RetrieveSkill"
