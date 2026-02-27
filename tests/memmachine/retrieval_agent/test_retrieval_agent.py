from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

import pytest

from memmachine.common.episode_store import Episode, EpisodeResponse
from memmachine.common.language_model.language_model import LanguageModel
from memmachine.common.reranker.reranker import Reranker
from memmachine.episodic_memory import EpisodicMemory
from memmachine.retrieval_agent.agents import (
    ChainOfQueryAgent,
    MemMachineAgent,
    SplitQueryAgent,
    ToolSelectAgent,
)
from memmachine.retrieval_agent.common.agent_api import (
    AgentToolBaseParam,
    QueryParam,
    QueryPolicy,
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
async def test_memmachine_agent_returns_episodes(query_policy: QueryPolicy) -> None:
    now = datetime.now(tz=UTC)
    episode = _build_episode(uid="e1", content="hello", created_at=now)
    memory = FakeEpisodicMemory({"hello": [episode]})
    reranker = DummyReranker()
    agent = MemMachineAgent(
        AgentToolBaseParam(
            model=None,
            children_tools=[],
            extra_params={},
            reranker=reranker,
        ),
    )

    result, metrics = await agent.do_query(
        query_policy,
        QueryParam(query="hello", limit=5, memory=memory),
    )

    assert result == [episode]
    assert metrics["memory_search_called"] == 1


@pytest.mark.asyncio
async def test_memmachine_agent_queries_long_term_memory_only(
    query_policy: QueryPolicy,
) -> None:
    now = datetime.now(tz=UTC)
    episode = _build_episode(uid="callback-e1", content="from-memory", created_at=now)
    memory = FakeEpisodicMemory({"callback-query": [episode]})
    agent = MemMachineAgent(
        AgentToolBaseParam(
            model=None,
            children_tools=[],
            extra_params={},
            reranker=DummyReranker(),
        ),
    )

    result, metrics = await agent.do_query(
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


@pytest.mark.asyncio
async def test_split_query_agent_aggregates_sub_queries(
    query_policy: QueryPolicy,
) -> None:
    now = datetime.now(tz=UTC)
    episode_a = _build_episode(uid="a", content="alpha", created_at=now)
    episode_b = _build_episode(
        uid="b",
        content="beta",
        created_at=now + timedelta(seconds=1),
    )
    memory = FakeEpisodicMemory({"Q1?": [episode_a], "Q2?": [episode_b]})
    reranker = DummyReranker()
    memory_agent = MemMachineAgent(
        AgentToolBaseParam(
            model=None,
            children_tools=[],
            extra_params={},
            reranker=reranker,
        ),
    )
    split_model = DummyLanguageModel("Q1?\nQ2?")
    split_agent = SplitQueryAgent(
        AgentToolBaseParam(
            model=split_model,
            children_tools=[memory_agent],
            extra_params={},
            reranker=reranker,
        ),
    )

    results, metrics = await split_agent.do_query(
        query_policy,
        QueryParam(query="original?", limit=10, memory=memory),
    )

    assert results == [episode_a, episode_b]
    assert metrics["queries"] == ["Q1?", "Q2?"]
    assert metrics["memory_search_called"] == 2


@pytest.mark.asyncio
async def test_tool_select_agent_uses_selected_tool(
    query_policy: QueryPolicy,
) -> None:
    now = datetime.now(tz=UTC)
    episode = _build_episode(uid="tool", content="tool-select", created_at=now)
    memory = FakeEpisodicMemory({"tool query": [episode]})
    reranker = DummyReranker()
    memory_agent = MemMachineAgent(
        AgentToolBaseParam(
            model=None,
            children_tools=[],
            extra_params={},
            reranker=reranker,
        ),
    )

    # LLM for the selector picks ChainOfQueryAgent directly.
    selector_model = DummyLanguageModel("ChainOfQueryAgent")
    split_agent = SplitQueryAgent(
        AgentToolBaseParam(
            model=DummyLanguageModel("unused"),
            children_tools=[memory_agent],
            extra_params={},
            reranker=reranker,
        ),
    )
    coq_agent = ChainOfQueryAgent(
        AgentToolBaseParam(
            model=DummyLanguageModel(
                '{"is_sufficient": true, "evidence_indices": [], "new_query": "", "confidence_score": 1.0}'
            ),
            children_tools=[memory_agent],
            extra_params={},
            reranker=reranker,
        ),
    )
    tool_select_agent = ToolSelectAgent(
        AgentToolBaseParam(
            model=selector_model,
            children_tools=[coq_agent, split_agent, memory_agent],
            extra_params={"default_tool_name": "MemMachineAgent"},
            reranker=reranker,
        ),
    )

    results, metrics = await tool_select_agent.do_query(
        query_policy,
        QueryParam(query="tool query", limit=5, memory=memory),
    )

    assert results == [episode]
    assert metrics["selected_tool"] == "ChainOfQueryAgent"
    assert selector_model.call_count == 1


@pytest.mark.asyncio
async def test_chain_of_query_agent_rewrites_and_accumulates_evidence(
    query_policy: QueryPolicy,
) -> None:
    now = datetime.now(tz=UTC)
    fact1 = _build_episode(uid="fact1", content="fact1", created_at=now)
    fact2 = _build_episode(
        uid="fact2",
        content="fact2",
        created_at=now + timedelta(seconds=1),
    )
    fact3 = _build_episode(
        uid="fact3",
        content="fact3",
        created_at=now + timedelta(seconds=2),
    )

    memory = FakeEpisodicMemory(
        {
            "original_query?": [fact1],
            "sub_query_1": [fact2],
            "sub_query_2": [fact3],
        },
    )
    reranker = DummyReranker()
    memory_agent = MemMachineAgent(
        AgentToolBaseParam(
            model=None,
            children_tools=[],
            extra_params={},
            reranker=reranker,
        ),
    )
    coq_model = DummyLanguageModel(
        [
            '{"is_sufficient": false, "evidence_indices": [0], "new_query": "sub_query_1", "confidence_score": 1.0}',
            '{"is_sufficient": false, "evidence_indices": [0, 1], "new_query": "sub_query_2", "confidence_score": 1.0}',
            '{"is_sufficient": true, "evidence_indices": [0, 1, 2], "new_query": "", "confidence_score": 1.0}',
        ]
    )
    coq_agent = ChainOfQueryAgent(
        AgentToolBaseParam(
            model=coq_model,
            children_tools=[memory_agent],
            extra_params={"max_attempts": 3},
            reranker=reranker,
        ),
    )

    results, metrics = await coq_agent.do_query(
        query_policy,
        QueryParam(query="original_query?", limit=10, memory=memory),
    )

    assert {episode.uid for episode in results} == {"fact1", "fact2", "fact3"}
    assert coq_model.call_count == 3
    assert memory.queries == ["original_query?", "sub_query_1", "sub_query_2"]
    assert metrics["queries"] == ["original_query?", "sub_query_1", "sub_query_2"]
    assert metrics["memory_search_called"] == 3


@pytest.mark.asyncio
async def test_chain_of_query_agent_handles_empty_query_without_retrieval(
    query_policy: QueryPolicy,
) -> None:
    memory = FakeEpisodicMemory({})
    reranker = DummyReranker()
    memory_agent = MemMachineAgent(
        AgentToolBaseParam(
            model=None,
            children_tools=[],
            extra_params={},
            reranker=reranker,
        ),
    )
    coq_model = DummyLanguageModel(
        '{"is_sufficient": true, "evidence_indices": [], "new_query": "", "confidence_score": 1.0}'
    )
    coq_agent = ChainOfQueryAgent(
        AgentToolBaseParam(
            model=coq_model,
            children_tools=[memory_agent],
            extra_params={"max_attempts": 3},
            reranker=reranker,
        ),
    )

    results, metrics = await coq_agent.do_query(
        query_policy,
        QueryParam(query="", limit=10, memory=memory),
    )

    assert results == []
    assert metrics["queries"] == []
    assert metrics["memory_search_called"] == 0
    assert coq_model.call_count == 0
    assert memory.queries == []


@pytest.mark.asyncio
async def test_rerank_logic(
    query_policy: QueryPolicy,
) -> None:
    now = datetime.now(tz=UTC)
    episode_a = _build_episode(
        uid="a",
        content="alpha",
        created_at=now + timedelta(seconds=1),
    )
    episode_b = _build_episode(
        uid="b",
        content="beta",
        created_at=now + timedelta(seconds=3),
    )
    episode_c = _build_episode(
        uid="c",
        content="gamma",
        created_at=now + timedelta(seconds=2),
    )
    memory_agent = MemMachineAgent(
        AgentToolBaseParam(
            model=None,
            children_tools=[],
            extra_params={},
        ),
    )

    reranker = DummyReranker([0.2, 0.9, 0.5])
    coq_model = DummyLanguageModel(
        [
            '{"is_sufficient": true, "evidence_indices": [0, 1, 2], "new_query": "", "confidence_score": 1.0}',
        ]
    )
    coq_agent = ChainOfQueryAgent(
        AgentToolBaseParam(
            model=coq_model,
            children_tools=[memory_agent],
            extra_params={"max_attempts": 3},
            reranker=reranker,
        ),
    )

    reranked = await coq_agent._do_rerank(
        QueryParam(query="rerank?", limit=2, memory=FakeEpisodicMemory({})),
        [episode_a, episode_b, episode_c],
    )

    # Top scores are episode_b (0.9) and episode_c (0.5); returned sorted by time.
    assert reranked == [episode_c, episode_b]
