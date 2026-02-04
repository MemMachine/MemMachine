from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

import pytest

from memmachine.common.language_model.language_model import LanguageModel
from memmachine.common.reranker.reranker import Reranker
from memmachine.episodic_memory.declarative_memory import DeclarativeMemory
from memmachine.episodic_memory.declarative_memory.data_types import (
    ContentType,
    Episode,
)
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

    async def generate_parsed_response(  # type: ignore[override]
        self,
        output_format: type[Any],
        system_prompt: str | None = None,
        user_prompt: str | None = None,
        max_attempts: int = 1,
    ) -> Any | None:
        return None

    async def generate_response(  # type: ignore[override]
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


class FakeDeclarativeMemory(DeclarativeMemory):
    """DeclarativeMemory stub that returns preset episodes by query."""

    def __init__(self, episodes_by_query: dict[str, list[Episode]]) -> None:
        self._episodes_by_query = episodes_by_query
        self.queries: list[str] = []

    async def search_scored(
        self,
        query: str,
        *,
        max_num_episodes: int = 20,
        expand_context: int = 0,
        property_filter: Any | None = None,
    ) -> list[tuple[float, Episode]]:
        self.queries.append(query)
        episodes = self._episodes_by_query.get(query, [])
        return [(1.0, episode) for episode in episodes[:max_num_episodes]]


@pytest.fixture
def query_policy() -> QueryPolicy:
    return QueryPolicy(
        token_cost=0,
        time_cost=0,
        accuracy_score=0.0,
        confidence_score=0.0,
    )


@pytest.mark.asyncio
async def test_memmachine_agent_returns_episodes(query_policy: QueryPolicy) -> None:
    now = datetime.now(tz=UTC)
    episode = Episode(
        uid="e1",
        timestamp=now,
        source="unit-test",
        content_type=ContentType.TEXT,
        content="hello",
    )
    memory = FakeDeclarativeMemory({"hello": [episode]})
    reranker = DummyReranker()
    agent = MemMachineAgent(
        AgentToolBaseParam(
            model=None,
            children_tools=[],
            extra_params={"memory": memory},
            reranker=reranker,
        ),
    )

    result, metrics = await agent.do_query(
        query_policy,
        QueryParam(query="hello", limit=5),
    )

    assert result == [episode]
    assert metrics["memory_retrieval_time"] > 0.0


@pytest.mark.asyncio
async def test_split_query_agent_aggregates_sub_queries(
    query_policy: QueryPolicy,
) -> None:
    now = datetime.now(tz=UTC)
    episode_a = Episode(
        uid="a",
        timestamp=now,
        source="unit-test",
        content_type=ContentType.TEXT,
        content="alpha",
    )
    episode_b = Episode(
        uid="b",
        timestamp=now + timedelta(seconds=1),
        source="unit-test",
        content_type=ContentType.TEXT,
        content="beta",
    )
    memory = FakeDeclarativeMemory({"Q1?": [episode_a], "Q2?": [episode_b]})
    reranker = DummyReranker()
    memory_agent = MemMachineAgent(
        AgentToolBaseParam(
            model=None,
            children_tools=[],
            extra_params={"memory": memory},
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
        QueryParam(query="original?", limit=10),
    )

    assert results == [episode_a, episode_b]
    assert metrics["queries"] == ["Q1?", "Q2?"]


@pytest.mark.asyncio
async def test_tool_select_agent_uses_selected_tool(
    query_policy: QueryPolicy,
) -> None:
    now = datetime.now(tz=UTC)
    episode = Episode(
        uid="tool",
        timestamp=now,
        source="unit-test",
        content_type=ContentType.TEXT,
        content="tool-select",
    )
    memory = FakeDeclarativeMemory({"tool query": [episode]})
    reranker = DummyReranker()
    memory_agent = MemMachineAgent(
        AgentToolBaseParam(
            model=None,
            children_tools=[],
            extra_params={"memory": memory},
            reranker=reranker,
        ),
    )

    # LLM for the selector picks MemMachineAgent directly.
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
        QueryParam(query="tool query", limit=5),
    )

    assert results == [episode]
    assert metrics["selected_tool"] == "ChainOfQueryAgent"
    assert selector_model.call_count == 1


@pytest.mark.asyncio
async def test_chain_of_query_agent_rewrites_and_accumulates_evidence(
    query_policy: QueryPolicy,
) -> None:
    now = datetime.now(tz=UTC)
    fact1 = Episode(
        uid="fact1",
        timestamp=now,
        source="unit-test",
        content_type=ContentType.TEXT,
        content="fact1",
    )
    fact2 = Episode(
        uid="fact2",
        timestamp=now + timedelta(seconds=1),
        source="unit-test",
        content_type=ContentType.TEXT,
        content="fact2",
    )
    fact3 = Episode(
        uid="fact3",
        timestamp=now + timedelta(seconds=2),
        source="unit-test",
        content_type=ContentType.TEXT,
        content="fact3",
    )

    memory = FakeDeclarativeMemory(
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
            extra_params={"memory": memory},
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
        QueryParam(query="original_query?", limit=10),
    )

    assert {episode.uid for episode in results} == {"fact1", "fact2", "fact3"}
    assert coq_model.call_count == 3
    assert memory.queries == ["original_query?", "sub_query_1", "sub_query_2"]
    assert metrics["queries"] == ["original_query?", "sub_query_1", "sub_query_2"]


@pytest.mark.asyncio
async def test_rerank_logic(
    query_policy: QueryPolicy,
) -> None:
    now = datetime.now(tz=UTC)
    episode_a = Episode(
        uid="a",
        timestamp=now + timedelta(seconds=1),
        source="unit-test",
        content_type=ContentType.TEXT,
        content="alpha",
    )
    episode_b = Episode(
        uid="b",
        timestamp=now + timedelta(seconds=3),
        source="unit-test",
        content_type=ContentType.TEXT,
        content="beta",
    )
    episode_c = Episode(
        uid="c",
        timestamp=now + timedelta(seconds=2),
        source="unit-test",
        content_type=ContentType.TEXT,
        content="gamma",
    )
    memory = FakeDeclarativeMemory({"rerank?": [episode_a, episode_b, episode_c]})
    memory_agent = MemMachineAgent(
        AgentToolBaseParam(
            model=None,
            children_tools=[],
            extra_params={"memory": memory},
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
        QueryParam(query="rerank?", limit=2),
        [episode_a, episode_b, episode_c],
    )

    # Top scores are episode_b (0.9) and episode_c (0.5); returned sorted by time.
    assert reranked == [episode_c, episode_b]
