"""Deterministic contracts for bounded progressive episodic retrieval."""

from __future__ import annotations

import asyncio
import json
from datetime import UTC, datetime, timedelta
from typing import Any
from unittest.mock import AsyncMock

import pytest

from memmachine_server.common.configuration.retrieval_config import (
    ProgressiveRetrievalConf,
)
from memmachine_server.common.episode_store import Episode, EpisodeResponse
from memmachine_server.common.filter.filter_parser import Comparison
from memmachine_server.episodic_memory import EpisodicMemory
from memmachine_server.retrieval_agent.agents import MemMachineAgent
from memmachine_server.retrieval_agent.agents.progressive_query_agent import (
    ProgressiveQueryAgent,
)
from memmachine_server.retrieval_agent.common.agent_api import (
    AgentToolBaseParam,
    QueryParam,
    QueryPolicy,
)

from .test_retrieval_agent import (
    DummyLanguageModel,
    DummyReranker,
    FakeEpisodicMemory,
    _build_episode,
)

pytestmark = pytest.mark.asyncio


def _episodes(count: int, prefix: str = "fact") -> list[Episode]:
    return [
        _build_episode(
            uid=f"{prefix}-{index}",
            content=f"{prefix}-{index}",
            created_at=datetime(2026, 1, 1, tzinfo=UTC) + timedelta(seconds=index),
        )
        for index in range(count)
    ]


def _decision(sufficient: bool = False, **changes: Any) -> str:
    return json.dumps(
        {
            "is_sufficient": sufficient,
            "evidence_indices": [0],
            "confidence_score": 0.9,
            "new_query": "question",
            **changes,
        }
    )


def _agent(
    responses: str | list[str], **config: Any
) -> tuple[ProgressiveQueryAgent, DummyLanguageModel, DummyReranker]:
    model = DummyLanguageModel(responses)
    reranker = DummyReranker()
    agent = ProgressiveQueryAgent(
        AgentToolBaseParam(
            model=model,
            reranker=reranker,
            children_tools=[MemMachineAgent(AgentToolBaseParam())],
        ),
        ProgressiveRetrievalConf(**config),
    )
    return agent, model, reranker


@pytest.fixture
def policy() -> QueryPolicy:
    return QueryPolicy(token_cost=0, time_cost=0, accuracy_score=0, confidence_score=0)


class RoundMemory(FakeEpisodicMemory):
    """Return complete batches, including neighbors beyond the seed limit."""

    def __init__(self, batches: list[list[Episode]]) -> None:
        super().__init__({})
        self.batches = batches

    async def query_memory(
        self, query: str, **kwargs: Any
    ) -> EpisodicMemory.QueryResponse:
        batch = self.batches[min(len(self.calls), len(self.batches) - 1)]
        self.queries.append(query)
        self.calls.append({"query": query, **kwargs})
        await asyncio.sleep(0)
        return EpisodicMemory.QueryResponse(
            long_term_memory=EpisodicMemory.QueryResponse.LongTermMemoryResponse(
                episodes=[
                    EpisodeResponse(score=1.0, **episode.model_dump())
                    for episode in batch
                ]
            ),
            short_term_memory=EpisodicMemory.QueryResponse.ShortTermMemoryResponse(
                episodes=[], episode_summary=[]
            ),
        )


async def test_sufficient_result_contains_exactly_assessed_support(
    policy: QueryPolicy,
) -> None:
    episodes = _episodes(4)
    agent, model, reranker = _agent(_decision(True, evidence_indices=[0, 3]))
    memory = FakeEpisodicMemory({"question": episodes})

    result, metrics = await agent.do_query(
        policy, QueryParam(query="question", memory=memory, limit=2)
    )

    assert {episode.uid for episode in result} == {episodes[0].uid, episodes[3].uid}
    assert metrics["stop_reason"] == "sufficient"
    assert metrics["assessed_sufficient"] is True
    assert metrics["returned_sufficient"] is True
    assert set(metrics["evidence_uids"]) == {str(episode.uid) for episode in result}
    assert metrics["returned_count"] == 2
    assert model.call_count == 1
    assert reranker.call_count == 0


async def test_support_larger_than_return_cap_is_not_claimed_sufficient(
    policy: QueryPolicy,
) -> None:
    episodes = _episodes(3)
    agent, model, _ = _agent(_decision(True, evidence_indices=[0, 1, 2]))
    result, metrics = await agent.do_query(
        policy,
        QueryParam(
            query="question", memory=FakeEpisodicMemory({"question": episodes}), limit=2
        ),
    )

    assert len(result) == 2
    assert metrics["stop_reason"] == "return_limit"
    assert metrics["assessed_sufficient"] is True
    assert metrics["returned_sufficient"] is None
    assert model.call_count == 1


async def test_same_query_widens_until_new_evidence_is_sufficient(
    policy: QueryPolicy,
) -> None:
    episodes = _episodes(6)
    agent, model, _ = _agent(
        [_decision(), _decision(), _decision(True, evidence_indices=[5])],
        initial_limit=2,
    )
    memory = FakeEpisodicMemory({"question": episodes})
    result, metrics = await agent.do_query(
        policy, QueryParam(query="question", memory=memory, limit=1)
    )

    assert result == [episodes[5]]
    assert memory.queries == ["question"] * 3
    assert metrics["retrieval_limits"] == [2, 4, 8]
    assert metrics["unique_candidates"] == 6
    assert metrics["raw_candidates"] == 12
    assert metrics["memory_search_called"] == model.call_count == 3
    assert metrics["input_token"] == metrics["output_token"] == 3


@pytest.mark.parametrize("query_limit", [0, -1, 100])
async def test_candidate_cap_includes_expanded_neighbors(
    policy: QueryPolicy, query_limit: int
) -> None:
    episodes = _episodes(9)
    memory = RoundMemory([episodes])
    agent, model, _ = _agent(_decision(), initial_limit=2, max_candidates=4)
    result, metrics = await agent.do_query(
        policy,
        QueryParam(
            query="question", memory=memory, limit=query_limit, expand_context=3
        ),
    )

    assert result == episodes[:4]
    assert metrics["unique_candidates"] == metrics["returned_count"] == 4
    assert metrics["raw_candidates"] == 9
    assert metrics["stop_reason"] == "candidate_limit"
    assert metrics["returned_sufficient"] is None
    assert model.call_count == 1
    assert memory.calls[0]["limit"] == 2


async def test_initial_limit_is_clamped_to_candidate_ceiling(
    policy: QueryPolicy,
) -> None:
    agent, _, _ = _agent(_decision(), initial_limit=8, max_candidates=3)
    memory = FakeEpisodicMemory({"question": _episodes(8)})
    _, metrics = await agent.do_query(
        policy, QueryParam(query="question", memory=memory)
    )
    assert metrics["retrieval_limits"] == [3]
    assert metrics["unique_candidates"] == 3


@pytest.mark.parametrize(
    ("rounds", "attempts", "expected"), [(3, 1, 1), (1, 5, 1), (2, 3, 2)]
)
async def test_retrieval_rounds_respect_both_budgets(
    policy: QueryPolicy, rounds: int, attempts: int, expected: int
) -> None:
    policy.max_attempts = attempts
    agent, model, _ = _agent(_decision(), initial_limit=1, max_rounds=rounds)
    memory = FakeEpisodicMemory({"question": _episodes(20)})
    _, metrics = await agent.do_query(
        policy, QueryParam(query="question", memory=memory)
    )
    assert len(memory.calls) == model.call_count == expected
    assert metrics["stop_reason"] == "round_limit"


async def test_uuid_dedup_keeps_first_snapshot_and_stops_stalled_search(
    policy: QueryPolicy,
) -> None:
    original = _episodes(1)[0]
    changed = original.model_copy(update={"content": "changed payload"})
    memory = RoundMemory([[original], [changed]])
    agent, model, _ = _agent(_decision(), initial_limit=1)
    result, metrics = await agent.do_query(
        policy, QueryParam(query="question", memory=memory)
    )
    assert result == [original]
    assert metrics["unique_candidates"] == 1
    assert metrics["raw_candidates"] == 2
    assert metrics["stop_reason"] == "stalled"
    assert metrics["memory_search_called"] == 2
    assert model.call_count == 1


async def test_unsuccessful_fallback_prioritizes_selected_evidence(
    policy: QueryPolicy,
) -> None:
    episodes = _episodes(3)
    agent, _, _ = _agent(_decision(evidence_indices=[2]), max_rounds=1)
    result, metrics = await agent.do_query(
        policy,
        QueryParam(
            query="question", memory=FakeEpisodicMemory({"question": episodes}), limit=1
        ),
    )
    assert result == [episodes[2]]
    assert metrics["returned_sufficient"] is None


@pytest.mark.parametrize(
    "changes",
    [
        {"is_sufficient": "false"},
        {"is_sufficient": 1},
        {"evidence_indices": [True]},
        {"evidence_indices": [-1]},
        {"evidence_indices": [2]},
        {"evidence_indices": [0, 0]},
        {"evidence_indices": [0.0]},
        {"evidence_indices": "0"},
        {"confidence_score": float("nan")},
        {"confidence_score": float("inf")},
        {"confidence_score": -0.1},
        {"confidence_score": 1.1},
        {"confidence_score": True},
        {"new_query": None},
    ],
)
async def test_invalid_decisions_stop_with_bounded_fallback(
    policy: QueryPolicy, changes: dict[str, Any]
) -> None:
    episodes = _episodes(2)
    agent, model, _ = _agent(_decision(True, **changes))
    result, metrics = await agent.do_query(
        policy,
        QueryParam(
            query="question", memory=FakeEpisodicMemory({"question": episodes}), limit=1
        ),
    )
    assert result == episodes[:1]
    assert metrics["stop_reason"] == "invalid_decision"
    assert metrics["returned_sufficient"] is None
    assert model.call_count == 1


@pytest.mark.parametrize("response", ["not json", "{}", "[]"])
async def test_malformed_or_missing_decision_fields(
    policy: QueryPolicy, response: str
) -> None:
    agent, _, _ = _agent(response)
    _, metrics = await agent.do_query(
        policy,
        QueryParam(
            query="question", memory=FakeEpisodicMemory({"question": _episodes(2)})
        ),
    )
    assert metrics["stop_reason"] == "invalid_decision"


@pytest.mark.parametrize(
    ("changes", "reason"),
    [
        ({"confidence_score": 0.79}, "round_limit"),
        ({"evidence_indices": []}, "invalid_decision"),
    ],
)
async def test_sufficient_requires_confidence_and_nonempty_support(
    policy: QueryPolicy, changes: dict[str, Any], reason: str
) -> None:
    agent, _, _ = _agent(_decision(True, **changes), max_rounds=1)
    _, metrics = await agent.do_query(
        policy,
        QueryParam(
            query="question", memory=FakeEpisodicMemory({"question": _episodes(2)})
        ),
    )
    assert metrics["assessed_sufficient"] is True
    assert metrics["returned_sufficient"] is None
    assert metrics["stop_reason"] == reason


@pytest.mark.parametrize("query", ["", "   "])
async def test_empty_query_never_calls_backend_or_model(
    policy: QueryPolicy, query: str
) -> None:
    agent, model, _ = _agent(_decision())
    memory = FakeEpisodicMemory({})
    result, metrics = await agent.do_query(
        policy, QueryParam(query=query, memory=memory)
    )
    assert result == memory.calls == []
    assert metrics["stop_reason"] == "empty_query"
    assert metrics["assessed_sufficient"] is None
    assert model.call_count == 0


@pytest.mark.parametrize(
    "failure", [RuntimeError("backend failed"), asyncio.CancelledError()]
)
@pytest.mark.parametrize("stage", ["retrieval", "model"])
async def test_errors_and_cancellation_propagate(
    policy: QueryPolicy,
    monkeypatch: pytest.MonkeyPatch,
    failure: BaseException,
    stage: str,
) -> None:
    agent, model, _ = _agent(_decision())
    memory = FakeEpisodicMemory({"question": _episodes(1)})
    target, method = (
        (memory, "query_memory")
        if stage == "retrieval"
        else (model, "generate_response_with_token_usage")
    )
    monkeypatch.setattr(target, method, AsyncMock(side_effect=failure))
    with pytest.raises(type(failure)):
        await agent.do_query(policy, QueryParam(query="question", memory=memory))


async def test_concurrent_requests_preserve_scope_and_do_not_mutate_queries(
    policy: QueryPolicy,
) -> None:
    agent, _, _ = _agent(
        _decision(new_query="rewritten"), initial_limit=1, max_rounds=2
    )
    memories = []
    for prefix in ("tenant-a", "tenant-b"):
        episodes = [
            episode.model_copy(
                update={"content": f"{prefix}:{episode.content}", "session_key": prefix}
            )
            for episode in _episodes(2)
        ]
        memory = RoundMemory([episodes[:1], episodes])
        memory._session_key = prefix
        memories.append(memory)
    queries = [
        QueryParam(
            query="question",
            memory=memory,
            limit=2,
            expand_context=2,
            score_threshold=0.65,
            property_filter=Comparison("tenant", "=", str(index)),
        )
        for index, memory in enumerate(memories)
    ]
    responses = await asyncio.gather(
        *(agent.do_query(policy, query) for query in queries)
    )
    for index, ((result, metrics), query, memory) in enumerate(
        zip(responses, queries, memories, strict=True)
    ):
        assert all(
            episode.content.startswith(("tenant-a", "tenant-b")[index])
            for episode in result
        )
        assert {episode.uid for episode in result} == {
            episode.uid for episode in _episodes(2)
        }
        assert all(
            episode.session_key == ("tenant-a", "tenant-b")[index] for episode in result
        )
        assert metrics["unique_candidates"] == 2
        assert metrics["retrieval_limits"] == [1, 2]
        assert metrics["queries"] == ["question", "rewritten"]
        assert query.query == "question"
        assert query.limit == 2
        assert query.memory is memory
        assert all(
            call["property_filter"] == query.property_filter for call in memory.calls
        )
        assert all(
            call["expand_context"] == 2 and call["score_threshold"] == 0.65
            for call in memory.calls
        )
        assert all(
            call["mode"] == EpisodicMemory.QueryMode.LONG_TERM_ONLY
            for call in memory.calls
        )


async def test_low_confidence_support_over_return_cap_continues_widening(
    policy: QueryPolicy,
) -> None:
    episodes = _episodes(3)
    agent, model, _ = _agent(
        [
            _decision(True, confidence_score=0.1, evidence_indices=[0, 1]),
            _decision(True, evidence_indices=[2]),
        ],
        initial_limit=2,
    )
    memory = FakeEpisodicMemory({"question": episodes})

    result, metrics = await agent.do_query(
        policy, QueryParam(query="question", memory=memory, limit=1)
    )

    assert result == [episodes[2]]
    assert metrics["retrieval_limits"] == [2, 4]
    assert metrics["stop_reason"] == "sufficient"
    assert metrics["returned_sufficient"] is True
    assert model.call_count == 2


async def test_nonprefix_batches_union_is_capped_without_shrinking_search_windows(
    policy: QueryPolicy,
) -> None:
    episodes = _episodes(7)
    memory = RoundMemory([episodes[:2], episodes[2:4], episodes[4:]])
    agent, _, _ = _agent(_decision(), initial_limit=2, max_candidates=5)

    result, metrics = await agent.do_query(
        policy, QueryParam(query="question", memory=memory)
    )

    assert result == episodes[:5]
    assert metrics["retrieval_limits"] == [2, 4, 5]
    assert [call["limit"] for call in memory.calls] == [2, 4, 5]
    assert metrics["unique_candidates"] == 5
    assert metrics["raw_candidates"] == 7
    assert metrics["stop_reason"] == "candidate_limit"


async def test_invalid_later_decision_preserves_previous_selected_evidence(
    policy: QueryPolicy,
) -> None:
    episodes = _episodes(3)
    agent, _, _ = _agent(
        [_decision(evidence_indices=[1]), "invalid response"], initial_limit=2
    )
    memory = FakeEpisodicMemory({"question": episodes})

    result, metrics = await agent.do_query(
        policy, QueryParam(query="question", memory=memory, limit=1)
    )

    assert result == [episodes[1]]
    assert metrics["evidence_uids"] == [str(episodes[1].uid)]
    assert metrics["retrieval_limits"] == [2, 4]
    assert metrics["unique_candidates"] == 3
    assert metrics["returned_count"] == 1
    assert metrics["stop_reason"] == "invalid_decision"
    assert metrics["returned_sufficient"] is None


async def test_mutating_constructor_config_does_not_change_agent_budgets(
    policy: QueryPolicy,
) -> None:
    config = ProgressiveRetrievalConf(initial_limit=2, max_candidates=3, max_rounds=2)
    agent = ProgressiveQueryAgent(
        AgentToolBaseParam(
            model=DummyLanguageModel(_decision()),
            children_tools=[MemMachineAgent(AgentToolBaseParam())],
        ),
        config,
    )
    config.initial_limit = config.max_candidates = 20
    config.max_rounds = 10
    memory = FakeEpisodicMemory({"question": _episodes(20)})

    result, metrics = await agent.do_query(
        policy, QueryParam(query="question", memory=memory)
    )

    assert len(result) == metrics["unique_candidates"] == 3
    assert metrics["retrieval_limits"] == [2, 3]
    assert metrics["stop_reason"] == "candidate_limit"


async def test_confirmed_sufficiency_takes_precedence_at_candidate_ceiling(
    policy: QueryPolicy,
) -> None:
    episodes = _episodes(4)
    agent, model, _ = _agent(
        [_decision(), _decision(True, evidence_indices=[1, 3])],
        initial_limit=2,
        max_candidates=4,
    )
    memory = FakeEpisodicMemory({"question": episodes})

    result, metrics = await agent.do_query(
        policy, QueryParam(query="question", memory=memory, limit=2)
    )

    assert result == [episodes[1], episodes[3]]
    assert metrics["unique_candidates"] == 4
    assert metrics["retrieval_limits"] == [2, 4]
    assert metrics["stop_reason"] == "sufficient"
    assert metrics["returned_sufficient"] is True
    assert model.call_count == 2
