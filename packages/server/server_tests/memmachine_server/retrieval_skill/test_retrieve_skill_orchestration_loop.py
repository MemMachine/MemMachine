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
async def test_tool_protocol_direct_memory_and_return_final(
    query_policy: QueryPolicy,
) -> None:
    episode = _build_episode("ep-direct", "direct")
    memory = FakeEpisodicMemory({"hello": [episode]})
    model = ScriptedLanguageModel(
        [
            (
                "policy text",
                [
                    {
                        "function": {
                            "name": "direct_memory_search",
                            "arguments": {
                                "query": "hello",
                                "rationale": "fetch evidence",
                            },
                        }
                    },
                    {
                        "function": {
                            "name": "return_final",
                            "arguments": {
                                "final_response": "finalized",
                                "rationale": "evidence sufficient",
                            },
                        }
                    },
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
    assert metrics["orchestrator_tool_call_count"] == 2
    assert metrics["orchestrator_sub_skill_count"] == 0
    assert metrics["orchestrator_final_response"] == "finalized"
    assert metrics["top_level_session_invocation_count"] == 1
    assert metrics["top_level_session_turn_count"] == 1
    trace = metrics["orchestrator_trace"]
    assert isinstance(trace, dict)
    direct_call = trace["tool_calls"][0]
    assert direct_call["tool_name"] == "direct_memory_search"
    assert direct_call["raw_result"]["episodes_returned"] == 1
    assert direct_call["raw_result"]["query"] == "hello"
    assert isinstance(direct_call["raw_result"]["wall_time_seconds"], float)
    assert isinstance(direct_call["raw_result"]["memory_search_latency_seconds"], list)
    assert "episodes_human_readable" not in direct_call["raw_result"]


@pytest.mark.asyncio
async def test_direct_memory_search_and_state_tracking(
    query_policy: QueryPolicy,
) -> None:
    sub_episode = _build_episode("ep-sub", "sub evidence")
    top_episode = _build_episode("ep-top", "top evidence")
    memory = FakeEpisodicMemory({"branch query": [sub_episode], "hello": [top_episode]})
    model = ScriptedLanguageModel(
        [
            (
                "top-level",
                [
                    {
                        "function": {
                            "name": "direct_memory_search",
                            "arguments": {
                                "query": "branch query",
                                "rationale": "branch retrieval",
                            },
                        }
                    },
                    {
                        "function": {
                            "name": "direct_memory_search",
                            "arguments": {"query": "hello"},
                        }
                    },
                    {
                        "function": {
                            "name": "return_final",
                            "arguments": {"final_response": "combined"},
                        }
                    },
                ],
            ),
        ]
    )
    retrieve_skill = _build_skill(model)

    episodes, metrics = await retrieve_skill.do_query(
        query_policy,
        QueryParam(query="hello", limit=5, memory=memory),
    )

    assert [item.uid for item in episodes] == ["ep-sub", "ep-top"]
    assert metrics["orchestrator_sub_skill_count"] == 0
    sub_skill_runs = metrics["orchestrator_sub_skill_runs"]
    assert isinstance(sub_skill_runs, list)
    assert sub_skill_runs == []
    trace = metrics["orchestrator_trace"]
    assert isinstance(trace, dict)
    calls = trace["tool_calls"]
    assert calls[0]["tool_name"] == "direct_memory_search"
    assert calls[0]["arguments"]["query"] == "branch query"
    assert calls[1]["tool_name"] == "direct_memory_search"
    assert calls[1]["arguments"]["query"] == "hello"
    assert metrics["memory_search_called"] == 2
    assert float(metrics["memory_retrieval_time"]) > 0.0
    assert metrics["top_level_session_invocation_count"] == 1


@pytest.mark.asyncio
async def test_return_final_captures_top_level_sufficiency_fields(
    query_policy: QueryPolicy,
) -> None:
    episode = _build_episode("ep-suff", "suff evidence")
    memory = FakeEpisodicMemory({"hello": [episode]})
    model = ScriptedLanguageModel(
        [
            (
                "policy text",
                [
                    {
                        "function": {
                            "name": "direct_memory_search",
                            "arguments": {"query": "hello"},
                        }
                    },
                    {
                        "function": {
                            "name": "return_final",
                            "arguments": {
                                "final_response": "finalized",
                                "is_sufficient": True,
                                "confidence_score": 0.91,
                                "reason_code": "sufficient_cumulative_evidence",
                                "reason_note": "supporting evidence found",
                                "related_episode_indices": [0],
                                "selected_episode_indices": [0],
                            },
                        }
                    },
                ],
            )
        ]
    )
    retrieve_skill = _build_skill(model)

    episodes, metrics = await retrieve_skill.do_query(
        query_policy,
        QueryParam(query="hello", limit=5, memory=memory),
    )

    assert [item.uid for item in episodes] == ["ep-suff"]
    assert metrics["top_level_sufficiency_signal_seen"] is True
    assert metrics["top_level_is_sufficient"] is True
    assert metrics["top_level_confidence_score"] == pytest.approx(0.91)
    assert metrics["top_level_reason_code"] == "sufficient_cumulative_evidence"
    assert metrics["top_level_reason_note"] == "supporting evidence found"
    assert metrics["top_level_related_episode_indices"] == [0]
    assert metrics["top_level_selected_episode_indices"] == [0]


@pytest.mark.asyncio
async def test_return_final_can_emit_stage_result_memory_payload(
    query_policy: QueryPolicy,
) -> None:
    episode = _build_episode("ep-stage-raw", "raw retrieval evidence")
    memory = FakeEpisodicMemory({"hello": [episode]})
    model = ScriptedLanguageModel(
        [
            (
                "policy text",
                [
                    {
                        "function": {
                            "name": "direct_memory_search",
                            "arguments": {"query": "hello"},
                        }
                    },
                    {
                        "function": {
                            "name": "return_final",
                            "arguments": {
                                "final_response": "finalized",
                                "is_sufficient": True,
                                "confidence_score": 0.92,
                                "stage_results": [
                                    {
                                        "query": "Who discovered penicillin?",
                                        "stage_result": "Alexander Fleming",
                                        "confidence_score": 0.92,
                                    }
                                ],
                                "sub_queries": [
                                    "Who discovered penicillin?",
                                    "What prize did he receive?",
                                ],
                            },
                        }
                    },
                ],
            )
        ]
    )
    retrieve_skill = _build_skill(model)

    episodes, metrics = await retrieve_skill.do_query(
        query_policy,
        QueryParam(query="hello", limit=5, memory=memory),
    )

    assert len(episodes) == 3
    assert all(item.producer_id == "retrieve-skill-stage-result" for item in episodes)
    assert "StageResult" in episodes[0].content
    assert "SubQuery 1" in episodes[1].content
    assert "SubQuery 2" in episodes[2].content
    assert metrics["stage_result_memory_returned"] is True
    assert metrics["returned_stage_result_count"] == 1
    assert metrics["returned_sub_query_count"] == 2


@pytest.mark.asyncio
async def test_stage_result_memory_payload_requires_confidence_threshold(
    query_policy: QueryPolicy,
) -> None:
    episode = _build_episode("ep-stage-low-confidence", "raw retrieval evidence")
    memory = FakeEpisodicMemory({"hello": [episode]})
    model = ScriptedLanguageModel(
        [
            (
                "policy text",
                [
                    {
                        "function": {
                            "name": "direct_memory_search",
                            "arguments": {"query": "hello"},
                        }
                    },
                    {
                        "function": {
                            "name": "return_final",
                            "arguments": {
                                "final_response": "finalized",
                                "is_sufficient": True,
                                "confidence_score": 0.72,
                                "stage_results": [
                                    {
                                        "query": "Who discovered penicillin?",
                                        "stage_result": "Alexander Fleming",
                                        "confidence_score": 0.72,
                                    }
                                ],
                                "sub_queries": ["Who discovered penicillin?"],
                            },
                        }
                    },
                ],
            )
        ]
    )
    retrieve_skill = _build_skill(model)

    episodes, metrics = await retrieve_skill.do_query(
        query_policy,
        QueryParam(query="hello", limit=5, memory=memory),
    )

    assert [item.uid for item in episodes] == ["ep-stage-low-confidence"]
    assert metrics["stage_result_memory_returned"] is False


@pytest.mark.asyncio
async def test_tool_protocol_invalid_action_triggers_fallback(
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
                            "name": "unknown_tool",
                            "arguments": {},
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
    assert metrics["skill_contract_error_code"] == "SKILL_CONTRACT_INVALID_OUTPUT"
    assert metrics["top_level_session_invocation_count"] == 1


@pytest.mark.asyncio
async def test_legacy_direct_memory_sub_skill_name_is_rejected(
    query_policy: QueryPolicy,
) -> None:
    fallback_episode = _build_episode("ep-direct-memory-legacy", "fallback")
    memory = FakeEpisodicMemory({"hello": [fallback_episode]})
    model = ScriptedLanguageModel(
        [
            (
                "legacy tool payload",
                [
                    {
                        "function": {
                            "name": "spawn_sub_skill",
                            "arguments": {
                                "skill_name": "direct_memory",
                                "query": "hello",
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

    assert [item.uid for item in episodes] == ["ep-direct-memory-legacy"]
    assert metrics["fallback_trigger_reason"] == "invalid_tool_call"
    assert metrics["skill_contract_error_code"] == "SKILL_CONTRACT_INVALID_OUTPUT"


@pytest.mark.asyncio
async def test_top_level_internal_split_branching_and_applies_final_rerank(
    query_policy: QueryPolicy,
) -> None:
    branch_a = _build_episode("branch-a", "branch evidence a")
    branch_b = _build_episode("branch-b", "branch evidence b")
    branch_c = _build_episode("branch-c", "branch evidence c")
    memory = FakeEpisodicMemory(
        {
            "branch direct": [branch_a, branch_b],
            "branch coq then detail": [branch_c],
        }
    )
    model = ScriptedLanguageModel(
        [
            (
                "top-level",
                [
                    {
                        "function": {
                            "name": "direct_memory_search",
                            "arguments": {
                                "query": "branch direct",
                            },
                        }
                    },
                    {
                        "function": {
                            "name": "spawn_sub_skill",
                            "arguments": {
                                "skill_name": "coq",
                                "query": "branch coq then detail",
                            },
                        }
                    },
                    {
                        "function": {
                            "name": "return_final",
                            "arguments": {
                                "final_response": "done",
                                "sub_queries": [
                                    "branch direct",
                                    "branch coq then detail",
                                ],
                            },
                        }
                    },
                ],
            ),
            (
                "coq branch",
                [
                    {
                        "function": {
                            "name": "memmachine_search",
                            "arguments": {"query": "branch coq then detail"},
                        }
                    },
                    {
                        "function": {
                            "name": "return_sub_skill_result",
                            "arguments": {
                                "summary": (
                                    '{"is_sufficient":true,'
                                    '"confidence_score":0.93,'
                                    '"final_query":"branch coq then detail"}'
                                )
                            },
                        }
                    },
                ],
            ),
        ]
    )
    retrieve_skill = _build_skill(model)

    episodes, metrics = await retrieve_skill.do_query(
        query_policy,
        QueryParam(query="hello", limit=1, memory=memory),
    )

    assert len(episodes) == 1
    assert metrics["branch_total"] == 0
    assert metrics["branch_success_count"] == 0
    assert metrics["branch_failure_count"] == 0
    assert metrics["rerank_applied"] is True
    sub_skill_runs = metrics["orchestrator_sub_skill_runs"]
    assert isinstance(sub_skill_runs, list)
    assert [run["skill_name"] for run in sub_skill_runs] == ["coq"]
    assert metrics.get("top_level_sub_queries") == [
        "branch direct",
        "branch coq then detail",
    ]


@pytest.mark.asyncio
async def test_top_level_internal_split_executes_multiple_direct_branches(
    query_policy: QueryPolicy,
) -> None:
    branch_a_query = "When did Fleetwood Sheppard die?"
    branch_b_query = "When did George William Whitaker die?"
    branch_a = _build_episode("branch-v1-a", "Fleetwood branch evidence")
    branch_b = _build_episode("branch-v1-b", "Whitaker branch evidence")
    memory = FakeEpisodicMemory(
        {
            branch_a_query: [branch_a],
            branch_b_query: [branch_b],
        }
    )
    model = ScriptedLanguageModel(
        [
            (
                "top-level",
                [
                    {
                        "function": {
                            "name": "direct_memory_search",
                            "arguments": {
                                "query": branch_a_query,
                            },
                        }
                    },
                    {
                        "function": {
                            "name": "direct_memory_search",
                            "arguments": {
                                "query": branch_b_query,
                            },
                        }
                    },
                    {
                        "function": {
                            "name": "return_final",
                            "arguments": {
                                "final_response": "done",
                                "sub_queries": [branch_a_query, branch_b_query],
                            },
                        }
                    },
                ],
            ),
        ]
    )
    retrieve_skill = _build_skill(model)

    _episodes, metrics = await retrieve_skill.do_query(
        query_policy,
        QueryParam(query="hello", limit=5, memory=memory),
    )

    assert metrics["branch_total"] == 0
    assert metrics["branch_success_count"] == 0
    assert branch_a_query in memory.queries
    assert branch_b_query in memory.queries
    sub_skill_runs = metrics["orchestrator_sub_skill_runs"]
    assert isinstance(sub_skill_runs, list)
    assert sub_skill_runs == []
    assert metrics.get("top_level_sub_queries") == [branch_a_query, branch_b_query]


@pytest.mark.asyncio
async def test_top_level_sub_queries_do_not_drive_stage_result_aggregation(
    query_policy: QueryPolicy,
) -> None:
    branch_episode = _build_episode("branch-only", "branch evidence")
    memory = FakeEpisodicMemory({"branch one": [branch_episode]})
    model = ScriptedLanguageModel(
        [
            (
                "top-level",
                [
                    {
                        "function": {
                            "name": "direct_memory_search",
                            "arguments": {
                                "query": "branch one",
                            },
                        }
                    },
                    {
                        "function": {
                            "name": "return_final",
                            "arguments": {
                                "final_response": "done",
                                "is_sufficient": True,
                                "confidence_score": 0.95,
                                "sub_queries": ["branch one", "branch two"],
                            },
                        }
                    },
                ],
            ),
        ]
    )
    retrieve_skill = _build_skill(model)

    episodes, metrics = await retrieve_skill.do_query(
        query_policy,
        QueryParam(query="compare", limit=5, memory=memory),
    )

    assert [item.uid for item in episodes] == ["branch-only"]
    assert metrics["stage_result_memory_returned"] is False
    assert metrics.get("top_level_stage_results") in (None, [])
    assert metrics.get("top_level_sub_queries") == ["branch one", "branch two"]


@pytest.mark.asyncio
async def test_top_level_can_issue_follow_up_internal_branch(
    query_policy: QueryPolicy,
) -> None:
    branch_initial = _build_episode("branch-rerun-initial", "initial branch evidence")
    branch_rerun = _build_episode("branch-rerun-second", "rerun branch evidence")
    memory = FakeEpisodicMemory(
        {
            "branch one": [branch_initial],
            "branch rerun": [branch_rerun],
        }
    )
    model = ScriptedLanguageModel(
        [
            (
                "top-level",
                [
                    {
                        "function": {
                            "name": "direct_memory_search",
                            "arguments": {
                                "query": "branch one",
                            },
                        }
                    },
                    {
                        "function": {
                            "name": "direct_memory_search",
                            "arguments": {
                                "query": "branch rerun",
                            },
                        }
                    },
                    {
                        "function": {
                            "name": "return_final",
                            "arguments": {
                                "final_response": "done",
                                "sub_queries": ["branch one", "branch rerun"],
                            },
                        }
                    },
                ],
            ),
        ]
    )
    retrieve_skill = _build_skill(model)

    _episodes, metrics = await retrieve_skill.do_query(
        query_policy,
        QueryParam(query="hello", limit=5, memory=memory),
    )

    assert "branch one" in memory.queries
    assert "branch rerun" in memory.queries
    assert metrics["branch_total"] == 0
    assert metrics["branch_success_count"] == 0
    sub_runs = metrics["orchestrator_sub_skill_runs"]
    assert isinstance(sub_runs, list)
    assert sub_runs == []
    assert metrics.get("top_level_sub_queries") == ["branch one", "branch rerun"]


@pytest.mark.asyncio
async def test_coq_sub_skill_reuses_cached_results_for_near_duplicate_queries(
    query_policy: QueryPolicy,
) -> None:
    q1 = "Who performed the song 'Sunday Papers' 'Sunday Papers' song performer"
    q2 = 'Who performed the song "Sunday Papers" performer artist "Sunday Papers" song'
    sub_episode = _build_episode(
        "coq-sub",
        'Sunday Papers: "Sunday Papers" is a song written and performed by British new wave musician Joe Jackson.',
    )
    memory = FakeEpisodicMemory({q1: [sub_episode]})
    model = ScriptedLanguageModel(
        [
            (
                "top-level",
                [
                    {
                        "function": {
                            "name": "spawn_sub_skill",
                            "arguments": {
                                "skill_name": "coq",
                                "query": "Find performer then award",
                            },
                        }
                    },
                    {
                        "function": {
                            "name": "return_final",
                            "arguments": {"final_response": "done"},
                        }
                    },
                ],
            ),
            (
                "coq",
                [
                    {
                        "function": {
                            "name": "memmachine_search",
                            "arguments": {"query": q1},
                        }
                    },
                    {
                        "function": {
                            "name": "memmachine_search",
                            "arguments": {"query": q2},
                        }
                    },
                    {
                        "function": {
                            "name": "return_sub_skill_result",
                            "arguments": {
                                "summary": (
                                    '{"is_sufficient":true,'
                                    '"evidence_indices":[0],'
                                    '"new_query":"Find performer then award",'
                                    '"confidence_score":0.91,'
                                    '"reason_code":"sufficient_explicit_fact"}'
                                )
                            },
                        }
                    },
                ],
            ),
        ]
    )
    retrieve_skill = _build_skill(model)

    _episodes, metrics = await retrieve_skill.do_query(
        query_policy,
        QueryParam(query="hello", limit=5, memory=memory),
    )

    # Only the first near-duplicate query should hit memory.
    assert memory.queries == [q1]
    sub_runs = metrics["orchestrator_sub_skill_runs"]
    assert isinstance(sub_runs, list)
    coq_run = sub_runs[0]
    memmachine_calls = [
        call
        for call in coq_run["tool_calls"]
        if call["tool_name"] == "memmachine_search"
    ]
    assert len(memmachine_calls) == 2
    assert memmachine_calls[0]["raw_result"]["cached"] is False
    assert memmachine_calls[1]["raw_result"]["cached"] is True
    assert memmachine_calls[1]["raw_result"]["cached_from_query"] == q1
    assert metrics["memory_search_called"] == 1
    assert float(metrics["memory_retrieval_time"]) > 0.0


@pytest.mark.asyncio
async def test_top_level_coq_spawn_preserves_explicit_query_override(
    query_policy: QueryPolicy,
) -> None:
    original_query = (
        "Where did Prince Gustav of Thurn and Taxis (1848-1914)'s mother die?"
    )
    rewritten_query = (
        "Decompose: 1) Identify the mother of Prince Gustav of Thurn and Taxis "
        "(1848-1914). 2) Find where she died."
    )
    original_episode = _build_episode(
        "coq-original",
        "Princess Mathilde Sophie ... died in Obermais, Meran.",
    )
    rewritten_episode = _build_episode(
        "coq-rewritten",
        "This should not be returned if coq query override works.",
    )
    memory = FakeEpisodicMemory(
        {
            original_query: [original_episode],
            rewritten_query: [rewritten_episode],
        }
    )
    model = ScriptedLanguageModel(
        [
            (
                "top-level",
                [
                    {
                        "function": {
                            "name": "spawn_sub_skill",
                            "arguments": {
                                "skill_name": "coq",
                                "query": rewritten_query,
                            },
                        }
                    },
                    {
                        "function": {
                            "name": "return_final",
                            "arguments": {"final_response": "done"},
                        }
                    },
                ],
            ),
            (
                "coq",
                [
                    {
                        "function": {
                            "name": "memmachine_search",
                            "arguments": {},
                        }
                    },
                    {
                        "function": {
                            "name": "return_sub_skill_result",
                            "arguments": {
                                "summary": (
                                    '{"is_sufficient":true,'
                                    '"evidence_indices":[0],'
                                    '"new_query":"Where did Prince Gustav of Thurn and '
                                    "Taxis (1848-1914)'s mother die?\","
                                    '"confidence_score":0.95,'
                                    '"reason_code":"sufficient_cumulative_evidence",'
                                    '"reason_note":"mother and death place found"}'
                                )
                            },
                        }
                    },
                ],
            ),
        ]
    )
    retrieve_skill = _build_skill(model)

    episodes, metrics = await retrieve_skill.do_query(
        query_policy,
        QueryParam(query=original_query, limit=5, memory=memory),
    )

    assert [item.uid for item in episodes] == ["coq-rewritten"]
    assert memory.queries == [rewritten_query]
    sub_runs = metrics["orchestrator_sub_skill_runs"]
    assert isinstance(sub_runs, list)
    assert sub_runs[0]["skill_name"] == "coq"
    assert sub_runs[0]["query"] == rewritten_query


@pytest.mark.asyncio
async def test_llm_time_accumulates_top_level_and_sub_skill_sessions(
    query_policy: QueryPolicy,
) -> None:
    sub_episode = _build_episode("llm-time-sub", "sub evidence")
    memory = FakeEpisodicMemory({"branch query": [sub_episode]})
    model = ScriptedLanguageModel(
        [
            (
                "top-level",
                [
                    {
                        "function": {
                            "name": "spawn_sub_skill",
                            "arguments": {
                                "skill_name": "coq",
                                "query": "branch query",
                            },
                        }
                    },
                    {
                        "function": {
                            "name": "return_final",
                            "arguments": {"final_response": "done"},
                        }
                    },
                ],
            ),
            (
                "sub-skill",
                [
                    {
                        "function": {
                            "name": "memmachine_search",
                            "arguments": {"query": "branch query"},
                        }
                    }
                ],
            ),
        ]
    )
    model.session_llm_times = [0.14, 0.31]
    model.session_llm_input_tokens = [7, 11]
    model.session_llm_output_tokens = [3, 5]
    retrieve_skill = _build_skill(model)

    _episodes, metrics = await retrieve_skill.do_query(
        query_policy,
        QueryParam(query="hello", limit=5, memory=memory),
    )

    assert metrics["llm_time"] == pytest.approx(0.45)
    assert metrics["llm_call_count"] == 2
    assert metrics["input_token"] == 18
    assert metrics["output_token"] == 8
    assert metrics["top_level_llm_call_count"] == 1
    assert metrics["top_level_input_token"] == 7
    assert metrics["top_level_output_token"] == 3
