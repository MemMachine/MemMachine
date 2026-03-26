from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import requests

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from memmachine_common import Skill, SkillRunner  # noqa: E402

from evaluation.utils import skill_utils  # noqa: E402


class _FakeUsage:
    def __init__(self, input_tokens: int = 0, output_tokens: int = 0) -> None:
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens


class _FakeResponse:
    def __init__(
        self,
        *,
        response_id: str,
        output_text: str,
        output: list[dict[str, object]],
        usage: _FakeUsage | None = None,
    ) -> None:
        self.id = response_id
        self.output_text = output_text
        self.output = output
        self.usage = usage


class _FakeResponsesAPI:
    def __init__(self, responses: list[_FakeResponse]) -> None:
        self.calls: list[dict[str, object]] = []
        self._responses = list(responses)

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        return self._responses.pop(0)


class _FakeOpenAIClient:
    def __init__(self, responses: list[_FakeResponse]) -> None:
        self.responses = _FakeResponsesAPI(responses)


class _FakeRestMemory:
    def __init__(self, search_results: list[dict[str, object]]) -> None:
        self.calls: list[dict[str, object]] = []
        self._search_results = list(search_results)

    async def search(self, query: str, **kwargs):
        self.calls.append({"query": query, **kwargs})
        return self._search_results.pop(0)


class _FakeWarmupMemory:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def search(
        self,
        query: str,
        limit: int | None = None,
        expand_context: int = 0,
        score_threshold: float | None = None,
        filter_dict: dict[str, str] | None = None,
        timeout: int | None = None,
        *,
        set_metadata: dict[str, object] | None = None,
        agent_mode: bool = False,
    ) -> dict[str, object]:
        self.calls.append(
            {
                "query": query,
                "limit": limit,
                "expand_context": expand_context,
                "score_threshold": score_threshold,
                "filter_dict": filter_dict,
                "timeout": timeout,
                "set_metadata": set_metadata,
                "agent_mode": agent_mode,
            }
        )
        return {}


class _FlakyWarmupMemory(_FakeWarmupMemory):
    def __init__(self, failures: int) -> None:
        super().__init__()
        self._failures = failures

    def search(
        self,
        query: str,
        limit: int | None = None,
        expand_context: int = 0,
        score_threshold: float | None = None,
        filter_dict: dict[str, str] | None = None,
        timeout: int | None = None,
        *,
        set_metadata: dict[str, object] | None = None,
        agent_mode: bool = False,
    ) -> dict[str, object]:
        result = super().search(
            query,
            limit,
            expand_context,
            score_threshold,
            filter_dict,
            timeout,
            set_metadata=set_metadata,
            agent_mode=agent_mode,
        )
        if self._failures > 0:
            self._failures -= 1
            raise requests.ReadTimeout("timed out")
        return result


def _skill() -> Skill:
    return Skill(
        provider="openai",
        skill_name="retrieve-skill",
        file_ids=("file-1",),
        content_hashes=("hash-1",),
    )


def _search_result(*episodes: tuple[str, str]) -> dict[str, object]:
    return {
        "content": {
            "episodic_memory": {
                "short_term_memory": {
                    "episodes": [
                        {"uid": episode_uid, "content": episode_content}
                        for episode_uid, episode_content in episodes
                    ],
                    "episode_summary": [],
                },
                "long_term_memory": {"episodes": []},
            }
        }
    }


def _semantic_feature(
    *,
    category: str,
    tag: str,
    feature_name: str,
    value: str,
) -> dict[str, str]:
    return {
        "category": category,
        "tag": tag,
        "feature_name": feature_name,
        "value": value,
    }


def _search_result_with_short_and_long(
    *,
    short_term: tuple[tuple[str, str], ...] = (),
    long_term: tuple[tuple[str, str], ...] = (),
    summary: str = "",
) -> dict[str, object]:
    return {
        "content": {
            "episodic_memory": {
                "short_term_memory": {
                    "episodes": [
                        {"uid": episode_uid, "content": episode_content}
                        for episode_uid, episode_content in short_term
                    ],
                    "episode_summary": [summary] if summary else [],
                },
                "long_term_memory": {
                    "episodes": [
                        {"uid": episode_uid, "content": episode_content}
                        for episode_uid, episode_content in long_term
                    ]
                },
            }
        }
    }


def test_strip_short_term_memory_from_search_result_clears_short_term_payload():
    payload = _search_result_with_short_and_long(
        short_term=(("ep-short", "short memory"),),
        long_term=(("ep-long", "long memory"),),
        summary="summary line",
    )

    result = skill_utils._strip_short_term_memory_from_search_result(payload)  # noqa: SLF001
    episodic_memory = result["content"]["episodic_memory"]

    assert episodic_memory["short_term_memory"]["episodes"] == []
    assert episodic_memory["short_term_memory"]["episode_summary"] == []
    assert episodic_memory["long_term_memory"]["episodes"] == [
        {"uid": "ep-long", "content": "long memory"}
    ]


def test_strip_short_term_memory_from_search_result_preserves_semantic_payload():
    payload = _search_result_with_short_and_long(
        short_term=(("ep-short", "short memory"),),
        long_term=(("ep-long", "long memory"),),
    )
    payload["content"]["semantic_memory"] = [
        _semantic_feature(
            category="profile",
            tag="food",
            feature_name="favorite_food",
            value="pizza",
        )
    ]

    result = skill_utils._strip_short_term_memory_from_search_result(payload)  # noqa: SLF001

    assert result["content"]["semantic_memory"] == [
        {
            "category": "profile",
            "tag": "food",
            "feature_name": "favorite_food",
            "value": "pizza",
        }
    ]


@pytest.mark.asyncio
async def test_warmup_rest_evaluation_search_uses_low_cost_search(monkeypatch):
    warmup_memory = _FakeWarmupMemory()

    monkeypatch.setattr(
        skill_utils,
        "init_rest_evaluation_memory",
        lambda session_id: warmup_memory,
    )
    monkeypatch.setenv("MEMMACHINE_SEARCH_TIMEOUT_SECONDS", "77")
    monkeypatch.setenv("MEMMACHINE_SEARCH_WARMUP_TIMEOUT_SECONDS", "13")

    elapsed = await skill_utils.warmup_rest_evaluation_search(
        session_id="session-1",
        query="warm up query",
    )

    assert elapsed >= 0.0
    assert warmup_memory.calls == [
        {
            "query": "warm up query",
            "limit": 1,
            "expand_context": 0,
            "score_threshold": None,
            "filter_dict": None,
            "timeout": 13,
            "set_metadata": None,
            "agent_mode": False,
        }
    ]


@pytest.mark.asyncio
async def test_warmup_rest_evaluation_search_skips_blank_inputs(monkeypatch):
    monkeypatch.setattr(
        skill_utils,
        "init_rest_evaluation_memory",
        lambda session_id: (_ for _ in ()).throw(
            AssertionError("should not initialize")
        ),
    )

    assert (
        await skill_utils.warmup_rest_evaluation_search(
            session_id="",
            query="warm up query",
        )
        == 0.0
    )
    assert (
        await skill_utils.warmup_rest_evaluation_search(
            session_id="session-1",
            query="   ",
        )
        == 0.0
    )


@pytest.mark.asyncio
async def test_warmup_rest_evaluation_search_retries_transient_timeouts(monkeypatch):
    warmup_memory = _FlakyWarmupMemory(failures=2)
    sleep_calls: list[float] = []

    async def _fake_sleep(delay: float) -> None:
        sleep_calls.append(delay)

    monkeypatch.setattr(
        skill_utils,
        "init_rest_evaluation_memory",
        lambda session_id: warmup_memory,
    )
    monkeypatch.setattr(skill_utils.asyncio, "sleep", _fake_sleep)

    elapsed = await skill_utils.warmup_rest_evaluation_search(
        session_id="session-1",
        query="warm up query",
        timeout_seconds=9,
        max_attempts=4,
        retry_delay_seconds=0.25,
    )

    assert elapsed is not None
    assert len(warmup_memory.calls) == 3
    assert [call["timeout"] for call in warmup_memory.calls] == [9, 9, 9]
    assert sleep_calls == [0.25, 0.25]


@pytest.mark.asyncio
async def test_warmup_rest_evaluation_search_returns_none_after_retry_budget(
    monkeypatch,
):
    warmup_memory = _FlakyWarmupMemory(failures=3)
    sleep_calls: list[float] = []

    async def _fake_sleep(delay: float) -> None:
        sleep_calls.append(delay)

    monkeypatch.setattr(
        skill_utils,
        "init_rest_evaluation_memory",
        lambda session_id: warmup_memory,
    )
    monkeypatch.setattr(skill_utils.asyncio, "sleep", _fake_sleep)

    elapsed = await skill_utils.warmup_rest_evaluation_search(
        session_id="session-1",
        query="warm up query",
        timeout_seconds=9,
        max_attempts=3,
        retry_delay_seconds=0.5,
    )

    assert elapsed is None
    assert len(warmup_memory.calls) == 3
    assert sleep_calls == [0.5, 0.5]


@pytest.mark.asyncio
async def test_warmup_rest_evaluation_search_raise_on_failure_propagates(
    monkeypatch,
):
    warmup_memory = _FlakyWarmupMemory(failures=1)

    monkeypatch.setattr(
        skill_utils,
        "init_rest_evaluation_memory",
        lambda session_id: warmup_memory,
    )

    with pytest.raises(requests.ReadTimeout):
        await skill_utils.warmup_rest_evaluation_search(
            session_id="session-1",
            query="warm up query",
            timeout_seconds=9,
            max_attempts=1,
            raise_on_failure=True,
        )


@pytest.mark.asyncio
async def test_process_question_with_runner_counts_unique_retrieved_episodes():
    client = _FakeOpenAIClient(
        [
            _FakeResponse(
                response_id="resp-1",
                output_text="",
                output=[
                    {
                        "type": "function_call",
                        "name": "memmachine_search",
                        "arguments": json.dumps({"query": "first hop"}),
                        "call_id": "call-1",
                    }
                ],
                usage=_FakeUsage(11, 3),
            ),
            _FakeResponse(
                response_id="resp-2",
                output_text=(
                    "[StageResult] Query: first hop | Answer: Paris | "
                    "Confidence: 0.93 | Reason: explicit memory"
                ),
                output=[
                    {
                        "type": "function_call",
                        "name": "memmachine_search",
                        "arguments": json.dumps({"query": "second hop"}),
                        "call_id": "call-2",
                    }
                ],
                usage=_FakeUsage(7, 2),
            ),
            _FakeResponse(
                response_id="resp-3",
                output_text="Paris",
                output=[],
                usage=_FakeUsage(5, 1),
            ),
            _FakeResponse(
                response_id="resp-4",
                output_text="Paris",
                output=[],
                usage=_FakeUsage(9, 2),
            ),
            _FakeResponse(
                response_id="resp-5",
                output_text="Paris",
                output=[],
                usage=_FakeUsage(9, 2),
            ),
        ]
    )
    memory = _FakeRestMemory(
        [
            _search_result(
                ("ep-1", "Paris is in France"), ("ep-2", "France is in Europe")
            ),
            _search_result(
                ("ep-1", "Paris is in France"), ("ep-2", "France is in Europe")
            ),
        ]
    )
    runner = SkillRunner(
        _skill(),
        client=client,
        model="gpt-5-mini",
        rest_memory=memory,
        stage_result_mode=True,
    )

    category, result = await skill_utils.process_question_with_runner(
        answer_prompt="Question: {question}\nMemories:\n{memories}",
        runner=runner,
        model=client,
        question="What is the capital of France?",
        answer="Paris",
        category="wiki",
        supporting_facts=["Paris is in France"],
        model_name="gpt-5-mini",
    )

    assert category == "wiki"
    assert result["model_answer"] == "Paris"
    assert result["memory_search_called"] == 2
    assert result["num_episodes_retrieved"] == 2
    assert result["llm_call_count"] == 5
    assert runner.last_search_results == []
    assert memory.calls[0]["limit"] == 20
    assert memory.calls[1]["limit"] == 20
    assert result["open_domain_rescue_used"] is False
    assert result["answer_verification_used"] is True


@pytest.mark.asyncio
async def test_process_question_with_runner_passes_semantic_memory_to_llm():
    client = _FakeOpenAIClient(
        [
            _FakeResponse(
                response_id="resp-1",
                output_text="",
                output=[
                    {
                        "type": "function_call",
                        "name": "memmachine_search",
                        "arguments": json.dumps({"query": "first hop"}),
                        "call_id": "call-1",
                    }
                ],
                usage=_FakeUsage(11, 3),
            ),
            _FakeResponse(
                response_id="resp-2",
                output_text="France",
                output=[],
                usage=_FakeUsage(7, 2),
            ),
            _FakeResponse(
                response_id="resp-3",
                output_text="France",
                output=[],
                usage=_FakeUsage(5, 1),
            ),
            _FakeResponse(
                response_id="resp-4",
                output_text="France",
                output=[],
                usage=_FakeUsage(5, 1),
            ),
        ]
    )
    memory = _FakeRestMemory(
        [
            {
                "content": {
                    "episodic_memory": {
                        "short_term_memory": {
                            "episodes": [],
                            "episode_summary": [],
                        },
                        "long_term_memory": {"episodes": []},
                    },
                    "semantic_memory": [
                        _semantic_feature(
                            category="location",
                            tag="capital",
                            feature_name="Paris",
                            value="France",
                        )
                    ],
                }
            }
        ]
    )
    runner = SkillRunner(
        _skill(),
        client=client,
        model="gpt-5-mini",
        rest_memory=memory,
    )

    _, result = await skill_utils.process_question_with_runner(
        answer_prompt="Question: {question}\nMemories:\n{memories}",
        runner=runner,
        model=client,
        question="What country is Paris in?",
        answer="France",
        category="wiki",
        supporting_facts=["Paris is in France"],
        model_name="gpt-5-mini",
    )

    semantic_line = "[Semantic] location/capital | Paris: France"
    assert semantic_line in result["conversation_memories"]
    assert semantic_line in client.responses.calls[2]["input"][0]["content"]
    assert semantic_line in client.responses.calls[3]["input"][0]["content"]


@pytest.mark.asyncio
async def test_process_question_with_runner_skips_memory_for_llm_mode():
    client = _FakeOpenAIClient(
        [
            _FakeResponse(
                response_id="resp-1",
                output_text="Ada Lovelace",
                output=[],
                usage=_FakeUsage(9, 2),
            ),
        ]
    )
    full_content = "Ada Lovelace wrote the first algorithm notes."

    _, result = await skill_utils.process_question_with_runner(
        answer_prompt="Question: {question}\nMemories:\n{memories}",
        runner=None,
        model=client,
        question="Who wrote the first algorithm notes?",
        answer="Ada Lovelace",
        category="wiki",
        supporting_facts=["Ada Lovelace wrote the first algorithm notes."],
        model_name="gpt-5-mini",
        full_content=full_content,
    )

    assert result["model_answer"] == "Ada Lovelace"
    assert result["conversation_memories"] == full_content
    assert result["memory_search_called"] == 0
    assert result["num_episodes_retrieved"] == 0
    assert result["selected_skill_name"] == "PureLLM"
    assert result["llm_call_count"] == 1
    assert "open_domain_rescue_used" not in result
    assert "answer_verification_used" not in result


@pytest.mark.asyncio
async def test_process_question_with_runner_uses_final_payload_and_serializes_stage_data():
    client = _FakeOpenAIClient(
        [
            _FakeResponse(
                response_id="resp-1",
                output_text="",
                output=[
                    {
                        "type": "function_call",
                        "name": "memmachine_search",
                        "arguments": json.dumps({"query": "first hop"}),
                        "call_id": "call-1",
                    }
                ],
                usage=_FakeUsage(8, 2),
            ),
            _FakeResponse(
                response_id="resp-2",
                output_text=(
                    "[StageResult] Query: first hop | Answer: Paris | "
                    "Confidence: 0.93 | Reason: explicit memory\n"
                    "[SubQuery] second hop"
                ),
                output=[
                    {
                        "type": "function_call",
                        "name": "memmachine_search",
                        "arguments": json.dumps({"query": "second hop"}),
                        "call_id": "call-2",
                    }
                ],
                usage=_FakeUsage(7, 2),
            ),
            _FakeResponse(
                response_id="resp-3",
                output_text=(
                    "[StageResult] Query: second hop | Answer: France | "
                    "Confidence: 0.94 | Reason: explicit memory\n"
                    "France"
                ),
                output=[],
                usage=_FakeUsage(5, 1),
            ),
            _FakeResponse(
                response_id="resp-4",
                output_text="France",
                output=[],
                usage=_FakeUsage(9, 2),
            ),
            _FakeResponse(
                response_id="resp-5",
                output_text="France",
                output=[],
                usage=_FakeUsage(9, 2),
            ),
        ]
    )
    memory = _FakeRestMemory(
        [
            _search_result(("ep-1", "Paris is a capital city")),
            _search_result(("ep-2", "Paris is in France")),
        ]
    )
    runner = SkillRunner(
        _skill(),
        client=client,
        model="gpt-5-mini",
        rest_memory=memory,
        stage_result_mode=True,
    )

    _, result = await skill_utils.process_question_with_runner(
        answer_prompt="Question: {question}\nMemories:\n{memories}",
        runner=runner,
        model=client,
        question="What country is Paris in?",
        answer="France",
        category="wiki",
        supporting_facts=["Paris is in France"],
        model_name="gpt-5-mini",
    )

    assert result["num_episodes_retrieved"] == 1
    assert (
        "[StageResult 1] Query: first hop | Answer: Paris"
        in result["conversation_memories"]
    )
    assert (
        "[StageResult 2] Query: second hop | Answer: France"
        in result["conversation_memories"]
    )
    assert "Paris is in France" in result["conversation_memories"]
    assert "Paris is a capital city" not in result["conversation_memories"]
    assert result["stage_results"] == [
        {
            "query": "first hop",
            "stage_result": "Paris",
            "confidence_score": 0.93,
            "reason_note": "explicit memory",
        },
        {
            "query": "second hop",
            "stage_result": "France",
            "confidence_score": 0.94,
            "reason_note": "explicit memory",
        },
    ]
    assert result["latest_stage_results"] == result["stage_results"]
    assert result["stage_sub_queries"] == ["second hop"]
    assert result["returned_stage_result_count"] == 2
    assert result["returned_sub_query_count"] == 1
    assert result["stage_result_memory_returned"] is True
    assert result["open_domain_rescue_used"] is False
    assert result["answer_verification_used"] is True


@pytest.mark.asyncio
async def test_process_question_with_runner_records_search_queries_and_all_hops_for_fact_matching():
    client = _FakeOpenAIClient(
        [
            _FakeResponse(
                response_id="resp-1",
                output_text="",
                output=[
                    {
                        "type": "function_call",
                        "name": "memmachine_search",
                        "arguments": json.dumps({"query": "first hop"}),
                        "call_id": "call-1",
                    }
                ],
                usage=_FakeUsage(10, 2),
            ),
            _FakeResponse(
                response_id="resp-2",
                output_text="",
                output=[
                    {
                        "type": "function_call",
                        "name": "memmachine_search",
                        "arguments": json.dumps({"query": "second hop"}),
                        "call_id": "call-2",
                    }
                ],
                usage=_FakeUsage(8, 2),
            ),
            _FakeResponse(
                response_id="resp-3",
                output_text="France",
                output=[],
                usage=_FakeUsage(5, 1),
            ),
            _FakeResponse(
                response_id="resp-4",
                output_text="France",
                output=[],
                usage=_FakeUsage(9, 2),
            ),
            _FakeResponse(
                response_id="resp-5",
                output_text="France",
                output=[],
                usage=_FakeUsage(9, 2),
            ),
        ]
    )
    memory = _FakeRestMemory(
        [
            _search_result(("ep-1", "Paris is in France")),
            _search_result(("ep-2", "France is in Europe")),
        ]
    )
    runner = SkillRunner(
        _skill(),
        client=client,
        model="gpt-5-mini",
        rest_memory=memory,
    )

    category, result = await skill_utils.process_question_with_runner(
        answer_prompt="Question: {question}\nMemories:\n{memories}",
        runner=runner,
        model=client,
        question="What country is Paris in?",
        answer="France",
        category="wiki",
        supporting_facts=["Paris is in France", "France is in Europe"],
        model_name="gpt-5-mini",
    )

    assert category == "wiki"
    assert result["memory_search_queries"] == ["first hop", "second hop"]
    assert [item["query"] for item in result["memory_search_trace"]] == [
        "first hop",
        "second hop",
    ]
    assert "Paris is in France" in result["conversation_memories"]
    assert "France is in Europe" in result["conversation_memories"]
    assert "Paris is in France" in result["all_retrieved_memories"]
    assert "France is in Europe" in result["all_retrieved_memories"]

    attribute_matrix = skill_utils.init_attribute_matrix()
    results: dict[str, object] = {}
    skill_utils.update_results([(category, result)], attribute_matrix, results)

    response = results["wiki"][0]
    assert len(response["fact_hits"]) == 2
    assert response["fact_miss"] == []


def test_build_runner_perf_metrics_marks_low_confidence_stage_results_not_returned():
    runner = SkillRunner(
        _skill(),
        client=_FakeOpenAIClient([]),
        model="gpt-5-mini",
        rest_memory=_FakeRestMemory([]),
        stage_result_mode=True,
        stage_result_confidence_threshold=0.9,
    )
    runner.last_search_results = [{"query": "first hop", "count": 1, "total_count": 1}]
    runner.last_memory_search_latency_seconds = [0.25]
    runner.last_stage_results = [
        {
            "query": "first hop",
            "stage_result": "tentative answer",
            "confidence_score": 0.55,
            "reason_note": "not enough evidence",
        }
    ]

    metrics = skill_utils._build_runner_perf_metrics(  # noqa: SLF001
        runner=runner,
        final_answer="tentative answer",
        retrieval_duration=1.0,
    )

    assert metrics["memory_search_queries"] == ["first hop"]
    assert metrics["memory_search_trace"] == [
        {"query": "first hop", "latency_seconds": 0.25, "count": 1, "total_count": 1}
    ]
    assert metrics["memory_retrieval_time"] == 0.25
    assert metrics["retrieval_wall_time"] == 1.0
    assert metrics["llm_time"] == 0.75
    assert metrics["stage_results"][0]["confidence_score"] == 0.55
    assert metrics["stage_result_memory_returned"] is False
    assert metrics["returned_stage_result_count"] == 0


def test_needs_answer_verification_for_multi_hop_retrieval() -> None:
    assert skill_utils._needs_answer_verification(  # noqa: SLF001
        question="When did [person]'s husband die?",
        perf_metrics={
            "memory_search_called": 2,
            "top_level_is_sufficient": True,
            "top_level_confidence_score": 0.99,
            "stage_results": [],
        },
        draft_answer="March 7, 1983",
    )


def test_needs_answer_verification_for_stage_results() -> None:
    assert skill_utils._needs_answer_verification(  # noqa: SLF001
        question="Where was the director of [film] born?",
        perf_metrics={
            "memory_search_called": 1,
            "top_level_is_sufficient": True,
            "top_level_confidence_score": 0.99,
            "stage_results": [
                {
                    "query": "director of [film]",
                    "stage_result": "Alex Example",
                    "confidence_score": 0.95,
                    "reason_note": "explicit memory",
                }
            ],
        },
        draft_answer="Alexandria, Egypt",
    )
