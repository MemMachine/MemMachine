from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

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
        ]
    )
    memory = _FakeRestMemory(
        [
            _search_result(("ep-1", "Paris is in France"), ("ep-2", "France is in Europe")),
            _search_result(("ep-1", "Paris is in France"), ("ep-2", "France is in Europe")),
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
    assert result["llm_call_count"] == 3
    assert runner.last_search_results == []
    assert memory.calls[0]["limit"] == 20
    assert memory.calls[1]["limit"] == 20


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
