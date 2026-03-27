"""Unit tests for shared provider-native SkillRunner behavior."""

from __future__ import annotations

import json
import time
from typing import Any, cast

import pytest

from memmachine_common import Skill, SkillRunner


class _FakeResponsesAPI:
    def __init__(self, follow_up_responses: list[object]) -> None:
        self.calls: list[dict[str, object]] = []
        self._follow_up_responses = list(follow_up_responses)

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        return self._follow_up_responses.pop(0)


class _FakeOpenAIClient:
    def __init__(self, follow_up_responses: list[object] | None = None) -> None:
        self.responses = _FakeResponsesAPI(follow_up_responses or [])


class _FakeMessagesAPI:
    def __init__(self, follow_up_responses: list[object]) -> None:
        self.calls: list[dict[str, object]] = []
        self._follow_up_responses = list(follow_up_responses)

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        return self._follow_up_responses.pop(0)


class _FakeAnthropicClient:
    def __init__(self, follow_up_responses: list[object] | None = None) -> None:
        self.messages = _FakeMessagesAPI(follow_up_responses or [])


class _FakeRestMemory:
    def __init__(self, result: dict[str, object] | None = None) -> None:
        self.calls: list[dict[str, object]] = []
        self._result = result or {
            "content": {
                "episodic_memory": {
                    "short_term_memory": {
                        "episodes": [{"uid": "s1", "content": "short memory"}],
                        "episode_summary": ["summary line"],
                    },
                    "long_term_memory": {
                        "episodes": [{"uid": "l1", "content": "long memory"}]
                    },
                }
            }
        }

    def search(self, query: str, **kwargs):
        self.calls.append({"query": query, **kwargs})
        return self._result


def _skill(provider: str = "openai") -> Skill:
    return Skill(
        provider=provider,  # type: ignore[arg-type]
        skill_name="retrieve-skill",
        file_ids=("file-1", "file-2"),
        content_hashes=("hash-1", "hash-2"),
    )


def _as_any_dict(value: object) -> dict[str, Any]:
    assert isinstance(value, dict)
    return cast(dict[str, Any], value)


def _as_any_list(value: object) -> list[Any]:
    assert isinstance(value, list)
    return cast(list[Any], value)


def test_skill_messages_openai_uses_input_file_blocks():
    runner = SkillRunner(
        _skill("openai"),
        client=_FakeOpenAIClient(),
        model="gpt-5-mini",
        rest_memory=_FakeRestMemory(),
    )

    messages = runner.skill_messages("Where is the answer?")

    assert messages == [
        {"type": "input_file", "file_id": "file-1"},
        {"type": "input_file", "file_id": "file-2"},
        {"type": "input_text", "text": "Where is the answer?"},
    ]


def test_skill_messages_anthropic_uses_document_blocks():
    runner = SkillRunner(
        _skill("anthropic"),
        client=_FakeAnthropicClient(),
        model="claude-sonnet",
        rest_memory=_FakeRestMemory(),
    )

    messages = runner.skill_messages("Where is the answer?", system_prompt="system")

    assert messages == [
        {"type": "document", "source": {"type": "file", "file_id": "file-1"}},
        {"type": "document", "source": {"type": "file", "file_id": "file-2"}},
        {"type": "text", "text": "Where is the answer?"},
    ]


def test_tools_are_provider_specific():
    openai_runner = SkillRunner(
        _skill("openai"),
        client=_FakeOpenAIClient(),
        model="gpt-5-mini",
        rest_memory=_FakeRestMemory(),
    )
    anthropic_runner = SkillRunner(
        _skill("anthropic"),
        client=_FakeAnthropicClient(),
        model="claude-sonnet",
        rest_memory=_FakeRestMemory(),
    )

    openai_tool = openai_runner.tools()[0]
    anthropic_tool = anthropic_runner.tools()[0]
    openai_parameters = _as_any_dict(openai_tool["parameters"])
    anthropic_input_schema = _as_any_dict(anthropic_tool["input_schema"])

    assert openai_parameters["required"] == ["query"]
    assert anthropic_input_schema["required"] == ["query"]


@pytest.mark.asyncio
async def test_handle_tool_loop_openai_rest_mode_uses_previous_response_id():
    client = _FakeOpenAIClient(
        follow_up_responses=[
            {"id": "resp-2", "output_text": "Final answer", "output": []}
        ]
    )
    memory = _FakeRestMemory()
    runner = SkillRunner(
        _skill("openai"),
        client=client,
        model="gpt-5-mini",
        rest_memory=memory,
    )

    response = {
        "id": "resp-1",
        "output_text": "",
        "output": [
            {
                "type": "function_call",
                "name": "memmachine_search",
                "arguments": json.dumps({"query": "hello"}),
                "call_id": "call-1",
            }
        ],
    }

    result = await runner.handle_tool_loop(response)

    assert result == "Final answer"
    assert runner.last_llm_call_count == 2
    assert runner.last_tool_call_count == 1
    assert runner.last_memory_search_called == 1
    assert memory.calls == [
        {
            "query": "hello",
            "limit": 20,
            "expand_context": 0,
            "score_threshold": None,
            "agent_mode": False,
        }
    ]
    follow_up_call = client.responses.calls[0]
    assert follow_up_call["previous_response_id"] == "resp-1"
    follow_up_inputs = _as_any_list(follow_up_call["input"])
    raw_output = _as_any_dict(follow_up_inputs[0])["output"]
    assert isinstance(raw_output, str)
    tool_output = json.loads(raw_output)
    assert (
        tool_output["episodes_text"] == "summary line\n1. short memory\n2. long memory"
    )
    assert "episodes" not in tool_output


@pytest.mark.asyncio
async def test_handle_tool_loop_openai_rest_mode_includes_semantic_memory():
    client = _FakeOpenAIClient(
        follow_up_responses=[
            {"id": "resp-2", "output_text": "Final answer", "output": []}
        ]
    )
    memory = _FakeRestMemory(
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
                    {
                        "category": "profile",
                        "tag": "food",
                        "feature_name": "favorite_food",
                        "value": "pizza",
                    }
                ],
            }
        }
    )
    runner = SkillRunner(
        _skill("openai"),
        client=client,
        model="gpt-5-mini",
        rest_memory=memory,
    )

    response = {
        "id": "resp-1",
        "output_text": "",
        "output": [
            {
                "type": "function_call",
                "name": "memmachine_search",
                "arguments": json.dumps({"query": "hello"}),
                "call_id": "call-1",
            }
        ],
    }

    result = await runner.handle_tool_loop(response)

    assert result == "Final answer"
    follow_up_call = client.responses.calls[0]
    follow_up_inputs = _as_any_list(follow_up_call["input"])
    raw_output = _as_any_dict(follow_up_inputs[0])["output"]
    assert isinstance(raw_output, str)
    tool_output = json.loads(raw_output)
    assert tool_output["semantic_memory"] == [
        {
            "category": "profile",
            "tag": "food",
            "feature_name": "favorite_food",
            "value": "pizza",
        }
    ]
    assert tool_output["semantic_text"] == (
        "[Semantic] profile/food | favorite_food: pizza"
    )
    assert tool_output["episodes_text"] == (
        "[Semantic] profile/food | favorite_food: pizza"
    )


@pytest.mark.asyncio
async def test_handle_tool_loop_anthropic_rest_mode_rebuilds_messages():
    client = _FakeAnthropicClient(
        follow_up_responses=[{"content": [{"type": "text", "text": "Final answer"}]}]
    )
    memory = _FakeRestMemory(
        {
            "content": {
                "episodic_memory": {
                    "short_term_memory": {
                        "episodes": [{"uid": "s1", "content": "short memory"}],
                        "episode_summary": ["summary line"],
                    },
                    "long_term_memory": {
                        "episodes": [{"uid": "l1", "content": "long memory"}]
                    },
                }
            }
        }
    )
    runner = SkillRunner(
        _skill("anthropic"),
        client=client,
        model="claude-sonnet",
        rest_memory=memory,
    )
    runner.skill_messages("Where is the answer?", system_prompt="system")

    response = {
        "content": [
            {
                "type": "tool_use",
                "name": "memmachine_search",
                "input": {"query": "hello"},
                "id": "tool-1",
            }
        ]
    }

    result = await runner.handle_tool_loop(response)

    assert result == "Final answer"
    assert runner.last_llm_call_count == 2
    assert runner.last_tool_call_count == 1
    assert runner.last_memory_search_called == 1
    assert memory.calls == [
        {
            "query": "hello",
            "limit": 20,
            "expand_context": 0,
            "score_threshold": None,
            "agent_mode": False,
        }
    ]
    follow_up_call = client.messages.calls[0]
    assert follow_up_call["system"] == "system"
    messages = _as_any_list(follow_up_call["messages"])
    assert len(messages) == 3
    tool_result_container = _as_any_dict(messages[-1])
    tool_result_content = _as_any_list(tool_result_container["content"])
    tool_result = _as_any_dict(tool_result_content[0])
    assert tool_result["type"] == "tool_result"
    tool_result_text = tool_result["content"]
    assert isinstance(tool_result_text, str)
    assert "short memory" in tool_result_text


@pytest.mark.asyncio
async def test_handle_tool_loop_returns_partial_on_max_turns():
    client = _FakeOpenAIClient()
    runner = SkillRunner(
        _skill("openai"),
        client=client,
        model="gpt-5-mini",
        rest_memory=_FakeRestMemory(),
        max_turns=1,
    )

    response = {
        "id": "resp-1",
        "output_text": "Working",
        "output": [
            {
                "type": "function_call",
                "name": "memmachine_search",
                "arguments": json.dumps({"query": "hello"}),
                "call_id": "call-1",
            }
        ],
    }

    result = await runner.handle_tool_loop(response)

    assert result == "Working"
    assert client.responses.calls == []


class _FakeRestMemoryWithScores:
    def __init__(self, episodes: list[dict]) -> None:
        self.calls: list[dict] = []
        self._episodes = episodes

    def search(self, query: str, **kwargs):
        self.calls.append({"query": query, **kwargs})
        return {
            "content": {
                "episodic_memory": {
                    "short_term_memory": {
                        "episodes": self._episodes,
                        "episode_summary": [],
                    },
                    "long_term_memory": {"episodes": []},
                }
            }
        }


class _SlowRestMemory(_FakeRestMemory):
    def search(self, query: str, **kwargs):
        time.sleep(0.001)
        return super().search(query, **kwargs)


@pytest.mark.asyncio
async def test_adaptive_search_limit():
    memory = _FakeRestMemory()
    client = _FakeOpenAIClient(
        follow_up_responses=[
            {
                "id": "resp-2",
                "output_text": "",
                "output": [
                    {
                        "type": "function_call",
                        "name": "memmachine_search",
                        "arguments": json.dumps({"query": "second query"}),
                        "call_id": "call-2",
                    }
                ],
            },
            {
                "id": "resp-3",
                "output_text": "Done",
                "output": [],
            },
        ]
    )
    runner = SkillRunner(
        _skill("openai"),
        client=client,
        model="gpt-5-mini",
        rest_memory=memory,
        adaptive_search_limit={"initial": 3, "escalated": 15},
    )

    response = {
        "id": "resp-1",
        "output_text": "",
        "output": [
            {
                "type": "function_call",
                "name": "memmachine_search",
                "arguments": json.dumps({"query": "first query"}),
                "call_id": "call-1",
            }
        ],
    }

    result = await runner.handle_tool_loop(response)

    assert result == "Done"
    assert memory.calls[0]["limit"] == 3
    assert memory.calls[1]["limit"] == 15


def test_fork_copies_configuration_but_resets_state():
    runner = SkillRunner(
        _skill("openai"),
        client=_FakeOpenAIClient(),
        model="gpt-5-mini",
        rest_memory=_FakeRestMemory(),
        search_limit=7,
        expand_context=2,
        score_threshold=0.6,
        adaptive_search_limit={"initial": 3, "escalated": 9},
        max_episode_chars=100,
        early_exit_confidence=True,
        query_dedup=True,
        stage_result_mode=True,
        stage_result_confidence_threshold=0.9,
        omit_episode_text_on_confident_stage_result=True,
        use_answer_prompt_template=True,
    )
    runner.last_search_results = [
        {"query": "hello", "episodes_text": "cached", "count": 1}
    ]
    runner.last_memory_search_called = 2
    runner.last_memory_search_latency_seconds = [0.1, 0.2]
    runner.last_stage_results = [{"query": "hello", "stage_result": "world"}]
    runner.last_stage_sub_queries = ["next question"]
    runner.last_llm_call_count = 3
    runner.last_tool_call_count = 4
    runner._seen_queries.add(frozenset({"hello"}))

    forked = runner.fork()

    assert forked is not runner
    assert forked.skill == runner.skill
    assert forked.provider == runner.provider
    assert forked.search_limit == 7
    assert forked.expand_context == 2
    assert forked.score_threshold == 0.6
    assert forked.adaptive_search_limit == {"initial": 3, "escalated": 9}
    assert forked.max_episode_chars == 100
    assert forked.early_exit_confidence is True
    assert forked.query_dedup is True
    assert forked.stage_result_mode is True
    assert forked.stage_result_confidence_threshold == 0.9
    assert forked.omit_episode_text_on_confident_stage_result is True
    assert forked.use_answer_prompt_template is True
    assert forked.last_search_results == []
    assert forked.last_memory_search_called == 0
    assert forked.last_memory_search_latency_seconds == []
    assert forked.last_stage_results == []
    assert forked.last_stage_sub_queries == []
    assert forked.last_llm_call_count == 0
    assert forked.last_tool_call_count == 0
    assert forked._seen_queries == set()


@pytest.mark.asyncio
async def test_run_search_records_latency():
    runner = SkillRunner(
        _skill("openai"),
        client=_FakeOpenAIClient(),
        model="gpt-5-mini",
        rest_memory=_SlowRestMemory(),
    )

    result = await runner._run_search("hello")

    assert result["query"] == "hello"
    assert "episodes" not in result
    assert result["episodes_text"] == "summary line\n1. short memory\n2. long memory"
    assert runner.last_memory_search_called == 1
    assert len(runner.last_memory_search_latency_seconds) == 1
    assert runner.last_memory_search_latency_seconds[0] > 0
    episodes = _as_any_list(runner.last_search_results[0]["episodes"])
    assert _as_any_dict(episodes[0])["uid"] == "s1"


@pytest.mark.asyncio
async def test_max_episode_chars():
    memory = _FakeRestMemoryWithScores(
        episodes=[{"uid": "e1", "content": "hello world this is long"}]
    )
    runner = SkillRunner(
        _skill("openai"),
        client=_FakeOpenAIClient(),
        model="gpt-5-mini",
        rest_memory=memory,
        max_episode_chars=5,
    )

    result = runner._normalize_search_result(
        query="test",
        raw_result=memory.search("test"),
    )

    episodes_text = result["episodes_text"]
    assert isinstance(episodes_text, str)
    assert "hello" in episodes_text
    assert "hello world" not in episodes_text


@pytest.mark.asyncio
async def test_client_side_score_threshold():
    memory = _FakeRestMemoryWithScores(
        episodes=[
            {"uid": "high", "content": "high score ep", "score": 0.9},
            {"uid": "low", "content": "low score ep", "score": 0.5},
        ]
    )
    runner = SkillRunner(
        _skill("openai"),
        client=_FakeOpenAIClient(),
        model="gpt-5-mini",
        rest_memory=memory,
        score_threshold=0.7,
    )

    result = runner._normalize_search_result(
        query="test",
        raw_result=memory.search("test"),
    )

    episodes = _as_any_list(result["episodes"])
    assert len(episodes) == 1
    assert _as_any_dict(episodes[0])["uid"] == "high"
    episodes_text = result["episodes_text"]
    assert isinstance(episodes_text, str)
    assert "low score ep" not in episodes_text


@pytest.mark.asyncio
async def test_early_exit_on_high_confidence():
    memory = _FakeRestMemory()
    client = _FakeOpenAIClient(
        follow_up_responses=[
            {
                "id": "resp-2",
                "output_text": "The answer is Paris.",
                "output": [],
            },
            {
                "id": "resp-3",
                "output_text": "Should not reach here",
                "output": [],
            },
        ]
    )
    runner = SkillRunner(
        _skill("openai"),
        client=client,
        model="gpt-5-mini",
        rest_memory=memory,
        early_exit_confidence=True,
    )

    response = {
        "id": "resp-1",
        "output_text": "",
        "output": [
            {
                "type": "function_call",
                "name": "memmachine_search",
                "arguments": json.dumps({"query": "capital of France"}),
                "call_id": "call-1",
            }
        ],
    }

    result = await runner.handle_tool_loop(response)

    assert result == "The answer is Paris."
    assert runner.last_llm_call_count == 2
    assert runner.last_memory_search_called == 1
    assert len(client.responses.calls) == 1


@pytest.mark.asyncio
async def test_query_deduplication():
    memory = _FakeRestMemory()
    client = _FakeOpenAIClient(
        follow_up_responses=[
            {
                "id": "resp-2",
                "output_text": "",
                "output": [
                    {
                        "type": "function_call",
                        "name": "memmachine_search",
                        "arguments": json.dumps({"query": "hello"}),
                        "call_id": "call-2",
                    }
                ],
            },
            {
                "id": "resp-3",
                "output_text": "Deduped answer",
                "output": [],
            },
        ]
    )
    runner = SkillRunner(
        _skill("openai"),
        client=client,
        model="gpt-5-mini",
        rest_memory=memory,
        query_dedup=True,
    )

    response = {
        "id": "resp-1",
        "output_text": "",
        "output": [
            {
                "type": "function_call",
                "name": "memmachine_search",
                "arguments": json.dumps({"query": "hello"}),
                "call_id": "call-1",
            }
        ],
    }

    result = await runner.handle_tool_loop(response)

    assert result == "Deduped answer"
    assert len(memory.calls) == 1
    assert runner.last_memory_search_called == 1


@pytest.mark.asyncio
async def test_query_dedup_disabled_by_default():
    memory = _FakeRestMemory()
    client = _FakeOpenAIClient(
        follow_up_responses=[
            {
                "id": "resp-2",
                "output_text": "",
                "output": [
                    {
                        "type": "function_call",
                        "name": "memmachine_search",
                        "arguments": json.dumps({"query": "hello"}),
                        "call_id": "call-2",
                    }
                ],
            },
            {
                "id": "resp-3",
                "output_text": "No dedup",
                "output": [],
            },
        ]
    )
    runner = SkillRunner(
        _skill("openai"),
        client=client,
        model="gpt-5-mini",
        rest_memory=memory,
    )

    response = {
        "id": "resp-1",
        "output_text": "",
        "output": [
            {
                "type": "function_call",
                "name": "memmachine_search",
                "arguments": json.dumps({"query": "hello"}),
                "call_id": "call-1",
            }
        ],
    }

    result = await runner.handle_tool_loop(response)

    assert result == "No dedup"
    assert len(memory.calls) == 2
    assert runner.last_memory_search_called == 2


@pytest.mark.asyncio
async def test_dedup_does_not_escalate():
    memory = _FakeRestMemory()
    client = _FakeOpenAIClient(
        follow_up_responses=[
            {
                "id": "resp-2",
                "output_text": "",
                "output": [
                    {
                        "type": "function_call",
                        "name": "memmachine_search",
                        "arguments": json.dumps({"query": "hello"}),
                        "call_id": "call-2",
                    }
                ],
            },
            {
                "id": "resp-3",
                "output_text": "Done",
                "output": [],
            },
        ]
    )
    runner = SkillRunner(
        _skill("openai"),
        client=client,
        model="gpt-5-mini",
        rest_memory=memory,
        adaptive_search_limit={"initial": 3, "escalated": 15},
        query_dedup=True,
    )

    response = {
        "id": "resp-1",
        "output_text": "",
        "output": [
            {
                "type": "function_call",
                "name": "memmachine_search",
                "arguments": json.dumps({"query": "hello"}),
                "call_id": "call-1",
            }
        ],
    }

    await runner.handle_tool_loop(response)

    assert len(memory.calls) == 1
    assert memory.calls[0]["limit"] == 3


def test_stage_result_mode_appends_guidance_to_prompt():
    runner = SkillRunner(
        _skill("openai"),
        client=_FakeOpenAIClient(),
        model="gpt-5-mini",
        rest_memory=_FakeRestMemory(),
        stage_result_mode=True,
    )

    messages = runner.skill_messages("Where is the answer?")
    message = _as_any_dict(messages[-1])
    text = message["text"]
    assert isinstance(text, str)

    assert "[StageResult]" in text
    assert "Where is the answer?" in text


@pytest.mark.asyncio
async def test_stage_result_mode_records_stage_progress_and_cleans_final_answer():
    client = _FakeOpenAIClient(
        follow_up_responses=[
            {
                "id": "resp-2",
                "output_text": (
                    "[StageResult] Query: capital of France | Answer: Paris | "
                    "Confidence: 0.93 | Reason: explicit memory\n"
                    "[SubQuery] country of Paris\n"
                    "Paris"
                ),
                "output": [],
                "usage": {"input_tokens": 13, "output_tokens": 7},
            }
        ]
    )
    runner = SkillRunner(
        _skill("openai"),
        client=client,
        model="gpt-5-mini",
        rest_memory=_FakeRestMemory(),
        stage_result_mode=True,
    )

    response = {
        "id": "resp-1",
        "output_text": "",
        "output": [
            {
                "type": "function_call",
                "name": "memmachine_search",
                "arguments": json.dumps({"query": "capital of France"}),
                "call_id": "call-1",
            }
        ],
    }

    result = await runner.handle_tool_loop(response)

    assert result == "Paris"
    assert runner.last_stage_results == [
        {
            "query": "capital of France",
            "stage_result": "Paris",
            "confidence_score": 0.93,
            "reason_note": "explicit memory",
        }
    ]
    assert runner.last_stage_sub_queries == ["country of Paris"]
    assert runner.last_follow_up_input_tokens == 13
    assert runner.last_follow_up_output_tokens == 7


@pytest.mark.asyncio
async def test_stage_result_mode_reads_stage_text_from_message_blocks_when_output_text_empty():
    client = _FakeOpenAIClient(
        follow_up_responses=[
            {
                "id": "resp-2",
                "output_text": "Paris",
                "output": [],
                "usage": {"input_tokens": 13, "output_tokens": 7},
            }
        ]
    )
    runner = SkillRunner(
        _skill("openai"),
        client=client,
        model="gpt-5-mini",
        rest_memory=_FakeRestMemory(),
        stage_result_mode=True,
    )

    response = {
        "id": "resp-1",
        "output_text": "",
        "output": [
            {
                "type": "message",
                "content": [
                    {
                        "type": "output_text",
                        "text": (
                            "[StageResult] Query: capital of France | Answer: Paris | "
                            "Confidence: 0.93 | Reason: explicit memory\n"
                            "[SubQuery] country of Paris"
                        ),
                    }
                ],
            },
            {
                "type": "function_call",
                "name": "memmachine_search",
                "arguments": json.dumps({"query": "capital of France"}),
                "call_id": "call-1",
            },
        ],
    }

    result = await runner.handle_tool_loop(response)

    assert result == "Paris"
    assert runner.last_stage_results == [
        {
            "query": "capital of France",
            "stage_result": "Paris",
            "confidence_score": 0.93,
            "reason_note": "explicit memory",
        }
    ]
    assert runner.last_stage_sub_queries == ["country of Paris"]


@pytest.mark.asyncio
async def test_run_search_canonicalizes_relation_terminal_queries():
    memory = _FakeRestMemory()
    runner = SkillRunner(
        _skill("openai"),
        client=_FakeOpenAIClient(),
        model="gpt-5-mini",
        rest_memory=memory,
    )

    result = await runner._run_search("Alex Carter husband died when")

    assert memory.calls[0]["query"] == "Alex Carter husband"
    assert result["query"] == "Alex Carter husband"
    assert runner.last_search_results[0]["query"] == "Alex Carter husband"

    result = await runner._run_search("Alex Carter (1919 2004) husband death")

    assert memory.calls[1]["query"] == "Alex Carter (1919 2004) husband"
    assert result["query"] == "Alex Carter (1919 2004) husband"

    result = await runner._run_search("cause of death Eleanor Vale")

    assert memory.calls[2]["query"] == "What was the cause of death of Eleanor Vale?"
    assert result["query"] == "What was the cause of death of Eleanor Vale?"

    result = await runner._run_search("cause of death of Eleanor Vale")

    assert memory.calls[3]["query"] == "What was the cause of death of Eleanor Vale?"
    assert result["query"] == "What was the cause of death of Eleanor Vale?"


@pytest.mark.asyncio
async def test_stage_result_mode_canonicalizes_stage_queries_and_subqueries():
    client = _FakeOpenAIClient(
        follow_up_responses=[
            {
                "id": "resp-2",
                "output_text": (
                    "[StageResult] Query: Casey Nolan born where | "
                    "Answer: Brooklyn, New York | Confidence: 0.55 | "
                    "Reason: partial evidence\n"
                    "[SubQuery] Dana Brooks husband died when\n"
                    "Brooklyn, New York"
                ),
                "output": [],
                "usage": {"input_tokens": 13, "output_tokens": 7},
            }
        ]
    )
    runner = SkillRunner(
        _skill("openai"),
        client=client,
        model="gpt-5-mini",
        rest_memory=_FakeRestMemory(),
        stage_result_mode=True,
    )

    response = {
        "id": "resp-1",
        "output_text": "",
        "output": [
            {
                "type": "function_call",
                "name": "memmachine_search",
                "arguments": json.dumps({"query": "director of Thomas Jefferson"}),
                "call_id": "call-1",
            }
        ],
    }

    result = await runner.handle_tool_loop(response)

    assert result == "Brooklyn, New York"
    assert runner.last_stage_results == [
        {
            "query": "Where was Casey Nolan born?",
            "stage_result": "Brooklyn, New York",
            "confidence_score": 0.55,
            "reason_note": "partial evidence",
        }
    ]
    assert runner.last_stage_sub_queries == ["Dana Brooks husband"]


def test_normalize_search_result_includes_stage_result_memory():
    memory = _FakeRestMemory()
    runner = SkillRunner(
        _skill("openai"),
        client=_FakeOpenAIClient(),
        model="gpt-5-mini",
        rest_memory=memory,
        stage_result_mode=True,
    )
    runner.last_stage_results = [
        {
            "query": "capital of France",
            "stage_result": "Paris",
            "confidence_score": 0.93,
            "reason_note": "explicit memory",
        }
    ]
    runner.last_stage_sub_queries = ["country of Paris"]

    result = runner._normalize_search_result(
        query="test",
        raw_result=memory.search("test"),
    )

    assert result["stage_result_memory"] == [
        (
            "[StageResult 1] Query: capital of France | Answer: Paris | "
            "Confidence: 0.93 | Reason: explicit memory"
        ),
        "[SubQuery 1] country of Paris",
    ]


@pytest.mark.asyncio
async def test_run_search_omits_episode_text_when_stage_result_is_confident():
    runner = SkillRunner(
        _skill("openai"),
        client=_FakeOpenAIClient(),
        model="gpt-5-mini",
        rest_memory=_FakeRestMemory(),
        stage_result_mode=True,
        omit_episode_text_on_confident_stage_result=True,
        stage_result_confidence_threshold=0.9,
    )
    runner.last_stage_results = [
        {
            "query": "capital of France",
            "stage_result": "Paris",
            "confidence_score": 0.93,
            "reason_note": "explicit memory",
        }
    ]

    result = await runner._run_search("capital of France")

    assert "episodes_text" not in result
    assert result["stage_result_memory"] == [
        (
            "[StageResult 1] Query: capital of France | Answer: Paris | "
            "Confidence: 0.93 | Reason: explicit memory"
        )
    ]
    assert runner.last_search_results[0]["episodes_text"] == (
        "summary line\n1. short memory\n2. long memory"
    )


@pytest.mark.asyncio
async def test_run_search_keeps_episode_text_when_stage_result_is_not_confident():
    runner = SkillRunner(
        _skill("openai"),
        client=_FakeOpenAIClient(),
        model="gpt-5-mini",
        rest_memory=_FakeRestMemory(),
        stage_result_mode=True,
        omit_episode_text_on_confident_stage_result=True,
        stage_result_confidence_threshold=0.9,
    )
    runner.last_stage_results = [
        {
            "query": "capital of France",
            "stage_result": "Paris",
            "confidence_score": 0.6,
        }
    ]

    result = await runner._run_search("capital of France")

    assert result["episodes_text"] == "summary line\n1. short memory\n2. long memory"
