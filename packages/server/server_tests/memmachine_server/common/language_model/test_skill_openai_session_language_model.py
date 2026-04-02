from __future__ import annotations

from typing import Any, cast
from unittest.mock import AsyncMock, patch

import openai
import pytest

from memmachine_server.common.language_model import (
    ProviderSkillBundle,
    SkillLanguageModel,
    SkillLanguageModelError,
    SkillOpenAISessionLanguageModelParams,
    SkillSessionLimitError,
    SkillToolNotFoundError,
)


@pytest.fixture
def mock_async_openai_client():
    with patch("openai.AsyncOpenAI", spec=openai.AsyncOpenAI):
        client = openai.AsyncOpenAI(api_key="test-key")
        cast(Any, client.responses).create = AsyncMock()
        yield client


def _as_any_dict(value: object) -> dict[str, Any]:
    assert isinstance(value, dict)
    return cast(dict[str, Any], value)


def _as_any_list(value: object) -> list[Any]:
    assert isinstance(value, list)
    return cast(list[Any], value)


@pytest.mark.asyncio
async def test_openai_live_session_chains_previous_response_id(
    mock_async_openai_client: Any,
) -> None:
    mock_async_openai_client.responses.create.side_effect = [
        {
            "id": "resp_1",
            "output_text": "",
            "usage": {"input_tokens": 2, "output_tokens": 1},
            "output": [
                {
                    "type": "function_call",
                    "name": "lookup",
                    "arguments": '{"query":"alpha"}',
                    "call_id": "c1",
                }
            ],
        },
        {
            "id": "resp_2",
            "output_text": "final answer",
            "usage": {"input_tokens": 3, "output_tokens": 2},
            "output": [],
        },
    ]

    model = SkillLanguageModel(
        SkillOpenAISessionLanguageModelParams(
            client=mock_async_openai_client,
            model="gpt-5",
        )
    )

    async def lookup(arguments: dict[str, Any]) -> dict[str, Any]:
        return {"query": arguments["query"], "hits": 2}

    result = await model.run_live_session(
        system_prompt="sys",
        user_prompt="find alpha",
        tools=[
            {"type": "function", "name": "lookup", "parameters": {"type": "object"}}
        ],
        tool_registry={"lookup": lookup},
    )

    assert result.final_response == "final answer"
    assert result.turn_count == 2
    assert result.llm_input_tokens == 5
    assert result.llm_output_tokens == 3
    assert result.llm_time_seconds >= 0.0
    assert len(result.tool_executions) == 1
    assert result.tool_executions[0].name == "lookup"
    assert result.tool_executions[0].call_id == "c1"

    assert mock_async_openai_client.responses.create.await_count == 2
    second_call = mock_async_openai_client.responses.create.await_args_list[1]
    kwargs = second_call.kwargs
    assert kwargs["previous_response_id"] == "resp_1"
    followup_input = _as_any_list(kwargs["input"])
    assert len(followup_input) == 1
    first_followup = _as_any_dict(followup_input[0])
    assert first_followup["type"] == "function_call_output"
    assert first_followup["call_id"] == "c1"


@pytest.mark.asyncio
async def test_openai_live_session_uses_only_function_call_outputs_in_followup_input(
    mock_async_openai_client: Any,
) -> None:
    mock_async_openai_client.responses.create.side_effect = [
        {
            "id": "resp_1",
            "output_text": "",
            "usage": {"input_tokens": 1, "output_tokens": 1},
            "output": [
                {"type": "reasoning", "id": "rs_1", "summary": []},
                {
                    "type": "function_call",
                    "name": "lookup",
                    "arguments": '{"query":"alpha"}',
                    "call_id": "c1",
                },
            ],
        },
        {
            "id": "resp_2",
            "output_text": "done",
            "usage": {"input_tokens": 1, "output_tokens": 1},
            "output": [],
        },
    ]

    model = SkillLanguageModel(
        SkillOpenAISessionLanguageModelParams(
            client=mock_async_openai_client,
            model="gpt-5",
        )
    )

    async def lookup(arguments: dict[str, Any]) -> dict[str, Any]:
        return {"query": arguments["query"], "hits": 9}

    _result = await model.run_live_session(
        system_prompt="sys",
        user_prompt="find alpha",
        tools=[
            {"type": "function", "name": "lookup", "parameters": {"type": "object"}}
        ],
        tool_registry={"lookup": lookup},
    )

    second_call = mock_async_openai_client.responses.create.await_args_list[1]
    followup_input = _as_any_list(second_call.kwargs["input"])
    assert len(followup_input) == 1
    first_followup = _as_any_dict(followup_input[0])
    assert first_followup["type"] == "function_call_output"
    assert first_followup["call_id"] == "c1"
    assert '"hits": 9' in str(first_followup["output"])
    assert not any(item.get("type") == "reasoning" for item in followup_input)


@pytest.mark.asyncio
async def test_openai_live_session_raises_for_unknown_tool(
    mock_async_openai_client: Any,
) -> None:
    mock_async_openai_client.responses.create.return_value = {
        "id": "resp_1",
        "output_text": "",
        "usage": {"input_tokens": 1, "output_tokens": 1},
        "output": [
            {
                "type": "function_call",
                "name": "lookup",
                "arguments": '{"query":"alpha"}',
                "call_id": "c1",
            }
        ],
    }

    model = SkillLanguageModel(
        SkillOpenAISessionLanguageModelParams(
            client=mock_async_openai_client,
            model="gpt-5",
        )
    )

    with pytest.raises(SkillToolNotFoundError):
        await model.run_live_session(
            system_prompt="sys",
            user_prompt="find alpha",
            tools=[
                {"type": "function", "name": "lookup", "parameters": {"type": "object"}}
            ],
            tool_registry={},
        )


@pytest.mark.asyncio
async def test_openai_live_session_respects_max_turns(
    mock_async_openai_client: Any,
) -> None:
    mock_async_openai_client.responses.create.return_value = {
        "id": "resp_1",
        "output_text": "",
        "usage": {"input_tokens": 1, "output_tokens": 1},
        "output": [
            {
                "type": "function_call",
                "name": "lookup",
                "arguments": '{"query":"alpha"}',
                "call_id": "c1",
            }
        ],
    }

    model = SkillLanguageModel(
        SkillOpenAISessionLanguageModelParams(
            client=mock_async_openai_client,
            model="gpt-5",
        )
    )

    async def lookup(arguments: dict[str, Any]) -> dict[str, Any]:
        return {"query": arguments["query"], "hits": 1}

    with pytest.raises(SkillSessionLimitError):
        await model.run_live_session(
            system_prompt="sys",
            user_prompt="find alpha",
            tools=[
                {"type": "function", "name": "lookup", "parameters": {"type": "object"}}
            ],
            tool_registry={"lookup": lookup},
            max_turns=1,
        )


@pytest.mark.asyncio
async def test_openai_live_session_attaches_local_skills_when_enabled(
    mock_async_openai_client: Any,
    tmp_path,
) -> None:
    skill_dir = tmp_path / "retrieve-skill"
    skill_dir.mkdir()
    (skill_dir / "SKILL.md").write_text("# retrieve skill", encoding="utf-8")
    mock_async_openai_client.responses.create.return_value = {
        "id": "resp_1",
        "output_text": "done",
        "usage": {"input_tokens": 1, "output_tokens": 1},
        "output": [],
    }

    model = SkillLanguageModel(
        SkillOpenAISessionLanguageModelParams(
            client=mock_async_openai_client,
            model="gpt-5",
        )
    )
    bundles = [
        ProviderSkillBundle(
            name="retrieve-skill",
            description="Retrieve skill bundle",
            path=str(skill_dir),
        )
    ]

    _ = await model.run_live_session(
        system_prompt="sys",
        user_prompt="hello",
        tools=[
            {"type": "function", "name": "lookup", "parameters": {"type": "object"}}
        ],
        tool_registry={},
        provider_skill_bundles=bundles,
    )

    first_call = mock_async_openai_client.responses.create.await_args_list[0]
    request_tools = _as_any_list(first_call.kwargs["tools"])
    first_tool = _as_any_dict(request_tools[0])
    assert first_tool["type"] == "shell"
    environment = _as_any_dict(first_tool["environment"])
    assert environment["type"] == "local"
    skills = environment["skills"]
    assert isinstance(skills, list)
    assert len(skills) == 1
    first_skill = _as_any_dict(skills[0])
    assert first_skill["name"] == "retrieve-skill"
    assert first_skill["description"] == "Retrieve skill bundle"
    assert first_skill["path"] == str(skill_dir)
    first_input = _as_any_list(first_call.kwargs["input"])
    first_message = _as_any_dict(first_input[0])
    assert first_message["role"] == "system"
    assert first_message["content"] == "sys"


@pytest.mark.asyncio
async def test_openai_live_session_surfaces_error_diagnostics(
    mock_async_openai_client: Any,
) -> None:
    mock_async_openai_client.responses.create.side_effect = openai.OpenAIError(
        "non-retryable failure"
    )
    model = SkillLanguageModel(
        SkillOpenAISessionLanguageModelParams(
            client=mock_async_openai_client,
            model="gpt-5",
        )
    )

    with pytest.raises(SkillLanguageModelError) as exc_info:
        await model.run_live_session(
            system_prompt="sys",
            user_prompt="hello",
            tools=[],
            tool_registry={},
        )

    diagnostics = exc_info.value.diagnostics
    assert diagnostics.get("provider") == "openai"
    assert diagnostics.get("operation") == "responses.create"
    assert diagnostics.get("error_type") == "OpenAIError"


@pytest.mark.asyncio
async def test_openai_live_session_executes_shell_call_in_local_mode(
    mock_async_openai_client: Any,
    tmp_path,
) -> None:
    skill_dir = tmp_path / "retrieve-skill"
    skill_dir.mkdir()
    (skill_dir / "SKILL.md").write_text("# retrieve skill", encoding="utf-8")
    mock_async_openai_client.responses.create.side_effect = [
        {
            "id": "resp_1",
            "output_text": "",
            "usage": {"input_tokens": 1, "output_tokens": 1},
            "output": [
                {
                    "type": "shell_call",
                    "id": "sh_1",
                    "action": {
                        "type": "exec",
                        "commands": ["echo hello"],
                        "timeout_ms": 5000,
                        "max_output_length": 2048,
                    },
                }
            ],
        },
        {
            "id": "resp_2",
            "output_text": "done",
            "usage": {"input_tokens": 1, "output_tokens": 1},
            "output": [],
        },
    ]

    model = SkillLanguageModel(
        SkillOpenAISessionLanguageModelParams(
            client=mock_async_openai_client,
            model="gpt-5",
        )
    )
    bundles = [
        ProviderSkillBundle(
            name="retrieve-skill",
            description="Retrieve skill bundle",
            path=str(skill_dir),
        )
    ]

    result = await model.run_live_session(
        system_prompt="sys",
        user_prompt="hello",
        tools=[],
        tool_registry={},
        provider_skill_bundles=bundles,
    )

    assert result.final_response == "done"
    assert result.turn_count == 2
    assert len(result.tool_executions) == 1
    assert result.tool_executions[0].name == "shell_call"
    assert isinstance(result.tool_executions[0].output, dict)
    tool_output = _as_any_dict(result.tool_executions[0].output)
    output_items = _as_any_list(tool_output.get("output"))
    outcome = _as_any_dict(_as_any_dict(output_items[0])["outcome"])
    assert outcome["type"] == "exit"
    second_call = mock_async_openai_client.responses.create.await_args_list[1]
    followup_input = _as_any_list(second_call.kwargs["input"])
    first_followup = _as_any_dict(followup_input[0])
    assert first_followup["type"] == "shell_call_output"
    assert first_followup["call_id"] == "sh_1"
