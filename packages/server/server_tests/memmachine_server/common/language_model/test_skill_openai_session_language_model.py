from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, patch

import openai
import pytest

from memmachine_server.common.language_model import (
    SkillLanguageModel,
    SkillOpenAISessionLanguageModelParams,
    SkillSessionLimitError,
    SkillToolNotFoundError,
)


@pytest.fixture
def mock_async_openai_client():
    with patch("openai.AsyncOpenAI", spec=openai.AsyncOpenAI):
        client = openai.AsyncOpenAI(api_key="test-key")
        client.responses.create = AsyncMock()
        yield client


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
    followup_input = kwargs["input"]
    assert followup_input[0]["role"] == "system"
    assert followup_input[1]["role"] == "user"
    assert followup_input[2]["type"] == "function_call_output"
    assert followup_input[2]["call_id"] == "c1"


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
    followup_input = second_call.kwargs["input"]
    assert len(followup_input) == 3
    assert followup_input[0]["role"] == "system"
    assert followup_input[1]["role"] == "user"
    assert followup_input[2]["type"] == "function_call_output"
    assert followup_input[2]["call_id"] == "c1"
    assert '"hits": 9' in followup_input[2]["output"]
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
