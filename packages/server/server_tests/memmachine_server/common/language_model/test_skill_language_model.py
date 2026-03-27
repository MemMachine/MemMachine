from __future__ import annotations

from typing import Any, cast
from unittest.mock import AsyncMock, MagicMock, call, patch

import openai
import pytest

from memmachine_server.common.language_model import (
    SkillLanguageModel,
    SkillLanguageModelError,
    SkillOpenAISessionLanguageModelParams,
    SkillToolCallFormatError,
)
from memmachine_server.common.language_model.openai_responses_language_model import (
    OpenAIResponsesLanguageModel,
    OpenAIResponsesLanguageModelParams,
)


@pytest.fixture
def mock_async_openai_client():
    with patch("openai.AsyncOpenAI", spec=openai.AsyncOpenAI):
        client = openai.AsyncOpenAI(api_key="test-key")
        cast(Any, client.responses).create = AsyncMock()
        yield client


def test_from_openai_responses_language_model_success(
    mock_async_openai_client: Any,
) -> None:
    base = OpenAIResponsesLanguageModel(
        OpenAIResponsesLanguageModelParams(
            client=mock_async_openai_client,
            model="gpt-5-mini",
            max_retry_interval_seconds=17,
            reasoning_effort="low",
        )
    )

    runtime = SkillLanguageModel.from_openai_responses_language_model(base)
    assert isinstance(runtime, SkillLanguageModel)


def test_from_openai_responses_language_model_rejects_non_openai() -> None:
    with pytest.raises(TypeError, match="requires OpenAIResponsesLanguageModel"):
        _ = SkillLanguageModel.from_openai_responses_language_model(object())


@pytest.mark.asyncio
@patch("asyncio.sleep", new_callable=AsyncMock)
async def test_run_live_session_retries_on_rate_limit(
    mock_sleep: AsyncMock,
    mock_async_openai_client: Any,
) -> None:
    mock_async_openai_client.responses.create.side_effect = [
        openai.RateLimitError("rate limited", response=MagicMock(), body=None),
        {
            "id": "resp_2",
            "output_text": "done",
            "usage": {"input_tokens": 3, "output_tokens": 2},
            "output": [],
        },
    ]
    runtime = SkillLanguageModel(
        SkillOpenAISessionLanguageModelParams(
            client=mock_async_openai_client,
            model="gpt-5-mini",
            max_retry_interval_seconds=8,
        )
    )

    result = await runtime.run_live_session(
        system_prompt="sys",
        user_prompt="hello",
        tools=[],
        tool_registry={},
    )

    assert result.final_response == "done"
    assert mock_async_openai_client.responses.create.await_count == 2
    mock_sleep.assert_awaited_once_with(1)


@pytest.mark.asyncio
async def test_run_live_session_raises_for_invalid_arguments_shape(
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
                "arguments": "[]",
                "call_id": "c1",
            }
        ],
    }
    runtime = SkillLanguageModel(
        SkillOpenAISessionLanguageModelParams(
            client=mock_async_openai_client,
            model="gpt-5-mini",
        )
    )

    with pytest.raises(SkillToolCallFormatError):
        await runtime.run_live_session(
            system_prompt="sys",
            user_prompt="hello",
            tools=[{"type": "function", "name": "lookup", "parameters": {"type": "object"}}],
            tool_registry={"lookup": lambda args: args},
        )


@pytest.mark.asyncio
@patch("asyncio.sleep", new_callable=AsyncMock)
async def test_run_live_session_raises_after_retry_exhausted(
    mock_sleep: AsyncMock,
    mock_async_openai_client: Any,
) -> None:
    mock_async_openai_client.responses.create.side_effect = openai.RateLimitError(
        "rate limited",
        response=MagicMock(),
        body=None,
    )
    runtime = SkillLanguageModel(
        SkillOpenAISessionLanguageModelParams(
            client=mock_async_openai_client,
            model="gpt-5-mini",
            max_retry_interval_seconds=2,
        )
    )

    with pytest.raises(SkillLanguageModelError):
        await runtime.run_live_session(
            system_prompt="sys",
            user_prompt="hello",
            tools=[],
            tool_registry={},
        )

    assert mock_async_openai_client.responses.create.await_count == 3
    mock_sleep.assert_has_awaits([call(1), call(2)])
