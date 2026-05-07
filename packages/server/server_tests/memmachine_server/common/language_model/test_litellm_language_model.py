"""Unit tests for LiteLLMLanguageModel.

LiteLLM normalizes every backing's response to OpenAI shape, so the test
fixtures here construct OpenAI-shaped `ChatCompletion` mocks and assert that
`LiteLLMLanguageModel` (subclass of `OpenAIChatCompletionsLanguageModel`)
delegates the parsing to the parent class while routing the request through
`litellm.acompletion`.
"""

import sys
import types
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest
from openai.types import chat as openai_chat

# Inject a fake litellm module so tests don't require the real package.
if "litellm" not in sys.modules:
    _fake_litellm = types.ModuleType("litellm")
    _fake_litellm.acompletion = AsyncMock()
    sys.modules["litellm"] = _fake_litellm

from memmachine_server.common.data_types import ExternalServiceAPIError  # noqa: E402
from memmachine_server.common.language_model.litellm_language_model import (  # noqa: E402
    LiteLLMLanguageModel,
    LiteLLMLanguageModelParams,
    _is_retryable_litellm_error,
)


def _make_chat_completion(content: str = "hello", **extra: Any) -> openai_chat.ChatCompletion:
    payload: dict[str, Any] = {
        "id": "resp-1",
        "object": "chat.completion",
        "created": 1,
        "model": "anthropic/claude-sonnet-4-6",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 7, "completion_tokens": 3, "total_tokens": 10},
    }
    payload.update(extra)
    return openai_chat.ChatCompletion.model_validate(payload)


@pytest.fixture
def model_params() -> LiteLLMLanguageModelParams:
    return LiteLLMLanguageModelParams(
        model="anthropic/claude-sonnet-4-6",
        max_retry_interval_seconds=1,
    )


def test_init_does_not_require_openai_client(model_params: LiteLLMLanguageModelParams) -> None:
    """LiteLLMLanguageModel skips the parent's AsyncOpenAI requirement."""
    lm = LiteLLMLanguageModel(model_params)
    assert lm._model == "anthropic/claude-sonnet-4-6"
    assert lm._client is None
    assert lm._litellm_drop_params is True


@pytest.mark.asyncio
async def test_request_chat_completion_dispatches_to_litellm(
    model_params: LiteLLMLanguageModelParams,
) -> None:
    lm = LiteLLMLanguageModel(model_params)
    fake_response = _make_chat_completion("pong")

    with patch(
        "litellm.acompletion",
        new_callable=AsyncMock,
        return_value=fake_response,
    ) as fake_acomp:
        result = await lm._request_chat_completion(
            args={"model": "anthropic/claude-sonnet-4-6", "messages": [{"role": "user", "content": "x"}]},
            max_attempts=1,
            generate_response_call_uuid="uuid",
        )

    assert result is fake_response
    fake_acomp.assert_awaited_once()
    assert fake_acomp.await_args is not None
    kwargs = fake_acomp.await_args.kwargs
    assert kwargs["model"] == "anthropic/claude-sonnet-4-6"
    assert kwargs["drop_params"] is True


@pytest.mark.asyncio
async def test_request_chat_completion_forwards_api_base_and_key() -> None:
    params = LiteLLMLanguageModelParams(
        model="openai/gpt-4o",
        api_key="sk-test",
        api_base="http://localhost:4000",
        api_version="2024-02-01",
        max_retry_interval_seconds=1,
    )
    lm = LiteLLMLanguageModel(params)
    fake_response = _make_chat_completion("ok")

    with patch(
        "litellm.acompletion",
        new_callable=AsyncMock,
        return_value=fake_response,
    ) as fake_acomp:
        await lm._request_chat_completion(
            args={"model": "openai/gpt-4o", "messages": []},
            max_attempts=1,
            generate_response_call_uuid="uuid",
        )

    assert fake_acomp.await_args is not None
    kwargs = fake_acomp.await_args.kwargs
    assert kwargs["api_key"] == "sk-test"
    assert kwargs["api_base"] == "http://localhost:4000"
    assert kwargs["api_version"] == "2024-02-01"


@pytest.mark.asyncio
async def test_request_chat_completion_does_not_overwrite_explicit_kwargs() -> None:
    params = LiteLLMLanguageModelParams(
        model="openai/gpt-4o",
        api_key="from-shim",
        max_retry_interval_seconds=1,
    )
    lm = LiteLLMLanguageModel(params)
    fake_response = _make_chat_completion()

    with patch(
        "litellm.acompletion",
        new_callable=AsyncMock,
        return_value=fake_response,
    ) as fake_acomp:
        await lm._request_chat_completion(
            args={
                "model": "openai/gpt-4o",
                "messages": [],
                "api_key": "from-caller",
            },
            max_attempts=1,
            generate_response_call_uuid="uuid",
        )

    assert fake_acomp.await_args is not None
    assert fake_acomp.await_args.kwargs["api_key"] == "from-caller"


@pytest.mark.asyncio
async def test_request_chat_completion_extra_kwargs_forwarded() -> None:
    params = LiteLLMLanguageModelParams(
        model="openai/gpt-4o",
        extra_kwargs={"metadata": {"tag": "memmachine"}},
        max_retry_interval_seconds=1,
    )
    lm = LiteLLMLanguageModel(params)
    fake_response = _make_chat_completion()

    with patch(
        "litellm.acompletion",
        new_callable=AsyncMock,
        return_value=fake_response,
    ) as fake_acomp:
        await lm._request_chat_completion(
            args={"model": "openai/gpt-4o", "messages": []},
            max_attempts=1,
            generate_response_call_uuid="uuid",
        )

    assert fake_acomp.await_args is not None
    assert fake_acomp.await_args.kwargs["metadata"] == {"tag": "memmachine"}


@pytest.mark.asyncio
async def test_request_chat_completion_retries_on_retryable_error(
    model_params: LiteLLMLanguageModelParams,
) -> None:
    class TransientRateLimitError(Exception):
        pass

    TransientRateLimitError.__name__ = "RateLimitError"
    fake_response = _make_chat_completion()
    side_effects = [TransientRateLimitError("slow down"), fake_response]

    lm = LiteLLMLanguageModel(model_params)
    with patch(
        "litellm.acompletion",
        new_callable=AsyncMock,
        side_effect=side_effects,
    ) as fake_acomp:
        result = await lm._request_chat_completion(
            args={"model": "x", "messages": []},
            max_attempts=2,
            generate_response_call_uuid="uuid",
        )

    assert result is fake_response
    assert fake_acomp.await_count == 2


@pytest.mark.asyncio
async def test_request_chat_completion_raises_external_service_error_after_max_attempts(
    model_params: LiteLLMLanguageModelParams,
) -> None:
    class TransientRateLimitError(Exception):
        pass

    TransientRateLimitError.__name__ = "RateLimitError"

    lm = LiteLLMLanguageModel(model_params)
    with (
        patch(
            "litellm.acompletion",
            new_callable=AsyncMock,
            side_effect=TransientRateLimitError("nope"),
        ),
        pytest.raises(ExternalServiceAPIError),
    ):
        await lm._request_chat_completion(
            args={"model": "x", "messages": []},
            max_attempts=2,
            generate_response_call_uuid="uuid",
        )


@pytest.mark.asyncio
async def test_request_chat_completion_non_retryable_error_raises_immediately(
    model_params: LiteLLMLanguageModelParams,
) -> None:
    class HardRequestError(Exception):
        pass

    HardRequestError.__name__ = "BadRequestError"

    lm = LiteLLMLanguageModel(model_params)
    with (
        patch(
            "litellm.acompletion",
            new_callable=AsyncMock,
            side_effect=HardRequestError("bad input"),
        ) as fake_acomp,
        pytest.raises(ExternalServiceAPIError),
    ):
        await lm._request_chat_completion(
            args={"model": "x", "messages": []},
            max_attempts=3,
            generate_response_call_uuid="uuid",
        )
    assert fake_acomp.await_count == 1


def test_is_retryable_litellm_error_recognizes_known_classes() -> None:
    cases = [
        ("RateLimitError", True),
        ("APITimeoutError", True),
        ("APIConnectionError", True),
        ("InternalServerError", True),
        ("ServiceUnavailableError", True),
        ("BadRequestError", False),
        ("AuthenticationError", False),
        ("ValueError", False),
    ]
    for name, expected in cases:
        cls = type(name, (Exception,), {})
        assert _is_retryable_litellm_error(cls()) is expected, name


@pytest.mark.asyncio
async def test_generate_response_uses_parent_parsing(
    model_params: LiteLLMLanguageModelParams,
) -> None:
    """End-to-end: generate_response inherits parent parsing of OpenAI-shape
    response; only the request fn was swapped."""
    lm = LiteLLMLanguageModel(model_params)
    fake_response = _make_chat_completion("hello world")

    with patch(
        "litellm.acompletion",
        new_callable=AsyncMock,
        return_value=fake_response,
    ):
        text, tool_calls = await lm.generate_response(
            system_prompt="be helpful",
            user_prompt="say hi",
        )

    assert text == "hello world"
    assert tool_calls == []
