"""
Unit tests for OpenAIChatCompletionsLanguageModel.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import openai
import pytest
from pydantic import ValidationError

from memmachine.common.data_types import ExternalServiceAPIError
from memmachine.common.language_model.openai_chat_completions_language_model import (
    OpenAIChatCompletionsLanguageModel,
    OpenAIChatCompletionsLanguageModelParams,
)
from memmachine.common.metrics_factory.metrics_factory import MetricsFactory


@pytest.fixture
def mock_metrics_factory():
    """Fixture for a mocked MetricsFactory."""

    class MockMetricsFactory(MetricsFactory):
        def __init__(self):
            self.counters = MagicMock()
            self.gauge = MagicMock()
            self.histogram = MagicMock()
            self.summaries = MagicMock()

        def get_counter(self, name, description, label_names=...):
            return self.counters

        def get_summary(self, name, description, label_names=...):
            return self.summaries

        def get_gauge(self, name, description, label_names=...):
            return self.gauge

        def get_histogram(self, name, description, label_names=...):
            return self.histogram

        def reset(self):
            return MagicMock()

    factory = MockMetricsFactory()
    return factory


@pytest.fixture
def mock_tool_call_impl():
    """Fixture for a mocked tool call."""

    class MockToolCall(
        MagicMock, openai.types.chat.ChatCompletionMessageFunctionToolCall
    ):
        pass

    return MockToolCall()


@pytest.fixture
def mock_async_openai():
    """Fixture for a mocked AsyncOpenAI client."""
    with patch("openai.AsyncOpenAI", spec=openai.AsyncOpenAI) as mock_async_openai:
        mock_client = mock_async_openai.return_value
        mock_client.chat.completions.create = AsyncMock()
        yield mock_async_openai


@pytest.fixture
def minimal_config(mock_async_openai):
    """Fixture for a minimal valid configuration."""
    return OpenAIChatCompletionsLanguageModelParams(
        client=openai.AsyncOpenAI(
            api_key="test_api_key",
        ),
        model="test-model",
    )


@pytest.fixture
def full_config(mock_async_openai, mock_metrics_factory):
    """Fixture for a full valid configuration with metrics."""
    return OpenAIChatCompletionsLanguageModelParams(
        client=openai.AsyncOpenAI(
            api_key="test_api_key",
            base_url="http://localhost:8080",
        ),
        model="test-model",
        max_retry_interval_seconds=60,
        metrics_factory=mock_metrics_factory,
        user_metrics_labels={"user": "test-user"},
    )


@pytest.fixture
def max_retry_interval_seconds_config(mock_async_openai):
    """Fixture for a valid configuration with small max_retry_interval_seconds."""
    return OpenAIChatCompletionsLanguageModelParams(
        client=openai.AsyncOpenAI(
            api_key="test_api_key",
        ),
        model="test-model",
        max_retry_interval_seconds=4,
    )


def test_init_success(mock_async_openai, minimal_config):
    """Test successful initialization."""
    OpenAIChatCompletionsLanguageModel(minimal_config)


def test_init_with_full_config(mock_async_openai, full_config):
    """Test successful initialization with all optional parameters."""
    _ = OpenAIChatCompletionsLanguageModel(full_config)
    mock_async_openai.assert_called_once_with(
        api_key="test_api_key", base_url="http://localhost:8080"
    )


def test_init_missing_client():
    """Test initialization fails if client is missing."""
    with pytest.raises(ValidationError):
        OpenAIChatCompletionsLanguageModel(
            OpenAIChatCompletionsLanguageModelParams(
                model="test-model",
            )
        )


def test_init_missing_model():
    """Test initialization fails if model is missing."""
    with pytest.raises(ValidationError):
        OpenAIChatCompletionsLanguageModel(
            OpenAIChatCompletionsLanguageModelParams(
                client=openai.AsyncOpenAI(
                    api_key="test_api_key",
                ),
            )
        )


def test_init_invalid_max_retry_interval_seconds_type(minimal_config):
    """Test initialization fails with non-integer max_retry_interval_seconds."""
    with pytest.raises(ValidationError):
        OpenAIChatCompletionsLanguageModelParams(
            **(
                minimal_config.model_dump()
                | {"max_retry_interval_seconds": "not-an-int"}
            ),
        )


def test_init_invalid_max_retry_interval_seconds_value(minimal_config):
    """Test initialization fails with non-positive max_retry_interval_seconds."""
    with pytest.raises(ValidationError):
        OpenAIChatCompletionsLanguageModelParams(
            **(minimal_config.model_dump() | {"max_retry_interval_seconds": 0}),
        )


def test_init_invalid_metrics_factory_type(minimal_config):
    """Test initialization fails with invalid metrics_factory type."""
    with pytest.raises(ValidationError):
        OpenAIChatCompletionsLanguageModelParams(
            **(minimal_config.model_dump() | {"metrics_factory": "not-a-factory"}),
        )


def test_init_invalid_user_metrics_labels_type(minimal_config, mock_metrics_factory):
    """Test initialization fails with invalid user_metrics_labels type."""
    with pytest.raises(ValidationError):
        OpenAIChatCompletionsLanguageModelParams(
            **(
                minimal_config.model_dump()
                | {
                    "metrics_factory": mock_metrics_factory,
                    "user_metrics_labels": "not-a-dict",
                }
            ),
        )


@pytest.mark.asyncio
async def test_generate_response_invalid_max_attempts(minimal_config):
    """Test generate_response fails with non-positive max_attempts."""
    lm = OpenAIChatCompletionsLanguageModel(minimal_config)
    with pytest.raises(ValueError, match="max_attempts must be a positive integer"):
        await lm.generate_response(max_attempts=0)


@pytest.mark.asyncio
async def test_generate_response_success(mock_async_openai, minimal_config):
    """Test a successful call to generate_response."""
    mock_response = MagicMock()
    mock_response.choices[0].message.content = "Hello, world!"
    mock_response.choices[0].message.tool_calls = None
    mock_response.usage = None

    mock_client = mock_async_openai.return_value
    mock_client.chat.completions.create.return_value = mock_response

    lm = OpenAIChatCompletionsLanguageModel(minimal_config)
    content, tool_calls = await lm.generate_response(
        system_prompt="System prompt", user_prompt="User prompt"
    )

    assert content == "Hello, world!"
    assert tool_calls == []
    mock_client.chat.completions.create.assert_awaited_once()
    call_args = mock_client.chat.completions.create.call_args
    assert call_args.kwargs["model"] == "test-model"
    assert call_args.kwargs["messages"] == [
        {"role": "system", "content": "System prompt"},
        {"role": "user", "content": "User prompt"},
    ]


@pytest.mark.asyncio
async def test_generate_response_with_tool_calls(
    mock_async_openai, minimal_config, mock_tool_call_impl
):
    """Test a successful call that returns tool calls."""
    mock_tool_call = mock_tool_call_impl()
    mock_tool_call.id = "call_123"
    mock_tool_call.function.name = "get_weather"
    mock_tool_call.function.arguments = '{"location": "Boston"}'

    invalid_mock_tool_call = MagicMock()
    invalid_mock_tool_call.id = "invalid_call_123"
    invalid_mock_tool_call.function.name = "get_weather"
    invalid_mock_tool_call.function.arguments = '{"location": "Boston"}'

    mock_response = MagicMock()
    mock_response.choices[0].message.content = None
    mock_response.choices[0].message.tool_calls = [
        mock_tool_call,
        invalid_mock_tool_call,
    ]
    mock_response.usage = None

    mock_client = mock_async_openai.return_value
    mock_client.chat.completions.create.return_value = mock_response

    lm = OpenAIChatCompletionsLanguageModel(minimal_config)
    content, tool_calls = await lm.generate_response()

    assert content == ""
    assert tool_calls == [
        {
            "call_id": "call_123",
            "function": {
                "name": "get_weather",
                "arguments": {"location": "Boston"},
            },
        }
    ]


@pytest.mark.asyncio
async def test_generate_response_tool_call_json_error(
    mock_async_openai, minimal_config, mock_tool_call_impl
):
    """Test handling of invalid JSON in tool call arguments."""
    mock_tool_call = mock_tool_call_impl()
    mock_tool_call.id = "call_123"
    mock_tool_call.function.name = "get_weather"
    mock_tool_call.function.arguments = '{"location": "Boston",}'

    mock_response = MagicMock()
    mock_response.choices[0].message.tool_calls = [mock_tool_call]
    mock_response.usage = None

    mock_client = mock_async_openai.return_value
    mock_client.chat.completions.create.return_value = mock_response

    lm = OpenAIChatCompletionsLanguageModel(minimal_config)
    with pytest.raises(ValueError, match="JSON decode error"):
        await lm.generate_response()


@pytest.mark.asyncio
@patch("asyncio.sleep", new_callable=AsyncMock)
async def test_generate_response_retry_on_rate_limit(
    mock_sleep, mock_async_openai, minimal_config
):
    """Test retry logic on RateLimitError."""
    mock_response = MagicMock()
    mock_response.choices[0].message.content = "Success after retry"
    mock_response.choices[0].message.tool_calls = None
    mock_response.usage = None

    mock_client = mock_async_openai.return_value
    mock_client.chat.completions.create.side_effect = [
        openai.RateLimitError("rate limited", response=MagicMock(), body=None),
        mock_response,
    ]

    lm = OpenAIChatCompletionsLanguageModel(minimal_config)
    content, _ = await lm.generate_response(max_attempts=2)

    assert content == "Success after retry"
    assert mock_client.chat.completions.create.call_count == 2
    mock_sleep.assert_awaited_once_with(1)


@pytest.mark.asyncio
@patch("asyncio.sleep", new_callable=AsyncMock)
async def test_generate_response_retry_on_rate_limit_with_max_retry_interval_seconds(
    mock_sleep, mock_async_openai, max_retry_interval_seconds_config
):
    """Test retry logic on RateLimitError."""
    mock_response = MagicMock()
    mock_response.choices[0].message.content = "Success after retry"
    mock_response.choices[0].message.tool_calls = None
    mock_response.usage = None

    mock_client = mock_async_openai.return_value
    mock_client.chat.completions.create.side_effect = openai.RateLimitError(
        "rate limited", response=MagicMock(), body=None
    )

    lm = OpenAIChatCompletionsLanguageModel(max_retry_interval_seconds_config)
    with pytest.raises(ExternalServiceAPIError):
        await lm.generate_response(max_attempts=6)

    assert mock_client.chat.completions.create.call_count == 6
    mock_sleep.assert_has_awaits(
        [
            ((1,),),
            ((2,),),
            ((4,),),
            ((4,),),
            ((4,),),
        ]
    )


@pytest.mark.asyncio
@patch("asyncio.sleep", new_callable=AsyncMock)
async def test_generate_response_fail_after_max_retries(
    mock_sleep, mock_async_openai, minimal_config
):
    """Test that an IOError is raised after max_attempts are exhausted."""
    mock_client = mock_async_openai.return_value
    mock_client.chat.completions.create.side_effect = openai.APITimeoutError(None)

    lm = OpenAIChatCompletionsLanguageModel(minimal_config)
    with pytest.raises(ExternalServiceAPIError, match=r"max attempts"):
        await lm.generate_response(max_attempts=3)

    assert mock_client.chat.completions.create.call_count == 3
    assert mock_sleep.call_count == 2
    mock_sleep.assert_any_await(1)
    mock_sleep.assert_any_await(2)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "exception",
    [
        (openai.AuthenticationError("auth error", response=MagicMock(), body=None),),
        (openai.BadRequestError("bad request", response=MagicMock(), body=None),),
        (openai.APIError("api error", request=MagicMock(), body=None),),
        (openai.OpenAIError("generic error")),
    ],
)
async def test_generate_response_runtime_exception_mapping(
    mock_async_openai, minimal_config, exception
):
    """Test that OpenAI exceptions are correctly mapped to generic
    exceptions.
    """
    mock_client = mock_async_openai.return_value
    mock_client.chat.completions.create.side_effect = exception

    lm = OpenAIChatCompletionsLanguageModel(minimal_config)
    with pytest.raises(ExternalServiceAPIError):
        await lm.generate_response()


@pytest.mark.asyncio
async def test_metrics_collection(mock_async_openai, full_config):
    """Test that metrics are collected on a successful call."""
    mock_usage = MagicMock()
    mock_usage.prompt_tokens = 100
    mock_usage.completion_tokens = 50
    mock_usage.total_tokens = 150

    mock_response = MagicMock()
    mock_response.choices[0].message.content = "Metrics test"
    mock_response.choices[0].message.tool_calls = None
    mock_response.usage = mock_usage

    mock_client = mock_async_openai.return_value
    mock_client.chat.completions.create.return_value = mock_response

    lm = OpenAIChatCompletionsLanguageModel(full_config)
    await lm.generate_response()

    metrics_factory = full_config.metrics_factory
    labels = full_config.user_metrics_labels

    input_counter = metrics_factory.get_counter("test", "test")
    output_counter = metrics_factory.get_counter("test", "test")
    total_counter = metrics_factory.get_counter("test", "test")
    latency_summary = metrics_factory.get_summary("test", "test")

    # Note: Since get_counter returns the same mock, we check the calls on that mock.
    input_counter.increment.assert_any_call(value=100, labels=labels)
    output_counter.increment.assert_any_call(value=50, labels=labels)
    total_counter.increment.assert_any_call(value=150, labels=labels)

    latency_summary.observe.assert_called_once()
    observed_latency = latency_summary.observe.call_args.kwargs["value"]
    assert observed_latency > 0
    assert latency_summary.observe.call_args.kwargs["labels"] == labels
