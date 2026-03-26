"""OpenAI-completions API based language model implementation."""

import asyncio
import logging
from collections.abc import AsyncIterator
from typing import Any, TypeVar, cast
from uuid import uuid4

import json_repair
import openai
from openai.lib._parsing import parse_chat_completion, type_to_response_format_param
from openai.types.chat import (
    ChatCompletion,
    ChatCompletionChunk,
    ChatCompletionMessageFunctionToolCall,
)
from pydantic import BaseModel, Field, InstanceOf, TypeAdapter

from memmachine_server.common.data_types import ExternalServiceAPIError
from memmachine_server.common.metrics_factory import MetricsFactory, OperationTracker

from .language_model import LanguageModel

T = TypeVar("T")

logger = logging.getLogger(__name__)


class _StreamedToolCallState(BaseModel):
    id: str | None = None
    type: str | None = None
    function_name: str = ""
    function_arguments: str = ""


class OpenAIChatCompletionsLanguageModelParams(BaseModel):
    """
    Parameters for OpenAIChatCompletionsLanguageModel.

    Attributes:
        client (openai.AsyncOpenAI):
            AsyncOpenAI client to use for making API calls.
        model (str):
            Name of the OpenAI model to use
            (e.g. 'gpt-5-nano').
        max_retry_interval_seconds (int):
            Maximal retry interval in seconds when retrying API calls
            (default: 120).
        metrics_factory (MetricsFactory | None):
            An instance of MetricsFactory
            for collecting usage metrics
            (default: None).
    """

    client: InstanceOf[openai.AsyncOpenAI] = Field(
        ...,
        description="AsyncOpenAI client to use for making API calls",
    )
    model: str = Field(
        ...,
        description="Name of the OpenAI model to use (e.g. 'gpt-5-nano')",
    )
    max_retry_interval_seconds: int = Field(
        120,
        description="Maximal retry interval in seconds when retrying API calls",
        gt=0,
    )
    metrics_factory: InstanceOf[MetricsFactory] | None = Field(
        None,
        description="An instance of MetricsFactory for collecting usage metrics",
    )


class OpenAIChatCompletionsLanguageModel(LanguageModel):
    """Language model that uses OpenAI's chat completions API."""

    def __init__(self, params: OpenAIChatCompletionsLanguageModelParams) -> None:
        """
        Initialize the chat completions language model.

        Args:
            params (OpenAIChatCompletionsLanguageModelParams):
                Parameters for the OpenAIChatCompletionsLanguageModel.

        """
        super().__init__()

        self._client = params.client

        self._model = params.model

        self._max_retry_interval_seconds = params.max_retry_interval_seconds

        metrics_factory = params.metrics_factory

        self._tracker = OperationTracker(
            metrics_factory, prefix="language_model_openai_chat_completions"
        )

        self._should_collect_metrics = False
        if metrics_factory is not None:
            self._should_collect_metrics = True

            self._input_tokens_usage_counter = metrics_factory.get_counter(
                "language_model_openai_chat_completions_usage_input_tokens",
                "Number of input tokens used for OpenAI language model",
            )
            self._output_tokens_usage_counter = metrics_factory.get_counter(
                "language_model_openai_chat_completions_usage_output_tokens",
                "Number of output tokens used for OpenAI language model",
            )
            self._total_tokens_usage_counter = metrics_factory.get_counter(
                "language_model_openai_chat_completions_usage_total_tokens",
                "Number of tokens used for OpenAI language model",
            )

    async def generate_parsed_response(
        self,
        output_format: type[T],
        system_prompt: str | None = None,
        user_prompt: str | None = None,
        max_attempts: int = 1,
    ) -> T | None:
        """Generate a structured response parsed into the given model."""
        async with self._tracker("generate_parsed_response"):
            if max_attempts <= 0:
                raise ValueError("max_attempts must be a positive integer")

            input_prompts = cast(
                Any,
                [
                    {"role": "system", "content": system_prompt or ""},
                    {"role": "user", "content": user_prompt or ""},
                ],
            )

            generate_response_call_uuid = uuid4()

            response = await self._request_chat_completion(
                args={
                    "model": self._model,
                    "messages": input_prompts,
                    "response_format": type_to_response_format_param(output_format),
                },
                max_attempts=max_attempts,
                generate_response_call_uuid=generate_response_call_uuid,
            )

            return await self._normalize_parsed_chat_completion_response(
                response=response,
                output_format=output_format,
            )

    async def generate_response(
        self,
        system_prompt: str | None = None,
        user_prompt: str | None = None,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, str] | None = None,
        max_attempts: int = 1,
    ) -> tuple[str, Any]:
        output, function_calls_arguments, _, _ = await self._generate_response(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            tools=tools,
            tool_choice=tool_choice,
            max_attempts=max_attempts,
        )
        return output, function_calls_arguments

    async def generate_response_with_token_usage(
        self,
        system_prompt: str | None = None,
        user_prompt: str | None = None,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, str] | None = None,
        max_attempts: int = 1,
    ) -> tuple[str, Any, int, int]:
        return await self._generate_response(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            tools=tools,
            tool_choice=tool_choice,
            max_attempts=max_attempts,
        )

    async def _generate_response(
        self,
        system_prompt: str | None = None,
        user_prompt: str | None = None,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, str] | None = None,
        max_attempts: int = 1,
    ) -> tuple[str, list[dict[str, Any]], int, int]:
        """Generate a chat completion response (and optional tool call)."""
        async with self._tracker("generate_response"):
            if max_attempts <= 0:
                raise ValueError("max_attempts must be a positive integer")

            input_prompts = cast(
                Any,
                [
                    {"role": "system", "content": system_prompt or ""},
                    {"role": "user", "content": user_prompt or ""},
                ],
            )
            generate_response_call_uuid = uuid4()

            args: dict[str, Any] = {
                "model": self._model,
                "messages": input_prompts,
            }
            if tools:
                args["tools"] = tools
                args["tool_choice"] = tool_choice if tool_choice is not None else "auto"

            response = await self._request_chat_completion(
                args=args,
                max_attempts=max_attempts,
                generate_response_call_uuid=generate_response_call_uuid,
            )

            return await self._normalize_chat_completion_response(response)

    async def _request_chat_completion(
        self,
        args: dict[str, Any],
        max_attempts: int,
        generate_response_call_uuid: object,
    ) -> ChatCompletion | AsyncIterator[object] | object:
        sleep_seconds = 1
        for attempt in range(1, max_attempts + 1):
            try:
                return await self._client.chat.completions.create(**args)
            except (
                openai.RateLimitError,
                openai.APITimeoutError,
                openai.APIConnectionError,
            ) as e:
                if attempt >= max_attempts:
                    error_message = (
                        f"[call uuid: {generate_response_call_uuid}] "
                        "Giving up generating response "
                        f"after failed attempt {attempt} "
                        f"due to retryable {type(e).__name__}: "
                        f"max attempts {max_attempts} reached"
                    )
                    logger.exception(error_message)
                    raise ExternalServiceAPIError(error_message) from e

                logger.info(
                    "[call uuid: %s] "
                    "Retrying generating response in %d seconds "
                    "after failed attempt %d due to retryable %s...",
                    generate_response_call_uuid,
                    sleep_seconds,
                    attempt,
                    type(e).__name__,
                )
                await asyncio.sleep(sleep_seconds)
                sleep_seconds *= 2
                sleep_seconds = min(sleep_seconds, self._max_retry_interval_seconds)
                continue
            except openai.OpenAIError as e:
                error_message = (
                    f"[call uuid: {generate_response_call_uuid}] "
                    "Giving up generating response "
                    f"after failed attempt {attempt} "
                    f"due to non-retryable {type(e).__name__}"
                )
                logger.exception(error_message)
                raise ExternalServiceAPIError(error_message) from e

        raise RuntimeError("unreachable")

    def _collect_usage_metrics(self, response: ChatCompletion) -> None:
        if not self._should_collect_metrics:
            return

        if response.usage is None:
            logger.debug("No usage information found in response")
            return

        try:
            self._input_tokens_usage_counter.increment(
                value=response.usage.prompt_tokens,
            )
            self._output_tokens_usage_counter.increment(
                value=response.usage.completion_tokens,
            )
            self._total_tokens_usage_counter.increment(
                value=response.usage.total_tokens,
            )
        except Exception:
            logger.exception("Failed to collect usage metrics")

    async def _normalize_chat_completion_response(
        self,
        response: ChatCompletion | AsyncIterator[object] | object,
    ) -> tuple[str, list[dict[str, Any]], int, int]:
        if not self._is_streaming_response(response):
            response = cast(ChatCompletion, response)
            self._collect_usage_metrics(response)
            function_calls_arguments = self._extract_tool_calls_from_completion(
                response
            )
            return (
                response.choices[0].message.content or "",
                function_calls_arguments,
                response.usage.prompt_tokens if response.usage else 0,
                response.usage.completion_tokens if response.usage else 0,
            )

        return await self._collect_streaming_response(cast(Any, response))

    async def _normalize_parsed_chat_completion_response(
        self,
        response: ChatCompletion | AsyncIterator[object] | object,
        output_format: type[T],
    ) -> T:
        if not self._is_streaming_response(response):
            response = cast(ChatCompletion, response)
            self._collect_usage_metrics(response)
            parsed_response = parse_chat_completion(
                response_format=output_format,
                input_tools=[],
                chat_completion=response,
            )
            return TypeAdapter(output_format).validate_python(
                parsed_response.choices[0].message.parsed
            )

        content, _, _, _ = await self._collect_streaming_response(cast(Any, response))
        return TypeAdapter(output_format).validate_python(json_repair.loads(content))

    def _extract_tool_calls_from_completion(
        self,
        response: ChatCompletion,
    ) -> list[dict[str, Any]]:
        function_calls_arguments = []
        try:
            if response.choices[0].message.tool_calls:
                for tool_call in response.choices[0].message.tool_calls:
                    if isinstance(
                        tool_call,
                        ChatCompletionMessageFunctionToolCall,
                    ):
                        function_calls_arguments.append(
                            {
                                "call_id": tool_call.id,
                                "function": {
                                    "name": tool_call.function.name,
                                    "arguments": json_repair.loads(
                                        tool_call.function.arguments,
                                    ),
                                },
                            },
                        )
                    else:
                        logger.info(
                            "Unsupported tool call type: %s",
                            type(tool_call).__name__,
                        )
        except (TypeError, ValueError) as e:
            raise ValueError(
                "Failed to repair or parse JSON from function call arguments"
            ) from e

        return function_calls_arguments

    async def _collect_streaming_response(
        self,
        response_stream: AsyncIterator[object],
    ) -> tuple[str, list[dict[str, Any]], int, int]:
        output_chunks: list[str] = []
        streamed_tool_calls: dict[int, _StreamedToolCallState] = {}
        prompt_tokens = 0
        completion_tokens = 0
        total_tokens = 0

        async for chunk in response_stream:
            if not self._is_supported_stream_chunk(chunk):
                continue

            chat_chunk = cast(ChatCompletionChunk, chunk)
            prompt_tokens, completion_tokens, total_tokens = (
                self._update_stream_usage_counters(
                    chat_chunk,
                    prompt_tokens,
                    completion_tokens,
                    total_tokens,
                )
            )
            self._append_stream_chunk_data(
                chat_chunk,
                output_chunks,
                streamed_tool_calls,
            )

        self._collect_stream_usage_metrics(
            prompt_tokens,
            completion_tokens,
            total_tokens,
        )

        function_calls_arguments = self._build_streamed_tool_calls(streamed_tool_calls)

        return (
            "".join(output_chunks),
            function_calls_arguments,
            prompt_tokens,
            completion_tokens,
        )

    def _build_streamed_tool_calls(
        self,
        streamed_tool_calls: dict[int, _StreamedToolCallState],
    ) -> list[dict[str, Any]]:
        function_calls_arguments = []

        try:
            for index in sorted(streamed_tool_calls):
                tool_call = streamed_tool_calls[index]
                if tool_call.type != "function":
                    logger.info(
                        "Unsupported streamed tool call type: %s",
                        tool_call.type,
                    )
                    continue
                function_calls_arguments.append(
                    {
                        "call_id": tool_call.id,
                        "function": {
                            "name": tool_call.function_name,
                            "arguments": json_repair.loads(
                                tool_call.function_arguments,
                            ),
                        },
                    },
                )
        except (TypeError, ValueError) as e:
            raise ValueError(
                "Failed to repair or parse JSON from function call arguments"
            ) from e

        return function_calls_arguments

    def _is_reasoning_chunk(self, chunk: object) -> bool:
        chunk_type = getattr(chunk, "type", None)
        if isinstance(chunk_type, str) and "reason" in chunk_type.lower():
            logger.debug("Discarding streamed reasoning item of type %s", chunk_type)
            return True

        if isinstance(chunk, ChatCompletionChunk) and chunk.choices:
            delta = chunk.choices[0].delta
            if hasattr(delta, "reasoning_content"):
                logger.debug(
                    "Discarding streamed reasoning chunk with reasoning content"
                )
                return True

        return False

    def _is_streaming_response(self, response: object) -> bool:
        if isinstance(response, ChatCompletion) or hasattr(response, "choices"):
            return False
        if type(response).__module__.startswith("unittest.mock"):
            return False
        return isinstance(response, openai.AsyncStream) or hasattr(
            response, "__anext__"
        )

    def _is_supported_stream_chunk(self, chunk: object) -> bool:
        if self._is_reasoning_chunk(chunk):
            return False

        if isinstance(chunk, ChatCompletionChunk):
            return True

        logger.info(
            "Discarding unsupported streamed chat completion item: %s",
            type(chunk).__name__,
        )
        return False

    def _update_stream_usage_counters(
        self,
        chunk: ChatCompletionChunk,
        prompt_tokens: int,
        completion_tokens: int,
        total_tokens: int,
    ) -> tuple[int, int, int]:
        if chunk.usage is None:
            return prompt_tokens, completion_tokens, total_tokens
        return (
            chunk.usage.prompt_tokens + prompt_tokens,
            chunk.usage.completion_tokens + completion_tokens,
            chunk.usage.total_tokens + total_tokens,
        )

    def _append_stream_chunk_data(
        self,
        chunk: ChatCompletionChunk,
        output_chunks: list[str],
        streamed_tool_calls: dict[int, _StreamedToolCallState],
    ) -> None:
        for choice in chunk.choices:
            delta = choice.delta

            if delta.content:
                output_chunks.append(delta.content)

            if not delta.tool_calls:
                continue

            for tool_call in delta.tool_calls:
                state = streamed_tool_calls.setdefault(
                    tool_call.index,
                    _StreamedToolCallState(),
                )
                if tool_call.id:
                    state.id = tool_call.id
                if tool_call.type:
                    state.type = tool_call.type
                if tool_call.function is not None:
                    if tool_call.function.name:
                        state.function_name += tool_call.function.name
                    if tool_call.function.arguments:
                        state.function_arguments += tool_call.function.arguments

    def _collect_stream_usage_metrics(
        self,
        prompt_tokens: int,
        completion_tokens: int,
        total_tokens: int,
    ) -> None:
        if not self._should_collect_metrics or total_tokens <= 0:
            return

        try:
            self._input_tokens_usage_counter.increment(value=prompt_tokens)
            self._output_tokens_usage_counter.increment(value=completion_tokens)
            self._total_tokens_usage_counter.increment(value=total_tokens)
        except Exception:
            logger.exception("Failed to collect usage metrics")
