"""OpenAI-completions API based language model implementation."""

import asyncio
import logging
from typing import Any, TypeVar, cast
from uuid import uuid4

import json_repair
import openai
from openai.types.chat import (
    ChatCompletion,
    ChatCompletionMessageFunctionToolCall,
)
from pydantic import BaseModel, Field, InstanceOf, TypeAdapter

from memmachine_server.common.data_types import ExternalServiceAPIError
from memmachine_server.common.metrics_factory import MetricsFactory, OperationTracker

from .language_model import LanguageModel

T = TypeVar("T")

logger = logging.getLogger(__name__)


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

            try:
                response = await self._client.with_options(
                    max_retries=max_attempts,
                ).chat.completions.parse(
                    model=self._model,
                    messages=input_prompts,
                    response_format=output_format,
                )
            except openai.OpenAIError as e:
                error_message = (
                    f"[call uuid: {generate_response_call_uuid}] "
                    "Giving up generating response "
                    f"due to non-retryable {type(e).__name__}"
                )
                logger.exception(error_message)
                raise ExternalServiceAPIError(error_message) from e

            self._collect_usage_metrics(response)

            return TypeAdapter(output_format).validate_python(
                response.choices[0].message.parsed
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

    async def _generate_response(  # noqa: C901
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

            sleep_seconds = 1
            for attempt in range(1, max_attempts + 1):
                try:
                    args: dict = {
                        "model": self._model,
                        "messages": input_prompts,
                    }
                    if tools:
                        args["tools"] = tools
                        args["tool_choice"] = (
                            tool_choice if tool_choice is not None else "auto"
                        )
                    response = await self._client.chat.completions.create(**args)
                    break
                except (
                    openai.RateLimitError,
                    openai.APITimeoutError,
                    openai.APIConnectionError,
                ) as e:
                    # Exception may be retried.
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

            self._collect_usage_metrics(response)

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

            return (
                response.choices[0].message.content or "",
                function_calls_arguments,
                response.usage.prompt_tokens if response.usage else 0,
                response.usage.completion_tokens if response.usage else 0,
            )

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
