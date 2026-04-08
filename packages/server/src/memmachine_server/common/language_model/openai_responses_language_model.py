"""OpenAI-based language model implementation."""

import asyncio
import logging
from typing import Any, TypeVar, cast
from uuid import uuid4

import json_repair
import openai
from openai.types.responses import (
    Response,
    ResponseFunctionToolCall,
    ResponseInputParam,
    ToolParam,
)
from pydantic import BaseModel, Field, InstanceOf

from memmachine_server.common.data_types import ExternalServiceAPIError
from memmachine_server.common.metrics_factory import MetricsFactory, OperationTracker

from .language_model import LanguageModel

T = TypeVar("T")

logger = logging.getLogger(__name__)


class OpenAIResponsesLanguageModelParams(BaseModel):
    """
    Parameters for OpenAIResponsesLanguageModel.

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
        reasoning_effort (str | None):
            Reasoning effort level for supported models
            (e.g. "minimal", "low", "medium", "high", "none").
            If None, the API default is used.

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
    reasoning_effort: str | None = Field(
        None,
        description=(
            "Reasoning effort level for supported models "
            "(e.g. 'minimal', 'low', 'medium', 'high', 'none' depend on model). "
            "If None, API default is used."
        ),
    )


class OpenAIResponsesLanguageModel(LanguageModel):
    """Language model that uses OpenAI's responses API."""

    def __init__(self, params: OpenAIResponsesLanguageModelParams) -> None:
        """
        Initialize the responses language model with configuration.

        Args:
            params (OpenAIResponsesLanguageModelParams):
                Parameters for the OpenAIResponsesLanguageModel.

        """
        super().__init__()

        self._client = params.client

        self._model = params.model

        self._max_retry_interval_seconds = params.max_retry_interval_seconds
        self._reasoning_effort = params.reasoning_effort

        metrics_factory = params.metrics_factory

        self._tracker = OperationTracker(
            metrics_factory, prefix="language_model_openai_responses"
        )

        self._should_collect_metrics = False
        if metrics_factory is not None:
            self._should_collect_metrics = True

            self._input_tokens_usage_counter = metrics_factory.get_counter(
                "language_model_openai_responses_usage_input_tokens",
                "Number of input tokens used for OpenAI language model",
            )
            self._input_cached_tokens_usage_counter = metrics_factory.get_counter(
                "language_model_openai_responses_usage_input_cached_tokens",
                (
                    "Number of tokens retrieved from cache "
                    "used for OpenAI language model"
                ),
            )
            self._output_tokens_usage_counter = metrics_factory.get_counter(
                "language_model_openai_responses_usage_output_tokens",
                "Number of output tokens used for OpenAI language model",
            )
            self._output_reasoning_tokens_usage_counter = metrics_factory.get_counter(
                "language_model_openai_responses_usage_output_reasoning_tokens",
                ("Number of reasoning tokens used for OpenAI language model"),
            )
            self._total_tokens_usage_counter = metrics_factory.get_counter(
                "language_model_openai_responses_usage_total_tokens",
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
                ResponseInputParam,
                [
                    {"role": "system", "content": system_prompt or ""},
                    {"role": "user", "content": user_prompt or ""},
                ],
            )

            generate_response_call_uuid = uuid4()
            request_data: dict[str, Any] = {
                "model": self._model,
                "input": input_prompts,
                "text_format": output_format.__name__,
                "max_attempts": max_attempts,
            }

            try:
                response = await self._client.with_options(
                    max_retries=max_attempts,
                ).responses.parse(
                    model=self._model,
                    input=input_prompts,
                    store=False,
                    text_format=output_format,
                )
            except openai.APITimeoutError as e:
                error_message = (
                    f"[call uuid: {generate_response_call_uuid}] "
                    "OpenAI Responses API request timed out while generating "
                    "a parsed response."
                )
                logger.exception("%s Request data: %s", error_message, request_data)
                raise ExternalServiceAPIError(error_message) from e
            except openai.OpenAIError as e:
                error_message = (
                    f"[call uuid: {generate_response_call_uuid}] "
                    "Giving up generating response "
                    f"due to non-retryable {type(e).__name__}"
                )
                logger.exception(error_message)
                raise ExternalServiceAPIError(error_message) from e

            self._collect_usage_metrics(response)

            return response.output_parsed

    async def generate_response(
        self,
        system_prompt: str | None = None,
        user_prompt: str | None = None,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, str] | None = None,
        max_attempts: int = 1,
    ) -> tuple[str, list[dict[str, Any]]]:
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
    ) -> tuple[str, list[dict[str, Any]], int, int]:
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
        """Generate a raw text response (and optional tool call)."""
        async with self._tracker("generate_response"):
            if max_attempts <= 0:
                raise ValueError("max_attempts must be a positive integer")

            input_prompts = cast(
                ResponseInputParam,
                [
                    {"role": "system", "content": system_prompt or ""},
                    {"role": "user", "content": user_prompt or ""},
                ],
            )

            generate_response_call_uuid = uuid4()

            response: Response | None = None
            request_data_base: dict[str, Any] = {
                "model": self._model,
                "input": input_prompts,
            }
            if tools is not None:
                request_data_base["tools"] = tools
                request_data_base["tool_choice"] = (
                    tool_choice if tool_choice is not None else "auto"
                )

            sleep_seconds = 1
            for attempt in range(1, max_attempts + 1):
                try:
                    logger.debug(
                        "[call uuid: %s] "
                        "Attempting to generate response using %s OpenAI language model: "
                        "on attempt %d with max attempts %d",
                        generate_response_call_uuid,
                        self._model,
                        attempt,
                        max_attempts,
                    )
                    if tools is None:
                        response = await self._client.responses.create(
                            model=self._model,
                            input=input_prompts,
                            store=False,
                        )
                    else:
                        response = await self._client.responses.create(
                            model=self._model,
                            input=input_prompts,
                            store=False,
                            tools=cast(list[ToolParam], tools),
                            tool_choice=cast(
                                Any,
                                tool_choice if tool_choice is not None else "auto",
                            ),
                        )
                    break
                except openai.APITimeoutError as e:
                    request_data = {
                        **request_data_base,
                        "attempt": attempt,
                        "max_attempts": max_attempts,
                    }
                    logger.warning(
                        "[call uuid: %s] "
                        "OpenAI Responses API request timed out on attempt %d/%d. "
                        "Request data: %s",
                        generate_response_call_uuid,
                        attempt,
                        max_attempts,
                        request_data,
                    )
                    if attempt >= max_attempts:
                        error_message = (
                            f"[call uuid: {generate_response_call_uuid}] "
                            "OpenAI Responses API request timed out. "
                            f"Giving up after {attempt} attempts."
                        )
                        logger.exception(error_message)
                        raise ExternalServiceAPIError(error_message) from e

                    await asyncio.sleep(sleep_seconds)
                    sleep_seconds *= 2
                    sleep_seconds = min(sleep_seconds, self._max_retry_interval_seconds)
                    continue
                except (
                    openai.RateLimitError,
                    openai.APIConnectionError,
                    openai.InternalServerError,
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

            if response is None:
                raise RuntimeError("OpenAI response was not generated")

            self._collect_usage_metrics(response)

            if response.output is None:
                return (response.output_text or "", [], 0, 0)

            function_calls_arguments: list[dict[str, Any]] = []
            try:
                for output in response.output:
                    if output.type != "function_call":
                        continue
                    function_call = cast(ResponseFunctionToolCall, output)
                    function_calls_arguments.append(
                        {
                            "call_id": function_call.call_id,
                            "function": {
                                "name": function_call.name,
                                "arguments": json_repair.loads(function_call.arguments),
                            },
                        }
                    )
            except (TypeError, ValueError) as e:
                raise ValueError(
                    "Failed to repair or parse JSON from function call arguments"
                ) from e

            return (
                response.output_text or "",
                function_calls_arguments,
                response.usage.input_tokens if response.usage else 0,
                response.usage.output_tokens if response.usage else 0,
            )

    def _collect_usage_metrics(self, response: Response) -> None:
        if not self._should_collect_metrics:
            return

        if response.usage is None:
            logger.debug("No usage information found in response")
            return

        try:
            self._input_tokens_usage_counter.increment(
                value=response.usage.input_tokens,
            )
            self._input_cached_tokens_usage_counter.increment(
                value=response.usage.input_tokens_details.cached_tokens,
            )
            self._output_tokens_usage_counter.increment(
                value=response.usage.output_tokens,
            )
            self._output_reasoning_tokens_usage_counter.increment(
                value=response.usage.output_tokens_details.reasoning_tokens,
            )
            self._total_tokens_usage_counter.increment(
                value=response.usage.total_tokens,
            )

        except Exception:
            logger.exception("Failed to collect usage metrics")
