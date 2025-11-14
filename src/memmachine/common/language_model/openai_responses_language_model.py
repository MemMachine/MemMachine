"""
OpenAI-based language model implementation.
"""

import asyncio
import json
import logging
import time
from typing import Any
from uuid import uuid4

import openai
from pydantic import BaseModel, Field, InstanceOf

from memmachine.common.data_types import ExternalServiceAPIError
from memmachine.common.metrics_factory.metrics_factory import MetricsFactory

from .language_model import LanguageModel

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
        user_metrics_labels (dict[str, str]):
            Labels to attach to the collected metrics
            (default: {}).
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
    user_metrics_labels: dict[str, str] = Field(
        default_factory=dict,
        description="Labels to attach to the collected metrics",
    )


class OpenAIResponsesLanguageModel(LanguageModel):
    """
    Language model that uses OpenAI's models
    to generate responses based on prompts and tools.
    """

    def __init__(self, params: OpenAIResponsesLanguageModelParams):
        """
        Initialize an OpenAIResponsesLanguageModel
        with the provided parameters.

        Args:
            params (OpenAIResponsesLanguageModelParams):
                Parameters for the OpenAIResponsesLanguageModel.
        """
        super().__init__()

        self._client = params.client

        self._model = params.model

        self._max_retry_interval_seconds = params.max_retry_interval_seconds

        metrics_factory = params.metrics_factory

        self._collect_metrics = False
        if metrics_factory is not None:
            self._collect_metrics = True
            self._user_metrics_labels = params.user_metrics_labels
            label_names = self._user_metrics_labels.keys()

            self._input_tokens_usage_counter = metrics_factory.get_counter(
                "language_model_openai_usage_input_tokens",
                "Number of input tokens used for OpenAI language model",
                label_names=label_names,
            )
            self._input_cached_tokens_usage_counter = metrics_factory.get_counter(
                "language_model_openai_usage_input_cached_tokens",
                (
                    "Number of tokens retrieved from cache "
                    "used for OpenAI language model"
                ),
                label_names=label_names,
            )
            self._output_tokens_usage_counter = metrics_factory.get_counter(
                "language_model_openai_usage_output_tokens",
                "Number of output tokens used for OpenAI language model",
                label_names=label_names,
            )
            self._output_reasoning_tokens_usage_counter = metrics_factory.get_counter(
                "language_model_openai_usage_output_reasoning_tokens",
                ("Number of reasoning tokens used for OpenAI language model"),
                label_names=label_names,
            )
            self._total_tokens_usage_counter = metrics_factory.get_counter(
                "language_model_openai_usage_total_tokens",
                "Number of tokens used for OpenAI language model",
                label_names=label_names,
            )
            self._latency_summary = metrics_factory.get_summary(
                "language_model_openai_latency_seconds",
                "Latency in seconds for OpenAI language model requests",
                label_names=label_names,
            )

    async def generate_response(
        self,
        system_prompt: str | None = None,
        user_prompt: str | None = None,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, str] | None = None,
        max_attempts: int = 1,
    ) -> tuple[str, Any]:
        if max_attempts <= 0:
            raise ValueError("max_attempts must be a positive integer")

        input_prompts = [
            {"role": "system", "content": system_prompt or ""},
            {"role": "user", "content": user_prompt or ""},
        ]

        generate_response_call_uuid = uuid4()

        start_time = time.monotonic()

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
                response = await self._client.responses.create(
                    model=self._model,
                    input=input_prompts,
                    tools=tools,
                    tool_choice=tool_choice if tool_choice is not None else "auto",
                )  # type: ignore
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
                    logger.error(error_message)
                    raise ExternalServiceAPIError(error_message)

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
                logger.error(error_message)
                raise ExternalServiceAPIError(error_message)

        end_time = time.monotonic()
        logger.debug(
            "[call uuid: %s] Response generated in %.3f seconds",
            generate_response_call_uuid,
            end_time - start_time,
        )

        if self._collect_metrics:
            if response.usage is not None:
                self._input_tokens_usage_counter.increment(
                    value=response.usage.input_tokens,
                    labels=self._user_metrics_labels,
                )
                self._input_cached_tokens_usage_counter.increment(
                    value=response.usage.input_tokens_details.cached_tokens,
                    labels=self._user_metrics_labels,
                )
                self._output_tokens_usage_counter.increment(
                    value=response.usage.output_tokens,
                    labels=self._user_metrics_labels,
                )
                self._output_reasoning_tokens_usage_counter.increment(
                    value=response.usage.output_tokens_details.reasoning_tokens,
                    labels=self._user_metrics_labels,
                )
                self._total_tokens_usage_counter.increment(
                    value=response.usage.total_tokens,
                    labels=self._user_metrics_labels,
                )

            self._latency_summary.observe(
                value=end_time - start_time,
                labels=self._user_metrics_labels,
            )

        if response.output is None:
            return (response.output_text or "", [])

        try:
            function_calls_arguments = [
                {
                    "call_id": output.call_id,
                    "function": {
                        "name": output.name,
                        "arguments": json.loads(output.arguments),
                    },
                }
                for output in response.output
                if output.type == "function_call"
            ]
        except json.JSONDecodeError as e:
            raise ValueError("JSON decode error") from e

        return (
            response.output_text,
            function_calls_arguments,
        )
