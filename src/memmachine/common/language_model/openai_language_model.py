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

from memmachine.common.data_types import ExternalServiceAPIError
from memmachine.common.metrics_factory.metrics_factory import MetricsFactory

from .language_model import LanguageModel

logger = logging.getLogger(__name__)


class OpenAILanguageModel(LanguageModel):
    """
    Language model that uses OpenAI's models
    to generate responses based on prompts and tools.
    """

    def __init__(self, config: dict[str, Any]):
        """
        Initialize an OpenAILanguageModel
        with the provided configuration.

        Args:
            config (dict[str, Any]):
                Configuration dictionary containing:
                - api_key (str):
                  API key for accessing the OpenAI service.
                - model (str, optional):
                  Name of the OpenAI model to use
                - metrics_factory (MetricsFactory, optional):
                  An instance of MetricsFactory
                  for collecting usage metrics.
                - user_metrics_labels (dict[str, str], optional):
                  Labels to attach to the collected metrics.
                - max_retry_interval_seconds(int, optional):
                  Maximal retry interval in seconds when retrying API calls.
                  The default value is 120 seconds.

        Raises:
            ValueError:
                If configuration argument values are missing or invalid.
            TypeError:
                If configuration argument values are of incorrect type.
        """
        super().__init__()

        self._model = config.get("model")
        if self._model is None:
            raise ValueError("The model name must be configured")
        if not isinstance(self._model, str):
            raise TypeError("The model name must be a string")

        api_key = config.get("api_key")
        if api_key is None:
            raise ValueError("Language API key must be provided")

        self._client = openai.AsyncOpenAI(api_key=api_key)

        self._max_retry_interval_seconds = config.get("max_retry_interval_seconds", 120)
        if not isinstance(self._max_retry_interval_seconds, int):
            raise TypeError("max_retry_interval_seconds must be an integer")

        if self._max_retry_interval_seconds <= 0:
            raise ValueError("max_retry_interval_seconds must be a positive integer")

        metrics_factory = config.get("metrics_factory")
        if metrics_factory is not None and not isinstance(
            metrics_factory, MetricsFactory
        ):
            raise TypeError("Metrics factory must be an instance of MetricsFactory")

        self._collect_metrics = False
        if metrics_factory is not None:
            self._collect_metrics = True
            self._user_metrics_labels = config.get("user_metrics_labels", {})
            if not isinstance(self._user_metrics_labels, dict):
                raise TypeError("user_metrics_labels must be a dictionary")
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
