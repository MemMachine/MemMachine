"""
OpenAI-based language model implementation.
"""

import asyncio
import json
import time
from typing import Any

import openai
from openai import AsyncOpenAI

from memmachine.common.metrics_factory.metrics_factory import MetricsFactory

from .language_model import LanguageModel


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
                  (default: "gpt-5-nano").
                - metrics_factory (MetricsFactory, optional):
                  An instance of MetricsFactory
                  for collecting usage metrics.
                - user_metrics_labels (dict[str, str], optional):
                  Labels to attach to the collected metrics.

        Raises:
            ValueError:
                If configuration argument values are missing or invalid.
            TypeError:
                If configuration argument values are of incorrect type.
        """
        super().__init__()

        self._model = config.get("model", "gpt-5-nano")

        api_key = config.get("api_key")
        if api_key is None:
            raise ValueError("Language API key must be provided")

        self._client = AsyncOpenAI(api_key=api_key)

        metrics_factory = config.get("metrics_factory")
        if metrics_factory is not None and not isinstance(
            metrics_factory, MetricsFactory
        ):
            raise TypeError("Metrics factory must be an instance of MetricsFactory")

        self._collect_metrics = False
        if metrics_factory is not None:
            self._collect_metrics = True
            self._user_metrics_labels = config.get("user_metrics_labels", {})
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
        tool_choice: str | dict[str, str] = "auto",
        max_attempts: int = 1,
    ) -> tuple[str, Any]:
        if max_attempts <= 0:
            raise ValueError("max_attempts must be a positive integer")

        input_prompts = [
            {"role": "system", "content": system_prompt or ""},
            {"role": "user", "content": user_prompt or ""},
        ]

        start_time = time.monotonic()
        sleep_seconds = 1
        for attempt in range(max_attempts):
            sleep_seconds *= 2
            try:
                response = await self._client.responses.create(
                    model=self._model,
                    input=input_prompts,
                    tools=tools,
                    tool_choice=tool_choice,
                )  # type: ignore
            # translate vendor specific exeception to common error
            except openai.AuthenticationError as e:
                raise ValueError("Invalid OpenAI API key") from e
            except openai.RateLimitError as e:
                if attempt + 1 >= max_attempts:
                    raise IOError("OpenAI rate limit exceeded") from e
                await asyncio.sleep(sleep_seconds)
                continue
            except openai.APITimeoutError as e:
                if attempt + 1 >= max_attempts:
                    raise IOError("OpenAI API timeout") from e
                await asyncio.sleep(sleep_seconds)
                continue
            except openai.APIConnectionError as e:
                if attempt + 1 >= max_attempts:
                    raise IOError("OpenAI API connection error") from e
                await asyncio.sleep(sleep_seconds)
                continue
            except openai.BadRequestError as e:
                raise ValueError("OpenAI invalid request") from e
            except openai.APIError as e:
                raise ValueError("OpenAI API error") from e
            except openai.OpenAIError as e:
                raise ValueError("OpenAI error") from e
            break

        end_time = time.monotonic()

        if self._collect_metrics and response.usage is not None:
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

        function_calls_arguments = [
            json.loads(output.arguments)
            for output in response.output
            if output.type == "function_call"
        ]

        return (
            response.output_text,
            function_calls_arguments,
        )
