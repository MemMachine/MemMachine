"""
OpenAI-completions API based language model implementation.
"""

import asyncio
import json
import time
from typing import Any
from urllib.parse import urlparse

from openai import AsyncOpenAI
import openai
from memmachine.common.metrics_factory.metrics_factory import MetricsFactory

from .language_model import LanguageModel


class OpenAICompatibleLanguageModel(LanguageModel):
    """
    Language model that uses OpenAI's completions API
    to generate responses based on prompts and tools.
    """

    def __init__(self, config: dict[str, Any]):
        """
        Initialize an OpenAICompatibleLanguageModel
        with the provided configuration.

        Args:
            config (dict[str, Any]):
                Configuration dictionary containing:
                - api_key (str):
                  API key for accessing the OpenAI service.
                - model (str):
                  Name of the OpenAI model to use
                - metrics_factory (MetricsFactory, optional):
                  An instance of MetricsFactory
                  for collecting usage metrics.
                - user_metrics_labels (dict[str, str], optional):
                  Labels to attach to the collected metrics.
                - base_url: The base URL of the model
                - max_delay: maximal seconds to delay when retrying API calls.
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

        base_url = config.get("base_url")
        if base_url is not None:
            try:
                parsed_url = urlparse(base_url)
                if not parsed_url.scheme or not parsed_url.netloc:
                    raise ValueError(f"Invalid base URL: {base_url}")
            except ValueError as e:
                raise ValueError(f"Invalid base URL: {base_url}") from e

        self._client = AsyncOpenAI(
            api_key=api_key,
            base_url=base_url
        )

        self._max_delay = config.get("max_delay", 120)
        if not isinstance(self._max_delay, int):
            raise TypeError("max_delay must be an integer")
        if self._max_delay <= 0:
            raise ValueError("max_delay must be a positive integer")

        metrics_factory = config.get("metrics_factory")
        if metrics_factory is not None and not isinstance(
            metrics_factory, MetricsFactory
        ):
            raise TypeError(
                "Metrics factory must be an instance of MetricsFactory"
            )

        self._collect_metrics = False
        if metrics_factory is not None:
            self._collect_metrics = True
            self._user_metrics_labels = config.get("user_metrics_labels", {})
            if not isinstance(self._user_metrics_labels, dict):
                raise TypeError(
                    "user_metrics_labels must be a dictionary"
                )
            label_names = self._user_metrics_labels.keys()

            self._input_tokens_usage_counter = metrics_factory.get_counter(
                "language_model_openai_usage_input_tokens",
                "Number of input tokens used for OpenAI language model",
                label_names=label_names,
            )
            self._output_tokens_usage_counter = metrics_factory.get_counter(
                "language_model_openai_usage_output_tokens",
                "Number of output tokens used for OpenAI language model",
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

    @property
    def model(self) -> str:
        """Get the configured model name"""
        return self._model

    @property
    def max_delay(self) -> int:
        """Get the configured maximum delay"""
        return self._max_delay

    @property
    def collect_metrics(self) -> bool:
        """Get whether metrics should be collected"""
        return self._collect_metrics

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
            sleep_seconds = min(sleep_seconds, self._max_delay)
            try:
                response = await self._client.chat.completions.create(
                    model=self._model,
                    messages=input_prompts,
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
                raise ValueError(f"OpenAI API error {str(e)}") from e
            except openai.OpenAIError as e:
                raise ValueError("OpenAI error") from e
            break

        end_time = time.monotonic()

        if self._collect_metrics and response.usage is not None:
            self._input_tokens_usage_counter.increment(
                value=response.usage.prompt_tokens,
                labels=self._user_metrics_labels,
            )
            self._output_tokens_usage_counter.increment(
                value=response.usage.completion_tokens,
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

        function_calls_arguments = []
        try:
            if response.choices[0].message.tool_calls:
                for tool_call in response.choices[0].message.tool_calls:
                    function_calls_arguments.append({
                        "call_id": tool_call.id,
                        "function": {
                            "name": tool_call.function.name,
                            "arguments": json.loads(
                                tool_call.function.arguments
                            ),
                        }
                    })
        except (json.JSONDecodeError) as e:
            raise ValueError("JSON decode error") from e

        return (
            response.choices[0].message.content,
            function_calls_arguments,
        )
