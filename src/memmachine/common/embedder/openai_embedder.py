"""
OpenAI-based embedder implementation.
"""

import asyncio
import logging
import time
from typing import Any

import openai

from memmachine.common.metrics_factory.metrics_factory import MetricsFactory

from .embedder import Embedder

logger = logging.getLogger(__name__)


class OpenAIEmbedder(Embedder):
    """
    Embedder that uses OpenAI's embedding models
    to generate embeddings for inputs and queries.
    """

    def __init__(self, config: dict[str, Any]):
        """
        Initialize an OpenAIEmbedder with the provided configuration.

        Args:
            config (dict[str, Any]):
                Configuration dictionary containing:
                - api_key (str):
                  API key for accessing the OpenAI service.
                - model (str, optional):
                  Name of the OpenAI embedding model to use
                  (default: "text-embedding-3-small").
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

        self._model = config.get("model", "text-embedding-3-small")

        api_key = config.get("api_key")
        if api_key is None:
            raise ValueError("Embedder API key must be provided")

        self._client = openai.AsyncOpenAI(api_key=api_key)

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
            label_names = self._user_metrics_labels.keys()

            self._prompt_tokens_usage_counter = metrics_factory.get_counter(
                "embedder_openai_usage_prompt_tokens",
                "Number of tokens used by prompts to OpenAI embedder",
                label_names=label_names,
            )
            self._total_tokens_usage_counter = metrics_factory.get_counter(
                "embedder_openai_usage_total_tokens",
                "Number of tokens used by requests to OpenAI embedder",
                label_names=label_names,
            )
            self._latency_summary = metrics_factory.get_summary(
                "embedder_openai_latency_seconds",
                "Latency in seconds for OpenAI embedder requests",
                label_names=label_names,
            )

    async def ingest_embed(
        self,
        inputs: list[Any],
        max_attempts: int = 1
    ) -> list[list[float]]:
        return await self._embed(inputs, max_attempts)

    async def search_embed(
        self,
        queries: list[Any],
        max_attempts: int = 1
    ) -> list[list[float]]:
        return await self._embed(queries, max_attempts)

    async def _embed(
            self,
            inputs: list[Any],
            max_attempts: int = 1) -> list[list[float]]:
        if not inputs:
            return []
        if max_attempts <= 0:
            raise ValueError("max_attempts must be a positive integer")

        inputs = [
            input.replace("\n", " ") if input else "\n" for input in inputs
        ]

        start_time = time.monotonic()
        sleep_seconds = 1
        for attempt in range(max_attempts):
            sleep_seconds *= 2
            try:
                response = await self._client.embeddings.create(
                    input=inputs, model=self._model
                )
            # translate vendor specific exeception to common error
            # for rate limit and timeout error, may retry the request
            except openai.AuthenticationError as e:
                raise ValueError("Invalid OpenAI API key") from e
            except openai.RateLimitError as e:
                logger.warning("OpenAI rate limit exceeded")
                if attempt + 1 >= max_attempts:
                    raise IOError("OpenAI rate limit exceeded") from e
                await asyncio.sleep(sleep_seconds)
                continue
            except openai.APITimeoutError as e:
                logger.warning("OpenAI API timeout")
                if attempt + 1 >= max_attempts:
                    raise IOError("OpenAI API timeout") from e
                await asyncio.sleep(sleep_seconds)
                continue
            except openai.APIConnectionError as e:
                logger.warning("OpenAI API connection error")
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

        if self._collect_metrics:
            self._prompt_tokens_usage_counter.increment(
                value=response.usage.prompt_tokens,
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

        return [datum.embedding for datum in response.data]
