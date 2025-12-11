"""OpenAI-based embedder implementation."""

import asyncio
import logging
import time
from typing import Any
from uuid import uuid4

import openai
from pydantic import BaseModel, Field, InstanceOf

from memmachine.common.data_types import ExternalServiceAPIError, SimilarityMetric
from memmachine.common.metrics_factory.metrics_factory import MetricsFactory

from .embedder import Embedder

logger = logging.getLogger(__name__)


class OpenAIEmbedderParams(BaseModel):
    """Parameters for OpenAIEmbedder."""

    client: InstanceOf[openai.AsyncOpenAI] = Field(
        ...,
        description="AsyncOpenAI client to use for making API calls",
    )
    model: str = Field(
        ...,
        description=(
            "Name of the OpenAI embedding model to use (e.g. 'text-embedding-3-small')"
        ),
    )
    dimensions: int = Field(
        ...,
        description=(
            "Dimensionality of the embedding vectors "
            "produced by the OpenAI embedding model"
        ),
        gt=0,
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


class OpenAIEmbedder(Embedder):
    """Embedder that uses OpenAI embedding models."""

    def __init__(self, params: OpenAIEmbedderParams) -> None:
        """Initialize the OpenAI embedder with configuration parameters."""
        super().__init__()

        self._client = params.client

        # https://platform.openai.com/docs/guides/embeddings#embedding-models
        self._model = params.model

        self._dimensions = params.dimensions
        self._use_dimensions_parameter = True

        self._max_retry_interval_seconds = params.max_retry_interval_seconds

        metrics_factory = params.metrics_factory

        self._collect_metrics = False
        if metrics_factory is not None:
            self._collect_metrics = True
            self._user_metrics_labels = params.user_metrics_labels
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
        max_attempts: int = 1,
    ) -> list[list[float]]:
        """Embed the provided inputs with retries."""
        return await self._embed(inputs, max_attempts)

    async def search_embed(
        self,
        queries: list[Any],
        max_attempts: int = 1,
    ) -> list[list[float]]:
        """Embed search queries with retries."""
        return await self._embed(queries, max_attempts)

    async def _embed(  # noqa: C901
        self,
        inputs: list[Any],
        max_attempts: int = 1,
    ) -> list[list[float]]:
        """Shared retrying embed logic."""
        if not inputs:
            return []
        if max_attempts <= 0:
            raise ValueError("max_attempts must be a positive integer")

        inputs = [item.replace("\n", " ") if item else "\n" for item in inputs]

        embed_call_uuid = uuid4()

        start_time = time.monotonic()

        sleep_seconds = 1
        for attempt in range(1, max_attempts + 1):
            try:
                logger.debug(
                    "[call uuid: %s] "
                    "Attempting to create embeddings using %s OpenAI model: "
                    "on attempt %d with max attempts %d",
                    embed_call_uuid,
                    self._model,
                    attempt,
                    max_attempts,
                )
                # Internal try-except is required
                # for models that do not support dimensions parameter
                try:
                    response = (
                        await self._client.embeddings.create(
                            input=inputs,
                            model=self._model,
                            dimensions=self._dimensions,
                        )
                        if self._use_dimensions_parameter
                        else await self._client.embeddings.create(
                            input=inputs,
                            model=self._model,
                        )
                    )
                except openai.BadRequestError as err:
                    if (
                        "dimension" in str(err).lower()
                        and self._use_dimensions_parameter
                    ):
                        response = await self._client.embeddings.create(
                            input=inputs,
                            model=self._model,
                        )
                        self._use_dimensions_parameter = False
                        break
                    raise
                break
            except (
                openai.RateLimitError,
                openai.APITimeoutError,
                openai.APIConnectionError,
            ) as err:
                # Exception may be retried.
                if attempt >= max_attempts:
                    error_message = (
                        f"[call uuid: {embed_call_uuid}] "
                        "Giving up creating embeddings "
                        f"after failed attempt {attempt} "
                        f"due to retryable {type(err).__name__}: "
                        f"max attempts {max_attempts} reached"
                    )
                    logger.exception(error_message)
                    raise ExternalServiceAPIError(error_message) from err

                logger.info(
                    "[call uuid: %s] "
                    "Retrying creating embeddings in %d seconds "
                    "after failed attempt %d due to retryable %s...",
                    embed_call_uuid,
                    sleep_seconds,
                    attempt,
                    type(err).__name__,
                )
                await asyncio.sleep(
                    min(sleep_seconds, self._max_retry_interval_seconds),
                )
                sleep_seconds *= 2
                continue
            except (openai.APIError, openai.OpenAIError) as err:
                error_message = (
                    f"[call uuid: {embed_call_uuid}] "
                    "Giving up creating embeddings "
                    f"after failed attempt {attempt} "
                    f"due to non-retryable {type(err).__name__}"
                )
                logger.exception(error_message)
                raise ExternalServiceAPIError(error_message) from err

        end_time = time.monotonic()
        logger.debug(
            "[call uuid: %s] Embeddings created in %.3f seconds",
            embed_call_uuid,
            end_time - start_time,
        )

        if len(response.data[0].embedding) != self._dimensions:
            error_message = (
                f"[call uuid: {embed_call_uuid}] "
                f"Received embedding dimensionality {len(response.data[0].embedding)} "
                f"does not match expected dimensionality {self._dimensions}"
            )
            logger.exception(error_message)
            raise ExternalServiceAPIError(error_message)

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

    @property
    def model_id(self) -> str:
        """Return the embedding model identifier."""
        return self._model

    @property
    def dimensions(self) -> int:
        """Return the embedding dimensionality."""
        return self._dimensions

    @property
    def similarity_metric(self) -> SimilarityMetric:
        """Return the similarity metric used by this embedder."""
        # https://platform.openai.com/docs/guides/embeddings
        return SimilarityMetric.COSINE
