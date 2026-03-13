"""OpenAI-based embedder implementation."""

import asyncio
import logging
from typing import Any
from uuid import UUID, uuid4

import numpy as np
import openai
from pydantic import BaseModel, Field, InstanceOf

from memmachine_server.common.data_types import (
    ExternalServiceAPIError,
    SimilarityMetric,
)
from memmachine_server.common.metrics_factory import MetricsFactory, OperationTracker
from memmachine_server.common.utils import (
    chunk_text_balanced,
    cluster_texts,
    unflatten_like,
)

from .embedder import Embedder

logger = logging.getLogger(__name__)


class OpenAIEmbedderParams(BaseModel):
    """Parameters for OpenAIEmbedder."""

    client: InstanceOf[openai.AsyncOpenAI] = Field(
        ...,
        description="AsyncOpenAI client to use for making API calls.",
    )
    model: str = Field(
        ...,
        description=(
            "Name of the OpenAI embedding model to use (e.g. 'text-embedding-3-small')."
        ),
    )
    dimensions: int = Field(
        ...,
        description=(
            "Dimensionality of the embedding vectors "
            "produced by the OpenAI embedding model."
        ),
        gt=0,
    )
    max_input_length: int | None = Field(
        default=None,
        description="Maximum input length for the model (in Unicode code points).",
        gt=0,
    )
    max_retry_interval_seconds: int = Field(
        default=120,
        description="Maximal retry interval in seconds when retrying API calls.",
        gt=0,
    )
    metrics_factory: InstanceOf[MetricsFactory] | None = Field(
        default=None,
        description="An instance of MetricsFactory for collecting usage metrics.",
    )


class OpenAIEmbedder(Embedder):
    """Embedder that uses OpenAI embedding models."""

    # https://platform.openai.com/docs/api-reference/embeddings/create#embeddings_create-input
    max_num_inputs_per_request = 2048
    max_total_input_length_per_request = (
        75000  # Assume at most 4 tokens per Unicode code point.
    )

    def __init__(self, params: OpenAIEmbedderParams) -> None:
        """Initialize the OpenAI embedder with configuration parameters."""
        super().__init__()

        self._client = params.client

        # https://platform.openai.com/docs/guides/embeddings#embedding-models
        self._model = params.model

        self._dimensions = params.dimensions
        self._use_dimensions_parameter = True

        self._max_retry_interval_seconds = params.max_retry_interval_seconds

        self._max_input_length = params.max_input_length

        metrics_factory = params.metrics_factory

        self._tracker = OperationTracker(metrics_factory, prefix="embedder_openai")

        self._should_collect_metrics = False
        if metrics_factory is not None:
            self._should_collect_metrics = True

            self._prompt_tokens_usage_counter = metrics_factory.get_counter(
                "embedder_openai_usage_prompt_tokens",
                "Number of tokens used by prompts to OpenAI embedder",
            )
            self._total_tokens_usage_counter = metrics_factory.get_counter(
                "embedder_openai_usage_total_tokens",
                "Number of tokens used by requests to OpenAI embedder",
            )

    async def ingest_embed(
        self,
        inputs: list[Any],
        max_attempts: int = 1,
    ) -> list[list[float]]:
        """Embed the provided inputs with retries."""
        async with self._tracker("ingest_embed"):
            return await self._embed(inputs, max_attempts)

    async def search_embed(
        self,
        queries: list[Any],
        max_attempts: int = 1,
    ) -> list[list[float]]:
        """Embed search queries with retries."""
        async with self._tracker("search_embed"):
            return await self._embed(queries, max_attempts)

    async def _embed(
        self,
        inputs: list[Any],
        max_attempts: int = 1,
    ) -> list[list[float]]:
        """Shared retrying embed logic."""
        if not inputs:
            return []
        if max_attempts <= 0:
            raise ValueError("max_attempts must be a positive integer")

        inputs = [input_text or "." for input_text in inputs]

        inputs_chunks = [
            chunk_text_balanced(input_text, self._max_input_length)
            if self._max_input_length is not None
            else [input_text]
            for input_text in inputs
        ]

        chunks = [chunk for input_chunks in inputs_chunks for chunk in input_chunks]
        chunk_clusters = cluster_texts(
            chunks,
            self.max_num_inputs_per_request,
            self.max_total_input_length_per_request,
        )

        embed_call_uuid = uuid4()

        logger.debug(
            "[call uuid: %s] "
            "Attempting to create embeddings using %s OpenAI model: "
            "%d total chunks in %d clusters with max attempts %d",
            embed_call_uuid,
            self._model,
            len(chunks),
            len(chunk_clusters),
            max_attempts,
        )
        clusters_chunk_embeddings_awaitables = [
            self._embed_chunk_cluster(
                embed_call_uuid=embed_call_uuid,
                cluster_number=cluster_number,
                chunk_cluster=chunk_cluster,
                max_attempts=max_attempts,
            )
            for cluster_number, chunk_cluster in enumerate(chunk_clusters)
        ]
        clusters_chunk_embeddings = await asyncio.gather(
            *clusters_chunk_embeddings_awaitables
        )

        chunk_embeddings = [
            chunk_embedding
            for cluster_chunk_embeddings in clusters_chunk_embeddings
            for chunk_embedding in cluster_chunk_embeddings
        ]
        inputs_chunk_embeddings = unflatten_like(
            chunk_embeddings,
            inputs_chunks,
        )

        # Average chunk embeddings to get input embeddings.
        return [
            np.mean(chunk_embeddings, axis=0).astype(float).tolist()
            for chunk_embeddings in inputs_chunk_embeddings
        ]

    async def _embed_chunk_cluster(
        self,
        embed_call_uuid: UUID,
        cluster_number: int,
        chunk_cluster: list[str],
        max_attempts: int = 1,
    ) -> list[list[float]]:
        sleep_seconds = 1
        for attempt in range(1, max_attempts + 1):
            logger.debug(
                "[call uuid: %s] "
                "Attempting to create embeddings for cluster number %d: "
                "on attempt %d with max attempts %d",
                embed_call_uuid,
                cluster_number,
                attempt,
                max_attempts,
            )

            try:
                # Internal try-except is required
                # for models that do not support dimensions parameter

                # Avoid concurrency issues by tracking whether dimensions parameter is used for this request only.
                dimensions_parameter_used = self._use_dimensions_parameter
                try:
                    response = (
                        await self._client.embeddings.create(
                            input=chunk_cluster,
                            model=self._model,
                            dimensions=self._dimensions,
                        )
                        if dimensions_parameter_used
                        else await self._client.embeddings.create(
                            input=chunk_cluster,
                            model=self._model,
                        )
                    )
                except openai.BadRequestError as err:
                    if "dimension" in str(err).lower() and dimensions_parameter_used:
                        response = await self._client.embeddings.create(
                            input=chunk_cluster,
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
                openai.InternalServerError,
            ) as err:
                # Exception may be retried.
                if attempt >= max_attempts:
                    error_message = (
                        f"[call uuid: {embed_call_uuid}] "
                        "Giving up creating embeddings "
                        f"for cluster number {cluster_number} "
                        f"after failed attempt {attempt} "
                        f"due to retryable {type(err).__name__}: "
                        f"max attempts {max_attempts} reached"
                    )
                    logger.exception(error_message)
                    raise ExternalServiceAPIError(error_message) from err

                logger.info(
                    "[call uuid: %s] "
                    "Retrying creating embeddings for cluster number %d "
                    "in %d seconds "
                    "after failed attempt %d due to retryable %s...",
                    embed_call_uuid,
                    cluster_number,
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
                    f"for cluster number {cluster_number} "
                    f"after failed attempt {attempt} "
                    f"due to non-retryable {type(err).__name__}"
                )
                logger.exception(error_message)
                raise ExternalServiceAPIError(error_message) from err

        if self._should_collect_metrics:
            self._prompt_tokens_usage_counter.increment(
                value=response.usage.prompt_tokens,
            )
            self._total_tokens_usage_counter.increment(
                value=response.usage.total_tokens,
            )

        if len(response.data[0].embedding) != self._dimensions:
            error_message = (
                f"[call uuid: {embed_call_uuid}] "
                f"Received embedding dimensionality {len(response.data[0].embedding)} "
                f"does not match expected dimensionality {self._dimensions}"
            )
            logger.exception(error_message)
            raise ExternalServiceAPIError(error_message)

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
