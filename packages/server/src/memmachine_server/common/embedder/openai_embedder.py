"""OpenAI-based embedder implementation."""

import asyncio
import logging
import random
import time
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
    chunk_text,
    cluster_texts,
    unflatten_like,
)

from .embedder import Embedder
from .openai_retry import (
    RetryGate,
    describe_error,
    error_headers,
    is_retryable,
    parse_openai_wait,
)

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
        description=(
            "Longest backoff in seconds between attempts when the provider "
            "does not say how long to wait."
        ),
        gt=0,
    )
    search_retry_budget_seconds: float = Field(
        default=10.0,
        description=(
            "Seconds a search embedding may take, attempts and waits "
            "between them included."
        ),
        gt=0,
    )
    ingest_retry_budget_seconds: float = Field(
        default=60.0,
        description=(
            "Seconds an ingestion embedding may take, attempts and waits "
            "between them included."
        ),
        gt=0,
    )
    min_request_time_seconds: float = Field(
        default=0.25,
        description=(
            "An attempt starts only if at least this many seconds of the "
            "budget remain after any wait."
        ),
        gt=0,
    )
    connect_timeout_seconds: float = Field(
        default=2.0,
        description=(
            "Seconds an attempt may spend connecting, so an unreachable "
            "endpoint fails without spending the budget."
        ),
        gt=0,
    )
    metrics_factory: InstanceOf[MetricsFactory] | None = Field(
        default=None,
        description="An instance of MetricsFactory for collecting usage metrics.",
    )
    batch_size: int | None = Field(
        None,
        description="Batch size for embedding requests.",
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
        super().__init__(batch_size=params.batch_size)

        self._client = params.client

        # https://platform.openai.com/docs/guides/embeddings#embedding-models
        self._model = params.model

        self._dimensions = params.dimensions
        self._use_dimensions_parameter = True

        self._max_retry_interval_seconds = params.max_retry_interval_seconds
        self._search_retry_budget_seconds = params.search_retry_budget_seconds
        self._ingest_retry_budget_seconds = params.ingest_retry_budget_seconds
        self._min_request_time_seconds = params.min_request_time_seconds
        self._connect_timeout_seconds = params.connect_timeout_seconds
        # One gate for every request of this embedder, so that while the
        # provider rate-limits it, new calls cannot cut ahead of older ones.
        # The lambdas look time and asyncio up at each call.
        self._gate = RetryGate(
            max_pace=params.max_retry_interval_seconds,
            clock=lambda: time.monotonic(),
            sleep=lambda seconds: asyncio.sleep(seconds),
        )

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

    async def _ingest_embed(
        self,
        inputs: list[Any],
        max_attempts: int = 1,  # noqa: ARG002
    ) -> list[list[float]]:
        """
        Embed the provided inputs, retrying within the ingestion budget.

        `max_attempts` is ignored: the time budget bounds the retries.
        """
        async with self._tracker("ingest_embed"):
            return await self._embed(inputs, self._ingest_retry_budget_seconds)

    async def _search_embed(
        self,
        queries: list[Any],
        max_attempts: int = 1,  # noqa: ARG002
    ) -> list[list[float]]:
        """
        Embed search queries, retrying within the search budget.

        `max_attempts` is ignored: the time budget bounds the retries.
        """
        async with self._tracker("search_embed"):
            return await self._embed(queries, self._search_retry_budget_seconds)

    async def _embed(
        self,
        inputs: list[Any],
        retry_budget_seconds: float,
    ) -> list[list[float]]:
        """Shared embed logic: every request of one call shares one deadline."""
        if not inputs:
            return []
        deadline = time.monotonic() + retry_budget_seconds

        # The OpenAI API rejects empty inputs,
        # and averaging over zero chunks is undefined.
        inputs = [input_text or "." for input_text in inputs]

        # No chunk may exceed the per-request cap, or cluster_texts rejects it:
        # a configured limit above the cap can never be honored.
        effective_max = min(
            self._max_input_length or self.max_total_input_length_per_request,
            self.max_total_input_length_per_request,
        )
        # Greedy max-size chunks measured more faithful to the whole-text
        # embedding than balanced chunks when combined with the
        # length-weighted average below: most characters get the largest
        # available context window.
        inputs_chunks = [chunk_text(input_text, effective_max) for input_text in inputs]

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
            "%d total chunks in %d clusters within %.1f s",
            embed_call_uuid,
            self._model,
            len(chunks),
            len(chunk_clusters),
            retry_budget_seconds,
        )
        clusters_chunk_embeddings_awaitables = [
            self._embed_chunk_cluster(
                embed_call_uuid=embed_call_uuid,
                cluster_number=cluster_number,
                chunk_cluster=chunk_cluster,
                deadline=deadline,
                retry_budget_seconds=retry_budget_seconds,
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

        # Average chunk embeddings to get input embeddings, weighting each
        # chunk by its length: under mean pooling the whole-text embedding is
        # approximately a length-weighted average of its chunks' embeddings,
        # and greedy chunking makes chunk lengths uneven.
        return [
            np.average(
                chunk_embeddings,
                axis=0,
                weights=[max(len(chunk), 1) for chunk in input_chunks],
            )
            .astype(float)
            .tolist()
            for input_chunks, chunk_embeddings in zip(
                inputs_chunks, inputs_chunk_embeddings, strict=True
            )
        ]

    async def _embed_chunk_cluster(  # noqa: C901
        self,
        embed_call_uuid: UUID,
        cluster_number: int,
        chunk_cluster: list[str],
        deadline: float,
        retry_budget_seconds: float,
    ) -> list[list[float]]:
        """
        Embed one cluster, retrying transient failures until `deadline`.

        Every attempt waits its turn at the embedder's gate. When the provider
        says how long to wait (retry-after-ms, Retry-After, or
        x-ratelimit-reset-*), the gate holds every request for that long, then
        gives turns oldest call first; the call gives up at once if the
        deadline leaves no time for the wait. A failure without a wait backs
        off exponentially with full jitter, cut to fit the deadline. No
        attempt starts that the deadline leaves no time for.
        """
        prefix = (
            f"[call uuid: {embed_call_uuid}] "
            f"Giving up creating embeddings for cluster number {cluster_number}"
        )
        attempt = 0
        last_error: openai.APIError | None = None
        while True:
            try:
                await self._gate.admit(
                    deadline - self._min_request_time_seconds,
                    since=deadline - retry_budget_seconds,
                )
            except TimeoutError:
                error_message = (
                    f"{prefix} before attempt {attempt + 1}: "
                    f"the {retry_budget_seconds:g} s budget is spent; "
                    f"last error {describe_error(last_error)}"
                )
                logger.exception(error_message)
                raise ExternalServiceAPIError(error_message) from last_error

            attempt += 1
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
                            timeout=self._request_timeout(deadline),
                        )
                        if dimensions_parameter_used
                        else await self._client.embeddings.create(
                            input=chunk_cluster,
                            model=self._model,
                            timeout=self._request_timeout(deadline),
                        )
                    )
                except openai.BadRequestError as err:
                    if "dimension" in str(err).lower() and dimensions_parameter_used:
                        response = await self._client.embeddings.create(
                            input=chunk_cluster,
                            model=self._model,
                            timeout=self._request_timeout(deadline),
                        )
                        self._use_dimensions_parameter = False
                        break
                    raise
                break
            except openai.APIError as err:
                if not is_retryable(err):
                    error_message = (
                        f"{prefix} after attempt {attempt}: "
                        f"non-retryable {describe_error(err)}"
                    )
                    logger.exception(error_message)
                    raise ExternalServiceAPIError(error_message) from err

                last_error = err
                wait = self._retry_wait(err, attempt, deadline)
                if wait is None:
                    error_message = (
                        f"{prefix} after attempt {attempt}: {describe_error(err)}; "
                        "waiting as long as it asks would pass "
                        f"the {retry_budget_seconds:g} s budget"
                    )
                    logger.exception(error_message)
                    raise ExternalServiceAPIError(error_message) from err

                logger.warning(
                    "[call uuid: %s] "
                    "Attempt %d to create embeddings for cluster number %d failed: "
                    "%s; retrying in %.2f s",
                    embed_call_uuid,
                    attempt,
                    cluster_number,
                    describe_error(err),
                    max(wait, self._gate.holding()),
                )
                if wait:
                    await asyncio.sleep(wait)
            except openai.OpenAIError as err:
                error_message = (
                    f"{prefix} after attempt {attempt}: "
                    f"non-retryable {type(err).__name__}: {err}"
                )
                logger.exception(error_message)
                raise ExternalServiceAPIError(error_message) from err

        self._gate.succeeded()

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
            logger.error(error_message)
            raise ExternalServiceAPIError(error_message)

        return [datum.embedding for datum in response.data]

    def _retry_wait(
        self, err: openai.APIError, attempt: int, deadline: float
    ) -> float | None:
        """
        Return the seconds this call sleeps after failed attempt number `attempt`.

        A provider that says how long to wait holds the gate for every request
        of this embedder, so the call sleeps 0 and waits its turn at the gate.
        Returns None when that wait leaves no time before `deadline`: retrying
        sooner than the provider asks would be refused. A failure without a
        wait backs off exponentially with full jitter, cut to fit the budget.
        """
        # Time left for waiting, keeping enough for one more attempt.
        available = deadline - time.monotonic() - self._min_request_time_seconds
        server_wait = parse_openai_wait(error_headers(err))
        if server_wait is not None:
            if server_wait > available:
                return None
            self._gate.limited(server_wait)
            return 0.0
        backoff = random.uniform(
            0, min(2 ** (attempt - 1) * 0.25, self._max_retry_interval_seconds)
        )
        return max(min(backoff + random.uniform(0, 0.1), available), 0.0)

    def _request_timeout(self, deadline: float) -> openai.Timeout:
        """Return the timeout of a request that must finish by `deadline`, with connecting capped."""
        remaining = max(deadline - time.monotonic(), 0.001)
        return openai.Timeout(
            remaining, connect=min(remaining, self._connect_timeout_seconds)
        )

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
