"""Amazon Bedrock-based embedder implementation."""

import asyncio
import logging
import time
from collections.abc import Callable, Coroutine
from typing import Any
from uuid import uuid4

import numpy as np
from botocore.exceptions import ClientError
from langchain_aws import BedrockEmbeddings
from pydantic import BaseModel, Field, InstanceOf

from memmachine.common.data_types import ExternalServiceAPIError, SimilarityMetric
from memmachine.common.utils import chunk_text_balanced, unflatten_like

from .embedder import Embedder

logger = logging.getLogger(__name__)


class AmazonBedrockEmbedderParams(BaseModel):
    """Parameters for AmazonBedrockEmbedder."""

    client: InstanceOf[BedrockEmbeddings] = Field(
        ...,
        description="BedrockEmbeddings client instance.",
    )
    model_id: str = Field(
        ...,
        description=(
            "ID of the Bedrock model to use for embedding "
            "(e.g. 'amazon.titan-embed-text-v2:0')."
        ),
    )
    max_input_length: int | None = Field(
        default=None,
        description="Maximum input length for the model (in Unicode code points).",
        gt=0,
    )
    similarity_metric: SimilarityMetric = Field(
        default=SimilarityMetric.COSINE,
        description="Similarity metric to use for comparing embeddings.",
    )
    max_retry_interval_seconds: int = Field(
        default=120,
        description="Maximal retry interval in seconds (defualt: 120).",
        gt=0,
    )


class AmazonBedrockEmbedder(Embedder):
    """Embedder that uses Amazon Bedrock models for embeddings."""

    def __init__(self, params: AmazonBedrockEmbedderParams) -> None:
        """Initialize the embedder with Bedrock client parameters."""
        super().__init__()

        self._client = params.client

        self._model_id = params.model_id
        self._max_input_length = params.max_input_length
        self._similarity_metric = params.similarity_metric
        self._max_retry_interval_seconds = params.max_retry_interval_seconds

        # Get dimensions by embedding a dummy string.
        try:
            response = self._client.embed_documents(["."])
            self._dimensions = len(response[0])
        except ClientError:
            logger.exception("Failed to get embedding dimensions")

    @property
    def embeddings(self) -> BedrockEmbeddings:
        """Return the underlying BedrockEmbeddings client."""
        return self._client

    async def ingest_embed(
        self,
        inputs: list[Any],
        max_attempts: int = 1,
    ) -> list[list[float]]:
        """Embed input documents."""
        return await self._embed(
            inputs,
            self._ingest_embed_func,
            max_attempts,
        )

    async def _ingest_embed_func(
        self,
        inputs: list[Any],
    ) -> list[list[float]]:
        """Call Bedrock for document embeddings."""
        return await self._client.aembed_documents(inputs)

    async def search_embed(
        self,
        queries: list[Any],
        max_attempts: int = 1,
    ) -> list[list[float]]:
        """Embed search queries."""
        return await self._embed(
            queries,
            self._search_embed_func,
            max_attempts,
        )

    async def _search_embed_func(
        self,
        queries: list[Any],
    ) -> list[list[float]]:
        embed_queries_tasks = [self._client.aembed_query(query) for query in queries]
        return await asyncio.gather(*embed_queries_tasks)

    async def _embed(
        self,
        inputs: list[Any],
        async_embed_func: Callable[[list[Any]], Coroutine[Any, Any, list[list[float]]]],
        max_attempts: int = 1,
    ) -> list[list[float]]:
        """Shared retry logic for embedding requests."""
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

        embed_call_uuid = uuid4()

        start_time = time.monotonic()

        chunk_embeddings = []
        sleep_seconds = 1
        for attempt in range(1, max_attempts + 1):
            logger.debug(
                "[call uuid: %s] "
                "Attempting to create embeddings using %s Amazon Bedrock model: "
                "on attempt %d with max attempts %d",
                embed_call_uuid,
                self.model_id,
                attempt,
                max_attempts,
            )

            try:
                chunk_embeddings = await async_embed_func(chunks)
                break
            except Exception as e:
                # Assume all exceptions may be retried.
                if attempt >= max_attempts:
                    error_message = (
                        f"[call uuid: {embed_call_uuid}] "
                        f"Giving up creating embeddings "
                        f"after failed attempt {attempt} "
                        f"due to assumed retryable {type(e).__name__}: "
                        f"max attempts {max_attempts} reached"
                    )
                    logger.exception(error_message)
                    raise ExternalServiceAPIError(error_message) from e

                logger.info(
                    "[call uuid: %s] "
                    "Retrying creating embeddings in %d seconds "
                    "after failed attempt %d due to assumed retryable %s...",
                    embed_call_uuid,
                    sleep_seconds,
                    attempt,
                    type(e).__name__,
                )
                await asyncio.sleep(
                    min(sleep_seconds, self._max_retry_interval_seconds),
                )
                sleep_seconds *= 2

        end_time = time.monotonic()
        logger.debug(
            "[call uuid: %s] Embeddings created in %.3f seconds",
            embed_call_uuid,
            end_time - start_time,
        )

        inputs_chunk_embeddings = unflatten_like(
            chunk_embeddings,
            inputs_chunks,
        )

        # Average chunk embeddings to get input embeddings.
        return [
            np.mean(chunk_embeddings, axis=0).astype(float).tolist()
            for chunk_embeddings in inputs_chunk_embeddings
        ]

    @property
    def model_id(self) -> str:
        """Return the identifier for the embedding model."""
        return self._model_id

    @property
    def dimensions(self) -> int:
        """Return the embedding dimensionality."""
        return self._dimensions

    @property
    def similarity_metric(self) -> SimilarityMetric:
        """Return the similarity metric used by the embedder."""
        return self._similarity_metric
