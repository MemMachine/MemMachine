"""
Amazon Bedrock-based embedder implementation.
"""

import asyncio
import logging
import time
from collections.abc import Callable, Coroutine
from typing import Any
from uuid import uuid4

import botocore
from langchain_aws import BedrockEmbeddings
from pydantic import BaseModel, Field, SecretStr

from memmachine.common.data_types import ExternalServiceAPIError

from .data_types import SimilarityMetric
from .embedder import Embedder

logger = logging.getLogger(__name__)


class AmazonBedrockEmbedderConfig(BaseModel):
    """
    Configuration for AmazonBedrockEmbedder.

    Attributes:
        region (str):
            AWS region where Bedrock is hosted.
        aws_access_key_id (SecretStr):
            AWS access key ID for authentication.
        aws_secret_access_key (SecretStr):
            AWS secret access key for authentication.
        model_id (str):
            ID of the Bedrock model to use for embedding
            (e.g. 'amazon.titan-embed-text-v2:0').
        dimensions (int):
            Dimensionality of the embedding vectors.
        similarity_metric (SimilarityMetric):
            Similarity metric to use for comparing embeddings
            (default: SimilarityMetric.COSINE).
        max_retry_interval_seconds (int, optional):
            Maximal retry interval in seconds
            (default: 120).
    """

    region: str = Field(
        "us-west-2",
        description="AWS region where Bedrock is hosted.",
    )
    aws_access_key_id: SecretStr = Field(
        description=("AWS access key ID for authentication."),
    )
    aws_secret_access_key: SecretStr = Field(
        description=("AWS secret access key for authentication."),
    )
    model_id: str = Field(
        description=(
            "ID of the Bedrock model to use for embedding "
            "(e.g. 'amazon.titan-embed-text-v2:0')."
        ),
    )
    similarity_metric: SimilarityMetric = Field(
        SimilarityMetric.COSINE,
        description="Similarity metric to use for comparing embeddings.",
    )
    max_retry_interval_seconds: int = Field(
        120,
        description="Maximal retry interval in seconds (defualt: 120).",
        gt=0,
    )


class AmazonBedrockEmbedder(Embedder):
    """
    Embedder that uses Amazon Bedrock models
    to generate embeddings for inputs and queries.
    """

    def __init__(self, config: AmazonBedrockEmbedderConfig):
        """
        Initialize an AmazonBedrockEmbedder
        with the provided configuration.

        Args:
            config (AmazonBedrockEmbedderConfig):
                Configuration for the embedder.
        """
        super().__init__()

        region = config.region
        aws_access_key_id = config.aws_access_key_id
        aws_secret_access_key = config.aws_secret_access_key
        self._model_id = config.model_id
        self._similarity_metric = config.similarity_metric
        self._max_retry_interval_seconds = config.max_retry_interval_seconds

        self._embeddings = BedrockEmbeddings(
            region_name=region,
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key,
            model_id=self._model_id,
            config=botocore.config.Config(
                retries={
                    "total_max_attempts": 1,
                    "mode": "standard",
                }
            ),
        )

        # Get dimensions by embedding a dummy string.
        response = self._embeddings.embed_documents(["."])
        self._dimensions = len(response[0])

    async def ingest_embed(
        self,
        inputs: list[Any],
        max_attempts: int = 1,
    ) -> list[list[float]]:
        return await self._embed(
            inputs,
            self._ingest_embed_func,
            max_attempts,
        )

    async def _ingest_embed_func(
        self,
        inputs: list[Any],
    ) -> list[list[float]]:
        return await self._embeddings.aembed_documents(inputs)

    async def search_embed(
        self,
        queries: list[Any],
        max_attempts: int = 1,
    ) -> list[list[float]]:
        return await self._embed(
            queries,
            self._search_embed_func,
            max_attempts,
        )

    async def _search_embed_func(
        self,
        queries: list[Any],
    ) -> list[list[float]]:
        embed_queries_tasks = [
            self._embeddings.aembed_query(query) for query in queries
        ]
        return await asyncio.gather(*embed_queries_tasks)

    async def _embed(
        self,
        inputs: list[Any],
        async_embed_func: Callable[[list[Any]], Coroutine[Any, Any, list[list[float]]]],
        max_attempts: int = 1,
    ) -> list[list[float]]:
        if max_attempts <= 0:
            raise ValueError("max_attempts must be a positive integer")

        embed_call_uuid = uuid4()

        start_time = time.monotonic()

        sleep_seconds = 1
        for attempt in range(1, max_attempts + 1):
            logger.debug(
                "[call uuid: %s] "
                "Attempting to create embeddings using %s Amazon Bedrock model: "
                "on attempt %d with max attempts %d",
                embed_call_uuid,
                self._model_id,
                attempt,
                max_attempts,
            )

            try:
                embeddings = await async_embed_func(inputs)
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
                    logger.error(error_message)
                    raise ExternalServiceAPIError(error_message)

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
                    min(sleep_seconds, self._max_retry_interval_seconds)
                )
                sleep_seconds *= 2

        end_time = time.monotonic()
        logger.debug(
            "[call uuid: %s] Embeddings created in %.3f seconds",
            embed_call_uuid,
            end_time - start_time,
        )

        return embeddings

    @property
    def model_id(self) -> str:
        return self._model_id

    @property
    def dimensions(self) -> int:
        return self._dimensions

    @property
    def similarity_metric(self) -> SimilarityMetric:
        return self._similarity_metric
