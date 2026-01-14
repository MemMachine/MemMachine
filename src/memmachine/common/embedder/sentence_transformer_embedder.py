"""Sentence transformer-based embedder implementation."""

import asyncio
import logging
import time
from typing import Any
from uuid import uuid4

import numpy as np
from pydantic import BaseModel, Field, InstanceOf
from sentence_transformers import SentenceTransformer

from memmachine.common.data_types import ExternalServiceAPIError, SimilarityMetric
from memmachine.common.utils import chunk_text_balanced, unflatten_like

from .embedder import Embedder

logger = logging.getLogger(__name__)


class SentenceTransformerEmbedderParams(BaseModel):
    """Parameters for SentenceTransformerEmbedder."""

    model_name: str = Field(
        ...,
        description="The name of the sentence transformer model.",
    )
    sentence_transformer: InstanceOf[SentenceTransformer] = Field(
        ...,
        description="The sentence transformer model to use for generating embeddings.",
    )
    max_input_length: int | None = Field(
        default=None,
        description="Maximum input length for the model (in Unicode code points).",
        gt=0,
    )


class SentenceTransformerEmbedder(Embedder):
    """Embedder powered by a sentence transformer model."""

    def __init__(self, params: SentenceTransformerEmbedderParams) -> None:
        """Initialize the sentence transformer embedder."""
        super().__init__()

        self._model_name = params.model_name
        self._sentence_transformer = params.sentence_transformer

        self._dimensions = (
            self._sentence_transformer.get_sentence_embedding_dimension()
            or len(self._sentence_transformer.encode(""))
        )
        match self._sentence_transformer.similarity_fn_name:
            case "cosine":
                self._similarity_metric = SimilarityMetric.COSINE
            case "dot":
                self._similarity_metric = SimilarityMetric.DOT
            case "euclidean":
                self._similarity_metric = SimilarityMetric.EUCLIDEAN
            case "manhattan":
                self._similarity_metric = SimilarityMetric.MANHATTAN
            case _:
                logger.warning(
                    "Unknown similarity function name '%s', defaulting to cosine",
                    self._sentence_transformer.similarity_fn_name,
                )
                self._similarity_metric = SimilarityMetric.COSINE

        self._max_input_length = params.max_input_length

    async def ingest_embed(
        self,
        inputs: list[Any],
        max_attempts: int = 1,
    ) -> list[list[float]]:
        """Embed input documents using the sentence transformer."""
        return await self._embed(inputs, max_attempts)

    async def search_embed(
        self,
        queries: list[Any],
        max_attempts: int = 1,
    ) -> list[list[float]]:
        """Embed search queries using the sentence transformer."""
        return await self._embed(queries, max_attempts, prompt_name="query")

    async def _embed(
        self,
        inputs: list[Any],
        max_attempts: int = 1,
        prompt_name: str | None = None,
    ) -> list[list[float]]:
        """Generate embeddings with retry logic."""
        if not inputs:
            return []
        if max_attempts <= 0:
            raise ValueError("max_attempts must be a positive integer")

        inputs_chunks = [
            chunk_text_balanced(input_text, self._max_input_length)
            if self._max_input_length is not None and input_text
            else [input_text]
            for input_text in inputs
        ]

        chunks = [chunk for input_chunks in inputs_chunks for chunk in input_chunks]

        embed_call_uuid = uuid4()

        start_time = time.monotonic()

        try:
            logger.debug(
                "[call uuid: %s] "
                "Attempting to create embeddings using %s sentence transformer model",
                embed_call_uuid,
                self._model_name,
            )
            response = await asyncio.to_thread(
                self._sentence_transformer.encode,
                chunks,
                prompt_name=prompt_name,
                show_progress_bar=False,
            )
        except Exception as e:
            # Exception may not be retried.
            error_message = (
                f"[call uuid: {embed_call_uuid}] "
                "Giving up creating embeddings "
                f"due to assumed non-retryable {type(e).__name__}"
            )
            logger.exception(error_message)
            raise ExternalServiceAPIError(error_message) from e

        end_time = time.monotonic()
        logger.debug(
            "[call uuid: %s] Embeddings created in %.3f seconds",
            embed_call_uuid,
            end_time - start_time,
        )

        chunk_embeddings = np.asarray(response, dtype=float).tolist()
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
        """Return the underlying model identifier."""
        return self._model_name

    @property
    def dimensions(self) -> int:
        """Return the embedding dimensionality."""
        return self._dimensions

    @property
    def similarity_metric(self) -> SimilarityMetric:
        """Return the similarity metric used."""
        return self._similarity_metric
