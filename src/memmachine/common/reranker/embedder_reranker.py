"""
Embedder-based reranker implementation.
"""

import numpy as np
from pydantic import BaseModel, Field, InstanceOf

from memmachine.common.embedder import Embedder, SimilarityMetric

from .reranker import Reranker


class EmbedderRerankerParams(BaseModel):
    """
    Parameters for EmbedderReranker.

    Attributes:
        embedder (Embedder):
            Embedder instance.
    """

    embedder: InstanceOf[Embedder] = Field(
        ..., description="An instance of an Embedder to use for generating embeddings"
    )


class EmbedderReranker(Reranker):
    """
    Reranker that uses an embedder
    to score relevance of candidates to a query.
    """

    def __init__(self, params: EmbedderRerankerParams):
        """
        Initialize an EmbedderReranker with the provided configuration.

        Args:
            params (EmbedderRerankerParams):
                Parameters for the EmbedderReranker.
        """
        super().__init__()

        self._embedder = params.embedder

    async def score(self, query: str, candidates: list[str]) -> list[float]:
        if len(candidates) == 0:
            return []

        query_embedding = np.array(await self._embedder.search_embed([query])).flatten()
        candidate_embeddings = np.array(await self._embedder.ingest_embed(candidates))

        match self._embedder.similarity_metric:
            case SimilarityMetric.COSINE:
                magnitude_products = np.linalg.norm(
                    candidate_embeddings, axis=-1
                ) * np.linalg.norm(query_embedding)
                magnitude_products[magnitude_products == 0] = float("inf")

                scores = (
                    np.dot(candidate_embeddings, query_embedding) / magnitude_products
                )
            case SimilarityMetric.DOT:
                scores = np.dot(candidate_embeddings, query_embedding)
            case SimilarityMetric.EUCLIDEAN:
                scores = -np.linalg.norm(
                    candidate_embeddings - query_embedding, axis=-1
                )
            case SimilarityMetric.MANHATTAN:
                scores = -np.sum(
                    np.abs(candidate_embeddings - query_embedding), axis=-1
                )
            case _:
                # Default to cosine similarity.
                magnitude_products = np.linalg.norm(
                    candidate_embeddings, axis=-1
                ) * np.linalg.norm(query_embedding)
                magnitude_products[magnitude_products == 0] = float("inf")

                scores = (
                    np.dot(candidate_embeddings, query_embedding) / magnitude_products
                )

        return scores.astype(float).tolist()
