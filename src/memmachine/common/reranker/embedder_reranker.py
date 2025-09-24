"""
Embedder-based reranker implementation.
"""

from typing import Any

import numpy as np

from memmachine.common.embedder import Embedder

from .reranker import Reranker


class EmbedderReranker(Reranker):
    """
    Reranker that uses an embedder and cosine similarity
    to score relevance of candidates to a query.
    """

    def __init__(self, config: dict[str, Any]):
        """
        Initialize an EmbedderReranker with the provided configuration.

        Args:
            config (dict[str, Any]):
                Configuration dictionary containing:
                - embedder (Embedder):
                    An instance of an Embedder
                    to use for generating embeddings.

        Raises:
            ValueError: If embedder is not provided.
            TypeError: If embedder is not an instance of Embedder.
        """
        super().__init__()

        embedder = config.get("embedder")
        if embedder is None:
            raise ValueError("Embedder must be provided")
        if not isinstance(embedder, Embedder):
            raise TypeError("Embedder must be an instance of Embedder")

        self._embedder = embedder

    async def score(self, query: str, candidates: list[str]) -> list[float]:
        if len(candidates) == 0:
            return []

        query_embedding = np.array(
            await self._embedder.search_embed([query])
        ).flatten()
        candidate_embeddings = np.array(
            await self._embedder.ingest_embed(candidates)
        )

        magnitude_products = np.linalg.norm(
            candidate_embeddings, axis=-1
        ) * np.linalg.norm(query_embedding)

        magnitude_products[magnitude_products == 0] = float("inf")
        scores = (
            np.dot(candidate_embeddings, query_embedding) / magnitude_products
        )
        return scores.astype(float).tolist()
