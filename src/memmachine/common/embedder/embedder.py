"""Abstract base class for an embedder."""

import asyncio
from abc import ABC, abstractmethod
from typing import Any

from memmachine.common.data_types import SimilarityMetric


class Embedder(ABC):
    """Abstract base class for an embedder."""

    def __init__(self, batch_size: int | None = None) -> None:
        """Initialize the embedder with an optional batch size."""
        self.batch_size = batch_size

    async def ingest_embed(
        self,
        inputs: list[Any],
        max_attempts: int = 1,
    ) -> list[list[float]]:
        """Generate embeddings for ingestion, handling batching if configured."""
        if not inputs:
            return []

        if self.batch_size and len(inputs) > self.batch_size:
            tasks = [
                self._ingest_embed(inputs[i : i + self.batch_size], max_attempts)
                for i in range(0, len(inputs), self.batch_size)
            ]
            batch_results = await asyncio.gather(*tasks)
            results = []
            for batch in batch_results:
                results.extend(batch)
            return results

        return await self._ingest_embed(inputs, max_attempts)

    async def search_embed(
        self,
        queries: list[Any],
        max_attempts: int = 1,
    ) -> list[list[float]]:
        """Generate embeddings for search queries, handling batching if configured."""
        if not queries:
            return []

        if self.batch_size and len(queries) > self.batch_size:
            tasks = [
                self._search_embed(queries[i : i + self.batch_size], max_attempts)
                for i in range(0, len(queries), self.batch_size)
            ]
            batch_results = await asyncio.gather(*tasks)
            results = []
            for batch in batch_results:
                results.extend(batch)
            return results

        return await self._search_embed(queries, max_attempts)

    @abstractmethod
    async def _ingest_embed(
        self,
        inputs: list[Any],
        max_attempts: int = 1,
    ) -> list[list[float]]:
        """Actual implementation of embedding ingestion."""

    @abstractmethod
    async def _search_embed(
        self,
        queries: list[Any],
        max_attempts: int = 1,
    ) -> list[list[float]]:
        """Actual implementation of search embedding."""

    @property
    @abstractmethod
    def model_id(self) -> str:
        """Return the identifier for the embedding model."""
        raise NotImplementedError

    @property
    @abstractmethod
    def dimensions(self) -> int:
        """Return the embedding dimensionality."""
        raise NotImplementedError

    @property
    @abstractmethod
    def similarity_metric(self) -> SimilarityMetric:
        """Return the similarity metric used by this embedder."""
        raise NotImplementedError
