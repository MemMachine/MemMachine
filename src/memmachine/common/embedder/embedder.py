"""Abstract base class for an embedder."""

from abc import ABC, abstractmethod
from typing import Any

from memmachine.common.data_types import SimilarityMetric


class Embedder(ABC):
    """Abstract base class for an embedder."""

    @abstractmethod
    async def ingest_embed(
        self,
        inputs: list[Any],
        max_attempts: int = 1,
    ) -> list[list[float]]:
        """Generate embeddings for the provided inputs."""
        raise NotImplementedError

    @abstractmethod
    async def search_embed(
        self,
        queries: list[Any],
        max_attempts: int = 1,
    ) -> list[list[float]]:
        """Generate embeddings for the provided queries."""
        raise NotImplementedError

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
