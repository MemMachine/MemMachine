"""
Abstract base class for an embedder.
"""

from abc import ABC, abstractmethod
from typing import Any


class Embedder(ABC):
    """
    Abstract base class for an embedder.
    """

    @abstractmethod
    async def ingest_embed(
        self,
        inputs: list[Any],
        retry_limit: int = 1,
    ) -> list[list[float]]:
        """
        Generate embeddings for the provided inputs.

        Args:
            inputs (list[Any]):
                A list of inputs to be embedded.
            retry_limit (int):
                The maximum number of retries to attempt.


        Returns:
            list[list[float]]:
                A list of embedding vectors corresponding to each input.
        """
        raise NotImplementedError

    @abstractmethod
    async def search_embed(self, queries: list[Any]) -> list[list[float]]:
        """
        Generate embeddings for the provided queries.

        Args:
            queries (list[Any]):
                A list of queries to be embedded.

        Returns:
            list[list[float]]:
                A list of embedding vectors corresponding to each query.
        """
        raise NotImplementedError
