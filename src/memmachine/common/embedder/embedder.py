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
        max_attempts: int = 1,
    ) -> list[list[float]]:
        """
        Generate embeddings for the provided inputs.

        Args:
            inputs (list[Any]):
                A list of inputs to be embedded.
            max_attempts (int):
                The maximum number of attempts to make before giving up.
                Defaults to 1.


        Returns:
            list[list[float]]:
                A list of embedding vectors corresponding to each input.
        """
        raise NotImplementedError

    @abstractmethod
    async def search_embed(
        self,
        queries: list[Any],
        max_attempts: int = 1,
    ) -> list[list[float]]:
        """
        Generate embeddings for the provided queries.

        Args:
            queries (list[Any]):
                A list of queries to be embedded.
            max_attempts (int):
                The maximum number of attempts to make before giving up.
                Defaults to 1.

        Returns:
            list[list[float]]:
                A list of embedding vectors corresponding to each query.

        Raises:
            IOError:
                If IO error happens. The IO errors can include: Netowrk Error,
                Rate Litmit, Timeout, etc.
            ValueError:
                Any other errors except the IOError.
        """
        raise NotImplementedError
