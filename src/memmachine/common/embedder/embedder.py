"""
Abstract base class for an embedder.
"""

from abc import ABC, abstractmethod
from typing import Any

from .data_types import SimilarityMetric


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
                The maximum number of attempts to make before giving up
                (default: 1).


        Returns:
            list[list[float]]:
                A list of embedding vectors corresponding to each input.

        Raises:
            ExternalServiceAPIError:
                Errors from the underlying embedding API.
            ValueError:
                Invalid input or max_attempts.
            RuntimeError:
                Catch-all for any other errors.
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
                The maximum number of attempts to make before giving up
                (default: 1).

        Returns:
            list[list[float]]:
                A list of embedding vectors corresponding to each query.

        Raises:
            ExternalServiceAPIError:
                Errors from the underlying embedding API.
            ValueError:
                Invalid input or max_attempts.
            RuntimeError:
                Catch-all for any other errors.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def model_id(self) -> str:
        """
        Get an identifier for the embedding model.
        Identifier-dimensionality pairs must be unique.

        Returns:
            str: The model identifier.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def dimensions(self) -> int:
        """
        Get the dimensionality for embeddings
        produced by this embedder.

        Returns:
            int: The dimensionality.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def similarity_metric(self) -> SimilarityMetric:
        """
        Get the similarity metric for embeddings
        produced by this embedder.

        Returns:
            SimilarityMetric: The similarity metric.
        """
        raise NotImplementedError
