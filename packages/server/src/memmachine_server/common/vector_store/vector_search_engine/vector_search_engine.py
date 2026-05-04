"""Abstract base class for a vector search engine."""

from abc import ABC, abstractmethod
from collections.abc import Container, Iterable, Mapping, Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class SearchMatch:
    """
    A single search match.

    Attributes:
        score (float):
            The meaning depends on the collection's `SimilarityMetric`:
            - *cosine*: cosine similarity in [-1, 1].
            - *dot*: raw dot product [0, inf).
            - *euclidean*: Euclidean distance [0, inf).
            - *manhattan*: Manhattan distance [0, inf).
        key (int): Engine key for the matched vector.
    """

    score: float
    key: int


@dataclass(frozen=True)
class SearchResult:
    """
    Result of a nearest-neighbor search for a single query vector.

    Matches are ordered from best to worst.
    """

    matches: list[SearchMatch]


class VectorSearchEngine(ABC):
    """
    A vector search engine that indexes vectors by integer key.

    Provides nearest-neighbor search (exact or approximate) over vectors
    identified by caller-provided integer keys.

    Results are returned ordered from best to worst.

    Safe for concurrent use from async tasks (single event loop).
    Not safe across threads or processes.
    """

    @abstractmethod
    async def add(self, vectors: Mapping[int, Sequence[float]]) -> None:
        """
        Add vectors.

        Keys must not already exist. The caller is responsible for
        removing existing keys before adding. Behavior on duplicate
        keys is undefined.

        Args:
            vectors (Mapping[int, Sequence[float]]):
                Mapping of integer keys to vectors.
        """

    @abstractmethod
    async def search(
        self,
        vectors: Iterable[Sequence[float]],
        *,
        limit: int,
        allowed_keys: Container[int] | None = None,
    ) -> list[SearchResult]:
        """
        Search for vectors similar to the query vectors.

        Results may be approximate depending on the engine implementation.

        Args:
            vectors (Iterable[Sequence[float]]):
                Query vectors.
            limit (int):
                Maximum number of results per query.
            allowed_keys (Container[int] | None):
                If provided, only return results whose keys
                are in this container. The container's ``__contains__``
                is called synchronously per candidate during search
                (default: None).

        Returns:
            list[SearchResult]:
                Results for each query vector,
                ordered as in the input iterable.
        """

    @abstractmethod
    async def get_vectors(self, keys: Iterable[int]) -> dict[int, list[float]]:
        """
        Retrieve vectors by key.

        Args:
            keys (Iterable[int]):
                Keys of vectors to retrieve.

        Returns:
            dict[int, list[float]]:
                Mapping of key to vector for keys that exist.
                Missing keys are omitted.
        """

    @abstractmethod
    async def remove(self, keys: Iterable[int]) -> None:
        """
        Remove vectors by key.

        Missing keys are silently ignored.

        Args:
            keys (Iterable[int]):
                Keys of vectors to remove.
        """

    @abstractmethod
    async def save(self, path: str) -> None:
        """
        Persist the index to disk.

        Args:
            path (str):
                File path to write the index to.
        """

    @abstractmethod
    async def load(self, path: str) -> None:
        """
        Load the index from disk.

        Args:
            path (str):
                File path to read the index from.
        """
