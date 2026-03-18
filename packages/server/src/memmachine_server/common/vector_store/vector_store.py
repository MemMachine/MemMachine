"""
Abstract base class for a vector store.

Defines the interface for adding, querying, and deleting records.
"""

from abc import ABC, abstractmethod
from collections.abc import Iterable, Mapping, Sequence
from uuid import UUID

from memmachine_server.common.data_types import PropertyValue, SimilarityMetric
from memmachine_server.common.filter.filter_parser import (
    FilterExpr,
)

from .data_types import QueryResult, Record


class Collection(ABC):
    """Collection in a vector store."""

    @abstractmethod
    async def upsert(
        self,
        *,
        records: Iterable[Record],
    ) -> None:
        """
        Upsert records in the collection.

        Args:
            records (Iterable[Record]):
                Iterable of records to upsert.

        """
        raise NotImplementedError

    @abstractmethod
    async def query(
        self,
        *,
        query_vectors: Iterable[Sequence[float]],
        score_threshold: float | None = None,
        limit: int | None = None,
        property_filter: FilterExpr | None = None,
        return_vector: bool = True,
        return_properties: bool = True,
    ) -> Iterable[Iterable[QueryResult]]:
        """
        Query for records matching the criteria by query vectors.

        Args:
            query_vectors (Iterable[Sequence[float]]):
                The vectors to compare against.
            score_threshold (float | None):
                Score threshold to consider a match
                (default: None).
            limit (int | None):
                Maximum number of matching records to return per query vector.
                If None, return as many matching records as possible
                (default: None).
            property_filter (FilterExpr | None):
                Filter expression tree.
                If None or empty, no property filtering is applied
                (default: None).
            return_vector (bool):
                Whether to include the vector in the returned records
                (default: False).
            return_properties (bool):
                Whether to include the properties in the returned records
                (default: True).

        Returns:
            Iterable[Iterable[QueryResult]]:
                Iterables of results matching the criteria for each query vector,
                each ordered by score from best to worst.

        """
        raise NotImplementedError

    @abstractmethod
    async def get(
        self,
        *,
        record_uuids: Iterable[UUID],
        return_vector: bool = True,
        return_properties: bool = True,
    ) -> Iterable[Record]:
        """
        Get records from the collection by their UUIDs.

        Args:
            record_uuids (Iterable[UUID]):
                Iterable of UUIDs of the records to retrieve.
            return_vector (bool):
                Whether to include the vector in the returned records
                (default: False).
            return_properties (bool):
                Whether to include the properties in the returned records
                (default: True).

        Returns:
            Iterable[Record]:
                Iterable of records with the specified UUIDs,
                ordered as in the input iterable.

        """
        raise NotImplementedError

    @abstractmethod
    async def delete(
        self,
        *,
        record_uuids: Iterable[UUID],
    ) -> None:
        """
        Delete records from the collection by their UUIDs.

        Args:
            record_uuids (Iterable[UUID]):
                Iterable of UUIDs of the records to delete.

        """
        raise NotImplementedError


class VectorStore(ABC):
    """Abstract base class for a vector store."""

    @abstractmethod
    async def startup(self) -> None:
        """Startup."""
        raise NotImplementedError

    @abstractmethod
    async def shutdown(self) -> None:
        """Shutdown."""
        raise NotImplementedError

    @abstractmethod
    async def create_collection(
        self,
        collection_name: str,
        *,
        vector_dimensions: int,
        similarity_metric: SimilarityMetric = SimilarityMetric.COSINE,
        properties_schema: Mapping[str, type[PropertyValue]] | None = None,
    ) -> None:
        """
        Create a collection in the vector store.

        Args:
            collection_name (str):
                Name of the collection to create.
            vector_dimensions (int):
                Number of dimensions for the vectors.
            similarity_metric (SimilarityMetric):
                Similarity metric to use for vector comparisons
                (default: SimilarityMetric.COSINE).
            properties_schema (Mapping[str, type] | None):
                Mapping of property names to their types
                (default: None).

        """
        raise NotImplementedError

    @abstractmethod
    async def get_collection(self, collection_name: str) -> Collection:
        """
        Get a collection from the vector store.

        Args:
            collection_name (str):
                Name of the collection to get.

        Returns:
            Collection:
                The requested collection.

        """
        raise NotImplementedError

    @abstractmethod
    async def delete_collection(self, collection_name: str) -> None:
        """
        Delete a collection from the vector store.

        Args:
            collection_name (str):
                Name of the collection to delete.

        """
        raise NotImplementedError
