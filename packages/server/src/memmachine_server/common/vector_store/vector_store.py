"""Abstract base class for a vector store."""

from abc import ABC, abstractmethod
from collections.abc import Iterable, Sequence
from uuid import UUID

from pydantic import BaseModel, Field, field_validator

from memmachine_server.common.filter.filter_parser import (
    FilterExpr,
)
from memmachine_server.common.partition_registry import PartitionedStore

from .data_types import (
    QueryResult,
    Record,
    VectorStoreSchema,
    validate_vector_store_name,
)


class VectorStorePartition(ABC):
    """
    A handle on one partition's records in a vector store.

    All data operations are scoped to the partition: a record's UUID names it
    in this partition only, and the same UUID in another partition names
    another record. The handle owns nothing: a caller builds one with
    `VectorStore.get_partition`, drops it, and builds another at will. The
    handle checks its inputs, and nothing about whether the partition is
    still live: its owner fences each operation by the partition's
    registration.

    An `upsert` or `delete` is durable once it returns; queries may not
    reflect it right away. A store that guarantees more states it.

    A partition stores the properties its store's schema declares, typed
    and indexed for filtering during a search, and no others: a record or a
    filter naming an undeclared key is rejected, so an undeclared key never
    exists in the store, neither stored write-only nor scanned for. Every
    store evaluates the whole filter language during a search.
    """

    @abstractmethod
    async def upsert(
        self,
        *,
        records: Iterable[Record],
    ) -> None:
        """
        Upsert records in the partition.

        Insert records with new UUIDs, and replace records with existing UUIDs:
        a replaced record holds only the vector and properties it is upserted
        with.

        Args:
            records (Iterable[Record]):
                Iterable of records to upsert.

        Raises:
            PropertyTypeMismatchError:
                If a record's value is not of its key's declared type;
                raised before anything is sent.
            UndeclaredPropertyKeyError:
                If a record carries a key the store has not declared;
                raised before anything is sent.
            ValueError:
                If a record's vector does not have the store's dimensions.
        """
        raise NotImplementedError

    @abstractmethod
    async def query(
        self,
        *,
        query_vectors: Iterable[Sequence[float]],
        limit: int,
        score_threshold: float | None = None,
        property_filter: FilterExpr | None = None,
    ) -> list[QueryResult]:
        """
        Query for records matching the criteria by query vectors.

        Each match holds a record's UUID and score.

        Args:
            query_vectors (Iterable[Sequence[float]]):
                The vectors to compare against.
            limit (int):
                Maximum number of matching records to return per query vector;
                positive. A filtered search returns fewer when the filter
                admits fewer.
            score_threshold (float | None):
                The worst score a match may have, by the store's
                similarity metric; a match scoring exactly the threshold is
                returned (default: None).
            property_filter (FilterExpr | None):
                Filter expression tree over declared keys, evaluated during
                the search.
                If None, no property filtering is applied
                (default: None).

        Returns:
            list[QueryResult]:
                Results for each query vector,
                ordered as in the input iterable.

        Raises:
            UndeclaredPropertyKeyError:
                If the filter names a key the store has not declared.
            ValueError:
                If the limit is not positive, a query vector does not have
                the store's dimensions or has a coordinate that is not
                finite, or the score threshold is not finite.
        """
        raise NotImplementedError

    @abstractmethod
    async def delete(
        self,
        *,
        record_uuids: Iterable[UUID],
    ) -> None:
        """
        Delete records from the partition by their UUIDs.

        Args:
            record_uuids (Iterable[UUID]):
                Iterable of UUIDs of the records to delete.
        """
        raise NotImplementedError


class VectorStore(PartitionedStore[VectorStorePartition]):
    """
    Abstract base class for a vector store.

    A store is constructed with a name and a schema: a dimensionality, a
    similarity metric, and declared properties. Every record it holds has a
    vector of that dimensionality, and any declared property it carries has
    its declared type. Stores of different names hold separate records.
    The schema is fixed for the life of the store's data: changing it is a
    migration.

    The store holds each partition's records under the partition's UUID. Its
    `PartitionedStore` methods create, get, purge, and drop partitions, which
    its owner's partition registry drives.

    Naming constraints:
        - Vector store names and property keys must match `[a-z0-9_]+`
          (lowercase alphanumeric and underscores only) and be at most 32
          bytes. Every such name works on every backend.
    """

    @property
    @abstractmethod
    def store_schema(self) -> VectorStoreSchema:
        """The schema the store was built with, which every partition's records and queries follow."""
        raise NotImplementedError

    @abstractmethod
    async def startup(self) -> None:
        """Start the store, creating idempotently what must exist before a partition is created."""
        raise NotImplementedError

    @abstractmethod
    async def shutdown(self) -> None:
        """Shutdown."""
        raise NotImplementedError

    @abstractmethod
    async def get_partition(self, partition_uuid: UUID) -> VectorStorePartition:
        """
        Get a handle on a created partition.

        A vector store answers a handle for any UUID.

        Args:
            partition_uuid (UUID):
                The partition's UUID.

        Returns:
            VectorStorePartition:
                A handle bound to the partition.
        """
        raise NotImplementedError


class VectorStoreParams(BaseModel):
    """
    The parameters every vector store is constructed with.

    Attributes:
        vector_store_name (str):
            The name of this store.
        store_schema (VectorStoreSchema):
            The shape of this store's records and the metric its queries score by.
    """

    vector_store_name: str = Field(..., description="The name of this store")
    store_schema: VectorStoreSchema = Field(
        ...,
        description=(
            "The shape of this store's records and the metric its queries score by"
        ),
    )

    @field_validator("vector_store_name")
    @classmethod
    def _validate_vector_store_name(cls, vector_store_name: str) -> str:
        validate_vector_store_name(vector_store_name)
        return vector_store_name
