"""Abstract base class for a vector store."""

from abc import ABC, abstractmethod
from collections.abc import Iterable, Sequence
from uuid import UUID

from pydantic import BaseModel, Field, field_validator

from memmachine_server.common.filter.filter_parser import (
    FilterExpr,
)

from .data_types import (
    QueryResult,
    Record,
    VectorStoreSchema,
    validate_vector_store_name,
)


class VectorStorePartition(ABC):
    """
    One partition of a vector store, bound to its key.

    All data operations are scoped to the partition: a record's UUID names
    it in this partition only, and the same UUID in another partition names
    another record. The handle owns nothing: a caller builds one with
    `VectorStore.open_partition`, drops it, and builds another at will.

    A partition created again under a deleted one's key starts empty. An
    operation through a handle begun after its partition was deleted raises
    VectorStorePartitionHandleStaleError, whether or not a partition was
    created again under the key since.

    An `upsert` or `delete` is durable once it returns; queries may not
    reflect it right away. A store that guarantees more states it.

    A partition stores the properties its store declares
    (`store_schema.indexed_properties`), typed and indexed for filtering
    during a search, and no others: a record or a filter naming an
    undeclared key is rejected, so an undeclared key never exists in the
    store, neither stored write-only nor scanned for. Every store evaluates
    the whole filter language during a search.
    """

    @property
    @abstractmethod
    def store_schema(self) -> VectorStoreSchema:
        """The schema every record and query of this partition follows."""
        raise NotImplementedError

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


class VectorStore(ABC):
    """
    Abstract base class for a vector store.

    A store is constructed with a name and a schema: a dimensionality, a
    similarity metric, and declared properties. Every record it holds has a
    vector of that dimensionality, and any declared property it carries has
    its declared type. Stores of different names hold separate partitions.
    The schema is fixed for the life of the store's data: changing it is a
    migration.

    A partition is identified by its key. A partition deleted and created
    again under the same key starts empty, and reclaiming the deleted one's
    storage leaves it untouched. Each store states which processes may share
    its partitions.

    Naming constraints:
        - Vector store names and property keys must match `[a-z0-9_]+`
          (lowercase alphanumeric and underscores only) and be at most 32
          bytes. Every such name works on every backend.
        - Partition keys are non-empty strings of at most 255 bytes, free of
          NUL characters. Every such key works on every backend.
    """

    @abstractmethod
    async def startup(self) -> None:
        """Start the store, creating idempotently what must exist before a partition is created."""
        raise NotImplementedError

    @abstractmethod
    async def shutdown(self) -> None:
        """Shutdown."""
        raise NotImplementedError

    @abstractmethod
    async def create_partition(self, partition_key: str) -> None:
        """
        Create a partition.

        Args:
            partition_key (str):
                The key of the partition.

        Raises:
            VectorStoreAttemptsExhaustedError: If the store gave up creating the
                partition after repeated attempts that made no progress.
            VectorStorePartitionAlreadyExistsError: If the partition already exists,
                or is being created.
            VectorStorePartitionDeletedError: If the partition was deleted
                before its creation completed.
        """
        raise NotImplementedError

    @abstractmethod
    async def open_or_create_partition(
        self, partition_key: str
    ) -> VectorStorePartition:
        """
        Get a handle for the partition, creating the partition if it does not exist.

        Args:
            partition_key (str):
                The key of the partition.

        Returns:
            VectorStorePartition:
                A handle bound to the partition's current life.

        Raises:
            VectorStoreAttemptsExhaustedError: If the store gave up opening or
                creating the partition after repeated attempts that made no
                progress.
            VectorStorePartitionPendingError: If the partition's creation, by
                another caller, did not complete within the store's attempts
                to open it.
        """
        raise NotImplementedError

    @abstractmethod
    async def open_partition(self, partition_key: str) -> VectorStorePartition | None:
        """
        Get a handle for an existing partition.

        Args:
            partition_key (str):
                The key of the partition.

        Returns:
            VectorStorePartition | None:
                A handle bound to the partition, or None if the partition
                does not exist.

        Raises:
            VectorStorePartitionPendingError: If the partition's creation
                has not completed.
        """
        raise NotImplementedError

    @abstractmethod
    async def delete_partition(self, partition_key: str) -> None:
        """
        Delete a partition and all of its records.

        When this returns, the partition is unreachable. A store that
        reclaims storage later leaves the partition's data for
        `purge_deleted_partitions`. Idempotent.

        Args:
            partition_key (str):
                The key of the partition to delete.
        """
        raise NotImplementedError

    @abstractmethod
    async def purge_deleted_partitions(self) -> bool:
        """
        Reclaim some of the storage of deleted partitions.

        Each call does a bounded amount of work. Call it until it returns
        False, and again from time to time: a deleted partition's storage
        may become reclaimable some time after the deletion. A store that
        reclaims storage in `delete_partition` returns False. Safe to call
        from several processes at once.

        Returns:
            bool:
                Whether the call ran a round; False when nothing was due.
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
