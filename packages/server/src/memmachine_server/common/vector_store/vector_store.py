"""
Abstract base class for a vector store.

A store is one collection: a body of records searched together, with one
dimensionality and one declared schema, named at construction. Within it, a partition holds one tenant's records, and
`VectorStorePartition` is the handle a data consumer holds for it.
"""

from abc import ABC, abstractmethod
from collections.abc import Iterable, Mapping, Sequence
from uuid import UUID

from memmachine_server.common.data_types import PropertyType
from memmachine_server.common.filter.filter_parser import (
    FilterExpr,
)

from .data_types import QueryResult, Record


class VectorStorePartition(ABC):
    """
    One partition of a vector store, bound to its key.

    All data operations are scoped to the partition: a record's UUID names
    it in this partition only, and the same UUID in another partition names
    another record. The handle owns nothing: a caller builds one with
    `VectorStore.get_partition`, drops it, and builds another at will.

    A partition created again under a deleted one's key starts empty.
    Operations through a handle whose partition was deleted may raise
    VectorStorePartitionHandleStaleError, or, on an implementation that
    cannot tell, act on a partition created again under the same key.

    An `upsert` or `delete` is durable once it returns; queries may not
    reflect it right away. A store that guarantees more states it.

    A partition stores the properties its store declares
    (`indexed_properties`), typed and indexed for filtering during a
    search, and no others: a record or a filter naming an undeclared key is
    rejected, so an undeclared key never exists in the store, neither
    stored write-only nor scanned for.
    """

    @property
    @abstractmethod
    def partition_key(self) -> str:
        """The key this handle is bound to."""
        raise NotImplementedError

    @property
    @abstractmethod
    def indexed_properties(self) -> Mapping[str, PropertyType]:
        """The declared schema: every key this partition indexes for filtering."""
        raise NotImplementedError

    @property
    @abstractmethod
    def supported_filter_nodes(self) -> frozenset[type]:
        """
        The filter node classes the backend evaluates during a search.

        A `query` whose filter uses any other node raises
        `UnsupportedFilterError`; a caller routes such a predicate to a store
        that evaluates it afterward.
        """
        raise NotImplementedError

    @abstractmethod
    async def upsert(
        self,
        *,
        records: Iterable[Record],
    ) -> None:
        """
        Upsert records in the partition.

        Insert records with new UUIDs,
        and update records with existing UUIDs.

        Args:
            records (Iterable[Record]):
                Iterable of records to upsert.

        Raises:
            UndeclaredPropertyKeyError:
                If a record carries a key the store has not declared;
                raised before anything is sent.
            PropertyTypeMismatchError:
                If a record's value is not of its key's declared type;
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
        min_cosine_similarity: float | None = None,
        property_filter: FilterExpr | None = None,
    ) -> list[QueryResult]:
        """
        Query for records matching the criteria by query vectors.

        Each match holds a record's UUID and cosine similarity.

        Args:
            query_vectors (Iterable[Sequence[float]]):
                The vectors to compare against.
            limit (int):
                Maximum number of matching records to return per query vector;
                positive. A filtered search returns fewer when the filter
                admits fewer.
            min_cosine_similarity (float | None):
                If provided, only return matches whose cosine similarity
                is greater than or equal to this value
                (default: None).
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
            ValueError:
                If the limit is not positive, a query vector does not have
                the store's dimensions or has a coordinate that is not
                finite, or the minimum cosine similarity is not finite.
            UndeclaredPropertyKeyError:
                If the filter names a key the store has not declared.
            UnsupportedFilterError:
                If the filter uses a node outside `supported_filter_nodes`.
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

    A store is one collection, named at construction with its vector
    dimensions and its declared schema; the composition root builds one
    store per collection it needs, and the name is what keeps two stores
    over one engine or one client apart. Every partition of the store
    shares the collection's dimensions and schema, and the schema is fixed
    for the life of the store's data: changing it is a migration. Every
    store scores by cosine similarity.

    A partition is identified by its key. A partition deleted and created
    again under the same key starts empty, and reclaiming the deleted one's
    storage leaves it untouched. Each store states which processes may share
    its partitions.

    Naming constraints:
        - Vector store names, partition keys and property keys must match
          `[a-z0-9_]+` (lowercase alphanumeric and underscores only) and be
          at most 32 bytes. Every such name works on every backend: a store
          whose backend restricts its native names further maps the name to
          one it accepts.
    """

    @property
    @abstractmethod
    def vector_store_name(self) -> str:
        """The name of this store, and of the one collection it is."""
        raise NotImplementedError

    @property
    @abstractmethod
    def vector_dimensions(self) -> int:
        """Dimensionality of every vector in the store."""
        raise NotImplementedError

    @property
    @abstractmethod
    def indexed_properties(self) -> Mapping[str, PropertyType]:
        """The declared schema every partition of this store carries."""
        raise NotImplementedError

    @abstractmethod
    @abstractmethod
    async def startup(self) -> None:
        """
        Start the store, creating its durable resources idempotently.

        The native collection and its payload indexes, the registry
        tables, or the tables and indexes beside the data: whatever must
        exist before a partition can be created.
        """
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
            VectorStorePartitionAlreadyExistsError: If the partition already exists,
                or is being created.
            VectorStorePartitionDeletedError: If the partition was deleted
                before its creation completed.
            VectorStoreAttemptsExhaustedError: If the store gave up creating the
                partition after repeated attempts that made no progress.
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
            VectorStorePartitionSchemaMismatchError:
                If the partition exists and was created under other
                dimensions or another declared schema than this store's.
            VectorStorePartitionPendingError: If the partition's creation, by
                another caller, did not complete within the store's attempts
                to open it.
            VectorStoreAttemptsExhaustedError: If the store gave up opening or
                creating the partition after repeated attempts that made no
                progress.
        """
        raise NotImplementedError

    @abstractmethod
    async def get_partition(self, partition_key: str) -> VectorStorePartition | None:
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
            VectorStorePartitionSchemaMismatchError:
                If the partition was created under other dimensions or
                another declared schema than this store's.
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
