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
from memmachine_server.common.filter import FilterExpr

from .data_types import QueryResult, Record


class VectorStorePartition(ABC):
    """
    One partition of a vector store, bound to its key.

    All data operations are scoped to the partition. The handle owns
    nothing: a caller builds one with `VectorStore.get_partition`, drops it,
    and builds another at will.

    A handle is bound to one life of the partition: after the partition is
    deleted, its operations raise VectorStorePartitionHandleStaleError, and
    a partition created again under the same key is a new life. A read
    concurrent with the deletion may take effect before it, returning the
    content from before the deletion. A store that cannot detect a stale
    handle says so in its own contract.

    A partition stores the properties its store declares
    (`indexed_properties`), typed and indexed for filtering during a
    search, and no others: a record or a filter naming an undeclared key is
    rejected, so an undeclared key never exists in the store, neither
    stored write-only nor scanned for. Neither a vector nor a property is
    read back out: a query answers with UUIDs and cosine similarities.
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

        Answers with UUIDs and scores. Stored properties are filterable but
        never returned: this store is not the authority for a record's
        content, and its copy is only as fresh as the last write to it -- a
        caller that needs a record's fields reads them from whatever owns
        them.

        Args:
            query_vectors (Iterable[Sequence[float]]):
                The vectors to compare against.
            limit (int):
                Maximum number of matching records to return per query vector.
                A filtered search returns fewer when the filter admits fewer.
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

    A partition is identified to callers by its key and inside the store
    by an incarnation minted per life of the key, so nothing written under
    one life of a key is ever seen by, or reclaimed out from under,
    another. Which processes may share a store's partitions is the store's
    own contract, stated on the store.

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
    async def provision(self) -> None:
        """
        Create the collection's durable resources, idempotently.

        The native collection and its payload indexes, the registry
        tables, or the tables and indexes beside the data; whatever must
        exist before a partition can be created. Run once per deployment
        change by whoever owns the schema, before `startup`; never by a
        request.
        """
        raise NotImplementedError

    @abstractmethod
    async def startup(self) -> None:
        """Startup."""
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
            VectorStorePartitionAlreadyExistsError: If the partition already exists.
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
        """
        raise NotImplementedError

    @abstractmethod
    async def close_partition(self, *, partition: VectorStorePartition) -> None:
        """
        Close a partition handle.

        Args:
            partition (VectorStorePartition):
                The handle to close.
        """
        raise NotImplementedError

    @abstractmethod
    async def delete_partition(self, partition_key: str) -> None:
        """
        Delete a partition and all of its records.

        When this returns, the partition is unreachable and its data is
        deleted or, on a store that reclaims it later, left for
        `purge_deleted_partitions`. Idempotent.

        Args:
            partition_key (str):
                The key of the partition to delete.
        """
        raise NotImplementedError

    @abstractmethod
    async def purge_deleted_partitions(self) -> bool:
        """
        Reclaim, bounded, some of the storage of deleted partitions.

        A store whose deletion reclaims physically returns False. A store
        that defers reclamation does one bounded round per call and
        returns True when the round found records to reclaim, so the
        caller's protocol is "call until False"; a False may still leave
        tombstones that come due later. Safe to repeat, and safe from
        several processes at once. The store never schedules this itself.

        Returns:
            bool:
                Whether the round reclaimed records.
        """
        raise NotImplementedError
