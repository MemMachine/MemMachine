"""
Abstract base class for a vector store.

A store is one collection: a body of records searched together, with one
dimensionality and one declared schema, named at construction. Within it,
a partition holds one tenant's records, and `VectorStorePartition` is the
handle a data consumer holds for it.
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
    One partition of a vector store, bound to one incarnation of its key.

    All data operations are scoped to the partition. The handle owns
    nothing: a caller builds one with `VectorStore.get_partition`, drops it,
    and builds another at will. It is bound to the incarnation the store
    minted when the partition was created: once the partition is deleted,
    or deleted and re-created under the same key, every operation of a
    handle bound to the old incarnation raises
    `VectorStorePartitionHandleStaleError`, and none of them can reach the
    successor's records.

    A partition stores every property of a record and filters on any key;
    the keys its store declares (`indexed_properties`) are indexed for
    filtering during a search, and their values are typed.
    """

    @property
    @abstractmethod
    def partition_key(self) -> str:
        """The key this handle is bound to."""
        raise NotImplementedError

    @property
    @abstractmethod
    def indexed_properties(self) -> Mapping[str, PropertyType]:
        """The declared schema: every key this partition stores and filters on."""
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
                Records containing properties
                not in the declared schema
                are allowed.
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
            min_cosine_similarity (float | None):
                If provided, only return matches whose cosine similarity
                is greater than or equal to this value
                (default: None).
            property_filter (FilterExpr | None):
                Filter expression tree.
                If None, no property filtering is applied
                (default: None).

        Returns:
            list[QueryResult]:
                Results for each query vector,
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
    for the life of the store's data: changing it is a migration.

    A partition is identified to callers by a string key and inside the
    store by an incarnation the store mints per partition life: records,
    points and index files are keyed by the incarnation, never by the key,
    so nothing written under one life of a key is ever seen by, or reclaimed
    out from under, another. Deletion is a registry write that makes the
    partition unreachable at once; its storage is reclaimed afterward by
    `purge_deleted_partitions`, which the deployment runs.

    A given partition must be managed by at most one process at a time.
    The consumer is responsible for sharding partition keys across
    processes.

    Naming constraints:
        - Collection names must match `[a-z0-9_]+` and be at most 64 bytes.
        - Partition keys and property keys must match `[a-z0-9_]+`
          (lowercase alphanumeric and underscores only) and be at most
          32 bytes.
    """

    @property
    @abstractmethod
    def collection(self) -> str:
        """The collection this store is."""
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

        The native collection and its payload indexes, or the tables and
        indexes beside the data; whatever must exist before a partition can
        be created. Run once per deployment change by whoever owns the
        schema, before `startup`; never by a request.
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
            VectorStoreAttemptsExhaustedError:
                If creation exhausted its internal attempts on a failure
                that should not recur; an immediate retry is unlikely to
                succeed -- diagnose the chained cause.
        """
        raise NotImplementedError

    @abstractmethod
    async def get_partition(self, partition_key: str) -> VectorStorePartition | None:
        """
        Get a handle for an existing partition.

        Args:
            partition_key (str):
                The key of the partition.

        Staleness is a property of a handle already held, raised by its
        operations, never of this lookup.

        Returns:
            VectorStorePartition | None:
                A handle bound to the partition's current incarnation, or
                None if the partition does not exist.

        Raises:
            VectorStorePartitionSchemaMismatchError:
                If the partition was created under other dimensions or
                another declared schema than this store's.
        """
        raise NotImplementedError

    @abstractmethod
    async def delete_partition(self, partition_key: str) -> None:
        """
        Delete a partition and all of its records.

        The partition becomes unreachable immediately: `get_partition`
        returns None for it, and handles bound to it raise from then on.
        Implementations may defer physically reclaiming its records to
        `purge_deleted_partitions`. Idempotent.

        Args:
            partition_key (str):
                The key of the partition to delete.
        """
        raise NotImplementedError

    @abstractmethod
    async def purge_deleted_partitions(self) -> bool:
        """
        Physically reclaim storage for deleted partitions, bounded per call.

        The sweeper: reclaims what `delete_partition` deferred, for every
        partition of this collection, oldest deletion first. Each call does
        a bounded amount of work, sized so it does not noticeably degrade
        concurrent request serving, commits what it did or nothing, and is
        safe to repeat, including after a failure on backend contention
        with another writer, and to run concurrently from any process. The
        store never schedules it; a deployment must run it somewhere (the
        server's resource manager runs it in the background).
        Implementations that reclaim physically in `delete_partition` may
        return False without doing anything.

        Returns:
            bool:
                True if another call may reclaim more. False if this call
                found nothing to claim; entries a concurrent purger holds
                are that purger's to finish, so the caller may back off.
        """
        raise NotImplementedError
