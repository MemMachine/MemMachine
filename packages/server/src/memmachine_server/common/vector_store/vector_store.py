"""
Abstract base class for a vector store.

Defines the interface for adding, querying, and deleting records.
"""

from abc import ABC, abstractmethod
from collections.abc import Iterable, Sequence
from uuid import UUID

from memmachine_server.common.filter.filter_parser import (
    FilterExpr,
)

from .data_types import (
    QueryResult,
    Record,
    VectorStoreCollectionConfig,
)


class VectorStoreCollection(ABC):
    """
    A logical collection in a vector store.

    Identified by a (namespace, name) pair.
    All data operations are scoped to this logical collection: a record's
    UUID names it in this collection only, and the same UUID in another
    collection names another record.

    A collection created again under a deleted one's (namespace, name)
    starts empty. Operations through a handle whose collection was deleted
    may raise VectorStoreCollectionHandleStaleError, or, on an
    implementation that cannot tell, act on a collection created again
    under the same (namespace, name).

    An `upsert` or `delete` is durable once it returns; queries may not
    reflect it right away. A store that guarantees more states it.

    Implementations must support storing and filtering on record properties
    not declared in the configured indexed properties schema.

    The schema exists to support indexing on fixed-type record properties.
    Record properties not declared in the schema may have mixed-type values.

    A store keeps a datetime property's instant, taking a naive datetime as
    UTC, and may drop its UTC offset. A filter compares datetime values by
    their instant, for equality and ordering alike.
    """

    @property
    @abstractmethod
    def config(self) -> VectorStoreCollectionConfig:
        """The configuration for this collection."""
        raise NotImplementedError

    @abstractmethod
    async def upsert(
        self,
        *,
        records: Iterable[Record],
    ) -> None:
        """
        Upsert records in the collection.

        Insert records with new UUIDs,
        and update records with existing UUIDs.

        Args:
            records (Iterable[Record]):
                Iterable of records to upsert.
                Records containing properties
                not in the indexed properties schema
                are allowed.

        Raises:
            ValueError:
                If two records share a UUID, a record's declared property
                holds a value of another type, a record's float property
                value is not finite, or a record's vector does not have
                the collection's dimensions; no record is written.
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
                positive.
            score_threshold (float | None):
                The worst score a match may have, by the collection's
                similarity metric; a match scoring exactly the threshold is
                returned (default: None).
            property_filter (FilterExpr | None):
                Filter expression tree.
                If None or empty, no property filtering is applied
                (default: None).

        Returns:
            list[QueryResult]:
                Results for each query vector,
                ordered as in the input iterable.

        Raises:
            ValueError:
                If the limit is not positive, a query vector does not have
                the collection's dimensions or has a coordinate that is not
                finite, the score threshold is not finite, or the property
                filter names an invalid property key.
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
    """
    Abstract base class for a vector store.

    A logical collection is identified by a (namespace, name) pair. A
    collection deleted and created again under the same pair starts empty,
    and reclaiming the deleted one's storage leaves it untouched. Each store
    states which processes may share its collections.

    Different namespaces are fully independent (separate native collections).
    Multiple logical collections with the same (namespace, vector dimensions, similarity metric, indexed properties schema)
    may share a native collection to reduce overhead.

    Naming constraints:
        - Namespaces, names, and property keys must match `[a-z0-9_]+`
          (lowercase alphanumeric and underscores only).
        - Each identifier must be at most 32 bytes.
    """

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
        *,
        namespace: str,
        name: str,
        config: VectorStoreCollectionConfig,
    ) -> None:
        """
        Create a logical collection in the vector store.

        A (namespace, name) pair uniquely identifies a collection.
        The configuration (dimensions, similarity metric, schema)
        is fixed at creation time.

        Args:
            namespace (str):
                Groups related collections and guarantees storage
                isolation at the native collection level.
            name (str):
                Name to identify the collection within a namespace.
            config (VectorStoreCollectionConfig):
                Configuration for the collection.

        Raises:
            VectorStoreCollectionAlreadyExistsError: If a collection with the same
                (namespace, name) already exists, or is being created.
            VectorStoreCollectionDeletedError: If the collection was deleted
                before its creation completed.
            VectorStoreAttemptsExhaustedError: If the store gave up creating the
                collection after repeated attempts that made no progress.
        """
        raise NotImplementedError

    @abstractmethod
    async def open_or_create_collection(
        self,
        *,
        namespace: str,
        name: str,
        config: VectorStoreCollectionConfig,
    ) -> VectorStoreCollection:
        """
        Open the collection if it exists, or create it if it does not.

        Args:
            namespace (str):
                Groups related collections and guarantees storage
                isolation at the native collection level.
            name (str):
                Name to identify the collection within a namespace.
            config (VectorStoreCollectionConfig):
                Configuration for the collection.

        Returns:
            VectorStoreCollection:
                A handle to the opened or created collection.

        Raises:
            VectorStoreCollectionConfigMismatchError: If a collection with the same
                (namespace, name) already exists with a different configuration.
            VectorStoreCollectionPendingError: If the collection's creation, by
                another caller, did not complete within the store's attempts
                to open it.
            VectorStoreAttemptsExhaustedError: If the store gave up opening or
                creating the collection after repeated attempts that made no
                progress.
        """
        raise NotImplementedError

    @abstractmethod
    async def open_collection(
        self, *, namespace: str, name: str
    ) -> VectorStoreCollection | None:
        """
        Get a handle to a logical collection in the vector store.

        Args:
            namespace (str):
                Namespace of the collection.
            name (str):
                Name of the collection within the namespace.

        Returns:
            VectorStoreCollection | None:
                A handle to the opened collection, or None if it does not exist.

        Raises:
            VectorStoreCollectionPendingError: If the collection's creation
                has not completed.
        """
        raise NotImplementedError

    @abstractmethod
    async def delete_collection(self, *, namespace: str, name: str) -> None:
        """
        Delete a logical collection from the vector store.

        When this returns, the collection is unreachable. A store that
        reclaims storage later leaves the collection's data for
        `purge_deleted_collections`. Idempotent.

        Args:
            namespace (str):
                Namespace of the collection.
            name (str):
                Name of the collection within the namespace.
        """
        raise NotImplementedError

    @abstractmethod
    async def purge_deleted_collections(self) -> bool:
        """
        Reclaim some of the storage of deleted collections.

        Each call does a bounded amount of work. Call it until it returns
        False, and again from time to time: a deleted collection's storage
        may become reclaimable some time after the deletion. A store that
        reclaims storage in `delete_collection` returns False. Safe to call
        from several processes at once.

        Returns:
            bool:
                Whether the call ran a round; False when nothing was due.
        """
        raise NotImplementedError
