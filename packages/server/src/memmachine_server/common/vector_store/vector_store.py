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
    All data operations are scoped to this logical collection.

    A handle is bound to one life of the collection: after the collection
    is deleted, its operations raise VectorStoreCollectionHandleStaleError,
    and a collection created again under the same (namespace, name) is a
    new life. A read concurrent with the deletion may take effect before
    it, returning the content from before the deletion. A store that cannot
    detect a stale handle says so in its own contract.

    Implementations must support storing, filtering on, and returning
    record properties not declared in the configured indexed properties schema.

    The schema exists to support indexing on fixed-type record properties.
    Record properties not declared in the schema may have mixed-type values.
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
        return_vector: bool = False,
        return_properties: bool = True,
    ) -> list[QueryResult]:
        """
        Query for records matching the criteria by query vectors.

        Args:
            query_vectors (Iterable[Sequence[float]]):
                The vectors to compare against.
            limit (int):
                Maximum number of matching records to return per query vector.
            score_threshold (float | None):
                Score threshold to consider a match
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
            list[QueryResult]:
                Results for each query vector,
                ordered as in the input iterable.
        """
        raise NotImplementedError

    @abstractmethod
    async def get(
        self,
        *,
        record_uuids: Iterable[UUID],
        return_vector: bool = False,
        return_properties: bool = True,
    ) -> list[Record]:
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
            list[Record]:
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
    """
    Abstract base class for a vector store.

    A logical collection is identified to callers by a (namespace, name)
    pair and inside the store by an incarnation minted per life of the
    pair, so nothing written under one life of a name is ever seen by, or
    reclaimed out from under, another. Which processes may share a store's
    collections is the store's own contract, stated on the store.

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
        Create a logical collection in the vector store and return a handle to it.

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
                (namespace, name) already exists.
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
        """
        raise NotImplementedError

    @abstractmethod
    async def close_collection(self, *, collection: VectorStoreCollection) -> None:
        """
        Close a collection handle.

        Args:
            collection (Collection):
                The handle of the collection to close.
        """
        raise NotImplementedError

    @abstractmethod
    async def delete_collection(self, *, namespace: str, name: str) -> None:
        """
        Delete a logical collection from the vector store.

        When this returns, the collection is unreachable and its data is
        deleted or, on a store that reclaims it later, left for
        `purge_deleted_collections`. It is idempotent.

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
        Reclaim, bounded, some of the storage of deleted collections.

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
