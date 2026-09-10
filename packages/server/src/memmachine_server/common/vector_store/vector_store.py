"""
Abstract base class for a vector store.

Defines the interface for adding, querying, and deleting records.
"""

from abc import ABC, abstractmethod
from collections.abc import Iterable, Mapping, Sequence
from uuid import UUID

from memmachine_server.common.data_types import PropertyType
from memmachine_server.common.filter import FilterExpr

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

    A collection stores the properties its store declares
    (`indexed_properties`), typed and indexed for filtering during a
    search, and no others: a record or a filter naming an undeclared key is
    rejected, so an undeclared key never exists in the store, neither
    stored write-only nor scanned for.
    """

    @property
    @abstractmethod
    def config(self) -> VectorStoreCollectionConfig:
        """The configuration for this collection."""
        raise NotImplementedError

    @property
    @abstractmethod
    def indexed_properties(self) -> Mapping[str, PropertyType]:
        """The declared schema: every key this collection stores and filters on."""
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
        Upsert records in the collection.

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
        Delete records from the collection by their UUIDs.

        Args:
            record_uuids (Iterable[UUID]):
                Iterable of UUIDs of the records to delete.
        """
        raise NotImplementedError


class VectorStore(ABC):
    """
    Abstract base class for a vector store.

    A given logical collection identified by a (namespace, name) pair
    must be managed by at most one process at a time.
    The consumer is responsible for sharding names across processes.

    Different namespaces are fully independent (separate native collections).
    Multiple logical collections with the same (namespace, vector dimensions)
    may share a native collection to reduce overhead.

    A store declares one indexed-property schema for every collection it
    holds, from deployment configuration merged with the system keys of the
    consumer it is built for; the schema is fixed for the life of the
    store's data, and changing it is a migration.

    Naming constraints:
        - Namespaces, names, and property keys must match `[a-z0-9_]+`
          (lowercase alphanumeric and underscores only).
        - Each identifier must be at most 32 bytes.
    """

    @property
    @abstractmethod
    def indexed_properties(self) -> Mapping[str, PropertyType]:
        """The declared schema every collection of this store carries."""
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
        The configuration (dimensions) is fixed at creation time.

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

        This will delete all data in the collection.
        It is idempotent.

        Args:
            namespace (str):
                Namespace of the collection.
            name (str):
                Name of the collection within the namespace.
        """
        raise NotImplementedError
