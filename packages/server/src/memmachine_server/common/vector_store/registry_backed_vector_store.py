"""
Base classes for a vector store whose collections a collection registry arbitrates.

The registry mints each collection's incarnation and arbitrates creation,
deletion, and reclamation across processes. A collection's name is
reserved, its storage is prepared, and the reservation is confirmed, which
makes the collection live; only a live collection is opened. The backend
holds records, each carrying its collection's incarnation, and a subclass
decides how: it prepares a new collection's storage, builds a handle for
one collection, and purges a deleted incarnation's records.
"""

import asyncio
import contextlib
import logging
from abc import abstractmethod
from collections.abc import Iterable, Sequence
from typing import override
from uuid import UUID

from pydantic import BaseModel, Field, InstanceOf

from memmachine_server.common.filter.filter_parser import FilterExpr
from memmachine_server.common.metrics_factory import MetricsFactory, OperationTracker

from .collection_registry import (
    Registration,
    Reservation,
    VectorStoreCollectionRegistry,
)
from .data_types import (
    QueryResult,
    Record,
    VectorStoreAttemptsExhaustedError,
    VectorStoreCollectionAlreadyExistsError,
    VectorStoreCollectionConfig,
    VectorStoreCollectionConfigMismatchError,
    VectorStoreCollectionDeletedError,
    VectorStoreCollectionPendingError,
)
from .utils import (
    require_declared_types,
    require_dimensions,
    require_distinct_record_uuids,
    require_finite_properties,
    require_identifiers,
    require_valid_limit,
    require_valid_query_vector,
    require_valid_score_threshold,
    validate_filter,
)
from .vector_store import VectorStore, VectorStoreCollection

logger = logging.getLogger(__name__)

# Attempts open-or-create makes, _OPEN_OR_CREATE_RETRY_DELAY_SECONDS apart,
# before it gives up on a collection that stays pending or a name it keeps
# losing.
_MAX_OPEN_OR_CREATE_ATTEMPTS = 10
_OPEN_OR_CREATE_RETRY_DELAY_SECONDS = 1


class RegistryBackedVectorStoreCollection(VectorStoreCollection):
    """A handle bound to one incarnation of a logical collection.

    Each operation checks its inputs and that the collection is still live:
    `upsert` before and after its backend call, `query` before it, and
    `delete` after it.

    A check after a write makes one that raced the collection's deletion
    raise instead of reporting success; whatever such a write landed, the
    purge reclaims.

    For subclasses: `_incarnation` is the incarnation the handle is bound to,
    and a subclass implements the backend calls `_upsert`, `_query`, and
    `_delete`.
    """

    def __init__(
        self, *, registration: Registration, tracker: OperationTracker
    ) -> None:
        """Initialize with the live registration the handle is bound to."""
        self._registration = registration
        self._incarnation = registration.incarnation
        self._config = registration.config
        self._tracker = tracker

    @property
    @override
    def config(self) -> VectorStoreCollectionConfig:
        return self._config

    @override
    async def upsert(self, *, records: Iterable[Record]) -> None:
        async with self._tracker("upsert"):
            records = list(records)
            for record in records:
                require_declared_types(
                    record.properties, self._config.indexed_properties_schema
                )
                require_finite_properties(record.properties)
                require_dimensions(record.vector, self._config.vector_dimensions)
            require_distinct_record_uuids(record.uuid for record in records)
            await self._registration.require_current()
            if not records:
                return
            await self._upsert(records)
            await self._registration.require_current()

    @override
    async def query(
        self,
        *,
        query_vectors: Iterable[Sequence[float]],
        limit: int,
        score_threshold: float | None = None,
        property_filter: FilterExpr | None = None,
    ) -> list[QueryResult]:
        async with self._tracker("query"):
            query_vectors = [list(query_vector) for query_vector in query_vectors]
            for query_vector in query_vectors:
                require_valid_query_vector(query_vector, self._config.vector_dimensions)
            require_valid_score_threshold(score_threshold)
            require_valid_limit(limit)
            if property_filter is not None and not validate_filter(property_filter):
                raise ValueError("Filter contains an invalid property key")
            await self._registration.require_current()
            if not query_vectors:
                return []
            return await self._query(
                query_vectors,
                limit=limit,
                score_threshold=score_threshold,
                property_filter=property_filter,
            )

    @override
    async def delete(self, *, record_uuids: Iterable[UUID]) -> None:
        async with self._tracker("delete"):
            # One check, after: a delete adds nothing a purge must reclaim,
            # and a stale handle's delete reaches only its own incarnation.
            record_uuids = list(record_uuids)
            if record_uuids:
                await self._delete(record_uuids)
            await self._registration.require_current()

    @abstractmethod
    async def _upsert(self, records: list[Record]) -> None:
        """
        Write records to the backend under the handle's incarnation.

        Called between two liveness checks, with at least one record and no
        two records of the same UUID, each already checked against the
        collection's configuration and holding only finite float property
        values. A record replaces the one with its UUID.
        The records are durable when it returns; a call that raises may have
        written some of them.

        Args:
            records (list[Record]): The records to write.

        Raises:
            Exception: Whatever the backend raises; the upsert raises it.
        """
        raise NotImplementedError

    @abstractmethod
    async def _query(
        self,
        query_vectors: list[list[float]],
        *,
        limit: int,
        score_threshold: float | None,
        property_filter: FilterExpr | None,
    ) -> list[QueryResult]:
        """
        Search the handle's incarnation's records for each query vector.

        Called after a liveness check, with at least one query vector and a
        positive limit, the vectors, threshold, and filter already checked.

        Args:
            query_vectors (list[list[float]]): The vectors to search for.
            limit (int): The most matches to return per query vector.
            score_threshold (float | None):
                The score a match must reach, or None for any.
            property_filter (FilterExpr | None):
                The condition a match's properties must meet, or None.

        Returns:
            list[QueryResult]:
                One result per query vector, in order, its matches best
                first.

        Raises:
            Exception: Whatever the backend raises; the query raises it.
        """
        raise NotImplementedError

    @abstractmethod
    async def _delete(self, record_uuids: list[UUID]) -> None:
        """
        Delete the handle's incarnation's records with these UUIDs.

        Called before a liveness check, with at least one UUID, possibly
        through a handle whose collection was deleted. A UUID with no record
        under the incarnation is skipped. The deletions are durable when it
        returns; a call that raises may have deleted some of the records.

        Args:
            record_uuids (list[UUID]): The UUIDs of the records to delete.

        Raises:
            Exception: Whatever the backend raises; the delete raises it.
        """
        raise NotImplementedError


class RegistryBackedVectorStoreParams(BaseModel):
    """
    Parameters for a RegistryBackedVectorStore.

    Attributes:
        collection_registry (VectorStoreCollectionRegistry):
            Registry of the store's collections, shared by the stores, in any
            process, whose clients connect to the same database, and by no
            other store. Started by the caller.
        metrics_factory (MetricsFactory | None):
            An instance of MetricsFactory for collecting usage metrics
            (default: None).
    """

    collection_registry: InstanceOf[VectorStoreCollectionRegistry] = Field(
        ...,
        description=(
            "Registry of the store's collections, shared by the stores, in any "
            "process, whose clients connect to the same database, and by no "
            "other store. Started by the caller"
        ),
    )
    metrics_factory: InstanceOf[MetricsFactory] | None = Field(
        None,
        description="An instance of MetricsFactory for collecting usage metrics",
    )


class RegistryBackedVectorStore[CollectionT: RegistryBackedVectorStoreCollection](
    VectorStore
):
    """A vector store whose collections a VectorStoreCollectionRegistry arbitrates.

    Any process connected to the same database, with the same registry, may
    serve any collection. A collection is pending until its storage is
    prepared: meanwhile opening it raises VectorStoreCollectionPendingError,
    `open_or_create_collection` waits a bounded time for it, and creating its
    name raises VectorStoreCollectionAlreadyExistsError. A collection a crash
    left pending stays pending until `delete_collection` deletes it.

    For subclasses: `_collection_registry` is the registry and `_tracker` times
    each operation, and a subclass implements `_prepare_storage`,
    `_build_collection_handle`, and `_purge_round`.
    """

    def __init__(
        self, params: RegistryBackedVectorStoreParams, *, metrics_prefix: str
    ) -> None:
        """Initialize with the registry and the prefix of the store's metrics."""
        super().__init__()
        self._collection_registry = params.collection_registry
        self._tracker = OperationTracker(params.metrics_factory, prefix=metrics_prefix)
        # Reservations cancelled after a failed preparation, held until done
        # so the garbage collector cannot drop one whose creation was
        # cancelled.
        self._reservation_cancellations: set[asyncio.Task[None]] = set()

    @override
    async def startup(self) -> None:
        # The caller owns the registry's lifecycle and that of any client a
        # subclass is given.
        pass

    @override
    async def shutdown(self) -> None:
        # The caller owns the registry's lifecycle and that of any client a
        # subclass is given.
        pass

    @override
    async def create_collection(
        self,
        *,
        namespace: str,
        name: str,
        config: VectorStoreCollectionConfig,
    ) -> None:
        require_identifiers(namespace, name)
        async with self._tracker("create_collection"):
            # The registry decides a creation race. The collection stays
            # pending until its storage is prepared; if it is deleted
            # meanwhile, confirming the reservation raises.
            reservation = await self._collection_registry.reserve(
                namespace, name, config
            )
            await self._prepare_and_confirm_or_cancel(reservation)

    @override
    async def open_or_create_collection(
        self,
        *,
        namespace: str,
        name: str,
        config: VectorStoreCollectionConfig,
    ) -> CollectionT:
        require_identifiers(namespace, name)
        async with self._tracker("open_or_create_collection"):
            # Read-then-create, retried: a pending collection is another
            # creator's (open it once it is live), losing the reservation
            # means another creator took the name meanwhile, and losing the
            # confirmation means a deleter removed this one while its storage
            # was prepared (create again).
            pending_error: VectorStoreCollectionPendingError | None = None
            lost_race: (
                VectorStoreCollectionAlreadyExistsError
                | VectorStoreCollectionDeletedError
                | None
            ) = None
            for attempt in range(_MAX_OPEN_OR_CREATE_ATTEMPTS):
                if attempt:
                    await asyncio.sleep(_OPEN_OR_CREATE_RETRY_DELAY_SECONDS)
                try:
                    registration = await self._collection_registry.resolve(
                        namespace, name
                    )
                except VectorStoreCollectionPendingError as err:
                    if err.config != config:
                        raise VectorStoreCollectionConfigMismatchError(
                            namespace, name, err.config, config
                        ) from err
                    pending_error = err
                    continue
                pending_error = None
                if registration is not None:
                    if registration.config != config:
                        raise VectorStoreCollectionConfigMismatchError(
                            namespace, name, registration.config, config
                        )
                    return self._build_collection_handle(registration)
                try:
                    reservation = await self._collection_registry.reserve(
                        namespace, name, config
                    )
                except VectorStoreCollectionAlreadyExistsError as err:
                    lost_race = err
                    continue
                try:
                    registration = await self._prepare_and_confirm_or_cancel(
                        reservation
                    )
                except VectorStoreCollectionDeletedError as err:
                    lost_race = err
                    continue
                return self._build_collection_handle(registration)
            # The last lookup found the collection pending.
            if pending_error is not None:
                raise pending_error
            raise VectorStoreAttemptsExhaustedError(
                f"Opening or creating collection ({namespace!r}, {name!r}) made "
                f"no progress after {_MAX_OPEN_OR_CREATE_ATTEMPTS} attempts"
            ) from lost_race

    async def _prepare_and_confirm_or_cancel(
        self, reservation: Reservation
    ) -> Registration:
        """Prepare a reserved collection's storage and confirm the reservation, cancelling it if either raises or the creation is cancelled.

        The cancel acts only on a pending collection, so a confirmation that
        committed before its failure or cancellation was observed stands. A
        collection whose reservation the registry cannot cancel stays pending
        until it is deleted.
        """
        try:
            await self._prepare_storage(
                reservation.namespace, reservation.config, reservation.incarnation
            )
            return await reservation.confirm()
        except BaseException:
            # Shielded, so a cancelled creation still frees the name. The task
            # reports its own failure, since a creation cancelled again stops
            # awaiting it before it ends.
            cancellation = asyncio.create_task(reservation.cancel())
            self._reservation_cancellations.add(cancellation)

            def finish(task: asyncio.Task[None]) -> None:
                self._reservation_cancellations.discard(task)
                if not task.cancelled() and task.exception() is not None:
                    logger.exception(
                        "Could not cancel the reservation of collection (%r, %r) "
                        "after its creation failed or was cancelled; it stays "
                        "pending until deleted",
                        reservation.namespace,
                        reservation.name,
                        exc_info=task.exception(),
                    )

            cancellation.add_done_callback(finish)
            with contextlib.suppress(Exception):
                await asyncio.shield(cancellation)
            raise

    @override
    async def open_collection(self, *, namespace: str, name: str) -> CollectionT | None:
        require_identifiers(namespace, name)
        async with self._tracker("open_collection"):
            registration = await self._collection_registry.resolve(namespace, name)
            if registration is None:
                return None
            return self._build_collection_handle(registration)

    @override
    async def delete_collection(self, *, namespace: str, name: str) -> None:
        require_identifiers(namespace, name)
        async with self._tracker("delete_collection"):
            # The collection is unreachable once unregister returns, and its
            # incarnation awaits the purge.
            await self._collection_registry.unregister(namespace, name)

    @override
    async def purge_deleted_collections(self) -> bool:
        # One purge round per call, on a due tombstone.
        async with self._tracker("purge_deleted_collections"):
            return await self._collection_registry.run_purge_round(self._purge_round)

    @abstractmethod
    async def _prepare_storage(
        self,
        namespace: str,
        config: VectorStoreCollectionConfig,
        incarnation: UUID,
    ) -> None:
        """
        Prepare the storage a newly reserved collection needs.

        The collection is reserved, pending, under the incarnation, and its
        reservation is confirmed once this returns. Its storage may be its own or shared
        with the other collections of its namespace and configuration. Shared
        storage is prepared by any number of processes at once, so preparing
        it must be idempotent and safe to race; what serves one namespace and
        configuration serves no other. Whatever a failed or interrupted call
        leaves must be recoverable: completed by a later call for the same
        namespace and configuration, or reclaimed by the purge rounds of the
        incarnation once its pending collection is deleted.

        Args:
            namespace (str):
                Namespace of the collection being created.
            config (VectorStoreCollectionConfig):
                Configuration of the collection being created.
            incarnation (UUID):
                The incarnation the collection's records will carry.

        Raises:
            Exception:
                Whatever the backend raises. The reservation is then
                cancelled, or, if the registry cannot cancel it, the
                collection stays pending until it is deleted.
        """
        raise NotImplementedError

    @abstractmethod
    def _build_collection_handle(self, registration: Registration) -> CollectionT:
        """
        Build a handle bound to a live collection's registration.

        The collection's storage was prepared before its reservation was
        confirmed.

        Args:
            registration (Registration):
                The live collection's registration, which carries its
                namespace, name, configuration, and incarnation.

        Returns:
            CollectionT:
                A handle bound to the registration, whose operations raise
                once the collection is deleted.
        """
        raise NotImplementedError

    @abstractmethod
    async def _purge_round(
        self, namespace: str, config: VectorStoreCollectionConfig, incarnation: UUID
    ) -> bool:
        """
        Delete records carrying a deleted incarnation, and return whether it found any.

        Runs under a purge claim, after the tombstone's retention. Returning
        False removes the tombstone, so it must return False only when no
        record under the incarnation remains; a round may delete some of the
        records and return True, to be run again. It may delete storage that
        holds only the incarnation's records, which a failed or interrupted
        preparation may have left part-made. It must be safe to repeat, and
        to run on two purgers at once. A round that finds the incarnation's
        storage missing returns False.

        Args:
            namespace (str):
                Namespace of the deleted collection.
            config (VectorStoreCollectionConfig):
                Configuration of the deleted collection; with the namespace,
                it locates the collection's records.
            incarnation (UUID):
                The incarnation the deleted collection's records carry.

        Returns:
            bool:
                Whether the round found records under the incarnation.

        Raises:
            Exception:
                Whatever the backend raises. The round then counts as
                failed, and the tombstone stays for a later claim.
        """
        raise NotImplementedError
