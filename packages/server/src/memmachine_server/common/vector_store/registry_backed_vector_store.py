"""
Base classes for a vector store whose partitions a partition registry arbitrates.

The registry mints each partition's incarnation and arbitrates creation,
deletion and reclamation across processes. A partition's key is reserved,
its storage is prepared, and the reservation is confirmed, which makes the
partition live; only a live partition is opened. The backend holds records, each carrying its partition's incarnation,
and a subclass decides how: it prepares the storage the store's partitions
share and the storage a new partition needs of its own, builds a handle for
one partition, and purges a deleted incarnation's records.
"""

import asyncio
import contextlib
import logging
from abc import abstractmethod
from collections.abc import Iterable, Mapping, Sequence
from typing import override
from uuid import UUID

from pydantic import BaseModel, Field, InstanceOf, field_validator

from memmachine_server.common.data_types import PropertyType, SimilarityMetric
from memmachine_server.common.filter.filter_parser import FilterExpr
from memmachine_server.common.metrics_factory import MetricsFactory, OperationTracker

from .data_types import (
    IndexedProperties,
    PartitionSchema,
    QueryResult,
    Record,
    VectorStorePartitionAlreadyExistsError,
    VectorStorePartitionPendingError,
    VectorStorePartitionSchemaMismatchError,
    indexed_property_names,
    validate_vector_store_name,
)
from .partition_registry import (
    Registration,
    Reservation,
    VectorStorePartitionRegistry,
)
from .utils import (
    require_declared_types,
    require_dimensions,
    require_partition_key,
    require_valid_limit,
    require_valid_query_vector,
    require_valid_score_threshold,
    validate_filter,
)
from .vector_store import VectorStore, VectorStorePartition

logger = logging.getLogger(__name__)


class RegistryBackedVectorStorePartition(VectorStorePartition):
    """A handle bound to one incarnation of a partition.

    Each operation checks its inputs and that the partition is still live:
    `upsert` before and after its backend call, `query` before it, and
    `delete` after it.

    A check after a write makes one that raced the partition's deletion
    raise instead of reporting success; whatever such a write landed, the
    purge reclaims.

    For subclasses: `_vector_store_name` is the store's name and
    `_incarnation` the incarnation the handle is bound to, and a subclass
    implements the backend calls `_upsert`, `_query` and `_delete`.
    """

    def __init__(
        self,
        *,
        vector_store_name: str,
        registration: Registration,
        vector_dimensions: int,
        similarity_metric: SimilarityMetric,
        indexed_properties: Mapping[str, PropertyType],
        tracker: OperationTracker,
    ) -> None:
        """Initialize with the registration the handle is bound to."""
        self._vector_store_name = vector_store_name
        self._registration = registration
        self._partition_key = registration.partition_key
        self._incarnation = registration.incarnation
        self._vector_dimensions = vector_dimensions
        self._similarity_metric = similarity_metric
        self._indexed_properties = dict(indexed_properties)
        self._tracker = tracker

    @property
    @override
    def partition_key(self) -> str:
        return self._partition_key

    @property
    @override
    def similarity_metric(self) -> SimilarityMetric:
        return self._similarity_metric

    @property
    @override
    def indexed_properties(self) -> Mapping[str, PropertyType]:
        return self._indexed_properties

    @override
    async def upsert(self, *, records: Iterable[Record]) -> None:
        async with self._tracker("upsert"):
            records = list(records)
            for record in records:
                require_declared_types(record.properties, self._indexed_properties)
                require_dimensions(record.vector, self._vector_dimensions)
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
                require_valid_query_vector(query_vector, self._vector_dimensions)
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

        Called between two liveness checks, with at least one record, each
        already checked against the store's declared schema and dimensions.
        A record replaces the one with its UUID. The records are durable when
        it returns; a call that raises may have written some of them.

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
        positive limit, the vectors, threshold and filter already checked.

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
        through a handle whose partition was deleted. A UUID with no record
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
        partition_registry (VectorStorePartitionRegistry):
            Registry of the store's partitions, shared by every process
            serving this store, and by no other store. Started by the
            caller.
        vector_store_name (str):
            The name of this store, which names its storage, so stores of
            different names may share a client.
        vector_dimensions (int):
            Dimensionality of every vector in the store.
        similarity_metric (SimilarityMetric):
            The metric every query of the store scores by (default: cosine).
        indexed_properties (IndexedProperties):
            The declared schema every partition of this store carries: each
            key is indexed by its declared type, which a search filters on.
        metrics_factory (MetricsFactory | None):
            An instance of MetricsFactory for collecting usage metrics
            (default: None).
    """

    partition_registry: InstanceOf[VectorStorePartitionRegistry] = Field(
        ...,
        description=(
            "Registry of the store's partitions, shared by every process serving "
            "this store, and by no other store. Started by the caller"
        ),
    )
    vector_store_name: str = Field(
        ...,
        description=(
            "The name of this store, which names its storage, so stores of "
            "different names may share a client"
        ),
    )
    vector_dimensions: int = Field(
        ..., gt=0, description="Dimensionality of every vector in the store"
    )
    similarity_metric: SimilarityMetric = Field(
        SimilarityMetric.COSINE,
        description="The metric every query of the store scores by",
    )
    indexed_properties: IndexedProperties = Field(
        ...,
        description=(
            "The declared schema every partition of this store carries: each key "
            "is indexed by its declared type, which a search filters on"
        ),
    )
    metrics_factory: InstanceOf[MetricsFactory] | None = Field(
        None,
        description="An instance of MetricsFactory for collecting usage metrics",
    )

    @field_validator("vector_store_name")
    @classmethod
    def _validate_vector_store_name(cls, vector_store_name: str) -> str:
        validate_vector_store_name(vector_store_name)
        return vector_store_name


class RegistryBackedVectorStore[PartitionT: RegistryBackedVectorStorePartition](
    VectorStore
):
    """A vector store whose partitions a VectorStorePartitionRegistry arbitrates.

    Any process sharing the backend and the registry may serve any partition.
    A partition is pending until its storage is prepared: meanwhile opening it
    raises VectorStorePartitionPendingError, and creating its key raises
    VectorStorePartitionAlreadyExistsError. A partition a crash left pending
    stays pending until `delete_partition` deletes it.

    For subclasses: `_partition_registry` is the registry and `_tracker` times
    each operation, and a subclass implements `_prepare_storage`,
    `_prepare_partition_storage`, `_partition_handle` and `_purge_round`.
    """

    def __init__(
        self, params: RegistryBackedVectorStoreParams, *, metrics_prefix: str
    ) -> None:
        """Initialize with the store's schema and registry, and the prefix of its metrics."""
        super().__init__()
        self._vector_store_name = params.vector_store_name
        self._vector_dimensions = params.vector_dimensions
        self._similarity_metric = params.similarity_metric
        self._indexed_properties = params.indexed_properties
        self._partition_registry = params.partition_registry
        self._tracker = OperationTracker(params.metrics_factory, prefix=metrics_prefix)
        # Reservations cancelled after a failed preparation, held until done
        # so the garbage collector cannot drop one whose creation was
        # cancelled.
        self._cancellations: set[asyncio.Task[None]] = set()

    @property
    @override
    def vector_store_name(self) -> str:
        return self._vector_store_name

    @property
    @override
    def vector_dimensions(self) -> int:
        return self._vector_dimensions

    @property
    @override
    def similarity_metric(self) -> SimilarityMetric:
        return self._similarity_metric

    @property
    @override
    def indexed_properties(self) -> Mapping[str, PropertyType]:
        return self._indexed_properties

    @override
    async def startup(self) -> None:
        # The caller owns the registry's lifecycle and that of any client a
        # subclass is given; starting the store prepares the storage its
        # partitions share.
        async with self._tracker("startup"):
            await self._prepare_storage()

    @override
    async def shutdown(self) -> None:
        # The caller owns the registry's lifecycle and that of any client a
        # subclass is given.
        pass

    @override
    async def create_partition(self, partition_key: str) -> None:
        require_partition_key(partition_key)
        async with self._tracker("create_partition"):
            # The registry decides a creation race. The partition stays
            # pending until its storage is prepared; if it is deleted
            # meanwhile, confirming the reservation raises.
            try:
                reservation = await self._partition_registry.reserve(
                    partition_key, self._declared_schema()
                )
            except VectorStorePartitionAlreadyExistsError:
                # A key taken under another schema is reported as such.
                with contextlib.suppress(VectorStorePartitionPendingError):
                    await self._checked_entry(partition_key)
                raise
            await self._prepare_and_confirm_or_cancel(reservation)

    async def _prepare_and_confirm_or_cancel(
        self, reservation: Reservation
    ) -> Registration:
        """Prepare a reserved partition's storage and confirm the reservation, cancelling it if either raises or the creation is cancelled.

        The cancel acts only on a pending partition, so a confirmation that
        committed before its failure or cancellation was observed stands. A
        partition whose reservation the registry cannot cancel stays pending
        until it is deleted.
        """
        try:
            await self._prepare_partition_storage(
                reservation.partition_key, reservation.incarnation
            )
            return await reservation.confirm()
        except BaseException:
            # Shielded, so a cancelled creation still frees the key. The task
            # reports its own failure, since a creation cancelled again stops
            # awaiting it before it ends.
            cancellation = asyncio.create_task(reservation.cancel())
            self._cancellations.add(cancellation)

            def finish(task: asyncio.Task[None]) -> None:
                self._cancellations.discard(task)
                if not task.cancelled() and task.exception() is not None:
                    logger.exception(
                        "Could not cancel the reservation of partition %r of "
                        "vector store %r after its creation failed or was "
                        "cancelled; it stays pending until deleted",
                        reservation.partition_key,
                        self._vector_store_name,
                        exc_info=task.exception(),
                    )

            cancellation.add_done_callback(finish)
            with contextlib.suppress(Exception):
                await asyncio.shield(cancellation)
            raise

    @override
    async def get_partition(self, partition_key: str) -> PartitionT | None:
        require_partition_key(partition_key)
        async with self._tracker("get_partition"):
            registration = await self._checked_entry(partition_key)
            if registration is None:
                return None
            return self._partition_handle(registration)

    async def _checked_entry(self, partition_key: str) -> Registration | None:
        """
        The live partition under the key, or None.

        Raises VectorStorePartitionSchemaMismatchError if the partition under
        the key, pending or live, was created under another schema than this
        store's, and VectorStorePartitionPendingError if it is pending under
        this store's.
        """
        declared_schema = self._declared_schema()
        try:
            registration = await self._partition_registry.resolve(partition_key)
        except VectorStorePartitionPendingError as err:
            if err.schema != declared_schema:
                raise VectorStorePartitionSchemaMismatchError(
                    self._vector_store_name,
                    partition_key,
                    err.schema,
                    declared_schema,
                ) from err
            raise
        if registration is not None and registration.schema != declared_schema:
            raise VectorStorePartitionSchemaMismatchError(
                self._vector_store_name,
                partition_key,
                registration.schema,
                declared_schema,
            )
        return registration

    def _declared_schema(self) -> PartitionSchema:
        return PartitionSchema(
            vector_dimensions=self._vector_dimensions,
            similarity_metric=self._similarity_metric,
            indexed_properties=indexed_property_names(self._indexed_properties),
        )

    @override
    async def delete_partition(self, partition_key: str) -> None:
        require_partition_key(partition_key)
        async with self._tracker("delete_partition"):
            # The partition is unreachable once unregister returns, and its
            # incarnation awaits the purge.
            await self._partition_registry.unregister(partition_key)

    @override
    async def purge_deleted_partitions(self) -> bool:
        # One purge round per call, on a due tombstone.
        async with self._tracker("purge_deleted_partitions"):
            return await self._partition_registry.run_purge_round(self._purge_round)

    @abstractmethod
    async def _prepare_storage(self) -> None:
        """
        Prepare the storage the store's partitions share.

        That is whatever a partition needs besides its incarnation and its
        own storage: a native collection they all live in, a container of
        per-partition units, or nothing. Storage of one partition's own is
        prepared when it registers, by `_prepare_partition_storage`. What is
        prepared for one store serves no other. It runs when the store starts,
        by any number of processes at once, so it must be idempotent and safe to race, and must
        complete what a failed call left part-made.

        Raises:
            Exception:
                Whatever the backend raises. Startup then fails, and the next
                startup completes what this call left.
        """
        raise NotImplementedError

    @abstractmethod
    async def _prepare_partition_storage(
        self, partition_key: str, incarnation: UUID
    ) -> None:
        """
        Prepare the storage a newly reserved partition needs of its own.

        The partition is reserved, pending, under the incarnation, and its
        reservation is confirmed once this returns. The storage the store's partitions
        share was prepared at startup; this prepares what the partition
        keeps of its own, such as a unit named by its
        incarnation, or nothing. Whatever a failed or interrupted call leaves
        must be recoverable: reclaimed by the purge rounds of the incarnation
        once its pending partition is deleted.

        Args:
            partition_key (str):
                Key of the partition being created.
            incarnation (UUID):
                The incarnation the partition's records will carry.

        Raises:
            Exception:
                Whatever the backend raises. The reservation is then
                cancelled, or, if the registry cannot cancel it, the
                partition stays pending until it is deleted.
        """
        raise NotImplementedError

    @abstractmethod
    def _partition_handle(self, registration: Registration) -> PartitionT:
        """
        Build a handle bound to a live partition's registration.

        The storage the store's partitions share was prepared at startup,
        and the partition's own before it was marked live.

        Args:
            registration (Registration):
                The live partition's registration, which carries its key,
                schema and incarnation.

        Returns:
            PartitionT:
                A handle bound to the registration, whose operations raise
                once the partition is deleted.
        """
        raise NotImplementedError

    @abstractmethod
    async def _purge_round(self, incarnation: UUID) -> bool:
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
            incarnation (UUID):
                The incarnation the deleted partition's records carry.

        Returns:
            bool:
                Whether the round found records under the incarnation.

        Raises:
            Exception:
                Whatever the backend raises. The round then counts as
                failed, and the tombstone stays for a later claim.
        """
        raise NotImplementedError
