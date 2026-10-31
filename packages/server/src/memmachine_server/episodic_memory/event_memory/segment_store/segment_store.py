"""
Abstract base class for a segment store.

Defines an interface for adding, retrieving, and forgetting events with
their segments, and for the work a derived index of their derivatives
leaves behind.
"""

from abc import ABC, abstractmethod
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from uuid import UUID

from memmachine_server.common.filter.filter_parser import FilterExpr
from memmachine_server.episodic_memory.event_memory.data_types import (
    Segment,
)
from memmachine_server.episodic_memory.event_memory.segment_store.data_types import (
    PendingDerivatives,
    SegmentStorePartitionConfig,
    Tombstone,
)


class SegmentStorePartition(ABC):
    """Partition-scoped handle for a segment store.

    A handle is bound to the partition incarnation it was opened on:
    deleting the partition permanently invalidates the handle, and its
    data operations raise `SegmentStorePartitionHandleStaleError` from
    then on, even if a partition is later created under the same key.
    A call with empty input may do no work and return without checking
    the handle.

    The partition holds an event at most once, and its events and their
    segments are immutable: a changed event is forgotten and added again.
    A derivative stays linked to its segment until its event is
    forgotten, and a forget records the derivatives it removed in a
    tombstone, so their derived index entries can still be deleted.
    """

    @property
    @abstractmethod
    def config(self) -> SegmentStorePartitionConfig:
        """The configuration for this partition."""
        raise NotImplementedError

    @abstractmethod
    async def add_events(
        self,
        segments_by_event: Mapping[UUID, Iterable[Segment]],
        pending_derivatives: Iterable[PendingDerivatives],
    ) -> None:
        """
        Add events with their segments, derivative links, and pending derivatives.

        All of it commits together or not at all. A batch naming an event
        the partition already holds is rejected whole, before anything is
        stored.

        Args:
            segments_by_event (Mapping[UUID, Iterable[Segment]]):
                A mapping from each event's UUID to its segments.
            pending_derivatives (Iterable[PendingDerivatives]):
                Each event's derivatives, each linked to the segment it
                names, and held as pending until
                `delete_pending_derivatives` removes them.

        Raises:
            SegmentStoreEventAlreadyStoredError:
                If the partition already holds any of the events; it names
                them all.
            ValueError:
                If a segment names an event other than the one it is listed
                under, or pending derivatives name an event or a segment
                that is not in the batch.
        """
        raise NotImplementedError

    @abstractmethod
    async def delete_pending_derivatives(
        self,
        pending_derivatives_uuids: Iterable[UUID],
    ) -> None:
        """
        Delete pending derivatives given by their UUIDs. Idempotent.

        The derivatives stay linked; only their pending record goes.

        Args:
            pending_derivatives_uuids (Iterable[UUID]):
                The UUIDs of the pending derivatives records to delete.
        """
        raise NotImplementedError

    @abstractmethod
    async def get_segment_contexts(
        self,
        seed_segment_uuids: Iterable[UUID],
        *,
        max_backward_segments: int = 0,
        max_forward_segments: int = 0,
        property_filter: FilterExpr | None = None,
    ) -> dict[UUID, list[Segment]]:
        """
        Get a window of segments around each of the seed segments.

        An implementation may bound how far from each seed it looks for
        context, so a side can come back with fewer segments than requested
        even when more exist further away.

        Args:
            seed_segment_uuids (Iterable[UUID]):
                The UUIDs of the seed segments for which to retrieve contexts.
            max_backward_segments (int):
                The maximum number of segments to include before each seed segment (default: 0).
            max_forward_segments (int):
                The maximum number of segments to include after each seed segment (default: 0).
            property_filter (FilterExpr | None):
                An optional filter expression to apply to the segments (default: None).

        Returns:
            dict[UUID, list[Segment]]:
                A mapping from each seed segment UUID to its context segments.
                A seed that does not match the filter has no entry, and a
                context holds only segments that match it.
        """
        raise NotImplementedError

    @abstractmethod
    async def get_segment_uuids_by_derivative_uuids(
        self,
        derivative_uuids: Iterable[UUID],
    ) -> dict[UUID, UUID]:
        """
        Get the segment each of the given derivatives belongs to.

        A derivative belongs to exactly one segment.

        Args:
            derivative_uuids (Iterable[UUID]):
                The UUIDs of the derivatives whose owning segments to look up.

        Returns:
            dict[UUID, UUID]:
                A mapping from each given derivative UUID the partition holds
                to its segment's UUID.
        """
        raise NotImplementedError

    @abstractmethod
    async def forget_events(
        self,
        event_uuids: Iterable[UUID],
    ) -> Tombstone | None:
        """
        Forget events, with their segments, derivative links, and pending derivatives.

        All of it commits together or not at all, with a tombstone naming
        every derivative the forget unlinked. The tombstone's time is taken
        after the events are locked, so a write that claims pending
        derivatives of these events either finishes its claim before that
        time or finds them gone. An event the partition does not hold is
        skipped. Idempotent.

        Args:
            event_uuids (Iterable[UUID]):
                The UUIDs of the events to forget.

        Returns:
            Tombstone | None:
                The tombstone, awaiting its first delete, or None if the
                forget unlinked no derivative.
        """
        raise NotImplementedError

    @abstractmethod
    async def record_tombstone_first_delete(self, tombstone_uuid: UUID) -> None:
        """
        Record that a tombstone's derivatives were deleted from their index once.

        The tombstone then awaits only its second delete, which falls due
        once the store's retention has passed since the forget. Idempotent,
        and a no-op for a tombstone no longer awaiting its first delete.

        Args:
            tombstone_uuid (UUID):
                The UUID of the tombstone.
        """
        raise NotImplementedError

    @abstractmethod
    async def require_current(self) -> None:
        """
        Raise unless the partition this handle was opened on still exists.

        Raises:
            SegmentStorePartitionHandleStaleError:
                If the partition was deleted, whether or not another was
                created under its key since.
        """
        raise NotImplementedError


@dataclass(frozen=True)
class PendingDerivativesClaim:
    """
    A claim on pending derivatives whose writer did not delete them.

    Attributes:
        partition_key (str):
            The key of the partition holding the derivatives.
        partition (SegmentStorePartition):
            A handle on the partition the derivatives belong to.
        pending_derivatives (PendingDerivatives):
            The pending derivatives.
        generation (int):
            The claim's generation; a later claim on the same record has a
            higher one.
        attempt (int):
            The claim's attempt number since the record was claimed without
            failing, from 1.
    """

    partition_key: str
    partition: SegmentStorePartition
    pending_derivatives: PendingDerivatives
    generation: int
    attempt: int


@dataclass(frozen=True)
class TombstoneClaim:
    """
    A claim on a tombstone that is due for a delete.

    Attributes:
        partition_key (str):
            The key of the partition the tombstone belongs to.
        tombstone (Tombstone):
            The tombstone.
        second_delete (bool):
            Whether the claim is for the tombstone's second delete; it is
            for its first otherwise.
        generation (int):
            The claim's generation; a later claim on the same tombstone has
            a higher one.
        attempt (int):
            The claim's attempt number since the tombstone last made
            progress, from 1.
    """

    partition_key: str
    tombstone: Tombstone
    second_delete: bool
    generation: int
    attempt: int


class SegmentStore(ABC):
    """
    Abstract base class for a segment store.

    Manages partition-scoped handles.

    Partition keys must match `[a-z0-9_]+`
    (lowercase alphanumeric and underscores only)
    and be at most 32 bytes.
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
    async def create_partition(
        self,
        partition_key: str,
        config: SegmentStorePartitionConfig,
    ) -> None:
        """
        Create a new partition.

        Args:
            partition_key (str):
                The key of the partition.
            config (SegmentStorePartitionConfig):
                Configuration for the partition.

        Raises:
            SegmentStorePartitionAlreadyExistsError: If the partition already exists.
            SegmentStoreAttemptsExhaustedError:
                If creation exhausted its internal attempts on a
                failure that should not recur; an immediate retry is
                unlikely to succeed -- diagnose the chained cause.
        """
        raise NotImplementedError

    @abstractmethod
    async def open_partition(self, partition_key: str) -> SegmentStorePartition | None:
        """
        Open a partition-scoped handle for an existing partition.

        Args:
            partition_key (str):
                The key of the partition.

        Returns:
            SegmentStorePartition | None:
                A partition-scoped handle, or None if the partition does not exist.
        """
        raise NotImplementedError

    @abstractmethod
    async def open_or_create_partition(
        self,
        partition_key: str,
        config: SegmentStorePartitionConfig,
    ) -> SegmentStorePartition:
        """
        Open the partition if it exists, or create it if it does not.

        Args:
            partition_key (str):
                The key of the partition.
            config (SegmentStorePartitionConfig):
                Configuration for the partition.

        Returns:
            SegmentStorePartition:
                A partition-scoped handle.

        Raises:
            SegmentStorePartitionConfigMismatchError:
                If the partition already exists with a different configuration.
            SegmentStoreAttemptsExhaustedError:
                If creation exhausted its internal attempts on a
                failure that should not recur; an immediate retry is
                unlikely to succeed -- diagnose the chained cause.
        """
        raise NotImplementedError

    @abstractmethod
    async def close_partition(
        self, segment_store_partition: SegmentStorePartition
    ) -> None:
        """
        Close a partition-scoped handle.

        Args:
            segment_store_partition (SegmentStorePartition):
                The partition-scoped handle to close.
        """
        raise NotImplementedError

    @abstractmethod
    async def delete_partition(self, partition_key: str) -> None:
        """
        Delete a partition.

        The partition becomes unreachable immediately: it can no longer be
        opened, and handles opened on it raise from then on.
        Implementations may defer physically reclaiming its rows to
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
        partition, oldest deletion first at the database clock's
        resolution. Each call does a bounded amount
        of work, sized so it does not noticeably degrade concurrent
        request serving, commits what it did or nothing, and is safe to
        repeat, including after a failure on backend contention with
        another writer, and to run concurrently from any process. The
        store never schedules
        it; a deployment must run it somewhere (the server's resource
        manager runs it in the background). Implementations that reclaim
        physically in
        `delete_partition` may return False without doing anything.

        Returns:
            bool:
                True if another call may reclaim more. False if this call
                found nothing to claim; entries a concurrent purger holds
                are that purger's to finish, so the caller may back off.
        """
        raise NotImplementedError

    @abstractmethod
    async def claim_pending_derivatives(
        self,
        *,
        min_age_seconds: int,
        lease_seconds: int,
    ) -> PendingDerivativesClaim | None:
        """
        Claim the oldest pending derivatives that have waited long enough.

        Pending derivatives are due once `min_age_seconds` have passed
        since they were added, on the store's clock. A due record is
        claimable unless a claim on it holds its lease, or its last claim
        failed and its backoff has not passed, or its claims failed the
        store's maximum number of times in a row. A claim holds its record
        for `lease_seconds`. Claims skip deleted partitions. Safe to call
        from several processes at once.

        Args:
            min_age_seconds (int):
                Seconds pending derivatives must have waited.
            lease_seconds (int):
                Seconds the claim holds its record.

        Returns:
            PendingDerivativesClaim | None:
                The claim, or None if no record is claimable.
        """
        raise NotImplementedError

    @abstractmethod
    async def finish_pending_derivatives_claim(
        self, claim: PendingDerivativesClaim
    ) -> None:
        """
        Delete claimed pending derivatives, if the claim is the latest on them.

        A claim that outlasted its lease leaves the record to the later
        claim.

        Args:
            claim (PendingDerivativesClaim):
                The claim.
        """
        raise NotImplementedError

    @abstractmethod
    async def fail_pending_derivatives_claim(
        self, claim: PendingDerivativesClaim, error: Exception
    ) -> None:
        """
        End a claim whose work raised, if the claim is the latest on its record.

        The record is claimable again after a backoff that grows with each
        failed claim in a row. After the store's maximum, it is no longer
        claimed, and the store reports it.

        Args:
            claim (PendingDerivativesClaim):
                The claim.
            error (Exception):
                What the work raised.
        """
        raise NotImplementedError

    @abstractmethod
    async def claim_tombstone(self) -> TombstoneClaim | None:
        """
        Claim the oldest tombstone due for a delete.

        A tombstone awaiting its first delete is due at once. One awaiting
        its second is due once the store's retention has passed since its
        forget, on the store's clock. A due tombstone is claimable unless a
        claim on it holds its lease, or its last claim failed and its
        backoff has not passed, or its claims failed the store's maximum
        number of times in a row. Claims skip deleted partitions. Safe to
        call from several processes at once.

        Returns:
            TombstoneClaim | None:
                The claim, or None if no tombstone is claimable.
        """
        raise NotImplementedError

    @abstractmethod
    async def finish_tombstone_claim(self, claim: TombstoneClaim) -> None:
        """
        Record a claimed delete as done.

        After its first delete, the tombstone awaits its second; after its
        second, it is removed.

        Args:
            claim (TombstoneClaim):
                The claim.
        """
        raise NotImplementedError

    @abstractmethod
    async def fail_tombstone_claim(
        self, claim: TombstoneClaim, error: Exception
    ) -> None:
        """
        End a claim whose delete raised, if the claim is the latest on its tombstone.

        The tombstone is claimable again after a backoff that grows with
        each failed claim in a row. After the store's maximum, it is no
        longer claimed, and the store reports it.

        Args:
            claim (TombstoneClaim):
                The claim.
            error (Exception):
                What the delete raised.
        """
        raise NotImplementedError
