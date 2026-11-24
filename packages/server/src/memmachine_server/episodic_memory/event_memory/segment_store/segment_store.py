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
from memmachine_server.common.partition_registry import PartitionedStore
from memmachine_server.episodic_memory.event_memory.data_types import (
    Derivative,
    Segment,
)
from memmachine_server.episodic_memory.event_memory.segment_store.data_types import (
    SegmentStorePartitionConfig,
)


class SegmentStorePartition(ABC):
    """
    A handle on one partition's events and segments in a segment store.

    Every operation reads and writes the handle's partition alone, without
    checking whether the partition was deleted.

    The partition holds an event at most once, and its events and their
    segments are immutable: a changed event is forgotten and added again.
    A forget deletes the events with their segments and derivatives, and
    leaves a tombstone for each segment holding its derivatives' UUIDs, so
    their derived index entries can still be deleted.
    """

    @property
    @abstractmethod
    def config(self) -> SegmentStorePartitionConfig:
        """The configuration for this partition."""
        raise NotImplementedError

    @abstractmethod
    async def add_event_segments(
        self,
        segments_by_event: Mapping[UUID, Iterable[Segment]],
        derivatives: Iterable[Derivative],
        *,
        lease_seconds: int,
    ) -> None:
        """
        Add events with their segments and derivatives, and an indexing job for each segment.

        All of it commits together or not at all. A batch naming an event
        the partition already holds is rejected whole, before anything is
        stored. An indexing job stands until `delete_indexing_jobs`
        removes it; the caller holds the indexing jobs for
        `lease_seconds`, on the store's clock, before a claim may take
        them.

        Args:
            segments_by_event (Mapping[UUID, Iterable[Segment]]):
                A mapping from each event's UUID to its segments.
            derivatives (Iterable[Derivative]):
                The derivatives, each naming a segment in the batch.
            lease_seconds (int):
                Seconds the caller holds the indexing jobs.

        Raises:
            SegmentStoreEventAlreadyExistsError:
                If the partition already holds any of the events; it names
                them all.
            ValueError:
                If a segment names an event other than the one it is listed
                under, or a derivative names a segment that is not in the
                batch.
        """
        raise NotImplementedError

    @abstractmethod
    async def delete_indexing_jobs(
        self,
        segment_uuids: Iterable[UUID],
    ) -> None:
        """
        Delete the indexing jobs of the given segments. Idempotent.

        Only the segments' indexing jobs are deleted. A segment forgotten
        since keeps its tombstone. Never waits on another transaction: an
        indexing job another one holds at the time is left in place, for a
        forget to delete or the background work to write again.

        Args:
            segment_uuids (Iterable[UUID]):
                The UUIDs of the segments whose indexing jobs to delete.
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

        Raises:
            ValueError:
                If `property_filter` names an unknown field.
        """
        raise NotImplementedError

    @abstractmethod
    async def get_segment_uuids_by_derivative_uuids(
        self,
        derivative_uuids: Iterable[UUID],
    ) -> dict[UUID, UUID]:
        """
        Get the segment each of the given derivatives belongs to.

        A derivative belongs to exactly one segment, and is deleted with it,
        so a forgotten segment's derivatives are left out.

        Args:
            derivative_uuids (Iterable[UUID]):
                The UUIDs of the derivatives whose owning segments to look up.

        Returns:
            dict[UUID, UUID]:
                A mapping from each given derivative UUID whose segment the
                partition holds to its segment's UUID.
        """
        raise NotImplementedError

    @abstractmethod
    async def delete_event_segments(self, event_uuids: Iterable[UUID]) -> None:
        """
        Delete events with their segments, derivatives, and indexing jobs, leaving a tombstone for each segment.

        All of it commits together or not at all. A tombstone holds its
        segment's derivative UUIDs until its second delete, and is due for
        its first delete once this commits. A renewal of a claim on these
        events' indexing jobs either commits first or finds them gone. An
        event the partition does not hold is skipped. Once the partition is
        dropped, nothing is deleted, and its purge reclaims the events.
        Idempotent.

        Args:
            event_uuids (Iterable[UUID]):
                The UUIDs of the events to forget.
        """
        raise NotImplementedError


@dataclass(frozen=True)
class IndexingJobClaim(ABC):
    """
    A claim on a segment's indexing job, which its ingest did not delete, held under a lease.

    Its holder renews it before writing the segment's derivatives, then ends
    it with `finish` once they are written, or with `fail` and what the
    write raised. Each claim counts an attempt on the record. A claim never
    ended keeps its attempt, and its record is claimable again once the
    claim's lease, and then the backoff, have passed.

    Attributes:
        partition_uuid (UUID):
            The UUID of the partition holding the segment.
        segment_uuid (UUID):
            The UUID of the segment.
        derivatives (list[Derivative]):
            The segment's derivatives, to write to their index.
        attempt (int):
            The claim's attempt number on the record, from 1: one more than
            the claims on it before, each of which failed or was abandoned.
    """

    partition_uuid: UUID
    segment_uuid: UUID
    derivatives: list[Derivative]
    attempt: int

    @abstractmethod
    async def renew(self, *, lease_seconds: int) -> bool:
        """
        Hold the record for `lease_seconds` more, if the claim still holds.

        The claim holds while it is the latest on its record and the record
        exists. A forget of the record's event either takes effect after a
        renewal that returned True, or makes the renewal return False.

        Args:
            lease_seconds (int):
                Seconds from now the claim holds its record.

        Returns:
            bool:
                Whether the claim held and was renewed.
        """
        raise NotImplementedError

    @abstractmethod
    async def finish(self) -> None:
        """
        Delete the record, once the claim has written its derivatives.

        A claim that outlasted its lease deletes the record too: the
        derivatives are written, whichever claim wrote them.
        """
        raise NotImplementedError

    @abstractmethod
    async def fail(self, error: Exception) -> None:
        """
        End a claim whose work raised, if the claim is the latest on its record.

        The record is claimable again after a backoff that grows with each
        failed claim in a row. After the store's maximum, it is no longer
        claimed, and the store reports it.

        Args:
            error (Exception):
                What the work raised.
        """
        raise NotImplementedError


@dataclass(frozen=True)
class TombstoneClaim(ABC):
    """
    A claim on a forgotten segment's tombstone, due for its first or second delete, held under a lease.

    Its holder deletes the segment's derivatives from their index, then ends
    the claim with `finish`, or with `fail` and what the delete raised. Each
    claim counts an attempt on the tombstone. A claim never ended keeps its
    attempt, and its tombstone is claimable again once the claim's lease,
    and then the backoff, have passed.

    Attributes:
        partition_uuid (UUID):
            The UUID of the partition the tombstone belongs to.
        segment_uuid (UUID):
            The UUID of the forgotten segment.
        derivative_uuids (list[UUID]):
            The UUIDs of the segment's derivatives, whose index entries the
            delete removes.
        second_delete (bool):
            Whether the claim is for the tombstone's second delete; it is
            for its first otherwise.
        attempt (int):
            The claim's attempt number since the tombstone last made
            progress, from 1: one more than the claims on it since, each of
            which failed or was abandoned.
    """

    partition_uuid: UUID
    segment_uuid: UUID
    derivative_uuids: list[UUID]
    second_delete: bool
    attempt: int

    @abstractmethod
    async def finish(self) -> None:
        """
        Record the claimed delete as done.

        After its first delete, the tombstone awaits its second, due once
        the store's retention has passed since this record, and any other
        open claim on it ends; after its second, it is deleted with the
        derivative UUIDs it holds.
        """
        raise NotImplementedError

    @abstractmethod
    async def fail(self, error: Exception) -> None:
        """
        End a claim whose delete raised, if the claim is the latest on its tombstone.

        The tombstone is claimable again after a backoff that grows with
        each failed claim in a row. After the store's maximum, it is no
        longer claimed, and the store reports it.

        Args:
            error (Exception):
                What the delete raised.
        """
        raise NotImplementedError


class SegmentStore(PartitionedStore):
    """
    Abstract base class for a segment store.

    The store holds each partition's events and segments under the
    partition's UUID. A partition keeps the configuration it was created
    with, which every handle on the partition uses. Creating the partition
    again with a different configuration raises
    SegmentStorePartitionConfigMismatchError.

    Indexing jobs and tombstones of every partition are claimed from the
    store, deleted partitions' included until their purge deletes them: a
    claim's holder checks whether its partition is live.
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
    async def get_partition(self, partition_uuid: UUID) -> SegmentStorePartition | None:
        """
        Get a handle on a created partition.

        Args:
            partition_uuid (UUID):
                The partition's UUID.

        Returns:
            SegmentStorePartition | None:
                A handle bound to the partition, or None when the store finds
                nothing under the UUID: the partition was never created, or
                was dropped.
        """
        raise NotImplementedError

    @abstractmethod
    async def claim_indexing_job(
        self,
        *,
        lease_seconds: int,
    ) -> IndexingJobClaim | None:
        """
        Claim the indexing job that came due first.

        An indexing job is due once its ingest's lease has passed, on the
        store's clock. It is claimable then unless a claim on it holds its
        lease, or its last claim failed and its backoff has not passed, or
        its claims failed the store's maximum number of times in a row. A
        claim holds its record for `lease_seconds`. Safe to call from
        several processes at once.

        Args:
            lease_seconds (int):
                Seconds the claim holds its record.

        Returns:
            IndexingJobClaim | None:
                The claim, or None if no record is claimable.
        """
        raise NotImplementedError

    @abstractmethod
    async def claim_tombstone(self, *, lease_seconds: int) -> TombstoneClaim | None:
        """
        Claim the oldest tombstone due for a delete.

        A tombstone awaiting its first delete is due once its forget
        commits. One awaiting its second is due once the store's retention
        has passed since its first delete was recorded; no claim's schedule
        brings that forward. A due tombstone is claimable unless a claim on
        it holds its lease, or its last claim failed and its backoff has not
        passed, or its claims failed the store's maximum number of times in
        a row. A claim holds its tombstone for `lease_seconds`. Safe to call
        from several processes at once.

        Args:
            lease_seconds (int):
                Seconds the claim holds its tombstone.

        Returns:
            TombstoneClaim | None:
                The claim, or None if no tombstone is claimable.
        """
        raise NotImplementedError
