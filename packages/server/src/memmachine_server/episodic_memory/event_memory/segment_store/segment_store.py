"""
Abstract base class for a segment store.

Defines an interface for adding, retrieving, and deleting segments of events.
"""

from abc import ABC, abstractmethod
from collections.abc import Iterable, Mapping
from datetime import datetime
from uuid import UUID

from memmachine_server.common.filter.filter_parser import FilterExpr
from memmachine_server.episodic_memory.event_memory.data_types import (
    Neighborhood,
    Segment,
)
from memmachine_server.episodic_memory.event_memory.segment_store.data_types import (
    SegmentStorePartitionConfig,
)


class SegmentStorePartition(ABC):
    """Partition-scoped handle for a segment store.

    A handle is bound to the partition incarnation it was opened on:
    deleting the partition permanently invalidates the handle, and its
    data operations raise `SegmentStorePartitionHandleStaleError` from
    then on, even if a partition is later created under the same key.
    A call with empty input may do no work and return without checking
    the handle.

    Segments are immutable. `add_segments` must reject a segment uuid
    that is already stored rather than replace it, and no operation that
    edits a stored segment may be added to this contract; a changed
    event is forgotten and encoded again.

    Segments within a partition are in one total order,
    `(timestamp, event_uuid, index, offset)`. `get_segments` fetches
    segments by uuid, subject to the filters. `get_segment_neighborhoods`
    walks the order outward from seeds named by uuid, within their
    session, and returns the neighbors that pass the filters; the seed
    itself is located, never filtered, and never returned.
    """

    @property
    @abstractmethod
    def config(self) -> SegmentStorePartitionConfig:
        """The configuration for this partition."""
        raise NotImplementedError

    @abstractmethod
    async def add_segments(
        self,
        segments_to_derivative_uuids: Mapping[Segment, Iterable[UUID]],
    ) -> None:
        """
        Add segments and their associated derivative UUIDs to the partition.

        Args:
            segments_to_derivative_uuids (Mapping[Segment, Iterable[UUID]]):
                A mapping from each segment to the UUIDs of its derivatives.
        """
        raise NotImplementedError

    @abstractmethod
    async def get_segments(
        self,
        segment_uuids: Iterable[UUID],
        *,
        since: datetime | None = None,
        until: datetime | None = None,
        session_ids: Iterable[str] | None = None,
        source_ids: Iterable[str] | None = None,
        block_kinds: Iterable[str] | None = None,
        property_filter: FilterExpr | None = None,
    ) -> dict[UUID, Segment]:
        """
        Get the segments among these that the partition holds and that pass the filters.

        Args:
            segment_uuids (Iterable[UUID]):
                The uuids to look up.
            since (datetime | None):
                Inclusive lower bound on the segment timestamp, timezone-aware
                (default: None).
            until (datetime | None):
                Exclusive upper bound on the segment timestamp, timezone-aware,
                so ranges meet without overlap (default: None).
            session_ids (Iterable[str] | None):
                Keep only segments whose session id is one of these; an
                empty list keeps none (default: None, every session).
            source_ids (Iterable[str] | None):
                Keep only segments whose source id is one of these; an
                empty list keeps none (default: None, every source).
            block_kinds (Iterable[str] | None):
                Keep only segments whose block is of one of these kinds;
                an empty list keeps none (default: None, every kind).
            property_filter (FilterExpr | None):
                A filter expression over segment properties (default: None).

        Returns:
            dict[UUID, Segment]:
                A mapping from each uuid found and admitted to its segment.
                A uuid the partition does not hold, or whose segment fails
                a filter, is absent.

        Raises:
            ValueError: If `since` or `until` is naive.
        """
        raise NotImplementedError

    @abstractmethod
    async def get_segment_neighborhoods(
        self,
        seed_segment_uuids: Iterable[UUID],
        *,
        before: int = 0,
        after: int = 0,
        since: datetime | None = None,
        until: datetime | None = None,
        source_ids: Iterable[str] | None = None,
        block_kinds: Iterable[str] | None = None,
        property_filter: FilterExpr | None = None,
    ) -> dict[UUID, Neighborhood]:
        """
        Get the segments around each seed segment, never the seed itself.

        The seed is an address: it is located whether or not it passes any
        filter, the walk stays within its session, and the filters select
        the neighbors. Other segments of the seed's own event are ordinary
        neighbors.

        Args:
            seed_segment_uuids (Iterable[UUID]):
                The UUIDs of the segments to gather neighbors around.
            before (int):
                The maximum number of neighbors before each seed (default: 0).
            after (int):
                The maximum number of neighbors after each seed (default: 0).
            since (datetime | None):
                Inclusive lower bound on the neighbors' timestamp, timezone-aware
                (default: None).
            until (datetime | None):
                Exclusive upper bound on the neighbors' timestamp, timezone-aware,
                so ranges meet without overlap (default: None).
            source_ids (Iterable[str] | None):
                Keep only neighbors whose source id is one of these; an
                empty list keeps none (default: None, every source).
            block_kinds (Iterable[str] | None):
                Keep only neighbors whose block is of one of these kinds;
                an empty list keeps none (default: None, every kind).
            property_filter (FilterExpr | None):
                A filter expression over the neighbors' properties
                (default: None).

        Returns:
            dict[UUID, Neighborhood]:
                A mapping from each known seed to its neighbors: `before`
                in order ending just before the seed, `after` in order
                starting just after it. A seed with no neighbors to show
                maps to two empty lists; an unknown seed is absent.

        Raises:
            ValueError: If `since` or `until` is naive.
        """
        raise NotImplementedError

    @abstractmethod
    async def get_segment_uuids_by_event_uuids(
        self,
        event_uuids: Iterable[UUID],
    ) -> dict[UUID, list[UUID]]:
        """
        Get segment UUIDs associated with the events given by their UUIDs.

        Args:
            event_uuids (Iterable[UUID]):
                The UUIDs of the events for which to retrieve the UUIDs of associated segments.

        Returns:
            dict[UUID, list[UUID]]:
                A mapping from each event UUID to the UUIDs of its
                associated segments, in the event's own order (by index,
                then offset).
        """
        raise NotImplementedError

    @abstractmethod
    async def get_derivative_uuids_by_segment_uuids(
        self,
        segment_uuids: Iterable[UUID],
    ) -> dict[UUID, list[UUID]]:
        """
        Get derivative UUIDs associated with the segments given by their UUIDs.

        Args:
            segment_uuids (Iterable[UUID]):
                The UUIDs of the segments for which to retrieve the UUIDs of associated derivatives.

        Returns:
            dict[UUID, list[UUID]]:
                A mapping from each segment UUID to the UUIDs of its associated derivatives.
        """
        raise NotImplementedError

    @abstractmethod
    async def get_segment_uuids_by_derivative_uuids(
        self,
        derivative_uuids: Iterable[UUID],
    ) -> dict[UUID, UUID]:
        """
        Get the segment each of the given derivatives belongs to.

        A derivative belongs to exactly one segment, so this is the inverse of
        `get_derivative_uuids_by_segment_uuids` and answers one UUID rather
        than a list. UUIDs the partition does not hold are omitted.

        Args:
            derivative_uuids (Iterable[UUID]):
                The UUIDs of the derivatives whose owning segments to look up.

        Returns:
            dict[UUID, UUID]:
                A mapping from each derivative UUID to its segment's UUID.
        """
        raise NotImplementedError

    @abstractmethod
    async def delete_segments(
        self,
        segment_uuids: Iterable[UUID],
    ) -> None:
        """
        Delete segments and their associated derivatives given by segment UUIDs.

        Args:
            segment_uuids (Iterable[UUID]):
                The UUIDs of the segments to delete.
        """
        raise NotImplementedError

    @abstractmethod
    async def delete_derivatives(
        self,
        derivative_uuids: Iterable[UUID],
    ) -> None:
        """
        Delete derivative links by derivative UUID, leaving their segments.

        Args:
            derivative_uuids (Iterable[UUID]):
                The UUIDs of the derivatives to unlink.
        """
        raise NotImplementedError


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
        Implementations may defer physically reclaiming its storage to
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
