"""
Abstract base class for a segment store.

Defines an interface for adding, retrieving, and deleting segments of events.
"""

from abc import ABC, abstractmethod
from collections.abc import Iterable, Mapping
from datetime import datetime
from uuid import UUID

from memmachine_server.common.filter import FilterExpr
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

    Segments within a partition are in one total order,
    `(session_id, timestamp, event_uuid, index, offset)`: a null session
    id compares equal to a null session id and to nothing else, so the
    ungrouped stream is one stream. Context windows and neighborhoods
    walk this order and are confined to their seed's session, so they
    never cross into another conversation interleaved in time.

    The two reads, `get_segment_contexts` and `get_neighbourhoods`, take
    the same parameters and differ in what they return. `before` and
    `after` count segments on each side of a seed; `since` (inclusive)
    and `until` (exclusive) bound the segment timestamp, so ranges meet
    without overlap; `source_ids`, `block_kinds` and `property_filter`
    select rows, and an empty id or kind list admits nothing.
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

        A segment's `session_id`, `source_id` and its block's `kind` are
        stored beside the encoded block, so the store can filter on them.

        Args:
            segments_to_derivative_uuids (Mapping[Segment, Iterable[UUID]]):
                A mapping from each segment to the UUIDs of its derivatives.
        """
        raise NotImplementedError

    @abstractmethod
    async def get_segment_contexts(
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
    ) -> dict[UUID, list[Segment]]:
        """
        Get a window of segments around each of the seed segments.

        The search window: the seed is a result, so every filter applies
        to the seed as to the window rows, and a seed that fails a filter
        or is unknown has no entry.

        Args:
            seed_segment_uuids (Iterable[UUID]):
                The UUIDs of the seed segments for which to retrieve contexts.
            before (int):
                The maximum number of segments to include before each seed segment (default: 0).
            after (int):
                The maximum number of segments to include after each seed segment (default: 0).
            since (datetime | None):
                Inclusive lower bound on the segment timestamp (default: None).
            until (datetime | None):
                Exclusive upper bound on the segment timestamp (default: None).
            source_ids (Iterable[str] | None):
                Keep only segments whose source id is one of these (default: None).
            block_kinds (Iterable[str] | None):
                Keep only segments whose block is of one of these kinds (default: None).
            property_filter (FilterExpr | None):
                An optional filter expression over segment properties (default: None).

        Returns:
            dict[UUID, list[Segment]]:
                A mapping from each seed segment UUID to its context segments,
                in the store's order, the seed among them.
        """
        raise NotImplementedError

    @abstractmethod
    async def get_neighbourhoods(
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

        Expansion: the seed is an address the caller named and holds. It
        is located whether or not it passes any filter, the filters apply
        to the neighbors only, and it is never in the result. Other
        segments of the seed's own event are ordinary neighbors.

        Args:
            seed_segment_uuids (Iterable[UUID]):
                The UUIDs of the segments to gather neighbors around.
            before (int):
                The maximum number of segments to include before each seed (default: 0).
            after (int):
                The maximum number of segments to include after each seed (default: 0).
            since (datetime | None):
                Inclusive lower bound on the neighbors' timestamp (default: None).
            until (datetime | None):
                Exclusive upper bound on the neighbors' timestamp (default: None).
            source_ids (Iterable[str] | None):
                Keep only neighbors whose source id is one of these (default: None).
            block_kinds (Iterable[str] | None):
                Keep only neighbors whose block is of one of these kinds (default: None).
            property_filter (FilterExpr | None):
                An optional filter expression over the neighbors' properties
                (default: None).

        Returns:
            dict[UUID, Neighborhood]:
                A mapping from each known seed to its neighbors: `before`
                in order ending just before the seed, `after` in order
                starting just after it. A seed with no neighbors to show
                maps to two empty lists; an unknown seed is absent.
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
