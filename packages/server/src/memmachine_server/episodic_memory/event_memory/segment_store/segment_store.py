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
    Segment,
)
from memmachine_server.episodic_memory.event_memory.segment_store.data_types import (
    EventHeader,
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
        """
        raise NotImplementedError

    @abstractmethod
    async def get_neighbor_segments(
        self,
        seed_segment_uuids: Iterable[UUID],
        *,
        max_backward_segments: int = 0,
        max_forward_segments: int = 0,
        property_filter: FilterExpr | None = None,
    ) -> dict[UUID, list[Segment]]:
        """
        Get the segments AROUND each seed segment, never the seed itself.

        The counterpart to ``get_segment_contexts`` for a caller that already holds
        the seed. There, the filter says which segments are valid results and the
        seed is one of them, so a seed that fails it has no context to return. Here
        the seed is an address: it is located whether or not it passes, the filter
        says only which surrounding segments are wanted, and the seed is excluded
        from the result unconditionally rather than sometimes appearing in it.

        Other segments of the seed's own event are ordinary neighbours -- only the
        seed segment itself is withheld.

        Args:
            seed_segment_uuids (Iterable[UUID]):
                The UUIDs of the segments to gather neighbours around.
            max_backward_segments (int):
                The maximum number of segments to include before each seed (default: 0).
            max_forward_segments (int):
                The maximum number of segments to include after each seed (default: 0).
            property_filter (FilterExpr | None):
                An optional filter over the NEIGHBOURS (default: None). It is not
                applied to the seed, which is an address rather than a result.

        Returns:
            dict[UUID, list[Segment]]:
                A mapping from each seed segment UUID to its neighbouring segments.
                A seed with no neighbours to show is absent from the mapping.
        """
        raise NotImplementedError

    @abstractmethod
    async def get_neighbor_events(
        self,
        seed_segment_uuids: Iterable[UUID],
        *,
        max_backward_events: int = 0,
        max_forward_events: int = 0,
        property_filter: FilterExpr | None = None,
    ) -> dict[UUID, list[Segment]]:
        """
        Get the segments of the whole events around each seed, never the seed's own.

        ``get_neighbor_segments`` measured in the other unit. A segment is a chunk
        of one event, so a segment-bounded window can begin and end mid-event --
        right for a flat budget, since every call then costs about the same. This
        one counts whole events, for when the question is "so many turns either
        side" and the length of what is in the way should not decide how far the
        window reaches.

        The seed's ENTIRE event is excluded, not merely the seed segment: in this
        unit the seed's event is the anchor the neighbours are counted from.

        Args:
            seed_segment_uuids (Iterable[UUID]):
                The UUIDs of the segments to gather neighbouring events around.
            max_backward_events (int):
                The maximum number of whole events to include before the seed's own
                event (default: 0).
            max_forward_events (int):
                The maximum number of whole events to include after it (default: 0).
            property_filter (FilterExpr | None):
                An optional filter over the neighbouring events (default: None).

        Returns:
            dict[UUID, list[Segment]]:
                A mapping from each seed segment UUID to the segments of its
                neighbouring events, in timeline order.
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
                A mapping from each event UUID to the UUIDs of its associated segments.
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
    async def find_segment_uuids_by_prefix(
        self,
        uuid_prefix: str,
        *,
        limit: int,
    ) -> list[UUID]:
        """
        Get the stored segment UUIDs whose hexadecimal form starts with a prefix.

        Segment UUIDs are the store's public addresses, and an address a person
        or a model has to read or retype is abbreviated in practice. Resolving an
        abbreviation is a store question -- only the store knows which UUIDs
        exist -- so it is answered here rather than by a caller guessing.

        Args:
            uuid_prefix (str):
                A hexadecimal prefix, without dashes. An empty prefix matches
                every segment.
            limit (int):
                The maximum number of matches to return. A caller reporting an
                ambiguous abbreviation should ask for one more than it intends
                to show, so it can tell a truncated list from a complete one.

        Returns:
            list[UUID]:
                The matching segment UUIDs, in ascending UUID order.
        """
        raise NotImplementedError

    @abstractmethod
    async def get_adjacent_segment_uuids(
        self,
        segment_uuids: Iterable[UUID],
    ) -> dict[UUID, tuple[UUID | None, UUID | None]]:
        """
        Get each UUID's nearest stored neighbours in ascending UUID order.

        The counterpart to `find_segment_uuids_by_prefix`: it says how short an
        abbreviation of a UUID can be while still naming only that segment.
        Whatever prefix separates a UUID from the two UUIDs adjacent to it
        separates it from every stored UUID, because anything sharing more
        would have sorted between them -- so two index lookups answer a
        question that otherwise means scanning the partition.

        The UUIDs need not themselves be stored: an absent one still has
        neighbours, which is what makes this usable before a segment is
        written.

        Args:
            segment_uuids (Iterable[UUID]):
                The UUIDs to find neighbours for.

        Returns:
            dict[UUID, tuple[UUID | None, UUID | None]]:
                A mapping from each requested UUID to the greatest stored UUID
                below it and the least stored UUID above it. Either is None
                when nothing is stored on that side.
        """
        raise NotImplementedError

    @abstractmethod
    async def list_event_headers(
        self,
        *,
        property_filter: FilterExpr | None = None,
        start: tuple[datetime, UUID] | None = None,
        end: tuple[datetime, UUID] | None = None,
        limit: int | None = None,
        descending: bool = False,
    ) -> list[EventHeader]:
        """
        List events on the timeline by their position and size, without content.

        Reading a conversation's shape -- which turns it has, where the work
        happened, how long each event is -- is a question about the timeline
        rather than about text, and answering it by fetching segments means
        decoding every block only to discard it. An event's whole content can
        be thousands of segments, so the cost is not marginal.

        Bounds are (timestamp, event_uuid) pairs naming an event, matching the
        order segments are stored in, and both are INCLUSIVE. `descending`
        reverses the walk, so a bounded `limit` takes the events nearest `end`
        rather than nearest `start`; the returned list is in timeline order
        either way.

        Args:
            property_filter (FilterExpr | None):
                An optional filter over segments. An event is listed when any
                of its segments matches, and its counts then describe only the
                matching segments (default: None).
            start (tuple[datetime, UUID] | None):
                Earliest event to include, inclusive. None starts at the
                beginning of the timeline (default: None).
            end (tuple[datetime, UUID] | None):
                Latest event to include, inclusive. None runs to the end of the
                timeline (default: None).
            limit (int | None):
                The maximum number of events to return. None returns every
                event in range (default: None).
            descending (bool):
                Walk from `end` towards `start`, so a `limit` keeps the latest
                events rather than the earliest (default: False).

        Returns:
            list[EventHeader]:
                The matching events in timeline order.
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
