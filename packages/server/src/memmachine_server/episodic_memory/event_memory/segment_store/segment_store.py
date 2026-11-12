"""
Abstract base class for a segment store.

Defines an interface for adding, retrieving, and deleting segments of events.
"""

from abc import ABC, abstractmethod
from collections.abc import Iterable, Mapping
from uuid import UUID

from memmachine_server.common.filter.filter_parser import FilterExpr
from memmachine_server.episodic_memory.event_memory.data_types import (
    Segment,
)
from memmachine_server.episodic_memory.event_memory.segment_store.data_types import (
    SegmentStorePartitionConfig,
)


class SegmentStorePartition(ABC):
    """A handle on one partition's segments in a segment store.

    Every operation reads and writes the partition it was opened on, and no
    other. The handle checks nothing about whether the partition is still
    live: its owner fences each operation by the partition's
    registration.
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

    The store holds each partition's segments under the partition's UUID, and
    holds no partition names: its owner's partition registry names each
    partition, mints its UUID, and decides which is live. Partitions are
    isolated from one another, including partitions created under the same
    name: segments written to one are never read from another.
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
        partition_uuid: UUID,
        config: SegmentStorePartitionConfig,
    ) -> None:
        """
        Create the storage a partition needs of its own.

        Records the partition's configuration, which every handle opened
        on it uses. Safe to repeat with the same configuration.

        Args:
            partition_uuid (UUID):
                The partition's UUID, minted by the owner's registry.
            config (SegmentStorePartitionConfig):
                Configuration for the partition.

        Raises:
            SegmentStorePartitionConfigMismatchError:
                If the partition was created with a different
                configuration.
        """
        raise NotImplementedError

    @abstractmethod
    async def open_partition(
        self, partition_uuid: UUID
    ) -> SegmentStorePartition | None:
        """
        Open a handle on a created partition.

        Args:
            partition_uuid (UUID):
                The partition's UUID.

        Returns:
            SegmentStorePartition | None:
                A handle bound to the partition, or None if the partition was
                never created or its purge has removed it.
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
    async def purge_partition(self, partition_uuid: UUID) -> bool:
        """
        Delete a bounded batch of what a partition holds: its segments, then its own storage.

        The owner's purge rounds call it once the partition is deleted and
        its retention has passed, until it returns False. Each
        call does a bounded amount of work, sized so it does not noticeably
        degrade concurrent request serving, commits what it did or nothing,
        and is safe to repeat and to run from several processes at once.

        Args:
            partition_uuid (UUID):
                The deleted partition's UUID.

        Returns:
            bool:
                Whether the call found anything of the partition's, so a
                call that returns False proves the partition gone from the
                store.
        """
        raise NotImplementedError
