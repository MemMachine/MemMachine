"""
Abstract base class for a segment linker.

Defines an interface for adding, retrieving, and deleting segments of episodes.
"""

from abc import ABC, abstractmethod
from collections.abc import Iterable, Mapping
from uuid import UUID

from memmachine_server.common.filter.filter_parser import FilterExpr
from memmachine_server.episodic_memory.extra_memory.data_types import (
    Segment,
)


class SegmentLinkerPartition(ABC):
    """
    Partition-scoped handle for a segment linker.

    Manages the relationship between episodes, segments, and their derivatives
    within a single partition.
    """

    @abstractmethod
    async def register_segments(
        self,
        links: Mapping[Segment, Iterable[UUID]],
    ) -> None:
        """
        Register links between segments and their derivatives.

        Args:
            links (Mapping[Segment, Iterable[UUID]]):
                Mapping from each segment to the UUIDs of its derivatives.
        """
        raise NotImplementedError

    @abstractmethod
    async def get_segments_by_derivatives(
        self,
        derivative_uuids: Iterable[UUID],
        *,
        limit_per_derivative: int | None = None,
        property_filter: FilterExpr | None = None,
    ) -> dict[UUID, list[Segment]]:
        """
        Get segments associated with the derivatives given by their UUIDs.

        Segments are ordered chronologically by
        (timestamp, episode_uuid, block, index).

        Only derivatives that have at least one linked segment in the
        partition are included in the result; unknown or unlinked
        derivative UUIDs are silently omitted.

        Segments are deduplicated per derivative.

        Args:
            derivative_uuids (Iterable[UUID]):
                The UUIDs of the derivatives for which to retrieve linked segments.
            limit_per_derivative (int | None):
                The maximum number of segments to return per derivative.
                The limit is distributed evenly between the oldest and newest
                segments, favoring the most recent when the limit is odd.
                If None, return all linked segments (default: None).
            property_filter (FilterExpr | None):
                An optional filter expression to apply to the segments (default: None).

        Returns:
            dict[UUID, list[Segment]]:
                A mapping from each derivative UUID to its linked segments.
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
    async def delete_segments_by_episodes(
        self,
        episode_uuids: Iterable[UUID],
    ) -> None:
        """
        Delete all segments associated with the episodes given by their UUIDs.

        Args:
            episode_uuids (Iterable[UUID]):
                The UUIDs of the episodes for which to delete segments.
        """
        raise NotImplementedError

    @abstractmethod
    async def mark_orphaned_derivatives_for_purging(self, limit: int = 10_000) -> None:
        """
        Mark derivatives that are orphaned for purging.

        A derivative is orphaned if it has no linked segments in this partition.

        Args:
            limit (int):
                The maximum number of orphaned derivatives to mark in one call (default: 10_000).
        """
        raise NotImplementedError

    @abstractmethod
    async def get_derivatives_pending_purge(self, limit: int = 10_000) -> set[UUID]:
        """
        Get derivatives marked for purging but not yet purged.

        Args:
            limit (int):
                The maximum number of UUIDs of derivatives to return (default: 10_000).

        Returns:
            set[UUID]:
                The UUIDs of derivatives in the purging state.
        """
        raise NotImplementedError

    @abstractmethod
    async def purge_derivatives(self, derivative_uuids: Iterable[UUID]) -> None:
        """
        Physically remove derivatives from the segment linker.

        Should be called after the derivatives have been deleted from external systems.

        Args:
            derivative_uuids (Iterable[UUID]):
                The UUIDs of the derivatives to purge.
        """
        raise NotImplementedError


class SegmentLinker(ABC):
    """
    Abstract base class for a segment linker.

    Factory that creates partition-scoped handles.
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
    async def open_partition(self, partition_key: str) -> SegmentLinkerPartition:
        """
        Open a partition-scoped handle for the given partition key.

        Args:
            partition_key (str):
                The key of the partition.

        Returns:
            SegmentLinkerPartition:
                A partition-scoped handle.
        """
        raise NotImplementedError

    @abstractmethod
    async def close_partition(
        self, segment_linker_partition: SegmentLinkerPartition
    ) -> None:
        """
        Close a partition-scoped handle.

        Args:
            segment_linker_partition (SegmentLinkerPartition):
                The partition-scoped handle to close.
        """
        raise NotImplementedError
