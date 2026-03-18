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


class DerivativeNotActiveError(Exception):
    """Raised when an operation that requires an active derivative is attempted on a derivative that is not active."""

    def __init__(self, not_active: Iterable[UUID]) -> None:
        """Initialize the error with the UUIDs of the derivatives that are not active."""
        self.not_active = set(not_active)
        message = f"{len(self.not_active)} derivatives are not active"
        super().__init__(message)


class SegmentLinkerPartition(ABC):
    """
    Partition-scoped handle for a segment linker.

    Manages the relationship between episodes, segments, and their derivatives
    within a single partition.
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
    async def register_segments(
        self,
        links: Mapping[Segment, Iterable[UUID]],
        *,
        active: Iterable[UUID] | None = None,
    ) -> None:
        """
        Register links between segments and their derivatives.

        All UUIDs in `links` that are not in `active` will be registered in 'active' state.

        Args:
            links (Mapping[Segment, Iterable[UUID]]):
                Mapping from each segment to the UUIDs of its derivatives.
            active (Iterable[UUID] | None):
                UUIDs of derivatives believed to be active (default: None).

        Raises:
            DerivativeNotActiveError:
                If any derivative in `active` is not active.

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
    async def delete_all_segments(self) -> None:
        """Delete all segments for the partition."""
        raise NotImplementedError

    @abstractmethod
    async def get_orphaned_derivatives(self, limit: int = 1000) -> set[UUID]:
        """
        Identify derivatives that are orphaned.

        Args:
            limit (int):
                The maximum number of orphaned derivatives to return (default: 1000).

        Returns:
            set[UUID]:
                The UUIDs of the orphaned derivatives.

        """
        raise NotImplementedError

    @abstractmethod
    async def mark_orphaned_derivatives_for_purging(
        self, potential_orphan_uuids: Iterable[UUID]
    ) -> set[UUID]:
        """
        Transition derivatives from 'active' to 'purging' state if they are orphaned.

        Args:
            potential_orphan_uuids (Iterable[UUID]):
                The UUIDs of potentially orphaned derivatives.

        Returns:
            set[UUID]:
                The UUIDs of derivatives successfully marked for purging.

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
    def get_partition(self, partition_key: str) -> SegmentLinkerPartition:
        """
        Get a partition-scoped handle for the given partition key.

        Args:
            partition_key (str):
                The key of the partition.

        Returns:
            SegmentLinkerPartition:
                A partition-scoped handle.

        """
        raise NotImplementedError
