"""
Abstract base class for a segment store.

Defines an interface for adding, retrieving, and deleting segments of episodes.
"""

from abc import ABC, abstractmethod
from collections.abc import Iterable, Mapping
from uuid import UUID

from memmachine.common.filter.filter_parser import FilterExpr

from .data_types import Segment


class SegmentStore(ABC):
    """Abstract base class for a segment store."""

    @abstractmethod
    async def startup(self) -> None:
        """Startup."""
        raise NotImplementedError

    @abstractmethod
    async def shutdown(self) -> None:
        """Shutdown."""
        raise NotImplementedError

    @abstractmethod
    async def add_segments(
        self,
        session_key: str,
        segments: Mapping[Segment, Iterable[UUID]],
    ) -> None:
        """
        Add a mapping of segments to the UUIDs of all embedding records.

        Args:
            session_key (str):
                The key of the session to which the segments belong.
            segments (Mapping[Segment, Iterable[UUID]]):
                A mapping of segments to the UUID of all embedding records associated with the segment.

        """
        raise NotImplementedError

    @abstractmethod
    async def get_segment_contexts(
        self,
        session_key: str,
        seed_segment_uuids: Iterable[UUID],
        *,
        max_backward_segments: int = 0,
        max_forward_segments: int = 0,
        property_filter: FilterExpr | None = None,
    ) -> Iterable[Iterable[Segment]]:
        """
        Returns a window of segments around each of the seed segments.

        Args:
            session_key (str):
                The key of the session to which the segments belong.
            seed_segment_uuids (Iterable[UUID]):
                The UUIDs of the seed segments for which to retrieve contexts.
            max_backward_segments (int):
                The maximum number of segments to include before each seed segment (default: 0).
            max_forward_segments (int):
                The maximum number of segments to include after each seed segment (default: 0).
            property_filter (FilterExpr | None):
                An optional filter expression to apply to the segments (default: None).

        Returns:
            Iterable[Iterable[Segment]]:
                An iterable of segment contexts.

        """
        raise NotImplementedError

    @abstractmethod
    async def delete_episodes_segments(
        self,
        session_key: str,
        episode_uuids: Iterable[UUID],
    ) -> Iterable[UUID]:
        """
        Delete all segments associated with the episodes given by their UUIDs.

        Args:
            session_key (str):
                The key of the session to which the segments belong.
            episode_uuids (Iterable[UUID]):
                The UUIDs of the episodes for which to delete segments.

        Returns:
            Iterable[UUID]:
                The UUIDs of the embedding records associated with the deleted segments.
        """
        raise NotImplementedError
