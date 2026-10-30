"""Segmenter ABC for segmenting events into segments."""

from abc import ABC, abstractmethod

from memmachine_server.episodic_memory.event_memory.data_types import (
    Event,
    Segment,
)


class Segmenter(ABC):
    """Segmenter ABC for segmenting events into segments."""

    @abstractmethod
    async def segment(self, event: Event) -> list[Segment]:
        """
        Segment an event into segments.

        Every segment carries its event's timestamp, source id, context and
        properties unchanged: the vector stage filters on the event's
        properties a record carries, the segment stage on the segment's,
        and the two agree only if they are the same.

        Args:
            event (Event): The event to segment.
        """
        raise NotImplementedError
