"""Segmenter ABC for segmenting events into segments."""

from abc import ABC, abstractmethod

from memmachine_server.episodic_memory.event_memory.data_types import (
    Event,
    FormatOptions,
    Segment,
)


class Segmenter(ABC):
    """Segmenter ABC for segmenting events into segments."""

    @abstractmethod
    async def segment(
        self,
        event: Event,
        *,
        format_options: FormatOptions | None = None,
    ) -> list[Segment]:
        """
        Segment an event into segments.

        Args:
            event (Event): The event to segment.
            format_options (FormatOptions | None):
                Options for formatting.
                (default: None).
        """
        raise NotImplementedError
