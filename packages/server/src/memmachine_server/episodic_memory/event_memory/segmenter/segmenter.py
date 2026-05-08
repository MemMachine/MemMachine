"""Segmenter ABC for segmenting events into segments."""

from abc import ABC, abstractmethod

from memmachine_server.episodic_memory.event_memory.data_types import (
    Event,
    Segment,
)


class Segmenter(ABC):
    """Segmenter ABC for segmenting events into segments."""

    @abstractmethod
    def segment(self, event: Event) -> list[Segment]:
        """Segment an event into segments."""
        raise NotImplementedError
