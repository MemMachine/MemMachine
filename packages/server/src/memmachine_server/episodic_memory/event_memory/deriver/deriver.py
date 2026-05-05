"""Deriver ABC for deriving derivatives from segments."""

from abc import ABC, abstractmethod

from memmachine_server.episodic_memory.event_memory.data_types import (
    Derivative,
    Segment,
)


class Deriver(ABC):
    """Deriver ABC for deriving derivatives from segments."""

    @abstractmethod
    def derive(self, segment: Segment) -> list[Derivative]:
        """Derive derivatives from a segment."""
        raise NotImplementedError
