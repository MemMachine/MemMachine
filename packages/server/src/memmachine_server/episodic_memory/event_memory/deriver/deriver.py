"""Deriver ABC for deriving derivatives from segments."""

from abc import ABC, abstractmethod

from memmachine_server.episodic_memory.event_memory.data_types import (
    Derivative,
    FormatOptions,
    Segment,
)


class Deriver(ABC):
    """Deriver ABC for deriving derivatives from segments."""

    @abstractmethod
    async def derive(
        self,
        segment: Segment,
        *,
        format_options: FormatOptions | None = None,
    ) -> list[Derivative]:
        """
        Derive derivatives from a segment.

        Args:
            segment (Segment): The segment to derive from.
            format_options (FormatOptions | None):
                Options for formatting.
                (default: None).
        """
        raise NotImplementedError
