"""Abstract base class for a temporal extractor."""

from abc import ABC, abstractmethod
from datetime import datetime

from memmachine_server.temporal.time_range import TimeRange


class TemporalExtractor(ABC):
    """Abstract base class for a temporal extractor."""

    @abstractmethod
    async def extract(
        self, text: str, *, ref_time: datetime | None = None
    ) -> list[TimeRange]:
        """
        Extract time ranges.

        Args:
            text (str):
                The text to extract time ranges from.
            ref_time (datetime):
                The reference time for resolving relative time references.

        Returns:
            list[TimeRange]:
                Time ranges extracted from the text.
        """
        raise NotImplementedError
