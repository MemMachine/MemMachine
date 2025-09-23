"""
Abstract base class for a related episode postulator.

Defines an interface for postulating related episodes given an episode.
"""

from abc import ABC, abstractmethod

from ..data_types import Episode


class RelatedEpisodePostulator(ABC):
    """
    Abstract base class for a related episode postulator.
    """

    @abstractmethod
    async def postulate(self, episode: Episode) -> list[Episode]:
        """
        Postulate related episodes given an episode.

        Args:
            episode (Episode):
                The input episode
                for which to postulate related episodes.

        Returns:
            list[Episode]:
                A list of postulated related episodes.
        """
        raise NotImplementedError
