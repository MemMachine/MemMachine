"""
A related episode postulator
that postulates no related episodes.

This can be used as a default implementation
when postulation is not appropriate.
"""

from ..data_types import Episode
from .related_episode_postulator import RelatedEpisodePostulator


class NullRelatedEpisodePostulator(RelatedEpisodePostulator):
    """
    RelatedEpisodePostulator implementation
    that postulates no related episodes.
    """

    async def postulate(self, episode: Episode) -> list[Episode]:
        return []
