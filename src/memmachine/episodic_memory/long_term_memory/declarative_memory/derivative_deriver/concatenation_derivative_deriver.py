"""
A derivative deriver that concatenates the content
of all episodes in an episode cluster into a single derivative.
"""

from typing import Any
from uuid import uuid4

from ..data_types import ContentType, Derivative, EpisodeCluster
from .derivative_deriver import DerivativeDeriver


class ConcatenationDerivativeDeriver(DerivativeDeriver):
    """
    Derivative deriver that concatenates the content
    of all episodes in an episode cluster into a single derivative.
    """

    def __init__(self, config: dict[str, Any] = {}):
        """
        Initialize a ConcatenationDerivativeDeriver
        with the provided configuration.

        Args:
            config (dict[str, Any], optional):
                Configuration dictionary containing:
                - derivative_type (str, optional):
                  The type to assign
                  to the derived derivative (default: "concatenation").
                - separator (str, optional):
                  The string to use to separate episode contents
                  in the concatenated derivative (default: "\n").
        """
        super().__init__()

        self._derivative_type = config.get("derivative_type", "concatenation")
        self._separator = config.get("separator", "\n")

    async def derive(
        self, episode_cluster: EpisodeCluster
    ) -> list[Derivative]:
        return [
            Derivative(
                uuid=uuid4(),
                derivative_type=self._derivative_type,
                content_type=ContentType.STRING,
                content=self._separator.join(
                    episode.content for episode in episode_cluster.episodes
                ),
                timestamp=episode_cluster.timestamp,
                filterable_properties=episode_cluster.filterable_properties,
                user_metadata=episode_cluster.user_metadata,
            )
        ]
