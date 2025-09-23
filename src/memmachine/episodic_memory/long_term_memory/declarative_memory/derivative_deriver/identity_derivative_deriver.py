"""
A derivative deriver that creates derivatives
identical to the episodes in the episode cluster.
"""

from typing import Any
from uuid import uuid4

from ..data_types import Derivative, EpisodeCluster
from .derivative_deriver import DerivativeDeriver


class IdentityDerivativeDeriver(DerivativeDeriver):
    """
    Derivative deriver that creates derivatives
    identical to the episodes in the episode cluster.
    """

    def __init__(self, config: dict[str, Any] = {}):
        """
        Initialize an IdentityDerivativeDeriver
        with the provided configuration.

        Args:
            config (dict[str, Any], optional):
                Configuration dictionary containing:
                - derivative_type (str, optional):
                  The type to assign
                  to the derived derivatives (default: "identity").
        """
        super().__init__()

        self._derivative_type = config.get("derivative_type", "identity")

    async def derive(
        self, episode_cluster: EpisodeCluster
    ) -> list[Derivative]:
        return [
            Derivative(
                uuid=uuid4(),
                derivative_type=self._derivative_type,
                content_type=episode.content_type,
                content=episode.content,
                timestamp=episode.timestamp,
                filterable_properties=episode.filterable_properties,
                user_metadata=episode.user_metadata,
            )
            for episode in episode_cluster.episodes
        ]
