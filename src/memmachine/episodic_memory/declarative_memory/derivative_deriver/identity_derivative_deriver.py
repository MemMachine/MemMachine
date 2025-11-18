"""
A derivative deriver that creates derivatives
identical to the episodes in the episode cluster.
"""

from uuid import uuid4

from pydantic import Field

from ..data_types import Derivative, EpisodeCluster
from .derivative_deriver import DerivativeDeriver


class IdentityDerivativeDeriverParams:
    """
    Parameters for IdentityDerivativeDeriver.

    Attributes:
        derivative_type (str):
            The type to assign to the derived derivatives
            (default: "identity").
    """

    derivative_type: str = Field(
        "identity",
        description="The type to assign to the derived derivatives",
    )


class IdentityDerivativeDeriver(DerivativeDeriver):
    """
    Derivative deriver that creates derivatives
    identical to the episodes in the episode cluster.
    """

    def __init__(self, params: IdentityDerivativeDeriverParams):
        """
        Initialize an IdentityDerivativeDeriver
        with the provided parameters.

        Args:
            params (IdentityDerivativeDeriverParams):
                Parameters for the IdentityDerivativeDeriver.
        """
        super().__init__()

        self._derivative_type = params.derivative_type

    async def derive(self, episode_cluster: EpisodeCluster) -> list[Derivative]:
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
