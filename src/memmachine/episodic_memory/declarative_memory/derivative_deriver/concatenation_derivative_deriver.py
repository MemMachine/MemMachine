"""
A derivative deriver that concatenates the content
of all episodes in an episode cluster into a single derivative.
"""

from uuid import uuid4

from pydantic import BaseModel, Field

from ..data_types import ContentType, Derivative, EpisodeCluster
from .derivative_deriver import DerivativeDeriver


class ConcatenationDerivativeDeriverParams(BaseModel):
    """
    Parameters for ConcatenationDerivativeDeriver.

    Attributes:
        derivative_type (str):
            The type to assign to the derived derivatives
            (default: "concatenation").
        separator (str):
            The string to use to separate episode contents
            in the concatenated derivative
            (default: "\n").
    """

    derivative_type: str = Field(
        "concatenation",
        description="The type to assign to the derived derivatives",
    )
    separator: str = Field(
        "\n",
        description="The string to use to separate episode contents "
        "in the concatenated derivative",
    )


class ConcatenationDerivativeDeriver(DerivativeDeriver):
    """
    Derivative deriver that concatenates the content
    of all episodes in an episode cluster into a single derivative.
    """

    def __init__(self, params: ConcatenationDerivativeDeriverParams):
        """
        Initialize a ConcatenationDerivativeDeriver
        with the provided parameters.

        Args:
            params (ConcatenationDerivativeDeriverParams):
                Parameters for the ConcatenationDerivativeDeriver.
        """
        super().__init__()

        self._derivative_type = params.derivative_type
        self._separator = params.separator

    async def derive(self, episode_cluster: EpisodeCluster) -> list[Derivative]:
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
