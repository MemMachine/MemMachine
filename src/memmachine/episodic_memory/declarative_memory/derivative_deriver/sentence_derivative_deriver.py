"""
A derivative deriver that splits episode content into sentences
and creates derivatives for each sentence.
"""

from uuid import uuid4

from nltk import sent_tokenize
from pydantic import BaseModel, Field

from ..data_types import ContentType, Derivative, EpisodeCluster
from .derivative_deriver import DerivativeDeriver


class SentenceDerivativeDeriverParams(BaseModel):
    """
    Parameters for SentenceDerivativeDeriver.

    Attributes:
        derivative_type (str):
            The type to assign to the derived derivatives
            (default: "sentence").
    """

    derivative_type: str = Field(
        "sentence",
        description="The type to assign to the derived derivatives",
    )


class SentenceDerivativeDeriver(DerivativeDeriver):
    """
    Derivative deriver that splits episode content into sentences
    and creates derivatives for each sentence.
    """

    def __init__(self, params: SentenceDerivativeDeriverParams):
        """
        Initialize a SentenceDerivativeDeriver
        with the provided parameters.

        Args:
            params (SentenceDerivativeDeriverParams):
                Parameters for the SentenceDerivativeDeriver.
        """
        super().__init__()

        self._derivative_type = params.derivative_type

    async def derive(self, episode_cluster: EpisodeCluster) -> list[Derivative]:
        return [
            Derivative(
                uuid=uuid4(),
                derivative_type=self._derivative_type,
                content_type=ContentType.STRING,
                content=sentence,
                timestamp=episode.timestamp,
                filterable_properties=episode.filterable_properties,
                user_metadata=episode.user_metadata,
            )
            for episode in episode_cluster.episodes
            for line in episode.content.splitlines()
            for sentence in sent_tokenize(line)
        ]
