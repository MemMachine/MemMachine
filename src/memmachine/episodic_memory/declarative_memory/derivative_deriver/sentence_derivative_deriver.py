"""
A derivative deriver that splits episode content into sentences
and creates derivatives for each sentence.
"""

from typing import Any
from uuid import uuid4

from nltk import sent_tokenize

from ..data_types import ContentType, Derivative, EpisodeCluster
from .derivative_deriver import DerivativeDeriver


class SentenceDerivativeDeriver(DerivativeDeriver):
    """
    Derivative deriver that splits episode content into sentences
    and creates derivatives for each sentence.
    """

    def __init__(self, config: dict[str, Any] = {}):
        """
        Initialize a SentenceDerivativeDeriver
        with the provided configuration.

        Args:
            config (dict[str, Any], optional):
                Configuration dictionary containing:
                - derivative_type (str, optional):
                  The type to assign
                  to the derived derivatives (default: "sentence").
        """
        super().__init__()

        self._derivative_type = config.get("derivative_type", "sentence")

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
            for sentence in sent_tokenize(episode.content)
        ]
