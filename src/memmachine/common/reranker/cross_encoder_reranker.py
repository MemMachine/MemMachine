"""
Cross-encoder based reranker implementation.
"""

import asyncio
from typing import Any

from ..configuration.reranker_conf import CrossEncoderRerankerConf
from .reranker import Reranker


class CrossEncoderReranker(Reranker):
    _cross_encoders: dict[str, Any] = {}

    """
    Reranker that uses a cross-encoder model to score candidates
    based on their relevance to the query.
    """

    def __init__(self, params: CrossEncoderRerankerConf):
        """
        Initialize a CrossEncoderReranker
        with the provided parameters.

        Args:
            params (CrossEncoderRerankerParams):
                Parameters for the CrossEncoderReranker.
        """
        super().__init__()
        self._params = params

    @property
    def cross_encoder(self):
        model_name = self._params.model_name
        if model_name not in self._cross_encoders:
            from sentence_transformers import CrossEncoder

            self._cross_encoders[model_name] = CrossEncoder(model_name)
        return self._cross_encoders[model_name]

    async def score(self, query: str, candidates: list[str]) -> list[float]:
        scores = [
            float(score)
            for score in await asyncio.to_thread(
                self.cross_encoder.predict,
                [(query, candidate) for candidate in candidates],
                show_progress_bar=False,
            )
        ]
        return scores
