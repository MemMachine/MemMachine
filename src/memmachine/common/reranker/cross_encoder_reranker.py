"""Cross-encoder based reranker implementation."""

import asyncio

from pydantic import BaseModel, Field, InstanceOf
from sentence_transformers import CrossEncoder

from .reranker import Reranker


class CrossEncoderRerankerParams(BaseModel):
    """Parameters for CrossEncoderReranker."""

    cross_encoder: InstanceOf[CrossEncoder] = Field(
        ...,
        description="The cross-encoder model to use for reranking",
    )


class CrossEncoderReranker(Reranker):
    """Reranker that uses a cross-encoder model to score candidates."""

    def __init__(self, params: CrossEncoderRerankerParams) -> None:
        """Initialize a CrossEncoderReranker with provided parameters."""
        super().__init__()

        self._cross_encoder = params.cross_encoder

    async def score(self, query: str, candidates: list[str]) -> list[float]:
        """Score candidates for a query using the cross-encoder."""
        scores = [
            float(score)
            for score in await asyncio.to_thread(
                self._cross_encoder.predict,
                [(query, candidate) for candidate in candidates],
                show_progress_bar=False,
            )
        ]
        return scores
