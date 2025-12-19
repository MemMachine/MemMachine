"""Cross-encoder based reranker implementation."""

import asyncio

from pydantic import BaseModel, Field, InstanceOf
from sentence_transformers import CrossEncoder

from memmachine.common.utils import chunk_text, unflatten_like

from .reranker import Reranker


class CrossEncoderRerankerParams(BaseModel):
    """Parameters for CrossEncoderReranker."""

    cross_encoder: InstanceOf[CrossEncoder] = Field(
        ...,
        description="The cross-encoder model to use for reranking",
    )
    max_input_length: int | None = Field(
        default=None,
        description="Maximum input length for the model (in Unicode code points)",
        gt=0,
    )


class CrossEncoderReranker(Reranker):
    """Reranker that uses a cross-encoder model to score candidates."""

    def __init__(self, params: CrossEncoderRerankerParams) -> None:
        """Initialize a CrossEncoderReranker with provided parameters."""
        super().__init__()

        self._cross_encoder = params.cross_encoder
        self._max_input_length = params.max_input_length

    async def score(self, query: str, candidates: list[str]) -> list[float]:
        """Score candidates for a query using the cross-encoder."""
        query = query[: self._max_input_length] if self._max_input_length else query

        candidates_chunks = [
            chunk_text(candidate, self._max_input_length)
            if self._max_input_length and candidate
            else [candidate]
            for candidate in candidates
        ]

        chunks = [
            chunk
            for candidate_chunks in candidates_chunks
            for chunk in candidate_chunks
        ]

        chunk_scores = [
            float(score)
            for score in await asyncio.to_thread(
                self._cross_encoder.predict,
                [(query, chunk) for chunk in chunks],
                show_progress_bar=False,
            )
        ]

        candidates_chunk_scores = unflatten_like(chunk_scores, candidates_chunks)

        # Take the maximum score among chunks for each candidate.
        return [
            max(candidate_chunk_scores)
            for candidate_chunk_scores in candidates_chunk_scores
        ]
