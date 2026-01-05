"""Cohere reranker implementation."""

import asyncio
import logging
from typing import Any

import cohere
from pydantic import BaseModel, Field

from memmachine.common.data_types import ExternalServiceAPIError

from .reranker import Reranker

logger = logging.getLogger(__name__)


class CohereRerankerParams(BaseModel):
    """Configuration parameters for CohereReranker."""

    client: Any = Field(
        ...,
        description="Cohere client instance for making API calls",
    )
    model: str = Field(
        "rerank-english-v3.0",
        description="Cohere rerank model",
    )


class CohereReranker(Reranker):
    """Reranker using Cohere's rerank API."""

    def __init__(self, params: CohereRerankerParams) -> None:
        """Initialize a CohereReranker with the provided parameters."""
        super().__init__()

        self._client = params.client
        self._model = params.model

    async def score(self, query: str, candidates: list[str]) -> list[float]:
        """Score candidates using Cohere's rerank API."""
        if len(candidates) == 0:
            return []

        query = query.strip() or "."
        if all(not candidate.strip() for candidate in candidates):
            return [0.0] * len(candidates)

        # Build request parameters
        def _call_rerank() -> cohere.RerankResponse:
            return self._client.rerank(
                model=self._model,
                query=query,
                documents=candidates,
            )

        try:
            response = await asyncio.to_thread(_call_rerank)
        except Exception as e:
            error_message = (
                f"Failed to score candidates with Cohere model {self._model} "
                f"due to {type(e).__name__}: {e}"
            )
            logger.exception(error_message)
            raise ExternalServiceAPIError(error_message) from e

        # Cohere returns ranked order — map scores back to original positions
        scores = [0.0] * len(candidates)
        for result in response.results:
            scores[result.index] = float(result.relevance_score)

        return scores
