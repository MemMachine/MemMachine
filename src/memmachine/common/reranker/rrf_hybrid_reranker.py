"""
RRF hybrid reranker implementation.
"""

import asyncio
from collections import defaultdict

from pydantic import BaseModel, Field, InstanceOf

from .reranker import Reranker


class RRFHybridRerankerParams(BaseModel):
    """
    Parameters for RRFHybridReranker.

    Attributes:
        rerankers (list[Reranker]):
            List of rerankers to combine.
        k (int):
            The k parameter for Reciprocal Rank Fusion (default: 60).
    """

    rerankers: list[InstanceOf[Reranker]] = Field(
        ..., description="List of rerankers to combine", min_length=1
    )
    k: int = Field(60, description="The k parameter for Reciprocal Rank Fusion", ge=0)


class RRFHybridReranker(Reranker):
    """
    Reranker that combines scores from multiple rerankers
    using Reciprocal Rank Fusion (RRF).
    """

    def __init__(self, params: RRFHybridRerankerParams):
        """
        Initialize a RRFHybridReranker with the provided parameters.

        Args:
            params (RRFHybridRerankerParams):
                Parameters for the RRFHybridReranker.
        """
        super().__init__()

        self._rerankers = params.rerankers
        self._k = params.k

    async def score(self, query: str, candidates: list[str]) -> list[float]:
        rerank_tasks = [
            reranker.rerank(query, candidates) for reranker in self._rerankers
        ]
        rankings = await asyncio.gather(*rerank_tasks)

        score_map: defaultdict[str, float] = defaultdict(float)

        for ranking in rankings:
            for rank, candidate in enumerate(ranking, start=1):
                score_map[candidate] += 1 / (self._k + rank)

        scores = [score_map[candidate] for candidate in candidates]

        return scores
