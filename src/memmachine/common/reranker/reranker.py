"""
Abstract base class for a reranker.

Defines the interface for scoring and reranking candidates
based on their relevance to a query.
"""

from abc import ABC, abstractmethod


class Reranker(ABC):
    """
    Abstract base class for a reranker.
    """

    async def rerank(self, query: str, candidates: list[str]) -> list[str]:
        """
        Rerank the candidates based on their relevance to the query.

        Args:
            query (str):
                The input query string.
            candidates (list[str]):
                A list of candidate strings to be reranked.

        Returns:
            list[str]:
                The reranked list of candidates,
                sorted by score in descending order.
        """
        scores = await self.score(query, candidates)
        score_map = dict(zip(candidates, scores))

        return sorted(
            candidates,
            key=lambda candidate: score_map[candidate],
            reverse=True,
        )

    @abstractmethod
    async def score(self, query: str, candidates: list[str]) -> list[float]:
        """
        Compute relevance scores for each candidate
        with respect to the query.

        Args:
            query (str):
                The input query string.
            candidates (list[str]):
                A list of candidate strings to be scored.

        Returns:
            list[float]:
                A list of scores corresponding to each candidate.
        """
        raise NotImplementedError
