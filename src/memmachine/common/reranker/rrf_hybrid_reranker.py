"""
RRF hybrid reranker implementation.
"""

import asyncio
from typing import Any

from .reranker import Reranker


class RRFHybridReranker(Reranker):
    """
    Reranker that combines scores from multiple rerankers
    using Reciprocal Rank Fusion (RRF).
    """

    def __init__(self, config: dict[str, Any]):
        """
        Initialize a RRFHybridReranker with the provided configuration.

        Args:
            config (dict[str, Any]):
                Configuration dictionary containing:
                - rerankers (list[Reranker]):
                    List of reranker instances to combine.
                - k (int, optional):
                    The k parameter for RRF (default: 60).

        Raises:
            ValueError:
                If configuration argument values are missing or invalid.
            TypeError:
                If configuration argument values are of incorrect type.
        """
        super().__init__()

        self._rerankers = config.get("rerankers", [])
        if not isinstance(self._rerankers, list):
            raise TypeError("Rerankers must be provided in a list")
        if not all(isinstance(reranker, Reranker) for reranker in self._rerankers):
            raise TypeError("All items in rerankers list must be Reranker instances")
        if len(self._rerankers) == 0:
            raise ValueError("At least one reranker must be provided")

        self._k = config.get("k", 60)
        if not isinstance(self._k, int) or self._k < 0:
            raise ValueError("k must be a nonnegative integer")

    async def score(self, query: str, candidates: list[str]) -> list[float]:
        score_tasks = [
            reranker.score(query, candidates) for reranker in self._rerankers
        ]
        score_lists = await asyncio.gather(*score_tasks)

        rank_lists = [
            RRFHybridReranker._get_ranks(score_list) for score_list in score_lists
        ]

        scores = [
            sum(
                [1 / (self._k + rank_list[candidate_index]) for rank_list in rank_lists]
            )
            for candidate_index in range(len(candidates))
        ]
        return scores

    @staticmethod
    def _get_ranks(scores: list[float]) -> list[int]:
        """
        Convert a list of scores into ranks,
        with rank 1 being the highest score.

        Args:
            scores (list[float]):
                List of scores to convert to ranks.

        Returns:
            list[int]:
                List of ranks corresponding to the input scores.
        """
        n = len(scores)
        sorted_indices = sorted(range(n), key=lambda index: scores[index], reverse=True)
        ranks = [0] * n
        for rank, index in enumerate(sorted_indices):
            ranks[index] = rank + 1

        return ranks
