"""Abstract base class for rerankers and scoring interfaces."""

from abc import ABC, abstractmethod


class Reranker(ABC):
    """Abstract base class for a reranker."""

    async def rerank(self, query: str, candidates: list[str]) -> list[str]:
        """Rerank candidates based on their relevance to the query."""
        scores = await self.score(query, candidates)
        score_map = dict(zip(candidates, scores, strict=True))

        return sorted(
            candidates,
            key=lambda candidate: score_map[candidate],
            reverse=True,
        )

    @abstractmethod
    async def score(self, query: str, candidates: list[str]) -> list[float]:
        """Compute relevance scores for each candidate."""
        raise NotImplementedError
