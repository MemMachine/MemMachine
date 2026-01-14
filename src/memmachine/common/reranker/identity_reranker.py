"""Identity reranker implementation."""

from .reranker import Reranker


class IdentityReranker(Reranker):
    """Reranker that returns candidates in their original order."""

    async def score(self, query: str, candidates: list[str]) -> list[float]:  # noqa: ARG002
        """Assign descending scores to preserve original order."""
        scores = list(map(float, reversed(range(len(candidates)))))
        return scores
