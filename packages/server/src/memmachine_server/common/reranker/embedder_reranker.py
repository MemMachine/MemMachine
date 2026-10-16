"""Embedder-based reranker implementation."""

from pydantic import BaseModel, Field, InstanceOf

from memmachine_server.common.embedder import Embedder
from memmachine_server.common.utils import compute_cosine_similarity

from .reranker import Reranker


class EmbedderRerankerParams(BaseModel):
    """Parameters for EmbedderReranker."""

    embedder: InstanceOf[Embedder] = Field(
        ...,
        description="An instance of an Embedder to use for generating embeddings",
    )


class EmbedderReranker(Reranker):
    """Reranker that uses an embedder to score candidate relevance."""

    def __init__(self, params: EmbedderRerankerParams) -> None:
        """Initialize an EmbedderReranker with the provided configuration."""
        super().__init__()

        self._embedder = params.embedder

    async def score(self, query: str, candidates: list[str]) -> list[float]:
        """Score candidates for a query using embedder similarity."""
        if len(candidates) == 0:
            return []

        query_embedding = (await self._embedder.search_embed([query]))[0]
        candidate_embeddings = await self._embedder.ingest_embed(candidates)

        return compute_cosine_similarity(query_embedding, candidate_embeddings)
