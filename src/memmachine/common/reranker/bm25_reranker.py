"""BM25-based reranker implementation."""

import asyncio
from collections.abc import Callable

from pydantic import BaseModel, Field
from rank_bm25 import BM25Okapi

from .reranker import Reranker


class BM25RerankerParams(BaseModel):
    """Parameters for BM25Reranker."""

    k1: float = Field(1.5, description="BM25 k1 parameter")
    b: float = Field(0.75, description="BM25 b parameter")
    epsilon: float = Field(0.25, description="BM25 epsilon parameter")
    tokenize: Callable[[str], list[str]] = Field(
        ...,
        description="Tokenizer function to split text into tokens",
    )


class BM25Reranker(Reranker):
    """Reranker that uses the BM25 algorithm to score candidates."""

    def __init__(self, params: BM25RerankerParams) -> None:
        """Initialize a BM25Reranker with the provided parameters."""
        super().__init__()

        self._k1 = params.k1
        self._b = params.b
        self._epsilon = params.epsilon
        self._tokenize = params.tokenize

    async def score(self, query: str, candidates: list[str]) -> list[float]:
        """Score candidates for a query using BM25."""
        tokenized_query_future = asyncio.to_thread(self._tokenize, query)
        tokenized_candidates_future = asyncio.to_thread(
            self._tokenize_multiple,
            candidates,
        )

        tokenized_query = await tokenized_query_future
        tokenized_candidates = await tokenized_candidates_future

        if not any(tokenized_candidates):
            # There are no tokens in the corpus.
            return [0.0 for _ in candidates]

        # There is at least one token in the corpus.
        bm25 = BM25Okapi(
            tokenized_candidates,
            k1=self._k1,
            b=self._b,
            epsilon=self._epsilon,
        )

        scores = [float(score) for score in bm25.get_scores(tokenized_query)]

        return scores

    def _tokenize_multiple(self, corpus: list[str]) -> list[list[str]]:
        return [self._tokenize(document) for document in corpus]
