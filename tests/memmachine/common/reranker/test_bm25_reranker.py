import re

import pytest

from memmachine import setup_nltk
from memmachine.common.reranker.bm25_reranker import BM25Reranker, BM25RerankerParams


@pytest.fixture
def reranker():
    setup_nltk()
    return BM25Reranker(
        BM25RerankerParams(
            tokenize=lambda text: re.sub(r"\W+", " ", text).lower().split()
        )
    )


@pytest.fixture(params=["Are tomatoes fruits?", ""])
def query(request):
    return request.param


@pytest.fixture(
    params=[
        ["Apples are fruits.", "Tomatoes are red."],
        ["Apples are fruits.", "Tomatoes are red.", ""],
        [""],
        [],
    ]
)
def candidates(request):
    return request.param


@pytest.mark.asyncio
async def test_score(reranker, query, candidates):
    scores = await reranker.score(query, candidates)
    assert isinstance(scores, list)
    assert len(scores) == len(candidates)
    assert all(isinstance(score, float) for score in scores)


@pytest.mark.asyncio
async def test_score_values(reranker):
    query = "What is the capital of France?"
    candidates = [
        "Hello, world!",
        "Berlin is the capital of Germany.",
        "Paris is the capital of France.",
    ]

    scores = await reranker.score(query, candidates)

    assert scores == sorted(scores)
    assert scores != reversed(scores)
