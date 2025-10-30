import pytest

from memmachine.common.reranker import Reranker
from memmachine.common.reranker.rrf_hybrid_reranker import (
    RRFHybridReranker,
    RRFHybridRerankerParams,
)


class FakeReranker(Reranker):
    def __init__(self, scores: list[float]):
        super().__init__()
        self._scores = scores

    async def score(self, query: str, candidates: list[str]) -> list[float]:
        return self._scores


@pytest.fixture
def reranker():
    return RRFHybridReranker(
        RRFHybridRerankerParams(
            rerankers=[
                FakeReranker([1.0, 2.0, 4.0]),
                FakeReranker([2.0, 1.0, 4.0]),
            ]
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
async def test_shape(query, candidates):
    reranker = RRFHybridReranker(
        RRFHybridRerankerParams(
            rerankers=[
                FakeReranker([-1.0] * len(candidates)),
                FakeReranker([1.0] * len(candidates)),
            ]
        )
    )

    scores = await reranker.score(query, candidates)
    assert isinstance(scores, list)
    assert len(scores) == len(candidates)
    assert all(isinstance(score, float) for score in scores)


@pytest.mark.asyncio
async def test_score(reranker):
    query = "query"
    candidates = ["candidate1", "candidate2", "candidate3"]
    scores = await reranker.score(query, candidates)

    assert scores[0] == scores[1] < scores[2]
