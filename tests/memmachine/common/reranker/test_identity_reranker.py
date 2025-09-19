import pytest

from memmachine.common.reranker.identity_reranker import IdentityReranker


@pytest.fixture
def reranker():
    return IdentityReranker()


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

    in_decreasing_order = True
    previous_score = float("inf")
    for score in scores:
        if score >= previous_score:
            in_decreasing_order = False
            break
        previous_score = score

    assert in_decreasing_order
