import pytest
from sentence_transformers import CrossEncoder

from memmachine.common.reranker.cross_encoder_reranker import (
    CrossEncoderReranker,
    CrossEncoderRerankerParams,
)

pytestmark = pytest.mark.integration


@pytest.fixture
def cross_encoder():
    return CrossEncoder("cross-encoder/ms-marco-MiniLM-L6-v2")


@pytest.fixture
def reranker(cross_encoder):
    return CrossEncoderReranker(
        CrossEncoderRerankerParams(
            cross_encoder=cross_encoder,
            max_input_length=2000,
        ),
    )


@pytest.fixture(params=["Are tomatoes fruits?", ".", " ", ""])
def query(request):
    return request.param


@pytest.fixture(
    params=[
        ["Apples are fruits.", "Tomatoes are red."],
        ["Apples are fruits.", "Tomatoes are red.", ""],
        ["."],
        [" "],
        [""],
        [],
    ],
)
def candidates(request):
    return request.param


@pytest.mark.asyncio
async def test_shape(reranker, query, candidates):
    scores = await reranker.score(query, candidates)
    assert isinstance(scores, list)
    assert len(scores) == len(candidates)
    assert all(isinstance(score, float) for score in scores)


@pytest.mark.asyncio
async def test_score(reranker):
    query = "Are tomatoes fruits?"
    candidates = ["Apples are fruits.", "Tomatoes are red.", "Tomatoes are fruits."]
    scores = await reranker.score(query, candidates)

    assert scores[2] > scores[0]
    assert scores[2] > scores[1]


@pytest.mark.asyncio
async def test_large_query(reranker):
    query = "👩‍💻" * 100000
    candidates = ["Candidate 1", "Candidate 2"]

    await reranker.rerank(query, candidates)


@pytest.mark.asyncio
async def test_large_document(reranker):
    query = "Query"
    candidates = ["👩‍💻" * 100000, "Candidate 2"]

    await reranker.rerank(query, candidates)
