import pytest

from memmachine.common.reranker.cohere_reranker import (
    CohereReranker,
    CohereRerankerParams,
)

pytestmark = pytest.mark.integration


@pytest.fixture
def reranker(cohere_client):
    return CohereReranker(
        CohereRerankerParams(
            client=cohere_client,
            model="rerank-english-v3.0",
        )
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
async def test_rerank_sanity(reranker):
    query = "What is the capital of France?"
    candidates = [
        "The capital of France is Paris.",
        "The capital of Germany is Berlin.",
        "Some apples are red.",
    ]

    scores = await reranker.score(query, candidates)

    assert len(scores) == len(candidates)
    assert scores[0] > scores[1] > scores[2]


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
