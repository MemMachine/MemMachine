import pytest

from memmachine.common.reranker.embedder_reranker import EmbedderReranker


@pytest.fixture
def reranker(mock_llm_embedder):
    return EmbedderReranker({"embedder": mock_llm_embedder})


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


def test_invalid_embedder():
    with pytest.raises(ValueError):
        EmbedderReranker(config={"embedder": None})

    with pytest.raises(TypeError):
        EmbedderReranker(config={"embedder": "a string: not an Embedder"})
