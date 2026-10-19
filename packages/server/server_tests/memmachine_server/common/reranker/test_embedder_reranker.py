from unittest.mock import MagicMock

import pytest

from memmachine_server.common.embedder import Embedder
from memmachine_server.common.reranker.embedder_reranker import (
    EmbedderReranker,
    EmbedderRerankerParams,
)
from server_tests.memmachine_server.common.reranker.fake_embedder import FakeEmbedder


@pytest.fixture
def embedder():
    return FakeEmbedder()


@pytest.fixture
def reranker(embedder):
    return EmbedderReranker(EmbedderRerankerParams(embedder=embedder))


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
async def test_score():
    embedder = MagicMock(spec=Embedder)
    reranker = EmbedderReranker(EmbedderRerankerParams(embedder=embedder))

    embedder.ingest_embed.return_value = [[1.0, 2.0], [1.5, 1.5]]
    embedder.search_embed.return_value = [[1.0, 1.0]]

    scores = await reranker.score("query", ["candidate1", "candidate2"])
    assert scores[0] < scores[1]
