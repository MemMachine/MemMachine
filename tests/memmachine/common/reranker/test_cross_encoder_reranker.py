from unittest.mock import MagicMock

import pytest
from sentence_transformers import CrossEncoder

from memmachine.common.reranker.cross_encoder_reranker import (
    CrossEncoderReranker,
    CrossEncoderRerankerParams,
)


@pytest.fixture
def cross_encoder():
    mock_cross_encoder = MagicMock(spec=CrossEncoder)
    return mock_cross_encoder


@pytest.fixture
def reranker(cross_encoder):
    return CrossEncoderReranker(
        CrossEncoderRerankerParams(
            cross_encoder=cross_encoder,
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
async def test_shape(reranker, cross_encoder, query, candidates):
    mock_scores = [0.0] * len(candidates)
    cross_encoder.predict.return_value = mock_scores

    scores = await reranker.score(query, candidates)
    assert isinstance(scores, list)
    assert len(scores) == len(candidates)
    assert all(isinstance(score, float) for score in scores)


@pytest.mark.asyncio
async def test_score(reranker, cross_encoder):
    mock_scores = [0.9, 0.1, 0.5]
    cross_encoder.predict.return_value = mock_scores

    query = "query"
    candidates = ["candidate1", "candidate2", "candidate3"]
    scores = await reranker.score(query, candidates)

    assert scores == mock_scores
