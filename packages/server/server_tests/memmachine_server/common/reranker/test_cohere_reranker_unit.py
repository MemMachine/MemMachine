import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import cohere
import pytest

from memmachine_server.common.data_types import ExternalServiceAPIError
from memmachine_server.common.reranker.cohere_reranker import (
    CohereReranker,
    CohereRerankerParams,
)


def stub_client(relevance_scores: list[float] | None = None) -> AsyncMock:
    """Stub of cohere.AsyncClientV2 whose rerank returns canned results."""
    client = AsyncMock(spec=cohere.AsyncClientV2)

    async def rerank(*, model, query, documents):
        scores = relevance_scores or [
            1.0 / (index + 1) for index in range(len(documents))
        ]

        # Return results in descending relevance order like the real API.
        results = sorted(
            (
                SimpleNamespace(index=index, relevance_score=score)
                for index, score in enumerate(scores)
            ),
            key=lambda result: result.relevance_score,
            reverse=True,
        )
        return SimpleNamespace(results=results)

    client.rerank.side_effect = rerank
    return client


@pytest.fixture
def client():
    return stub_client()


@pytest.fixture
def reranker(client):
    return CohereReranker(CohereRerankerParams(client=client, model="rerank-v3.5"))


@pytest.mark.asyncio
async def test_scores_map_back_to_original_positions():
    reranker = CohereReranker(
        CohereRerankerParams(
            client=stub_client(relevance_scores=[0.2, 0.9, 0.5]),
            model="rerank-v3.5",
        )
    )

    scores = await reranker.score("query", ["a", "b", "c"])

    assert scores == [0.2, 0.9, 0.5]


@pytest.mark.asyncio
async def test_empty_candidates_do_not_call_api(reranker, client):
    assert await reranker.score("query", []) == []
    client.rerank.assert_not_awaited()


@pytest.mark.asyncio
async def test_blank_candidates_do_not_call_api(reranker, client):
    assert await reranker.score("query", ["", "  "]) == [0.0, 0.0]
    client.rerank.assert_not_awaited()


@pytest.mark.asyncio
async def test_blank_query_is_replaced(reranker, client):
    await reranker.score("  ", ["a"])

    client.rerank.assert_awaited_once_with(
        model="rerank-v3.5", query=".", documents=["a"]
    )


@pytest.mark.asyncio
async def test_request_parameters_passed_through(reranker, client):
    await reranker.score("query", ["a", "b"])

    client.rerank.assert_awaited_once_with(
        model="rerank-v3.5", query="query", documents=["a", "b"]
    )


@pytest.mark.asyncio
async def test_client_error_wrapped(reranker, client):
    client.rerank.side_effect = RuntimeError("boom")

    with pytest.raises(ExternalServiceAPIError):
        await reranker.score("query", ["a"])


@pytest.mark.asyncio
async def test_concurrent_scores_are_in_flight_simultaneously(reranker, client):
    num_calls = 64
    in_flight = 0
    all_in_flight = asyncio.Event()

    async def rerank(*, model, query, documents):
        nonlocal in_flight
        in_flight += 1
        if in_flight == num_calls:
            all_in_flight.set()
        # Deadlocks (and times out) if calls are serialized by a
        # bounded worker pool instead of running concurrently.
        await all_in_flight.wait()
        return SimpleNamespace(results=[SimpleNamespace(index=0, relevance_score=1.0)])

    client.rerank.side_effect = rerank

    await asyncio.wait_for(
        asyncio.gather(*(reranker.score("query", ["a"]) for _ in range(num_calls))),
        timeout=5,
    )
