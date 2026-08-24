import asyncio
from types import SimpleNamespace

import pytest

from memmachine_server.common.data_types import ExternalServiceAPIError
from memmachine_server.common.reranker.cohere_reranker import (
    CohereReranker,
    CohereRerankerParams,
)


class StubAsyncCohereClient:
    """Stub of cohere.AsyncClientV2 returning canned rerank results."""

    def __init__(self, relevance_scores: list[float] | None = None):
        self.calls: list[dict] = []
        self._relevance_scores = relevance_scores

    async def rerank(self, *, model, query, documents):
        self.calls.append({"model": model, "query": query, "documents": documents})

        relevance_scores = self._relevance_scores or [
            1.0 / (index + 1) for index in range(len(documents))
        ]

        # Return results in descending relevance order like the real API.
        results = sorted(
            (
                SimpleNamespace(index=index, relevance_score=relevance_score)
                for index, relevance_score in enumerate(relevance_scores)
            ),
            key=lambda result: result.relevance_score,
            reverse=True,
        )
        return SimpleNamespace(results=results)


@pytest.fixture
def client():
    return StubAsyncCohereClient()


@pytest.fixture
def reranker(client):
    return CohereReranker(
        CohereRerankerParams(
            client=client,
            model="rerank-v3.5",
        )
    )


@pytest.mark.asyncio
async def test_scores_map_back_to_original_positions(client):
    reranker = CohereReranker(
        CohereRerankerParams(
            client=StubAsyncCohereClient(relevance_scores=[0.2, 0.9, 0.5]),
            model="rerank-v3.5",
        )
    )

    scores = await reranker.score("query", ["a", "b", "c"])

    assert scores == [0.2, 0.9, 0.5]


@pytest.mark.asyncio
async def test_empty_candidates_do_not_call_api(reranker, client):
    assert await reranker.score("query", []) == []
    assert client.calls == []


@pytest.mark.asyncio
async def test_blank_candidates_do_not_call_api(reranker, client):
    assert await reranker.score("query", ["", "  "]) == [0.0, 0.0]
    assert client.calls == []


@pytest.mark.asyncio
async def test_blank_query_is_replaced(reranker, client):
    await reranker.score("  ", ["a"])

    assert client.calls[0]["query"] == "."


@pytest.mark.asyncio
async def test_request_parameters_passed_through(reranker, client):
    await reranker.score("query", ["a", "b"])

    assert client.calls == [
        {"model": "rerank-v3.5", "query": "query", "documents": ["a", "b"]}
    ]


@pytest.mark.asyncio
async def test_client_error_wrapped(reranker, client):
    class FailingClient:
        async def rerank(self, *, model, query, documents):
            raise RuntimeError("boom")

    reranker = CohereReranker(
        CohereRerankerParams(client=FailingClient(), model="rerank-v3.5")
    )

    with pytest.raises(ExternalServiceAPIError):
        await reranker.score("query", ["a"])


@pytest.mark.asyncio
async def test_concurrent_scores_are_in_flight_simultaneously():
    num_calls = 64
    in_flight = 0
    all_in_flight = asyncio.Event()

    class BlockingClient(StubAsyncCohereClient):
        async def rerank(self, *, model, query, documents):
            nonlocal in_flight
            in_flight += 1
            if in_flight == num_calls:
                all_in_flight.set()
            # Deadlocks (and times out) if calls are serialized by a
            # bounded worker pool instead of running concurrently.
            await all_in_flight.wait()
            return await super().rerank(model=model, query=query, documents=documents)

    reranker = CohereReranker(
        CohereRerankerParams(client=BlockingClient(), model="rerank-v3.5")
    )

    await asyncio.wait_for(
        asyncio.gather(*(reranker.score("query", ["a"]) for _ in range(num_calls))),
        timeout=5,
    )
