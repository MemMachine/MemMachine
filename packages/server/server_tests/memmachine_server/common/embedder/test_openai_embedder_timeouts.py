"""Operation deadlines, cancellation, and retry behavior without provider I/O."""

import asyncio
from contextlib import asynccontextmanager
from types import SimpleNamespace
from unittest.mock import ANY, AsyncMock, MagicMock

import httpx
import openai
import pytest

from memmachine_server.common.data_types import ExternalServiceAPIError
from memmachine_server.common.embedder.openai_embedder import (
    OpenAIEmbedder,
    OpenAIEmbedderParams,
)
from memmachine_server.common.metrics_factory import MetricsFactory

pytestmark = pytest.mark.asyncio


def _response(count=1):
    return SimpleNamespace(
        data=[SimpleNamespace(embedding=[1.0, 0.0]) for _ in range(count)],
        usage=SimpleNamespace(prompt_tokens=count, total_tokens=count),
    )


def _embedder(create, *, search_timeout=None, ingest_timeout=None, **kwargs):
    client = AsyncMock(spec=openai.AsyncOpenAI)
    client.embeddings.create = AsyncMock(side_effect=create)
    return OpenAIEmbedder(
        OpenAIEmbedderParams(
            client=client,
            model="test-model",
            dimensions=2,
            search_timeout_seconds=search_timeout,
            ingest_timeout_seconds=ingest_timeout,
            **kwargs,
        )
    )


@asynccontextmanager
async def _running(coroutine):
    """Clean up stalled fake requests even when an assertion fails."""
    task = asyncio.create_task(coroutine)
    try:
        async with asyncio.timeout(5):
            yield task
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)


@pytest.mark.parametrize("operation", ["search_embed", "ingest_embed"])
@pytest.mark.parametrize("split", ["batches", "clusters"])
async def test_timeout_discards_partial_results_and_cleans_up(operation, split):
    started = asyncio.Event()
    pending = asyncio.Event()
    requests = []
    cancelled = []
    metrics = MagicMock(spec=MetricsFactory)

    async def create(**kwargs):
        inputs = kwargs["input"]
        requests.append(asyncio.current_task())
        if len(requests) == 3:
            started.set()
        if inputs[0].startswith("a"):
            return _response(len(inputs))
        try:
            await pending.wait()
        except asyncio.CancelledError:
            cancelled.append(inputs[0][0])
            raise

    embedder = _embedder(
        create,
        search_timeout=0.1,
        ingest_timeout=0.1,
        batch_size=1 if split == "batches" else None,
        metrics_factory=metrics,
    )
    # Each long input fills one provider request without changing its limits.
    inputs = ["a", "b", "c"] if split == "batches" else [s * 75_000 for s in "abc"]

    async with _running(getattr(embedder, operation)(inputs)) as task:
        await started.wait()
        with pytest.raises(ExternalServiceAPIError) as exc_info:
            await task

    assert isinstance(exc_info.value.__cause__, TimeoutError)
    assert sorted(cancelled) == ["b", "c"]
    assert all(request is not None and request.done() for request in requests)
    # A partially successful batch is still one failed embedding operation.
    metrics.get_histogram.return_value.observe.assert_called_once_with(
        value=ANY,
        labels={"operation": operation, "status": "error"},
    )


async def test_application_retry_backoff_uses_the_operation_budget(monkeypatch):
    backoff_started = asyncio.Event()
    backoff_cancelled = asyncio.Event()
    pending = asyncio.Event()
    request = httpx.Request("POST", "https://embedder.invalid/embeddings")
    error = openai.InternalServerError(
        "temporary failure", response=httpx.Response(500, request=request), body=None
    )
    create = AsyncMock(side_effect=error)

    async def backoff(delay):
        assert delay == 1
        backoff_started.set()
        try:
            await pending.wait()
        finally:
            backoff_cancelled.set()

    monkeypatch.setattr(asyncio, "sleep", backoff)
    embedder = _embedder(create, search_timeout=0.1)
    async with _running(embedder.search_embed(["query"], max_attempts=3)) as task:
        await backoff_started.wait()
        with pytest.raises(ExternalServiceAPIError) as exc_info:
            await task

    assert isinstance(exc_info.value.__cause__, TimeoutError)
    assert backoff_cancelled.is_set()
    create.assert_awaited_once()


async def test_search_timeout_does_not_cancel_concurrent_ingestion():
    search_started = asyncio.Event()
    ingest_started = asyncio.Event()
    release_ingest = asyncio.Event()
    search_pending = asyncio.Event()

    async def create(**kwargs):
        if kwargs["input"] == ["query"]:
            search_started.set()
            await search_pending.wait()
        else:
            ingest_started.set()
            await release_ingest.wait()
        return _response()

    embedder = _embedder(create, search_timeout=0.1, ingest_timeout=3)
    async with (
        _running(embedder.ingest_embed(["memory"])) as ingest_task,
        _running(embedder.search_embed(["query"])) as search_task,
    ):
        await search_started.wait()
        await ingest_started.wait()
        with pytest.raises(ExternalServiceAPIError):
            await search_task
        assert not ingest_task.done()
        release_ingest.set()
        assert await ingest_task == [[1.0, 0.0]]


@pytest.mark.parametrize("operation", ["search_embed", "ingest_embed"])
async def test_caller_cancellation_is_preserved_and_cleans_up(operation):
    started = asyncio.Event()
    cancelled = asyncio.Event()
    pending = asyncio.Event()

    async def create(**kwargs):
        started.set()
        try:
            await pending.wait()
        finally:
            cancelled.set()

    embedder = _embedder(create, search_timeout=3, ingest_timeout=3)
    async with _running(getattr(embedder, operation)(["text"])) as task:
        await started.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    assert cancelled.is_set()


async def test_dimensions_fallback_shares_the_original_deadline():
    request = httpx.Request("POST", "https://embedder.invalid/embeddings")
    calls = []
    fallback_cancelled = asyncio.Event()
    # Either response fits the budget alone; the two responses together do not.
    # Use timer-driven fake responses, without asserting wall-clock durations.
    response_delay = 0.3
    budget = 0.5

    async def create(**kwargs):
        calls.append(kwargs)
        ready = asyncio.Event()
        timer = asyncio.get_running_loop().call_later(response_delay, ready.set)
        try:
            await ready.wait()
        except asyncio.CancelledError:
            fallback_cancelled.set()
            raise
        finally:
            timer.cancel()
        if "dimensions" in kwargs:
            raise openai.BadRequestError(
                "dimensions is unsupported",
                response=httpx.Response(400, request=request),
                body=None,
            )
        return _response()

    embedder = _embedder(create, search_timeout=budget)
    async with _running(embedder.search_embed(["query"])) as task:
        with pytest.raises(ExternalServiceAPIError) as exc_info:
            await task

    assert isinstance(exc_info.value.__cause__, TimeoutError)
    assert len(calls) == 2
    assert "dimensions" in calls[0]
    assert "dimensions" not in calls[1]
    assert fallback_cancelled.is_set()


@pytest.mark.parametrize("operation", ["search_embed", "ingest_embed"])
async def test_unset_budget_preserves_success_and_batch_order(operation):
    first_started = asyncio.Event()
    second_finished = asyncio.Event()
    release_first = asyncio.Event()

    async def create(**kwargs):
        if kwargs["input"] == ["first"]:
            first_started.set()
            await release_first.wait()
            return _response()
        second_finished.set()
        response = _response()
        response.data[0].embedding = [0.0, 1.0]
        return response

    embedder = _embedder(create, batch_size=1)
    async with _running(getattr(embedder, operation)(["first", "second"])) as task:
        await first_started.wait()
        await second_finished.wait()
        assert not task.done()
        release_first.set()
        assert await task == [[1.0, 0.0], [0.0, 1.0]]


@pytest.mark.parametrize(
    "error", [ValueError("bad response"), TimeoutError("provider")]
)
async def test_provider_errors_are_not_relabelled_as_deadline_expiry(error):
    embedder = _embedder(AsyncMock(side_effect=error), search_timeout=3)
    with pytest.raises(type(error)) as exc_info:
        await embedder.search_embed(["query"])
    assert exc_info.value is error


async def test_real_sdk_retry_cannot_outlive_the_operation_budget():
    calls = 0
    warming_up = True
    retry_started = asyncio.Event()
    retry_cancelled = asyncio.Event()
    pending = asyncio.Event()

    async def handle(request):
        nonlocal calls
        if warming_up:
            return httpx.Response(
                200,
                json={
                    "object": "list",
                    "model": "test-model",
                    "data": [
                        {"object": "embedding", "index": 0, "embedding": [1.0, 0.0]}
                    ],
                    "usage": {"prompt_tokens": 1, "total_tokens": 1},
                },
            )
        calls += 1
        if calls == 1:
            return httpx.Response(
                429,
                headers={"retry-after-ms": "1"},
                json={"error": {"message": "retry", "type": "rate_limit_error"}},
            )
        retry_started.set()
        try:
            await pending.wait()
        finally:
            retry_cancelled.set()
        return httpx.Response(200)

    # MockTransport never opens a socket; this exercises the real SDK retry loop.
    async with openai.AsyncOpenAI(
        api_key="local-test-key",
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(handle)),
        max_retries=2,
    ) as client:
        # Keep SDK first-use initialization outside the retry scenario's budget.
        await client.embeddings.create(input=["warmup"], model="test-model")
        warming_up = False
        embedder = OpenAIEmbedder(
            OpenAIEmbedderParams(
                client=client,
                model="test-model",
                dimensions=2,
                search_timeout_seconds=0.5,
            )
        )
        async with _running(embedder.search_embed(["query"])) as task:
            with pytest.raises(ExternalServiceAPIError) as exc_info:
                await task

    assert isinstance(exc_info.value.__cause__, TimeoutError)
    assert calls == 2
    assert retry_started.is_set()
    assert retry_cancelled.is_set()
