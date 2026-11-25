import asyncio
import json
import logging
import math
import time
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import openai
import pytest

from memmachine_server.common.data_types import ExternalServiceAPIError
from memmachine_server.common.embedder.openai_embedder import (
    OpenAIEmbedder,
    OpenAIEmbedderParams,
)


@pytest.fixture(
    params=[
        ["Are tomatoes fruits?", "Tomatoes are red."],
        ["Are tomatoes fruits?", "Tomatoes are red.", ""],
        ["."],
        [" "],
        [""],
        [],
    ],
)
def inputs(request):
    return request.param


@pytest.mark.integration
@pytest.mark.asyncio
async def test_ingest_embed(openai_embedder, inputs):
    embeddings = await openai_embedder.ingest_embed(inputs)
    assert isinstance(embeddings, list)
    assert len(embeddings) == len(inputs)
    assert all(len(embedding) == openai_embedder.dimensions for embedding in embeddings)


@pytest.mark.integration
@pytest.mark.asyncio
async def test_search_embed(openai_embedder, inputs):
    embeddings = await openai_embedder.search_embed(inputs)
    assert isinstance(embeddings, list)
    assert len(embeddings) == len(inputs)
    assert all(len(embedding) == openai_embedder.dimensions for embedding in embeddings)


@pytest.mark.integration
@pytest.mark.asyncio
async def test_large_input(openai_embedder):
    input_text = "👩‍💻" * 10000

    assert len(await openai_embedder.ingest_embed([input_text])) == 1
    assert len(await openai_embedder.search_embed([input_text])) == 1


@pytest.mark.integration
@pytest.mark.asyncio
async def test_many_inputs(openai_embedder):
    input_texts = ["Hello, world!"] * 10000
    assert len(await openai_embedder.ingest_embed(input_texts)) == 10000
    assert len(await openai_embedder.search_embed(input_texts)) == 10000


class _FakeClock:
    """A monotonic clock that only advances when the embedder sleeps."""

    def __init__(self) -> None:
        self.now = 1000.0
        self.sleeps: list[float] = []

    def monotonic(self) -> float:
        return self.now

    async def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds


_OK = {
    "object": "list",
    "data": [{"object": "embedding", "index": 0, "embedding": [0.1] * 4}],
    "model": "test-model",
    "usage": {"prompt_tokens": 1, "total_tokens": 1},
}

# Captured from real 429s (#1974): OpenAI over its token limit, DeepInfra overloaded.
_OPENAI_429_HEADERS = {
    "Retry-After": "6",
    "retry-after-ms": "5389",
    "x-ratelimit-reset-requests": "12ms",
    "x-ratelimit-reset-tokens": "55.931s",
    "x-request-id": "req_ee2633c8",
}
_OPENAI_429_BODY = {
    "error": {
        "message": "Rate limit reached for text-embedding-3-small on tokens per min (TPM)",
        "type": "tokens",
        "param": None,
        "code": "rate_limit_exceeded",
    }
}
_DEEPINFRA_429_BODY = {
    "error": {
        "message": "Model busy, retry later",
        "type": "invalid_request_error",
        "param": None,
        "code": "engine_overloaded",
    }
}
# Replies recorded on 2026-11-25 for keys the provider refuses. Neither has a
# wait header; OpenAI labels its JSON text/plain.
_OPENAI_REVOKED_KEY = (
    401,
    {"Content-Type": "text/plain", "x-request-id": "req_test"},
    (
        b'{\n  "error": {\n    "message": "Your API key has been invalidated.",\n'
        b'    "type": null,\n    "code": "token_invalidated",\n    "param": null\n'
        b'  },\n  "status": 401\n}'
    ),
)
_DEEPINFRA_UNSCOPED_KEY = (
    403,
    {"Content-Type": "application/json"},
    (
        b'{"error":{"message":"this API key does not have the \'inference\' scope",'
        b'"type":"invalid_request_error","param":null,"code":null}}'
    ),
)


def _replies(*replies):
    """
    Make an httpx handler that answers with `replies` in order, and records the requests.

    A reply is (status, headers, body), or an exception to raise. A body of
    bytes is sent as is; any other body is sent as JSON.
    """
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        reply = replies[min(len(requests), len(replies)) - 1]
        if isinstance(reply, Exception):
            raise reply
        status, headers, body = reply
        if isinstance(body, bytes):
            return httpx.Response(status, headers=headers, content=body)
        return httpx.Response(status, headers=headers, json=body)

    return handler, requests


def _embedder(handler, **params) -> OpenAIEmbedder:
    client = openai.AsyncOpenAI(
        api_key="k",
        base_url="http://test/v1",
        max_retries=0,
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )
    return OpenAIEmbedder(
        OpenAIEmbedderParams(client=client, model="test-model", dimensions=4, **params)
    )


@pytest.fixture
def clock():
    clock = _FakeClock()
    with (
        patch(
            "memmachine_server.common.embedder.openai_embedder.time",
            SimpleNamespace(monotonic=clock.monotonic),
        ),
        patch(
            "memmachine_server.common.embedder.openai_embedder.asyncio",
            SimpleNamespace(sleep=clock.sleep, gather=asyncio.gather),
        ),
    ):
        yield clock


@pytest.mark.asyncio
async def test_retry_waits_as_long_as_the_provider_asks(clock):
    handler, requests = _replies(
        (429, _OPENAI_429_HEADERS, _OPENAI_429_BODY), (200, {}, _OK)
    )
    result = await _embedder(handler).ingest_embed(["test input"])

    assert len(result) == 1
    assert len(requests) == 2
    # retry-after-ms (5.389 s), not the rounded-up Retry-After or the quota reset.
    assert len(clock.sleeps) == 1
    assert 5.389 <= clock.sleeps[0] <= 5.489


@pytest.mark.asyncio
async def test_retry_backs_off_when_the_provider_gives_no_wait(clock):
    handler, requests = _replies(
        (429, {}, _DEEPINFRA_429_BODY),
        (503, {}, {"error": {"message": "unavailable"}}),
        (200, {}, _OK),
    )
    result = await _embedder(handler).ingest_embed(["test input"])

    assert len(result) == 1
    assert len(requests) == 3
    assert len(clock.sleeps) == 2
    # Full-jitter backoff (up to 0.25 s, then 0.5 s) plus up to 0.1 s of jitter.
    assert 0 <= clock.sleeps[0] <= 0.35
    assert 0 <= clock.sleeps[1] <= 0.6


@pytest.fixture
def longest_draws():
    """Make every random draw its upper bound, so backoffs are deterministic."""
    with patch(
        "memmachine_server.common.embedder.openai_embedder.random",
        SimpleNamespace(uniform=lambda _low, high: high),
    ):
        yield


@pytest.mark.asyncio
async def test_retry_after_zero_still_backs_off(clock):
    # A provider answering "retry now" every time must not be called in a tight loop.
    handler, requests = _replies((429, {"Retry-After": "0"}, _OPENAI_429_BODY))

    with pytest.raises(ExternalServiceAPIError, match=r"budget is spent"):
        await _embedder(handler).search_embed(["q"])

    # The gate's pace grows by half at most once per pace interval, so each
    # pace is waited twice: 1, 1, 1.5, 1.5, 2.25, 2.25, ... ms, until the next
    # wait would pass the deadline (10 s, less 0.25 s for an attempt).
    growing = [0.001 * 1.5**k for k in range(19) for _ in range(2)]
    assert clock.sleeps == pytest.approx([*growing, 9.75 - sum(growing)])
    assert len(requests) == 40


@pytest.mark.asyncio
async def test_backoff_shrinks_to_fit_the_budget(clock, longest_draws):
    # With no wait from the provider, a long backoff draw must not end the
    # retries while budget remains: it is cut to what is left.
    handler, requests = _replies((503, {}, {"error": {"message": "unavailable"}}))

    with pytest.raises(ExternalServiceAPIError, match=r"budget is spent"):
        await _embedder(handler).ingest_embed(["text"])

    # Waits 0.35 ... 16.1 s, then 32.1 s cut to the rest of the 60 s budget,
    # less the time kept for the last attempt.
    assert clock.sleeps[-1] < 32.1
    assert sum(clock.sleeps) == pytest.approx(60 - 0.25)
    assert len(requests) == 8


@pytest.mark.asyncio
async def test_retry_after_connection_error(clock):
    handler, requests = _replies(httpx.ConnectError("refused"), (200, {}, _OK))
    assert len(await _embedder(handler).search_embed(["q"])) == 1
    assert len(requests) == 2


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("status", "body"),
    [
        (429, {"error": {"message": "out of credit", "code": "insufficient_quota"}}),
        (400, {"error": {"message": "bad input"}}),
    ],
)
async def test_no_retry_when_waiting_cannot_help(clock, status, body):
    handler, requests = _replies((status, {}, body), (200, {}, _OK))

    with pytest.raises(ExternalServiceAPIError, match=r"non-retryable"):
        await _embedder(handler).ingest_embed(["test input"])

    assert len(requests) == 1
    assert clock.sleeps == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("reply", "error"),
    [
        (
            _OPENAI_REVOKED_KEY,
            (
                r"AuthenticationError"
                r" \(status=401 request_id=req_test code=token_invalidated\)"
            ),
        ),
        (_DEEPINFRA_UNSCOPED_KEY, r"PermissionDeniedError \(status=403\)"),
    ],
)
async def test_no_retry_for_a_key_the_provider_refuses(clock, reply, error):
    handler, requests = _replies(reply, (200, {}, _OK))

    with pytest.raises(ExternalServiceAPIError, match=rf"non-retryable {error}"):
        await _embedder(handler).ingest_embed(["test input"])

    assert len(requests) == 1
    assert clock.sleeps == []


@pytest.mark.asyncio
async def test_gives_up_when_the_asked_wait_passes_the_budget(clock):
    # The search budget is 10 s; the provider asks for 30 s.
    handler, requests = _replies((429, {"Retry-After": "30"}, _OPENAI_429_BODY))

    with pytest.raises(ExternalServiceAPIError, match=r"would pass the 10 s budget"):
        await _embedder(handler).search_embed(["q"])

    assert len(requests) == 1
    assert clock.sleeps == []


@pytest.mark.asyncio
async def test_gives_up_when_the_budget_is_spent(clock):
    handler, requests = _replies(httpx.ConnectError("refused"))

    with pytest.raises(ExternalServiceAPIError) as exc_info:
        await _embedder(handler).search_embed(["q"])

    message = str(exc_info.value)
    assert "budget" in message
    assert "APIConnectionError" in message
    assert len(requests) > 1
    assert sum(clock.sleeps) <= 10


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("embed", "budget"), [("search_embed", 10.0), ("ingest_embed", 60.0)]
)
async def test_requests_carry_the_remaining_budget_and_connect_cap(
    clock, embed, budget
):
    handler, requests = _replies((200, {}, _OK))
    await getattr(_embedder(handler), embed)(["text"])

    timeout = requests[0].extensions["timeout"]
    assert timeout["read"] == pytest.approx(budget)
    assert timeout["connect"] == pytest.approx(2.0)


def _embeddings_server(*, accepts_dimensions: bool, embedding_length: int = 4):
    """
    Make an httpx handler that behaves like an embeddings server, and records the requests.

    A server that does not accept the dimensions parameter answers any
    request carrying it with 400, as OpenAI does for a model without it.
    """
    requests: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        requests.append(body)
        if "dimensions" in body and not accepts_dimensions:
            return httpx.Response(
                400,
                json={
                    "error": {
                        "message": "This model does not support specifying dimensions.",
                        "type": "invalid_request_error",
                        "param": "dimensions",
                        "code": None,
                    }
                },
            )
        return httpx.Response(
            200,
            json={
                "object": "list",
                "data": [
                    {
                        "object": "embedding",
                        "index": i,
                        "embedding": [0.1] * embedding_length,
                    }
                    for i in range(len(body["input"]))
                ],
                "model": body["model"],
                "usage": {"prompt_tokens": 1, "total_tokens": 1},
            },
        )

    return handler, requests


@pytest.mark.asyncio
async def test_sends_dimensions_to_a_server_that_accepts_them(clock):
    handler, requests = _embeddings_server(accepts_dimensions=True)
    embedder = _embedder(handler)

    assert len(await embedder.ingest_embed(["text"])) == 1
    assert len(await embedder.search_embed(["query"])) == 1

    # One request per call, each asking for the configured dimensions.
    assert [body.get("dimensions") for body in requests] == [4, 4]
    assert clock.sleeps == []


@pytest.mark.asyncio
async def test_drops_dimensions_for_a_server_that_rejects_them(clock):
    handler, requests = _embeddings_server(accepts_dimensions=False)
    embedder = _embedder(handler)

    assert len(await embedder.ingest_embed(["text"])) == 1
    # The first call is refused with dimensions and resent without them, at once.
    assert ["dimensions" in body for body in requests] == [True, False]
    assert clock.sleeps == []

    assert len(await embedder.search_embed(["query"])) == 1
    # Later calls leave dimensions out from the start.
    assert ["dimensions" in body for body in requests] == [True, False, False]


@pytest.mark.asyncio
async def test_rejects_embeddings_of_the_wrong_size(clock):
    handler, requests = _embeddings_server(accepts_dimensions=True, embedding_length=3)

    with pytest.raises(ExternalServiceAPIError, match=r"dimensionality 3"):
        await _embedder(handler).ingest_embed(["text"])

    assert len(requests) == 1


@pytest.mark.asyncio
async def test_embed_oversized_input_with_no_max_input_length():
    """Regression for #1298: text > 75,000 chars must not raise ValueError.

    When max_input_length is None, _embed() must fall back to
    max_total_input_length_per_request (75,000) as the chunking bound so that
    cluster_texts() never receives a text that exceeds its hard limit.
    Prior to the fix, this raised ValueError("Text length ... exceeds
    max_total_length_per_cluster ..."), which surfaced as HTTP 500.
    """
    mock_client = AsyncMock(spec=openai.AsyncOpenAI)

    mock_embedding = MagicMock()
    mock_embedding.embedding = [0.1] * 256

    mock_response = MagicMock()
    mock_response.data = [mock_embedding]
    mock_response.usage.prompt_tokens = 10
    mock_response.usage.total_tokens = 10

    mock_client.embeddings.create = AsyncMock(return_value=mock_response)

    embedder = OpenAIEmbedder(
        OpenAIEmbedderParams(
            client=mock_client,
            model="test-model",
            dimensions=256,
            max_input_length=None,  # default — reproduces the crash scenario
        )
    )

    # 80,000 chars exceeds the 75,000-char cluster hard limit by 5,000.
    long_text = "x" * 80_000

    # Before the fix this raised:
    #   ValueError: Text length 80000 exceeds max_total_length_per_cluster 75000
    result = await embedder.ingest_embed([long_text])

    assert len(result) == 1
    assert len(result[0]) == 256
    # The text must have been split: greedy chunks of 80,000 chars at limit
    # 75,000 produce chunks of 75,000 and 5,000, each in its own cluster.
    assert mock_client.embeddings.create.call_count >= 2


@pytest.mark.asyncio
async def test_embed_max_input_length_above_request_cap():
    """max_input_length above max_total_input_length_per_request must not raise.

    A configured chunk limit above the per-request cap can never be honored
    (no chunk larger than the cap fits in any request), so chunking clamps
    to the cap instead of handing cluster_texts() a chunk it rejects.
    """
    mock_client = AsyncMock(spec=openai.AsyncOpenAI)

    mock_embedding = MagicMock()
    mock_embedding.embedding = [0.1] * 256

    mock_response = MagicMock()
    mock_response.data = [mock_embedding]
    mock_response.usage.prompt_tokens = 10
    mock_response.usage.total_tokens = 10

    mock_client.embeddings.create = AsyncMock(return_value=mock_response)

    embedder = OpenAIEmbedder(
        OpenAIEmbedderParams(
            client=mock_client,
            model="test-model",
            dimensions=256,
            max_input_length=80_000,  # above the 75,000-char per-request cap
        )
    )

    long_text = "x" * 100_000

    result = await embedder.ingest_embed([long_text])

    assert len(result) == 1
    assert len(result[0]) == 256
    # Clamped to 75,000, the text splits into 75,000 + 25,000 chars, which
    # exceed one request's total together, so each chunk gets its own call.
    assert mock_client.embeddings.create.call_count == 2


def _duration(seconds: float) -> str:
    """Format seconds as OpenAI's x-ratelimit-reset-* headers do: "20ms", "1.5s"."""
    return f"{round(seconds * 1000)}ms" if seconds < 1 else f"{seconds:.3f}s"


class _RateLimitedServer:
    """
    An embeddings server that limits requests and tokens per second, as OpenAI does per minute.

    Each limit is a token bucket that starts full and refills continuously.
    A request is charged one request and its input's tokens (a quarter of
    its characters) when it is accepted. A request that does not fit is
    refused with 429 and OpenAI's headers: retry-after-ms (when both buckets
    will have room), Retry-After (that, rounded up to whole seconds), and
    the x-ratelimit-* limits, remainders and resets.
    """

    def __init__(
        self,
        *,
        requests_per_second: float,
        tokens_per_second: float,
        request_capacity: float,
        token_capacity: float,
    ) -> None:
        self.requests_per_second = requests_per_second
        self.tokens_per_second = tokens_per_second
        self.request_capacity = request_capacity
        self.token_capacity = token_capacity
        self.reset()

    def reset(self) -> None:
        """Fill both buckets and forget the requests seen so far."""
        self.requests_left = self.request_capacity
        self.tokens_left = self.token_capacity
        self.last_refill = time.monotonic()
        # (time, tokens) per accepted request
        self.accepted: list[tuple[float, int]] = []
        self.refused = 0

    def handler(self, request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        now = time.monotonic()
        elapsed, self.last_refill = now - self.last_refill, now
        self.requests_left = min(
            self.request_capacity,
            self.requests_left + elapsed * self.requests_per_second,
        )
        self.tokens_left = min(
            self.token_capacity, self.tokens_left + elapsed * self.tokens_per_second
        )

        tokens = sum(len(text) // 4 for text in body["input"])
        request_wait = max(0.0, (1 - self.requests_left) / self.requests_per_second)
        token_wait = max(0.0, (tokens - self.tokens_left) / self.tokens_per_second)
        if request_wait > 0 or token_wait > 0:
            self.refused += 1
            wait = max(request_wait, token_wait)
            headers = {
                "retry-after-ms": str(math.ceil(wait * 1000)),
                "Retry-After": str(math.ceil(wait)),
                "x-ratelimit-limit-requests": str(self.requests_per_second),
                "x-ratelimit-limit-tokens": str(self.tokens_per_second),
                "x-ratelimit-remaining-requests": str(int(self.requests_left)),
                "x-ratelimit-remaining-tokens": str(int(self.tokens_left)),
                "x-ratelimit-reset-requests": _duration(
                    (self.request_capacity - self.requests_left)
                    / self.requests_per_second
                ),
                "x-ratelimit-reset-tokens": _duration(
                    (self.token_capacity - self.tokens_left) / self.tokens_per_second
                ),
            }
            limit = "tokens" if token_wait >= request_wait else "requests"
            return httpx.Response(
                429,
                headers=headers,
                json={
                    "error": {
                        "message": f"Rate limit reached on {limit} per second.",
                        "type": limit,
                        "param": None,
                        "code": "rate_limit_exceeded",
                    }
                },
            )

        self.requests_left -= 1
        self.tokens_left -= tokens
        self.accepted.append((now, tokens))
        return httpx.Response(
            200,
            json={
                "object": "list",
                "data": [
                    {"object": "embedding", "index": i, "embedding": [0.1] * 4}
                    for i in range(len(body["input"]))
                ],
                "model": body["model"],
                "usage": {"prompt_tokens": tokens, "total_tokens": tokens},
            },
        )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("requests_per_second", "tokens_per_second", "bound", "latency", "floor"),
    [
        # 40 requests/s binds; tokens are plentiful.
        (40, 1_000_000, "requests", 0.0, 0.9),
        # 2,000 tokens/s (20 requests of 100 tokens) binds; requests are plentiful.
        (1_000_000, 2_000, "tokens", 0.0, 0.9),
        # As the first, with the server answering after 150 ms: turns must not
        # wait for answers, or one client could not keep up with the limit.
        # Refusals arrive 150 ms late, so the pace lags (91% measured).
        (40, 1_000_000, "requests", 0.15, 0.85),
    ],
)
async def test_retries_keep_throughput_near_the_rate_limit(
    caplog, requests_per_second, tokens_per_second, bound, latency, floor
):
    """
    Callers asking far more than the limits allow get close to the limits, not far under them.

    Twenty callers embed 100-token texts back to back for two seconds of
    real time, retrying as the server's 429 headers say; the server's
    buckets hold a tenth of a second of each limit.
    """
    caplog.set_level(
        logging.ERROR, logger="memmachine_server.common.embedder.openai_embedder"
    )
    server = _RateLimitedServer(
        requests_per_second=requests_per_second,
        tokens_per_second=tokens_per_second,
        request_capacity=max(1.0, requests_per_second / 10),
        token_capacity=max(100.0, tokens_per_second / 10),
    )

    async def handler(request: httpx.Request) -> httpx.Response:
        await asyncio.sleep(latency)
        return server.handler(request)

    embedder = _embedder(handler)
    text = "word " * 80  # 400 characters: 100 tokens
    # The first call pays one-time setup (~300 ms), which would count against
    # the rate: make it before the clock starts.
    await embedder.ingest_embed([text])
    server.reset()
    duration = 2.0
    start = time.monotonic()

    async def caller() -> None:
        while time.monotonic() - start < duration:
            await embedder.ingest_embed([text])

    callers = [asyncio.create_task(caller()) for _ in range(20)]
    await asyncio.sleep(duration)
    for task in callers:
        task.cancel()
    results = await asyncio.gather(*callers, return_exceptions=True)
    assert not [r for r in results if not isinstance(r, asyncio.CancelledError)]

    accepted = [(t, tokens) for t, tokens in server.accepted if t - start <= duration]
    if bound == "requests":
        limit, burst = requests_per_second, server.request_capacity
        achieved = len(accepted) / duration
    else:
        limit, burst = tokens_per_second, server.token_capacity
        achieved = sum(tokens for _, tokens in accepted) / duration
    assert server.refused > 0, "the callers never reached the limit"
    assert floor * limit <= achieved <= limit + burst / duration, (
        f"{bound}: achieved {achieved:.1f}/s against a limit of {limit}/s "
        f"({achieved / limit:.0%}), {server.refused} requests refused"
    )
