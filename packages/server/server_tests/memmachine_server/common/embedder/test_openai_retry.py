import asyncio
import time
from types import SimpleNamespace

import httpx
import openai
import pytest

from memmachine_server.common.embedder.openai_retry import (
    RetryGate,
    describe_error,
    is_retryable,
    parse_openai_wait,
)


def _status_error(status: int, headers=None, body=None) -> openai.APIStatusError:
    request = httpx.Request("POST", "http://test/v1/embeddings")
    response = httpx.Response(status, request=request, headers=headers or {})
    return openai.AsyncOpenAI(api_key="k")._make_status_error(
        "error", body=body, response=response
    )


@pytest.mark.parametrize(
    ("headers", "expected"),
    [
        (None, None),
        ({}, None),
        # Captured from a real OpenAI 429: retry-after-ms wins over the rounded-up Retry-After.
        (
            {
                "Retry-After": "6",
                "retry-after-ms": "5389",
                "x-ratelimit-reset-requests": "12ms",
                "x-ratelimit-reset-tokens": "55.931s",
            },
            5.389,
        ),
        ({"Retry-After": "6", "x-ratelimit-reset-tokens": "55.931s"}, 6.0),
        # Retry-After may be an HTTP date, which is ignored in favor of the reset headers.
        (
            {
                "Retry-After": "Wed, 21 Oct 2026 07:28:00 GMT",
                "x-ratelimit-reset-tokens": "1s",
            },
            1.0,
        ),
        (
            {"x-ratelimit-reset-requests": "12ms", "x-ratelimit-reset-tokens": "0s"},
            0.012,
        ),
        ({"x-ratelimit-reset-tokens": "55.931s"}, 55.931),
        ({"x-ratelimit-reset-tokens": "6m0s"}, 360.0),
        ({"x-ratelimit-reset-tokens": "1h2m3.5s"}, 3723.5),
        ({"x-ratelimit-reset-tokens": "soon"}, None),
        ({"retry-after-ms": "not-a-number", "Retry-After": "2"}, 2.0),
    ],
)
def test_parse_openai_wait(headers, expected):
    wait = parse_openai_wait(httpx.Headers(headers) if headers is not None else None)
    assert wait == pytest.approx(expected) if expected is not None else wait is None


@pytest.mark.parametrize(
    ("status", "code", "retryable"),
    [
        (429, "rate_limit_exceeded", True),
        (429, "engine_overloaded", True),
        (429, None, True),
        (429, "insufficient_quota", False),
        (500, None, True),
        (502, None, True),
        (503, None, True),
        (504, None, True),
        (400, None, False),
        (401, None, False),
        (404, None, False),
        (422, None, False),
    ],
)
def test_is_retryable_status(status, code, retryable):
    body = {"message": "m", "code": code} if code else None
    assert is_retryable(_status_error(status, body=body)) is retryable


def test_connection_errors_are_retryable():
    request = httpx.Request("POST", "http://test/v1/embeddings")
    assert is_retryable(openai.APIConnectionError(request=request))
    assert is_retryable(openai.APITimeoutError(request=request))


def test_describe_error():
    err = _status_error(
        429,
        headers={"x-request-id": "req_1", "retry-after-ms": "5389", "Retry-After": "6"},
        body={"message": "m", "code": "rate_limit_exceeded"},
    )
    assert describe_error(err) == (
        "RateLimitError (status=429 request_id=req_1 code=rate_limit_exceeded "
        "retry_after_ms=5389 retry_after=6)"
    )


def test_describe_no_error():
    assert describe_error(None) == "none"


def _gate(max_pace: float = 120.0) -> RetryGate:
    return RetryGate(max_pace=max_pace, clock=time.monotonic, sleep=asyncio.sleep)


@pytest.mark.asyncio
async def test_gate_admits_at_once_when_not_limited():
    gate = _gate()
    start = time.monotonic()
    for _ in range(100):
        await gate.admit(start + 10, since=start)
    assert time.monotonic() - start < 0.05
    assert gate.pace == 0


@pytest.mark.asyncio
async def test_gate_holds_every_request_for_the_providers_wait():
    gate = _gate()
    gate.limited(0.05)
    start = time.monotonic()
    await gate.admit(start + 10, since=start)
    assert time.monotonic() - start >= 0.05


@pytest.mark.asyncio
async def test_gate_turns_go_oldest_call_first_and_newcomers_wait():
    gate = _gate()
    gate.limited(0.05)
    now = time.monotonic()
    turns: list[str] = []

    async def request(name: str, since: float, deadline: float) -> None:
        await gate.admit(deadline, since=since)
        turns.append(name)

    waiting = [
        asyncio.create_task(request("A", now, now + 60)),
        asyncio.create_task(request("B", now + 0.001, now + 10)),
    ]
    await asyncio.sleep(0.01)
    # While A and B wait: C is a new call with the earliest deadline, and E is
    # the retry of a call that started before A.
    waiting += [
        asyncio.create_task(request("C", now + 0.01, now + 5)),
        asyncio.create_task(request("E", now - 1, now + 59)),
    ]
    await asyncio.gather(*waiting)
    assert turns == ["E", "A", "B", "C"]


@pytest.mark.asyncio
async def test_gate_waiter_leaves_at_its_deadline():
    gate = _gate()
    gate.limited(1.0)
    start = time.monotonic()
    # An older call waits ahead of it, with a later deadline.
    older = asyncio.create_task(gate.admit(start + 10, since=start - 1))
    with pytest.raises(TimeoutError):
        await gate.admit(start + 0.05, since=start)
    assert 0.05 <= time.monotonic() - start < 0.5
    older.cancel()


@pytest.mark.asyncio
async def test_gate_skips_a_cancelled_waiter():
    gate = _gate()
    gate.limited(0.05)
    now = time.monotonic()
    first = asyncio.create_task(gate.admit(now + 10, since=now))
    second = asyncio.create_task(gate.admit(now + 20, since=now + 0.001))
    await asyncio.sleep(0.01)
    first.cancel()
    await asyncio.wait_for(second, timeout=1)


def test_gate_pace_grows_on_refusals_and_shrinks_on_successes():
    clock = SimpleNamespace(now=0.0)
    gate = RetryGate(max_pace=0.002, clock=lambda: clock.now, sleep=asyncio.sleep)
    gate.limited(0)
    assert gate.pace == pytest.approx(0.001)
    gate.limited(0)  # within the pace interval: no change
    assert gate.pace == pytest.approx(0.001)
    clock.now += gate.pace
    gate.limited(0)
    assert gate.pace == pytest.approx(0.0015)
    clock.now += gate.pace
    gate.limited(0)
    assert gate.pace == pytest.approx(0.002)  # capped
    gate.succeeded()
    assert gate.pace == pytest.approx(0.0016)
    for _ in range(3):
        gate.succeeded()
    assert gate.pace == 0


def test_gate_refusals_close_together_grow_the_pace_once():
    clock = SimpleNamespace(now=0.0)
    gate = RetryGate(max_pace=120.0, clock=lambda: clock.now, sleep=asyncio.sleep)
    gate.limited(0)
    # Sixteen more refusals within the pace interval, as from requests sent together.
    for _ in range(16):
        clock.now += gate.pace / 20
        gate.limited(0)
    assert gate.pace == pytest.approx(0.001)
    clock.now += gate.pace
    gate.limited(0)
    assert gate.pace == pytest.approx(0.0015)
