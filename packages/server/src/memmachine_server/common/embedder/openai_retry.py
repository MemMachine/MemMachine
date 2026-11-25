"""Retry decisions for calls to OpenAI-compatible APIs, from their errors and response headers."""

import asyncio
import heapq
import itertools
import re
from collections.abc import Awaitable, Callable, Mapping

import openai

# Statuses worth another attempt: the provider is limiting us or failing, not
# rejecting the request.
RETRYABLE_STATUSES = frozenset({429, 500, 502, 503, 504})

# Error codes that come with a retryable status but that waiting does not fix.
# OpenAI answers 429 `insufficient_quota` when the account is out of credit.
NON_RETRYABLE_CODES = frozenset({"insufficient_quota"})

# Go-style durations, as OpenAI sends in x-ratelimit-reset-*: "12ms", "55.931s", "6m0s".
_DURATION_RE = re.compile(
    r"(?:(\d+)h)?(?:(\d+)m(?!s))?(?:(\d+(?:\.\d+)?)s)?(?:(\d+)ms)?"
)


def parse_openai_wait(headers: Mapping[str, str] | None) -> float | None:
    """
    Return the seconds the provider asks us to wait, or None if it does not say.

    Reads, in order: retry-after-ms (exact), Retry-After (whole seconds,
    rounded up), and the longest of x-ratelimit-reset-requests and
    x-ratelimit-reset-tokens (time until the whole quota refills, so an
    overestimate). Header names match case-insensitively on httpx headers.
    """
    if not headers:
        return None

    if ms := headers.get("retry-after-ms"):
        try:
            return float(ms) / 1000.0
        except ValueError:
            pass

    if sec := headers.get("retry-after"):
        try:
            return float(sec)
        except ValueError:
            pass

    waits = []
    for key in ("x-ratelimit-reset-requests", "x-ratelimit-reset-tokens"):
        if (raw := headers.get(key)) and (m := _DURATION_RE.fullmatch(raw.strip())):
            h, mins, s, ms = m.groups()
            waits.append(
                int(h or 0) * 3600
                + int(mins or 0) * 60
                + float(s or 0)
                + int(ms or 0) / 1000.0
            )
    return max(waits) if waits else None


def error_code(err: openai.APIError) -> str | None:
    """Return the provider's `error.code` (e.g. rate_limit_exceeded, engine_overloaded), if any."""
    code = getattr(err, "code", None)
    return code if isinstance(code, str) else None


def is_retryable(err: openai.APIError) -> bool:
    """
    Return whether another attempt can succeed.

    Connection errors and timeouts can; so can the statuses in
    RETRYABLE_STATUSES, unless the error code says waiting will not help.
    """
    if isinstance(err, openai.APIConnectionError):
        return True
    if isinstance(err, openai.APIStatusError):
        return (
            err.status_code in RETRYABLE_STATUSES
            and error_code(err) not in NON_RETRYABLE_CODES
        )
    return False


def error_headers(err: openai.APIError) -> Mapping[str, str] | None:
    """Return the response headers of a failed call, if it got a response."""
    response = getattr(err, "response", None)
    return getattr(response, "headers", None)


def describe_error(err: openai.APIError | None) -> str:
    """Summarize a failed attempt for the log: status, request id, error code, and the provider's wait headers."""
    if err is None:
        return "none"
    headers = error_headers(err) or {}
    fields = {
        "status": getattr(err, "status_code", None),
        "request_id": headers.get("x-request-id"),
        "code": error_code(err),
        "retry_after_ms": headers.get("retry-after-ms"),
        "retry_after": headers.get("retry-after"),
    }
    details = " ".join(f"{k}={v}" for k, v in fields.items() if v is not None)
    return f"{type(err).__name__}" + (f" ({details})" if details else "")


class RetryGate:
    """
    Turns for one client's requests while its provider rate-limits it.

    Until the provider refuses a request with a wait, admit() returns at
    once. After a refusal (limited()), no request, new or retried, is sent
    before the provider's wait ends, and requests that must wait go one at a
    time, oldest call first, so a new call cannot take the turn of one that
    has waited longer. Turns are spaced by a pace that grows by half on a
    refusal and shrinks by a fifth on each success, so a provider that keeps
    refusing, even with a wait of 0, is sent fewer requests, and one that
    stops refusing is soon sent requests without spacing again. The pace grows
    at most once per pace interval, so refusals that come close together, of
    requests sent under the same pace, grow it once.
    """

    MIN_PACE = 0.001
    GROWTH = 1.5
    DECAY = 0.8

    def __init__(
        self,
        *,
        max_pace: float,
        clock: Callable[[], float],
        sleep: Callable[[float], Awaitable[None]],
    ) -> None:
        """Create a gate whose pace never exceeds `max_pace`, on the given monotonic clock and sleep."""
        self._max_pace = max_pace
        self._clock = clock
        self._sleep = sleep
        # (call start, arrival order, deadline, waiter), oldest call first
        self._queue: list[tuple[float, int, float, asyncio.Future[None]]] = []
        self._seq = itertools.count()
        self._not_before = float("-inf")
        self._pace = 0.0
        self._paced_at = float("-inf")  # when the pace last grew
        self._releaser: asyncio.Task[None] | None = None

    @property
    def pace(self) -> float:
        """Seconds between turns; 0 when the provider is not refusing."""
        return self._pace

    def holding(self) -> float:
        """Seconds until the provider's last wait ends."""
        return max(self._not_before - self._clock(), 0.0)

    async def admit(self, deadline: float, *, since: float) -> None:
        """
        Wait for the turn of a request of a call that started at `since`.

        Raises:
            TimeoutError: If `deadline`, on the gate's clock, comes first.

        """
        now = self._clock()
        if deadline <= now:
            raise TimeoutError
        if not self._queue and now >= self._not_before:
            self._not_before = now + self._pace
            return
        loop = asyncio.get_running_loop()
        waiter: asyncio.Future[None] = loop.create_future()
        heapq.heappush(self._queue, (since, next(self._seq), deadline, waiter))
        if (
            self._releaser is None
            or self._releaser.done()
            or self._releaser.get_loop() is not loop
        ):
            self._releaser = loop.create_task(self._release())
        await waiter

    def limited(self, wait: float) -> None:
        """Record that the provider refused a request and asked for `wait` seconds."""
        now = self._clock()
        self._not_before = max(self._not_before, now + wait)
        if now >= self._paced_at + self._pace:
            self._pace = min(
                max(self._pace * self.GROWTH, self.MIN_PACE), self._max_pace
            )
            self._paced_at = now

    def succeeded(self) -> None:
        """Record that the provider accepted a request."""
        self._pace *= self.DECAY
        if self._pace < self.MIN_PACE:
            self._pace = 0.0

    async def _release(self) -> None:
        """Give waiting requests their turns, oldest call first, until none wait."""
        while self._queue:
            now = self._clock()
            # Drop waiters whose callers were cancelled or whose deadlines passed.
            live = []
            for entry in self._queue:
                waiter = entry[3]
                if waiter.done():
                    continue
                if entry[2] <= now:
                    waiter.set_exception(TimeoutError())
                    continue
                live.append(entry)
            if len(live) != len(self._queue):
                heapq.heapify(live)
                self._queue = live
                continue
            if now < self._not_before:
                next_deadline = min(entry[2] for entry in self._queue)
                await self._sleep(min(self._not_before, next_deadline) - now)
                continue
            heapq.heappop(self._queue)[3].set_result(None)
            self._not_before = now + self._pace
