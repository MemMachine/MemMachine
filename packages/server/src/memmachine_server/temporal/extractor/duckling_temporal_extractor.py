"""Duckling-based temporal extractor."""

import json
from datetime import UTC, datetime, timedelta
from typing import override

import httpx
from pydantic import BaseModel, Field, InstanceOf

from memmachine_server.common.request_context import get_request_locale
from memmachine_server.common.utils import ensure_tz_aware
from memmachine_server.temporal.time_range import TimeInterval, TimeRange

from .temporal_extractor import TemporalExtractor


def _parse_iso(value: str) -> datetime:
    """Parse a Duckling ISO-8601 timestamp into a UTC-aware datetime."""
    return ensure_tz_aware(datetime.fromisoformat(value)).astimezone(UTC)


def _end_of(start: datetime, grain: str) -> datetime | None:
    """End of the period containing ``start`` at the given ``grain``.

    Returns ``None`` if the grain is unknown.
    """
    if grain == "second":
        return start + timedelta(seconds=1)
    if grain == "minute":
        return start + timedelta(minutes=1)
    if grain == "hour":
        return start + timedelta(hours=1)
    if grain == "day":
        return start + timedelta(days=1)
    if grain == "week":
        # Duckling returns the week's start instant (Sunday for en_US,
        # Monday for en_GB, etc.) -- add 7 days regardless of locale.
        return start + timedelta(days=7)
    if grain == "month":
        first_next_month = (
            datetime(start.year + 1, 1, 1, tzinfo=start.tzinfo)
            if start.month == 12
            else datetime(start.year, start.month + 1, 1, tzinfo=start.tzinfo)
        )
        # The boundary day at this grain is the first of the next month.
        return first_next_month
    if grain == "quarter":
        # Quarter starts: 1, 4, 7, 10.
        q_start_month = ((start.month - 1) // 3) * 3 + 1
        end_month = q_start_month + 3
        if end_month > 12:
            return datetime(start.year + 1, end_month - 12, 1, tzinfo=start.tzinfo)
        return datetime(start.year, end_month, 1, tzinfo=start.tzinfo)
    if grain == "year":
        return datetime(start.year + 1, 1, 1, tzinfo=start.tzinfo)
    return None


def _value_to_time_range(value: dict) -> TimeRange | None:
    """Convert a Duckling ``type="value"`` (instant + grain) to a range."""
    raw = value.get("value")
    grain = value.get("grain", "day")
    if raw is None:
        return None
    try:
        start = _parse_iso(raw)
    except ValueError:
        return None
    end = _end_of(start, grain)
    if end is None or end <= start:
        return None
    return TimeRange(intervals=[TimeInterval(start=start, end=end)])


def _interval_bound(part: dict | None) -> datetime | None:
    """Parse one interval bound to UTC; ``None`` if the bound is absent.

    Raises ``KeyError`` / ``ValueError`` if the bound is present but
    malformed, so the caller can reject the whole interval.
    """
    if part is None:
        return None
    return _parse_iso(part["value"])


def _interval_to_time_range(value: dict) -> TimeRange | None:
    """Convert a Duckling ``type="interval"`` (from / to bounds) to a range.

    Half-bounded intervals keep the missing side as ``None``.
    """
    try:
        start = _interval_bound(value.get("from"))
        end = _interval_bound(value.get("to"))
    except (ValueError, KeyError):
        return None
    # Empty interval (no bounds at all) is meaningless.
    if start is None and end is None:
        return None
    # Closed interval must be non-empty.
    if start is not None and end is not None and end <= start:
        return None
    return TimeRange(intervals=[TimeInterval(start=start, end=end)])


def _entity_to_time_range(entity: dict) -> TimeRange | None:
    """Convert one Duckling ``dim=time`` entity into a ``TimeRange``.

    Uses ONLY the primary value (``value.value`` or ``value.from`` /
    ``value.to``); ignores the ``values[]`` alternatives.
    """
    value = entity.get("value") or {}
    kind = value.get("type")
    if kind == "value":
        return _value_to_time_range(value)
    if kind == "interval":
        return _interval_to_time_range(value)
    return None


class DucklingTemporalExtractorParams(BaseModel):
    """
    Parameters for DucklingTemporalExtractor.

    Attributes:
        client (httpx.AsyncClient):
            HTTP client used to call the Duckling server (caller-owned).
        url (str):
            Duckling parse endpoint URL.
    """

    client: InstanceOf[httpx.AsyncClient] = Field(
        ...,
        description="HTTP client used to call the Duckling server (caller-owned)",
    )
    url: str = Field(
        ...,
        description="Duckling parse endpoint URL",
    )


class DucklingTemporalExtractor(TemporalExtractor):
    """``TimeRange`` extractor backed by a local Duckling HTTP server."""

    def __init__(self, params: DucklingTemporalExtractorParams) -> None:
        """Initialize from parameters."""
        self._client = params.client
        self._url = params.url

    @override
    async def extract(
        self, text: str, *, ref_time: datetime | None = None
    ) -> list[TimeRange]:
        if ref_time is None:
            ref_time = datetime.now(UTC)

        ref_time = ensure_tz_aware(ref_time)

        # Duckling resolves local expressions in this tz; derive it from the
        # input ref_time (its IANA name when available) rather than configuring
        # it once.
        timezone_name = (
            getattr(ref_time.tzinfo, "key", None) or ref_time.tzname() or "UTC"
        )
        reftime_ms = int(ref_time.astimezone(UTC).timestamp() * 1000)
        # Duckling and babel both render locales as ``lang_REGION`` (``en_US``).
        duckling_locale = str(get_request_locale())
        try:
            response = await self._client.post(
                self._url,
                data={
                    "locale": duckling_locale,
                    "tz": timezone_name,
                    "reftime": reftime_ms,
                    "dims": json.dumps(["time"]),
                    "text": text,
                },
            )
        except httpx.HTTPError:
            return []
        if response.status_code != 200:
            return []
        try:
            entities = response.json()
        except ValueError:
            return []
        ranges: list[TimeRange] = []
        for entity in entities:
            if entity.get("dim") != "time":
                continue
            time_range = _entity_to_time_range(entity)
            if time_range is not None:
                ranges.append(time_range)
        return ranges
