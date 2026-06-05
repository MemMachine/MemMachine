"""Interface / structure tests for the Duckling-backed temporal extractor.

These exercise the structural contract -- return types, request shape,
error-to-empty-list handling, and field wiring -- using a mocked
HTTP transport and synthetic Duckling entities (whose shape is the server's
documented API contract). They deliberately do NOT assert on resolved date
values, which depend on the Duckling server version.
"""

import json
import urllib.parse
from contextlib import asynccontextmanager
from datetime import UTC, datetime, timedelta
from zoneinfo import ZoneInfo

import pytest

pytest.importorskip("httpx")

import httpx

from memmachine_server.temporal.extractor.duckling_temporal_extractor import (
    DucklingTemporalExtractor,
    DucklingTemporalExtractorParams,
    _end_of,
    _entity_to_time_range,
    _interval_bound,
    _interval_to_time_range,
    _parse_iso,
    _value_to_time_range,
)
from memmachine_server.temporal.time_range import TimeRange

URL = "http://duck.test/parse"


def value_entity(stamp="2024-03-05T00:00:00.000Z", grain="day"):
    """A synthetic Duckling ``type="value"`` time entity."""
    return {"dim": "time", "value": {"type": "value", "value": stamp, "grain": grain}}


def interval_entity(from_stamp=None, to_stamp=None):
    """A synthetic Duckling ``type="interval"`` time entity."""
    value: dict[str, object] = {"type": "interval"}
    if from_stamp is not None:
        value["from"] = {"value": from_stamp, "grain": "day"}
    if to_stamp is not None:
        value["to"] = {"value": to_stamp, "grain": "day"}
    return {"dim": "time", "value": value}


@asynccontextmanager
async def mock_extractor(handler, **kwargs):
    """A ``DucklingTemporalExtractor`` whose injected client is backed by ``handler``."""
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        yield DucklingTemporalExtractor(
            DucklingTemporalExtractorParams(client=client, url=URL, **kwargs)
        )


# --- pure mapping helpers (synthetic entity -> TimeRange) ------------------


class TestEntityMapping:
    def test_value_entity_is_single_bounded_interval(self):
        result = _entity_to_time_range(value_entity())
        assert isinstance(result, TimeRange)
        assert len(result.intervals) == 1
        interval = result.intervals[0]
        assert interval.start is not None
        assert interval.end is not None
        assert interval.start < interval.end  # half-open, non-empty

    def test_closed_interval_wires_from_to_start_and_to_to_end(self):
        start_stamp = "2020-01-01T00:00:00.000Z"
        end_stamp = "2024-01-01T00:00:00.000Z"
        result = _entity_to_time_range(interval_entity(start_stamp, end_stamp))
        assert isinstance(result, TimeRange)
        interval = result.intervals[0]
        # Field wiring (oracle is our own parser, not a version-specific value).
        assert interval.start == _parse_iso(start_stamp)
        assert interval.end == _parse_iso(end_stamp)

    def test_from_only_interval_is_left_bounded(self):
        result = _entity_to_time_range(
            interval_entity(from_stamp="2020-01-01T00:00:00.000Z")
        )
        assert isinstance(result, TimeRange)
        assert result.intervals[0].start is not None
        assert result.intervals[0].end is None

    def test_to_only_interval_is_right_bounded(self):
        result = _entity_to_time_range(
            interval_entity(to_stamp="2024-01-01T00:00:00.000Z")
        )
        assert isinstance(result, TimeRange)
        assert result.intervals[0].start is None
        assert result.intervals[0].end is not None

    def test_empty_interval_is_dropped(self):
        assert _entity_to_time_range(interval_entity()) is None

    def test_unknown_value_type_is_dropped(self):
        assert (
            _entity_to_time_range({"dim": "time", "value": {"type": "duration"}})
            is None
        )

    def test_missing_value_is_dropped(self):
        assert _entity_to_time_range({"dim": "time"}) is None

    def test_value_missing_timestamp_is_dropped(self):
        assert _value_to_time_range({"type": "value", "grain": "day"}) is None

    def test_value_unparseable_timestamp_is_dropped(self):
        assert (
            _value_to_time_range({"type": "value", "value": "nope", "grain": "day"})
            is None
        )

    def test_value_unknown_grain_is_dropped(self):
        entity = {
            "type": "value",
            "value": "2024-03-05T00:00:00.000Z",
            "grain": "fortnight",
        }
        assert _value_to_time_range(entity) is None


class TestIntervalBound:
    def test_absent_bound_is_none(self):
        assert _interval_bound(None) is None

    def test_present_bound_parses(self):
        stamp = "2024-01-01T00:00:00.000Z"
        assert _interval_bound({"value": stamp}) == _parse_iso(stamp)

    def test_malformed_bound_missing_value_raises(self):
        # Present-but-malformed must surface (so the caller can reject it),
        # distinct from an absent bound.
        with pytest.raises(KeyError):
            _interval_bound({"grain": "day"})

    def test_malformed_bound_bad_iso_raises(self):
        with pytest.raises(ValueError, match="isoformat"):
            _interval_bound({"value": "not-a-date"})

    def test_interval_with_malformed_bound_is_dropped(self):
        # The raise from a present-but-malformed bound is caught here -> the
        # whole interval is dropped rather than half-parsed.
        result = _interval_to_time_range({"type": "interval", "from": {"grain": "day"}})
        assert result is None


class TestEndOf:
    @pytest.mark.parametrize(
        "grain", ["second", "minute", "hour", "day", "week", "month", "quarter", "year"]
    )
    def test_known_grain_returns_later_utc_instant(self, grain):
        start = datetime(2024, 2, 15, tzinfo=UTC)
        end = _end_of(start, grain)
        assert end is not None
        assert end > start  # period end is strictly after its start
        assert end.utcoffset() == timedelta(0)

    def test_unknown_grain_is_none(self):
        assert _end_of(datetime(2024, 2, 15, tzinfo=UTC), "fortnight") is None


class TestParseIso:
    def test_returns_utc_aware(self):
        assert _parse_iso("2024-03-05T00:00:00.000Z").utcoffset() == timedelta(0)

    def test_normalizes_offset_to_utc(self):
        # A +08:00 stamp and its UTC equivalent denote the same instant.
        assert _parse_iso("2024-03-05T08:00:00.000+08:00") == _parse_iso(
            "2024-03-05T00:00:00.000Z"
        )


# --- DucklingTemporalExtractor (HTTP interface, via MockTransport) ---------


@pytest.mark.asyncio
class TestDucklingExtractor:
    async def test_returns_list_of_time_ranges(self):
        def handler(request):
            return httpx.Response(200, json=[value_entity()])

        async with mock_extractor(handler) as extractor:
            result = await extractor.extract(
                "anything", ref_time=datetime(2024, 6, 15, tzinfo=UTC)
            )
            assert isinstance(result, list)
            assert all(isinstance(r, TimeRange) for r in result)
            assert len(result) == 1

    async def test_request_shape(self):
        captured = {}

        def handler(request):
            captured["method"] = request.method
            captured["url"] = str(request.url)
            captured["form"] = urllib.parse.parse_qs(request.content.decode())
            return httpx.Response(200, json=[])

        ref = datetime(2024, 6, 15, 12, tzinfo=UTC)
        async with mock_extractor(handler, locale="en_GB") as extractor:
            await extractor.extract("when did x happen", ref_time=ref)

        assert captured["method"] == "POST"
        assert captured["url"] == URL
        form = captured["form"]
        assert form["text"] == ["when did x happen"]
        assert form["locale"] == ["en_GB"]
        assert form["tz"] == ["UTC"]
        assert json.loads(form["dims"][0]) == ["time"]
        # reftime is epoch-milliseconds of ref_time (our deterministic contract).
        assert form["reftime"] == [str(int(ref.timestamp() * 1000))]

    async def test_tz_is_derived_from_ref_time(self):
        captured = {}

        def handler(request):
            captured["form"] = urllib.parse.parse_qs(request.content.decode())
            return httpx.Response(200, json=[])

        ref = datetime(2024, 6, 15, 12, tzinfo=ZoneInfo("America/Los_Angeles"))
        async with mock_extractor(handler) as extractor:
            await extractor.extract("when did x happen", ref_time=ref)

        # tz comes from the input datetime's zone, not a config parameter.
        assert captured["form"]["tz"] == ["America/Los_Angeles"]
        # reftime stays the absolute UTC epoch of that instant.
        assert captured["form"]["reftime"] == [str(int(ref.timestamp() * 1000))]

    async def test_filters_out_non_time_dims(self):
        def handler(request):
            return httpx.Response(
                200, json=[value_entity(), {"dim": "number", "value": {"value": 7}}]
            )

        async with mock_extractor(handler) as extractor:
            result = await extractor.extract(
                "x", ref_time=datetime(2024, 1, 1, tzinfo=UTC)
            )
            assert len(result) == 1  # the non-time entity is ignored

    async def test_non_200_returns_empty(self):
        def handler(request):
            return httpx.Response(500)

        async with mock_extractor(handler) as extractor:
            assert (
                await extractor.extract("x", ref_time=datetime(2024, 1, 1, tzinfo=UTC))
                == []
            )

    async def test_invalid_json_returns_empty(self):
        def handler(request):
            return httpx.Response(200, text="not json")

        async with mock_extractor(handler) as extractor:
            assert (
                await extractor.extract("x", ref_time=datetime(2024, 1, 1, tzinfo=UTC))
                == []
            )

    async def test_transport_error_returns_empty(self):
        def handler(request):
            raise httpx.ConnectError("boom")

        async with mock_extractor(handler) as extractor:
            assert (
                await extractor.extract("x", ref_time=datetime(2024, 1, 1, tzinfo=UTC))
                == []
            )
