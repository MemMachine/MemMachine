"""Interface / structure tests for the dateparser-backed temporal extractor.

These assert the structural contract -- return types, the extract
contract and the pure interval-expansion helpers -- not the
resolved date values, which depend on the ``dateparser`` library version.
"""

from datetime import UTC, datetime, timedelta

import pytest

pytest.importorskip("dateparser")

from babel import Locale

from memmachine_server.common.request_context import (
    reset_request_locale,
    set_request_locale,
)
from memmachine_server.temporal.extractor.dateparser_temporal_extractor import (
    DateparserTemporalExtractor,
    _interval_for,
)
from memmachine_server.temporal.time_range import TimeRange

# --- pure interval-expansion helpers ---------------------------------------


class TestIntervalFor:
    @pytest.mark.parametrize("period", ["day", "week", "month", "year"])
    def test_known_period_half_open_contains_instant(self, period):
        instant = datetime(2024, 2, 15, 9, 30, tzinfo=UTC)
        interval = _interval_for(instant, period)
        assert interval is not None
        assert interval.start is not None
        assert interval.end is not None
        # Half-open containment: start <= instant < end.
        assert interval.start <= instant < interval.end
        assert interval.start.utcoffset() == timedelta(0)

    def test_unknown_period_is_none(self):
        assert _interval_for(datetime(2024, 2, 15, tzinfo=UTC), "decade") is None

    def test_naive_instant_treated_as_utc(self):
        naive = datetime(2024, 2, 15, tzinfo=UTC).replace(tzinfo=None)
        interval = _interval_for(naive, "day")
        assert interval is not None
        assert interval.start is not None
        assert interval.start.utcoffset() == timedelta(0)


# --- DateparserTemporalExtractor --------------------------------------------


@pytest.mark.asyncio
class TestDateparserExtractor:
    async def test_returns_list_of_time_ranges(self):
        extractor = DateparserTemporalExtractor()
        result = await extractor.extract(
            "the launch was in 2020", ref_time=datetime(2024, 6, 15, tzinfo=UTC)
        )
        assert isinstance(result, list)
        assert all(isinstance(r, TimeRange) for r in result)

    async def test_no_date_text_returns_empty_list(self):
        extractor = DateparserTemporalExtractor()
        result = await extractor.extract(
            "no date here at all", ref_time=datetime(2024, 6, 15, tzinfo=UTC)
        )
        assert result == []

    async def test_uses_request_locale_language(self, monkeypatch):
        # dateparser is fed the language subtag of the request locale.
        extractor = DateparserTemporalExtractor()
        captured = {}

        def fake_extract(text, ref_time, languages):
            captured["languages"] = languages
            return []

        monkeypatch.setattr(extractor, "_extract_time_ranges", fake_extract)
        token = set_request_locale(Locale("fr", "FR"))
        try:
            await extractor.extract("hier", ref_time=datetime(2024, 6, 15, tzinfo=UTC))
        finally:
            reset_request_locale(token)

        assert captured["languages"] == ["fr"]
