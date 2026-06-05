"""Rule-based temporal extractor using dateparser."""

import calendar
from datetime import UTC, datetime, timedelta
from typing import override

from dateparser.date import DateDataParser
from dateparser.search import search_dates

from memmachine_server.common.utils import ensure_tz_aware
from memmachine_server.temporal.time_range import TimeInterval, TimeRange

from .temporal_extractor import TemporalExtractor

_VALID_PERIODS = ("day", "week", "month", "year")


def _interval_for(instant: datetime, period: str) -> TimeInterval | None:
    """Expand a parsed ``instant`` into the interval containing it.

    The returned ``[start, end)`` interval spans the ``period`` grain
    (day / week / month / year) around ``instant``. Returns ``None`` for
    unknown periods.
    """
    instant = ensure_tz_aware(instant).astimezone(UTC)
    if period == "day":
        start = instant.replace(hour=0, minute=0, second=0, microsecond=0)
        end = start + timedelta(days=1)
    elif period == "week":
        # Expand to the ISO week containing ``instant`` (Monday-Sunday).
        midnight = instant.replace(hour=0, minute=0, second=0, microsecond=0)
        monday = midnight - timedelta(days=instant.weekday())
        start = monday
        end = monday + timedelta(days=7)
    elif period == "month":
        start = datetime(instant.year, instant.month, 1, tzinfo=UTC)
        last_day = calendar.monthrange(instant.year, instant.month)[1]
        end = datetime(instant.year, instant.month, last_day, tzinfo=UTC) + timedelta(
            days=1
        )
    elif period == "year":
        start = datetime(instant.year, 1, 1, tzinfo=UTC)
        end = datetime(instant.year + 1, 1, 1, tzinfo=UTC)
    else:
        return None
    return TimeInterval(start=start, end=end)


class DateparserTemporalExtractor(TemporalExtractor):
    """Rule-based ``TimeRange`` extractor (no LLM).

    Configured with English language restriction and past-leaning
    disambiguation, matching the LoCoMo-style "describe a past event"
    profile.
    """

    def _settings(self, ref_time: datetime) -> dict:
        return {
            "RELATIVE_BASE": ensure_tz_aware(ref_time).astimezone(UTC),
            "PREFER_DATES_FROM": "past",
        }

    def _extract_time_ranges(self, text: str, ref_time: datetime) -> list[TimeRange]:
        settings = self._settings(ref_time)
        # DateDataParser takes settings only at construction time, so we
        # build a fresh parser per call (its construction cost is small;
        # ``RELATIVE_BASE`` is the per-call signal).
        parser = DateDataParser(languages=["en"], settings=settings)
        candidates = search_dates(text, languages=["en"], settings=settings)
        if not candidates:
            return []
        intervals: list[TimeInterval] = []
        for matched, _instant in candidates:
            data = parser.get_date_data(matched)
            if data.date_obj is None:
                continue
            if data.period not in _VALID_PERIODS:
                continue
            interval = _interval_for(data.date_obj, data.period)
            if interval is not None:
                intervals.append(interval)
        if not intervals:
            return []
        # Deduplicate exact-match intervals (the same date phrase can
        # match multiple substrings) and emit each as its own reference.
        seen: set[tuple] = set()
        ranges: list[TimeRange] = []
        for interval in intervals:
            key = (
                interval.start.isoformat() if interval.start else None,
                interval.end.isoformat() if interval.end else None,
            )
            if key in seen:
                continue
            seen.add(key)
            ranges.append(TimeRange(intervals=[interval]))
        return ranges

    @override
    async def extract(
        self, text: str, *, ref_time: datetime | None = None
    ) -> list[TimeRange]:
        if ref_time is None:
            ref_time = datetime.now(UTC)

        return self._extract_time_ranges(text, ref_time)
