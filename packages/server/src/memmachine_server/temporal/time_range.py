"""Time range types."""

import math
from collections.abc import Iterable
from datetime import UTC, datetime

from pydantic import BaseModel, Field, field_validator


def epoch_seconds(moment: datetime) -> float:
    """Seconds since the Unix epoch (naive datetimes are treated as UTC)."""
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=UTC)
    return moment.timestamp()


class TimeInterval(BaseModel):
    """
    A time interval.

    The interval is half-open when start and end are not None: [start, end).
    No lower bound when start is None.
    No upper bound when end is None.
    """

    start: datetime | None = None
    end: datetime | None = None

    @property
    def lower_endpoint(self) -> float:
        """Lower endpoint as a comparable real (`-inf` when unbounded)."""
        return -math.inf if self.start is None else epoch_seconds(self.start)

    @property
    def upper_endpoint(self) -> float:
        """Upper endpoint as a comparable real (`+inf` when unbounded)."""
        return math.inf if self.end is None else epoch_seconds(self.end)

    @property
    def measure_seconds(self) -> float:
        """Measure of the interval in seconds (`inf` if unbounded)."""
        return self.upper_endpoint - self.lower_endpoint

    def __and__(self, other: "TimeInterval") -> "TimeRange":
        """Intersection with another interval, as a time range."""
        intersection = _intersect(self, other)
        intervals = [intersection] if intersection is not None else []
        return TimeRange(intervals=intervals)

    def __or__(self, other: "TimeInterval") -> "TimeRange":
        """Union with another interval, as a time range."""
        return TimeRange(intervals=[self, other])


def _intersect(a: TimeInterval, b: TimeInterval) -> TimeInterval | None:
    """Intersection of two intervals as one interval, or `None` if disjoint."""
    start = a.start if a.lower_endpoint >= b.lower_endpoint else b.start
    end = a.end if a.upper_endpoint <= b.upper_endpoint else b.end
    intersection = TimeInterval(start=start, end=end)
    if intersection.lower_endpoint < intersection.upper_endpoint:
        return intersection
    return None


def _merge_intervals(intervals: Iterable[TimeInterval]) -> list[TimeInterval]:
    """Sort and merge intervals into a union of disjoint, non-adjacent time intervals, in chronological order."""
    endpoints_keyed_intervals = [
        (interval.lower_endpoint, interval.upper_endpoint, interval)
        for interval in intervals
        if interval.lower_endpoint < interval.upper_endpoint
    ]
    endpoints_keyed_intervals.sort(key=lambda item: (item[0], item[1]))

    merged: list[TimeInterval] = []
    run_upper = -math.inf
    for lower, upper, interval in endpoints_keyed_intervals:
        if merged and lower <= run_upper:
            # Overlap or adjacency: extend the run's end if this reaches further.
            if upper > run_upper:
                merged[-1] = TimeInterval(start=merged[-1].start, end=interval.end)
                run_upper = upper
        else:
            merged.append(interval)
            run_upper = upper
    return merged


class TimeRange(BaseModel):
    """A union of disjoint, non-adjacent time intervals, in chronological order."""

    intervals: list[TimeInterval] = Field(default_factory=list)

    @field_validator("intervals", mode="after")
    @classmethod
    def _canonicalize(cls, intervals: list[TimeInterval]) -> list[TimeInterval]:
        """Merge intervals into a disjoint, chronologically ordered union."""
        return _merge_intervals(intervals)

    @property
    def measure_seconds(self) -> float:
        """Measure of the range in seconds (`inf` if unbounded)."""
        return sum(interval.measure_seconds for interval in self.intervals)

    def __and__(self, other: "TimeRange") -> "TimeRange":
        """Set intersection with another range."""
        intersections: list[TimeInterval] = []
        for a in self.intervals:
            for b in other.intervals:
                intersection = _intersect(a, b)
                if intersection is not None:
                    intersections.append(intersection)
        return TimeRange(intervals=intersections)

    def __or__(self, other: "TimeRange") -> "TimeRange":
        """Set union with another range."""
        return TimeRange(intervals=[*self.intervals, *other.intervals])
