"""Tests for TimeRange canonicalization, & (intersection), and .measure_seconds."""

import math
from datetime import UTC, datetime

from memmachine_server.temporal.time_range import TimeInterval, TimeRange

DAY = 86400.0  # seconds


def dt(day: int) -> datetime:
    return datetime(2024, 1, day, tzinfo=UTC)


def bounds(time_range: TimeRange) -> list[tuple[datetime | None, datetime | None]]:
    return [(iv.start, iv.end) for iv in time_range.intervals]


def test_empty_stays_empty():
    assert TimeRange().intervals == []
    assert TimeRange(intervals=[]).intervals == []


def test_single_interval_unchanged():
    tr = TimeRange(intervals=[TimeInterval(start=dt(1), end=dt(2))])
    assert bounds(tr) == [(dt(1), dt(2))]


def test_orders_chronologically():
    tr = TimeRange(
        intervals=[
            TimeInterval(start=dt(5), end=dt(6)),
            TimeInterval(start=dt(1), end=dt(2)),
            TimeInterval(start=dt(3), end=dt(4)),
        ]
    )
    assert bounds(tr) == [(dt(1), dt(2)), (dt(3), dt(4)), (dt(5), dt(6))]


def test_merges_overlapping():
    tr = TimeRange(
        intervals=[
            TimeInterval(start=dt(1), end=dt(4)),
            TimeInterval(start=dt(3), end=dt(6)),
        ]
    )
    assert bounds(tr) == [(dt(1), dt(6))]


def test_merges_adjacent_half_open():
    # [1, 3) and [3, 5) touch only at the excluded boundary -> one [1, 5).
    tr = TimeRange(
        intervals=[
            TimeInterval(start=dt(1), end=dt(3)),
            TimeInterval(start=dt(3), end=dt(5)),
        ]
    )
    assert bounds(tr) == [(dt(1), dt(5))]


def test_keeps_disjoint_nonadjacent():
    tr = TimeRange(
        intervals=[
            TimeInterval(start=dt(1), end=dt(2)),
            TimeInterval(start=dt(4), end=dt(5)),
        ]
    )
    assert bounds(tr) == [(dt(1), dt(2)), (dt(4), dt(5))]


def test_absorbs_contained_interval():
    tr = TimeRange(
        intervals=[
            TimeInterval(start=dt(1), end=dt(9)),
            TimeInterval(start=dt(3), end=dt(4)),
        ]
    )
    assert bounds(tr) == [(dt(1), dt(9))]


def test_collapses_duplicates():
    tr = TimeRange(
        intervals=[
            TimeInterval(start=dt(1), end=dt(2)),
            TimeInterval(start=dt(1), end=dt(2)),
        ]
    )
    assert bounds(tr) == [(dt(1), dt(2))]


def test_drops_empty_and_inverted_intervals():
    tr = TimeRange(
        intervals=[
            TimeInterval(start=dt(2), end=dt(2)),  # zero width
            TimeInterval(start=dt(5), end=dt(3)),  # inverted
            TimeInterval(start=dt(1), end=dt(2)),
        ]
    )
    assert bounds(tr) == [(dt(1), dt(2))]


def test_unbounded_endpoints_merge_to_universal():
    # (-inf, 3) and [2, +inf) overlap -> the whole line.
    tr = TimeRange(
        intervals=[
            TimeInterval(start=None, end=dt(3)),
            TimeInterval(start=dt(2), end=None),
        ]
    )
    assert bounds(tr) == [(None, None)]


def test_unbounded_left_sorts_first():
    tr = TimeRange(
        intervals=[
            TimeInterval(start=dt(5), end=dt(6)),
            TimeInterval(start=None, end=dt(2)),
        ]
    )
    assert bounds(tr) == [(None, dt(2)), (dt(5), dt(6))]


def test_canonicalization_is_idempotent_across_round_trip():
    tr = TimeRange(
        intervals=[
            TimeInterval(start=dt(1), end=dt(4)),
            TimeInterval(start=dt(3), end=dt(6)),
        ]
    )
    again = TimeRange.model_validate(tr.model_dump())
    assert again == tr
    assert bounds(again) == [(dt(1), dt(6))]


# --- measure_seconds ------------------------------------------------------


def test_interval_measure_is_length_in_seconds():
    assert TimeInterval(start=dt(1), end=dt(3)).measure_seconds == 2 * DAY


def test_interval_measure_unbounded_is_inf():
    assert TimeInterval(start=dt(1), end=None).measure_seconds == math.inf
    assert TimeInterval(start=None, end=dt(1)).measure_seconds == math.inf


def test_range_measure_sums_intervals():
    tr = TimeRange(
        intervals=[
            TimeInterval(start=dt(1), end=dt(2)),
            TimeInterval(start=dt(4), end=dt(6)),
        ]
    )
    assert tr.measure_seconds == 3 * DAY


def test_empty_range_measure_is_zero():
    assert TimeRange().measure_seconds == 0


# --- & (intersection) -----------------------------------------------------


def test_interval_intersection_overlapping():
    a = TimeInterval(start=dt(1), end=dt(4))
    b = TimeInterval(start=dt(3), end=dt(6))
    assert bounds(a & b) == [(dt(3), dt(4))]


def test_interval_intersection_disjoint_is_empty():
    a = TimeInterval(start=dt(1), end=dt(2))
    b = TimeInterval(start=dt(4), end=dt(5))
    assert (a & b).intervals == []


def test_range_intersection_keeps_overlap_per_piece():
    a = TimeRange(
        intervals=[
            TimeInterval(start=dt(1), end=dt(4)),
            TimeInterval(start=dt(6), end=dt(9)),
        ]
    )
    b = TimeRange(intervals=[TimeInterval(start=dt(3), end=dt(7))])
    # a & b keeps each piece's overlap: [3, 4) and [6, 7)
    assert bounds(a & b) == [(dt(3), dt(4)), (dt(6), dt(7))]


def test_range_intersection_disjoint_is_empty():
    a = TimeRange(intervals=[TimeInterval(start=dt(1), end=dt(2))])
    b = TimeRange(intervals=[TimeInterval(start=dt(5), end=dt(6))])
    assert (a & b).intervals == []


def test_range_intersection_unbounded():
    a = TimeRange(intervals=[TimeInterval(start=None, end=dt(5))])
    b = TimeRange(intervals=[TimeInterval(start=dt(3), end=None)])
    assert bounds(a & b) == [(dt(3), dt(5))]


def test_intersection_then_measure_is_overlap_length():
    a = TimeRange(intervals=[TimeInterval(start=dt(1), end=dt(4))])
    b = TimeRange(intervals=[TimeInterval(start=dt(3), end=dt(6))])
    assert (a & b).measure_seconds == DAY  # overlap [3, 4) is one day


# --- | (union) ------------------------------------------------------------


def test_interval_union_overlapping_merges():
    a = TimeInterval(start=dt(1), end=dt(4))
    b = TimeInterval(start=dt(3), end=dt(6))
    assert bounds(a | b) == [(dt(1), dt(6))]


def test_interval_union_adjacent_merges():
    a = TimeInterval(start=dt(1), end=dt(3))
    b = TimeInterval(start=dt(3), end=dt(5))
    assert bounds(a | b) == [(dt(1), dt(5))]


def test_interval_union_disjoint_keeps_both():
    a = TimeInterval(start=dt(1), end=dt(2))
    b = TimeInterval(start=dt(4), end=dt(5))
    assert bounds(a | b) == [(dt(1), dt(2)), (dt(4), dt(5))]


def test_range_union_merges_across_ranges():
    a = TimeRange(
        intervals=[
            TimeInterval(start=dt(1), end=dt(3)),
            TimeInterval(start=dt(7), end=dt(9)),
        ]
    )
    b = TimeRange(intervals=[TimeInterval(start=dt(2), end=dt(5))])
    # [1, 3) and [2, 5) merge to [1, 5); [7, 9) stays separate.
    assert bounds(a | b) == [(dt(1), dt(5)), (dt(7), dt(9))]


def test_range_union_disjoint_keeps_separate():
    a = TimeRange(intervals=[TimeInterval(start=dt(1), end=dt(2))])
    b = TimeRange(intervals=[TimeInterval(start=dt(5), end=dt(6))])
    assert bounds(a | b) == [(dt(1), dt(2)), (dt(5), dt(6))]


def test_union_then_measure_is_covered_length():
    a = TimeRange(intervals=[TimeInterval(start=dt(1), end=dt(4))])
    b = TimeRange(intervals=[TimeInterval(start=dt(3), end=dt(6))])
    assert (a | b).measure_seconds == 5 * DAY  # union [1, 6) covers five days
