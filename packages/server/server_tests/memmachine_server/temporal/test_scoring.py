"""Tests for temporal match scoring and the generic TemporalScorer."""

from datetime import UTC, datetime

from memmachine_server.temporal.query_planner import TemporalQueryPlan
from memmachine_server.temporal.scoring import (
    TemporalScorer,
    TemporalScorerParams,
    TemporalScoringCandidate,
    TemporalScoringResult,
    document_temporal_match_score,
    temporal_match_score,
)
from memmachine_server.temporal.time_range import TimeInterval, TimeRange


def dt(year: int, month: int = 1, day: int = 1) -> datetime:
    return datetime(year, month, day, tzinfo=UTC)


def interval(start: datetime | None, end: datetime | None) -> TimeInterval:
    return TimeInterval(start=start, end=end)


def month(year: int, month_num: int) -> TimeRange:
    """A single-month range (one interval)."""
    end = dt(year + 1, 1, 1) if month_num == 12 else dt(year, month_num + 1, 1)
    return TimeRange(intervals=[interval(dt(year, month_num, 1), end)])


def year(y: int) -> TimeRange:
    """A whole-year range (one interval)."""
    return TimeRange(intervals=[interval(dt(y, 1, 1), dt(y + 1, 1, 1))])


def anchor(start: datetime, end: datetime) -> TimeRange:
    """A single-interval ``TimeRange`` anchor."""
    return TimeRange(intervals=[interval(start, end)])


def candidate(base: float, anchors: list[TimeRange]) -> TemporalScoringCandidate:
    return TemporalScoringCandidate(base_score=base, anchors=anchors)


def ranked_indices(scores: list[TemporalScoringResult]) -> list[int]:
    """Candidate indices ordered by ``final_score`` descending -- the ranking."""
    return sorted(range(len(scores)), key=lambda i: scores[i].final_score, reverse=True)


# --- temporal_match_score ----------------------------------------------------


def test_temporal_match_score_identical() -> None:
    assert temporal_match_score(year(2024), year(2024)) == 1.0


def test_temporal_match_score_anchor_narrower_than_target() -> None:
    # Anchor (one month) fully inside the target year -> full overlap of min.
    assert temporal_match_score(year(2024), month(2024, 6)) == 1.0


def test_temporal_match_score_disjoint_is_zero() -> None:
    assert temporal_match_score(year(2024), month(2020, 3)) == 0.0


def test_temporal_match_score_partial_is_fraction_of_min() -> None:
    target = TimeRange(intervals=[interval(dt(2024, 1, 1), dt(2024, 1, 3))])  # 2 days
    anchor_ = TimeRange(intervals=[interval(dt(2024, 1, 2), dt(2024, 1, 4))])  # 2 days
    # Intersection is one day; min measure is two days -> 0.5.
    assert temporal_match_score(target, anchor_) == 0.5


def test_temporal_match_score_both_unbounded_is_one() -> None:
    target = TimeRange(intervals=[interval(dt(2020), None)])
    anchor_ = TimeRange(intervals=[interval(dt(2021), None)])
    assert temporal_match_score(target, anchor_) == 1.0


def test_temporal_match_score_multi_interval_target_matches_any_piece() -> None:
    target = TimeRange(
        intervals=[
            interval(dt(2020, 1), dt(2020, 2)),
            interval(dt(2024, 11), dt(2024, 12)),
        ]
    )
    assert temporal_match_score(target, month(2024, 11)) == 1.0


def test_temporal_match_score_multi_interval_anchor_honors_gap() -> None:
    # The doc-side fix: an anchor that is itself a period with a gap, e.g.
    # "active in 2024 except over the summer" -> [Jan, Jun) and [Sep, Jan).
    anchor_ = TimeRange(
        intervals=[
            interval(dt(2024, 1), dt(2024, 6)),
            interval(dt(2024, 9), dt(2025, 1)),
        ]
    )
    # A June query lands in the gap (misses); November lands in the 2nd piece.
    assert temporal_match_score(month(2024, 6), anchor_) == 0.0
    assert temporal_match_score(month(2024, 11), anchor_) == 1.0


# --- document_temporal_match_score (graded coverage) -------------------------


def test_document_temporal_match_score_no_targets_is_zero() -> None:
    # No temporal constraint -> 0.0 for every doc (a constant), so ranking
    # falls back to base.
    assert document_temporal_match_score([], [month(2024, 3)]) == 0.0


def test_document_temporal_match_score_no_anchors_is_zero() -> None:
    # A timeless doc (no anchors) gets no temporal lift: 0.0, the same as a
    # date mismatch. Additive fusion leaves it on its base score, not buried.
    assert document_temporal_match_score([year(2024)], []) == 0.0


def test_document_temporal_match_score_two_targets_one_match_is_half() -> None:
    targets = [year(2020), year(2024)]
    assert document_temporal_match_score(targets, [month(2020, 3)]) == 0.5


def test_document_temporal_match_score_two_targets_both_match_is_one() -> None:
    targets = [year(2020), year(2024)]
    assert (
        document_temporal_match_score(targets, [month(2020, 3), month(2024, 11)]) == 1.0
    )


# --- TemporalScorer ----------------------------------------------------------


def test_temporal_match_lifts_lower_base_candidate():
    a = candidate(0.50, [anchor(dt(2024, 3), dt(2024, 4))])  # matches 2024
    b = candidate(0.60, [anchor(dt(2020, 1), dt(2020, 2))])
    c = candidate(0.40, [anchor(dt(2019, 1), dt(2019, 2))])

    plan = TemporalQueryPlan(targets=[year(2024)])
    scores = TemporalScorer(TemporalScorerParams()).score(plan, [a, b, c])

    # The 2024 match takes the top final_score despite its lower base.
    assert ranked_indices(scores)[0] == 0
    assert scores[0].temporal_match_score == 1.0


def test_scores_are_returned_in_candidate_order():
    a = candidate(0.50, [anchor(dt(2024, 3), dt(2024, 4))])  # matches 2024
    b = candidate(0.60, [anchor(dt(2020, 1), dt(2020, 2))])  # does not

    plan = TemporalQueryPlan(targets=[year(2024)])
    scores = TemporalScorer(TemporalScorerParams()).score(plan, [a, b])

    assert scores[0].temporal_match_score == 1.0
    assert scores[1].temporal_match_score == 0.0


def test_empty_plan_ranks_by_base():
    a = candidate(0.30, [anchor(dt(2024, 3), dt(2024, 4))])
    b = candidate(0.90, [anchor(dt(2020, 1), dt(2020, 2))])
    c = candidate(0.60, [])

    plan = TemporalQueryPlan(targets=[])
    scores = TemporalScorer(TemporalScorerParams()).score(plan, [a, b, c])

    # No targets -> no temporal lift -> ordered by base: b, c, a.
    assert ranked_indices(scores) == [1, 2, 0]
    assert all(s.temporal_match_score == 0.0 for s in scores)


def test_empty_candidates():
    plan = TemporalQueryPlan(targets=[year(2024)])
    assert TemporalScorer(TemporalScorerParams()).score(plan, []) == []
