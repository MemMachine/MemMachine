"""Tests for the EventMemory temporal-ranking adapter."""

from datetime import UTC, datetime
from uuid import uuid4

import pytest

from memmachine_server.episodic_memory.event_memory.data_types import (
    CompositeContext,
    NullContext,
    ProducerContext,
    QueryResult,
    ScoredSegmentContext,
    Segment,
    TextBlock,
    TimeRangesContext,
)
from memmachine_server.episodic_memory.event_memory.temporal_ranking import (
    _temporal_anchors_from_context,
    temporal_rerank_query_results,
)
from memmachine_server.temporal.query_planner import TemporalQueryPlan
from memmachine_server.temporal.scoring import TemporalScorer
from memmachine_server.temporal.time_range import TimeInterval, TimeRange

BASE_TIME = datetime(2024, 1, 1, tzinfo=UTC)


def dt(year: int, month: int = 1, day: int = 1) -> datetime:
    return datetime(year, month, day, tzinfo=UTC)


def anchor(start: datetime | None, end: datetime | None) -> TimeRange:
    """A single-interval ``TimeRange``."""
    return TimeRange(intervals=[TimeInterval(start=start, end=end)])


def year_target(y: int) -> TimeRange:
    return TimeRange(intervals=[TimeInterval(start=dt(y, 1, 1), end=dt(y + 1, 1, 1))])


def scored(base: float, anchors: list[TimeRange]) -> ScoredSegmentContext:
    seg = Segment(
        uuid=uuid4(),
        event_uuid=uuid4(),
        index=0,
        offset=0,
        timestamp=BASE_TIME,
        context=TimeRangesContext(time_ranges=anchors),
        block=TextBlock(text="x"),
    )
    return ScoredSegmentContext(score=base, seed_segment_uuid=seg.uuid, segments=[seg])


class TestAnchorsFromContext:
    def test_atomic_and_composed(self):
        temporal = TimeRangesContext(time_ranges=[anchor(dt(2024, 1), dt(2024, 2))])
        assert len(_temporal_anchors_from_context(temporal)) == 1

        composed = CompositeContext(
            contexts=[ProducerContext(producer="Alice"), temporal]
        )
        anchors = _temporal_anchors_from_context(composed)
        assert len(anchors) == 1
        assert anchors[0].intervals[0].start == dt(2024, 1)

        assert _temporal_anchors_from_context(NullContext()) == []
        assert _temporal_anchors_from_context(None) == []

    def test_aggregates_multiple_temporal_aspects(self):
        first = TimeRangesContext(time_ranges=[anchor(dt(2020), dt(2021))])
        second = TimeRangesContext(time_ranges=[anchor(dt(2024), dt(2025))])
        composed = CompositeContext(contexts=[first, second])

        anchors = _temporal_anchors_from_context(composed)
        assert [a.intervals[0].start for a in anchors] == [dt(2020), dt(2024)]

    def test_unbounded_endpoint_preserved(self):
        ctx = TimeRangesContext(time_ranges=[anchor(None, dt(2020, 1))])
        anchors = _temporal_anchors_from_context(ctx)
        assert anchors[0].intervals[0].start is None
        assert anchors[0].intervals[0].end == dt(2020, 1)


class TestRerankQueryResult:
    def test_returns_query_result_reordered_and_rescored(self):
        # A third doc widens the pool spread so the 2024 match's lift lands it
        # above the higher-base off-year doc.
        a = scored(0.50, [anchor(dt(2024, 3), dt(2024, 4))])
        b = scored(0.60, [anchor(dt(2020, 1), dt(2020, 2))])
        c = scored(0.40, [anchor(dt(2019, 1), dt(2019, 2))])
        query_result = QueryResult(scored_segment_contexts=[b, a, c])

        out = temporal_rerank_query_results(
            TemporalScorer(),
            TemporalQueryPlan(targets=[year_target(2024)]),
            query_result,
        )

        # Same shape: a QueryResult of the same contexts, nothing added/dropped.
        assert isinstance(out, QueryResult)
        assert {s.seed_segment_uuid for s in out.scored_segment_contexts} == {
            a.seed_segment_uuid,
            b.seed_segment_uuid,
            c.seed_segment_uuid,
        }
        # The 2024 match is first, carrying the temporally-fused score.
        assert out.scored_segment_contexts[0].seed_segment_uuid == a.seed_segment_uuid
        assert out.scored_segment_contexts[0].score == pytest.approx(
            0.70
        )  # 0.50 + 1.0*(0.60-0.40)

    def test_does_not_mutate_caller_input(self):
        a = scored(0.50, [anchor(dt(2024, 3), dt(2024, 4))])
        temporal_rerank_query_results(
            TemporalScorer(),
            TemporalQueryPlan(targets=[year_target(2024)]),
            QueryResult(scored_segment_contexts=[a]),
        )
        assert a.score == 0.50  # the caller's context keeps its original score

    def test_empty_plan_orders_by_base_and_keeps_scores(self):
        a = scored(0.30, [anchor(dt(2024, 3), dt(2024, 4))])
        b = scored(0.90, [anchor(dt(2020, 1), dt(2020, 2))])
        out = temporal_rerank_query_results(
            TemporalScorer(),
            TemporalQueryPlan(targets=[]),
            QueryResult(scored_segment_contexts=[a, b]),
        )
        # No targets -> no temporal lift -> ordered by base, scores unchanged.
        assert [s.seed_segment_uuid for s in out.scored_segment_contexts] == [
            b.seed_segment_uuid,
            a.seed_segment_uuid,
        ]
        assert out.scored_segment_contexts[0].score == 0.90

    def test_scores_over_all_segments_not_just_seed(self):
        # The seed carries no date; only a neighbor segment does. Temporal fit
        # is scored over the whole unit, so the neighbor's 2024 anchor lifts the
        # unit above a higher-base off-year doc -- which seed-only scoring would
        # not do.
        seed = Segment(
            uuid=uuid4(),
            event_uuid=uuid4(),
            index=0,
            offset=0,
            timestamp=BASE_TIME,
            context=NullContext(),
            block=TextBlock(text="seed"),
        )
        neighbor = Segment(
            uuid=uuid4(),
            event_uuid=uuid4(),
            index=1,
            offset=0,
            timestamp=BASE_TIME,
            context=TimeRangesContext(time_ranges=[anchor(dt(2024, 3), dt(2024, 4))]),
            block=TextBlock(text="neighbor"),
        )
        unit = ScoredSegmentContext(
            score=0.50, seed_segment_uuid=seed.uuid, segments=[seed, neighbor]
        )
        higher = scored(0.70, [anchor(dt(2020, 1), dt(2020, 2))])
        lower = scored(0.40, [anchor(dt(2019, 1), dt(2019, 2))])

        out = temporal_rerank_query_results(
            TemporalScorer(),
            TemporalQueryPlan(targets=[year_target(2024)]),
            QueryResult(scored_segment_contexts=[higher, unit, lower]),
        )

        # Pool spread = 0.70 - 0.40 = 0.30; the neighbor's 2024 match lifts the
        # unit to 0.50 + 1.0*0.30 = 0.80, above the higher-base (0.70) off-year doc.
        assert out.scored_segment_contexts[0].seed_segment_uuid == seed.uuid
        assert out.scored_segment_contexts[0].score == pytest.approx(0.80)
