"""Transparent temporal reranking layer over EventMemory query results."""

from collections.abc import Iterable

from memmachine_server.episodic_memory.event_memory.data_types import (
    Context,
    QueryResult,
    ScoredSegmentContext,
    Segment,
    TimeRangesContext,
    find_contexts,
)
from memmachine_server.temporal.query_planner import TemporalQueryPlan
from memmachine_server.temporal.scoring import (
    TemporalScorer,
    TemporalScoringCandidate,
    TemporalScoringResult,
)
from memmachine_server.temporal.time_range import TimeRange


def _temporal_anchors_from_context(context: Context | None) -> list[TimeRange]:
    """Return the time ranges of the context."""
    return [
        time_range
        for time_range_context in find_contexts(context, TimeRangesContext)
        for time_range in time_range_context.time_ranges
    ]


def _temporal_anchors_from_segments(segments: Iterable[Segment]) -> list[TimeRange]:
    """Merge the time ranges of the segments."""
    return [
        anchor
        for segment in segments
        for anchor in _temporal_anchors_from_context(segment.context)
    ]


def temporal_rerank_query_results(
    scorer: TemporalScorer,
    plan: TemporalQueryPlan,
    query_result: QueryResult,
) -> QueryResult:
    """Rerank query results."""
    scores = scorer.score(
        plan, _temporal_scoring_candidates_from_query_result(query_result)
    )
    return QueryResult(
        scored_segment_contexts=_temporal_rerank_rescored_segment_contexts(
            query_result.scored_segment_contexts, scores
        )
    )


def _temporal_scoring_candidates_from_query_result(
    query_result: QueryResult,
) -> list[TemporalScoringCandidate]:
    """Produce temporal scoring candidates from query results."""
    return [
        TemporalScoringCandidate(
            base_score=scored.score,
            anchors=_temporal_anchors_from_segments(scored.segments),
        )
        for scored in query_result.scored_segment_contexts
    ]


def _temporal_rerank_rescored_segment_contexts(
    contexts: list[ScoredSegmentContext],
    scores: list[TemporalScoringResult],
) -> list[ScoredSegmentContext]:
    """Order segment contexts by their new scores."""
    ranked = sorted(
        zip(contexts, scores, strict=True),
        key=lambda pair: pair[1].final_score,
        reverse=True,
    )
    return [
        context.model_copy(update={"score": score.final_score})
        for context, score in ranked
    ]
