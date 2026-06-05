"""Temporal scoring."""

import math

from pydantic import BaseModel, Field

from .query_planner import TemporalQueryPlan
from .time_range import TimeRange


def temporal_match_score(target: TimeRange, anchor: TimeRange) -> float:
    """Return the temporal match score between a target and an anchor."""
    intersection_measure = (target & anchor).measure_seconds
    if intersection_measure <= 0.0:
        return 0.0

    target_measure = target.measure_seconds
    anchor_measure = anchor.measure_seconds

    if math.isinf(target_measure) and math.isinf(anchor_measure):
        return 1.0

    denominator = min(target_measure, anchor_measure)
    if denominator <= 0.0:
        return 0.0

    return min(intersection_measure / denominator, 1.0)


def document_temporal_match_score(
    targets: list[TimeRange],
    anchors: list[TimeRange],
) -> float:
    """
    Return the temporal match score for a document.

    The temporal match score for a document
    is the arithmetic mean over the query's targets
    of the max over the document's anchors of the
    per-pair match score between one target and one anchor.
    """
    if not targets:
        return 0.0

    return sum(
        max(
            (temporal_match_score(target, anchor) for anchor in anchors),
            default=0.0,
        )
        for target in targets
    ) / len(targets)


class TemporalScoringCandidate(BaseModel):
    """
    Temporal scoring candidate.

    Attributes:
        base_score (float): Base relevance score.
        anchors (list[TimeRange]): Anchor time ranges.
    """

    base_score: float
    anchors: list[TimeRange]


class TemporalScoringResult(BaseModel):
    """
    Temporal scoring result.

    Attributes:
        final_score (float): Final score for ordering candidates.
        temporal_match_score (float): Temporal match score.
    """

    final_score: float
    temporal_match_score: float


class TemporalScorerParams(BaseModel):
    """
    Parameters for TemporalScorer.

    Attributes:
        match_weight (float):
            Weight on the temporal match term in the additive fusion.
    """

    match_weight: float = Field(
        default=1.0,
        description=("Weight on the temporal match term in the additive fusion"),
    )


class TemporalScorer:
    """Temporal scorer."""

    def __init__(self, params: TemporalScorerParams) -> None:
        """Initialize from parameters."""
        self._match_weight = params.match_weight

    def score(
        self,
        plan: TemporalQueryPlan,
        candidates: list[TemporalScoringCandidate],
    ) -> list[TemporalScoringResult]:
        """
        Score candidates against a temporal query plan.

        Args:
            plan (TemporalQueryPlan):
                Temporal query plan for scoring candidates.
            candidates (list[TemporalScoringCandidate]):
                Candidates for temporal scoring.
        """
        if not candidates:
            return []

        base_scores = [candidate.base_score for candidate in candidates]
        match_scores = [
            document_temporal_match_score(plan.targets, candidate.anchors)
            for candidate in candidates
        ]

        # Scale the temporal match score by the pool's base score spread so the
        # base-match balance is stable across pools.
        pool_spread = (max(base_scores) - min(base_scores)) if base_scores else 1.0
        if pool_spread <= 0:
            pool_spread = 1.0

        return [
            TemporalScoringResult(
                final_score=base_scores[i]
                + self._match_weight * match_scores[i] * pool_spread,
                temporal_match_score=match_scores[i],
            )
            for i in range(len(candidates))
        ]
