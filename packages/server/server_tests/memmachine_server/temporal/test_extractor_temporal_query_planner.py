"""Interface / structure tests for the extractor-backed temporal query planner.

Tests the unified planner (which wraps any ``TemporalExtractor``) with a
recording stub extractor: plan shape and ref-time pass-through --
independent of any specific extractor backend.
"""

from datetime import UTC, datetime
from typing import override

import pytest

from memmachine_server.temporal.extractor import TemporalExtractor
from memmachine_server.temporal.extractor.extractor_temporal_query_planner import (
    ExtractorTemporalQueryPlanner,
    ExtractorTemporalQueryPlannerParams,
)
from memmachine_server.temporal.query_planner import TemporalQueryPlan
from memmachine_server.temporal.time_range import TimeInterval, TimeRange


def year_range(year=2024):
    return TimeRange(
        intervals=[
            TimeInterval(
                start=datetime(year, 1, 1, tzinfo=UTC),
                end=datetime(year + 1, 1, 1, tzinfo=UTC),
            )
        ]
    )


class RecordingExtractor(TemporalExtractor):
    """A ``TemporalExtractor`` stub that records calls and returns fixed ranges."""

    def __init__(self, ranges):
        self._ranges = ranges
        self.calls = []

    @override
    async def extract(self, text, ref_time=None):
        self.calls.append((text, ref_time))
        return list(self._ranges)


def planner_with(extractor):
    return ExtractorTemporalQueryPlanner(
        ExtractorTemporalQueryPlannerParams(extractor=extractor)
    )


@pytest.mark.asyncio
class TestExtractorTemporalQueryPlanner:
    async def test_plan_passes_targets_through(self):
        ranges = [year_range()]
        planner = planner_with(RecordingExtractor(ranges))
        plan = await planner.plan(
            "when did x happen", ref_time=datetime(2024, 6, 15, tzinfo=UTC)
        )
        assert isinstance(plan, TemporalQueryPlan)
        assert plan.targets == ranges  # extractor output passed through

    async def test_passes_ref_time_through_to_extractor(self):
        extractor = RecordingExtractor([])
        planner = planner_with(extractor)
        ref_time = datetime(2024, 6, 15, tzinfo=UTC)
        await planner.plan("q", ref_time=ref_time)
        assert len(extractor.calls) == 1
        _text, passed = extractor.calls[0]
        assert passed is ref_time  # datetime passed straight through, no parsing
