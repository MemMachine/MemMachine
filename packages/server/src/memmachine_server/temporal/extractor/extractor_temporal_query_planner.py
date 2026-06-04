"""Temporal query planner backed by a temporal extractor."""

from datetime import UTC, datetime
from typing import override

from pydantic import BaseModel, Field, InstanceOf

from memmachine_server.temporal.query_planner import (
    TemporalQueryPlan,
    TemporalQueryPlanner,
)

from .temporal_extractor import TemporalExtractor


class ExtractorTemporalQueryPlannerParams(BaseModel):
    """Parameters for ExtractorTemporalQueryPlanner."""

    extractor: InstanceOf[TemporalExtractor] = Field(
        ...,
        description="Temporal extractor used to resolve the query's dates into targets",
    )


class ExtractorTemporalQueryPlanner(TemporalQueryPlanner):
    """Temporal query planner backed by an injected temporal extractor."""

    def __init__(self, params: ExtractorTemporalQueryPlannerParams) -> None:
        """Initialize with the injected extractor."""
        self._extractor = params.extractor

    @override
    async def plan(
        self, query: str, *, ref_time: datetime | None = None
    ) -> TemporalQueryPlan:
        if ref_time is None:
            ref_time = datetime.now(UTC)

        targets = await self._extractor.extract(query, ref_time=ref_time)
        return TemporalQueryPlan(targets=targets)
