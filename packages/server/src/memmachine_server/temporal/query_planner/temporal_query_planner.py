"""Temporal query planning."""

from abc import ABC, abstractmethod
from datetime import datetime

from pydantic import BaseModel, Field

from memmachine_server.temporal.time_range import TimeRange


class TemporalQueryPlan(BaseModel):
    """Temporal query plan."""

    targets: list[TimeRange] = Field(default_factory=list)


class TemporalQueryPlanner(ABC):
    """Abstract base class for a temporal query planner."""

    @abstractmethod
    async def plan(
        self, query: str, *, ref_time: datetime | None = None
    ) -> TemporalQueryPlan:
        """
        Make a temporal query plan for the query.

        Args:
            query (str):
                The query to make a plan for.
            ref_time (str):
                The reference time for resolving relative time references.

        Returns:
            TemporalQueryPlan:
                The temporal query plan.
        """
        raise NotImplementedError
