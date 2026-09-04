"""LLM-based temporal query planner."""

from datetime import UTC, datetime
from typing import override

from pydantic import BaseModel, Field, InstanceOf

from memmachine_server.common.language_model.language_model import LanguageModel
from memmachine_server.temporal.time_range import TimeInterval, TimeRange

from .temporal_query_planner import TemporalQueryPlan, TemporalQueryPlanner

_PROMPT = """
You translate a natural-language query into a TIME RANGE PLAN.

A query describes WHAT MOMENTS IN TIME a matching document's date anchor
should fall inside. You describe that set as a list of TARGETS. Each target
is a SET of allowed moments expressed as one or more half-open intervals
[start, end). Null endpoints mean unbounded (start=null is -infinity; end=null
is +infinity).

OUTPUT SHAPE
============
{{
  "targets": [
    {{"intervals": [{{"start": "YYYY-MM-DD"|null, "end": "YYYY-MM-DD"|null}}, ...]}}
  ]
}}

`targets` are concrete time sets you have resolved. A doc anchor scores
higher when it satisfies MORE targets -- mean of per-target overlap.

KEY CONCEPTS
============
- An INTERVAL is half-open [start, end). end is EXCLUSIVE: "March 2024" =
  start "2024-03-01", end "2024-04-01".
- A TARGET is a SET of allowed moments. The doc anchor satisfies a target
  if it falls in ANY of the target's intervals (intra-target OR via
  multi-interval).
- Multiple TARGETS = each target is a SEPARATE constraint. The doc is
  scored by how many it satisfies (graded coverage).

WHEN TO EMIT ONE TARGET (MULTI-INTERVAL) vs MULTIPLE TARGETS
============================================================
This is the planner's only structural decision.

ONE target with multiple intervals -- use when the query describes a SINGLE
allowed REGION that happens to have holes or discontinuities. The
intervals are interchangeable: ANY ONE of them satisfies the user.
  - "not in 2023" -> one target = (-inf, 2023) and (2024, +inf)
  - "in 2024 not in summer" -> one target = [Jan-Jun 2024] and [Sep-Dec 2024]
  - "between A and B" -> one target = [A.start, B.end)
  - any single contiguous period ("in 2024", "after March 2020") -> one target

MULTIPLE targets -- use when the query lists SEPARATE periods the doc should
match independently. Coverage matters: matching both > matching one.
  - "in 2020 and 2024" (colloquial, disjoint) -> two targets (one per year)
  - "in 2020 or 2024" (explicit OR, disjoint) -> two targets (graded coverage)
  - "in Q1 or Q4 of 2023" -> two targets (each quarter is its own period)

The litmus: if a doc that mentions BOTH periods should rank higher than
one that mentions ONE -> emit multiple targets. If they're interchangeable
(any one is fine) -> emit one multi-interval target.

REF_TIME is provided for resolving relative phrases ("recently", "two
weeks ago", "last quarter"). For absolute dates you don't need it.

VERB-POLARITY RULE -- CRITICAL
==============================
"not" / "didn't" / "did not" / "wasn't" attached to a VERB is EVENT
POLARITY, not temporal scoping. IGNORE it. Emit the same plan as if the
verb were affirmative. ONLY treat "not" as temporal scoping when it
attaches DIRECTLY to a temporal preposition ("not in X", "not during Y",
"not before Z").

  "what did not happen in 2024" -- "not" attaches to the verb "happen",
    NOT to "in 2024". Treat as: "what happened in 2024" -> one target [2024].
  "what wasn't completed by March" -- "wasn't" is verb polarity. Treat as
    "what was completed by March" -> one target (-inf, March).

  Contrast with temporal-scoping negation:
  "what happened NOT in 2024" -> complement of 2024 (rare phrasing).
  "what happened outside 2024" -> complement of 2024.
  "what happened excluding 2024" -> complement of 2024.

COMPOSITION RULES (do these AT THE LLM LEVEL -- emit the composed result)
========================================================================
- "in X" -> ONE target = [X.start, X.end).
- "after X" -> ONE target = [X.end, null). (X.end because "after X" excludes X.)
- "before X" -> ONE target = [null, X.start).
- "not in X" / "outside X" / "excluding X" -> ONE target = complement of [X]
  = two intervals [null, X.start) and [X.end, null).
- "in A not in B" (with B inside A) -> ONE target = A minus B = two intervals
  [A.start, B.start) and [B.end, A.end).
- "not in A or B" (A and B disjoint) -> ONE target = three intervals
  [null, A.start), [A.end, B.start), [B.end, null).
- "in A and B" (colloquial; A and B disjoint dates) -> TWO targets (one for
  A, one for B). DO NOT intersect them -- the intersection is empty.
- "in A or B" / "either A or B" -> TWO targets.
- "between A and B" -> ONE target = [A.start, B.end) (inclusive of both endpoints).
- "since X" / "starting X" / "from X onwards" -> ONE target = [X.start, null).
- "until X" -> ONE target = [null, X.end).

EMPTY OUTPUT
============
If the query has NO temporal scope at all (e.g., "how do I plan my
morning?", "lessons from the launch", "how did the migration go?") emit
{{"targets": []}}.

"What did I do recently" / "show me what happened lately" -> deictic;
resolve to a recent window (e.g. last 60-90 days from REF_TIME) as a
target.

DEICTIC RESOLUTION
==================
Resolve deictic phrases against REF_TIME and emit them as targets.

Resolutions:
- "this year" -> [Jan 1 of ref_time year, Jan 1 of next year)
- "last year" -> year before
- "this quarter" / "last quarter" / "next quarter" -> corresponding calendar quarter
- "this month" / "last month" / "next month" -> corresponding calendar month
- "yesterday" -> 1-day interval before ref_time's date
- "today" -> ref_time's date (1-day)
- "this week" / "last week" / "next week" -> Mon-Sun week-of
- "two weeks ago" / "three months ago" -> resolve arithmetically

EXAMPLES
========

Query: "in March 2024"
{{"targets":[{{"intervals":[{{"start":"2024-03-01","end":"2024-04-01"}}]}}]}}

Query: "after 2020"
{{"targets":[{{"intervals":[{{"start":"2021-01-01","end":null}}]}}]}}

Query: "not in 2023"
{{"targets":[{{"intervals":[{{"start":null,"end":"2023-01-01"}},{{"start":"2024-01-01","end":null}}]}}]}}

Query: "in Q1 or Q4 of 2023"
{{"targets":[
  {{"intervals":[{{"start":"2023-01-01","end":"2023-04-01"}}]}},
  {{"intervals":[{{"start":"2023-10-01","end":"2024-01-01"}}]}}
]}}

Query: "between 2020 and 2024"
{{"targets":[{{"intervals":[{{"start":"2020-01-01","end":"2025-01-01"}}]}}]}}

Query: "How do I plan my morning?"
{{"targets":[]}}

RETRIEVAL ORIENTATION
=====================
You are translating queries for a SEARCH/RETRIEVAL system over a memory
of PAST events. When a query asks about recurring or ambiguous-direction
patterns ("what do I do on Thursdays?", "when do I exercise?"),
enumerate PAST occurrences (looking back from REF_TIME), not future
ones. Future-tense queries ("what is scheduled for next Thursday") of
course remain future-oriented.

NOW PRODUCE THE PLAN FOR:

Query: {query}
Reference time: {ref_time}
"""


# Data models for structured outputs specification.


class _Interval(BaseModel):
    start: str | None = None
    end: str | None = None


class _Target(BaseModel):
    intervals: list[_Interval] = Field(default_factory=list)


class _PlanResponse(BaseModel):
    targets: list[_Target] = Field(default_factory=list)


def _iso_to_datetime(s: str) -> datetime:
    """Parse an ISO date/datetime into a UTC-aware datetime."""
    try:
        dt = datetime.fromisoformat(s)
    except (ValueError, TypeError) as e:
        raise ValueError(f"Cannot parse date {s!r}: {e}") from e
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return dt


def _json_intervals_to_time_range(json_intervals: list[dict]) -> TimeRange:
    """Convert the LLM's interval JSON into one ``TimeRange`` (a union).

    A ``null`` endpoint maps to an unbounded (``None``) endpoint; invalid
    (``start >= end``) intervals are dropped.
    """
    intervals: list[TimeInterval] = []
    for entry in json_intervals:
        start_raw = entry.get("start")
        end_raw = entry.get("end")
        try:
            start = None if start_raw is None else _iso_to_datetime(start_raw)
            end = None if end_raw is None else _iso_to_datetime(end_raw)
        except ValueError:
            continue
        if start is not None and end is not None and start >= end:
            continue
        intervals.append(TimeInterval(start=start, end=end))
    return TimeRange(intervals=intervals)


def _json_to_targets(json_targets: list[dict]) -> list[TimeRange]:
    """Convert the LLM's target list into a flat list of ``TimeRange`` targets."""
    out: list[TimeRange] = []
    for entry in json_targets:
        target = _json_intervals_to_time_range(entry.get("intervals", []))
        if target.intervals:
            out.append(target)
    return out


class LanguageModelTemporalQueryPlannerParams(BaseModel):
    """
    Parameters for LanguageModelTemporalQueryPlanner.

    Attributes:
        language_model (LanguageModel):
            Language model used to resolve the query into time range targets.
    """

    language_model: InstanceOf[LanguageModel] = Field(
        ...,
        description="Language model used to resolve the query into time range targets",
    )


class LanguageModelTemporalQueryPlanner(TemporalQueryPlanner):
    """LLM-based temporal query planner."""

    def __init__(self, params: LanguageModelTemporalQueryPlannerParams) -> None:
        """Initialize from parameters."""
        self._language_model = params.language_model

    @override
    async def plan(
        self, query: str, *, ref_time: datetime | None = None
    ) -> TemporalQueryPlan:
        if ref_time is None:
            ref_time = datetime.now(UTC)
        prompt = _PROMPT.format(query=query, ref_time=ref_time.isoformat())
        try:
            parsed = await self._language_model.generate_parsed_response(
                output_format=_PlanResponse,
                user_prompt=prompt,
            )
        except Exception:
            # Produce an empty plan on failure.
            return TemporalQueryPlan()

        if parsed is None:
            return TemporalQueryPlan()

        return TemporalQueryPlan(
            targets=_json_to_targets(
                [target.model_dump() for target in parsed.targets]
            ),
        )
