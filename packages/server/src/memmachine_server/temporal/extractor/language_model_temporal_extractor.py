"""LLM-based temporal extractor."""

from datetime import UTC, datetime, timedelta
from typing import override

from pydantic import BaseModel, Field, InstanceOf

from memmachine_server.common.language_model.language_model import LanguageModel
from memmachine_server.common.utils import ensure_tz_aware
from memmachine_server.temporal.time_range import TimeInterval, TimeRange

from .temporal_extractor import TemporalExtractor


def _parse_iso(s: str) -> datetime:
    """Parse an ISO-8601 string into a UTC-aware datetime."""
    return ensure_tz_aware(datetime.fromisoformat(s)).astimezone(UTC)


def _deictic_context(ref_time: datetime) -> str:
    """
    Render a verbose deictic context relative to the reference time.

    This allows the model to resolve deictic phrases more easily. The deictic
    dates (today, yesterday, this week / month / quarter / year) are computed
    in ``ref_time``'s own timezone, so "yesterday" is the reader's yesterday,
    not UTC's; the rendered reference time carries its real offset. A naive
    ``ref_time`` is treated as UTC.
    """
    ref_time = ensure_tz_aware(ref_time)
    weekday = ref_time.strftime("%A")
    iso_ref = ref_time.isoformat(timespec="seconds")
    yesterday = ref_time - timedelta(days=1)
    tomorrow = ref_time + timedelta(days=1)
    iso_weekday = ref_time.isoweekday()
    this_week_start = (ref_time - timedelta(days=iso_weekday - 1)).date()
    this_week_end = this_week_start + timedelta(days=6)
    last_week_start = this_week_start - timedelta(days=7)
    last_week_end = last_week_start + timedelta(days=6)
    next_week_start = this_week_start + timedelta(days=7)
    next_week_end = next_week_start + timedelta(days=6)

    def month_label(dt: datetime) -> str:
        return dt.strftime("%B %Y")

    this_month = month_label(ref_time)
    if ref_time.month == 1:
        last_month = ref_time.replace(year=ref_time.year - 1, month=12, day=1)
    else:
        last_month = ref_time.replace(month=ref_time.month - 1, day=1)
    if ref_time.month == 12:
        next_month = ref_time.replace(year=ref_time.year + 1, month=1, day=1)
    else:
        next_month = ref_time.replace(month=ref_time.month + 1, day=1)
    this_quarter = (ref_time.month - 1) // 3 + 1
    this_year = ref_time.year
    return (
        f"Reference time: {iso_ref} ({weekday}).\n"
        f"Today = {ref_time.strftime('%A, %B %d, %Y')}. "
        f"Yesterday = {yesterday.strftime('%A, %b %d, %Y')}. "
        f"Tomorrow = {tomorrow.strftime('%A, %b %d, %Y')}.\n"
        f"This week = {this_week_start.strftime('%b %d')}"
        f"-{this_week_end.strftime('%b %d, %Y')} (Mon-Sun).\n"
        f"Last week = {last_week_start.strftime('%b %d')}"
        f"-{last_week_end.strftime('%b %d, %Y')}. "
        f"Next week = {next_week_start.strftime('%b %d')}"
        f"-{next_week_end.strftime('%b %d, %Y')}.\n"
        f"This month = {this_month}. "
        f"Last month = {month_label(last_month)}. "
        f"Next month = {month_label(next_month)}.\n"
        f"This quarter = Q{this_quarter} {this_year}. "
        f"This year = {this_year}. Last year = {this_year - 1}. "
        f"Next year = {this_year + 1}."
    )


_SYSTEM_PROMPT = """You are a temporal-anchor extractor.

Your job: identify EVERY span in a passage that names a specific point,
span, or recurring schedule of time, AND directly resolve each one
into a temporal anchor (a half-open interval on the calendar).

# Critical test before emitting anything

Does the passage USE this phrase to locate a specific occurrence on
the calendar -- one that the reader could later recall, search for,
or reference by date? Or does the phrase describe a constraint, a
requirement, a format placeholder, or a rule that applies generally
across many possible occurrences?

- Specific occurrence ("yesterday we deployed", "shipped on March 15",
  "Q1 was rough", "during the pandemic", "every Thursday at 3pm I
  have therapy") -> EMIT.
- Constraint / rule / placeholder ("Policy: backups within the last
  hour", "every release requires a 30-minute window", "Subject
  format: [Date]") -> SKIP, even if temporal-shaped.

This is the deciding test for borderline cases. The retriever's job
is to surface timeless rule docs on non-temporal queries; anchoring
a policy at the reference time defeats that.

# What counts as a temporal anchor

A span is a temporal anchor if and only if, given the reference
time and any explicit anchoring in the passage, you could state WHEN
it is on a calendar -- AND the critical test above puts it on the
"specific occurrence" side:

- Absolute dates: "March 5, 2026", "1986", "Q3 2025".
- Relative deictics: "yesterday", "2 weeks ago", "next Thursday".
- Approximations: "around 2010", "a few weeks ago", "recently".
- Eras with a calendar anchor: "the 90s", "back in college", "during
  the pandemic".
- Recurring schedules tied to a real standing pattern: "every
  Thursday at 3pm". Emit the first/nearest known occurrence.
- Durations: emit ONLY if attached to a specific calendar anchor.
  In particular, IMPACT-MAGNITUDE durations describe how long an
  effect lasted, not when it was on the calendar -- skip them.
  Examples: "over-reported for 6 weeks", "froze for 12 minutes",
  "delayed 3 hours", "outage lasted 45 minutes". These are
  measurements of an event's effect, not retrievable time anchors;
  even when an anchor exists nearby (e.g. "Postmortem from May 12:
  ... for 6 weeks"), the duration itself isn't a separately
  retrievable anchor -- the date anchor ("May 12") is.

# What does NOT count (skip)

- Bare names of recurring annual events without a year-anchor:
  "summer", "Christmas", "Easter", "graduation day". (EXCEPTION:
  when the phrase IS the recurring schedule itself in a
  standing-arrangement context -- "every summer we visit the lake".)
- Vague descriptors: "recent", "modern", "old", "new", "ancient".
- Bare frequency words: "often", "always", "sometimes", "rarely".
- Bare approximators without concrete reference: "about", "around",
  "roughly" used alone.

# Policy / rule / template contexts -- skip everything inside

When the surrounding sentence describes a generic policy, rule,
convention, requirement, or format, even temporal-shaped phrases
inside it are CONSTRAINTS or PLACEHOLDERS, not events. Cue patterns:

- Explicit policy header: "policy:", "convention:", "rule:",
  "guideline:", "standard:".
- Prescriptive modals as main predicate: "must X", "should X",
  "requires X", "never X without Y", "always X before Y".
- Recurrence over an event-CLASS without naming a specific instance:
  "every release", "every deploy", "every PR", "every sprint".
- Template placeholders: "[Date]", "{date}", "<date>", "YYYY-MM-DD".

If you skip on these cues, do not silently emit other temporal
phrases from the same sentence either; the whole sentence is
policy/rule content.

# How to think about start / end

- A pinpoint anchor (e.g. "March 15, 2024") -> single-day
  interval: start = 2024-03-15T00:00:00Z, end = 2024-03-16T00:00:00Z.
- A span ("Q1 2024") -> start = 2024-01-01T00:00:00Z, end =
  2024-04-01T00:00:00Z.
- A fuzzy anchor ("around 2008") -> widen by one unit: start
  = 2006-01-01T00:00:00Z, end = 2011-01-01T00:00:00Z.
- A relative anchor resolves against ref time. "yesterday" ->
  day before ref. "last month" -> calendar month before ref. "the
  90s" -> [1990-01-01, 2000-01-01).
- A recurring phrase ("every Thursday at 3pm"): emit FIRST known
  occurrence. If the passage anchors the schedule earlier ("every
  Thursday since March"), use that start. Otherwise pick the
  nearest past/upcoming occurrence from ref time.
- A duration only counts if attached to an anchor ("for 3 weeks
  starting June 1") -> [anchor, anchor+duration].

# One anchor (multiple intervals) vs multiple anchors

Most anchors are a single interval. Emit MULTIPLE intervals inside
ONE anchor only when the passage frames them as ONE claim whose
allowed moments form a set with holes:

- A bounded period with an internal gap ("active all year except over
  the break") -> one anchor, two intervals: the span before the
  gap and the span after it.
- A complement ("anytime except that one month") -> one anchor
  covering the region outside the span. Endpoints are required, so
  bound an open side with a far-past / far-future sentinel
  (start 0001-01-01T00:00:00Z, end 9999-12-31T00:00:00Z).
- An intrinsic disjunction stated as one fact ("only ever in the two
  shoulder seasons") -> one anchor, one interval per piece.

Emit SEPARATE anchors (each a single interval) when the passage
reports DISTINCT occurrences -- separate events, even if they share a
pattern ("shipped once early in the year and again late" -> two
anchors).

The litmus: would the passage call these "one thing with gaps" (one
multi-interval anchor) or "several separate things" (several
anchors)?

# Rules

- Use UTC ISO 8601 with "Z" suffix.
- start is inclusive, end is exclusive (half-open).
- For "about" / "around" / "roughly" / "a few" / "a couple", widen by
  one granularity level (day -> week, month -> year, etc.).

# Skip -- do not emit

If you cannot place a surface on the calendar without falling back
to ref time as a fabricated anchor, DO NOT emit it. Omit the
anchor from your output entirely.

This applies to:
- Policy / rule constraints with no specific occurrence.
- Generic recurrences over an event class with no named instance.
- Template placeholders.
- Phrases that look temporal but lack a calendar anchor (e.g.,
  bare "the launch" or "grad school" without other context).

In those cases, do not invent an interval around ref_time -- just
skip the phrase.

# Output

A single JSON object: {"anchors": [...]}. Each anchor carries a list
of one or more half-open intervals:
{
  "intervals": [
    {
      "start": ISO-8601 UTC datetime with "Z",
      "end":   ISO-8601 UTC datetime with "Z"
    }
  ]
}

start is inclusive, end is exclusive. Use multiple intervals in
one anchor ONLY for a single claim with internal structure (gap,
complement, disjunction); otherwise one interval per anchor. If the
passage has no temporal anchors that meet the bar, output
{"anchors": []}.
"""


# Data models for structured outputs specification.


# A half-open calendar interval as ISO-8601 UTC datetimes (with "Z").
# No docstring on these schema models: a model docstring renders into the JSON
# schema's "description" and would be sent to the LLM. The original hand-written
# schema carried none.
class _Interval(BaseModel):
    start: str
    end: str


# One temporal anchor: a set of intervals treated as a single claim.
class _Anchor(BaseModel):
    intervals: list[_Interval] = Field(default_factory=list)


# The extractor's structured output: the resolved temporal anchors.
class _ExtractResponse(BaseModel):
    anchors: list[_Anchor] = Field(default_factory=list)


def _interval_to_time_interval(interval: dict) -> TimeInterval | None:
    """Convert one ``{start, end}`` interval to a ``TimeInterval``.

    Returns ``None`` for malformed or empty (``end <= start``) intervals
    so the caller can drop them.
    """
    try:
        start = _parse_iso(interval["start"])
        end = _parse_iso(interval["end"])
    except (KeyError, ValueError, TypeError):
        return None
    if end <= start:
        return None
    return TimeInterval(start=start, end=end)


def _anchor_to_time_range(anchor: dict) -> TimeRange | None:
    """Convert one anchor (a set of intervals) to a ``TimeRange``.

    Drops malformed or empty intervals; returns ``None`` when none survive
    so the caller can drop the whole anchor.
    """
    intervals = [
        time_interval
        for interval in anchor.get("intervals", [])
        if (time_interval := _interval_to_time_interval(interval)) is not None
    ]
    if not intervals:
        return None
    return TimeRange(intervals=intervals)


class LanguageModelTemporalExtractorParams(BaseModel):
    """
    Parameters for LanguageModelTemporalExtractor.

    Attributes:
        language_model (LanguageModel):
            Language model used to extract time ranges.
    """

    language_model: InstanceOf[LanguageModel] = Field(
        ...,
        description="Language model used to extract time ranges",
    )


class LanguageModelTemporalExtractor(TemporalExtractor):
    """LLM-based temporal extractor."""

    def __init__(self, params: LanguageModelTemporalExtractorParams) -> None:
        """Initialize from parameters."""
        self._language_model = params.language_model

    @override
    async def extract(
        self, text: str, *, ref_time: datetime | None = None
    ) -> list[TimeRange]:
        if ref_time is None:
            ref_time = datetime.now(UTC)

        deictic_context = _deictic_context(ref_time)
        user_prompt = f"{deictic_context}\n\nPassage:\n{text}"
        parsed = await self._language_model.generate_parsed_response(
            output_format=_ExtractResponse,
            system_prompt=_SYSTEM_PROMPT,
            user_prompt=user_prompt,
        )
        if parsed is None:
            return []
        return [
            time_range
            for anchor in parsed.anchors
            if (time_range := _anchor_to_time_range(anchor.model_dump())) is not None
        ]
