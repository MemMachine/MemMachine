"""Timeline service: segment-level reads over episodic memory.

The memory API speaks in episodes, which is the right unit for "what is
relevant to this query". Reading a stored conversation is a different
question -- what happened around this point, and in what order -- and
episodes are the wrong grain for it: one episode's content can be thousands
of segments, and the window worth reading is often *inside* one.

So these operations address segments. An address is a segment id, and every
response reports it abbreviated to the shortest prefix that was unambiguous
when it was rendered, because an address a person or a model has to read and
pass back should be short. Abbreviations are resolved on the way in, so a
caller can hand back whatever it was given.
"""

import logging
from collections.abc import Sequence
from uuid import UUID

from memmachine_common.api.spec import (
    ExpandTimelineResponse,
    ExpandTimelineSpec,
    OutlineTimelineResponse,
    OutlineTimelineSpec,
    ResolveTimelineAddressResponse,
    ResolveTimelineAddressSpec,
    SearchTimelineResponse,
    SearchTimelineSpec,
    TimelineEvent,
    TimelineMatch,
    TimelineSegment,
)

from memmachine_server import MemMachine
from memmachine_server.common.filter.filter_parser import FilterExpr, parse_filter
from memmachine_server.episodic_memory.event_memory.data_types import (
    ProducerContext,
    Segment,
)
from memmachine_server.episodic_memory.event_memory.event_memory import EventMemory
from memmachine_server.episodic_memory.long_term_memory.long_term_memory import (
    LongTermMemory,
)
from memmachine_server.server.api_v2.service import _SessionData

logger = logging.getLogger(__name__)

# One more candidate than a caller is shown, so an ambiguity report can say
# whether the list it prints is the whole of it.
_ADDRESS_CANDIDATE_LIMIT = 8


class TimelineAddressError(ValueError):
    """An address named no segment, or more than one."""

    def __init__(self, handle: str, candidates: Sequence[UUID]) -> None:
        """Record the address and whatever it could have named."""
        self.handle = handle
        self.candidates = list(candidates)
        if candidates:
            shown = ", ".join(candidate.hex for candidate in candidates)
            super().__init__(f"Address {handle!r} is ambiguous; it could name: {shown}")
        else:
            super().__init__(f"Address {handle!r} names no stored segment.")


def _parse_optional_filter(filter_string: str) -> FilterExpr | None:
    """Parse a filter string, treating the empty string as no filter."""
    return parse_filter(filter_string) if filter_string else None


def _producer_of(segment: Segment) -> str | None:
    """The segment's speaker, where it has one."""
    return (
        segment.context.producer
        if isinstance(segment.context, ProducerContext)
        else None
    )


async def _resolve_address(
    long_term_memory: LongTermMemory,
    handle: str,
) -> UUID:
    """Resolve an address to the one segment it names, or say why it cannot."""
    candidates = await long_term_memory.resolve_segment_address(
        handle,
        limit=_ADDRESS_CANDIDATE_LIMIT + 1,
    )
    if len(candidates) != 1:
        raise TimelineAddressError(handle, candidates[:_ADDRESS_CANDIDATE_LIMIT])
    return candidates[0]


async def _as_timeline_segments(
    long_term_memory: LongTermMemory,
    segments: Sequence[Segment],
) -> list[TimelineSegment]:
    """Render segments as addressable results, abbreviating in one round trip."""
    handles = await long_term_memory.abbreviate_segment_addresses(
        segment.uuid for segment in segments
    )
    return [
        TimelineSegment(
            handle=handles[segment.uuid],
            segment_uid=segment.uuid.hex,
            event_uid=segment.event_uuid.hex,
            timestamp=segment.timestamp,
            producer=_producer_of(segment),
        )
        for segment in segments
    ]


async def _search_timeline(
    spec: SearchTimelineSpec,
    memmachine: MemMachine,
) -> SearchTimelineResponse:
    session_data = _SessionData(org_id=spec.org_id, project_id=spec.project_id)
    async with memmachine.open_timeline(session_data) as long_term_memory:
        result = await long_term_memory.search_timeline(
            spec.query,
            limit=spec.limit,
            expand_context=spec.expand_context,
            property_filter=_parse_optional_filter(spec.filter),
            score_threshold=spec.score_threshold,
            query_vector=spec.query_vector,
        )
        scored = result.scored_segment_contexts
        seeds = await long_term_memory.get_timeline_segments(
            context.seed_segment_uuid for context in scored
        )
        # A seed whose segment has since been deleted cannot be addressed, so
        # it is dropped rather than reported with an address that resolves to
        # nothing.
        addressable = [
            context for context in scored if context.seed_segment_uuid in seeds
        ]
        rendered_seeds = await _as_timeline_segments(
            long_term_memory,
            [seeds[context.seed_segment_uuid] for context in addressable],
        )
        return SearchTimelineResponse(
            matches=[
                TimelineMatch(
                    score=context.score,
                    seed=seed,
                    rendered=EventMemory.string_from_segment_context(context.segments),
                )
                for context, seed in zip(addressable, rendered_seeds, strict=True)
            ]
        )


async def _expand_timeline(
    spec: ExpandTimelineSpec,
    memmachine: MemMachine,
) -> ExpandTimelineResponse:
    session_data = _SessionData(org_id=spec.org_id, project_id=spec.project_id)
    async with memmachine.open_timeline(session_data) as long_term_memory:
        seed_uuid = await _resolve_address(long_term_memory, spec.handle)
        neighbors = await long_term_memory.expand_timeline(
            seed_uuid,
            before=spec.before,
            after=spec.after,
            unit=spec.unit,
            property_filter=_parse_optional_filter(spec.filter),
        )
        seeds = await long_term_memory.get_timeline_segments([seed_uuid])
        if seed_uuid not in seeds:
            raise TimelineAddressError(spec.handle, [])
        seed_segment = seeds[seed_uuid]
        [seed] = await _as_timeline_segments(long_term_memory, [seed_segment])
        # The seed is rendered with its neighbours rather than alongside them:
        # a window whose centre is missing cannot be read as a timeline, and
        # the reader would have to work out where it anchored.
        window = sorted(
            [*neighbors, seed_segment],
            key=lambda segment: (
                segment.timestamp,
                segment.event_uuid,
                segment.index,
                segment.offset,
            ),
        )
        return ExpandTimelineResponse(
            seed=seed,
            rendered=EventMemory.string_from_segment_context(window),
        )


async def _outline_timeline(
    spec: OutlineTimelineSpec,
    memmachine: MemMachine,
) -> OutlineTimelineResponse:
    session_data = _SessionData(org_id=spec.org_id, project_id=spec.project_id)
    async with memmachine.open_timeline(session_data) as long_term_memory:
        property_filter = _parse_optional_filter(spec.filter)
        if spec.handle is None:
            headers = await long_term_memory.outline_timeline(
                property_filter=property_filter,
                limit=spec.before + spec.after + 1,
            )
        else:
            seed_uuid = await _resolve_address(long_term_memory, spec.handle)
            seeds = await long_term_memory.get_timeline_segments([seed_uuid])
            if seed_uuid not in seeds:
                raise TimelineAddressError(spec.handle, [])
            anchor = (seeds[seed_uuid].timestamp, seeds[seed_uuid].event_uuid)
            # Two bounded walks outward from the anchor rather than one wide
            # window: the anchor's own event is the centre, and each side
            # should get the count it asked for however dense the other is.
            earlier = await long_term_memory.outline_timeline(
                property_filter=property_filter,
                end=anchor,
                limit=spec.before + 1,
                descending=True,
            )
            later = await long_term_memory.outline_timeline(
                property_filter=property_filter,
                start=anchor,
                limit=spec.after + 1,
            )
            seen: set[UUID] = set()
            headers = [
                header
                for header in [*earlier, *later]
                if header.event_uuid not in seen and not seen.add(header.event_uuid)
            ]
            headers.sort(key=lambda header: (header.timestamp, header.event_uuid))

        handles = await long_term_memory.abbreviate_segment_addresses(
            header.first_segment_uuid for header in headers
        )
        return OutlineTimelineResponse(
            events=[
                TimelineEvent(
                    handle=handles[header.first_segment_uuid],
                    event_uid=header.event_uuid.hex,
                    timestamp=header.timestamp,
                    segment_count=header.segment_count,
                    encoded_length=header.encoded_length,
                )
                for header in headers
            ]
        )


async def _resolve_timeline_address(
    spec: ResolveTimelineAddressSpec,
    memmachine: MemMachine,
) -> ResolveTimelineAddressResponse:
    session_data = _SessionData(org_id=spec.org_id, project_id=spec.project_id)
    async with memmachine.open_timeline(session_data) as long_term_memory:
        candidates = await long_term_memory.resolve_segment_address(
            spec.handle,
            limit=_ADDRESS_CANDIDATE_LIMIT + 1,
        )
        return ResolveTimelineAddressResponse(
            segment_uid=candidates[0].hex if len(candidates) == 1 else None,
            candidates=[
                candidate.hex for candidate in candidates[:_ADDRESS_CANDIDATE_LIMIT]
            ]
            if len(candidates) != 1
            else [],
        )
