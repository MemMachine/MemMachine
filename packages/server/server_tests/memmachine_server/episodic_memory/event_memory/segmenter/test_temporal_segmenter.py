"""Tests for TemporalSegmenter (base-segmenter injection + composition)."""

from datetime import UTC, datetime
from typing import override
from uuid import uuid4

import pytest

from memmachine_server.episodic_memory.event_memory.data_types import (
    CompositeContext,
    Event,
    NullContext,
    ProducerContext,
    TextBlock,
    TimeRangesContext,
    find_contexts,
)
from memmachine_server.episodic_memory.event_memory.segmenter.passthrough_segmenter import (
    PassthroughSegmenter,
)
from memmachine_server.episodic_memory.event_memory.segmenter.temporal_segmenter import (
    TemporalSegmenter,
    TemporalSegmenterParams,
)
from memmachine_server.temporal.extractor import TemporalExtractor
from memmachine_server.temporal.time_range import TimeInterval, TimeRange

BASE_TIME = datetime(2024, 6, 1, tzinfo=UTC)
EXTRACTED_RANGES = [
    TimeRange(
        intervals=[
            TimeInterval(
                start=datetime(2024, 1, 1, tzinfo=UTC),
                end=datetime(2024, 1, 2, tzinfo=UTC),
            )
        ]
    )
]


class RecordingExtractor(TemporalExtractor):
    def __init__(self, ranges, calls):
        self._ranges = ranges
        self.calls = calls

    @override
    async def extract(self, text, *, ref_time=None):
        self.calls.append((text, ref_time))
        return list(self._ranges)


def make_temporal_extractor(calls):
    return RecordingExtractor(EXTRACTED_RANGES, calls)


def make_event(text, context=None):
    return Event(
        uuid=uuid4(),
        timestamp=BASE_TIME,
        context=context or NullContext(),
        blocks=[TextBlock(text=text)],
        properties={"doc_id": "d1"},
    )


@pytest.mark.asyncio
async def test_attaches_time_range_context():
    calls = []
    segmenter = TemporalSegmenter(
        TemporalSegmenterParams(temporal_extractor=make_temporal_extractor(calls))
    )
    segments = await segmenter.segment(make_event("Shipped on March 15, 2024."))

    assert len(segments) == 1
    ctx = segments[0].context
    temporal = find_contexts(ctx, TimeRangesContext)
    assert len(temporal) == 1
    assert temporal[0].time_ranges == EXTRACTED_RANGES
    # Event timestamp is the extraction reference time.
    assert calls == [("Shipped on March 15, 2024.", BASE_TIME)]
    assert segments[0].properties == {"doc_id": "d1"}


@pytest.mark.asyncio
async def test_composes_with_existing_producer_context():
    segmenter = TemporalSegmenter(
        TemporalSegmenterParams(temporal_extractor=make_temporal_extractor([]))
    )
    event = make_event("Shipped March 2024.", context=ProducerContext(producer="Alice"))
    segments = await segmenter.segment(event)

    ctx = segments[0].context
    assert isinstance(ctx, CompositeContext)
    producers = find_contexts(ctx, ProducerContext)
    temporal = find_contexts(ctx, TimeRangesContext)
    assert len(producers) == 1
    assert producers[0].producer == "Alice"
    assert len(temporal) == 1
    assert len(temporal[0].time_ranges) == 1


@pytest.mark.asyncio
async def test_uses_injected_base_segmenter():
    # PassthroughSegmenter emits one segment per block (no chunking), so a
    # long text stays a single segment -> one extraction call.
    calls = []
    segmenter = TemporalSegmenter(
        TemporalSegmenterParams(
            temporal_extractor=make_temporal_extractor(calls),
            base_segmenter=PassthroughSegmenter(),
        )
    )
    long_text = "First sentence here. " * 50
    segments = await segmenter.segment(make_event(long_text))

    assert len(segments) == 1
    assert len(calls) == 1
    assert find_contexts(segments[0].context, TimeRangesContext)


@pytest.mark.asyncio
async def test_empty_extraction_yields_empty_ranges():
    segmenter = TemporalSegmenter(
        TemporalSegmenterParams(temporal_extractor=RecordingExtractor([], []))
    )
    segments = await segmenter.segment(make_event("No dates here."))

    ctx = segments[0].context
    temporal = find_contexts(ctx, TimeRangesContext)
    assert len(temporal) == 1
    assert temporal[0].time_ranges == []
