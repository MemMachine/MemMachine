"""Tests for PassthroughSegmenter."""

from datetime import UTC, datetime
from uuid import uuid4

import pytest

from memmachine_server.episodic_memory.event_memory.data_types import (
    Event,
    NullContext,
    ProducerContext,
    TextBlock,
)
from memmachine_server.episodic_memory.event_memory.segmenter.passthrough_segmenter import (
    PassthroughSegmenter,
)

pytestmark = pytest.mark.asyncio

_TS = datetime(2026, 1, 15, 10, 30, tzinfo=UTC)


async def test_passthrough_emits_one_segment_per_block():
    event = Event(
        uuid=uuid4(),
        timestamp=_TS,
        context=ProducerContext(producer="alice"),
        blocks=[
            TextBlock(text="first block"),
            TextBlock(text="second block"),
            TextBlock(text="third block"),
        ],
        properties={"my_field": "value"},
    )
    segmenter = PassthroughSegmenter()

    segments = await segmenter.segment(event)

    assert len(segments) == 3
    for index, segment in enumerate(segments):
        assert segment.event_uuid == event.uuid
        assert segment.index == index
        assert segment.offset == 0
        assert segment.timestamp == event.timestamp
        assert segment.context == event.context
        assert segment.properties == event.properties
        assert segment.block.text == event.blocks[index].text


async def test_passthrough_does_not_split_long_text():
    """Each block is preserved verbatim, regardless of length."""
    long_text = "lorem ipsum " * 1000
    event = Event(
        uuid=uuid4(),
        timestamp=_TS,
        context=NullContext(),
        blocks=[TextBlock(text=long_text)],
    )
    segmenter = PassthroughSegmenter()

    segments = await segmenter.segment(event)

    assert len(segments) == 1
    assert segments[0].block.text == long_text


async def test_passthrough_empty_blocks_yields_no_segments():
    event = Event(
        uuid=uuid4(),
        timestamp=_TS,
        context=NullContext(),
        blocks=[],
    )
    segmenter = PassthroughSegmenter()

    segments = await segmenter.segment(event)

    assert segments == []


async def test_passthrough_each_segment_has_unique_uuid():
    event = Event(
        uuid=uuid4(),
        timestamp=_TS,
        context=NullContext(),
        blocks=[TextBlock(text="a"), TextBlock(text="b")],
    )
    segmenter = PassthroughSegmenter()

    segments = await segmenter.segment(event)

    assert len({s.uuid for s in segments}) == len(segments)
