"""Tests for TextSegmenter."""

from datetime import UTC, datetime
from typing import cast
from uuid import uuid4

import pytest
from pydantic import BaseModel

from memmachine_server.episodic_memory.event_memory.data_types import (
    Block,
    Event,
    NullContext,
    ProducerContext,
    TextBlock,
)
from memmachine_server.episodic_memory.event_memory.segmenter.text_segmenter import (
    TextSegmenter,
)

_TS = datetime(2026, 1, 15, 10, 30, tzinfo=UTC)


def _make_event(
    *,
    blocks: list[Block],
    context=None,
    properties=None,
) -> Event:
    return Event(
        uuid=uuid4(),
        timestamp=_TS,
        context=context if context is not None else NullContext(),
        blocks=blocks,
        properties=properties or {},
    )


def _make_event_with_unsupported_block() -> Event:
    """Bypass discriminated-union validation to exercise the segmenter's fallback arm."""

    class _OtherBlock(BaseModel):
        block_type: str = "other"

    return Event.model_construct(
        uuid=uuid4(),
        timestamp=_TS,
        context=NullContext(),
        blocks=[cast(Block, _OtherBlock())],
        properties={},
    )


class TestTextSegmenter:
    def test_short_text_emits_single_segment(self):
        event = _make_event(blocks=[TextBlock(text="hello world")])

        result = TextSegmenter().segment(event)

        assert len(result) == 1
        segment = result[0]
        assert segment.block == TextBlock(text="hello world")
        assert segment.event_uuid == event.uuid
        assert segment.index == 0
        assert segment.offset == 0
        assert segment.timestamp == event.timestamp
        assert segment.context == event.context

    def test_long_text_splits_into_multiple_segments(self):
        long_text = "word " * 1000  # ~5000 chars
        event = _make_event(blocks=[TextBlock(text=long_text.strip())])

        result = TextSegmenter().segment(event)

        assert len(result) > 1
        offsets = sorted(s.offset for s in result)
        assert offsets == list(range(len(result)))
        for segment in result:
            assert segment.event_uuid == event.uuid
            assert segment.index == 0
            assert isinstance(segment.block, TextBlock)
            assert len(segment.block.text) <= 2000

    def test_max_chunk_length_controls_split_size(self):
        event = _make_event(blocks=[TextBlock(text="word " * 200)])

        small_result = TextSegmenter(max_chunk_length=50).segment(event)
        large_result = TextSegmenter(max_chunk_length=5000).segment(event)

        assert len(small_result) > len(large_result)
        assert len(large_result) == 1

    def test_multiple_blocks_get_distinct_indexes(self):
        event = _make_event(blocks=[TextBlock(text="first"), TextBlock(text="second")])

        result = TextSegmenter().segment(event)

        assert len(result) == 2
        result.sort(key=lambda s: s.index)
        assert result[0].index == 0
        assert result[1].index == 1
        assert result[0].offset == 0
        assert result[1].offset == 0
        assert result[0].block == TextBlock(text="first")
        assert result[1].block == TextBlock(text="second")

    def test_propagates_event_context(self):
        event = _make_event(
            blocks=[TextBlock(text="hi")],
            context=ProducerContext(producer="Alice"),
        )

        result = TextSegmenter().segment(event)

        assert result[0].context == ProducerContext(producer="Alice")

    def test_propagates_event_properties(self):
        event = _make_event(
            blocks=[TextBlock(text="hi")],
            properties={"color": "red", "score": 7},
        )

        result = TextSegmenter().segment(event)

        assert result[0].properties == {"color": "red", "score": 7}

    def test_empty_blocks_emits_no_segments(self):
        event = _make_event(blocks=[])

        result = TextSegmenter().segment(event)

        assert result == []

    def test_each_segment_call_emits_unique_uuids(self):
        event = _make_event(blocks=[TextBlock(text="x")])

        result1 = TextSegmenter().segment(event)
        result2 = TextSegmenter().segment(event)

        assert result1[0].uuid != result2[0].uuid

    def test_non_text_block_raises(self):
        event = _make_event_with_unsupported_block()

        with pytest.raises(NotImplementedError, match="Unsupported block type"):
            TextSegmenter().segment(event)
