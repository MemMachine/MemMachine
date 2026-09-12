"""Tests for TextSegmenter."""

from datetime import UTC, datetime
from uuid import uuid4

import pytest

from memmachine_server.episodic_memory.event_memory.data_types import (
    Author,
    Block,
    Event,
    TextBlock,
)
from memmachine_server.episodic_memory.event_memory.segmenter import Segmenter
from memmachine_server.episodic_memory.event_memory.segmenter.text_segmenter import (
    TextSegmenter,
)

pytestmark = pytest.mark.asyncio

_TS = datetime(2026, 1, 15, 10, 30, tzinfo=UTC)


def _make_event(
    *,
    blocks: list[Block],
    context=None,
    properties=None,
) -> Event:
    return Event(
        session_id="s",
        source_id="src",
        uuid=uuid4(),
        timestamp=_TS,
        context=context if context is not None else {},
        blocks=blocks,
        properties=properties or {},
    )


def _block_text(block: Block) -> str:
    if isinstance(block, TextBlock):
        return block.text
    pytest.fail(f"Unexpected block type: {type(block).__name__}")


async def _segment(event: Event, max_chunk_length: int = 500):
    table = Segmenter([TextSegmenter(max_chunk_length=max_chunk_length)])
    return await table.segment(event)


def test_declares_the_text_kind():
    assert TextSegmenter.kind == TextBlock(text="").kind


class TestTextSegmenter:
    async def test_short_text_emits_single_piece(self):
        block = TextBlock(text="hello world")
        event = _make_event(blocks=[block])
        pieces = await TextSegmenter().split(event, block)
        assert [(p.offset, p.block) for p in pieces] == [
            (0, TextBlock(text="hello world"))
        ]

    async def test_short_text_emits_single_segment(self):
        event = _make_event(blocks=[TextBlock(text="hello world")])
        result = await _segment(event)
        assert len(result) == 1
        segment = result[0]
        assert segment.block == TextBlock(text="hello world")
        assert segment.event_uuid == event.uuid
        assert segment.index == 0
        assert segment.offset == 0
        assert segment.timestamp == event.timestamp
        assert segment.context == event.context

    async def test_long_text_splits_into_multiple_segments(self):
        long_text = "word " * 1000  # ~5000 chars
        event = _make_event(blocks=[TextBlock(text=long_text.strip())])
        result = await _segment(event)
        assert len(result) > 1
        offsets = sorted(s.offset for s in result)
        assert offsets == list(range(len(result)))
        for segment in result:
            assert segment.event_uuid == event.uuid
            assert segment.index == 0
            assert isinstance(segment.block, TextBlock)
            assert len(segment.block.text) <= 2000

    async def test_pieces_join_back_into_the_text(self):
        text = ("The quick brown fox jumps over the lazy dog. " * 40).strip()
        block = TextBlock(text=text)
        event = _make_event(blocks=[block])
        pieces = await TextSegmenter(max_chunk_length=100).split(event, block)
        assert len(pieces) > 1
        assert [p.offset for p in pieces] == list(range(len(pieces)))
        # The splitter strips whitespace at chunk boundaries (its
        # `strip_whitespace` default), so the pieces rebuild the text up to
        # that whitespace, not byte for byte.
        joined = "".join(_block_text(p.block) for p in pieces)
        assert joined.replace(" ", "") == text.replace(" ", "")

    async def test_max_chunk_length_controls_split_size(self):
        event = _make_event(blocks=[TextBlock(text="word " * 200)])
        small_result = await _segment(event, max_chunk_length=50)
        large_result = await _segment(event, max_chunk_length=5000)
        assert len(small_result) > len(large_result)
        assert len(large_result) == 1

    async def test_multiple_blocks_get_distinct_indexes(self):
        event = _make_event(blocks=[TextBlock(text="first"), TextBlock(text="second")])
        result = await _segment(event)
        assert len(result) == 2
        result.sort(key=lambda s: s.index)
        assert result[0].index == 0
        assert result[1].index == 1
        assert result[0].offset == 0
        assert result[1].offset == 0
        assert result[0].block == TextBlock(text="first")
        assert result[1].block == TextBlock(text="second")

    async def test_propagates_event_context(self):
        event = _make_event(
            blocks=[TextBlock(text="hi")],
            context={"author": Author(name="Alice")},
        )
        result = await _segment(event)
        assert result[0].context == {"author": Author(name="Alice")}

    async def test_propagates_event_properties(self):
        event = _make_event(
            blocks=[TextBlock(text="hi")],
            properties={"color": "red", "score": 7},
        )
        result = await _segment(event)
        assert result[0].properties == {"color": "red", "score": 7}

    async def test_empty_blocks_emits_no_segments(self):
        event = _make_event(blocks=[])
        result = await _segment(event)
        assert result == []

    async def test_each_segment_call_emits_unique_uuids(self):
        event = _make_event(blocks=[TextBlock(text="x")])
        result1 = await _segment(event)
        result2 = await _segment(event)
        assert result1[0].uuid != result2[0].uuid
