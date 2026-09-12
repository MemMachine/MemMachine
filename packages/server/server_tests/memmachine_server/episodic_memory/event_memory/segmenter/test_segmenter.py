"""Tests for the Segmenter table: dispatch by kind, override order, envelopes."""

from datetime import UTC, datetime
from typing import override
from uuid import uuid4

import pytest

from memmachine_server.episodic_memory.event_memory.data_types import (
    Author,
    Block,
    Event,
    TextBlock,
)
from memmachine_server.episodic_memory.event_memory.segmenter import (
    BlockSegmenter,
    Piece,
    Segmenter,
)

pytestmark = pytest.mark.asyncio

_TS = datetime(2026, 1, 15, 10, 30, tzinfo=UTC)


def _text(block: Block) -> str:
    assert isinstance(block, TextBlock)
    return block.text


def _event(*texts: str, context=None, properties=None) -> Event:
    return Event(
        uuid=uuid4(),
        timestamp=_TS,
        session_id="s1",
        source_id="chat",
        context=context if context is not None else {},
        blocks=[TextBlock(text=text) for text in texts],
        properties=properties or {},
    )


class _Upper(BlockSegmenter[TextBlock]):
    kind = "text"

    @override
    async def split(self, event, block):
        return [Piece(offset=0, block=TextBlock(text=block.text.upper()))]


class _Halves(BlockSegmenter[TextBlock]):
    kind = "text"

    @override
    async def split(self, event, block):
        middle = len(block.text) // 2
        return [
            Piece(offset=0, block=TextBlock(text=block.text[:middle])),
            Piece(offset=1, block=TextBlock(text=block.text[middle:])),
        ]


class _Recording(BlockSegmenter[TextBlock]):
    kind = "text"

    def __init__(self) -> None:
        self.calls = []

    @override
    async def split(self, event, block):
        self.calls.append((event, block))
        return [Piece(offset=0, block=block)]


async def test_no_handler_passes_each_block_through_unchanged():
    event = _event(
        "first block",
        "second block",
        "third block",
        context={"author": Author(name="alice")},
        properties={"my_field": "value"},
    )
    segments = await Segmenter().segment(event)
    assert [s.block for s in segments] == list(event.blocks)
    for index, segment in enumerate(segments):
        assert segment.event_uuid == event.uuid
        assert segment.index == index
        assert segment.offset == 0
        assert segment.timestamp == event.timestamp
        assert segment.session_id == "s1"
        assert segment.source_id == "chat"
        assert segment.context == event.context
        assert segment.properties == event.properties
    assert len({s.uuid for s in segments}) == 3


async def test_passthrough_does_not_split_long_text():
    long_text = "lorem ipsum " * 1000
    [segment] = await Segmenter().segment(_event(long_text))
    assert segment.block == TextBlock(text=long_text)


async def test_empty_blocks_yield_no_segments():
    assert await Segmenter().segment(_event()) == []


async def test_handler_of_the_kind_splits_its_blocks():
    segments = await Segmenter([_Halves()]).segment(_event("abcd", "wxyz"))
    assert [(s.index, s.offset, _text(s.block)) for s in segments] == [
        (0, 0, "ab"),
        (0, 1, "cd"),
        (1, 0, "wx"),
        (1, 1, "yz"),
    ]


async def test_later_handler_replaces_earlier_for_its_kind():
    segments = await Segmenter([_Upper(), _Halves()]).segment(_event("abcd"))
    assert [_text(s.block) for s in segments] == ["ab", "cd"]


async def test_handler_receives_the_event_and_its_block():
    recording = _Recording()
    event = _event("a", "b")
    await Segmenter([recording]).segment(event)
    assert recording.calls == [
        (event, event.blocks[0]),
        (event, event.blocks[1]),
    ]
