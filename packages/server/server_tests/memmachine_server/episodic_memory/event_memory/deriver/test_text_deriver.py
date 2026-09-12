"""Tests for WholeTextDeriver and SentenceTextDeriver."""

from datetime import UTC, datetime
from uuid import uuid4

import pytest

from memmachine_server.episodic_memory.event_memory.data_types import (
    Segment,
    TextBlock,
)
from memmachine_server.episodic_memory.event_memory.deriver import Deriver
from memmachine_server.episodic_memory.event_memory.deriver.text_deriver import (
    SentenceTextDeriver,
    WholeTextDeriver,
)

pytestmark = pytest.mark.asyncio

_TS = datetime(2026, 1, 15, 10, 30, tzinfo=UTC)


def _segment(block: TextBlock, properties=None) -> Segment:
    return Segment(
        session_id="s",
        source_id="src",
        uuid=uuid4(),
        event_uuid=uuid4(),
        index=0,
        offset=0,
        timestamp=_TS,
        block=block,
        properties=properties or {},
    )


def test_declare_the_text_kind():
    assert WholeTextDeriver.kind == TextBlock(text="").kind
    assert SentenceTextDeriver.kind == TextBlock(text="").kind


class TestWholeTextDeriver:
    async def test_derives_the_whole_text_bare(self):
        """The handler returns content only; the memory adds the header."""
        block = TextBlock(text="hello world. And more.")
        seg = _segment(block)
        assert await WholeTextDeriver().derive(seg, block) == ["hello world. And more."]

    async def test_through_the_table(self):
        seg = _segment(TextBlock(text="x"), properties={"color": "red", "score": 7})
        [derivative] = await Deriver([WholeTextDeriver()]).derive(seg)
        assert derivative.text == "x"
        assert derivative.block_kind == "text"
        assert derivative.segment_uuid == seg.uuid
        assert derivative.properties == {"color": "red", "score": 7}


class TestSentenceTextDeriver:
    async def test_single_sentence(self):
        block = TextBlock(text="Hello world.")
        seg = _segment(block)
        assert await SentenceTextDeriver().derive(seg, block) == ["Hello world."]

    async def test_one_text_per_sentence(self):
        block = TextBlock(text="First sentence. Second sentence. Third sentence.")
        seg = _segment(block)
        # extract_sentences returns a set; assert on the set of texts.
        texts = set(await SentenceTextDeriver().derive(seg, block))
        assert texts == {"First sentence.", "Second sentence.", "Third sentence."}

    async def test_through_the_table(self):
        seg = _segment(TextBlock(text="A. B."), properties={"k": "v"})
        derivatives = await Deriver([SentenceTextDeriver()]).derive(seg)
        assert {d.text for d in derivatives} == {"A.", "B."}
        for derivative in derivatives:
            assert derivative.segment_uuid == seg.uuid
            assert derivative.properties == {"k": "v"}
