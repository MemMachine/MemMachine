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


def _segment(text: str) -> Segment:
    return Segment(
        source_id="src",
        uuid=uuid4(),
        event_uuid=uuid4(),
        index=0,
        offset=0,
        timestamp=_TS,
        block=TextBlock(text=text),
    )


def test_declare_the_text_kind():
    assert WholeTextDeriver.kind == TextBlock(text="").kind
    assert SentenceTextDeriver.kind == TextBlock(text="").kind


class TestWholeTextDeriver:
    async def test_derives_the_whole_text_bare(self):
        """The handler returns content only; the memory adds the header."""
        seg = _segment("hello world. And more.")
        assert await WholeTextDeriver().derive(seg, seg.block) == [
            "hello world. And more."
        ]

    async def test_through_the_table(self):
        seg = _segment("x")
        [derivative] = await Deriver([WholeTextDeriver()]).derive(seg)
        assert derivative.text == '[Thursday, January 15, 2026] "x"'
        assert derivative.block_kind == "text"
        assert derivative.segment_uuid == seg.uuid


class TestSentenceTextDeriver:
    async def test_single_sentence(self):
        seg = _segment("Hello world.")
        assert await SentenceTextDeriver().derive(seg, seg.block) == ["Hello world."]

    async def test_one_text_per_sentence(self):
        seg = _segment("First sentence. Second sentence. Third sentence.")
        # extract_sentences returns a set; assert on the set of texts.
        texts = set(await SentenceTextDeriver().derive(seg, seg.block))
        assert texts == {"First sentence.", "Second sentence.", "Third sentence."}

    async def test_through_the_table(self):
        seg = _segment("A. B.")
        derivatives = await Deriver([SentenceTextDeriver()]).derive(seg)
        assert {d.text for d in derivatives} == {
            '[Thursday, January 15, 2026] "A."',
            '[Thursday, January 15, 2026] "B."',
        }
        for derivative in derivatives:
            assert derivative.segment_uuid == seg.uuid
