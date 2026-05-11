"""Tests for WholeTextDeriver and SentenceTextDeriver."""

from datetime import UTC, datetime
from typing import cast
from uuid import uuid4

import pytest
from pydantic import BaseModel

from memmachine_server.episodic_memory.event_memory.data_types import (
    Block,
    NullContext,
    ProducerContext,
    Segment,
    TextBlock,
)
from memmachine_server.episodic_memory.event_memory.deriver.text_deriver import (
    SentenceTextDeriver,
    WholeTextDeriver,
)

pytestmark = pytest.mark.asyncio

_TS = datetime(2026, 1, 15, 10, 30, tzinfo=UTC)


def _make_segment(
    *,
    block: Block,
    context=None,
    properties=None,
) -> Segment:
    return Segment(
        uuid=uuid4(),
        event_uuid=uuid4(),
        index=0,
        offset=0,
        timestamp=_TS,
        context=context if context is not None else NullContext(),
        block=block,
        properties=properties or {},
    )


def _make_segment_with_unsupported_block() -> Segment:
    """Bypass discriminated-union validation to exercise the deriver's fallback arm."""

    class _OtherBlock(BaseModel):
        block_type: str = "other"

    return Segment.model_construct(
        uuid=uuid4(),
        event_uuid=uuid4(),
        index=0,
        offset=0,
        timestamp=_TS,
        context=NullContext(),
        block=cast(Block, _OtherBlock()),
        properties={},
    )


def _block_text(block: Block) -> str:
    if isinstance(block, TextBlock):
        return block.text
    pytest.fail(f"Unexpected block type: {type(block).__name__}")


class TestWholeTextDeriver:
    async def test_null_context_emits_one_derivative_with_raw_text(self):
        seg = _make_segment(block=TextBlock(text="hello world"))

        result = await WholeTextDeriver().derive(seg)

        assert len(result) == 1
        derivative = result[0]
        assert derivative.block == TextBlock(text="hello world")
        assert derivative.segment_uuid == seg.uuid
        assert derivative.timestamp == seg.timestamp
        assert derivative.context == seg.context

    async def test_producer_context_prefixes_text(self):
        seg = _make_segment(
            block=TextBlock(text="hi there"),
            context=ProducerContext(producer="Alice"),
        )

        result = await WholeTextDeriver().derive(seg)

        assert len(result) == 1
        assert result[0].block == TextBlock(text="Alice: hi there")

    async def test_propagates_segment_properties(self):
        seg = _make_segment(
            block=TextBlock(text="x"),
            properties={"color": "red", "score": 7},
        )

        result = await WholeTextDeriver().derive(seg)

        assert result[0].properties == {"color": "red", "score": 7}

    async def test_non_text_block_raises(self):
        seg = _make_segment_with_unsupported_block()

        with pytest.raises(NotImplementedError, match="Unsupported block type"):
            await WholeTextDeriver().derive(seg)

    async def test_each_derive_call_emits_unique_uuid(self):
        seg = _make_segment(block=TextBlock(text="x"))

        result1 = await WholeTextDeriver().derive(seg)
        result2 = await WholeTextDeriver().derive(seg)

        assert result1[0].uuid != result2[0].uuid


class TestSentenceTextDeriver:
    async def test_single_sentence_emits_one_derivative(self):
        seg = _make_segment(block=TextBlock(text="Hello world."))

        result = await SentenceTextDeriver().derive(seg)

        assert len(result) == 1
        assert result[0].block == TextBlock(text="Hello world.")

    async def test_multi_sentence_emits_one_per_sentence(self):
        seg = _make_segment(
            block=TextBlock(text="First sentence. Second sentence. Third sentence.")
        )

        result = await SentenceTextDeriver().derive(seg)

        # extract_sentences returns a set; assert on the set of derivative texts.
        texts = {_block_text(d.block) for d in result}
        assert texts == {"First sentence.", "Second sentence.", "Third sentence."}

    async def test_producer_context_prefixes_each_sentence(self):
        seg = _make_segment(
            block=TextBlock(text="One. Two."),
            context=ProducerContext(producer="Bob"),
        )

        result = await SentenceTextDeriver().derive(seg)

        texts = {_block_text(d.block) for d in result}
        assert texts == {"Bob: One.", "Bob: Two."}

    async def test_propagates_segment_metadata(self):
        seg = _make_segment(
            block=TextBlock(text="A. B."),
            properties={"k": "v"},
        )

        result = await SentenceTextDeriver().derive(seg)

        for derivative in result:
            assert derivative.segment_uuid == seg.uuid
            assert derivative.timestamp == seg.timestamp
            assert derivative.context == seg.context
            assert derivative.properties == {"k": "v"}

    async def test_non_text_block_raises(self):
        seg = _make_segment_with_unsupported_block()

        with pytest.raises(NotImplementedError, match="Unsupported block type"):
            await SentenceTextDeriver().derive(seg)
