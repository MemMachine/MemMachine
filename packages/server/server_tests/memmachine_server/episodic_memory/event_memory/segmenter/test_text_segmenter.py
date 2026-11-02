"""Tests for TextSegmenter."""

import random
import re
from datetime import UTC, datetime
from itertools import accumulate
from typing import cast
from uuid import uuid4

import pytest
from pydantic import BaseModel

from memmachine_server.episodic_memory.event_memory.data_types import (
    Block,
    Event,
    NullContext,
    ProducerContext,
    Segment,
    TextBlock,
)
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
    async def test_short_text_emits_single_segment(self):
        event = _make_event(blocks=[TextBlock(text="hello world")])

        result = await TextSegmenter().segment(event)

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

        result = await TextSegmenter().segment(event)

        assert len(result) > 1
        offsets = sorted(s.offset for s in result)
        assert offsets == list(range(len(result)))
        for segment in result:
            assert segment.event_uuid == event.uuid
            assert segment.index == 0
            assert isinstance(segment.block, TextBlock)
            assert len(segment.block.text) <= 2000

    async def test_max_chunk_length_controls_split_size(self):
        event = _make_event(blocks=[TextBlock(text="word " * 200)])

        small_result = await TextSegmenter(max_chunk_length=50).segment(event)
        large_result = await TextSegmenter(max_chunk_length=5000).segment(event)

        assert len(small_result) > len(large_result)
        assert len(large_result) == 1

    async def test_multiple_blocks_get_distinct_indexes(self):
        event = _make_event(blocks=[TextBlock(text="first"), TextBlock(text="second")])

        result = await TextSegmenter().segment(event)

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
            context=ProducerContext(producer="Alice"),
        )

        result = await TextSegmenter().segment(event)

        assert result[0].context == ProducerContext(producer="Alice")

    async def test_propagates_event_properties(self):
        event = _make_event(
            blocks=[TextBlock(text="hi")],
            properties={"color": "red", "score": 7},
        )

        result = await TextSegmenter().segment(event)

        assert result[0].properties == {"color": "red", "score": 7}

    async def test_empty_blocks_emits_no_segments(self):
        event = _make_event(blocks=[])

        result = await TextSegmenter().segment(event)

        assert result == []

    async def test_each_segment_call_emits_unique_uuids(self):
        event = _make_event(blocks=[TextBlock(text="x")])

        result1 = await TextSegmenter().segment(event)
        result2 = await TextSegmenter().segment(event)

        assert result1[0].uuid != result2[0].uuid

    async def test_non_text_block_raises(self):
        event = _make_event_with_unsupported_block()

        with pytest.raises(NotImplementedError, match="Unsupported block type"):
            await TextSegmenter().segment(event)

    @pytest.mark.parametrize(
        "text",
        [
            "First sentence. Second sentence! Third sentence? Fourth.",
            "  Leading and trailing whitespace.  ",
            "Paragraph one.\n\n\n\nParagraph two, after blank lines.\n",
            "def f():\n    return [\n        1,\n        2,\n    ]\n",
            "第一句话\u3002第二句话\uff0c还有更多的内容\u3002第三句话\uff1f",
            "x" * 75,
            "tab\tseparated\tvalues and\u00a0non-breaking\u3000spaces",
            " \n\t ",
        ],
    )
    async def test_segments_join_to_the_text(self, text: str):
        event = _make_event(blocks=[TextBlock(text=text)])

        result = await TextSegmenter(max_chunk_length=20).segment(event)

        assert "".join(_texts(result)) == text

    async def test_segments_of_random_texts_join_fit_and_follow_the_rule(self):
        rng = random.Random(0)
        words = [
            "alpha",
            "beta.",
            "gamma,",
            "delta!",
            "eps?",
            "zeta;",
            "eta:",
            "\u3002",
            "--",
            "x" * 30,
        ]
        runs = [" ", "  ", "\t", "\n", "\n\n", "\n    ", "  \n  "]
        segmenter = TextSegmenter(max_chunk_length=25)

        for _ in range(200):
            text = rng.choice(["", *runs]) + "".join(
                rng.choice(words) + rng.choice(runs) for _ in range(rng.randint(0, 60))
            )
            result = await segmenter.segment(_make_event(blocks=[TextBlock(text=text)]))

            texts = _texts(result)
            assert "".join(texts) == text
            assert all(len(chunk) <= 25 for chunk in texts)
            assert _boundaries_follow_the_whitespace_rule(text, texts)

    async def test_whitespace_run_is_cut_after_its_last_newline(self):
        event = _make_event(
            blocks=[TextBlock(text="Line one is here.\n    Indented line.")]
        )

        result = await TextSegmenter(max_chunk_length=20).segment(event)

        assert _texts(result) == ["Line one is here.\n", "    Indented line."]

    async def test_content_is_cut_so_whitespace_fits_the_limit(self):
        event = _make_event(blocks=[TextBlock(text="Sixteen letters.\n\n\n\nNext.")])

        result = await TextSegmenter(max_chunk_length=19).segment(event)

        assert _texts(result) == ["Sixteen ", "letters.\n\n\n\n", "Next."]

    async def test_splits_at_the_most_preferred_separator(self):
        event = _make_event(
            blocks=[
                TextBlock(text="One two three four.\n\nFive six seven eight. Nine ten.")
            ]
        )

        result = await TextSegmenter(max_chunk_length=40).segment(event)

        assert _texts(result) == [
            "One two three four.\n\n",
            "Five six seven eight. Nine ten.",
        ]

    async def test_leading_whitespace_begins_the_first_segment(self):
        event = _make_event(blocks=[TextBlock(text="\n\nHello there. General Kenobi.")])

        result = await TextSegmenter(max_chunk_length=15).segment(event)

        assert _texts(result) == ["\n\nHello there. ", "General Kenobi."]

    async def test_trailing_whitespace_ends_the_last_segment(self):
        event = _make_event(blocks=[TextBlock(text="Hello there.  \n\n")])

        result = await TextSegmenter(max_chunk_length=10).segment(event)

        assert _texts(result) == ["Hello ", "there.  \n\n"]

    async def test_whitespace_too_long_for_a_segment_is_cut_to_fit(self):
        text = "x" + "\n" * 30 + "y"
        event = _make_event(blocks=[TextBlock(text=text)])

        result = await TextSegmenter(max_chunk_length=12).segment(event)

        texts = _texts(result)
        assert "".join(texts) == text
        assert all(len(chunk) <= 12 for chunk in texts)

    async def test_whitespace_only_text_emits_one_segment(self):
        event = _make_event(blocks=[TextBlock(text=" \n\t ")])

        result = await TextSegmenter().segment(event)

        assert _texts(result) == [" \n\t "]

    async def test_empty_text_emits_no_segments(self):
        event = _make_event(blocks=[TextBlock(text="")])

        result = await TextSegmenter().segment(event)

        assert result == []

    @pytest.mark.parametrize("max_chunk_length", [0, -1])
    async def test_nonpositive_max_chunk_length_raises(self, max_chunk_length: int):
        with pytest.raises(ValueError, match="max_chunk_length must be positive"):
            TextSegmenter(max_chunk_length=max_chunk_length)


def _texts(segments: list[Segment]) -> list[str]:
    """The segments' texts, in offset order."""
    return [segment.block.text for segment in sorted(segments, key=lambda s: s.offset)]


def _boundaries_follow_the_whitespace_rule(text: str, texts: list[str]) -> bool:
    """Whether each boundary touching whitespace cuts it after its last newline.

    A whitespace run without a newline is cut at its end, and a run at either
    end of the text is not cut.
    """
    boundaries = set(accumulate(len(chunk) for chunk in texts[:-1]))
    for run in re.finditer(r"\s+", text):
        touching = {b for b in boundaries if run.start() <= b <= run.end()}
        if run.start() == 0 or run.end() == len(text):
            allowed = set()
        else:
            newline = run.group().rfind("\n")
            allowed = {run.start() + newline + 1 if newline != -1 else run.end()}
        if not touching <= allowed:
            return False
    return True
