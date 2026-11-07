"""Derivers for text blocks."""

from typing import override

from memmachine_server.common.utils import extract_sentences
from memmachine_server.episodic_memory.event_memory.data_types import (
    Segment,
    TextBlock,
)
from memmachine_server.episodic_memory.event_memory.deriver.deriver import BlockDeriver


class WholeTextDeriver(BlockDeriver[TextBlock]):
    """Derives one text: the segment's whole text."""

    kind = "text"

    @override
    async def derive(self, segment: Segment, block: TextBlock) -> list[str]:
        _ = segment
        return [block.text]


class SentenceTextDeriver(BlockDeriver[TextBlock]):
    """Derives one text per sentence of the segment's text."""

    kind = "text"

    @override
    async def derive(self, segment: Segment, block: TextBlock) -> list[str]:
        _ = segment
        return list(extract_sentences(block.text))
