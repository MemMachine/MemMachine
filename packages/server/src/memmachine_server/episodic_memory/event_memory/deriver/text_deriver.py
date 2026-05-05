"""Derivers for TextBlock segments."""

from collections.abc import Iterable
from typing import override
from uuid import uuid4

from memmachine_server.common.utils import extract_sentences
from memmachine_server.episodic_memory.event_memory.data_types import (
    Context,
    Derivative,
    NullContext,
    ProducerContext,
    Segment,
    TextBlock,
)
from memmachine_server.episodic_memory.event_memory.deriver.deriver import Deriver


def _format_with_context(context: Context, text: str) -> str:
    """Format text within its context."""
    match context:
        case ProducerContext(producer=producer):
            return f"{producer}: {text}"
        case NullContext():
            return text
        case _:
            raise NotImplementedError(
                f"Unsupported context type: {type(context).__name__}"
            )


def _build_text_derivatives(segment: Segment, texts: Iterable[str]) -> list[Derivative]:
    """Build derivatives from a segment and text strings."""
    return [
        Derivative(
            uuid=uuid4(),
            segment_uuid=segment.uuid,
            timestamp=segment.timestamp,
            context=segment.context,
            block=TextBlock(text=text),
            properties=segment.properties,
        )
        for text in texts
    ]


class WholeTextDeriver(Deriver):
    """Emits one derivative with the segment's whole text formatted in context."""

    @override
    def derive(self, segment: Segment) -> list[Derivative]:
        match segment.block:
            case TextBlock(text=text):
                return _build_text_derivatives(
                    segment, [_format_with_context(segment.context, text)]
                )
            case _:
                raise NotImplementedError(
                    f"Unsupported block type: {type(segment.block).__name__}"
                )


class SentenceTextDeriver(Deriver):
    """Emits one derivative per sentence in the segment's text, formatted in context."""

    @override
    def derive(self, segment: Segment) -> list[Derivative]:
        match segment.block:
            case TextBlock(text=text):
                return _build_text_derivatives(
                    segment,
                    [
                        _format_with_context(segment.context, sentence)
                        for sentence in extract_sentences(text)
                    ],
                )
            case _:
                raise NotImplementedError(
                    f"Unsupported block type: {type(segment.block).__name__}"
                )
