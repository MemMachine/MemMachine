"""Derivers for TextBlock segments."""

import json
from collections.abc import Iterable
from typing import override
from uuid import uuid4

from memmachine_server.common.utils import extract_sentences
from memmachine_server.episodic_memory.event_memory.data_types import (
    Context,
    DateTimeFormat,
    Derivative,
    NullContext,
    ProducerContext,
    Segment,
    TextBlock,
)
from memmachine_server.episodic_memory.event_memory.deriver.deriver import Deriver
from memmachine_server.episodic_memory.event_memory.formatting import (
    format_timestamp,
)


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


def _format_for_embedding(
    segment: Segment,
    text: str,
    datetime_format: DateTimeFormat,
) -> str:
    """Format a segment's text as an embedding anchor."""
    # Mirror the query result formatters: the message text is JSON-dumped
    # (ensure_ascii=False) so it is a single escaped token, while the
    # producer prefix stays outside the quotes.
    body = _format_with_context(segment.context, json.dumps(text, ensure_ascii=False))

    formatted_timestamp = format_timestamp(segment.timestamp, datetime_format)
    if not formatted_timestamp:
        return body
    return f"[{formatted_timestamp}] {body}"


# A full date and no time, the format of the derived text unless a
# deriver is given another.
_DATE_ONLY = DateTimeFormat(time_style=None)


def _build_text_derivatives(segment: Segment, texts: Iterable[str]) -> list[Derivative]:
    """Build derivatives from a segment and text strings."""
    return [
        Derivative(
            uuid=uuid4(),
            segment_uuid=segment.uuid,
            timestamp=segment.timestamp,
            session_id=segment.session_id,
            source_id=segment.source_id,
            block=TextBlock(text=text),
        )
        for text in texts
    ]


class WholeTextDeriver(Deriver):
    """Emits one derivative with the segment's whole text formatted in context.

    `datetime_format` decides how the timestamp and the author are written
    into the derived text; by default a full date and no time.
    """

    def __init__(self, datetime_format: DateTimeFormat = _DATE_ONLY) -> None:
        """Take the format of the derived text."""
        self._datetime_format = datetime_format

    @override
    async def derive(self, segment: Segment) -> list[Derivative]:
        match segment.block:
            case TextBlock(text=text):
                return _build_text_derivatives(
                    segment,
                    [_format_for_embedding(segment, text, self._datetime_format)],
                )
            case _:
                raise NotImplementedError(
                    f"Unsupported block type: {type(segment.block).__name__}"
                )


class SentenceTextDeriver(Deriver):
    """Emits one derivative per sentence in the segment's text, formatted in context.

    `datetime_format` decides how the timestamp and the author are written
    into the derived text; by default a full date and no time.
    """

    def __init__(self, datetime_format: DateTimeFormat = _DATE_ONLY) -> None:
        """Take the format of the derived text."""
        self._datetime_format = datetime_format

    @override
    async def derive(self, segment: Segment) -> list[Derivative]:
        match segment.block:
            case TextBlock(text=text):
                return _build_text_derivatives(
                    segment,
                    [
                        _format_for_embedding(segment, sentence, self._datetime_format)
                        for sentence in extract_sentences(text)
                    ],
                )
            case _:
                raise NotImplementedError(
                    f"Unsupported block type: {type(segment.block).__name__}"
                )
