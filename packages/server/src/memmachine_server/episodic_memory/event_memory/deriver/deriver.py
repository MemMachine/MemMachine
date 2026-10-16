"""Deriver: a segment into derivatives, by the handler for its block's kind."""

import json
from abc import ABC, abstractmethod
from collections.abc import Iterable
from typing import Any, ClassVar
from uuid import uuid4

from memmachine_server.episodic_memory.event_memory.data_types import (
    Block,
    DateTimeFormat,
    Derivative,
    Segment,
    part_kinds,
)
from memmachine_server.episodic_memory.event_memory.formatting import format_header

# A full date and no time, the format of the embedded text unless a
# handler is given another.
_DATE_ONLY = DateTimeFormat(time_style=None)


class BlockDeriver[B: Block](ABC):
    """Derives texts from the segments whose block is of one kind.

    `kind` names that kind; `derive` receives segments of that kind only
    and returns their content, nothing composed around it. The handler
    owns how the timestamp and the context parts are composed before each
    text it derives, `datetime_format` and `parts`, so a kind or a
    handler can embed under its own composition.
    """

    kind: ClassVar[str]

    def __init__(
        self,
        datetime_format: DateTimeFormat = _DATE_ONLY,
        parts: Iterable[str] = ("author",),
    ) -> None:
        """Take the composition of the text this handler embeds.

        `datetime_format` is by default a full date and no time; `parts`
        names the context part kinds composed after the timestamp, in
        order.
        """
        self.datetime_format = datetime_format
        self.parts = part_kinds(parts)

    @abstractmethod
    async def derive(self, segment: Segment, block: B) -> list[str]:
        """
        Derive texts from a segment.

        Args:
            segment (Segment): The segment to derive from.
            block (B): The segment's block.

        Returns:
            list[str]: The derived texts, each embedded as one derivative.
        """
        raise NotImplementedError


class Deriver:
    """Derives derivatives from a segment.

    A table from block kind to `BlockDeriver`, a later handler replacing
    an earlier one for its kind; a segment whose kind has no handler
    yields no derivatives. The table builds every derivative's envelope
    from the segment and composes the text it embeds, the header under
    the handler's format options and then the derived text as one
    escaped token, so a handler decides the texts and nothing else.
    """

    def __init__(self, handlers: Iterable[BlockDeriver[Any]] = ()) -> None:
        """Build the table; a later handler replaces an earlier one for its kind."""
        self._handlers: dict[str, BlockDeriver[Any]] = {}
        for handler in handlers:
            self._handlers[handler.kind] = handler

    async def derive(self, segment: Segment) -> list[Derivative]:
        """
        Derive derivatives from a segment.

        Args:
            segment (Segment): The segment to derive from.

        Returns:
            list[Derivative]: One derivative per derived text, its `text`
                the text to embed.
        """
        handler = self._handlers.get(segment.block.kind)
        if handler is None:
            return []
        texts = await handler.derive(segment, segment.block)
        header = format_header(
            segment.timestamp, segment.context, handler.datetime_format, handler.parts
        )
        return [
            Derivative(
                uuid=uuid4(),
                segment_uuid=segment.uuid,
                timestamp=segment.timestamp,
                session_id=segment.session_id,
                source_id=segment.source_id,
                block_kind=segment.block.kind,
                text=header + json.dumps(text, ensure_ascii=False),
            )
            for text in texts
        ]
