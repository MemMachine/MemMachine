"""Deriver: a segment into derivatives, by the handler for its block's kind."""

from abc import ABC, abstractmethod
from collections.abc import Iterable
from typing import Any, ClassVar
from uuid import uuid4

from memmachine_server.episodic_memory.event_memory.data_types import (
    Block,
    Derivative,
    Segment,
)


class BlockDeriver[B: Block](ABC):
    """Derives texts from the segments whose block is of one kind.

    `kind` names that kind; `derive` receives segments of that kind only
    and returns their content, nothing composed around it.
    """

    kind: ClassVar[str]

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
    yields no derivatives. The table builds
    every derivative's envelope from the segment, so a handler decides the
    texts and nothing else.
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
            list[Derivative]: One derivative per derived text.
        """
        handler = self._handlers.get(segment.block.kind)
        if handler is None:
            return []
        texts = await handler.derive(segment, segment.block)
        return [
            Derivative(
                uuid=uuid4(),
                segment_uuid=segment.uuid,
                timestamp=segment.timestamp,
                session_id=segment.session_id,
                source_id=segment.source_id,
                context=segment.context,
                block_kind=segment.block.kind,
                text=text,
                properties=segment.properties,
            )
            for text in texts
        ]
