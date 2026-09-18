"""Segmenter: an event's blocks into segments, each block by its kind's handler."""

from abc import ABC, abstractmethod
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any, ClassVar
from uuid import uuid4

from memmachine_server.episodic_memory.event_memory.data_types import (
    Block,
    Event,
    Segment,
)


@dataclass(frozen=True, slots=True)
class Piece:
    """A piece of one block.

    `offset` is the piece's position among its block's pieces, which
    reconstruct the block in offset order.
    """

    offset: int
    block: Block


class BlockSegmenter[B: Block](ABC):
    """Splits the blocks of one kind into pieces.

    `kind` names that kind; `split` receives blocks of that kind only.
    """

    kind: ClassVar[str]

    @abstractmethod
    async def split(self, event: Event, block: B) -> list[Piece]:
        """
        Split one of an event's blocks into pieces.

        Args:
            event (Event): The event the block belongs to.
            block (B): The block to split.

        Returns:
            list[Piece]: The block's pieces, in offset order.
        """
        raise NotImplementedError


class Segmenter:
    """Splits an event's blocks into segments.

    A table from block kind to `BlockSegmenter`, a later handler
    replacing an earlier one for its kind; a kind with no handler passes
    through as one segment, unchanged. The
    table builds every segment's envelope from the event, so a handler
    decides the pieces and nothing else.
    """

    def __init__(self, handlers: Iterable[BlockSegmenter[Any]] = ()) -> None:
        """Build the table; a later handler replaces an earlier one for its kind."""
        self._handlers: dict[str, BlockSegmenter[Any]] = {}
        for handler in handlers:
            self._handlers[handler.kind] = handler

    async def segment(self, event: Event) -> list[Segment]:
        """
        Segment an event.

        Args:
            event (Event): The event to segment.

        Returns:
            list[Segment]: The segments of every block, in block and offset order.
        """
        segments: list[Segment] = []
        for index, block in enumerate(event.blocks):
            handler = self._handlers.get(block.kind)
            if handler is None:
                pieces = [Piece(offset=0, block=block)]
            else:
                pieces = await handler.split(event, block)
            segments.extend(
                Segment(
                    uuid=uuid4(),
                    event_uuid=event.uuid,
                    index=index,
                    offset=piece.offset,
                    timestamp=event.timestamp,
                    session_id=event.session_id,
                    source_id=event.source_id,
                    context=event.context,
                    block=piece.block,
                    properties=event.properties,
                )
                for piece in pieces
            )
        return segments
