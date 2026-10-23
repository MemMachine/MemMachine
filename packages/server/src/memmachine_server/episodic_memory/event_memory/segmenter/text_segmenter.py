"""Segmenters for events containing TextBlocks."""

import re
from collections.abc import Sequence
from itertools import pairwise
from typing import override
from uuid import uuid4

from memmachine_server.episodic_memory.event_memory.data_types import (
    Event,
    FormatOptions,
    Segment,
    TextBlock,
)
from memmachine_server.episodic_memory.event_memory.segmenter.segmenter import (
    Segmenter,
)

# Split points in order of preference; the empty string splits between
# characters. A separator stays at the end of the text it follows.
_SEPARATORS = (
    "\n\n",
    "],\n",
    "},\n",
    "),\n",
    "]\n",
    "}\n",
    ")\n",
    ",\n",
    "\uff1f\n",  # Fullwidth question mark
    "?\n",
    "\uff01\n",  # Fullwidth exclamation mark
    "!\n",
    "\u3002\n",  # Ideographic full stop
    ".\n",
    "\uff1f",  # Fullwidth question mark
    "? ",
    "\uff01",  # Fullwidth exclamation mark
    "! ",
    "\u3002",  # Ideographic full stop
    ". ",
    "; ",
    ": ",
    "—",
    "--",
    "\uff0c",  # Fullwidth comma
    "\u3001",  # Ideographic comma
    ", ",
    "\u200b",  # Zero-width space
    " ",
    "",
)


class TextSegmenter(Segmenter):
    """Segments events via recursive character splitting.

    A text block's segments, joined in offset order, are its text exactly,
    and none is longer than `max_chunk_length`. Whitespace between two
    segments ends the earlier one through its last newline, and the rest, a
    line's indentation, begins the later one, unless the whitespace is too
    long for a segment to hold, in which case it is cut to fit.
    """

    def __init__(self, max_chunk_length: int = 500) -> None:
        """
        Initialize the segmenter.

        Args:
            max_chunk_length (int):
                Max code-point length for text chunks
                (default: 500).

        Raises:
            ValueError: If `max_chunk_length` is not positive.
        """
        if max_chunk_length <= 0:
            raise ValueError(
                f"max_chunk_length must be positive, got {max_chunk_length}"
            )
        self._max_chunk_length = max_chunk_length

    @override
    async def segment(
        self,
        event: Event,
        *,
        format_options: FormatOptions | None = None,
    ) -> list[Segment]:
        segments: list[Segment] = []
        for index, block in enumerate(event.blocks):
            match block:
                case TextBlock(text=text):
                    chunks = _split(text, _SEPARATORS, self._max_chunk_length)
                    segments.extend(
                        Segment(
                            uuid=uuid4(),
                            event_uuid=event.uuid,
                            index=index,
                            offset=offset,
                            timestamp=event.timestamp,
                            block=TextBlock(text=chunk),
                            context=event.context,
                            properties=event.properties,
                        )
                        for offset, chunk in enumerate(chunks)
                    )
                case _:
                    raise NotImplementedError(
                        f"Unsupported block type: {type(block).__name__}"
                    )
        return segments


def _split(text: str, separators: Sequence[str], max_length: int) -> list[str]:
    """Split text into chunks of at most `max_length` that join to it.

    Cuts at `_cuts` for the first separator the text contains, then merges
    adjacent pieces while they fit. A piece of `max_length` or more is split
    again at the separators after that one. The last separator must be the
    empty string, which cuts between characters; a piece still over
    `max_length` there is cut every `max_length` code points.
    """
    position, separator = next(
        (position, separator)
        for position, separator in enumerate(separators)
        if not separator or separator in text
    )
    remaining = separators[position + 1 :]

    chunks: list[str] = []
    chunk = ""
    for start, end in pairwise([0, *_cuts(text, separator), len(text)]):
        piece = text[start:end]
        if len(piece) >= max_length:
            if chunk:
                chunks.append(chunk)
                chunk = ""
            if remaining:
                chunks.extend(_split(piece, remaining, max_length))
            else:
                chunks.extend(
                    piece[offset : offset + max_length]
                    for offset in range(0, len(piece), max_length)
                )
        elif len(chunk) + len(piece) > max_length:
            chunks.append(chunk)
            chunk = piece
        else:
            chunk += piece
    if chunk:
        chunks.append(chunk)
    return chunks


def _cuts(text: str, separator: str) -> list[int]:
    """Cut after each occurrence of `separator`, in ascending order.

    A cut that touches whitespace moves to where `_whitespace_cut` cuts that
    whitespace, and is dropped if the whitespace has no cut.
    """
    cuts: list[int] = []
    run_end = -1
    run_cut: int | None = None
    for match in re.finditer(re.escape(separator), text):
        cut: int | None = match.end()
        if (cut > 0 and text[cut - 1].isspace()) or (
            cut < len(text) and text[cut].isspace()
        ):
            if cut > run_end:
                run_end, run_cut = _whitespace_cut(text, cut)
            cut = run_cut
        if cut is not None and 0 < cut < len(text) and (not cuts or cut > cuts[-1]):
            cuts.append(cut)
    return cuts


def _whitespace_cut(text: str, position: int) -> tuple[int, int | None]:
    """Find where the whitespace run at `position` ends and where it is cut.

    The text before the run takes it through its last newline, or all of it
    if it has none. A run at either end of the text belongs to the text it
    touches, so it has no cut (None).
    """
    start = position
    while start > 0 and text[start - 1].isspace():
        start -= 1
    end = position
    while end < len(text) and text[end].isspace():
        end += 1
    if start == 0 or end == len(text):
        return end, None
    return end, text.rfind("\n", start, end) + 1 or end
