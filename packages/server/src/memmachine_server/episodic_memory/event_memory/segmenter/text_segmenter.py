"""Segmenter for text blocks."""

from typing import override

from langchain_text_splitters import RecursiveCharacterTextSplitter

from memmachine_server.episodic_memory.event_memory.data_types import (
    Event,
    TextBlock,
)
from memmachine_server.episodic_memory.event_memory.segmenter.segmenter import (
    BlockSegmenter,
    Piece,
)


class TextSegmenter(BlockSegmenter[TextBlock]):
    """Splits text blocks by recursive character splitting."""

    kind = "text"

    def __init__(self, max_chunk_length: int = 500) -> None:
        """
        Initialize the segmenter.

        Args:
            max_chunk_length (int):
                Max code-point length for text chunks
                (default: 500).
        """
        self._text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=max_chunk_length,
            chunk_overlap=0,
            separators=[
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
            ],
            keep_separator="end",
        )

    @override
    async def split(self, event: Event, block: TextBlock) -> list[Piece]:
        _ = event
        return [
            Piece(offset=offset, block=TextBlock(text=chunk))
            for offset, chunk in enumerate(self._text_splitter.split_text(block.text))
        ]
