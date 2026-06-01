"""Segmenter for temporal retrieval."""

import asyncio
from typing import override

from pydantic import BaseModel, Field, InstanceOf

from memmachine_server.episodic_memory.event_memory.data_types import (
    CompositeContext,
    Event,
    Segment,
    TextBlock,
    TimeRangesContext,
)
from memmachine_server.episodic_memory.event_memory.segmenter.segmenter import (
    Segmenter,
)
from memmachine_server.episodic_memory.event_memory.segmenter.text_segmenter import (
    TextSegmenter,
)
from memmachine_server.temporal.extractor import TemporalExtractor


class TemporalSegmenterParams(BaseModel):
    """Parameters for TemporalSegmenter.

    Attributes:
        temporal_extractor (TemporalExtractor):
            Temporal extractor resolving text and a reference time into time ranges.
        base_segmenter (Segmenter):
            Base segmenter for the initial segmentation.
    """

    temporal_extractor: InstanceOf[TemporalExtractor] = Field(
        ...,
        description="Temporal extractor resolving text and a reference time into time ranges",
    )
    base_segmenter: InstanceOf[Segmenter] = Field(
        default_factory=TextSegmenter,
        description="Base segmenter for the initial segmentation",
    )


class TemporalSegmenter(Segmenter):
    """Wraps a base segmenter and augments each segment with extracted time ranges."""

    def __init__(self, params: TemporalSegmenterParams) -> None:
        """Initialize the segmenter."""
        self._temporal_extractor = params.temporal_extractor
        self._base_segmenter = params.base_segmenter

    @override
    async def segment(self, event: Event) -> list[Segment]:
        base_segments = await self._base_segmenter.segment(event)
        if not base_segments:
            return []

        ref_time = event.timestamp
        texts: list[str] = []
        for segment in base_segments:
            match segment.block:
                case TextBlock(text=text):
                    texts.append(text)
                case _:
                    raise NotImplementedError(
                        f"Unsupported block type: {type(segment.block).__name__}"
                    )

        time_ranges_per_segment = await asyncio.gather(
            *(
                self._temporal_extractor.extract(text, ref_time=ref_time)
                for text in texts
            )
        )

        return [
            base_segment.model_copy(
                update={
                    "context": CompositeContext(
                        contexts=[
                            base_segment.context,
                            TimeRangesContext(time_ranges=list(ranges)),
                        ]
                    )
                }
            )
            for base_segment, ranges in zip(
                base_segments, time_ranges_per_segment, strict=True
            )
        ]
