"""Passthrough segmenter that emits one segment per block, unchanged."""

from typing import override
from uuid import uuid4

from memmachine_server.episodic_memory.event_memory.data_types import (
    Event,
    Segment,
)
from memmachine_server.episodic_memory.event_memory.segmenter.segmenter import (
    Segmenter,
)


class PassthroughSegmenter(Segmenter):
    """Emit one segment per block; do not split text."""

    @override
    async def segment(self, event: Event) -> list[Segment]:
        return [
            Segment(
                uuid=uuid4(),
                event_uuid=event.uuid,
                index=index,
                offset=0,
                timestamp=event.timestamp,
                block=block,
                context=event.context,
                properties=event.properties,
            )
            for index, block in enumerate(event.blocks)
        ]
