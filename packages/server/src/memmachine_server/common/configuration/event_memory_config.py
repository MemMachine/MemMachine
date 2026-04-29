"""Configuration for event memory."""

from memmachine_common.api.event_memory.config import EventMemoryConf
from pydantic import BaseModel, Field


class EventMemoryStoreConf(BaseModel):
    """Server-level storage configuration for event memory."""

    vector_store: str = Field(
        ...,
        description="Resource ID of the VectorStore instance",
    )
    segment_store: str = Field(
        ...,
        description="Resource ID of the SegmentStore instance",
    )


__all__ = ["EventMemoryConf", "EventMemoryStoreConf"]
