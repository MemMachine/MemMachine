"""Data models for ExtraMemory."""

from datetime import datetime
from typing import Annotated, Literal
from uuid import UUID

from pydantic import BaseModel, Field, JsonValue

from memmachine_server.common.data_types import PropertyValue


class MessageContent(BaseModel):
    """A text message from a source."""

    type: Literal["message"] = "message"
    source: str
    text: str


class TextContent(BaseModel):
    """A plain text block."""

    type: Literal["text"] = "text"
    text: str


class ImageContent(BaseModel):
    """An image."""

    type: Literal["image"] = "image"
    url: str | None = None
    data: str | None = None  # base64
    mime_type: str | None = None


Content = Annotated[
    MessageContent | TextContent | ImageContent,
    Field(discriminator="type"),
]


class Episode(BaseModel):
    """Experience of an event."""

    uuid: UUID
    timestamp: datetime
    content: list[Content]
    properties: dict[str, PropertyValue] = Field(default_factory=dict)
    metadata: dict[str, JsonValue] = Field(default_factory=dict)

    def __hash__(self) -> int:
        """Hash an episode by its UUID."""
        return hash(self.uuid)


class Segment(BaseModel):
    """Snapshot of an episode, representing a smaller unit of experience."""

    uuid: UUID
    episode_uuid: UUID
    block: int
    index: int
    timestamp: datetime
    content: MessageContent | TextContent | ImageContent
    properties: dict[str, PropertyValue] = Field(default_factory=dict)

    def __hash__(self) -> int:
        """Hash a segment by its UUID."""
        return hash(self.uuid)


class Derivative(BaseModel):
    """Information derived from a segment."""

    uuid: UUID
    content: str

    def __hash__(self) -> int:
        """Hash a derivative by its UUID."""
        return hash(self.uuid)


class QueryResult(BaseModel):
    """Memory query result."""

    unified_segment_context: list[Segment]
    unified_segment_context_string: str
