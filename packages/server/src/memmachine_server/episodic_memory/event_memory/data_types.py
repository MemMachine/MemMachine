"""Data types for EventMemory."""

from datetime import datetime
from typing import Annotated, Literal
from uuid import UUID

from pydantic import BaseModel, Field, JsonValue

from memmachine_server.common.data_types import PropertyValue

# Block: leaf content type


class Text(BaseModel):
    """Plain text."""

    type: Literal["text"] = "text"
    text: str


class Image(BaseModel):
    """An image."""

    type: Literal["image"] = "image"


class Audio(BaseModel):
    """Audio content."""

    type: Literal["audio"] = "audio"


class Video(BaseModel):
    """Video content."""

    type: Literal["video"] = "video"


class FileRef(BaseModel):
    """Reference to a file."""

    type: Literal["file_ref"] = "file_ref"


Block = Annotated[
    Text | Image | Audio | Video | FileRef,
    Field(discriminator="type"),
]


# Context: contextual information about the content


class MessageContext(BaseModel):
    """The content is communicated by a source."""

    type: Literal["message"] = "message"
    source: str


class CitationContext(BaseModel):
    """The content is cited from a source."""

    type: Literal["citation"] = "citation"
    source: str
    source_type: str | None = None
    location: str | None = None


Context = Annotated[
    MessageContext | CitationContext,
    Field(discriminator="type"),
]


# Body: top-level event payload


class Content(BaseModel):
    """A list of item blocks with optional context."""

    type: Literal["content"] = "content"
    context: Context | None = None
    items: list[Block]


class ReadFile(BaseModel):
    """Request the system to read a file."""

    type: Literal["read_file"] = "read_file"
    file: FileRef


Body = Annotated[
    Content | ReadFile,
    Field(discriminator="type"),
]


# Event, Segment, Derivative: core data models for EventMemory


class Event(BaseModel):
    """An event."""

    uuid: UUID
    timestamp: datetime
    body: Body
    properties: dict[str, PropertyValue] = Field(default_factory=dict)
    metadata: dict[str, JsonValue] = Field(default_factory=dict)

    def __hash__(self) -> int:
        """Hash an event by its UUID."""
        return hash(self.uuid)


class Segment(BaseModel):
    """Snapshot of an event, representing a smaller unit of content."""

    uuid: UUID
    event_uuid: UUID
    index: int
    offset: int
    timestamp: datetime
    context: Context | None = None
    block: Block
    properties: dict[str, PropertyValue] = Field(default_factory=dict)

    def __hash__(self) -> int:
        """Hash a segment by its UUID."""
        return hash(self.uuid)


class Derivative(BaseModel):
    """Information derived from a segment."""

    uuid: UUID
    segment_uuid: UUID
    timestamp: datetime
    context: Context | None = None
    text: str
    properties: dict[str, PropertyValue] = Field(default_factory=dict)

    def __hash__(self) -> int:
        """Hash a derivative by its UUID."""
        return hash(self.uuid)


# QueryResult: the result of a memory query, containing relevant segments and their context-ready string representation.


class QueryResult(BaseModel):
    """Memory query result."""

    unified_segment_context: list[Segment]
    unified_segment_context_string: str
