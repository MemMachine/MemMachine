"""Data types for EventMemory."""

from collections.abc import Mapping
from datetime import datetime, tzinfo
from typing import (
    Annotated,
    Literal,
)
from uuid import UUID

from pydantic import (
    BaseModel,
    Field,
    InstanceOf,
    JsonValue,
    field_serializer,
    field_validator,
)

from memmachine_server.common.data_types import PropertyValue
from memmachine_server.common.properties_json import (
    decode_properties,
    encode_properties,
)

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


class NullContext(BaseModel):
    """No context is attached."""

    type: Literal["null"] = "null"


ContextUnion = MessageContext | CitationContext | NullContext

Context = Annotated[
    ContextUnion,
    Field(discriminator="type"),
]


# Body: top-level event payload


class Content(BaseModel):
    """A list of item blocks with context."""

    type: Literal["content"] = "content"
    context: Context = Field(default_factory=NullContext)
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

    @field_validator("properties", mode="before")
    @classmethod
    def _deserialize_properties(cls, v: object) -> object:
        if not isinstance(v, Mapping):
            return v
        try:
            return decode_properties(v)
        except (TypeError, ValueError):
            return v

    @field_serializer("properties")
    def _serialize_properties(
        self, v: dict[str, PropertyValue]
    ) -> dict[str, dict[str, bool | int | float | str]]:
        return encode_properties(v)

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
    context: Context = Field(default_factory=NullContext)
    block: Block
    properties: dict[str, PropertyValue] = Field(default_factory=dict)

    @field_validator("properties", mode="before")
    @classmethod
    def _deserialize_properties(cls, v: object) -> object:
        if not isinstance(v, Mapping):
            return v
        try:
            return decode_properties(v)
        except (TypeError, ValueError):
            # Not type-tagged data (e.g. plain PropertyValue from code).
            return v

    @field_serializer("properties")
    def _serialize_properties(
        self, v: dict[str, PropertyValue]
    ) -> dict[str, dict[str, bool | int | float | str]]:
        return encode_properties(v)

    def __hash__(self) -> int:
        """Hash a segment by its UUID."""
        return hash(self.uuid)


class Derivative(BaseModel):
    """Information derived from a segment."""

    uuid: UUID
    segment_uuid: UUID
    timestamp: datetime
    context: Context = Field(default_factory=NullContext)
    text: str
    properties: dict[str, PropertyValue] = Field(default_factory=dict)

    @field_validator("properties", mode="before")
    @classmethod
    def _deserialize_properties(cls, v: object) -> object:
        if not isinstance(v, Mapping):
            return v
        try:
            return decode_properties(v)
        except (TypeError, ValueError):
            return v

    @field_serializer("properties")
    def _serialize_properties(
        self, v: dict[str, PropertyValue]
    ) -> dict[str, dict[str, bool | int | float | str]]:
        return encode_properties(v)

    def __hash__(self) -> int:
        """Hash a derivative by its UUID."""
        return hash(self.uuid)


# FormatOptions: options for formatting query result.

# CLDR datetime style. Ordered from compact to verbose.
DateTimeStyle = Literal["short", "medium", "long", "full"]


class FormatOptions(BaseModel):
    """Options for formatting."""

    date_style: DateTimeStyle | None = "full"
    time_style: DateTimeStyle | None = "long"
    locale: str = "en_US"
    timezone: InstanceOf[tzinfo] | None = None


# QueryResult: the result of a memory query.


class ScoredSegmentContext(BaseModel):
    """A segment context anchored on a seed segment, with a score."""

    score: float
    seed_segment_uuid: UUID
    segments: list[Segment]


class QueryResult(BaseModel):
    """Memory query result, ordered by reranker score."""

    scored_segment_contexts: list[ScoredSegmentContext]
