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
    TypeAdapter,
    field_serializer,
    field_validator,
)

from memmachine_server.common.data_types import PropertyValue
from memmachine_server.common.properties_json import (
    decode_properties,
    encode_properties,
)
from memmachine_server.temporal.time_range import TimeRange

# Block: leaf content type.
#
# Different Block types do not just represent different modalities;
# they represent different content types,
# each requiring distinct downstream processing logic.
# Plain text, JSON, and HTML may all be processed differently
# despite sharing the text modality.


class TextBlock(BaseModel):
    """Plain text block."""

    block_type: Literal["text"] = "text"
    text: str


Block = Annotated[
    TextBlock,
    Field(discriminator="block_type"),
]


class NullContext(BaseModel):
    """No context is attached."""

    context_type: Literal["null"] = "null"


class ProducerContext(BaseModel):
    """The content is produced by a producer."""

    context_type: Literal["producer"] = "producer"
    producer: str


class TimeRangesContext(BaseModel):
    """Time ranges associated with the content."""

    context_type: Literal["time_ranges"] = "time_ranges"
    time_ranges: list[TimeRange] = Field(default_factory=list)


class CompositeContext(BaseModel):
    """A sequence of contexts applied to the same content, in order."""

    context_type: Literal["composite"] = "composite"
    contexts: list["Context"] = Field(default_factory=list)


Context = Annotated[
    NullContext | ProducerContext | TimeRangesContext | CompositeContext,
    Field(discriminator="context_type"),
]

# Resolve forward reference to Context.
CompositeContext.model_rebuild()

_CONTEXT_ADAPTER = TypeAdapter(Context | None)
_BLOCK_ADAPTER = TypeAdapter(Block)


def find_contexts[ContextT: Context](
    context: Context | None,
    context_type: type[ContextT],
) -> list[ContextT]:
    """Return every context of the given type, in depth-first order."""
    if isinstance(context, CompositeContext):
        return [
            found
            for member in context.contexts
            for found in find_contexts(member, context_type)
        ]
    return [context] if isinstance(context, context_type) else []


def encode_context(context: Context | None) -> dict[str, JsonValue] | None:
    """Encode a context into JSON-compatible data."""
    return _CONTEXT_ADAPTER.dump_python(context, mode="json")


def decode_context(encoded: Mapping[str, JsonValue] | None) -> Context | None:
    """Decode a context from JSON-compatible data."""
    return _CONTEXT_ADAPTER.validate_python(encoded)


def encode_block(block: Block) -> dict[str, JsonValue]:
    """Encode a block into JSON-compatible data."""
    return _BLOCK_ADAPTER.dump_python(block, mode="json")


def decode_block(encoded: Mapping[str, JsonValue]) -> Block:
    """Decode a block from JSON-compatible data."""
    return _BLOCK_ADAPTER.validate_python(encoded)


# Event, Segment, Derivative: core data models for EventMemory.


class Event(BaseModel):
    """An event."""

    uuid: UUID
    timestamp: datetime
    context: Context = Field(default_factory=NullContext)
    blocks: list[Block]
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
