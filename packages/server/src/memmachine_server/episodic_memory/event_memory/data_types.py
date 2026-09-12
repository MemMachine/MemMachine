"""Data types for EventMemory."""

from collections.abc import Mapping
from datetime import tzinfo
from typing import (
    Annotated,
    Literal,
)
from uuid import UUID

from pydantic import (
    AfterValidator,
    AwareDatetime,
    BaseModel,
    Field,
    InstanceOf,
    JsonValue,
    StringConstraints,
    TypeAdapter,
    field_serializer,
    field_validator,
)

from memmachine_server.common.data_types import PropertyValue
from memmachine_server.common.properties_json import (
    decode_properties,
    encode_properties,
)

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


class ProducerContext(BaseModel):
    """The content is produced by a producer."""

    context_type: Literal["producer"] = "producer"
    producer: str


class NullContext(BaseModel):
    """No context is attached."""

    context_type: Literal["null"] = "null"


ContextUnion = ProducerContext | NullContext

Context = Annotated[
    ContextUnion,
    Field(discriminator="context_type"),
]

_CONTEXT_ADAPTER = TypeAdapter(Context | None)
_BLOCK_ADAPTER = TypeAdapter(Block)


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

ID_MAX_BYTES = 255
"""Bound on a session id and a source id, in bytes: the width of the store's key columns."""


def _bounded_id(value: str) -> str:
    size = len(value.encode())
    if size > ID_MAX_BYTES:
        raise ValueError(f"is {size} bytes; the maximum is {ID_MAX_BYTES}")
    return value


_BoundedId = Annotated[
    str, StringConstraints(min_length=1), AfterValidator(_bounded_id)
]


class Event(BaseModel):
    """Something that happened at a point in time, and the content it produced.

    Immutable once stored: no operation may edit a stored event. A change
    is a forget and a re-encode under the same uuid.
    """

    uuid: UUID = Field(description="Identity of the event")
    timestamp: AwareDatetime = Field(
        description="When the event happened, with its zone; a naive value is rejected"
    )
    session_id: _BoundedId = Field(
        description="The conversation or stream the event belongs to"
    )
    source_id: _BoundedId | None = Field(
        default=None,
        description="The entity responsible for the content; None for none",
    )
    context: Context = Field(
        default_factory=NullContext,
        description="The circumstances the content was produced in",
    )
    blocks: list[Block] = Field(description="The content, in order")
    properties: dict[str, PropertyValue] = Field(
        default_factory=dict,
        description="Caller-defined values the event can be filtered by",
    )

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
    """A piece of one of an event's blocks, carrying the event's fields."""

    uuid: UUID = Field(description="Identity of the segment")
    event_uuid: UUID = Field(description="The event the segment is a piece of")
    index: int = Field(
        ge=0, description="Position of the block among the event's blocks"
    )
    offset: int = Field(
        ge=0, description="Position of the piece among the block's pieces"
    )
    timestamp: AwareDatetime = Field(description="The event's timestamp")
    session_id: _BoundedId = Field(description="The event's session id")
    source_id: _BoundedId | None = Field(
        default=None, description="The event's source id"
    )
    context: Context = Field(
        default_factory=NullContext, description="The event's context"
    )
    block: Block = Field(description="The piece of the event's block")
    properties: dict[str, PropertyValue] = Field(
        default_factory=dict, description="The event's properties"
    )

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
    """Content derived from a segment to be embedded in its place, carrying the segment's fields."""

    uuid: UUID = Field(description="Identity of the derivative")
    segment_uuid: UUID = Field(description="The segment the content was derived from")
    timestamp: AwareDatetime = Field(description="The segment's timestamp")
    session_id: _BoundedId = Field(description="The segment's session id")
    source_id: _BoundedId | None = Field(
        default=None, description="The segment's source id"
    )
    context: Context = Field(
        default_factory=NullContext, description="The segment's context"
    )
    block: Block = Field(description="The derived content")
    properties: dict[str, PropertyValue] = Field(
        default_factory=dict, description="The segment's properties"
    )

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


# Results and options.


class Neighborhood(BaseModel):
    """The segments around an anchor, never the anchor itself: its open neighborhood."""

    before: list[Segment] = Field(
        description="In the store's order, ending just before the anchor"
    )
    after: list[Segment] = Field(
        description="In the store's order, starting just after the anchor"
    )


class QueryHit(BaseModel):
    """A segment a query matched, scored, with the neighborhood around it."""

    score: float = Field(
        description="Relevance of the seed segment to the query; higher is better"
    )
    seed: Segment = Field(description="The segment the query matched")
    neighborhood: Neighborhood = Field(
        description="The segments around the seed, in the store's order"
    )

    def window(self) -> list[Segment]:
        """The seed and its neighbors, in the store's order."""
        return [*self.neighborhood.before, self.seed, *self.neighborhood.after]


class EvictionOptions(BaseModel):
    """Eviction, at ingest, of stored derivatives that a new derivative nearly duplicates."""

    cosine_similarity_threshold: float = Field(
        ge=-1.0,
        le=1.0,
        description=(
            "Cosine similarity between a new derivative and a stored one "
            "at or above which eviction is considered"
        ),
    )
    search_limit: int = Field(
        gt=0,
        description=(
            "Maximum number of stored derivatives at or above the threshold "
            "fetched per new derivative; only those can be evicted"
        ),
    )
    target_size: int = Field(
        gt=0,
        description=(
            "How many derivatives to keep out of a new derivative and the "
            "stored ones at or above the threshold with it, when there are more"
        ),
    )
