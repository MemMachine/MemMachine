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
    ConfigDict,
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
"""Bound on a session id and a source id, in bytes, fitting the store's 255-character key columns."""


def _bounded_id(value: str) -> str:
    size = len(value.encode())
    if size > ID_MAX_BYTES:
        raise ValueError(f"is {size} bytes; the maximum is {ID_MAX_BYTES}")
    return value


_BoundedId = Annotated[
    str, StringConstraints(min_length=1), AfterValidator(_bounded_id)
]


class Event(BaseModel):
    """An entry in a session's timeline, with the content to remember about it.

    Immutable once stored: no operation may edit a stored event. A change
    is a forget and a re-encode under the same uuid.
    """

    uuid: UUID = Field(description="The UUID of the event")
    timestamp: AwareDatetime = Field(
        description="When the event happened, timezone-aware"
    )
    session_id: _BoundedId = Field(description="The session the event belongs to")
    source_id: _BoundedId | None = Field(
        default=None,
        description="The source of the event, if any",
    )
    context: Context = Field(
        default_factory=NullContext,
        description="The context surrounding the event",
    )
    blocks: list[Block] = Field(description="The blocks of the event, in order")
    properties: dict[str, PropertyValue] = Field(
        default_factory=dict,
        description="User-defined values the event can be filtered by",
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

    uuid: UUID = Field(description="The UUID of the segment")
    event_uuid: UUID = Field(
        description="The UUID of the event the segment is a piece of"
    )
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
    """Content derived from a segment for embedding, with the segment's fields a vector record carries."""

    uuid: UUID = Field(description="The UUID of the derivative")
    segment_uuid: UUID = Field(
        description="The UUID of the segment the derivative was derived from"
    )
    timestamp: AwareDatetime = Field(description="The segment's timestamp")
    session_id: _BoundedId = Field(description="The segment's session id")
    source_id: _BoundedId | None = Field(
        default=None, description="The segment's source id"
    )
    block: Block = Field(description="The derived content")

    def __hash__(self) -> int:
        """Hash a derivative by its UUID."""
        return hash(self.uuid)


# DateTimeFormat: how a timestamp is written into text.

# The CLDR format length of a date or of a time (`dateFormatLength`,
# `timeFormatLength`). A date and a time are two words in CamelCase, as
# CLDR spells `dateTime`, and one in snake_case, as Python spells
# `datetime`: `DateTimeFormat`, `datetime_format`. Ordered from compact
# to verbose.
DateTimeStyle = Literal["short", "medium", "long", "full"]


class DateTimeFormat(BaseModel):
    """
    How a timestamp is written into text.

    Attributes:
        date_style (DateTimeStyle | None):
            The CLDR style of the date, or None to omit the date
            (default: "full").
        time_style (DateTimeStyle | None):
            The CLDR style of the time, or None to omit the time
            (default: "long").
        locale (str):
            The CLDR locale the date and time are written in
            (default: "en_US").
        timezone (tzinfo | None):
            The zone the timestamp is converted to before it is written,
            or None to write it in the zone it carries (default: None).
    """

    model_config = ConfigDict(frozen=True)

    date_style: DateTimeStyle | None = Field(
        "full", description="The CLDR style of the date, or None to omit the date"
    )
    time_style: DateTimeStyle | None = Field(
        "long", description="The CLDR style of the time, or None to omit the time"
    )
    locale: str = Field(
        "en_US", description="The CLDR locale the date and time are written in"
    )
    timezone: InstanceOf[tzinfo] | None = Field(
        None,
        description=(
            "The zone the timestamp is converted to before it is written, "
            "or None to write it in the zone it carries"
        ),
    )


# Results and options.


class Neighborhood(BaseModel):
    """The segments around a seed, excluding the seed itself: its open neighborhood."""

    before: list[Segment] = Field(
        description="The segments before the seed, in the store's order"
    )
    after: list[Segment] = Field(
        description="The segments after the seed, in the store's order"
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
