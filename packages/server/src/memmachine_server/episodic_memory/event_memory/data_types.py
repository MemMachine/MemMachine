"""Data types for EventMemory."""

import logging
from abc import ABC, abstractmethod
from collections.abc import Iterable, Iterator, Mapping
from datetime import tzinfo
from typing import (
    Annotated,
    Any,
    ClassVar,
    Literal,
    cast,
    override,
)
from uuid import UUID

from pydantic import (
    AfterValidator,
    AwareDatetime,
    BaseModel,
    ConfigDict,
    Field,
    GetCoreSchemaHandler,
    InstanceOf,
    JsonValue,
    SerializeAsAny,
    StringConstraints,
    TypeAdapter,
    field_serializer,
    field_validator,
)
from pydantic_core import core_schema

from memmachine_server.common.data_types import PropertyValue
from memmachine_server.common.properties_json import (
    decode_properties,
    encode_properties,
)
from memmachine_server.common.vector_store.utils import validate_identifier

logger = logging.getLogger(__name__)


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


# Block: leaf content type.
#
# Different Block kinds do not just represent different modalities;
# they represent different content types,
# each requiring distinct downstream processing logic.
# Plain text, JSON, and HTML may all be processed differently
# despite sharing the text modality.


class Block(BaseModel, ABC):
    """One unit of an event's content.

    `kind` is the discriminator every registered family uses: a subclass
    narrows it to a `Literal` naming its kind. A kind name matches
    `[a-z0-9_]` and is bounded like a property key.
    """

    kind: str

    @abstractmethod
    def render(self, datetime_format: DateTimeFormat) -> str | None:
        """The reader's text for this block; None renders nothing."""
        raise NotImplementedError


class TextBlock(Block):
    """Plain text block."""

    kind: Literal["text"] = "text"
    text: str

    @override
    def render(self, datetime_format: DateTimeFormat) -> str | None:
        return self.text


RegisteredBlock = Annotated[
    TextBlock,
    Field(discriminator="kind"),
]
"""The union of registered block kinds; closed over `TextBlock` until the
kind table lands."""

_BLOCK_ADAPTER = TypeAdapter(RegisteredBlock)


def encode_block(block: Block) -> dict[str, JsonValue]:
    """Encode a block into JSON-compatible data."""
    return _BLOCK_ADAPTER.dump_python(block, mode="json")


def decode_block(encoded: Mapping[str, JsonValue]) -> Block:
    """Decode a block from JSON-compatible data."""
    return _BLOCK_ADAPTER.validate_python(encoded)


def _block_from_field_input(value: object) -> object:
    """Decode encoded block data given for a block field; an instance passes through."""
    return decode_block(value) if isinstance(value, Mapping) else value


# Context: typed, non-filterable data attached to content, keyed by part kind.


class ContextPart(BaseModel, ABC):
    """One registered kind of context, keyed by `kind` in a `Context`.

    A part carries data read by kind, never anything to filter on.
    """

    kind: ClassVar[str]

    @abstractmethod
    def render(self, datetime_format: DateTimeFormat) -> str | None:
        """This part's contribution to a rendered segment; None contributes nothing."""
        raise NotImplementedError


class Author(ContextPart):
    """The readable name of the content's author, as it was at the event."""

    kind: ClassVar[str] = "author"
    name: str

    @override
    def render(self, datetime_format: DateTimeFormat) -> str | None:
        return self.name


class UnknownPart(ContextPart):
    """A part of a kind the running process does not register.

    Produced only by `Context.decode`; keeps the kind name and the data so the
    context round-trips unchanged, renders nothing and is read by no step.
    """

    kind_name: str
    data: dict[str, JsonValue]

    @override
    def render(self, datetime_format: DateTimeFormat) -> str | None:
        return None


_CONTEXT_PART_KINDS: dict[str, type[ContextPart]] = {Author.kind: Author}
"""The registered context part kinds. Built-ins register by import."""

for _kind in _CONTEXT_PART_KINDS:
    if not validate_identifier(_kind):
        raise ValueError(
            f"Context part kind {_kind!r} must match [a-z0-9_] and be at most 32 bytes."
        )


def part_kinds(kinds: Iterable[str]) -> tuple[str, ...]:
    """The kind names to compose, in order, each a valid kind name."""
    kinds = tuple(kinds)
    for kind in kinds:
        if not validate_identifier(kind):
            raise ValueError(
                f"Part kind {kind!r} must match [a-z0-9_] and be at most 32 bytes."
            )
    return kinds


def part_kind(part: ContextPart) -> str:
    """The kind name a part is keyed by."""
    if isinstance(part, UnknownPart):
        return part.kind_name
    return type(part).kind


class Context:
    """The context surrounding an event: at most one part of each kind.

    Built from parts, read by kind. Encodes as `{kind: fields}`; a kind
    this process does not register decodes to an `UnknownPart`, so nothing
    is dropped and the context encodes back unchanged.
    """

    __slots__ = ("_parts",)

    def __init__(self, *parts: ContextPart) -> None:
        """Build a context from parts; two parts of one kind are rejected."""
        by_kind: dict[str, ContextPart] = {}
        for part in parts:
            kind = part_kind(part)
            if kind in by_kind:
                raise ValueError(f"Two parts of kind {kind!r}")
            by_kind[kind] = part
        self._parts = by_kind

    def get(self, kind: str) -> ContextPart | None:
        """The part of this kind, or None."""
        return self._parts.get(kind)

    def with_part(self, part: ContextPart) -> "Context":
        """This context with `part`, replacing any part of its kind."""
        kind = part_kind(part)
        return Context(*(p for k, p in self._parts.items() if k != kind), part)

    def __iter__(self) -> Iterator[ContextPart]:
        """Iterate over the parts."""
        return iter(self._parts.values())

    def __len__(self) -> int:
        """The number of parts."""
        return len(self._parts)

    def __eq__(self, other: object) -> bool:
        """Equal to another context with the same parts."""
        return isinstance(other, Context) and self._parts == other._parts

    def __repr__(self) -> str:
        """The parts, as a constructor call."""
        return f"Context({', '.join(repr(part) for part in self)})"

    def encode(self) -> dict[str, JsonValue]:
        """This context as JSON-compatible data, `{kind: fields}`."""
        return {
            kind: part.data
            if isinstance(part, UnknownPart)
            else part.model_dump(mode="json")
            for kind, part in self._parts.items()
        }

    @classmethod
    def decode(cls, encoded: Mapping[str, JsonValue]) -> "Context":
        """A context from `{kind: fields}`."""
        parts: list[ContextPart] = []
        for kind, fields in encoded.items():
            if not isinstance(fields, Mapping):
                raise TypeError(
                    f"Context part {kind!r} must encode as an object, "
                    f"got {type(fields).__name__}"
                )
            part_type = _CONTEXT_PART_KINDS.get(kind)
            if part_type is None:
                logger.warning(
                    "Context part kind %r is not registered; keeping it as data", kind
                )
                parts.append(
                    UnknownPart(
                        kind_name=kind, data=cast(Mapping[str, JsonValue], fields)
                    )
                )
            else:
                parts.append(part_type.model_validate(fields))
        return cls(*parts)

    @classmethod
    def __get_pydantic_core_schema__(
        cls, source: type[Any], handler: GetCoreSchemaHandler
    ) -> core_schema.CoreSchema:
        """A model field of this type accepts an instance or the encoded form, and serializes to the encoded form."""
        from_encoded = core_schema.no_info_plain_validator_function(cls.decode)
        return core_schema.json_or_python_schema(
            json_schema=from_encoded,
            python_schema=core_schema.union_schema(
                [core_schema.is_instance_schema(cls), from_encoded]
            ),
            serialization=core_schema.plain_serializer_function_ser_schema(
                cls.encode, when_used="json"
            ),
        )


def _deserialize_properties(value: object) -> object:
    if not isinstance(value, Mapping):
        return value
    try:
        return decode_properties(value)
    except (TypeError, ValueError):
        # Not type-tagged data (e.g. plain PropertyValue from code).
        return value


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
        default_factory=Context,
        description="The context surrounding the event",
    )
    blocks: list[SerializeAsAny[Block]] = Field(
        description="The blocks of the event, in order"
    )
    properties: dict[str, PropertyValue] = Field(
        default_factory=dict,
        description="User-defined values the event can be filtered by",
    )

    @field_validator("blocks", mode="before")
    @classmethod
    def _decode_blocks(cls, v: object) -> object:
        if isinstance(v, list):
            return [_block_from_field_input(item) for item in v]
        return v

    @field_validator("properties", mode="before")
    @classmethod
    def _validate_properties(cls, v: object) -> object:
        return _deserialize_properties(v)

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
    context: Context = Field(default_factory=Context, description="The event's context")
    block: SerializeAsAny[Block] = Field(description="The piece of the event's block")
    properties: dict[str, PropertyValue] = Field(
        default_factory=dict, description="The event's properties"
    )

    @field_validator("block", mode="before")
    @classmethod
    def _decode_block(cls, v: object) -> object:
        return _block_from_field_input(v)

    @field_validator("properties", mode="before")
    @classmethod
    def _validate_properties(cls, v: object) -> object:
        return _deserialize_properties(v)

    @field_serializer("properties")
    def _serialize_properties(
        self, v: dict[str, PropertyValue]
    ) -> dict[str, dict[str, bool | int | float | str]]:
        return encode_properties(v)

    def __hash__(self) -> int:
        """Hash a segment by its UUID."""
        return hash(self.uuid)


class Derivative(BaseModel):
    """Text derived from a segment for embedding, with the segment's fields a vector record carries."""

    uuid: UUID = Field(description="The UUID of the derivative")
    segment_uuid: UUID = Field(
        description="The UUID of the segment the text was derived from"
    )
    timestamp: AwareDatetime = Field(description="The segment's timestamp")
    session_id: _BoundedId = Field(description="The segment's session id")
    source_id: _BoundedId | None = Field(
        default=None, description="The segment's source id"
    )
    block_kind: str = Field(description="The kind of the segment's block")
    text: str = Field(description="The text to embed")

    def __hash__(self) -> int:
        """Hash a derivative by its UUID."""
        return hash(self.uuid)


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
