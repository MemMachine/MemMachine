"""Data types for EventMemory."""

import logging
from abc import ABC, abstractmethod
from collections.abc import Mapping
from datetime import datetime, tzinfo
from typing import (
    Annotated,
    ClassVar,
    Literal,
    cast,
    override,
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
from memmachine_server.common.vector_store.utils import validate_identifier

logger = logging.getLogger(__name__)


# FormatOptions: options for rendering timestamps and names into text.

# CLDR datetime style. Ordered from compact to verbose.
DateTimeStyle = Literal["short", "medium", "long", "full"]


class FormatOptions(BaseModel):
    """Options for formatting."""

    date_style: DateTimeStyle | None = "full"
    time_style: DateTimeStyle | None = "long"
    locale: str = "en_US"
    timezone: InstanceOf[tzinfo] | None = None


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
    def render(self, options: FormatOptions) -> str | None:
        """The reader's text for this block; None renders nothing."""
        raise NotImplementedError


class TextBlock(Block):
    """Plain text block."""

    kind: Literal["text"] = "text"
    text: str

    @override
    def render(self, options: FormatOptions) -> str | None:
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


# Context: typed, non-filterable data attached to content, keyed by part kind.


class ContextPart(BaseModel, ABC):
    """One registered kind of context, keyed by `kind` in a `Context`.

    A part carries data the processing steps and the renderer read
    (`get_part`), never anything to filter on: identity is `Event.source_id`
    and everything else filterable is a property.
    """

    kind: ClassVar[str]

    @abstractmethod
    def render(self, options: FormatOptions) -> str | None:
        """This part's contribution to a rendered segment; None contributes nothing."""
        raise NotImplementedError


class Author(ContextPart):
    """The readable name of the content's author, as it was at the event."""

    kind: ClassVar[str] = "author"
    name: str

    @override
    def render(self, options: FormatOptions) -> str | None:
        return self.name


class UnknownPart(ContextPart):
    """A part of a kind the running process does not register.

    Produced only by `decode_context`; keeps the kind name and the data so the
    context round-trips unchanged, renders nothing and is read by no step.
    """

    kind_name: str
    data: dict[str, JsonValue]

    @override
    def render(self, options: FormatOptions) -> str | None:
        return None


Context = Mapping[str, ContextPart]
"""A mapping from part kind to the one part of that kind; no context is `{}`."""

_CONTEXT_PART_KINDS: dict[str, type[ContextPart]] = {Author.kind: Author}
"""The registered context part kinds. Built-ins register by import."""

for _kind in _CONTEXT_PART_KINDS:
    if not validate_identifier(_kind):
        raise ValueError(
            f"Context part kind {_kind!r} must match [a-z0-9_] and be at most 32 bytes."
        )


def part_kind(part: ContextPart) -> str:
    """The kind name a part is keyed by."""
    if isinstance(part, UnknownPart):
        return part.kind_name
    return type(part).kind


def get_part[P: ContextPart](context: Context, part: type[P]) -> P | None:
    """The part of a registered kind, or None when the context has none."""
    found = context.get(part.kind)
    if isinstance(found, part):
        return found
    return None


def with_part(context: Context, part: ContextPart) -> Context:
    """A context with `part` set under its kind, replacing any part of that kind."""
    return {**context, part_kind(part): part}


def without_part(context: Context, part: type[ContextPart]) -> Context:
    """A context without the part of a registered kind."""
    return {kind: found for kind, found in context.items() if kind != part.kind}


def encode_context(context: Context) -> dict[str, JsonValue]:
    """Encode a context into JSON-compatible data, as `{kind: fields}`."""
    return {
        kind: part.data
        if isinstance(part, UnknownPart)
        else part.model_dump(mode="json")
        for kind, part in context.items()
    }


def decode_context(encoded: Mapping[str, JsonValue]) -> Context:
    """Decode a context from JSON-compatible data.

    A kind this process does not register decodes to an `UnknownPart`, so
    nothing is dropped and the context encodes back unchanged.
    """
    context: dict[str, ContextPart] = {}
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
            context[kind] = UnknownPart(
                kind_name=kind, data=cast(Mapping[str, JsonValue], fields)
            )
        else:
            context[kind] = part_type.model_validate(fields)
    return context


def _deserialize_context(value: object) -> object:
    # Encoded contexts (`{kind: fields}`) arrive from JSON; a mapping of
    # parts is already decoded and passes through.
    if isinstance(value, Mapping) and not all(
        isinstance(part, ContextPart) for part in value.values()
    ):
        return decode_context(cast(Mapping[str, JsonValue], value))
    return value


def _deserialize_properties(value: object) -> object:
    if not isinstance(value, Mapping):
        return value
    try:
        return decode_properties(value)
    except (TypeError, ValueError):
        # Not type-tagged data (e.g. plain PropertyValue from code).
        return value


# Event, Segment, Derivative: core data models for EventMemory.


class Event(BaseModel):
    """An event."""

    uuid: UUID
    timestamp: datetime
    session_id: str | None = None
    source_id: str | None = None
    context: Context = Field(default_factory=dict)
    blocks: list[RegisteredBlock]
    properties: dict[str, PropertyValue] = Field(default_factory=dict)

    @field_validator("context", mode="before")
    @classmethod
    def _validate_context(cls, v: object) -> object:
        return _deserialize_context(v)

    @field_serializer("context")
    def _serialize_context(self, v: Context) -> dict[str, JsonValue]:
        return encode_context(v)

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
    """Snapshot of an event, representing a smaller unit of content.

    `session_id`, `source_id`, `context`, `timestamp` and `properties` are
    copied verbatim from the event by the segmenter; the segment store
    depends on the copy.
    """

    uuid: UUID
    event_uuid: UUID
    index: int
    offset: int
    timestamp: datetime
    session_id: str | None = None
    source_id: str | None = None
    context: Context = Field(default_factory=dict)
    block: RegisteredBlock
    properties: dict[str, PropertyValue] = Field(default_factory=dict)

    @field_validator("context", mode="before")
    @classmethod
    def _validate_context(cls, v: object) -> object:
        return _deserialize_context(v)

    @field_serializer("context")
    def _serialize_context(self, v: Context) -> dict[str, JsonValue]:
        return encode_context(v)

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
    """Information derived from a segment.

    `session_id`, `source_id`, `context`, `timestamp` and `properties` are
    copied verbatim from the segment by the deriver.
    """

    uuid: UUID
    segment_uuid: UUID
    timestamp: datetime
    session_id: str | None = None
    source_id: str | None = None
    context: Context = Field(default_factory=dict)
    block: RegisteredBlock
    properties: dict[str, PropertyValue] = Field(default_factory=dict)

    @field_validator("context", mode="before")
    @classmethod
    def _validate_context(cls, v: object) -> object:
        return _deserialize_context(v)

    @field_serializer("context")
    def _serialize_context(self, v: Context) -> dict[str, JsonValue]:
        return encode_context(v)

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
        """Hash a derivative by its UUID."""
        return hash(self.uuid)


# Results and options.


class SearchHit(BaseModel):
    """One matched derivative with the context window around its segment."""

    score: float
    """Cosine similarity of the matched derivative."""
    seed: int
    """Index in `segments` of the matched segment."""
    segments: list[Segment]
    """The context window, in the store's order."""


class Neighborhood(BaseModel):
    """The segments around an anchor, never the anchor itself."""

    before: list[Segment]
    """In order, ending just before the anchor."""
    after: list[Segment]
    """In order, starting just after the anchor."""


class FilterOptions(BaseModel):
    """How EventMemory applies the part of a filter the vector store does not evaluate."""

    max_overfetch_factor: int = Field(default=64, ge=1)
    """Cap on widening the vector search, as a multiple of `limit`.

    A predicate on a key the vector store does not declare is applied
    afterward by the segment store, and a seed it drops leaves the search
    short; the search is widened until `limit` hits survive or the fetch
    reaches `limit * max_overfetch_factor`, where it returns what survived.
    A query with no undeclared part never widens.
    """


class EvictionOptions(BaseModel):
    """How EventMemory trims clusters of near-duplicate derivatives."""

    similarity_threshold: float = Field(ge=-1.0, le=1.0)
    """Cosine similarity at or above which two derivatives are one cluster."""
    search_limit: int = Field(gt=0)
    """Stored neighbors consulted per new derivative."""
    target_size: int = Field(gt=0)
    """A cluster larger than this is trimmed to it."""
