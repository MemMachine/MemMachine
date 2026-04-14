"""Data types for EventMemory."""

import types
from collections.abc import Iterable, Mapping
from datetime import datetime, tzinfo
from typing import (
    Annotated,
    ClassVar,
    Final,
    Literal,
    Union,
    cast,
    get_args,
    get_origin,
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

from memmachine_server.common.data_types import (
    PROPERTY_TYPE_TO_PROPERTY_TYPE_NAME,
    PropertyValue,
)
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


# Context: contextual information about the content
#
# Every field must be resolvable to a PropertyValue type.
# Each Context type may declare a ClassVar `indexed_properties`,
# a frozenset of field names, as an indexing hint.


def _resolve_annotation_property_type(annotation: object) -> type[PropertyValue]:
    """
    Resolve a Pydantic field annotation to a PropertyValue type.

    Handles optional fields and single-typed literals.

    Raises TypeError if the annotation fails to resolve to a valid type.
    """
    origin = get_origin(annotation)

    # Unwrap optional.
    if origin is Union or origin is types.UnionType:
        non_none = [arg for arg in get_args(annotation) if arg is not type(None)]
        if len(non_none) == 1:
            annotation = non_none[0]
            origin = get_origin(annotation)

    # Unwrap literal.
    if origin is Literal:
        literal_args = get_args(annotation)
        if literal_args and all(
            type(arg) is type(literal_args[0]) for arg in literal_args
        ):
            annotation = type(literal_args[0])

    # Plain type.
    if (
        isinstance(annotation, type)
        and annotation in PROPERTY_TYPE_TO_PROPERTY_TYPE_NAME
    ):
        return cast(type[PropertyValue], annotation)

    raise TypeError(
        f"annotation {annotation!r} does not resolve to a "
        f"PropertyValue type (expected one of "
        f"{', '.join(PROPERTY_TYPE_TO_PROPERTY_TYPE_NAME.values())})"
    )


def _resolve_pydantic_field_property_types(
    cls: type[BaseModel],
) -> dict[str, type[PropertyValue]]:
    """
    Resolve every Pydantic field on `cls` to a PropertyValue type.

    Raises TypeError if any annotation fails to resolve to a valid type.
    """
    resolved: dict[str, type[PropertyValue]] = {}
    for name, field_info in cls.model_fields.items():
        try:
            resolved[name] = _resolve_annotation_property_type(field_info.annotation)
        except TypeError as e:
            raise TypeError(f"{cls.__name__}.{name}: {e}") from e
    return resolved


def _resolve_indexed_property_names(
    cls: type[BaseModel],
) -> frozenset[str]:
    """
    Resolve the set of property names hinted for indexing.

    Returns an empty frozenset if the class does not declare one.
    Raises TypeError if the declaration exists but is malformed.
    """
    raw = getattr(cls, "indexed_properties", None)
    if raw is None:
        return frozenset()

    if not isinstance(raw, frozenset):
        raise TypeError(
            f"{cls.__name__}.indexed_properties must be a frozenset "
            f"(got {type(raw).__name__})"
        )

    extra = raw - set(cls.model_fields)
    if extra:
        raise TypeError(
            f"{cls.__name__}.indexed_properties declares fields "
            f"{sorted(extra)} not present in Pydantic model_fields"
        )

    return raw


def resolve_indexed_context_properties_schema(
    context_types: Iterable[type[BaseModel]],
) -> dict[str, type[PropertyValue]]:
    """
    Resolve the schema of properties hinted for indexing.

    Returns a dict mapping the names of a subset of Context fields
    to their PropertyValue types, as an indexing hint.

    Raises TypeError if any Context type is invalid,
    or if Context types have mismatched annotations on a shared indexed field.
    """
    all_field_property_types: dict[str, dict[type[BaseModel], type[PropertyValue]]] = {}
    merged: dict[str, type[PropertyValue]] = {}

    for cls in context_types:
        field_property_types = _resolve_pydantic_field_property_types(cls)
        indexed_property_names = _resolve_indexed_property_names(cls)

        for name, field_type in field_property_types.items():
            all_field_property_types.setdefault(name, {})[cls] = field_type

        for name in indexed_property_names:
            indexed_type = field_property_types[name]
            existing = merged.get(name)
            if existing is not None and existing is not indexed_type:
                raise TypeError(
                    f"Context field {name!r} is indexed with conflicting "
                    f"PropertyValue types across Context types: "
                    f"{cls.__name__} resolves to "
                    f"{PROPERTY_TYPE_TO_PROPERTY_TYPE_NAME[indexed_type]}, "
                    f"but an earlier subtype resolves to "
                    f"{PROPERTY_TYPE_TO_PROPERTY_TYPE_NAME[existing]}"
                )
            merged[name] = indexed_type

    # Any indexed field must resolve to the same PropertyValue type
    # in every Context type that has the field.
    for name, indexed_type in merged.items():
        for cls, field_type in all_field_property_types.get(name, {}).items():
            if field_type is not indexed_type:
                raise TypeError(
                    f"Context field {name!r} is indexed as "
                    f"{PROPERTY_TYPE_TO_PROPERTY_TYPE_NAME[indexed_type]} "
                    f"but {cls.__name__}.{name} resolves to "
                    f"{PROPERTY_TYPE_TO_PROPERTY_TYPE_NAME[field_type]}"
                )

    return merged


class MessageContext(BaseModel):
    """The content is communicated by a source."""

    type: Literal["message"] = "message"
    source: str

    indexed_properties: ClassVar[frozenset[str]] = frozenset({"type", "source"})


class CitationContext(BaseModel):
    """The content is cited from a source."""

    type: Literal["citation"] = "citation"
    source: str
    source_type: str | None = None
    location: str | None = None

    indexed_properties: ClassVar[frozenset[str]] = frozenset(
        {"type", "source", "source_type", "location"}
    )


ContextUnion = MessageContext | CitationContext

Context = Annotated[
    ContextUnion,
    Field(discriminator="type"),
]


# Concrete subtypes in the Context discriminated union,
# fixed at module import.
_CONTEXT_UNION_MEMBERS: Final[tuple[type[ContextUnion], ...]] = get_args(ContextUnion)


# Merged indexed properties schema across all Context subtypes,
# fixed at module import.
INDEXED_CONTEXT_PROPERTIES_SCHEMA: Final[Mapping[str, type[PropertyValue]]] = (
    types.MappingProxyType(
        resolve_indexed_context_properties_schema(_CONTEXT_UNION_MEMBERS)
    )
)


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
    context: Context | None = None
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
    context: Context | None = None
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


class FormatOptions(BaseModel):
    """Options for formatting query results."""

    timezone: InstanceOf[tzinfo] | None = None
    show_timezone_label: bool = True


# QueryResult: the result of a memory query.


class ScoredSegmentContext(BaseModel):
    """A segment context anchored on a seed segment, with a score."""

    score: float
    seed_segment_uuid: UUID
    segments: list[Segment]


class QueryResult(BaseModel):
    """Memory query result, ordered by reranker score."""

    scored_segment_contexts: list[ScoredSegmentContext]
