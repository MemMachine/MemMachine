"""The request and response bodies of the v1 event-memory routes."""

from datetime import UTC, datetime
from typing import Annotated, Final, Self
from uuid import UUID, uuid4
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from pydantic import (
    AfterValidator,
    AwareDatetime,
    BaseModel,
    Field,
    JsonValue,
    StringConstraints,
)

from memmachine_server.common.data_types import PropertyValue
from memmachine_server.episodic_memory.event_memory.data_types import (
    Context,
    DateTimeFormat,
    DateTimeStyle,
    Event,
    RegisteredBlock,
    Segment,
    encode_block,
)

TENANT_NAME_MAX_BYTES: Final[int] = 255
"""Bound on a tenant name, in bytes, matching the bound on a session id."""


def _within_tenant_name_bound(name: str) -> str:
    """Return the name, unless it is longer than a tenant name may be."""
    size = len(name.encode())
    if size > TENANT_NAME_MAX_BYTES:
        raise ValueError(
            f"Tenant name is {size} bytes; the maximum is {TENANT_NAME_MAX_BYTES}"
        )
    return name


TenantName = Annotated[
    str,
    StringConstraints(min_length=1),
    AfterValidator(_within_tenant_name_bound),
    Field(description="The tenant's name, an opaque string the server resolves"),
]
"""A tenant's name in a path: never empty, at most `TENANT_NAME_MAX_BYTES` bytes."""


class TenantBody(BaseModel):
    """A tenant, as v1 knows it: a name, and nothing else yet."""

    name: str = Field(description="The tenant's name")


class DateTimeFormatSpec(BaseModel):
    """How a timestamp is written into the text a response renders."""

    date_style: DateTimeStyle | None = Field(
        "full", description="The CLDR style of the date, or null to omit the date"
    )
    time_style: DateTimeStyle | None = Field(
        "long", description="The CLDR style of the time, or null to omit the time"
    )
    locale: str = Field(
        "en_US", description="The CLDR locale the date and time are written in"
    )
    timezone: str | None = Field(
        None,
        description=(
            "The IANA name of the zone a timestamp is converted to before it "
            "is written, or null to write it in the zone it carries"
        ),
    )

    def to_datetime_format(self) -> DateTimeFormat:
        """This specification as the memory's format.

        Raises:
            ValueError: If `timezone` names no zone this system knows.
        """
        zone: ZoneInfo | None = None
        if self.timezone is not None:
            try:
                zone = ZoneInfo(self.timezone)
            except (ZoneInfoNotFoundError, ValueError) as error:
                raise ValueError(f"Unknown timezone: {self.timezone!r}") from error
        return DateTimeFormat(
            date_style=self.date_style,
            time_style=self.time_style,
            locale=self.locale,
            timezone=zone,
        )


class SegmentSelection(BaseModel):
    """What both a query and an expansion select and render with.

    The filters narrow the segments a request may reach; the format and
    the parts decide how the ones it reaches are written.
    """

    since: AwareDatetime | None = Field(
        None,
        description="Inclusive lower bound on the events' timestamps, with offset",
    )
    until: AwareDatetime | None = Field(
        None,
        description="Exclusive upper bound on the events' timestamps, with offset",
    )
    source_ids: list[str] | None = Field(
        None,
        description=(
            "Keep only the segments of these sources; an empty list keeps none, "
            "and null keeps every source"
        ),
    )
    block_kinds: list[str] | None = Field(
        None,
        description=(
            "Keep only the segments whose block is of these kinds; an empty list "
            "keeps none, and null keeps every kind"
        ),
    )
    filter: str | None = Field(
        None,
        description=(
            "A filter over the events' own properties, in the server's filter "
            'grammar, for example `m.project = "memmachine" and m.tool != "bash"`'
        ),
    )
    datetime_format: DateTimeFormatSpec = Field(
        default_factory=DateTimeFormatSpec,
        description="How a timestamp is written into the rendered text",
    )
    parts: list[str] = Field(
        default=["author"],
        description=(
            "The context part kinds composed into the rendered text, in the "
            "order they are written"
        ),
    )


class RerankSpec(BaseModel):
    """How the candidates of a query are reranked before it answers."""

    reranker: str | None = Field(
        None,
        description=(
            "The reranker to score with. This server scores with the tenant's "
            "own reranker and resolves no other, so naming one is rejected"
        ),
    )
    candidates: int | None = Field(
        None,
        gt=0,
        description=(
            "How many hits the vector search fetches for the reranker to score; "
            "null takes the tenant's default"
        ),
    )
    min_score: float | None = Field(
        None,
        description="Drop the hits the reranker scores below this; null drops none",
    )


class QueryRequest(SegmentSelection):
    """What a query asks for."""

    query: str = Field(min_length=1, description="The text to query for")
    limit: int | None = Field(
        None,
        gt=0,
        description="How many hits to answer with; null takes the tenant's default",
    )
    min_cosine_similarity: float | None = Field(
        None,
        ge=-1.0,
        le=1.0,
        description=(
            "Drop the matches whose cosine similarity is below this; null drops none"
        ),
    )
    expand_context: int | None = Field(
        None,
        ge=0,
        description=(
            "How many neighbors to render around each hit; null takes the "
            "tenant's default"
        ),
    )
    rerank: RerankSpec | None = Field(
        default_factory=RerankSpec,
        description=(
            "How to rerank the candidates; null reranks nothing, and omitting "
            "it reranks with the tenant's reranker when it has one"
        ),
    )
    session_ids: list[str] | None = Field(
        None,
        description=(
            "Keep only the segments of these sessions; an empty list keeps none, "
            "and null keeps every session"
        ),
    )


class ExpandRequest(SegmentSelection):
    """What an expansion around one segment asks for."""

    anchor: UUID = Field(description="The uuid of the segment to expand around")
    before: int | None = Field(
        None,
        ge=0,
        description=(
            "How many segments to walk backward from the anchor; null takes the "
            "tenant's default"
        ),
    )
    after: int | None = Field(
        None,
        ge=0,
        description=(
            "How many segments to walk forward from the anchor; null takes the "
            "tenant's default"
        ),
    )


class SegmentBody(BaseModel):
    """One piece of one of an event's blocks, carrying the event's fields.

    A segment carries no ingestion position: this deployment holds no
    event store, so there is no ingestion order to report.
    """

    uuid: UUID = Field(description="The uuid of the segment")
    event_uuid: UUID = Field(description="The uuid of the event it is a piece of")
    index: int = Field(description="Position of the block among the event's blocks")
    offset: int = Field(description="Position of the piece among the block's pieces")
    timestamp: datetime = Field(description="The event's timestamp, with offset")
    session_id: str = Field(description="The event's session id")
    source_id: str | None = Field(description="The event's source id, if it has one")
    context: dict[str, JsonValue] = Field(
        description="The event's context, its parts keyed by kind"
    )
    block: dict[str, JsonValue] = Field(description="The piece of the event's block")
    properties: dict[str, PropertyValue] = Field(
        description="The event's own properties"
    )

    @classmethod
    def from_segment(cls, segment: Segment) -> Self:
        """The body of a stored segment."""
        return cls(
            uuid=segment.uuid,
            event_uuid=segment.event_uuid,
            index=segment.index,
            offset=segment.offset,
            timestamp=segment.timestamp,
            session_id=segment.session_id,
            source_id=segment.source_id,
            context=segment.context.encode(),
            block=encode_block(segment.block),
            properties=segment.properties,
        )


class QueryHitBody(BaseModel):
    """A segment a query matched, with the window rendered around it."""

    score: float = Field(description="Relevance of the seed segment; higher is better")
    seed: int = Field(description="The index in `segments` of the matched segment")
    segments: list[SegmentBody] = Field(
        description="The seed and its neighbors, in the store's order"
    )
    text: str = Field(
        description="The window, rendered with its session and segment ids"
    )


class QueryResponse(BaseModel):
    """What a query answers with."""

    hits: list[QueryHitBody] = Field(description="The hits, in descending score")


class ExpandResponse(BaseModel):
    """What an expansion answers with: the two sides of the anchor."""

    before: list[SegmentBody] = Field(
        description="The segments before the anchor, in the store's order"
    )
    after: list[SegmentBody] = Field(
        description="The segments after the anchor, in the store's order"
    )
    before_text: str = Field(
        description="The segments before the anchor, rendered with their ids"
    )
    after_text: str = Field(
        description="The segments after the anchor, rendered with their ids"
    )


class EventSpec(BaseModel):
    """An entry to remember on a session's timeline."""

    id: UUID | None = Field(
        None,
        description=(
            "The event's uuid, which is how a client deduplicates its writes; "
            "null mints one, and a retry then stores the event again"
        ),
    )
    timestamp: AwareDatetime | None = Field(
        None,
        description=(
            "When the event happened, with offset; null takes the server's "
            "current UTC time"
        ),
    )
    session_id: str = Field(
        min_length=1, description="The session the event belongs to"
    )
    source_id: str | None = Field(
        None, description="The source of the event, if it has one"
    )
    context: dict[str, dict[str, JsonValue]] = Field(
        default_factory=dict,
        description=(
            "The context surrounding the event, its parts keyed by kind, for "
            'example `{"author": {"name": "Alice"}}`'
        ),
    )
    blocks: list[RegisteredBlock] = Field(
        description=(
            "The event's content, in order, each block of a registered kind; "
            '`{"kind": "text", "text": ...}` is built in'
        )
    )
    properties: dict[str, PropertyValue] = Field(
        default_factory=dict,
        description="Scalar values the event can be filtered by",
    )

    def to_event(self) -> Event:
        """The event to encode, with what the request left out filled in.

        Raises:
            ValueError:
                If a field is outside what an event may carry, such as a
                session id longer than the store's key.
        """
        return Event(
            uuid=self.id if self.id is not None else uuid4(),
            timestamp=self.timestamp
            if self.timestamp is not None
            else datetime.now(UTC),
            session_id=self.session_id,
            source_id=self.source_id,
            context=Context.decode(self.context),
            blocks=list(self.blocks),
            properties=self.properties,
        )


class StoredEvents(BaseModel):
    """What an ingest answers with."""

    stored: list[UUID] = Field(
        description="The uuids of the events stored, in the order they were given"
    )


class ForgetEventsRequest(BaseModel):
    """Which events to forget."""

    ids: list[UUID] = Field(description="The uuids of the events to forget")
