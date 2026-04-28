"""Event memory system for storing and retrieving events."""

import datetime
import json
import logging
import time
from collections.abc import Iterable, Sequence
from typing import ClassVar, cast
from uuid import UUID, uuid4

from babel.dates import format_date, format_time, get_datetime_format
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pydantic import BaseModel, Field, InstanceOf

from memmachine_server.common.data_types import PropertyValue
from memmachine_server.common.embedder import Embedder
from memmachine_server.common.filter.filter_parser import (
    FilterExpr,
    demangle_user_metadata_key,
    map_filter_fields,
    normalize_filter_field,
)
from memmachine_server.common.reranker import Reranker
from memmachine_server.common.utils import (
    extract_sentences,
)
from memmachine_server.common.vector_store import (
    Record,
    VectorStoreCollection,
)

from .data_types import (
    Block,
    CitationContext,
    Content,
    Context,
    DateTimeStyle,
    Derivative,
    Event,
    FormatOptions,
    MessageContext,
    QueryResult,
    ScoredSegmentContext,
    Segment,
    Text,
)
from .segment_store import SegmentStorePartition

logger = logging.getLogger(__name__)

# CLDR datetime style levels, ordered from compact to verbose.
_DATETIME_STYLE_LEVELS: tuple[DateTimeStyle, ...] = ("short", "medium", "long", "full")


class EventMemoryParams(BaseModel):
    """
    Parameters for EventMemory.

    Attributes:
        vector_store_collection (VectorStoreCollection):
            Vector store collection.
        segment_store_partition (SegmentStorePartition):
            Segment store partition handle for managing segments.
        embedder (Embedder):
            Embedder instance for creating embeddings.
        reranker (Reranker | None):
            Reranker instance for scoring search results.
            If None, embedding similarity scores are used instead
            (default: None).
        derive_sentences (bool):
            Whether to derive sentence-level derivatives from content
            (default: False).
        max_text_chunk_length (int):
            Max code-point length for text chunking in segment creation
            (default: 500).
    """

    vector_store_collection: InstanceOf[VectorStoreCollection] = Field(
        ...,
        description="Vector store collection",
    )
    segment_store_partition: InstanceOf[SegmentStorePartition] = Field(
        ...,
        description="Segment store partition handle for managing segments",
    )
    embedder: InstanceOf[Embedder] = Field(
        ...,
        description="Embedder instance for creating embeddings",
    )
    reranker: InstanceOf[Reranker] | None = Field(
        None,
        description="Reranker instance for scoring search results. "
        "If None, embedding similarity scores are used instead",
    )
    derive_sentences: bool = Field(
        False,
        description="Whether to derive sentence-level derivatives from content",
    )
    max_text_chunk_length: int = Field(
        500,
        description="Max code-point length for text chunking in segment creation",
    )


class EventMemory:
    """Event memory system."""

    # System-defined metadata field names. Reserved.
    _SEGMENT_UUID_FIELD_NAME = "_segment_uuid"
    _TIMESTAMP_FIELD_NAME = "_timestamp"

    _BASE_EVENT_MEMORY_FIELD_NAMES: ClassVar[frozenset[str]] = frozenset(
        {_SEGMENT_UUID_FIELD_NAME, _TIMESTAMP_FIELD_NAME}
    )

    @classmethod
    def expected_vector_store_collection_schema(cls) -> dict[str, type[PropertyValue]]:
        """
        Return the vector store collection schema expected by EventMemory.

        Callers should merge this with any user or external system-defined properties
        when creating the collection so that EventMemory's reserved fields are efficiently filterable.
        """
        return {
            cls._SEGMENT_UUID_FIELD_NAME: cast(type[PropertyValue], str),
            cls._TIMESTAMP_FIELD_NAME: cast(type[PropertyValue], datetime.datetime),
        }

    def __init__(self, params: EventMemoryParams) -> None:
        """
        Initialize an EventMemory with the provided parameters.

        Args:
            params (EventMemoryParams):
                Parameters for the EventMemory.

        """
        self._vector_store_collection = params.vector_store_collection
        self._segment_store_partition = params.segment_store_partition
        self._schema_fields = frozenset(
            params.vector_store_collection.config.properties_schema
        )

        missing_base_fields = (
            EventMemory._BASE_EVENT_MEMORY_FIELD_NAMES - self._schema_fields
        )
        if missing_base_fields:
            raise ValueError(
                f"Collection schema missing fields required by EventMemory: "
                f"{', '.join(sorted(missing_base_fields))}"
            )

        self._embedder = params.embedder
        self._reranker = params.reranker

        self._derive_sentences = params.derive_sentences

        self._text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=params.max_text_chunk_length,
            chunk_overlap=0,
            separators=[
                "\n\n",
                "],\n",
                "},\n",
                "),\n",
                "]\n",
                "}\n",
                ")\n",
                ",\n",
                "\uff1f\n",  # Fullwidth question mark
                "?\n",
                "\uff01\n",  # Fullwidth exclamation mark
                "!\n",
                "\u3002\n",  # Ideographic full stop
                ".\n",
                "\uff1f",  # Fullwidth question mark
                "? ",
                "\uff01",  # Fullwidth exclamation mark
                "! ",
                "\u3002",  # Ideographic full stop
                ". ",
                "; ",
                ": ",
                "—",
                "--",
                "\uff0c",  # Fullwidth comma
                "\u3001",  # Ideographic comma
                ", ",
                "\u200b",  # Zero-width space
                " ",
                "",
            ],
            keep_separator="end",
        )

    @classmethod
    def _is_reserved_field(cls, field: str) -> bool:
        """Returns whether a property field is reserved."""
        return field in cls._BASE_EVENT_MEMORY_FIELD_NAMES

    def _validate_events(self, events: Iterable[Event]) -> None:
        """
        Validate a batch of events before encoding.

        Raises ValueError if any event supplies a reserved field name in its properties,
        or if the collection schema is missing fields required by EventMemory.
        """
        events = list(events)

        reserved_fields = {
            field
            for event in events
            for field in event.properties
            if EventMemory._is_reserved_field(field)
        }
        if reserved_fields:
            raise ValueError(
                f"Event properties must not contain reserved fields: "
                f"{', '.join(sorted(reserved_fields))}"
            )

        missing_fields = (
            EventMemory._BASE_EVENT_MEMORY_FIELD_NAMES - self._schema_fields
        )
        if missing_fields:
            raise ValueError(
                f"Events require properties missing from the collection schema: "
                f"{', '.join(sorted(missing_fields))}"
            )

    async def encode_events(
        self,
        events: Iterable[Event],
    ) -> None:
        """
        Encode events.

        Args:
            events (Iterable[Event]): The events to encode.

        Raises:
            ValueError:
                If any event supplies a reserved field name in its properties,
                or if the collection schema is missing fields required by any event's Context type.
        """
        t_start = time.monotonic()

        events = list(events)
        self._validate_events(events)

        events = sorted(
            events,
            key=lambda event: (event.timestamp, event.uuid),
        )

        segments = [
            segment for event in events for segment in self._create_segments(event)
        ]
        t_segment = time.monotonic()

        segments_to_derivatives: dict[Segment, list[Derivative]] = {
            segment: self._derive_derivatives(segment) for segment in segments
        }

        derivatives = [
            derivative
            for segment_derivatives in segments_to_derivatives.values()
            for derivative in segment_derivatives
        ]
        t_derive = time.monotonic()

        derivative_embeddings = await self._embedder.ingest_embed(
            [derivative.text for derivative in derivatives],
        )
        t_embed = time.monotonic()

        await self._segment_store_partition.add_segments(
            {
                segment: [derivative.uuid for derivative in segment_derivatives]
                for segment, segment_derivatives in segments_to_derivatives.items()
            }
        )
        t_seg_store = time.monotonic()

        derivative_records = [
            EventMemory._build_derivative_record(derivative, derivative_embedding)
            for derivative, derivative_embedding in zip(
                derivatives,
                derivative_embeddings,
                strict=True,
            )
        ]

        if derivative_records:
            await self._vector_store_collection.upsert(records=derivative_records)
        t_v_store = time.monotonic()

        logger.debug(
            "Ingest timing: "
            "segment=%.3fs "
            "derive=%.3fs "
            "embed=%.3fs "
            "seg_store=%.3fs "
            "v_store=%.3fs "
            "total=%.3fs",
            t_segment - t_start,
            t_derive - t_segment,
            t_embed - t_derive,
            t_seg_store - t_embed,
            t_v_store - t_seg_store,
            t_v_store - t_start,
        )

    @classmethod
    def _build_derivative_record(
        cls,
        derivative: Derivative,
        derivative_embedding: Sequence[float],
    ) -> Record:
        """Build a vector record from a derivative and its embedding."""
        properties: dict[str, PropertyValue] = {}

        # System-defined metadata (underscore-prefixed).
        properties[cls._SEGMENT_UUID_FIELD_NAME] = str(derivative.segment_uuid)
        properties[cls._TIMESTAMP_FIELD_NAME] = derivative.timestamp

        # User-defined properties.
        properties.update(derivative.properties)

        return Record(
            uuid=derivative.uuid,
            vector=list(derivative_embedding),
            properties=properties,
        )

    def _create_segments(
        self,
        event: Event,
    ) -> list[Segment]:
        """
        Create segments from an event.

        Args:
            event (Event):
                The event from which to create segments.

        Returns:
            list[Segment]: A list of created segments.

        """
        match event.body:
            case Content(context=context, items=primitives):
                return self._segment_event_content_items(
                    event=event,
                    items=primitives,
                    context=context,
                )
            case _:
                logger.warning("Unsupported body type: %s", type(event.body))
                return []

    def _segment_event_content_items(
        self,
        event: Event,
        items: Iterable[Block],
        context: Context,
    ) -> list[Segment]:
        """Split content items into single-block segments, propagating context."""
        segments: list[Segment] = []
        for index, item in enumerate(items):
            match item:
                case Text(text=text):
                    chunks = self._text_splitter.split_text(text)
                    segments.extend(
                        Segment(
                            uuid=uuid4(),
                            event_uuid=event.uuid,
                            index=index,
                            offset=offset,
                            timestamp=event.timestamp,
                            block=Text(text=chunk),
                            context=context,
                            properties=event.properties,
                        )
                        for offset, chunk in enumerate(chunks)
                    )
                case _:
                    segments.append(
                        Segment(
                            uuid=uuid4(),
                            event_uuid=event.uuid,
                            index=index,
                            offset=0,
                            timestamp=event.timestamp,
                            block=item,
                            context=context,
                            properties=event.properties,
                        )
                    )
        return segments

    def _derive_derivatives(
        self,
        segment: Segment,
    ) -> list[Derivative]:
        """
        Derive derivatives from a segment.

        Args:
            segment (Segment):
                The segment from which to derive derivatives.

        Returns:
            list[Derivative]: A list of derived derivatives.

        """
        match segment.block:
            case Text(text=text):
                derivative_texts = self._extract_derivative_texts(
                    context=segment.context, text=text
                )
            case _:
                logger.warning("Non-text primitive derivatives are not yet supported")
                return []

        return [
            Derivative(
                uuid=uuid4(),
                segment_uuid=segment.uuid,
                timestamp=segment.timestamp,
                context=segment.context,
                text=text,
                properties=segment.properties,
            )
            for text in derivative_texts
        ]

    @staticmethod
    def _format_with_context(text: str, context: Context) -> str:
        """Format text within its context."""
        match context:
            case MessageContext(source=source):
                return f"{source}: {text}"
            case CitationContext(source=source):
                return f"From '{source}': {text}"
            case _:
                return text

    def _extract_derivative_texts(
        self,
        *,
        context: Context,
        text: str,
    ) -> list[str]:
        """Derive formatted text strings from a text block."""
        if not self._derive_sentences:
            return [EventMemory._format_with_context(text, context)]
        return [
            EventMemory._format_with_context(sentence, context)
            for sentence in extract_sentences(text)
        ]

    @classmethod
    def _to_vector_record_property(cls, field: str) -> str:
        """
        Translates canonical filter field name to vector record property.

        Event memory base properties (`foo`) translate to `_foo`.
        User-defined properties (`m.foo` / `metadata.foo`) translate to `foo`.
        """
        internal_name, is_user_metadata = normalize_filter_field(field)
        if is_user_metadata:
            return demangle_user_metadata_key(internal_name)
        return f"_{field}"

    async def query(
        self,
        query: str,
        *,
        vector_search_limit: int = 20,
        expand_context: int = 0,
        property_filter: FilterExpr | None = None,
        format_options: FormatOptions | None = None,
    ) -> QueryResult:
        """
        Query event memory for segments relevant to the query.

        Args:
            query (str):
                The search query.
            vector_search_limit (int):
                The maximum number of seed segments
                to retrieve from the vector search
                (default: 20).
            expand_context (int):
                The number of additional segments to include
                around each matched segment for additional context
                (default: 0).
            property_filter (FilterExpr | None):
                Property fields and values
                to use for filtering segments
                (default: None).
            format_options (FormatOptions | None):
                Options for formatting.
                (default: None).

        Returns:
            QueryResult:
                The query result.

        """
        if format_options is None:
            format_options = FormatOptions()

        t_start = time.monotonic()
        query_embedding = (
            await self._embedder.search_embed(
                [query],
            )
        )[0]
        t_embed = time.monotonic()

        # Translate filter fields for vector store.
        collection_filter = (
            map_filter_fields(property_filter, EventMemory._to_vector_record_property)
            if property_filter is not None
            else None
        )

        # Search derivative collection for matches.
        [query_result] = await self._vector_store_collection.query(
            query_vectors=[query_embedding],
            limit=vector_search_limit,
            property_filter=collection_filter,
            return_vector=False,
            return_properties=True,
        )
        t_v_query = time.monotonic()

        # Extract seed segment UUIDs and their best embedding scores.
        # Deduplicate by first occurrence (multiple derivatives can map to the same segment).
        # First occurrence has the best score since matches are ordered best-to-worst.
        seed_embedding_scores: dict[UUID, float] = {}
        for match in query_result.matches:
            segment_uuid = UUID(
                str(
                    cast(
                        dict[str, PropertyValue],
                        match.record.properties,
                    )[EventMemory._SEGMENT_UUID_FIELD_NAME]
                )
            )
            if segment_uuid not in seed_embedding_scores:
                seed_embedding_scores[segment_uuid] = match.score

        seed_segment_uuids = list(seed_embedding_scores)

        max_backward_segments = expand_context // 3
        max_forward_segments = expand_context - max_backward_segments

        segment_contexts_by_seed = (
            await self._segment_store_partition.get_segment_contexts(
                seed_segment_uuids=seed_segment_uuids,
                max_backward_segments=max_backward_segments,
                max_forward_segments=max_forward_segments,
                property_filter=property_filter,
            )
        )
        t_s_query = time.monotonic()

        # Filter to seeds with results, preserving similarity order.
        kept_seed_segment_uuids = [
            seed_segment_uuid
            for seed_segment_uuid in seed_segment_uuids
            if seed_segment_uuid in segment_contexts_by_seed
        ]
        segment_contexts: list[list[Segment]] = [
            segment_contexts_by_seed[seed_segment_uuid]
            for seed_segment_uuid in kept_seed_segment_uuids
        ]

        # Use embedding scores if reranker is not available.
        if self._reranker is None:
            scores = [
                seed_embedding_scores[seed_uuid]
                for seed_uuid in kept_seed_segment_uuids
            ]
        else:
            scores = await self._score_segment_contexts(
                query, segment_contexts, format_options
            )
        t_score = time.monotonic()

        # Reranker scores are always higher-is-better.
        # Embedding scores depend on the similarity metric.
        higher_is_better = (
            self._reranker is not None
            or self._vector_store_collection.config.similarity_metric.higher_is_better
        )

        # Return scored contexts ordered by score.
        scored_segment_contexts = [
            ScoredSegmentContext(
                score=score, seed_segment_uuid=seed_uuid, segments=context
            )
            for score, seed_uuid, context in sorted(
                zip(
                    scores,
                    kept_seed_segment_uuids,
                    segment_contexts,
                    strict=True,
                ),
                key=lambda triple: triple[0],
                reverse=higher_is_better,
            )
        ]
        t_result = time.monotonic()

        logger.debug(
            "Query timing: "
            "embed=%.3fs "
            "v_query=%.3fs "
            "s_query=%.3fs "
            "score=%.3fs "
            "result=%.3fs "
            "total=%.3fs",
            t_embed - t_start,
            t_v_query - t_embed,
            t_s_query - t_v_query,
            t_score - t_s_query,
            t_result - t_score,
            t_result - t_start,
        )

        return QueryResult(scored_segment_contexts=scored_segment_contexts)

    async def _score_segment_contexts(
        self,
        query: str,
        segment_contexts: Iterable[Iterable[Segment]],
        format_options: FormatOptions,
    ) -> list[float]:
        """Score segment contexts using the reranker. Requires reranker."""
        assert self._reranker is not None
        context_strings = [
            EventMemory.string_from_segment_context(
                segment_context, format_options=format_options
            )
            for segment_context in segment_contexts
        ]
        return await self._reranker.score(query, context_strings)

    @staticmethod
    def string_from_segment_context(
        segment_context: Iterable[Segment],
        *,
        format_options: FormatOptions | None = None,
    ) -> str:
        """Format segment context as a string."""
        if format_options is None:
            format_options = FormatOptions()

        context_string = ""
        last_segment: Segment | None = None
        accumulated_text = ""
        first = True

        for segment in segment_context:
            is_continuation = (
                last_segment is not None
                and segment.event_uuid == last_segment.event_uuid
                and segment.index == last_segment.index
            )

            if not is_continuation:
                if not first:
                    context_string += (
                        json.dumps(accumulated_text, ensure_ascii=False) + "\n"
                    )
                first = False
                accumulated_text = ""
                context_string += EventMemory._segment_header(segment, format_options)

            text = EventMemory._extract_text(segment.block)
            if text is not None:
                accumulated_text += text
            elif not is_continuation:
                context_string += f"[{segment.block.type}]\n"

            last_segment = segment

        if not first:
            context_string += json.dumps(accumulated_text, ensure_ascii=False) + "\n"

        return context_string.strip()

    @staticmethod
    def string_from_segment_contexts(
        segment_contexts: Iterable[Iterable[Segment]],
        *,
        format_options: FormatOptions | None = None,
    ) -> str:
        """Format multiple segment contexts as a string, separating disconnected components."""
        segment_contexts = [list(context) for context in segment_contexts]

        # Deduplicate segments and build union-find over their UUIDs in one pass.
        segments_by_uuid: dict[UUID, Segment] = {}
        component_parent: dict[UUID, UUID] = {}

        def find(uuid: UUID) -> UUID:
            component_parent.setdefault(uuid, uuid)
            root = uuid
            while component_parent[root] != root:
                root = component_parent[root]
            while component_parent[uuid] != root:
                parent = component_parent[uuid]
                component_parent[uuid] = root
                uuid = parent
            return root

        for context in segment_contexts:
            first_segment_root: UUID | None = None
            for segment in context:
                segments_by_uuid.setdefault(segment.uuid, segment)
                if first_segment_root is None:
                    first_segment_root = find(segment.uuid)
                else:
                    segment_root = find(segment.uuid)
                    component_parent[segment_root] = first_segment_root

        # Group unique segments by component root.
        segments_by_root: dict[UUID, list[Segment]] = {}
        for segment_uuid, segment in segments_by_uuid.items():
            segments_by_root.setdefault(find(segment_uuid), []).append(segment)

        # Sort segments within each component, then order components chronologically.
        def segment_key(segment: Segment) -> tuple:
            return (
                segment.timestamp,
                segment.event_uuid,
                segment.index,
                segment.offset,
            )

        components = list(segments_by_root.values())
        for component in components:
            component.sort(key=segment_key)
        components.sort(key=lambda segments: segment_key(segments[0]))

        return "\n\n".join(
            EventMemory.string_from_segment_context(
                segments, format_options=format_options
            )
            for segments in components
        )

    @staticmethod
    def _segment_header(segment: Segment, format_options: FormatOptions) -> str:
        """Build the header emitted before a segment."""
        formatted_timestamp = EventMemory._format_timestamp(
            segment.timestamp,
            date_style=format_options.date_style,
            time_style=format_options.time_style,
            locale=format_options.locale,
            timezone=format_options.timezone,
        )
        timestamp_prefix = f"[{formatted_timestamp}] " if formatted_timestamp else ""

        match segment.context:
            case MessageContext(source=source):
                return f"{timestamp_prefix}{source}: "
            case CitationContext(source=source):
                return f"{timestamp_prefix}From '{source}': "
            case _:
                return timestamp_prefix

    @staticmethod
    def _format_timestamp(
        timestamp: datetime.datetime,
        *,
        date_style: DateTimeStyle | None,
        time_style: DateTimeStyle | None,
        locale: str,
        timezone: datetime.tzinfo | None,
    ) -> str:
        """Format a timestamp."""
        if date_style is None and time_style is None:
            return ""

        normalized_timestamp = (
            timestamp.astimezone(timezone) if timezone is not None else timestamp
        )

        date_string = ""
        time_string = ""

        if date_style is not None:
            date_string = format_date(
                normalized_timestamp, format=date_style, locale=locale
            )
        if time_style is not None:
            time_string = format_time(
                normalized_timestamp, format=time_style, locale=locale
            )

        if not time_string:
            return date_string
        if not date_string:
            return time_string

        connector_style = _DATETIME_STYLE_LEVELS[
            max(
                _DATETIME_STYLE_LEVELS.index(date_style),
                _DATETIME_STYLE_LEVELS.index(time_style),
            )
        ]

        template = str(get_datetime_format(connector_style, locale=locale))
        return template.replace("{1}", date_string).replace("{0}", time_string)

    @staticmethod
    def _extract_text(block: Block) -> str | None:
        """Extract text from a block, if it contains text."""
        match block:
            case Text(text=text):
                return text
            case _:
                return None

    async def forget_events(self, event_uuids: Iterable[UUID]) -> None:
        """Forget events by their UUIDs."""
        event_uuids = set(event_uuids)
        if not event_uuids:
            return

        # Snapshot segment UUIDs for these events.
        segments_by_event = (
            await self._segment_store_partition.get_segment_uuids_by_event_uuids(
                event_uuids=event_uuids,
            )
        )
        segment_uuids = {
            segment_uuid
            for event_segment_uuids in segments_by_event.values()
            for segment_uuid in event_segment_uuids
        }
        if not segment_uuids:
            return

        # Get derivative UUIDs for those segments.
        derivatives_by_segment = (
            await self._segment_store_partition.get_derivative_uuids_by_segment_uuids(
                segment_uuids=segment_uuids,
            )
        )
        derivative_uuids = {
            derivative_uuid
            for segment_derivative_uuids in derivatives_by_segment.values()
            for derivative_uuid in segment_derivative_uuids
        }

        # Delete from vector DB first, then segment store.
        if derivative_uuids:
            await self._vector_store_collection.delete(record_uuids=derivative_uuids)

        await self._segment_store_partition.delete_segments(
            segment_uuids=segment_uuids,
        )

    @staticmethod
    def build_query_result_context(
        query_result: QueryResult,
        max_num_segments: int,
    ) -> list[Segment]:
        """
        Build a single segment context from the query result within the limit.

        Iterates contexts in score order, accumulating segments until the limit is reached.
        When a context would exceed the limit, segments nearest the seed are prioritized.
        Deduplicates across segment contexts in the query result.

        Args:
            query_result (QueryResult):
                The query result with scored anchored segment contexts.
            max_num_segments (int):
                The maximum number of segments to return.

        Returns:
            list[Segment]:
                Deduplicated segments ordered chronologically.
        """
        unified: set[Segment] = set()

        for scored_context in query_result.scored_segment_contexts:
            context = scored_context.segments

            if len(unified) >= max_num_segments:
                break
            if (len(unified) + len(context)) <= max_num_segments:
                unified.update(context)
            else:
                # Prioritize segments near the seed segment.
                seed_index = next(
                    index
                    for index, segment in enumerate(context)
                    if segment.uuid == scored_context.seed_segment_uuid
                )

                for segment in sorted(
                    context,
                    key=lambda s: EventMemory._seed_proximity(s, context, seed_index),
                ):
                    if len(unified) >= max_num_segments:
                        break
                    unified.add(segment)

        return sorted(
            unified,
            key=lambda segment: (
                segment.timestamp,
                segment.event_uuid,
                segment.index,
                segment.offset,
            ),
        )

    @staticmethod
    def string_from_query_result(
        query_result: QueryResult,
        *,
        max_num_segments: int | None = None,
        format_options: FormatOptions | None = None,
    ) -> str:
        """Format a query result as a string with breaks between disconnected contexts."""
        contexts: list[list[Segment]] = [
            list(scored_context.segments)
            for scored_context in query_result.scored_segment_contexts
        ]

        if max_num_segments is not None:
            included = {
                segment.uuid
                for segment in EventMemory.build_query_result_context(
                    query_result, max_num_segments
                )
            }
            contexts = [
                [segment for segment in context if segment.uuid in included]
                for context in contexts
            ]

        return EventMemory.string_from_segment_contexts(
            contexts, format_options=format_options
        )

    @staticmethod
    def _seed_proximity(
        segment: Segment,
        context: list[Segment],
        seed_index: int,
    ) -> float:
        """Score a segment by its proximity to the seed. Lower is closer."""
        offset = context.index(segment) - seed_index
        if offset >= 0:
            # Forward context is more useful than backward.
            return (offset - 0.5) / 2
        return -offset
