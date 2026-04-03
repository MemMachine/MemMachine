"""Event memory system for storing and retrieving events."""

import datetime
import json
import logging
from collections.abc import Iterable, Sequence
from typing import ClassVar, cast
from uuid import UUID, uuid4

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
    Collection,
    Record,
)

from .data_types import (
    Block,
    CitationContext,
    Content,
    Context,
    Derivative,
    Event,
    FileRef,
    FormatOptions,
    MessageContext,
    QueryResult,
    ReadFile,
    Segment,
    Text,
)
from .segment_store import SegmentStorePartition

logger = logging.getLogger(__name__)


class EventMemoryParams(BaseModel):
    """
    Parameters for EventMemory.

    Attributes:
        partition_key (str):
            Partition key for scoping the memory.
        collection (Collection):
            Collection instance in a vector store.
        segment_store_partition (SegmentStorePartition):
            Segment store partition handle for managing segments.
        embedder (Embedder):
            Embedder instance for creating embeddings.
        reranker (Reranker):
            Reranker instance for reranking search results.
        derive_sentences (bool):
            Whether to derive sentence-level derivatives from content (default: False).
        max_text_chunk_length (int):
            Max code-point length for text chunking in segment creation (default: 2000).
    """

    partition_key: str = Field(
        ...,
        description="Partition key for scoping the memory",
    )
    collection: InstanceOf[Collection] = Field(
        ...,
        description="Collection instance in a vector store",
    )
    segment_store_partition: InstanceOf[SegmentStorePartition] = Field(
        ...,
        description="Segment store partition handle for managing segments",
    )
    embedder: InstanceOf[Embedder] = Field(
        ...,
        description="Embedder instance for creating embeddings",
    )
    reranker: InstanceOf[Reranker] = Field(
        ...,
        description="Reranker instance for reranking search results",
    )
    derive_sentences: bool = Field(
        False,
        description="Whether to derive sentence-level derivatives from content",
    )
    max_text_chunk_length: int = Field(
        2000,
        description="Max code-point length for text chunking in segment creation",
    )


class EventMemory:
    """Event memory system."""

    # System-defined metadata keys. Underscore prefix is reserved.
    _SEGMENT_UUID_KEY = "_segment_uuid"
    _TIMESTAMP_KEY = "_timestamp"
    _CONTEXT_TYPE_KEY = "_context_type"
    _CONTEXT_SOURCE_KEY = "_context_source"

    _COLLECTION_SYSTEM_FIELD_MAPPING: ClassVar[dict[str, str]] = {
        "timestamp": _TIMESTAMP_KEY,
        "context.type": _CONTEXT_TYPE_KEY,
        "context.source": _CONTEXT_SOURCE_KEY,
    }

    def __init__(self, params: EventMemoryParams) -> None:
        """
        Initialize an EventMemory with the provided parameters.

        Args:
            params (EventMemoryParams):
                Parameters for the EventMemory.

        """
        self._partition_key = params.partition_key

        self._collection = params.collection
        self._segment_store_partition = params.segment_store_partition

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

    async def encode_events(
        self,
        events: Iterable[Event],
    ) -> None:
        """
        Encode events.

        Args:
            events (Iterable[Event]): The events to encode.

        """
        events = sorted(
            events,
            key=lambda event: (event.timestamp, event.uuid),
        )

        segments = [
            segment for event in events for segment in self._create_segments(event)
        ]

        segments_to_derivatives: dict[Segment, list[Derivative]] = {
            segment: self._derive_derivatives(segment) for segment in segments
        }

        derivatives = [
            derivative
            for segment_derivatives in segments_to_derivatives.values()
            for derivative in segment_derivatives
        ]

        derivative_embeddings = await self._embedder.ingest_embed(
            [derivative.text for derivative in derivatives],
        )

        await self._segment_store_partition.add_segments(
            {
                segment: [derivative.uuid for derivative in segment_derivatives]
                for segment, segment_derivatives in segments_to_derivatives.items()
            }
        )

        derivative_records = [
            EventMemory._build_derivative_record(derivative, derivative_embedding)
            for derivative, derivative_embedding in zip(
                derivatives,
                derivative_embeddings,
                strict=True,
            )
        ]

        if derivative_records:
            await self._collection.upsert(records=derivative_records)

    @staticmethod
    def _build_derivative_record(
        derivative: Derivative,
        derivative_embedding: Sequence[float],
    ) -> Record:
        """Build a vector record from a derivative and its embedding."""
        properties: dict[str, PropertyValue] = {}

        # System-defined metadata (underscore-prefixed).
        properties[EventMemory._SEGMENT_UUID_KEY] = str(derivative.segment_uuid)
        properties[EventMemory._TIMESTAMP_KEY] = derivative.timestamp

        match derivative.context:
            case MessageContext(source=source):
                properties[EventMemory._CONTEXT_TYPE_KEY] = "message"
                properties[EventMemory._CONTEXT_SOURCE_KEY] = source
            case CitationContext(source=source):
                properties[EventMemory._CONTEXT_TYPE_KEY] = "citation"
                properties[EventMemory._CONTEXT_SOURCE_KEY] = source

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
            case ReadFile(file=file_ref):
                return [
                    Segment(
                        uuid=uuid4(),
                        event_uuid=event.uuid,
                        index=0,
                        offset=0,
                        timestamp=event.timestamp,
                        block=file_ref,
                        properties=event.properties,
                    )
                ]
            case _:
                logger.warning("Unsupported body type: %s", type(event.body))
                return []

    def _segment_event_content_items(
        self,
        event: Event,
        items: Iterable[Block],
        context: Context | None,
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
            case FileRef():
                return []
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
    def _format_with_context(text: str, context: Context | None) -> str:
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
        context: Context | None,
        text: str,
    ) -> list[str]:
        """Derive formatted text strings from a text block."""
        if not self._derive_sentences:
            return [EventMemory._format_with_context(text, context)]
        return [
            EventMemory._format_with_context(sentence, context)
            for sentence in extract_sentences(text)
        ]

    @staticmethod
    def _to_collection_field(field: str) -> str:
        """Translate canonical filter field names to collection property keys."""
        internal_name, is_user_metadata = normalize_filter_field(field)
        if is_user_metadata:
            return demangle_user_metadata_key(internal_name)
        if field in EventMemory._COLLECTION_SYSTEM_FIELD_MAPPING:
            return EventMemory._COLLECTION_SYSTEM_FIELD_MAPPING[field]
        raise ValueError(f"Unknown filter field: {field!r}")

    async def query(
        self,
        query: str,
        *,
        max_num_segments: int = 20,
        expand_context: int = 0,
        property_filter: FilterExpr | None = None,
        format_options: FormatOptions | None = None,
    ) -> QueryResult:
        """
        Query event memory for segments relevant to the query.

        Args:
            query (str):
                The search query.
            max_num_segments (int):
                The maximum number of segments to return
                (default: 20).
            expand_context (int):
                The number of additional segments to include
                around each matched segment for additional context
                (default: 0).
            property_filter (FilterExpr | None):
                Attribute keys and values
                to use for filtering segments
                (default: None).
            format_options (FormatOptions | None):
                Options for formatting timestamps in output
                (default: None).

        Returns:
            QueryResult:
                Query result containing segments relevant to the query, ordered chronologically.

        """
        if format_options is None:
            format_options = FormatOptions()
        query_embedding = (
            await self._embedder.search_embed(
                [query],
            )
        )[0]

        # Translate filter fields for vector store.
        collection_filter = (
            map_filter_fields(property_filter, EventMemory._to_collection_field)
            if property_filter is not None
            else None
        )

        # Search derivative collection for matches.
        [query_result] = await self._collection.query(
            query_vectors=[query_embedding],
            limit=min(5 * max_num_segments, 200),
            property_filter=collection_filter,
            return_vector=False,
            return_properties=True,
        )

        # Extract seed segment UUIDs from vector metadata, preserving similarity order.
        # Deduplicate by first occurrence (multiple derivatives can map to the same segment).
        seed_segment_uuids = list(
            dict.fromkeys(
                UUID(
                    str(
                        cast(
                            dict[str, PropertyValue],
                            match.record.properties,
                        )[EventMemory._SEGMENT_UUID_KEY]
                    )
                )
                for match in query_result.matches
            )
        )

        expand_context = min(max(0, expand_context), max_num_segments - 1)
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

        # Rerank segment contexts.
        segment_context_scores = await self._score_segment_contexts(
            query,
            segment_contexts,
            format_options,
        )

        reranked_anchored_segment_contexts = [
            (seed_segment_uuid, segment_context)
            for _, seed_segment_uuid, segment_context in sorted(
                zip(
                    segment_context_scores,
                    kept_seed_segment_uuids,
                    segment_contexts,
                    strict=True,
                ),
                key=lambda triple: triple[0],
                reverse=True,
            )
        ]

        # Unify segment contexts.
        unified_segment_context = EventMemory._unify_anchored_segment_contexts(
            reranked_anchored_segment_contexts,
            max_num_segments=max_num_segments,
        )

        unified_segment_context_string = EventMemory.string_from_segment_context(
            unified_segment_context, format_options=format_options
        )

        return QueryResult(
            unified_segment_context=unified_segment_context,
            unified_segment_context_string=unified_segment_context_string,
        )

    async def _score_segment_contexts(
        self,
        query: str,
        segment_contexts: Iterable[Iterable[Segment]],
        format_options: FormatOptions,
    ) -> list[float]:
        """Score segment contexts based on their relevance to the query."""
        context_strings = []
        for segment_context in segment_contexts:
            context_string = EventMemory.string_from_segment_context(
                segment_context, format_options=format_options
            )
            context_strings.append(context_string)

        segment_context_scores = await self._reranker.score(query, context_strings)
        return segment_context_scores

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
                    context_string += json.dumps(accumulated_text) + "\n"
                first = False
                accumulated_text = ""

                timestamp = EventMemory._format_timestamp(
                    segment.timestamp, format_options
                )

                match segment.context:
                    case MessageContext(source=source):
                        context_string += f"{timestamp} {source}: "
                    case CitationContext(source=source):
                        context_string += f"{timestamp} From '{source}': "
                    case _:
                        context_string += f"{timestamp} "

            text = EventMemory._extract_text(segment.block)
            if text is not None:
                accumulated_text += text
            elif not is_continuation:
                context_string += f"[{segment.block.type}]\n"

            last_segment = segment

        if not first:
            context_string += json.dumps(accumulated_text) + "\n"

        return context_string.strip()

    @staticmethod
    def _extract_text(block: Block) -> str | None:
        """Extract text from a block, if it contains text."""
        match block:
            case Text(text=text):
                return text
            case _:
                return None

    @staticmethod
    def _format_timestamp(
        timestamp: datetime.datetime,
        format_options: FormatOptions,
    ) -> str:
        """Format a timestamp as a bracketed date/time string."""
        display_timestamp = (
            timestamp.astimezone(format_options.timezone)
            if format_options.timezone is not None
            else timestamp
        )
        date = EventMemory._format_date(display_timestamp.date())
        time = EventMemory._format_time(display_timestamp.time())
        if format_options.show_timezone_label:
            tz_label = EventMemory._format_timezone(display_timestamp)
            if tz_label:
                time += " " + tz_label
        return f"[{date} at {time}]"

    @staticmethod
    def _format_date(date: datetime.date) -> str:
        """Format the date as a string."""
        return date.strftime("%A, %B %d, %Y")

    @staticmethod
    def _format_time(time: datetime.time) -> str:
        """Format the time as a string."""
        return time.strftime("%I:%M %p")

    @staticmethod
    def _format_timezone(timestamp: datetime.datetime) -> str:
        """Format the timezone of a datetime as a UTC offset string."""
        offset = timestamp.utcoffset()
        if offset is None:
            return ""
        total_seconds = int(offset.total_seconds())
        sign = "+" if total_seconds >= 0 else "-"
        total_seconds = abs(total_seconds)
        hours, remainder = divmod(total_seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        if seconds:
            return f"UTC{sign}{hours:02d}:{minutes:02d}:{seconds:02d}"
        return f"UTC{sign}{hours:02d}:{minutes:02d}"

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
            await self._collection.delete(record_uuids=derivative_uuids)

        await self._segment_store_partition.delete_segments(
            segment_uuids=segment_uuids,
        )

    @staticmethod
    def _unify_anchored_segment_contexts(
        anchored_segment_contexts: Iterable[tuple[UUID, Iterable[Segment]]],
        max_num_segments: int,
    ) -> list[Segment]:
        """Unify anchored segment contexts into a single list within the limit."""
        unified_segment_context_set: set[Segment] = set()

        for seed_segment_uuid, context in anchored_segment_contexts:
            context = list(context)

            if len(unified_segment_context_set) >= max_num_segments:
                break
            if (len(unified_segment_context_set) + len(context)) <= max_num_segments:
                # It is impossible that the context exceeds the limit.
                unified_segment_context_set.update(context)
            else:
                # It is possible that the context exceeds the limit.
                # Prioritize segments near the seed segment.

                # Sort chronological segments by weighted index-proximity to the seed segment.
                seed_index = next(
                    index
                    for index, segment in enumerate(context)
                    if segment.uuid == seed_segment_uuid
                )

                seed_context = sorted(
                    context,
                    key=lambda segment: EventMemory._weighted_index_proximity(
                        segment=segment,
                        context=context,
                        seed_index=seed_index,
                    ),
                )

                # Add segments to unified context until limit is reached,
                # or until the context is exhausted.
                for segment in seed_context:
                    if len(unified_segment_context_set) >= max_num_segments:
                        break
                    unified_segment_context_set.add(segment)

        unified_segment_context = sorted(
            unified_segment_context_set,
            key=lambda segment: (
                segment.timestamp,
                segment.event_uuid,
                segment.index,
                segment.offset,
            ),
        )

        return unified_segment_context

    @staticmethod
    def _weighted_index_proximity(
        segment: Segment,
        context: Sequence[Segment],
        seed_index: int,
    ) -> float:
        proximity = context.index(segment) - seed_index
        if proximity >= 0:
            # Forward recall is better than backward recall.
            return (proximity - 0.5) / 2
        return -proximity
