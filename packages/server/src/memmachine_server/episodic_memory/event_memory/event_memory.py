"""Event memory system for storing and retrieving events."""

import asyncio
import datetime
import json
import logging
from collections.abc import Iterable, Sequence
from contextlib import AbstractAsyncContextManager, nullcontext
from typing import ClassVar, cast
from uuid import UUID, uuid4

import numpy as np
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pydantic import BaseModel, Field, InstanceOf

from memmachine_server.common.data_types import PropertyValue, SimilarityMetric
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
from memmachine_server.common.vector_store.data_types import (
    QueryResult as VectorStoreQueryResult,
)

from .data_types import (
    INDEXED_CONTEXT_PROPERTIES_SCHEMA,
    Block,
    CitationContext,
    Content,
    Context,
    ContextUnion,
    Derivative,
    Event,
    FileRef,
    FormatOptions,
    MessageContext,
    QueryResult,
    ReadFile,
    ScoredSegmentContext,
    Segment,
    Text,
)
from .segment_store import SegmentStorePartition

logger = logging.getLogger(__name__)


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
            (default: 2000).
        eviction_similarity_threshold (float | None):
            Similarity threshold at which vectors are considered part of the same cluster
            for eviction evaluation. None disables eviction.
            (default: None).
        eviction_search_limit (int):
            Maximum number of similar vectors to retrieve per derivative for eviction evaluation
            (default: 20).
        eviction_target_size (int):
            Target cluster size; eviction starts when the number of similar vectors exceeds this
            (default: 15).
        serialize_encode (bool):
            Serialize encode_events calls with an async lock
            (default: False).
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
        2000,
        description="Max code-point length for text chunking in segment creation",
    )
    eviction_similarity_threshold: float | None = Field(
        None,
        description="Similarity threshold at which vectors are considered part of the same cluster "
        "for eviction consideration. None disables eviction",
    )
    eviction_search_limit: int = Field(
        20,
        description="Maximum number of similar vectors to retrieve per derivative for eviction evaluation",
    )
    eviction_target_size: int = Field(
        15,
        description="Target cluster size; eviction starts when the number of similar vectors exceeds this",
    )
    serialize_encode: bool = Field(
        False,
        description="Serialize encode_events calls with an async lock",
    )


class EventMemory:
    """Event memory system."""

    # System-defined metadata field names. Reserved.
    _SEGMENT_UUID_FIELD_NAME = "_segment_uuid"
    _TIMESTAMP_FIELD_NAME = "_timestamp"

    _BASE_EVENT_MEMORY_FIELD_NAMES: ClassVar[frozenset[str]] = frozenset(
        {_SEGMENT_UUID_FIELD_NAME, _TIMESTAMP_FIELD_NAME}
    )

    # Prefix applied to Context field names in the vector store. Reserved.
    _CONTEXT_VECTOR_RECORD_FIELD_PREFIX: ClassVar[str] = "_context_"

    @classmethod
    def _context_vector_record_property_name(cls, field_name: str) -> str:
        """Return the vector record property name for a Context field name."""
        return f"{cls._CONTEXT_VECTOR_RECORD_FIELD_PREFIX}{field_name}"

    @classmethod
    def _required_fields_for_context_type(
        cls,
        context_type: type[ContextUnion],
    ) -> frozenset[str]:
        """Return the storage fields required by a concrete Context type."""
        field_names = set(context_type.model_fields)
        return frozenset(
            cls._context_vector_record_property_name(name)
            for name in INDEXED_CONTEXT_PROPERTIES_SCHEMA
            if name in field_names
        )

    @classmethod
    def expected_vector_store_collection_schema(cls) -> dict[str, type[PropertyValue]]:
        """
        Return the vector store collection schema expected by EventMemory.

        Callers should merge this with any user or external system-defined properties
        when creating the collection so that EventMemory's reserved fields are filterable.
        """
        schema: dict[str, type[PropertyValue]] = {
            cls._SEGMENT_UUID_FIELD_NAME: cast(type[PropertyValue], str),
            cls._TIMESTAMP_FIELD_NAME: cast(type[PropertyValue], datetime.datetime),
        }
        for name, storage_type in INDEXED_CONTEXT_PROPERTIES_SCHEMA.items():
            schema[cls._context_vector_record_property_name(name)] = storage_type
        return schema

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
        missing_context_fields = (
            frozenset(EventMemory.expected_vector_store_collection_schema())
            - self._schema_fields
            - EventMemory._BASE_EVENT_MEMORY_FIELD_NAMES
        )
        if missing_context_fields:
            logger.warning(
                "EventMemory collection schema is missing context fields: "
                "%s. Ingesting events with the corresponding Context "
                "subtypes will fail until the schema is updated.",
                ", ".join(sorted(missing_context_fields)),
            )

        self._embedder = params.embedder
        self._reranker = params.reranker

        self._derive_sentences = params.derive_sentences

        self._eviction_similarity_threshold = params.eviction_similarity_threshold
        self._eviction_search_limit = params.eviction_search_limit
        self._eviction_target_size = params.eviction_target_size

        self._encode_lock: AbstractAsyncContextManager[None] = (
            asyncio.Lock() if params.serialize_encode else nullcontext()
        )

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
        return field in cls._BASE_EVENT_MEMORY_FIELD_NAMES or field.startswith(
            cls._CONTEXT_VECTOR_RECORD_FIELD_PREFIX
        )

    def _validate_events(self, events: Iterable[Event]) -> None:
        """
        Validate a batch of events before encoding.

        Raises ValueError if any event supplies a reserved field name in its properties,
        or if the collection schema is missing fields required by any event's Context type.
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

        required_fields: set[str] = set(EventMemory._BASE_EVENT_MEMORY_FIELD_NAMES)
        for event in events:
            match event.body:
                case Content(context=context) if context is not None:
                    required_fields.update(
                        EventMemory._required_fields_for_context_type(type(context))
                    )
        missing_fields = required_fields - self._schema_fields
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
        events = list(events)
        self._validate_events(events)

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

        batch_predecessors: list[set[int]] | None = None
        if self._eviction_similarity_threshold is not None and derivative_embeddings:
            batch_predecessors = EventMemory._compute_batch_predecessors(
                derivative_embeddings,
                self._eviction_similarity_threshold,
                self._embedder.similarity_metric,
            )

        async with self._encode_lock:
            skipped_uuids: set[UUID] = set()
            db_eviction_uuids: set[UUID] = set()

            if batch_predecessors is not None:
                eviction_query_results = await self._vector_store_collection.query(
                    query_vectors=derivative_embeddings,
                    score_threshold=self._eviction_similarity_threshold,
                    limit=self._eviction_search_limit,
                    return_vector=False,
                    return_properties=True,
                )

                db_eviction_uuids, skipped_uuids = EventMemory._select_eviction_targets(
                    derivatives=derivatives,
                    query_results=eviction_query_results,
                    batch_predecessors=batch_predecessors,
                    eviction_target_size=self._eviction_target_size,
                )

            await self._segment_store_partition.add_segments(
                {
                    segment: [
                        derivative.uuid
                        for derivative in segment_derivatives
                        if derivative.uuid not in skipped_uuids
                    ]
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
                if derivative.uuid not in skipped_uuids
            ]

            if derivative_records:
                await self._vector_store_collection.upsert(records=derivative_records)

            if db_eviction_uuids:
                await self._vector_store_collection.delete(
                    record_uuids=db_eviction_uuids
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

        if derivative.context is not None:
            for name, value in derivative.context.model_dump(exclude_none=True).items():
                properties[cls._context_vector_record_property_name(name)] = value

        # User-defined properties.
        properties.update(derivative.properties)

        return Record(
            uuid=derivative.uuid,
            vector=list(derivative_embedding),
            properties=properties,
        )

    @staticmethod
    def _compute_batch_predecessors(
        derivative_embeddings: Iterable[Sequence[float]],
        eviction_similarity_threshold: float,
        similarity_metric: SimilarityMetric,
    ) -> list[set[int]]:
        """
        Compute batch predecessors for each derivative embedding.

        The ith entry contains indices j < i that are similar to i.
        Only predecessors are included to mimic serial ingestion.
        """
        embeddings = np.asarray(list(derivative_embeddings), dtype=np.float64)
        num_embeddings = len(embeddings)
        higher_is_better = similarity_metric.higher_is_better

        match similarity_metric:
            case SimilarityMetric.COSINE:
                norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
                norms[norms == 0] = 1.0
                normalized = embeddings / norms
                similarity_matrix = normalized @ normalized.T
            case SimilarityMetric.DOT:
                similarity_matrix = embeddings @ embeddings.T
            case SimilarityMetric.EUCLIDEAN:
                norms_squared = np.sum(embeddings**2, axis=1)
                distance_squared = np.add.outer(norms_squared, norms_squared) - 2.0 * (
                    embeddings @ embeddings.T
                )
                similarity_matrix = np.sqrt(np.maximum(distance_squared, 0.0))
            case SimilarityMetric.MANHATTAN:
                # Compute row-by-row to avoid using excessive memory.
                similarity_matrix = np.zeros(
                    (num_embeddings, num_embeddings), dtype=np.float64
                )
                for i in range(1, num_embeddings):
                    diff = embeddings[i] - embeddings[:i]
                    similarity_matrix[i, :i] = np.sum(np.abs(diff), axis=1)

        # Exclude diagonal and upper triangle by setting them to a value
        # that can never pass the threshold comparison.
        upper_indices = np.triu_indices(num_embeddings)
        if higher_is_better:
            similarity_matrix[upper_indices] = -np.inf
            mask = similarity_matrix >= eviction_similarity_threshold
        else:
            similarity_matrix[upper_indices] = np.inf
            mask = similarity_matrix <= eviction_similarity_threshold

        return [set(np.where(mask[i])[0].tolist()) for i in range(num_embeddings)]

    @staticmethod
    def _select_eviction_targets(
        derivatives: Iterable[Derivative],
        query_results: Iterable[VectorStoreQueryResult],
        batch_predecessors: list[set[int]],
        eviction_target_size: int,
    ) -> tuple[set[UUID], set[UUID]]:
        """
        Select eviction targets considering both DB and intra-batch similarity.

        Returns a tuple of:
        - a set of DB UUIDs to delete
        - a set of batch UUIDs to skip
        """
        derivatives = list(derivatives)
        query_results = list(query_results)

        db_eviction_uuids: set[UUID] = set()
        batch_skip_uuids: set[UUID] = set()

        for derivative, query_result, predecessor_indexes in zip(
            derivatives, query_results, batch_predecessors, strict=True
        ):
            # Build cluster members: (timestamp, uuid, is_from_db).
            members: list[tuple[datetime.datetime, UUID, bool]] = []

            # DB cluster members, excluding already-evicted.
            for match in query_result.matches:
                if match.record.uuid in db_eviction_uuids:
                    continue
                timestamp = cast(
                    datetime.datetime,
                    cast(dict[str, PropertyValue], match.record.properties)[
                        EventMemory._TIMESTAMP_FIELD_NAME
                    ],
                )
                members.append((timestamp, match.record.uuid, True))

            # Batch predecessors, excluding already-skipped.
            for index in predecessor_indexes:
                neighbor = derivatives[index]
                if neighbor.uuid not in batch_skip_uuids:
                    members.append((neighbor.timestamp, neighbor.uuid, False))

            # Self.
            members.append((derivative.timestamp, derivative.uuid, False))

            total_size = len(members)
            if total_size <= eviction_target_size:
                continue

            # Cluster is oversized — evict from the temporal middle.
            members.sort()
            keep_early = eviction_target_size // 2
            keep_late = eviction_target_size - keep_early

            for _, uuid, is_from_db in members[keep_early : total_size - keep_late]:
                if is_from_db:
                    db_eviction_uuids.add(uuid)
                else:
                    batch_skip_uuids.add(uuid)

        return db_eviction_uuids, batch_skip_uuids

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

    @classmethod
    def _to_vector_record_property(cls, field: str) -> str:
        """
        Translates canonical filter field name to vector record property.

        Event memory base properties (`foo`) translate to `_foo`..
        Context properties (`context.foo`) translate to `_context_foo`.
        User-defined properties (`m.foo` / `metadata.foo`) translate to `foo`.
        """
        internal_name, is_user_metadata = normalize_filter_field(field)
        if is_user_metadata:
            return demangle_user_metadata_key(internal_name)
        context_prefix = "context."
        if field.startswith(context_prefix):
            subfield = field.removeprefix(context_prefix)
            return cls._context_vector_record_property_name(subfield)
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
                Options for formatting timestamps in output
                (default: None).

        Returns:
            QueryResult:
                The query result.

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
