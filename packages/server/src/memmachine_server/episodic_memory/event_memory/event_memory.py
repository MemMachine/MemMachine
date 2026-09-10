"""Event memory system for storing and retrieving events."""

import asyncio
import datetime
import json
import logging
import time
from collections.abc import Iterable, Mapping, Sequence
from typing import ClassVar, cast
from uuid import UUID

import numpy as np
from pydantic import BaseModel, Field, InstanceOf

from memmachine_server.common.data_types import (
    PROPERTY_TYPE_TO_PROPERTY_TYPE_NAME,
    PropertyType,
    PropertyValue,
)
from memmachine_server.common.embedder import Embedder
from memmachine_server.common.filter import (
    FilterExpr,
    conjoin,
    map_filter_fields,
    split_declared,
)
from memmachine_server.common.filter.filter_parser import (
    demangle_user_metadata_key,
    normalize_filter_field,
)
from memmachine_server.common.metrics_factory import (
    MetricsFactory,
    OperationTracker,
)
from memmachine_server.common.property_keys import validate_caller_property_key
from memmachine_server.common.reranker import Reranker
from memmachine_server.common.vector_store import (
    QueryResult,
    Record,
    VectorStoreCollection,
)

from .data_types import (
    Author,
    ContextPart,
    Derivative,
    Event,
    EvictionOptions,
    FilterOptions,
    FormatOptions,
    Neighborhood,
    SearchHit,
    Segment,
    get_part,
)
from .deriver import Deriver
from .formatting import format_timestamp
from .segment_store import SegmentStorePartition
from .segmenter import Segmenter
from .system_filters import (
    BLOCK_KIND_KEY,
    EVENT_SESSION_KEY,
    EVENT_SOURCE_KEY,
    EVENT_TIMESTAMP_KEY,
    system_predicates,
)

logger = logging.getLogger(__name__)

ID_MAX_BYTES = 255
"""Bound on `Event.session_id` and `Event.source_id`, in bytes.

The width the store's key columns use; an id is an identifier a caller
mints, never content.
"""

# The context part kinds rendering prints, in the order they are printed.
_RENDERED_PART_KINDS: tuple[type[ContextPart], ...] = (Author,)

# Each widening step multiplies the vector fetch by this.
_OVERFETCH_BASE = 4


class InvalidCollectionSchemaError(ValueError):
    """Raised when a vector store collection does not declare EventMemory's system keys."""

    def __init__(
        self,
        missing: Mapping[str, PropertyType],
        declared: Mapping[str, PropertyType],
    ) -> None:
        """Initialize with the keys the collection lacks and the ones it declares."""
        self.missing = dict(missing)
        self.declared = dict(declared)
        super().__init__(
            "The vector store collection does not declare the system keys EventMemory "
            f"writes: missing {_schema_names(missing)}, declared {_schema_names(declared)}."
        )


def _schema_names(schema: Mapping[str, PropertyType]) -> dict[str, str]:
    return {
        key: PROPERTY_TYPE_TO_PROPERTY_TYPE_NAME[property_type]
        for key, property_type in sorted(schema.items())
    }


class EventMemoryParams(BaseModel):
    """
    Parameters for EventMemory.

    Attributes:
        segment_store_partition (SegmentStorePartition):
            Segment store partition.
        vector_store_collection (VectorStoreCollection):
            Vector store collection.
        segmenter (Segmenter):
            Segmenter that segments events into segments.
        deriver (Deriver):
            Deriver that derives derivatives from segments.
        embedder (Embedder):
            Embedder instance for creating embeddings.
        format_options (FormatOptions):
            How the deriver formats a segment's timestamp and author into
            the text it embeds. Fixed per memory because a memory's
            derivatives must be formatted one way; a display format is a
            call argument (default: a full date and no time).
        eviction (EvictionOptions | None):
            Trim clusters of near-duplicate derivatives at ingest. None
            keeps every derivative and issues no eviction query
            (default: None).
        filter (FilterOptions):
            How far a search widens to make up for seeds the segment store's
            post-filter drops (default: `FilterOptions()`).
        metrics_factory (MetricsFactory | None):
            An instance of MetricsFactory for collecting usage metrics
            (default: None).
    """

    segment_store_partition: InstanceOf[SegmentStorePartition] = Field(
        ...,
        description="Segment store partition",
    )
    vector_store_collection: InstanceOf[VectorStoreCollection] = Field(
        ...,
        description="Vector store collection",
    )
    segmenter: InstanceOf[Segmenter] = Field(
        ...,
        description="Segmenter that segments events into segments",
    )
    deriver: InstanceOf[Deriver] = Field(
        ...,
        description="Deriver that derives derivatives from segments",
    )
    embedder: InstanceOf[Embedder] = Field(
        ...,
        description="Embedder instance for creating embeddings",
    )
    format_options: FormatOptions = Field(
        default_factory=lambda: FormatOptions(time_style=None),
        description="How the deriver formats the text it embeds",
    )
    eviction: EvictionOptions | None = Field(
        None,
        description="Trim clusters of near-duplicate derivatives at ingest",
    )
    filter: FilterOptions = Field(
        default_factory=FilterOptions,
        description="How far a search widens over the segment store's post-filter",
    )
    metrics_factory: InstanceOf[MetricsFactory] | None = Field(
        None,
        description="An instance of MetricsFactory for collecting usage metrics",
    )


class EventMemory:
    """Event memory system."""

    # Every system value written into a vector record, under the reserved
    # keys `system_filters` owns, with the type the collection declares:
    # the fields a search filters on at the vector stage, and nothing
    # else. The derivative's segment and event are not among them: the
    # segment store owns those mappings, and a copy here could only go
    # stale.
    _RESERVED_PROPERTY_SCHEMA: ClassVar[dict[str, type[PropertyValue]]] = {
        EVENT_TIMESTAMP_KEY: cast(type[PropertyValue], datetime.datetime),
        EVENT_SESSION_KEY: cast(type[PropertyValue], str),
        EVENT_SOURCE_KEY: cast(type[PropertyValue], str),
        BLOCK_KIND_KEY: cast(type[PropertyValue], str),
    }

    @classmethod
    def expected_vector_store_collection_schema(cls) -> dict[str, type[PropertyValue]]:
        """
        Return the vector store collection schema expected by EventMemory.

        Callers should merge this with any user or external system-defined properties
        when creating the collection so that EventMemory's reserved fields are efficiently filterable.
        """
        return dict(cls._RESERVED_PROPERTY_SCHEMA)

    def __init__(self, params: EventMemoryParams) -> None:
        """
        Initialize an EventMemory with the provided parameters.

        Args:
            params (EventMemoryParams):
                Parameters for the EventMemory.

        """
        self._segment_store_partition = params.segment_store_partition
        self._vector_store_collection = params.vector_store_collection
        self._segmenter = params.segmenter
        self._deriver = params.deriver
        self._embedder = params.embedder
        self._format_options = params.format_options
        self._eviction = params.eviction
        self._filter = params.filter

        self._tracker = OperationTracker(
            params.metrics_factory,
            prefix="event_memory",
        )

        # The store declares what it indexes; the system keys must be among
        # them, with the types this memory writes.
        declared = dict(params.vector_store_collection.indexed_properties)
        missing = {
            key: property_type
            for key, property_type in EventMemory._RESERVED_PROPERTY_SCHEMA.items()
            if declared.get(key) is not property_type
        }
        if missing:
            raise InvalidCollectionSchemaError(missing, declared)
        self._declared_properties = declared

        self._encode_events_phase_seconds: MetricsFactory.Histogram | None = None
        self._query_phase_seconds: MetricsFactory.Histogram | None = None
        if params.metrics_factory is not None:
            self._encode_events_phase_seconds = params.metrics_factory.get_histogram(
                "event_memory_encode_events_phase_seconds",
                "Time spent in each phase of encode_events",
                label_names=("phase",),
            )
            self._query_phase_seconds = params.metrics_factory.get_histogram(
                "event_memory_query_phase_seconds",
                "Time spent in each phase of query",
                label_names=("phase",),
            )

    def _validate_events(self, events: Iterable[Event]) -> None:
        """
        Validate a batch of events before encoding.

        Raises ValueError if any event supplies a property key in the
        reserved namespace or outside the naming contract, a value of another
        type than the vector store declares for its key, or a session or
        source id longer than `ID_MAX_BYTES`.
        """
        for event in events:
            for key, value in event.properties.items():
                validate_caller_property_key(key)
                declared = self._declared_properties.get(key)
                if declared is not None and type(value) is not declared:
                    raise ValueError(
                        f"Event {event.uuid} property {key!r} is a "
                        f"{type(value).__name__}; the vector store declares it as "
                        f"{PROPERTY_TYPE_TO_PROPERTY_TYPE_NAME[declared]}."
                    )
            for name, value in (
                ("session_id", event.session_id),
                ("source_id", event.source_id),
            ):
                if value is not None and len(value.encode()) > ID_MAX_BYTES:
                    raise ValueError(
                        f"Event {event.uuid} {name} is {len(value.encode())} bytes; "
                        f"the maximum is {ID_MAX_BYTES}."
                    )

    async def encode_events(self, events: Iterable[Event]) -> None:
        """
        Encode events.

        A batch is written whole: each event's earlier encoding is forgotten
        first, so a repeated batch leaves one copy.

        Args:
            events (Iterable[Event]): The events to encode.

        Raises:
            ValueError:
                If any event supplies a reserved or illegal property key, or
                a session or source id over `ID_MAX_BYTES`.
        """
        async with self._tracker("encode_events"):
            await self._encode_events(events)

    async def _encode_events(self, events: Iterable[Event]) -> None:
        t_start = time.monotonic()

        events = list(events)
        self._validate_events(events)
        if not events:
            return

        await self._forget_events({event.uuid for event in events})

        # Temporal order within the batch, so that eviction among the
        # batch's own derivatives is what serial ingestion would decide.
        events = sorted(events, key=lambda event: (event.timestamp, event.uuid))

        segment_lists = await asyncio.gather(
            *(
                self._segmenter.segment(event, format_options=self._format_options)
                for event in events
            )
        )
        segments = [
            segment for segment_list in segment_lists for segment in segment_list
        ]
        t_segmentation = time.monotonic()

        derivative_lists = await asyncio.gather(
            *(
                self._deriver.derive(segment, format_options=self._format_options)
                for segment in segments
            )
        )
        segments_to_derivatives: dict[Segment, list[Derivative]] = dict(
            zip(segments, derivative_lists, strict=True)
        )

        derivatives = [
            derivative
            for segment_derivatives in segments_to_derivatives.values()
            for derivative in segment_derivatives
        ]
        t_derivation = time.monotonic()

        derivative_texts: list[str] = []
        for derivative in derivatives:
            text = derivative.block.render(self._format_options)
            if text is None:
                raise NotImplementedError(
                    f"Cannot embed a derivative of block kind {derivative.block.kind!r}"
                )
            derivative_texts.append(text)

        derivative_embeddings = await self._embedder.ingest_embed(derivative_texts)
        t_embedding = time.monotonic()

        displaced_uuids: set[UUID] = set()
        skipped_uuids: set[UUID] = set()
        if self._eviction is not None and derivatives:
            batch_predecessors = EventMemory._compute_batch_predecessors(
                derivative_embeddings,
                self._eviction.similarity_threshold,
            )
            stored_neighbors = await self._vector_store_collection.query(
                query_vectors=derivative_embeddings,
                min_cosine_similarity=self._eviction.similarity_threshold,
                limit=self._eviction.search_limit,
            )
            stored_timestamps = await self._stored_derivative_timestamps(
                match.record_uuid
                for query_result in stored_neighbors
                for match in query_result.matches
            )
            displaced_uuids, skipped_uuids = EventMemory._select_eviction_targets(
                derivatives,
                stored_neighbors,
                batch_predecessors,
                stored_timestamps,
                self._eviction.target_size,
            )
        t_eviction = time.monotonic()

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
        t_segment_store = time.monotonic()

        embeddings_by_derivative = dict(
            zip(
                (derivative.uuid for derivative in derivatives),
                derivative_embeddings,
                strict=True,
            )
        )
        derivative_records = [
            self._build_derivative_record(
                derivative, embeddings_by_derivative[derivative.uuid]
            )
            for derivative in derivatives
            if derivative.uuid not in skipped_uuids
        ]
        if derivative_records:
            await self._vector_store_collection.upsert(records=derivative_records)
        if displaced_uuids:
            await self._vector_store_collection.delete(record_uuids=displaced_uuids)
            await self._segment_store_partition.delete_derivatives(displaced_uuids)
        t_vector_store = time.monotonic()

        phase_durations = {
            "segmentation": t_segmentation - t_start,
            "derivation": t_derivation - t_segmentation,
            "embedding": t_embedding - t_derivation,
            "eviction": t_eviction - t_embedding,
            "segment_store": t_segment_store - t_eviction,
            "vector_store": t_vector_store - t_segment_store,
        }

        logger.debug(
            "encode_events timing: %s total=%.3fs",
            " ".join(
                f"{phase}={duration:.3f}s"
                for phase, duration in phase_durations.items()
            ),
            t_vector_store - t_start,
        )

        if self._encode_events_phase_seconds is not None:
            for phase, duration in phase_durations.items():
                self._encode_events_phase_seconds.observe(
                    duration, labels={"phase": phase}
                )

    async def _stored_derivative_timestamps(
        self, derivative_uuids: Iterable[UUID]
    ) -> dict[UUID, datetime.datetime]:
        """The event timestamp of each stored derivative, read from its segment.

        The vector store answers uuids and scores only; the segment store
        owns the derivative's segment, and the segment its timestamp. A
        derivative whose segment is gone is omitted.
        """
        segment_by_derivative = (
            await self._segment_store_partition.get_segment_uuids_by_derivative_uuids(
                derivative_uuids
            )
        )
        if not segment_by_derivative:
            return {}
        segments_by_uuid = await self._segment_store_partition.get_segment_contexts(
            seed_segment_uuids=set(segment_by_derivative.values())
        )
        return {
            derivative_uuid: segments_by_uuid[segment_uuid][0].timestamp
            for derivative_uuid, segment_uuid in segment_by_derivative.items()
            if segment_uuid in segments_by_uuid
        }

    def _build_derivative_record(
        self,
        derivative: Derivative,
        derivative_embedding: Sequence[float],
    ) -> Record:
        """Build a vector record from a derivative and its embedding.

        Only the caller properties the store declares go in: an undeclared
        key never exists in the vector store, and the segment store holds
        every property for the filtering the vector store does not do.
        """
        # Caller properties first: a reserved key cannot reach here
        # (encode_events validates), and building in this order makes the
        # merge itself enforce that a system value always wins.
        properties: dict[str, PropertyValue] = {
            key: value
            for key, value in derivative.properties.items()
            if key in self._declared_properties
        }
        properties[EVENT_TIMESTAMP_KEY] = derivative.timestamp
        if derivative.session_id is not None:
            properties[EVENT_SESSION_KEY] = derivative.session_id
        if derivative.source_id is not None:
            properties[EVENT_SOURCE_KEY] = derivative.source_id
        properties[BLOCK_KIND_KEY] = derivative.block.kind

        return Record(
            uuid=derivative.uuid,
            vector=list(derivative_embedding),
            properties=properties,
        )

    @staticmethod
    def _compute_batch_predecessors(
        derivative_embeddings: Iterable[Sequence[float]],
        similarity_threshold: float,
    ) -> list[set[int]]:
        """
        Compute batch predecessors for each derivative embedding.

        The ith entry holds the indices j < i whose cosine similarity to i
        is at or above the threshold. Only earlier indices count, so a
        batch evicts exactly what serial ingestion would.
        """
        embeddings = np.asarray(list(derivative_embeddings), dtype=np.float64)
        num_embeddings = len(embeddings)
        if num_embeddings == 0:
            return []

        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        normalized = embeddings / norms
        similarity_matrix = normalized @ normalized.T

        # The diagonal and the upper triangle can never pass the threshold.
        similarity_matrix[np.triu_indices(num_embeddings)] = -np.inf
        mask = similarity_matrix >= similarity_threshold

        return [set(np.where(mask[i])[0].tolist()) for i in range(num_embeddings)]

    @staticmethod
    def _select_eviction_targets(
        derivatives: Iterable[Derivative],
        query_results: Iterable[QueryResult],
        batch_predecessors: list[set[int]],
        stored_timestamps: Mapping[UUID, datetime.datetime],
        target_size: int,
    ) -> tuple[set[UUID], set[UUID]]:
        """
        Select eviction targets considering both stored and batch similarity.

        The cluster of a derivative is its stored neighbors not already
        displaced in this batch, its batch predecessors not already skipped,
        and itself. Within `target_size` nothing happens; over it, the
        cluster is sorted by event timestamp and the earliest
        `target_size // 2` and the latest remainder are kept: the middle's
        stored members are displaced, its batch members skipped. A stored
        neighbor with no timestamp in `stored_timestamps` (its segment is
        gone) is not a member.

        Returns a tuple of:
        - the stored derivative UUIDs to displace
        - the batch derivative UUIDs to skip
        """
        derivatives = list(derivatives)
        query_results = list(query_results)

        displaced_uuids: set[UUID] = set()
        skipped_uuids: set[UUID] = set()

        for derivative, query_result, predecessor_indexes in zip(
            derivatives, query_results, batch_predecessors, strict=True
        ):
            # Cluster members: (timestamp, uuid, is_stored).
            members: list[tuple[datetime.datetime, UUID, bool]] = []

            for match in query_result.matches:
                if match.record_uuid in displaced_uuids:
                    continue
                timestamp = stored_timestamps.get(match.record_uuid)
                if timestamp is None:
                    continue
                members.append((timestamp, match.record_uuid, True))

            for index in predecessor_indexes:
                neighbor = derivatives[index]
                if neighbor.uuid not in skipped_uuids:
                    members.append((neighbor.timestamp, neighbor.uuid, False))

            members.append((derivative.timestamp, derivative.uuid, False))

            total_size = len(members)
            if total_size <= target_size:
                continue

            # Oversized: trim the temporal middle.
            members.sort()
            keep_early = target_size // 2
            keep_late = target_size - keep_early

            for _, uuid, is_stored in members[keep_early : total_size - keep_late]:
                if is_stored:
                    displaced_uuids.add(uuid)
                else:
                    skipped_uuids.add(uuid)

        return displaced_uuids, skipped_uuids

    @staticmethod
    def _to_vector_record_property(field: str) -> str:
        """
        Translate a caller's filter field name to a vector record property.

        User-defined properties (`m.foo` / `metadata.foo`) translate to `foo`.
        `timestamp` is the event timestamp, under its reserved key. Any
        other bare name (`foo`) translates to `_foo`, the layout the server
        writes its own fields in.
        """
        internal_name, is_user_metadata = normalize_filter_field(field)
        if is_user_metadata:
            return demangle_user_metadata_key(internal_name)
        if field == "timestamp":
            return EVENT_TIMESTAMP_KEY
        return f"_{field}"

    async def query(
        self,
        query: str,
        *,
        limit: int = 20,
        min_cosine_similarity: float | None = None,
        expand_context: int = 0,
        since: datetime.datetime | None = None,
        until: datetime.datetime | None = None,
        session_ids: Iterable[str] | None = None,
        source_ids: Iterable[str] | None = None,
        block_kinds: Iterable[str] | None = None,
        property_filter: FilterExpr | None = None,
    ) -> list[SearchHit]:
        """
        Query event memory for segments relevant to the query.

        The vector stage: `rerank` is the second stage, for a caller that
        has a reranker.

        Args:
            query (str):
                The search query.
            limit (int):
                The maximum number of hits (default: 20).
            min_cosine_similarity (float | None):
                Drop matches whose cosine similarity is below this
                (default: None).
            expand_context (int):
                The number of additional segments to include
                around each matched segment for additional context
                (default: 0).
            since (datetime | None):
                Inclusive lower bound on the event timestamp (default: None).
            until (datetime | None):
                Exclusive upper bound on the event timestamp (default: None).
            session_ids (Iterable[str] | None):
                Keep only events of these sessions (default: None).
            source_ids (Iterable[str] | None):
                Keep only events of these sources (default: None).
            block_kinds (Iterable[str] | None):
                Keep only segments whose block is of these kinds (default: None).
            property_filter (FilterExpr | None):
                Property fields and values
                to use for filtering segments
                (default: None).

        Returns:
            list[SearchHit]:
                At most `limit` hits in descending similarity, each with
                its context window and the index of the matched segment in
                it. Windows of different hits may overlap; each hit is
                returned whole. Every count is a maximum.

        """
        async with self._tracker("query"):
            return await self._query(
                query,
                limit=limit,
                min_cosine_similarity=min_cosine_similarity,
                expand_context=expand_context,
                since=since,
                until=until,
                session_ids=session_ids,
                source_ids=source_ids,
                block_kinds=block_kinds,
                property_filter=property_filter,
            )

    async def _query(
        self,
        query: str,
        *,
        limit: int,
        min_cosine_similarity: float | None,
        expand_context: int,
        since: datetime.datetime | None,
        until: datetime.datetime | None,
        session_ids: Iterable[str] | None,
        source_ids: Iterable[str] | None,
        block_kinds: Iterable[str] | None,
        property_filter: FilterExpr | None,
    ) -> list[SearchHit]:
        t_start = time.monotonic()
        session_ids = list(session_ids) if session_ids is not None else None
        source_ids = list(source_ids) if source_ids is not None else None
        block_kinds = list(block_kinds) if block_kinds is not None else None
        # An empty id or kind list admits nothing, and needs no query to say so.
        if any(
            ids is not None and not ids
            for ids in (session_ids, source_ids, block_kinds)
        ):
            return []

        query_embedding = (
            await self._embedder.search_embed(
                [query],
            )
        )[0]
        t_embedding = time.monotonic()

        # One plan. The vector store gets the system predicates and the part
        # of the caller's filter naming keys it declares, evaluated during
        # the search; the rest is the segment store's, applied to the seeds
        # afterward, and the search is widened to make up for what it drops.
        declared_part, undeclared_part = split_declared(
            map_filter_fields(property_filter, EventMemory._to_vector_record_property)
            if property_filter is not None
            else None,
            self._declared_properties,
        )
        collection_filter = conjoin(
            [
                system_predicates(
                    since=since,
                    until=until,
                    session_ids=session_ids,
                    source_ids=source_ids,
                    block_kinds=block_kinds,
                ),
                declared_part,
            ]
        )

        before = expand_context // 3
        after = expand_context - before

        max_fetch = limit * self._filter.max_overfetch_factor
        fetch_limit = limit
        vector_query_seconds = 0.0
        segment_query_seconds = 0.0
        while True:
            t_vector_start = time.monotonic()
            [query_result] = await self._vector_store_collection.query(
                query_vectors=[query_embedding],
                limit=fetch_limit,
                min_cosine_similarity=min_cosine_similarity,
                property_filter=collection_filter,
            )
            t_vector_query = time.monotonic()
            vector_query_seconds += t_vector_query - t_vector_start

            hits = await self._hits(
                query_result,
                before=before,
                after=after,
                since=since,
                until=until,
                source_ids=source_ids,
                block_kinds=block_kinds,
                property_filter=property_filter,
            )
            segment_query_seconds += time.monotonic() - t_vector_query

            exhausted = len(query_result.matches) < fetch_limit
            if (
                undeclared_part is None
                or len(hits) >= limit
                or exhausted
                or fetch_limit >= max_fetch
            ):
                break
            fetch_limit = min(fetch_limit * _OVERFETCH_BASE, max_fetch)

        phase_durations = {
            "embedding": t_embedding - t_start,
            "vector_query": vector_query_seconds,
            "segment_query": segment_query_seconds,
        }

        logger.debug(
            "query timing: %s total=%.3fs",
            " ".join(
                f"{phase}={duration:.3f}s"
                for phase, duration in phase_durations.items()
            ),
            time.monotonic() - t_start,
        )

        if self._query_phase_seconds is not None:
            for phase, duration in phase_durations.items():
                self._query_phase_seconds.observe(duration, labels={"phase": phase})

        return hits[:limit]

    async def _hits(
        self,
        query_result: QueryResult,
        *,
        before: int,
        after: int,
        since: datetime.datetime | None,
        until: datetime.datetime | None,
        source_ids: list[str] | None,
        block_kinds: list[str] | None,
        property_filter: FilterExpr | None,
    ) -> list[SearchHit]:
        """The hits of one vector query: each seed's window, in similarity order.

        The segment store applies the whole caller filter to the seed and its
        window, so a neighbor satisfies what the seed satisfies; a seed the
        store does not return is dropped.
        """
        segment_by_derivative = (
            await self._segment_store_partition.get_segment_uuids_by_derivative_uuids(
                match.record_uuid for match in query_result.matches
            )
        )

        # Deduplicate by first occurrence (multiple derivatives can map to the same segment).
        # First occurrence has the best score since matches are ordered best-to-worst.
        seed_cosine_similarities: dict[UUID, float] = {}
        for match in query_result.matches:
            segment_uuid = segment_by_derivative.get(match.record_uuid)
            if segment_uuid is None:
                # The derivative's segment is gone; its vector outlived it.
                continue
            if segment_uuid not in seed_cosine_similarities:
                seed_cosine_similarities[segment_uuid] = match.cosine_similarity

        segment_contexts_by_seed = (
            await self._segment_store_partition.get_segment_contexts(
                seed_segment_uuids=seed_cosine_similarities.keys(),
                before=before,
                after=after,
                since=since,
                until=until,
                source_ids=source_ids,
                block_kinds=block_kinds,
                property_filter=property_filter,
            )
        )

        hits: list[SearchHit] = []
        for seed_uuid, score in seed_cosine_similarities.items():
            segments = segment_contexts_by_seed.get(seed_uuid)
            if segments is None:
                continue
            seed_index = next(
                index
                for index, segment in enumerate(segments)
                if segment.uuid == seed_uuid
            )
            hits.append(SearchHit(score=score, seed=seed_index, segments=segments))
        return hits

    async def expand(
        self,
        anchor: UUID,
        *,
        before: int = 0,
        after: int = 0,
        since: datetime.datetime | None = None,
        until: datetime.datetime | None = None,
        source_ids: Iterable[str] | None = None,
        block_kinds: Iterable[str] | None = None,
        property_filter: FilterExpr | None = None,
    ) -> Neighborhood:
        """
        Get the neighborhood of an anchor in its session's order.

        The anchor is a segment uuid (from a hit) or an event uuid (its
        first segment). The filters apply to the neighbors only, and the
        anchor is never returned: its place is between the two lists. A
        caller walks further by calling again from the first of `before`
        or the last of `after` with one side zero.

        Args:
            anchor (UUID):
                A segment or event uuid.
            before (int):
                The maximum number of segments before the anchor (default: 0).
            after (int):
                The maximum number of segments after the anchor (default: 0).
            since (datetime | None):
                Inclusive lower bound on the neighbors' timestamp (default: None).
            until (datetime | None):
                Exclusive upper bound on the neighbors' timestamp (default: None).
            source_ids (Iterable[str] | None):
                Keep only neighbors of these sources (default: None).
            block_kinds (Iterable[str] | None):
                Keep only neighbors whose block is of these kinds (default: None).
            property_filter (FilterExpr | None):
                Property fields and values to filter the neighbors by
                (default: None).

        Returns:
            Neighborhood:
                The neighbors before and after the anchor, in the store's order.

        Raises:
            LookupError: If the anchor is neither a segment nor an event of this memory.
        """
        async with self._tracker("expand"):
            segment_uuids_by_event = (
                await self._segment_store_partition.get_segment_uuids_by_event_uuids(
                    event_uuids=[anchor],
                )
            )
            event_segment_uuids = segment_uuids_by_event.get(anchor)
            seed = event_segment_uuids[0] if event_segment_uuids else anchor

            neighborhoods = await self._segment_store_partition.get_neighbourhoods(
                seed_segment_uuids=[seed],
                before=before,
                after=after,
                since=since,
                until=until,
                source_ids=source_ids,
                block_kinds=block_kinds,
                property_filter=property_filter,
            )
            neighborhood = neighborhoods.get(seed)
            if neighborhood is None:
                raise LookupError(
                    f"Anchor {anchor} is neither a segment nor an event of this memory"
                )
            return neighborhood

    @staticmethod
    async def rerank(
        query: str,
        hits: Sequence[SearchHit],
        *,
        reranker: Reranker,
        limit: int,
        min_score: float | None = None,
        format_options: FormatOptions,
    ) -> list[SearchHit]:
        """
        Rerank hits by a reranker's score of their rendered windows.

        The second stage after `query`, for a caller that has a reranker:
        each hit's window is rendered with `format_options` and scored
        against the query; hits below `min_score` are dropped and at most
        `limit` are returned in descending score, each with its score
        replaced by the reranker's.
        """
        hits = list(hits)
        if not hits:
            return []
        scores = await reranker.score(
            query,
            [
                EventMemory.render(hit.segments, format_options=format_options)
                for hit in hits
            ],
        )
        reranked = [
            SearchHit(score=score, seed=hit.seed, segments=hit.segments)
            for hit, score in zip(hits, scores, strict=True)
            if min_score is None or score >= min_score
        ]
        reranked.sort(key=lambda hit: hit.score, reverse=True)
        return reranked[:limit]

    @staticmethod
    def _immediately_follows(previous: Segment, segment: Segment) -> bool:
        """Whether `segment` is the very next piece of the same event as `previous`.

        Two pieces are adjacent either within a block (the next chunk of it)
        or across blocks (the first chunk of the next one). Anything else
        means something between them is not being shown. A block's chunk
        count is not known here, so the end of one block is recognized by
        the next block starting at its own beginning rather than by
        counting up to it.
        """
        if segment.event_uuid != previous.event_uuid:
            return False
        if segment.index == previous.index:
            return segment.offset == previous.offset + 1
        if segment.index == previous.index + 1:
            return segment.offset == 0
        return False

    @staticmethod
    def render(
        segments: Iterable[Segment],
        *,
        format_options: FormatOptions,
    ) -> str:
        """
        The reader's text for a run of segments, in their order.

        A header (the timestamp formatted by `format_options`, then the
        context parts' contributions) starts each run of adjacent pieces
        of one event; the pieces' block renderings are joined under it.
        """
        context_string = ""
        previous: Segment | None = None
        accumulated_text = ""

        for segment in segments:
            is_continuation = previous is not None and EventMemory._immediately_follows(
                previous, segment
            )

            if not is_continuation:
                if previous is not None:
                    context_string += (
                        json.dumps(accumulated_text, ensure_ascii=False) + "\n"
                    )
                accumulated_text = ""
                context_string += EventMemory._segment_header(segment, format_options)

            text = segment.block.render(format_options)
            if text is not None:
                accumulated_text += text
            elif not is_continuation:
                context_string += f"[{segment.block.kind}]\n"

            previous = segment

        if previous is not None:
            context_string += json.dumps(accumulated_text, ensure_ascii=False) + "\n"

        return context_string.strip()

    @staticmethod
    def _segment_header(segment: Segment, format_options: FormatOptions) -> str:
        """Build the header emitted before a segment."""
        formatted_timestamp = format_timestamp(segment.timestamp, format_options)
        header = f"[{formatted_timestamp}] " if formatted_timestamp else ""
        for part_type in _RENDERED_PART_KINDS:
            part = get_part(segment.context, part_type)
            if part is None:
                continue
            contribution = part.render(format_options)
            if contribution:
                header += f"{contribution}: "
        return header

    async def forget_events(self, event_uuids: Iterable[UUID]) -> None:
        """Forget events by their UUIDs."""
        event_uuids = set(event_uuids)
        if not event_uuids:
            return

        async with self._tracker("forget_events"):
            await self._forget_events(event_uuids)

    async def _forget_events(self, event_uuids: set[UUID]) -> None:
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
