"""Event memory system for storing and retrieving events."""

import asyncio
import datetime
import json
import logging
import time
from collections.abc import Iterable, Mapping, Sequence
from typing import ClassVar, Final, cast
from uuid import UUID

import numpy as np
from pydantic import BaseModel, Field, InstanceOf

from memmachine_server.common.data_types import PropertyValue
from memmachine_server.common.embedder import Embedder
from memmachine_server.common.filter.filter_parser import (
    And,
    Comparison,
    FilterExpr,
    In,
)
from memmachine_server.common.metrics_factory import (
    MetricsFactory,
    OperationTracker,
)
from memmachine_server.common.property_keys import (
    reserved_property_key,
    validate_user_property_key,
)
from memmachine_server.common.reranker import Reranker
from memmachine_server.common.vector_store import (
    QueryResult,
    Record,
    VectorStoreCollection,
)

from .data_types import (
    Block,
    DateTimeFormat,
    Derivative,
    Event,
    EvictionOptions,
    Neighborhood,
    NullContext,
    ProducerContext,
    QueryHit,
    Segment,
    TextBlock,
)
from .deriver import Deriver
from .event_memory_store import EventMemoryStorePartition
from .formatting import format_timestamp
from .segmenter import Segmenter

logger = logging.getLogger(__name__)

# The keys the memory writes into a vector record; `em` names this memory,
# short because the keys share the identifier budget.
EVENT_TIMESTAMP_KEY: Final[str] = reserved_property_key("em", "timestamp")
EVENT_SESSION_KEY: Final[str] = reserved_property_key("em", "session")
EVENT_SOURCE_KEY: Final[str] = reserved_property_key("em", "source")
BLOCK_KIND_KEY: Final[str] = reserved_property_key("em", "block_kind")


def _conjoin(clauses: Iterable[FilterExpr | None]) -> FilterExpr | None:
    """The conjunction of the given clauses; None when there are none."""
    combined: FilterExpr | None = None
    for clause in clauses:
        if clause is None:
            continue
        combined = clause if combined is None else And(left=combined, right=clause)
    return combined


def _system_predicates(
    *,
    since: datetime.datetime | None = None,
    until: datetime.datetime | None = None,
    session_ids: Iterable[str] | None = None,
    source_ids: Iterable[str] | None = None,
    block_kinds: Iterable[str] | None = None,
) -> FilterExpr | None:
    """The predicates on reserved keys that a vector store evaluates.

    `since` is inclusive and `until` exclusive, so ranges meet without
    overlap. A list admits its members and nothing else, so an empty list
    admits nothing; a list left `None` admits everything.
    """
    clauses: list[FilterExpr | None] = [
        Comparison(field=EVENT_TIMESTAMP_KEY, op=">=", value=since)
        if since is not None
        else None,
        Comparison(field=EVENT_TIMESTAMP_KEY, op="<", value=until)
        if until is not None
        else None,
        In(field=EVENT_SESSION_KEY, values=list(session_ids))
        if session_ids is not None
        else None,
        In(field=EVENT_SOURCE_KEY, values=list(source_ids))
        if source_ids is not None
        else None,
        In(field=BLOCK_KIND_KEY, values=list(block_kinds))
        if block_kinds is not None
        else None,
    ]
    return _conjoin(clauses)


class EventMemoryParams(BaseModel):
    """
    Parameters for EventMemory.

    Attributes:
        event_memory_store_partition (EventMemoryStorePartition):
            Event memory store partition.
        vector_store_collection (VectorStoreCollection):
            Vector store collection.
        segmenter (Segmenter):
            Segmenter that segments events into segments.
        deriver (Deriver):
            Deriver that derives derivatives from segments.
        embedder (Embedder):
            Embedder instance for creating embeddings.
        eviction (EvictionOptions | None):
            Evict stored derivatives a new one nearly duplicates, at
            ingest: their links go with the encode's own write, their
            records once it has committed. None keeps every derivative
            and issues no eviction query (default: None).
        metrics_factory (MetricsFactory | None):
            An instance of MetricsFactory for collecting usage metrics
            (default: None).
    """

    event_memory_store_partition: InstanceOf[EventMemoryStorePartition] = Field(
        ...,
        description="Event memory store partition",
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
    eviction: EvictionOptions | None = Field(
        None,
        description="Evict stored derivatives a new one nearly duplicates, at ingest",
    )
    metrics_factory: InstanceOf[MetricsFactory] | None = Field(
        None,
        description="An instance of MetricsFactory for collecting usage metrics",
    )


class EventMemory:
    """Event memory: encodes events into segments and derivatives, and searches them.

    Stored data is immutable: no operation edits a stored segment or
    vector record, and none may be added. A change is `forget_events`
    and `encode_events` again.
    """

    # Every property a vector record carries, with the type the collection
    # declares: the fields a search filters on at the vector stage. User
    # properties stay in the event memory store.
    _RESERVED_PROPERTY_SCHEMA: ClassVar[dict[str, type[PropertyValue]]] = {
        EVENT_TIMESTAMP_KEY: cast(type[PropertyValue], datetime.datetime),
        EVENT_SESSION_KEY: cast(type[PropertyValue], str),
        EVENT_SOURCE_KEY: cast(type[PropertyValue], str),
        BLOCK_KIND_KEY: cast(type[PropertyValue], str),
    }

    @classmethod
    def expected_vector_store_collection_schema(cls) -> dict[str, type[PropertyValue]]:
        """
        Return the vector store collection schema EventMemory requires.

        Every key the memory writes into a vector record, with its type; a
        collection must declare each of them.
        """
        return dict(cls._RESERVED_PROPERTY_SCHEMA)

    def __init__(self, params: EventMemoryParams) -> None:
        """
        Initialize an EventMemory with the provided parameters.

        Args:
            params (EventMemoryParams):
                Parameters for the EventMemory.

        """
        self._event_memory_store_partition = params.event_memory_store_partition
        self._vector_store_collection = params.vector_store_collection
        self._segmenter = params.segmenter
        self._deriver = params.deriver
        self._embedder = params.embedder
        self._eviction = params.eviction

        self._tracker = OperationTracker(
            params.metrics_factory,
            prefix="event_memory",
        )

        declared_fields = frozenset(
            params.vector_store_collection.config.indexed_properties_schema
        )
        missing_fields = EventMemory._RESERVED_PROPERTY_SCHEMA.keys() - declared_fields
        if missing_fields:
            raise ValueError(
                f"Collection schema missing fields required by EventMemory: "
                f"{', '.join(sorted(missing_fields))}"
            )

        self._encode_events_phase_seconds: MetricsFactory.Histogram | None = None
        self._query_phase_seconds: MetricsFactory.Histogram | None = None
        self._orphan_records_deleted: MetricsFactory.Counter | None = None
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
            self._orphan_records_deleted = params.metrics_factory.get_counter(
                "event_memory_orphan_records_deleted_total",
                "Vector records a query deleted because their segment was gone",
            )

    def _validate_events(self, events: Iterable[Event]) -> None:
        """
        Validate a batch of events before encoding.

        Raises ValueError if any event supplies a property key in the
        reserved namespace or outside the naming contract.
        """
        for event in events:
            for key in event.properties:
                validate_user_property_key(key)

    async def encode_events(self, events: Iterable[Event]) -> None:
        """
        Encode events.

        An event is encoded once: a batch naming an event the memory
        already holds is rejected whole, and nothing is stored. Forget
        the event to encode it again.

        With eviction configured, a stored derivative a new one
        displaces loses its link in the same transaction that adds the
        new links, and its record leaves the vector store once that
        transaction has committed: a record no link names is reclaimed
        when a search returns it, while a link naming a record that is
        gone is not. The displaced derivative's segment stays stored.

        Args:
            events (Iterable[Event]): The events to encode.

        Raises:
            ValueError:
                If any event supplies a reserved or illegal property key.
            EventMemoryStoreEventAlreadyStoredError:
                If the memory already holds any of the events.
        """
        async with self._tracker("encode_events"):
            await self._encode_events(events)

    async def _encode_events(self, events: Iterable[Event]) -> None:
        t_start = time.monotonic()

        events = list(events)
        self._validate_events(events)
        if not events:
            return

        # Temporal order within the batch, so that eviction among the
        # batch's own derivatives is what serial ingestion would decide.
        events = sorted(events, key=lambda event: (event.timestamp, event.uuid))

        segment_lists = await asyncio.gather(
            *(self._segmenter.segment(event) for event in events)
        )
        segments = [
            segment for segment_list in segment_lists for segment in segment_list
        ]
        t_segmentation = time.monotonic()

        derivative_lists = await asyncio.gather(
            *(self._deriver.derive(segment) for segment in segments)
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
            text = EventMemory._extract_text(derivative.block)
            if text is None:
                raise NotImplementedError(
                    f"Cannot embed a derivative of block type {derivative.block.block_type!r}"
                )
            derivative_texts.append(text)

        derivative_embeddings = await self._embedder.ingest_embed(derivative_texts)
        t_embedding = time.monotonic()

        displaced_uuids, skipped_uuids = await self._decide_eviction(
            derivatives, derivative_embeddings
        )
        t_eviction = time.monotonic()

        derivative_records = [
            EventMemory._build_derivative_record(derivative, embedding)
            for derivative, embedding in zip(
                derivatives, derivative_embeddings, strict=True
            )
            if derivative.uuid not in skipped_uuids
        ]
        events_to_segments = {
            event.uuid: {
                segment: [
                    derivative.uuid
                    for derivative in segments_to_derivatives[segment]
                    if derivative.uuid not in skipped_uuids
                ]
                for segment in segment_list
            }
            for event, segment_list in zip(events, segment_lists, strict=True)
        }

        # The surviving records are written inside the event memory store's
        # transaction: their links commit only once the vector store has
        # acknowledged them, and an upsert that fails rolls the links back.
        # The displaced records leave the vector store after the commit:
        # a delete that fails then leaves records no link names, which read
        # repair reclaims, rather than links naming records that are gone.
        async with self._event_memory_store_partition.write() as writer:
            await writer.add_events(events_to_segments)
            # Empty without eviction: the writer ignores an empty set.
            await writer.delete_derivatives(displaced_uuids)
            t_event_memory_store = time.monotonic()
            if derivative_records:
                try:
                    await self._vector_store_collection.upsert(
                        records=derivative_records
                    )
                except Exception as upsert_error:
                    # The upsert may have been applied before it failed;
                    # delete what it may have written, so the rollback
                    # leaves no record behind.
                    try:
                        await self._vector_store_collection.delete(
                            record_uuids=[record.uuid for record in derivative_records]
                        )
                    except Exception as delete_error:
                        upsert_error.add_note(
                            "deleting the records the upsert may have written "
                            f"failed too: {delete_error!r}"
                        )
                    raise
            t_vector_store = time.monotonic()
        t_commit = time.monotonic()

        if displaced_uuids:
            await self._vector_store_collection.delete(record_uuids=displaced_uuids)
        t_displaced_records = time.monotonic()

        phase_durations = {
            "segmentation": t_segmentation - t_start,
            "derivation": t_derivation - t_segmentation,
            "embedding": t_embedding - t_derivation,
            "eviction": t_eviction - t_embedding,
            "event_memory_store": (t_event_memory_store - t_eviction)
            + (t_commit - t_vector_store),
            "vector_store": (t_vector_store - t_event_memory_store)
            + (t_displaced_records - t_commit),
        }

        logger.debug(
            "encode_events timing: %s total=%.3fs",
            " ".join(
                f"{phase}={duration:.3f}s"
                for phase, duration in phase_durations.items()
            ),
            t_displaced_records - t_start,
        )

        if self._encode_events_phase_seconds is not None:
            for phase, duration in phase_durations.items():
                self._encode_events_phase_seconds.observe(
                    duration, labels={"phase": phase}
                )

    async def _decide_eviction(
        self,
        derivatives: Sequence[Derivative],
        derivative_embeddings: Sequence[Sequence[float]],
    ) -> tuple[set[UUID], set[UUID]]:
        """The stored derivatives to displace and the batch's to skip.

        Decided before the encode writes anything, from the batch's own
        embeddings and one neighbor query per derivative. Both sets are
        empty when eviction is not configured.

        Returns:
            tuple[set[UUID], set[UUID]]:
                The stored derivative UUIDs to displace and the batch
                derivative UUIDs to leave unwritten.
        """
        if self._eviction is None or not derivatives:
            return set(), set()

        batch_predecessors = EventMemory._compute_batch_predecessors(
            derivative_embeddings,
            self._eviction.cosine_similarity_threshold,
        )
        stored_neighbors = await self._vector_store_collection.query(
            query_vectors=list(derivative_embeddings),
            min_cosine_similarity=self._eviction.cosine_similarity_threshold,
            limit=self._eviction.search_limit,
        )
        stored_timestamps = await self._stored_derivative_timestamps(
            match.record_uuid
            for query_result in stored_neighbors
            for match in query_result.matches
        )
        return EventMemory._select_eviction_targets(
            derivatives,
            stored_neighbors,
            batch_predecessors,
            stored_timestamps,
            self._eviction.target_size,
        )

    async def _stored_derivative_timestamps(
        self, derivative_uuids: Iterable[UUID]
    ) -> dict[UUID, datetime.datetime]:
        """The event timestamp of each stored derivative, read from its segment.

        The vector store answers uuids and scores only; the segment store
        owns the derivative's segment, and the segment its timestamp. A
        derivative whose segment is gone is omitted.
        """
        segment_by_derivative = await self._event_memory_store_partition.get_segment_uuids_by_derivative_uuids(
            derivative_uuids
        )
        if not segment_by_derivative:
            return {}
        segments_by_uuid = await self._event_memory_store_partition.get_segments(
            set(segment_by_derivative.values())
        )
        return {
            derivative_uuid: segments_by_uuid[segment_uuid].timestamp
            for derivative_uuid, segment_uuid in segment_by_derivative.items()
            if segment_uuid in segments_by_uuid
        }

    @staticmethod
    def _build_derivative_record(
        derivative: Derivative,
        derivative_embedding: Sequence[float],
    ) -> Record:
        """Build a vector record from a derivative and its embedding."""
        properties: dict[str, PropertyValue] = {
            EVENT_TIMESTAMP_KEY: derivative.timestamp,
            EVENT_SESSION_KEY: derivative.session_id,
            BLOCK_KIND_KEY: derivative.block.block_type,
        }
        if derivative.source_id is not None:
            properties[EVENT_SOURCE_KEY] = derivative.source_id

        return Record(
            uuid=derivative.uuid,
            vector=list(derivative_embedding),
            properties=properties,
        )

    @staticmethod
    def _compute_batch_predecessors(
        derivative_embeddings: Iterable[Sequence[float]],
        cosine_similarity_threshold: float,
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
        cosine_similarity_matrix = normalized @ normalized.T

        # The diagonal and the upper triangle can never pass the threshold.
        cosine_similarity_matrix[np.triu_indices(num_embeddings)] = -np.inf
        mask = cosine_similarity_matrix >= cosine_similarity_threshold

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
        Select eviction targets by cosine similarity to stored and batch derivatives.

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

    async def query(
        self,
        query: str,
        *,
        vector_search_limit: int = 20,
        min_cosine_similarity: float | None = None,
        expand_context: int = 0,
        since: datetime.datetime | None = None,
        until: datetime.datetime | None = None,
        session_ids: Iterable[str] | None = None,
        source_ids: Iterable[str] | None = None,
        block_kinds: Iterable[str] | None = None,
        property_filter: FilterExpr | None = None,
    ) -> list[QueryHit]:
        """
        Query event memory for segments relevant to the query.

        Args:
            query (str):
                The search query.
            vector_search_limit (int):
                The maximum number of matches the vector search returns,
                and so of hits (default: 20).
            min_cosine_similarity (float | None):
                Drop matches whose cosine similarity is below this
                (default: None).
            expand_context (int):
                The maximum number of neighbors to include around each
                hit, nonnegative (default: 0).
            since (datetime | None):
                Inclusive lower bound on the events' timestamps, timezone-aware
                (default: None).
            until (datetime | None):
                Exclusive upper bound on the events' timestamps, timezone-aware
                (default: None).
            session_ids (Iterable[str] | None):
                Keep only events of these sessions; an empty list keeps
                none, and None keeps every session (default: None).
            source_ids (Iterable[str] | None):
                Keep only events of these sources; an empty list keeps
                none, and None keeps every source (default: None).
            block_kinds (Iterable[str] | None):
                Keep only segments whose block is of these kinds; an empty
                list keeps none, and None keeps every kind (default: None).
            property_filter (FilterExpr | None):
                A filter over the segments' user properties; None filters
                nothing (default: None).

        Returns:
            list[QueryHit]:
                At most `vector_search_limit` hits in descending cosine
                similarity, each with its seed and the neighborhood around
                it; neighborhoods of different hits may overlap.

        Raises:
            ValueError:
                If `expand_context` is negative, or `since` or `until` is
                naive.

        """
        async with self._tracker("query"):
            return await self._query(
                query,
                vector_search_limit=vector_search_limit,
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
        vector_search_limit: int,
        min_cosine_similarity: float | None,
        expand_context: int,
        since: datetime.datetime | None,
        until: datetime.datetime | None,
        session_ids: Iterable[str] | None,
        source_ids: Iterable[str] | None,
        block_kinds: Iterable[str] | None,
        property_filter: FilterExpr | None,
    ) -> list[QueryHit]:
        if expand_context < 0:
            raise ValueError(f"expand_context must be nonnegative: {expand_context}")
        t_start = time.monotonic()
        session_ids = list(session_ids) if session_ids is not None else None
        source_ids = list(source_ids) if source_ids is not None else None
        block_kinds = list(block_kinds) if block_kinds is not None else None

        query_embedding = (
            await self._embedder.search_embed(
                [query],
            )
        )[0]
        t_embedding = time.monotonic()

        # The vector stage evaluates the system fields only; the user
        # property filter is the event memory store's, applied to the seeds.
        collection_filter = _system_predicates(
            since=since,
            until=until,
            session_ids=session_ids,
            source_ids=source_ids,
            block_kinds=block_kinds,
        )

        # Search derivative collection for matches.
        [query_result] = await self._vector_store_collection.query(
            query_vectors=[query_embedding],
            limit=vector_search_limit,
            min_cosine_similarity=min_cosine_similarity,
            property_filter=collection_filter,
        )
        t_vector_query = time.monotonic()

        segment_by_derivative = await self._segment_uuids_by_record_uuids(
            [match.record_uuid for match in query_result.matches]
        )

        # Deduplicate by first occurrence (multiple derivatives can map to the same segment).
        # First occurrence has the best score since matches are ordered best-to-worst.
        cosine_similarity_by_seed_uuid: dict[UUID, float] = {}
        for match in query_result.matches:
            segment_uuid = segment_by_derivative.get(match.record_uuid)
            if segment_uuid is None:
                # An orphan, deleted by the repair.
                continue
            if segment_uuid not in cosine_similarity_by_seed_uuid:
                cosine_similarity_by_seed_uuid[segment_uuid] = match.cosine_similarity

        seed_segments = await self._event_memory_store_partition.get_segments(
            cosine_similarity_by_seed_uuid.keys(),
            since=since,
            until=until,
            session_ids=session_ids,
            source_ids=source_ids,
            block_kinds=block_kinds,
            property_filter=property_filter,
        )
        before = expand_context // 3
        after = expand_context - before
        neighborhoods: dict[UUID, Neighborhood] = {}
        if expand_context > 0 and seed_segments:
            neighborhoods = (
                await self._event_memory_store_partition.get_segment_neighborhoods(
                    seed_segments.keys(),
                    before=before,
                    after=after,
                    since=since,
                    until=until,
                    source_ids=source_ids,
                    block_kinds=block_kinds,
                    property_filter=property_filter,
                )
            )
        t_segment_query = time.monotonic()

        # Seeds the store did not return are dropped; cosine similarity order is kept.
        hits: list[QueryHit] = []
        for seed_uuid, score in cosine_similarity_by_seed_uuid.items():
            seed = seed_segments.get(seed_uuid)
            if seed is None:
                continue
            neighborhood = neighborhoods.get(
                seed_uuid, Neighborhood(before=[], after=[])
            )
            hits.append(QueryHit(score=score, seed=seed, neighborhood=neighborhood))

        phase_durations = {
            "embedding": t_embedding - t_start,
            "vector_query": t_vector_query - t_embedding,
            "segment_query": t_segment_query - t_vector_query,
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

        return hits

    async def _segment_uuids_by_record_uuids(
        self, record_uuids: list[UUID]
    ) -> dict[UUID, UUID]:
        """The segment of each record, repairing records found without one."""
        linked = await self._event_memory_store_partition.get_segment_uuids_by_derivative_uuids(
            record_uuids
        )
        unlinked = [uuid for uuid in record_uuids if uuid not in linked]
        if unlinked:
            linked.update(await self._repair_unlinked_records(unlinked))
        return linked

    async def _repair_unlinked_records(
        self, record_uuids: list[UUID]
    ) -> dict[UUID, UUID]:
        """Settle records the vector search returned without a link.

        Such a record is either an encode's, between its upsert and its
        commit, or an orphan: its encode rolled back after the upsert, or
        its event was forgotten and the record delete never landed. Under
        the exclusive fence no encode is in flight, so a link found now
        is kept, and a record still without one is an orphan, deleted
        once the fence is released; derivative uuids are never reused,
        so it can never gain a link later.

        Returns:
            dict[UUID, UUID]:
                The links found, from record uuid to segment uuid.
        """
        async with self._event_memory_store_partition.write(exclusive=True) as writer:
            linked = await writer.get_segment_uuids_by_derivative_uuids(record_uuids)
        orphans = [uuid for uuid in record_uuids if uuid not in linked]
        if orphans:
            await self._vector_store_collection.delete(record_uuids=orphans)
            if self._orphan_records_deleted is not None:
                self._orphan_records_deleted.increment(len(orphans))
        return linked

    async def expand(
        self,
        seed_uuid: UUID,
        *,
        before: int = 0,
        after: int = 0,
        since: datetime.datetime | None = None,
        until: datetime.datetime | None = None,
        session_ids: Iterable[str] | None = None,
        source_ids: Iterable[str] | None = None,
        block_kinds: Iterable[str] | None = None,
        property_filter: FilterExpr | None = None,
    ) -> Neighborhood:
        """
        Get the neighborhood of a seed segment: the segments before and after it in its session, in the store's order.

        The filters select the neighbors; the seed itself is excluded from
        the result.

        Args:
            seed_uuid (UUID):
                The UUID of the seed segment.
            before (int):
                The maximum number of neighbors before the seed, nonnegative
                (default: 0).
            after (int):
                The maximum number of neighbors after the seed, nonnegative
                (default: 0).
            since (datetime | None):
                Inclusive lower bound on the neighbors' timestamps, timezone-aware
                (default: None).
            until (datetime | None):
                Exclusive upper bound on the neighbors' timestamps, timezone-aware
                (default: None).
            session_ids (Iterable[str] | None):
                The sessions the seed may be in; a seed in another session
                is not found, and None allows any session (default: None).
            source_ids (Iterable[str] | None):
                Keep only neighbors of these sources; an empty list keeps
                none, and None keeps every source (default: None).
            block_kinds (Iterable[str] | None):
                Keep only neighbors whose block is of these kinds; an empty
                list keeps none, and None keeps every kind (default: None).
            property_filter (FilterExpr | None):
                A filter over the neighbors' user properties; None filters
                nothing (default: None).

        Returns:
            Neighborhood:
                The neighbors before and after the seed, in the store's order.

        Raises:
            LookupError:
                If the seed is not a segment of this memory, or its
                session is not among `session_ids`.
            ValueError:
                If `before` or `after` is negative, or `since` or `until`
                is naive.
        """
        async with self._tracker("expand"):
            if session_ids is not None:
                visible = await self._event_memory_store_partition.get_segments(
                    [seed_uuid], session_ids=session_ids
                )
                if seed_uuid not in visible:
                    raise LookupError(
                        f"Seed segment {seed_uuid} is not in the named sessions"
                    )
            neighborhoods = (
                await self._event_memory_store_partition.get_segment_neighborhoods(
                    [seed_uuid],
                    before=before,
                    after=after,
                    since=since,
                    until=until,
                    source_ids=source_ids,
                    block_kinds=block_kinds,
                    property_filter=property_filter,
                )
            )
            neighborhood = neighborhoods.get(seed_uuid)
            if neighborhood is None:
                raise LookupError(f"Seed segment {seed_uuid} is not in this memory")
            return neighborhood

    @staticmethod
    async def rerank(
        query: str,
        hits: Sequence[QueryHit],
        *,
        reranker: Reranker,
        datetime_format: DateTimeFormat,
    ) -> list[QueryHit]:
        """
        Rerank hits by a reranker's score of their rendered windows.

        Every hit is returned, in descending score, with its score
        replaced by the reranker's.
        """
        hits = list(hits)
        if not hits:
            return []
        scores = await reranker.score(
            query,
            [
                EventMemory.render_segments(
                    hit.window(), datetime_format=datetime_format
                )
                for hit in hits
            ],
        )
        reranked = [
            QueryHit(score=score, seed=hit.seed, neighborhood=hit.neighborhood)
            for hit, score in zip(hits, scores, strict=True)
        ]
        reranked.sort(key=lambda hit: hit.score, reverse=True)
        return reranked

    @staticmethod
    def render_segments(
        segments: Iterable[Segment],
        *,
        datetime_format: DateTimeFormat,
    ) -> str:
        """
        The reader's text for a run of segments, in their order.

        A header (the timestamp formatted by `datetime_format`, then the
        producer's name) starts each run of adjacent pieces of one event;
        the pieces' text is joined under it.
        """
        context_string = ""
        previous: Segment | None = None
        accumulated_text = ""

        for segment in segments:
            is_continuation = previous is not None and EventMemory._is_continuation(
                previous, segment
            )

            if not is_continuation:
                if previous is not None:
                    context_string += (
                        json.dumps(accumulated_text, ensure_ascii=False) + "\n"
                    )
                accumulated_text = ""
                context_string += EventMemory._segment_header(segment, datetime_format)

            text = EventMemory._extract_text(segment.block)
            if text is not None:
                accumulated_text += text
            elif not is_continuation:
                context_string += f"[{segment.block.block_type}]\n"

            previous = segment

        if previous is not None:
            context_string += json.dumps(accumulated_text, ensure_ascii=False) + "\n"

        return context_string.strip()

    @staticmethod
    def _is_continuation(previous: Segment, segment: Segment) -> bool:
        """Whether `segment` continues `previous`: the next piece of the same event.

        The next chunk of the same block, or the first chunk of the next
        block; a block's chunk count is not known here, so the end of a
        block is recognized by the next block starting at its beginning.
        """
        if segment.event_uuid != previous.event_uuid:
            return False
        if segment.index == previous.index:
            return segment.offset == previous.offset + 1
        if segment.index == previous.index + 1:
            return segment.offset == 0
        return False

    @staticmethod
    def _segment_header(segment: Segment, datetime_format: DateTimeFormat) -> str:
        """Build the header emitted before a segment."""
        formatted_timestamp = format_timestamp(segment.timestamp, datetime_format)
        timestamp_prefix = f"[{formatted_timestamp}] " if formatted_timestamp else ""

        match segment.context:
            case ProducerContext(producer=producer):
                return f"{timestamp_prefix}{producer}: "
            case NullContext():
                return timestamp_prefix
            case _:
                raise NotImplementedError(
                    f"Unsupported context type: {type(segment.context).__name__}"
                )

    @staticmethod
    def _extract_text(block: Block) -> str | None:
        """Extract text from a block, if it contains text."""
        match block:
            case TextBlock(text=text):
                return text
            case _:
                return None

    async def forget_events(self, event_uuids: Iterable[UUID]) -> None:
        """Forget events by their UUIDs."""
        event_uuids = set(event_uuids)
        if not event_uuids:
            return

        async with self._tracker("forget_events"):
            await self._forget_events(event_uuids)

    async def _forget_events(self, event_uuids: set[UUID]) -> None:
        derivatives_by_event = await self._event_memory_store_partition.get_derivative_uuids_by_event_uuids(
            event_uuids
        )
        derivative_uuids = {
            derivative_uuid
            for event_derivative_uuids in derivatives_by_event.values()
            for derivative_uuid in event_derivative_uuids
        }

        # Records before events: an event whose records are gone is
        # deleted next, while a failed record delete leaves the event
        # whole for a retry.
        if derivative_uuids:
            await self._vector_store_collection.delete(record_uuids=derivative_uuids)

        await self._event_memory_store_partition.delete_events(event_uuids)
