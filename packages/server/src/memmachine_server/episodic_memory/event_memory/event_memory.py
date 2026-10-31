"""Event memory system for storing and retrieving events."""

import asyncio
import contextlib
import datetime
import json
import logging
import time
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Self, cast
from uuid import UUID, uuid4

from pydantic import BaseModel, Field, InstanceOf

from memmachine_server.common.data_types import (
    PropertyType,
    PropertyValue,
    SimilarityMetric,
)
from memmachine_server.common.embedder import Embedder
from memmachine_server.common.filter.filter_parser import (
    FilterExpr,
    demangle_user_metadata_key,
    extract_filter,
    map_filter_fields,
    normalize_filter_field,
)
from memmachine_server.common.metrics_factory import (
    MetricsFactory,
    OperationTracker,
)
from memmachine_server.common.reranker import Reranker
from memmachine_server.common.vector_store import (
    Record,
    VectorStore,
    VectorStorePartition,
    VectorStoreSchema,
)

from .data_types import (
    Block,
    Derivative,
    Event,
    FormatOptions,
    NullContext,
    ProducerContext,
    QueryResult,
    ScoredSegmentContext,
    Segment,
    TextBlock,
)
from .deriver import Deriver
from .formatting import format_timestamp
from .segment_store import (
    PendingDerivatives,
    PendingDerivativesClaim,
    SegmentStore,
    SegmentStorePartition,
    SegmentStorePartitionConfig,
)
from .segmenter import Segmenter

logger = logging.getLogger(__name__)

# System-defined metadata field names. Reserved.
_TIMESTAMP_FIELD_NAME = "_timestamp"

_BASE_EVENT_MEMORY_FIELD_NAMES: frozenset[str] = frozenset({_TIMESTAMP_FIELD_NAME})

# Seconds a claim on pending derivatives waits past their writer's budget,
# and holds them past its own: the margin between the database's clock,
# which ages the records, and the writers' clocks, which run the budgets.
_CLAIM_MARGIN_SECONDS = 60


class EventMemoryParams(BaseModel):
    """
    Parameters for EventMemory.

    Attributes:
        segment_store (SegmentStore):
            Segment store holding the events' segments.
        vector_store (VectorStore):
            Vector store holding the derivatives' vectors.
        store_schema (VectorStoreSchema):
            Schema the vector store was built with.
        segmenter (Segmenter):
            Segmenter that segments events into segments.
        deriver (Deriver):
            Deriver that derives derivatives from segments.
        embedder (Embedder):
            Embedder instance for creating embeddings.
        reranker (Reranker | None):
            Reranker instance for scoring search results.
            If None, embedding similarity scores are used instead
            (default: None).
        metrics_factory (MetricsFactory | None):
            An instance of MetricsFactory for collecting usage metrics
            (default: None).
        write_budget_seconds (int):
            Seconds from its start within which a write of derivatives to
            the vector store sends its last request. A write still running
            then is cancelled, and its derivatives are written again in the
            background. The segment store's tombstone retention must exceed
            it by a wide margin (default: 120).
        background_idle_seconds (int):
            Seconds the background work waits after finding nothing to do
            (default: 5).
    """

    segment_store: InstanceOf[SegmentStore] = Field(
        ...,
        description="Segment store holding the events' segments",
    )
    vector_store: InstanceOf[VectorStore] = Field(
        ...,
        description="Vector store holding the derivatives' vectors",
    )
    store_schema: VectorStoreSchema = Field(
        ...,
        description="Schema the vector store was built with",
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
    reranker: InstanceOf[Reranker] | None = Field(
        None,
        description="Reranker instance for scoring search results. "
        "If None, embedding similarity scores are used instead",
    )
    metrics_factory: InstanceOf[MetricsFactory] | None = Field(
        None,
        description="An instance of MetricsFactory for collecting usage metrics",
    )
    write_budget_seconds: int = Field(
        120,
        gt=0,
        description=(
            "Seconds from its start within which a write of derivatives to "
            "the vector store sends its last request. A write still running "
            "then is cancelled, and its derivatives are written again in the "
            "background. The segment store's tombstone retention must exceed "
            "it by a wide margin"
        ),
    )
    background_idle_seconds: int = Field(
        5,
        gt=0,
        description="Seconds the background work waits after finding nothing to do",
    )


class EventMemory:
    """
    Event memory over a segment store and a vector store.

    A partition key names one partition in each store, and must be valid
    in both. Handles on a partition read and write that pair only.

    Between `startup` and `shutdown`, it finishes in the background what
    its partitions' writes and forgets leave undone: derivatives whose
    writer stopped before the vector store had them, and the deletes of
    forgotten derivatives, once at once and once more after the segment
    store's retention. Any number of processes may run this work over the
    same stores at once.
    """

    @classmethod
    def expected_vector_store_indexed_properties(cls) -> dict[str, PropertyType]:
        """
        Return the properties EventMemory requires its vector store to declare.

        A vector record carries only the properties the vector store
        declares, so the vector search can filter only on those; a condition
        on any other property applies to the seeds the search returns.
        """
        return {
            _TIMESTAMP_FIELD_NAME: cast(type[PropertyValue], datetime.datetime),
        }

    def __init__(self, params: EventMemoryParams) -> None:
        """
        Initialize an EventMemory with the provided parameters.

        Args:
            params (EventMemoryParams):
                Parameters for the EventMemory.

        Raises:
            ValueError:
                If the vector store schema is missing fields required by
                EventMemory.
        """
        self._params = params
        self._schema_fields = frozenset(params.store_schema.indexed_properties)

        missing_base_fields = _BASE_EVENT_MEMORY_FIELD_NAMES - self._schema_fields
        if missing_base_fields:
            raise ValueError(
                f"Vector store schema missing fields required by EventMemory: "
                f"{', '.join(sorted(missing_base_fields))}"
            )

        self._similarity_metric = params.store_schema.similarity_metric
        self._background_work: asyncio.Task[None] | None = None

    async def startup(self) -> None:
        """Start the background work."""
        self._background_work = asyncio.create_task(self._work_in_background())

    async def shutdown(self) -> None:
        """
        Stop the background work.

        What it leaves unfinished is taken up again by background work
        started later, in this process or another.
        """
        if self._background_work is None:
            return
        self._background_work.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await self._background_work
        self._background_work = None

    async def open_or_create_partition(
        self, partition_key: str
    ) -> "EventMemoryPartition":
        """
        Open the partition if it exists, or create it if it does not.

        The vector store's partition is created before the segment store's,
        so a segment store partition always has its vector store partition.

        Args:
            partition_key (str):
                The key of the partition.

        Returns:
            EventMemoryPartition:
                A handle on the partition.
        """
        vector_store_partition = (
            await self._params.vector_store.open_or_create_partition(partition_key)
        )
        segment_store_partition = (
            await self._params.segment_store.open_or_create_partition(
                partition_key,
                SegmentStorePartitionConfig(),
            )
        )
        return EventMemoryPartition(
            self._params,
            segment_store_partition=segment_store_partition,
            vector_store_partition=vector_store_partition,
            schema_fields=self._schema_fields,
            similarity_metric=self._similarity_metric,
        )

    async def delete_partition(self, partition_key: str) -> None:
        """
        Delete the partition from both stores. Idempotent.

        The segment store's partition is deleted first. Its text is gone
        then, no write can start on it, and the background work stops
        taking up its pending derivatives and tombstones, whose vectors
        the vector store's partition deletion removes.

        Args:
            partition_key (str):
                The key of the partition to delete.
        """
        await self._params.segment_store.delete_partition(partition_key)
        await self._params.vector_store.delete_partition(partition_key)

    async def _work_in_background(self) -> None:
        """Finish pending derivatives and delete tombstoned ones until cancelled."""
        while True:
            try:
                worked = await self._finish_pending_derivatives()
                worked = await self._delete_tombstoned_derivatives() or worked
            except Exception:
                logger.exception("Event memory background work failed; pausing")
                worked = False
            if not worked:
                await asyncio.sleep(self._params.background_idle_seconds)

    async def _finish_pending_derivatives(self) -> bool:
        """Write one claimed record of pending derivatives to the vector store; whether one was claimed."""
        # Started before the claim: a forget of these derivatives stamps its
        # tombstone after the claim, so after the start.
        budget = _WriteBudget.start(self._params.write_budget_seconds)
        claim = await self._params.segment_store.claim_pending_derivatives(
            min_age_seconds=self._params.write_budget_seconds + _CLAIM_MARGIN_SECONDS,
            lease_seconds=self._params.write_budget_seconds + _CLAIM_MARGIN_SECONDS,
        )
        if claim is None:
            return False
        try:
            await self._write_pending_derivatives(claim, budget)
        except Exception as error:
            logger.warning(
                "Writing pending derivatives %s of partition %r failed on attempt %d",
                claim.pending_derivatives.uuid,
                claim.partition_key,
                claim.attempt,
                exc_info=True,
            )
            await self._params.segment_store.fail_pending_derivatives_claim(
                claim, error
            )
            return True
        await self._params.segment_store.finish_pending_derivatives_claim(claim)
        return True

    async def _write_pending_derivatives(
        self, claim: PendingDerivativesClaim, budget: "_WriteBudget"
    ) -> None:
        """Write a claim's derivatives to the vector store within the budget."""
        vector_store_partition = await self._params.vector_store.open_partition(
            claim.partition_key
        )
        # The vector store partition opened under the key belongs to the
        # claimed partition only while that partition exists, since a
        # partition leaves the segment store first.
        await claim.partition.require_current()
        if vector_store_partition is None:
            raise RuntimeError(
                f"Partition {claim.partition_key!r} has no vector store partition"
            )
        derivatives = claim.pending_derivatives.derivatives
        embeddings = await self._params.embedder.ingest_embed(
            _derivative_texts(derivatives)
        )
        await _upsert_within_budget(
            vector_store_partition,
            [
                _build_derivative_record(derivative, embedding, self._schema_fields)
                for derivative, embedding in zip(derivatives, embeddings, strict=True)
            ],
            budget,
        )

    async def _delete_tombstoned_derivatives(self) -> bool:
        """Delete one claimed tombstone's derivatives from the vector store; whether one was claimed."""
        claim = await self._params.segment_store.claim_tombstone()
        if claim is None:
            return False
        try:
            vector_store_partition = await self._params.vector_store.open_partition(
                claim.partition_key
            )
            # A partition without its vector store partition has no vectors
            # left to delete.
            if vector_store_partition is not None:
                await vector_store_partition.delete(
                    record_uuids=claim.tombstone.derivative_uuids
                )
        except Exception as error:
            logger.warning(
                "Deleting tombstoned derivatives %s of partition %r failed on "
                "attempt %d",
                claim.tombstone.uuid,
                claim.partition_key,
                claim.attempt,
                exc_info=True,
            )
            await self._params.segment_store.fail_tombstone_claim(claim, error)
            return True
        await self._params.segment_store.finish_tombstone_claim(claim)
        return True

    @staticmethod
    def string_from_segment_context(
        segment_context: Iterable[Segment],
        *,
        format_options: FormatOptions | None = None,
    ) -> str:
        """Format segment context as a string."""
        if format_options is None:
            format_options = FormatOptions(time_style="short")

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

            text = _extract_text(segment.block)
            if text is not None:
                accumulated_text += text
            elif not is_continuation:
                context_string += f"[{segment.block.block_type}]\n"

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
        formatted_timestamp = format_timestamp(segment.timestamp, format_options)
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


class EventMemoryPartition:
    """
    A handle on one partition of an EventMemory.

    Reads and writes only its partition in each store. Obtained from
    `EventMemory.open_or_create_partition`.
    """

    def __init__(
        self,
        params: EventMemoryParams,
        *,
        segment_store_partition: SegmentStorePartition,
        vector_store_partition: VectorStorePartition,
        schema_fields: frozenset[str],
        similarity_metric: SimilarityMetric,
    ) -> None:
        """
        Initialize a handle on one partition of an EventMemory.

        Args:
            params (EventMemoryParams):
                Parameters of the EventMemory.
            segment_store_partition (SegmentStorePartition):
                The partition's segment store partition.
            vector_store_partition (VectorStorePartition):
                The partition's vector store partition.
            schema_fields (frozenset[str]):
                The properties the vector store declares.
            similarity_metric (SimilarityMetric):
                The metric the vector store scores matches by.
        """
        self._segment_store_partition = segment_store_partition
        self._vector_store_partition = vector_store_partition
        self._segmenter = params.segmenter
        self._deriver = params.deriver
        self._embedder = params.embedder
        self._reranker = params.reranker
        self._write_budget_seconds = params.write_budget_seconds

        self._tracker = OperationTracker(
            params.metrics_factory,
            prefix="event_memory",
        )

        self._schema_fields = schema_fields
        self._similarity_metric = similarity_metric

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

    @property
    def similarity_metric(self) -> SimilarityMetric:
        """Similarity metric by which the vector search scores matches."""
        return self._similarity_metric

    def _validate_events(self, events: Iterable[Event]) -> None:
        """
        Validate a batch of events before encoding.

        Raises ValueError if any event supplies a reserved field name in its properties.
        """
        events = list(events)

        reserved_fields = {
            field
            for event in events
            for field in event.properties
            if field in _BASE_EVENT_MEMORY_FIELD_NAMES
        }
        if reserved_fields:
            raise ValueError(
                f"Event properties must not contain reserved fields: "
                f"{', '.join(sorted(reserved_fields))}"
            )

    async def encode_events(
        self,
        events: Iterable[Event],
        *,
        format_options: FormatOptions | None = None,
    ) -> None:
        """
        Encode events.

        The events are stored, with their segments and derivatives, before
        any derivative is written to the vector store. An error after that
        leaves them stored, and their derivatives are written in the
        background.

        Args:
            events (Iterable[Event]): The events to encode.
            format_options (FormatOptions | None):
                Options for formatting.
                (default: None).

        Raises:
            ValueError:
                If any event supplies a reserved field name in its properties.
            SegmentStoreEventAlreadyStoredError:
                If the partition already holds any of the events; nothing
                is stored.
            TimeoutError:
                If the write budget ran out before the vector store had the
                derivatives; the events stay stored.
        """
        async with self._tracker("encode_events"):
            await self._encode_events(events, format_options=format_options)

    async def _encode_events(
        self,
        events: Iterable[Event],
        *,
        format_options: FormatOptions | None,
    ) -> None:
        t_start = time.monotonic()

        events = list(events)
        self._validate_events(events)

        segment_lists = await asyncio.gather(
            *(
                self._segmenter.segment(event, format_options=format_options)
                for event in events
            )
        )
        segments = [
            segment for segment_list in segment_lists for segment in segment_list
        ]
        t_segmentation = time.monotonic()

        derivative_lists = await asyncio.gather(
            *(
                self._deriver.derive(segment, format_options=format_options)
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

        derivative_embeddings = await self._embedder.ingest_embed(
            _derivative_texts(derivatives)
        )
        t_embedding = time.monotonic()

        # Started before the events are stored, so a forget of them either
        # finds nothing or stamps its tombstone after the start.
        budget = _WriteBudget.start(self._write_budget_seconds)
        derivatives_by_event: dict[UUID, list[Derivative]] = {
            event.uuid: [] for event in events
        }
        for segment, segment_derivatives in segments_to_derivatives.items():
            derivatives_by_event[segment.event_uuid].extend(segment_derivatives)
        pending_derivatives = [
            PendingDerivatives(
                uuid=uuid4(), event_uuid=event_uuid, derivatives=event_derivatives
            )
            for event_uuid, event_derivatives in derivatives_by_event.items()
            if event_derivatives
        ]
        await self._segment_store_partition.add_events(
            {
                event.uuid: segment_list
                for event, segment_list in zip(events, segment_lists, strict=True)
            },
            pending_derivatives,
        )
        t_segment_store = time.monotonic()

        derivative_records = [
            _build_derivative_record(
                derivative, derivative_embedding, self._schema_fields
            )
            for derivative, derivative_embedding in zip(
                derivatives,
                derivative_embeddings,
                strict=True,
            )
        ]

        if derivative_records:
            await _upsert_within_budget(
                self._vector_store_partition, derivative_records, budget
            )
        t_vector_store = time.monotonic()

        await self._segment_store_partition.delete_pending_derivatives(
            pending.uuid for pending in pending_derivatives
        )
        t_pending_derivatives = time.monotonic()

        phase_durations = {
            "segmentation": t_segmentation - t_start,
            "derivation": t_derivation - t_segmentation,
            "embedding": t_embedding - t_derivation,
            "segment_store": t_segment_store - t_embedding,
            "vector_store": t_vector_store - t_segment_store,
            "pending_derivatives": t_pending_derivatives - t_vector_store,
        }

        logger.debug(
            "encode_events timing: %s total=%.3fs",
            " ".join(
                f"{phase}={duration:.3f}s"
                for phase, duration in phase_durations.items()
            ),
            t_pending_derivatives - t_start,
        )

        if self._encode_events_phase_seconds is not None:
            for phase, duration in phase_durations.items():
                self._encode_events_phase_seconds.observe(
                    duration, labels={"phase": phase}
                )

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
                The vector search applies only the part of the filter
                that names declared fields alone (see `extract_filter`);
                the whole filter then applies to the seeds it returns,
                so fewer than `vector_search_limit` seeds, or none,
                may survive it.
            format_options (FormatOptions | None):
                Options for formatting.
                (default: None).

        Returns:
            QueryResult:
                The query result.

        """
        async with self._tracker("query"):
            return await self._query(
                query,
                vector_search_limit=vector_search_limit,
                expand_context=expand_context,
                property_filter=property_filter,
                format_options=format_options,
            )

    async def _query(
        self,
        query: str,
        *,
        vector_search_limit: int,
        expand_context: int,
        property_filter: FilterExpr | None,
        format_options: FormatOptions | None,
    ) -> QueryResult:
        t_start = time.monotonic()
        query_embedding = (
            await self._embedder.search_embed(
                [query],
            )
        )[0]
        t_embedding = time.monotonic()

        # A vector record carries the properties the vector store declares, so
        # a condition naming any other field has nothing to match there. The
        # vector store gets the part of the filter on declared fields, which
        # only ever widens its search; the segment store, which holds every
        # property, gets the whole filter below and narrows it back.
        vector_store_filter = None
        if property_filter is not None:
            vector_store_filter = extract_filter(
                map_filter_fields(property_filter, self._to_vector_record_property),
                allowed_fields=self._schema_fields,
            )

        [query_result] = await self._vector_store_partition.query(
            query_vectors=[query_embedding],
            limit=vector_search_limit,
            property_filter=vector_store_filter,
        )
        t_vector_query = time.monotonic()

        segments_by_derivatives = (
            await self._segment_store_partition.get_segment_uuids_by_derivative_uuids(
                match.record_uuid for match in query_result.matches
            )
        )

        # Extract seed segment UUIDs and their best embedding scores.
        # Deduplicate by first occurrence (multiple derivatives can map to the same segment).
        # First occurrence has the best score since matches are ordered best-to-worst.
        seed_embedding_scores: dict[UUID, float] = {}
        unlinked_record_uuids: list[UUID] = []
        for match in query_result.matches:
            segment_uuid = segments_by_derivatives.get(match.record_uuid)
            if segment_uuid is None:
                # The derivative's segment is gone; its vector outlived it.
                unlinked_record_uuids.append(match.record_uuid)
                continue
            if segment_uuid not in seed_embedding_scores:
                seed_embedding_scores[segment_uuid] = match.score
        if unlinked_record_uuids:
            await self._delete_unlinked_records(unlinked_record_uuids)

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
        t_segment_query = time.monotonic()

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
            reranker_format_options = format_options or FormatOptions(
                time_style="short"
            )
            scores = await self._score_segment_contexts(
                query, segment_contexts, reranker_format_options
            )
        t_scoring = time.monotonic()

        # Reranker scores are always higher-is-better.
        # Embedding scores depend on the similarity metric.
        higher_is_better = (
            self._reranker is not None or self._similarity_metric.higher_is_better
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

        phase_durations = {
            "embedding": t_embedding - t_start,
            "vector_query": t_vector_query - t_embedding,
            "segment_query": t_segment_query - t_vector_query,
            "scoring": t_scoring - t_segment_query,
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

        return QueryResult(scored_segment_contexts=scored_segment_contexts)

    async def _delete_unlinked_records(self, record_uuids: list[UUID]) -> None:
        """Delete vector records no derivative link names, once, best effort.

        A link is written before its record and removed only by a forget,
        which deletes its record in the background, so such a record was
        left by a delete that did not reach the vector store. A delete
        that fails here is tried again by the next query that finds it.
        """
        try:
            await self._vector_store_partition.delete(record_uuids=record_uuids)
        except Exception:
            logger.warning(
                "Deleting %d vector records without derivative links failed",
                len(record_uuids),
                exc_info=True,
            )

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

    async def forget_events(self, event_uuids: Iterable[UUID]) -> None:
        """
        Forget events by their UUIDs.

        When this returns, the events, their segments and their derivatives
        are gone from the segment store, and their vectors are deleted from
        the vector store at once and again after the segment store's
        retention, by the background work if not here.

        Args:
            event_uuids (Iterable[UUID]):
                The UUIDs of the events to forget.
        """
        event_uuids = set(event_uuids)
        if not event_uuids:
            return

        async with self._tracker("forget_events"):
            await self._forget_events(event_uuids)

    async def _forget_events(self, event_uuids: set[UUID]) -> None:
        tombstone = await self._segment_store_partition.forget_events(event_uuids)
        if tombstone is None:
            return
        # The forget is complete once its tombstone commits: the background
        # work deletes the vectors whether or not this first delete does.
        try:
            await self._vector_store_partition.delete(
                record_uuids=tombstone.derivative_uuids
            )
            await self._segment_store_partition.record_tombstone_first_delete(
                tombstone.uuid
            )
        except Exception:
            logger.warning(
                "Deleting the vectors of forgotten derivatives failed; the "
                "background work deletes them",
                exc_info=True,
            )


def _extract_text(block: Block) -> str | None:
    """Extract text from a block, if it contains text."""
    match block:
        case TextBlock(text=text):
            return text
        case _:
            return None


def _derivative_texts(derivatives: Iterable[Derivative]) -> list[str]:
    """The text each derivative embeds."""
    derivative_texts: list[str] = []
    for derivative in derivatives:
        text = _extract_text(derivative.block)
        if text is None:
            raise NotImplementedError(
                f"Unsupported block type: {type(derivative.block).__name__}"
            )
        derivative_texts.append(text)
    return derivative_texts


def _build_derivative_record(
    derivative: Derivative,
    derivative_embedding: Sequence[float],
    schema_fields: frozenset[str],
) -> Record:
    """Build a vector record from a derivative and its embedding.

    The record carries the reserved timestamp and the properties the vector
    store declares. Every other property stays on the segment: the segment
    store holds them all and evaluates the whole filter (see `query`).
    """
    properties: dict[str, PropertyValue] = {
        _TIMESTAMP_FIELD_NAME: derivative.timestamp,
        **{
            key: value
            for key, value in derivative.properties.items()
            if key in schema_fields
        },
    }

    return Record(
        uuid=derivative.uuid,
        vector=list(derivative_embedding),
        properties=properties,
    )


async def _upsert_within_budget(
    vector_store_partition: VectorStorePartition,
    records: list[Record],
    budget: "_WriteBudget",
) -> None:
    """
    Upsert records, sending nothing once the budget has run out.

    The upsert is cancelled when the budget runs out, which stops it from
    sending further requests; a request already sent may still be applied.

    Raises:
        TimeoutError: If the budget ran out before the upsert finished.
    """
    remaining = budget.remaining_seconds()
    if remaining <= 0:
        raise TimeoutError("The write budget ran out before the upsert started")
    async with asyncio.timeout(remaining):
        await vector_store_partition.upsert(records=records)


@dataclass(frozen=True)
class _WriteBudget:
    """
    The time a writer of derivatives may keep sending, from its start.

    Elapsed time is the larger of the monotonic and the wall clock's, so a
    process suspended mid-write counts the time it slept, and a wall clock
    set back does not lengthen the budget.
    """

    seconds: float
    started_monotonic: float
    started_wall: float

    @classmethod
    def start(cls, seconds: float) -> Self:
        """Start a budget of `seconds` now."""
        return cls(
            seconds=seconds,
            started_monotonic=time.monotonic(),
            started_wall=time.time(),
        )

    def remaining_seconds(self) -> float:
        """Seconds left in the budget; zero or less once it has run out."""
        elapsed = max(
            time.monotonic() - self.started_monotonic,
            time.time() - self.started_wall,
        )
        return self.seconds - elapsed
