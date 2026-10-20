"""Event memory system for storing and retrieving events."""

import asyncio
import contextlib
import datetime
import json
import logging
import time
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Final, Self, cast
from uuid import UUID, uuid4

from pydantic import BaseModel, Field, InstanceOf

from memmachine_server.common.data_types import PropertyValue, SimilarityMetric
from memmachine_server.common.embedder import Embedder
from memmachine_server.common.filter.filter_parser import (
    And,
    Comparison,
    FilterExpr,
    In,
    demangle_user_metadata_key,
    extract_filter,
    filter_fields,
    map_filter_fields,
    normalize_filter_field,
)
from memmachine_server.common.metrics_factory import (
    MetricsFactory,
    OperationTracker,
)
from memmachine_server.common.property_keys import (
    is_reserved_property_key,
    reserved_property_key,
)
from memmachine_server.common.reranker import Reranker
from memmachine_server.common.vector_store import (
    QueryResult,
    Record,
    VectorStore,
    VectorStorePartition,
)
from memmachine_server.common.vector_store.utils import is_valid_identifier

from .data_types import (
    Block,
    DateTimeFormat,
    Derivative,
    Event,
    Neighborhood,
    NullContext,
    ProducerContext,
    QueryHit,
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

# The keys the memory writes into a vector record; `em` names this memory,
# short because the keys share the identifier budget.
_EVENT_MEMORY_SYSTEM: Final[str] = "em"
EVENT_TIMESTAMP_KEY: Final[str] = reserved_property_key(
    _EVENT_MEMORY_SYSTEM, "timestamp"
)
EVENT_SOURCE_KEY: Final[str] = reserved_property_key(_EVENT_MEMORY_SYSTEM, "source")

# Every property a vector record carries, with the type the collection
# declares: the fields a search filters on at the vector stage. User
# properties stay in the segment store.
_RESERVED_PROPERTY_SCHEMA: Final[dict[str, type[PropertyValue]]] = {
    EVENT_TIMESTAMP_KEY: cast(type[PropertyValue], datetime.datetime),
    EVENT_SOURCE_KEY: cast(type[PropertyValue], str),
}

# Seconds a claim on pending derivatives waits past their writer's budget,
# and holds them past its own: the margin between the database's clock,
# which ages the records, and the writers' clocks, which run the budgets.
_CLAIM_MARGIN_SECONDS = 60

# The filter field naming the event's timestamp; any other field names a
# property, as `m.<key>` or `metadata.<key>`.
_TIMESTAMP_FIELD: Final[str] = "timestamp"


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
    source_ids: Iterable[str] | None = None,
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
        In(field=EVENT_SOURCE_KEY, values=list(source_ids))
        if source_ids is not None
        else None,
    ]
    return _conjoin(clauses)


def _validate_property_filter(property_filter: FilterExpr) -> None:
    """
    Raise ValueError unless every field of a filter names a property or the timestamp.

    The memory's own reserved keys are reached through its typed search
    parameters alone.
    """
    for field in filter_fields(property_filter):
        if field == _TIMESTAMP_FIELD:
            continue
        key = _filter_field_property_key(field)
        if key is None:
            raise ValueError(
                f"Filter field {field!r} names no property: a property is "
                f"'m.<key>', and {_TIMESTAMP_FIELD!r} is the event's timestamp."
            )
        if is_reserved_property_key(key, system=_EVENT_MEMORY_SYSTEM):
            raise ValueError(
                f"Filter field {field!r} names a key reserved for the event memory."
            )


def _filter_field_property_key(field: str) -> str | None:
    """The property key a filter field names; None when it names none."""
    internal_name, is_user_metadata = normalize_filter_field(field)
    if not is_user_metadata:
        return None
    return demangle_user_metadata_key(internal_name)


class EventMemoryParams(BaseModel):
    """
    Parameters for EventMemory.

    Attributes:
        segment_store (SegmentStore):
            Segment store holding the events' segments.
        vector_store (VectorStore):
            Vector store holding the derivatives' vectors.
        segmenter (Segmenter):
            Segmenter that segments events into segments.
        deriver (Deriver):
            Deriver that derives derivatives from segments.
        embedder (Embedder):
            Embedder instance for creating embeddings.
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

    Stored data is immutable: no operation edits a stored segment or
    vector record, and none may be added. A change is a forget and an
    encode again.
    """

    @classmethod
    def expected_vector_store_collection_schema(cls) -> dict[str, type[PropertyValue]]:
        """
        Return the vector store collection schema EventMemory requires.

        Every reserved key the memory writes into a vector record, with its
        type; the vector store must declare each of them. A vector record
        carries only the properties the vector store declares, so the vector
        search can filter only on those; a condition on any other property
        applies to the seeds the search returns.
        """
        return dict(_RESERVED_PROPERTY_SCHEMA)

    def __init__(self, params: EventMemoryParams) -> None:
        """
        Initialize an EventMemory with the provided parameters.

        Args:
            params (EventMemoryParams):
                Parameters for the EventMemory.

        """
        self._params = params
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

        Raises:
            ValueError:
                If the vector store schema is missing fields required by
                EventMemory.
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
        schema_fields = frozenset(vector_store_partition.indexed_properties)
        await _upsert_within_budget(
            vector_store_partition,
            [
                _build_derivative_record(
                    derivative,
                    embedding,
                    claim.pending_derivatives.properties,
                    schema_fields,
                )
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

            text = _extract_text(segment.block)
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

        Raises:
            ValueError:
                If the vector store schema is missing fields required by
                EventMemory.
        """
        self._segment_store_partition = segment_store_partition
        self._vector_store_partition = vector_store_partition
        self._segmenter = params.segmenter
        self._deriver = params.deriver
        self._embedder = params.embedder
        self._write_budget_seconds = params.write_budget_seconds

        self._tracker = OperationTracker(
            params.metrics_factory,
            prefix="event_memory",
        )

        self._schema_fields = frozenset(vector_store_partition.indexed_properties)

        missing_fields = _RESERVED_PROPERTY_SCHEMA.keys() - self._schema_fields
        if missing_fields:
            raise ValueError(
                f"Vector store schema missing fields required by EventMemory: "
                f"{', '.join(sorted(missing_fields))}"
            )

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
        return self._vector_store_partition.similarity_metric

    def _validate_events(self, events: Iterable[Event]) -> None:
        """
        Validate a batch of events before encoding.

        Raises ValueError if any event supplies a property key outside the
        naming contract or under the memory's own reserved keys. A key
        reserved for another system belongs to the component that embeds
        the memory and is kept as given.
        """
        for event in events:
            for key in event.properties:
                if not is_valid_identifier(key):
                    raise ValueError(
                        f"Property key {key!r} must match [a-z0-9_] and be at "
                        "most 32 bytes."
                    )
                if is_reserved_property_key(key, system=_EVENT_MEMORY_SYSTEM):
                    raise ValueError(
                        f"Property key {key!r} is reserved for the event memory."
                    )

    async def encode_events(self, events: Iterable[Event]) -> None:
        """
        Encode events.

        The events are stored, with their segments and derivatives, before
        any derivative is written to the vector store. An error after that
        leaves them stored, and their derivatives are written in the
        background.

        Args:
            events (Iterable[Event]): The events to encode.

        Raises:
            ValueError:
                If any event supplies an illegal property key, or one of
                the memory's own reserved keys.
            SegmentStoreEventAlreadyStoredError:
                If the partition already holds any of the events; nothing
                is stored.
            TimeoutError:
                If the write budget ran out before the vector store had the
                derivatives; the events stay stored.
        """
        async with self._tracker("encode_events"):
            await self._encode_events(events)

    async def _encode_events(self, events: Iterable[Event]) -> None:
        t_start = time.monotonic()

        events = list(events)
        self._validate_events(events)
        if not events:
            return

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
                uuid=uuid4(),
                event_uuid=event.uuid,
                properties=event.properties,
                derivatives=derivatives_by_event[event.uuid],
            )
            for event in events
            if derivatives_by_event[event.uuid]
        ]
        await self._segment_store_partition.add_events(
            {
                event.uuid: segment_list
                for event, segment_list in zip(events, segment_lists, strict=True)
            },
            pending_derivatives,
        )
        t_segment_store = time.monotonic()

        # Built from the pending records, as the background work builds them,
        # so a record has the same properties whichever writes it.
        embedding_by_derivative = dict(
            zip(
                (derivative.uuid for derivative in derivatives),
                derivative_embeddings,
                strict=True,
            )
        )
        derivative_records = [
            _build_derivative_record(
                derivative,
                embedding_by_derivative[derivative.uuid],
                pending.properties,
                self._schema_fields,
            )
            for pending in pending_derivatives
            for derivative in pending.derivatives
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

    def _vector_store_filter(self, property_filter: FilterExpr) -> FilterExpr | None:
        """The part of a filter that names declared properties alone; None when there is none.

        A vector record carries only the properties the vector store
        declares, so a condition on any other property, or on the event's
        timestamp, is left to the segment store.
        """
        declared_keys = {
            field: key
            for field in filter_fields(property_filter)
            if (key := _filter_field_property_key(field)) is not None
            and key in self._schema_fields
        }
        vector_store_filter = extract_filter(
            property_filter, allowed_fields=declared_keys.keys()
        )
        if vector_store_filter is None:
            return None
        return map_filter_fields(vector_store_filter, declared_keys.__getitem__)

    async def query(
        self,
        query: str,
        *,
        vector_search_limit: int = 20,
        score_threshold: float | None = None,
        expand_context: int = 0,
        since: datetime.datetime | None = None,
        until: datetime.datetime | None = None,
        source_ids: Iterable[str] | None = None,
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
            score_threshold (float | None):
                The worst score a match may have, by the vector store
                partition's similarity metric; a match scoring exactly the
                threshold is kept (default: None).
            expand_context (int):
                The maximum number of neighbors to include around each
                hit, nonnegative (default: 0).
            since (datetime | None):
                Inclusive lower bound on the events' timestamps, timezone-aware
                (default: None).
            until (datetime | None):
                Exclusive upper bound on the events' timestamps, timezone-aware
                (default: None).
            source_ids (Iterable[str] | None):
                Keep only events of these sources; an empty list keeps
                none, and None keeps every source (default: None).
            property_filter (FilterExpr | None):
                A filter over the segments' properties, each named
                `m.<key>` or `metadata.<key>`, and their `timestamp`;
                None filters nothing (default: None). The vector search
                applies only the part of the filter that names declared
                properties alone (see `extract_filter`); the whole filter
                then applies to the seeds it returns, so fewer than
                `vector_search_limit` seeds, or none, may survive it.

        Returns:
            list[QueryHit]:
                At most `vector_search_limit` hits, best first by the
                vector search's score, each with its seed and the
                neighborhood around it; neighborhoods of different hits
                may overlap.

        Raises:
            ValueError:
                If `expand_context` is negative, `since` or `until` is
                naive, `score_threshold` is not finite, or `property_filter`
                names a field that is neither a property nor the timestamp,
                or a key reserved for the memory.

        """
        if property_filter is not None:
            _validate_property_filter(property_filter)
        async with self._tracker("query"):
            return await self._query(
                query,
                vector_search_limit=vector_search_limit,
                score_threshold=score_threshold,
                expand_context=expand_context,
                since=since,
                until=until,
                source_ids=source_ids,
                property_filter=property_filter,
            )

    async def _query(
        self,
        query: str,
        *,
        vector_search_limit: int,
        score_threshold: float | None,
        expand_context: int,
        since: datetime.datetime | None,
        until: datetime.datetime | None,
        source_ids: Iterable[str] | None,
        property_filter: FilterExpr | None,
    ) -> list[QueryHit]:
        if expand_context < 0:
            raise ValueError(f"expand_context must be nonnegative: {expand_context}")
        t_start = time.monotonic()
        source_ids = list(source_ids) if source_ids is not None else None

        query_embedding = (
            await self._embedder.search_embed(
                [query],
            )
        )[0]
        t_embedding = time.monotonic()

        # A vector record carries the properties the vector store declares, so
        # a condition naming any other field has nothing to match there. The
        # vector store gets the predicates on reserved keys and the part of
        # the filter on declared fields, which only ever widens its search;
        # the segment store, which holds every property, gets the whole
        # filter below and narrows it back.
        collection_filter = _conjoin(
            [
                _system_predicates(
                    since=since,
                    until=until,
                    source_ids=source_ids,
                ),
                self._vector_store_filter(property_filter)
                if property_filter is not None
                else None,
            ]
        )

        # Search the session's vector store partition for matching derivatives.
        [query_result] = await self._vector_store_partition.query(
            query_vectors=[query_embedding],
            limit=vector_search_limit,
            score_threshold=score_threshold,
            property_filter=collection_filter,
        )
        t_vector_query = time.monotonic()

        segments_by_derivatives = (
            await self._segment_store_partition.get_segment_uuids_by_derivative_uuids(
                match.record_uuid for match in query_result.matches
            )
        )

        score_by_seed_uuid = await self._score_seeds(
            query_result, segments_by_derivatives
        )

        seed_segments = await self._segment_store_partition.get_segments(
            score_by_seed_uuid.keys(),
            since=since,
            until=until,
            source_ids=source_ids,
            property_filter=property_filter,
        )
        before = expand_context // 3
        after = expand_context - before
        neighborhoods: dict[UUID, Neighborhood] = {}
        if expand_context > 0 and seed_segments:
            neighborhoods = (
                await self._segment_store_partition.get_segment_neighborhoods(
                    seed_segments.keys(),
                    before=before,
                    after=after,
                    since=since,
                    until=until,
                    source_ids=source_ids,
                    property_filter=property_filter,
                )
            )
        t_segment_query = time.monotonic()

        # Seeds the store did not return are dropped; the best-first order is kept.
        hits: list[QueryHit] = []
        for seed_uuid, score in score_by_seed_uuid.items():
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

    async def _score_seeds(
        self,
        query_result: QueryResult,
        segments_by_derivatives: Mapping[UUID, UUID],
    ) -> dict[UUID, float]:
        """Each matched segment's best score, in the matches' best-first order.

        Several derivatives can match one segment, and matches come best
        first, so a segment's first match has its best score. A match whose
        record no derivative link names is deleted
        (`_delete_unlinked_records`).
        """
        score_by_seed_uuid: dict[UUID, float] = {}
        unlinked_record_uuids: list[UUID] = []
        for match in query_result.matches:
            segment_uuid = segments_by_derivatives.get(match.record_uuid)
            if segment_uuid is None:
                # The derivative's segment is gone; its vector outlived it.
                unlinked_record_uuids.append(match.record_uuid)
                continue
            if segment_uuid not in score_by_seed_uuid:
                score_by_seed_uuid[segment_uuid] = match.score
        if unlinked_record_uuids:
            await self._delete_unlinked_records(unlinked_record_uuids)
        return score_by_seed_uuid

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

    async def expand(
        self,
        seed_uuid: UUID,
        *,
        before: int = 0,
        after: int = 0,
        since: datetime.datetime | None = None,
        until: datetime.datetime | None = None,
        source_ids: Iterable[str] | None = None,
        property_filter: FilterExpr | None = None,
    ) -> Neighborhood:
        """
        Get the neighborhood of a seed segment: the segments before and after it in the store's order.

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
            source_ids (Iterable[str] | None):
                Keep only neighbors of these sources; an empty list keeps
                none, and None keeps every source (default: None).
            property_filter (FilterExpr | None):
                A filter over the neighbors' properties, named as in
                `query`; None filters nothing (default: None).

        Returns:
            Neighborhood:
                The neighbors before and after the seed, in the store's order.

        Raises:
            LookupError:
                If the seed is not a segment of this memory.
            ValueError:
                If `before` or `after` is negative, `since` or `until` is
                naive, or `property_filter` names a field `query` refuses.
        """
        if property_filter is not None:
            _validate_property_filter(property_filter)
        async with self._tracker("expand"):
            neighborhoods = (
                await self._segment_store_partition.get_segment_neighborhoods(
                    [seed_uuid],
                    before=before,
                    after=after,
                    since=since,
                    until=until,
                    source_ids=source_ids,
                    property_filter=property_filter,
                )
            )
            neighborhood = neighborhoods.get(seed_uuid)
            if neighborhood is None:
                raise LookupError(f"Seed segment {seed_uuid} is not in this memory")
            return neighborhood

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
                f"Cannot embed a derivative of block type {derivative.block.block_type!r}"
            )
        derivative_texts.append(text)
    return derivative_texts


def _build_derivative_record(
    derivative: Derivative,
    derivative_embedding: Sequence[float],
    event_properties: Mapping[str, PropertyValue],
    schema_fields: frozenset[str],
) -> Record:
    """Build a vector record from a derivative, its embedding and its event's properties.

    The record carries the reserved keys and the event properties the
    vector store declares, which are its segment's, since a segment
    carries its event's properties unchanged. Every other property stays
    on the segment: the segment store holds them all and evaluates the
    whole filter (see `EventMemoryPartition.query`).
    """
    properties: dict[str, PropertyValue] = {
        **{
            key: value
            for key, value in event_properties.items()
            if key in schema_fields
        },
        EVENT_TIMESTAMP_KEY: derivative.timestamp,
    }
    if derivative.source_id is not None:
        properties[EVENT_SOURCE_KEY] = derivative.source_id

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
