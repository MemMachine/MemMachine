"""Event memory system for storing and retrieving events."""

import asyncio
import contextlib
import datetime
import json
import logging
import time
from collections.abc import Collection, Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Final, Literal, Self, cast
from uuid import UUID, uuid4

import numpy as np
import numpy.typing as npt
from pydantic import BaseModel, Field, InstanceOf

from memmachine_server.common.data_types import (
    PropertyType,
    PropertyValue,
    SimilarityMetric,
)
from memmachine_server.common.embedder import Embedder
from memmachine_server.common.filter.filter_parser import (
    And,
    Comparison,
    FilterExpr,
    In,
    demangle_user_metadata_key,
    extract_filter,
    map_filter_fields,
    normalize_filter_field,
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
    VectorStore,
    VectorStorePartition,
    VectorStoreSchema,
)

from .data_types import (
    DateTimeFormat,
    Derivative,
    Event,
    EvictionOptions,
    Neighborhood,
    QueryHit,
    Segment,
    part_kinds,
)
from .deriver import Deriver
from .event_memory_store import (
    EventMemoryStore,
    EventMemoryStorePartition,
    EventMemoryStorePartitionConfig,
    PendingDerivatives,
    PendingDerivativesClaim,
)
from .formatting import format_header
from .segmenter import Segmenter

logger = logging.getLogger(__name__)


# The keys the memory writes into a vector record; `em` names this memory,
# short because the keys share the identifier budget.
EVENT_TIMESTAMP_KEY: Final[str] = reserved_property_key("em", "timestamp")
EVENT_SOURCE_KEY: Final[str] = reserved_property_key("em", "source")
BLOCK_KIND_KEY: Final[str] = reserved_property_key("em", "block_kind")

# Every property a vector record carries, with the type the collection
# declares: the fields a search filters on at the vector stage. User
# properties stay in the event memory store.
_RESERVED_PROPERTY_SCHEMA: Final[dict[str, type[PropertyValue]]] = {
    EVENT_TIMESTAMP_KEY: cast(type[PropertyValue], datetime.datetime),
    EVENT_SOURCE_KEY: cast(type[PropertyValue], str),
    BLOCK_KIND_KEY: cast(type[PropertyValue], str),
}

# Seconds a claim on pending derivatives waits past their writer's budget,
# and holds them past its own: the margin between the database's clock,
# which ages the records, and the writers' clocks, which run the budgets.
_CLAIM_MARGIN_SECONDS = 60

IdKind = Literal["segment"]
"""What `render_segments` marks with ids: each line's segments."""

_ID_KINDS: Final[frozenset[str]] = frozenset({"segment"})


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
        event_memory_store (EventMemoryStore):
            Event memory store holding the events' segments.
        vector_store (VectorStore):
            Vector store holding the derivatives' vectors.
        store_schema (VectorStoreSchema):
            Schema the vector store was built with.
        segmenter (Segmenter):
            The table from block kind to handler that segments events.
        deriver (Deriver):
            The table from block kind to handler that derives from segments.
        embedder (Embedder):
            Embedder instance for creating embeddings.
        eviction (EvictionOptions | None):
            Evict stored derivatives a new one nearly duplicates, at
            ingest: the batch's own displaced derivatives are never
            stored, and the stored ones' records leave the vector store
            once the batch's records are written. None keeps every
            derivative and issues no eviction query (default: None).
        metrics_factory (MetricsFactory | None):
            An instance of MetricsFactory for collecting usage metrics
            (default: None).
        write_budget_seconds (int):
            Seconds from its start within which a write of derivatives to
            the vector store sends its last request. A write still running
            then is cancelled, and its derivatives are written again in the
            background. The event memory store's tombstone retention must exceed
            it by a wide margin (default: 120).
        background_idle_seconds (int):
            Seconds the background work waits after finding nothing to do
            (default: 5).
    """

    event_memory_store: InstanceOf[EventMemoryStore] = Field(
        ...,
        description="Event memory store holding the events' segments",
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
        description="The table from block kind to handler that segments events",
    )
    deriver: InstanceOf[Deriver] = Field(
        ...,
        description="The table from block kind to handler that derives from segments",
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
    write_budget_seconds: int = Field(
        120,
        gt=0,
        description=(
            "Seconds from its start within which a write of derivatives to "
            "the vector store sends its last request. A write still running "
            "then is cancelled, and its derivatives are written again in the "
            "background. The event memory store's tombstone retention must exceed "
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
    Event memory over an event memory store and a vector store.

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
    def expected_vector_store_indexed_properties(cls) -> dict[str, PropertyType]:
        """
        Return the properties EventMemory requires its vector store to declare.

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

        Raises:
            ValueError:
                If the vector store schema is missing fields required by
                EventMemory.
        """
        self._params = params
        self._schema_fields = frozenset(params.store_schema.indexed_properties)

        missing_fields = _RESERVED_PROPERTY_SCHEMA.keys() - self._schema_fields
        if missing_fields:
            raise ValueError(
                f"Vector store schema missing fields required by EventMemory: "
                f"{', '.join(sorted(missing_fields))}"
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

        The vector store's partition is created before the event memory store's,
        so an event memory store partition always has its vector store partition.

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
        event_memory_store_partition = (
            await self._params.event_memory_store.open_or_create_partition(
                partition_key,
                EventMemoryStorePartitionConfig(),
            )
        )
        return EventMemoryPartition(
            self._params,
            event_memory_store_partition=event_memory_store_partition,
            vector_store_partition=vector_store_partition,
            schema_fields=self._schema_fields,
            similarity_metric=self._similarity_metric,
        )

    async def delete_partition(self, partition_key: str) -> None:
        """
        Delete the partition from both stores. Idempotent.

        The event memory store's partition is deleted first. Its text is gone
        then, no write can start on it, and the background work stops
        taking up its pending derivatives and tombstones, whose vectors
        the vector store's partition deletion removes.

        Args:
            partition_key (str):
                The key of the partition to delete.
        """
        await self._params.event_memory_store.delete_partition(partition_key)
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
        claim = await self._params.event_memory_store.claim_pending_derivatives(
            min_age_seconds=self._params.write_budget_seconds + _CLAIM_MARGIN_SECONDS,
            lease_seconds=self._params.write_budget_seconds + _CLAIM_MARGIN_SECONDS,
        )
        if claim is None:
            return False
        try:
            (
                vector_store_partition,
                evicted_uuids,
            ) = await self._write_pending_derivatives(claim, budget)
        except Exception as error:
            logger.warning(
                "Writing pending derivatives %s of partition %r failed on attempt %d",
                claim.pending_derivatives.uuid,
                claim.partition_key,
                claim.attempt,
                exc_info=True,
            )
            await self._params.event_memory_store.fail_pending_derivatives_claim(
                claim, error
            )
            return True
        await self._params.event_memory_store.finish_pending_derivatives_claim(claim)
        await _delete_evicted_records(vector_store_partition, evicted_uuids)
        return True

    async def _write_pending_derivatives(
        self, claim: PendingDerivativesClaim, budget: "_WriteBudget"
    ) -> tuple[VectorStorePartition, set[UUID]]:
        """Write a claim's derivatives to the vector store within the budget, deciding eviction again.

        Returns:
            tuple[VectorStorePartition, set[UUID]]:
                The partition's vector store partition, and the derivative
                UUIDs whose records eviction removes from it.
        """
        vector_store_partition = await self._params.vector_store.open_partition(
            claim.partition_key
        )
        # The vector store partition opened under the key belongs to the
        # claimed partition only while that partition exists, since a
        # partition leaves the event memory store first.
        await claim.partition.require_current()
        if vector_store_partition is None:
            raise RuntimeError(
                f"Partition {claim.partition_key!r} has no vector store partition"
            )
        # In temporal order, as an encode decides eviction.
        derivatives = sorted(
            claim.pending_derivatives.derivatives,
            key=lambda derivative: (derivative.timestamp, derivative.uuid),
        )
        embeddings = await self._params.embedder.ingest_embed(
            [derivative.text for derivative in derivatives]
        )
        # Decided again, against what is stored now. The stranded write may
        # have stored some of the claim's own derivatives, which are not
        # their own neighbors, and a skipped one's record leaves with the
        # displaced ones.
        displaced_uuids, skipped_uuids = await _decide_eviction(
            self._params.eviction,
            vector_store_partition,
            self._similarity_metric,
            claim.partition,
            derivatives,
            embeddings,
            excluded_uuids={derivative.uuid for derivative in derivatives},
        )
        records = [
            _build_derivative_record(
                derivative,
                embedding,
                claim.pending_derivatives.properties,
                self._schema_fields,
            )
            for derivative, embedding in zip(derivatives, embeddings, strict=True)
            if derivative.uuid not in skipped_uuids
        ]
        if records:
            await _upsert_within_budget(vector_store_partition, records, budget)
        return vector_store_partition, displaced_uuids | skipped_uuids

    async def _delete_tombstoned_derivatives(self) -> bool:
        """Delete one claimed tombstone's derivatives from the vector store; whether one was claimed."""
        claim = await self._params.event_memory_store.claim_tombstone()
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
            await self._params.event_memory_store.fail_tombstone_claim(claim, error)
            return True
        await self._params.event_memory_store.finish_tombstone_claim(claim)
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
        parts: Iterable[str] = ("author",),
        ids: Iterable[IdKind] = (),
    ) -> str:
        """
        The reader's text for segments: a line per event, in the store's order.

        A segment given twice is rendered once. A line is a run of adjacent
        pieces of one event: a header (`format_header`: the timestamp, then
        the context parts `parts` names, in order), then the pieces' block
        renderings. Nothing in the text says whether two lines are adjacent
        in the store.

        `ids` names what is marked: `"segment"` starts each line with
        `[segment:<hex>]`, or `[segments:<first>..<last>]` when the line
        holds more than one segment, a uuid as 32 hex digits.

        Raises:
            ValueError:
                If `ids` names a kind other than `"segment"`.
        """
        parts = part_kinds(parts)
        ids = frozenset(ids)
        unknown = ids - _ID_KINDS
        if unknown:
            raise ValueError(f"Unknown id kinds: {sorted(unknown)}")
        unique: dict[UUID, Segment] = {}
        for segment in segments:
            unique.setdefault(segment.uuid, segment)
        lines: list[str] = []
        for run in EventMemory._runs(sorted(unique.values(), key=_store_order)):
            line = EventMemory._render_run(run, datetime_format, parts)
            if "segment" in ids:
                line = f"{_segment_marker(run)} {line}"
            lines.append(line)
        return "\n".join(lines)

    @staticmethod
    def _runs(segments: Iterable[Segment]) -> list[list[Segment]]:
        """Split segments, in the store's order, into runs of adjacent pieces of one event."""
        runs: list[list[Segment]] = []
        for segment in segments:
            if runs and EventMemory._is_continuation(runs[-1][-1], segment):
                runs[-1].append(segment)
            else:
                runs.append([segment])
        return runs

    @staticmethod
    def _render_run(
        run: Sequence[Segment],
        datetime_format: DateTimeFormat,
        parts: Sequence[str],
    ) -> str:
        """One event's adjacent pieces as one line: the header, then the block renderings."""
        first = run[0]
        text = format_header(first.timestamp, first.context, datetime_format, parts)
        accumulated = ""
        for segment in run:
            rendered = segment.block.render(datetime_format)
            if rendered is not None:
                accumulated += rendered
            elif segment is first:
                text += f"[{segment.block.kind}]\n"
        return text + json.dumps(accumulated, ensure_ascii=False)

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
        event_memory_store_partition: EventMemoryStorePartition,
        vector_store_partition: VectorStorePartition,
        schema_fields: frozenset[str],
        similarity_metric: SimilarityMetric,
    ) -> None:
        """
        Initialize a handle on one partition of an EventMemory.

        Args:
            params (EventMemoryParams):
                Parameters of the EventMemory.
            event_memory_store_partition (EventMemoryStorePartition):
                The partition's event memory store partition.
            vector_store_partition (VectorStorePartition):
                The partition's vector store partition.
            schema_fields (frozenset[str]):
                The properties the vector store declares.
            similarity_metric (SimilarityMetric):
                The metric the vector store scores matches by.
        """
        self._event_memory_store_partition = event_memory_store_partition
        self._vector_store_partition = vector_store_partition
        self._segmenter = params.segmenter
        self._deriver = params.deriver
        self._embedder = params.embedder
        self._write_budget_seconds = params.write_budget_seconds
        self._eviction = params.eviction

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

        Raises ValueError if any event supplies a property key in the
        reserved namespace or outside the naming contract.
        """
        for event in events:
            for key in event.properties:
                validate_user_property_key(key)

    async def encode_events(self, events: Iterable[Event]) -> None:
        """
        Encode events.

        The events are stored, with their segments and derivatives, before
        any derivative is written to the vector store. An error after that
        leaves them stored, and their derivatives are written in the
        background.

        With eviction configured, a derivative of the batch that a cluster
        trims is never stored, and a stored derivative a new one displaces
        keeps its link and its segment while its record leaves the vector
        store once the batch's records are written, best effort.

        Args:
            events (Iterable[Event]): The events to encode.

        Raises:
            ValueError:
                If any event supplies a reserved or illegal property key.
            EventMemoryStoreEventAlreadyStoredError:
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

        derivative_embeddings = await self._embedder.ingest_embed(
            [derivative.text for derivative in derivatives]
        )
        t_embedding = time.monotonic()

        displaced_uuids, skipped_uuids = await _decide_eviction(
            self._eviction,
            self._vector_store_partition,
            self._similarity_metric,
            self._event_memory_store_partition,
            derivatives,
            derivative_embeddings,
            # The batch's derivatives are new, so none is stored yet.
            excluded_uuids=frozenset(),
        )
        t_eviction = time.monotonic()

        # Started before the events are stored, so a forget of them either
        # finds nothing or stamps its tombstone after the start.
        budget = _WriteBudget.start(self._write_budget_seconds)
        derivatives_by_event: dict[UUID, list[Derivative]] = {
            event.uuid: [] for event in events
        }
        for segment, segment_derivatives in segments_to_derivatives.items():
            derivatives_by_event[segment.event_uuid].extend(
                derivative
                for derivative in segment_derivatives
                if derivative.uuid not in skipped_uuids
            )
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
        await self._event_memory_store_partition.add_events(
            {
                event.uuid: segment_list
                for event, segment_list in zip(events, segment_lists, strict=True)
            },
            pending_derivatives,
        )
        t_event_memory_store = time.monotonic()

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

        await self._event_memory_store_partition.delete_pending_derivatives(
            pending.uuid for pending in pending_derivatives
        )
        t_pending_derivatives = time.monotonic()

        # A displaced derivative keeps its link, and its record leaves the
        # vector store once the batch's records are written.
        await _delete_evicted_records(self._vector_store_partition, displaced_uuids)
        t_displaced_records = time.monotonic()

        phase_durations = {
            "segmentation": t_segmentation - t_start,
            "derivation": t_derivation - t_segmentation,
            "embedding": t_embedding - t_derivation,
            "eviction": t_eviction - t_embedding,
            "event_memory_store": t_event_memory_store - t_eviction,
            "vector_store": t_vector_store - t_event_memory_store,
            "pending_derivatives": t_pending_derivatives - t_vector_store,
            "displaced_records": t_displaced_records - t_pending_derivatives,
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
        score_threshold: float | None = None,
        expand_context: int = 0,
        since: datetime.datetime | None = None,
        until: datetime.datetime | None = None,
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
            block_kinds (Iterable[str] | None):
                Keep only segments whose block is of these kinds; an empty
                list keeps none, and None keeps every kind (default: None).
            property_filter (FilterExpr | None):
                A filter over the segments' properties; None filters
                nothing (default: None). The vector search applies only
                the part of the filter that names declared fields alone
                (see `extract_filter`); the whole filter then applies to
                the seeds it returns, so fewer than `vector_search_limit`
                seeds, or none, may survive it.

        Returns:
            list[QueryHit]:
                At most `vector_search_limit` hits, best first by the
                vector search's score, each with its seed and the
                neighborhood around it; neighborhoods of different hits
                may overlap.

        Raises:
            ValueError:
                If `expand_context` is negative, `since` or `until` is
                naive, or `score_threshold` is not finite.

        """
        async with self._tracker("query"):
            return await self._query(
                query,
                vector_search_limit=vector_search_limit,
                score_threshold=score_threshold,
                expand_context=expand_context,
                since=since,
                until=until,
                source_ids=source_ids,
                block_kinds=block_kinds,
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
        block_kinds: Iterable[str] | None,
        property_filter: FilterExpr | None,
    ) -> list[QueryHit]:
        if expand_context < 0:
            raise ValueError(f"expand_context must be nonnegative: {expand_context}")
        t_start = time.monotonic()
        source_ids = list(source_ids) if source_ids is not None else None
        block_kinds = list(block_kinds) if block_kinds is not None else None

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
        # the event memory store, which holds every property, gets the whole
        # filter below and narrows it back.
        vector_store_filter = _conjoin(
            [
                _system_predicates(
                    since=since,
                    until=until,
                    source_ids=source_ids,
                    block_kinds=block_kinds,
                ),
                extract_filter(
                    map_filter_fields(property_filter, self._to_vector_record_property),
                    allowed_fields=self._schema_fields,
                )
                if property_filter is not None
                else None,
            ]
        )

        [query_result] = await self._vector_store_partition.query(
            query_vectors=[query_embedding],
            limit=vector_search_limit,
            score_threshold=score_threshold,
            property_filter=vector_store_filter,
        )
        t_vector_query = time.monotonic()

        segments_by_derivatives = await self._event_memory_store_partition.get_segment_uuids_by_derivative_uuids(
            match.record_uuid for match in query_result.matches
        )

        score_by_seed_uuid = await self._score_seeds(
            query_result, segments_by_derivatives
        )

        seed_segments = await self._event_memory_store_partition.get_segments(
            score_by_seed_uuid.keys(),
            since=since,
            until=until,
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
        block_kinds: Iterable[str] | None = None,
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
                If the seed is not a segment of this memory.
            ValueError:
                If `before` or `after` is negative, or `since` or `until`
                is naive.
        """
        async with self._tracker("expand"):
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

    async def forget_events(self, event_uuids: Iterable[UUID]) -> None:
        """
        Forget events by their UUIDs.

        When this returns, the events, their segments and their derivatives
        are gone from the event memory store, and their vectors are deleted from
        the vector store at once and again after the event memory store's
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
        tombstone = await self._event_memory_store_partition.forget_events(event_uuids)
        if tombstone is None:
            return
        # The forget is complete once its tombstone commits: the background
        # work deletes the vectors whether or not this first delete does.
        try:
            await self._vector_store_partition.delete(
                record_uuids=tombstone.derivative_uuids
            )
            await self._event_memory_store_partition.record_tombstone_first_delete(
                tombstone.uuid
            )
        except Exception:
            logger.warning(
                "Deleting the vectors of forgotten derivatives failed; the "
                "background work deletes them",
                exc_info=True,
            )


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
    on the segment: the event memory store holds them all and evaluates the
    whole filter (see `EventMemoryPartition.query`).
    """
    properties: dict[str, PropertyValue] = {
        **{
            key: value
            for key, value in event_properties.items()
            if key in schema_fields
        },
        EVENT_TIMESTAMP_KEY: derivative.timestamp,
        BLOCK_KIND_KEY: derivative.block_kind,
    }
    if derivative.source_id is not None:
        properties[EVENT_SOURCE_KEY] = derivative.source_id

    return Record(
        uuid=derivative.uuid,
        vector=list(derivative_embedding),
        properties=properties,
    )


async def _delete_evicted_records(
    vector_store_partition: VectorStorePartition, record_uuids: set[UUID]
) -> None:
    """Delete evicted derivatives' records, best effort.

    An evicted derivative keeps its link, so a record a failed delete leaves
    stays searchable until a later eviction or a forget removes it.
    """
    if not record_uuids:
        return
    try:
        await vector_store_partition.delete(record_uuids=record_uuids)
    except Exception:
        logger.warning(
            "Deleting the records of %d evicted derivatives failed",
            len(record_uuids),
            exc_info=True,
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


async def _decide_eviction(
    eviction: EvictionOptions | None,
    vector_store_partition: VectorStorePartition,
    similarity_metric: SimilarityMetric,
    event_memory_store_partition: EventMemoryStorePartition,
    derivatives: Sequence[Derivative],
    derivative_embeddings: Sequence[Sequence[float]],
    *,
    excluded_uuids: Collection[UUID],
) -> tuple[set[UUID], set[UUID]]:
    """The stored derivatives to displace and the batch's to skip.

    Decided before the batch's records are written, from the batch's own
    embeddings and one neighbor query per derivative. A stored record whose
    UUID is in `excluded_uuids` is no derivative's neighbor. Both sets are
    empty when eviction is not configured.

    Returns:
        tuple[set[UUID], set[UUID]]:
            The stored derivative UUIDs to displace and the batch
            derivative UUIDs to leave unwritten.
    """
    if eviction is None or not derivatives:
        return set(), set()

    batch_predecessors = _compute_batch_predecessors(
        derivative_embeddings,
        similarity_metric,
        eviction.score_threshold,
    )
    stored_neighbors = [
        QueryResult(
            matches=[
                match
                for match in query_result.matches
                if match.record_uuid not in excluded_uuids
            ][: eviction.search_limit]
        )
        for query_result in await vector_store_partition.query(
            query_vectors=list(derivative_embeddings),
            score_threshold=eviction.score_threshold,
            # Room for the excluded records besides `search_limit` others.
            limit=eviction.search_limit + len(excluded_uuids),
        )
    ]
    stored_timestamps = await _stored_derivative_timestamps(
        event_memory_store_partition,
        (
            match.record_uuid
            for query_result in stored_neighbors
            for match in query_result.matches
        ),
    )
    return _select_eviction_targets(
        derivatives,
        stored_neighbors,
        batch_predecessors,
        stored_timestamps,
        eviction.target_size,
    )


async def _stored_derivative_timestamps(
    event_memory_store_partition: EventMemoryStorePartition,
    derivative_uuids: Iterable[UUID],
) -> dict[UUID, datetime.datetime]:
    """The event timestamp of each stored derivative, read from its segment.

    The vector store answers uuids and scores only; the event memory
    store owns the derivative's segment, and the segment its timestamp.
    A derivative whose segment is gone is omitted.
    """
    segment_by_derivative = (
        await event_memory_store_partition.get_segment_uuids_by_derivative_uuids(
            derivative_uuids
        )
    )
    if not segment_by_derivative:
        return {}
    segments_by_uuid = await event_memory_store_partition.get_segments(
        set(segment_by_derivative.values())
    )
    return {
        derivative_uuid: segments_by_uuid[segment_uuid].timestamp
        for derivative_uuid, segment_uuid in segment_by_derivative.items()
        if segment_uuid in segments_by_uuid
    }


def _compute_batch_predecessors(
    derivative_embeddings: Iterable[Sequence[float]],
    similarity_metric: SimilarityMetric,
    score_threshold: float,
) -> list[set[int]]:
    """
    Compute batch predecessors for each derivative embedding.

    The ith entry holds the indices j < i whose score with i, by
    `similarity_metric`, is at least as good as the threshold, the test
    the vector store applies to a stored neighbor. Only earlier indices
    count, so a batch evicts exactly what serial ingestion would.
    """
    embeddings = np.asarray(list(derivative_embeddings), dtype=np.float64)
    num_embeddings = len(embeddings)
    if num_embeddings == 0:
        return []

    scores = _pairwise_scores(embeddings, similarity_metric)
    if similarity_metric.higher_is_better:
        mask = scores >= score_threshold
    else:
        mask = scores <= score_threshold
    # The diagonal and the upper triangle never count.
    mask[np.triu_indices(num_embeddings)] = False

    return [set(np.where(mask[i])[0].tolist()) for i in range(num_embeddings)]


def _select_eviction_targets(
    derivatives: Iterable[Derivative],
    query_results: Iterable[QueryResult],
    batch_predecessors: list[set[int]],
    stored_timestamps: Mapping[UUID, datetime.datetime],
    target_size: int,
) -> tuple[set[UUID], set[UUID]]:
    """
    Select eviction targets among near-duplicate stored and batch derivatives.

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


def _pairwise_scores(
    embeddings: npt.NDArray[np.float64], similarity_metric: SimilarityMetric
) -> npt.NDArray[np.float64]:
    """Every pair's score by `similarity_metric`, in the units the vector store reports.

    Entry (i, j) scores embeddings i and j: cosine similarity or dot product
    under the higher-is-better metrics, Euclidean or Manhattan distance
    under the others.
    """
    match similarity_metric:
        case SimilarityMetric.COSINE:
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            normalized = embeddings / norms
            return normalized @ normalized.T
        case SimilarityMetric.DOT:
            return embeddings @ embeddings.T
        case SimilarityMetric.EUCLIDEAN:
            return np.stack(
                [np.linalg.norm(embeddings - row, axis=1) for row in embeddings]
            )
        case SimilarityMetric.MANHATTAN:
            return np.stack(
                [np.abs(embeddings - row).sum(axis=1) for row in embeddings]
            )


def _store_order(segment: Segment) -> tuple[datetime.datetime, UUID, int, int]:
    """The event memory store's total order."""
    return (segment.timestamp, segment.event_uuid, segment.index, segment.offset)


def _segment_marker(run: Sequence[Segment]) -> str:
    """The id marker of a rendered line: its one segment, or its first and last."""
    if len(run) == 1:
        return f"[segment:{run[0].uuid.hex}]"
    return f"[segments:{run[0].uuid.hex}..{run[-1].uuid.hex}]"
