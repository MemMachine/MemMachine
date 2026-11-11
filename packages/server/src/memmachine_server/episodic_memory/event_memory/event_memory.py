"""Event memory system for storing and retrieving events."""

import asyncio
import contextlib
import datetime
import json
import logging
import time
from collections.abc import Awaitable, Iterable, Sequence
from typing import Final, cast
from uuid import UUID

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
from memmachine_server.common.partition_registry import (
    PartitionAlreadyExistsError,
    PartitionCreateAttemptsExhaustedError,
    PartitionIncarnationEndedError,
    PartitionPendingError,
    PartitionRegistry,
    Registration,
    Reservation,
    validate_partition_name,
)
from memmachine_server.common.property_keys import (
    is_reserved_property_key,
    reserved_property_key,
)
from memmachine_server.common.reranker import Reranker
from memmachine_server.common.vector_store import (
    Record,
    VectorStore,
    VectorStorePartition,
)
from memmachine_server.common.vector_store.data_types import is_valid_identifier

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
    SegmentStore,
    SegmentStorePartition,
    SegmentStorePartitionConfig,
)
from .segmenter import Segmenter

logger = logging.getLogger(__name__)

# The key the memory writes into a vector record; `em` names this memory,
# short because every key shares the identifier budget.
_EVENT_MEMORY_SYSTEM: Final[str] = "em"
EVENT_TIMESTAMP_KEY: Final[str] = reserved_property_key(
    _EVENT_MEMORY_SYSTEM, "timestamp"
)

_BASE_EVENT_MEMORY_FIELD_NAMES: frozenset[str] = frozenset({EVENT_TIMESTAMP_KEY})

# Attempts open-or-create makes, _OPEN_OR_CREATE_RETRY_DELAY_SECONDS apart,
# before it gives up on a partition that stays pending or a name it keeps
# losing.
_MAX_OPEN_OR_CREATE_ATTEMPTS = 10
_OPEN_OR_CREATE_RETRY_DELAY_SECONDS = 1

# Reservations cancelled after a failed creation, held until done so the
# garbage collector cannot drop one whose creation was cancelled.
_reservation_cancellations: set[asyncio.Task[None]] = set()


class EventMemoryParams(BaseModel):
    """
    Parameters for EventMemory.

    Attributes:
        partition_registry (PartitionRegistry):
            Registry of event memory's partition namespace, shared by every
            process serving it. Started by the caller.
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
        reranker (Reranker | None):
            Reranker instance for scoring search results.
            If None, embedding similarity scores are used instead
            (default: None).
        metrics_factory (MetricsFactory | None):
            An instance of MetricsFactory for collecting usage metrics
            (default: None).
    """

    partition_registry: InstanceOf[PartitionRegistry] = Field(
        ...,
        description=(
            "Registry of event memory's partition namespace, shared by every "
            "process serving it. Started by the caller"
        ),
    )
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
    reranker: InstanceOf[Reranker] | None = Field(
        None,
        description="Reranker instance for scoring search results. "
        "If None, embedding similarity scores are used instead",
    )
    metrics_factory: InstanceOf[MetricsFactory] | None = Field(
        None,
        description="An instance of MetricsFactory for collecting usage metrics",
    )


class EventMemory:
    """
    Event memory over a segment store and a vector store.

    Event memory owns its partition namespace: its registry names each
    partition and mints an incarnation for each of its lives, and each store
    holds a life's records under the incarnation. Handles on a partition
    read and write that life only.
    """

    PARTITION_NAMESPACE: Final[str] = "event_memory"
    """The partition namespace event memory owns, the same in every deployment."""

    @classmethod
    def expected_vector_store_indexed_properties(cls) -> dict[str, PropertyType]:
        """
        Return the properties EventMemory requires its vector store to declare.

        A vector record carries only the properties the vector store
        declares, so the vector search can filter only on those; a condition
        on any other property applies to the seeds the search returns.
        """
        return {
            EVENT_TIMESTAMP_KEY: cast(type[PropertyValue], datetime.datetime),
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
        store_schema = params.vector_store.store_schema
        self._schema_fields = frozenset(store_schema.indexed_properties)

        missing_base_fields = _BASE_EVENT_MEMORY_FIELD_NAMES - self._schema_fields
        if missing_base_fields:
            raise ValueError(
                f"Vector store schema missing fields required by EventMemory: "
                f"{', '.join(sorted(missing_base_fields))}"
            )

        self._similarity_metric = store_schema.similarity_metric

    async def open_or_create_partition(
        self, partition_name: str
    ) -> "EventMemoryPartition":
        """
        Open the partition if it exists, or create it if it does not.

        Creating reserves the name in the registry, prepares every store's
        storage under the minted incarnation, and confirms the reservation.

        Args:
            partition_name (str):
                The name of the partition.

        Returns:
            EventMemoryPartition:
                A handle bound to the partition's current life.

        Raises:
            ValueError:
                If the partition name breaks the registry's rule.
            PartitionPendingError:
                If the partition's creation, by another caller, did not
                complete within the attempts to open it.
            PartitionCreateAttemptsExhaustedError:
                If every attempt to open or create the partition lost the
                name to another creator or deleter.
            PartitionIncarnationEndedError:
                If the partition was deleted before its handle was built.
        """
        validate_partition_name(partition_name)
        registration = await self._open_or_create_registration(partition_name)
        segment_store_partition = (
            await self._params.segment_store.open_partition_incarnation(
                registration.incarnation
            )
        )
        if segment_store_partition is None:
            raise PartitionIncarnationEndedError(
                self._params.partition_registry.partition_namespace,
                registration.partition_name,
                registration.incarnation,
            )
        vector_store_partition = (
            await self._params.vector_store.open_partition_incarnation(
                registration.incarnation
            )
        )
        return EventMemoryPartition(
            self._params,
            registration=registration,
            segment_store_partition=segment_store_partition,
            vector_store_partition=vector_store_partition,
            schema_fields=self._schema_fields,
            similarity_metric=self._similarity_metric,
        )

    async def _open_or_create_registration(self, partition_name: str) -> Registration:
        """
        Answer the registration of the live partition under a name, creating the partition if no partition holds the name.

        Read-then-create, retried a bounded number of times: a live partition
        is answered; a pending one is another creator's, so the call waits for
        it; no partition means create; losing the reservation means another
        creator took the name meanwhile; and losing the confirmation means a
        deleter removed the partition while its storage was prepared, so the
        call creates again.
        """
        partition_registry = self._params.partition_registry
        pending_error: PartitionPendingError | None = None
        lost_race: (
            PartitionAlreadyExistsError | PartitionIncarnationEndedError | None
        ) = None
        for attempt in range(_MAX_OPEN_OR_CREATE_ATTEMPTS):
            if attempt:
                await asyncio.sleep(_OPEN_OR_CREATE_RETRY_DELAY_SECONDS)
            try:
                registration = await partition_registry.resolve(partition_name)
            except PartitionPendingError as err:
                pending_error = err
                continue
            pending_error = None
            if registration is not None:
                return registration
            try:
                reservation = await partition_registry.reserve(partition_name)
            except PartitionAlreadyExistsError as err:
                lost_race = err
                continue
            try:
                return await self._prepare_and_confirm_or_cancel(reservation)
            except PartitionIncarnationEndedError as err:
                lost_race = err
                continue
        # The last lookup found the partition pending.
        if pending_error is not None:
            raise pending_error
        raise PartitionCreateAttemptsExhaustedError(
            f"Opening or creating partition {partition_name!r} made no progress "
            f"after {_MAX_OPEN_OR_CREATE_ATTEMPTS} attempts"
        ) from lost_race

    async def _prepare_and_confirm_or_cancel(
        self, reservation: Reservation
    ) -> Registration:
        """Prepare a reserved partition's storage and confirm the reservation, cancelling it if either raises or the creation is cancelled.

        The cancel acts only on a pending partition, so a confirmation that
        committed before its failure or cancellation was observed stands. A
        partition whose reservation the registry cannot cancel stays pending
        until it is deleted.
        """
        try:
            await self._prepare_partition_incarnation(reservation.incarnation)
            return await reservation.confirm()
        except BaseException:
            # Shielded, so a cancelled creation still frees the name. The task
            # reports its own failure, since a creation cancelled again stops
            # awaiting it before it ends.
            cancellation = asyncio.create_task(reservation.cancel())
            _reservation_cancellations.add(cancellation)

            def finish(task: asyncio.Task[None]) -> None:
                _reservation_cancellations.discard(task)
                if not task.cancelled() and task.exception() is not None:
                    logger.exception(
                        "Could not cancel the reservation of partition %r after "
                        "its creation failed or was cancelled; it stays pending "
                        "until deleted",
                        reservation.partition_name,
                        exc_info=task.exception(),
                    )

            cancellation.add_done_callback(finish)
            with contextlib.suppress(Exception):
                await asyncio.shield(cancellation)
            raise

    async def _prepare_partition_incarnation(self, incarnation: UUID) -> None:
        """Prepare every store's storage for a reserved partition incarnation."""
        await _on_every_store(
            "prepare",
            incarnation,
            [
                self._params.segment_store.prepare_partition_incarnation(
                    incarnation, SegmentStorePartitionConfig()
                ),
                self._params.vector_store.prepare_partition_incarnation(incarnation),
            ],
        )

    async def delete_partition(self, partition_name: str) -> None:
        """
        Delete the partition. Idempotent.

        The partition is unreachable once its name is unregistered, and its
        incarnation's records await the purge.

        Args:
            partition_name (str):
                The name of the partition to delete.
        """
        await self._params.partition_registry.unregister(partition_name)

    @staticmethod
    async def purge_partition_incarnation(
        segment_store: SegmentStore,
        vector_store: VectorStore,
        incarnation: UUID,
    ) -> bool:
        """
        Purge a bounded batch of a deleted partition incarnation's records from every store.

        The purge round of event memory's partition namespace: the registry
        calls it once the incarnation's retention has passed, until it
        returns False. Every store finishes its batch before the round's
        outcome is decided, so the stores that answer keep purging while
        another is down; the round then raises the first store's error.

        Args:
            segment_store (SegmentStore):
                The segment store holding the incarnation's segments.
            vector_store (VectorStore):
                The vector store holding the incarnation's derivatives.
            incarnation (UUID):
                The incarnation of a deleted partition.

        Returns:
            bool:
                Whether any store found anything of the incarnation's.
        """
        found = await _on_every_store(
            "purge",
            incarnation,
            [
                segment_store.purge_partition_incarnation(incarnation),
                vector_store.purge_partition_incarnation(incarnation),
            ],
        )
        return any(found)

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

    Reads and writes only its partition's life in each store, and fences
    each operation by the life's registration: an operation begun after the
    partition was deleted raises PartitionIncarnationEndedError, whether or
    not a partition was created again under the name since. Obtained from
    `EventMemory.open_or_create_partition`.
    """

    def __init__(
        self,
        params: EventMemoryParams,
        *,
        registration: Registration,
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
            registration (Registration):
                The registration of the partition's life.
            segment_store_partition (SegmentStorePartition):
                The partition's segment store partition.
            vector_store_partition (VectorStorePartition):
                The partition's vector store partition.
            schema_fields (frozenset[str]):
                The properties the vector store declares.
            similarity_metric (SimilarityMetric):
                The metric the vector store scores matches by.
        """
        self._registration = registration
        self._segment_store_partition = segment_store_partition
        self._vector_store_partition = vector_store_partition
        self._segmenter = params.segmenter
        self._deriver = params.deriver
        self._embedder = params.embedder
        self._reranker = params.reranker

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

    async def encode_events(
        self,
        events: Iterable[Event],
        *,
        format_options: FormatOptions | None = None,
    ) -> None:
        """
        Encode events.

        Args:
            events (Iterable[Event]): The events to encode.
            format_options (FormatOptions | None):
                Options for formatting.
                (default: None).

        Raises:
            ValueError:
                If any event supplies an illegal property key, or one of
                the memory's own reserved keys.
            PartitionIncarnationEndedError:
                If the partition was deleted before the events were stored,
                or while they were.
        """
        async with self._tracker("encode_events"):
            # Checked before and after the writes: a write that completed
            # under a life that died meanwhile raises instead of reporting
            # success, and the purge reclaims what it wrote.
            await self._registration.require_current()
            await self._encode_events(events, format_options=format_options)
            await self._registration.require_current()

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

        derivative_texts: list[str] = []
        for derivative in derivatives:
            text = _extract_text(derivative.block)
            if text is None:
                raise NotImplementedError(
                    f"Unsupported block type: {type(derivative.block).__name__}"
                )
            derivative_texts.append(text)

        derivative_embeddings = await self._embedder.ingest_embed(derivative_texts)
        t_embedding = time.monotonic()

        await self._segment_store_partition.add_segments(
            {
                segment: [derivative.uuid for derivative in segment_derivatives]
                for segment, segment_derivatives in segments_to_derivatives.items()
            }
        )
        t_segment_store = time.monotonic()

        derivative_records = [
            self._build_derivative_record(derivative, derivative_embedding)
            for derivative, derivative_embedding in zip(
                derivatives,
                derivative_embeddings,
                strict=True,
            )
        ]

        if derivative_records:
            await self._vector_store_partition.upsert(records=derivative_records)
        t_vector_store = time.monotonic()

        phase_durations = {
            "segmentation": t_segmentation - t_start,
            "derivation": t_derivation - t_segmentation,
            "embedding": t_embedding - t_derivation,
            "segment_store": t_segment_store - t_embedding,
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

    def _build_derivative_record(
        self,
        derivative: Derivative,
        derivative_embedding: Sequence[float],
    ) -> Record:
        """Build a vector record from a derivative and its embedding.

        The record carries the reserved timestamp and the properties the
        vector store declares. Every other property stays on the
        segment: the segment store holds them all and evaluates the whole
        filter (see `query`).
        """
        properties: dict[str, PropertyValue] = {
            EVENT_TIMESTAMP_KEY: derivative.timestamp,
            **{
                key: value
                for key, value in derivative.properties.items()
                if key in self._schema_fields
            },
        }

        return Record(
            uuid=derivative.uuid,
            vector=list(derivative_embedding),
            properties=properties,
        )

    @classmethod
    def _to_vector_record_property(cls, field: str) -> str:
        """
        Translates canonical filter field name to vector record property.

        Event memory base properties (`foo`) translate to their reserved keys.
        User-defined properties (`m.foo` / `metadata.foo`) translate to `foo`.
        Any other name translates unchanged.
        """
        internal_name, is_user_metadata = normalize_filter_field(field)
        if is_user_metadata:
            return demangle_user_metadata_key(internal_name)
        if field == "timestamp":
            return EVENT_TIMESTAMP_KEY
        return field

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

        Raises:
            PartitionIncarnationEndedError:
                If the partition was deleted before the query began.

        """
        async with self._tracker("query"):
            # A read in flight when the deletion commits answers from before
            # it, since the purge waits out a retention, so it is checked
            # only before.
            await self._registration.require_current()
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
        for match in query_result.matches:
            segment_uuid = segments_by_derivatives.get(match.record_uuid)
            if segment_uuid is None:
                # The derivative's segment is gone; its vector outlived it.
                continue
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

        Raises:
            PartitionIncarnationEndedError:
                If the partition was deleted before the events were
                forgotten, or while they were.
        """
        event_uuids = set(event_uuids)
        if not event_uuids:
            return

        async with self._tracker("forget_events"):
            # Checked after the deletes only: a delete adds nothing a purge
            # must reclaim, and reaches only this life's records.
            await self._forget_events(event_uuids)
            await self._registration.require_current()

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
            await self._vector_store_partition.delete(record_uuids=derivative_uuids)

        await self._segment_store_partition.delete_segments(
            segment_uuids=segment_uuids,
        )


async def _on_every_store[T](
    step: str, incarnation: UUID, calls: Sequence[Awaitable[T]]
) -> list[T]:
    """Run one step of a partition incarnation's lifecycle on every store at once, and raise the first error once every store has finished."""
    results = await asyncio.gather(*calls, return_exceptions=True)
    errors = [result for result in results if isinstance(result, BaseException)]
    for error in errors:
        logger.error(
            "A store failed to %s partition incarnation %s",
            step,
            incarnation,
            exc_info=error,
        )
    if errors:
        raise errors[0]
    return [result for result in results if not isinstance(result, BaseException)]


def _extract_text(block: Block) -> str | None:
    """Extract text from a block, if it contains text."""
    match block:
        case TextBlock(text=text):
            return text
        case _:
            return None
