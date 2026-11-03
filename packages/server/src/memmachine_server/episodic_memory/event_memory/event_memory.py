"""Event memory system for storing and retrieving events."""

import asyncio
import datetime
import json
import logging
import time
from collections.abc import Iterable, Mapping, Sequence
from typing import Final, cast
from uuid import UUID

from pydantic import BaseModel, Field, InstanceOf

from memmachine_server.common.data_types import (
    PropertyType,
    PropertyValue,
    SimilarityMetric,
)
from memmachine_server.common.embedder import Embedder
from memmachine_server.common.filter import (
    FilterExpr,
    Property,
    SystemField,
    extract_filter,
    filter_fields,
    map_filter_fields,
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
    Record,
    VectorStore,
    VectorStorePartition,
)
from memmachine_server.common.vector_store.data_types import is_valid_identifier
from memmachine_server.common.vector_store.utils import bind_properties

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
        reranker (Reranker | None):
            Reranker instance for scoring search results.
            If None, embedding similarity scores are used instead
            (default: None).
        metrics_factory (MetricsFactory | None):
            An instance of MetricsFactory for collecting usage metrics
            (default: None).
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

    A partition key names one partition in each store, and must be valid
    in both. Handles on a partition read and write that pair only.
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
        self._indexed_properties = store_schema.indexed_properties

        missing_base_fields = (
            _BASE_EVENT_MEMORY_FIELD_NAMES - self._indexed_properties.keys()
        )
        if missing_base_fields:
            raise ValueError(
                f"Vector store schema missing fields required by EventMemory: "
                f"{', '.join(sorted(missing_base_fields))}"
            )

        self._similarity_metric = store_schema.similarity_metric

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
            indexed_properties=self._indexed_properties,
            similarity_metric=self._similarity_metric,
        )

    async def delete_partition(self, partition_key: str) -> None:
        """
        Delete the partition from both stores. Idempotent.

        Args:
            partition_key (str):
                The key of the partition to delete.
        """
        await self._params.vector_store.delete_partition(partition_key)
        await self._params.segment_store.delete_partition(partition_key)

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
        indexed_properties: Mapping[str, PropertyType],
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
            indexed_properties (Mapping[str, PropertyType]):
                The properties the vector store declares, each to its type.
            similarity_metric (SimilarityMetric):
                The metric the vector store scores matches by.
        """
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

        self._indexed_properties = indexed_properties
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

    def _validate_events(self, events: Iterable[Event]) -> list[Event]:
        """
        Validate a batch of events before encoding: the events encoded in their place.

        Raises ValueError if any event supplies a property key outside the
        naming contract or under the memory's own reserved keys. A key
        reserved for another system belongs to the component that embeds
        the memory and is kept as given. Each property the vector store
        declares is bound to its declared type (`bind_properties`), so the
        segment store holds the value the vector record holds; a value its
        type does not hold raises PropertyTypeMismatchError before anything
        is written.
        """
        events = list(events)
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

        return [
            event.model_copy(
                update={
                    "properties": bind_properties(
                        event.properties, self._indexed_properties
                    )
                }
            )
            for event in events
        ]

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
            PropertyTypeMismatchError:
                If an event's property the vector store declares holds a
                value its declared type does not, an int counting as a float.
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

        events = self._validate_events(events)

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
                if key in self._indexed_properties
            },
        }

        return Record(
            uuid=derivative.uuid,
            vector=list(derivative_embedding),
            properties=properties,
        )

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
            ValueError:
                If the filter names a system field the memory does not
                define.

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

        # A vector record carries the properties the vector store declares, so
        # a condition naming any other field has nothing to match there. The
        # vector store gets the part of the filter on declared fields, which
        # only ever widens its search; the segment store, which holds every
        # property, gets the whole filter below and narrows it back.
        vector_store_filter = None
        if property_filter is not None:
            vector_record_filter = map_filter_fields(
                property_filter, self._vector_record_field
            )
            vector_store_filter = extract_filter(
                vector_record_filter,
                allowed_fields={
                    field
                    for field in filter_fields(vector_record_filter)
                    if isinstance(field, Property)
                    and field.key in self._indexed_properties
                },
            )

        query_embedding = (
            await self._embedder.search_embed(
                [query],
            )
        )[0]
        t_embedding = time.monotonic()

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

    @staticmethod
    def _vector_record_field(field: SystemField | Property) -> Property:
        """The vector record property a filter's field compares: an event memory base field's reserved key, or the property itself."""
        match field:
            case SystemField(name="timestamp"):
                return Property(EVENT_TIMESTAMP_KEY, datetime.datetime)
            case Property():
                return field
            case SystemField(name=name):
                raise ValueError(
                    f"EventMemory filters on properties and the timestamp only, "
                    f"not the system field {name!r}"
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
            await self._vector_store_partition.delete(record_uuids=derivative_uuids)

        await self._segment_store_partition.delete_segments(
            segment_uuids=segment_uuids,
        )


def _extract_text(block: Block) -> str | None:
    """Extract text from a block, if it contains text."""
    match block:
        case TextBlock(text=text):
            return text
        case _:
            return None
