"""Shared fakes and fixtures for event memory tests."""

import math
from collections.abc import Callable, Iterable, Mapping
from datetime import datetime
from typing import Any, override
from unittest.mock import create_autospec
from uuid import UUID, uuid4

import pytest
import pytest_asyncio

from memmachine_server.common.data_types import SimilarityMetric
from memmachine_server.common.embedder import Embedder
from memmachine_server.common.filter.filter_parser import (
    FilterExpr,
    demangle_user_metadata_key,
    map_filter_fields,
    normalize_filter_field,
)
from memmachine_server.common.metrics_factory import MetricsFactory
from memmachine_server.common.reranker import Reranker
from memmachine_server.common.vector_store import VectorStore, VectorStorePartition
from memmachine_server.episodic_memory.event_memory.data_types import (
    Neighborhood,
    Segment,
)
from memmachine_server.episodic_memory.event_memory.deriver import Deriver
from memmachine_server.episodic_memory.event_memory.deriver.text_deriver import (
    SentenceTextDeriver,
    WholeTextDeriver,
)
from memmachine_server.episodic_memory.event_memory.event_memory import (
    EventMemory,
    EventMemoryParams,
    EventMemoryPartition,
)
from memmachine_server.episodic_memory.event_memory.segment_store import (
    PendingDerivatives,
    SegmentStore,
    SegmentStoreEventAlreadyStoredError,
    SegmentStorePartition,
    SegmentStorePartitionConfig,
    Tombstone,
)
from memmachine_server.episodic_memory.event_memory.segmenter import Segmenter
from memmachine_server.episodic_memory.event_memory.segmenter.text_segmenter import (
    TextSegmenter,
)
from server_tests.memmachine_server.common.reranker.fake_embedder import (
    FakeEmbedder,
)
from server_tests.memmachine_server.common.vector_store.in_memory_vector_store_partition import (
    InMemoryVectorStorePartition,
    evaluate_filter,
)

# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


def _order_key(segment: Segment) -> tuple:
    # The store's total order.
    return (segment.timestamp, segment.event_uuid, segment.index, segment.offset)


class InMemorySegmentStorePartition(SegmentStorePartition):
    """Minimal in-memory segment store partition for testing.

    Mirrors the SQLAlchemy store's reads: one total order, a filtered
    lookup by uuid, and a walk around a given segment that never returns
    the segment.
    """

    def __init__(
        self,
        config: SegmentStorePartitionConfig | None = None,
    ) -> None:
        self._config = config or SegmentStorePartitionConfig()
        self.segments: dict[UUID, Segment] = {}
        self.event_to_segments: dict[UUID, list[UUID]] = {}
        self.segment_to_derivatives: dict[UUID, list[UUID]] = {}
        self.pending_derivatives: dict[UUID, PendingDerivatives] = {}
        self.tombstones: dict[UUID, Tombstone] = {}
        self.tombstones_awaiting_second_delete: set[UUID] = set()

    @override
    @property
    def config(self) -> SegmentStorePartitionConfig:
        return self._config

    @override
    async def add_events(
        self,
        segments_by_event: Mapping[UUID, Iterable[Segment]],
        pending_derivatives: Iterable[PendingDerivatives],
    ) -> None:
        segments_by_event = {
            event_uuid: list(segments)
            for event_uuid, segments in segments_by_event.items()
        }
        already_stored = segments_by_event.keys() & self.event_to_segments.keys()
        if already_stored:
            raise SegmentStoreEventAlreadyStoredError(already_stored)
        for event_uuid, segments in segments_by_event.items():
            self.event_to_segments[event_uuid] = []
            for segment in segments:
                self.segments[segment.uuid] = segment
                self.event_to_segments[event_uuid].append(segment.uuid)
                self.segment_to_derivatives[segment.uuid] = []
        for pending in pending_derivatives:
            self.pending_derivatives[pending.uuid] = pending
            for derivative in pending.derivatives:
                self.segment_to_derivatives[derivative.segment_uuid].append(
                    derivative.uuid
                )

    @override
    async def delete_pending_derivatives(
        self,
        pending_derivatives_uuids: Iterable[UUID],
    ) -> None:
        for pending_uuid in pending_derivatives_uuids:
            self.pending_derivatives.pop(pending_uuid, None)

    def _ordered(self) -> list[Segment]:
        return sorted(self.segments.values(), key=_order_key)

    @staticmethod
    def _passes(
        segment: Segment,
        *,
        since: datetime | None,
        until: datetime | None,
        source_ids: list[str] | None,
        normalized_filter: FilterExpr | None,
    ) -> bool:
        for name, bound in (("since", since), ("until", until)):
            if bound is not None and bound.tzinfo is None:
                raise ValueError(f"{name} must be timezone-aware: {bound!r}")
        if since is not None and segment.timestamp < since:
            return False
        if until is not None and segment.timestamp >= until:
            return False
        if source_ids is not None and segment.source_id not in source_ids:
            return False
        if normalized_filter is not None:
            evaluated = {**segment.properties, "timestamp": segment.timestamp}
            return evaluate_filter(normalized_filter, evaluated)
        return True

    @staticmethod
    def _normalize_segment_field(field: str) -> str:
        """Translate canonical filter field names to raw segment property keys.

        Mirrors SQLAlchemySegmentStorePartition._resolve_segment_field:
        - `timestamp` -> the segment's own timestamp.
        - `m.<key>` / `metadata.<key>` -> the property, bare key.
        - any other name -> ValueError.
        """
        if field == "timestamp":
            return field
        internal_name, is_user_metadata = normalize_filter_field(field)
        if not is_user_metadata:
            raise ValueError(f"Filter field {field!r} names no segment property.")
        return demangle_user_metadata_key(internal_name)

    def _admits(
        self,
        *,
        since: datetime | None,
        until: datetime | None,
        source_ids: Iterable[str] | None,
        property_filter: FilterExpr | None,
    ) -> Callable[[Segment], bool]:
        normalized_filter = (
            map_filter_fields(property_filter, self._normalize_segment_field)
            if property_filter is not None
            else None
        )
        listed_sources = list(source_ids) if source_ids is not None else None

        def passes(segment: Segment) -> bool:
            return self._passes(
                segment,
                since=since,
                until=until,
                source_ids=listed_sources,
                normalized_filter=normalized_filter,
            )

        return passes

    @override
    async def get_segments(
        self,
        segment_uuids: Iterable[UUID],
        *,
        since: datetime | None = None,
        until: datetime | None = None,
        source_ids: Iterable[str] | None = None,
        property_filter: FilterExpr | None = None,
    ) -> dict[UUID, Segment]:
        passes = self._admits(
            since=since,
            until=until,
            source_ids=source_ids,
            property_filter=property_filter,
        )
        found: dict[UUID, Segment] = {}
        for segment_uuid in segment_uuids:
            segment = self.segments.get(segment_uuid)
            if segment is not None and passes(segment):
                found[segment_uuid] = segment
        return found

    @override
    async def get_segment_neighborhoods(
        self,
        seed_uuids: Iterable[UUID],
        *,
        before: int = 0,
        after: int = 0,
        since: datetime | None = None,
        until: datetime | None = None,
        source_ids: Iterable[str] | None = None,
        property_filter: FilterExpr | None = None,
    ) -> dict[UUID, Neighborhood]:
        if before < 0:
            raise ValueError(f"before must be nonnegative: {before}")
        if after < 0:
            raise ValueError(f"after must be nonnegative: {after}")
        passes = self._admits(
            since=since,
            until=until,
            source_ids=source_ids,
            property_filter=property_filter,
        )
        neighborhoods: dict[UUID, Neighborhood] = {}
        for seed_uuid in seed_uuids:
            seed = self.segments.get(seed_uuid)
            if seed is None:
                continue
            key = _order_key(seed)
            walk = self._ordered()
            backward = [s for s in walk if _order_key(s) < key and passes(s)]
            forward = [s for s in walk if _order_key(s) > key and passes(s)]
            neighborhoods[seed.uuid] = Neighborhood(
                before=backward[-before:] if before > 0 else [],
                after=forward[:after],
            )
        return neighborhoods

    @override
    async def get_segment_uuids_by_derivative_uuids(
        self,
        derivative_uuids: Iterable[UUID],
    ) -> dict[UUID, UUID]:
        wanted = set(derivative_uuids)
        return {
            derivative_uuid: segment_uuid
            for segment_uuid, owned in self.segment_to_derivatives.items()
            for derivative_uuid in owned
            if derivative_uuid in wanted
        }

    @override
    async def forget_events(
        self,
        event_uuids: Iterable[UUID],
    ) -> Tombstone | None:
        held_event_uuids = set(event_uuids) & self.event_to_segments.keys()
        derivative_uuids: list[UUID] = []
        for event_uuid in held_event_uuids:
            for segment_uuid in self.event_to_segments.pop(event_uuid):
                self.segments.pop(segment_uuid)
                derivative_uuids.extend(self.segment_to_derivatives.pop(segment_uuid))
        for pending_uuid, pending in list(self.pending_derivatives.items()):
            if pending.event_uuid in held_event_uuids:
                del self.pending_derivatives[pending_uuid]
        if not derivative_uuids:
            return None
        tombstone = Tombstone(uuid=uuid4(), derivative_uuids=derivative_uuids)
        self.tombstones[tombstone.uuid] = tombstone
        return tombstone

    @override
    async def record_tombstone_first_delete(self, tombstone_uuid: UUID) -> None:
        if tombstone_uuid in self.tombstones:
            self.tombstones_awaiting_second_delete.add(tombstone_uuid)

    @override
    async def require_current(self) -> None:
        return


class FakeReranker(Reranker):
    """Reranker that scores by candidate string length."""

    async def score(self, query: str, candidates: list[str]) -> list[float]:
        return [float(len(c)) for c in candidates]


async def open_event_memory_partition(
    *,
    segment_store_partition: SegmentStorePartition,
    vector_store_partition: VectorStorePartition,
    segmenter: Segmenter,
    deriver: Deriver,
    embedder: Embedder,
    metrics_factory: MetricsFactory | None = None,
) -> EventMemoryPartition:
    """Open a partition of an EventMemory whose stores hand out the given partitions."""
    segment_store = create_autospec(SegmentStore, instance=True)
    segment_store.open_or_create_partition.return_value = segment_store_partition
    vector_store = create_autospec(VectorStore, instance=True)
    vector_store.open_or_create_partition.return_value = vector_store_partition
    event_memory = EventMemory(
        EventMemoryParams(
            segment_store=segment_store,
            vector_store=vector_store,
            segmenter=segmenter,
            deriver=deriver,
            embedder=embedder,
            metrics_factory=metrics_factory,
        )
    )
    return await event_memory.open_or_create_partition("partition")


class AngleEmbedder(FakeEmbedder):
    """Embeds a text on the unit circle at the angle of the first token it carries.

    `FakeEmbedder` maps every text onto one direction, so under cosine every
    score is 1.0 and an ordering assertion is vacuous. This fake gives a
    text its own angle by a token it contains, so two texts sharing a token
    are one cluster (cosine 1.0), texts a quarter turn apart are unrelated
    (cosine 0.0), and a query ranks texts by angular distance. The vectors
    are unit vectors, so every similarity metric ranks them by angle.
    """

    def __init__(
        self,
        token_angles: Mapping[str, float],
        *,
        default_angle: float = math.pi / 4,
    ) -> None:
        super().__init__()
        self._token_angles = dict(token_angles)
        self._default_angle = default_angle

    def _vector(self, text: Any) -> list[float]:
        angle = next(
            (
                angle
                for token, angle in self._token_angles.items()
                if token in str(text)
            ),
            self._default_angle,
        )
        return [math.cos(angle), math.sin(angle)]

    @override
    async def _ingest_embed(
        self,
        inputs: list[Any],
        max_attempts: int = 1,
    ) -> list[list[float]]:
        return [self._vector(text) for text in inputs]

    @override
    async def _search_embed(
        self,
        queries: list[Any],
        max_attempts: int = 1,
    ) -> list[list[float]]:
        return [self._vector(query) for query in queries]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def make_partition(
    similarity_metric: SimilarityMetric = SimilarityMetric.COSINE,
) -> InMemoryVectorStorePartition:
    """A partition declaring EventMemory's reserved keys and one property.

    It scores by `similarity_metric`. A declared property is copied into
    the records, and anything else stays with the segment.
    """
    return InMemoryVectorStorePartition(
        similarity_metric=similarity_metric,
        indexed_properties={
            **EventMemory.expected_vector_store_collection_schema(),
            "episode_uid": str,
        },
    )


@pytest.fixture
def fake_embedder():
    return FakeEmbedder()


@pytest.fixture
def fake_segment_store_partition():
    return InMemorySegmentStorePartition()


@pytest.fixture
def fake_vector_store_partition(fake_embedder):
    return make_partition(fake_embedder.similarity_metric)


@pytest_asyncio.fixture
async def event_memory_partition(
    fake_vector_store_partition,
    fake_segment_store_partition,
    fake_embedder,
):
    return await open_event_memory_partition(
        segment_store_partition=fake_segment_store_partition,
        vector_store_partition=fake_vector_store_partition,
        segmenter=TextSegmenter(),
        deriver=WholeTextDeriver(),
        embedder=fake_embedder,
    )


@pytest_asyncio.fixture
async def event_memory_partition_with_sentences(
    fake_vector_store_partition,
    fake_segment_store_partition,
    fake_embedder,
):
    return await open_event_memory_partition(
        segment_store_partition=fake_segment_store_partition,
        vector_store_partition=fake_vector_store_partition,
        segmenter=TextSegmenter(),
        deriver=SentenceTextDeriver(),
        embedder=fake_embedder,
    )
