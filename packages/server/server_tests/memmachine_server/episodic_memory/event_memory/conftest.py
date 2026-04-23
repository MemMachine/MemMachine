"""Shared fakes and fixtures for event memory tests."""

from collections import defaultdict
from collections.abc import Iterable, Mapping
from typing import override
from uuid import UUID

import pytest

from memmachine_server.common.filter.filter_parser import (
    FilterExpr,
    demangle_user_metadata_key,
    map_filter_fields,
    normalize_filter_field,
)
from memmachine_server.common.reranker import Reranker
from memmachine_server.common.vector_store.data_types import (
    VectorStoreCollectionConfig,
)
from memmachine_server.episodic_memory.event_memory.data_types import Segment
from memmachine_server.episodic_memory.event_memory.event_memory import (
    EventMemory,
    EventMemoryParams,
)
from memmachine_server.episodic_memory.event_memory.segment_store import (
    SegmentStorePartition,
    SegmentStorePartitionConfig,
)
from server_tests.memmachine_server.common.reranker.fake_embedder import (
    FakeEmbedder,
)
from server_tests.memmachine_server.common.vector_store.in_memory_vector_store_collection import (
    InMemoryVectorStoreCollection,
    evaluate_filter,
)

# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class InMemorySegmentStorePartition(SegmentStorePartition):
    """Minimal in-memory segment store partition for testing."""

    def __init__(
        self,
        config: SegmentStorePartitionConfig | None = None,
    ) -> None:
        self._config = config or SegmentStorePartitionConfig()
        self.segments: dict[UUID, Segment] = {}
        self.segment_order: list[UUID] = []
        self.event_to_segments: dict[UUID, list[UUID]] = defaultdict(list)
        self.segment_to_derivatives: dict[UUID, list[UUID]] = {}

    @override
    @property
    def config(self) -> SegmentStorePartitionConfig:
        return self._config

    @override
    async def add_segments(
        self,
        segments_to_derivative_uuids: Mapping[Segment, Iterable[UUID]],
    ) -> None:
        for segment, derivative_uuids in segments_to_derivative_uuids.items():
            self.segments[segment.uuid] = segment
            self.segment_order.append(segment.uuid)
            self.event_to_segments[segment.event_uuid].append(segment.uuid)
            self.segment_to_derivatives[segment.uuid] = list(derivative_uuids)

    @override
    async def get_segment_contexts(
        self,
        seed_segment_uuids: Iterable[UUID],
        *,
        max_backward_segments: int = 0,
        max_forward_segments: int = 0,
        property_filter: FilterExpr | None = None,
    ) -> dict[UUID, list[Segment]]:
        # Normalize canonical field names (e.g. "m.color" → "color") to
        # match the raw keys stored in segment.properties.
        normalized_filter = (
            map_filter_fields(property_filter, self._normalize_segment_field)
            if property_filter is not None
            else None
        )
        result: dict[UUID, list[Segment]] = {}
        for seed_uuid in seed_segment_uuids:
            if seed_uuid not in self.segments:
                continue
            try:
                pos = self.segment_order.index(seed_uuid)
            except ValueError:
                continue
            start = max(0, pos - max_backward_segments)
            end = min(len(self.segment_order), pos + max_forward_segments + 1)
            context = [
                self.segments[uid]
                for uid in self.segment_order[start:end]
                if uid in self.segments
                and (
                    normalized_filter is None
                    or evaluate_filter(normalized_filter, self.segments[uid].properties)
                )
            ]
            if context:
                result[seed_uuid] = context
        return result

    @staticmethod
    def _normalize_segment_field(field: str) -> str:
        """Translate canonical filter field names to raw segment property keys."""
        internal_name, is_user_metadata = normalize_filter_field(field)
        if is_user_metadata:
            return demangle_user_metadata_key(internal_name)
        return field

    @override
    async def get_segment_uuids_by_event_uuids(
        self,
        event_uuids: Iterable[UUID],
    ) -> dict[UUID, list[UUID]]:
        result: dict[UUID, list[UUID]] = {}
        for event_uuid in event_uuids:
            segment_uuids = self.event_to_segments.get(event_uuid)
            if segment_uuids:
                result[event_uuid] = list(segment_uuids)
        return result

    @override
    async def get_derivative_uuids_by_segment_uuids(
        self,
        segment_uuids: Iterable[UUID],
    ) -> dict[UUID, list[UUID]]:
        result: dict[UUID, list[UUID]] = {}
        for segment_uuid in segment_uuids:
            derivative_uuids = self.segment_to_derivatives.get(segment_uuid)
            if derivative_uuids:
                result[segment_uuid] = list(derivative_uuids)
        return result

    @override
    async def delete_segments(
        self,
        segment_uuids: Iterable[UUID],
    ) -> None:
        for segment_uuid in set(segment_uuids):
            segment = self.segments.pop(segment_uuid, None)
            if segment is None:
                continue
            self.segment_order = [
                uid for uid in self.segment_order if uid != segment_uuid
            ]
            event_list = self.event_to_segments.get(segment.event_uuid)
            if event_list is not None:
                event_list[:] = [uid for uid in event_list if uid != segment_uuid]
                if not event_list:
                    del self.event_to_segments[segment.event_uuid]
            self.segment_to_derivatives.pop(segment_uuid, None)


class FakeReranker(Reranker):
    """Reranker that scores by candidate string length."""

    async def score(self, query: str, candidates: list[str]) -> list[float]:
        return [float(len(c)) for c in candidates]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def fake_embedder():
    return FakeEmbedder()


@pytest.fixture
def fake_segment_store_partition():
    return InMemorySegmentStorePartition()


@pytest.fixture
def fake_vector_store_collection(fake_embedder):
    config = VectorStoreCollectionConfig(
        vector_dimensions=fake_embedder.dimensions,
        similarity_metric=fake_embedder.similarity_metric,
        properties_schema={
            **EventMemory.expected_vector_store_collection_schema(),
            "color": str,
        },
    )
    return InMemoryVectorStoreCollection(config)


@pytest.fixture
def event_memory(
    fake_vector_store_collection,
    fake_segment_store_partition,
    fake_embedder,
):
    return EventMemory(
        EventMemoryParams(
            vector_store_collection=fake_vector_store_collection,
            segment_store_partition=fake_segment_store_partition,
            embedder=fake_embedder,
        )
    )


@pytest.fixture
def event_memory_with_reranker(
    fake_vector_store_collection,
    fake_segment_store_partition,
    fake_embedder,
):
    return EventMemory(
        EventMemoryParams(
            vector_store_collection=fake_vector_store_collection,
            segment_store_partition=fake_segment_store_partition,
            embedder=fake_embedder,
            reranker=FakeReranker(),
        )
    )


@pytest.fixture
def event_memory_with_sentences(
    fake_vector_store_collection,
    fake_segment_store_partition,
    fake_embedder,
):
    return EventMemory(
        EventMemoryParams(
            vector_store_collection=fake_vector_store_collection,
            segment_store_partition=fake_segment_store_partition,
            embedder=fake_embedder,
            derive_sentences=True,
        )
    )


@pytest.fixture
def event_memory_with_eviction(
    fake_vector_store_collection,
    fake_segment_store_partition,
    fake_embedder,
):
    return EventMemory(
        EventMemoryParams(
            vector_store_collection=fake_vector_store_collection,
            segment_store_partition=fake_segment_store_partition,
            embedder=fake_embedder,
            eviction_similarity_threshold=0.5,
            eviction_target_size=3,
            eviction_search_limit=20,
        )
    )
