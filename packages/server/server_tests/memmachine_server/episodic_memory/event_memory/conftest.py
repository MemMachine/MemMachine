"""Shared fakes and fixtures for event memory tests."""

import math
from collections import defaultdict
from collections.abc import Iterable, Mapping
from datetime import datetime
from typing import Any, override
from uuid import UUID

import pytest

from memmachine_server.common.filter import (
    FilterExpr,
    map_filter_fields,
)
from memmachine_server.common.filter.filter_parser import (
    demangle_user_metadata_key,
    normalize_filter_field,
)
from memmachine_server.common.reranker import Reranker
from memmachine_server.common.vector_store.data_types import (
    VectorStoreCollectionConfig,
)
from memmachine_server.episodic_memory.event_memory.data_types import (
    EvictionOptions,
    Neighborhood,
    Segment,
)
from memmachine_server.episodic_memory.event_memory.deriver.text_deriver import (
    SentenceTextDeriver,
    WholeTextDeriver,
)
from memmachine_server.episodic_memory.event_memory.event_memory import (
    EventMemory,
    EventMemoryParams,
)
from memmachine_server.episodic_memory.event_memory.segment_store import (
    SegmentStorePartition,
    SegmentStorePartitionConfig,
)
from memmachine_server.episodic_memory.event_memory.segmenter.text_segmenter import (
    TextSegmenter,
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


def _order_key(segment: Segment) -> tuple:
    # The store's total order, with the ungrouped (null session) stream first.
    return (
        segment.session_id is not None,
        segment.session_id or "",
        segment.timestamp,
        segment.event_uuid,
        segment.index,
        segment.offset,
    )


class InMemorySegmentStorePartition(SegmentStorePartition):
    """Minimal in-memory segment store partition for testing.

    Mirrors the SQLAlchemy store's reads: one total order per session,
    windows confined to the seed's session, the seed filtered for a
    context and never returned for a neighborhood.
    """

    def __init__(
        self,
        config: SegmentStorePartitionConfig | None = None,
    ) -> None:
        self._config = config or SegmentStorePartitionConfig()
        self.segments: dict[UUID, Segment] = {}
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
            self.event_to_segments[segment.event_uuid].append(segment.uuid)
            self.segment_to_derivatives[segment.uuid] = list(derivative_uuids)

    def _ordered(self) -> list[Segment]:
        return sorted(self.segments.values(), key=_order_key)

    @staticmethod
    def _passes(
        segment: Segment,
        *,
        since: datetime | None,
        until: datetime | None,
        source_ids: list[str] | None,
        block_kinds: list[str] | None,
        normalized_filter: FilterExpr | None,
    ) -> bool:
        if since is not None and segment.timestamp < since:
            return False
        if until is not None and segment.timestamp >= until:
            return False
        if source_ids is not None and segment.source_id not in source_ids:
            return False
        if block_kinds is not None and segment.block.kind not in block_kinds:
            return False
        if normalized_filter is not None:
            evaluated = {**segment.properties, "timestamp": segment.timestamp}
            return evaluate_filter(normalized_filter, evaluated)
        return True

    def _window(
        self,
        seed: Segment,
        before: int,
        after: int,
        passes: Any,
    ) -> tuple[list[Segment], list[Segment]]:
        ordered = [s for s in self._ordered() if s.session_id == seed.session_id]
        position = next(i for i, s in enumerate(ordered) if s.uuid == seed.uuid)
        backward = (
            [s for s in ordered[:position] if passes(s)][-before:] if before > 0 else []
        )
        forward = [s for s in ordered[position + 1 :] if passes(s)][:after]
        return backward, forward

    def _read(
        self,
        seed_segment_uuids: Iterable[UUID],
        *,
        before: int,
        after: int,
        since: datetime | None,
        until: datetime | None,
        source_ids: Iterable[str] | None,
        block_kinds: Iterable[str] | None,
        property_filter: FilterExpr | None,
        seed_must_pass: bool,
    ) -> dict[UUID, tuple[list[Segment], Segment, list[Segment]]]:
        normalized_filter = (
            map_filter_fields(property_filter, self._normalize_segment_field)
            if property_filter is not None
            else None
        )
        source_ids = list(source_ids) if source_ids is not None else None
        block_kinds = list(block_kinds) if block_kinds is not None else None

        def passes(segment: Segment) -> bool:
            return self._passes(
                segment,
                since=since,
                until=until,
                source_ids=source_ids,
                block_kinds=block_kinds,
                normalized_filter=normalized_filter,
            )

        result: dict[UUID, tuple[list[Segment], Segment, list[Segment]]] = {}
        for seed_uuid in seed_segment_uuids:
            seed = self.segments.get(seed_uuid)
            if seed is None or (seed_must_pass and not passes(seed)):
                continue
            backward, forward = self._window(seed, before, after, passes)
            result[seed_uuid] = (backward, seed, forward)
        return result

    @override
    async def get_segment_contexts(
        self,
        seed_segment_uuids: Iterable[UUID],
        *,
        before: int = 0,
        after: int = 0,
        since: datetime | None = None,
        until: datetime | None = None,
        source_ids: Iterable[str] | None = None,
        block_kinds: Iterable[str] | None = None,
        property_filter: FilterExpr | None = None,
    ) -> dict[UUID, list[Segment]]:
        windows = self._read(
            seed_segment_uuids,
            before=before,
            after=after,
            since=since,
            until=until,
            source_ids=source_ids,
            block_kinds=block_kinds,
            property_filter=property_filter,
            seed_must_pass=True,
        )
        return {
            seed_uuid: [*backward, seed, *forward]
            for seed_uuid, (backward, seed, forward) in windows.items()
        }

    @override
    async def get_neighbourhoods(
        self,
        seed_segment_uuids: Iterable[UUID],
        *,
        before: int = 0,
        after: int = 0,
        since: datetime | None = None,
        until: datetime | None = None,
        source_ids: Iterable[str] | None = None,
        block_kinds: Iterable[str] | None = None,
        property_filter: FilterExpr | None = None,
    ) -> dict[UUID, Neighborhood]:
        windows = self._read(
            seed_segment_uuids,
            before=before,
            after=after,
            since=since,
            until=until,
            source_ids=source_ids,
            block_kinds=block_kinds,
            property_filter=property_filter,
            seed_must_pass=False,
        )
        return {
            seed_uuid: Neighborhood(before=backward, after=forward)
            for seed_uuid, (backward, _, forward) in windows.items()
        }

    @staticmethod
    def _normalize_segment_field(field: str) -> str:
        """Translate canonical filter field names to raw segment property keys.

        Mirrors SQLAlchemySegmentStorePartition._resolve_segment_field:
        - `timestamp` -> the segment's own timestamp.
        - `m.<key>` / `metadata.<key>` -> user metadata, bare key.
        - any other bare name -> system field, `_<field>` (matches the
          LongTermMemory event-backend property layout).
        """
        if field == "timestamp":
            return field
        internal_name, is_user_metadata = normalize_filter_field(field)
        if is_user_metadata:
            return demangle_user_metadata_key(internal_name)
        return f"_{field}"

    @override
    async def get_segment_uuids_by_event_uuids(
        self,
        event_uuids: Iterable[UUID],
    ) -> dict[UUID, list[UUID]]:
        result: dict[UUID, list[UUID]] = {}
        for event_uuid in event_uuids:
            segment_uuids = self.event_to_segments.get(event_uuid)
            if segment_uuids:
                result[event_uuid] = sorted(
                    segment_uuids,
                    key=lambda uid: (
                        self.segments[uid].index,
                        self.segments[uid].offset,
                    ),
                )
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
    async def delete_segments(
        self,
        segment_uuids: Iterable[UUID],
    ) -> None:
        for segment_uuid in set(segment_uuids):
            segment = self.segments.pop(segment_uuid, None)
            if segment is None:
                continue
            event_list = self.event_to_segments.get(segment.event_uuid)
            if event_list is not None:
                event_list[:] = [uid for uid in event_list if uid != segment_uuid]
                if not event_list:
                    del self.event_to_segments[segment.event_uuid]
            self.segment_to_derivatives.pop(segment_uuid, None)

    @override
    async def delete_derivatives(
        self,
        derivative_uuids: Iterable[UUID],
    ) -> None:
        derivative_uuids = set(derivative_uuids)
        for segment_uuid, linked in self.segment_to_derivatives.items():
            self.segment_to_derivatives[segment_uuid] = [
                uid for uid in linked if uid not in derivative_uuids
            ]


class FakeReranker(Reranker):
    """Reranker that scores by candidate string length."""

    async def score(self, query: str, candidates: list[str]) -> list[float]:
        return [float(len(c)) for c in candidates]


class AngleEmbedder(FakeEmbedder):
    """Embeds a text on the unit circle at the angle of the first token it carries.

    `FakeEmbedder` maps every text onto one direction, so under cosine every
    score is 1.0 and an ordering assertion is vacuous. This fake gives a
    text its own angle by a token it contains, so two texts sharing a token
    are one cluster (cosine 1.0), texts a quarter turn apart are unrelated
    (cosine 0.0), and a query ranks texts by angular distance.
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


def make_collection(embedder: FakeEmbedder) -> InMemoryVectorStoreCollection:
    """A collection declaring EventMemory's reserved keys and a `color` property."""
    return InMemoryVectorStoreCollection(
        VectorStoreCollectionConfig(vector_dimensions=embedder.dimensions),
        {**EventMemory.expected_vector_store_collection_schema(), "color": str},
    )


@pytest.fixture
def fake_embedder():
    return FakeEmbedder()


@pytest.fixture
def fake_segment_store_partition():
    return InMemorySegmentStorePartition()


@pytest.fixture
def fake_vector_store_collection(fake_embedder):
    return make_collection(fake_embedder)


@pytest.fixture
def event_memory(
    fake_vector_store_collection,
    fake_segment_store_partition,
    fake_embedder,
):
    return EventMemory(
        EventMemoryParams(
            segment_store_partition=fake_segment_store_partition,
            vector_store_collection=fake_vector_store_collection,
            segmenter=TextSegmenter(),
            deriver=WholeTextDeriver(),
            embedder=fake_embedder,
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
            segment_store_partition=fake_segment_store_partition,
            vector_store_collection=fake_vector_store_collection,
            segmenter=TextSegmenter(),
            deriver=SentenceTextDeriver(),
            embedder=fake_embedder,
        )
    )


@pytest.fixture
def event_memory_with_eviction(
    fake_vector_store_collection,
    fake_segment_store_partition,
    fake_embedder,
):
    # FakeEmbedder maps every text onto one direction, so all derivatives
    # are cosine-similar (1.0): any batch forms a single eviction cluster.
    return EventMemory(
        EventMemoryParams(
            segment_store_partition=fake_segment_store_partition,
            vector_store_collection=fake_vector_store_collection,
            segmenter=TextSegmenter(),
            deriver=WholeTextDeriver(),
            embedder=fake_embedder,
            eviction=EvictionOptions(
                similarity_threshold=0.5, search_limit=100, target_size=5
            ),
        )
    )
