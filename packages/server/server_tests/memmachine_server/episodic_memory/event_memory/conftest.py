"""Shared fakes and fixtures for event memory tests."""

import asyncio
import math
from collections import defaultdict
from collections.abc import AsyncIterator, Callable, Iterable, Mapping
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Any, override
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
from memmachine_server.episodic_memory.event_memory.event_memory_store import (
    EventMemoryStoreEventAlreadyStoredError,
    EventMemoryStorePartition,
    EventMemoryStorePartitionConfig,
    EventMemoryStorePartitionWriter,
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
    # The store's total order.
    return (segment.timestamp, segment.event_uuid, segment.index, segment.offset)


class InMemoryEventMemoryStorePartition(EventMemoryStorePartition):
    """Minimal in-memory event memory store partition for testing.

    Mirrors the SQLAlchemy store's reads: one total order, a filtered
    lookup by uuid, and a walk around a given segment within its session
    that never returns the segment.
    """

    def __init__(
        self,
        config: EventMemoryStorePartitionConfig | None = None,
    ) -> None:
        self._config = config or EventMemoryStorePartitionConfig()
        self.events: set[UUID] = set()
        self.segments: dict[UUID, Segment] = {}
        self.event_to_segments: dict[UUID, list[UUID]] = defaultdict(list)
        self.segment_to_derivatives: dict[UUID, list[UUID]] = {}
        # The fence: shared writes count, an exclusive one waits for zero
        # and excludes new ones, as the registry row lock does.
        self._writes_in_flight = 0
        self._exclusive_held = False
        self._fence = asyncio.Condition()

    @override
    @property
    def config(self) -> EventMemoryStorePartitionConfig:
        return self._config

    @override
    @asynccontextmanager
    async def write(
        self, *, exclusive: bool = False
    ) -> AsyncIterator[EventMemoryStorePartitionWriter]:
        async with self._fence:
            await self._fence.wait_for(
                lambda: (
                    not self._exclusive_held
                    and (not exclusive or self._writes_in_flight == 0)
                )
            )
            if exclusive:
                self._exclusive_held = True
            else:
                self._writes_in_flight += 1
        writer = InMemoryEventMemoryStorePartitionWriter(self)
        try:
            yield writer
            writer.apply()
        finally:
            async with self._fence:
                if exclusive:
                    self._exclusive_held = False
                else:
                    self._writes_in_flight -= 1
                self._fence.notify_all()

    def _ordered(self) -> list[Segment]:
        return sorted(self.segments.values(), key=_order_key)

    @staticmethod
    def _passes(
        segment: Segment,
        *,
        since: datetime | None,
        until: datetime | None,
        session_ids: list[str] | None,
        source_ids: list[str] | None,
        block_kinds: list[str] | None,
        normalized_filter: FilterExpr | None,
    ) -> bool:
        for name, bound in (("since", since), ("until", until)):
            if bound is not None and bound.tzinfo is None:
                raise ValueError(f"{name} must be timezone-aware: {bound!r}")
        if since is not None and segment.timestamp < since:
            return False
        if until is not None and segment.timestamp >= until:
            return False
        if session_ids is not None and segment.session_id not in session_ids:
            return False
        if source_ids is not None and segment.source_id not in source_ids:
            return False
        if block_kinds is not None and segment.block.block_type not in block_kinds:
            return False
        if normalized_filter is not None:
            evaluated = {**segment.properties, "timestamp": segment.timestamp}
            return evaluate_filter(normalized_filter, evaluated)
        return True

    @staticmethod
    def _normalize_segment_field(field: str) -> str:
        """Translate canonical filter field names to raw segment property keys.

        Mirrors SQLAlchemyEventMemoryStorePartition._resolve_segment_field:
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

    def _admits(
        self,
        *,
        since: datetime | None,
        until: datetime | None,
        session_ids: Iterable[str] | None,
        source_ids: Iterable[str] | None,
        block_kinds: Iterable[str] | None,
        property_filter: FilterExpr | None,
    ) -> Callable[[Segment], bool]:
        normalized_filter = (
            map_filter_fields(property_filter, self._normalize_segment_field)
            if property_filter is not None
            else None
        )
        listed_sessions = list(session_ids) if session_ids is not None else None
        listed_sources = list(source_ids) if source_ids is not None else None
        listed_kinds = list(block_kinds) if block_kinds is not None else None

        def passes(segment: Segment) -> bool:
            return self._passes(
                segment,
                since=since,
                until=until,
                session_ids=listed_sessions,
                source_ids=listed_sources,
                block_kinds=listed_kinds,
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
        session_ids: Iterable[str] | None = None,
        source_ids: Iterable[str] | None = None,
        block_kinds: Iterable[str] | None = None,
        property_filter: FilterExpr | None = None,
    ) -> dict[UUID, Segment]:
        passes = self._admits(
            since=since,
            until=until,
            session_ids=session_ids,
            source_ids=source_ids,
            block_kinds=block_kinds,
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
        block_kinds: Iterable[str] | None = None,
        property_filter: FilterExpr | None = None,
    ) -> dict[UUID, Neighborhood]:
        if before < 0:
            raise ValueError(f"before must be nonnegative: {before}")
        if after < 0:
            raise ValueError(f"after must be nonnegative: {after}")
        passes = self._admits(
            since=since,
            until=until,
            session_ids=None,
            source_ids=source_ids,
            block_kinds=block_kinds,
            property_filter=property_filter,
        )
        neighborhoods: dict[UUID, Neighborhood] = {}
        for seed_uuid in seed_uuids:
            seed = self.segments.get(seed_uuid)
            if seed is None:
                continue
            key = _order_key(seed)
            walk = [s for s in self._ordered() if s.session_id == seed.session_id]
            backward = [s for s in walk if _order_key(s) < key and passes(s)]
            forward = [s for s in walk if _order_key(s) > key and passes(s)]
            neighborhoods[seed.uuid] = Neighborhood(
                before=backward[-before:] if before > 0 else [],
                after=forward[:after],
            )
        return neighborhoods

    @override
    async def get_derivative_uuids_by_event_uuids(
        self,
        event_uuids: Iterable[UUID],
    ) -> dict[UUID, list[UUID]]:
        return {
            event_uuid: [
                derivative_uuid
                for segment_uuid in self.event_to_segments.get(event_uuid, [])
                for derivative_uuid in self.segment_to_derivatives.get(segment_uuid, [])
            ]
            for event_uuid in event_uuids
            if event_uuid in self.events
        }

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
    async def delete_events(
        self,
        event_uuids: Iterable[UUID],
    ) -> None:
        for event_uuid in set(event_uuids):
            self.events.discard(event_uuid)
            for segment_uuid in self.event_to_segments.pop(event_uuid, []):
                self.segments.pop(segment_uuid, None)
                self.segment_to_derivatives.pop(segment_uuid, None)

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


class InMemoryEventMemoryStorePartitionWriter(EventMemoryStorePartitionWriter):
    """Stages a write; the partition applies it when the block exits normally."""

    def __init__(self, partition: InMemoryEventMemoryStorePartition) -> None:
        self._partition = partition
        self._staged: dict[UUID, dict[Segment, list[UUID]]] = {}
        self._unlinked: set[UUID] = set()

    @override
    async def add_events(
        self,
        events: Mapping[UUID, Mapping[Segment, Iterable[UUID]]],
    ) -> None:
        events = {
            event_uuid: {
                segment: list(derivative_uuids)
                for segment, derivative_uuids in segments.items()
            }
            for event_uuid, segments in events.items()
        }
        for event_uuid, segments in events.items():
            for segment in segments:
                if segment.event_uuid != event_uuid:
                    raise ValueError(
                        f"segment {segment.uuid} names event {segment.event_uuid}, "
                        f"listed under {event_uuid}"
                    )
        already_stored = {
            event_uuid
            for event_uuid in events
            if event_uuid in self._partition.events or event_uuid in self._staged
        }
        if already_stored:
            raise EventMemoryStoreEventAlreadyStoredError(already_stored)
        for event_uuid, segments in events.items():
            for segment in segments:
                if segment.uuid in self._partition.segments:
                    raise ValueError(f"segment {segment.uuid} is already stored")
            self._staged[event_uuid] = segments

    @override
    async def delete_derivatives(
        self,
        derivative_uuids: Iterable[UUID],
    ) -> None:
        self._unlinked |= set(derivative_uuids)

    @override
    async def get_segment_uuids_by_derivative_uuids(
        self,
        derivative_uuids: Iterable[UUID],
    ) -> dict[UUID, UUID]:
        return await self._partition.get_segment_uuids_by_derivative_uuids(
            derivative_uuids
        )

    def apply(self) -> None:
        """Commit the staged events and unlinkings into the partition."""
        for event_uuid, segments in self._staged.items():
            self._partition.events.add(event_uuid)
            for segment, derivative_uuids in segments.items():
                self._partition.segments[segment.uuid] = segment
                self._partition.event_to_segments[event_uuid].append(segment.uuid)
                self._partition.segment_to_derivatives[segment.uuid] = derivative_uuids
        self._staged = {}
        if self._unlinked:
            for segment_uuid, linked in self._partition.segment_to_derivatives.items():
                self._partition.segment_to_derivatives[segment_uuid] = [
                    derivative_uuid
                    for derivative_uuid in linked
                    if derivative_uuid not in self._unlinked
                ]
            self._unlinked = set()


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
        VectorStoreCollectionConfig(
            vector_dimensions=embedder.dimensions,
            indexed_properties_schema={
                **EventMemory.expected_vector_store_collection_schema(),
                "color": str,
            },
        )
    )


@pytest.fixture
def fake_embedder():
    return FakeEmbedder()


@pytest.fixture
def fake_event_memory_store_partition():
    return InMemoryEventMemoryStorePartition()


@pytest.fixture
def fake_vector_store_collection(fake_embedder):
    return make_collection(fake_embedder)


@pytest.fixture
def event_memory(
    fake_vector_store_collection,
    fake_event_memory_store_partition,
    fake_embedder,
):
    return EventMemory(
        EventMemoryParams(
            event_memory_store_partition=fake_event_memory_store_partition,
            vector_store_collection=fake_vector_store_collection,
            segmenter=TextSegmenter(),
            deriver=WholeTextDeriver(),
            embedder=fake_embedder,
        )
    )


@pytest.fixture
def event_memory_with_sentences(
    fake_vector_store_collection,
    fake_event_memory_store_partition,
    fake_embedder,
):
    return EventMemory(
        EventMemoryParams(
            event_memory_store_partition=fake_event_memory_store_partition,
            vector_store_collection=fake_vector_store_collection,
            segmenter=TextSegmenter(),
            deriver=SentenceTextDeriver(),
            embedder=fake_embedder,
        )
    )


@pytest.fixture
def event_memory_with_eviction(
    fake_vector_store_collection,
    fake_event_memory_store_partition,
    fake_embedder,
):
    # FakeEmbedder maps every text onto one direction, so all derivatives
    # are cosine-similar (1.0): any batch forms a single eviction cluster.
    return EventMemory(
        EventMemoryParams(
            event_memory_store_partition=fake_event_memory_store_partition,
            vector_store_collection=fake_vector_store_collection,
            segmenter=TextSegmenter(),
            deriver=WholeTextDeriver(),
            embedder=fake_embedder,
            eviction=EvictionOptions(
                cosine_similarity_threshold=0.5, search_limit=100, target_size=5
            ),
        )
    )
