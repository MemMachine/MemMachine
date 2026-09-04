"""Timeline access over the event-backed LongTermMemory.

Exercises the segment-level reads -- search, expand, outline, and address
abbreviation -- against a real LongTermMemory wired to the in-memory
collection and segment store, so the assertions cover the actual store walk
rather than a stubbed one.
"""

from datetime import UTC, datetime, timedelta
from unittest.mock import create_autospec

import pytest

from memmachine_server.common.episode_store import Episode, EpisodeStorage
from memmachine_server.common.filter.filter_parser import parse_filter
from memmachine_server.common.reranker import Reranker
from memmachine_server.common.vector_graph_store import VectorGraphStore
from memmachine_server.common.vector_store import VectorStore
from memmachine_server.common.vector_store.data_types import (
    VectorStoreCollectionConfig,
)
from memmachine_server.episodic_memory.event_memory.deriver.text_deriver import (
    WholeTextDeriver,
)
from memmachine_server.episodic_memory.event_memory.event_memory import EventMemory
from memmachine_server.episodic_memory.event_memory.segment_store import SegmentStore
from memmachine_server.episodic_memory.event_memory.segmenter.passthrough_segmenter import (
    PassthroughSegmenter,
)
from memmachine_server.episodic_memory.long_term_memory import (
    EVENT_BACKEND_SYSTEM_FIELDS,
    DeclarativeBackendParams,
    EventBackendParams,
    LongTermMemory,
)
from server_tests.memmachine_server.common.reranker.fake_embedder import FakeEmbedder
from server_tests.memmachine_server.common.vector_store.in_memory_vector_store_collection import (
    InMemoryVectorStoreCollection,
)
from server_tests.memmachine_server.episodic_memory.event_memory.conftest import (
    InMemorySegmentStorePartition,
)

pytestmark = pytest.mark.asyncio

BASE_TIME = datetime(2026, 1, 15, 12, 0, tzinfo=UTC)


def _episode(
    uid: str, content: str, *, minute: int, producer: str = "alice"
) -> Episode:
    return Episode(
        uid=uid,
        content=content,
        session_key="sess1",
        created_at=BASE_TIME + timedelta(minutes=minute),
        producer_id=producer,
        producer_role="user",
        sequence_num=minute,
        filterable_metadata={"source": producer},
    )


@pytest.fixture
def episodes() -> list[Episode]:
    return [
        _episode("ep-1", "first thing said", minute=0),
        _episode("ep-2", "a reply to it", minute=1, producer="bob"),
        _episode("ep-3", "the middle of the conversation", minute=2),
        _episode("ep-4", "another reply", minute=3, producer="bob"),
        _episode("ep-5", "the last thing said", minute=4),
    ]


@pytest.fixture
def fake_embedder() -> FakeEmbedder:
    return FakeEmbedder()


@pytest.fixture
def segment_store_partition() -> InMemorySegmentStorePartition:
    return InMemorySegmentStorePartition()


@pytest.fixture
def long_term_memory(fake_embedder, segment_store_partition) -> LongTermMemory:
    collection = InMemoryVectorStoreCollection(
        VectorStoreCollectionConfig(
            vector_dimensions=fake_embedder.dimensions,
            similarity_metric=fake_embedder.similarity_metric,
            indexed_properties_schema={
                **EventMemory.expected_vector_store_collection_schema(),
                **EVENT_BACKEND_SYSTEM_FIELDS,
            },
        )
    )
    return LongTermMemory(
        EventBackendParams(
            session_id="sess1",
            vector_store=create_autospec(VectorStore, instance=True),
            vector_store_collection=collection,
            vector_store_collection_namespace="long_term_memory",
            segment_store=create_autospec(SegmentStore, instance=True),
            segment_store_partition=segment_store_partition,
            partition_key="sess1",
            episode_storage=create_autospec(EpisodeStorage, instance=True),
            embedder=fake_embedder,
            segmenter=PassthroughSegmenter(),
            deriver=WholeTextDeriver(),
        ),
    )


async def _seed(long_term_memory: LongTermMemory, episodes: list[Episode]) -> list:
    await long_term_memory.add_episodes(episodes)
    headers = await long_term_memory.outline_timeline()
    return headers


async def test_outline_reports_events_in_timeline_order(long_term_memory, episodes):
    headers = await _seed(long_term_memory, episodes)

    assert len(headers) == len(episodes)
    assert [header.timestamp for header in headers] == sorted(
        header.timestamp for header in headers
    )
    assert all(header.segment_count >= 1 for header in headers)
    assert all(header.encoded_length > 0 for header in headers)


async def test_outline_windows_around_an_anchor(long_term_memory, episodes):
    headers = await _seed(long_term_memory, episodes)
    anchor = headers[2]

    earlier = await long_term_memory.outline_timeline(
        end=(anchor.timestamp, anchor.event_uuid), limit=2, descending=True
    )
    later = await long_term_memory.outline_timeline(
        start=(anchor.timestamp, anchor.event_uuid), limit=2
    )

    assert [header.event_uuid for header in earlier] == [
        headers[1].event_uuid,
        anchor.event_uuid,
    ]
    assert [header.event_uuid for header in later] == [
        anchor.event_uuid,
        headers[3].event_uuid,
    ]


async def test_outline_filters_to_one_speaker(long_term_memory, episodes):
    await _seed(long_term_memory, episodes)

    headers = await long_term_memory.outline_timeline(
        property_filter=parse_filter("m.source = 'bob'")
    )

    assert len(headers) == 2


async def test_expand_returns_neighbours_without_the_seed(long_term_memory, episodes):
    headers = await _seed(long_term_memory, episodes)
    seed_uuid = headers[2].first_segment_uuid

    neighbours = await long_term_memory.expand_timeline(
        seed_uuid, before=1, after=1, unit="segments"
    )

    assert seed_uuid not in {segment.uuid for segment in neighbours}
    assert {segment.event_uuid for segment in neighbours} == {
        headers[1].event_uuid,
        headers[3].event_uuid,
    }


async def test_expand_by_events_excludes_the_seeds_whole_event(
    long_term_memory, episodes
):
    headers = await _seed(long_term_memory, episodes)
    seed_uuid = headers[0].first_segment_uuid

    neighbours = await long_term_memory.expand_timeline(
        seed_uuid, before=0, after=2, unit="events"
    )

    assert headers[0].event_uuid not in {segment.event_uuid for segment in neighbours}
    assert {segment.event_uuid for segment in neighbours} == {
        headers[1].event_uuid,
        headers[2].event_uuid,
    }


async def test_expand_locates_a_seed_the_filter_excludes(long_term_memory, episodes):
    headers = await _seed(long_term_memory, episodes)
    # headers[1] is bob's; the filter keeps only alice's, so the seed itself
    # fails it and must still work as an address.
    seed_uuid = headers[1].first_segment_uuid

    neighbours = await long_term_memory.expand_timeline(
        seed_uuid,
        before=1,
        after=1,
        unit="segments",
        property_filter=parse_filter("m.source = 'alice'"),
    )

    assert neighbours
    assert seed_uuid not in {segment.uuid for segment in neighbours}


async def test_addresses_abbreviate_to_something_that_resolves(
    long_term_memory, episodes
):
    headers = await _seed(long_term_memory, episodes)
    segment_uuids = [header.first_segment_uuid for header in headers]

    handles = await long_term_memory.abbreviate_segment_addresses(segment_uuids)

    assert len(set(handles.values())) == len(segment_uuids)
    for segment_uuid, handle in handles.items():
        assert handle
        assert segment_uuid.hex.startswith(handle)
        assert await long_term_memory.resolve_segment_address(handle, limit=5) == [
            segment_uuid
        ]


async def test_search_returns_addressable_seeds(long_term_memory, episodes):
    await _seed(long_term_memory, episodes)

    result = await long_term_memory.search_timeline(
        "the middle of the conversation", limit=3
    )

    assert result.scored_segment_contexts
    for scored in result.scored_segment_contexts:
        [resolved] = await long_term_memory.resolve_segment_address(
            scored.seed_segment_uuid.hex, limit=2
        )
        assert resolved == scored.seed_segment_uuid


async def test_search_honours_a_supplied_query_vector(
    long_term_memory, episodes, fake_embedder
):
    await _seed(long_term_memory, episodes)
    [vector] = await fake_embedder.search_embed(["anything"])

    result = await long_term_memory.search_timeline(
        "unused text", limit=3, query_vector=vector
    )

    assert result.scored_segment_contexts


async def test_timeline_access_rejects_the_declarative_backend(fake_embedder):
    declarative = LongTermMemory(
        DeclarativeBackendParams(
            session_id="sess1",
            vector_graph_store=create_autospec(VectorGraphStore, instance=True),
            embedder=fake_embedder,
            reranker=create_autospec(Reranker, instance=True),
        ),
    )

    with pytest.raises(ValueError, match="event backend"):
        await declarative.outline_timeline()

    with pytest.raises(ValueError, match="event backend"):
        await declarative.resolve_segment_address("ab", limit=1)


async def test_unknown_filter_field_is_rejected(long_term_memory, episodes):
    await _seed(long_term_memory, episodes)

    with pytest.raises(ValueError, match="Unknown filter field"):
        await long_term_memory.outline_timeline(
            property_filter=parse_filter("not_a_field = 'x'")
        )
