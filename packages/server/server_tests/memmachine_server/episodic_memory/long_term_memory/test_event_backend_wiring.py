"""End-to-end wiring test for the event-backed LongTermMemory.

Builds a LongTermMemory(EventBackendParams(...)) using:
- the in-memory vector_store collection from event_memory tests
- the in-memory segment_store partition from event_memory tests
- a fake embedder
- a fake EpisodeStorage that satisfies the get_episode(uid) lookup used during
  search_scored hydration.

Verifies that add_episodes / search_scored / delete_episodes /
drop_session_partition all dispatch correctly through the event backend.
"""

from datetime import UTC, datetime
from typing import override
from unittest.mock import create_autospec

import pytest

from memmachine_server.common.episode_store import (
    Episode,
    EpisodeEntry,
    EpisodeIdT,
    EpisodeStorage,
)
from memmachine_server.common.filter.filter_parser import (
    Comparison as FilterComparison,
)
from memmachine_server.common.vector_store import VectorStore
from memmachine_server.common.vector_store.data_types import (
    VectorStoreCollectionConfig,
)
from memmachine_server.episodic_memory.event_memory.deriver.text_deriver import (
    WholeTextDeriver,
)
from memmachine_server.episodic_memory.event_memory.event_memory import EventMemory
from memmachine_server.episodic_memory.event_memory.segment_store import (
    SegmentStore,
)
from memmachine_server.episodic_memory.event_memory.segmenter.passthrough_segmenter import (
    PassthroughSegmenter,
)
from memmachine_server.episodic_memory.long_term_memory import (
    EVENT_BACKEND_SYSTEM_FIELDS,
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


class FakeEpisodeStorage(EpisodeStorage):
    """In-memory EpisodeStorage; only get_episode is exercised here."""

    def __init__(self, episodes: dict[str, Episode]):
        self._episodes = dict(episodes)

    @override
    async def startup(self) -> None: ...

    @override
    async def delete_all(self) -> None:
        self._episodes.clear()

    @override
    async def add_episodes(
        self, session_key: str, episodes: list[EpisodeEntry]
    ) -> list[Episode]:
        raise NotImplementedError

    @override
    async def get_episode(self, episode_id: EpisodeIdT) -> Episode | None:
        return self._episodes.get(episode_id)

    @override
    async def get_episode_messages(self, **kwargs) -> list[Episode]:
        raise NotImplementedError

    @override
    async def get_episode_messages_count(self, **kwargs) -> int:
        raise NotImplementedError

    @override
    async def get_episode_ids(self, **kwargs) -> list[EpisodeIdT]:
        raise NotImplementedError

    @override
    async def delete_episodes(self, episode_ids: list[EpisodeIdT]) -> None:
        for uid in episode_ids:
            self._episodes.pop(uid, None)

    @override
    async def delete_episode_messages(self, **kwargs) -> None:
        raise NotImplementedError


def _episode(uid: str, content: str, *, producer_id: str = "alice") -> Episode:
    return Episode(
        uid=uid,
        content=content,
        session_key="sess1",
        created_at=datetime(2026, 1, 15, 12, 0, tzinfo=UTC),
        producer_id=producer_id,
        producer_role="user",
        sequence_num=0,
    )


@pytest.fixture
def episodes() -> list[Episode]:
    return [
        _episode("ep-1", "the mitochondria is the powerhouse"),
        _episode("ep-2", "george washington was the first president"),
        _episode("ep-3", "lorem ipsum dolor sit amet"),
    ]


@pytest.fixture
def fake_episode_storage(episodes) -> FakeEpisodeStorage:
    return FakeEpisodeStorage({e.uid: e for e in episodes})


@pytest.fixture
def fake_embedder() -> FakeEmbedder:
    return FakeEmbedder()


@pytest.fixture
def vector_store():
    """Stand-in for the parent VectorStore: only delete_collection is invoked."""
    return create_autospec(VectorStore, instance=True)


@pytest.fixture
def vector_store_collection(fake_embedder):
    config = VectorStoreCollectionConfig(
        vector_dimensions=fake_embedder.dimensions,
        similarity_metric=fake_embedder.similarity_metric,
        indexed_properties_schema={
            **EventMemory.expected_vector_store_collection_schema(),
            **EVENT_BACKEND_SYSTEM_FIELDS,
        },
    )
    return InMemoryVectorStoreCollection(config)


@pytest.fixture
def segment_store():
    """Stand-in for the parent SegmentStore: only delete_partition is invoked."""
    return create_autospec(SegmentStore, instance=True)


@pytest.fixture
def segment_store_partition() -> InMemorySegmentStorePartition:
    return InMemorySegmentStorePartition()


@pytest.fixture
def long_term_memory(
    fake_embedder,
    vector_store,
    vector_store_collection,
    segment_store,
    segment_store_partition,
    fake_episode_storage,
) -> LongTermMemory:
    return LongTermMemory(
        EventBackendParams(
            session_id="sess1",
            vector_store=vector_store,
            vector_store_collection=vector_store_collection,
            vector_store_collection_namespace="long_term_memory",
            segment_store=segment_store,
            segment_store_partition=segment_store_partition,
            partition_key="sess1",
            episode_storage=fake_episode_storage,
            embedder=fake_embedder,
            segmenter=PassthroughSegmenter(),
            deriver=WholeTextDeriver(),
        ),
    )


async def test_add_then_search_returns_full_episodes(long_term_memory, episodes):
    await long_term_memory.add_episodes(episodes)

    # FakeEmbedder maps query length -> vector; the longest content scores best.
    scored = await long_term_memory.search_scored(
        "george washington",
        num_episodes_limit=3,
    )
    returned = [ep.uid for _, ep in scored]
    assert set(returned) <= {e.uid for e in episodes}
    # All returned items are full Episode objects (not segments).
    for _, ep in scored:
        assert isinstance(ep, Episode)
        assert ep.content  # round-tripped from the episode store


async def test_search_dedupes_by_episode_uid(
    long_term_memory,
    fake_episode_storage,
    episodes,
):
    """Even if a single episode produces multiple segments/derivatives, only
    one tuple per episode_uid is returned."""
    await long_term_memory.add_episodes(episodes)
    scored = await long_term_memory.search_scored(
        "powerhouse",
        num_episodes_limit=10,
    )
    uids = [ep.uid for _, ep in scored]
    assert len(uids) == len(set(uids))


async def test_delete_episodes_removes_from_event_memory(
    long_term_memory,
    segment_store_partition,
    episodes,
):
    await long_term_memory.add_episodes(episodes)
    # Sanity: 3 events, each with 1 segment under PassthroughSegmenter.
    assert len(segment_store_partition.segments) == 3

    await long_term_memory.delete_episodes(["ep-1"])

    # ep-1's segment should be gone; the others should remain.
    assert len(segment_store_partition.segments) == 2
    # Map back: ep-1's event_uuid is uuid5(NS, "ep-1"); easier to assert by
    # checking the *_episode_uid* property on remaining segments.
    remaining_episode_uids = {
        s.properties["_episode_uid"] for s in segment_store_partition.segments.values()
    }
    assert "ep-1" not in remaining_episode_uids


async def test_drop_session_partition_calls_parent_lifecycle_hooks(
    long_term_memory,
    vector_store,
    segment_store,
):
    await long_term_memory.drop_session_partition()
    vector_store.delete_collection.assert_awaited_once_with(
        namespace="long_term_memory",
        name="sess1",
    )
    segment_store.delete_partition.assert_awaited_once_with("sess1")


async def test_user_metadata_filter_round_trips(
    long_term_memory,
    fake_episode_storage,
):
    """`m.<field>` filter on the client-API translates to bare field on storage."""
    episodes = [
        Episode(
            uid="m-1",
            content="apple",
            session_key="sess1",
            created_at=datetime(2026, 1, 15, 12, 0, tzinfo=UTC),
            producer_id="alice",
            producer_role="user",
            filterable_metadata={"color": "red"},
        ),
        Episode(
            uid="m-2",
            content="banana",
            session_key="sess1",
            created_at=datetime(2026, 1, 15, 12, 1, tzinfo=UTC),
            producer_id="alice",
            producer_role="user",
            filterable_metadata={"color": "yellow"},
        ),
    ]
    fake_episode_storage._episodes.update({e.uid: e for e in episodes})
    await long_term_memory.add_episodes(episodes)

    scored = await long_term_memory.search_scored(
        "fruit",
        num_episodes_limit=10,
        property_filter=FilterComparison(field="m.color", op="=", value="red"),
    )
    uids = {ep.uid for _, ep in scored}
    assert uids == {"m-1"}


async def test_close_is_a_noop(long_term_memory):
    # Should not raise.
    await long_term_memory.close()
