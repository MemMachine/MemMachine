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

import logging
import math
from collections.abc import Iterable
from datetime import UTC, datetime
from typing import Any, override
from unittest.mock import create_autospec
from uuid import uuid4

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
from memmachine_server.episodic_memory.event_memory.data_types import (
    Neighborhood,
    QueryHit,
    Segment,
    TextBlock,
)
from memmachine_server.episodic_memory.event_memory.deriver import Deriver
from memmachine_server.episodic_memory.event_memory.deriver.text_deriver import (
    WholeTextDeriver,
)
from memmachine_server.episodic_memory.event_memory.event_memory import EventMemory
from memmachine_server.episodic_memory.event_memory.segment_store import (
    SegmentStore,
)
from memmachine_server.episodic_memory.event_memory.segmenter import Segmenter
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
    async def get_episodes(self, episode_ids: Iterable[EpisodeIdT]) -> list[Episode]:
        return [self._episodes[uid] for uid in episode_ids if uid in self._episodes]

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
        indexed_properties_schema={
            **EventMemory.expected_vector_store_collection_schema(),
            **EVENT_BACKEND_SYSTEM_FIELDS,
        },
    )
    return InMemoryVectorStoreCollection(config)


@pytest.fixture
def segment_store():
    """Stand-in for the parent SegmentStore lifecycle methods."""
    store = create_autospec(SegmentStore, instance=True)
    store.purge_deleted_partitions.return_value = False
    return store


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
            segmenter=Segmenter(),
            deriver=Deriver([WholeTextDeriver()]),
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


async def test_search_warns_on_index_storage_drift(
    long_term_memory,
    fake_episode_storage,
    episodes,
    caplog,
):
    """If the event index references an episode UID that EpisodeStorage no
    longer has (index/storage drift), the dropped UID is logged as a warning
    and the remaining episodes are still returned."""

    await long_term_memory.add_episodes(episodes)
    # Simulate drift: index keeps ep-2's segment, but EpisodeStorage forgets it.
    await fake_episode_storage.delete_episodes(["ep-2"])

    with caplog.at_level(
        logging.WARNING,
        logger="memmachine_server.episodic_memory.long_term_memory.long_term_memory",
    ):
        scored = await long_term_memory.search_scored(
            "george washington",
            num_episodes_limit=3,
        )

    returned_uids = {ep.uid for _, ep in scored}
    assert "ep-2" not in returned_uids
    assert returned_uids <= {"ep-1", "ep-3"}

    drift_records = [
        r for r in caplog.records if "index/storage drift" in r.getMessage()
    ]
    assert drift_records, "expected a drift warning"
    assert "ep-2" in drift_records[0].getMessage()


async def test_delete_episodes_removes_from_event_memory(
    long_term_memory,
    segment_store_partition,
    episodes,
):
    await long_term_memory.add_episodes(episodes)
    # Sanity: 3 events, each with 1 segment under a segmenter with no handler.
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
    # Reclamation is the sweeper's; the delete path never purges.
    segment_store.purge_deleted_partitions.assert_not_awaited()


async def test_event_backend_unusable_after_drop_session_partition(
    long_term_memory,
    episodes,
):
    """After dropping the partition, the EventMemory handles point at deleted
    resources. Reusing the LongTermMemory must fail loudly rather than
    silently operate on a stale (or recreated) collection.
    """
    await long_term_memory.drop_session_partition()
    with pytest.raises(RuntimeError, match="drop_session_partition"):
        await long_term_memory.add_episodes(episodes)
    with pytest.raises(RuntimeError, match="drop_session_partition"):
        await long_term_memory.search_scored("anything", num_episodes_limit=1)
    with pytest.raises(RuntimeError, match="drop_session_partition"):
        await long_term_memory.delete_episodes(["ep-1"])


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


async def test_system_field_filter_round_trips(
    long_term_memory,
    fake_episode_storage,
):
    """Bare client-API field (`producer_id`) translates to storage key `_producer_id`.

    EventMemory translates the filter consistently for both vector_store and
    segment_store stages so a system-field filter actually narrows results.
    """
    episodes = [
        Episode(
            uid="s-1",
            content="alice msg",
            session_key="sess1",
            created_at=datetime(2026, 1, 15, 12, 0, tzinfo=UTC),
            producer_id="alice",
            producer_role="user",
        ),
        Episode(
            uid="s-2",
            content="bob msg",
            session_key="sess1",
            created_at=datetime(2026, 1, 15, 12, 1, tzinfo=UTC),
            producer_id="bob",
            producer_role="user",
        ),
    ]
    fake_episode_storage._episodes.update({e.uid: e for e in episodes})
    await long_term_memory.add_episodes(episodes)

    scored = await long_term_memory.search_scored(
        "msg",
        num_episodes_limit=10,
        property_filter=FilterComparison(field="producer_id", op="=", value="alice"),
    )
    uids = {ep.uid for _, ep in scored}
    assert uids == {"s-1"}


async def test_close_is_a_noop(long_term_memory):
    # Should not raise.
    await long_term_memory.close()


async def test_unknown_bare_filter_field_raises(long_term_memory):
    """Typo'd bare system field surfaces as ValueError, not silent empty.

    Without this guard, segment store / vector store would treat the unknown
    name as an exact JSON property lookup and silently return zero results.
    """
    with pytest.raises(ValueError, match="Unknown filter field 'producre_id'"):
        await long_term_memory.search_scored(
            "msg",
            num_episodes_limit=10,
            property_filter=FilterComparison(
                field="producre_id", op="=", value="alice"
            ),
        )


async def test_unknown_user_metadata_field_passes_when_no_schema(long_term_memory):
    """With empty `user_property_keys`, any `m.<x>` is accepted.

    The default fixture leaves `properties_schema` unset, so validation is
    permissive on user metadata. Matches the documented behavior in
    `_validate_event_backend_filter`.
    """
    # Doesn't raise.
    scored = await long_term_memory.search_scored(
        "msg",
        num_episodes_limit=10,
        property_filter=FilterComparison(field="m.anything", op="=", value="x"),
    )
    assert scored == []


async def test_unknown_user_metadata_field_raises_when_schema_configured(
    fake_embedder,
    vector_store,
    vector_store_collection,
    segment_store,
    segment_store_partition,
    fake_episode_storage,
):
    """With a configured schema, typo'd `m.<x>` surfaces as ValueError."""
    ltm = LongTermMemory(
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
            segmenter=Segmenter(),
            deriver=Deriver([WholeTextDeriver()]),
            user_property_keys=frozenset({"color"}),
        ),
    )
    with pytest.raises(
        ValueError, match=r"Unknown user-metadata filter field 'm\.coloor'"
    ):
        await ltm.search_scored(
            "msg",
            num_episodes_limit=10,
            property_filter=FilterComparison(field="m.coloor", op="=", value="red"),
        )


async def test_timestamp_filter_field_is_accepted(long_term_memory, episodes):
    """`timestamp` is a valid bare filter field (segment store has it as a column)."""
    await long_term_memory.add_episodes(episodes)
    # Doesn't raise; whether anything matches depends on the embedder/score path.
    await long_term_memory.search_scored(
        "anything",
        num_episodes_limit=10,
        property_filter=FilterComparison(
            field="timestamp",
            op=">=",
            value=datetime(2000, 1, 1, tzinfo=UTC),
        ),
    )


def _make_ltm(episodes: list[Episode]) -> LongTermMemory:
    """Build a self-contained LongTermMemory, bypassing the shared fixtures.

    No reranker is configured, so scores come straight from the vector store.
    """
    fake_embedder = FakeEmbedder()
    vector_store_collection = InMemoryVectorStoreCollection(
        VectorStoreCollectionConfig(
            vector_dimensions=fake_embedder.dimensions,
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
            vector_store_collection=vector_store_collection,
            vector_store_collection_namespace="long_term_memory",
            segment_store=create_autospec(SegmentStore, instance=True),
            segment_store_partition=InMemorySegmentStorePartition(),
            partition_key="sess1",
            episode_storage=FakeEpisodeStorage({e.uid: e for e in episodes}),
            embedder=fake_embedder,
            segmenter=Segmenter(),
            deriver=Deriver([WholeTextDeriver()]),
        ),
    )


async def test_score_threshold_drops_low_scores_under_cosine():
    """Cosine: higher score = better match. With FakeEmbedder, "abc" → [3,-3]
    and "abc def" → [7,-7] are colinear and score ~1.0 each. A threshold
    above 1 should drop everything; default None should keep everything.
    """
    episodes = [
        _episode("near", "abc"),
        _episode("far", "abcdefghij"),
    ]
    ltm = _make_ltm(episodes)
    await ltm.add_episodes(episodes)

    kept_all = await ltm.search_scored("abc", num_episodes_limit=10)
    assert {ep.uid for _, ep in kept_all} == {"near", "far"}

    kept_none = await ltm.search_scored(
        "abc", num_episodes_limit=10, score_threshold=2.0
    )
    assert kept_none == []


def _timeline_episode(uid: str, content: str, minute: int) -> Episode:
    return Episode(
        uid=uid,
        content=content,
        session_key="sess1",
        created_at=datetime(2026, 1, 15, 12, minute, tzinfo=UTC),
        producer_id="alice",
        producer_role="user",
        sequence_num=0,
    )


# `FakeEmbedder` maps text to `[len(text), -len(text)]`, so under cosine every
# document scores exactly 1.0 against every query. That tie makes every stored
# episode a seed of equal rank, which hides whether context expansion
# contributed anything: a search at `num_episodes_limit=N` returns the first N
# episodes in store order whether or not the windows are folded in.
#
# The expansion tests below give each episode its own cosine similarity instead, by an
# explicit search rank. The rank order is chosen so that the timeline
# neighbours of the one matching episode are the LEAST similar of all, which is
# what lets the tests assert on the contract ("expansion returns timeline
# neighbours the search itself would not return") rather than on a particular
# ranking: any correct top-k leaves those neighbours out, and any nonzero
# window around the match reaches at least one of them, whatever the
# backward/forward split.
_TIMELINE_LENGTH = 7
_MATCH_INDEX = 3
# Timeline index -> search rank (0 = most similar). The match ranks first, the
# two episodes farthest from it next, its four neighbours last.
_SEARCH_RANK_BY_INDEX = {3: 0, 0: 1, 6: 2, 2: 3, 4: 4, 1: 5, 5: 6}
_NEIGHBOUR_UIDS = frozenset({"tl-1", "tl-2", "tl-4", "tl-5"})


def _timeline_token(index: int) -> str:
    return f"tok-{index}"


class RankedEmbedder(FakeEmbedder):
    """Embeds each timeline episode at its own distance from the query.

    Every text carries exactly one `tok-<index>`; the vector is placed at an
    angle proportional to that episode's search rank, so cosine similarity is
    strictly decreasing in the rank and no two episodes tie.
    """

    @override
    async def _ingest_embed(
        self,
        inputs: list[Any],
        max_attempts: int = 1,
    ) -> list[list[float]]:
        return [RankedEmbedder._vector(text) for text in inputs]

    @override
    async def _search_embed(
        self,
        queries: list[Any],
        max_attempts: int = 1,
    ) -> list[list[float]]:
        return [RankedEmbedder._vector(query) for query in queries]

    @staticmethod
    def _vector(text: Any) -> list[float]:
        rank = next(
            (
                _SEARCH_RANK_BY_INDEX[index]
                for index in range(_TIMELINE_LENGTH)
                if _timeline_token(index) in str(text)
            ),
            _TIMELINE_LENGTH,
        )
        angle = rank * (math.pi / 2) / _TIMELINE_LENGTH
        return [math.cos(angle), math.sin(angle)]


@pytest.fixture
def timeline_episodes() -> list[Episode]:
    return [
        _timeline_episode(
            f"tl-{index}",
            f"timeline message {_timeline_token(index)}",
            index,
        )
        for index in range(_TIMELINE_LENGTH)
    ]


@pytest.fixture
def timeline_storage(timeline_episodes) -> FakeEpisodeStorage:
    return FakeEpisodeStorage({e.uid: e for e in timeline_episodes})


@pytest.fixture
def timeline_long_term_memory(
    vector_store,
    vector_store_collection,
    segment_store,
    segment_store_partition,
    timeline_storage,
) -> LongTermMemory:
    # `RankedEmbedder` shares FakeEmbedder's dimensions, so the shared
    # `vector_store_collection` config still applies.
    return LongTermMemory(
        EventBackendParams(
            session_id="sess1",
            vector_store=vector_store,
            vector_store_collection=vector_store_collection,
            vector_store_collection_namespace="long_term_memory",
            segment_store=segment_store,
            segment_store_partition=segment_store_partition,
            partition_key="sess1",
            episode_storage=timeline_storage,
            embedder=RankedEmbedder(),
            segmenter=Segmenter(),
            deriver=Deriver([WholeTextDeriver()]),
        ),
    )


async def test_expand_context_returns_neighbours_the_search_would_not(
    timeline_long_term_memory,
    timeline_episodes,
):
    """expand_context folds the match's timeline neighbours into the result.

    The match's neighbours are the least similar episodes in the fixture, so
    no top-k can return them; if they come back, the expansion put them there.
    """
    await timeline_long_term_memory.add_episodes(timeline_episodes)

    query = _timeline_token(_MATCH_INDEX)
    plain = await timeline_long_term_memory.search_scored(
        query,
        num_episodes_limit=3,
    )
    expanded = await timeline_long_term_memory.search_scored(
        query,
        num_episodes_limit=3,
        expand_context=2,
    )

    plain_uids = {ep.uid for _, ep in plain}
    expanded_uids = {ep.uid for _, ep in expanded}
    assert f"tl-{_MATCH_INDEX}" in plain_uids
    assert not (plain_uids & _NEIGHBOUR_UIDS)
    assert expanded_uids & _NEIGHBOUR_UIDS

    # Expanded results come back chronologically, within the episode limit.
    created = [ep.created_at for _, ep in expanded]
    assert created == sorted(created)
    assert len(expanded) <= 3


async def test_expand_context_zero_returns_matches_in_score_order(
    timeline_long_term_memory,
    timeline_episodes,
):
    """Without expansion, the result is the search's own matches, best first."""
    await timeline_long_term_memory.add_episodes(timeline_episodes)

    scored = await timeline_long_term_memory.search_scored(
        _timeline_token(_MATCH_INDEX),
        num_episodes_limit=3,
    )

    scores = [score for score, _ in scored]
    assert scores == sorted(scores, reverse=True)
    assert not ({ep.uid for _, ep in scored} & _NEIGHBOUR_UIDS)


async def test_expand_context_window_stays_within_the_episode_limit(
    timeline_long_term_memory,
    timeline_episodes,
    segment_store_partition,
    monkeypatch,
):
    """The window asked of the segment store is clamped to [0, limit - 1].

    Asserted on the store call rather than on which episodes come back, so it
    holds however the window is split between the two directions and however
    results are ranked. The lower bound matters on its own: at
    `num_episodes_limit == 0` (which `SearchMemoriesSpec.top_k` allows)
    `min(expand_context, num_episodes_limit - 1)` is -1, and a negative window
    is outside the SegmentStorePartition contract.
    """
    await timeline_long_term_memory.add_episodes(timeline_episodes)

    walks: list[tuple[int, int]] = []
    get_segment_neighborhoods = segment_store_partition.get_segment_neighborhoods

    async def recording_get_segment_neighborhoods(seed_segment_uuids, **kwargs):
        walks.append(
            (
                kwargs.get("before", 0),
                kwargs.get("after", 0),
            )
        )
        return await get_segment_neighborhoods(seed_segment_uuids, **kwargs)

    monkeypatch.setattr(
        segment_store_partition,
        "get_segment_neighborhoods",
        recording_get_segment_neighborhoods,
    )

    for num_episodes_limit, expand_context in ((0, 5), (1, 5), (3, 99), (5, 2)):
        walks.clear()
        scored = await timeline_long_term_memory.search_scored(
            _timeline_token(_MATCH_INDEX),
            num_episodes_limit=num_episodes_limit,
            expand_context=expand_context,
        )
        assert len(scored) <= num_episodes_limit
        allowed = max(0, num_episodes_limit - 1)
        if allowed == 0:
            # Nothing to expand into: no walk is asked of the store.
            assert walks == []
            continue
        assert walks
        for backward, forward in walks:
            assert backward >= 0
            assert forward >= 0
            assert backward + forward <= allowed


def test_unify_takes_whole_contexts_while_they_fit():
    unified = LongTermMemory._unify_scored_uid_contexts(
        [
            (0.9, "b", ["a", "b", "c"]),
            (0.5, "e", ["d", "e"]),
        ],
        max_num_episodes=10,
    )
    assert unified == {"a": 0.9, "b": 0.9, "c": 0.9, "d": 0.5, "e": 0.5}


def test_unify_overflow_prefers_nucleus_then_forward():
    unified = LongTermMemory._unify_scored_uid_contexts(
        [(0.9, "c", ["a", "b", "c", "d", "e"])],
        max_num_episodes=3,
    )
    # Nucleus first, then forward neighbor, then next-forward beats backward
    # at equal distance (forward recall preferred).
    assert set(unified) == {"c", "d", "e"}


def test_unify_first_window_keeps_the_score():
    unified = LongTermMemory._unify_scored_uid_contexts(
        [
            (0.9, "b", ["a", "b"]),
            (0.4, "a", ["a", "z"]),
        ],
        max_num_episodes=10,
    )
    assert unified["a"] == 0.9  # first (best) window wins
    assert unified["z"] == 0.4


def test_episode_uid_context_dedup_and_nucleus():
    def _seg(uid: str) -> Segment:
        return Segment(
            session_id="s",
            source_id="src",
            uuid=uuid4(),
            event_uuid=uuid4(),
            index=0,
            offset=0,
            timestamp=datetime(2026, 1, 15, 12, 0, tzinfo=UTC),
            block=TextBlock(text=uid),
            properties={"_episode_uid": uid},
        )

    hit = QueryHit(
        score=1.0,
        seed=_seg("e2"),
        neighborhood=Neighborhood(before=[_seg("e1")], after=[_seg("e2"), _seg("e3")]),
    )

    nucleus, context = LongTermMemory._episode_uid_context(hit)
    assert nucleus == "e2"
    assert context == ["e1", "e2", "e3"]
