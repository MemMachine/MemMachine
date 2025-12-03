from unittest.mock import AsyncMock, MagicMock

import pytest

from memmachine.common.episode_store import CountCachingEpisodeStorage, EpisodeStorage
from memmachine.common.filter.filter_parser import Comparison


@pytest.fixture
def wrapped_store():
    store = MagicMock(spec=EpisodeStorage)
    store.startup = AsyncMock()
    store.add_episodes = AsyncMock(return_value=[])
    store.get_episode = AsyncMock()
    store.get_episode_messages = AsyncMock()
    store.get_episode_messages_count = AsyncMock()
    store.delete_episodes = AsyncMock()
    store.delete_episode_messages = AsyncMock()
    return store


@pytest.mark.asyncio
async def test_caches_counts_per_key(wrapped_store):
    wrapped_store.get_episode_messages_count = AsyncMock(side_effect=[2, 5])
    storage = CountCachingEpisodeStorage(wrapped_store)

    filter_a = Comparison(field="session_key", op="=", value="a")
    filter_b = Comparison(field="session_key", op="=", value="b")

    first_a = await storage.get_episode_messages_count(filter_expr=filter_a)
    second_a = await storage.get_episode_messages_count(filter_expr=filter_a)
    b_count = await storage.get_episode_messages_count(filter_expr=filter_b)

    assert first_a == 2
    assert second_a == 2
    assert b_count == 5
    assert wrapped_store.get_episode_messages_count.await_count == 2


@pytest.mark.asyncio
async def test_mutations_invalidate_cache(wrapped_store):
    wrapped_store.get_episode_messages_count = AsyncMock(side_effect=[1, 3])
    storage = CountCachingEpisodeStorage(wrapped_store)

    session_filter = Comparison(field="session_key", op="=", value="abc")

    first = await storage.get_episode_messages_count(filter_expr=session_filter)
    second = await storage.get_episode_messages_count(filter_expr=session_filter)

    assert first == 1
    assert second == 1
    assert wrapped_store.get_episode_messages_count.await_count == 1

    await storage.add_episodes("abc", [{"msg": "x"}])

    updated = await storage.get_episode_messages_count(filter_expr=session_filter)
    assert updated == 2
    assert wrapped_store.get_episode_messages_count.await_count == 1
    wrapped_store.add_episodes.assert_awaited_once_with("abc", [{"msg": "x"}])


@pytest.mark.asyncio
async def test_deletes_clear_cached_counts(wrapped_store):
    wrapped_store.get_episode_messages_count = AsyncMock(side_effect=[4, 6, 8])
    storage = CountCachingEpisodeStorage(wrapped_store)

    session_filter = Comparison(field="session_key", op="=", value="s")

    initial = await storage.get_episode_messages_count(filter_expr=session_filter)
    assert initial == 4

    await storage.delete_episode_messages()
    after_delete_messages = await storage.get_episode_messages_count(
        filter_expr=session_filter
    )

    await storage.delete_episodes([1, 2])
    after_delete_ids = await storage.get_episode_messages_count(
        filter_expr=session_filter
    )

    assert after_delete_messages == 6
    assert after_delete_ids == 8
    assert wrapped_store.get_episode_messages_count.await_count == 3
    wrapped_store.delete_episode_messages.assert_awaited_once_with(
        filter_expr=None,
        start_time=None,
        end_time=None,
    )
    wrapped_store.delete_episodes.assert_awaited_once_with([1, 2])


@pytest.mark.asyncio
async def test_non_session_filters_bypass_cache(wrapped_store):
    wrapped_store.get_episode_messages_count = AsyncMock(side_effect=[7, 9])
    storage = CountCachingEpisodeStorage(wrapped_store)

    topic_filter = Comparison(field="topic", op="=", value="alpha")

    first = await storage.get_episode_messages_count(filter_expr=topic_filter)
    second = await storage.get_episode_messages_count(filter_expr=topic_filter)

    assert first == 7
    assert second == 9
    assert wrapped_store.get_episode_messages_count.await_count == 2
