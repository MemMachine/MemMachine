"""Unit tests for :mod:`memmachine.main.memmachine`."""

from __future__ import annotations

from contextlib import asynccontextmanager
from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock

import pytest

from memmachine.common.configuration import (
    Configuration,
    EpisodicMemoryConfPartial,
)
from memmachine.common.configuration.episodic_config import (
    LongTermMemoryConfPartial,
    ShortTermMemoryConfPartial,
)
from memmachine.common.episode_store import Episode, EpisodeEntry
from memmachine.episodic_memory import EpisodicMemory
from memmachine.main.memmachine import MemMachine, MemoryType
from memmachine.semantic_memory.semantic_model import SemanticFeature


class DummySessionData:
    """Simple SessionData implementation for tests."""

    def __init__(self, session_key: str) -> None:
        self._session_key = session_key

    @property
    def session_id(self) -> str:  # pragma: no cover - trivial accessor
        return self._session_key

    @property
    def session_key(self) -> str:  # pragma: no cover - trivial accessor
        return self._session_key


@pytest.fixture
def minimal_conf() -> Configuration:
    """Provide the minimal subset of configuration accessed in tests."""
    mock_rerankers = MagicMock()
    mock_rerankers.contains_reranker.return_value = True

    mock_embedders = MagicMock()
    mock_embedders.contains_embedder.return_value = True

    resource_conf = MagicMock()
    resource_conf.embedders = mock_embedders
    resource_conf.rerankers = mock_rerankers

    ret = MagicMock()
    ret.resources = resource_conf
    ret.episodic_memory = EpisodicMemoryConfPartial(
        short_term_memory=ShortTermMemoryConfPartial(
            summary_prompt_system=None,
            summary_prompt_user=None,
            llm_model=None,
        ),
        long_term_memory=LongTermMemoryConfPartial(
            vector_graph_store=None,
            embedder="default-embedder",
            reranker="default-reranker",
        ),
    )
    ret.default_long_term_memory_embedder = "default-embedder"
    ret.default_long_term_memory_reranker = "default-reranker"
    return ret


@pytest.fixture
def patched_resource_manager(monkeypatch):
    """Replace :class:`ResourceManagerImpl` with a controllable double."""

    fake_manager = AsyncMock()
    monkeypatch.setattr(
        "memmachine.main.memmachine.ResourceManagerImpl",
        MagicMock(return_value=fake_manager),
    )
    return fake_manager


def _make_episode(uid: str, session_key: str) -> Episode:
    return Episode(
        uid=uid,
        content="content",
        session_key=session_key,
        created_at=datetime.now(UTC),
        producer_id="user",
        producer_role="assistant",
    )


def _async_cm(value):
    @asynccontextmanager
    async def _manager():
        yield value

    return _manager()


def test_with_default_episodic_memory_conf_uses_fallbacks(
    minimal_conf, patched_resource_manager
):
    memmachine = MemMachine(minimal_conf, patched_resource_manager)

    conf = memmachine._with_default_episodic_memory_conf(session_key="session-1")

    assert conf.session_key == "session-1"
    assert conf.long_term_memory.embedder == "default-embedder"
    assert conf.long_term_memory.reranker == "default-reranker"
    assert conf.long_term_memory.vector_graph_store == "default_store"
    assert conf.short_term_memory.llm_model == "gpt-4.1"
    assert conf.short_term_memory.summary_prompt_system.startswith(
        "You are a helpful assistant."
    )
    assert (
        "Based on the following episodes" in conf.short_term_memory.summary_prompt_user
    )


@pytest.mark.asyncio
async def test_create_session_passes_generated_config(
    minimal_conf, patched_resource_manager
):
    session_manager = AsyncMock()
    patched_resource_manager.get_session_data_manager = AsyncMock(
        return_value=session_manager
    )

    memmachine = MemMachine(minimal_conf, patched_resource_manager)

    await memmachine.create_session(
        "alpha",
        description="demo",
        embedder_name="custom-embed",
        reranker_name="custom-reranker",
    )

    session_manager.create_new_session.assert_awaited_once()
    _, kwargs = session_manager.create_new_session.await_args
    episodic_conf = kwargs["param"]

    assert episodic_conf.long_term_memory.embedder == "custom-embed"
    assert episodic_conf.long_term_memory.reranker == "custom-reranker"
    assert episodic_conf.short_term_memory.session_key == "alpha"
    assert kwargs["description"] == "demo"


@pytest.mark.asyncio
async def test_query_search_runs_targeted_memory_tasks(
    minimal_conf, patched_resource_manager, monkeypatch
):
    dummy_session = DummySessionData("s1")

    async_episodic = AsyncMock(
        return_value=EpisodicMemory.QueryResponse(
            long_term_memory=EpisodicMemory.QueryResponse.LongTermMemoryResponse(
                episodes=[]
            ),
            short_term_memory=EpisodicMemory.QueryResponse.ShortTermMemoryResponse(
                episodes=[],
                episode_summary=[],
            ),
        )
    )
    monkeypatch.setattr(MemMachine, "_search_episodic_memory", async_episodic)

    semantic_manager = MagicMock()
    semantic_manager.search = AsyncMock(
        return_value=[
            SemanticFeature(
                category="profile",
                tag="name",
                feature_name="value",
                value="semantic-response",
            )
        ]
    )
    patched_resource_manager.get_semantic_session_manager = AsyncMock(
        return_value=semantic_manager
    )

    memmachine = MemMachine(minimal_conf, patched_resource_manager)

    result = await memmachine.query_search(
        dummy_session,
        target_memories=[MemoryType.Episodic, MemoryType.Semantic],
        query="hello world",
    )

    async_episodic.assert_awaited_once()
    semantic_manager.search.assert_awaited_once()

    assert result.episodic_memory is async_episodic.return_value
    assert result.semantic_memory == semantic_manager.search.return_value


@pytest.mark.asyncio
async def test_query_search_skips_unrequested_memories(
    minimal_conf, patched_resource_manager, monkeypatch
):
    dummy_session = DummySessionData("s2")

    async_episodic = AsyncMock(
        return_value=EpisodicMemory.QueryResponse(
            long_term_memory=EpisodicMemory.QueryResponse.LongTermMemoryResponse(
                episodes=[]
            ),
            short_term_memory=EpisodicMemory.QueryResponse.ShortTermMemoryResponse(
                episodes=[],
                episode_summary=[],
            ),
        )
    )
    monkeypatch.setattr(MemMachine, "_search_episodic_memory", async_episodic)

    semantic_manager = MagicMock()
    semantic_manager.search = AsyncMock(
        return_value=[
            SemanticFeature(
                category="profile",
                tag="name",
                feature_name="value",
                value="semantic-response",
            )
        ]
    )
    patched_resource_manager.get_semantic_session_manager = AsyncMock(
        return_value=semantic_manager
    )

    memmachine = MemMachine(minimal_conf, patched_resource_manager)

    result = await memmachine.query_search(
        dummy_session,
        target_memories=[MemoryType.Semantic],
        query="find",
    )

    async_episodic.assert_not_called()
    semantic_manager.search.assert_awaited_once()

    assert result.episodic_memory is None
    assert result.semantic_memory == semantic_manager.search.return_value


@pytest.mark.asyncio
async def test_add_episodes_dispatches_to_all_memories(
    minimal_conf, patched_resource_manager
):
    memmachine = MemMachine(minimal_conf, patched_resource_manager)
    session = DummySessionData("session-42")

    entries = [
        EpisodeEntry(content="hello", producer_id="user", producer_role="assistant"),
    ]
    stored_episodes = [
        _make_episode("e1", session.session_key),
        _make_episode("e2", session.session_key),
    ]

    episode_storage = MagicMock()
    episode_storage.add_episodes = AsyncMock(return_value=stored_episodes)
    patched_resource_manager.get_episode_storage = AsyncMock(
        return_value=episode_storage
    )

    episodic_session = AsyncMock()
    episodic_manager = MagicMock()
    episodic_manager.open_episodic_memory.return_value = _async_cm(episodic_session)
    episodic_manager.open_or_create_episodic_memory.return_value = _async_cm(
        episodic_session
    )
    patched_resource_manager.get_episodic_memory_manager = AsyncMock(
        return_value=episodic_manager
    )

    semantic_manager_service = MagicMock()
    semantic_manager_service.simple_semantic_session_id_manager._generate_session_data.return_value = session

    patched_resource_manager.get_semantic_manager = AsyncMock(
        return_value=semantic_manager_service
    )

    semantic_manager = MagicMock()
    semantic_manager.add_message = AsyncMock()
    patched_resource_manager.get_semantic_session_manager = AsyncMock(
        return_value=semantic_manager
    )

    await memmachine.add_episodes(session, entries)

    episode_storage.add_episodes.assert_awaited_once_with(session.session_key, entries)
    episodic_session.add_memory_episodes.assert_awaited_once_with(stored_episodes)
    semantic_manager.add_message.assert_awaited_once_with(
        episode_ids=["e1", "e2"],
        session_data=session,
    )


@pytest.mark.asyncio
async def test_add_episodes_skips_memories_not_requested(
    minimal_conf, patched_resource_manager
):
    memmachine = MemMachine(minimal_conf, patched_resource_manager)
    session = DummySessionData("only-semantic")

    entries = [
        EpisodeEntry(content="hello", producer_id="user", producer_role="assistant"),
    ]
    stored_episodes = [_make_episode("e1", session.session_key)]

    episode_storage = MagicMock()
    episode_storage.add_episodes = AsyncMock(return_value=stored_episodes)
    patched_resource_manager.get_episode_storage = AsyncMock(
        return_value=episode_storage
    )

    episodic_manager = MagicMock()
    episodic_manager.open_episodic_memory.return_value = _async_cm(MagicMock())
    patched_resource_manager.get_episodic_memory_manager = AsyncMock(
        return_value=episodic_manager
    )

    semantic_manager = MagicMock()
    semantic_manager.add_message = AsyncMock()
    patched_resource_manager.get_semantic_session_manager = AsyncMock(
        return_value=semantic_manager
    )

    await memmachine.add_episodes(
        session,
        entries,
        target_memories=[MemoryType.Semantic],
    )

    episodic_manager.open_episodic_memory.assert_not_called()
    semantic_manager.add_message.assert_awaited_once()


@pytest.mark.asyncio
async def test_list_search_fetches_episode_history(
    minimal_conf, patched_resource_manager
):
    memmachine = MemMachine(minimal_conf, patched_resource_manager)
    session = DummySessionData("session-list")

    episode_storage = MagicMock()
    episodes = [_make_episode("e1", session.session_key)]
    episode_storage.get_episode_messages = AsyncMock(return_value=episodes)
    patched_resource_manager.get_episode_storage = AsyncMock(
        return_value=episode_storage
    )

    result = await memmachine.list_search(
        session,
        target_memories=[MemoryType.Episodic],
        search_filter="meta_key = value",
    )

    episode_storage.get_episode_messages.assert_awaited_once()
    assert result.episodic_memory == episodes
    assert result.semantic_memory is None


@pytest.mark.asyncio
async def test_delete_episodes_forwards_to_storage_and_memories(
    minimal_conf, patched_resource_manager
):
    memmachine = MemMachine(minimal_conf, patched_resource_manager)
    session = DummySessionData("session-del")

    episode_storage = MagicMock()
    episode_storage.delete_episodes = AsyncMock()
    patched_resource_manager.get_episode_storage = AsyncMock(
        return_value=episode_storage
    )

    episodic_session = AsyncMock()
    episodic_manager = MagicMock()
    episodic_manager.open_episodic_memory.return_value = _async_cm(episodic_session)
    patched_resource_manager.get_episodic_memory_manager = AsyncMock(
        return_value=episodic_manager
    )

    await memmachine.delete_episodes(["ep1", "ep2"], session_data=session)

    episode_storage.delete_episodes.assert_awaited_once_with(["ep1", "ep2"])
    episodic_session.delete_episodes.assert_awaited_once_with(["ep1", "ep2"])


@pytest.mark.asyncio
async def test_delete_episodes_without_session_only_hits_storage(
    minimal_conf, patched_resource_manager
):
    memmachine = MemMachine(minimal_conf, patched_resource_manager)

    episode_storage = MagicMock()
    episode_storage.delete_episodes = AsyncMock()
    patched_resource_manager.get_episode_storage = AsyncMock(
        return_value=episode_storage
    )

    episodic_manager = MagicMock()
    episodic_manager.open_episodic_memory.return_value = _async_cm(MagicMock())
    patched_resource_manager.get_episodic_memory_manager = AsyncMock(
        return_value=episodic_manager
    )

    await memmachine.delete_episodes(["ep1"], session_data=None)

    episode_storage.delete_episodes.assert_awaited_once_with(["ep1"])
    episodic_manager.open_episodic_memory.assert_not_called()


@pytest.mark.asyncio
async def test_delete_features_forwards_to_semantic_manager(
    minimal_conf, patched_resource_manager
):
    memmachine = MemMachine(minimal_conf, patched_resource_manager)

    semantic_manager = MagicMock()
    semantic_manager.delete_features = AsyncMock()
    patched_resource_manager.get_semantic_session_manager = AsyncMock(
        return_value=semantic_manager
    )

    await memmachine.delete_features(["feat1", "feat2"])

    semantic_manager.delete_features.assert_awaited_once_with(["feat1", "feat2"])
