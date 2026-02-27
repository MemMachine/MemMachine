"""Unit tests for :mod:`memmachine.main.memmachine`."""

from __future__ import annotations

from contextlib import asynccontextmanager
from datetime import UTC, datetime
from typing import cast
from unittest.mock import AsyncMock, MagicMock

import pytest

from memmachine.common.configuration import (
    Configuration,
)
from memmachine.common.configuration.episodic_config import (
    EpisodicMemoryConfPartial,
    LongTermMemoryConfPartial,
    ShortTermMemoryConfPartial,
)
from memmachine.common.configuration.retrieval_config import RetrievalAgentConf
from memmachine.common.episode_store import Episode, EpisodeEntry, EpisodeResponse
from memmachine.common.errors import SessionNotFoundError
from memmachine.common.filter.filter_parser import And as FilterAnd
from memmachine.common.filter.filter_parser import Comparison as FilterComparison
from memmachine.episodic_memory import EpisodicMemory
from memmachine.main.memmachine import MemMachine, MemoryType
from memmachine.retrieval_agent.common.agent_api import AgentToolBase
from memmachine.semantic_memory.semantic_model import SemanticFeature


class DummySessionData:
    """Simple SessionData implementation for tests."""

    def __init__(
        self, session_key: str, org_id: str = "org", project_id: str = "project"
    ) -> None:
        self._session_key = session_key
        self._org_id = org_id
        self._project_id = project_id

    @property
    def session_id(self) -> str:  # pragma: no cover - trivial accessor
        return self._session_key

    @property
    def session_key(self) -> str:  # pragma: no cover - trivial accessor
        return self._session_key

    @property
    def org_id(self) -> str:  # pragma: no cover - trivial accessor
        return self._org_id

    @property
    def project_id(self) -> str:  # pragma: no cover - trivial accessor
        return self._project_id


@pytest.mark.asyncio
async def test_delete_session_raises_session_not_found(
    minimal_conf, patched_resource_manager
):
    session_manager = AsyncMock()
    session_manager.get_session_info = AsyncMock(return_value=None)
    patched_resource_manager.get_session_data_manager = AsyncMock(
        return_value=session_manager
    )

    memmachine = MemMachine(minimal_conf, patched_resource_manager)

    with pytest.raises(
        SessionNotFoundError, match=r"Session 'missing-session' does not exist"
    ):
        await memmachine.delete_session(DummySessionData("missing-session"))


def _minimal_conf(
    short_memory_enabled: bool = True, long_term_memory_enabled: bool = True
) -> Configuration:
    """Provide the minimal subset of configuration accessed in tests."""
    mock_rerankers = MagicMock()
    mock_rerankers.contains_reranker.return_value = True

    mock_embedders = MagicMock()
    mock_embedders.contains_embedder.return_value = True

    mock_language_models = MagicMock()
    mock_language_models.contains_language_model.return_value = True

    resource_conf = MagicMock()
    resource_conf.embedders = mock_embedders
    resource_conf.language_models = mock_language_models
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
        short_term_memory_enabled=short_memory_enabled,
        long_term_memory_enabled=long_term_memory_enabled,
    )
    ret.default_long_term_memory_embedder = "default-embedder"
    ret.default_long_term_memory_reranker = "default-reranker"
    ret.retrieval_agent = RetrievalAgentConf()
    semantic_conf = MagicMock()
    semantic_conf.llm_model = None
    ret.semantic_memory = semantic_conf
    prompt_conf = MagicMock()
    prompt_conf.episode_summary_system_prompt = "You are a helpful assistant."
    prompt_conf.episode_summary_user_prompt = (
        "Based on the following episodes: {episodes}, and the previous summary: {summary}, "
        "please update the summary. Keep it under {max_length} characters."
    )
    ret.prompt = prompt_conf
    return ret


@pytest.fixture
def minimal_conf() -> Configuration:
    return _minimal_conf()


@pytest.fixture
def minimal_conf_factory():
    return _minimal_conf


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
    assert conf.long_term_memory is not None
    assert conf.short_term_memory is not None
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


def test_with_default_retrieval_agent_llm_falls_back_to_stm_model(
    minimal_conf_factory, patched_resource_manager
):
    min_conf = minimal_conf_factory()
    assert min_conf.episodic_memory.short_term_memory is not None

    min_conf.episodic_memory.short_term_memory.llm_model = "fallback-llm"
    min_conf.semantic_memory.llm_model = None
    min_conf.retrieval_agent.llm_model = None
    min_conf.resources.language_models.contains_language_model.side_effect = (
        lambda model_name: model_name == "fallback-llm"
    )

    memmachine = MemMachine(min_conf, patched_resource_manager)

    assert memmachine._conf.retrieval_agent.llm_model == "fallback-llm"


def test_with_default_short_conf_enable_status(
    minimal_conf_factory, patched_resource_manager
):
    min_conf = minimal_conf_factory(
        short_memory_enabled=False, long_term_memory_enabled=True
    )
    memmachine = MemMachine(min_conf, patched_resource_manager)
    conf = memmachine._with_default_episodic_memory_conf(session_key="session-2")
    assert min_conf.episodic_memory.short_term_memory_enabled is False
    assert min_conf.episodic_memory.long_term_memory_enabled is True
    assert conf.short_term_memory_enabled is False
    assert conf.long_term_memory_enabled is True
    user_conf = EpisodicMemoryConfPartial(
        short_term_memory_enabled=True,
        long_term_memory_enabled=False,
    )
    conf = memmachine._with_default_episodic_memory_conf(
        session_key="session-2", user_conf=user_conf
    )
    assert conf.short_term_memory_enabled is True
    assert conf.long_term_memory_enabled is False


def test_with_default_long_conf_enable_status(
    minimal_conf_factory, patched_resource_manager
):
    min_conf = minimal_conf_factory(
        short_memory_enabled=True, long_term_memory_enabled=False
    )
    memmachine = MemMachine(min_conf, patched_resource_manager)
    conf = memmachine._with_default_episodic_memory_conf(session_key="session-2")
    assert min_conf.episodic_memory.short_term_memory_enabled is True
    assert min_conf.episodic_memory.long_term_memory_enabled is False
    assert conf.short_term_memory_enabled is True
    assert conf.long_term_memory_enabled is False
    user_conf = EpisodicMemoryConfPartial(
        short_term_memory_enabled=False,
        long_term_memory_enabled=True,
    )
    conf = memmachine._with_default_episodic_memory_conf(
        session_key="session-2", user_conf=user_conf
    )
    assert conf.short_term_memory_enabled is False
    assert conf.long_term_memory_enabled is True


@pytest.mark.asyncio
async def test_create_session_passes_generated_config(
    minimal_conf, patched_resource_manager
):
    session_manager = AsyncMock()
    patched_resource_manager.get_session_data_manager = AsyncMock(
        return_value=session_manager
    )

    memmachine = MemMachine(minimal_conf, patched_resource_manager)

    user_conf = EpisodicMemoryConfPartial(
        long_term_memory=LongTermMemoryConfPartial(
            embedder="custom-embed",
            reranker="custom-reranker",
        )
    )
    await memmachine.create_session(
        "alpha",
        description="demo",
        user_conf=user_conf,
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
    await_args = async_episodic.await_args
    assert await_args is not None
    assert await_args.kwargs["retrieval_agent"] is None

    assert result.episodic_memory is async_episodic.return_value
    assert result.semantic_memory == semantic_manager.search.return_value


@pytest.mark.asyncio
async def test_query_search_uses_retrieval_agent_when_agent_mode_enabled(
    minimal_conf, patched_resource_manager, monkeypatch
):
    dummy_session = DummySessionData("s1")
    expected_retrieval_agent = object()
    get_retrieval_agent = AsyncMock(return_value=expected_retrieval_agent)
    monkeypatch.setattr(MemMachine, "_get_retrieval_agent", get_retrieval_agent)

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

    memmachine = MemMachine(minimal_conf, patched_resource_manager)
    await memmachine.query_search(
        dummy_session,
        target_memories=[MemoryType.Episodic],
        query="hello world",
        agent_mode=True,
    )

    get_retrieval_agent.assert_awaited_once()
    await_args = async_episodic.await_args
    assert await_args is not None
    assert await_args.kwargs["retrieval_agent"] is expected_retrieval_agent


@pytest.mark.asyncio
async def test_query_episodic_with_retrieval_agent_searches_long_then_short(
    minimal_conf, patched_resource_manager
):
    memmachine = MemMachine(minimal_conf, patched_resource_manager)

    long_episode = _make_episode("long-1", "s1")
    short_episode = _make_episode("short-1", "s1")

    long_only_response = EpisodicMemory.QueryResponse(
        long_term_memory=EpisodicMemory.QueryResponse.LongTermMemoryResponse(
            episodes=[EpisodeResponse(score=0.8, **long_episode.model_dump())]
        ),
        short_term_memory=EpisodicMemory.QueryResponse.ShortTermMemoryResponse(
            episodes=[],
            episode_summary=[""],
        ),
    )
    short_only_response = EpisodicMemory.QueryResponse(
        long_term_memory=EpisodicMemory.QueryResponse.LongTermMemoryResponse(
            episodes=[]
        ),
        short_term_memory=EpisodicMemory.QueryResponse.ShortTermMemoryResponse(
            episodes=[EpisodeResponse(**short_episode.model_dump())],
            episode_summary=["short-summary"],
        ),
    )

    episodic_session = object.__new__(EpisodicMemory)
    episodic_session._session_key = "s1"
    episodic_session._long_term_memory = MagicMock()
    episodic_session._short_term_memory = MagicMock()

    async def _query_memory_side_effect(*_args, **kwargs):
        mode = kwargs["mode"]
        if mode is EpisodicMemory.QueryMode.LONG_TERM_ONLY:
            return long_only_response
        if mode is EpisodicMemory.QueryMode.SHORT_TERM_ONLY:
            return short_only_response
        raise AssertionError(f"Unexpected mode: {mode}")

    episodic_session.query_memory = AsyncMock(side_effect=_query_memory_side_effect)

    class _TestRetrievalAgent:
        async def do_query(self, _policy, query_param):
            assert query_param.memory is episodic_session
            long_term_response = await query_param.memory.query_memory(
                query=query_param.query,
                limit=query_param.limit,
                expand_context=query_param.expand_context,
                score_threshold=query_param.score_threshold,
                property_filter=query_param.property_filter,
                mode=EpisodicMemory.QueryMode.LONG_TERM_ONLY,
            )
            assert long_term_response is not None
            episodes = [
                Episode(
                    uid=episode.uid,
                    content=episode.content,
                    session_key=query_param.memory.session_key,
                    created_at=episode.created_at or datetime.now(UTC),
                    producer_id=episode.producer_id,
                    producer_role=episode.producer_role,
                    produced_for_id=episode.produced_for_id,
                    metadata=episode.metadata,
                )
                for episode in long_term_response.long_term_memory.episodes
            ]
            return episodes, {}

    response = await memmachine._query_episodic_with_retrieval_agent(
        episodic_session=episodic_session,
        retrieval_agent=cast(AgentToolBase, _TestRetrievalAgent()),
        query="hello world",
        limit=5,
        expand_context=0,
        score_threshold=-float("inf"),
        search_filter=None,
    )

    assert response is not None
    assert [episode.uid for episode in response.long_term_memory.episodes] == ["long-1"]
    assert [episode.uid for episode in response.short_term_memory.episodes] == [
        "short-1"
    ]
    assert response.short_term_memory.episode_summary == ["short-summary"]
    assert episodic_session.query_memory.await_count == 2
    assert (
        episodic_session.query_memory.await_args_list[0].kwargs["mode"]
        is EpisodicMemory.QueryMode.LONG_TERM_ONLY
    )
    assert (
        episodic_session.query_memory.await_args_list[1].kwargs["mode"]
        is EpisodicMemory.QueryMode.SHORT_TERM_ONLY
    )


@pytest.mark.asyncio
async def test_query_episodic_with_retrieval_agent_skips_short_term_when_disabled(
    minimal_conf, patched_resource_manager
):
    memmachine = MemMachine(minimal_conf, patched_resource_manager)
    long_episode = _make_episode("long-2", "s1")
    long_only_response = EpisodicMemory.QueryResponse(
        long_term_memory=EpisodicMemory.QueryResponse.LongTermMemoryResponse(
            episodes=[EpisodeResponse(score=0.9, **long_episode.model_dump())]
        ),
        short_term_memory=EpisodicMemory.QueryResponse.ShortTermMemoryResponse(
            episodes=[],
            episode_summary=[""],
        ),
    )

    episodic_session = object.__new__(EpisodicMemory)
    episodic_session._session_key = "s1"
    episodic_session._long_term_memory = MagicMock()
    episodic_session._short_term_memory = None
    episodic_session.query_memory = AsyncMock(return_value=long_only_response)

    class _TestRetrievalAgent:
        async def do_query(self, _policy, query_param):
            assert query_param.memory is episodic_session
            long_term_response = await query_param.memory.query_memory(
                query=query_param.query,
                limit=query_param.limit,
                expand_context=query_param.expand_context,
                score_threshold=query_param.score_threshold,
                property_filter=query_param.property_filter,
                mode=EpisodicMemory.QueryMode.LONG_TERM_ONLY,
            )
            assert long_term_response is not None
            episodes = [
                Episode(
                    uid=episode.uid,
                    content=episode.content,
                    session_key=query_param.memory.session_key,
                    created_at=episode.created_at or datetime.now(UTC),
                    producer_id=episode.producer_id,
                    producer_role=episode.producer_role,
                    produced_for_id=episode.produced_for_id,
                    metadata=episode.metadata,
                )
                for episode in long_term_response.long_term_memory.episodes
            ]
            return episodes, {}

    response = await memmachine._query_episodic_with_retrieval_agent(
        episodic_session=episodic_session,
        retrieval_agent=cast(AgentToolBase, _TestRetrievalAgent()),
        query="hello world",
        limit=5,
        expand_context=0,
        score_threshold=-float("inf"),
        search_filter=None,
    )

    assert response is not None
    assert [episode.uid for episode in response.long_term_memory.episodes] == ["long-2"]
    assert response.short_term_memory.episodes == []
    assert episodic_session.query_memory.await_count == 1
    assert (
        episodic_session.query_memory.await_args.kwargs["mode"]
        is EpisodicMemory.QueryMode.LONG_TERM_ONLY
    )


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
        episodes=stored_episodes,
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
async def test_count_episodes_filters_by_session_only(
    minimal_conf, patched_resource_manager
):
    memmachine = MemMachine(minimal_conf, patched_resource_manager)
    session = DummySessionData("session-count")

    episode_storage = MagicMock()
    episode_storage.get_episode_messages_count = AsyncMock(return_value=7)
    patched_resource_manager.get_episode_storage = AsyncMock(
        return_value=episode_storage
    )

    result = await memmachine.episodes_count(session, search_filter=None)

    assert result == 7
    episode_storage.get_episode_messages_count.assert_awaited_once_with(
        filter_expr=FilterComparison(
            field="session_key",
            op="=",
            value=session.session_key,
        )
    )


@pytest.mark.asyncio
async def test_count_episodes_combines_search_filter(
    minimal_conf, patched_resource_manager, monkeypatch
):
    memmachine = MemMachine(minimal_conf, patched_resource_manager)
    session = DummySessionData("session-with-filter")
    custom_filter = FilterComparison(field="topic", op="=", value="alpha")
    parsed_specs: list[str] = []

    def _fake_parse(spec: str | None):
        parsed_specs.append(spec or "")
        return custom_filter

    monkeypatch.setattr("memmachine.main.memmachine.parse_filter", _fake_parse)

    episode_storage = MagicMock()
    episode_storage.get_episode_messages_count = AsyncMock(return_value=3)
    patched_resource_manager.get_episode_storage = AsyncMock(
        return_value=episode_storage
    )

    result = await memmachine.episodes_count(session, search_filter="topic = 'alpha'")

    assert result == 3
    assert parsed_specs == ["topic = 'alpha'"]

    combined_filter = episode_storage.get_episode_messages_count.await_args.kwargs[
        "filter_expr"
    ]
    assert combined_filter == FilterAnd(
        left=FilterComparison(
            field="session_key",
            op="=",
            value=session.session_key,
        ),
        right=custom_filter,
    )


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
