"""Unit tests for :mod:`memmachine_server.main.memmachine`."""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from typing import cast
from unittest.mock import (
    AsyncMock,
    MagicMock,
    NonCallableMagicMock,
    create_autospec,
)
from uuid import NAMESPACE_URL, UUID, uuid5

import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import create_async_engine

from memmachine_server.common.configuration import (
    Configuration,
)
from memmachine_server.common.configuration.episodic_config import (
    DeclarativeLongTermMemoryConf,
    EpisodicMemoryConfPartial,
    EventLongTermMemoryConf,
    LongTermMemoryConfPartial,
    ShortTermMemoryConfPartial,
)
from memmachine_server.common.configuration.retrieval_config import RetrievalAgentConf
from memmachine_server.common.episode_store import (
    Episode,
    EpisodeEntry,
    EpisodeResponse,
)
from memmachine_server.common.errors import (
    ConfigurationError,
    InvalidArgumentError,
    ResourceNotFoundError,
    SessionAlreadyExistsError,
    SessionNotFoundError,
)
from memmachine_server.common.filter.filter_parser import And as FilterAnd
from memmachine_server.common.filter.filter_parser import Comparison as FilterComparison
from memmachine_server.common.partition_registry import (
    PartitionDeletedError,
    PartitionedStore,
    PartitionPendingError,
)
from memmachine_server.common.partition_registry.sqlalchemy_partition_registry import (
    SQLAlchemyPartitionRegistry,
    SQLAlchemyPartitionRegistryParams,
)
from memmachine_server.common.session_manager.session_data_manager import (
    SessionDataManager,
)
from memmachine_server.common.session_manager.session_data_manager_sql_impl import (
    SessionDataManagerSQL,
)
from memmachine_server.common.sql import enable_sqlite_explicit_begin
from memmachine_server.episodic_memory import EpisodicMemory
from memmachine_server.main import memmachine as memmachine_module
from memmachine_server.main.memmachine import MemMachine, MemoryType
from memmachine_server.main.partitioned_stores import SessionDataPartitionedStore
from memmachine_server.retrieval_agent.common.agent_api import AgentToolBase
from memmachine_server.semantic_memory.semantic_memory import with_has_set_ids
from memmachine_server.semantic_memory.semantic_model import SemanticFeature
from memmachine_server.semantic_memory.semantic_session_manager import (
    SemanticSessionManager,
)

RETENTION_SECONDS = 86400


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
    semantic_conf.storage_configured.return_value = False
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


@pytest_asyncio.fixture
async def partition_registry(tmp_path):
    """The registry of MemMachine's partition namespace, on a SQLite file."""
    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'registry.db'}")
    enable_sqlite_explicit_begin(engine)
    registry = SQLAlchemyPartitionRegistry(
        SQLAlchemyPartitionRegistryParams(
            engine=engine,
            partition_namespace="memmachine",
            tombstone_retention_seconds=RETENTION_SECONDS,
        ),
    )
    await registry.prepare()
    await registry.startup()
    yield registry
    await engine.dispose()


@pytest.fixture
def partitioned_store(monkeypatch) -> NonCallableMagicMock:
    """The one store MemMachine builds its partition manager over, in place of its three."""
    store = create_autospec(PartitionedStore, instance=True)
    store.purge_partition.return_value = False
    for name in (
        "SessionDataPartitionedStore",
        "EpisodePartitionedStore",
        "LongTermMemoryPartitionedStore",
    ):
        monkeypatch.setattr(
            f"memmachine_server.main.memmachine.{name}",
            MagicMock(return_value=store),
        )
    return store


@pytest.fixture
def patched_resource_manager(monkeypatch, partition_registry, partitioned_store):
    """Replace :class:`ResourceManagerImpl` with a controllable double, over a real partition registry."""

    fake_manager = AsyncMock()
    fake_manager.get_partition_registry = AsyncMock(return_value=partition_registry)
    fake_manager.get_metrics_factory = AsyncMock(return_value=None)
    monkeypatch.setattr(
        "memmachine_server.main.memmachine.ResourceManagerImpl",
        MagicMock(return_value=fake_manager),
    )
    return fake_manager


@pytest.fixture
def session_data_store(monkeypatch, partitioned_store) -> None:
    """MemMachine's own session data store, in place of the stores' double."""
    monkeypatch.setattr(
        memmachine_module, "SessionDataPartitionedStore", SessionDataPartitionedStore
    )


async def _partition_key(
    partition_registry: SQLAlchemyPartitionRegistry, session_key: str
) -> str:
    """The key the components hold a live session's data under."""
    registration = await partition_registry.resolve(session_key)
    assert registration is not None
    return str(registration.partition_uuid)


async def _create_partition(
    partition_registry: SQLAlchemyPartitionRegistry, session_key: str
) -> str:
    """Register a live partition under a session key, and answer its key."""
    await (await partition_registry.reserve(session_key)).confirm()
    return await _partition_key(partition_registry, session_key)


def _uid(value: str | UUID) -> UUID:
    return value if isinstance(value, UUID) else uuid5(NAMESPACE_URL, value)


def _make_episode(uid: str | UUID, session_key: str) -> Episode:
    return Episode(
        uid=_uid(uid),
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


class _ClosingEpisodicSession:
    """Fail if an episodic operation runs after its context closes."""

    def __init__(self) -> None:
        self.closed = False
        self.operations: list[str] = []

    async def add_memory_episodes(self, episodes: list[Episode]) -> None:
        assert not self.closed
        await asyncio.sleep(0)
        assert not self.closed
        self.operations.append("add")

    async def delete_episodes(self, episode_ids: list[str]) -> None:
        assert not self.closed
        await asyncio.sleep(0)
        assert not self.closed
        self.operations.append("delete")

    async def close(self) -> None:
        self.closed = True
        self.operations.append("close")


@asynccontextmanager
async def _closing_episodic_cm(memory: _ClosingEpisodicSession):
    try:
        yield memory
    finally:
        await memory.close()


def test_with_default_episodic_memory_conf_uses_fallbacks(
    minimal_conf, patched_resource_manager
):
    memmachine = MemMachine(minimal_conf, patched_resource_manager)

    conf = memmachine._with_default_episodic_memory_conf(session_key="session-1")

    assert conf.session_key == "session-1"
    assert conf.long_term_memory is not None
    assert conf.short_term_memory is not None
    assert isinstance(conf.long_term_memory, DeclarativeLongTermMemoryConf)
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


_EVENT_PROJECT = EpisodicMemoryConfPartial(
    long_term_memory=LongTermMemoryConfPartial(backend="event")
)


def test_an_event_project_takes_the_deployments_event_memory(
    minimal_conf_factory, patched_resource_manager
):
    conf = minimal_conf_factory()
    conf.episodic_memory.long_term_memory = LongTermMemoryConfPartial(
        backend="event",
        vector_store="vs",
        segment_store="ss",
        partition_registry="pr",
        embedder="default-embedder",
        reranker="default-reranker",
    )
    memmachine = MemMachine(conf, patched_resource_manager)

    merged = memmachine._with_default_episodic_memory_conf(
        session_key="s", user_conf=_EVENT_PROJECT
    )

    assert isinstance(merged.long_term_memory, EventLongTermMemoryConf)
    assert merged.long_term_memory.reranker == "default-reranker"


@pytest.mark.parametrize(
    "missing", ["vector_store", "segment_store", "partition_registry"]
)
def test_an_event_deployment_missing_a_store_or_registry_disables_long_term_memory(
    minimal_conf_factory, patched_resource_manager, missing
):
    conf = minimal_conf_factory()
    long_term_memory = LongTermMemoryConfPartial(
        backend="event",
        embedder="default-embedder",
        vector_store="vs",
        segment_store="ss",
        partition_registry="pr",
    )
    setattr(long_term_memory, missing, None)
    conf.episodic_memory.long_term_memory = long_term_memory

    MemMachine(conf, patched_resource_manager)

    assert conf.episodic_memory.long_term_memory_enabled is False


def test_a_deployment_without_event_memory_refuses_an_event_project(
    minimal_conf, patched_resource_manager
):
    memmachine = MemMachine(minimal_conf, patched_resource_manager)

    with pytest.raises(ConfigurationError, match="no event memory"):
        memmachine._with_default_episodic_memory_conf(
            session_key="s", user_conf=_EVENT_PROJECT
        )


def test_a_deployment_without_a_partition_registry_refuses_an_event_project(
    minimal_conf_factory, patched_resource_manager
):
    """A deployment without a partition registry refuses an event session, whose event memory it could neither serve nor delete."""
    conf = minimal_conf_factory()
    conf.episodic_memory.long_term_memory = LongTermMemoryConfPartial(
        backend="event",
        vector_store="vs",
        segment_store="ss",
        embedder="default-embedder",
    )
    memmachine = MemMachine(conf, patched_resource_manager)

    with pytest.raises(ConfigurationError, match="no event memory"):
        memmachine._with_default_episodic_memory_conf(
            session_key="s", user_conf=_EVENT_PROJECT
        )


@pytest.mark.asyncio
async def test_create_session_passes_generated_config(
    minimal_conf, patched_resource_manager, partition_registry
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

    session_manager.create_or_validate_session.assert_awaited_once()
    _, kwargs = session_manager.create_or_validate_session.await_args
    episodic_conf = kwargs["param"]

    assert episodic_conf.long_term_memory.embedder == "custom-embed"
    assert episodic_conf.long_term_memory.reranker == "custom-reranker"
    assert episodic_conf.short_term_memory.session_key == await _partition_key(
        partition_registry, "alpha"
    )
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

    semantic_features = [
        SemanticFeature(
            category="profile",
            tag="name",
            feature_name="value",
            value="semantic-response",
        )
    ]

    async def mock_search(*args, **kwargs):
        for feature in semantic_features:
            yield feature

    semantic_manager = MagicMock()
    semantic_manager.search = mock_search
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

    assert result.episodic_memory is async_episodic.return_value
    assert result.semantic_memory == semantic_features


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
    assert [episode.uid for episode in response.long_term_memory.episodes] == [
        _uid("long-1")
    ]
    assert [episode.uid for episode in response.short_term_memory.episodes] == [
        _uid("short-1")
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
    assert [episode.uid for episode in response.long_term_memory.episodes] == [
        _uid("long-2")
    ]
    assert response.short_term_memory.episodes == []
    assert episodic_session.query_memory.await_count == 1
    await_args = episodic_session.query_memory.await_args
    assert await_args is not None
    assert await_args.kwargs["mode"] is EpisodicMemory.QueryMode.LONG_TERM_ONLY


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

    semantic_features = [
        SemanticFeature(
            category="profile",
            tag="name",
            feature_name="value",
            value="semantic-response",
        )
    ]

    async def mock_search(*args, **kwargs):
        for feature in semantic_features:
            yield feature

    semantic_manager = MagicMock()
    semantic_manager.search = mock_search
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

    assert result.episodic_memory is None
    assert result.semantic_memory == semantic_features


@pytest.mark.asyncio
async def test_add_episodes_dispatches_same_id_to_all_backends(
    minimal_conf, patched_resource_manager, monkeypatch, partition_registry
):
    memmachine = MemMachine(minimal_conf, patched_resource_manager)
    minted_uid = _uid("minted")
    monkeypatch.setattr("memmachine_server.main.memmachine.uuid4", lambda: minted_uid)
    session = DummySessionData("session-42")
    partition_key = await _create_partition(partition_registry, session.session_key)
    created_at = datetime(2026, 1, 2, tzinfo=UTC)

    entries = [
        EpisodeEntry(
            uid=_uid("e1"),
            content="hello",
            producer_id="user",
            producer_role="assistant",
            created_at=created_at,
        ),
    ]
    episodes = [
        Episode(
            uid=_uid("e1"),
            content="hello",
            session_key=partition_key,
            created_at=created_at,
            producer_id="user",
            producer_role="assistant",
        )
    ]

    episode_storage = MagicMock()
    episode_storage.add_episodes = AsyncMock(return_value=episodes)
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

    store_call = episode_storage.add_episodes.await_args
    assert store_call is not None
    assert store_call.args[0] == partition_key
    assert [entry.uid for entry in store_call.args[1]] == [minted_uid]
    expected_episodes = [episodes[0].model_copy(update={"uid": minted_uid})]
    episodic_session.add_memory_episodes.assert_awaited_once_with(expected_episodes)
    semantic_manager.add_message.assert_awaited_once_with(
        episodes=expected_episodes,
        session_data=session,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("key", ["userId", "user-id", "memmachine_mine"])
async def test_add_episodes_refuses_an_illegal_metadata_key_before_any_write(
    minimal_conf, patched_resource_manager, key
):
    memmachine = MemMachine(minimal_conf, patched_resource_manager)
    episode_storage = MagicMock()
    episode_storage.add_episodes = AsyncMock()
    patched_resource_manager.get_episode_storage = AsyncMock(
        return_value=episode_storage
    )
    patched_resource_manager.get_episodic_memory_manager = AsyncMock()
    patched_resource_manager.get_semantic_session_manager = AsyncMock()
    entries = [
        EpisodeEntry(
            content="hello",
            producer_id="user",
            producer_role="user",
            metadata={"color": "red"},
        ),
        EpisodeEntry(
            content="world",
            producer_id="user",
            producer_role="user",
            metadata={key: "v"},
        ),
    ]

    with pytest.raises(ValueError, match=key):
        await memmachine.add_episodes(DummySessionData("session-42"), entries)

    episode_storage.add_episodes.assert_not_awaited()
    patched_resource_manager.get_episodic_memory_manager.assert_not_awaited()
    patched_resource_manager.get_semantic_session_manager.assert_not_awaited()


@pytest.mark.asyncio
async def test_add_episodes_single_entry_replaces_supplied_uid(
    minimal_conf, patched_resource_manager, monkeypatch
):
    memmachine = MemMachine(minimal_conf, patched_resource_manager)
    episode_storage = MagicMock()
    episode_storage.add_episodes = AsyncMock()
    patched_resource_manager.get_episode_storage = AsyncMock(
        return_value=episode_storage
    )
    supplied_uid = _uid("supplied")
    minted_uid = _uid("minted")
    monkeypatch.setattr("memmachine_server.main.memmachine.uuid4", lambda: minted_uid)

    result = await memmachine.add_episodes(
        DummySessionData("single"),
        [
            EpisodeEntry(
                uid=supplied_uid,
                content="hello",
                producer_id="user",
                producer_role="user",
            )
        ],
        target_memories=[],
    )

    store_call = episode_storage.add_episodes.await_args
    assert store_call is not None
    stored_entries = store_call.args[1]
    assert result == [minted_uid]
    assert stored_entries[0].uid == minted_uid


@pytest.mark.asyncio
async def test_add_episodes_batch_assigns_sorted_uuids_in_input_order(
    minimal_conf, patched_resource_manager, monkeypatch
):
    memmachine = MemMachine(minimal_conf, patched_resource_manager)
    episode_storage = MagicMock()
    episode_storage.add_episodes = AsyncMock()
    patched_resource_manager.get_episode_storage = AsyncMock(
        return_value=episode_storage
    )
    entries = [
        EpisodeEntry(content=str(index), producer_id="user", producer_role="user")
        for index in range(3)
    ]
    generated_ids = [
        UUID(f"{prefix}0000000-0000-4000-8000-000000000000") for prefix in "fa1"
    ]
    generated = iter(generated_ids)
    monkeypatch.setattr(
        "memmachine_server.main.memmachine.uuid4", lambda: next(generated)
    )
    result = await memmachine.add_episodes(
        DummySessionData("batch"), entries, target_memories=[]
    )

    store_call = episode_storage.add_episodes.await_args
    assert store_call is not None
    stored = store_call.args[1]
    assert [entry.content for entry in stored] == ["0", "1", "2"]
    assert result == [entry.uid for entry in stored]
    assert result == sorted(generated_ids)


@pytest.mark.asyncio
async def test_add_episodes_writes_to_all_backends_concurrently(
    minimal_conf, patched_resource_manager
):
    memmachine = MemMachine(minimal_conf, patched_resource_manager)
    session = DummySessionData("parallel-writes")
    entries = [
        EpisodeEntry(
            content="hello",
            producer_id="user",
            producer_role="assistant",
            metadata={},
        )
    ]
    stored_episodes = [_make_episode(entries[0].uid, session.session_key)]
    started: set[str] = set()
    stored_entries: list[EpisodeEntry] = []
    episodic_episodes: list[Episode] = []
    semantic_episodes: list[Episode] = []
    all_started = asyncio.Event()
    release = asyncio.Event()

    async def hold_write(name: str, result=None):
        started.add(name)
        if len(started) == 3:
            all_started.set()
        await release.wait()
        return result

    async def hold_store(
        _session_key: str,
        normalized_entries: list[EpisodeEntry],
    ):
        stored_entries.extend(normalized_entries)
        return await hold_write("store", stored_episodes)

    async def hold_episodic(episodes: list[Episode]):
        episodic_episodes.extend(episodes)
        await hold_write("episodic")

    async def hold_semantic(
        *,
        episodes: list[Episode],
        session_data: DummySessionData,
    ):
        assert session_data is session
        semantic_episodes.extend(episodes)
        await hold_write("semantic")

    episode_storage = MagicMock()
    episode_storage.add_episodes = AsyncMock(side_effect=hold_store)
    patched_resource_manager.get_episode_storage = AsyncMock(
        return_value=episode_storage
    )

    episodic_session = MagicMock()
    episodic_session.add_memory_episodes = AsyncMock(side_effect=hold_episodic)
    episodic_manager = MagicMock()
    episodic_manager.open_or_create_episodic_memory.return_value = _async_cm(
        episodic_session
    )
    patched_resource_manager.get_episodic_memory_manager = AsyncMock(
        return_value=episodic_manager
    )

    semantic_manager = MagicMock()
    semantic_manager.add_message = AsyncMock(side_effect=hold_semantic)
    patched_resource_manager.get_semantic_session_manager = AsyncMock(
        return_value=semantic_manager
    )

    add_task = asyncio.create_task(memmachine.add_episodes(session, entries))
    try:
        await asyncio.wait_for(all_started.wait(), timeout=5)
        assert started == {"store", "episodic", "semantic"}
    finally:
        release.set()
        episode_ids = await add_task

    assert episode_ids == [stored_entries[0].uid]
    assert stored_entries[0].created_at is not None
    assert stored_entries[0].created_at == episodic_episodes[0].created_at
    assert stored_entries[0].created_at == semantic_episodes[0].created_at
    assert episodic_episodes[0].uid == episode_ids[0]
    assert semantic_episodes[0].uid == episode_ids[0]
    assert episodic_episodes[0].metadata is None
    assert semantic_episodes[0].metadata is None


@pytest.mark.asyncio
async def test_add_episodes_waits_for_started_writes_before_raising(
    minimal_conf, patched_resource_manager
):
    memmachine = MemMachine(minimal_conf, patched_resource_manager)
    session = DummySessionData("failed-parallel-write")
    entries = [
        EpisodeEntry(
            content="hello",
            producer_id="user",
            producer_role="assistant",
        )
    ]
    episodic_started = asyncio.Event()
    episodic_release = asyncio.Event()
    episodic_finished = asyncio.Event()

    episode_storage = MagicMock()
    episode_storage.add_episodes = AsyncMock(
        side_effect=RuntimeError("store write failed")
    )
    patched_resource_manager.get_episode_storage = AsyncMock(
        return_value=episode_storage
    )

    async def hold_episodic_write(*_args):
        episodic_started.set()
        await episodic_release.wait()
        episodic_finished.set()

    episodic_session = MagicMock()
    episodic_session.add_memory_episodes = AsyncMock(side_effect=hold_episodic_write)
    episodic_manager = MagicMock()
    episodic_manager.open_or_create_episodic_memory.return_value = _async_cm(
        episodic_session
    )
    patched_resource_manager.get_episodic_memory_manager = AsyncMock(
        return_value=episodic_manager
    )

    add_task = asyncio.create_task(
        memmachine.add_episodes(
            session,
            entries,
            target_memories=[MemoryType.Episodic],
        )
    )
    await asyncio.wait_for(episodic_started.wait(), timeout=5)
    await asyncio.sleep(0)

    try:
        assert not add_task.done()
    finally:
        episodic_release.set()

    with pytest.raises(RuntimeError, match="store write failed"):
        await add_task
    assert episodic_finished.is_set()


@pytest.mark.asyncio
async def test_add_episodes_reports_every_failed_write(
    minimal_conf, patched_resource_manager, caplog, monkeypatch
):
    memmachine = MemMachine(minimal_conf, patched_resource_manager)
    minted_uid = _uid("minted")
    monkeypatch.setattr("memmachine_server.main.memmachine.uuid4", lambda: minted_uid)
    episode_storage = MagicMock()
    episode_storage.add_episodes = AsyncMock(side_effect=RuntimeError("store failed"))
    patched_resource_manager.get_episode_storage = AsyncMock(
        return_value=episode_storage
    )
    episodic_session = MagicMock()
    episodic_session.add_memory_episodes = AsyncMock(
        side_effect=ValueError("episodic failed")
    )
    episodic_manager = MagicMock()
    episodic_manager.open_or_create_episodic_memory.return_value = _async_cm(
        episodic_session
    )
    patched_resource_manager.get_episodic_memory_manager = AsyncMock(
        return_value=episodic_manager
    )

    entry = EpisodeEntry(content="hello", producer_id="user", producer_role="user")
    with pytest.raises(RuntimeError, match="store failed"):
        await memmachine.add_episodes(
            DummySessionData("failed-writes"),
            [entry],
            target_memories=[MemoryType.Episodic],
        )

    assert "episodic failed" in caplog.text
    assert str(minted_uid) in caplog.text


@pytest.mark.asyncio
async def test_add_episodes_finishes_episodic_write_before_closing(
    minimal_conf, patched_resource_manager
):
    memmachine = MemMachine(minimal_conf, patched_resource_manager)
    session = DummySessionData("session-add-close")
    memory = _ClosingEpisodicSession()
    episode_storage = MagicMock()
    episode_storage.add_episodes = AsyncMock(
        return_value=[_make_episode(_uid("ep1"), session.session_key)]
    )
    patched_resource_manager.get_episode_storage = AsyncMock(
        return_value=episode_storage
    )
    episodic_manager = MagicMock()
    episodic_manager.open_or_create_episodic_memory.side_effect = lambda **kwargs: (
        _closing_episodic_cm(memory)
    )
    patched_resource_manager.get_episodic_memory_manager = AsyncMock(
        return_value=episodic_manager
    )

    await memmachine.add_episodes(
        session,
        [EpisodeEntry(content="hello", producer_id="user", producer_role="user")],
        target_memories=[MemoryType.Episodic],
    )

    assert memory.operations == ["add", "close"]


@pytest.mark.asyncio
async def test_add_episodes_skips_memories_not_requested(
    minimal_conf, patched_resource_manager
):
    memmachine = MemMachine(minimal_conf, patched_resource_manager)
    session = DummySessionData("only-semantic")

    entries = [
        EpisodeEntry(content="hello", producer_id="user", producer_role="assistant"),
    ]
    stored_episodes = [_make_episode(_uid("e1"), session.session_key)]

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
    minimal_conf, patched_resource_manager, partition_registry
):
    memmachine = MemMachine(minimal_conf, patched_resource_manager)
    session = DummySessionData("session-list")

    episode_storage = MagicMock()
    partition_key = await _create_partition(partition_registry, session.session_key)
    episodes = [_make_episode(_uid("e1"), partition_key)]
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
    minimal_conf, patched_resource_manager, partition_registry
):
    memmachine = MemMachine(minimal_conf, patched_resource_manager)
    session = DummySessionData("session-count")

    partition_key = await _create_partition(partition_registry, session.session_key)
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
            value=partition_key,
        )
    )


@pytest.mark.asyncio
async def test_count_episodes_combines_search_filter(
    minimal_conf, patched_resource_manager, monkeypatch, partition_registry
):
    memmachine = MemMachine(minimal_conf, patched_resource_manager)
    session = DummySessionData("session-with-filter")
    custom_filter = FilterComparison(field="topic", op="=", value="alpha")
    parsed_specs: list[str] = []

    def _fake_parse(spec: str | None):
        parsed_specs.append(spec or "")
        return custom_filter

    monkeypatch.setattr("memmachine_server.main.memmachine.parse_filter", _fake_parse)

    partition_key = await _create_partition(partition_registry, session.session_key)
    episode_storage = MagicMock()
    episode_storage.get_episode_messages_count = AsyncMock(return_value=3)
    patched_resource_manager.get_episode_storage = AsyncMock(
        return_value=episode_storage
    )

    result = await memmachine.episodes_count(session, search_filter="topic = 'alpha'")

    assert result == 3
    assert parsed_specs == ["topic = 'alpha'"]

    await_args = episode_storage.get_episode_messages_count.await_args
    assert await_args is not None
    combined_filter = await_args.kwargs["filter_expr"]
    assert combined_filter == FilterAnd(
        left=FilterComparison(
            field="session_key",
            op="=",
            value=partition_key,
        ),
        right=custom_filter,
    )


@pytest.mark.asyncio
async def test_delete_episodes_cleans_missing_store_id(
    minimal_conf, patched_resource_manager, partition_registry
):
    memmachine = MemMachine(minimal_conf, patched_resource_manager)
    episode_storage = MagicMock()
    episode_storage.get_episodes = AsyncMock(return_value=[])
    episode_storage.delete_episodes = AsyncMock()
    patched_resource_manager.get_episode_storage = AsyncMock(
        return_value=episode_storage
    )
    episodic_manager = MagicMock()
    episodic_manager.open_episodic_memory.return_value = _async_cm(AsyncMock())
    patched_resource_manager.get_episodic_memory_manager = AsyncMock(
        return_value=episodic_manager
    )

    await _create_partition(partition_registry, "session")

    await memmachine.delete_episodes(
        [_uid("missing-episode")], session_data=DummySessionData("session")
    )

    episode_storage.get_episodes.assert_awaited_once_with([_uid("missing-episode")])
    episode_storage.delete_episodes.assert_awaited_once_with([_uid("missing-episode")])


@pytest.mark.asyncio
async def test_delete_episodes_rejects_ids_from_another_session_before_deleting(
    minimal_conf, patched_resource_manager, partition_registry
):
    memmachine = MemMachine(minimal_conf, patched_resource_manager)
    requested_session = DummySessionData("requested-session")
    partition_key = await _create_partition(
        partition_registry, requested_session.session_key
    )
    own_id = _uid("own-episode")
    other_id = _uid("other-episode")

    episode_storage = MagicMock()
    episode_storage.get_episodes = AsyncMock(
        return_value=[
            _make_episode(own_id, partition_key),
            _make_episode(other_id, "other-session"),
        ]
    )
    episode_storage.delete_episodes = AsyncMock()
    patched_resource_manager.get_episode_storage = AsyncMock(
        return_value=episode_storage
    )
    episodic_manager = MagicMock()
    episodic_manager.open_episodic_memory.return_value = _async_cm(AsyncMock())
    patched_resource_manager.get_episodic_memory_manager = AsyncMock(
        return_value=episodic_manager
    )

    with pytest.raises(ResourceNotFoundError):
        await memmachine.delete_episodes(
            [own_id, other_id], session_data=requested_session
        )

    episode_storage.delete_episodes.assert_not_awaited()
    patched_resource_manager.get_episodic_memory_manager.assert_not_awaited()
    patched_resource_manager.get_semantic_storage.assert_not_awaited()


@pytest.mark.asyncio
async def test_delete_missing_episode_without_episodic_session(
    minimal_conf, patched_resource_manager, partition_registry
):
    memmachine = MemMachine(minimal_conf, patched_resource_manager)
    episode_storage = MagicMock()
    episode_storage.get_episodes = AsyncMock(return_value=[])
    episode_storage.delete_episodes = AsyncMock()
    patched_resource_manager.get_episode_storage = AsyncMock(
        return_value=episode_storage
    )

    @asynccontextmanager
    async def missing_session():
        raise SessionNotFoundError("missing")
        yield

    episodic_manager = MagicMock()
    episodic_manager.open_episodic_memory.return_value = missing_session()
    patched_resource_manager.get_episodic_memory_manager = AsyncMock(
        return_value=episodic_manager
    )

    await _create_partition(partition_registry, "missing")
    uid = _uid("missing-episode")
    await memmachine.delete_episodes([uid], session_data=DummySessionData("missing"))

    episode_storage.delete_episodes.assert_awaited_once_with([uid])


@pytest.mark.asyncio
async def test_delete_episodes_forwards_to_storage_and_memories(
    minimal_conf, patched_resource_manager, partition_registry
):
    memmachine = MemMachine(minimal_conf, patched_resource_manager)
    session = DummySessionData("session-del")

    partition_key = await _create_partition(partition_registry, session.session_key)

    episode_storage = MagicMock()
    episode_storage.get_episodes = AsyncMock(
        return_value=[
            _make_episode("ep1", partition_key),
            _make_episode("ep2", partition_key),
        ]
    )
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

    await memmachine.delete_episodes([_uid("ep1"), _uid("ep2")], session_data=session)

    episode_storage.delete_episodes.assert_awaited_once_with([_uid("ep1"), _uid("ep2")])
    episodic_session.delete_episodes.assert_awaited_once_with(
        [_uid("ep1"), _uid("ep2")]
    )


@pytest.mark.asyncio
async def test_delete_episodes_finishes_episodic_write_before_closing(
    minimal_conf, patched_resource_manager, partition_registry
):
    memmachine = MemMachine(minimal_conf, patched_resource_manager)
    session = DummySessionData("session-delete-close")
    partition_key = await _create_partition(partition_registry, session.session_key)
    memory = _ClosingEpisodicSession()
    episode_storage = MagicMock()
    episode_storage.get_episodes = AsyncMock(
        return_value=[_make_episode("ep1", partition_key)]
    )
    episode_storage.delete_episodes = AsyncMock()
    patched_resource_manager.get_episode_storage = AsyncMock(
        return_value=episode_storage
    )
    episodic_manager = MagicMock()
    episodic_manager.open_episodic_memory.side_effect = lambda session_key: (
        _closing_episodic_cm(memory)
    )
    patched_resource_manager.get_episodic_memory_manager = AsyncMock(
        return_value=episodic_manager
    )

    await memmachine.delete_episodes([_uid("ep1")], session_data=session)

    assert memory.operations == ["delete", "close"]


@pytest.mark.asyncio
async def test_delete_episodes_requires_session_data(
    minimal_conf, patched_resource_manager
):
    memmachine = MemMachine(minimal_conf, patched_resource_manager)

    with pytest.raises(TypeError, match="session_data"):
        await memmachine.delete_episodes([_uid("ep1")])  # ty: ignore[missing-argument]

    patched_resource_manager.get_episode_storage.assert_not_awaited()


@pytest.mark.asyncio
async def test_delete_episodes_rejects_none_session_data(
    minimal_conf, patched_resource_manager
):
    memmachine = MemMachine(minimal_conf, patched_resource_manager)

    with pytest.raises(ValueError, match="session_data is required"):
        await memmachine.delete_episodes([_uid("ep1")], session_data=None)  # ty: ignore[invalid-argument-type]

    patched_resource_manager.get_episode_storage.assert_not_awaited()


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


def _session_machine(minimal_conf, patched_resource_manager):
    """A MemMachine over a session data manager double."""
    session_manager = AsyncMock()
    patched_resource_manager.get_session_data_manager = AsyncMock(
        return_value=session_manager
    )
    return MemMachine(minimal_conf, patched_resource_manager), session_manager


@pytest.fixture
def no_retry_delay(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(memmachine_module, "_GET_OR_CREATE_RETRY_DELAY_SECONDS", 0)


@pytest.mark.asyncio
async def test_start_starts_the_registry_of_memmachines_partition_namespace(
    minimal_conf, patched_resource_manager, partitioned_store
):
    minimal_conf.session_manager.database = "session_db"
    minimal_conf.session_manager.tombstone_retention_seconds = 600
    minimal_conf.semantic_memory.enabled = False
    machine = MemMachine(minimal_conf, patched_resource_manager)

    await machine.start()

    patched_resource_manager.get_partition_registry.assert_awaited_once_with(
        "session_db",
        partition_namespace="memmachine",
        tombstone_retention_seconds=600,
        partitioned_stores=[partitioned_store] * 3,
    )
    await machine.stop()


@pytest.mark.asyncio
async def test_a_sessions_record_is_written_while_its_partition_is_pending(
    minimal_conf, patched_resource_manager, partition_registry, session_data_store
):
    machine, session_manager = _session_machine(minimal_conf, patched_resource_manager)
    states: list[str] = []

    async def record_written(**kwargs) -> None:
        try:
            registration = await partition_registry.resolve("alpha")
        except PartitionPendingError:
            states.append("pending")
        else:
            states.append("absent" if registration is None else "live")

    session_manager.create_or_validate_session.side_effect = record_written

    await machine.create_session("alpha", description="demo")

    # Written by the creation, then checked once the partition is live.
    assert states == ["pending", "live"]
    partition_key = await _partition_key(partition_registry, "alpha")
    for call in session_manager.create_or_validate_session.await_args_list:
        assert call.kwargs["session_key"] == partition_key
        assert call.kwargs["param"].session_key == partition_key
        assert call.kwargs["description"] == "demo"
        assert call.kwargs["metadata"] == {}
    session_manager.get_session_info.assert_awaited_once_with(
        partition_key, status=SessionDataManager.SessionStatus.Active
    )


@pytest.mark.asyncio
async def test_creating_a_session_whose_partition_is_live_checks_its_record(
    minimal_conf, patched_resource_manager, partition_registry, partitioned_store
):
    machine, session_manager = _session_machine(minimal_conf, patched_resource_manager)
    partition_key = await _create_partition(partition_registry, "alpha")

    await machine.create_session("alpha")

    partitioned_store.create_partition.assert_not_awaited()
    kwargs = session_manager.create_or_validate_session.await_args.kwargs
    assert kwargs["session_key"] == partition_key


@pytest.mark.asyncio
async def test_creating_a_session_another_creation_holds_pending_raises_pending(
    minimal_conf, patched_resource_manager, partition_registry
):
    machine, session_manager = _session_machine(minimal_conf, patched_resource_manager)
    await partition_registry.reserve("alpha")

    with pytest.raises(PartitionPendingError):
        await machine.create_session("alpha")

    session_manager.create_or_validate_session.assert_not_awaited()
    assert await machine.get_session("alpha") is None


@pytest.mark.asyncio
async def test_a_stored_record_that_conflicts_names_the_session_not_its_partition(
    minimal_conf, patched_resource_manager
):
    machine, session_manager = _session_machine(minimal_conf, patched_resource_manager)
    session_manager.create_or_validate_session.side_effect = SessionAlreadyExistsError(
        "partition-key"
    )

    with pytest.raises(SessionAlreadyExistsError, match="'alpha'"):
        await machine.create_session("alpha")


@pytest.mark.asyncio
async def test_an_invalid_configuration_creates_no_partition(
    minimal_conf_factory, patched_resource_manager, partition_registry
):
    conf = minimal_conf_factory()
    conf.episodic_memory.long_term_memory = LongTermMemoryConfPartial(
        backend="event",
        vector_store="vs",
        segment_store="ss",
        embedder="default-embedder",
    )
    machine, _ = _session_machine(conf, patched_resource_manager)

    with pytest.raises(ConfigurationError, match="no event memory"):
        await machine.create_session("alpha", user_conf=_EVENT_PROJECT)

    assert await partition_registry.resolve("alpha") is None


@pytest.mark.asyncio
async def test_a_session_key_the_registry_refuses_is_an_invalid_argument(
    minimal_conf, patched_resource_manager
):
    machine, _ = _session_machine(minimal_conf, patched_resource_manager)

    with pytest.raises(InvalidArgumentError, match="Partition name"):
        await machine.create_session("x" * 256)

    assert await machine.get_session("x" * 256) is None
    with pytest.raises(SessionNotFoundError):
        await machine.delete_session(DummySessionData("x" * 256))


@pytest.mark.asyncio
async def test_deleting_a_session_frees_its_key_for_a_session_with_another_partition(
    minimal_conf, patched_resource_manager, partition_registry
):
    machine, session_manager = _session_machine(minimal_conf, patched_resource_manager)
    await machine.create_session("alpha")
    first_key = await _partition_key(partition_registry, "alpha")

    await machine.delete_session(DummySessionData("alpha"))

    assert await machine.get_session("alpha") is None
    # The record stays for the partition's purge, under the deleted key.
    session_manager.delete_session.assert_not_awaited()

    await machine.create_session("alpha")

    second_key = await _partition_key(partition_registry, "alpha")
    assert second_key != first_key
    assert (
        session_manager.create_or_validate_session.await_args.kwargs["session_key"]
        == second_key
    )


@pytest.mark.asyncio
async def test_deleting_a_pending_session_frees_its_key(
    minimal_conf, patched_resource_manager, partition_registry
):
    minimal_conf.semantic_memory.storage_configured.return_value = True
    machine, _ = _session_machine(minimal_conf, patched_resource_manager)
    reservation = await partition_registry.reserve("alpha")

    await machine.delete_session(DummySessionData("alpha"))

    assert await partition_registry.resolve("alpha") is None
    with pytest.raises(PartitionDeletedError):
        await reservation.confirm()
    patched_resource_manager.get_semantic_storage.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("enabled", [True, False])
async def test_deleting_a_session_deletes_its_projects_semantic_sets_before_freeing_its_key(
    minimal_conf, patched_resource_manager, partition_registry, enabled
):
    minimal_conf.semantic_memory.enabled = enabled
    minimal_conf.semantic_memory.storage_configured.return_value = True
    semantic_storage = AsyncMock()

    async def while_the_key_names_the_session(**kwargs) -> None:
        assert await partition_registry.resolve("alpha") is not None

    semantic_storage.delete_feature_set.side_effect = while_the_key_names_the_session
    patched_resource_manager.get_semantic_storage = AsyncMock(
        return_value=semantic_storage
    )
    machine, _ = _session_machine(minimal_conf, patched_resource_manager)
    await _create_partition(partition_registry, "alpha")

    await machine.delete_session(
        DummySessionData("alpha", org_id="org-a", project_id="proj-a")
    )

    semantic_storage.delete_feature_set.assert_awaited_once_with(
        filter_expr=with_has_set_ids(
            SemanticSessionManager.project_deletion_set_ids(
                org_id="org-a", project_id="proj-a"
            ),
            None,
        )
    )
    assert await partition_registry.resolve("alpha") is None


@pytest.mark.asyncio
async def test_an_unreachable_semantic_store_leaves_the_session_live(
    minimal_conf, patched_resource_manager, partition_registry
):
    minimal_conf.semantic_memory.storage_configured.return_value = True
    patched_resource_manager.get_semantic_storage = AsyncMock(
        side_effect=ConnectionError("semantic store unreachable")
    )
    machine, _ = _session_machine(minimal_conf, patched_resource_manager)
    await _create_partition(partition_registry, "alpha")

    with pytest.raises(ConnectionError):
        await machine.delete_session(DummySessionData("alpha"))

    assert await partition_registry.resolve("alpha") is not None


@pytest_asyncio.fixture
async def session_data_manager(tmp_path):
    """A session data manager on a SQLite file."""
    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'sessions.db'}")
    manager = SessionDataManagerSQL(engine=engine)
    await manager.prepare()
    yield manager
    await engine.dispose()


def _machine_with_records(
    minimal_conf, patched_resource_manager, session_data_manager
) -> MemMachine:
    """A MemMachine that writes its session records to a database."""
    patched_resource_manager.get_session_data_manager = AsyncMock(
        return_value=session_data_manager
    )
    return MemMachine(minimal_conf, patched_resource_manager)


@pytest.mark.asyncio
async def test_a_session_created_by_an_add_has_its_record(
    minimal_conf, patched_resource_manager, session_data_manager, session_data_store
):
    episode_storage = MagicMock()
    episode_storage.add_episodes = AsyncMock()
    patched_resource_manager.get_episode_storage = AsyncMock(
        return_value=episode_storage
    )
    machine = _machine_with_records(
        minimal_conf, patched_resource_manager, session_data_manager
    )

    await machine.add_episodes(
        DummySessionData("org/a", org_id="org"), [_ENTRY], target_memories=[]
    )

    assert await machine.get_session("org/a") is not None


@pytest.mark.asyncio
async def test_search_sessions_lists_the_pending_and_live_sessions_with_a_key_prefix(
    minimal_conf, patched_resource_manager, partition_registry
):
    machine, _ = _session_machine(minimal_conf, patched_resource_manager)
    for session_key in ("org/a", "org/deleted", "other/c"):
        await machine.create_session(session_key)
    await machine.delete_session(DummySessionData("org/deleted"))
    await partition_registry.reserve("org/pending")

    listed = await machine.search_sessions("org/")

    assert [(session.partition_name, session.live) for session in listed] == [
        ("org/a", True),
        ("org/pending", False),
    ]
    assert listed[0].partition_uuid == UUID(
        await _partition_key(partition_registry, "org/a")
    )


@pytest.mark.asyncio
async def test_search_sessions_reads_a_listing_longer_than_one_registry_read(
    minimal_conf, patched_resource_manager, partition_registry
):
    partition_registry._max_list_limit = 2
    machine, _ = _session_machine(minimal_conf, patched_resource_manager)
    session_keys = [f"org/{index}" for index in range(5)]
    for session_key in session_keys:
        await machine.create_session(session_key)

    listed = await machine.search_sessions("org/")

    assert [session.partition_name for session in listed] == session_keys


@pytest.mark.asyncio
async def test_a_sessions_flags_are_updated_under_its_partition_key(
    minimal_conf, patched_resource_manager, partition_registry
):
    machine, session_manager = _session_machine(minimal_conf, patched_resource_manager)
    partition_key = await _create_partition(partition_registry, "alpha")

    await machine.update_session_episodic_config("alpha", enabled=False)

    session_manager.update_session_episodic_config.assert_awaited_once_with(
        session_key=partition_key,
        enabled=False,
        long_term_memory_enabled=None,
        short_term_memory_enabled=None,
    )
    with pytest.raises(SessionNotFoundError, match="'beta'"):
        await machine.update_session_episodic_config("beta", enabled=False)


def _add_episodes_machine(minimal_conf, patched_resource_manager):
    """A MemMachine whose episode store and episodic memory are doubles."""
    episode_storage = MagicMock()
    episode_storage.add_episodes = AsyncMock()
    patched_resource_manager.get_episode_storage = AsyncMock(
        return_value=episode_storage
    )
    episodic_manager = MagicMock()
    episodic_manager.open_or_create_episodic_memory.side_effect = lambda **kwargs: (
        _async_cm(AsyncMock())
    )
    patched_resource_manager.get_episodic_memory_manager = AsyncMock(
        return_value=episodic_manager
    )
    machine = MemMachine(minimal_conf, patched_resource_manager)
    return machine, episode_storage, episodic_manager


_ENTRY = EpisodeEntry(content="hello", producer_id="user", producer_role="assistant")


@pytest.mark.asyncio
async def test_an_add_creates_its_session_and_writes_under_the_partition_key(
    minimal_conf, patched_resource_manager, partition_registry
):
    machine, episode_storage, episodic_manager = _add_episodes_machine(
        minimal_conf, patched_resource_manager
    )

    await machine.add_episodes(
        DummySessionData("alpha"), [_ENTRY], target_memories=[MemoryType.Episodic]
    )

    partition_key = await _partition_key(partition_registry, "alpha")
    assert episode_storage.add_episodes.await_args.args[0] == partition_key
    opened = episodic_manager.open_or_create_episodic_memory.call_args.kwargs
    assert opened["session_key"] == partition_key
    assert opened["episodic_memory_config"].session_key == partition_key


@pytest.mark.asyncio
async def test_an_add_to_a_session_deleted_meanwhile_is_not_acknowledged(
    minimal_conf, patched_resource_manager, partition_registry
):
    machine, episode_storage, _ = _add_episodes_machine(
        minimal_conf, patched_resource_manager
    )
    await _create_partition(partition_registry, "alpha")

    async def deleted_while_the_episodes_are_written(*args) -> None:
        await partition_registry.unregister("alpha")

    episode_storage.add_episodes.side_effect = deleted_while_the_episodes_are_written

    with pytest.raises(SessionNotFoundError, match="'alpha'"):
        await machine.add_episodes(
            DummySessionData("alpha"), [_ENTRY], target_memories=[MemoryType.Episodic]
        )


@pytest.mark.asyncio
@pytest.mark.usefixtures("no_retry_delay")
async def test_an_add_waits_for_a_session_another_creation_holds_pending(
    minimal_conf, patched_resource_manager, partition_registry
):
    machine, episode_storage, _ = _add_episodes_machine(
        minimal_conf, patched_resource_manager
    )
    reservation = await partition_registry.reserve("alpha")
    resolve = partition_registry.resolve
    lookups = 0

    async def confirmed_after_the_first_lookup(partition_name: str):
        nonlocal lookups
        lookups += 1
        if lookups == 2:
            await reservation.confirm()
        return await resolve(partition_name)

    partition_registry.resolve = confirmed_after_the_first_lookup

    await machine.add_episodes(
        DummySessionData("alpha"), [_ENTRY], target_memories=[MemoryType.Episodic]
    )

    assert episode_storage.add_episodes.await_args.args[0] == str(
        reservation.partition_uuid
    )


@pytest.mark.asyncio
@pytest.mark.usefixtures("no_retry_delay")
async def test_an_add_reports_a_session_that_stays_pending(
    minimal_conf, patched_resource_manager, partition_registry
):
    machine, episode_storage, _ = _add_episodes_machine(
        minimal_conf, patched_resource_manager
    )
    await partition_registry.reserve("alpha")

    with pytest.raises(PartitionPendingError):
        await machine.add_episodes(
            DummySessionData("alpha"), [_ENTRY], target_memories=[MemoryType.Episodic]
        )

    episode_storage.add_episodes.assert_not_awaited()


@pytest.mark.asyncio
async def test_a_session_no_partition_holds_lists_and_counts_no_episodes(
    minimal_conf, patched_resource_manager
):
    episode_storage = AsyncMock()
    patched_resource_manager.get_episode_storage = AsyncMock(
        return_value=episode_storage
    )
    machine = MemMachine(minimal_conf, patched_resource_manager)

    listed = await machine.list_search(
        DummySessionData("alpha"), target_memories=[MemoryType.Episodic]
    )

    assert listed.episodic_memory == []
    assert await machine.episodes_count(DummySessionData("alpha")) == 0
    episode_storage.get_episode_messages.assert_not_awaited()
    episode_storage.get_episode_messages_count.assert_not_awaited()


@pytest.mark.asyncio
async def test_add_episodes_skips_semantic_memory_when_disabled(
    minimal_conf, patched_resource_manager, monkeypatch, partition_registry
):
    """A write must not reach semantic memory while it is disabled.

    ``target_memories`` defaults to every memory type, so a plain
    ``POST /memories`` asks for semantic memory without knowing whether it is
    configured. Reaching the backend here raises ``ResourceNotReadyError``,
    which no route handles.
    """
    minimal_conf.semantic_memory.enabled = False
    memmachine = MemMachine(minimal_conf, patched_resource_manager)
    minted_uid = _uid("minted")
    monkeypatch.setattr("memmachine_server.main.memmachine.uuid4", lambda: minted_uid)
    session = DummySessionData("disabled-write")
    partition_key = await _create_partition(partition_registry, session.session_key)
    created_at = datetime(2026, 1, 2, tzinfo=UTC)

    entries = [
        EpisodeEntry(
            uid=_uid("e1"),
            content="hello",
            producer_id="user",
            producer_role="assistant",
            created_at=created_at,
        ),
    ]
    episodes = [
        Episode(
            uid=_uid("e1"),
            content="hello",
            session_key=partition_key,
            created_at=created_at,
            producer_id="user",
            producer_role="assistant",
        )
    ]

    episode_storage = MagicMock()
    episode_storage.add_episodes = AsyncMock(return_value=episodes)
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
    patched_resource_manager.get_semantic_session_manager = AsyncMock()

    episode_ids = await memmachine.add_episodes(session, entries)

    patched_resource_manager.get_semantic_session_manager.assert_not_called()
    expected_episodes = [episodes[0].model_copy(update={"uid": minted_uid})]
    episodic_session.add_memory_episodes.assert_awaited_once_with(expected_episodes)
    assert episode_ids == [minted_uid]


@pytest.mark.asyncio
async def test_query_search_skips_semantic_memory_when_disabled(
    minimal_conf, patched_resource_manager, monkeypatch
):
    """A search must degrade to episodic-only while semantic memory is disabled."""
    minimal_conf.semantic_memory.enabled = False
    dummy_session = DummySessionData("disabled-search")

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
    patched_resource_manager.get_semantic_session_manager = AsyncMock()

    memmachine = MemMachine(minimal_conf, patched_resource_manager)

    result = await memmachine.query_search(dummy_session, query="find")

    patched_resource_manager.get_semantic_session_manager.assert_not_called()
    async_episodic.assert_awaited_once()
    assert result.semantic_memory is None
    assert result.episodic_memory is not None


@pytest.mark.asyncio
async def test_query_search_uses_semantic_memory_when_enabled(
    minimal_conf, patched_resource_manager, monkeypatch
):
    """The enabled counterpart of the disabled tests above.

    ``minimal_conf.semantic_memory`` is a ``MagicMock``, so ``enabled`` is
    already truthy and the assignment below states intent rather than changing
    state. The discrimination therefore comes from the disabled siblings, which
    set the flag to ``False``; this test only pins that an enabled server still
    reaches semantic memory.
    """
    minimal_conf.semantic_memory.enabled = True
    dummy_session = DummySessionData("enabled-search")

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

    semantic_features = [
        SemanticFeature(
            category="profile",
            tag="name",
            feature_name="value",
            value="semantic-response",
        )
    ]

    async def mock_search(*args, **kwargs):
        for feature in semantic_features:
            yield feature

    semantic_manager = MagicMock()
    semantic_manager.search = mock_search
    patched_resource_manager.get_semantic_session_manager = AsyncMock(
        return_value=semantic_manager
    )

    memmachine = MemMachine(minimal_conf, patched_resource_manager)

    result = await memmachine.query_search(dummy_session, query="find")

    patched_resource_manager.get_semantic_session_manager.assert_awaited_once()
    assert result.semantic_memory == semantic_features


@pytest.mark.asyncio
async def test_list_search_skips_semantic_memory_when_disabled(
    minimal_conf, patched_resource_manager, partition_registry
):
    """A list must degrade to episodic-only while semantic memory is disabled."""
    minimal_conf.semantic_memory.enabled = False
    memmachine = MemMachine(minimal_conf, patched_resource_manager)
    session = DummySessionData("disabled-list")

    episode_storage = MagicMock()
    partition_key = await _create_partition(partition_registry, session.session_key)
    episodes = [_make_episode(_uid("e1"), partition_key)]
    episode_storage.get_episode_messages = AsyncMock(return_value=episodes)
    patched_resource_manager.get_episode_storage = AsyncMock(
        return_value=episode_storage
    )
    patched_resource_manager.get_semantic_session_manager = AsyncMock()

    result = await memmachine.list_search(session)

    patched_resource_manager.get_semantic_session_manager.assert_not_called()
    assert result.episodic_memory == episodes
    assert result.semantic_memory is None


@pytest.mark.asyncio
async def test_list_search_uses_semantic_memory_when_enabled(
    minimal_conf, patched_resource_manager
):
    """The enabled counterpart for `list_search`.

    Without this, removing the semantic branch at the `list_search` call site
    leaves the suite green: the only other `list_search` test asks for
    episodic memory alone.
    """
    minimal_conf.semantic_memory.enabled = True
    memmachine = MemMachine(minimal_conf, patched_resource_manager)
    session = DummySessionData("enabled-list")

    episode_storage = MagicMock()
    episode_storage.get_episode_messages = AsyncMock(return_value=[])
    patched_resource_manager.get_episode_storage = AsyncMock(
        return_value=episode_storage
    )

    semantic_features = [
        SemanticFeature(
            category="profile",
            tag="name",
            feature_name="value",
            value="semantic-response",
        )
    ]

    async def mock_get_set_features(*args, **kwargs):
        for feature in semantic_features:
            yield feature

    semantic_manager = MagicMock()
    semantic_manager.get_set_features = mock_get_set_features
    patched_resource_manager.get_semantic_session_manager = AsyncMock(
        return_value=semantic_manager
    )

    result = await memmachine.list_search(session)

    patched_resource_manager.get_semantic_session_manager.assert_awaited_once()
    assert result.semantic_memory == semantic_features


def _delete_episodes_machine(minimal_conf, patched_resource_manager):
    episode_storage = MagicMock()
    episode_storage.get_episodes = AsyncMock(return_value=[])
    episode_storage.delete_episodes = AsyncMock()
    patched_resource_manager.get_episode_storage = AsyncMock(
        return_value=episode_storage
    )
    episodic_manager = MagicMock()
    episodic_manager.open_episodic_memory.return_value = _async_cm(AsyncMock())
    patched_resource_manager.get_episodic_memory_manager = AsyncMock(
        return_value=episodic_manager
    )
    return MemMachine(minimal_conf, patched_resource_manager), episode_storage


@pytest.mark.asyncio
async def test_delete_episodes_skips_semantic_when_no_storage_is_configured(
    minimal_conf, patched_resource_manager, partition_registry
):
    minimal_conf.semantic_memory.storage_configured.return_value = False
    patched_resource_manager.get_semantic_storage = AsyncMock()
    memmachine, episode_storage = _delete_episodes_machine(
        minimal_conf, patched_resource_manager
    )
    await _create_partition(partition_registry, "session")

    await memmachine.delete_episodes(
        [_uid("e1")], session_data=DummySessionData("session")
    )

    patched_resource_manager.get_semantic_storage.assert_not_awaited()
    episode_storage.delete_episodes.assert_awaited_once_with([_uid("e1")])


@pytest.mark.asyncio
@pytest.mark.parametrize("enabled", [True, False])
async def test_delete_episodes_deletes_semantic_history_when_storage_is_configured(
    minimal_conf, patched_resource_manager, enabled, partition_registry
):
    minimal_conf.semantic_memory.enabled = enabled
    minimal_conf.semantic_memory.storage_configured.return_value = True
    semantic_storage = AsyncMock()
    patched_resource_manager.get_semantic_storage = AsyncMock(
        return_value=semantic_storage
    )
    memmachine, episode_storage = _delete_episodes_machine(
        minimal_conf, patched_resource_manager
    )
    await _create_partition(partition_registry, "session")

    await memmachine.delete_episodes(
        [_uid("e1")], session_data=DummySessionData("session")
    )

    semantic_storage.delete_history.assert_awaited_once_with([_uid("e1")])
    episode_storage.delete_episodes.assert_awaited_once_with([_uid("e1")])


@pytest.mark.asyncio
async def test_delete_episodes_deletes_nothing_when_semantic_storage_fails(
    minimal_conf, patched_resource_manager, partition_registry
):
    minimal_conf.semantic_memory.storage_configured.return_value = True
    patched_resource_manager.get_semantic_storage = AsyncMock(
        side_effect=ConnectionError("semantic store unreachable")
    )
    memmachine, episode_storage = _delete_episodes_machine(
        minimal_conf, patched_resource_manager
    )
    await _create_partition(partition_registry, "session")

    with pytest.raises(ConnectionError):
        await memmachine.delete_episodes(
            [_uid("e1")], session_data=DummySessionData("session")
        )

    episode_storage.delete_episodes.assert_not_called()
    patched_resource_manager.get_episodic_memory_manager.assert_not_awaited()


@pytest.mark.asyncio
async def test_delete_episodes_keeps_episodes_when_semantic_history_delete_fails(
    minimal_conf, patched_resource_manager, partition_registry
):
    minimal_conf.semantic_memory.storage_configured.return_value = True
    semantic_storage = AsyncMock()
    semantic_storage.delete_history.side_effect = ConnectionError(
        "semantic store unreachable"
    )
    patched_resource_manager.get_semantic_storage = AsyncMock(
        return_value=semantic_storage
    )
    memmachine, episode_storage = _delete_episodes_machine(
        minimal_conf, patched_resource_manager
    )
    await _create_partition(partition_registry, "session")

    with pytest.raises(ConnectionError):
        await memmachine.delete_episodes(
            [_uid("e1")], session_data=DummySessionData("session")
        )

    episode_storage.delete_episodes.assert_not_called()
    patched_resource_manager.get_episodic_memory_manager.assert_not_awaited()


@pytest.mark.asyncio
async def test_empty_project_set_ids_refuse_to_delete(
    minimal_conf, patched_resource_manager, monkeypatch
):
    monkeypatch.setattr(
        SemanticSessionManager,
        "project_deletion_set_ids",
        classmethod(lambda cls, *, org_id, project_id: []),
    )
    semantic_storage = AsyncMock()
    machine = MemMachine(minimal_conf, patched_resource_manager)

    with pytest.raises(RuntimeError, match="no semantic set ids"):
        await machine._delete_session_semantic_memory(
            DummySessionData("pending"), semantic_storage
        )

    semantic_storage.delete_feature_set.assert_not_awaited()
