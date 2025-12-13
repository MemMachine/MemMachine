"""Tests for the EpisodicMemoryManager class."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import create_async_engine

from memmachine.common.configuration.episodic_config import EpisodicMemoryConf
from memmachine.common.errors import SessionAlreadyExistsError
from memmachine.common.language_model import LanguageModel
from memmachine.common.metrics_factory import MetricsFactory
from memmachine.common.resource_manager import CommonResourceManager
from memmachine.common.session_manager.session_data_manager_sql_impl import (
    SessionDataManagerSQL,
)
from memmachine.episodic_memory.episodic_memory import (
    EpisodicMemory,
    EpisodicMemoryParams,
)
from memmachine.episodic_memory.episodic_memory_manager import (
    EpisodicMemoryManager,
    EpisodicMemoryManagerParams,
)


@pytest_asyncio.fixture
async def db_engine():
    """Fixture for an in-memory SQLite async engine."""
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    yield engine
    await engine.dispose()


@pytest_asyncio.fixture
async def mock_session_storage(db_engine):
    """Fixture for a mocked SessionDataManager."""
    storage = SessionDataManagerSQL(engine=db_engine)
    await storage.create_tables()
    return storage


@pytest.fixture
def mock_metrics_factory():
    """Fixture for a mocked MetricsFactory."""
    global MockMetricsFactory

    class MockMetricsFactory(MetricsFactory):
        def __init__(self):
            self.counters = MagicMock()
            self.gauge = MagicMock()
            self.histogram = MagicMock()
            self.summaries = MagicMock()

        def get_counter(self, name, description, label_names=...):
            return self.counters

        def get_summary(self, name, description, label_names=...):
            return self.summaries

        def get_gauge(self, name, description, label_names=...):
            return self.gauge

        def get_histogram(self, name, description, label_names=...):
            return self.histogram

        def reset(self):
            pass

        def __getstate__(self):
            return {}

        def __setstate__(self, state):
            pass

    factory = MockMetricsFactory()
    return factory


@pytest.fixture
def mock_resource_manager(mock_metrics_factory):
    """Fixture for a mocked ResourceManager."""
    resource_manager = MagicMock(spec=CommonResourceManager)
    resource_manager.get_language_model.return_value = AsyncMock(spec=LanguageModel)
    resource_manager.get_metrics_factory.return_value = mock_metrics_factory
    return resource_manager


@pytest.fixture
def mock_episodic_memory_instance():
    """Fixture for a mocked EpisodicMemory instance."""
    mock_instance = AsyncMock(spec=EpisodicMemory)
    mock_instance.close = AsyncMock()
    mock_instance.delete_data = AsyncMock()
    return mock_instance


@pytest.fixture
def mock_episodic_memory_conf():
    """Fixture for a dummy EpisodicMemoryParams object."""
    return EpisodicMemoryConf(session_key="test_session", enabled=False)


@pytest.fixture
def mock_episodic_memory_manager_param(mock_session_storage, mock_resource_manager):
    """Fixture for EpisodicMemoryManagerParam."""
    return EpisodicMemoryManagerParams(
        instance_cache_size=10,
        max_life_time=3600,
        resource_manager=mock_resource_manager,
        session_data_manager=mock_session_storage,
    )


@pytest_asyncio.fixture
async def manager(mock_episodic_memory_manager_param):
    """Fixture for an EpisodicMemoryManager instance."""
    return EpisodicMemoryManager(params=mock_episodic_memory_manager_param)


@pytest.mark.asyncio
@patch("memmachine.episodic_memory.episodic_memory_manager.EpisodicMemory")
async def test_create_episodic_memory_success(
    mock_episodic_memory_cls,
    manager: EpisodicMemoryManager,
    mock_episodic_memory_conf,
    mock_episodic_memory_instance,
):
    """Test successfully creating a new episodic memory instance."""
    session_key = "new_session"
    description = "A new test session"
    metadata = {"owner": "tester"}
    mock_episodic_memory_cls.return_value = mock_episodic_memory_instance

    # Patch the service locator function
    with patch(
        "memmachine.episodic_memory.episodic_memory_manager.episodic_memory_params_from_config",
        new_callable=AsyncMock,
    ) as mock_params_from_config:
        mock_params_from_config.return_value = MagicMock(spec=EpisodicMemoryParams)

        async with manager.create_episodic_memory(
            session_key,
            mock_episodic_memory_conf,
            description,
            metadata,
        ) as instance:
            assert instance is mock_episodic_memory_instance
            mock_params_from_config.assert_awaited_once_with(
                mock_episodic_memory_conf,
                manager._resource_manager,
            )
            mock_episodic_memory_cls.assert_called_once_with(
                mock_params_from_config.return_value,
            )
            assert manager._instance_cache.get_ref_count(session_key) == 1  # 1 from add

    assert manager._instance_cache.get_ref_count(session_key) == 0  # put is called


@pytest.mark.asyncio
async def test_create_episodic_memory_already_exists(
    manager: EpisodicMemoryManager,
    mock_episodic_memory_conf,
):
    """Test that creating a session that already exists raises an error."""
    session_key = "existing_session"
    async with manager.create_episodic_memory(
        session_key,
        mock_episodic_memory_conf,
        "",
        {},
    ):
        with pytest.raises(
            SessionAlreadyExistsError, match=f"Session '{session_key}' already exists"
        ):
            async with manager.create_episodic_memory(
                session_key,
                mock_episodic_memory_conf,
                "",
                {},
            ):
                pass  # This part should not be reached


@pytest.mark.asyncio
async def test_create_or_open_episodic_memory_success(
    manager: EpisodicMemoryManager,
    mock_episodic_memory_conf,
):
    """Test successfully creating or opening an episodic memory instance."""
    session_key = "create_or_open_session"
    description = "A test session"
    metadata = {"owner": "tester"}

    # Create a new session
    async with manager.open_or_create_episodic_memory(
        session_key,
        mock_episodic_memory_conf,
        description,
        metadata,
    ) as instance:
        assert instance is not None
        assert manager._instance_cache.get_ref_count(session_key) == 1  # 1 from add

    # Open the same session again
    async with manager.open_or_create_episodic_memory(
        session_key,
        mock_episodic_memory_conf,
        description,
        metadata,
    ) as instance:
        assert instance is not None
        assert manager._instance_cache.get_ref_count(session_key) == 1  # 1 from open

    assert manager._instance_cache.get_ref_count(session_key) == 0  # put is called
    await manager.close_session(session_key)

    # Open the same session again, should load from storage
    async with manager.open_or_create_episodic_memory(
        session_key,
        mock_episodic_memory_conf,
        description,
        metadata,
    ) as instance:
        assert instance is not None
        assert manager._instance_cache.get_ref_count(session_key) == 1  # 1 from add


@pytest.mark.asyncio
@patch(
    "memmachine.episodic_memory.episodic_memory_manager.episodic_memory_params_from_config",
    new_callable=AsyncMock,
)
@patch("memmachine.episodic_memory.episodic_memory_manager.EpisodicMemory")
async def test_open_episodic_memory_new_instance(
    mock_episodic_memory_cls,
    mock_params_from_config,
    manager: EpisodicMemoryManager,
    mock_episodic_memory_conf,
    mock_episodic_memory_instance,
):
    """Test opening a session for the first time, loading it from storage."""
    session_key = "session_to_open"
    mock_episodic_memory_cls.return_value = mock_episodic_memory_instance
    mock_params = MagicMock(spec=EpisodicMemoryParams)
    mock_params_from_config.return_value = mock_params

    async with manager.create_episodic_memory(
        session_key,
        mock_episodic_memory_conf,
        "",
        {},
    ) as instance:
        assert instance is mock_episodic_memory_instance
    await manager.close_session(session_key)

    # Reset mocks to test the 'open' logic in isolation
    mock_episodic_memory_cls.reset_mock()
    mock_params_from_config.reset_mock()

    async with manager.open_episodic_memory(session_key) as instance:
        assert instance is mock_episodic_memory_instance
        mock_params_from_config.assert_awaited_once()
        mock_episodic_memory_cls.assert_called_once_with(mock_params)
        assert manager._instance_cache.get_ref_count(session_key) == 1

    assert manager._instance_cache.get_ref_count(session_key) == 0


@pytest.mark.asyncio
@patch("memmachine.episodic_memory.episodic_memory_manager.EpisodicMemory")
async def test_open_episodic_memory_cached_instance(
    mock_episodic_memory_cls,
    manager: EpisodicMemoryManager,
    mock_episodic_memory_conf,
    mock_episodic_memory_instance,
):
    """Test opening a session that is already in the cache."""
    session_key = "cached_session"
    mock_episodic_memory_cls.return_value = mock_episodic_memory_instance

    # Pre-populate the cache
    async with manager.create_episodic_memory(
        session_key,
        mock_episodic_memory_conf,
        "",
        {},
    ):
        pass

    mock_episodic_memory_cls.assert_called_once()
    mock_episodic_memory_cls.reset_mock()

    # Open it again
    async with manager.open_episodic_memory(session_key) as instance:
        assert instance is mock_episodic_memory_instance
        # Should not call storage or create again
        mock_episodic_memory_cls.assert_not_called()
        assert manager._instance_cache.get_ref_count(session_key) == 1

    assert manager._instance_cache.get_ref_count(session_key) == 0


@pytest.mark.asyncio
@patch("memmachine.episodic_memory.episodic_memory_manager.EpisodicMemory")
async def test_delete_episodic_session_not_in_use(
    mock_episodic_memory_cls,
    manager: EpisodicMemoryManager,
    mock_episodic_memory_conf,
    mock_episodic_memory_instance,
):
    """Test deleting a session that is not currently in use."""
    session_key = "session_to_delete"
    mock_episodic_memory_cls.return_value = mock_episodic_memory_instance

    # Create and release the session so it's in cache but not in use
    async with manager.create_episodic_memory(
        session_key,
        mock_episodic_memory_conf,
        "",
        {},
    ):
        pass

    assert manager._instance_cache.get_ref_count(session_key) == 0

    await manager.delete_episodic_session(session_key)

    # Verify it's gone from cache and storage
    assert manager._instance_cache.get(session_key) is None
    mock_episodic_memory_instance.delete_session_episodes.assert_awaited_once()
    mock_episodic_memory_instance.close.assert_awaited_once()


@pytest.mark.asyncio
@patch("memmachine.episodic_memory.episodic_memory_manager.EpisodicMemory")
async def test_delete_episodic_session_in_use_raises_error(
    mock_episodic_memory_cls,
    manager: EpisodicMemoryManager,
    mock_episodic_memory_conf,
    mock_episodic_memory_instance,
):
    """Test deleting a session that is in use."""
    session_key = "session_to_delete"
    mock_episodic_memory_cls.return_value = mock_episodic_memory_instance

    # Create and release the session so it's in cache but not in use
    async with manager.create_episodic_memory(
        session_key,
        mock_episodic_memory_conf,
        "",
        {},
    ):
        assert manager._instance_cache.get_ref_count(session_key) == 1
        with pytest.raises(
            RuntimeError,
            match=f"Session {session_key} is still in use",
        ):
            await manager.delete_episodic_session(session_key)


@pytest.mark.asyncio
@patch("memmachine.episodic_memory.episodic_memory_manager.EpisodicMemory")
async def test_delete_episodic_session_not_in_cache(
    mock_episodic_memory_cls,
    manager: EpisodicMemoryManager,
    mock_episodic_memory_conf,
    mock_episodic_memory_instance,
):
    """Test deleting a session that exists in storage but not in the cache."""
    session_key = "not_in_cache_session"
    mock_episodic_memory_cls.return_value = mock_episodic_memory_instance

    async with manager.create_episodic_memory(
        session_key,
        mock_episodic_memory_conf,
        "",
        {},
    ):
        pass
    await manager.close_session(session_key)

    mock_episodic_memory_cls.assert_called_once()
    mock_episodic_memory_instance.close.assert_awaited_once()
    mock_episodic_memory_cls.reset_mock()
    mock_episodic_memory_instance.reset_mock()

    await manager.delete_episodic_session(session_key)

    # Should load from storage to delete
    mock_episodic_memory_cls.assert_called_once()
    mock_episodic_memory_instance.delete_session_episodes.assert_awaited_once()
    mock_episodic_memory_instance.close.assert_awaited_once()


@pytest.mark.asyncio
@patch("memmachine.episodic_memory.episodic_memory_manager.EpisodicMemory")
async def test_close_session_not_in_use(
    mock_episodic_memory_cls,
    manager: EpisodicMemoryManager,
    mock_episodic_memory_conf,
    mock_episodic_memory_instance,
):
    """Test closing a session that is cached but not in use."""
    session_key = "session_to_close"
    mock_episodic_memory_cls.return_value = mock_episodic_memory_instance

    async with manager.create_episodic_memory(
        session_key,
        mock_episodic_memory_conf,
        "",
        {},
    ):
        pass  # Enters and exits context, ref_count becomes 0

    await manager.close_session(session_key)

    mock_episodic_memory_instance.close.assert_awaited_once()
    assert manager._instance_cache.get(session_key) is None


@pytest.mark.asyncio
@patch("memmachine.episodic_memory.episodic_memory_manager.EpisodicMemory")
async def test_close_session_in_use_raises_error(
    mock_episodic_memory_cls,
    manager: EpisodicMemoryManager,
    mock_episodic_memory_conf,
    mock_episodic_memory_instance,
):
    """Test closing a session in use raise a RuntimeError."""
    session_key = "session_to_close"
    mock_episodic_memory_cls.return_value = mock_episodic_memory_instance

    async with manager.create_episodic_memory(
        session_key,
        mock_episodic_memory_conf,
        "",
        {},
    ):
        with pytest.raises(RuntimeError, match=f"Session {session_key} is busy"):
            await manager.close_session(session_key)


@pytest.mark.asyncio
async def test_manager_close(manager: EpisodicMemoryManager, mock_episodic_memory_conf):
    """Test the main close method of the manager."""
    session_key1 = "s1"
    session_key2 = "s2"
    mock_instance1 = AsyncMock(spec=EpisodicMemory, name="instance1")
    mock_instance2 = AsyncMock(spec=EpisodicMemory, name="instance2")

    with patch(
        "memmachine.episodic_memory.episodic_memory_manager.EpisodicMemory",
        side_effect=[mock_instance1, mock_instance2],
    ):
        # Create two sessions and leave them in the cache
        async with manager.create_episodic_memory(
            session_key1,
            mock_episodic_memory_conf,
            "",
            {},
        ):
            pass
        async with manager.create_episodic_memory(
            session_key2,
            mock_episodic_memory_conf,
            "",
            {},
        ):
            pass

    await manager.close()

    # Verify instances were closed and removed from cache
    mock_instance1.close.assert_awaited_once()
    mock_instance2.close.assert_awaited_once()
    assert manager._instance_cache.get(session_key1) is None
    assert manager._instance_cache.get(session_key2) is None

    # Verify manager is in a closed state
    with pytest.raises(RuntimeError, match="Memory is closed"):
        async with manager.open_episodic_memory("any_session"):
            pass


@pytest.mark.asyncio
async def test_manager_methods_after_close_raise_error(manager: EpisodicMemoryManager):
    """Test that all public methods raise RuntimeError after the manager is closed."""
    await manager.close()

    with pytest.raises(RuntimeError, match="Memory is closed"):
        async with manager.create_episodic_memory("s", MagicMock(), "", {}):
            pass

    with pytest.raises(RuntimeError, match="Memory is closed"):
        async with manager.open_episodic_memory("s"):
            pass

    with pytest.raises(RuntimeError, match="Memory is closed"):
        await manager.delete_episodic_session("s")

    with pytest.raises(RuntimeError, match="Memory is closed"):
        await manager.close_session("s")
