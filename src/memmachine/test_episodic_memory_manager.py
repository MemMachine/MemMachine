import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine
from unittest.mock import AsyncMock, MagicMock, patch

from memmachine.configuration.episodic_config import (
    EpisodicMemoryManagerParam,
    EpisodicMemoryParams,
)
from memmachine.common.metrics_factory import MetricsFactory
from memmachine.episodic_memory.episodic_memory import EpisodicMemory
from memmachine.episodic_memory_manager import EpisodicMemoryManager
from memmachine.session_manager import SessionDataManagerImpl

 
@pytest_asyncio.fixture
async def db_engine():
    """Fixture for an in-memory SQLite async engine."""
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    yield engine
    await engine.dispose()

@pytest_asyncio.fixture
async def mock_session_storage(db_engine):
    """Fixture for a mocked SessionDataManager."""
    storage = SessionDataManagerImpl(engine=db_engine)
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
def mock_episodic_memory_params(mock_metrics_factory):
    """Fixture for a dummy EpisodicMemoryParams object."""
    return EpisodicMemoryParams(
        session_key="test_session",
        metrics_factory=mock_metrics_factory,
        enabled=False
    )


@pytest.fixture
def mock_episodic_memory_manager_param(mock_session_storage):
    """Fixture for EpisodicMemoryManagerParam."""
    return EpisodicMemoryManagerParam(
        session_storage=mock_session_storage,
        instance_cache_size=10,
        max_life_time=3600,
    )


@pytest_asyncio.fixture
async def manager(mock_episodic_memory_manager_param):
    """Fixture for an EpisodicMemoryManager instance."""
    return EpisodicMemoryManager(param=mock_episodic_memory_manager_param)


@pytest.mark.asyncio
@patch("memmachine.episodic_memory_manager.EpisodicMemory.create", new_callable=AsyncMock)
async def test_create_episodic_memory_success(
    mock_create, manager: EpisodicMemoryManager, mock_session_storage, mock_episodic_memory_params
):
    """Test successfully creating a new episodic memory instance."""
    session_key = "new_session"
    description = "A new test session"
    metadata = {"owner": "tester"}
    mock_instance = AsyncMock(spec=EpisodicMemory)
    mock_create.return_value = mock_instance

    async with manager.create_episodic_memory(
        session_key, mock_episodic_memory_params, description, metadata
    ) as instance:
        assert instance is mock_instance
        mock_create.assert_awaited_once_with(mock_episodic_memory_params)
        assert manager._instance_cache.get_ref_count(session_key) == 1  # 1 from add

    assert manager._instance_cache.get_ref_count(session_key) == 0 # put is called


@pytest.mark.asyncio
async def test_create_episodic_memory_already_exists(
    manager: EpisodicMemoryManager, mock_session_storage, mock_episodic_memory_params
):
    """Test that creating a session that already exists raises an error."""
    session_key = "existing_session"
    async with manager.create_episodic_memory(
        session_key, mock_episodic_memory_params, "", {}
    ):

        with pytest.raises(ValueError, match=f"Session {session_key} already exists"):
            async with manager.create_episodic_memory(
                session_key, mock_episodic_memory_params, "", {}
            ):
                pass  # This part should not be reached


@pytest.mark.asyncio
@patch("memmachine.episodic_memory_manager.EpisodicMemory.create", new_callable=AsyncMock)
async def test_open_episodic_memory_new_instance(
    mock_create, manager: EpisodicMemoryManager, mock_session_storage, mock_episodic_memory_params
):
    """Test opening a session for the first time, loading it from storage."""
    session_key = "session_to_open"
    mock_instance = AsyncMock(spec=EpisodicMemory)
    mock_create.return_value = mock_instance
    async with manager.create_episodic_memory(
        session_key, mock_episodic_memory_params, "", {}
    ) as instance:
        assert instance is mock_instance
    await manager.close_session(session_key)

    async with manager.open_episodic_memory(session_key) as instance:
        assert instance is mock_instance
        mock_create.assert_awaited_once_with(mock_episodic_memory_params)
        assert manager._instance_cache.get_ref_count(session_key) == 1

    assert manager._instance_cache.get_ref_count(session_key) == 0


@pytest.mark.asyncio
@patch("memmachine.episodic_memory_manager.EpisodicMemory.create", new_callable=AsyncMock)
async def test_open_episodic_memory_cached_instance(
    mock_create, manager: EpisodicMemoryManager, mock_session_storage, mock_episodic_memory_params
):
    """Test opening a session that is already in the cache."""
    session_key = "cached_session"
    mock_instance = AsyncMock(spec=EpisodicMemory)
    mock_create.return_value = mock_instance

    # Pre-populate the cache
    async with manager.create_episodic_memory(session_key, mock_episodic_memory_params, "", {}):
        pass

    mock_create.assert_awaited_once()
    mock_create.reset_mock()

    # Open it again
    async with manager.open_episodic_memory(session_key) as instance:
        assert instance is mock_instance
        # Should not call storage or create again
        mock_create.assert_not_awaited()
        assert manager._instance_cache.get_ref_count(session_key) == 1

    assert manager._instance_cache.get_ref_count(session_key) == 0


@pytest.mark.asyncio
async def test_delete_episodic_session_not_in_use(
    manager: EpisodicMemoryManager, mock_session_storage, mock_episodic_memory_params
):
    """Test deleting a session that is not currently in use."""
    session_key = "session_to_delete"
    mock_instance = AsyncMock(spec=EpisodicMemory)

    with patch("memmachine.episodic_memory_manager.EpisodicMemory.create", return_value=mock_instance):
        # Create and release the session so it's in cache but not in use
        async with manager.create_episodic_memory(session_key, mock_episodic_memory_params, "", {}):
            pass

    assert manager._instance_cache.get_ref_count(session_key) == 0

    await manager.delete_episodic_session(session_key)

    # Verify it's gone from cache and storage
    assert manager._instance_cache.get(session_key) is None
    mock_instance.delete_data.assert_awaited_once()
    mock_instance.close.assert_awaited_once()

@pytest.mark.asyncio
async def test_delete_episodic_session_in_use_raises_error(
    manager: EpisodicMemoryManager, mock_episodic_memory_params
):
    """Test that deleting a session currently in use raises a RuntimeError."""
    session_key = "session_in_use"
    mock_instance = AsyncMock(spec=EpisodicMemory)

    with patch("memmachine.episodic_memory_manager.EpisodicMemory.create", return_value=mock_instance):
        async with manager.create_episodic_memory(session_key, mock_episodic_memory_params, "", {}):
            with pytest.raises(RuntimeError, match=f"Session {session_key} is still in use"):
                await manager.delete_episodic_session(session_key)


@pytest.mark.asyncio
@patch("memmachine.episodic_memory_manager.EpisodicMemory.create", new_callable=AsyncMock)
async def test_delete_episodic_session_not_in_cache(
    mock_create, manager: EpisodicMemoryManager, mock_session_storage, mock_episodic_memory_params
):
    """Test deleting a session that exists in storage but not in the cache."""
    session_key = "not_in_cache_session"
    mock_instance = AsyncMock(spec=EpisodicMemory)
    mock_create.return_value = mock_instance
    async with manager.create_episodic_memory(session_key, mock_episodic_memory_params, "", {}):
        pass
    await manager.close_session(session_key)
    mock_create.assert_awaited_once_with(mock_episodic_memory_params)
    mock_instance.close.assert_awaited_once()
    mock_create.reset_mock()
    mock_instance.reset_mock()

    await manager.delete_episodic_session(session_key)

    # Should load from storage to delete
    mock_instance.delete_data.assert_awaited_once()
    mock_instance.close.assert_awaited_once()


@pytest.mark.asyncio
async def test_close_session_not_in_use(manager: EpisodicMemoryManager, mock_episodic_memory_params):
    """Test closing a session that is cached but not in use."""
    session_key = "session_to_close"
    mock_instance = AsyncMock(spec=EpisodicMemory)

    with patch("memmachine.episodic_memory_manager.EpisodicMemory.create", return_value=mock_instance):
        async with manager.create_episodic_memory(session_key, mock_episodic_memory_params, "", {}):
            pass # Enters and exits context, ref_count becomes 1

    await manager.close_session(session_key)

    mock_instance.close.assert_awaited_once()
    assert manager._instance_cache.get(session_key) is None


@pytest.mark.asyncio
async def test_close_session_in_use_raises_error(manager: EpisodicMemoryManager, mock_episodic_memory_params):
    """Test that closing a session in use raises a RuntimeError."""
    session_key = "busy_session"
    with patch("memmachine.episodic_memory_manager.EpisodicMemory.create", new_callable=AsyncMock):
        async with manager.create_episodic_memory(session_key, mock_episodic_memory_params, "", {}):
            with pytest.raises(RuntimeError, match=f"Session {session_key} is busy"):
                await manager.close_session(session_key)


@pytest.mark.asyncio
async def test_manager_close(manager: EpisodicMemoryManager, mock_session_storage, mock_episodic_memory_params):
    """Test the main close method of the manager."""
    session_key1 = "s1"
    session_key2 = "s2"
    mock_instance1 = AsyncMock(spec=EpisodicMemory)
    mock_instance2 = AsyncMock(spec=EpisodicMemory)

    with patch("memmachine.episodic_memory_manager.EpisodicMemory.create", side_effect=[mock_instance1, mock_instance2]):
        # Create two sessions and leave them in the cache
        async with manager.create_episodic_memory(session_key1, mock_episodic_memory_params, "", {}):
            pass
        async with manager.create_episodic_memory(session_key2, mock_episodic_memory_params, "", {}):
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