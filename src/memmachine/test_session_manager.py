from unittest.mock import MagicMock

import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine

from memmachine.common.metrics_factory import MetricsFactory
from memmachine.configuration.episodic_config import EpisodicMemoryParams
from memmachine.session_manager import SessionDataManagerImpl


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

        def __getstate__(self):
            return {}

        def __setstate__(self, state):
            pass


    factory = MockMetricsFactory()
    return factory

@pytest.fixture
def episodic_memory_params(mock_metrics_factory):
    """Fixture for a dummy EpisodicMemoryParams object."""
    return EpisodicMemoryParams(
        session_key="test_session",
        metrics_factory=mock_metrics_factory,
        enabled=False
    )


@pytest_asyncio.fixture
async def db_engine():
    """Fixture for an in-memory SQLite async engine."""
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    yield engine
    await engine.dispose()


@pytest_asyncio.fixture
async def session_manager(db_engine: AsyncEngine):
    """Fixture for SessionDataManagerImpl, with tables created."""
    manager = SessionDataManagerImpl(engine=db_engine)
    await manager.create_tables()
    yield manager
    await manager.close()


@pytest.mark.asyncio
async def test_create_tables(db_engine: AsyncEngine):
    """Test that create_tables creates the expected tables."""
    manager = SessionDataManagerImpl(engine=db_engine)
    await manager.create_tables()

@pytest.mark.asyncio
async def test_create_new_session(session_manager: SessionDataManagerImpl, episodic_memory_params: EpisodicMemoryParams):
    """Test creating a new session successfully."""
    session_key = "session1"
    config = {"key": "value"}
    description = "A test session"
    metadata = {"user": "tester"}

    await session_manager.create_new_session(session_key, config, episodic_memory_params, description, metadata)

    ret_config, ret_desc, ret_meta, ret_param = await session_manager.get_session_info(session_key)

    assert ret_config == config
    assert ret_desc == description
    assert ret_meta == metadata
    assert ret_param.session_key == episodic_memory_params.session_key


@pytest.mark.asyncio
async def test_create_existing_session_raises_error(session_manager: SessionDataManagerImpl, episodic_memory_params: EpisodicMemoryParams):
    """Test that creating a session that already exists raises a ValueError."""
    session_key = "session1"
    await session_manager.create_new_session(session_key, {}, episodic_memory_params, "", {})

    with pytest.raises(ValueError, match=f"Session {session_key} already exists"):
        await session_manager.create_new_session(session_key, {}, episodic_memory_params, "", {})


@pytest.mark.asyncio
async def test_delete_session(session_manager: SessionDataManagerImpl, episodic_memory_params: EpisodicMemoryParams):
    """Test deleting an existing session."""
    session_key = "session_to_delete"
    await session_manager.create_new_session(session_key, {}, episodic_memory_params, "", {})

    # Verify it exists
    await session_manager.get_session_info(session_key)

    # Delete it
    await session_manager.delete_session(session_key)

    # Verify it's gone
    with pytest.raises(ValueError, match=f"Session {session_key} does not exists"):
        await session_manager.get_session_info(session_key)


@pytest.mark.asyncio
async def test_delete_nonexistent_session_raises_error(session_manager: SessionDataManagerImpl):
    """Test that deleting a non-existent session raises a ValueError."""
    session_key = "nonexistent_session"
    with pytest.raises(ValueError, match=f"Session {session_key} does not exists"):
        await session_manager.delete_session(session_key)


@pytest.mark.asyncio
async def test_get_session_info_nonexistent_raises_error(session_manager: SessionDataManagerImpl):
    """Test that getting info for a non-existent session raises a ValueError."""
    session_key = "nonexistent_session"
    with pytest.raises(ValueError, match=f"Session {session_key} does not exists"):
        await session_manager.get_session_info(session_key)


@pytest.mark.asyncio
async def test_get_sessions(session_manager: SessionDataManagerImpl, episodic_memory_params: EpisodicMemoryParams):
    """Test retrieving session keys with and without filters."""
    # Create some sessions
    await session_manager.create_new_session("session1", {}, episodic_memory_params, "", {"tag": "A", "user": "1"})
    await session_manager.create_new_session("session2", {}, episodic_memory_params, "", {"tag": "B", "user": "1"})
    await session_manager.create_new_session("session3", {}, episodic_memory_params, "", {"tag": "A", "user": "2"})

    # Get all sessions
    all_sessions = await session_manager.get_sessions()
    assert sorted(all_sessions) == ["session1", "session2", "session3"]

    # Filter by tag 'A'
    sessions_A = await session_manager.get_sessions(filter={"tag": "A"})
    assert sorted(sessions_A) == ["session1", "session3"]

    # Filter by user '1'
    sessions_user1 = await session_manager.get_sessions(filter={"user": "1"})
    assert sorted(sessions_user1) == ["session1", "session2"]

    # Filter by tag 'B' and user '1'
    sessions_B_user1 = await session_manager.get_sessions(filter={"tag": "B", "user": "1"})
    assert sessions_B_user1 == ["session2"]

    # Filter with no matches
    no_match = await session_manager.get_sessions(filter={"tag": "C"})
    assert no_match == []


@pytest.mark.asyncio
async def test_get_sessions_empty(session_manager: SessionDataManagerImpl):
    """Test retrieving sessions when none exist."""
    sessions = await session_manager.get_sessions()
    assert sessions == []


@pytest.mark.asyncio
async def test_save_short_term_memory_new(session_manager: SessionDataManagerImpl, episodic_memory_params: EpisodicMemoryParams):
    """Test saving short-term memory for a session for the first time."""
    session_key = "stm_session_1"
    await session_manager.create_new_session(session_key, {}, episodic_memory_params, "", {})

    summary = "This is a summary."
    last_seq = 10
    episode_num = 5

    await session_manager.save_short_term_memory(session_key, summary, last_seq, episode_num)

    ret_summary, ret_ep_num, ret_last_seq = await session_manager.get_short_term_memory(session_key)

    assert ret_summary == summary
    assert ret_last_seq == last_seq
    assert ret_ep_num == episode_num


@pytest.mark.asyncio
async def test_save_short_term_memory_update(session_manager: SessionDataManagerImpl, episodic_memory_params: EpisodicMemoryParams):
    """Test updating existing short-term memory for a session."""
    session_key = "stm_session_2"
    await session_manager.create_new_session(session_key, {}, episodic_memory_params, "", {})

    # First save
    await session_manager.save_short_term_memory(session_key, "summary1", 1, 1)

    # Second save (update)
    summary = "This is an updated summary."
    last_seq = 20
    episode_num = 10
    await session_manager.save_short_term_memory(session_key, summary, last_seq, episode_num)

    ret_summary, ret_ep_num, ret_last_seq = await session_manager.get_short_term_memory(session_key)

    assert ret_summary == summary
    assert ret_last_seq == last_seq
    assert ret_ep_num == episode_num


@pytest.mark.asyncio
async def test_save_short_term_memory_for_nonexistent_session(session_manager: SessionDataManagerImpl):
    """Test that saving STM for a non-existent session raises a ValueError."""
    session_key = "nonexistent_session"
    with pytest.raises(ValueError, match=f"Session {session_key} does not exists"):
        await session_manager.save_short_term_memory(session_key, "summary", 1, 1)


@pytest.mark.asyncio
async def test_get_short_term_memory_nonexistent(session_manager: SessionDataManagerImpl, episodic_memory_params: EpisodicMemoryParams):
    """Test that getting STM for which none has been saved raises a ValueError."""
    session_key = "session_no_stm"
    await session_manager.create_new_session(session_key, {}, episodic_memory_params, "", {})

    with pytest.raises(ValueError, match=f"session {session_key} does not have short term memory"):
        await session_manager.get_short_term_memory(session_key)


@pytest.mark.asyncio
async def test_delete_session_cascades_to_short_term_memory(session_manager: SessionDataManagerImpl, episodic_memory_params: EpisodicMemoryParams):
    """Test that deleting a session also deletes its associated short-term memory data."""
    session_key = "cascade_delete_session"
    await session_manager.create_new_session(session_key, {}, episodic_memory_params, "", {})
    await session_manager.save_short_term_memory(session_key, "summary", 1, 1)

    # Verify STM exists
    await session_manager.get_short_term_memory(session_key)

    # Delete the parent session
    await session_manager.delete_session(session_key)

    # Verify STM is also gone
    with pytest.raises(ValueError, match=f"session {session_key} does not have short term memory"):
        await session_manager.get_short_term_memory(session_key)
