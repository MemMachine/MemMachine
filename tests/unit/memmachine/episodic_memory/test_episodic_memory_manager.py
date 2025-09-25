"""Unit tests for the EpisodicMemoryManager class."""

from unittest.mock import AsyncMock, MagicMock, mock_open, patch

import pytest
import pytest_asyncio
import yaml

from memmachine.episodic_memory.data_types import MemoryContext
from memmachine.episodic_memory.episodic_memory_manager import (
    EpisodicMemoryManager,
)

# Mark all tests in this file as asyncio
pytestmark = pytest.mark.asyncio


@pytest.fixture
def mock_config():
    """Provides a default mock configuration dictionary."""
    return {
        "sessiondb": {"uri": "sqlite:///:memory:"},
        "logging": {"path": "/tmp/test.log", "level": "info"},
        "storage": {"uri": "sqlite:///:memory:"},
    }


@pytest_asyncio.fixture
async def manager(mock_config):
    """
    Fixture to create and clean up an EpisodicMemoryManager instance for each test.
    It mocks file I/O and logging to isolate the manager.
    """
    # We patch `open` to simulate reading the config file, and `logging` to
    # prevent it from affecting the test environment.
    with (
        patch("builtins.open", mock_open(read_data=yaml.dump(mock_config))),
        patch("logging.basicConfig"),
    ):
        # The public factory method to create an instance
        instance = EpisodicMemoryManager.create_episodic_memory_manager(
            "dummy_path.yaml"
        )
        yield instance
        # The reset method ensures a clean state for the next test by shutting
        # down the singleton instance.
        await EpisodicMemoryManager.reset()


async def test_create_episodic_memory_manager_singleton(manager, mock_config):
    """
    Test that create_episodic_memory_manager follows the singleton pattern.
    """

    # Calling the factory method again should return the exact same instance
    with patch("builtins.open", mock_open(read_data=yaml.dump(mock_config))):
        same_instance = EpisodicMemoryManager.create_episodic_memory_manager(
            "dummy_path.yaml"
        )
        assert same_instance is manager


async def test_configuration(manager, mock_config):
    """Test that configuration returns the loaded configuration."""
    # The config keys are lowercased during loading
    config = manager.configuration
    assert config["sessiondb"]["uri"] == mock_config["sessiondb"]["uri"]
    assert config["logging"]["path"] == mock_config["logging"]["path"]
    assert config["logging"]["level"] == mock_config["logging"]["level"]
    assert config["storage"]["uri"] == mock_config["storage"]["uri"]


@patch("memmachine.episodic_memory.episodic_memory_manager.EpisodicMemory")
async def test_get_episodic_memory_instance_new(MockEpisodicMemory, manager):
    """Test creating a new EpisodicMemory instance for a new context."""
    # Mock the async methods of EpisodicMemory
    mock_instance = MockEpisodicMemory.return_value
    mock_instance.reference = AsyncMock(return_value=True)
    mock_instance.close = AsyncMock()

    group_id = "group1"
    agent_id = ["agent1"]
    user_id = ["user1"]
    session_id = "session1"

    instance = await manager.get_episodic_memory_instance(
        group_id, agent_id, user_id, session_id
    )

    assert instance is not None
    # Check that a new EpisodicMemory object was created with the correct
    # context
    MockEpisodicMemory.assert_called_once()
    called_context = MockEpisodicMemory.call_args[0][2]
    assert isinstance(called_context, MemoryContext)
    assert called_context.group_id == group_id
    assert called_context.session_id == session_id

    # Check that the instance's reference count was incremented
    mock_instance.reference.assert_awaited_once()


@patch("memmachine.episodic_memory.episodic_memory_manager.EpisodicMemory")
async def test_get_episodic_memory_instance_existing(MockEpisodicMemory, manager):
    """Test retrieving an existing EpisodicMemory instance."""
    mock_instance = MockEpisodicMemory.return_value
    mock_instance.reference = AsyncMock(return_value=True)
    mock_instance.close = AsyncMock()

    # First call creates the instance
    instance1 = await manager.get_episodic_memory_instance("g1", ["a1"], ["u1"], "s1")
    assert instance1 is not None
    MockEpisodicMemory.assert_called_once()
    mock_instance.reference.assert_awaited_once()

    # Second call for the same context should return the same instance
    instance2 = await manager.get_episodic_memory_instance("g1", ["a1"], ["u1"], "s1")
    assert instance2 is instance1
    # No new EpisodicMemory should be created
    MockEpisodicMemory.assert_called_once()
    # Reference count should be incremented again
    assert mock_instance.reference.await_count == 2


async def test_create_group(manager):
    """Test create group for episodic memory"""
    # Create group with invalid parameters
    with pytest.raises(ValueError):
        await manager.create_group("g1", None, None)
    # Create group with correct parameter
    await manager.create_group("g1", ["u1"], None)
    # Create group with same ID
    with pytest.raises(ValueError):
        await manager.create_group("g1", ["u2"], ["A2"])
    group = manager.get_group_configuration("g1")
    assert group is not None
    assert group.group_id == "g1"


@patch("memmachine.episodic_memory.episodic_memory_manager.EpisodicMemory")
async def test_create_episodic_memory_instance(MockEpisodicMemory, manager):
    """Test create episodic memory"""
    mock_instance = MockEpisodicMemory.return_value
    mock_instance.reference = AsyncMock(return_value=True)
    mock_instance.close = AsyncMock()
    # Test create instance without group
    with pytest.raises(ValueError):
        await manager.create_episodic_memory_instance("g1", "s1")
    # Create a group first
    await manager.create_group("g1", ["a1"], ["u1"])
    # Opne a non-existing session
    with pytest.raises(ValueError):
        await manager.open_episodic_memory_instance("g1", "s1")
    # Create a memory instance
    inst = await manager.create_episodic_memory_instance("g1", "s1")
    assert inst is not None
    MockEpisodicMemory.assert_called_once()
    await inst.close()

    # Create a memory instance with the same group, session ID
    with pytest.raises(ValueError):
        await manager.create_episodic_memory_instance("g1", "s1")
    # Try to close it
    res = await manager.close_episodic_memory_instance("g1", "s1")
    assert res is True
    # Open the memory instance
    inst = await manager.open_episodic_memory_instance("g1", "s1")
    assert inst is not None
    await inst.close()
    # Create an instance with different session ID
    inst = await manager.create_episodic_memory_instance("g1", "s2")
    assert inst is not None
    await inst.close()


@patch("memmachine.episodic_memory.episodic_memory_manager.EpisodicMemory")
async def test_async_open_episodic_memory_instance(MockEpisodicMemory, manager):
    """Test retrieving an existing EpisodicMemory instance."""
    mock_instance = MockEpisodicMemory.return_value
    mock_instance.reference = AsyncMock(return_value=True)
    mock_instance.close = AsyncMock()
    # First create the group and session
    await manager.create_group("g1", ["a1"], ["u1"])
    # Create the seesion
    inst = await manager.create_episodic_memory_instance("g1", "s1")
    await inst.close()
    # First call creates the instance
    async with manager.async_open_episodic_memory_instance("g1", "s1") as instance1:
        assert instance1 is not None
        MockEpisodicMemory.assert_called_once()
        assert mock_instance.reference.await_count == 2

        # Second call for the same context should return the same instance
        instance2 = await manager.open_episodic_memory_instance("g1", "s1")
        assert instance2 is instance1
        # No new EpisodicMemory should be created
        MockEpisodicMemory.assert_called_once()
        # Reference count should be incremented again
        assert mock_instance.reference.await_count == 3
        await instance2.close()
        assert mock_instance.close.await_count == 2
    assert mock_instance.close.await_count == 3


@patch("memmachine.episodic_memory.episodic_memory_manager.EpisodicMemory")
async def test_async_create_episodic_memory_instance(MockEpisodicMemory, manager):
    """Test retrieving an existing EpisodicMemory instance."""
    mock_instance = MockEpisodicMemory.return_value
    mock_instance.reference = AsyncMock(return_value=True)
    mock_instance.close = AsyncMock()
    # First create the group and session
    await manager.create_group("g1", ["a1"], ["u1"])
    async with manager.async_create_episodic_memory_instance("g1", "s1") as instance:
        assert instance is not None
        MockEpisodicMemory.assert_called_once()
        assert mock_instance.reference.await_count == 1
        assert mock_instance.close.await_count == 0
    assert mock_instance.close.await_count == 1


async def test_get_episodic_memory_instance_invalid_context(manager):
    """Test that getting an instance with an invalid context raises an error."""
    with pytest.raises(ValueError):
        await manager.get_episodic_memory_instance(None, None)

    with pytest.raises(ValueError):
        await manager.get_episodic_memory_instance(group_id="g1", session_id=None)


@patch("memmachine.episodic_memory.episodic_memory_manager.EpisodicMemory")
async def test_close_episodic_memory_instance(MockEpisodicMemory, manager):
    """Test closing an active EpisodicMemory instance."""
    mock_instance = MockEpisodicMemory.return_value
    mock_instance.reference = AsyncMock(return_value=True)
    mock_instance.close = AsyncMock()

    # Create an instance
    await manager.get_episodic_memory_instance("g1", ["a1"], ["u1"], "s1")

    # Close it
    result = await manager.close_episodic_memory_instance("g1", "s1")

    assert result is True
    mock_instance.close.assert_awaited_once()


async def test_close_non_existent_instance(manager):
    """Test that closing a non-existent instance returns False."""
    result = await manager.close_episodic_memory_instance("non-existent", "s1")
    assert result is False


async def test_shut_down(manager):
    """Test that shut_down closes all active instances and cleans up."""

    with patch(
        "memmachine.episodic_memory.episodic_memory_manager.EpisodicMemory"
    ) as MockEpisodicMemory:
        mock_instance1 = MagicMock()
        mock_instance1.reference = AsyncMock(return_value=True)
        mock_instance1.close = AsyncMock()

        mock_instance2 = MagicMock()
        mock_instance2.reference = AsyncMock(return_value=True)
        mock_instance2.close = AsyncMock()

        # Create two separate instances
        MockEpisodicMemory.side_effect = [mock_instance1, mock_instance2]
        await manager.get_episodic_memory_instance("g1", ["a1"], ["u1"], "s1")
        await manager.get_episodic_memory_instance("g2", ["a2"], ["u2"], "s2")

        # Now, shut down the manager
        await manager.shut_down()

        # Verify that close was called on both instances
        mock_instance1.close.assert_called_once()
        mock_instance2.close.assert_called_once()


# --- Test Session Proxy Methods ---


async def test_get_all_sessions(manager):
    """Test the get_all_sessions proxy method."""
    # Create some sessions directly via the session manager for this test
    sm = manager.session_manager
    sm.create_session_if_not_exist("g1", ["a1"], ["u1"], "s1")
    sm.create_session_if_not_exist("g2", ["a2"], ["u2"], "s2")

    sessions = manager.get_all_sessions()
    assert len(sessions) == 2
    assert {s.session_id for s in sessions} == {"s1", "s2"}


async def test_get_user_sessions(manager):
    """Test the get_user_sessions proxy method."""
    sm = manager.session_manager
    sm.create_session_if_not_exist("g1", ["a1"], ["u1", "u2"], "s1")
    sm.create_session_if_not_exist("g2", ["a2"], ["u2"], "s2")

    # User in one session
    user1_sessions = manager.get_user_sessions("u1")
    assert len(user1_sessions) == 1
    assert user1_sessions[0].session_id == "s1"

    # User in two sessions
    user2_sessions = manager.get_user_sessions("u2")
    assert len(user2_sessions) == 2
    assert {s.session_id for s in user2_sessions} == {"s1", "s2"}

    # User in no sessions
    assert manager.get_user_sessions("u3") == []


async def test_get_agent_sessions(manager):
    """Test the get_agent_sessions proxy method."""
    sm = manager.session_manager
    sm.create_session_if_not_exist("g1", ["a1", "a2"], ["u1"], "s1")
    sm.create_session_if_not_exist("g2", ["a2"], ["u2"], "s2")

    agent2_sessions = manager.get_agent_sessions("a2")
    assert len(agent2_sessions) == 2
    assert {s.session_id for s in agent2_sessions} == {"s1", "s2"}


async def test_get_group_sessions(manager):
    """Test the get_group_sessions proxy method."""
    sm = manager.session_manager
    sm.create_session_if_not_exist("g1", ["a1"], ["u1"], "s1")
    sm.create_session_if_not_exist("g1", ["a2"], ["u2"], "s2")

    group1_sessions = manager.get_group_sessions("g1")
    assert len(group1_sessions) == 2
    assert {s.session_id for s in group1_sessions} == {"s1", "s2"}

    assert manager.get_group_sessions("g2") == []
