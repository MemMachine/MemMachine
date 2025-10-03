import uuid
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock

import pytest

from memmachine.episodic_memory.data_types import (
    ContentType,
    Episode,
    MemoryContext,
)
from memmachine.episodic_memory.short_term_memory.session_memory import (
    SessionMemory,
)


def create_test_episode(**kwargs):
    """Helper function to create a valid Episode for testing."""
    defaults = {
        "uuid": uuid.uuid4(),
        "episode_type": "message",
        "content_type": ContentType.STRING,
        "content": "default content",
        "timestamp": datetime.now(),
        "group_id": "group1",
        "session_id": "session1",
        "producer_id": "user1",
    }
    defaults.update(kwargs)
    return Episode(**defaults)


@pytest.fixture
def mock_model():
    """Fixture for a mocked language model."""
    model = MagicMock()
    model.generate_response = AsyncMock(return_value=["summary"])
    return model


@pytest.fixture
def memory_context():
    """Fixture for a sample MemoryContext."""
    return MemoryContext(
        group_id="group1",
        agent_id={"agent1"},
        user_id={"user1"},
        session_id="session1",
    )


@pytest.fixture
def memory(mock_model, memory_context):
    """Fixture for a SessionMemory instance."""
    return SessionMemory(
        model=mock_model,
        summary_system_prompt="System prompt",
        summary_user_prompt="User prompt: {episodes} {summary}",
        capacity=3,
        max_message_len=100,
        max_token_num=50,
        memory_context=memory_context,
    )


@pytest.mark.asyncio
class TestSessionMemoryPublicAPI:
    """Test suite for the public API of SessionMemory."""

    async def test_initial_state(self, memory):
        """Test that the SessionMemory instance is initialized correctly."""
        episodes, summary = await memory.get_session_memory_context(query="test")
        assert episodes == []
        assert summary == ""

    async def test_add_episode(self, memory):
        """Test adding an episode to the session memory."""
        episode1 = create_test_episode(content="Hello")
        await memory.add_episode(episode1)

        episodes, summary = await memory.get_session_memory_context(query="test")
        # session memory is not full
        assert episodes == [episode1]
        assert summary == ""

        episode2 = create_test_episode(content="World")
        await memory.add_episode(episode2)

        episodes, summary = await memory.get_session_memory_context(query="test")
        assert episodes == [episode1, episode2]
        assert summary == ""

        # session memory is full
        episode3 = create_test_episode(content="!")
        await memory.add_episode(episode3)
        episodes, summary = await memory.get_session_memory_context(query="test")
        assert episodes == [episode1, episode2, episode3]
        assert summary == "summary"

        # New episode push out the oldest one: episode1
        episode4 = create_test_episode(content="?")
        await memory.add_episode(episode4)
        episodes, summary = await memory.get_session_memory_context(query="test")
        assert episodes == [episode2, episode3, episode4]
        assert summary == "summary"

    async def test_clear_memory(self, memory):
        """Test clearing the memory."""
        await memory.add_episode(create_test_episode(content="test"))

        await memory.clear_memory()

        episodes, summary = await memory.get_session_memory_context(query="test")
        assert episodes == []
        assert summary == ""

    async def test_close(self, memory):
        """Test closing the memory."""
        await memory.add_episode(create_test_episode(content="test"))
        await memory.close()
        episodes, summary = await memory.get_session_memory_context(query="test")
        assert episodes == []
        assert summary == ""

    async def test_get_session_memory_context(self, memory):
        """Test retrieving session memory context."""
        ep1 = create_test_episode(content="a" * 20)  # 5 tokens
        ep2 = create_test_episode(content="b" * 20)  # 5 tokens
        ep3 = create_test_episode(content="c" * 20)  # 5 tokens
        await memory.add_episode(ep1)
        await memory.add_episode(ep2)
        await memory.add_episode(ep3)

        # Test with token limit that fits all
        # summary (5) + ep1 (5) + ep2 (5) + ep3 (5) = 20 tokens
        episodes, summary = await memory.get_session_memory_context(
            query="test", max_token_num=21
        )
        assert len(episodes) == 3
        assert episodes == [ep1, ep2, ep3]
        assert summary == "summary"

        # Test with a tighter token limit. Episodes are retrieved newest first.
        # length=5 (summary)
        # add ep1 (5 tokens), length=10.
        # add ep2 (5 tokens), length=15. Now length >= 14, so loop breaks.
        # Should return [ep1, ep2]
        episodes, summary = await memory.get_session_memory_context(
            query="test", max_token_num=14
        )
        assert len(episodes) == 2
        assert episodes == [ep2, ep3]

        # Test with episode limit
        episodes, summary = await memory.get_session_memory_context(
            query="test", limit=1
        )
        assert len(episodes) == 1
        assert episodes == [ep3]
