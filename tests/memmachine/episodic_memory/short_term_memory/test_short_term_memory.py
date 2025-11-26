import uuid
from datetime import UTC, datetime
from typing import Any, TypeVar

import pytest
import pytest_asyncio

from memmachine.common.configuration.episodic_config import (
    EpisodicMemoryConf,
)
from memmachine.common.episode_store import (
    ContentType,
    Episode,
)
from memmachine.common.filter.filter_parser import parse_filter
from memmachine.common.language_model import LanguageModel
from memmachine.common.session_manager.session_data_manager import SessionDataManager
from memmachine.episodic_memory.short_term_memory.short_term_memory import (
    ShortTermMemory,
    ShortTermMemoryParams,
)


def create_test_episode(**kwargs):
    """Helper function to create a valid Episode for testing."""
    defaults = {
        "uid": str(uuid.uuid4()),
        "sequence_num": 1,
        "session_key": "session1",
        "episode_type": "message",
        "content_type": ContentType.STRING,
        "content": "default content",
        "created_at": datetime.now(tz=UTC),
        "producer_id": "user1",
        "producer_role": "user",
        "produced_for_id": None,
        "user_metadata": None,
    }
    defaults.update(kwargs)
    return Episode(**defaults)


class MockShortTermMemoryDataManager(SessionDataManager):
    """Mock implementation of SessionDataManager for testing."""

    def __init__(self):
        self.data = {}
        self.tables_created = False

    async def create_tables(self):
        self.tables_created = True

    async def drop_tables(self):
        pass

    async def save_short_term_memory(
        self,
        session_key: str,
        summary: str,
        seq: int,
        num: int,
    ):
        self.data[session_key] = (summary, seq, num)

    async def get_short_term_memory(self, session_key: str) -> tuple[str, int, int]:
        if session_key not in self.data:
            raise ValueError(f"No data for session key {session_key}")
        return self.data[session_key]

    async def close(self):
        self.data = {}
        self.tables_created = False

    async def create_new_session(
        self,
        session_key: str,
        configuration: dict,
        param: EpisodicMemoryConf,
        description: str,
        metadata: dict,
    ):
        pass

    async def delete_session(self, session_key: str):
        pass

    async def get_session_info(
        self,
        session_key: str,
    ) -> SessionDataManager.SessionInfo:
        return SessionDataManager.SessionInfo(
            description="",
            configuration={},
            user_metadata={},
            episode_memory_conf=EpisodicMemoryConf(
                metrics_factory_id="prometheus", session_key=session_key
            ),
        )

    async def get_sessions(self, filters: dict[str, object] | None = None) -> list[str]:
        return []


T = TypeVar("T")


class MockLanguageModel(LanguageModel):
    """Mock implementation of LanguageModel for testing."""

    async def generate_response(
        self,
        system_prompt: str | None = None,
        user_prompt: str | None = None,
        tools: list | None = None,
        tool_choice: str | dict[str, str] | None = None,
        max_attempts: int = 1,
    ) -> tuple[str, Any]:
        return "summary", ""

    async def generate_parsed_response(
        self,
        output_format: type[T],
        system_prompt: str | None = None,
        user_prompt: str | None = None,
        max_attempts: int = 1,
    ) -> T:
        return "summary"


@pytest.fixture
def mock_model():
    """Fixture for a mocked language model."""
    model = MockLanguageModel()
    return model


@pytest.fixture
def mock_data_manager():
    """Fixture for a mocked ShortTermMemoryDataManager."""
    return MockShortTermMemoryDataManager()


@pytest.fixture
def short_term_memory_param(mock_model, mock_data_manager):
    """Fixture for short_term_memory_params."""
    return ShortTermMemoryParams(
        session_key="session1",
        llm_model=mock_model,
        data_manager=mock_data_manager,
        summary_prompt_system="System prompt",
        summary_prompt_user="User prompt: {episodes} {summary} {max_length}",
        message_capacity=16,
    )


@pytest_asyncio.fixture
async def memory(short_term_memory_param):
    """Fixture for a SessionMemory instance."""
    return await ShortTermMemory.create(short_term_memory_param)


@pytest.mark.asyncio
class TestSessionMemoryPublicAPI:
    """Test suite for the public API of SessionMemory."""

    async def test_initial_state(self, memory):
        """Test that the SessionMemory instance is initialized correctly."""
        episodes, summary = await memory.get_short_term_memory_context(query="test")
        assert episodes == []
        assert summary == ""

    async def test_add_episode(self, memory):
        """Test adding an episode to the session memory."""
        episode1 = create_test_episode(content="Hello")
        await memory.add_episodes([episode1])

        episodes, summary = await memory.get_short_term_memory_context(query="test")
        # session memory is not full
        assert episodes == [episode1]
        assert summary == ""

        episode2 = create_test_episode(content="World")
        await memory.add_episodes([episode2])

        episodes, summary = await memory.get_short_term_memory_context(query="test")
        assert episodes == [episode1, episode2]
        assert summary == ""

        # session memory is full
        episode3 = create_test_episode(content="!" * 7)
        await memory.add_episodes([episode3])

        episodes, summary = await memory.get_short_term_memory_context(query="test")
        assert episodes == [episode1, episode2, episode3]
        assert summary == "summary"

        # New episode push out the oldest one: episode1
        episode4 = create_test_episode(content="??")
        await memory.add_episodes([episode4])
        episodes, summary = await memory.get_short_term_memory_context(query="test")
        assert episodes == [episode3, episode4]
        assert summary == "summary"

    async def test_clear_memory(self, memory):
        """Test clearing the memory."""
        await memory.add_episodes([create_test_episode(content="test")])

        await memory.clear_memory()

        episodes, summary = await memory.get_short_term_memory_context(query="test")
        assert episodes == []
        assert summary == ""

    async def test_delete_episode(self, memory):
        """Test deleting an episode from the memory."""
        ep1 = create_test_episode(content="a")
        ep2 = create_test_episode(content="b")
        ep3 = create_test_episode(content="c")
        await memory.add_episodes([ep1, ep2, ep3])

        await memory.delete_episode(ep2.uid)
        episodes, _ = await memory.get_short_term_memory_context(query="test")
        assert episodes == [ep1, ep3]

    async def test_close(self, memory):
        """Test closing the memory."""
        await memory.add_episodes([create_test_episode(content="test")])
        await memory.close()
        with pytest.raises(RuntimeError):
            await memory.add_episodes([create_test_episode(content="test")])
        with pytest.raises(RuntimeError):
            _, _ = await memory.get_short_term_memory_context(query="test")

    async def test_get_short_term_memory_context(self, memory):
        """Test retrieving session memory context."""
        ep1 = create_test_episode(content="a" * 6)
        ep2 = create_test_episode(content="b" * 6)
        ep3 = create_test_episode(content="c" * 6)
        await memory.add_episodes([ep1, ep2, ep3])

        # Test with message length limit that fits all
        episodes, summary = await memory.get_short_term_memory_context(
            query="test",
            max_message_length=100,
        )
        assert len(episodes) == 3
        assert episodes == [ep1, ep2, ep3]
        assert summary == "summary"

        # Test with a tighter message length limit. Episodes are retrieved newest first.
        # length=7 (summary)
        # add ep1 (length 6), length=13.
        # add ep2 (length 6), length=19. Now length >= 19, so loop breaks.
        # Should return [ep1, ep2]
        episodes, summary = await memory.get_short_term_memory_context(
            query="test",
            max_message_length=19,
        )
        assert len(episodes) == 2
        assert episodes == [ep2, ep3]

        # Test with episode limit
        episodes, summary = await memory.get_short_term_memory_context(
            query="test",
            limit=1,
        )
        assert len(episodes) == 1
        assert episodes == [ep3]

    async def test_get_short_term_memory_context_with_filters(self, memory):
        """Test retrieving session memory context."""
        ep1 = create_test_episode(
            content="a" * 6,
            producer_id="user1",
            producer_role="user",
            metadata={"type": "message"},
        )
        ep2 = create_test_episode(
            content="b" * 6,
            producer_id="user2",
            producer_role="assistant",
            metadata={"type": "message", "category": "greeting"},
        )
        await memory.add_episodes([ep1, ep2])

        filter_str = "producer_id = 'user1'"
        filters = parse_filter(filter_str)
        # Test with filter that matches one episode
        episodes, _ = await memory.get_short_term_memory_context(
            "test", filters=filters
        )
        assert len(episodes) == 1
        assert episodes == [ep1]

        # Test with filter that matches no episodes
        filter_str = "producer_id = 'nonexistent'"
        filters = parse_filter(filter_str)
        episodes, _ = await memory.get_short_term_memory_context(
            "test", filters=filters
        )
        assert len(episodes) == 0
        assert episodes == []

        # Test with filter that matches both episodes
        filter_str = "m.type = 'message'"
        filters = parse_filter(filter_str)
        episodes, _ = await memory.get_short_term_memory_context(
            "test", filters=filters
        )
        assert len(episodes) == 2
        assert episodes == [ep1, ep2]

        # Test with filter that matches one episode based on filterable metadata
        filter_str = "m.category = 'greeting'"
        filters = parse_filter(filter_str)
        episodes, _ = await memory.get_short_term_memory_context(
            "test", filters=filters
        )
        assert len(episodes) == 1
        assert episodes == [ep2]

        # Test with filter that matches one episodes based on filterable metadata with "metadata." as prefix
        filter_str = "metadata.category = 'greeting'"
        filters = parse_filter(filter_str)
        episodes, _ = await memory.get_short_term_memory_context(
            "test", filters=filters
        )
        assert len(episodes) == 1
        assert episodes == [ep2]

        # Test with complex filter
        filter_str = "producer_role = 'assistant' AND m.type = 'message'"
        filters = parse_filter(filter_str)
        episodes, _ = await memory.get_short_term_memory_context(
            "test", filters=filters
        )
        assert len(episodes) == 1
        assert episodes == [ep2]

        filter_str = "producer_id = 'user1' OR m.category = 'greeting'"
        filters = parse_filter(filter_str)
        episodes, _ = await memory.get_short_term_memory_context(
            "test", filters=filters
        )
        assert len(episodes) == 2
        assert episodes == [ep1, ep2]
