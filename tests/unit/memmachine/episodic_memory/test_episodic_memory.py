"""Unit tests for the EpisodicMemory class."""

import uuid
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from memmachine.episodic_memory.data_types import ContentType, Episode, MemoryContext
from memmachine.episodic_memory.episodic_memory import (
    AsyncEpisodicMemory,
    EpisodicMemory,
)

# Since we are not using a test class, we'll use pytest's features.
# The 'asyncio' marker is used to run tests with an asyncio event loop.
pytestmark = pytest.mark.asyncio


@pytest.fixture
def memory_context():
    """Provides a sample MemoryContext for tests."""
    return MemoryContext(
        group_id="test_group",
        agent_id={"test_agent"},
        user_id={"test_user"},
        session_id="test_session",
    )


@pytest.fixture
def mock_manager():
    """Provides a mock EpisodicMemoryManager."""
    manager = MagicMock()
    manager.delete_context_memory = AsyncMock()
    return manager


@pytest.fixture
def mock_config():
    """Provides a mock configuration dictionary."""
    return {
        "model": {
            "test_model": {
                "model_vendor": "mock_vendor",
            }
        },
        "sessionmemory": {"model_name": "test_model"},
        "prompts": {},
        "long_term_memory": {},
    }


@pytest.fixture
def mock_config_without_longterm_memory():
    """Provides a mock configuration dictionary."""
    return {
        "model": {
            "test_model": {
                "model_vendor": "mock_vendor",
            }
        },
        "sessionmemory": {"model_name": "test_model"},
        "prompts": {},
    }


@pytest.fixture
def mock_config_without_session_memory():
    """Provides a mock configuration dictionary."""
    return {
        "model": {
            "test_model": {
                "model_vendor": "mock_vendor",
            }
        },
        "prompts": {},
        "long_term_memory": {"type": "long"},
    }


@pytest.fixture
def episodic_memory_instance(mock_manager, mock_config, memory_context):
    """Provides an EpisodicMemory instance with mocked dependencies."""
    with (
        patch(
            "memmachine.episodic_memory.episodic_memory.LanguageModelBuilder"
        ) as MockLMB,
        patch(
            "memmachine.episodic_memory.episodic_memory.MetricsFactoryBuilder"
        ) as MockMFB,
        patch(
            "memmachine.episodic_memory.episodic_memory.SessionMemory"
        ) as MockSessionMemory,
        patch(
            "memmachine.episodic_memory.episodic_memory.LongTermMemory"
        ) as MockLongTermMemory,
    ):
        # Mock the builders and their build methods
        mock_model = MagicMock()
        MockLMB.build.return_value = mock_model

        mock_metrics_manager = MagicMock()
        mock_metrics_manager.get_summary.return_value = MagicMock()
        mock_metrics_manager.get_counter.return_value = MagicMock()
        MockMFB.build.return_value = mock_metrics_manager

        # Mock the memory stores
        mock_session_memory_instance = MagicMock()
        mock_session_memory_instance.add_episode = AsyncMock()
        mock_session_memory_instance.clear_memory = AsyncMock()
        mock_session_memory_instance.close = AsyncMock()
        mock_session_memory_instance.get_session_memory_context = AsyncMock()
        MockSessionMemory.return_value = mock_session_memory_instance

        mock_ltm_instance = MagicMock()
        mock_ltm_instance.add_episode = AsyncMock()
        mock_ltm_instance.forget_session = AsyncMock()
        mock_ltm_instance.close = AsyncMock()
        mock_ltm_instance.search = AsyncMock()
        MockLongTermMemory.return_value = mock_ltm_instance

        instance = EpisodicMemory(mock_manager, mock_config, memory_context)
        # Attach mocks for easy access in tests
        instance.short_term_memory = mock_session_memory_instance
        instance.long_term_memory = mock_ltm_instance
        yield instance


@pytest.fixture
def episodic_memory_instance_without_sessionmemory(
    mock_manager, mock_config_without_session_memory, memory_context
):
    """Provides an EpisodicMemory instance with mocked dependencies."""
    with (
        patch(
            "memmachine.episodic_memory.episodic_memory.LanguageModelBuilder"
        ) as MockLMB,
        patch(
            "memmachine.episodic_memory.episodic_memory.MetricsFactoryBuilder"
        ) as MockMFB,
        patch(
            "memmachine.episodic_memory.episodic_memory.LongTermMemory"
        ) as MockLongTermMemory,
    ):
        # Mock the builders and their build methods
        mock_model = MagicMock()
        MockLMB.build.return_value = mock_model

        mock_metrics_manager = MagicMock()
        mock_metrics_manager.get_summary.return_value = MagicMock()
        mock_metrics_manager.get_counter.return_value = MagicMock()
        MockMFB.build.return_value = mock_metrics_manager

        mock_ltm_instance = MagicMock()
        mock_ltm_instance.add_episode = AsyncMock()
        mock_ltm_instance.forget_session = AsyncMock()
        mock_ltm_instance.close = AsyncMock()
        mock_ltm_instance.search = AsyncMock()
        MockLongTermMemory.return_value = mock_ltm_instance

        instance = EpisodicMemory(
            mock_manager, mock_config_without_session_memory, memory_context
        )
        # Attach mocks for easy access in tests
        instance.long_term_memory = mock_ltm_instance
        yield instance


@pytest.fixture
def episodic_memory_instance_without_longterm(
    mock_manager, mock_config_without_longterm_memory, memory_context
):
    """Provides an EpisodicMemory instance with mocked dependencies."""
    with (
        patch(
            "memmachine.episodic_memory.episodic_memory.LanguageModelBuilder"
        ) as MockLMB,
        patch(
            "memmachine.episodic_memory.episodic_memory.MetricsFactoryBuilder"
        ) as MockMFB,
        patch(
            "memmachine.episodic_memory.episodic_memory.SessionMemory"
        ) as MockSessionMemory,
    ):
        # Mock the builders and their build methods
        mock_model = MagicMock()
        MockLMB.build.return_value = mock_model

        mock_metrics_manager = MagicMock()
        mock_metrics_manager.get_summary.return_value = MagicMock()
        mock_metrics_manager.get_counter.return_value = MagicMock()
        MockMFB.build.return_value = mock_metrics_manager

        # Mock the memory stores
        mock_session_memory_instance = MagicMock()
        mock_session_memory_instance.add_episode = AsyncMock()
        mock_session_memory_instance.clear_memory = AsyncMock()
        mock_session_memory_instance.close = AsyncMock()
        mock_session_memory_instance.get_session_memory_context = AsyncMock()
        MockSessionMemory.return_value = mock_session_memory_instance

        instance = EpisodicMemory(
            mock_manager, mock_config_without_longterm_memory, memory_context
        )
        # Attach mocks for easy access in tests
        instance.short_term_memory = mock_session_memory_instance
        yield instance


async def test_episodic_memory_initialization(episodic_memory_instance, memory_context):
    """Tests if the EpisodicMemory instance is initialized correctly."""
    assert episodic_memory_instance.get_memory_context() == memory_context


async def test_initialization_fails_with_invalid_config(mock_manager, memory_context):
    """Tests that initialization raises ValueError for bad configuration."""
    with pytest.raises(ValueError, match="No memory is configured"):
        EpisodicMemory(mock_manager, {"sessionmemory": {}}, memory_context)

    with pytest.raises(ValueError, match="Invalid model configuration"):
        EpisodicMemory(
            mock_manager,
            {"sessionmemory": {"model_name": "bad_model"}},
            memory_context,
        )


async def test_reference_and_close(episodic_memory_instance, mock_manager):
    """Tests the reference counting and closing mechanism."""
    # Initial ref count is 1
    assert await episodic_memory_instance.reference() is True
    assert episodic_memory_instance.get_reference_count() == 2

    await episodic_memory_instance.close()

    assert episodic_memory_instance.get_reference_count() == 1
    # This call will bring ref_count to 0 and trigger close logic
    await episodic_memory_instance.close()

    mock_manager.delete_context_memory.assert_awaited_once_with(
        episodic_memory_instance.get_memory_context()
    )

    # Cannot reference a closed instance
    assert await episodic_memory_instance.reference() is False


async def test_add_memory_episode_success(episodic_memory_instance):
    """Tests adding a valid memory episode."""
    result = await episodic_memory_instance.add_memory_episode(
        producer="test_user",
        produced_for="test_agent",
        episode_content="Hello world",
        episode_type="message",
        content_type=ContentType.STRING,
    )

    assert result is True


async def test_add_memory_episode_invalid_producer(episodic_memory_instance, caplog):
    """Tests that adding an episode with an invalid producer fails."""
    with pytest.raises(ValueError):
        await episodic_memory_instance.add_memory_episode(
            producer="invalid_user",
            produced_for="test_agent",
            episode_content="Hello world",
            episode_type="message",
            content_type=ContentType.STRING,
        )

    assert "The producer invalid_user does not belong to the session" in caplog.text


async def test_add_memory_episode_invalid_produced_for(
    episodic_memory_instance, caplog
):
    """Tests that adding an episode with an invalid recipient fails."""
    with pytest.raises(ValueError):
        await episodic_memory_instance.add_memory_episode(
            producer="test_user",
            produced_for="invalid_agent",
            episode_content="Hello world",
            episode_type="message",
            content_type=ContentType.STRING,
        )

    assert (
        "The produced_for invalid_agent does not belong to the session" in caplog.text
    )


async def test_memory_without_sessionmemory(
    episodic_memory_instance_without_sessionmemory, memory_context
):
    """
    Test memory without session memory configured
    """
    long_ep_unique = Episode(
        uuid=uuid.uuid4(),
        content="from long term",
        episode_type="message",
        content_type=ContentType.STRING,
        timestamp=datetime.now(),
        group_id=memory_context.group_id,
        session_id=memory_context.session_id,
        producer_id="test_user",
    )
    episodic_memory_instance_without_sessionmemory.long_term_memory.search.return_value = [
        long_ep_unique,
    ]
    assert episodic_memory_instance_without_sessionmemory.short_term_memory is None
    (
        short_res,
        long_res,
        summary_res,
    ) = await episodic_memory_instance_without_sessionmemory.query_memory(
        "test query", limit=10, property_filter={"key": "value"}
    )
    assert len(short_res) == 0
    assert len(summary_res) == 1
    assert len(summary_res[0]) == 0
    assert len(long_res) == 1
    assert long_res[0] == long_ep_unique
    result = await episodic_memory_instance_without_sessionmemory.add_memory_episode(
        producer="test_user",
        produced_for="test_agent",
        episode_content="Hello world",
        episode_type="message",
        content_type=ContentType.STRING,
    )

    assert result is True
    await episodic_memory_instance_without_sessionmemory.delete_data()
    episodic_memory_instance_without_sessionmemory.long_term_memory.forget_session.assert_awaited_once()


async def test_memory_without_ltm_memory(
    episodic_memory_instance_without_longterm, memory_context
):
    """
    Test memory without long term memory configured
    """
    session_ep_unique = Episode(
        uuid=uuid.uuid4(),
        content="from long term",
        episode_type="message",
        content_type=ContentType.STRING,
        timestamp=datetime.now(),
        group_id=memory_context.group_id,
        session_id=memory_context.session_id,
        producer_id="test_user",
    )
    episodic_memory_instance_without_longterm.short_term_memory.get_session_memory_context.return_value = [
        [session_ep_unique],
        "summary",
    ]
    assert episodic_memory_instance_without_longterm.long_term_memory is None
    (
        short_res,
        long_res,
        summary_res,
    ) = await episodic_memory_instance_without_longterm.query_memory(
        "test query", limit=10, property_filter={"key": "value"}
    )
    assert len(short_res) == 1
    assert len(summary_res) == 1
    assert summary_res[0] == "summary"
    assert len(long_res) == 0
    assert short_res[0] == session_ep_unique
    result = await episodic_memory_instance_without_longterm.add_memory_episode(
        producer="test_user",
        produced_for="test_agent",
        episode_content="Hello world",
        episode_type="message",
        content_type=ContentType.STRING,
    )

    assert result is True
    await episodic_memory_instance_without_longterm.delete_data()
    episodic_memory_instance_without_longterm.short_term_memory.clear_memory.assert_awaited_once()


async def test_query_memory(episodic_memory_instance, memory_context):
    """Tests querying memory and the deduplication of results."""
    common_uuid = uuid.uuid4()
    short_ep = Episode(
        uuid=common_uuid,
        content="from short term",
        episode_type="message",
        content_type=ContentType.STRING,
        timestamp=datetime.now(),
        group_id=memory_context.group_id,
        session_id=memory_context.session_id,
        producer_id="test_user",
    )
    long_ep_unique = Episode(
        uuid=uuid.uuid4(),
        content="from long term",
        episode_type="message",
        content_type=ContentType.STRING,
        timestamp=datetime.now(),
        group_id=memory_context.group_id,
        session_id=memory_context.session_id,
        producer_id="test_user",
    )
    long_ep_common = Episode(
        uuid=common_uuid,
        content="from long term (dupe)",
        episode_type="message",
        content_type=ContentType.STRING,
        timestamp=datetime.now(),
        group_id=memory_context.group_id,
        session_id=memory_context.session_id,
        producer_id="test_user",
    )

    episodic_memory_instance.short_term_memory.get_session_memory_context.return_value = (
        [short_ep],
        "summary",
    )
    episodic_memory_instance.long_term_memory.search.return_value = [
        long_ep_unique,
        long_ep_common,
    ]

    short_res, long_res, summary_res = await episodic_memory_instance.query_memory(
        "test query", limit=10, property_filter={"key": "value"}
    )

    episodic_memory_instance.short_term_memory.get_session_memory_context.assert_awaited_with(
        "test query", limit=10
    )
    expected_filter = {"key": "value", "group_id": memory_context.group_id}
    episodic_memory_instance.long_term_memory.search.assert_awaited_with(
        "test query", 10, expected_filter
    )

    assert short_res == [short_ep]
    assert long_res == [long_ep_unique]  # Deduplicated
    assert summary_res == ["summary"]


async def test_formalize_query_with_context(episodic_memory_instance):
    """Tests the formatting of a query with context from memory."""
    mock_episode = Episode(
        uuid=uuid.uuid4(),
        content="episode content",
        episode_type="message",
        content_type=ContentType.STRING,
        timestamp=datetime.now(),
        group_id="group",
        session_id="session",
        producer_id="user",
    )

    mock_long_term_episode = Episode(
        uuid=uuid.uuid4(),
        content="long term episode content",
        episode_type="message",
        content_type=ContentType.STRING,
        timestamp=datetime.now(),
        group_id="group",
        session_id="session",
        producer_id="user",
    )

    # Patch the instance's own query_memory method
    with patch.object(
        episodic_memory_instance, "query_memory", new=AsyncMock()
    ) as mock_query:
        mock_query.return_value = (
            [mock_episode],
            [mock_long_term_episode],
            ["my summary"],
        )

        result = await episodic_memory_instance.formalize_query_with_context(
            "original query"
        )

        expected = (
            "<Summary>\nmy summary\n\n</Summary>\n"
            "<Episodes>\nepisode content\nlong term episode content\n"
            "</Episodes>\n"
            "<Query>\noriginal query\n</Query>"
        )
        assert result == expected


async def test_formalize_query_with_ordering(episodic_memory_instance):
    """Tests the formatting of a query with context from memory."""
    current_date = datetime.now()
    mock_episode = Episode(
        uuid=uuid.uuid4(),
        content="episode content",
        episode_type="message",
        content_type=ContentType.STRING,
        # make the short memory newer
        timestamp=current_date + timedelta(days=2),
        group_id="group",
        session_id="session",
        producer_id="user",
    )

    mock_long_term_episode = Episode(
        uuid=uuid.uuid4(),
        content="long term episode content",
        episode_type="message",
        content_type=ContentType.STRING,
        timestamp=current_date,
        group_id="group",
        session_id="session",
        producer_id="user",
    )

    # Patch the instance's own query_memory method
    with patch.object(
        episodic_memory_instance, "query_memory", new=AsyncMock()
    ) as mock_query:
        mock_query.return_value = (
            [mock_episode],
            [mock_long_term_episode],
            ["my summary"],
        )

        result = await episodic_memory_instance.formalize_query_with_context(
            "original query"
        )

        expected = (
            "<Summary>\nmy summary\n\n</Summary>\n"
            "<Episodes>\nlong term episode content\nepisode content\n"
            "</Episodes>\n"
            "<Query>\noriginal query\n</Query>"
        )
        assert result == expected


async def test_formalize_query_with_empty_context(episodic_memory_instance):
    """Tests formalizing a query when memory returns no context."""
    with patch.object(
        episodic_memory_instance, "query_memory", new=AsyncMock()
    ) as mock_query:
        mock_query.return_value = ([], [], [])

        result = await episodic_memory_instance.formalize_query_with_context(
            "original query"
        )

        expected = "<Query>\noriginal query\n</Query>"
        assert result == expected


async def test_async_episodic_memory_context_manager(episodic_memory_instance):
    """Tests the AsyncEpisodicMemory context manager."""
    episodic_memory_instance.close = AsyncMock()

    async with AsyncEpisodicMemory(episodic_memory_instance) as mem:
        assert mem is episodic_memory_instance

    # __aexit__ should call close
    episodic_memory_instance.close.assert_awaited_once()


async def test_delete_data(episodic_memory_instance):
    """Tests the delete_data method."""
    await episodic_memory_instance.delete_data()
    episodic_memory_instance.short_term_memory.clear_memory.assert_awaited_once()
    episodic_memory_instance.long_term_memory.forget_session.assert_awaited_once()
