"""Tests for the EpisodicMemory class."""

from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock, call, create_autospec
from uuid import uuid4

import pytest

from memmachine.common.episode_store import Episode, EpisodeResponse
from memmachine.common.metrics_factory import MetricsFactory
from memmachine.episodic_memory.episodic_memory import (
    EpisodicMemory,
    EpisodicMemoryParams,
)
from memmachine.episodic_memory.long_term_memory.long_term_memory import (
    LongTermMemory,
)
from memmachine.episodic_memory.short_term_memory.short_term_memory import (
    ShortTermMemory,
)


def create_test_episode(**kwargs):
    """Helper function to create a valid Episode for testing."""
    defaults = {
        "uid": str(uuid4()),
        "session_key": "test_session",
        "sequence_num": 1,
        "content": "test content",
        "created_at": datetime.now(tz=UTC),
        "producer_id": "test_producer",
        "producer_role": "user",
    }
    defaults.update(kwargs)
    return Episode(**defaults)


@pytest.fixture
def mock_metrics_factory():
    return MagicMock(spec=MetricsFactory)


@pytest.fixture
def mock_short_term_memory():
    """Fixture for a mocked ShortTermMemory."""
    return create_autospec(ShortTermMemory, instance=True)


@pytest.fixture
def mock_long_term_memory():
    """Fixture for a mocked LongTermMemory."""
    return create_autospec(LongTermMemory, instance=True)


@pytest.fixture
def episodic_memory_params(
    mock_metrics_factory,
    mock_short_term_memory,
    mock_long_term_memory,
):
    """Fixture for EpisodicMemoryParams."""
    return EpisodicMemoryParams(
        session_key="test_session",
        metrics_factory=mock_metrics_factory,
        short_term_memory=mock_short_term_memory,
        long_term_memory=mock_long_term_memory,
        enabled=True,
    )


@pytest.fixture
def episodic_memory(episodic_memory_params):
    """Fixture for an EpisodicMemory instance."""
    return EpisodicMemory(episodic_memory_params)


@pytest.mark.asyncio
async def test_init_success(episodic_memory_params, mock_metrics_factory):
    """Test successful initialization of EpisodicMemory."""
    memory = EpisodicMemory(episodic_memory_params)
    assert memory.session_key == "test_session"
    assert memory.short_term_memory is not None
    assert memory.long_term_memory is not None
    mock_metrics_factory.get_summary.assert_any_call(
        "Ingestion_latency",
        "Latency of Episode ingestion in milliseconds",
    )
    mock_metrics_factory.get_counter.assert_any_call(
        "Ingestion_count",
        "Count of Episode ingestion",
    )


def test_init_no_memory_configured_raises_error(episodic_memory_params):
    """Test that initialization fails if no memory stores are provided when enabled."""
    episodic_memory_params.short_term_memory = None
    episodic_memory_params.long_term_memory = None
    with pytest.raises(ValueError, match="No memory is configured"):
        EpisodicMemory(episodic_memory_params)


def test_init_disabled(episodic_memory_params, mock_metrics_factory):
    """Test initialization when memory is disabled."""
    episodic_memory_params.enabled = False
    episodic_memory_params.short_term_memory = None
    episodic_memory_params.long_term_memory = None
    memory = EpisodicMemory(episodic_memory_params)
    assert not memory._enabled
    mock_metrics_factory.get_summary.assert_not_called()


@pytest.mark.asyncio
async def test_add_memory_episode(
    episodic_memory,
    mock_short_term_memory,
    mock_long_term_memory,
):
    """Test adding a memory episode."""
    episode = create_test_episode()
    await episodic_memory.add_memory_episodes([episode])
    mock_short_term_memory.add_episodes.assert_awaited_once_with([episode])
    mock_long_term_memory.add_episodes.assert_awaited_once_with([episode])


@pytest.mark.asyncio
async def test_add_memory_episode_when_disabled(
    episodic_memory_params,
    mock_short_term_memory,
    mock_long_term_memory,
):
    """Test that add_memory_episode does nothing when disabled."""
    episodic_memory_params.enabled = False
    memory = EpisodicMemory(episodic_memory_params)
    episode = create_test_episode()
    await memory.add_memory_episodes([episode])
    mock_short_term_memory.add_episodes.assert_not_awaited()
    mock_long_term_memory.add_episodes.assert_not_awaited()


@pytest.mark.asyncio
async def test_add_memory_episode_when_closed(episodic_memory):
    """Test that adding an episode to a closed memory raises RuntimeError."""
    await episodic_memory.close()
    with pytest.raises(RuntimeError, match="Memory is closed test_session"):
        await episodic_memory.add_memory_episodes([create_test_episode()])


@pytest.mark.asyncio
async def test_close(episodic_memory, mock_short_term_memory, mock_long_term_memory):
    """Test closing the episodic memory."""
    await episodic_memory.close()
    mock_short_term_memory.close.assert_awaited_once()
    mock_long_term_memory.close.assert_awaited_once()
    assert episodic_memory._closed


@pytest.mark.asyncio
async def test_close_when_disabled(episodic_memory_params, mock_short_term_memory):
    """Test that close does nothing when disabled."""
    episodic_memory_params.enabled = False
    memory = EpisodicMemory(episodic_memory_params)
    await memory.close()
    mock_short_term_memory.close.assert_not_awaited()
    assert memory._closed


@pytest.mark.asyncio
async def test_delete_episodes(
    episodic_memory,
    mock_short_term_memory,
    mock_long_term_memory,
):
    """Test deleting episodes by UID."""
    uid1, uid2 = str(uuid4()), str(uuid4())
    await episodic_memory.delete_episodes([uid1, uid2])
    mock_short_term_memory.delete_episode.assert_has_awaits([call(uid1), call(uid2)])
    mock_long_term_memory.delete_episodes.assert_awaited_once_with([uid1, uid2])


@pytest.mark.asyncio
async def test_delete_session_episodes(
    episodic_memory,
    mock_short_term_memory,
    mock_long_term_memory,
):
    """Test deleting all episodes in a session."""
    await episodic_memory.delete_session_episodes()
    mock_short_term_memory.clear_memory.assert_awaited_once()
    mock_long_term_memory.delete_matching_episodes.assert_awaited_once()


@pytest.mark.asyncio
async def test_query_memory_all_enabled(
    episodic_memory,
    mock_short_term_memory,
    mock_long_term_memory,
):
    """Test querying memory with both short and long term memory enabled."""
    ep1 = create_test_episode(uid=str(uuid4()), content="short")
    ep2 = create_test_episode(uid=str(uuid4()), content="long")
    ep3 = create_test_episode(uid=ep1.uid, content="duplicate")
    ep1_rsp = EpisodeResponse(**ep1.model_dump())
    ep2_rsp = EpisodeResponse(**ep2.model_dump())
    ep3_rsp = EpisodeResponse(**ep3.model_dump())

    mock_short_term_memory.get_short_term_memory_context.return_value = (
        [ep1_rsp],
        "summary",
    )
    mock_long_term_memory.search.return_value = [ep2_rsp, ep3_rsp]

    response = await episodic_memory.query_memory("test query")

    assert response is not None
    assert response.short_term_memory.episodes == [ep1_rsp]
    assert response.long_term_memory.episodes == [
        ep2_rsp
    ]  # ep3 is a duplicate and should be filtered out
    assert response.short_term_memory.episode_summary == ["summary"]
    mock_short_term_memory.get_short_term_memory_context.assert_awaited_once()
    mock_long_term_memory.search.assert_awaited_once()


@pytest.mark.asyncio
async def test_query_memory_short_term_only(
    episodic_memory_params,
    mock_short_term_memory,
):
    """Test querying when only short-term memory is configured."""
    episodic_memory_params.long_term_memory = None
    memory = EpisodicMemory(episodic_memory_params)

    ep1 = create_test_episode(content="short only")
    ep1_rsp = EpisodeResponse(**ep1.model_dump())
    mock_short_term_memory.get_short_term_memory_context.return_value = (
        [ep1_rsp],
        "summary",
    )

    response = await memory.query_memory("test query")
    assert response is not None
    assert response.short_term_memory.episodes == [ep1_rsp]
    assert response.long_term_memory.episodes == []
    assert response.short_term_memory.episode_summary == ["summary"]


@pytest.mark.asyncio
async def test_query_memory_long_term_only(
    episodic_memory_params,
    mock_long_term_memory,
):
    """Test querying when only long-term memory is configured."""
    episodic_memory_params.short_term_memory = None
    memory = EpisodicMemory(episodic_memory_params)

    ep1 = create_test_episode(content="long only")
    ep1_rsp = EpisodeResponse(**ep1.model_dump())
    mock_long_term_memory.search.return_value = [ep1_rsp]

    response = await memory.query_memory("test query")
    assert response is not None
    assert response.short_term_memory.episodes == []
    assert response.long_term_memory.episodes == [ep1_rsp]
    assert response.short_term_memory.episode_summary == [""]


@pytest.mark.asyncio
async def test_query_memory_when_disabled(episodic_memory_params):
    """Test that query_memory returns None when disabled."""
    episodic_memory_params.enabled = False
    memory = EpisodicMemory(episodic_memory_params)
    response = await memory.query_memory("test query")
    assert response is None


@pytest.mark.asyncio
async def test_formalize_query_with_context(
    episodic_memory,
    mock_short_term_memory,
    mock_long_term_memory,
):
    """Test formalizing a query with context from memory."""
    ep1 = create_test_episode(
        content="hello",
        created_at=datetime(1592, 3, 14, 6, 53, 59, tzinfo=UTC),
    )
    ep2 = create_test_episode(
        content="world",
        created_at=datetime(1828, 2, 7, 18, 28, 45, tzinfo=UTC),
    )

    mock_short_term_memory.get_short_term_memory_context.return_value = (
        [ep1],
        "summary text",
    )
    mock_long_term_memory.search.return_value = [ep2]

    # Mock the timestamp attribute for sorting
    ep1.created_at = ep1.created_at
    ep2.created_at = ep2.created_at

    final_query = await episodic_memory.formalize_query_with_context("original query")

    expected_query = (
        "<Summary>\n"
        "summary text\n"
        "</Summary>\n"
        "<Episodes>\n"
        '[Saturday, March 14, 1592 at 06:53 AM] test_producer: "hello"\n'
        '[Thursday, February 07, 1828 at 06:28 PM] test_producer: "world"\n'
        "</Episodes>\n"
        "<Query>\noriginal query\n</Query>"
    )
    assert final_query == expected_query


@pytest.mark.asyncio
async def test_formalize_query_no_context(
    episodic_memory,
    mock_short_term_memory,
    mock_long_term_memory,
):
    """Test formalizing a query when no context is found."""
    mock_short_term_memory.get_short_term_memory_context.return_value = ([], "")
    mock_long_term_memory.search.return_value = []

    final_query = await episodic_memory.formalize_query_with_context("original query")

    expected_query = "<Query>\noriginal query\n</Query>"
    assert final_query == expected_query


def test_property_setters(episodic_memory):
    """Test the property setters for memory stores."""
    new_stm = AsyncMock()
    new_ltm = AsyncMock()

    episodic_memory.short_term_memory = new_stm
    episodic_memory.long_term_memory = new_ltm

    assert episodic_memory.short_term_memory is new_stm
    assert episodic_memory.long_term_memory is new_ltm
