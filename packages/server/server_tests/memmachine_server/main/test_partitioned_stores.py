"""Tests of the partitioned stores over the components that hold a MemMachine partition's data."""

from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest

from memmachine_server.common.configuration.episodic_config import (
    DeclarativeLongTermMemoryConf,
    EventLongTermMemoryConf,
)
from memmachine_server.common.errors import SessionNotFoundError
from memmachine_server.common.filter.filter_parser import Comparison as FilterComparison
from memmachine_server.main import partitioned_stores
from memmachine_server.main.partitioned_stores import (
    EPISODE_DELETE_BATCH_SIZE,
    EpisodePartitionedStore,
    LongTermMemoryPartitionedStore,
    SessionDataPartitionedStore,
)


@pytest.fixture
def resource_manager() -> AsyncMock:
    """A resource manager whose deployment configures no semantic storage."""
    manager = AsyncMock()
    manager.config = MagicMock()
    manager.config.semantic_memory.storage_configured.return_value = False
    return manager


@pytest.fixture
def long_term_memory_storage(monkeypatch) -> MagicMock:
    """The long-term memory storage functions, replaced by doubles that record their calls in order."""
    storage = MagicMock()
    for name in (
        "create_event_memory_storage",
        "delete_event_memory_storage",
        "delete_long_term_memory_storage",
    ):
        double = AsyncMock()
        storage.attach_mock(double, name)
        monkeypatch.setattr(partitioned_stores, name, double)
    return storage


@pytest.mark.asyncio
async def test_dropping_a_partition_deletes_its_record_by_its_key(resource_manager):
    partition_uuid = uuid4()
    session_data_manager = resource_manager.get_session_data_manager.return_value

    await SessionDataPartitionedStore(resource_manager).drop_partition(partition_uuid)

    session_data_manager.delete_session.assert_awaited_once_with(str(partition_uuid))


@pytest.mark.asyncio
async def test_dropping_a_partition_without_a_record_succeeds(resource_manager):
    session_data_manager = resource_manager.get_session_data_manager.return_value
    session_data_manager.delete_session.side_effect = SessionNotFoundError("gone")

    await SessionDataPartitionedStore(resource_manager).drop_partition(uuid4())


@pytest.mark.asyncio
async def test_creating_a_partition_writes_its_record_under_its_key(resource_manager):
    create_session_record = AsyncMock()
    partition_uuid = uuid4()

    await SessionDataPartitionedStore(
        resource_manager, create_session_record
    ).create_partition(partition_uuid)

    create_session_record.assert_awaited_once_with(str(partition_uuid))


@pytest.mark.asyncio
async def test_a_store_built_without_a_record_refuses_to_create_a_partition(
    resource_manager,
):
    with pytest.raises(RuntimeError, match="without a session record"):
        await SessionDataPartitionedStore(resource_manager).create_partition(uuid4())


@pytest.mark.asyncio
async def test_a_partitions_record_is_left_to_the_drop(resource_manager):
    store = SessionDataPartitionedStore(resource_manager)

    assert await store.purge_partition(uuid4()) is False

    resource_manager.get_session_data_manager.assert_not_awaited()


@pytest.mark.asyncio
async def test_a_purge_deletes_one_batch_of_the_partitions_episodes(resource_manager):
    partition_uuid = uuid4()
    episode_ids = [uuid4(), uuid4()]
    episode_storage = resource_manager.get_episode_storage.return_value
    episode_storage.get_episode_ids.return_value = episode_ids

    found = await EpisodePartitionedStore(resource_manager).purge_partition(
        partition_uuid
    )

    assert found is True
    episode_storage.get_episode_ids.assert_awaited_once_with(
        filter_expr=FilterComparison(
            field="session_key", op="=", value=str(partition_uuid)
        ),
        page_size=EPISODE_DELETE_BATCH_SIZE,
    )
    episode_storage.delete_episodes.assert_awaited_once_with(episode_ids)
    resource_manager.get_semantic_storage.assert_not_awaited()


@pytest.mark.asyncio
async def test_a_purge_that_finds_no_episodes_reports_none(resource_manager):
    episode_storage = resource_manager.get_episode_storage.return_value
    episode_storage.get_episode_ids.return_value = []

    found = await EpisodePartitionedStore(resource_manager).purge_partition(uuid4())

    assert found is False
    episode_storage.delete_episodes.assert_not_awaited()


@pytest.mark.asyncio
async def test_a_purge_deletes_a_batchs_semantic_history_before_its_episodes(
    resource_manager,
):
    episode_ids = [uuid4()]
    resource_manager.config.semantic_memory.storage_configured.return_value = True
    calls = MagicMock()
    episode_storage = resource_manager.get_episode_storage.return_value
    episode_storage.get_episode_ids.return_value = episode_ids
    semantic_storage = resource_manager.get_semantic_storage.return_value
    calls.attach_mock(semantic_storage.delete_history, "delete_history")
    calls.attach_mock(episode_storage.delete_episodes, "delete_episodes")

    await EpisodePartitionedStore(resource_manager).purge_partition(uuid4())

    assert [name for name, _, _ in calls.mock_calls] == [
        "delete_history",
        "delete_episodes",
    ]
    semantic_storage.delete_history.assert_awaited_once_with(episode_ids)


@pytest.mark.asyncio
async def test_a_failed_semantic_history_delete_keeps_the_batchs_episodes(
    resource_manager,
):
    resource_manager.config.semantic_memory.storage_configured.return_value = True
    episode_storage = resource_manager.get_episode_storage.return_value
    episode_storage.get_episode_ids.return_value = [uuid4()]
    semantic_storage = resource_manager.get_semantic_storage.return_value
    semantic_storage.delete_history.side_effect = RuntimeError("semantic store down")

    with pytest.raises(RuntimeError, match="semantic store down"):
        await EpisodePartitionedStore(resource_manager).purge_partition(uuid4())

    episode_storage.delete_episodes.assert_not_awaited()


@pytest.mark.asyncio
async def test_creating_a_partition_creates_its_event_memory_storage(
    resource_manager, long_term_memory_storage
):
    partition_uuid = uuid4()

    await LongTermMemoryPartitionedStore(resource_manager).create_partition(
        partition_uuid
    )

    long_term_memory_storage.create_event_memory_storage.assert_awaited_once_with(
        str(partition_uuid), resource_manager
    )


@pytest.mark.asyncio
async def test_a_purge_deletes_a_partitions_event_memory_storage(
    resource_manager, long_term_memory_storage
):
    partition_uuid = uuid4()
    session_data_manager = resource_manager.get_session_data_manager.return_value
    session_data_manager.get_session_info.return_value = MagicMock(
        episode_memory_conf=MagicMock(
            long_term_memory=EventLongTermMemoryConf(session_id=str(partition_uuid))
        )
    )

    found = await LongTermMemoryPartitionedStore(resource_manager).purge_partition(
        partition_uuid
    )

    assert found is False
    long_term_memory_storage.delete_event_memory_storage.assert_awaited_once_with(
        str(partition_uuid), resource_manager
    )
    long_term_memory_storage.delete_long_term_memory_storage.assert_not_awaited()


@pytest.mark.asyncio
async def test_a_purge_deletes_the_declarative_memory_a_partitions_record_selects(
    resource_manager, long_term_memory_storage
):
    partition_uuid = uuid4()
    declarative_conf = DeclarativeLongTermMemoryConf(
        session_id=str(partition_uuid),
        vector_graph_store="g",
        embedder="e",
        reranker="r",
    )
    session_data_manager = resource_manager.get_session_data_manager.return_value
    session_data_manager.get_session_info.return_value = MagicMock(
        episode_memory_conf=MagicMock(long_term_memory=declarative_conf)
    )

    found = await LongTermMemoryPartitionedStore(resource_manager).purge_partition(
        partition_uuid
    )

    assert found is False
    session_data_manager.get_session_info.assert_awaited_once_with(str(partition_uuid))
    long_term_memory_storage.delete_long_term_memory_storage.assert_awaited_once_with(
        declarative_conf, resource_manager
    )


@pytest.mark.asyncio
async def test_a_purge_of_a_partition_without_a_record_deletes_its_event_memory_storage(
    resource_manager, long_term_memory_storage
):
    partition_uuid = uuid4()
    session_data_manager = resource_manager.get_session_data_manager.return_value
    session_data_manager.get_session_info.return_value = None

    found = await LongTermMemoryPartitionedStore(resource_manager).purge_partition(
        partition_uuid
    )

    assert found is False
    long_term_memory_storage.delete_event_memory_storage.assert_awaited_once_with(
        str(partition_uuid), resource_manager
    )
    long_term_memory_storage.delete_long_term_memory_storage.assert_not_awaited()
