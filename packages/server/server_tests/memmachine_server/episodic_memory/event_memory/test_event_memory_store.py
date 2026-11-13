"""Tests of event memory's stores as one partitioned store: each step runs on every store."""

import asyncio
from unittest.mock import NonCallableMagicMock, create_autospec, sentinel
from uuid import uuid4

import pytest

from memmachine_server.common.vector_store import VectorStore
from memmachine_server.episodic_memory.event_memory.event_memory_store import (
    EventMemoryStore,
    EventMemoryStorePartition,
)
from memmachine_server.episodic_memory.event_memory.segment_store import (
    SegmentStore,
)

PARTITION_UUID = uuid4()


@pytest.fixture
def segment_store() -> NonCallableMagicMock:
    store = create_autospec(SegmentStore, instance=True)
    store.get_partition.return_value = sentinel.segment_store_partition
    store.purge_partition.return_value = False
    return store


@pytest.fixture
def vector_store() -> NonCallableMagicMock:
    store = create_autospec(VectorStore, instance=True)
    store.get_partition.return_value = sentinel.vector_store_partition
    store.purge_partition.return_value = False
    return store


@pytest.fixture
def event_memory_store(segment_store, vector_store) -> EventMemoryStore:
    return EventMemoryStore(segment_store=segment_store, vector_store=vector_store)


@pytest.mark.asyncio
async def test_a_partition_is_created_in_every_store(
    event_memory_store, segment_store, vector_store
):
    await event_memory_store.create_partition(PARTITION_UUID)

    segment_store.create_partition.assert_awaited_once_with(PARTITION_UUID)
    vector_store.create_partition.assert_awaited_once_with(PARTITION_UUID)


@pytest.mark.asyncio
async def test_every_store_finishes_creating_before_the_first_error_is_raised(
    event_memory_store, segment_store, vector_store
):
    finished = asyncio.Event()

    async def slow(partition_uuid) -> None:
        await asyncio.sleep(0.05)
        finished.set()

    segment_store.create_partition.side_effect = RuntimeError("the backend refused")
    vector_store.create_partition.side_effect = slow

    with pytest.raises(RuntimeError, match="refused"):
        await event_memory_store.create_partition(PARTITION_UUID)
    assert finished.is_set()


@pytest.mark.asyncio
async def test_a_partition_gets_a_handle_from_every_store(
    event_memory_store, segment_store, vector_store
):
    assert await event_memory_store.get_partition(
        PARTITION_UUID
    ) == EventMemoryStorePartition(
        segment_store_partition=sentinel.segment_store_partition,
        vector_store_partition=sentinel.vector_store_partition,
    )
    segment_store.get_partition.assert_awaited_once_with(PARTITION_UUID)
    vector_store.get_partition.assert_awaited_once_with(PARTITION_UUID)


@pytest.mark.asyncio
async def test_a_partition_the_segment_store_holds_nothing_of_is_none(
    event_memory_store, segment_store, vector_store
):
    segment_store.get_partition.return_value = None

    assert await event_memory_store.get_partition(PARTITION_UUID) is None
    vector_store.get_partition.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("in_segment_store", "in_vector_store"),
    [(False, False), (True, False), (False, True), (True, True)],
)
async def test_a_purge_finds_records_when_any_store_does(
    event_memory_store, segment_store, vector_store, in_segment_store, in_vector_store
):
    segment_store.purge_partition.return_value = in_segment_store
    vector_store.purge_partition.return_value = in_vector_store

    assert await event_memory_store.purge_partition(PARTITION_UUID) is (
        in_segment_store or in_vector_store
    )
    segment_store.purge_partition.assert_awaited_once_with(PARTITION_UUID)
    vector_store.purge_partition.assert_awaited_once_with(PARTITION_UUID)


@pytest.mark.asyncio
async def test_every_store_purges_before_the_first_error_is_raised(
    event_memory_store, segment_store, vector_store
):
    segment_store.purge_partition.side_effect = ConnectionError("the store is down")

    with pytest.raises(ConnectionError):
        await event_memory_store.purge_partition(PARTITION_UUID)
    vector_store.purge_partition.assert_awaited_once_with(PARTITION_UUID)


@pytest.mark.asyncio
async def test_every_store_drops_before_the_first_error_is_raised(
    event_memory_store, segment_store, vector_store
):
    segment_store.drop_partition.side_effect = ConnectionError("the store is down")

    with pytest.raises(ConnectionError):
        await event_memory_store.drop_partition(PARTITION_UUID)
    vector_store.drop_partition.assert_awaited_once_with(PARTITION_UUID)
