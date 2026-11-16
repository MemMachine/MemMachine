"""Tests of event memory's steps in its partitions' lifecycle: each runs on every store."""

import asyncio
from collections.abc import Awaitable, Callable
from unittest.mock import NonCallableMagicMock, create_autospec, sentinel
from uuid import UUID, uuid4

import pytest

from memmachine_server.common.partition_registry import (
    PartitionDeletedError,
    PartitionRegistry,
    PurgeRound,
    Registration,
)
from memmachine_server.common.vector_store import VectorStore, VectorStoreSchema
from memmachine_server.episodic_memory.event_memory.deriver import Deriver
from memmachine_server.episodic_memory.event_memory.event_memory import (
    EventMemory,
    EventMemoryParams,
)
from memmachine_server.episodic_memory.event_memory.segment_store import (
    SegmentStore,
)
from memmachine_server.episodic_memory.event_memory.segmenter import Segmenter
from server_tests.memmachine_server.common.reranker.fake_embedder import (
    FakeEmbedder,
)

NAME = "p"
PARTITION_UUID = uuid4()


@pytest.fixture
def segment_store() -> NonCallableMagicMock:
    store = create_autospec(SegmentStore, instance=True)
    store.get_partition.return_value = sentinel.segment_store_partition
    store.purge_partition.return_value = False
    return store


@pytest.fixture
def vector_store() -> NonCallableMagicMock:
    embedder = FakeEmbedder()
    store = create_autospec(VectorStore, instance=True)
    store.store_schema = VectorStoreSchema(
        vector_dimensions=embedder.dimensions,
        similarity_metric=embedder.similarity_metric,
        indexed_properties=EventMemory.expected_vector_store_indexed_properties(),
    )
    store.get_partition.return_value = sentinel.vector_store_partition
    store.purge_partition.return_value = False
    return store


@pytest.fixture
def partition_registry() -> NonCallableMagicMock:
    """A registry that runs a creation's storage step and answers one partition's registration."""
    registration = create_autospec(Registration, instance=True)
    registration.partition_name = NAME
    registration.partition_uuid = PARTITION_UUID
    registry = create_autospec(PartitionRegistry, instance=True)
    registry.partition_namespace = EventMemory.PARTITION_NAMESPACE

    async def create_partition(
        partition_name: str,
        create_partition_storage: Callable[[UUID], Awaitable[None]],
    ) -> Registration:
        await create_partition_storage(PARTITION_UUID)
        return registration

    registry.create_partition.side_effect = create_partition
    registry.get_partition.return_value = registration
    return registry


@pytest.fixture
def event_memory(segment_store, vector_store, partition_registry) -> EventMemory:
    return EventMemory(
        EventMemoryParams(
            segment_store=segment_store,
            vector_store=vector_store,
            partition_registry=partition_registry,
            segmenter=create_autospec(Segmenter, instance=True),
            deriver=create_autospec(Deriver, instance=True),
            embedder=FakeEmbedder(),
        )
    )


@pytest.fixture
def purge_round(segment_store, vector_store) -> PurgeRound:
    return EventMemory.purge_round(segment_store, vector_store)


@pytest.mark.asyncio
async def test_a_partitions_storage_is_created_in_every_store(
    event_memory, segment_store, vector_store
):
    await event_memory.create_partition(NAME)

    segment_store.create_partition.assert_awaited_once_with(PARTITION_UUID)
    vector_store.create_partition.assert_awaited_once_with(PARTITION_UUID)


@pytest.mark.asyncio
async def test_every_store_finishes_creating_before_the_first_error_is_raised(
    event_memory, segment_store, vector_store
):
    finished = asyncio.Event()

    async def slow(partition_uuid) -> None:
        await asyncio.sleep(0.05)
        finished.set()

    segment_store.create_partition.side_effect = RuntimeError("the backend refused")
    vector_store.create_partition.side_effect = slow

    with pytest.raises(RuntimeError, match="refused"):
        await event_memory.create_partition(NAME)
    assert finished.is_set()


@pytest.mark.asyncio
async def test_a_partition_gets_a_handle_from_every_store(
    event_memory, segment_store, vector_store
):
    partition = await event_memory.get_partition(NAME)

    assert partition is not None
    assert partition._segment_store_partition is sentinel.segment_store_partition
    assert partition._vector_store_partition is sentinel.vector_store_partition
    segment_store.get_partition.assert_awaited_once_with(PARTITION_UUID)
    vector_store.get_partition.assert_awaited_once_with(PARTITION_UUID)


@pytest.mark.asyncio
async def test_a_partition_the_segment_store_holds_nothing_of_is_none(
    event_memory, segment_store, vector_store
):
    """A store finds nothing only of a partition deleted, and dropped, since the registry's read."""
    segment_store.get_partition.return_value = None

    assert await event_memory.get_partition(NAME) is None
    vector_store.get_partition.assert_not_awaited()


@pytest.mark.asyncio
async def test_a_partition_the_vector_store_holds_nothing_of_is_none(
    event_memory, vector_store
):
    vector_store.get_partition.return_value = None

    assert await event_memory.get_partition(NAME) is None


@pytest.mark.asyncio
async def test_a_created_partition_a_store_holds_nothing_of_was_deleted(
    event_memory, segment_store
):
    segment_store.get_partition.return_value = None

    with pytest.raises(PartitionDeletedError) as deleted:
        await event_memory.create_partition(NAME)
    assert deleted.value.partition_name == NAME
    assert deleted.value.partition_uuid == PARTITION_UUID


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("in_segment_store", "in_vector_store"),
    [(False, False), (True, False), (False, True), (True, True)],
)
async def test_a_purge_round_finds_records_when_any_store_does_and_drops_once_none_does(
    purge_round,
    segment_store,
    vector_store,
    in_segment_store,
    in_vector_store,
):
    segment_store.purge_partition.return_value = in_segment_store
    vector_store.purge_partition.return_value = in_vector_store
    found = in_segment_store or in_vector_store

    assert await purge_round(PARTITION_UUID) is found
    segment_store.purge_partition.assert_awaited_once_with(PARTITION_UUID)
    vector_store.purge_partition.assert_awaited_once_with(PARTITION_UUID)
    for store in (segment_store, vector_store):
        if found:
            store.drop_partition.assert_not_awaited()
        else:
            store.drop_partition.assert_awaited_once_with(PARTITION_UUID)


@pytest.mark.asyncio
async def test_every_store_purges_before_the_first_error_is_raised(
    purge_round, segment_store, vector_store
):
    segment_store.purge_partition.side_effect = ConnectionError("the store is down")

    with pytest.raises(ConnectionError):
        await purge_round(PARTITION_UUID)
    vector_store.purge_partition.assert_awaited_once_with(PARTITION_UUID)
    segment_store.drop_partition.assert_not_awaited()
    vector_store.drop_partition.assert_not_awaited()


@pytest.mark.asyncio
async def test_every_store_drops_before_the_first_error_is_raised(
    purge_round, segment_store, vector_store
):
    segment_store.drop_partition.side_effect = ConnectionError("the store is down")

    with pytest.raises(ConnectionError):
        await purge_round(PARTITION_UUID)
    vector_store.drop_partition.assert_awaited_once_with(PARTITION_UUID)
