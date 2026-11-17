"""Tests of event memory over its two stores: its partitions are created and purged in both, and a handle pairs both stores' handles."""

from unittest.mock import NonCallableMagicMock, create_autospec, sentinel
from uuid import uuid4

import pytest

from memmachine_server.common.partition_registry import (
    PartitionRegistry,
    Registration,
)
from memmachine_server.common.partition_registry.partition_registry import (
    Reservation,
    TombstoneClaim,
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
    """A registry that reserves, resolves, and claims the tombstone of one partition."""
    reservation = create_autospec(Reservation, instance=True)
    reservation.partition_name = NAME
    reservation.partition_uuid = PARTITION_UUID
    registration = create_autospec(Registration, instance=True)
    registration.partition_name = NAME
    registration.partition_uuid = PARTITION_UUID
    registry = create_autospec(PartitionRegistry, instance=True)
    registry.reserve.return_value = reservation
    registry.resolve.return_value = registration
    claim = create_autospec(TombstoneClaim, instance=True)
    claim.partition_uuid = PARTITION_UUID
    registry.claim_tombstone.return_value = claim
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


@pytest.mark.asyncio
async def test_a_partition_is_created_in_both_stores(
    event_memory, segment_store, vector_store, partition_registry
):
    await event_memory.create_partition(NAME)

    segment_store.create_partition.assert_awaited_once_with(PARTITION_UUID)
    vector_store.create_partition.assert_awaited_once_with(PARTITION_UUID)
    partition_registry.reserve.return_value.confirm.assert_awaited_once()


@pytest.mark.asyncio
async def test_a_deleted_partition_is_purged_and_dropped_from_both_stores(
    event_memory, segment_store, vector_store, partition_registry
):
    assert await event_memory._partition_manager.run_purge_round() is True

    for store in (segment_store, vector_store):
        store.purge_partition.assert_awaited_once_with(PARTITION_UUID)
        store.drop_partition.assert_awaited_once_with(PARTITION_UUID)
    partition_registry.claim_tombstone.return_value.finish.assert_awaited_once_with(
        False
    )


@pytest.mark.asyncio
async def test_a_partition_gets_a_handle_from_each_store(
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
async def test_a_name_no_partition_holds_is_none(
    event_memory, partition_registry, segment_store
):
    partition_registry.resolve.return_value = None

    assert await event_memory.get_partition(NAME) is None
    segment_store.get_partition.assert_not_awaited()
