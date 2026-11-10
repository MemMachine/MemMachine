"""Tests of event memory as the owner of its partition namespace, on a SQLite registry and segment store."""

import datetime
import functools
from unittest.mock import create_autospec
from uuid import UUID, uuid4

import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import AsyncEngine

from memmachine_server.common.partition_registry import (
    PartitionIncarnationEndedError,
)
from memmachine_server.common.partition_registry.sqlalchemy_partition_registry import (
    SQLAlchemyPartitionRegistry,
    SQLAlchemyPartitionRegistryParams,
)
from memmachine_server.common.vector_store import VectorStore, VectorStoreSchema
from memmachine_server.episodic_memory.event_memory.data_types import (
    Event,
    NullContext,
    TextBlock,
)
from memmachine_server.episodic_memory.event_memory.deriver.text_deriver import (
    WholeTextDeriver,
)
from memmachine_server.episodic_memory.event_memory.event_memory import (
    EventMemory,
    EventMemoryParams,
    EventMemoryPartition,
)
from memmachine_server.episodic_memory.event_memory.segment_store.sqlalchemy_segment_store import (
    SQLAlchemySegmentStore,
    SQLAlchemySegmentStoreParams,
)
from memmachine_server.episodic_memory.event_memory.segmenter.text_segmenter import (
    TextSegmenter,
)
from server_tests.memmachine_server.common.reranker.fake_embedder import (
    FakeEmbedder,
)
from server_tests.memmachine_server.common.vector_store.in_memory_vector_store_partition import (
    InMemoryVectorStorePartition,
)

NAME = "p"


def _event(text: str) -> Event:
    return Event(
        uuid=uuid4(),
        timestamp=datetime.datetime(2024, 1, 1, tzinfo=datetime.UTC),
        context=NullContext(),
        blocks=[TextBlock(text=text)],
        properties={},
    )


@pytest_asyncio.fixture
async def partition_registry(
    sqlalchemy_sqlite_engine: AsyncEngine,
) -> SQLAlchemyPartitionRegistry:
    registry = SQLAlchemyPartitionRegistry(
        SQLAlchemyPartitionRegistryParams(
            engine=sqlalchemy_sqlite_engine,
            partition_namespace=EventMemory.PARTITION_NAMESPACE,
            tombstone_retention_seconds=0,
        )
    )
    await registry.startup()
    return registry


@pytest_asyncio.fixture
async def segment_store(
    sqlalchemy_sqlite_engine: AsyncEngine,
) -> SQLAlchemySegmentStore:
    store = SQLAlchemySegmentStore(
        SQLAlchemySegmentStoreParams(engine=sqlalchemy_sqlite_engine)
    )
    await store.startup()
    return store


@pytest.fixture
def event_memory(
    partition_registry: SQLAlchemyPartitionRegistry,
    segment_store: SQLAlchemySegmentStore,
) -> EventMemory:
    embedder = FakeEmbedder()
    store_schema = VectorStoreSchema(
        vector_dimensions=embedder.dimensions,
        similarity_metric=embedder.similarity_metric,
        indexed_properties=EventMemory.expected_vector_store_indexed_properties(),
    )
    vector_store = create_autospec(VectorStore, instance=True)
    vector_store.store_schema = store_schema
    vector_store.open_partition_incarnation.side_effect = lambda incarnation: (
        InMemoryVectorStorePartition(store_schema=store_schema)
    )
    vector_store.purge_partition_incarnation.return_value = False
    return EventMemory(
        EventMemoryParams(
            partition_registry=partition_registry,
            segment_store=segment_store,
            vector_store=vector_store,
            segmenter=TextSegmenter(),
            deriver=WholeTextDeriver(),
            embedder=embedder,
        )
    )


async def _purge(event_memory: EventMemory) -> None:
    """Run the namespace's purge rounds until none is due."""
    params = event_memory._params
    purge_round = functools.partial(
        EventMemory.purge_partition_incarnation,
        params.segment_store,
        params.vector_store,
    )
    while await params.partition_registry.run_purge_round(purge_round):
        pass


def _incarnation(partition: EventMemoryPartition) -> UUID:
    return partition._registration.incarnation


@pytest.mark.asyncio
async def test_open_or_create_prepares_a_life_and_reopens_it(
    event_memory: EventMemory,
    segment_store: SQLAlchemySegmentStore,
) -> None:
    created = await event_memory.open_or_create_partition(NAME)
    event = _event("remember the milk")
    await created.encode_events([event])

    reopened = await event_memory.open_or_create_partition(NAME)

    assert _incarnation(reopened) == _incarnation(created)
    segments = await segment_store.open_partition_incarnation(_incarnation(created))
    assert segments is not None
    assert await segments.get_segment_uuids_by_event_uuids([event.uuid])


@pytest.mark.asyncio
async def test_a_deleted_partitions_handle_refuses_every_operation(
    event_memory: EventMemory,
) -> None:
    partition = await event_memory.open_or_create_partition(NAME)
    event = _event("remember the milk")
    await partition.encode_events([event])

    await event_memory.delete_partition(NAME)

    with pytest.raises(PartitionIncarnationEndedError):
        await partition.encode_events([_event("too late")])
    with pytest.raises(PartitionIncarnationEndedError):
        await partition.query("milk")
    with pytest.raises(PartitionIncarnationEndedError):
        await partition.forget_events([event.uuid])


@pytest.mark.asyncio
async def test_an_encode_that_outlives_its_life_raises(
    event_memory: EventMemory,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The check after the writes reports a life that died during them."""
    partition = await event_memory.open_or_create_partition(NAME)
    add_segments = partition._segment_store_partition.add_segments

    async def deleted_meanwhile(segments_to_derivative_uuids) -> None:
        await add_segments(segments_to_derivative_uuids)
        await event_memory.delete_partition(NAME)

    monkeypatch.setattr(
        partition._segment_store_partition, "add_segments", deleted_meanwhile
    )
    with pytest.raises(PartitionIncarnationEndedError) as ended:
        await partition.encode_events([_event("remember the milk")])
    assert ended.value.incarnation == _incarnation(partition)


@pytest.mark.asyncio
async def test_a_partition_created_again_is_a_new_life_that_starts_empty(
    event_memory: EventMemory,
    segment_store: SQLAlchemySegmentStore,
) -> None:
    old = await event_memory.open_or_create_partition(NAME)
    event = _event("remember the milk")
    await old.encode_events([event])
    await event_memory.delete_partition(NAME)

    new = await event_memory.open_or_create_partition(NAME)

    assert _incarnation(new) != _incarnation(old)
    segments = await segment_store.open_partition_incarnation(_incarnation(new))
    assert segments is not None
    assert await segments.get_segment_uuids_by_event_uuids([event.uuid]) == {}


@pytest.mark.asyncio
async def test_the_purge_round_reclaims_a_deleted_life(
    event_memory: EventMemory,
    segment_store: SQLAlchemySegmentStore,
) -> None:
    deleted = await event_memory.open_or_create_partition(NAME)
    await deleted.encode_events([_event("remember the milk")])
    await event_memory.delete_partition(NAME)
    kept = await event_memory.open_or_create_partition(NAME)
    kept_event = _event("buy bread")
    await kept.encode_events([kept_event])

    await _purge(event_memory)

    assert await segment_store.open_partition_incarnation(_incarnation(deleted)) is None
    segments = await segment_store.open_partition_incarnation(_incarnation(kept))
    assert segments is not None
    assert await segments.get_segment_uuids_by_event_uuids([kept_event.uuid])
