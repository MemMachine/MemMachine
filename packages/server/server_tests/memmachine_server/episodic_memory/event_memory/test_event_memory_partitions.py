"""Tests of event memory's partitions (creation, lookup, fencing, and purge) on a SQLite registry and segment store."""

import datetime
from unittest.mock import MagicMock, NonCallableMagicMock, create_autospec
from uuid import UUID, uuid4

import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import AsyncEngine

from memmachine_server.common.metrics_factory import MetricsFactory
from memmachine_server.common.partition_registry import (
    PartitionAlreadyExistsError,
    PartitionDeletedError,
    PartitionPendingError,
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
    EVENT_TIMESTAMP_KEY,
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
            partition_namespace="event_memory",
            tombstone_retention_seconds=0,
        ),
    )
    await registry.migrate()
    await registry.startup()
    return registry


@pytest_asyncio.fixture
async def segment_store(
    sqlalchemy_sqlite_engine: AsyncEngine,
) -> SQLAlchemySegmentStore:
    store = SQLAlchemySegmentStore(
        SQLAlchemySegmentStoreParams(engine=sqlalchemy_sqlite_engine)
    )
    await store.migrate()
    await store.startup()
    return store


@pytest.fixture
def vector_store() -> NonCallableMagicMock:
    """A vector store whose handles hold records in memory."""
    embedder = FakeEmbedder()
    store_schema = VectorStoreSchema(
        vector_dimensions=embedder.dimensions,
        similarity_metric=embedder.similarity_metric,
        indexed_properties=EventMemory.expected_vector_store_indexed_properties(),
    )
    vector_store = create_autospec(VectorStore, instance=True)
    vector_store.store_schema = store_schema
    vector_store.get_partition.side_effect = lambda partition_uuid: (
        InMemoryVectorStorePartition(store_schema=store_schema)
    )
    vector_store.purge_partition.return_value = False
    return vector_store


@pytest.fixture
def event_memory(
    segment_store: SQLAlchemySegmentStore,
    vector_store: NonCallableMagicMock,
    partition_registry: SQLAlchemyPartitionRegistry,
) -> EventMemory:
    return EventMemory(
        EventMemoryParams(
            segment_store=segment_store,
            vector_store=vector_store,
            partition_registry=partition_registry,
            segmenter=TextSegmenter(),
            deriver=WholeTextDeriver(),
            embedder=FakeEmbedder(),
        )
    )


async def _created(event_memory: EventMemory) -> EventMemoryPartition:
    """Create the partition named NAME, and get it."""
    await event_memory.create_partition(NAME)
    partition = await event_memory.get_partition(NAME)
    assert partition is not None
    return partition


async def _purge(event_memory: EventMemory) -> None:
    """Run the namespace's purge rounds until none is due."""
    while await event_memory._partition_manager.run_purge_round():
        pass


def _partition_uuid(partition: EventMemoryPartition) -> UUID:
    return partition._registration.partition_uuid


@pytest.mark.asyncio
async def test_a_created_partition_is_gotten_by_its_name(
    event_memory: EventMemory,
    segment_store: SQLAlchemySegmentStore,
    vector_store: NonCallableMagicMock,
) -> None:
    assert await event_memory.get_partition(NAME) is None
    created = await _created(event_memory)
    event = _event("remember the milk")
    await created.encode_events([event])

    gotten = await event_memory.get_partition(NAME)

    assert gotten is not None
    assert _partition_uuid(gotten) == _partition_uuid(created)
    vector_store.create_partition.assert_awaited_once_with(_partition_uuid(created))
    segments = await segment_store.get_partition(_partition_uuid(created))
    assert segments is not None
    assert await segments.get_segment_uuids_by_event_uuids([event.uuid])


@pytest.mark.asyncio
async def test_partition_operations_are_timed_under_the_memorys_metrics_prefix(
    segment_store: SQLAlchemySegmentStore,
    vector_store: NonCallableMagicMock,
    partition_registry: SQLAlchemyPartitionRegistry,
) -> None:
    metrics_factory = MagicMock(spec=MetricsFactory)
    event_memory = EventMemory(
        EventMemoryParams(
            segment_store=segment_store,
            vector_store=vector_store,
            partition_registry=partition_registry,
            segmenter=TextSegmenter(),
            deriver=WholeTextDeriver(),
            embedder=FakeEmbedder(),
            metrics_factory=metrics_factory,
        )
    )

    await event_memory.create_partition(NAME)

    assert "event_memory_latency_seconds" in [
        histogram.args[0] for histogram in metrics_factory.get_histogram.call_args_list
    ]
    assert {"operation": "create_partition", "status": "ok"} in [
        observation.kwargs["labels"]
        for observation in metrics_factory.get_histogram.return_value.observe.call_args_list
    ]


@pytest.mark.asyncio
async def test_creating_a_taken_name_raises(event_memory: EventMemory) -> None:
    await event_memory.create_partition(NAME)

    with pytest.raises(PartitionAlreadyExistsError):
        await event_memory.create_partition(NAME)


@pytest.mark.asyncio
async def test_getting_a_pending_partition_raises(
    event_memory: EventMemory,
    partition_registry: SQLAlchemyPartitionRegistry,
) -> None:
    pending = await partition_registry._reserve(NAME)

    with pytest.raises(PartitionPendingError) as still_pending:
        await event_memory.get_partition(NAME)
    assert still_pending.value.partition_name == NAME
    await pending.cancel()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "created_again", [False, True], ids=["deleted", "deleted and created again"]
)
async def test_a_deleted_partitions_handle_refuses_every_operation(
    event_memory: EventMemory,
    segment_store: SQLAlchemySegmentStore,
    created_again: bool,
) -> None:
    partition = await _created(event_memory)
    event = _event("remember the milk")
    await partition.encode_events([event])
    vector_partition = partition._vector_store_partition
    assert isinstance(vector_partition, InMemoryVectorStorePartition)
    vector_records = dict(vector_partition.records)

    await event_memory.delete_partition(NAME)
    if created_again:
        await event_memory.create_partition(NAME)

    too_late = _event("too late")
    with pytest.raises(PartitionDeletedError):
        await partition.encode_events([too_late])
    # The refused encode wrote nothing.
    segments = await segment_store.get_partition(_partition_uuid(partition))
    assert segments is not None
    assert await segments.get_segment_uuids_by_event_uuids([too_late.uuid]) == {}
    assert vector_partition.records == vector_records
    with pytest.raises(PartitionDeletedError):
        await partition.query("milk")
    with pytest.raises(PartitionDeletedError):
        await partition.forget_events([event.uuid])
    # An operation with nothing to write or delete is checked too.
    with pytest.raises(PartitionDeletedError):
        await partition.encode_events([])
    with pytest.raises(PartitionDeletedError):
        await partition.forget_events([])


@pytest.mark.asyncio
async def test_an_encode_checks_its_events_before_its_partition(
    event_memory: EventMemory,
) -> None:
    partition = await _created(event_memory)
    await event_memory.delete_partition(NAME)
    reserved = _event("remember the milk").model_copy(
        update={"properties": {EVENT_TIMESTAMP_KEY: "reserved"}}
    )

    with pytest.raises(ValueError, match="reserved for the event memory"):
        await partition.encode_events([reserved])


@pytest.mark.asyncio
async def test_an_encode_that_outlives_its_partition_raises(
    event_memory: EventMemory,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The check after the writes reports a partition deleted during them."""
    partition = await _created(event_memory)
    add_segments = partition._segment_store_partition.add_segments

    async def deleted_meanwhile(segments_to_derivative_uuids) -> None:
        await add_segments(segments_to_derivative_uuids)
        await event_memory.delete_partition(NAME)

    monkeypatch.setattr(
        partition._segment_store_partition, "add_segments", deleted_meanwhile
    )
    with pytest.raises(PartitionDeletedError) as ended:
        await partition.encode_events([_event("remember the milk")])
    assert ended.value.partition_uuid == _partition_uuid(partition)


@pytest.mark.asyncio
async def test_a_name_created_again_names_a_new_partition_that_starts_empty(
    event_memory: EventMemory,
    segment_store: SQLAlchemySegmentStore,
) -> None:
    old = await _created(event_memory)
    event = _event("remember the milk")
    await old.encode_events([event])
    await event_memory.delete_partition(NAME)

    new = await _created(event_memory)

    assert _partition_uuid(new) != _partition_uuid(old)
    segments = await segment_store.get_partition(_partition_uuid(new))
    assert segments is not None
    assert await segments.get_segment_uuids_by_event_uuids([event.uuid]) == {}


@pytest.mark.asyncio
async def test_the_purge_round_reclaims_a_deleted_partition_from_every_store(
    event_memory: EventMemory,
    segment_store: SQLAlchemySegmentStore,
    vector_store: NonCallableMagicMock,
) -> None:
    deleted = await _created(event_memory)
    await deleted.encode_events([_event("remember the milk")])
    await event_memory.delete_partition(NAME)
    kept = await _created(event_memory)
    kept_event = _event("buy bread")
    await kept.encode_events([kept_event])

    await _purge(event_memory)

    assert await segment_store.get_partition(_partition_uuid(deleted)) is None
    vector_store.purge_partition.assert_any_await(_partition_uuid(deleted))
    vector_store.drop_partition.assert_awaited_once_with(_partition_uuid(deleted))
    segments = await segment_store.get_partition(_partition_uuid(kept))
    assert segments is not None
    assert await segments.get_segment_uuids_by_event_uuids([kept_event.uuid])
