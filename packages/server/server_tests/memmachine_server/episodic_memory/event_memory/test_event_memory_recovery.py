"""EventMemory's writes, forgets and background work across its two stores."""

import datetime
import logging
from collections.abc import Awaitable, Callable, Iterable
from datetime import UTC, timedelta
from unittest.mock import create_autospec
from uuid import uuid4

import pytest
import pytest_asyncio
from sqlalchemy import ColumnElement, func, select, update
from sqlalchemy.ext.asyncio import AsyncEngine

from memmachine_server.common.partition_registry.sqlalchemy_partition_registry import (
    SQLAlchemyPartitionRegistry,
    SQLAlchemyPartitionRegistryParams,
)
from memmachine_server.common.vector_store import (
    Record,
    VectorStore,
    VectorStoreSchema,
)
from memmachine_server.episodic_memory.event_memory.data_types import (
    Event,
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
from memmachine_server.episodic_memory.event_memory.segment_store import (
    SegmentStoreEventAlreadyExistsError,
)
from memmachine_server.episodic_memory.event_memory.segment_store.sqlalchemy_segment_store import (
    IndexingJobRow,
    SQLAlchemySegmentStore,
    SQLAlchemySegmentStoreParams,
    TombstoneRow,
)
from memmachine_server.episodic_memory.event_memory.segmenter.text_segmenter import (
    TextSegmenter,
)
from server_tests.memmachine_server.common.reranker.fake_embedder import (
    FakeEmbedder,
)

from .conftest import (
    InMemorySegmentStorePartition,
    InMemoryVectorStorePartition,
    create_event_memory_partition,
)

pytestmark = pytest.mark.asyncio

_T0 = datetime.datetime(2025, 6, 1, tzinfo=UTC)


def _event(text: str) -> Event:
    return Event(uuid=uuid4(), timestamp=_T0, blocks=[TextBlock(text=text)])


class _FailingVectorStorePartition(InMemoryVectorStorePartition):
    """A vector store partition whose upserts raise while told to."""

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.fail_upserts = False

    async def upsert(self, *, records: Iterable[Record]) -> None:
        if self.fail_upserts:
            raise ConnectionError("upsert failed")
        await super().upsert(records=records)


@pytest.fixture
def store_schema(fake_embedder) -> VectorStoreSchema:
    return VectorStoreSchema(
        vector_dimensions=fake_embedder.dimensions,
        similarity_metric=fake_embedder.similarity_metric,
        indexed_properties=EventMemory.expected_vector_store_indexed_properties(),
    )


@pytest.fixture
def vector_store_partition(store_schema) -> _FailingVectorStorePartition:
    return _FailingVectorStorePartition(store_schema=store_schema)


@pytest_asyncio.fixture
async def partition(
    fake_segment_store_partition, vector_store_partition, store_schema, fake_embedder
) -> EventMemoryPartition:
    return await create_event_memory_partition(
        segment_store_partition=fake_segment_store_partition,
        vector_store_partition=vector_store_partition,
        store_schema=store_schema,
        segmenter=TextSegmenter(),
        deriver=WholeTextDeriver(),
        embedder=fake_embedder,
    )


# ---------------------------------------------------------------------------
# Encoding
# ---------------------------------------------------------------------------


async def test_encode_clears_the_indexing_jobs_once_the_vectors_are_written(
    partition: EventMemoryPartition,
    fake_segment_store_partition: InMemorySegmentStorePartition,
    vector_store_partition: _FailingVectorStorePartition,
):
    await partition.encode_events([_event("hello"), _event("world")])

    assert len(fake_segment_store_partition.event_to_segments) == 2
    assert len(vector_store_partition.records) == 2
    assert fake_segment_store_partition.indexing_jobs == {}


async def test_a_failed_upsert_leaves_the_events_stored_and_their_derivatives_pending(
    partition: EventMemoryPartition,
    fake_segment_store_partition: InMemorySegmentStorePartition,
    vector_store_partition: _FailingVectorStorePartition,
    caplog: pytest.LogCaptureFixture,
):
    """The encode succeeds: its events are stored, and it logs the failed write."""
    vector_store_partition.fail_upserts = True
    event = _event("hello")

    with caplog.at_level(logging.WARNING):
        await partition.encode_events([event])

    assert [
        record
        for record in caplog.records
        if record.levelno == logging.WARNING
        and isinstance(record.exc_info and record.exc_info[1], ConnectionError)
    ]

    assert event.uuid in fake_segment_store_partition.event_to_segments
    ((segment_uuid, derivatives),) = fake_segment_store_partition.indexing_jobs.items()
    assert segment_uuid in fake_segment_store_partition.event_to_segments[event.uuid]
    linked = await fake_segment_store_partition.get_segment_uuids_by_derivative_uuids(
        derivative.uuid for derivative in derivatives
    )
    assert set(linked) == {derivative.uuid for derivative in derivatives}


async def test_encoding_a_held_event_again_is_rejected_whole(
    partition: EventMemoryPartition,
    fake_segment_store_partition: InMemorySegmentStorePartition,
):
    held = _event("held")
    await partition.encode_events([held])
    new = _event("new")

    with pytest.raises(SegmentStoreEventAlreadyExistsError) as raised:
        await partition.encode_events([new, held])

    assert raised.value.event_uuids == {held.uuid}
    assert new.uuid not in fake_segment_store_partition.event_to_segments


# ---------------------------------------------------------------------------
# Forgetting
# ---------------------------------------------------------------------------


async def test_forget_tombstones_the_segments_and_leaves_the_vectors_to_the_background(
    partition: EventMemoryPartition,
    fake_segment_store_partition: InMemorySegmentStorePartition,
    vector_store_partition: _FailingVectorStorePartition,
):
    event = _event("hello")
    await partition.encode_events([event])

    await partition.forget_events([event.uuid])

    assert fake_segment_store_partition.event_to_segments == {}
    assert len(fake_segment_store_partition.tombstones) == 1
    assert len(vector_store_partition.records) == 1


async def test_forgetting_events_the_partition_does_not_hold_deletes_nothing(
    partition: EventMemoryPartition,
    fake_segment_store_partition: InMemorySegmentStorePartition,
    vector_store_partition: _FailingVectorStorePartition,
):
    await partition.encode_events([_event("hello")])

    await partition.forget_events([uuid4()])

    assert len(vector_store_partition.records) == 1
    assert fake_segment_store_partition.tombstones == {}


# ---------------------------------------------------------------------------
# Querying
# ---------------------------------------------------------------------------


async def test_a_query_drops_a_hit_whose_derivative_is_deleted(
    partition: EventMemoryPartition,
    vector_store_partition: _FailingVectorStorePartition,
    fake_embedder: FakeEmbedder,
):
    await partition.encode_events([_event("hello")])
    [embedding] = await fake_embedder.ingest_embed(["hello"])
    await vector_store_partition.upsert(
        records=[
            Record(
                uuid=uuid4(), vector=embedding, properties={EVENT_TIMESTAMP_KEY: _T0}
            )
        ]
    )

    result = await partition.query("hello")

    assert len(result.scored_segment_contexts) == 1


# ---------------------------------------------------------------------------
# Background work, over a SQLite segment store
# ---------------------------------------------------------------------------


@pytest_asyncio.fixture
async def segment_store(sqlalchemy_sqlite_engine: AsyncEngine):
    store = SQLAlchemySegmentStore(
        SQLAlchemySegmentStoreParams(engine=sqlalchemy_sqlite_engine)
    )
    await store.startup()
    yield store
    await store.shutdown()


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
    await registry.startup()
    return registry


def _event_memory(
    segment_store: SQLAlchemySegmentStore,
    partition_registry: SQLAlchemyPartitionRegistry,
    vector_store_partition: _FailingVectorStorePartition,
    store_schema: VectorStoreSchema,
    embedder: FakeEmbedder,
) -> EventMemory:
    """An EventMemory whose vector store holds every partition's records in `vector_store_partition`."""
    vector_store = create_autospec(VectorStore, instance=True)
    vector_store.store_schema = store_schema
    vector_store.get_partition.return_value = vector_store_partition
    return EventMemory(
        EventMemoryParams(
            segment_store=segment_store,
            vector_store=vector_store,
            partition_registry=partition_registry,
            segmenter=TextSegmenter(),
            deriver=WholeTextDeriver(),
            embedder=embedder,
        )
    )


@pytest.fixture
def event_memory(
    segment_store: SQLAlchemySegmentStore,
    partition_registry: SQLAlchemyPartitionRegistry,
    vector_store_partition: _FailingVectorStorePartition,
    store_schema: VectorStoreSchema,
    fake_embedder,
) -> EventMemory:
    return _event_memory(
        segment_store,
        partition_registry,
        vector_store_partition,
        store_schema,
        fake_embedder,
    )


async def _created(event_memory: EventMemory) -> EventMemoryPartition:
    """Create the partition named "partition", and get it."""
    await event_memory.create_partition("partition")
    partition = await event_memory.get_partition("partition")
    assert partition is not None
    return partition


def _database_time_ago(engine: AsyncEngine, delta: timedelta) -> ColumnElement:
    """The database clock's now less `delta`, in the form of its own stamps."""
    if engine.dialect.name == "sqlite":
        return func.datetime("now", f"-{int(delta.total_seconds())} seconds")
    return func.now() - delta


async def _age_indexing_jobs(engine: AsyncEngine, delta: timedelta) -> None:
    """Make every indexing job claimable since `delta` ago, past its ingest's lease."""
    async with engine.begin() as connection:
        await connection.execute(
            update(IndexingJobRow).values(
                claimable_at=_database_time_ago(engine, delta)
            )
        )


async def _age_tombstones(engine: AsyncEngine, delta: timedelta) -> None:
    """Move every recorded first delete back to `delta` ago."""
    async with engine.begin() as connection:
        await connection.execute(
            update(TombstoneRow)
            .where(TombstoneRow.first_deleted_at.is_not(None))
            .values(first_deleted_at=_database_time_ago(engine, delta))
        )


async def _indexing_job_count(engine: AsyncEngine) -> int:
    async with engine.connect() as connection:
        return (
            await connection.execute(select(func.count()).select_from(IndexingJobRow))
        ).scalar_one()


async def _tombstone_count(engine: AsyncEngine) -> int:
    async with engine.connect() as connection:
        return (
            await connection.execute(select(func.count()).select_from(TombstoneRow))
        ).scalar_one()


async def test_background_work_writes_the_derivatives_an_encode_left_pending(
    event_memory: EventMemory,
    vector_store_partition: _FailingVectorStorePartition,
    sqlalchemy_sqlite_engine: AsyncEngine,
):
    partition = await _created(event_memory)
    vector_store_partition.fail_upserts = True
    await partition.encode_events([_event("hello")])
    vector_store_partition.fail_upserts = False

    assert not await event_memory.index_pending_segments()
    await _age_indexing_jobs(sqlalchemy_sqlite_engine, timedelta(hours=1))
    assert await event_memory.index_pending_segments()

    assert len(vector_store_partition.records) == 1
    assert not await event_memory.index_pending_segments()
    result = await partition.query("hello")
    assert len(result.scored_segment_contexts) == 1


async def test_background_work_retries_indexing_jobs_after_a_failed_write(
    event_memory: EventMemory,
    vector_store_partition: _FailingVectorStorePartition,
    sqlalchemy_sqlite_engine: AsyncEngine,
):
    partition = await _created(event_memory)
    vector_store_partition.fail_upserts = True
    await partition.encode_events([_event("hello")])
    await _age_indexing_jobs(sqlalchemy_sqlite_engine, timedelta(hours=1))

    assert await event_memory.index_pending_segments()
    assert vector_store_partition.records == {}
    # In its backoff now.
    assert not await event_memory.index_pending_segments()


async def test_background_work_deletes_a_forgotten_events_vectors_twice(
    event_memory: EventMemory,
    vector_store_partition: _FailingVectorStorePartition,
    sqlalchemy_sqlite_engine: AsyncEngine,
):
    partition = await _created(event_memory)
    event = _event("hello")
    await partition.encode_events([event])
    await partition.forget_events([event.uuid])
    assert len(vector_store_partition.records) == 1

    assert await event_memory.purge_forgotten_segments()
    assert vector_store_partition.records == {}
    assert await _tombstone_count(sqlalchemy_sqlite_engine) == 1
    assert not await event_memory.purge_forgotten_segments()

    await _age_tombstones(sqlalchemy_sqlite_engine, timedelta(days=2))
    assert await event_memory.purge_forgotten_segments()
    assert await _tombstone_count(sqlalchemy_sqlite_engine) == 0


async def test_an_event_encoded_again_right_after_its_forget_keeps_its_new_vectors(
    event_memory: EventMemory,
    vector_store_partition: _FailingVectorStorePartition,
    sqlalchemy_sqlite_engine: AsyncEngine,
):
    """Both deletes of the forgotten life reach only its own derivatives, not those of the event encoded again under its UUID."""
    partition = await _created(event_memory)
    event = _event("hello")
    await partition.encode_events([event])
    await partition.forget_events([event.uuid])
    [forgotten_record_uuid] = vector_store_partition.records

    await partition.encode_events([event])
    assert await event_memory.purge_forgotten_segments()
    await _age_tombstones(sqlalchemy_sqlite_engine, timedelta(days=2))
    assert await event_memory.purge_forgotten_segments()
    assert await _tombstone_count(sqlalchemy_sqlite_engine) == 0

    assert len(vector_store_partition.records) == 1
    assert forgotten_record_uuid not in vector_store_partition.records
    result = await partition.query("hello")
    assert [
        [segment.event_uuid for segment in context.segments]
        for context in result.scored_segment_contexts
    ] == [[event.uuid]]


async def test_a_forget_ends_the_derivatives_pending_with_it(
    event_memory: EventMemory,
    vector_store_partition: _FailingVectorStorePartition,
    sqlalchemy_sqlite_engine: AsyncEngine,
):
    partition = await _created(event_memory)
    vector_store_partition.fail_upserts = True
    event = _event("hello")
    await partition.encode_events([event])
    vector_store_partition.fail_upserts = False

    await partition.forget_events([event.uuid])
    await _age_indexing_jobs(sqlalchemy_sqlite_engine, timedelta(hours=1))

    assert not await event_memory.index_pending_segments()
    assert vector_store_partition.records == {}


class _HookedEmbedder(FakeEmbedder):
    """An embedder that awaits a hook, while one is set, before embedding for ingestion."""

    def __init__(self) -> None:
        super().__init__()
        self.hook: Callable[[], Awaitable[None]] | None = None

    async def _ingest_embed(
        self, inputs: list, max_attempts: int = 1
    ) -> list[list[float]]:
        if self.hook is not None:
            await self.hook()
        return await super()._ingest_embed(inputs, max_attempts)


async def test_a_forget_while_the_background_work_embeds_ends_its_write(
    segment_store: SQLAlchemySegmentStore,
    vector_store_partition: _FailingVectorStorePartition,
    store_schema: VectorStoreSchema,
    sqlalchemy_sqlite_engine: AsyncEngine,
    partition_registry: SQLAlchemyPartitionRegistry,
):
    embedder = _HookedEmbedder()
    event_memory = _event_memory(
        segment_store,
        partition_registry,
        vector_store_partition,
        store_schema,
        embedder,
    )
    partition = await _created(event_memory)
    vector_store_partition.fail_upserts = True
    event = _event("hello")
    await partition.encode_events([event])
    vector_store_partition.fail_upserts = False
    await _age_indexing_jobs(sqlalchemy_sqlite_engine, timedelta(hours=1))

    async def forget_the_event() -> None:
        embedder.hook = None
        await partition.forget_events([event.uuid])

    embedder.hook = forget_the_event
    assert await event_memory.index_pending_segments()

    assert vector_store_partition.records == {}
    assert not await event_memory.index_pending_segments()


async def test_a_deletion_of_the_partition_while_the_background_work_embeds_ends_its_write(
    segment_store: SQLAlchemySegmentStore,
    partition_registry: SQLAlchemyPartitionRegistry,
    vector_store_partition: _FailingVectorStorePartition,
    store_schema: VectorStoreSchema,
    sqlalchemy_sqlite_engine: AsyncEngine,
):
    embedder = _HookedEmbedder()
    event_memory = _event_memory(
        segment_store,
        partition_registry,
        vector_store_partition,
        store_schema,
        embedder,
    )
    partition = await _created(event_memory)
    vector_store_partition.fail_upserts = True
    await partition.encode_events([_event("hello")])
    vector_store_partition.fail_upserts = False
    await _age_indexing_jobs(sqlalchemy_sqlite_engine, timedelta(hours=1))

    async def delete_the_partition() -> None:
        embedder.hook = None
        await event_memory.delete_partition("partition")

    embedder.hook = delete_the_partition
    assert await event_memory.index_pending_segments()

    assert vector_store_partition.records == {}
    assert await _indexing_job_count(sqlalchemy_sqlite_engine) == 0


async def test_background_work_ends_a_deleted_partitions_indexing_job_without_writing(
    event_memory: EventMemory,
    vector_store_partition: _FailingVectorStorePartition,
    sqlalchemy_sqlite_engine: AsyncEngine,
):
    """The partition's purge reclaims its records, so its pending segments are not written."""
    partition = await _created(event_memory)
    vector_store_partition.fail_upserts = True
    await partition.encode_events([_event("hello")])
    vector_store_partition.fail_upserts = False
    await event_memory.delete_partition("partition")
    await _age_indexing_jobs(sqlalchemy_sqlite_engine, timedelta(hours=1))

    assert await event_memory.index_pending_segments()

    assert vector_store_partition.records == {}
    assert await _indexing_job_count(sqlalchemy_sqlite_engine) == 0


async def test_background_work_leaves_a_deleted_partitions_forgotten_vectors_to_its_purge(
    event_memory: EventMemory,
    vector_store_partition: _FailingVectorStorePartition,
    sqlalchemy_sqlite_engine: AsyncEngine,
):
    partition = await _created(event_memory)
    event = _event("hello")
    await partition.encode_events([event])
    await partition.forget_events([event.uuid])
    await event_memory.delete_partition("partition")

    assert await event_memory.purge_forgotten_segments()
    assert len(vector_store_partition.records) == 1
    assert not await event_memory.purge_forgotten_segments()

    await _age_tombstones(sqlalchemy_sqlite_engine, timedelta(days=2))
    assert await event_memory.purge_forgotten_segments()
    assert await _tombstone_count(sqlalchemy_sqlite_engine) == 0
