"""EventMemory's writes, forgets and background work across its two stores."""

import asyncio
import datetime
from collections.abc import Iterable
from datetime import UTC, timedelta
from unittest.mock import call, create_autospec
from uuid import UUID, uuid4

import pytest
import pytest_asyncio
from sqlalchemy import ColumnElement, func, select, update
from sqlalchemy.ext.asyncio import AsyncEngine

from memmachine_server.common.vector_store import (
    Record,
    VectorStore,
    VectorStoreSchema,
)
from memmachine_server.episodic_memory.event_memory.data_types import (
    Event,
    EvictionOptions,
    TextBlock,
)
from memmachine_server.episodic_memory.event_memory.deriver import Deriver
from memmachine_server.episodic_memory.event_memory.deriver.text_deriver import (
    WholeTextDeriver,
)
from memmachine_server.episodic_memory.event_memory.event_memory import (
    EVENT_TIMESTAMP_KEY,
    EventMemory,
    EventMemoryParams,
    EventMemoryPartition,
)
from memmachine_server.episodic_memory.event_memory.event_memory_store import (
    EventMemoryStore,
    EventMemoryStoreEventAlreadyStoredError,
)
from memmachine_server.episodic_memory.event_memory.event_memory_store.sqlalchemy_event_memory_store import (
    PendingDerivativesRow,
    SQLAlchemyEventMemoryStore,
    SQLAlchemyEventMemoryStoreParams,
    TombstoneRow,
)
from memmachine_server.episodic_memory.event_memory.segmenter import Segmenter
from memmachine_server.episodic_memory.event_memory.segmenter.text_segmenter import (
    TextSegmenter,
)
from server_tests.memmachine_server.common.reranker.fake_embedder import (
    FakeEmbedder,
)

from .conftest import (
    InMemoryEventMemoryStorePartition,
    InMemoryVectorStorePartition,
    open_event_memory_partition,
)

pytestmark = pytest.mark.asyncio

_T0 = datetime.datetime(2025, 6, 1, tzinfo=UTC)


def _event(text: str, *, minutes: int = 0) -> Event:
    return Event(
        uuid=uuid4(),
        timestamp=_T0 + timedelta(minutes=minutes),
        blocks=[TextBlock(text=text)],
    )


class _FailingVectorStorePartition(InMemoryVectorStorePartition):
    """A vector store partition whose upserts or deletes raise while told to.

    An upsert told to fail after writing writes its records, then raises.
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.fail_upserts = False
        self.fail_upserts_after_writing = False
        self.fail_deletes = False
        self.upsert_delay_seconds = 0.0
        self.deleted = asyncio.Event()

    async def upsert(self, *, records: Iterable[Record]) -> None:
        if self.upsert_delay_seconds:
            await asyncio.sleep(self.upsert_delay_seconds)
        if self.fail_upserts:
            raise ConnectionError("upsert failed")
        await super().upsert(records=records)
        if self.fail_upserts_after_writing:
            raise ConnectionError("upsert failed after writing")

    async def delete(self, *, record_uuids: Iterable[UUID]) -> None:
        if self.fail_deletes:
            raise ConnectionError("delete failed")
        await super().delete(record_uuids=record_uuids)
        self.deleted.set()


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
    fake_event_memory_store_partition, vector_store_partition, fake_embedder
) -> EventMemoryPartition:
    return await open_event_memory_partition(
        event_memory_store_partition=fake_event_memory_store_partition,
        vector_store_partition=vector_store_partition,
        segmenter=Segmenter([TextSegmenter()]),
        deriver=Deriver([WholeTextDeriver()]),
        embedder=fake_embedder,
    )


# ---------------------------------------------------------------------------
# Encoding
# ---------------------------------------------------------------------------


async def test_encode_clears_the_pending_derivatives_once_the_vectors_are_written(
    partition: EventMemoryPartition,
    fake_event_memory_store_partition: InMemoryEventMemoryStorePartition,
    vector_store_partition: _FailingVectorStorePartition,
):
    await partition.encode_events([_event("hello"), _event("world")])

    assert len(fake_event_memory_store_partition.event_to_segments) == 2
    assert len(vector_store_partition.records) == 2
    assert fake_event_memory_store_partition.pending_derivatives == {}


async def test_a_failed_upsert_leaves_the_events_stored_and_their_derivatives_pending(
    partition: EventMemoryPartition,
    fake_event_memory_store_partition: InMemoryEventMemoryStorePartition,
    vector_store_partition: _FailingVectorStorePartition,
):
    vector_store_partition.fail_upserts = True
    event = _event("hello")

    with pytest.raises(ConnectionError):
        await partition.encode_events([event])

    assert event.uuid in fake_event_memory_store_partition.event_to_segments
    (pending,) = fake_event_memory_store_partition.pending_derivatives.values()
    assert pending.event_uuid == event.uuid
    linked = (
        await fake_event_memory_store_partition.get_segment_uuids_by_derivative_uuids(
            derivative.uuid for derivative in pending.derivatives
        )
    )
    assert set(linked) == {derivative.uuid for derivative in pending.derivatives}


async def test_an_upsert_outlasting_the_write_budget_is_cancelled(
    fake_event_memory_store_partition: InMemoryEventMemoryStorePartition,
    vector_store_partition: _FailingVectorStorePartition,
    store_schema: VectorStoreSchema,
    fake_embedder,
):
    event_memory_store = create_autospec(EventMemoryStore, instance=True)
    event_memory_store.open_or_create_partition.return_value = (
        fake_event_memory_store_partition
    )
    vector_store = create_autospec(VectorStore, instance=True)
    vector_store.open_or_create_partition.return_value = vector_store_partition
    event_memory = EventMemory(
        EventMemoryParams(
            event_memory_store=event_memory_store,
            vector_store=vector_store,
            store_schema=store_schema,
            segmenter=Segmenter([TextSegmenter()]),
            deriver=Deriver([WholeTextDeriver()]),
            embedder=fake_embedder,
            write_budget_seconds=1,
        )
    )
    partition = await event_memory.open_or_create_partition("partition")
    vector_store_partition.upsert_delay_seconds = 30

    with pytest.raises(TimeoutError):
        await asyncio.wait_for(partition.encode_events([_event("hello")]), timeout=10)

    assert vector_store_partition.records == {}
    assert len(fake_event_memory_store_partition.pending_derivatives) == 1


async def test_encoding_a_held_event_again_is_rejected_whole(
    partition: EventMemoryPartition,
    fake_event_memory_store_partition: InMemoryEventMemoryStorePartition,
):
    held = _event("held")
    await partition.encode_events([held])
    new = _event("new")

    with pytest.raises(EventMemoryStoreEventAlreadyStoredError) as raised:
        await partition.encode_events([new, held])

    assert raised.value.event_uuids == {held.uuid}
    assert new.uuid not in fake_event_memory_store_partition.event_to_segments


# ---------------------------------------------------------------------------
# Forgetting
# ---------------------------------------------------------------------------


async def test_forget_deletes_the_vectors_and_records_the_first_delete(
    partition: EventMemoryPartition,
    fake_event_memory_store_partition: InMemoryEventMemoryStorePartition,
    vector_store_partition: _FailingVectorStorePartition,
):
    event = _event("hello")
    await partition.encode_events([event])

    await partition.forget_events([event.uuid])

    assert vector_store_partition.records == {}
    assert fake_event_memory_store_partition.event_to_segments == {}
    (tombstone_uuid,) = fake_event_memory_store_partition.tombstones
    assert fake_event_memory_store_partition.tombstones_awaiting_second_delete == {
        tombstone_uuid
    }


async def test_a_failed_first_delete_leaves_the_forget_complete_and_its_deletes_to_the_background(
    partition: EventMemoryPartition,
    fake_event_memory_store_partition: InMemoryEventMemoryStorePartition,
    vector_store_partition: _FailingVectorStorePartition,
):
    event = _event("hello")
    await partition.encode_events([event])
    vector_store_partition.fail_deletes = True

    await partition.forget_events([event.uuid])

    assert fake_event_memory_store_partition.event_to_segments == {}
    assert len(fake_event_memory_store_partition.tombstones) == 1
    assert fake_event_memory_store_partition.tombstones_awaiting_second_delete == set()


async def test_forgetting_events_the_partition_does_not_hold_deletes_nothing(
    partition: EventMemoryPartition,
    fake_event_memory_store_partition: InMemoryEventMemoryStorePartition,
    vector_store_partition: _FailingVectorStorePartition,
):
    await partition.encode_events([_event("hello")])

    await partition.forget_events([uuid4()])

    assert len(vector_store_partition.records) == 1
    assert fake_event_memory_store_partition.tombstones == {}


# ---------------------------------------------------------------------------
# Querying
# ---------------------------------------------------------------------------


async def test_a_query_deletes_the_unlinked_records_it_finds(
    partition: EventMemoryPartition,
    vector_store_partition: _FailingVectorStorePartition,
    fake_embedder: FakeEmbedder,
):
    await partition.encode_events([_event("hello")])
    unlinked_uuid = uuid4()
    [embedding] = await fake_embedder.ingest_embed(["hello"])
    await vector_store_partition.upsert(
        records=[
            Record(
                uuid=unlinked_uuid,
                vector=embedding,
                properties={EVENT_TIMESTAMP_KEY: _T0},
            )
        ]
    )

    result = await partition.query("hello")

    assert unlinked_uuid not in vector_store_partition.records
    assert len(result) == 1


async def test_a_failed_cleanup_of_unlinked_records_leaves_the_query_answered(
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
    vector_store_partition.fail_deletes = True

    result = await partition.query("hello")

    assert len(result) == 1


# ---------------------------------------------------------------------------
# Partition deletion
# ---------------------------------------------------------------------------


async def test_delete_partition_deletes_the_event_memory_store_partition_first(
    store_schema: VectorStoreSchema,
    fake_embedder,
):
    stores = create_autospec(object)
    event_memory_store = create_autospec(EventMemoryStore, instance=True)
    vector_store = create_autospec(VectorStore, instance=True)
    stores.attach_mock(event_memory_store.delete_partition, "event_memory_store_delete")
    stores.attach_mock(vector_store.delete_partition, "vector_store_delete")
    event_memory = EventMemory(
        EventMemoryParams(
            event_memory_store=event_memory_store,
            vector_store=vector_store,
            store_schema=store_schema,
            segmenter=Segmenter([TextSegmenter()]),
            deriver=Deriver([WholeTextDeriver()]),
            embedder=fake_embedder,
        )
    )

    await event_memory.delete_partition("partition")

    assert stores.mock_calls == [
        call.event_memory_store_delete("partition"),
        call.vector_store_delete("partition"),
    ]


# ---------------------------------------------------------------------------
# Background work, over a SQLite event memory store
# ---------------------------------------------------------------------------


@pytest_asyncio.fixture
async def event_memory_store(sqlalchemy_sqlite_engine: AsyncEngine):
    store = SQLAlchemyEventMemoryStore(
        SQLAlchemyEventMemoryStoreParams(engine=sqlalchemy_sqlite_engine)
    )
    await store.startup()
    yield store
    await store.shutdown()


@pytest_asyncio.fixture
async def event_memory(
    event_memory_store: SQLAlchemyEventMemoryStore,
    vector_store_partition: _FailingVectorStorePartition,
    store_schema: VectorStoreSchema,
    fake_embedder,
) -> EventMemory:
    vector_store = create_autospec(VectorStore, instance=True)
    vector_store.open_or_create_partition.return_value = vector_store_partition
    vector_store.open_partition.return_value = vector_store_partition
    return EventMemory(
        EventMemoryParams(
            event_memory_store=event_memory_store,
            vector_store=vector_store,
            store_schema=store_schema,
            segmenter=Segmenter([TextSegmenter()]),
            deriver=Deriver([WholeTextDeriver()]),
            embedder=fake_embedder,
        )
    )


def _database_time_ago(engine: AsyncEngine, delta: timedelta) -> ColumnElement:
    """The database clock's now less `delta`, in the form of its own stamps."""
    if engine.dialect.name == "sqlite":
        return func.datetime("now", f"-{int(delta.total_seconds())} seconds")
    return func.now() - delta


async def _age_pending_derivatives(engine: AsyncEngine, delta: timedelta) -> None:
    async with engine.begin() as connection:
        await connection.execute(
            update(PendingDerivativesRow).values(
                created_at=_database_time_ago(engine, delta)
            )
        )


async def _age_tombstones(engine: AsyncEngine, delta: timedelta) -> None:
    async with engine.begin() as connection:
        await connection.execute(
            update(TombstoneRow).values(enqueued_at=_database_time_ago(engine, delta))
        )


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
    partition = await event_memory.open_or_create_partition("partition")
    vector_store_partition.fail_upserts = True
    with pytest.raises(ConnectionError):
        await partition.encode_events([_event("hello")])
    vector_store_partition.fail_upserts = False

    assert not await event_memory._finish_pending_derivatives()
    await _age_pending_derivatives(sqlalchemy_sqlite_engine, timedelta(hours=1))
    assert await event_memory._finish_pending_derivatives()

    assert len(vector_store_partition.records) == 1
    assert not await event_memory._finish_pending_derivatives()
    result = await partition.query("hello")
    assert len(result) == 1


async def test_background_work_retries_pending_derivatives_after_a_failed_write(
    event_memory: EventMemory,
    vector_store_partition: _FailingVectorStorePartition,
    sqlalchemy_sqlite_engine: AsyncEngine,
):
    partition = await event_memory.open_or_create_partition("partition")
    vector_store_partition.fail_upserts = True
    with pytest.raises(ConnectionError):
        await partition.encode_events([_event("hello")])
    await _age_pending_derivatives(sqlalchemy_sqlite_engine, timedelta(hours=1))

    assert await event_memory._finish_pending_derivatives()
    assert vector_store_partition.records == {}
    # In its backoff now.
    assert not await event_memory._finish_pending_derivatives()


async def test_background_work_deletes_a_forgotten_events_vectors_twice(
    event_memory: EventMemory,
    vector_store_partition: _FailingVectorStorePartition,
    sqlalchemy_sqlite_engine: AsyncEngine,
):
    partition = await event_memory.open_or_create_partition("partition")
    event = _event("hello")
    await partition.encode_events([event])
    vector_store_partition.fail_deletes = True
    await partition.forget_events([event.uuid])
    vector_store_partition.fail_deletes = False
    assert len(vector_store_partition.records) == 1

    assert await event_memory._delete_tombstoned_derivatives()
    assert vector_store_partition.records == {}
    assert await _tombstone_count(sqlalchemy_sqlite_engine) == 1
    assert not await event_memory._delete_tombstoned_derivatives()

    await _age_tombstones(sqlalchemy_sqlite_engine, timedelta(days=2))
    assert await event_memory._delete_tombstoned_derivatives()
    assert await _tombstone_count(sqlalchemy_sqlite_engine) == 0


async def test_a_forget_ends_the_derivatives_pending_with_it(
    event_memory: EventMemory,
    vector_store_partition: _FailingVectorStorePartition,
    sqlalchemy_sqlite_engine: AsyncEngine,
):
    partition = await event_memory.open_or_create_partition("partition")
    vector_store_partition.fail_upserts = True
    event = _event("hello")
    with pytest.raises(ConnectionError):
        await partition.encode_events([event])
    vector_store_partition.fail_upserts = False

    await partition.forget_events([event.uuid])
    await _age_pending_derivatives(sqlalchemy_sqlite_engine, timedelta(hours=1))

    assert not await event_memory._finish_pending_derivatives()
    assert vector_store_partition.records == {}


@pytest_asyncio.fixture
async def evicting_event_memory(
    event_memory_store: SQLAlchemyEventMemoryStore,
    vector_store_partition: _FailingVectorStorePartition,
    store_schema: VectorStoreSchema,
    fake_embedder,
) -> EventMemory:
    """An EventMemory that keeps five derivatives of a near-duplicate cluster.

    The FakeEmbedder maps every text onto one direction, so every
    derivative is in one cluster.
    """
    vector_store = create_autospec(VectorStore, instance=True)
    vector_store.open_or_create_partition.return_value = vector_store_partition
    vector_store.open_partition.return_value = vector_store_partition
    return EventMemory(
        EventMemoryParams(
            event_memory_store=event_memory_store,
            vector_store=vector_store,
            store_schema=store_schema,
            segmenter=Segmenter([TextSegmenter()]),
            deriver=Deriver([WholeTextDeriver()]),
            embedder=fake_embedder,
            eviction=EvictionOptions(
                score_threshold=0.5, search_limit=100, target_size=5
            ),
        )
    )


def _stored_minutes(vector_store_partition: _FailingVectorStorePartition) -> list[int]:
    """The minute, after _T0, of every record in the vector store."""
    minutes = []
    for record in vector_store_partition.records.values():
        timestamp = record.properties[EVENT_TIMESTAMP_KEY]
        assert isinstance(timestamp, datetime.datetime)
        minutes.append(int((timestamp - _T0).total_seconds() // 60))
    return sorted(minutes)


async def _encode_one_per_minute(
    partition: EventMemoryPartition, minutes: Iterable[int]
) -> None:
    for minute in minutes:
        await partition.encode_events([_event(f"at {minute}", minutes=minute)])


async def test_background_work_displaces_what_the_stranded_encode_would_have(
    evicting_event_memory: EventMemory,
    vector_store_partition: _FailingVectorStorePartition,
    sqlalchemy_sqlite_engine: AsyncEngine,
):
    """Writing a stranded batch evicts as its encode would have, once its records are written."""
    partition = await evicting_event_memory.open_or_create_partition("partition")
    await _encode_one_per_minute(partition, range(5))
    vector_store_partition.fail_upserts = True
    with pytest.raises(ConnectionError):
        await partition.encode_events([_event("at 5", minutes=5)])
    vector_store_partition.fail_upserts = False
    assert _stored_minutes(vector_store_partition) == [0, 1, 2, 3, 4]

    await _age_pending_derivatives(sqlalchemy_sqlite_engine, timedelta(hours=1))
    assert await evicting_event_memory._finish_pending_derivatives()

    assert _stored_minutes(vector_store_partition) == [0, 1, 3, 4, 5]


async def test_background_work_does_not_count_the_stranded_writes_records_as_neighbors(
    evicting_event_memory: EventMemory,
    vector_store_partition: _FailingVectorStorePartition,
    sqlalchemy_sqlite_engine: AsyncEngine,
):
    """A record the stranded write stored is the derivative itself, not a near-duplicate of it."""
    partition = await evicting_event_memory.open_or_create_partition("partition")
    await _encode_one_per_minute(partition, range(4))
    vector_store_partition.fail_upserts_after_writing = True
    with pytest.raises(ConnectionError):
        await partition.encode_events([_event("at 4", minutes=4)])
    vector_store_partition.fail_upserts_after_writing = False
    assert _stored_minutes(vector_store_partition) == [0, 1, 2, 3, 4]

    await _age_pending_derivatives(sqlalchemy_sqlite_engine, timedelta(hours=1))
    assert await evicting_event_memory._finish_pending_derivatives()

    assert _stored_minutes(vector_store_partition) == [0, 1, 2, 3, 4]


async def test_background_work_skips_a_derivative_now_in_its_clusters_middle(
    evicting_event_memory: EventMemory,
    vector_store_partition: _FailingVectorStorePartition,
    sqlalchemy_sqlite_engine: AsyncEngine,
):
    """A stranded derivative that later ingests put in the temporal middle is not written, and the stranded write's record of it leaves."""
    partition = await evicting_event_memory.open_or_create_partition("partition")
    await _encode_one_per_minute(partition, [0, 1, 8, 9])
    vector_store_partition.fail_upserts_after_writing = True
    with pytest.raises(ConnectionError):
        await partition.encode_events([_event("at 5", minutes=5)])
    vector_store_partition.fail_upserts_after_writing = False
    # The later encode displaces the stranded record, but its delete fails.
    vector_store_partition.fail_deletes = True
    await partition.encode_events([_event("at 10", minutes=10)])
    vector_store_partition.fail_deletes = False
    assert _stored_minutes(vector_store_partition) == [0, 1, 5, 8, 9, 10]

    await _age_pending_derivatives(sqlalchemy_sqlite_engine, timedelta(hours=1))
    assert await evicting_event_memory._finish_pending_derivatives()

    assert _stored_minutes(vector_store_partition) == [0, 1, 8, 9, 10]
    assert not await evicting_event_memory._finish_pending_derivatives()


async def test_startup_runs_the_background_work_until_shutdown(
    event_memory: EventMemory,
    vector_store_partition: _FailingVectorStorePartition,
):
    partition = await event_memory.open_or_create_partition("partition")
    event = _event("hello")
    await partition.encode_events([event])
    vector_store_partition.fail_deletes = True
    await partition.forget_events([event.uuid])
    vector_store_partition.fail_deletes = False
    vector_store_partition.deleted.clear()

    await event_memory.startup()
    try:
        await asyncio.wait_for(vector_store_partition.deleted.wait(), timeout=10)
    finally:
        await event_memory.shutdown()

    assert vector_store_partition.records == {}
