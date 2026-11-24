"""Tests for SQLAlchemySegmentStore — SQLite (unit) and PostgreSQL (integration)."""

import asyncio
import contextlib
import functools
import json
import logging
import operator
import random
from collections.abc import AsyncIterator, Generator, Iterable, Mapping
from datetime import UTC, datetime, timedelta, timezone
from uuid import UUID, uuid4

import pytest
import pytest_asyncio
from pydantic import ValidationError
from sqlalchemy import (
    ColumnElement,
    Delete,
    Select,
    delete,
    event,
    func,
    insert,
    select,
    text,
    update,
)
from sqlalchemy.dialects import postgresql
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    create_async_engine,
)
from sqlalchemy.orm import InstrumentedAttribute
from sqlalchemy.pool import StaticPool

from memmachine_server.common.filter.filter_parser import (
    And,
    Comparison,
    FilterExpr,
    In,
    IsNull,
    Not,
    Or,
    parse_filter,
)
from memmachine_server.common.payload_codec.payload_codec_config import (
    PlaintextPayloadCodecConfig,
)
from memmachine_server.episodic_memory.event_memory.data_types import (
    Derivative,
    NullContext,
    ProducerContext,
    Segment,
    TextBlock,
)
from memmachine_server.episodic_memory.event_memory.segment_store import (
    IndexingJobClaim,
    SegmentStoreEventAlreadyExistsError,
    SegmentStorePartitionConfig,
    SegmentStorePartitionConfigMismatchError,
    TombstoneClaim,
    sqlalchemy_segment_store,
)
from memmachine_server.episodic_memory.event_memory.segment_store.sqlalchemy_segment_store import (
    BaseSegmentStore,
    DerivativeRow,
    EventRow,
    IndexingJobRow,
    PartitionRow,
    SegmentRow,
    SQLAlchemySegmentStore,
    SQLAlchemySegmentStoreParams,
    SQLAlchemySegmentStorePartition,
    TombstoneDerivativeRow,
    TombstoneRow,
)

PARTITION_UUID = UUID("00000000-0000-4000-8000-000000000001")
OTHER_PARTITION_UUID = UUID("00000000-0000-4000-8000-000000000002")
BASE_TIME = datetime(2024, 1, 1, tzinfo=UTC)
_NULL_CONTEXT = NullContext()

# Every table holding a partition's rows, the partition's own row aside.
_PARTITION_TABLES = (
    EventRow,
    SegmentRow,
    DerivativeRow,
    IndexingJobRow,
    TombstoneRow,
    TombstoneDerivativeRow,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _seg(
    *,
    event_uuid: UUID | None = None,
    index: int = 0,
    offset: int = 0,
    ts_offset_seconds: int = 0,
    text: str = "hello",
    context: ProducerContext | NullContext = _NULL_CONTEXT,
    properties: dict | None = None,
) -> Segment:
    return Segment(
        uuid=uuid4(),
        event_uuid=event_uuid or uuid4(),
        index=index,
        offset=offset,
        timestamp=BASE_TIME + timedelta(seconds=ts_offset_seconds),
        block=TextBlock(text=text),
        context=context,
        properties=properties or {},
    )


def _links(*segments: Segment) -> dict[Segment, list[UUID]]:
    """Build a segment-to-derivative-UUIDs mapping with one derivative per segment."""
    return {seg: [uuid4()] for seg in segments}


async def _add_segments(
    partition: SQLAlchemySegmentStorePartition,
    segments_to_derivative_uuids: Mapping[Segment, Iterable[UUID]],
) -> None:
    """Add the segments' events as a finished ingest leaves them: linked, nothing pending."""
    segments_by_event, derivatives = _events(segments_to_derivative_uuids)
    await partition.add_event_segments(segments_by_event, derivatives, lease_seconds=0)
    await partition.delete_indexing_jobs(
        segment.uuid for segments in segments_by_event.values() for segment in segments
    )


def _events(
    segments_to_derivative_uuids: Mapping[Segment, Iterable[UUID]],
) -> tuple[dict[UUID, list[Segment]], list[Derivative]]:
    """`add_event_segments` arguments for the segments' events, with a derivative per given UUID.

    Each derivative gets the timestamp and block of its segment.
    """
    segments_by_event: dict[UUID, list[Segment]] = {}
    derivatives: list[Derivative] = []
    for segment, derivative_uuids in segments_to_derivative_uuids.items():
        segments_by_event.setdefault(segment.event_uuid, []).append(segment)
        derivatives.extend(
            Derivative(
                uuid=derivative_uuid,
                segment_uuid=segment.uuid,
                timestamp=segment.timestamp,
                block=segment.block,
            )
            for derivative_uuid in derivative_uuids
        )
    return segments_by_event, derivatives


def _plaintext_partition_config() -> SegmentStorePartitionConfig:
    """Return the default plaintext partition config."""
    return SegmentStorePartitionConfig()


async def _created(
    store: SQLAlchemySegmentStore, partition_uuid: UUID = PARTITION_UUID
) -> SQLAlchemySegmentStorePartition:
    """Create a partition with the store's plaintext config and get a handle on it."""
    await store.create_partition(partition_uuid)
    partition = await store.get_partition(partition_uuid)
    assert partition is not None
    return partition


def _database_time_ago(engine: AsyncEngine, delta: timedelta) -> ColumnElement:
    """The database clock's now less `delta`, written by the database in the form of its own stamps."""
    if engine.dialect.name == "sqlite":
        return func.datetime("now", f"-{int(delta.total_seconds())} seconds")
    return func.now() - delta


async def _wait_until_blocked_or_done(
    engine: AsyncEngine,
    task: asyncio.Task,
) -> str:
    """Wait until `task` finishes ("done") or its backend waits on a lock ("blocked").

    Decided by observed database state (pg_stat_activity), not elapsed
    wall-clock time: with correct locking the task enters a lock wait
    within a few round trips, and with a lock ablated it finishes instead.
    """
    deadline = asyncio.get_running_loop().time() + 30
    while True:
        if task.done():
            return "done"
        async with engine.connect() as connection:
            blocked = (
                await connection.execute(
                    text(
                        "SELECT count(*) FROM pg_stat_activity "
                        "WHERE wait_event_type = 'Lock' "
                        "AND datname = current_database() "
                        "AND pid != pg_backend_pid()"
                    )
                )
            ).scalar_one()
        if blocked:
            return "blocked"
        if asyncio.get_running_loop().time() > deadline:
            raise TimeoutError("task neither blocked on a lock nor finished")
        await asyncio.sleep(0.01)


async def _row_counts(
    owner: SQLAlchemySegmentStore | SQLAlchemySegmentStorePartition,
    partition_uuid: UUID,
) -> tuple[int, int]:
    """The partition's segment and derivative row counts."""
    async with owner._create_session() as session:
        segments = (
            await session.execute(
                select(func.count())
                .select_from(SegmentRow)
                .where(SegmentRow.partition_uuid == partition_uuid)
            )
        ).scalar_one()
        links = (
            await session.execute(
                select(func.count())
                .select_from(DerivativeRow)
                .where(DerivativeRow.partition_uuid == partition_uuid)
            )
        ).scalar_one()
    return segments, links


async def _table_counts(
    owner: SQLAlchemySegmentStore | SQLAlchemySegmentStorePartition,
    partition_uuid: UUID,
) -> list[int]:
    """The partition's row count in each of `_PARTITION_TABLES`."""
    return [
        await _count_rows(owner, table.partition_uuid == partition_uuid)
        for table in _PARTITION_TABLES
    ]


async def _count_rows(
    owner: SQLAlchemySegmentStore | SQLAlchemySegmentStorePartition,
    condition: ColumnElement[bool],
) -> int:
    """How many rows of the table `condition` reads satisfy it."""
    async with owner._create_session() as session:
        return (
            await session.execute(select(func.count()).where(condition))
        ).scalar_one()


async def _tombstones(
    partition: SQLAlchemySegmentStorePartition,
) -> dict[UUID, list[UUID]]:
    """The partition's tombstones by segment UUID, each with its derivative UUIDs sorted."""
    async with partition._create_session() as session:
        tombstones: dict[UUID, list[UUID]] = {
            segment_uuid: []
            for segment_uuid in (
                await session.execute(
                    select(TombstoneRow.segment_uuid).where(
                        TombstoneRow.partition_uuid == partition._partition_uuid
                    )
                )
            ).scalars()
        }
        derivatives = await session.execute(
            select(
                TombstoneDerivativeRow.segment_uuid, TombstoneDerivativeRow.uuid
            ).where(TombstoneDerivativeRow.partition_uuid == partition._partition_uuid)
        )
        for segment_uuid, derivative_uuid in derivatives.all():
            tombstones[segment_uuid].append(derivative_uuid)
    return {
        segment_uuid: sorted(derivative_uuids)
        for segment_uuid, derivative_uuids in tombstones.items()
    }


@contextlib.contextmanager
def _failing_next_commit(engine: AsyncEngine) -> Generator[None, None, None]:
    """Make the engine's next transaction commit raise instead of committing."""
    failed: list[bool] = []

    def _fail_once(_connection) -> None:
        if not failed:
            failed.append(True)
            raise RuntimeError("injected commit failure")

    event.listen(engine.sync_engine, "commit", _fail_once)
    try:
        yield
    finally:
        event.remove(engine.sync_engine, "commit", _fail_once)
    assert failed, "no commit was attempted"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest_asyncio.fixture
async def sqlite_store(
    sqlalchemy_sqlite_engine: AsyncEngine,
) -> AsyncIterator[SQLAlchemySegmentStore]:
    store = SQLAlchemySegmentStore(
        SQLAlchemySegmentStoreParams(engine=sqlalchemy_sqlite_engine)
    )
    await store.startup()
    yield store
    async with sqlalchemy_sqlite_engine.begin() as conn:
        await conn.run_sync(BaseSegmentStore.metadata.drop_all)


@pytest_asyncio.fixture
async def pg_store(
    sqlalchemy_pg_engine: AsyncEngine,
) -> AsyncIterator[SQLAlchemySegmentStore]:
    store = SQLAlchemySegmentStore(
        SQLAlchemySegmentStoreParams(engine=sqlalchemy_pg_engine)
    )
    await store.startup()
    yield store
    async with sqlalchemy_pg_engine.begin() as conn:
        await conn.run_sync(BaseSegmentStore.metadata.drop_all)


@pytest.fixture(
    params=[
        "sqlite_store",
        pytest.param("pg_store", marks=pytest.mark.integration),
    ],
)
def store(request) -> SQLAlchemySegmentStore:
    return request.getfixturevalue(request.param)


@pytest_asyncio.fixture
async def partition(
    store: SQLAlchemySegmentStore,
) -> SQLAlchemySegmentStorePartition:
    return await _created(store)


# ===================================================================
# add_event_segments
# ===================================================================


@pytest.mark.asyncio
async def test_add_event_segments_and_get_contexts(
    partition: SQLAlchemySegmentStorePartition,
) -> None:
    seg = _seg(text="a")
    await _add_segments(partition, _links(seg))

    result = await partition.get_segment_contexts([seg.uuid])
    assert seg.uuid in result
    assert len(result[seg.uuid]) == 1
    assert result[seg.uuid][0].uuid == seg.uuid


@pytest.mark.asyncio
async def test_add_event_segments_with_properties(
    partition: SQLAlchemySegmentStorePartition,
) -> None:
    seg = _seg(properties={"color": "red", "score": 42})
    await _add_segments(partition, _links(seg))

    result = await partition.get_segment_contexts([seg.uuid])
    returned = result[seg.uuid][0]
    assert returned.properties == {"color": "red", "score": 42}


@pytest.mark.asyncio
async def test_add_event_segments_with_producer_context(
    partition: SQLAlchemySegmentStorePartition,
) -> None:
    ctx = ProducerContext(producer="User")
    seg = _seg(context=ctx)
    await _add_segments(partition, _links(seg))

    result = await partition.get_segment_contexts([seg.uuid])
    assert result[seg.uuid][0].context == ctx

    async with partition._create_session() as session:
        row = (
            await session.execute(select(SegmentRow).where(SegmentRow.uuid == seg.uuid))
        ).scalar_one()
    assert json.loads(row.context) == {"context_type": "producer", "producer": "User"}
    assert json.loads(row.block) == {"block_type": "text", "text": "hello"}


@pytest.mark.asyncio
async def test_add_event_segments_with_no_context(
    partition: SQLAlchemySegmentStorePartition,
) -> None:
    seg = _seg()
    await _add_segments(partition, _links(seg))

    async with partition._create_session() as session:
        row = (
            await session.execute(select(SegmentRow).where(SegmentRow.uuid == seg.uuid))
        ).scalar_one()
    assert json.loads(row.context) == {"context_type": "null"}

    result = await partition.get_segment_contexts([seg.uuid])
    assert result[seg.uuid][0].context == NullContext()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "tz",
    [
        UTC,
        timezone(timedelta(hours=-8)),
        timezone(timedelta(hours=5, minutes=30)),
    ],
)
async def test_timestamp_roundtrips_with_timezone(
    partition: SQLAlchemySegmentStorePartition,
    tz: timezone,
) -> None:
    """A timezone-aware timestamp roundtrips with its instant and offset intact.

    SQLite's DateTime(timezone=True) discards tzinfo and stores the wall-clock
    fields verbatim, so a non-UTC timestamp that is not normalized to UTC before
    writing comes back shifted by its offset. Regression test for that bug.
    """
    ts = datetime(2024, 1, 1, 13, 30, 45, tzinfo=tz)
    seg = Segment(
        uuid=uuid4(),
        event_uuid=uuid4(),
        index=0,
        offset=0,
        timestamp=ts,
        block=TextBlock(text="tz"),
        context=_NULL_CONTEXT,
        properties={},
    )
    await _add_segments(partition, _links(seg))

    result = await partition.get_segment_contexts([seg.uuid])
    returned = result[seg.uuid][0].timestamp
    # Aware-datetime equality compares absolute instants.
    assert returned == ts
    # The original UTC offset is reconstructed, not collapsed to UTC.
    assert returned.utcoffset() == ts.utcoffset()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "bound",
    [
        "2024-01-01T00:00:30+00:00",
        "2024-01-01T08:00:30+08:00",  # the same instant, named in another zone
        "2023-12-31T16:00:30-08:00",  # and another
    ],
)
async def test_timestamp_filter_compares_instants_not_wall_clocks(
    partition: SQLAlchemySegmentStorePartition,
    bound: str,
) -> None:
    """A datetime bound means an instant, whatever zone it is written in.

    Companion to the write-side normalization: timestamps are stored as the UTC
    instant, so a bound compared with its own offset would be compared on its
    wall-clock digits, and `<= 08:00+08:00` would exclude a row stored at 00:00Z.
    Both backends must agree, which is why this runs against SQLite and Postgres.
    """
    early = _seg(ts_offset_seconds=0)
    late = _seg(ts_offset_seconds=60)
    await _add_segments(partition, _links(early, late))

    result = await partition.get_segment_contexts(
        [early.uuid],
        max_forward_segments=5,
        property_filter=parse_filter(f"timestamp <= date('{bound}')"),
    )
    assert [s.uuid for s in result[early.uuid]] == [early.uuid]


@pytest.mark.asyncio
async def test_add_event_segments_empty(
    partition: SQLAlchemySegmentStorePartition,
) -> None:
    await partition.add_event_segments({}, [], lease_seconds=0)


@pytest.mark.asyncio
async def test_add_event_segments_rejects_a_batch_naming_a_held_event(
    partition: SQLAlchemySegmentStorePartition,
) -> None:
    """The error names every held event, and the batch stores nothing, not even its new event."""
    held = [_seg(), _seg()]
    await _add_segments(partition, _links(*held))
    new = _seg()
    batch = _links(
        _seg(event_uuid=held[0].event_uuid, index=1),
        new,
        _seg(event_uuid=held[1].event_uuid, index=1),
    )
    segments_by_event, derivatives = _events(batch)

    with pytest.raises(SegmentStoreEventAlreadyExistsError) as exc_info:
        await partition.add_event_segments(
            segments_by_event, derivatives, lease_seconds=0
        )

    assert exc_info.value.event_uuids == {segment.event_uuid for segment in held}
    assert await _row_counts(partition, partition._partition_uuid) == (2, 2)
    assert (
        await _count_rows(
            partition, IndexingJobRow.partition_uuid == partition._partition_uuid
        )
        == 0
    )
    assert set(
        await partition.get_segment_contexts(
            segment.uuid for segment in (*held, *batch)
        )
    ) == {segment.uuid for segment in held}
    await _add_segments(partition, _links(new))


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "mismatch",
    [
        "segment_listed_under_another_event",
        "derivative_of_a_segment_outside_the_batch",
    ],
)
async def test_add_event_segments_rejects_mismatched_event_naming(
    partition: SQLAlchemySegmentStorePartition,
    mismatch: str,
) -> None:
    """The error names the segment or derivative at fault."""
    first, second = _seg(), _seg()
    segments_by_event, derivatives = _events(_links(first, second))
    if mismatch == "segment_listed_under_another_event":
        segments_by_event[first.event_uuid].append(second)
        at_fault = second.uuid
    else:
        del segments_by_event[second.event_uuid]
        at_fault = derivatives[1].uuid

    with pytest.raises(ValueError, match=str(at_fault)):
        await partition.add_event_segments(
            segments_by_event, derivatives, lease_seconds=0
        )

    assert await _row_counts(partition, partition._partition_uuid) == (0, 0)
    assert (
        await _count_rows(
            partition, EventRow.partition_uuid == partition._partition_uuid
        )
        == 0
    )


@pytest.mark.asyncio
async def test_add_event_segments_failing_partway_writes_nothing(
    partition: SQLAlchemySegmentStorePartition,
) -> None:
    """An add whose link rows are rejected keeps none of its events or segments either."""
    taken = uuid4()
    await _add_segments(partition, {_seg(): [taken]})
    rejected = [_seg(ts_offset_seconds=1), _seg(ts_offset_seconds=2)]
    segments_by_event, derivatives = _events(
        {rejected[0]: [uuid4()], rejected[1]: [taken]}
    )

    with pytest.raises(IntegrityError):
        await partition.add_event_segments(
            segments_by_event, derivatives, lease_seconds=0
        )

    assert await _row_counts(partition, partition._partition_uuid) == (1, 1)
    assert (
        await _count_rows(
            partition, EventRow.partition_uuid == partition._partition_uuid
        )
        == 1
    )
    assert (
        await _count_rows(
            partition, IndexingJobRow.partition_uuid == partition._partition_uuid
        )
        == 0
    )
    assert (
        await partition.get_segment_contexts([segment.uuid for segment in rejected])
        == {}
    )


@pytest.mark.asyncio
async def test_concurrent_adds_of_one_event_store_it_once(
    partition: SQLAlchemySegmentStorePartition,
) -> None:
    """Of two racing adds of one event, one stores it and the other is rejected whole."""
    event_uuid = uuid4()
    batches = [
        _links(*(_seg(event_uuid=event_uuid, index=index) for index in range(3)))
        for _ in range(2)
    ]

    outcomes = await asyncio.gather(
        *(
            partition.add_event_segments(*_events(batch), lease_seconds=0)
            for batch in batches
        ),
        return_exceptions=True,
    )

    [error] = [outcome for outcome in outcomes if outcome is not None]
    assert isinstance(error, SegmentStoreEventAlreadyExistsError)
    assert error.event_uuids == {event_uuid}
    stored = await partition.get_segment_contexts(
        segment.uuid for batch in batches for segment in batch
    )
    assert set(stored) in [{segment.uuid for segment in batch} for batch in batches]
    assert await _row_counts(partition, partition._partition_uuid) == (3, 3)
    # One indexing job for each of the winner's segments.
    assert (
        await _count_rows(
            partition, IndexingJobRow.partition_uuid == partition._partition_uuid
        )
        == 3
    )


@pytest.mark.asyncio
async def test_indexing_jobs_roundtrip_through_a_claim(
    store: SQLAlchemySegmentStore,
) -> None:
    """A claim returns the indexing job as added, timestamp offsets included."""
    partition = await _created(store)
    event_uuid = uuid4()
    segments = [_seg(event_uuid=event_uuid, index=index) for index in range(2)]
    derivatives = [
        Derivative(
            uuid=uuid4(),
            segment_uuid=segments[0].uuid,
            timestamp=datetime(
                2024, 1, 1, 13, 30, 45, tzinfo=timezone(timedelta(hours=5, minutes=30))
            ),
            context=ProducerContext(producer="User"),
            block=TextBlock(text="first"),
            properties={
                "color": "red",
                "score": 42,
                "ratio": 0.5,
                "flag": True,
                "seen": datetime(2024, 2, 1, tzinfo=timezone(timedelta(hours=-8))),
            },
        ),
        Derivative(
            uuid=uuid4(),
            segment_uuid=segments[0].uuid,
            timestamp=datetime(2024, 1, 1, tzinfo=timezone(timedelta(hours=-8))),
            block=TextBlock(text="second"),
        ),
        Derivative(
            uuid=uuid4(),
            segment_uuid=segments[1].uuid,
            timestamp=BASE_TIME,
            block=TextBlock(text="third"),
        ),
    ]
    await partition.add_event_segments(
        {event_uuid: segments}, derivatives, lease_seconds=0
    )

    claims = [await store.claim_indexing_job(lease_seconds=60) for _ in range(2)]

    assert await store.claim_indexing_job(lease_seconds=60) is None
    claimed: list[Derivative] = []
    for claim in claims:
        assert claim is not None
        assert claim.partition_uuid == PARTITION_UUID
        assert claim.attempt == 1
        assert {derivative.segment_uuid for derivative in claim.derivatives} == {
            claim.segment_uuid
        }
        claimed.extend(claim.derivatives)
    assert {claim.segment_uuid for claim in claims if claim is not None} == {
        segment.uuid for segment in segments
    }
    by_uuid = operator.attrgetter("uuid")
    assert sorted(claimed, key=by_uuid) == sorted(derivatives, key=by_uuid)
    # Aware-datetime equality compares instants; the offsets come back too.
    assert {
        derivative.uuid: derivative.timestamp.utcoffset() for derivative in claimed
    } == {
        derivative.uuid: derivative.timestamp.utcoffset() for derivative in derivatives
    }


# ===================================================================
# delete_indexing_jobs
# ===================================================================


@pytest.mark.asyncio
async def test_delete_indexing_jobs_removes_only_the_pending_record(
    store: SQLAlchemySegmentStore,
) -> None:
    """The segment's derivatives stay and other segments stay pending. Idempotent."""
    partition = await _created(store)
    finished, unfinished = _seg(), _seg()
    segments_by_event, derivatives = _events(
        {finished: [uuid4(), uuid4()], unfinished: [uuid4()]}
    )
    await partition.add_event_segments(segments_by_event, derivatives, lease_seconds=0)

    for _ in range(2):
        await partition.delete_indexing_jobs([finished.uuid])

    assert await partition.get_segment_uuids_by_derivative_uuids(
        derivative.uuid for derivative in derivatives
    ) == {derivative.uuid: derivative.segment_uuid for derivative in derivatives}
    assert (
        await _count_rows(partition, IndexingJobRow.segment_uuid == finished.uuid) == 0
    )
    claim = await store.claim_indexing_job(lease_seconds=60)
    assert claim is not None
    assert claim.segment_uuid == unfinished.uuid
    assert await store.claim_indexing_job(lease_seconds=60) is None


@pytest.mark.integration
@pytest.mark.asyncio
async def test_delete_indexing_jobs_skips_a_record_another_transaction_holds(
    pg_store: SQLAlchemySegmentStore,
    sqlalchemy_pg_engine: AsyncEngine,
) -> None:
    """The delete never waits: a held record is left for the background work."""
    partition = await _created(pg_store)
    held, free = _seg(), _seg()
    await partition.add_event_segments(*_events(_links(held, free)), lease_seconds=0)

    async with (
        partition._create_session() as remote_session,
        remote_session.begin(),
    ):
        await remote_session.execute(
            select(IndexingJobRow)
            .where(IndexingJobRow.segment_uuid == held.uuid)
            .with_for_update()
        )
        deletion = asyncio.create_task(
            partition.delete_indexing_jobs([held.uuid, free.uuid])
        )
        assert await _wait_until_blocked_or_done(sqlalchemy_pg_engine, deletion) == (
            "done"
        )
        await deletion

    assert await _count_rows(partition, IndexingJobRow.segment_uuid == held.uuid) == 1
    assert await _count_rows(partition, IndexingJobRow.segment_uuid == free.uuid) == 0


# ===================================================================
# get_segment_contexts
# ===================================================================


@pytest.mark.asyncio
async def test_contexts_empty_seeds(
    partition: SQLAlchemySegmentStorePartition,
) -> None:
    result = await partition.get_segment_contexts([])
    assert result == {}


@pytest.mark.asyncio
async def test_contexts_unknown_seed(
    partition: SQLAlchemySegmentStorePartition,
) -> None:
    result = await partition.get_segment_contexts(
        [uuid4()], max_backward_segments=2, max_forward_segments=2
    )
    assert result == {}


@pytest.mark.asyncio
async def test_contexts_seed_only(
    partition: SQLAlchemySegmentStorePartition,
) -> None:
    """When max_backward=0 and max_forward=0, return just the seed."""
    seg = _seg()
    await _add_segments(partition, _links(seg))

    result = await partition.get_segment_contexts([seg.uuid])
    assert seg.uuid in result
    ctx = result[seg.uuid]
    assert len(ctx) == 1
    assert ctx[0].uuid == seg.uuid


@pytest.mark.asyncio
async def test_contexts_backward(
    partition: SQLAlchemySegmentStorePartition,
) -> None:
    ep = uuid4()
    segs = [_seg(event_uuid=ep, offset=i, ts_offset_seconds=i) for i in range(5)]
    await _add_segments(partition, _links(*segs))

    seed = segs[3]
    result = await partition.get_segment_contexts([seed.uuid], max_backward_segments=2)
    ctx = result[seed.uuid]
    # backward(2) + seed = 3 segments
    assert len(ctx) == 3
    uuids = [s.uuid for s in ctx]
    assert uuids == [segs[1].uuid, segs[2].uuid, seed.uuid]


@pytest.mark.asyncio
async def test_contexts_forward(
    partition: SQLAlchemySegmentStorePartition,
) -> None:
    ep = uuid4()
    segs = [_seg(event_uuid=ep, offset=i, ts_offset_seconds=i) for i in range(5)]
    await _add_segments(partition, _links(*segs))

    seed = segs[1]
    result = await partition.get_segment_contexts([seed.uuid], max_forward_segments=2)
    ctx = result[seed.uuid]
    # seed + forward(2) = 3 segments
    assert len(ctx) == 3
    uuids = [s.uuid for s in ctx]
    assert uuids == [seed.uuid, segs[2].uuid, segs[3].uuid]


@pytest.mark.asyncio
async def test_contexts_backward_and_forward(
    partition: SQLAlchemySegmentStorePartition,
) -> None:
    ep = uuid4()
    segs = [_seg(event_uuid=ep, offset=i, ts_offset_seconds=i) for i in range(7)]
    await _add_segments(partition, _links(*segs))

    seed = segs[3]
    result = await partition.get_segment_contexts(
        [seed.uuid], max_backward_segments=2, max_forward_segments=2
    )
    ctx = result[seed.uuid]
    assert len(ctx) == 5
    uuids = [s.uuid for s in ctx]
    assert uuids == [
        segs[1].uuid,
        segs[2].uuid,
        seed.uuid,
        segs[4].uuid,
        segs[5].uuid,
    ]


@pytest.mark.asyncio
async def test_contexts_clamp_at_boundaries(
    partition: SQLAlchemySegmentStorePartition,
) -> None:
    """Requesting more context than available returns what exists."""
    ep = uuid4()
    segs = [_seg(event_uuid=ep, offset=i, ts_offset_seconds=i) for i in range(3)]
    await _add_segments(partition, _links(*segs))

    seed = segs[0]
    result = await partition.get_segment_contexts(
        [seed.uuid], max_backward_segments=10, max_forward_segments=10
    )
    ctx = result[seed.uuid]
    assert len(ctx) == 3
    uuids = [s.uuid for s in ctx]
    assert uuids == [segs[0].uuid, segs[1].uuid, segs[2].uuid]


@pytest.mark.asyncio
async def test_contexts_multiple_seeds(
    partition: SQLAlchemySegmentStorePartition,
) -> None:
    ep = uuid4()
    segs = [_seg(event_uuid=ep, offset=i, ts_offset_seconds=i) for i in range(10)]
    await _add_segments(partition, _links(*segs))

    seed_a, seed_b = segs[2], segs[7]
    result = await partition.get_segment_contexts(
        [seed_a.uuid, seed_b.uuid],
        max_backward_segments=1,
        max_forward_segments=1,
    )
    assert seed_a.uuid in result
    assert seed_b.uuid in result
    ctx_a = [s.uuid for s in result[seed_a.uuid]]
    ctx_b = [s.uuid for s in result[seed_b.uuid]]
    assert ctx_a == [segs[1].uuid, seed_a.uuid, segs[3].uuid]
    assert ctx_b == [segs[6].uuid, seed_b.uuid, segs[8].uuid]


@pytest.mark.asyncio
async def test_contexts_with_properties(
    partition: SQLAlchemySegmentStorePartition,
) -> None:
    """Properties are loaded for seed and context segments."""
    ep = uuid4()
    s0 = _seg(event_uuid=ep, offset=0, ts_offset_seconds=0, properties={"k": "v0"})
    s1 = _seg(event_uuid=ep, offset=1, ts_offset_seconds=1, properties={"k": "v1"})
    s2 = _seg(event_uuid=ep, offset=2, ts_offset_seconds=2, properties={"k": "v2"})
    await _add_segments(partition, _links(s0, s1, s2))

    result = await partition.get_segment_contexts(
        [s1.uuid], max_backward_segments=1, max_forward_segments=1
    )
    ctx = result[s1.uuid]
    assert len(ctx) == 3
    assert ctx[0].properties == {"k": "v0"}
    assert ctx[1].properties == {"k": "v1"}
    assert ctx[2].properties == {"k": "v2"}


@pytest.mark.asyncio
async def test_contexts_property_filter(
    partition: SQLAlchemySegmentStorePartition,
) -> None:
    """Property filter excludes context rows that don't match."""
    ep = uuid4()
    s0 = _seg(event_uuid=ep, offset=0, ts_offset_seconds=0, properties={"tag": "a"})
    s1 = _seg(event_uuid=ep, offset=1, ts_offset_seconds=1, properties={"tag": "b"})
    s2 = _seg(event_uuid=ep, offset=2, ts_offset_seconds=2, properties={"tag": "a"})
    s3 = _seg(event_uuid=ep, offset=3, ts_offset_seconds=3, properties={"tag": "a"})
    await _add_segments(partition, _links(s0, s1, s2, s3))

    filt = Comparison(field="m.tag", op="=", value="a")
    result = await partition.get_segment_contexts(
        [s2.uuid],
        max_backward_segments=5,
        max_forward_segments=5,
        property_filter=filt,
    )
    ctx = result[s2.uuid]
    uuids = [s.uuid for s in ctx]
    # s1 excluded (tag=b); s0 backward, s2 seed, s3 forward
    assert uuids == [s0.uuid, s2.uuid, s3.uuid]


@pytest.mark.asyncio
async def test_context_comes_from_the_segments_nearest_the_seed(
    partition: SQLAlchemySegmentStorePartition,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A context read takes its context from a bounded number of segments on each side.

    With the bound at 4 segments, a match 5 or more segments from the seed
    is not context, however few matches lie within the bound, and an
    unfiltered read returns no segment beyond it either.
    """
    monkeypatch.setattr(sqlalchemy_segment_store, "_MAX_CONTEXT_DISTANCE", 4)
    tags = {1: "a", 2: "b", 3: "b", 4: "a", 5: "a", 6: "a"}
    seed = _seg(ts_offset_seconds=0, properties={"tag": "a"})
    backward = {
        d: _seg(ts_offset_seconds=-d, properties={"tag": t}) for d, t in tags.items()
    }
    forward = {
        d: _seg(ts_offset_seconds=d, properties={"tag": t}) for d, t in tags.items()
    }
    await _add_segments(partition, _links(seed, *backward.values(), *forward.values()))

    filtered = await partition.get_segment_contexts(
        [seed.uuid],
        max_backward_segments=5,
        max_forward_segments=5,
        property_filter=Comparison(field="m.tag", op="=", value="a"),
    )
    assert [segment.uuid for segment in filtered[seed.uuid]] == [
        backward[4].uuid,
        backward[1].uuid,
        seed.uuid,
        forward[1].uuid,
        forward[4].uuid,
    ]

    unfiltered = await partition.get_segment_contexts(
        [seed.uuid], max_backward_segments=6, max_forward_segments=6
    )
    assert [segment.uuid for segment in unfiltered[seed.uuid]] == [
        *(backward[distance].uuid for distance in (4, 3, 2, 1)),
        seed.uuid,
        *(forward[distance].uuid for distance in (1, 2, 3, 4)),
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize("field", ["context.producer", "context.context_type"])
async def test_a_filter_on_the_context_is_refused(
    partition: SQLAlchemySegmentStorePartition,
    field: str,
) -> None:
    """The context is no property, and a filter field naming no property is refused."""
    s0 = _seg(
        event_uuid=uuid4(),
        offset=0,
        ts_offset_seconds=0,
        context=ProducerContext(producer="Alice"),
    )
    await _add_segments(partition, _links(s0))

    filt = Comparison(field=field, op="=", value="Alice")
    with pytest.raises(ValueError, match="names no segment property"):
        await partition.get_segment_contexts(
            [s0.uuid],
            max_backward_segments=5,
            max_forward_segments=5,
            property_filter=filt,
        )


@pytest.mark.asyncio
async def test_contexts_session_isolation(store: SQLAlchemySegmentStore) -> None:
    """Context only includes segments from the same partition."""
    ep = uuid4()
    s_other = _seg(event_uuid=ep, offset=0, ts_offset_seconds=0)
    s_seed = _seg(event_uuid=ep, offset=1, ts_offset_seconds=1)
    s_after = _seg(event_uuid=ep, offset=2, ts_offset_seconds=2)

    other_partition = await _created(store, uuid4())
    await _add_segments(other_partition, _links(s_other))

    partition = await _created(store)
    await _add_segments(partition, _links(s_seed, s_after))

    result = await partition.get_segment_contexts(
        [s_seed.uuid], max_backward_segments=5, max_forward_segments=5
    )
    ctx = result[s_seed.uuid]
    uuids = [s.uuid for s in ctx]
    assert s_other.uuid not in uuids
    assert uuids == [s_seed.uuid, s_after.uuid]


@pytest.mark.asyncio
async def test_contexts_chronological_order(
    partition: SQLAlchemySegmentStorePartition,
) -> None:
    """Context segments are returned in chronological order."""
    ep = uuid4()
    segs = [_seg(event_uuid=ep, offset=i, ts_offset_seconds=i) for i in range(5)]
    await _add_segments(partition, _links(*segs))

    result = await partition.get_segment_contexts(
        [segs[2].uuid], max_backward_segments=10, max_forward_segments=10
    )
    ctx = result[segs[2].uuid]
    timestamps = [s.timestamp for s in ctx]
    assert timestamps == sorted(timestamps)


@pytest.mark.asyncio
async def test_context_preserved_in_segment_contexts(
    partition: SQLAlchemySegmentStorePartition,
) -> None:
    """Context is preserved when retrieving segment contexts (backward/forward)."""
    ep = uuid4()
    ctx_user = ProducerContext(producer="User")
    ctx_assistant = ProducerContext(producer="Assistant")
    s0 = _seg(event_uuid=ep, offset=0, ts_offset_seconds=0, context=ctx_user)
    s1 = _seg(event_uuid=ep, offset=1, ts_offset_seconds=1, context=ctx_assistant)
    s2 = _seg(event_uuid=ep, offset=2, ts_offset_seconds=2, context=ctx_user)
    await _add_segments(partition, _links(s0, s1, s2))

    result = await partition.get_segment_contexts(
        [s1.uuid], max_backward_segments=1, max_forward_segments=1
    )
    ctx = result[s1.uuid]
    assert len(ctx) == 3
    assert ctx[0].context == ctx_user
    assert ctx[1].context == ctx_assistant
    assert ctx[2].context == ctx_user


@pytest.mark.asyncio
@pytest.mark.parametrize("random_seed", [1, 2, 3])
async def test_random_context_reads_agree_with_a_model(
    store: SQLAlchemySegmentStore,
    random_seed: int,
) -> None:
    """Random context reads agree with an in-memory model of the timeline.

    Several seeds per read, filters of every node type, and many segments
    sharing a timestamp, with another partition's segments at the same
    times. The partition is far smaller than a filtered read's window, so
    every match on a side is within it. Events take random UUIDs in sorted
    order, so events sharing a timestamp come in the same order every run.
    """
    rng = random.Random(random_seed)
    partition = await _created(store)
    other_partition = await _created(store, uuid4())
    event_uuids = iter(sorted(uuid4() for _ in range(45)))
    segments = [
        segment
        for _ in range(30)
        for segment in _random_event_segments(rng, next(event_uuids))
    ]
    await _add_segments(partition, _links(*segments))
    await _add_segments(
        other_partition,
        _links(
            *(
                segment
                for _ in range(15)
                for segment in _random_event_segments(rng, next(event_uuids))
            )
        ),
    )
    timeline = sorted(
        segments,
        key=lambda segment: (
            segment.timestamp,
            segment.event_uuid,
            segment.index,
            segment.offset,
        ),
    )

    filters: list[FilterExpr | None] = [
        None,
        Comparison(field="m.tag", op="=", value="a"),
        In(field="m.tag", values=["a", "b"]),
        Not(Comparison(field="m.opt", op="=", value="x")),
        And(
            Comparison(field="m.tag", op="=", value="b"),
            Comparison(field="m.role", op="=", value="user"),
        ),
        Or(Comparison(field="m.tag", op="=", value="c"), IsNull(field="m.opt")),
        Comparison(field="timestamp", op=">=", value=BASE_TIME + timedelta(seconds=8)),
    ]
    for _ in range(40):
        seeds = rng.sample(segments, rng.randint(1, 5))
        max_backward_segments = rng.randint(0, 4)
        max_forward_segments = rng.randint(0, 4)
        property_filter = rng.choice(filters)

        result = await partition.get_segment_contexts(
            [seed.uuid for seed in seeds],
            max_backward_segments=max_backward_segments,
            max_forward_segments=max_forward_segments,
            property_filter=property_filter,
        )
        assert {
            seed_uuid: [segment.uuid for segment in context]
            for seed_uuid, context in result.items()
        } == _model_context_uuids(
            timeline,
            seeds,
            max_backward_segments=max_backward_segments,
            max_forward_segments=max_forward_segments,
            property_filter=property_filter,
        ), (property_filter, max_backward_segments, max_forward_segments)


def _random_event_segments(rng: random.Random, event_uuid: UUID) -> list[Segment]:
    """One event's segments at one random time, each with random properties.

    Every segment has `tag` and `role`; about half have `opt`.
    """
    ts_offset_seconds = rng.randrange(20)
    segments = []
    for index in range(rng.randint(1, 2)):
        for offset in range(rng.randint(1, 2)):
            properties = {
                "tag": rng.choice("abc"),
                "role": rng.choice(["user", "assistant"]),
            }
            if rng.random() < 0.5:
                properties["opt"] = rng.choice("xy")
            segments.append(
                _seg(
                    event_uuid=event_uuid,
                    index=index,
                    offset=offset,
                    ts_offset_seconds=ts_offset_seconds,
                    properties=properties,
                )
            )
    return segments


def _model_context_uuids(
    timeline: list[Segment],
    seeds: list[Segment],
    *,
    max_backward_segments: int,
    max_forward_segments: int,
    property_filter: FilterExpr | None,
) -> dict[UUID, list[UUID]]:
    """What get_segment_contexts returns for `seeds`, as segment UUIDs.

    `timeline` is the partition's segments in chronological order.
    """

    def matches(segment: Segment) -> bool:
        return property_filter is None or _sql_truth(segment, property_filter) is True

    contexts: dict[UUID, list[UUID]] = {}
    for seed in seeds:
        if not matches(seed):
            continue
        position = timeline.index(seed)
        before = [segment for segment in timeline[:position] if matches(segment)]
        after = [segment for segment in timeline[position + 1 :] if matches(segment)]
        context = [
            *(before[-max_backward_segments:] if max_backward_segments else []),
            seed,
            *after[:max_forward_segments],
        ]
        contexts[seed.uuid] = [segment.uuid for segment in context]
    return contexts


def _sql_truth(segment: Segment, expr: FilterExpr) -> bool | None:
    """`expr`'s value for `segment` in SQL's logic: None where SQL has NULL."""
    if isinstance(expr, And):
        left, right = _sql_truth(segment, expr.left), _sql_truth(segment, expr.right)
        if left is False or right is False:
            return False
        return None if left is None or right is None else True
    if isinstance(expr, Or):
        left, right = _sql_truth(segment, expr.left), _sql_truth(segment, expr.right)
        if left is True or right is True:
            return True
        return None if left is None or right is None else False
    if isinstance(expr, Not):
        inner = _sql_truth(segment, expr.expr)
        return None if inner is None else not inner
    assert isinstance(expr, Comparison | In | IsNull)
    return _sql_leaf_truth(segment, expr)


def _sql_leaf_truth(segment: Segment, expr: Comparison | In | IsNull) -> bool | None:
    """A leaf's value for `segment`: a missing property compares as NULL."""
    if expr.field == "timestamp":
        actual = segment.timestamp
    else:
        key = expr.field.removeprefix("m.")
        if isinstance(expr, IsNull):
            return key not in segment.properties
        if key not in segment.properties:
            return None
        actual = segment.properties[key]
    if isinstance(expr, In):
        return actual in expr.values
    assert isinstance(expr, Comparison)
    return {"=": operator.eq, ">=": operator.ge}[expr.op](actual, expr.value)


# ===================================================================
# get_segment_uuids_by_derivative_uuids
# ===================================================================


@pytest.mark.asyncio
async def test_get_segment_uuids_by_derivative_uuids(
    partition: SQLAlchemySegmentStorePartition,
) -> None:
    seg = _seg()
    d1, d2 = uuid4(), uuid4()
    await _add_segments(partition, {seg: [d1, d2]})

    result = await partition.get_segment_uuids_by_derivative_uuids([d1, d2])
    assert result == {d1: seg.uuid, d2: seg.uuid}


@pytest.mark.asyncio
async def test_get_segment_uuids_by_derivative_uuids_empty(
    partition: SQLAlchemySegmentStorePartition,
) -> None:
    result = await partition.get_segment_uuids_by_derivative_uuids([])
    assert result == {}


@pytest.mark.asyncio
async def test_get_segment_uuids_by_derivative_uuids_omits_unknown(
    partition: SQLAlchemySegmentStorePartition,
) -> None:
    """UUIDs the partition does not hold are omitted, not raised on."""
    seg = _seg()
    known = uuid4()
    await _add_segments(partition, {seg: [known]})

    unknown = uuid4()
    result = await partition.get_segment_uuids_by_derivative_uuids([known, unknown])
    assert result == {known: seg.uuid}


@pytest.mark.asyncio
async def test_get_segment_uuids_by_derivative_uuids_session_isolation(
    store: SQLAlchemySegmentStore,
) -> None:
    """A derivative another partition owns is invisible here."""
    other_partition = await _created(store, uuid4())
    other_seg = _seg()
    other_derivative = uuid4()
    await _add_segments(other_partition, {other_seg: [other_derivative]})

    partition = await _created(store)
    seg = _seg()
    derivative = uuid4()
    await _add_segments(partition, {seg: [derivative]})

    result = await partition.get_segment_uuids_by_derivative_uuids(
        [derivative, other_derivative]
    )
    assert result == {derivative: seg.uuid}


@pytest.mark.asyncio
async def test_get_segment_uuids_by_derivative_uuids_after_delete_event_segments(
    partition: SQLAlchemySegmentStorePartition,
) -> None:
    """Forgetting an event takes its derivatives' mapping with it."""
    kept, dropped = _seg(offset=0), _seg(offset=1)
    kept_derivative, dropped_derivative = uuid4(), uuid4()
    await _add_segments(
        partition, {kept: [kept_derivative], dropped: [dropped_derivative]}
    )

    await partition.delete_event_segments([dropped.event_uuid])

    result = await partition.get_segment_uuids_by_derivative_uuids(
        [kept_derivative, dropped_derivative]
    )
    assert result == {kept_derivative: kept.uuid}


# ===================================================================
# delete_event_segments
# ===================================================================


@pytest.mark.asyncio
async def test_delete_event_segments_removes_segments_and_derivatives(
    partition: SQLAlchemySegmentStorePartition,
) -> None:
    seg = _seg()
    derivative = uuid4()
    await _add_segments(partition, {seg: [derivative]})

    await partition.delete_event_segments([seg.event_uuid])

    assert await partition.get_segment_contexts([seg.uuid]) == {}
    assert await partition.get_segment_uuids_by_derivative_uuids([derivative]) == {}
    assert await _row_counts(partition, partition._partition_uuid) == (0, 0)
    assert (
        await _count_rows(
            partition, EventRow.partition_uuid == partition._partition_uuid
        )
        == 0
    )


@pytest.mark.asyncio
async def test_delete_event_segments_leaves_other_events_intact(
    partition: SQLAlchemySegmentStorePartition,
) -> None:
    forgotten_event, kept_event = uuid4(), uuid4()
    forgotten = [
        _seg(event_uuid=forgotten_event, offset=i, ts_offset_seconds=i)
        for i in range(2)
    ]
    kept = [
        _seg(event_uuid=kept_event, offset=i, ts_offset_seconds=i) for i in range(2)
    ]
    await _add_segments(partition, _links(*forgotten, *kept))

    await partition.delete_event_segments([forgotten_event])

    result = await partition.get_segment_contexts(
        segment.uuid for segment in (*forgotten, *kept)
    )
    assert set(result) == {segment.uuid for segment in kept}


@pytest.mark.asyncio
async def test_delete_event_segments_leaves_a_tombstone_per_forgotten_segment_naming_its_derivatives(
    partition: SQLAlchemySegmentStorePartition,
) -> None:
    """Each forgotten segment gets a tombstone naming its derivatives, if any, and no other segment does."""
    forgotten = [uuid4(), uuid4()]
    links = {
        _seg(event_uuid=forgotten[0], index=0): [uuid4(), uuid4()],
        _seg(event_uuid=forgotten[0], index=1): [uuid4()],
        _seg(event_uuid=forgotten[0], index=2): [],
        _seg(event_uuid=forgotten[1]): [uuid4()],
        _seg(): [uuid4()],
    }
    await _add_segments(partition, links)

    await partition.delete_event_segments([*forgotten, uuid4()])

    assert await _tombstones(partition) == {
        segment.uuid: sorted(derivative_uuids)
        for segment, derivative_uuids in links.items()
        if segment.event_uuid in forgotten
    }


@pytest.mark.asyncio
async def test_delete_event_segments_replaces_the_indexing_job_with_a_tombstone_holding_the_derivative_uuids(
    store: SQLAlchemySegmentStore,
) -> None:
    """The derivatives are deleted with their segment, and its tombstone holds their UUIDs."""
    partition = await _created(store)
    segment = _seg()
    segments_by_event, derivatives = _events({segment: [uuid4(), uuid4()]})
    await partition.add_event_segments(segments_by_event, derivatives, lease_seconds=0)
    derivative_uuids = [derivative.uuid for derivative in derivatives]

    await partition.delete_event_segments([segment.event_uuid])

    assert await _tombstones(partition) == {segment.uuid: sorted(derivative_uuids)}
    assert await _row_counts(partition, partition._partition_uuid) == (0, 0)
    assert (
        await _count_rows(
            partition,
            TombstoneRow.segment_uuid == segment.uuid,
        )
        == 1
    )
    assert (
        await _count_rows(
            partition,
            TombstoneDerivativeRow.segment_uuid == segment.uuid,
        )
        == 2
    )
    assert await store.claim_indexing_job(lease_seconds=60) is None
    assert await partition.get_segment_uuids_by_derivative_uuids(derivative_uuids) == {}
    claim = await store.claim_tombstone(lease_seconds=60)
    assert claim is not None
    assert sorted(claim.derivative_uuids) == sorted(derivative_uuids)


@pytest.mark.asyncio
async def test_delete_event_segments_skips_unknown_events_and_forgets_events_without_derivatives(
    partition: SQLAlchemySegmentStorePartition,
) -> None:
    """A segment without derivatives gets a tombstone naming none; an event without segments gets none."""
    bare = _seg()
    await _add_segments(partition, {bare: []})
    empty_event_uuid = uuid4()
    await partition.add_event_segments({empty_event_uuid: []}, [], lease_seconds=0)

    await partition.delete_event_segments([])
    await partition.delete_event_segments([uuid4()])
    assert await _tombstones(partition) == {}
    await partition.delete_event_segments([bare.event_uuid])
    await partition.delete_event_segments([empty_event_uuid])
    assert await _tombstones(partition) == {bare.uuid: []}

    assert await partition.get_segment_contexts([bare.uuid]) == {}
    assert (
        await _count_rows(
            partition, EventRow.partition_uuid == partition._partition_uuid
        )
        == 0
    )
    assert (
        await _count_rows(
            partition,
            TombstoneRow.partition_uuid == partition._partition_uuid,
        )
        == 1
    )


@pytest.mark.asyncio
async def test_delete_event_segments_is_idempotent(
    partition: SQLAlchemySegmentStorePartition,
) -> None:
    segment = _seg()
    links = _links(segment)
    await _add_segments(partition, links)

    await partition.delete_event_segments([segment.event_uuid])
    await partition.delete_event_segments([segment.event_uuid])

    assert await _tombstones(partition) == {segment.uuid: links[segment]}


@pytest.mark.asyncio
async def test_forgotten_event_can_be_added_again(
    partition: SQLAlchemySegmentStorePartition,
) -> None:
    """A changed event is forgotten and added again under its UUID."""
    event_uuid = uuid4()
    await _add_segments(partition, _links(_seg(event_uuid=event_uuid)))
    await partition.delete_event_segments([event_uuid])
    changed = _seg(event_uuid=event_uuid, text="changed")

    await _add_segments(partition, _links(changed))

    contexts = await partition.get_segment_contexts([changed.uuid])
    assert contexts[changed.uuid][0].block == TextBlock(text="changed")


@pytest.mark.asyncio
async def test_a_tombstone_is_refused_unless_scheduled_until_its_first_delete(
    store: SQLAlchemySegmentStore,
) -> None:
    async with store._create_session() as session:
        with pytest.raises(
            IntegrityError, match="segment_store_tb__scheduled_until_first_delete"
        ):
            async with session.begin():
                await session.execute(
                    insert(TombstoneRow).values(
                        partition_uuid=uuid4(), segment_uuid=uuid4(), claimable_at=None
                    )
                )


@pytest.mark.asyncio
async def test_an_ingests_own_indexing_job_delete_leaves_the_tombstone_a_forget_made_since(
    store: SQLAlchemySegmentStore,
) -> None:
    """The ingest's delete of its indexing jobs leaves the forget's tombstone to its deletes."""
    partition = await _created(store)
    segment = _seg()
    segments_by_event, derivatives = _events(_links(segment))
    await partition.add_event_segments(
        segments_by_event, derivatives, lease_seconds=3600
    )
    await partition.delete_event_segments([segment.event_uuid])

    await partition.delete_indexing_jobs([segment.uuid])

    claim = await store.claim_tombstone(lease_seconds=60)
    assert claim is not None
    assert (claim.segment_uuid, claim.derivative_uuids) == (
        segment.uuid,
        [derivatives[0].uuid],
    )


@pytest.mark.asyncio
async def test_a_stale_indexing_job_claim_leaves_a_tombstone_at_its_generation_alone(
    store: SQLAlchemySegmentStore,
) -> None:
    """A tombstone written after its indexing job was deleted starts its generations over, and a claim on the indexing job still reaches none of it."""
    partition = await _created(store)
    segment = _seg()
    await partition.add_event_segments(*_events(_links(segment)), lease_seconds=0)
    stale = await store.claim_indexing_job(lease_seconds=60)
    assert stale is not None
    await partition.delete_indexing_jobs([segment.uuid])
    await partition.delete_event_segments([segment.event_uuid])
    await _finish_first_delete(store)

    assert not await stale.renew(lease_seconds=60)
    await stale.fail(RuntimeError("index unavailable"))
    await stale.finish()

    await _backdate(store, TombstoneRow.first_deleted_at, timedelta(days=2))
    second = await store.claim_tombstone(lease_seconds=60)
    assert second is not None
    assert (second.segment_uuid, second.second_delete, second.attempt) == (
        segment.uuid,
        True,
        1,
    )


@pytest.mark.asyncio
async def test_a_segment_without_derivatives_has_the_lifecycle_of_any_other(
    store: SQLAlchemySegmentStore,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Its indexing job is claimed with no derivatives, and its tombstone names none and is deleted by its second delete."""
    monkeypatch.setattr(store, "_tombstone_retention_seconds", 3600)
    partition = await _created(store)
    segment = _seg()
    await partition.add_event_segments(*_events({segment: []}), lease_seconds=0)

    pending = await store.claim_indexing_job(lease_seconds=60)
    assert pending is not None
    assert (pending.segment_uuid, pending.derivatives) == (segment.uuid, [])

    await partition.delete_event_segments([segment.event_uuid])
    first = await store.claim_tombstone(lease_seconds=60)
    assert first is not None
    assert (first.segment_uuid, first.derivative_uuids) == (segment.uuid, [])
    await first.finish()
    await _backdate(store, TombstoneRow.first_deleted_at, timedelta(hours=2))
    second = await store.claim_tombstone(lease_seconds=60)
    assert second is not None
    assert second.second_delete is True
    await second.finish()

    assert await _count_rows(store, TombstoneRow.segment_uuid == segment.uuid) == 0


@pytest.mark.asyncio
async def test_claim_on_an_indexing_job_of_a_forgotten_event_ends_without_effect(
    store: SQLAlchemySegmentStore,
) -> None:
    """The forget deleted the claim's record by cascade, and the tombstone names the claimed derivatives."""
    partition = await _created(store)
    segment = _seg()
    segments_by_event, derivatives = _events(_links(segment))
    await partition.add_event_segments(segments_by_event, derivatives, lease_seconds=0)
    claim = await store.claim_indexing_job(lease_seconds=60)
    assert claim is not None

    await partition.delete_event_segments([segment.event_uuid])
    assert not await claim.renew(lease_seconds=60)
    await claim.finish()
    await claim.fail(RuntimeError("index unavailable"))

    assert (
        await _count_rows(
            partition,
            IndexingJobRow.partition_uuid == partition._partition_uuid,
        )
        == 0
    )
    tombstone_claim = await store.claim_tombstone(lease_seconds=60)
    assert tombstone_claim is not None
    assert tombstone_claim.derivative_uuids == [
        derivative.uuid for derivative in claim.derivatives
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize("operation", ["add_event_segments", "delete_event_segments"])
async def test_event_write_whose_commit_fails_changes_nothing(
    store: SQLAlchemySegmentStore,
    operation: str,
) -> None:
    """An event write whose commit fails leaves the partition as it was."""
    partition = await _created(store, uuid4())
    kept = _seg()
    await _add_segments(partition, _links(kept))
    write = (
        partition.add_event_segments(
            *_events(_links(_seg(ts_offset_seconds=1))), lease_seconds=0
        )
        if operation == "add_event_segments"
        else partition.delete_event_segments([kept.event_uuid])
    )

    with (
        _failing_next_commit(store._engine),
        pytest.raises(RuntimeError, match="injected commit failure"),
    ):
        await write

    assert await _row_counts(store, partition._partition_uuid) == (1, 1)
    for table, count in (
        (EventRow, 1),
        (IndexingJobRow, 0),
        (TombstoneRow, 0),
    ):
        assert (
            await _count_rows(store, table.partition_uuid == partition._partition_uuid)
            == count
        )
    assert kept.uuid in await partition.get_segment_contexts([kept.uuid])


# ===================================================================
# Claims
# ===================================================================

_CLAIMED_TABLES = pytest.mark.parametrize(
    "table",
    [IndexingJobRow, TombstoneRow],
    ids=["indexing_job", "tombstone"],
)


@pytest.mark.asyncio
async def test_an_indexing_job_is_claimable_once_its_ingests_lease_has_passed(
    store: SQLAlchemySegmentStore,
) -> None:
    partition = await _created(store)
    segments_by_event, derivatives = _events(_links(_seg()))
    await partition.add_event_segments(
        segments_by_event, derivatives, lease_seconds=3600
    )

    assert await store.claim_indexing_job(lease_seconds=60) is None
    await _shift_earlier(store, IndexingJobRow.claimable_at, timedelta(minutes=50))
    assert await store.claim_indexing_job(lease_seconds=60) is None
    await _shift_earlier(store, IndexingJobRow.claimable_at, timedelta(minutes=20))
    claim = await store.claim_indexing_job(lease_seconds=60)
    assert _claimed_record_uuid(claim) == derivatives[0].segment_uuid


@pytest.mark.asyncio
@_CLAIMED_TABLES
async def test_claims_take_the_oldest_record_first(
    store: SQLAlchemySegmentStore,
    table: type[IndexingJobRow | TombstoneRow],
) -> None:
    partition = await _created(store)
    newer = await _add_claimable_record(partition, table)
    older = await _add_claimable_record(partition, table)
    await _backdate(
        store,
        table.claimable_at,
        timedelta(minutes=5),
        table.segment_uuid == older,
    )

    claimed = [_claimed_record_uuid(await _claim(store, table)) for _ in range(2)]
    assert claimed == [older, newer]


@pytest.mark.asyncio
@_CLAIMED_TABLES
async def test_claim_holds_its_record_for_the_lease(
    store: SQLAlchemySegmentStore,
    table: type[IndexingJobRow | TombstoneRow],
) -> None:
    """A record is claimed again only once its claim's lease is over, as a further attempt."""
    partition = await _created(store)
    record_uuid = await _add_claimable_record(partition, table)

    first = await _claim(store, table)
    assert first is not None
    assert (_claimed_record_uuid(first), first.attempt) == (record_uuid, 1)
    assert await _claim(store, table) is None

    await _backdate(store, table.claimable_at, timedelta(days=1))
    second = await _claim(store, table)
    assert second is not None
    assert (_claimed_record_uuid(second), second.attempt) == (record_uuid, 2)


@pytest.mark.asyncio
async def test_finishing_a_superseded_indexing_job_claim_deletes_the_record(
    store: SQLAlchemySegmentStore,
) -> None:
    """A claim that outlasted its lease wrote the derivatives, so its finish deletes the record, and the later claim's renewal finds it gone."""
    partition = await _created(store)
    pending_uuid = await _add_claimable_record(partition, IndexingJobRow)
    superseded = await store.claim_indexing_job(lease_seconds=60)
    await _backdate(store, IndexingJobRow.claimable_at, timedelta(days=1))
    latest = await store.claim_indexing_job(lease_seconds=60)
    assert superseded is not None
    assert latest is not None

    await superseded.finish()
    assert await _count_rows(store, IndexingJobRow.segment_uuid == pending_uuid) == 0
    assert not await latest.renew(lease_seconds=60)
    await latest.finish()
    # The segment's derivatives stay with it.
    assert await _count_rows(store, DerivativeRow.segment_uuid == pending_uuid) == 1


@pytest.mark.asyncio
@_CLAIMED_TABLES
async def test_failing_a_superseded_claim_changes_nothing(
    store: SQLAlchemySegmentStore,
    table: type[IndexingJobRow | TombstoneRow],
) -> None:
    """It neither ends the later claim nor starts a backoff."""
    partition = await _created(store)
    record_uuid = await _add_claimable_record(partition, table)
    superseded = await _claim(store, table)
    await _backdate(store, table.claimable_at, timedelta(days=1))
    assert await _claim(store, table) is not None
    assert superseded is not None
    scheduled = await _claimable_at(store, table, record_uuid)

    await _fail_claim(superseded)

    assert await _claimable_at(store, table, record_uuid) == scheduled


@pytest.mark.asyncio
@_CLAIMED_TABLES
async def test_failed_claim_backs_off_doubling_up_to_the_cap(
    store: SQLAlchemySegmentStore,
    monkeypatch: pytest.MonkeyPatch,
    table: type[IndexingJobRow | TombstoneRow],
) -> None:
    """Each failed claim in a row doubles the wait before the next, up to the cap."""
    monkeypatch.setattr(store, "_base_claim_retry_backoff_seconds", 60)
    monkeypatch.setattr(store, "_max_claim_retry_backoff_seconds", 150)
    partition = await _created(store)
    await _add_claimable_record(partition, table)

    for attempt, backoff_seconds in enumerate([60, 120, 150], start=1):
        claim = await _claim(store, table)
        assert claim is not None
        assert claim.attempt == attempt
        await _fail_claim(claim)
        assert await _claim(store, table) is None
        await _shift_earlier(
            store, table.claimable_at, timedelta(seconds=backoff_seconds - 15)
        )
        assert await _claim(store, table) is None
        await _shift_earlier(store, table.claimable_at, timedelta(seconds=30))
    claim = await _claim(store, table)
    assert claim is not None
    assert claim.attempt == 4


@pytest.mark.asyncio
@_CLAIMED_TABLES
async def test_a_failed_claim_after_an_operator_reset_its_attempts_backs_off_the_base_wait(
    store: SQLAlchemySegmentStore,
    monkeypatch: pytest.MonkeyPatch,
    table: type[IndexingJobRow | TombstoneRow],
) -> None:
    """An operator may reset a record's attempts while a claim on it is open; the claim's failure still waits the base backoff."""
    monkeypatch.setattr(store, "_base_claim_retry_backoff_seconds", 60)
    partition = await _created(store)
    await _add_claimable_record(partition, table)
    claim = await _claim(store, table)
    assert claim is not None
    await _reset_attempts(store, table)

    await _fail_claim(claim)

    assert await _claim(store, table) is None
    await _shift_earlier(store, table.claimable_at, timedelta(seconds=45))
    assert await _claim(store, table) is None
    await _shift_earlier(store, table.claimable_at, timedelta(seconds=30))
    assert await _claim(store, table) is not None


@pytest.mark.asyncio
async def test_renewing_a_claim_after_an_operator_reset_its_attempts_holds_it(
    store: SQLAlchemySegmentStore,
) -> None:
    partition = await _created(store)
    await _add_claimable_record(partition, IndexingJobRow)
    claim = await store.claim_indexing_job(lease_seconds=60)
    assert claim is not None
    await _reset_attempts(store, IndexingJobRow)

    assert await claim.renew(lease_seconds=60)


async def _reset_attempts(
    store: SQLAlchemySegmentStore,
    table: type[IndexingJobRow | TombstoneRow],
) -> None:
    """Set every record's attempts to 0, as an operator retrying a record does."""
    async with store._create_session() as session, session.begin():
        await session.execute(update(table).values(attempts_without_progress=0))


@pytest.mark.asyncio
@_CLAIMED_TABLES
async def test_record_is_no_longer_claimed_after_the_maximum_failed_claims(
    store: SQLAlchemySegmentStore,
    caplog: pytest.LogCaptureFixture,
    table: type[IndexingJobRow | TombstoneRow],
) -> None:
    """The store reports the record once its claims failed the maximum number of times in a row."""
    partition = await _created(store)
    await _add_claimable_record(partition, table)
    maximum = sqlalchemy_segment_store._MAX_CLAIM_ATTEMPTS_WITHOUT_PROGRESS

    with caplog.at_level(logging.ERROR, logger=sqlalchemy_segment_store.__name__):
        for attempt in range(1, maximum + 1):
            claim = await _claim(store, table)
            assert claim is not None
            assert claim.attempt == attempt
            await _fail_claim(claim)
            await _backdate(store, table.claimable_at, timedelta(days=1))

    assert await _claim(store, table) is None
    reports = [
        record for record in caplog.records if "no longer claimed" in record.message
    ]
    assert [record.levelno for record in reports] == [logging.ERROR]


@pytest.mark.asyncio
async def test_renewing_a_claim_holds_its_record_for_another_lease(
    store: SQLAlchemySegmentStore,
) -> None:
    partition = await _created(store)
    await partition.add_event_segments(*_events(_links(_seg())), lease_seconds=0)
    claim = await store.claim_indexing_job(lease_seconds=60)
    assert claim is not None
    await _backdate(store, IndexingJobRow.claimable_at, timedelta(days=1))

    assert await claim.renew(lease_seconds=60)

    # The renewed lease holds the record again, and then the backoff.
    assert await store.claim_indexing_job(lease_seconds=60) is None
    await _shift_earlier(store, IndexingJobRow.claimable_at, timedelta(seconds=80))
    assert await store.claim_indexing_job(lease_seconds=60) is None
    await _shift_earlier(store, IndexingJobRow.claimable_at, timedelta(seconds=20))
    assert await store.claim_indexing_job(lease_seconds=60) is not None


@pytest.mark.asyncio
async def test_a_superseded_claim_is_not_renewed(
    store: SQLAlchemySegmentStore,
) -> None:
    partition = await _created(store)
    await partition.add_event_segments(*_events(_links(_seg())), lease_seconds=0)
    superseded = await store.claim_indexing_job(lease_seconds=60)
    assert superseded is not None
    await _backdate(store, IndexingJobRow.claimable_at, timedelta(days=1))
    latest = await store.claim_indexing_job(lease_seconds=60)
    assert latest is not None

    assert not await superseded.renew(lease_seconds=60)
    assert await latest.renew(lease_seconds=60)


@pytest.mark.asyncio
async def test_a_claim_on_a_forgotten_event_is_not_renewed(
    store: SQLAlchemySegmentStore,
) -> None:
    partition = await _created(store)
    segment = _seg()
    await partition.add_event_segments(*_events(_links(segment)), lease_seconds=0)
    claim = await store.claim_indexing_job(lease_seconds=60)
    assert claim is not None

    await partition.delete_event_segments([segment.event_uuid])

    assert not await claim.renew(lease_seconds=60)


# ===================================================================
# Tombstones
# ===================================================================


@pytest.mark.asyncio
async def test_new_tombstone_is_claimable_at_once(
    store: SQLAlchemySegmentStore,
) -> None:
    partition = await _created(store)
    segment = _seg()
    derivative_uuids = [uuid4(), uuid4()]
    await _add_segments(partition, {segment: derivative_uuids})
    await partition.delete_event_segments([segment.event_uuid])

    claim = await store.claim_tombstone(lease_seconds=60)

    assert claim is not None
    assert claim.partition_uuid == PARTITION_UUID
    assert claim.segment_uuid == segment.uuid
    assert sorted(claim.derivative_uuids) == sorted(derivative_uuids)
    assert claim.second_delete is False
    assert claim.attempt == 1


@pytest.mark.asyncio
async def test_tombstone_awaits_the_retention_after_its_first_delete(
    store: SQLAlchemySegmentStore,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Its second delete falls due once the retention has passed since its first delete."""
    monkeypatch.setattr(store, "_tombstone_retention_seconds", 3600)
    partition = await _created(store)
    tombstone_uuid = await _add_claimable_record(partition, TombstoneRow)
    await _finish_first_delete(store)

    for age in (timedelta(0), timedelta(minutes=50)):
        await _backdate(store, TombstoneRow.first_deleted_at, age)
        assert await store.claim_tombstone(lease_seconds=60) is None
    await _backdate(store, TombstoneRow.first_deleted_at, timedelta(minutes=70))
    second = await store.claim_tombstone(lease_seconds=60)
    assert second is not None
    assert (second.segment_uuid, second.second_delete, second.attempt) == (
        tombstone_uuid,
        True,
        1,
    )


@pytest.mark.asyncio
async def test_finishing_a_second_delete_claim_removes_the_tombstone(
    store: SQLAlchemySegmentStore,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(store, "_tombstone_retention_seconds", 3600)
    partition = await _created(store)
    tombstone_uuid = await _add_claimable_record(partition, TombstoneRow)
    await _finish_first_delete(store)
    await _backdate(store, TombstoneRow.first_deleted_at, timedelta(hours=2))
    claim = await store.claim_tombstone(lease_seconds=60)
    assert claim is not None
    assert claim.second_delete is True
    holds = TombstoneDerivativeRow.segment_uuid == tombstone_uuid
    assert await _count_rows(store, holds) == 1

    await claim.finish()

    assert await _count_rows(store, TombstoneRow.segment_uuid == tombstone_uuid) == 0
    assert await _count_rows(store, holds) == 0
    assert await store.claim_tombstone(lease_seconds=60) is None


async def _claim_second_delete(
    store: SQLAlchemySegmentStore,
    partition: SQLAlchemySegmentStorePartition,
) -> tuple[UUID, TombstoneClaim]:
    """Leave a tombstone due for its second delete, claim that delete, and return its segment's UUID and the claim."""
    tombstone_uuid = await _add_claimable_record(partition, TombstoneRow)
    await _finish_first_delete(store)
    await _backdate(store, TombstoneRow.first_deleted_at, timedelta(days=2))
    claim = await store.claim_tombstone(lease_seconds=60)
    assert claim is not None
    assert claim.second_delete
    return tombstone_uuid, claim


async def _finish_first_delete(store: SQLAlchemySegmentStore) -> None:
    """Claim the oldest tombstone due for a delete, which must be its first, and finish it."""
    claim = await store.claim_tombstone(lease_seconds=60)
    assert claim is not None
    assert not claim.second_delete
    await claim.finish()


@pytest.mark.asyncio
async def test_no_schedule_brings_a_second_delete_before_the_retention(
    store: SQLAlchemySegmentStore,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The retention is its own condition: a tombstone scheduled as claimable
    still waits for the retention since its first delete."""
    monkeypatch.setattr(store, "_tombstone_retention_seconds", 3600)
    partition = await _created(store)
    await _add_claimable_record(partition, TombstoneRow)
    await _finish_first_delete(store)
    await _backdate(store, TombstoneRow.claimable_at, timedelta(days=1))

    await _backdate(store, TombstoneRow.first_deleted_at, timedelta(minutes=50))
    assert await store.claim_tombstone(lease_seconds=60) is None
    await _backdate(store, TombstoneRow.first_deleted_at, timedelta(minutes=70))
    second = await store.claim_tombstone(lease_seconds=60)
    assert second is not None
    assert second.second_delete is True


@pytest.mark.asyncio
@_CLAIMED_TABLES
async def test_a_claim_that_never_ends_is_claimed_again_once_its_lease_and_the_backoff_pass(
    store: SQLAlchemySegmentStore,
    table: type[IndexingJobRow | TombstoneRow],
) -> None:
    partition = await _created(store)
    await _add_claimable_record(partition, table)
    lease = timedelta(seconds=60)
    assert await _claim(store, table) is not None

    # Past the lease, but not the 30-second backoff after it.
    await _shift_earlier(store, table.claimable_at, lease + timedelta(seconds=20))
    assert await _claim(store, table) is None
    await _shift_earlier(store, table.claimable_at, timedelta(seconds=20))
    retry = await _claim(store, table)
    assert retry is not None
    assert retry.attempt == 2


@pytest.mark.asyncio
async def test_claim_timings_come_from_params(
    sqlalchemy_sqlite_engine: AsyncEngine,
) -> None:
    """The configured retry backoff and tombstone retention govern claims."""
    store = SQLAlchemySegmentStore(
        SQLAlchemySegmentStoreParams(
            engine=sqlalchemy_sqlite_engine,
            tombstone_retention_seconds=60,
            base_claim_retry_backoff_seconds=10,
            max_claim_retry_backoff_seconds=10,
        )
    )
    await store.startup()
    try:
        partition = await _created(store, uuid4())
        await _add_claimable_record(partition, TombstoneRow)
        failed = await store.claim_tombstone(lease_seconds=60)
        assert failed is not None
        await _fail_claim(failed)
        # Past the configured backoff, not the default's.
        await _shift_earlier(store, TombstoneRow.claimable_at, timedelta(seconds=20))
        first = await store.claim_tombstone(lease_seconds=60)
        assert first is not None
        await first.finish()

        await _backdate(store, TombstoneRow.first_deleted_at, timedelta(minutes=2))
        second = await store.claim_tombstone(lease_seconds=60)
        assert second is not None
        assert second.second_delete is True
    finally:
        async with sqlalchemy_sqlite_engine.begin() as conn:
            await conn.run_sync(BaseSegmentStore.metadata.drop_all)


async def _add_claimable_record(
    partition: SQLAlchemySegmentStorePartition,
    table: type[IndexingJobRow | TombstoneRow],
) -> UUID:
    """Leave a claimable record of `table` in the partition, and return its segment's UUID.

    An indexing job is an added segment's, left pending; a tombstone is a
    forgotten segment's.
    """
    segment = _seg()
    segments_by_event, derivatives = _events(_links(segment))
    await partition.add_event_segments(segments_by_event, derivatives, lease_seconds=0)
    if table is TombstoneRow:
        await partition.delete_event_segments([segment.event_uuid])
    return segment.uuid


async def _claim(
    store: SQLAlchemySegmentStore,
    table: type[IndexingJobRow | TombstoneRow],
) -> IndexingJobClaim | TombstoneClaim | None:
    """Claim a record of `table`, taking indexing jobs at any age."""
    if table is IndexingJobRow:
        return await store.claim_indexing_job(lease_seconds=60)
    return await store.claim_tombstone(lease_seconds=60)


def _claimed_record_uuid(
    claim: IndexingJobClaim | TombstoneClaim | None,
) -> UUID | None:
    """The UUID of the segment whose record `claim` holds; None for no claim."""
    if isinstance(claim, IndexingJobClaim):
        return claim.segment_uuid
    if isinstance(claim, TombstoneClaim):
        return claim.segment_uuid
    return None


async def _fail_claim(claim: IndexingJobClaim | TombstoneClaim) -> None:
    """End `claim` as failed."""
    await claim.fail(RuntimeError("index unavailable"))


async def _shift_earlier(
    store: SQLAlchemySegmentStore,
    column: InstrumentedAttribute[datetime] | InstrumentedAttribute[datetime | None],
    by: timedelta,
) -> None:
    """Move `column` back by `by` in every row of its table, as if that much more time had passed since it was set."""
    if store._is_sqlite:
        earlier = func.datetime(column, f"-{int(by.total_seconds())} seconds")
    else:
        earlier = column - by
    async with store._create_session() as session, session.begin():
        await session.execute(update(column.table).values({column.key: earlier}))


async def _claimable_at(
    store: SQLAlchemySegmentStore,
    table: type[IndexingJobRow | TombstoneRow],
    record_uuid: UUID,
) -> datetime | None:
    """When a claim may next take the record."""
    async with store._create_session() as session:
        return (
            await session.execute(
                select(table.claimable_at).where(table.segment_uuid == record_uuid)
            )
        ).scalar_one()


async def _backdate(
    store: SQLAlchemySegmentStore,
    column: InstrumentedAttribute[datetime] | InstrumentedAttribute[datetime | None],
    delta: timedelta,
    *conditions: ColumnElement[bool],
) -> None:
    """Set `column` to the database clock's now less `delta`, in its table's rows matching `conditions`."""
    async with store._create_session() as session, session.begin():
        await session.execute(
            update(column.table)
            .where(*conditions)
            .values({column.key: _database_time_ago(store._engine, delta)})
        )


# ===================================================================
# Concurrency tests
# ===================================================================


async def _get_partition(engine: AsyncEngine) -> SQLAlchemySegmentStorePartition:
    """Create a partition handle that shares the engine."""
    store = SQLAlchemySegmentStore(SQLAlchemySegmentStoreParams(engine=engine))
    return await _created(store)


@pytest.mark.asyncio
async def test_concurrent_add_disjoint(
    store: SQLAlchemySegmentStore,
) -> None:
    """Concurrent additions with disjoint segments should not interfere."""
    engine = store._engine

    async def add_batch(batch_id: int) -> None:
        part = await _get_partition(engine)
        segs = [
            _seg(ts_offset_seconds=batch_id * 10 + i, text=f"batch{batch_id}-{i}")
            for i in range(5)
        ]
        await _add_segments(part, _links(*segs))

    await asyncio.gather(*(add_batch(i) for i in range(10)))

    # Verify all segments were added.
    part = await _get_partition(engine)
    async with part._create_session() as session:
        count = (
            await session.execute(select(func.count()).select_from(SegmentRow))
        ).scalar()
    assert count == 50
    assert await _row_counts(part, part._partition_uuid) == (50, 50)


@pytest.mark.asyncio
async def test_concurrent_reads_during_writes(
    store: SQLAlchemySegmentStore,
) -> None:
    """Reads should not fail or block indefinitely while writes are happening."""
    engine = store._engine
    partition = await _created(store)

    # Seed some data.
    ep = uuid4()
    segs = [_seg(event_uuid=ep, offset=i, ts_offset_seconds=i) for i in range(10)]
    await _add_segments(partition, _links(*segs))

    read_results: list[int] = []

    async def reader() -> None:
        part = await _get_partition(engine)
        for _ in range(5):
            result = await part.get_segment_contexts(
                [segs[5].uuid], max_backward_segments=5, max_forward_segments=5
            )
            if segs[5].uuid in result:
                read_results.append(len(result[segs[5].uuid]))
            await asyncio.sleep(0.01)

    async def writer() -> None:
        part = await _get_partition(engine)
        for i in range(5):
            new_seg = _seg(ts_offset_seconds=100 + i)
            await _add_segments(part, _links(new_seg))
            await asyncio.sleep(0.01)

    await asyncio.gather(reader(), reader(), writer())

    assert len(read_results) > 0


@pytest.mark.asyncio
async def test_concurrent_context_reads_during_forgets(
    store: SQLAlchemySegmentStore,
) -> None:
    """get_segment_contexts should not crash if events are forgotten concurrently."""
    engine = store._engine
    partition = await _created(store)

    # Register segments across multiple events.
    events = [uuid4() for _ in range(5)]
    all_segs = [
        _seg(event_uuid=ep, offset=i, ts_offset_seconds=ep_idx * 10 + i)
        for ep_idx, ep in enumerate(events)
        for i in range(4)
    ]
    await _add_segments(partition, _links(*all_segs))

    errors: list[Exception] = []

    async def context_reader() -> None:
        part = await _get_partition(engine)
        for seg in all_segs[::3]:
            try:
                await part.get_segment_contexts(
                    [seg.uuid], max_backward_segments=2, max_forward_segments=2
                )
            except Exception as e:
                errors.append(e)
            await asyncio.sleep(0.01)

    async def event_forgetter() -> None:
        part = await _get_partition(engine)
        for ep in events[1:3]:
            await part.delete_event_segments([ep])
            await asyncio.sleep(0.02)

    await asyncio.gather(context_reader(), event_forgetter())

    assert errors == []


@pytest.mark.integration
@pytest.mark.asyncio
async def test_pg_context_preserved_via_lateral_join(
    pg_store: SQLAlchemySegmentStore,
) -> None:
    """Context is preserved when retrieved via the LATERAL join path (multiple seeds)."""
    partition = await _created(pg_store)
    ep = uuid4()
    ctx_user = ProducerContext(producer="User")
    ctx_assistant = ProducerContext(producer="Assistant")
    s0 = _seg(event_uuid=ep, offset=0, ts_offset_seconds=0, context=ctx_user)
    s1 = _seg(event_uuid=ep, offset=1, ts_offset_seconds=1, context=ctx_assistant)
    s2 = _seg(event_uuid=ep, offset=2, ts_offset_seconds=2, context=ctx_user)
    s3 = _seg(event_uuid=ep, offset=3, ts_offset_seconds=3, context=ctx_assistant)
    s4 = _seg(event_uuid=ep, offset=4, ts_offset_seconds=4, context=ctx_user)
    all_segs = [s0, s1, s2, s3, s4]
    await _add_segments(partition, _links(*all_segs))

    # Two seeds exercises the LATERAL join code path.
    result = await partition.get_segment_contexts(
        [s1.uuid, s3.uuid], max_backward_segments=1, max_forward_segments=1
    )

    ctx_a = result[s1.uuid]
    assert len(ctx_a) == 3
    assert ctx_a[0].context == ctx_user
    assert ctx_a[1].context == ctx_assistant
    assert ctx_a[2].context == ctx_user

    ctx_b = result[s3.uuid]
    assert len(ctx_b) == 3
    assert ctx_b[0].context == ctx_user
    assert ctx_b[1].context == ctx_assistant
    assert ctx_b[2].context == ctx_user


@pytest.mark.integration
@pytest.mark.asyncio
async def test_pg_mixed_context_types(
    pg_store: SQLAlchemySegmentStore,
) -> None:
    """Different context types (producer, None) round-trip correctly on PG."""
    partition = await _created(pg_store)
    ctx_msg = ProducerContext(producer="User")

    s_msg = _seg(ts_offset_seconds=0, context=ctx_msg)
    s_none = _seg(ts_offset_seconds=1)

    all_segs = [s_msg, s_none]
    await _add_segments(partition, _links(*all_segs))

    async with partition._create_session() as session:
        row = (
            await session.execute(
                select(SegmentRow).where(SegmentRow.uuid == s_none.uuid)
            )
        ).scalar_one()
    assert json.loads(row.context) == {"context_type": "null"}

    result = await partition.get_segment_contexts([s_msg.uuid])
    assert result[s_msg.uuid][0].context == ctx_msg

    result = await partition.get_segment_contexts([s_none.uuid])
    assert result[s_none.uuid][0].context == NullContext()


@pytest.mark.asyncio
async def test_static_pool_engine_is_rejected(tmp_path) -> None:
    """StaticPool shares one connection; the params must refuse it loudly."""
    engine = create_async_engine(
        f"sqlite+aiosqlite:///{tmp_path / 'static_pool.db'}", poolclass=StaticPool
    )
    try:
        with pytest.raises(ValidationError, match="StaticPool"):
            SQLAlchemySegmentStoreParams(engine=engine)
    finally:
        await engine.dispose()


@pytest.mark.asyncio
async def test_old_sqlite_runtime_is_rejected(
    sqlalchemy_sqlite_engine: AsyncEngine,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Adding events and claiming indexing jobs and tombstones depend on
    RETURNING; refuse an older SQLite loudly.

    The store's own minimum is raised past the runtime rather than the
    stdlib's version tuple being patched process-wide, which SQLAlchemy's
    dialect also reads.
    """
    monkeypatch.setattr(sqlalchemy_segment_store, "_MIN_SQLITE_VERSION", (99, 0))
    with pytest.raises(ValidationError, match="RETURNING"):
        SQLAlchemySegmentStoreParams(engine=sqlalchemy_sqlite_engine)


@pytest.mark.asyncio
async def test_unsupported_dialect_engine_is_rejected(
    sqlalchemy_sqlite_engine: AsyncEngine,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The store supports PostgreSQL and SQLite; refuse any other dialect loudly."""
    monkeypatch.setattr(sqlalchemy_sqlite_engine.dialect, "name", "mssql")
    with pytest.raises(ValidationError, match="mssql"):
        SQLAlchemySegmentStoreParams(engine=sqlalchemy_sqlite_engine)


# ===================================================================
# Partitions
# ===================================================================


@pytest.mark.asyncio
async def test_getting_an_uncreated_partition_returns_none(
    store: SQLAlchemySegmentStore,
) -> None:
    assert await store.get_partition(uuid4()) is None


@pytest.mark.asyncio
async def test_getting_a_created_partition_returns_a_handle_with_its_config(
    store: SQLAlchemySegmentStore,
) -> None:
    await store.create_partition(PARTITION_UUID)

    partition = await store.get_partition(PARTITION_UUID)
    assert partition is not None
    assert partition.config == _plaintext_partition_config()


@pytest.mark.asyncio
async def test_creating_again_with_the_same_config_changes_nothing(
    store: SQLAlchemySegmentStore,
) -> None:
    partition = await _created(store)
    segment = _seg()
    await _add_segments(partition, _links(segment))

    await store.create_partition(PARTITION_UUID)
    assert await _row_counts(store, PARTITION_UUID) == (1, 1)
    assert segment.uuid in await partition.get_segment_contexts([segment.uuid])


class _ForeignPayloadCodecConfig(PlaintextPayloadCodecConfig):
    """A codec-config type distinct from the store's plaintext one."""


@pytest.mark.asyncio
async def test_creating_again_with_a_different_config_raises_mismatch(
    store: SQLAlchemySegmentStore,
) -> None:
    """Creating a partition again under a different config is refused, not adapted."""
    await _created(store)

    requested = SegmentStorePartitionConfig(
        payload_codec_config=_ForeignPayloadCodecConfig()
    )
    foreign = SQLAlchemySegmentStore(
        SQLAlchemySegmentStoreParams(engine=store._engine, partition_config=requested)
    )
    with pytest.raises(SegmentStorePartitionConfigMismatchError) as exc_info:
        await foreign.create_partition(PARTITION_UUID)
    assert exc_info.value.partition_uuid == PARTITION_UUID
    assert exc_info.value.existing_config == _plaintext_partition_config()
    assert exc_info.value.requested_config == requested

    # The partition is untouched: getting it still returns a handle.
    assert await store.get_partition(PARTITION_UUID) is not None


@pytest.mark.asyncio
async def test_an_unloadable_codec_config_creates_nothing(
    store: SQLAlchemySegmentStore,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A codec that cannot be materialized leaves no partition row behind."""

    async def unloadable(config) -> None:
        raise NotImplementedError("unsupported codec")

    with monkeypatch.context() as patched:
        patched.setattr(store, "_load_payload_codec", unloadable)
        with pytest.raises(NotImplementedError):
            await store.create_partition(PARTITION_UUID)
    async with store._create_session() as session:
        rows = (
            await session.execute(select(func.count()).select_from(PartitionRow))
        ).scalar_one()
    assert rows == 0


@pytest.mark.asyncio
async def test_concurrent_creates_of_one_partition_all_succeed_and_record_it_once(
    store: SQLAlchemySegmentStore,
) -> None:
    """Racing creates of one partition with the same config all succeed."""
    await asyncio.wait_for(
        asyncio.gather(*(store.create_partition(PARTITION_UUID) for _ in range(8))),
        60,
    )

    async with store._create_session() as session:
        partition_uuids = (
            (await session.execute(select(PartitionRow.partition_uuid))).scalars().all()
        )
    assert partition_uuids == [PARTITION_UUID]
    partition = await store.get_partition(PARTITION_UUID)
    assert partition is not None
    assert partition.config == _plaintext_partition_config()


@pytest.mark.asyncio
async def test_a_create_whose_conflicting_row_is_dropped_before_it_reads_the_row_raises(
    store: SQLAlchemySegmentStore,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A create that cannot find the row its insert conflicted with raises and leaves the partition dropped."""
    await store.create_partition(PARTITION_UUID)
    read_partition_row = store._get_partition_row

    async def dropped_first(partition_uuid: UUID) -> PartitionRow | None:
        # Stands in for a drop that commits between the conflict and the read.
        await store.drop_partition(partition_uuid)
        return await read_partition_row(partition_uuid)

    with monkeypatch.context() as patched:
        patched.setattr(store, "_get_partition_row", dropped_first)
        with pytest.raises(IntegrityError):
            await store.create_partition(PARTITION_UUID)
    assert await store.get_partition(PARTITION_UUID) is None


async def _fill(partition: SQLAlchemySegmentStorePartition) -> None:
    """Leave rows in every table: two events with their segments and derivatives, a pending indexing job, and a forgotten segment's tombstone with its derivative's UUID."""
    kept, pending, forgotten = _seg(), _seg(ts_offset_seconds=1), _seg()
    await _add_segments(partition, _links(kept, forgotten))
    await partition.add_event_segments(*_events(_links(pending)), lease_seconds=0)
    await partition.delete_event_segments([forgotten.event_uuid])


async def _budgeted_rows(store: SQLAlchemySegmentStore, partition_uuid: UUID) -> int:
    """The partition's rows a purge call counts against its bound: segments, events, and tombstones."""
    return sum(
        [
            await _count_rows(store, table.partition_uuid == partition_uuid)
            for table in (SegmentRow, EventRow, TombstoneRow)
        ]
    )


async def _purge_cursors(
    store: SQLAlchemySegmentStore,
) -> tuple[UUID | None, UUID | None, UUID | None]:
    """The purge's cursors on the segments, the events, and the tombstones."""
    async with store._create_session() as session:
        return tuple(
            (
                await session.execute(
                    select(
                        PartitionRow.segments_purged_through,
                        PartitionRow.events_purged_through,
                        PartitionRow.tombstones_purged_through,
                    ).where(PartitionRow.partition_uuid == PARTITION_UUID)
                )
            ).one()
        )


@pytest.mark.asyncio
async def test_purge_deletes_every_row_of_the_partition_and_the_drop_the_partition(
    store: SQLAlchemySegmentStore,
) -> None:
    """Events, segments, derivatives, indexing jobs, tombstones, and their derivative UUIDs are all deleted; another partition keeps its own."""
    for partition_uuid in (PARTITION_UUID, OTHER_PARTITION_UUID):
        await _fill(await _created(store, partition_uuid))

    assert await store.purge_partition(PARTITION_UUID) is True
    assert await _table_counts(store, PARTITION_UUID) == [0, 0, 0, 0, 0, 0]
    assert await store.purge_partition(PARTITION_UUID) is False
    # The partition keeps its own row until it is dropped.
    assert await store.get_partition(PARTITION_UUID) is not None
    assert await _table_counts(store, OTHER_PARTITION_UUID) == [2, 2, 2, 1, 1, 1]

    await store.drop_partition(PARTITION_UUID)
    assert await store.get_partition(PARTITION_UUID) is None
    await store.drop_partition(PARTITION_UUID)


@pytest.mark.asyncio
async def test_a_partition_that_holds_no_rows_purges_nothing_and_drops(
    store: SQLAlchemySegmentStore,
) -> None:
    await _created(store)

    assert await store.purge_partition(PARTITION_UUID) is False
    await store.drop_partition(PARTITION_UUID)
    assert await store.get_partition(PARTITION_UUID) is None


@pytest.mark.asyncio
async def test_purge_of_an_uncreated_partition_finds_nothing(
    store: SQLAlchemySegmentStore,
) -> None:
    assert await store.purge_partition(uuid4()) is False


@pytest.mark.asyncio
async def test_purge_leaves_other_partitions_whole(
    store: SQLAlchemySegmentStore,
) -> None:
    keeper = await _created(store, OTHER_PARTITION_UUID)
    segment = _seg()
    derivative_uuid = uuid4()
    await _add_segments(keeper, {segment: [derivative_uuid]})
    doomed = await _created(store)
    await _fill(doomed)

    while await store.purge_partition(PARTITION_UUID):
        pass

    assert await _table_counts(store, OTHER_PARTITION_UUID) == [1, 1, 1, 0, 0, 0]
    # The cascades still work for the partition UUID the purge left.
    await keeper.delete_event_segments([segment.event_uuid])
    assert await keeper.get_segment_uuids_by_derivative_uuids([derivative_uuid]) == {}
    assert await _table_counts(store, OTHER_PARTITION_UUID) == [0, 0, 0, 0, 1, 1]


@pytest.mark.integration
@pytest.mark.asyncio
@pytest.mark.parametrize(
    "statement",
    [
        (
            "INSERT INTO segment_store_ij"
            " (partition_uuid, segment_uuid, claimable_at, attempts_without_progress,"
            " claim_generation)"
            " VALUES (:partition_uuid, :uuid, now(), 0, 0)"
        ),
        (
            "INSERT INTO segment_store_tb_dv (partition_uuid, uuid, segment_uuid)"
            " VALUES (:partition_uuid, :uuid, :uuid)"
        ),
    ],
    ids=["indexing_job", "tombstone_derivative"],
)
async def test_purge_keeps_the_foreign_keys_enforced(
    pg_store: SQLAlchemySegmentStore,
    sqlalchemy_pg_engine: AsyncEngine,
    statement: str,
) -> None:
    """Purging one partition UUID keeps the foreign keys the cascades follow."""
    await _created(pg_store, OTHER_PARTITION_UUID)
    await _fill(await _created(pg_store))

    while await pg_store.purge_partition(PARTITION_UUID):
        pass

    # A row pointing at no parent row must still be rejected in the partition
    # UUID the purge left.
    with pytest.raises(IntegrityError, match="foreign key"):
        async with sqlalchemy_pg_engine.begin() as connection:
            await connection.execute(
                text(statement),
                {"partition_uuid": OTHER_PARTITION_UUID, "uuid": uuid4()},
            )


@pytest.mark.asyncio
async def test_purge_is_bounded_and_continues_after_its_cursors(
    store: SQLAlchemySegmentStore,
) -> None:
    """The configured bound governs every call, spans the tables, and each call picks up where the last stopped."""
    bounded = SQLAlchemySegmentStore(
        SQLAlchemySegmentStoreParams(engine=store._engine, purge_max_rows=2)
    )
    partition = await _created(bounded)
    segments = [_seg(ts_offset_seconds=index) for index in range(5)]
    await _add_segments(partition, _links(*segments))
    segment_uuids = sorted(segment.uuid for segment in segments)
    event_uuids = sorted(segment.event_uuid for segment in segments)

    async def segments_and_events() -> tuple[int, int]:
        return (
            await _count_rows(bounded, SegmentRow.partition_uuid == PARTITION_UUID),
            await _count_rows(bounded, EventRow.partition_uuid == PARTITION_UUID),
        )

    # Two segments a call, then the last segment and the first event in one
    # call, then the events from their cursor.
    for rows, cursors in (
        ((3, 5), (segment_uuids[1], None, None)),
        ((1, 5), (segment_uuids[3], None, None)),
        ((0, 4), (segment_uuids[4], event_uuids[0], None)),
        ((0, 2), (segment_uuids[4], event_uuids[2], None)),
        ((0, 0), (segment_uuids[4], event_uuids[4], None)),
    ):
        assert await bounded.purge_partition(PARTITION_UUID) is True
        assert await segments_and_events() == rows
        assert await _purge_cursors(bounded) == cursors
    # The walk from the cursors reaches the end of every table, and the
    # cursors are cleared for a search from the first keys.
    assert await bounded.purge_partition(PARTITION_UUID) is True
    assert await _purge_cursors(bounded) == (None, None, None)
    assert await bounded.purge_partition(PARTITION_UUID) is False


@pytest.mark.asyncio
async def test_purge_finds_nothing_only_after_searching_from_the_first_keys(
    store: SQLAlchemySegmentStore,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A segment, event, and tombstone written behind the cursors are purged before the purge finds nothing."""
    monkeypatch.setattr(store, "_purge_max_rows", 2)
    partition = await _created(store)
    await _add_segments(
        partition, _links(*(_seg(ts_offset_seconds=index) for index in range(3)))
    )
    forgotten = [_seg(ts_offset_seconds=index) for index in range(3)]
    await _add_segments(partition, _links(*forgotten))
    await partition.delete_event_segments(segment.event_uuid for segment in forgotten)
    while None in await _purge_cursors(store):
        assert await store.purge_partition(PARTITION_UUID) is True

    # Late writes keyed below every key the purge has passed.
    behind = _seg(event_uuid=UUID(int=1)).model_copy(update={"uuid": UUID(int=1)})
    forgotten_behind = _seg(event_uuid=UUID(int=2)).model_copy(
        update={"uuid": UUID(int=2)}
    )
    await _add_segments(partition, _links(behind, forgotten_behind))
    await partition.delete_event_segments([forgotten_behind.event_uuid])

    while await store.purge_partition(PARTITION_UUID):
        pass
    assert await _table_counts(store, PARTITION_UUID) == [0, 0, 0, 0, 0, 0]


@pytest.mark.asyncio
@pytest.mark.parametrize("events", [1, 2])
async def test_a_purge_of_whole_batches_clears_its_cursors_before_it_finds_nothing(
    store: SQLAlchemySegmentStore,
    monkeypatch: pytest.MonkeyPatch,
    events: int,
) -> None:
    """After the last full batch, one call clears the cursors, and the next finds nothing."""
    monkeypatch.setattr(store, "_purge_max_rows", 2)
    partition = await _created(store)
    # An event and its segment are two rows: one full batch each.
    await _add_segments(
        partition, _links(*(_seg(ts_offset_seconds=index) for index in range(events)))
    )

    for _ in range(events):
        assert await store.purge_partition(PARTITION_UUID) is True
    assert await _budgeted_rows(store, PARTITION_UUID) == 0
    assert await _purge_cursors(store) != (None, None, None)

    assert await store.purge_partition(PARTITION_UUID) is True
    assert await _purge_cursors(store) == (None, None, None)
    assert await store.purge_partition(PARTITION_UUID) is False


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "table", [SegmentRow, TombstoneRow], ids=["segments", "tombstones"]
)
async def test_purge_finds_nothing_only_when_its_search_finds_nothing(
    store: SQLAlchemySegmentStore,
    monkeypatch: pytest.MonkeyPatch,
    table: type[SegmentRow | TombstoneRow],
) -> None:
    """A batch another transaction deletes before the purge's delete still makes the purge return True."""
    monkeypatch.setattr(store, "_purge_max_rows", 2)
    partition = await _created(store)
    segments = [_seg(ts_offset_seconds=index) for index in range(3)]
    await _add_segments(partition, _links(*segments))
    if table is TombstoneRow:
        await partition.delete_event_segments(
            segment.event_uuid for segment in segments
        )
    batch = sorted(segment.uuid for segment in segments)[:2]

    with _deleted_before_the_next_delete(store._engine, table, batch):
        assert await store.purge_partition(PARTITION_UUID) is True

    while await store.purge_partition(PARTITION_UUID):
        pass
    assert await _table_counts(store, PARTITION_UUID) == [0, 0, 0, 0, 0, 0]


@pytest.mark.asyncio
async def test_purge_advances_its_cursor_past_a_full_batch_another_transaction_shrank(
    store: SQLAlchemySegmentStore,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The search decides a full batch: the cursor advances even when another transaction deleted part of the batch first."""
    monkeypatch.setattr(store, "_purge_max_rows", 2)
    partition = await _created(store)
    segments = [_seg(ts_offset_seconds=index) for index in range(5)]
    await _add_segments(partition, _links(*segments))
    ordered = sorted(segment.uuid for segment in segments)

    with _deleted_before_the_next_delete(store._engine, SegmentRow, ordered[:1]):
        assert await store.purge_partition(PARTITION_UUID) is True

    assert await _purge_cursors(store) == (ordered[1], None, None)
    assert await _row_counts(store, PARTITION_UUID) == (3, 3)


@contextlib.contextmanager
def _deleted_before_the_next_delete(
    engine: AsyncEngine,
    table: type[SegmentRow | TombstoneRow],
    segment_uuids: list[UUID],
) -> Generator[None, None, None]:
    """Delete the rows of `table` of the given segments just before the engine's next delete of that table's rows runs, standing in for a concurrent delete."""
    key = SegmentRow.uuid if table is SegmentRow else TombstoneRow.segment_uuid
    deleted: list[bool] = []

    def _delete_first(connection, clauseelement, *_args) -> None:
        if deleted or not (
            isinstance(clauseelement, Delete)
            and clauseelement.table.name == table.__tablename__
        ):
            return
        deleted.append(True)
        connection.execute(delete(table).where(key.in_(segment_uuids)))

    event.listen(engine.sync_engine, "before_execute", _delete_first)
    try:
        yield
    finally:
        event.remove(engine.sync_engine, "before_execute", _delete_first)
    assert deleted, f"no delete of {table.__tablename__} ran"


@pytest.mark.asyncio
@pytest.mark.parametrize("bound", [1, 2, 3, 4, 5, 7, 13])
async def test_purge_continues_through_every_table_within_the_bound(
    store: SQLAlchemySegmentStore,
    monkeypatch: pytest.MonkeyPatch,
    bound: int,
) -> None:
    """Segments, events, and tombstones share each call's bound, and each table resumes where the last call left it.

    The partition holds five segments, three events, and two tombstones, so
    the bounds end calls inside each table and exactly on each table's last
    row. Deleting a segment deletes its derivatives and indexing job by
    cascade, and deleting a tombstone its derivative UUIDs.
    """
    partition = await _created(store)
    event_uuid = uuid4()
    await _add_segments(
        partition,
        _links(*(_seg(event_uuid=event_uuid, index=index) for index in range(3))),
    )
    await _add_segments(partition, _links(_seg()))
    await partition.add_event_segments(*_events(_links(_seg())), lease_seconds=0)
    for forgotten in (_seg(), _seg()):
        await _add_segments(partition, _links(forgotten))
        await partition.delete_event_segments([forgotten.event_uuid])
    monkeypatch.setattr(store, "_purge_max_rows", bound)

    left = await _budgeted_rows(store, PARTITION_UUID)
    assert left == 10
    # Calls with full batches, then at most one that clears the cursors and
    # one that finds nothing.
    for _ in range(-(-left // bound) + 2):
        more = await store.purge_partition(PARTITION_UUID)
        now = await _budgeted_rows(store, PARTITION_UUID)
        assert left - now <= bound
        left = now
        if not more:
            break
    else:
        pytest.fail("the purge did not finish within its calls")
    assert await _table_counts(store, PARTITION_UUID) == [0, 0, 0, 0, 0, 0]


@pytest.mark.asyncio
async def test_a_failed_purge_call_leaves_nothing_half_done(
    store: SQLAlchemySegmentStore,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A purge call whose commit fails changes nothing; later calls finish.

    A call's deletions and its record of how far it got commit together or
    not at all, so no row is skipped after a failure.
    """
    partition = await _created(store)
    await _add_segments(
        partition, _links(*(_seg(ts_offset_seconds=index) for index in range(5)))
    )
    await _fill(partition)
    monkeypatch.setattr(store, "_purge_max_rows", 2)
    before = await _table_counts(store, PARTITION_UUID)

    with (
        _failing_next_commit(store._engine),
        pytest.raises(RuntimeError, match="injected commit failure"),
    ):
        await store.purge_partition(PARTITION_UUID)
    assert await _table_counts(store, PARTITION_UUID) == before
    assert await _purge_cursors(store) == (None, None, None)

    while await store.purge_partition(PARTITION_UUID):
        pass
    assert await _table_counts(store, PARTITION_UUID) == [0, 0, 0, 0, 0, 0]


@pytest.mark.asyncio
async def test_concurrent_purges_of_one_partition_reclaim_everything(
    store: SQLAlchemySegmentStore,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Racing purgers of one partition neither error nor leave rows behind."""
    monkeypatch.setattr(store, "_purge_max_rows", 2)
    partition = await _created(store)
    await _add_segments(
        partition, _links(*(_seg(ts_offset_seconds=index) for index in range(7)))
    )
    for _ in range(3):
        await _fill(partition)

    async def drain() -> None:
        while await store.purge_partition(PARTITION_UUID):
            pass

    await asyncio.wait_for(asyncio.gather(drain(), drain(), drain()), 60)

    assert await _table_counts(store, PARTITION_UUID) == [0, 0, 0, 0, 0, 0]


@pytest.mark.integration
@pytest.mark.asyncio
async def test_pg_a_forget_waits_for_a_purge_of_its_partition(
    pg_store: SQLAlchemySegmentStore,
) -> None:
    """A forget waits for a purge of its partition in progress.

    A forget and a purge that ran at once would lock the partition's
    segments and events in different orders, could each wait for the other,
    and PostgreSQL would fail one of them.
    """
    partition = await _created(pg_store)
    segments = [_seg(ts_offset_seconds=index) for index in range(3)]
    await _add_segments(partition, _links(*segments))
    held, finished = asyncio.Event(), asyncio.Event()

    async def forget_once_the_purge_is_held() -> None:
        await held.wait()
        try:
            await partition.delete_event_segments(
                segment.event_uuid for segment in segments
            )
        finally:
            finished.set()

    with _held_at_the_next_delete_until_it_blocks_another_transaction(
        pg_store._engine, SegmentRow.__tablename__, held, finished
    ) as blocked:
        await asyncio.wait_for(
            asyncio.gather(
                pg_store.purge_partition(PARTITION_UUID),
                forget_once_the_purge_is_held(),
            ),
            60,
        )

    assert blocked == [True]
    assert await _table_counts(pg_store, PARTITION_UUID) == [0, 0, 0, 0, 0, 0]


@pytest.mark.integration
@pytest.mark.asyncio
async def test_pg_a_purge_waits_for_a_forget_of_its_partition(
    pg_store: SQLAlchemySegmentStore,
) -> None:
    """A purge waits for a forget of its partition in progress, and then purges the tombstones it left."""
    partition = await _created(pg_store)
    segments = [_seg(ts_offset_seconds=index) for index in range(3)]
    await _add_segments(partition, _links(*segments))
    held, finished = asyncio.Event(), asyncio.Event()

    async def purge_once_the_forget_is_held() -> None:
        await held.wait()
        try:
            await pg_store.purge_partition(PARTITION_UUID)
        finally:
            finished.set()

    with _held_at_the_next_delete_until_it_blocks_another_transaction(
        pg_store._engine, EventRow.__tablename__, held, finished
    ) as blocked:
        await asyncio.wait_for(
            asyncio.gather(
                partition.delete_event_segments(
                    segment.event_uuid for segment in segments
                ),
                purge_once_the_forget_is_held(),
            ),
            60,
        )

    assert blocked == [True]
    assert await _table_counts(pg_store, PARTITION_UUID) == [0, 0, 0, 0, 0, 0]


@contextlib.contextmanager
def _held_at_the_next_delete_until_it_blocks_another_transaction(
    engine: AsyncEngine, table_name: str, held: asyncio.Event, finished: asyncio.Event
) -> Generator[list[bool], None, None]:
    """Hold the engine's next delete from the table until its transaction blocks another, and yield whether it did.

    Sets `held` when the hold begins. The hold ends without having blocked
    a transaction once `finished` is set.
    """
    entered: list[bool] = []
    blocked: list[bool] = []

    def _hold(connection, clauseelement, *_args) -> None:
        if entered or not (
            isinstance(clauseelement, Delete) and clauseelement.table.name == table_name
        ):
            return
        entered.append(True)
        held.set()
        while not finished.is_set():
            if connection.execute(
                text(
                    "SELECT count(*) FROM pg_locks WHERE NOT granted "
                    "AND pg_backend_pid() = ANY (pg_blocking_pids(pid))"
                )
            ).scalar_one():
                blocked.append(True)
                return
            connection.execute(text("SELECT pg_sleep(0.01)"))
        blocked.append(False)

    event.listen(engine.sync_engine, "before_execute", _hold)
    try:
        yield blocked
    finally:
        event.remove(engine.sync_engine, "before_execute", _hold)
    assert entered, f"no delete from {table_name} ran"


@pytest.mark.asyncio
async def test_purge_reclaims_a_write_that_lands_after_the_partition_is_dropped(
    store: SQLAlchemySegmentStore,
) -> None:
    """A late write to a dropped partition is reclaimed by the next purge call."""
    partition = await _created(store)
    while await store.purge_partition(PARTITION_UUID):
        pass
    await store.drop_partition(PARTITION_UUID)

    await _fill(partition)

    assert await store.purge_partition(PARTITION_UUID) is True
    assert await _table_counts(store, PARTITION_UUID) == [0, 0, 0, 0, 0, 0]
    assert await store.purge_partition(PARTITION_UUID) is False


@pytest.mark.asyncio
async def test_forgetting_events_of_a_dropped_partition_leaves_them_to_the_purge(
    store: SQLAlchemySegmentStore,
) -> None:
    """A forget on a dropped partition deletes nothing and leaves no tombstone, and the purge reclaims the events."""
    partition = await _created(store)
    segment = _seg()
    await _add_segments(partition, _links(segment))
    await store.drop_partition(PARTITION_UUID)
    held = await _table_counts(store, PARTITION_UUID)

    await partition.delete_event_segments([segment.event_uuid])

    assert await _table_counts(store, PARTITION_UUID) == held
    assert await store.purge_partition(PARTITION_UUID) is True
    assert await _table_counts(store, PARTITION_UUID) == [0, 0, 0, 0, 0, 0]


@pytest.mark.asyncio
async def test_purge_of_a_dropped_partition_deletes_at_most_the_bound_per_call(
    store: SQLAlchemySegmentStore,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Rows written to a dropped partition are purged in bounded calls that leave the partition dropped."""
    monkeypatch.setattr(store, "_purge_max_rows", 2)
    partition = await _created(store)
    await store.drop_partition(PARTITION_UUID)
    await _add_segments(
        partition, _links(*(_seg(ts_offset_seconds=index) for index in range(5)))
    )
    await _fill(partition)

    left = await _budgeted_rows(store, PARTITION_UUID)
    while await store.purge_partition(PARTITION_UUID):
        now = await _budgeted_rows(store, PARTITION_UUID)
        assert 0 < left - now <= 2
        left = now
    assert await _table_counts(store, PARTITION_UUID) == [0, 0, 0, 0, 0, 0]
    assert await store.get_partition(PARTITION_UUID) is None


@pytest.mark.asyncio
async def test_purging_while_other_partitions_are_written_leaves_exactly_their_rows(
    store: SQLAlchemySegmentStore,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Purges of some partition UUIDs and writes to others interleave cleanly."""
    monkeypatch.setattr(store, "_purge_max_rows", 3)
    retired: list[UUID] = []
    for _ in range(3):
        partition_uuid = uuid4()
        partition = await _created(store, partition_uuid)
        await _add_segments(
            partition, _links(*(_seg(ts_offset_seconds=index) for index in range(5)))
        )
        await _fill(partition)
        retired.append(partition_uuid)
    live = [await _created(store, uuid4()) for _ in range(2)]

    async def purge(partition_uuid: UUID) -> None:
        while await store.purge_partition(partition_uuid):
            pass

    async def write(partition: SQLAlchemySegmentStorePartition, seed: int) -> int:
        rng = random.Random(seed)
        written = 0
        for step in range(20):
            count = rng.randint(1, 3)
            await _add_segments(
                partition,
                _links(
                    *(
                        _seg(ts_offset_seconds=step * 10 + index)
                        for index in range(count)
                    )
                ),
            )
            written += count
        return written

    results = await asyncio.wait_for(
        asyncio.gather(
            *(write(partition, seed) for seed, partition in enumerate(live)),
            *(purge(partition_uuid) for partition_uuid in retired),
        ),
        120,
    )

    for partition_uuid in retired:
        assert await _table_counts(store, partition_uuid) == [0, 0, 0, 0, 0, 0]
    for partition, written in zip(live, results[: len(live)], strict=True):
        assert await _table_counts(store, partition._partition_uuid) == [
            written,
            written,
            written,
            0,
            0,
            0,
        ]


# ===================================================================
# Turn-taking
# Ingests and forgets of several events take turns on the partition's row
# (FOR NO KEY UPDATE), so none takes its event locks in an order; one-event
# writes take FOR KEY SHARE, which that lock does not block. The purge takes
# FOR UPDATE, which waits for both, so writes and the purge take turns.
# ===================================================================


@pytest.mark.integration
@pytest.mark.asyncio
async def test_a_purge_waits_for_a_write_in_flight(
    pg_store: SQLAlchemySegmentStore,
    sqlalchemy_pg_engine: AsyncEngine,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A purge of the partition waits until a write that has locked the partition's row commits, and then purges what it wrote.

    Stages the interleaving the lock serializes: a write transaction that
    has locked the row but not yet committed, with a concurrent purge.
    """
    partition = await _created(pg_store)

    reached_pause = asyncio.Event()
    release = asyncio.Event()
    original_insert_segments = partition._insert_segments

    async def pausing_insert_segments(session, segments) -> None:
        # Runs inside the write transaction, after the row is locked.
        reached_pause.set()
        await release.wait()
        await original_insert_segments(session, segments)

    monkeypatch.setattr(partition, "_insert_segments", pausing_insert_segments)

    writer = asyncio.create_task(
        partition.add_event_segments(*_events(_links(_seg())), lease_seconds=0)
    )
    purge = None
    try:
        await asyncio.wait_for(reached_pause.wait(), 30)
        purge = asyncio.create_task(pg_store.purge_partition(PARTITION_UUID))
        assert await _wait_until_blocked_or_done(sqlalchemy_pg_engine, purge) == (
            "blocked"
        )
    finally:
        # Unpause the writer even on failure so its open transaction cannot
        # wedge fixture teardown.
        release.set()
        with contextlib.suppress(Exception):
            await asyncio.wait_for(writer, 30)
        if purge is not None:
            with contextlib.suppress(Exception):
                await asyncio.wait_for(purge, 30)

    await writer
    assert await purge is True
    assert await _table_counts(pg_store, PARTITION_UUID) == [0, 0, 0, 0, 0, 0]


@contextlib.asynccontextmanager
async def _partition_row_held(
    partition: SQLAlchemySegmentStorePartition, *, exclusive: bool
) -> AsyncIterator[None]:
    """Hold the partition's row in another transaction, as a write of several events locks it if `exclusive`, else as a write of one."""
    async with partition._create_session() as session, session.begin():
        await session.execute(
            select(PartitionRow)
            .where(PartitionRow.partition_uuid == partition._partition_uuid)
            .with_for_update(read=not exclusive, key_share=True)
        )
        yield


async def _ingest(
    _store: SQLAlchemySegmentStore,
    partition: SQLAlchemySegmentStorePartition,
    events: int,
):
    """An operation that ingests `events` new events."""
    batch = _events(_links(*(_seg() for _ in range(events))))
    return lambda: partition.add_event_segments(*batch, lease_seconds=0)


async def _forget(
    _store: SQLAlchemySegmentStore,
    partition: SQLAlchemySegmentStorePartition,
    events: int,
):
    """An operation that forgets `events` events added now."""
    segments = [_seg() for _ in range(events)]
    await _add_segments(partition, _links(*segments))
    return lambda: partition.delete_event_segments(
        [segment.event_uuid for segment in segments]
    )


@pytest.mark.integration
@pytest.mark.asyncio
@pytest.mark.parametrize("operation", [_ingest, _forget], ids=["ingest", "forget"])
async def test_a_write_of_several_events_waits_for_the_partitions_write_lock(
    pg_store: SQLAlchemySegmentStore,
    sqlalchemy_pg_engine: AsyncEngine,
    operation,
) -> None:
    """Writes of several events take turns, so none locks its events in an order."""
    partition = await _created(pg_store)
    write = await operation(pg_store, partition, 2)

    async with _partition_row_held(partition, exclusive=True):
        task = asyncio.create_task(write())
        assert await _wait_until_blocked_or_done(sqlalchemy_pg_engine, task) == (
            "blocked"
        )
    await task


async def _delete_indexing_jobs(
    _store: SQLAlchemySegmentStore, partition: SQLAlchemySegmentStorePartition
):
    segment = _seg()
    await partition.add_event_segments(*_events(_links(segment)), lease_seconds=0)
    return lambda: partition.delete_indexing_jobs([segment.uuid])


async def _finish_second_delete(
    store: SQLAlchemySegmentStore, partition: SQLAlchemySegmentStorePartition
):
    _, claim = await _claim_second_delete(store, partition)
    return claim.finish


@pytest.mark.integration
@pytest.mark.asyncio
@pytest.mark.parametrize(
    "operation",
    [
        functools.partial(_ingest, events=1),
        functools.partial(_forget, events=1),
        _delete_indexing_jobs,
        _finish_second_delete,
    ],
    ids=[
        "ingest_one",
        "forget_one",
        "delete_indexing_jobs",
        "second_delete",
    ],
)
async def test_writes_of_one_event_and_record_deletes_do_not_wait_for_the_write_lock(
    pg_store: SQLAlchemySegmentStore,
    sqlalchemy_pg_engine: AsyncEngine,
    operation,
) -> None:
    partition = await _created(pg_store)
    write = await operation(pg_store, partition)

    async with _partition_row_held(partition, exclusive=True):
        task = asyncio.create_task(write())
        assert await _wait_until_blocked_or_done(sqlalchemy_pg_engine, task) == ("done")
        await task


@pytest.mark.integration
@pytest.mark.asyncio
@pytest.mark.parametrize(
    "exclusive", [True, False], ids=["several_events", "one_event"]
)
async def test_a_purge_waits_for_a_write_holding_the_partitions_row(
    pg_store: SQLAlchemySegmentStore,
    sqlalchemy_pg_engine: AsyncEngine,
    exclusive: bool,
) -> None:
    partition = await _created(pg_store)

    async with _partition_row_held(partition, exclusive=exclusive):
        purge = asyncio.create_task(pg_store.purge_partition(PARTITION_UUID))
        assert await _wait_until_blocked_or_done(sqlalchemy_pg_engine, purge) == (
            "blocked"
        )
    await purge


@pytest.mark.asyncio
async def test_batches_naming_the_same_events_in_opposite_orders_store_them_once(
    store: SQLAlchemySegmentStore,
) -> None:
    """One stores the events and the other is refused whole, never a deadlock."""
    partition = await _created(store)
    for _ in range(10):
        segments = [_seg() for _ in range(6)]
        batches = [_links(*segments), _links(*reversed(segments))]

        outcomes = await asyncio.wait_for(
            asyncio.gather(
                *(
                    partition.add_event_segments(*_events(batch), lease_seconds=0)
                    for batch in batches
                ),
                return_exceptions=True,
            ),
            30,
        )

        [error] = [outcome for outcome in outcomes if outcome is not None]
        assert isinstance(error, SegmentStoreEventAlreadyExistsError)
        assert error.event_uuids == {segment.event_uuid for segment in segments}


@pytest.mark.asyncio
async def test_concurrent_forgets_of_one_event_tombstone_it_once(
    store: SQLAlchemySegmentStore,
) -> None:
    """The forget that waits for the other's lock on the event finds it gone."""
    partition = await _created(store)
    for _ in range(10):
        event_uuid = uuid4()
        segments = [_seg(event_uuid=event_uuid, index=index) for index in range(3)]
        await _add_segments(partition, _links(*segments))

        await asyncio.wait_for(
            asyncio.gather(
                *(partition.delete_event_segments([event_uuid]) for _ in range(2))
            ),
            30,
        )

        tombstones = await _tombstones(partition)
        assert all(segment.uuid in tombstones for segment in segments)


@pytest.mark.integration
@pytest.mark.asyncio
async def test_a_forget_acts_only_on_the_events_its_lock_returned(
    pg_store: SQLAlchemySegmentStore,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An event an ingest commits after the forget locked its events is neither tombstoned nor deleted.

    Each statement of the forget reads a snapshot of its own on PostgreSQL,
    so one that named the requested events rather than the locked ones would
    see the event committed meanwhile: it would tombstone segments that stay
    stored, or delete the event with its segments and derivatives and
    tombstone none of them.
    """
    partition = await _created(pg_store, uuid4())
    held = _seg()
    held_links = _links(held)
    await _add_segments(partition, held_links)
    late = _seg()
    late_links = _links(late)

    reached_pause = asyncio.Event()
    release = asyncio.Event()
    original_execute = AsyncSession.execute

    async def pausing_after_the_event_lock(self, statement, *args, **kwargs):
        result = await original_execute(self, statement, *args, **kwargs)
        if not reached_pause.is_set() and _locks_events(statement):
            reached_pause.set()
            await release.wait()
        return result

    monkeypatch.setattr(AsyncSession, "execute", pausing_after_the_event_lock)

    forget = asyncio.create_task(
        partition.delete_event_segments([held.event_uuid, late.event_uuid])
    )
    try:
        await asyncio.wait_for(reached_pause.wait(), 30)
        await asyncio.wait_for(_add_segments(partition, late_links), 30)
    finally:
        # Release the forget even on failure so its open transaction cannot
        # wedge fixture teardown.
        release.set()
        with contextlib.suppress(Exception):
            await asyncio.wait_for(forget, 30)
    await forget

    assert await _tombstones(partition) == {held.uuid: sorted(held_links[held])}
    assert set(await partition.get_segment_contexts([late.uuid])) == {late.uuid}
    assert await partition.get_segment_uuids_by_derivative_uuids(
        late_links[late]
    ) == dict.fromkeys(late_links[late], late.uuid)


def _locks_events(statement: object) -> bool:
    """Whether `statement` is a SELECT that locks event rows with FOR UPDATE."""
    if not isinstance(statement, Select):
        return False
    compiled = str(statement.compile(dialect=postgresql.dialect()))
    return EventRow.__tablename__ in compiled and "FOR UPDATE" in compiled


@pytest.mark.asyncio
async def test_overlapping_forgets_do_not_deadlock(
    store: SQLAlchemySegmentStore,
) -> None:
    """Overlapping concurrent `delete_event_segments` stay deadlock-free.

    The racing forgets of several events take turns on the partition's
    write lock. Each unlinked derivative is named by exactly one of their
    tombstones.
    """
    partition = await _created(store, uuid4())

    async def round_trip(rng: random.Random) -> None:
        links = _links(*(_seg(index=index) for index in range(24)))
        await _add_segments(partition, links)
        event_uuids = [segment.event_uuid for segment in links]
        forward = event_uuids[:16]
        backward = [*reversed(event_uuids[8:])]
        await asyncio.gather(
            partition.delete_event_segments(forward),
            partition.delete_event_segments(backward),
        )
        await partition.delete_event_segments(rng.sample(event_uuids, len(event_uuids)))
        derivative_uuids = [
            derivative_uuid
            for derivative_uuids in links.values()
            for derivative_uuid in derivative_uuids
        ]
        assert await _row_counts(partition, partition._partition_uuid) == (0, 0)
        # Held by the tombstones.
        assert (
            await _count_rows(
                partition, TombstoneDerivativeRow.uuid.in_(derivative_uuids)
            )
            == 24
        )
        tombstones = await _tombstones(partition)
        assert {segment.uuid: tombstones[segment.uuid] for segment in links} == {
            segment.uuid: sorted(derivative_uuids)
            for segment, derivative_uuids in links.items()
        }

    rng = random.Random(7)
    for _ in range(10):
        await asyncio.wait_for(round_trip(rng), 30)


@pytest.mark.asyncio
@pytest.mark.parametrize("seed", [1, 2])
async def test_random_operation_sequences_agree_with_a_model(
    store: SQLAlchemySegmentStore,
    monkeypatch: pytest.MonkeyPatch,
    seed: int,
) -> None:
    """Random sequences of every data operation agree with an in-memory model.

    Adds and forgets in a few slots' current partition UUIDs, retiring a
    slot's partition UUID for a new one, and bounded purge calls on retired
    partition UUIDs. Each forget's tombstones name exactly the derivatives it
    drops from the model, after every step each current partition UUID reads
    back exactly the model's segments and links, and once every retired
    partition UUID is purged, the tables hold exactly the current rows, and
    the current partitions' tombstones with the derivatives they name.
    """
    rng = random.Random(seed)
    monkeypatch.setattr(store, "_purge_max_rows", 3)
    slots = ["model_a", "model_b", "model_c"]
    handles: dict[str, SQLAlchemySegmentStorePartition] = {}
    # Each slot's events, with each event's segments and their links.
    model: dict[str, dict[UUID, dict[UUID, set[UUID]]]] = {}
    tombstones: dict[str, int] = {}
    tombstoned_derivatives: dict[str, int] = {}
    retired: list[UUID] = []

    for step in range(120):
        slot = rng.choice(slots)
        operation = rng.choice(["add", "add", "forget", "retire", "purge"])
        if operation == "purge" and retired:
            await store.purge_partition(rng.choice(retired))
        elif operation == "retire" and slot in handles:
            retired.append(handles.pop(slot)._partition_uuid)
            model.pop(slot)
            tombstones.pop(slot)
            tombstoned_derivatives.pop(slot)
        elif operation in ("add", "forget"):
            if slot not in handles:
                handles[slot] = await _created(store, uuid4())
                model[slot] = {}
                tombstones[slot] = 0
                tombstoned_derivatives[slot] = 0
            if operation == "add":
                segments = _random_model_events(rng, ts_offset_seconds=step * 10)
                await _add_segments(handles[slot], segments)
                for segment, links in segments.items():
                    model[slot].setdefault(segment.event_uuid, {})[segment.uuid] = links
            else:
                forgotten, forgotten_derivatives = await _forget_in_model(
                    handles[slot], model[slot], rng
                )
                tombstones[slot] += forgotten
                tombstoned_derivatives[slot] += forgotten_derivatives

        await asyncio.gather(
            *(
                _assert_reads_agree_with_the_model(handle, model[current_slot])
                for current_slot, handle in handles.items()
            )
        )

    for partition_uuid in retired:
        while await store.purge_partition(partition_uuid):
            pass
    await _assert_tables_agree_with_the_model(
        store,
        [event for slot_events in model.values() for event in slot_events.values()],
        tombstones=sum(tombstones.values()),
        tombstoned_derivatives=sum(tombstoned_derivatives.values()),
    )


async def _assert_tables_agree_with_the_model(
    store: SQLAlchemySegmentStore,
    events: list[dict[UUID, set[UUID]]],
    *,
    tombstones: int,
    tombstoned_derivatives: int,
) -> None:
    """The tables hold exactly the model's events, their segments and links, and the tombstones with the derivative UUIDs they hold.

    `events` holds each event's segments' links.
    """
    async with store._create_session() as session:
        row_counts = [
            (
                await session.execute(select(func.count()).select_from(table))
            ).scalar_one()
            for table in _PARTITION_TABLES
        ]
    assert row_counts == [
        len(events),
        sum(map(len, events)),
        sum(len(links) for event in events for links in event.values()),
        0,
        tombstones,
        tombstoned_derivatives,
    ]


async def _forget_in_model(
    partition: SQLAlchemySegmentStorePartition,
    events: dict[UUID, dict[UUID, set[UUID]]],
    rng: random.Random,
) -> tuple[int, int]:
    """Forget a few of the model's events, and an event the partition never held, checking the new tombstones name exactly the derivatives forgotten.

    `events` maps each event to its segments' links, and loses the forgotten
    events. Returns how many tombstones the forget left, and how many
    derivative UUIDs they hold.
    """
    victims = rng.sample(sorted(events), min(len(events), rng.randint(0, 3)))
    before = await _tombstones(partition)
    await partition.delete_event_segments([*victims, uuid4()])
    unlinked = {
        segment_uuid: sorted(links)
        for victim in victims
        for segment_uuid, links in events.pop(victim).items()
    }
    assert {
        segment_uuid: derivative_uuids
        for segment_uuid, derivative_uuids in (await _tombstones(partition)).items()
        if segment_uuid not in before
    } == unlinked
    return len(unlinked), sum(map(len, unlinked.values()))


def _random_model_events(
    rng: random.Random, *, ts_offset_seconds: int
) -> dict[Segment, set[UUID]]:
    """A few events' segments, from `ts_offset_seconds` on, each with random links."""
    segments: dict[Segment, set[UUID]] = {}
    for offset in range(rng.randint(1, 3)):
        event_uuid = uuid4()
        for index in range(rng.randint(1, 2)):
            segment = _seg(
                event_uuid=event_uuid,
                index=index,
                ts_offset_seconds=ts_offset_seconds + offset,
            )
            segments[segment] = {uuid4() for _ in range(rng.randint(0, 2))}
    return segments


async def _assert_reads_agree_with_the_model(
    partition: SQLAlchemySegmentStorePartition,
    events: dict[UUID, dict[UUID, set[UUID]]],
) -> None:
    """The partition reads back exactly the model's segments and links.

    `events` maps each event to its segments' links.
    """
    segment_links = {
        segment_uuid: links
        for segments in events.values()
        for segment_uuid, links in segments.items()
    }
    expected = {
        link: segment_uuid
        for segment_uuid, links in segment_links.items()
        for link in links
    }
    assert await partition.get_segment_uuids_by_derivative_uuids(expected) == expected
    assert set(await partition.get_segment_contexts(segment_links)) == set(
        segment_links
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("seed", [1, 2])
async def test_mixed_operation_churn_completes_and_strands_nothing(
    store: SQLAlchemySegmentStore,
    monkeypatch: pytest.MonkeyPatch,
    seed: int,
) -> None:
    """Every data operation at once completes and strands nothing.

    Writers add and forget their own events in live partitions, churners
    create, fill, and retire partitions, and two purgers purge and drop the
    retired partitions throughout. Everything finishes within the timeout, so
    no lock cycle stalls it, and after a final drain no row belongs to a
    retired partition, while each writer's partition holds exactly what it
    left, and the derivative UUIDs its forgets' tombstones hold.
    """
    rng = random.Random(seed)
    monkeypatch.setattr(store, "_purge_max_rows", 3)
    live = [await _created(store, uuid4()) for _ in range(2)]
    # Each writer's events, by UUID, with their link counts; an event has
    # one segment.
    written: dict[UUID, dict[UUID, int]] = {
        partition._partition_uuid: {} for partition in live
    }
    tombstoned_links: dict[UUID, int] = {
        partition._partition_uuid: 0 for partition in live
    }
    retired: list[UUID] = []
    finished = asyncio.Event()

    async def purger() -> None:
        while not finished.is_set():
            purged = False
            for partition_uuid in retired:
                if await store.purge_partition(partition_uuid):
                    purged = True
                else:
                    await store.drop_partition(partition_uuid)
            if not purged:
                await asyncio.sleep(0.01)

    async def work() -> None:
        await asyncio.gather(
            *(
                _write_and_forget_own_events(
                    partition,
                    written[partition._partition_uuid],
                    tombstoned_links,
                    rng,
                )
                for partition in live
            ),
            *(_churn_partitions(store, retired, rng) for _ in range(3)),
        )
        finished.set()

    await asyncio.wait_for(asyncio.gather(work(), purger(), purger()), 120)
    for partition_uuid in retired:
        while await store.purge_partition(partition_uuid):
            pass
        await store.drop_partition(partition_uuid)

    live_uuids = [partition._partition_uuid for partition in live]
    stranded = [
        await _count_rows(store, table.partition_uuid.not_in(live_uuids))
        for table in (PartitionRow, *_PARTITION_TABLES)
    ]
    assert stranded == [0, 0, 0, 0, 0, 0, 0]
    for partition_uuid, events in written.items():
        assert await _row_counts(store, partition_uuid) == (
            len(events),
            sum(events.values()),
        )
        assert (
            await _count_rows(
                store, TombstoneDerivativeRow.partition_uuid == partition_uuid
            )
            == tombstoned_links[partition_uuid]
        )


async def _write_and_forget_own_events(
    partition: SQLAlchemySegmentStorePartition,
    mine: dict[UUID, int],
    tombstoned_links: dict[UUID, int],
    rng: random.Random,
) -> None:
    """Add events, then forget some of this writer's own; `mine` tracks what is left, and `tombstoned_links` the derivative UUIDs the forgets' tombstones hold."""
    for round_index in range(12):
        segments = {
            _seg(ts_offset_seconds=round_index * 10 + offset): [
                uuid4() for _ in range(rng.randint(0, 2))
            ]
            for offset in range(rng.randint(1, 3))
        }
        await _add_segments(partition, segments)
        mine.update(
            {segment.event_uuid: len(links) for segment, links in segments.items()}
        )
        victims = rng.sample(sorted(mine), min(len(mine), rng.randint(0, 2)))
        await partition.delete_event_segments(victims)
        for victim in victims:
            tombstoned_links[partition._partition_uuid] += mine.pop(victim)


async def _churn_partitions(
    store: SQLAlchemySegmentStore,
    retired: list[UUID],
    rng: random.Random,
) -> None:
    """Create and fill partitions, forget some of their events, and retire them to `retired`."""
    for _ in range(8):
        partition = await _created(store, uuid4())
        segments = [
            _seg(ts_offset_seconds=offset) for offset in range(rng.randint(1, 4))
        ]
        await _add_segments(partition, _links(*segments))
        await partition.delete_event_segments(
            segment.event_uuid for segment in segments[: rng.randint(0, len(segments))]
        )
        retired.append(partition._partition_uuid)
