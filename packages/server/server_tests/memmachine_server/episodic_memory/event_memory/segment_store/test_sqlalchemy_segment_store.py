"""Tests for SQLAlchemySegmentStore — SQLite (unit) and PostgreSQL (integration)."""

import asyncio
import contextlib
import functools
import json
import logging
import operator
import random
from collections.abc import AsyncIterator, Generator, Iterable, Iterator, Mapping
from datetime import UTC, datetime, timedelta, timezone
from uuid import UUID, uuid4

import pytest
import pytest_asyncio
from pydantic import ValidationError
from sqlalchemy import (
    ColumnElement,
    Delete,
    delete,
    event,
    func,
    insert,
    select,
    text,
    update,
)
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncConnection, AsyncEngine, create_async_engine
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
    PendingWriteClaim,
    SegmentStoreAttemptsExhaustedError,
    SegmentStoreEventAlreadyExistsError,
    SegmentStorePartitionAlreadyExistsError,
    SegmentStorePartitionConfig,
    SegmentStorePartitionConfigMismatchError,
    SegmentStorePartitionHandleStaleError,
    Tombstone,
    TombstoneClaim,
    sqlalchemy_segment_store,
)
from memmachine_server.episodic_memory.event_memory.segment_store.sqlalchemy_segment_store import (
    BaseSegmentStore,
    DerivativeRow,
    EventRow,
    PartitionRow,
    PendingWriteRow,
    PurgeQueueRow,
    SegmentRow,
    SQLAlchemySegmentStore,
    SQLAlchemySegmentStoreParams,
    SQLAlchemySegmentStorePartition,
    TombstoneDerivativeRow,
    TombstoneRow,
)
from memmachine_server.episodic_memory.event_memory.segment_store.utils import (
    validate_partition_key,
)

PARTITION_KEY = "test_partition"
BASE_TIME = datetime(2024, 1, 1, tzinfo=UTC)
_NULL_CONTEXT = NullContext()


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
    await partition.delete_pending_writes(
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
    incarnation: UUID,
) -> tuple[int, int]:
    """The incarnation's segment and derivative row counts."""
    async with owner._create_session() as session:
        segments = (
            await session.execute(
                select(func.count())
                .select_from(SegmentRow)
                .where(SegmentRow.incarnation == incarnation)
            )
        ).scalar_one()
        links = (
            await session.execute(
                select(func.count())
                .select_from(DerivativeRow)
                .where(DerivativeRow.incarnation == incarnation)
            )
        ).scalar_one()
    return segments, links


async def _count_rows(
    owner: SQLAlchemySegmentStore | SQLAlchemySegmentStorePartition,
    condition: ColumnElement[bool],
) -> int:
    """How many rows of the table `condition` reads satisfy it."""
    async with owner._create_session() as session:
        return (
            await session.execute(select(func.count()).where(condition))
        ).scalar_one()


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


def _record_statements(engine: AsyncEngine) -> Iterator[list[str]]:
    """Record every statement the engine executes, normalized to one line.

    Recording starts immediately; tests clear the list after their own
    setup so assertions cover only the exercised operations.
    """
    statements: list[str] = []

    def _record(_conn, _cursor, statement, _parameters, _context, _executemany):
        statements.append(" ".join(statement.split()))

    event.listen(engine.sync_engine, "before_cursor_execute", _record)
    yield statements
    event.remove(engine.sync_engine, "before_cursor_execute", _record)


@pytest.fixture
def recorded_statements(
    sqlalchemy_pg_engine: AsyncEngine,
) -> Iterator[list[str]]:
    yield from _record_statements(sqlalchemy_pg_engine)


@pytest_asyncio.fixture
async def partition(
    store: SQLAlchemySegmentStore,
) -> SQLAlchemySegmentStorePartition:
    return await store.open_or_create_partition(
        PARTITION_KEY,
        _plaintext_partition_config(),
    )


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
    assert await _row_counts(partition, partition._incarnation) == (2, 2)
    assert (
        await _count_rows(
            partition, PendingWriteRow.incarnation == partition._incarnation
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

    assert await _row_counts(partition, partition._incarnation) == (0, 0)
    assert (
        await _count_rows(partition, EventRow.incarnation == partition._incarnation)
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

    assert await _row_counts(partition, partition._incarnation) == (1, 1)
    assert (
        await _count_rows(partition, EventRow.incarnation == partition._incarnation)
        == 1
    )
    assert (
        await _count_rows(
            partition, PendingWriteRow.incarnation == partition._incarnation
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
    assert await _row_counts(partition, partition._incarnation) == (3, 3)
    # One pending write for each of the winner's segments.
    assert (
        await _count_rows(
            partition, PendingWriteRow.incarnation == partition._incarnation
        )
        == 3
    )


@pytest.mark.asyncio
async def test_pending_writes_roundtrip_through_a_claim(
    store: SQLAlchemySegmentStore,
) -> None:
    """A claim returns the pending write as added, timestamp offsets included."""
    partition = await store.open_or_create_partition(
        PARTITION_KEY, _plaintext_partition_config()
    )
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

    claims = [await store.claim_pending_write(lease_seconds=60) for _ in range(2)]

    assert await store.claim_pending_write(lease_seconds=60) is None
    claimed: list[Derivative] = []
    for claim in claims:
        assert claim is not None
        assert claim.partition_key == PARTITION_KEY
        assert (claim.generation, claim.attempt) == (1, 1)
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
# delete_pending_writes
# ===================================================================


@pytest.mark.asyncio
async def test_delete_pending_writes_removes_only_the_pending_record(
    store: SQLAlchemySegmentStore,
) -> None:
    """The segment's derivatives stay and other segments stay pending. Idempotent."""
    partition = await store.open_or_create_partition(
        PARTITION_KEY, _plaintext_partition_config()
    )
    finished, unfinished = _seg(), _seg()
    segments_by_event, derivatives = _events(
        {finished: [uuid4(), uuid4()], unfinished: [uuid4()]}
    )
    await partition.add_event_segments(segments_by_event, derivatives, lease_seconds=0)

    for _ in range(2):
        await partition.delete_pending_writes([finished.uuid])

    assert await partition.get_segment_uuids_by_derivative_uuids(
        derivative.uuid for derivative in derivatives
    ) == {derivative.uuid: derivative.segment_uuid for derivative in derivatives}
    assert (
        await _count_rows(partition, PendingWriteRow.segment_uuid == finished.uuid) == 0
    )
    claim = await store.claim_pending_write(lease_seconds=60)
    assert claim is not None
    assert claim.segment_uuid == unfinished.uuid
    assert await store.claim_pending_write(lease_seconds=60) is None


@pytest.mark.integration
@pytest.mark.asyncio
async def test_delete_pending_writes_skips_a_record_another_transaction_holds(
    pg_store: SQLAlchemySegmentStore,
    sqlalchemy_pg_engine: AsyncEngine,
) -> None:
    """The delete never waits: a held record is left for the background work."""
    partition = await pg_store.open_or_create_partition(
        PARTITION_KEY, _plaintext_partition_config()
    )
    held, free = _seg(), _seg()
    await partition.add_event_segments(*_events(_links(held, free)), lease_seconds=0)

    async with (
        partition._create_session() as remote_session,
        remote_session.begin(),
    ):
        await remote_session.execute(
            select(PendingWriteRow)
            .where(PendingWriteRow.segment_uuid == held.uuid)
            .with_for_update()
        )
        deletion = asyncio.create_task(
            partition.delete_pending_writes([held.uuid, free.uuid])
        )
        assert await _wait_until_blocked_or_done(sqlalchemy_pg_engine, deletion) == (
            "done"
        )
        await deletion

    assert await _count_rows(partition, PendingWriteRow.segment_uuid == held.uuid) == 1
    assert await _count_rows(partition, PendingWriteRow.segment_uuid == free.uuid) == 0


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
    """Context only includes segments from the same partition_key."""
    ep = uuid4()
    s_other = _seg(event_uuid=ep, offset=0, ts_offset_seconds=0)
    s_seed = _seg(event_uuid=ep, offset=1, ts_offset_seconds=1)
    s_after = _seg(event_uuid=ep, offset=2, ts_offset_seconds=2)

    other_partition = await store.open_or_create_partition(
        "other_session",
        _plaintext_partition_config(),
    )
    await _add_segments(other_partition, _links(s_other))

    partition = await store.open_or_create_partition(
        PARTITION_KEY,
        _plaintext_partition_config(),
    )
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
    config = _plaintext_partition_config()
    partition = await store.open_or_create_partition(PARTITION_KEY, config)
    other_partition = await store.open_or_create_partition("other_partition", config)
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
    other_partition = await store.open_or_create_partition(
        "other_derivatives",
        _plaintext_partition_config(),
    )
    other_seg = _seg()
    other_derivative = uuid4()
    await _add_segments(other_partition, {other_seg: [other_derivative]})

    partition = await store.open_or_create_partition(
        PARTITION_KEY,
        _plaintext_partition_config(),
    )
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

    await partition.delete_event_segments([dropped.event_uuid], lease_seconds=0)

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

    await partition.delete_event_segments([seg.event_uuid], lease_seconds=0)

    assert await partition.get_segment_contexts([seg.uuid]) == {}
    assert await partition.get_segment_uuids_by_derivative_uuids([derivative]) == {}
    assert await _row_counts(partition, partition._incarnation) == (0, 0)
    assert (
        await _count_rows(partition, EventRow.incarnation == partition._incarnation)
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

    await partition.delete_event_segments([forgotten_event], lease_seconds=0)

    result = await partition.get_segment_contexts(
        segment.uuid for segment in (*forgotten, *kept)
    )
    assert set(result) == {segment.uuid for segment in kept}


@pytest.mark.asyncio
async def test_delete_event_segments_returns_a_tombstone_per_forgotten_segment_naming_its_derivatives(
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

    tombstones = await partition.delete_event_segments(
        [*forgotten, uuid4()], lease_seconds=0
    )

    assert {
        tombstone.segment_uuid: sorted(tombstone.derivative_uuids)
        for tombstone in tombstones
    } == {
        segment.uuid: sorted(derivative_uuids)
        for segment, derivative_uuids in links.items()
        if segment.event_uuid in forgotten
    }


@pytest.mark.asyncio
async def test_delete_event_segments_replaces_the_pending_write_with_a_tombstone_holding_the_derivative_uuids(
    store: SQLAlchemySegmentStore,
) -> None:
    """The derivatives are deleted with their segment, and its tombstone holds their UUIDs."""
    partition = await store.open_or_create_partition(
        PARTITION_KEY, _plaintext_partition_config()
    )
    segment = _seg()
    segments_by_event, derivatives = _events({segment: [uuid4(), uuid4()]})
    await partition.add_event_segments(segments_by_event, derivatives, lease_seconds=0)
    derivative_uuids = [derivative.uuid for derivative in derivatives]

    [tombstone] = await partition.delete_event_segments(
        [segment.event_uuid], lease_seconds=0
    )

    assert tombstone.segment_uuid == segment.uuid
    assert sorted(tombstone.derivative_uuids) == sorted(derivative_uuids)
    assert await _row_counts(partition, partition._incarnation) == (0, 0)
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
    assert await store.claim_pending_write(lease_seconds=60) is None
    assert await partition.get_segment_uuids_by_derivative_uuids(derivative_uuids) == {}
    claim = await store.claim_tombstone(lease_seconds=60)
    assert claim is not None
    assert sorted(claim.tombstone.derivative_uuids) == sorted(derivative_uuids)


@pytest.mark.asyncio
async def test_delete_event_segments_skips_unknown_events_and_forgets_events_without_derivatives(
    partition: SQLAlchemySegmentStorePartition,
) -> None:
    """A segment without derivatives gets a tombstone naming none; an event without segments gets none."""
    bare = _seg()
    await _add_segments(partition, {bare: []})
    empty_event_uuid = uuid4()
    await partition.add_event_segments({empty_event_uuid: []}, [], lease_seconds=0)

    assert await partition.delete_event_segments([], lease_seconds=0) == []
    assert await partition.delete_event_segments([uuid4()], lease_seconds=0) == []
    assert await partition.delete_event_segments(
        [bare.event_uuid], lease_seconds=0
    ) == [Tombstone(segment_uuid=bare.uuid, derivative_uuids=[])]
    assert (
        await partition.delete_event_segments([empty_event_uuid], lease_seconds=0) == []
    )

    assert await partition.get_segment_contexts([bare.uuid]) == {}
    assert (
        await _count_rows(partition, EventRow.incarnation == partition._incarnation)
        == 0
    )
    assert (
        await _count_rows(
            partition,
            TombstoneRow.incarnation == partition._incarnation,
        )
        == 1
    )


@pytest.mark.asyncio
async def test_delete_event_segments_is_idempotent(
    partition: SQLAlchemySegmentStorePartition,
) -> None:
    segment = _seg()
    await _add_segments(partition, _links(segment))

    assert (
        await partition.delete_event_segments([segment.event_uuid], lease_seconds=0)
        != []
    )
    assert (
        await partition.delete_event_segments([segment.event_uuid], lease_seconds=0)
        == []
    )
    assert (
        await _count_rows(partition, TombstoneRow.incarnation == partition._incarnation)
        == 1
    )


@pytest.mark.asyncio
async def test_forgotten_event_can_be_added_again(
    partition: SQLAlchemySegmentStorePartition,
) -> None:
    """A changed event is forgotten and added again under its UUID."""
    event_uuid = uuid4()
    await _add_segments(partition, _links(_seg(event_uuid=event_uuid)))
    await partition.delete_event_segments([event_uuid], lease_seconds=0)
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
                        incarnation=uuid4(), segment_uuid=uuid4(), claimable_at=None
                    )
                )


@pytest.mark.asyncio
async def test_a_writers_own_pending_write_delete_leaves_the_tombstone_a_forget_made_since(
    store: SQLAlchemySegmentStore,
) -> None:
    """The writer's delete of its pending writes leaves the forget's tombstone to its deletes."""
    partition = await store.open_or_create_partition(
        PARTITION_KEY, _plaintext_partition_config()
    )
    segment = _seg()
    segments_by_event, derivatives = _events(_links(segment))
    await partition.add_event_segments(
        segments_by_event, derivatives, lease_seconds=3600
    )
    await partition.delete_event_segments([segment.event_uuid], lease_seconds=0)

    await partition.delete_pending_writes([segment.uuid])

    claim = await store.claim_tombstone(lease_seconds=60)
    assert claim is not None
    assert claim.tombstone == Tombstone(
        segment_uuid=segment.uuid, derivative_uuids=[derivatives[0].uuid]
    )


@pytest.mark.asyncio
async def test_a_stale_pending_write_claim_leaves_a_tombstone_at_its_generation_alone(
    store: SQLAlchemySegmentStore,
) -> None:
    """A tombstone written after its pending write was deleted starts its generations over, and a claim on the pending write still reaches none of it."""
    partition = await store.open_or_create_partition(
        PARTITION_KEY, _plaintext_partition_config()
    )
    segment = _seg()
    await partition.add_event_segments(*_events(_links(segment)), lease_seconds=0)
    stale = await store.claim_pending_write(lease_seconds=60)
    assert stale is not None
    await partition.delete_pending_writes([segment.uuid])
    await partition.delete_event_segments([segment.event_uuid], lease_seconds=0)
    await partition.record_tombstone_first_delete([segment.uuid])

    assert not await store.renew_pending_write_claim(stale, lease_seconds=60)
    await store.fail_pending_write_claim(stale, RuntimeError("index unavailable"))
    await store.cancel_pending_write_claim(stale)
    await store.finish_pending_write_claim(stale)

    await _backdate(store, TombstoneRow.first_deleted_at, timedelta(days=2))
    second = await store.claim_tombstone(lease_seconds=60)
    assert second is not None
    assert (second.tombstone.segment_uuid, second.second_delete, second.attempt) == (
        segment.uuid,
        True,
        1,
    )


@pytest.mark.asyncio
async def test_a_segment_without_derivatives_has_the_lifecycle_of_any_other(
    store: SQLAlchemySegmentStore,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Its pending write is claimed with no derivatives, and its tombstone names none and is deleted by its second delete."""
    monkeypatch.setattr(store, "_tombstone_retention_seconds", 3600)
    partition = await store.open_or_create_partition(
        PARTITION_KEY, _plaintext_partition_config()
    )
    segment = _seg()
    await partition.add_event_segments(*_events({segment: []}), lease_seconds=0)

    pending = await store.claim_pending_write(lease_seconds=60)
    assert pending is not None
    assert (pending.segment_uuid, pending.derivatives) == (segment.uuid, [])

    await partition.delete_event_segments([segment.event_uuid], lease_seconds=0)
    first = await store.claim_tombstone(lease_seconds=60)
    assert first is not None
    assert first.tombstone == Tombstone(segment_uuid=segment.uuid, derivative_uuids=[])
    await store.finish_tombstone_claim(first)
    await _backdate(store, TombstoneRow.first_deleted_at, timedelta(hours=2))
    second = await store.claim_tombstone(lease_seconds=60)
    assert second is not None
    assert second.second_delete is True
    await store.finish_tombstone_claim(second)

    assert await _count_rows(store, TombstoneRow.segment_uuid == segment.uuid) == 0


@pytest.mark.asyncio
async def test_recording_a_first_delete_passes_over_segments_without_a_tombstone(
    store: SQLAlchemySegmentStore,
) -> None:
    """A segment that was not forgotten keeps its pending write as it was."""
    partition = await store.open_or_create_partition(
        PARTITION_KEY, _plaintext_partition_config()
    )
    segment = _seg()
    await partition.add_event_segments(*_events(_links(segment)), lease_seconds=0)

    await partition.record_tombstone_first_delete([segment.uuid])

    claim = await store.claim_pending_write(lease_seconds=60)
    assert claim is not None
    assert (claim.segment_uuid, claim.generation) == (segment.uuid, 1)


@pytest.mark.asyncio
async def test_claim_on_a_pending_write_of_a_forgotten_event_ends_without_effect(
    store: SQLAlchemySegmentStore,
) -> None:
    """The forget deleted the claim's record by cascade, and the tombstone names the claimed derivatives."""
    partition = await store.open_or_create_partition(
        PARTITION_KEY, _plaintext_partition_config()
    )
    segment = _seg()
    segments_by_event, derivatives = _events(_links(segment))
    await partition.add_event_segments(segments_by_event, derivatives, lease_seconds=0)
    claim = await store.claim_pending_write(lease_seconds=60)
    assert claim is not None

    await partition.delete_event_segments([segment.event_uuid], lease_seconds=0)
    assert not await store.renew_pending_write_claim(claim, lease_seconds=60)
    await store.finish_pending_write_claim(claim)
    await store.fail_pending_write_claim(claim, RuntimeError("index unavailable"))
    await store.cancel_pending_write_claim(claim)

    assert (
        await _count_rows(
            partition,
            PendingWriteRow.incarnation == partition._incarnation,
        )
        == 0
    )
    tombstone_claim = await store.claim_tombstone(lease_seconds=60)
    assert tombstone_claim is not None
    assert tombstone_claim.tombstone.derivative_uuids == [
        derivative.uuid for derivative in claim.derivatives
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize("operation", ["add_event_segments", "delete_event_segments"])
async def test_event_write_whose_commit_fails_changes_nothing(
    store: SQLAlchemySegmentStore,
    operation: str,
) -> None:
    """An event write whose commit fails leaves the partition as it was."""
    partition = await store.open_or_create_partition(
        "uncommitted", _plaintext_partition_config()
    )
    kept = _seg()
    await _add_segments(partition, _links(kept))
    write = (
        partition.add_event_segments(
            *_events(_links(_seg(ts_offset_seconds=1))), lease_seconds=0
        )
        if operation == "add_event_segments"
        else partition.delete_event_segments([kept.event_uuid], lease_seconds=0)
    )

    with (
        _failing_next_commit(store._engine),
        pytest.raises(RuntimeError, match="injected commit failure"),
    ):
        await write

    assert await _row_counts(store, partition._incarnation) == (1, 1)
    for table, count in (
        (EventRow, 1),
        (PendingWriteRow, 0),
        (TombstoneRow, 0),
    ):
        assert (
            await _count_rows(store, table.incarnation == partition._incarnation)
            == count
        )
    assert kept.uuid in await partition.get_segment_contexts([kept.uuid])


# ===================================================================
# Claims
# ===================================================================

_CLAIMED_TABLES = pytest.mark.parametrize(
    "table",
    [PendingWriteRow, TombstoneRow],
    ids=["pending_write", "tombstone"],
)


@pytest.mark.asyncio
async def test_a_pending_write_is_claimable_once_its_writers_lease_has_passed(
    store: SQLAlchemySegmentStore,
) -> None:
    partition = await store.open_or_create_partition(
        PARTITION_KEY, _plaintext_partition_config()
    )
    segments_by_event, derivatives = _events(_links(_seg()))
    await partition.add_event_segments(
        segments_by_event, derivatives, lease_seconds=3600
    )

    assert await store.claim_pending_write(lease_seconds=60) is None
    await _shift_earlier(store, PendingWriteRow.claimable_at, timedelta(minutes=50))
    assert await store.claim_pending_write(lease_seconds=60) is None
    await _shift_earlier(store, PendingWriteRow.claimable_at, timedelta(minutes=20))
    claim = await store.claim_pending_write(lease_seconds=60)
    assert _claimed_record_uuid(claim) == derivatives[0].segment_uuid


@pytest.mark.asyncio
@_CLAIMED_TABLES
async def test_claims_take_the_oldest_record_first(
    store: SQLAlchemySegmentStore,
    table: type[PendingWriteRow | TombstoneRow],
) -> None:
    partition = await store.open_or_create_partition(
        PARTITION_KEY, _plaintext_partition_config()
    )
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
    table: type[PendingWriteRow | TombstoneRow],
) -> None:
    """A record is claimed again only once its claim's lease is over, by a later generation."""
    partition = await store.open_or_create_partition(
        PARTITION_KEY, _plaintext_partition_config()
    )
    record_uuid = await _add_claimable_record(partition, table)

    first = await _claim(store, table)
    assert first is not None
    assert (_claimed_record_uuid(first), first.attempt) == (record_uuid, 1)
    assert await _claim(store, table) is None

    await _backdate(store, table.claimable_at, timedelta(days=1))
    second = await _claim(store, table)
    assert second is not None
    assert (_claimed_record_uuid(second), second.generation, second.attempt) == (
        record_uuid,
        first.generation + 1,
        2,
    )


@pytest.mark.asyncio
async def test_finishing_a_superseded_pending_write_claim_deletes_the_record(
    store: SQLAlchemySegmentStore,
) -> None:
    """A claim that outlasted its lease wrote the derivatives, so its finish deletes the record, and the later claim's renewal finds it gone."""
    partition = await store.open_or_create_partition(
        PARTITION_KEY, _plaintext_partition_config()
    )
    pending_uuid = await _add_claimable_record(partition, PendingWriteRow)
    superseded = await store.claim_pending_write(lease_seconds=60)
    await _backdate(store, PendingWriteRow.claimable_at, timedelta(days=1))
    latest = await store.claim_pending_write(lease_seconds=60)
    assert superseded is not None
    assert latest is not None

    await store.finish_pending_write_claim(superseded)
    assert await _count_rows(store, PendingWriteRow.segment_uuid == pending_uuid) == 0
    assert not await store.renew_pending_write_claim(latest, lease_seconds=60)
    await store.finish_pending_write_claim(latest)
    # The segment's derivatives stay with it.
    assert await _count_rows(store, DerivativeRow.segment_uuid == pending_uuid) == 1


@pytest.mark.asyncio
@_CLAIMED_TABLES
async def test_failing_a_superseded_claim_changes_nothing(
    store: SQLAlchemySegmentStore,
    table: type[PendingWriteRow | TombstoneRow],
) -> None:
    """It neither ends the later claim nor starts a backoff."""
    partition = await store.open_or_create_partition(
        PARTITION_KEY, _plaintext_partition_config()
    )
    record_uuid = await _add_claimable_record(partition, table)
    superseded = await _claim(store, table)
    await _backdate(store, table.claimable_at, timedelta(days=1))
    assert await _claim(store, table) is not None
    assert superseded is not None
    scheduled = await _claimable_at(store, table, record_uuid)

    await _fail_claim(store, superseded)

    assert await _claimable_at(store, table, record_uuid) == scheduled


@pytest.mark.asyncio
@_CLAIMED_TABLES
async def test_failed_claim_backs_off_doubling_up_to_the_cap(
    store: SQLAlchemySegmentStore,
    monkeypatch: pytest.MonkeyPatch,
    table: type[PendingWriteRow | TombstoneRow],
) -> None:
    """Each failed claim in a row doubles the wait before the next, up to the cap."""
    monkeypatch.setattr(store, "_base_claim_retry_backoff_seconds", 60)
    monkeypatch.setattr(store, "_max_claim_retry_backoff_seconds", 150)
    partition = await store.open_or_create_partition(
        PARTITION_KEY, _plaintext_partition_config()
    )
    await _add_claimable_record(partition, table)

    for attempt, backoff_seconds in enumerate([60, 120, 150], start=1):
        claim = await _claim(store, table)
        assert claim is not None
        assert claim.attempt == attempt
        await _fail_claim(store, claim)
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
async def test_a_cancelled_claim_takes_back_its_attempt(
    store: SQLAlchemySegmentStore,
    table: type[PendingWriteRow | TombstoneRow],
) -> None:
    """A cancelled claim is not a failed one: its record is claimable at once, under the same attempt."""
    partition = await store.open_or_create_partition(
        PARTITION_KEY, _plaintext_partition_config()
    )
    record_uuid = await _add_claimable_record(partition, table)
    cancelled = await _claim(store, table)
    assert cancelled is not None

    await _cancel_claim(store, cancelled)

    again = await _claim(store, table)
    assert again is not None
    assert (_claimed_record_uuid(again), again.generation, again.attempt) == (
        record_uuid,
        cancelled.generation + 1,
        1,
    )


@pytest.mark.asyncio
@_CLAIMED_TABLES
async def test_cancelling_a_superseded_claim_changes_nothing(
    store: SQLAlchemySegmentStore,
    table: type[PendingWriteRow | TombstoneRow],
) -> None:
    partition = await store.open_or_create_partition(
        PARTITION_KEY, _plaintext_partition_config()
    )
    record_uuid = await _add_claimable_record(partition, table)
    superseded = await _claim(store, table)
    assert superseded is not None
    await _backdate(store, table.claimable_at, timedelta(days=1))
    assert await _claim(store, table) is not None
    scheduled = await _claimable_at(store, table, record_uuid)

    await _cancel_claim(store, superseded)

    assert await _claimable_at(store, table, record_uuid) == scheduled
    await _backdate(store, table.claimable_at, timedelta(days=1))
    third = await _claim(store, table)
    assert third is not None
    assert third.attempt == 3


@pytest.mark.asyncio
@_CLAIMED_TABLES
async def test_record_is_no_longer_claimed_after_the_maximum_failed_claims(
    store: SQLAlchemySegmentStore,
    caplog: pytest.LogCaptureFixture,
    table: type[PendingWriteRow | TombstoneRow],
) -> None:
    """The store reports the record once its claims failed the maximum number of times in a row."""
    partition = await store.open_or_create_partition(
        PARTITION_KEY, _plaintext_partition_config()
    )
    await _add_claimable_record(partition, table)
    maximum = sqlalchemy_segment_store._MAX_CLAIM_ATTEMPTS_WITHOUT_PROGRESS

    with caplog.at_level(logging.ERROR, logger=sqlalchemy_segment_store.__name__):
        for attempt in range(1, maximum + 1):
            claim = await _claim(store, table)
            assert claim is not None
            assert claim.attempt == attempt
            await _fail_claim(store, claim)
            await _backdate(store, table.claimable_at, timedelta(days=1))

    assert await _claim(store, table) is None
    reports = [
        record for record in caplog.records if "no longer claimed" in record.message
    ]
    assert [record.levelno for record in reports] == [logging.ERROR]


@pytest.mark.asyncio
@_CLAIMED_TABLES
async def test_claims_skip_deleted_partitions(
    store: SQLAlchemySegmentStore,
    table: type[PendingWriteRow | TombstoneRow],
) -> None:
    """A deleted partition's records are passed over, however old."""
    config = _plaintext_partition_config()
    doomed = await store.open_or_create_partition("doomed_claims", config)
    live = await store.open_or_create_partition("live_claims", config)
    live_uuid = await _add_claimable_record(live, table)
    await _add_claimable_record(doomed, table)
    await _backdate(
        store,
        table.claimable_at,
        timedelta(minutes=5),
        table.incarnation == doomed._incarnation,
    )
    await store.delete_partition("doomed_claims")

    claim = await _claim(store, table)
    assert claim is not None
    assert (claim.partition_key, _claimed_record_uuid(claim)) == (
        "live_claims",
        live_uuid,
    )
    assert await _claim(store, table) is None


@pytest.mark.asyncio
async def test_renewing_a_claim_holds_its_record_for_another_lease(
    store: SQLAlchemySegmentStore,
) -> None:
    partition = await store.open_or_create_partition(
        PARTITION_KEY, _plaintext_partition_config()
    )
    await partition.add_event_segments(*_events(_links(_seg())), lease_seconds=0)
    claim = await store.claim_pending_write(lease_seconds=60)
    assert claim is not None
    await _backdate(store, PendingWriteRow.claimable_at, timedelta(days=1))

    assert await store.renew_pending_write_claim(claim, lease_seconds=60)

    # The renewed lease holds the record again, and then the backoff.
    assert await store.claim_pending_write(lease_seconds=60) is None
    await _shift_earlier(store, PendingWriteRow.claimable_at, timedelta(seconds=80))
    assert await store.claim_pending_write(lease_seconds=60) is None
    await _shift_earlier(store, PendingWriteRow.claimable_at, timedelta(seconds=20))
    assert await store.claim_pending_write(lease_seconds=60) is not None


@pytest.mark.asyncio
async def test_a_superseded_claim_is_not_renewed(
    store: SQLAlchemySegmentStore,
) -> None:
    partition = await store.open_or_create_partition(
        PARTITION_KEY, _plaintext_partition_config()
    )
    await partition.add_event_segments(*_events(_links(_seg())), lease_seconds=0)
    superseded = await store.claim_pending_write(lease_seconds=60)
    assert superseded is not None
    await _backdate(store, PendingWriteRow.claimable_at, timedelta(days=1))
    latest = await store.claim_pending_write(lease_seconds=60)
    assert latest is not None

    assert not await store.renew_pending_write_claim(superseded, lease_seconds=60)
    assert await store.renew_pending_write_claim(latest, lease_seconds=60)


@pytest.mark.asyncio
async def test_a_claim_on_a_forgotten_event_is_not_renewed(
    store: SQLAlchemySegmentStore,
) -> None:
    partition = await store.open_or_create_partition(
        PARTITION_KEY, _plaintext_partition_config()
    )
    segment = _seg()
    await partition.add_event_segments(*_events(_links(segment)), lease_seconds=0)
    claim = await store.claim_pending_write(lease_seconds=60)
    assert claim is not None

    await partition.delete_event_segments([segment.event_uuid], lease_seconds=0)

    assert not await store.renew_pending_write_claim(claim, lease_seconds=60)


@pytest.mark.asyncio
async def test_a_claim_in_a_deleted_partition_is_not_renewed(
    store: SQLAlchemySegmentStore,
) -> None:
    config = _plaintext_partition_config()
    partition = await store.open_or_create_partition("claimed_incarnation", config)
    await partition.add_event_segments(*_events(_links(_seg())), lease_seconds=0)
    claim = await store.claim_pending_write(lease_seconds=60)
    assert claim is not None

    await store.delete_partition("claimed_incarnation")
    await store.open_or_create_partition("claimed_incarnation", config)

    assert not await store.renew_pending_write_claim(claim, lease_seconds=60)


# ===================================================================
# Tombstones
# ===================================================================


@pytest.mark.asyncio
async def test_new_tombstone_is_claimable_at_once(
    store: SQLAlchemySegmentStore,
) -> None:
    partition = await store.open_or_create_partition(
        PARTITION_KEY, _plaintext_partition_config()
    )
    segment = _seg()
    derivative_uuids = [uuid4(), uuid4()]
    await _add_segments(partition, {segment: derivative_uuids})
    [tombstone] = await partition.delete_event_segments(
        [segment.event_uuid], lease_seconds=0
    )

    claim = await store.claim_tombstone(lease_seconds=60)

    assert claim is not None
    assert claim.partition_key == PARTITION_KEY
    assert claim.tombstone.segment_uuid == tombstone.segment_uuid == segment.uuid
    assert sorted(claim.tombstone.derivative_uuids) == sorted(derivative_uuids)
    assert claim.second_delete is False
    assert (claim.generation, claim.attempt) == (1, 1)


@pytest.mark.asyncio
@pytest.mark.parametrize("recorded_by", ["partition", "claim"])
async def test_tombstone_awaits_the_retention_after_its_first_delete(
    store: SQLAlchemySegmentStore,
    monkeypatch: pytest.MonkeyPatch,
    recorded_by: str,
) -> None:
    """Its second delete falls due once the retention has passed since its first delete.

    Recording the first delete again, once the second is claimed, changes
    nothing.
    """
    monkeypatch.setattr(store, "_tombstone_retention_seconds", 3600)
    partition = await store.open_or_create_partition(
        PARTITION_KEY, _plaintext_partition_config()
    )
    tombstone_uuid = await _add_claimable_record(partition, TombstoneRow)
    if recorded_by == "partition":
        await partition.record_tombstone_first_delete([tombstone_uuid])
    else:
        first = await store.claim_tombstone(lease_seconds=60)
        assert first is not None
        await store.finish_tombstone_claim(first)

    for age in (timedelta(0), timedelta(minutes=50)):
        await _backdate(store, TombstoneRow.first_deleted_at, age)
        assert await store.claim_tombstone(lease_seconds=60) is None
    await _backdate(store, TombstoneRow.first_deleted_at, timedelta(minutes=70))
    second = await store.claim_tombstone(lease_seconds=60)
    assert second is not None
    assert (second.tombstone.segment_uuid, second.second_delete, second.attempt) == (
        tombstone_uuid,
        True,
        1,
    )

    await partition.record_tombstone_first_delete([tombstone_uuid])
    assert await store.claim_tombstone(lease_seconds=60) is None


@pytest.mark.asyncio
async def test_a_forget_holds_its_tombstones_until_its_lease_passes(
    store: SQLAlchemySegmentStore,
) -> None:
    """The background work takes a first delete over only once the forget's own has had its lease."""
    partition = await store.open_or_create_partition(
        PARTITION_KEY, _plaintext_partition_config()
    )
    segment = _seg()
    await _add_segments(partition, _links(segment))
    await partition.delete_event_segments([segment.event_uuid], lease_seconds=60)

    assert await store.claim_tombstone(lease_seconds=60) is None
    await _backdate(store, TombstoneRow.claimable_at, timedelta(seconds=61))
    claim = await store.claim_tombstone(lease_seconds=60)
    assert claim is not None
    assert (claim.tombstone.segment_uuid, claim.second_delete) == (segment.uuid, False)


@pytest.mark.asyncio
async def test_finishing_a_second_delete_claim_removes_the_tombstone(
    store: SQLAlchemySegmentStore,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(store, "_tombstone_retention_seconds", 3600)
    partition = await store.open_or_create_partition(
        PARTITION_KEY, _plaintext_partition_config()
    )
    tombstone_uuid = await _add_claimable_record(partition, TombstoneRow)
    await partition.record_tombstone_first_delete([tombstone_uuid])
    await _backdate(store, TombstoneRow.first_deleted_at, timedelta(hours=2))
    claim = await store.claim_tombstone(lease_seconds=60)
    assert claim is not None
    assert claim.second_delete is True
    holds = TombstoneDerivativeRow.segment_uuid == tombstone_uuid
    assert await _count_rows(store, holds) == 1

    await store.finish_tombstone_claim(claim)

    assert await _count_rows(store, TombstoneRow.segment_uuid == tombstone_uuid) == 0
    assert await _count_rows(store, holds) == 0
    assert await store.claim_tombstone(lease_seconds=60) is None


async def _claim_second_delete(
    store: SQLAlchemySegmentStore,
    partition: SQLAlchemySegmentStorePartition,
) -> tuple[UUID, TombstoneClaim]:
    """Leave a tombstone due for its second delete, claim that delete, and return its segment's UUID and the claim."""
    tombstone_uuid = await _add_claimable_record(partition, TombstoneRow)
    await partition.record_tombstone_first_delete([tombstone_uuid])
    await _backdate(store, TombstoneRow.first_deleted_at, timedelta(days=2))
    claim = await store.claim_tombstone(lease_seconds=60)
    assert claim is not None
    assert claim.second_delete
    return tombstone_uuid, claim


@pytest.mark.asyncio
async def test_a_second_delete_in_a_deleted_partition_leaves_the_tombstone_to_the_purge(
    store: SQLAlchemySegmentStore,
) -> None:
    partition = await store.open_or_create_partition(
        PARTITION_KEY, _plaintext_partition_config()
    )
    tombstone_uuid, claim = await _claim_second_delete(store, partition)
    await store.delete_partition(PARTITION_KEY)

    await store.finish_tombstone_claim(claim)

    assert await _count_rows(store, TombstoneRow.segment_uuid == tombstone_uuid) == 1
    while await store.purge_deleted_partitions():
        pass
    assert await _count_rows(store, TombstoneRow.segment_uuid == tombstone_uuid) == 0
    assert (
        await _count_rows(store, TombstoneDerivativeRow.segment_uuid == tombstone_uuid)
        == 0
    )


@pytest.mark.integration
@pytest.mark.asyncio
async def test_a_second_delete_waits_for_a_deletion_of_its_partition(
    pg_store: SQLAlchemySegmentStore,
    sqlalchemy_pg_engine: AsyncEngine,
) -> None:
    """It pins the partition as a partition's writes do, so it never runs beside the purge."""
    partition = await pg_store.open_or_create_partition(
        PARTITION_KEY, _plaintext_partition_config()
    )
    tombstone_uuid, claim = await _claim_second_delete(pg_store, partition)

    async with (
        partition._create_session() as remote_session,
        remote_session.begin(),
    ):
        # As a deletion of the partition holds the row until it commits.
        await remote_session.execute(
            select(PartitionRow)
            .where(PartitionRow.incarnation == partition._incarnation)
            .with_for_update()
        )
        finish = asyncio.create_task(pg_store.finish_tombstone_claim(claim))
        assert await _wait_until_blocked_or_done(sqlalchemy_pg_engine, finish) == (
            "blocked"
        )
    await finish

    assert await _count_rows(pg_store, TombstoneRow.segment_uuid == tombstone_uuid) == 0


@pytest.mark.integration
@pytest.mark.asyncio
async def test_recording_a_first_delete_skips_a_tombstone_another_transaction_holds(
    pg_store: SQLAlchemySegmentStore,
    sqlalchemy_pg_engine: AsyncEngine,
) -> None:
    """The record never waits: a held tombstone is left for its claim to record."""
    partition = await pg_store.open_or_create_partition(
        PARTITION_KEY, _plaintext_partition_config()
    )
    held = await _add_claimable_record(partition, TombstoneRow)
    free = await _add_claimable_record(partition, TombstoneRow)

    async with (
        partition._create_session() as remote_session,
        remote_session.begin(),
    ):
        await remote_session.execute(
            select(TombstoneRow)
            .where(TombstoneRow.segment_uuid == held)
            .with_for_update()
        )
        record = asyncio.create_task(
            partition.record_tombstone_first_delete([held, free])
        )
        assert await _wait_until_blocked_or_done(sqlalchemy_pg_engine, record) == (
            "done"
        )
        await record

    recorded = TombstoneRow.first_deleted_at.is_not(None)
    assert (
        await _count_rows(pg_store, (TombstoneRow.segment_uuid == held) & recorded) == 0
    )
    assert (
        await _count_rows(pg_store, (TombstoneRow.segment_uuid == free) & recorded) == 1
    )


@pytest.mark.asyncio
async def test_no_schedule_brings_a_second_delete_before_the_retention(
    store: SQLAlchemySegmentStore,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The retention is its own condition: a tombstone scheduled as claimable
    still waits for the retention since its first delete."""
    monkeypatch.setattr(store, "_tombstone_retention_seconds", 3600)
    partition = await store.open_or_create_partition(
        PARTITION_KEY, _plaintext_partition_config()
    )
    tombstone_uuid = await _add_claimable_record(partition, TombstoneRow)
    await partition.record_tombstone_first_delete([tombstone_uuid])
    await _backdate(store, TombstoneRow.claimable_at, timedelta(days=1))

    await _backdate(store, TombstoneRow.first_deleted_at, timedelta(minutes=50))
    assert await store.claim_tombstone(lease_seconds=60) is None
    await _backdate(store, TombstoneRow.first_deleted_at, timedelta(minutes=70))
    second = await store.claim_tombstone(lease_seconds=60)
    assert second is not None
    assert second.second_delete is True


@pytest.mark.asyncio
@pytest.mark.parametrize("recorded_by", ["partition", "an_earlier_claim"])
async def test_recording_the_first_delete_ends_an_open_claim(
    store: SQLAlchemySegmentStore,
    monkeypatch: pytest.MonkeyPatch,
    recorded_by: str,
) -> None:
    """A claim on the first delete that fails after the delete was recorded,
    by the partition or by a claim its own superseded, changes nothing, so
    the second delete keeps waiting for the retention."""
    monkeypatch.setattr(store, "_tombstone_retention_seconds", 3600)
    partition = await store.open_or_create_partition(
        PARTITION_KEY, _plaintext_partition_config()
    )
    tombstone_uuid = await _add_claimable_record(partition, TombstoneRow)
    earlier = await store.claim_tombstone(lease_seconds=60)
    assert earlier is not None
    await _backdate(store, TombstoneRow.claimable_at, timedelta(days=1))
    open_claim = await store.claim_tombstone(lease_seconds=60)
    assert open_claim is not None
    if recorded_by == "partition":
        await partition.record_tombstone_first_delete([tombstone_uuid])
    else:
        await store.finish_tombstone_claim(earlier)

    await _fail_claim(store, open_claim)

    assert await _claimable_at(store, TombstoneRow, tombstone_uuid) is None
    assert await store.claim_tombstone(lease_seconds=60) is None


@pytest.mark.asyncio
@_CLAIMED_TABLES
async def test_a_claim_that_never_ends_is_claimed_again_once_its_lease_and_the_backoff_pass(
    store: SQLAlchemySegmentStore,
    table: type[PendingWriteRow | TombstoneRow],
) -> None:
    partition = await store.open_or_create_partition(
        PARTITION_KEY, _plaintext_partition_config()
    )
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
        partition = await store.open_or_create_partition(
            "timed_p", _plaintext_partition_config()
        )
        await _add_claimable_record(partition, TombstoneRow)
        failed = await store.claim_tombstone(lease_seconds=60)
        assert failed is not None
        await _fail_claim(store, failed)
        # Past the configured backoff, not the default's.
        await _shift_earlier(store, TombstoneRow.claimable_at, timedelta(seconds=20))
        first = await store.claim_tombstone(lease_seconds=60)
        assert first is not None
        await store.finish_tombstone_claim(first)

        await _backdate(store, TombstoneRow.first_deleted_at, timedelta(minutes=2))
        second = await store.claim_tombstone(lease_seconds=60)
        assert second is not None
        assert second.second_delete is True
    finally:
        async with sqlalchemy_sqlite_engine.begin() as conn:
            await conn.run_sync(BaseSegmentStore.metadata.drop_all)


async def _add_claimable_record(
    partition: SQLAlchemySegmentStorePartition,
    table: type[PendingWriteRow | TombstoneRow],
) -> UUID:
    """Leave a claimable record of `table` in the partition, and return its segment's UUID.

    A pending write is an added segment's, left pending; a tombstone is a
    forgotten segment's.
    """
    segment = _seg()
    segments_by_event, derivatives = _events(_links(segment))
    await partition.add_event_segments(segments_by_event, derivatives, lease_seconds=0)
    if table is TombstoneRow:
        assert (
            await partition.delete_event_segments([segment.event_uuid], lease_seconds=0)
            != []
        )
    return segment.uuid


async def _claim(
    store: SQLAlchemySegmentStore,
    table: type[PendingWriteRow | TombstoneRow],
) -> PendingWriteClaim | TombstoneClaim | None:
    """Claim a record of `table`, taking pending writes at any age."""
    if table is PendingWriteRow:
        return await store.claim_pending_write(lease_seconds=60)
    return await store.claim_tombstone(lease_seconds=60)


def _claimed_record_uuid(
    claim: PendingWriteClaim | TombstoneClaim | None,
) -> UUID | None:
    """The UUID of the segment whose record `claim` holds; None for no claim."""
    if isinstance(claim, PendingWriteClaim):
        return claim.segment_uuid
    if isinstance(claim, TombstoneClaim):
        return claim.tombstone.segment_uuid
    return None


async def _fail_claim(
    store: SQLAlchemySegmentStore,
    claim: PendingWriteClaim | TombstoneClaim,
) -> None:
    """End `claim` as failed."""
    error = RuntimeError("index unavailable")
    if isinstance(claim, PendingWriteClaim):
        await store.fail_pending_write_claim(claim, error)
    else:
        await store.fail_tombstone_claim(claim, error)


async def _cancel_claim(
    store: SQLAlchemySegmentStore,
    claim: PendingWriteClaim | TombstoneClaim,
) -> None:
    """End `claim` as cancelled."""
    if isinstance(claim, PendingWriteClaim):
        await store.cancel_pending_write_claim(claim)
    else:
        await store.cancel_tombstone_claim(claim)


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
    table: type[PendingWriteRow | TombstoneRow],
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
    return await store.open_or_create_partition(
        PARTITION_KEY,
        _plaintext_partition_config(),
    )


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
    assert await _row_counts(part, part._incarnation) == (50, 50)


@pytest.mark.asyncio
async def test_concurrent_reads_during_writes(
    store: SQLAlchemySegmentStore,
) -> None:
    """Reads should not fail or block indefinitely while writes are happening."""
    engine = store._engine
    partition = await store.open_or_create_partition(
        PARTITION_KEY,
        _plaintext_partition_config(),
    )

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
    partition = await store.open_or_create_partition(
        PARTITION_KEY,
        _plaintext_partition_config(),
    )

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
            await part.delete_event_segments([ep], lease_seconds=0)
            await asyncio.sleep(0.02)

    await asyncio.gather(context_reader(), event_forgetter())

    assert errors == []


# ===================================================================
# Partition lifecycle
# ===================================================================


@pytest.mark.asyncio
async def test_open_or_create_partition_defaults_to_plaintext_config(
    store: SQLAlchemySegmentStore,
) -> None:
    partition = await store.open_or_create_partition(
        "plaintext_default",
        _plaintext_partition_config(),
    )
    assert partition.config.payload_codec_config == PlaintextPayloadCodecConfig()


@pytest.mark.asyncio
async def test_create_partition(store: SQLAlchemySegmentStore) -> None:
    await store.create_partition("new_partition", _plaintext_partition_config())
    partition = await store.open_partition("new_partition")
    assert partition is not None


@pytest.mark.asyncio
async def test_create_partition_already_exists(store: SQLAlchemySegmentStore) -> None:
    await store.create_partition("dup_partition", _plaintext_partition_config())
    with pytest.raises(SegmentStorePartitionAlreadyExistsError):
        await store.create_partition("dup_partition", _plaintext_partition_config())


@pytest.mark.asyncio
async def test_open_partition_nonexistent(store: SQLAlchemySegmentStore) -> None:
    result = await store.open_partition("nonexistent")
    assert result is None


@pytest.mark.asyncio
async def test_open_partition_existing(store: SQLAlchemySegmentStore) -> None:
    await store.create_partition("existing", _plaintext_partition_config())
    partition = await store.open_partition("existing")
    assert partition is not None


@pytest.mark.asyncio
async def test_open_or_create_partition_creates(store: SQLAlchemySegmentStore) -> None:
    partition = await store.open_or_create_partition(
        "fresh",
        _plaintext_partition_config(),
    )
    assert partition is not None
    # Verify it was actually created.
    opened = await store.open_partition("fresh")
    assert opened is not None


@pytest.mark.asyncio
async def test_open_or_create_partition_idempotent(
    store: SQLAlchemySegmentStore,
) -> None:
    await store.create_partition("idem", _plaintext_partition_config())
    partition = await store.open_or_create_partition(
        "idem",
        _plaintext_partition_config(),
    )
    assert partition is not None


@pytest.mark.asyncio
async def test_delete_partition_removes_data(store: SQLAlchemySegmentStore) -> None:
    partition = await store.open_or_create_partition(
        "to_delete",
        _plaintext_partition_config(),
    )
    seg = _seg()
    await _add_segments(partition, _links(seg))

    await store.delete_partition("to_delete")

    # Partition no longer exists.
    assert await store.open_partition("to_delete") is None


@pytest.mark.integration
@pytest.mark.asyncio
async def test_delete_partition_keeps_foreign_key_enforced(
    pg_store: SQLAlchemySegmentStore,
    sqlalchemy_pg_engine: AsyncEngine,
) -> None:
    """Deleting one partition must not drop the segments' foreign key to events."""
    keeper = await pg_store.open_or_create_partition(
        "keeper_fk", _plaintext_partition_config()
    )
    await pg_store.open_or_create_partition("doomed_fk", _plaintext_partition_config())

    await pg_store.delete_partition("doomed_fk")

    # A segment pointing at no event must still be rejected in the
    # partition that survived.
    with pytest.raises(IntegrityError, match="foreign key"):
        async with sqlalchemy_pg_engine.begin() as connection:
            await connection.execute(
                text(
                    "INSERT INTO segment_store_sg"
                    ' (incarnation, uuid, event_uuid, "index", "offset", timestamp,'
                    " timestamp_timezone_offset, context, block, properties)"
                    " VALUES (:incarnation, :uuid, :event_uuid, 0, 0, now(), 0,"
                    " :payload, :payload, '{}')"
                ),
                {
                    "incarnation": keeper._incarnation,
                    "uuid": uuid4(),
                    "event_uuid": uuid4(),
                    "payload": b"{}",
                },
            )


@pytest.mark.asyncio
async def test_delete_partition_keeps_other_partitions_cascading(
    store: SQLAlchemySegmentStore,
) -> None:
    """Deleting one partition must not disable the cascade from events to segments."""
    keeper = await store.open_or_create_partition(
        "keeper",
        _plaintext_partition_config(),
    )
    await store.open_or_create_partition(
        "doomed",
        _plaintext_partition_config(),
    )
    segment = _seg()
    derivative_uuid = uuid4()
    await _add_segments(keeper, {segment: [derivative_uuid]})

    await store.delete_partition("doomed")

    # The foreign keys from segments to events and from derivatives to
    # segments must still cascade for the partitions that were not deleted.
    await keeper.delete_event_segments([segment.event_uuid], lease_seconds=0)
    assert await _row_counts(keeper, keeper._incarnation) == (0, 0)


@pytest.mark.asyncio
async def test_delete_partition_cascades_segments(
    store: SQLAlchemySegmentStore,
) -> None:
    partition = await store.open_or_create_partition(
        "cascade_test",
        _plaintext_partition_config(),
    )
    seg = _seg()
    d1 = uuid4()
    await _add_segments(partition, {seg: [d1]})

    await store.delete_partition("cascade_test")

    # Re-create the partition and verify data is gone.
    new_partition = await store.open_or_create_partition(
        "cascade_test",
        _plaintext_partition_config(),
    )
    result = await new_partition.get_segment_contexts([seg.uuid])
    assert result == {}
    deriv_result = await new_partition.get_segment_uuids_by_derivative_uuids([d1])
    assert deriv_result == {}


@pytest.mark.asyncio
async def test_delete_partition_idempotent(store: SQLAlchemySegmentStore) -> None:
    await store.delete_partition("never_existed")


@pytest.mark.asyncio
async def test_partition_key_validation_invalid_chars(
    store: SQLAlchemySegmentStore,
) -> None:
    with pytest.raises(ValueError, match="invalid characters"):
        await store.create_partition("UPPER", _plaintext_partition_config())
    with pytest.raises(ValueError, match="invalid characters"):
        await store.create_partition("has-hyphen", _plaintext_partition_config())
    with pytest.raises(ValueError, match="invalid characters"):
        await store.create_partition("has space", _plaintext_partition_config())


@pytest.mark.asyncio
async def test_partition_key_validation_too_long(
    store: SQLAlchemySegmentStore,
) -> None:
    with pytest.raises(ValueError, match="too long"):
        await store.create_partition("a" * 33, _plaintext_partition_config())


@pytest.mark.integration
@pytest.mark.asyncio
async def test_pg_context_preserved_via_lateral_join(
    pg_store: SQLAlchemySegmentStore,
) -> None:
    """Context is preserved when retrieved via the LATERAL join path (multiple seeds)."""
    partition = await pg_store.open_or_create_partition(
        PARTITION_KEY,
        _plaintext_partition_config(),
    )
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
    partition = await pg_store.open_or_create_partition(
        PARTITION_KEY,
        _plaintext_partition_config(),
    )
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


# ===================================================================
# Incarnation fencing and O(1) deletion
# ===================================================================


@pytest.mark.asyncio
async def test_stale_handle_raises_after_delete(
    store: SQLAlchemySegmentStore,
) -> None:
    """A handle held across deletion must fail loudly, not act."""
    partition = await store.open_or_create_partition(
        "fenced", _plaintext_partition_config()
    )
    seg = _seg()
    await _add_segments(partition, _links(seg))

    await store.delete_partition("fenced")

    # Empty input may skip the handle check (the ABC-permitted shortcut).
    await partition.add_event_segments({}, [], lease_seconds=0)
    assert await partition.delete_event_segments([], lease_seconds=0) == []

    with pytest.raises(SegmentStorePartitionHandleStaleError):
        await partition.add_event_segments(*_events(_links(_seg())), lease_seconds=0)
    with pytest.raises(SegmentStorePartitionHandleStaleError):
        await partition.delete_pending_writes([uuid4()])
    with pytest.raises(SegmentStorePartitionHandleStaleError):
        await partition.get_segment_contexts([seg.uuid])
    with pytest.raises(SegmentStorePartitionHandleStaleError):
        await partition.get_segment_uuids_by_derivative_uuids([uuid4()])
    with pytest.raises(SegmentStorePartitionHandleStaleError):
        await partition.delete_event_segments([seg.event_uuid], lease_seconds=0)
    with pytest.raises(SegmentStorePartitionHandleStaleError):
        await partition.record_tombstone_first_delete([uuid4()])
    # The stale handle deleted nothing; the rows wait for the purge.
    assert await _row_counts(store, partition._incarnation) == (1, 1)


@pytest.mark.integration
@pytest.mark.asyncio
async def test_reads_check_liveness_inside_the_data_statement(
    pg_store: SQLAlchemySegmentStore,
    recorded_statements: list[str],
) -> None:
    """A read that finds rows is one statement; the registry check rides in it.

    Only a read that finds nothing pays a second statement, to tell an
    empty partition from a stale handle.
    """
    partition = await pg_store.open_or_create_partition(
        "folded", _plaintext_partition_config()
    )
    seg = _seg()
    derivative_uuids = [uuid4()]
    await _add_segments(partition, {seg: derivative_uuids})
    recorded_statements.clear()

    await partition.get_segment_contexts([seg.uuid])
    await partition.get_segment_uuids_by_derivative_uuids(derivative_uuids)
    assert len(recorded_statements) == 2
    assert all(
        "EXISTS (SELECT" in s and "segment_store_pt" in s for s in recorded_statements
    )

    recorded_statements.clear()
    assert await partition.get_segment_contexts([uuid4()]) == {}
    assert len(recorded_statements) == 2


@pytest.mark.asyncio
async def test_stale_handle_raises_after_recreate(
    store: SQLAlchemySegmentStore,
) -> None:
    """Re-creating the key must not let an old handle act on the successor."""
    old_handle = await store.open_or_create_partition(
        "reborn", _plaintext_partition_config()
    )
    await store.delete_partition("reborn")
    new_handle = await store.open_or_create_partition(
        "reborn", _plaintext_partition_config()
    )

    with pytest.raises(SegmentStorePartitionHandleStaleError):
        await _add_segments(old_handle, _links(_seg()))
    await _add_segments(new_handle, _links(_seg()))  # the live handle works


@pytest.mark.asyncio
async def test_recreated_partition_is_isolated_from_old_rows(
    store: SQLAlchemySegmentStore,
) -> None:
    """Old-incarnation rows are invisible to the successor before purging."""
    partition = await store.open_or_create_partition(
        "isolated", _plaintext_partition_config()
    )
    seg = _seg()
    await _add_segments(partition, _links(seg))
    old_incarnation = partition._incarnation

    await store.delete_partition("isolated")
    successor = await store.open_or_create_partition(
        "isolated", _plaintext_partition_config()
    )

    assert await successor.get_segment_contexts([seg.uuid]) == {}
    # The old rows still physically exist until the purger reclaims them.
    async with successor._create_session() as session:
        remaining = (
            await session.execute(
                select(func.count())
                .select_from(SegmentRow)
                .where(SegmentRow.incarnation == old_incarnation)
            )
        ).scalar_one()
    assert remaining == 1


@pytest.mark.asyncio
async def test_purge_reclaims_only_dead_incarnations(
    store: SQLAlchemySegmentStore,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Purging erases queued incarnations and leaves live partitions alone."""
    # Small default bound, so draining takes the caller's loop.
    monkeypatch.setattr(store, "_purge_max_rows", 2)

    live = await store.open_or_create_partition("live_p", _plaintext_partition_config())
    live_seg = _seg()
    await _add_segments(live, _links(live_seg))

    doomed = await store.open_or_create_partition(
        "doomed_p", _plaintext_partition_config()
    )
    doomed_incarnation = doomed._incarnation
    await _add_segments(doomed, _links(_seg(), _seg(), _seg()))
    await store.delete_partition("doomed_p")

    calls = 0
    while await store.purge_deleted_partitions():
        calls += 1
        assert calls < 10
    assert calls > 0

    async with live._create_session() as session:
        dead_rows = (
            await session.execute(
                select(func.count())
                .select_from(SegmentRow)
                .where(SegmentRow.incarnation == doomed_incarnation)
            )
        ).scalar_one()
        queue_depth = (
            await session.execute(select(func.count()).select_from(PurgeQueueRow))
        ).scalar_one()
    assert dead_rows == 0
    assert queue_depth == 0
    assert await _row_counts(store, doomed_incarnation) == (0, 0)
    assert await _row_counts(store, live._incarnation) == (1, 1)
    assert (await live.get_segment_contexts([live_seg.uuid]))[live_seg.uuid]


@pytest.mark.asyncio
async def test_concurrent_purges_reclaim_everything(
    store: SQLAlchemySegmentStore,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Racing purgers neither error nor leave garbage behind.

    Queue-entry claiming partitions the work: whatever the interleaving,
    every dead incarnation is reclaimed exactly once and the drain loops
    all terminate cleanly.
    """
    monkeypatch.setattr(store, "_purge_max_rows", 2)
    incarnations = []
    for index in range(3):
        partition = await store.open_or_create_partition(
            f"gc_race_{index}", _plaintext_partition_config()
        )
        incarnations.append(partition._incarnation)
        await _add_segments(partition, _links(_seg(), _seg(), _seg()))
        await store.delete_partition(f"gc_race_{index}")

    async def drain() -> None:
        while await store.purge_deleted_partitions():
            pass

    await asyncio.wait_for(asyncio.gather(drain(), drain(), drain()), 60)

    async with partition._create_session() as session:
        dead_rows = (
            await session.execute(
                select(func.count())
                .select_from(SegmentRow)
                .where(SegmentRow.incarnation.in_(incarnations))
            )
        ).scalar_one()
        queue_depth = (
            await session.execute(select(func.count()).select_from(PurgeQueueRow))
        ).scalar_one()
    assert dead_rows == 0
    assert queue_depth == 0
    for incarnation in incarnations:
        assert await _row_counts(store, incarnation) == (0, 0)


@pytest.mark.asyncio
async def test_racing_purgers_never_retire_a_partly_purged_entry(
    store: SQLAlchemySegmentStore,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A purger racing another never retires rows it did not reach.

    The first purger to delete a batch is held after its DELETE, before it
    commits, until a second purger reads segment rows (or for a second).
    A second purger that planned its batch from the entry as it stood
    before the first committed would delete nothing and retire the entry
    with rows left. PostgreSQL's claim lock makes it skip the entry; on
    SQLite the call holds the write lock from BEGIN, so the second waits
    until the first commits.
    """
    partition = await store.open_or_create_partition(
        "raced", _plaintext_partition_config()
    )
    await _add_segments(
        partition, {_seg(ts_offset_seconds=index): [] for index in range(6)}
    )
    incarnation = partition._incarnation
    await store.delete_partition("raced")
    monkeypatch.setattr(store, "_purge_max_rows", 2)

    held: list[AsyncConnection] = []
    second_read = asyncio.Event()
    original_execute = AsyncConnection.execute

    async def interleaving_execute(self, statement, *args, **kwargs):
        result = await original_execute(self, statement, *args, **kwargs)
        if (
            isinstance(statement, Delete)
            and statement.table.name == SegmentRow.__tablename__
        ):
            if not held:
                held.append(self)
                with contextlib.suppress(TimeoutError):
                    await asyncio.wait_for(second_read.wait(), 1)
        elif held and self is not held[0] and "segment_store_sg" in str(statement):
            second_read.set()
        return result

    monkeypatch.setattr(AsyncConnection, "execute", interleaving_execute)
    await asyncio.wait_for(
        asyncio.gather(
            store.purge_deleted_partitions(), store.purge_deleted_partitions()
        ),
        30,
    )
    monkeypatch.setattr(AsyncConnection, "execute", original_execute)
    assert held, "no purger was held after deleting a batch"
    while await store.purge_deleted_partitions():
        pass

    assert await _row_counts(store, incarnation) == (0, 0)
    async with store._create_session() as session:
        queue_depth = (
            await session.execute(select(func.count()).select_from(PurgeQueueRow))
        ).scalar_one()
    assert queue_depth == 0


@pytest.mark.asyncio
async def test_failed_purge_call_leaves_nothing_half_done(
    store: SQLAlchemySegmentStore,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A purge call whose commit fails changes nothing; later calls finish.

    A call's deletions and its record of how far it got commit together or
    not at all, so no row is ever skipped after a failure.
    """
    partition = await store.open_or_create_partition(
        "interrupted", _plaintext_partition_config()
    )
    await _add_segments(
        partition, _links(*(_seg(ts_offset_seconds=index) for index in range(5)))
    )
    incarnation = partition._incarnation
    await store.delete_partition("interrupted")
    monkeypatch.setattr(store, "_purge_max_rows", 2)

    with (
        _failing_next_commit(store._engine),
        pytest.raises(RuntimeError, match="injected commit failure"),
    ):
        await store.purge_deleted_partitions()
    assert await _row_counts(store, incarnation) == (5, 5)

    while await store.purge_deleted_partitions():
        pass
    assert await _row_counts(store, incarnation) == (0, 0)


@pytest.mark.asyncio
async def test_purge_reclaims_every_size_within_the_bound(
    store: SQLAlchemySegmentStore,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Every incarnation size is reclaimed whole, never over the bound per call.

    Sizes straddle the bound: empty, below it, exactly it, just over it,
    and an exact multiple, where the last call finds nothing left.
    """
    bound = 3
    sizes = [0, 1, bound - 1, bound, bound + 1, 2 * bound]
    incarnations = []
    for index, size in enumerate(sizes):
        partition = await store.open_or_create_partition(
            f"sized_{index}", _plaintext_partition_config()
        )
        if size:
            await _add_segments(
                partition,
                _links(*(_seg(ts_offset_seconds=offset) for offset in range(size))),
            )
        incarnations.append(partition._incarnation)
        await store.delete_partition(f"sized_{index}")
    monkeypatch.setattr(store, "_purge_max_rows", bound)

    async def dead_segments() -> int:
        return sum(
            [(await _row_counts(store, incarnation))[0] for incarnation in incarnations]
        )

    left = await dead_segments()
    for _ in range(50):
        more = await store.purge_deleted_partitions()
        now = await dead_segments()
        assert 0 <= left - now <= bound
        left = now
        if not more:
            break
    else:
        pytest.fail("the purge did not finish within 50 calls")
    for incarnation in incarnations:
        assert await _row_counts(store, incarnation) == (0, 0)


@pytest.mark.asyncio
async def test_purge_reclaims_every_row_of_a_deleted_partition(
    store: SQLAlchemySegmentStore,
) -> None:
    """Events, segments, derivatives, pending writes, tombstones, and their derivative UUIDs are all deleted; a live partition keeps its own."""
    config = _plaintext_partition_config()
    doomed = await store.open_or_create_partition("doomed_rows", config)
    live = await store.open_or_create_partition("live_rows", config)
    for partition in (doomed, live):
        kept, pending_segment, forgotten = _seg(), _seg(), _seg()
        await _add_segments(partition, _links(kept, forgotten))
        segments_by_event, derivatives = _events(_links(pending_segment))
        await partition.add_event_segments(
            segments_by_event, derivatives, lease_seconds=0
        )
        assert (
            await partition.delete_event_segments(
                [forgotten.event_uuid], lease_seconds=0
            )
            != []
        )
    await store.delete_partition("doomed_rows")

    while await store.purge_deleted_partitions():
        pass

    async def row_counts(incarnation: UUID) -> list[int]:
        return [
            await _count_rows(store, condition)
            for condition in (
                EventRow.incarnation == incarnation,
                SegmentRow.incarnation == incarnation,
                DerivativeRow.incarnation == incarnation,
                PendingWriteRow.incarnation == incarnation,
                TombstoneRow.incarnation == incarnation,
                TombstoneDerivativeRow.incarnation == incarnation,
            )
        ]

    assert await row_counts(doomed._incarnation) == [0, 0, 0, 0, 0, 0]
    # The forgotten segment's tombstone holds its derivative's UUID.
    assert await row_counts(live._incarnation) == [2, 2, 2, 1, 1, 1]
    assert (
        await _count_rows(store, PurgeQueueRow.incarnation == doomed._incarnation) == 0
    )


@pytest.mark.asyncio
async def test_purge_ends_a_table_only_when_its_search_finds_fewer_than_the_bound(
    store: SQLAlchemySegmentStore,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A batch another transaction deletes before the purge's delete leaves the entry for the next call, not rows behind it."""
    partition = await store.open_or_create_partition(
        "raced_p", _plaintext_partition_config()
    )
    incarnation = partition._incarnation
    forgotten = [_seg() for _ in range(3)]
    for segment in forgotten:
        await _add_segments(partition, {segment: []})
        await partition.delete_event_segments([segment.event_uuid], lease_seconds=0)
    await store.delete_partition("raced_p")
    monkeypatch.setattr(store, "_purge_max_rows", 2)
    batch = sorted(segment.uuid for segment in forgotten)[:2]

    with _deleted_before_the_next_tombstone_delete(store._engine, batch):
        assert await store.purge_deleted_partitions() is True

    while await store.purge_deleted_partitions():
        pass
    assert await _count_rows(store, TombstoneRow.incarnation == incarnation) == 0
    assert await _count_rows(store, PurgeQueueRow.incarnation == incarnation) == 0


@contextlib.contextmanager
def _deleted_before_the_next_tombstone_delete(
    engine: AsyncEngine, segment_uuids: list[UUID]
) -> Generator[None, None, None]:
    """Delete the given tombstones just before the engine's next delete of tombstone rows runs, as a concurrent delete would."""
    deleted: list[bool] = []

    def _delete_first(connection, clauseelement, *_args) -> None:
        if deleted or not (
            isinstance(clauseelement, Delete)
            and clauseelement.table.name == TombstoneRow.__tablename__
        ):
            return
        deleted.append(True)
        connection.execute(
            delete(TombstoneRow).where(TombstoneRow.segment_uuid.in_(segment_uuids))
        )

    event.listen(engine.sync_engine, "before_execute", _delete_first)
    try:
        yield
    finally:
        event.remove(engine.sync_engine, "before_execute", _delete_first)
    assert deleted, "no tombstone delete ran"


@pytest.mark.asyncio
@pytest.mark.parametrize("bound", [1, 2, 3, 4, 5, 7, 13])
async def test_purge_continues_through_every_table_within_the_bound(
    store: SQLAlchemySegmentStore,
    monkeypatch: pytest.MonkeyPatch,
    bound: int,
) -> None:
    """Derivatives, segments, events, tombstones' derivative UUIDs, and tombstones share each call's bound, and each table resumes where the last call left it.

    The partition holds five derivatives, five segments, three events, and
    two tombstones holding a derivative UUID each, so the bounds end calls
    inside each table and exactly on each table's last row. Deleting a
    segment deletes its pending write by cascade.
    """
    partition = await store.open_or_create_partition(
        "tables_p", _plaintext_partition_config()
    )
    incarnation = partition._incarnation
    event_uuid = uuid4()
    await _add_segments(
        partition,
        _links(*(_seg(event_uuid=event_uuid, index=index) for index in range(3))),
    )
    await _add_segments(partition, _links(_seg()))
    await partition.add_event_segments(*_events(_links(_seg())), lease_seconds=0)
    for forgotten in (_seg(), _seg()):
        await _add_segments(partition, _links(forgotten))
        await partition.delete_event_segments([forgotten.event_uuid], lease_seconds=0)
    await store.delete_partition("tables_p")
    monkeypatch.setattr(store, "_purge_max_rows", bound)

    async def budgeted_rows() -> int:
        return sum(
            [
                await _count_rows(store, table.incarnation == incarnation)
                for table in (
                    DerivativeRow,
                    SegmentRow,
                    EventRow,
                    TombstoneDerivativeRow,
                    TombstoneRow,
                )
            ]
        )

    left = await budgeted_rows()
    assert left == 17
    for _ in range(20):
        more = await store.purge_deleted_partitions()
        now = await budgeted_rows()
        assert left - now <= bound
        assert left - now > 0 or not more
        left = now
        if not more:
            break
    else:
        pytest.fail("the purge did not finish within 20 calls")
    assert left == 0
    assert await _row_counts(store, incarnation) == (0, 0)
    assert await _count_rows(store, PendingWriteRow.incarnation == incarnation) == 0
    assert await _count_rows(store, PurgeQueueRow.incarnation == incarnation) == 0


@pytest.mark.asyncio
async def test_purging_a_recreated_key_reclaims_only_the_dead_incarnation(
    store: SQLAlchemySegmentStore,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A successor keeps every row it writes while its predecessor is purged."""
    predecessor = await store.open_or_create_partition(
        "reborn_purge", _plaintext_partition_config()
    )
    await _add_segments(
        predecessor, _links(*(_seg(ts_offset_seconds=index) for index in range(5)))
    )
    await store.delete_partition("reborn_purge")
    successor = await store.open_or_create_partition(
        "reborn_purge", _plaintext_partition_config()
    )
    early = [_seg(ts_offset_seconds=10 + index) for index in range(3)]
    await _add_segments(successor, _links(*early))
    monkeypatch.setattr(store, "_purge_max_rows", 2)

    assert await store.purge_deleted_partitions() is True
    late = [_seg(ts_offset_seconds=20 + index) for index in range(2)]
    await _add_segments(successor, _links(*late))
    while await store.purge_deleted_partitions():
        pass

    assert await _row_counts(store, predecessor._incarnation) == (0, 0)
    assert await _row_counts(store, successor._incarnation) == (5, 5)
    contexts = await successor.get_segment_contexts(
        [segment.uuid for segment in (*early, *late)]
    )
    assert len(contexts) == 5


@pytest.mark.asyncio
@pytest.mark.parametrize("seed", [1, 2, 3])
async def test_racing_purgers_and_writers_leave_exactly_the_live_rows(
    store: SQLAlchemySegmentStore,
    monkeypatch: pytest.MonkeyPatch,
    seed: int,
) -> None:
    """Concurrent purgers and writers leave exactly what live partitions hold.

    Dead incarnations of random sizes and link fan-outs are drained by
    three concurrent purgers under a small random bound while writers keep
    adding to live partitions. Afterwards no dead incarnation has a
    segment or link row, each live partition has exactly the rows written
    to it, and the queue is empty.
    """
    rng = random.Random(seed)
    monkeypatch.setattr(store, "_purge_max_rows", rng.randint(1, 4))

    def batch(count: int, base: int) -> dict[Segment, list[UUID]]:
        return {
            _seg(ts_offset_seconds=base + offset): [
                uuid4() for _ in range(rng.randint(0, 2))
            ]
            for offset in range(count)
        }

    dead = []
    for index in range(5):
        partition = await store.open_or_create_partition(
            f"stress_dead_{index}", _plaintext_partition_config()
        )
        segments = batch(rng.randint(0, 9), 0)
        if segments:
            await _add_segments(partition, segments)
        dead.append(partition._incarnation)
        await store.delete_partition(f"stress_dead_{index}")

    live = [
        await store.open_or_create_partition(
            f"stress_live_{index}", _plaintext_partition_config()
        )
        for index in range(2)
    ]
    written = {partition._incarnation: [0, 0] for partition in live}

    async def write(partition: SQLAlchemySegmentStorePartition) -> None:
        for round_index in range(5):
            segments = batch(rng.randint(1, 3), round_index * 10)
            await _add_segments(partition, segments)
            written[partition._incarnation][0] += len(segments)
            written[partition._incarnation][1] += sum(map(len, segments.values()))

    async def drain() -> None:
        while await store.purge_deleted_partitions():
            pass

    await asyncio.wait_for(
        asyncio.gather(drain(), drain(), drain(), *(write(p) for p in live)), 60
    )
    await drain()

    for incarnation in dead:
        assert await _row_counts(store, incarnation) == (0, 0)
    for incarnation, (segments, links) in written.items():
        assert await _row_counts(store, incarnation) == (segments, links)
    async with store._create_session() as session:
        queue_depth = (
            await session.execute(select(func.count()).select_from(PurgeQueueRow))
        ).scalar_one()
    assert queue_depth == 0


@pytest.mark.integration
@pytest.mark.asyncio
async def test_purge_skips_entries_claimed_by_concurrent_purger(
    pg_store: SQLAlchemySegmentStore,
    sqlalchemy_pg_engine: AsyncEngine,
) -> None:
    """A purge never waits on another purger's claimed queue entries.

    Emulates a purger from another process holding its claim (`FOR
    UPDATE` on a queue row, uncommitted): a concurrent purge must skip
    that entry -- reclaiming everything else and completing without
    blocking -- rather than deadlocking on or waiting for the other
    purger's work.
    """
    partitions = {}
    for key in ("gc_held", "gc_free"):
        partition = await pg_store.open_or_create_partition(
            key, _plaintext_partition_config()
        )
        partitions[key] = partition._incarnation
        await _add_segments(partition, _links(_seg()))
        await pg_store.delete_partition(key)

    async with (
        partition._create_session() as remote_session,
        remote_session.begin(),
    ):
        await remote_session.execute(
            select(PurgeQueueRow)
            .where(PurgeQueueRow.incarnation == partitions["gc_held"])
            .with_for_update()
        )
        purge = asyncio.create_task(pg_store.purge_deleted_partitions())
        outcome = await _wait_until_blocked_or_done(sqlalchemy_pg_engine, purge)
        assert outcome == "done", (
            "purge blocked on a queue entry claimed by a concurrent "
            "purger instead of skipping it"
        )
        assert await purge is False

    async with partition._create_session() as session:
        held_rows = (
            await session.execute(
                select(func.count())
                .select_from(SegmentRow)
                .where(SegmentRow.incarnation == partitions["gc_held"])
            )
        ).scalar_one()
        free_rows = (
            await session.execute(
                select(func.count())
                .select_from(SegmentRow)
                .where(SegmentRow.incarnation == partitions["gc_free"])
            )
        ).scalar_one()
    assert held_rows == 1
    assert free_rows == 0

    assert await pg_store.purge_deleted_partitions() is False
    async with partition._create_session() as session:
        queue_depth = (
            await session.execute(select(func.count()).select_from(PurgeQueueRow))
        ).scalar_one()
    assert queue_depth == 0


@pytest.mark.integration
@pytest.mark.asyncio
async def test_write_landing_during_delete_is_never_orphaned(
    pg_store: SQLAlchemySegmentStore,
    sqlalchemy_pg_engine: AsyncEngine,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A write racing a delete can never leave untracked rows behind.

    The writer's registry-row pin forces delete_partition to wait, so any
    write that takes its pin commits before the incarnation is enqueued
    for purging -- its rows are always covered by the queue entry. Without
    the pin, the delete (and a purge) can complete while the write is in
    flight, and the write then lands rows under an incarnation the queue
    no longer tracks: garbage no purge will ever reclaim.
    """
    partition = await pg_store.open_or_create_partition(
        "orphan_race", _plaintext_partition_config()
    )
    incarnation = partition._incarnation

    reached_pause = asyncio.Event()
    release = asyncio.Event()
    original_insert_segments = partition._insert_segments

    async def pausing_insert_segments(session, segments) -> None:
        reached_pause.set()
        await release.wait()
        await original_insert_segments(session, segments)

    monkeypatch.setattr(partition, "_insert_segments", pausing_insert_segments)

    writer = asyncio.create_task(
        partition.add_event_segments(*_events(_links(_seg())), lease_seconds=0)
    )
    deleter = None
    try:
        await asyncio.wait_for(reached_pause.wait(), 30)
        deleter = asyncio.create_task(pg_store.delete_partition("orphan_race"))
        outcome = await _wait_until_blocked_or_done(sqlalchemy_pg_engine, deleter)
        if outcome == "done":
            # Broken-locking world: the delete did not wait for the writer.
            # A purge in this window retires the queue entry before the
            # write lands.
            await pg_store.purge_deleted_partitions()
    finally:
        # Unpause the writer even on failure so its open transaction cannot
        # wedge fixture teardown.
        release.set()
        with contextlib.suppress(Exception):
            await asyncio.wait_for(writer, 30)
        if deleter is not None:
            with contextlib.suppress(Exception):
                await asyncio.wait_for(deleter, 30)

    with contextlib.suppress(SegmentStorePartitionHandleStaleError):
        await writer
    await deleter

    assert await pg_store.purge_deleted_partitions() is False
    async with partition._create_session() as session:
        leftover_rows = (
            await session.execute(
                select(func.count())
                .select_from(SegmentRow)
                .where(SegmentRow.incarnation == incarnation)
            )
        ).scalar_one()
    assert leftover_rows == 0, (
        "rows landed under an incarnation the purge queue no longer "
        "tracks: the writer's registry-row pin is not draining writes "
        "before deletion commits"
    )


@pytest.mark.integration
@pytest.mark.asyncio
async def test_concurrent_remote_delete_yields_single_queue_entry(
    pg_store: SQLAlchemySegmentStore,
    sqlalchemy_pg_engine: AsyncEngine,
) -> None:
    """Racing deletions enqueue a dead incarnation exactly once.

    Emulates a delete_partition from another process at the wire level
    (exclusive row pin, queue insert, registry delete, held uncommitted)
    and races the store's delete_partition against it: the store's delete
    must block on the row pin and, once the remote transaction commits,
    observe the registry row gone and no-op -- one queue entry, no
    integrity error surfacing to the caller.
    """
    partition = await pg_store.open_or_create_partition(
        "remote_del", _plaintext_partition_config()
    )
    incarnation = partition._incarnation

    async with partition._create_session() as remote_session:
        async with remote_session.begin():
            await remote_session.execute(
                select(PartitionRow.incarnation)
                .where(PartitionRow.partition_key == "remote_del")
                .with_for_update()
            )
            await remote_session.execute(
                insert(PurgeQueueRow).values(
                    incarnation=incarnation,
                    enqueued_at=func.now(),
                )
            )
            await remote_session.execute(
                delete(PartitionRow).where(PartitionRow.partition_key == "remote_del")
            )

            local_delete = asyncio.create_task(pg_store.delete_partition("remote_del"))
            # Wait for the local delete to block (on the registry-row pin
            # with correct locking; on the queue insert without it) so the
            # race is staged before the remote transaction commits.
            await _wait_until_blocked_or_done(sqlalchemy_pg_engine, local_delete)
        # Remote transaction committed on exiting begin().
        await asyncio.wait_for(local_delete, 30)

    async with partition._create_session() as session:
        queue_entries = (
            await session.execute(select(func.count()).select_from(PurgeQueueRow))
        ).scalar_one()
    assert queue_entries == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "create_via",
    ["create_partition", "open_or_create_partition"],
)
async def test_incarnation_with_garbage_left_is_never_reused(
    store: SQLAlchemySegmentStore,
    monkeypatch: pytest.MonkeyPatch,
    create_via: str,
) -> None:
    """A minted incarnation colliding with unpurged garbage is re-minted.

    Data rows are keyed by incarnation alone, so a new partition reusing a
    dead incarnation's uuid would adopt its garbage and then be erased by
    the purger. The mint transaction re-checks the purge queue and retries
    with a fresh uuid instead.
    """
    doomed = await store.open_or_create_partition(
        "gc_reuse", _plaintext_partition_config()
    )
    dead_incarnation = doomed._incarnation
    await _add_segments(doomed, _links(_seg()))
    await store.delete_partition("gc_reuse")

    offered = []

    def colliding_uuid4() -> UUID:
        if not offered:
            offered.append(dead_incarnation)
            return dead_incarnation
        return uuid4()

    monkeypatch.setattr(sqlalchemy_segment_store, "uuid4", colliding_uuid4)

    if create_via == "create_partition":
        await store.create_partition("fresh_p", _plaintext_partition_config())
        fresh = await store.open_partition("fresh_p")
    else:
        fresh = await store.open_or_create_partition(
            "fresh_p", _plaintext_partition_config()
        )
    assert fresh is not None
    assert offered, "the colliding uuid was never offered to the mint"
    assert fresh._incarnation != dead_incarnation

    # The dead incarnation's garbage is untouched and still tracked.
    async with fresh._create_session() as session:
        dead_rows = (
            await session.execute(
                select(func.count())
                .select_from(SegmentRow)
                .where(SegmentRow.incarnation == dead_incarnation)
            )
        ).scalar_one()
        queue_depth = (
            await session.execute(select(func.count()).select_from(PurgeQueueRow))
        ).scalar_one()
    assert dead_rows == 1
    assert queue_depth == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "create_via",
    ["create_partition", "open_or_create_partition"],
)
async def test_incarnation_colliding_with_live_partition_is_never_reused(
    store: SQLAlchemySegmentStore,
    monkeypatch: pytest.MonkeyPatch,
    create_via: str,
) -> None:
    """A minted incarnation colliding with a live one is re-minted.

    The registry's unique constraint rejects the insert; the mint must
    classify that as an incarnation collision (the key is free), mint a
    fresh uuid, and succeed -- not misreport the partition as already
    existing.
    """
    live = await store.open_or_create_partition(
        "live_src", _plaintext_partition_config()
    )
    live_incarnation = live._incarnation

    offered = []

    def colliding_uuid4() -> UUID:
        if not offered:
            offered.append(live_incarnation)
            return live_incarnation
        return uuid4()

    monkeypatch.setattr(sqlalchemy_segment_store, "uuid4", colliding_uuid4)

    if create_via == "create_partition":
        await store.create_partition("fresh_p", _plaintext_partition_config())
        fresh = await store.open_partition("fresh_p")
    else:
        fresh = await store.open_or_create_partition(
            "fresh_p", _plaintext_partition_config()
        )
    assert fresh is not None
    assert offered, "the colliding uuid was never offered to the mint"
    assert fresh._incarnation != live_incarnation

    # The live partition is unharmed.
    reopened = await store.open_partition("live_src")
    assert reopened is not None
    assert reopened._incarnation == live_incarnation


@pytest.mark.asyncio
async def test_purge_bound_comes_from_params(
    sqlalchemy_sqlite_engine: AsyncEngine,
) -> None:
    """The configured bound governs every purge call."""
    store = SQLAlchemySegmentStore(
        SQLAlchemySegmentStoreParams(
            engine=sqlalchemy_sqlite_engine,
            purge_max_rows=2,
        )
    )
    await store.startup()
    try:
        partition = await store.open_or_create_partition(
            "bound_p", _plaintext_partition_config()
        )
        event_uuid = uuid4()
        await _add_segments(
            partition,
            {_seg(event_uuid=event_uuid, index=index): [] for index in range(2)},
        )
        await store.delete_partition("bound_p")

        assert await store.purge_deleted_partitions() is True
        assert await store.purge_deleted_partitions() is False
    finally:
        async with sqlalchemy_sqlite_engine.begin() as conn:
            await conn.run_sync(BaseSegmentStore.metadata.drop_all)


@pytest.mark.integration
@pytest.mark.asyncio
async def test_mint_detects_collision_with_concurrent_deletion(
    pg_store: SQLAlchemySegmentStore,
    sqlalchemy_pg_engine: AsyncEngine,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The mint checks the purge queue AFTER inserting the registry row.

    Pins the statement ordering inside the mint transaction, which only a
    concurrent interleave can distinguish (the sequential collision test
    passes under either order): while a deletion holds its uncommitted
    registry delete, a colliding mint blocks on the unique index; once the
    deletion commits, the insert succeeds -- and only a check that runs
    AFTER the insert sees the just-committed queue entry. Checking before
    the insert reads the queue too early and commits a live partition
    whose incarnation is on the purge queue, handing its rows to the
    purger.
    """
    victim = await pg_store.open_or_create_partition(
        "mint_victim", _plaintext_partition_config()
    )
    victim_incarnation = victim._incarnation
    await _add_segments(victim, _links(_seg()))

    offered = []

    def colliding_uuid4() -> UUID:
        if not offered:
            offered.append(victim_incarnation)
            return victim_incarnation
        return uuid4()

    monkeypatch.setattr(sqlalchemy_segment_store, "uuid4", colliding_uuid4)

    async with (
        victim._create_session() as remote_session,
        remote_session.begin(),
    ):
        # Emulate a delete_partition from another process, held uncommitted.
        await remote_session.execute(
            select(PartitionRow)
            .where(PartitionRow.partition_key == "mint_victim")
            .with_for_update()
        )
        await remote_session.execute(
            insert(PurgeQueueRow).values(
                incarnation=victim_incarnation,
                enqueued_at=func.now(),
            )
        )
        await remote_session.execute(
            delete(PartitionRow).where(PartitionRow.partition_key == "mint_victim")
        )

        creator = asyncio.create_task(
            pg_store.create_partition("mint_fresh", _plaintext_partition_config())
        )
        # The colliding insert must be waiting on the uncommitted registry
        # delete before the deletion commits, or no interleave is staged.
        outcome = await _wait_until_blocked_or_done(sqlalchemy_pg_engine, creator)
        assert outcome == "blocked", (
            "the colliding mint did not block on the concurrent deletion"
        )
    # Deletion committed on exiting begin().
    await asyncio.wait_for(creator, 30)

    fresh = await pg_store.open_partition("mint_fresh")
    assert fresh is not None
    assert offered, "the colliding uuid was never offered to the mint"
    assert fresh._incarnation != victim_incarnation, (
        "the mint reused an incarnation that a concurrent deletion had "
        "just moved to the purge queue"
    )

    # The dead incarnation's garbage is still tracked; purging it leaves
    # the fresh partition alone.
    assert await pg_store.purge_deleted_partitions() is False
    async with fresh._create_session() as session:
        dead_rows = (
            await session.execute(
                select(func.count())
                .select_from(SegmentRow)
                .where(SegmentRow.incarnation == victim_incarnation)
            )
        ).scalar_one()
    assert dead_rows == 0
    assert await pg_store.open_partition("mint_fresh") is not None


@pytest.mark.integration
@pytest.mark.asyncio
async def test_purge_claims_queue_entries_incrementally(
    pg_store: SQLAlchemySegmentStore,
    recorded_statements: list[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A bounded purge never claims queue entries it will not process.

    Claims are one entry at a time, so a call whose bound exhausts on its
    first incarnation issues exactly one claim -- it neither materializes
    nor locks the rest of the backlog, leaving those entries claimable by
    concurrent purgers.
    """
    for index in range(3):
        partition = await pg_store.open_or_create_partition(
            f"inc_claim_{index}", _plaintext_partition_config()
        )
        await _add_segments(partition, _links(_seg(), _seg(), _seg()))
        await pg_store.delete_partition(f"inc_claim_{index}")
    monkeypatch.setattr(pg_store, "_purge_max_rows", 2)
    recorded_statements.clear()

    assert await pg_store.purge_deleted_partitions() is True

    claim_statements = [
        statement
        for statement in recorded_statements
        if statement.startswith("SELECT") and "segment_store_gc" in statement
    ]
    assert claim_statements, "no queue claim was recorded"
    assert all("LIMIT" in statement for statement in claim_statements), (
        "an unbounded queue claim materialized the whole backlog"
    )
    assert len(claim_statements) == 1

    while await pg_store.purge_deleted_partitions():
        pass
    async with partition._create_session() as session:
        queue_depth = (
            await session.execute(select(func.count()).select_from(PurgeQueueRow))
        ).scalar_one()
    assert queue_depth == 0


@pytest.mark.asyncio
async def test_sqlite_write_racing_delete_cannot_orphan_rows(
    sqlite_store: SQLAlchemySegmentStore,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """On SQLite the write fence checks inside the write transaction.

    A fence that ran outside it would let a delete and a full purge
    complete between the check and the insert, and the write would then
    commit rows no queue entry tracks. The write transaction holds the
    write lock from BEGIN, so the racing deletion must wait for the writer
    and the rows stay reclaimable.
    """
    partition = await sqlite_store.open_or_create_partition(
        "sqlite_fence", _plaintext_partition_config()
    )
    incarnation = partition._incarnation

    reached_pause = asyncio.Event()
    release = asyncio.Event()
    original_insert_segments = partition._insert_segments

    async def pausing_insert_segments(session, segments) -> None:
        reached_pause.set()
        await release.wait()
        await original_insert_segments(session, segments)

    monkeypatch.setattr(partition, "_insert_segments", pausing_insert_segments)

    writer = asyncio.create_task(
        partition.add_event_segments(*_events(_links(_seg())), lease_seconds=0)
    )
    await asyncio.wait_for(reached_pause.wait(), 30)

    deleter_started = asyncio.Event()
    original_delete_partition = sqlite_store.delete_partition

    async def signalling_delete_partition(partition_key: str) -> None:
        deleter_started.set()
        await original_delete_partition(partition_key)

    monkeypatch.setattr(sqlite_store, "delete_partition", signalling_delete_partition)
    deleter = asyncio.create_task(sqlite_store.delete_partition("sqlite_fence"))
    # The started event proves the deleter ran before the sample below,
    # so a loaded box cannot pass vacuously by never scheduling it. With
    # the fence transaction open, the deleter CANNOT finish while the
    # writer is paused; without it, it finishes immediately (broken
    # world), and a purge in this window would sweep the queue before
    # the write. The grace period is the one wall-clock element left:
    # SQLite exposes no lock-wait state to observe, so "still blocked"
    # can only be sampled.
    await asyncio.wait_for(deleter_started.wait(), 30)
    done, _pending = await asyncio.wait([deleter], timeout=1.0)
    if done:
        while await sqlite_store.purge_deleted_partitions():
            pass
    release.set()
    with contextlib.suppress(SegmentStorePartitionHandleStaleError):
        await asyncio.wait_for(writer, 30)
    await asyncio.wait_for(deleter, 30)

    while await sqlite_store.purge_deleted_partitions():
        pass
    async with partition._create_session() as session:
        leftover_rows = (
            await session.execute(
                select(func.count())
                .select_from(SegmentRow)
                .where(SegmentRow.incarnation == incarnation)
            )
        ).scalar_one()
    assert leftover_rows == 0, (
        "rows were committed under an incarnation the purge queue no "
        "longer tracks: the SQLite fence checked outside the write "
        "transaction"
    )


@pytest.mark.asyncio
async def test_persistent_mint_failure_raises_instead_of_looping(
    store: SQLAlchemySegmentStore,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A persistent constraint failure surfaces after bounded mint attempts.

    Every realistic trip through the collision-retry path beyond a few
    attempts is a persistent database error being retried, not a race;
    the mint raises SegmentStoreAttemptsExhaustedError instead of
    hot-looping, with the underlying error chained for diagnosis.
    """
    cause = IntegrityError("stmt", None, Exception("persistent"))
    attempts = []

    async def always_colliding(partition_key, incarnation, config) -> None:
        attempts.append(incarnation)
        raise sqlalchemy_segment_store._RegistryInsertRejectedError(
            str(incarnation)
        ) from cause

    monkeypatch.setattr(store, "_insert_partition_row", always_colliding)

    with pytest.raises(SegmentStoreAttemptsExhaustedError) as exc_info:
        await store.create_partition("mint_cap", _plaintext_partition_config())
    assert len(attempts) == sqlalchemy_segment_store._MAX_MINT_ATTEMPTS
    # The underlying database error stays reachable for diagnosis.
    collision = exc_info.value.__cause__
    assert collision is not None
    assert collision.__cause__ is cause

    attempts.clear()
    with pytest.raises(SegmentStoreAttemptsExhaustedError):
        await store.open_or_create_partition("mint_cap", _plaintext_partition_config())
    assert len(attempts) == sqlalchemy_segment_store._MAX_MINT_ATTEMPTS

    # The lost-race arm is bounded by the same cap: an insert that keeps
    # losing to a winner that keeps vanishing must not livelock.
    attempts.clear()

    async def always_losing(partition_key, incarnation, config) -> None:
        attempts.append(incarnation)
        raise SegmentStorePartitionAlreadyExistsError(partition_key)

    monkeypatch.setattr(store, "_insert_partition_row", always_losing)
    with pytest.raises(SegmentStoreAttemptsExhaustedError):
        await store.open_or_create_partition("mint_cap", _plaintext_partition_config())
    assert len(attempts) == sqlalchemy_segment_store._MAX_MINT_ATTEMPTS


@pytest.mark.asyncio
async def test_persistent_integrity_error_surfaces_with_cause(
    store: SQLAlchemySegmentStore,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Any integrity rejection is retried boundedly; the cause stays chained.

    A rejected registry insert is treated as a possible incarnation
    collision and re-minted up to the attempt bound; a persistent cause
    surfaces through SegmentStoreAttemptsExhaustedError with the
    driver's error chained for diagnosis. A NOT NULL violation on the
    incarnation column stands in for a cause that is not a collision.
    """
    monkeypatch.setattr(sqlalchemy_segment_store, "uuid4", lambda: None)

    for create in (store.create_partition, store.open_or_create_partition):
        with pytest.raises(SegmentStoreAttemptsExhaustedError) as exc_info:
            await create("not_null", _plaintext_partition_config())
        cause: BaseException | None = exc_info.value.__cause__
        while cause is not None and not isinstance(cause, IntegrityError):
            cause = cause.__cause__
        assert isinstance(cause, IntegrityError)


@pytest.mark.asyncio
async def test_purge_bounds_entries_processed_per_call(
    sqlalchemy_sqlite_engine: AsyncEngine,
) -> None:
    """Queue entries carry their own per-call bound, separate from rows.

    Empty partitions are cheap to create and delete, so a large backlog
    of empty entries is easy to accumulate; entries cost round trips
    rather than row deletions, so they are bounded by
    purge_max_partitions rather than charged against the row bound.
    """
    store = SQLAlchemySegmentStore(
        SQLAlchemySegmentStoreParams(
            engine=sqlalchemy_sqlite_engine,
            purge_max_partitions=2,
        )
    )
    await store.startup()
    try:
        for index in range(5):
            await store.create_partition(
                f"empty_{index}", _plaintext_partition_config()
            )
            await store.delete_partition(f"empty_{index}")

        assert await store.purge_deleted_partitions() is True
        async with store._create_session() as session:
            queue_depth = (
                await session.execute(select(func.count()).select_from(PurgeQueueRow))
            ).scalar_one()
        assert queue_depth == 3

        while await store.purge_deleted_partitions():
            pass
    finally:
        async with sqlalchemy_sqlite_engine.begin() as conn:
            await conn.run_sync(BaseSegmentStore.metadata.drop_all)


@pytest.mark.asyncio
async def test_empty_incarnations_do_not_consume_row_budget(
    store: SQLAlchemySegmentStore,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A backlog of empty entries leaves the row budget for real rows."""
    monkeypatch.setattr(store, "_purge_max_rows", 2)
    for index in range(3):
        await store.create_partition(f"noop_{index}", _plaintext_partition_config())
        await store.delete_partition(f"noop_{index}")
    rowful = await store.open_or_create_partition(
        "rowful", _plaintext_partition_config()
    )
    event_uuid = uuid4()
    await _add_segments(
        rowful, {_seg(event_uuid=event_uuid, index=index): [] for index in range(2)}
    )
    await store.delete_partition("rowful")

    # One call: three empty entries retired for free, then both segment rows;
    # the next takes their event's row.
    assert await store.purge_deleted_partitions() is True
    async with rowful._create_session() as session:
        remaining_rows = (
            await session.execute(select(func.count()).select_from(SegmentRow))
        ).scalar_one()
    assert remaining_rows == 0
    assert await store.purge_deleted_partitions() is False


@pytest.mark.asyncio
async def test_purge_reclaims_oldest_garbage_first(
    store: SQLAlchemySegmentStore,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The queue is FIFO: claims follow the enqueue stamp, not insertion."""
    incarnations = {}
    partition = None
    for key in ("fifo_a", "fifo_b", "fifo_c"):
        partition = await store.open_or_create_partition(
            key, _plaintext_partition_config()
        )
        incarnations[key] = partition._incarnation
        await _add_segments(partition, {_seg(): []})
        await store.delete_partition(key)

    # Stamps come from the database clock, whose resolution need not
    # separate back-to-back deletions; set them so the LAST-inserted entry
    # is the oldest, so only the ordering key can produce the expectation.
    async with partition._create_session() as session, session.begin():
        for age, key in enumerate(incarnations):
            await session.execute(
                update(PurgeQueueRow)
                .where(PurgeQueueRow.incarnation == incarnations[key])
                .values(
                    enqueued_at=_database_time_ago(
                        store._engine, timedelta(minutes=age)
                    )
                )
            )

    # One row of budget: only the oldest entry's row may die.
    monkeypatch.setattr(store, "_purge_max_rows", 1)
    assert await store.purge_deleted_partitions() is True
    async with partition._create_session() as session:
        counts = {
            key: (
                await session.execute(
                    select(func.count())
                    .select_from(SegmentRow)
                    .where(SegmentRow.incarnation == incarnations[key])
                )
            ).scalar_one()
            for key in incarnations
        }
    assert counts == {"fifo_a": 1, "fifo_b": 1, "fifo_c": 0}

    while await store.purge_deleted_partitions():
        pass


@pytest.mark.integration
@pytest.mark.asyncio
async def test_purge_queue_stamps_enqueue_time_from_database_clock(
    pg_store: SQLAlchemySegmentStore,
    recorded_statements: list[str],
) -> None:
    """The FIFO key is one clock for every server: the database's."""
    await pg_store.create_partition("db_clock", _plaintext_partition_config())
    recorded_statements.clear()

    await pg_store.delete_partition("db_clock")

    enqueues = [
        s for s in recorded_statements if s.startswith("INSERT INTO segment_store_gc")
    ]
    assert len(enqueues) == 1
    assert "now()" in enqueues[0]


class _ForeignPayloadCodecConfig(PlaintextPayloadCodecConfig):
    """A distinct codec-config type standing in for a future variant."""


@pytest.mark.asyncio
async def test_open_or_create_with_different_config_raises_mismatch(
    store: SQLAlchemySegmentStore,
) -> None:
    """Reopening a key under a different config is refused, not adapted."""
    await store.create_partition("cfg_guard", _plaintext_partition_config())

    requested = SegmentStorePartitionConfig(
        payload_codec_config=_ForeignPayloadCodecConfig()
    )
    with pytest.raises(SegmentStorePartitionConfigMismatchError) as exc_info:
        await store.open_or_create_partition("cfg_guard", requested)
    assert exc_info.value.partition_key == "cfg_guard"
    assert exc_info.value.existing_config == _plaintext_partition_config()
    assert exc_info.value.requested_config == requested

    # The partition itself is untouched and still opens.
    assert await store.open_partition("cfg_guard") is not None


@pytest.mark.asyncio
@pytest.mark.parametrize("create", ["create_partition", "open_or_create_partition"])
async def test_unloadable_codec_config_commits_no_registry_row(
    store: SQLAlchemySegmentStore,
    monkeypatch: pytest.MonkeyPatch,
    create: str,
) -> None:
    """A codec that cannot be materialized leaves no registry row behind."""

    async def unloadable(config) -> None:
        raise NotImplementedError("unsupported codec")

    monkeypatch.setattr(store, "_load_payload_codec", unloadable)
    with pytest.raises(NotImplementedError):
        await getattr(store, create)("codec_p", _plaintext_partition_config())
    async with store._create_session() as session:
        assert (
            await SQLAlchemySegmentStore._get_partition_row(session, "codec_p")
        ) is None


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
    """Adding events and claiming pending writes and tombstones depend on
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


@pytest.mark.integration
@pytest.mark.asyncio
async def test_windowed_read_raises_when_partition_dies_between_statements(
    pg_store: SQLAlchemySegmentStore,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Seeds found, then deletion commits before the context statements.

    Each statement takes its own snapshot, so the context statements
    would return nothing; the read must raise rather than hand back
    seeds with silently empty context. On SQLite a read's statements
    share one transaction, and so one snapshot, which a deletion cannot
    change.
    """
    partition = await pg_store.open_or_create_partition(
        "mid_read", _plaintext_partition_config()
    )
    segs = [_seg(ts_offset_seconds=i) for i in range(3)]
    await _add_segments(partition, _links(*segs))

    original = partition._get_context_rows_lateral

    async def delete_then_read(*args, **kwargs):
        await pg_store.delete_partition("mid_read")
        return await original(*args, **kwargs)

    monkeypatch.setattr(partition, "_get_context_rows_lateral", delete_then_read)
    with pytest.raises(SegmentStorePartitionHandleStaleError):
        await partition.get_segment_contexts(
            [segs[1].uuid], max_backward_segments=1, max_forward_segments=1
        )


@pytest.mark.asyncio
async def test_partition_key_with_trailing_newline_is_rejected(
    store: SQLAlchemySegmentStore,
) -> None:
    """`$` matches before a trailing newline; the validator must not."""
    with pytest.raises(ValueError, match="invalid characters"):
        await store.create_partition("bad_key\n", _plaintext_partition_config())


@pytest.mark.integration
@pytest.mark.asyncio
async def test_delete_partition_touches_only_registry_and_queue(
    pg_store: SQLAlchemySegmentStore,
    recorded_statements: list[str],
) -> None:
    """Deletion is O(1): no data-table statements, regardless of size."""
    partition = await pg_store.open_or_create_partition(
        "big_delete", _plaintext_partition_config()
    )
    await _add_segments(
        partition, _links(*(_seg(ts_offset_seconds=i) for i in range(20)))
    )
    recorded_statements.clear()

    await pg_store.delete_partition("big_delete")

    data_tables = BaseSegmentStore.metadata.tables.keys() - {
        PartitionRow.__tablename__,
        PurgeQueueRow.__tablename__,
    }
    touching_data = [
        s for s in recorded_statements if any(table in s for table in data_tables)
    ]
    assert not touching_data, touching_data
    assert any("segment_store_gc" in s for s in recorded_statements)


# ===================================================================
# Locking
#
# Each test stages the interleaving its lock exists to serialize.
# Coverage, as verified by ablating locks one at a time in
# source-patched variants of both this store and the pre-overhaul
# partitioned store:
# - The writer's shared registry-row pin (_ensure_partition_live with pin)
#   blocks partition deletion while a write transaction is in flight:
#   test_write_pin_blocks_partition_delete fails with the pin ablated
#   in either generation and passes with it present.
# - Deletion's exclusive registry-row pin serializes racing lifecycle
#   operations: the churn and concurrent-delete tests fail with it
#   ablated. On the partitioned store those two tests failed even with
#   all locks intact (create/delete DDL lock cycles through the shared
#   parents) -- the deadlock-freedom they pin is a property of the
#   shared-table layout.
# Ingests and forgets of several events take turns on the partition's write
# lock (FOR NO KEY UPDATE on the registry row), so none takes its event
# locks in an order; one-event writes and the other pinned writes take
# FOR KEY SHARE, which the write lock does not block, and deletion takes
# FOR UPDATE, which blocks both.
# ===================================================================


@pytest.mark.integration
@pytest.mark.asyncio
async def test_write_pin_blocks_partition_delete(
    pg_store: SQLAlchemySegmentStore,
    sqlalchemy_pg_engine: AsyncEngine,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The writer's shared registry-row pin makes deletion wait out writers.

    Stages the exact interleaving the pin serializes: a write transaction
    that has taken its pin but not yet committed, with a concurrent
    `delete_partition`. Deletion must block until the writer commits;
    without the pin it proceeds immediately and the write lands in a
    partition that no longer exists.
    """
    partition = await pg_store.open_or_create_partition(
        "lk_write_pin", _plaintext_partition_config()
    )

    reached_pause = asyncio.Event()
    release = asyncio.Event()
    original_insert_segments = partition._insert_segments

    async def pausing_insert_segments(session, segments) -> None:
        # Runs inside the write transaction, after the write pin is taken.
        reached_pause.set()
        await release.wait()
        await original_insert_segments(session, segments)

    monkeypatch.setattr(partition, "_insert_segments", pausing_insert_segments)

    writer = asyncio.create_task(
        partition.add_event_segments(*_events(_links(_seg())), lease_seconds=0)
    )
    deleter = None
    try:
        await asyncio.wait_for(reached_pause.wait(), 30)
        deleter = asyncio.create_task(pg_store.delete_partition("lk_write_pin"))
        outcome = await _wait_until_blocked_or_done(sqlalchemy_pg_engine, deleter)
        assert outcome == "blocked", (
            "delete_partition completed while a write transaction was in "
            "flight: the writer's registry-row pin is not blocking deletion"
        )
    finally:
        # Unpause the writer even on failure so its open transaction cannot
        # wedge fixture teardown.
        release.set()
        with contextlib.suppress(Exception):
            await asyncio.wait_for(writer, 30)
        if deleter is not None:
            with contextlib.suppress(Exception):
                await asyncio.wait_for(deleter, 30)

    await writer
    await deleter
    assert await pg_store.open_partition("lk_write_pin") is None


@contextlib.asynccontextmanager
async def _registry_row_held(
    partition: SQLAlchemySegmentStorePartition, *, exclusive: bool
) -> AsyncIterator[None]:
    """Hold the partition's registry row in another transaction, under the write lock if `exclusive`, else under a pin."""
    async with partition._create_session() as session, session.begin():
        await session.execute(
            select(PartitionRow)
            .where(PartitionRow.incarnation == partition._incarnation)
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
        [segment.event_uuid for segment in segments], lease_seconds=0
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
    partition = await pg_store.open_or_create_partition(
        PARTITION_KEY, _plaintext_partition_config()
    )
    write = await operation(pg_store, partition, 2)

    async with _registry_row_held(partition, exclusive=True):
        task = asyncio.create_task(write())
        assert await _wait_until_blocked_or_done(sqlalchemy_pg_engine, task) == (
            "blocked"
        )
    await task


async def _delete_pending_writes(
    _store: SQLAlchemySegmentStore, partition: SQLAlchemySegmentStorePartition
):
    segment = _seg()
    await partition.add_event_segments(*_events(_links(segment)), lease_seconds=0)
    return lambda: partition.delete_pending_writes([segment.uuid])


async def _record_first_delete(
    _store: SQLAlchemySegmentStore, partition: SQLAlchemySegmentStorePartition
):
    tombstone_uuid = await _add_claimable_record(partition, TombstoneRow)
    return lambda: partition.record_tombstone_first_delete([tombstone_uuid])


async def _finish_second_delete(
    store: SQLAlchemySegmentStore, partition: SQLAlchemySegmentStorePartition
):
    _, claim = await _claim_second_delete(store, partition)
    return lambda: store.finish_tombstone_claim(claim)


@pytest.mark.integration
@pytest.mark.asyncio
@pytest.mark.parametrize(
    "operation",
    [
        functools.partial(_ingest, events=1),
        functools.partial(_forget, events=1),
        _delete_pending_writes,
        _record_first_delete,
        _finish_second_delete,
    ],
    ids=[
        "ingest_one",
        "forget_one",
        "delete_pending_writes",
        "record_first_delete",
        "second_delete",
    ],
)
async def test_one_event_and_the_other_pinned_writes_do_not_wait_for_the_write_lock(
    pg_store: SQLAlchemySegmentStore,
    sqlalchemy_pg_engine: AsyncEngine,
    operation,
) -> None:
    partition = await pg_store.open_or_create_partition(
        PARTITION_KEY, _plaintext_partition_config()
    )
    write = await operation(pg_store, partition)

    async with _registry_row_held(partition, exclusive=True):
        task = asyncio.create_task(write())
        assert await _wait_until_blocked_or_done(sqlalchemy_pg_engine, task) == ("done")
        await task


@pytest.mark.integration
@pytest.mark.asyncio
@pytest.mark.parametrize("exclusive", [True, False], ids=["write_lock", "pin"])
async def test_deleting_a_partition_waits_for_its_write_lock_and_its_pins(
    pg_store: SQLAlchemySegmentStore,
    sqlalchemy_pg_engine: AsyncEngine,
    exclusive: bool,
) -> None:
    partition = await pg_store.open_or_create_partition(
        PARTITION_KEY, _plaintext_partition_config()
    )

    async with _registry_row_held(partition, exclusive=exclusive):
        deletion = asyncio.create_task(pg_store.delete_partition(PARTITION_KEY))
        assert await _wait_until_blocked_or_done(sqlalchemy_pg_engine, deletion) == (
            "blocked"
        )
    await deletion


@pytest.mark.asyncio
async def test_batches_naming_the_same_events_in_opposite_orders_store_them_once(
    store: SQLAlchemySegmentStore,
) -> None:
    """One stores the events and the other is refused whole, never a deadlock."""
    partition = await store.open_or_create_partition(
        PARTITION_KEY, _plaintext_partition_config()
    )
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
    partition = await store.open_or_create_partition(
        PARTITION_KEY, _plaintext_partition_config()
    )
    for _ in range(10):
        event_uuid = uuid4()
        segments = [_seg(event_uuid=event_uuid, index=index) for index in range(3)]
        await _add_segments(partition, _links(*segments))

        outcomes = await asyncio.wait_for(
            asyncio.gather(
                *(
                    partition.delete_event_segments([event_uuid], lease_seconds=0)
                    for _ in range(2)
                )
            ),
            30,
        )

        assert sorted(map(len, outcomes)) == [0, 3]


@pytest.mark.asyncio
async def test_overlapping_forgets_do_not_deadlock(
    store: SQLAlchemySegmentStore,
) -> None:
    """Overlapping concurrent `delete_event_segments` stay deadlock-free.

    The racing forgets of several events take turns on the partition's
    write lock. Each unlinked derivative is named by exactly one of their
    tombstones.
    """
    partition = await store.open_or_create_partition(
        "lk_row_order", _plaintext_partition_config()
    )

    async def round_trip(rng: random.Random) -> None:
        links = _links(*(_seg(index=index) for index in range(24)))
        await _add_segments(partition, links)
        event_uuids = [segment.event_uuid for segment in links]
        forward = event_uuids[:16]
        backward = [*reversed(event_uuids[8:])]
        tombstones = await asyncio.gather(
            partition.delete_event_segments(forward, lease_seconds=0),
            partition.delete_event_segments(backward, lease_seconds=0),
        )
        assert (
            await partition.delete_event_segments(
                rng.sample(event_uuids, len(event_uuids)), lease_seconds=0
            )
            == []
        )
        derivative_uuids = [
            derivative_uuid
            for derivative_uuids in links.values()
            for derivative_uuid in derivative_uuids
        ]
        assert await _row_counts(partition, partition._incarnation) == (0, 0)
        # Held by the tombstones.
        assert (
            await _count_rows(
                partition, TombstoneDerivativeRow.uuid.in_(derivative_uuids)
            )
            == 24
        )
        assert sorted(
            derivative_uuid
            for deleted in tombstones
            for tombstone in deleted
            for derivative_uuid in tombstone.derivative_uuids
        ) == sorted(
            derivative_uuid
            for derivative_uuids in links.values()
            for derivative_uuid in derivative_uuids
        )

    rng = random.Random(7)
    for _ in range(10):
        await asyncio.wait_for(round_trip(rng), 30)


@pytest.mark.asyncio
async def test_lifecycle_churn_completes_without_database_errors(
    store: SQLAlchemySegmentStore,
) -> None:
    """Concurrent create/open/delete churn never aborts on lock cycles.

    Lifecycle operations of different partitions, and repeated
    create/delete/re-create of the same partitions, must serialize
    through the store's partition-level locking: any database deadlock
    abort or constraint violation escaping the store API is a locking
    defect. Domain errors (already exists, config mismatch) are the only
    legitimate racing outcomes.
    """
    keys = [f"lk_churn_{index}" for index in range(4)]
    config = _plaintext_partition_config()

    async def worker(seed: int) -> None:
        rng = random.Random(seed)
        for _ in range(40):
            key = rng.choice(keys)
            operation = rng.randrange(4)
            try:
                if operation == 0:
                    await store.create_partition(key, config)
                elif operation == 1:
                    await store.open_or_create_partition(key, config)
                elif operation == 2:
                    await store.open_partition(key)
                else:
                    await store.delete_partition(key)
            except (
                SegmentStorePartitionAlreadyExistsError,
                SegmentStorePartitionConfigMismatchError,
            ):
                pass

    await asyncio.wait_for(
        asyncio.gather(*(worker(seed) for seed in range(8))),
        120,
    )


@pytest.mark.asyncio
async def test_concurrent_partition_deletes_are_clean(
    store: SQLAlchemySegmentStore,
) -> None:
    """Racing deletions of one partition serialize through the row pin.

    All racers must complete without database errors -- the losers observe
    the winner's deletion instead of double-processing the partition --
    and the partition must be cleanly re-creatable afterwards.
    """
    config = _plaintext_partition_config()
    for cycle in range(10):
        await store.create_partition("lk_del_race", config)
        await asyncio.wait_for(
            asyncio.gather(*(store.delete_partition("lk_del_race") for _ in range(4))),
            30,
        )
        assert await store.open_partition("lk_del_race") is None
        async with store._create_session() as session:
            queue_depth = (
                await session.execute(select(func.count()).select_from(PurgeQueueRow))
            ).scalar_one()
        # Racing deletions enqueued the dead incarnation exactly once.
        assert queue_depth == cycle + 1
    await store.create_partition("lk_del_race", config)


@pytest.mark.asyncio
@pytest.mark.parametrize("seed", [1, 2])
async def test_random_operation_sequences_agree_with_a_model(
    store: SQLAlchemySegmentStore,
    monkeypatch: pytest.MonkeyPatch,
    seed: int,
) -> None:
    """Random sequences of every data operation agree with an in-memory model.

    Adds, forgets, partition deletes and re-creates, and bounded purge
    calls over a few keys. Each forget's tombstones name exactly the
    derivatives it drops from the model, after every step each live key
    reads back exactly the model's segments and links, and once the purge
    drains, the tables hold exactly the live rows, and the live keys'
    tombstones with the derivatives they name.
    """
    rng = random.Random(seed)
    monkeypatch.setattr(store, "_purge_max_rows", 3)
    config = _plaintext_partition_config()
    keys = ["model_a", "model_b", "model_c"]
    handles: dict[str, SQLAlchemySegmentStorePartition] = {}
    # Each live key's events, with each event's segments and their links.
    model: dict[str, dict[UUID, dict[UUID, set[UUID]]]] = {}
    tombstones: dict[str, int] = {}
    tombstoned_derivatives: dict[str, int] = {}

    for step in range(120):
        key = rng.choice(keys)
        operation = rng.choice(["add", "add", "forget", "delete_partition", "purge"])
        if operation == "purge":
            await store.purge_deleted_partitions()
        elif operation == "delete_partition":
            await store.delete_partition(key)
            handles.pop(key, None)
            model.pop(key, None)
            tombstones.pop(key, None)
            tombstoned_derivatives.pop(key, None)
        else:
            if key not in handles:
                handles[key] = await store.open_or_create_partition(key, config)
                model[key] = {}
                tombstones[key] = 0
                tombstoned_derivatives[key] = 0
            if operation == "add":
                segments = _random_model_events(rng, ts_offset_seconds=step * 10)
                await _add_segments(handles[key], segments)
                for segment, links in segments.items():
                    model[key].setdefault(segment.event_uuid, {})[segment.uuid] = links
            else:
                victims = rng.sample(
                    sorted(model[key]), min(len(model[key]), rng.randint(0, 3))
                )
                deleted = await handles[key].delete_event_segments(
                    [*victims, uuid4()], lease_seconds=0
                )
                unlinked = {
                    segment_uuid: sorted(links)
                    for victim in victims
                    for segment_uuid, links in model[key].pop(victim).items()
                }
                assert {
                    tombstone.segment_uuid: sorted(tombstone.derivative_uuids)
                    for tombstone in deleted
                } == unlinked
                tombstones[key] += len(deleted)
                tombstoned_derivatives[key] += sum(map(len, unlinked.values()))

        for live_key, handle in handles.items():
            await _assert_reads_agree_with_the_model(handle, model[live_key])

    while await store.purge_deleted_partitions():
        pass
    events = [event for live_events in model.values() for event in live_events.values()]
    async with store._create_session() as session:
        row_counts = [
            (
                await session.execute(select(func.count()).select_from(table))
            ).scalar_one()
            for table in (
                EventRow,
                SegmentRow,
                DerivativeRow,
                PendingWriteRow,
                TombstoneRow,
                TombstoneDerivativeRow,
            )
        ]
    assert row_counts == [
        len(events),
        sum(map(len, events)),
        sum(len(links) for event in events for links in event.values()),
        0,
        sum(tombstones.values()),
        sum(tombstoned_derivatives.values()),
    ]


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
    """Every data operation at once completes, errs only as documented, and strands nothing.

    Writers add and forget their own events in live partitions,
    churners create, fill, and delete partitions under shared keys, and
    two purgers drain throughout. Everything finishes within the timeout,
    so no lock cycle stalls it; the only errors are the domain outcomes
    of churning shared keys; and after a final drain no row belongs to a
    dead incarnation, while each writer's partition holds exactly what it
    left, and the derivative UUIDs its forgets' tombstones hold.
    """
    rng = random.Random(seed)
    monkeypatch.setattr(store, "_purge_max_rows", 3)
    config = _plaintext_partition_config()
    live = [
        await store.open_or_create_partition(f"churn_live_{index}", config)
        for index in range(2)
    ]
    # Each writer's events, by UUID, with their link counts; an event has
    # one segment.
    written: dict[UUID, dict[UUID, int]] = {
        partition._incarnation: {} for partition in live
    }
    tombstoned_links: dict[UUID, int] = {
        partition._incarnation: 0 for partition in live
    }
    finished = asyncio.Event()

    async def purger() -> None:
        while not finished.is_set():
            if not await store.purge_deleted_partitions():
                await asyncio.sleep(0.01)

    async def work() -> None:
        await asyncio.gather(
            *(
                _write_and_forget_own_events(
                    partition, written[partition._incarnation], tombstoned_links, rng
                )
                for partition in live
            ),
            *(_churn_shared_keys(store, config, rng) for _ in range(3)),
        )
        finished.set()

    await asyncio.wait_for(asyncio.gather(work(), purger(), purger()), 120)
    while await store.purge_deleted_partitions():
        pass

    live_incarnations = select(PartitionRow.incarnation)
    stranded = [
        await _count_rows(store, table.incarnation.not_in(live_incarnations))
        for table in (
            EventRow,
            SegmentRow,
            DerivativeRow,
            PendingWriteRow,
            TombstoneRow,
            TombstoneDerivativeRow,
        )
    ]
    async with store._create_session() as session:
        queue_depth = (
            await session.execute(select(func.count()).select_from(PurgeQueueRow))
        ).scalar_one()
    assert (stranded, queue_depth) == ([0, 0, 0, 0, 0, 0], 0)
    for incarnation, events in written.items():
        assert await _row_counts(store, incarnation) == (
            len(events),
            sum(events.values()),
        )
        assert (
            await _count_rows(store, TombstoneDerivativeRow.incarnation == incarnation)
            == tombstoned_links[incarnation]
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
        await partition.delete_event_segments(victims, lease_seconds=0)
        for victim in victims:
            tombstoned_links[partition._incarnation] += mine.pop(victim)


async def _churn_shared_keys(
    store: SQLAlchemySegmentStore,
    config: SegmentStorePartitionConfig,
    rng: random.Random,
) -> None:
    """Create, fill, and delete partitions under keys other churners share."""
    for _ in range(8):
        key = f"churn_shared_{rng.randrange(2)}"
        try:
            partition = await store.open_or_create_partition(key, config)
            await _add_segments(
                partition,
                _links(
                    *(
                        _seg(ts_offset_seconds=offset)
                        for offset in range(rng.randint(1, 4))
                    )
                ),
            )
            await store.delete_partition(key)
        except (
            SegmentStorePartitionAlreadyExistsError,
            SegmentStorePartitionHandleStaleError,
        ):
            pass


@pytest.fixture
def sqlite_recorded_statements(
    sqlalchemy_sqlite_engine: AsyncEngine,
) -> Iterator[list[str]]:
    """SQLite counterpart of recorded_statements."""
    yield from _record_statements(sqlalchemy_sqlite_engine)


@pytest.mark.asyncio
async def test_sqlite_delete_partition_touches_only_registry_and_queue(
    sqlite_store: SQLAlchemySegmentStore,
    sqlite_recorded_statements: list[str],
) -> None:
    """Deletion is O(1) on SQLite too: no data-table statements."""
    partition = await sqlite_store.open_or_create_partition(
        "sqlite_big_delete", _plaintext_partition_config()
    )
    await _add_segments(
        partition, _links(*(_seg(ts_offset_seconds=i) for i in range(20)))
    )
    sqlite_recorded_statements.clear()

    await sqlite_store.delete_partition("sqlite_big_delete")

    data_tables = BaseSegmentStore.metadata.tables.keys() - {
        PartitionRow.__tablename__,
        PurgeQueueRow.__tablename__,
    }
    data_statements = [
        statement
        for statement in sqlite_recorded_statements
        if any(table in statement for table in data_tables)
    ]
    assert not data_statements
    assert any("segment_store_gc" in s for s in sqlite_recorded_statements)


@pytest.mark.asyncio
async def test_sqlite_mint_detects_collision_with_concurrent_deletion(
    sqlite_store: SQLAlchemySegmentStore,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """SQLite counterpart of the concurrent mint-collision test.

    A deletion holds its write transaction open; the colliding mint's
    registry insert must wait on SQLite's write lock, and once the
    deletion commits, the post-insert queue re-check sees the entry and
    re-mints.
    """
    victim = await sqlite_store.open_or_create_partition(
        "sq_mint_victim", _plaintext_partition_config()
    )
    victim_incarnation = victim._incarnation
    await _add_segments(victim, _links(_seg()))

    offered = []

    def colliding_uuid4() -> UUID:
        if not offered:
            offered.append(victim_incarnation)
            return victim_incarnation
        return uuid4()

    monkeypatch.setattr(sqlalchemy_segment_store, "uuid4", colliding_uuid4)

    creator_started = asyncio.Event()
    original_insert_partition_row = sqlite_store._insert_partition_row

    async def signalling_insert_partition_row(partition_key, incarnation, config):
        creator_started.set()
        await original_insert_partition_row(partition_key, incarnation, config)

    monkeypatch.setattr(
        sqlite_store, "_insert_partition_row", signalling_insert_partition_row
    )

    async with (
        victim._create_session() as remote_session,
        remote_session.begin(),
    ):
        # DML opens the write transaction and holds SQLite's write lock.
        await remote_session.execute(
            insert(PurgeQueueRow).values(
                incarnation=victim_incarnation,
                enqueued_at=func.now(),
            )
        )
        await remote_session.execute(
            delete(PartitionRow).where(PartitionRow.partition_key == "sq_mint_victim")
        )
        creator = asyncio.create_task(
            sqlite_store.create_partition(
                "sq_mint_fresh", _plaintext_partition_config()
            )
        )
        # The started event proves the mint reached its insert before the
        # sample below; the grace period is the unavoidable wall-clock
        # element (SQLite exposes no lock-wait state to observe). While
        # the deletion's write lock is held, the mint cannot finish.
        await asyncio.wait_for(creator_started.wait(), 30)
        done, _pending = await asyncio.wait([creator], timeout=0.8)
        assert not done, "the colliding mint did not wait for the deletion"
    # Deletion committed on exiting begin().
    await asyncio.wait_for(creator, 30)

    fresh = await sqlite_store.open_partition("sq_mint_fresh")
    assert fresh is not None
    assert offered
    assert fresh._incarnation != victim_incarnation

    while await sqlite_store.purge_deleted_partitions():
        pass
    assert (await sqlite_store.open_partition("sq_mint_fresh")) is not None


def test_empty_partition_key_is_named_as_empty() -> None:
    """An empty key is reported as empty, not as containing invalid characters."""
    with pytest.raises(ValueError, match="must not be empty"):
        validate_partition_key("")
