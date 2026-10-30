"""Tests for SQLAlchemyEventMemoryStore — SQLite (unit) and PostgreSQL (integration)."""

import asyncio
import contextlib
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
    Context,
    Derivative,
    Neighborhood,
    NullContext,
    ProducerContext,
    Segment,
    TextBlock,
)
from memmachine_server.episodic_memory.event_memory.event_memory_store import (
    EventMemoryStoreAttemptsExhaustedError,
    EventMemoryStoreEventAlreadyStoredError,
    EventMemoryStorePartition,
    EventMemoryStorePartitionAlreadyExistsError,
    EventMemoryStorePartitionConfig,
    EventMemoryStorePartitionConfigMismatchError,
    EventMemoryStorePartitionHandleStaleError,
    PendingDerivatives,
    PendingDerivativesClaim,
    TombstoneClaim,
    sqlalchemy_event_memory_store,
)
from memmachine_server.episodic_memory.event_memory.event_memory_store.sqlalchemy_event_memory_store import (
    BaseEventMemoryStore,
    DerivativeLinkRow,
    EventRow,
    PartitionRow,
    PendingDerivativeRow,
    PendingDerivativesRow,
    PurgeQueueRow,
    SegmentRow,
    SQLAlchemyEventMemoryStore,
    SQLAlchemyEventMemoryStoreParams,
    SQLAlchemyEventMemoryStorePartition,
    TombstoneDerivativeRow,
    TombstoneRow,
)
from memmachine_server.episodic_memory.event_memory.event_memory_store.utils import (
    validate_partition_key,
)

PARTITION_KEY = "test_partition"
BASE_TIME = datetime(2024, 1, 1, tzinfo=UTC)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _author(name: str) -> Context:
    return ProducerContext(producer=name)


def _seg(
    *,
    event_uuid: UUID | None = None,
    index: int = 0,
    offset: int = 0,
    ts_offset_seconds: int = 0,
    text: str = "hello",
    source_id: str | None = None,
    context: Context | None = None,
    properties: dict | None = None,
) -> Segment:
    return Segment(
        uuid=uuid4(),
        event_uuid=event_uuid or uuid4(),
        index=index,
        offset=offset,
        timestamp=BASE_TIME + timedelta(seconds=ts_offset_seconds),
        source_id=source_id,
        block=TextBlock(text=text),
        context=context if context is not None else NullContext(),
        properties=properties or {},
    )


async def _drop_schema(engine: AsyncEngine) -> None:
    """Drop the store's tables, so the next startup starts clean."""
    async with engine.begin() as conn:
        await conn.run_sync(BaseEventMemoryStore.metadata.drop_all)


def _links(*segments: Segment) -> dict[Segment, list[UUID]]:
    """Build a segment-to-derivative-UUIDs mapping with one derivative per segment."""
    return {seg: [uuid4()] for seg in segments}


async def _add_segments(
    partition: SQLAlchemyEventMemoryStorePartition,
    segments_to_derivative_uuids: Mapping[Segment, Iterable[UUID]],
) -> None:
    """Add the segments' events as a finished ingest leaves them: linked, nothing pending."""
    segments_by_event, pending_derivatives = _events(segments_to_derivative_uuids)
    await partition.add_events(segments_by_event, pending_derivatives)
    await partition.delete_pending_derivatives(
        pending.uuid for pending in pending_derivatives
    )


def _events(
    segments_to_derivative_uuids: Mapping[Segment, Iterable[UUID]],
) -> tuple[dict[UUID, list[Segment]], list[PendingDerivatives]]:
    """`add_events` arguments for the segments' events, with a derivative per given UUID.

    Each event gets one pending derivatives record, and each derivative the
    timestamp, source id and block of its segment, and the record the
    event's properties, which its segments carry.
    """
    segments_by_event: dict[UUID, list[Segment]] = {}
    derivatives_by_event: dict[UUID, list[Derivative]] = {}
    for segment, derivative_uuids in segments_to_derivative_uuids.items():
        segments_by_event.setdefault(segment.event_uuid, []).append(segment)
        derivatives_by_event.setdefault(segment.event_uuid, []).extend(
            Derivative(
                uuid=derivative_uuid,
                segment_uuid=segment.uuid,
                timestamp=segment.timestamp,
                source_id=segment.source_id,
                block=segment.block,
            )
            for derivative_uuid in derivative_uuids
        )
    pending_derivatives = [
        PendingDerivatives(
            uuid=uuid4(),
            event_uuid=event_uuid,
            properties=segments_by_event[event_uuid][0].properties,
            derivatives=derivatives,
        )
        for event_uuid, derivatives in derivatives_by_event.items()
    ]
    return segments_by_event, pending_derivatives


async def _windows(
    partition: EventMemoryStorePartition,
    seeds: list[Segment],
    **read_options,
) -> dict[UUID, list[Segment]]:
    """Each known seed inside its neighborhood, in the store's order: the walk, flattened."""
    neighborhoods = await partition.get_segment_neighborhoods(
        [seed.uuid for seed in seeds], **read_options
    )
    return {
        seed.uuid: [
            *neighborhoods[seed.uuid].before,
            seed,
            *neighborhoods[seed.uuid].after,
        ]
        for seed in seeds
        if seed.uuid in neighborhoods
    }


def _plaintext_partition_config() -> EventMemoryStorePartitionConfig:
    """Return the default plaintext partition config."""
    return EventMemoryStorePartitionConfig()


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
    owner: SQLAlchemyEventMemoryStore | SQLAlchemyEventMemoryStorePartition,
    incarnation: UUID,
) -> tuple[int, int]:
    """The incarnation's segment and derivative-link row counts."""
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
                .select_from(DerivativeLinkRow)
                .where(DerivativeLinkRow.incarnation == incarnation)
            )
        ).scalar_one()
    return segments, links


async def _count_rows(
    owner: SQLAlchemyEventMemoryStore | SQLAlchemyEventMemoryStorePartition,
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
) -> AsyncIterator[SQLAlchemyEventMemoryStore]:
    store = SQLAlchemyEventMemoryStore(
        SQLAlchemyEventMemoryStoreParams(engine=sqlalchemy_sqlite_engine)
    )
    await store.startup()
    yield store
    await _drop_schema(sqlalchemy_sqlite_engine)


@pytest_asyncio.fixture
async def pg_store(
    sqlalchemy_pg_engine: AsyncEngine,
) -> AsyncIterator[SQLAlchemyEventMemoryStore]:
    store = SQLAlchemyEventMemoryStore(
        SQLAlchemyEventMemoryStoreParams(engine=sqlalchemy_pg_engine)
    )
    await store.startup()
    yield store
    await _drop_schema(sqlalchemy_pg_engine)


@pytest.fixture(
    params=[
        "sqlite_store",
        pytest.param("pg_store", marks=pytest.mark.integration),
    ],
)
def store(request) -> SQLAlchemyEventMemoryStore:
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
    store: SQLAlchemyEventMemoryStore,
) -> SQLAlchemyEventMemoryStorePartition:
    return await store.open_or_create_partition(
        PARTITION_KEY,
        _plaintext_partition_config(),
    )


# ===================================================================
# add_events
# ===================================================================


@pytest.mark.asyncio
async def test_add_events_and_get_contexts(
    partition: SQLAlchemyEventMemoryStorePartition,
) -> None:
    seg = _seg(text="a")
    await _add_segments(partition, _links(seg))

    result = await partition.get_segments([seg.uuid])
    assert seg.uuid in result
    assert result[seg.uuid].uuid == seg.uuid


@pytest.mark.asyncio
async def test_add_events_with_properties(
    partition: SQLAlchemyEventMemoryStorePartition,
) -> None:
    seg = _seg(properties={"color": "red", "score": 42})
    await _add_segments(partition, _links(seg))

    result = await partition.get_segments([seg.uuid])
    returned = result[seg.uuid]
    assert returned.properties == {"color": "red", "score": 42}


@pytest.mark.asyncio
async def test_add_events_with_producer_context(
    partition: SQLAlchemyEventMemoryStorePartition,
) -> None:
    ctx = _author("User")
    seg = _seg(context=ctx)
    await _add_segments(partition, _links(seg))

    result = await partition.get_segments([seg.uuid])
    assert result[seg.uuid].context == ctx

    async with partition._create_session() as session:
        row = (
            await session.execute(select(SegmentRow).where(SegmentRow.uuid == seg.uuid))
        ).scalar_one()
    assert json.loads(row.context) == {"context_type": "producer", "producer": "User"}
    assert json.loads(row.block) == {"block_type": "text", "text": "hello"}


@pytest.mark.asyncio
async def test_add_events_with_no_context(
    partition: SQLAlchemyEventMemoryStorePartition,
) -> None:
    seg = _seg()
    await _add_segments(partition, _links(seg))

    async with partition._create_session() as session:
        row = (
            await session.execute(select(SegmentRow).where(SegmentRow.uuid == seg.uuid))
        ).scalar_one()
    assert json.loads(row.context) == {"context_type": "null"}

    result = await partition.get_segments([seg.uuid])
    assert result[seg.uuid].context == NullContext()


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
    partition: SQLAlchemyEventMemoryStorePartition,
    tz: timezone,
) -> None:
    """A timezone-aware timestamp roundtrips with its instant and offset intact.

    SQLite's DateTime(timezone=True) discards tzinfo and stores the wall-clock
    fields verbatim, so a non-UTC timestamp that is not normalized to UTC before
    writing comes back shifted by its offset. Regression test for that bug.
    """
    ts = datetime(2024, 1, 1, 13, 30, 45, tzinfo=tz)
    seg = Segment(
        source_id="src",
        uuid=uuid4(),
        event_uuid=uuid4(),
        index=0,
        offset=0,
        timestamp=ts,
        block=TextBlock(text="tz"),
        context=NullContext(),
        properties={},
    )
    await _add_segments(partition, _links(seg))

    result = await partition.get_segments([seg.uuid])
    returned = result[seg.uuid].timestamp
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
    partition: SQLAlchemyEventMemoryStorePartition,
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

    result = await _windows(
        partition,
        [early],
        after=5,
        property_filter=parse_filter(f"timestamp <= date('{bound}')"),
    )
    assert [s.uuid for s in result[early.uuid]] == [early.uuid]


@pytest.mark.asyncio
async def test_add_events_empty(
    partition: SQLAlchemyEventMemoryStorePartition,
) -> None:
    await partition.add_events({}, [])


@pytest.mark.asyncio
async def test_add_events_rejects_a_batch_naming_a_held_event(
    partition: SQLAlchemyEventMemoryStorePartition,
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
    segments_by_event, pending_derivatives = _events(batch)

    with pytest.raises(EventMemoryStoreEventAlreadyStoredError) as exc_info:
        await partition.add_events(segments_by_event, pending_derivatives)

    assert exc_info.value.event_uuids == {segment.event_uuid for segment in held}
    assert await _row_counts(partition, partition._incarnation) == (2, 2)
    assert (
        await _count_rows(
            partition, PendingDerivativesRow.incarnation == partition._incarnation
        )
        == 0
    )
    assert set(
        await partition.get_segments(segment.uuid for segment in (*held, *batch))
    ) == {segment.uuid for segment in held}
    await _add_segments(partition, _links(new))


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "mismatch",
    [
        "segment_listed_under_another_event",
        "pending_derivatives_of_an_event_outside_the_batch",
        "derivative_of_a_segment_outside_its_event",
    ],
)
async def test_add_events_rejects_mismatched_event_naming(
    partition: SQLAlchemyEventMemoryStorePartition,
    mismatch: str,
) -> None:
    """The error names the segment, pending derivatives, or derivative at fault."""
    first, second = _seg(), _seg()
    segments_by_event, pending_derivatives = _events(_links(first, second))
    if mismatch == "segment_listed_under_another_event":
        segments_by_event[first.event_uuid].append(second)
        at_fault = second.uuid
    elif mismatch == "pending_derivatives_of_an_event_outside_the_batch":
        del segments_by_event[second.event_uuid]
        at_fault = pending_derivatives[1].uuid
    else:
        pending_derivatives[0].derivatives.extend(pending_derivatives[1].derivatives)
        at_fault = pending_derivatives[1].derivatives[0].uuid

    with pytest.raises(ValueError, match=str(at_fault)):
        await partition.add_events(segments_by_event, pending_derivatives)

    assert await _row_counts(partition, partition._incarnation) == (0, 0)
    assert (
        await _count_rows(partition, EventRow.incarnation == partition._incarnation)
        == 0
    )


@pytest.mark.asyncio
async def test_add_events_failing_partway_writes_nothing(
    partition: SQLAlchemyEventMemoryStorePartition,
) -> None:
    """An add whose link rows are rejected keeps none of its events or segments either."""
    taken = uuid4()
    await _add_segments(partition, {_seg(): [taken]})
    rejected = [_seg(ts_offset_seconds=1), _seg(ts_offset_seconds=2)]
    segments_by_event, pending_derivatives = _events(
        {rejected[0]: [uuid4()], rejected[1]: [taken]}
    )

    with pytest.raises(IntegrityError):
        await partition.add_events(segments_by_event, pending_derivatives)

    assert await _row_counts(partition, partition._incarnation) == (1, 1)
    assert (
        await _count_rows(partition, EventRow.incarnation == partition._incarnation)
        == 1
    )
    assert (
        await _count_rows(
            partition, PendingDerivativesRow.incarnation == partition._incarnation
        )
        == 0
    )
    assert await partition.get_segments([segment.uuid for segment in rejected]) == {}


@pytest.mark.asyncio
async def test_concurrent_adds_of_one_event_store_it_once(
    partition: SQLAlchemyEventMemoryStorePartition,
) -> None:
    """Of two racing adds of one event, one stores it and the other is rejected whole."""
    event_uuid = uuid4()
    batches = [
        _links(*(_seg(event_uuid=event_uuid, index=index) for index in range(3)))
        for _ in range(2)
    ]

    outcomes = await asyncio.gather(
        *(partition.add_events(*_events(batch)) for batch in batches),
        return_exceptions=True,
    )

    [error] = [outcome for outcome in outcomes if outcome is not None]
    assert isinstance(error, EventMemoryStoreEventAlreadyStoredError)
    assert error.event_uuids == {event_uuid}
    stored = await partition.get_segments(
        segment.uuid for batch in batches for segment in batch
    )
    assert set(stored) in [{segment.uuid for segment in batch} for batch in batches]
    assert await _row_counts(partition, partition._incarnation) == (3, 3)
    assert (
        await _count_rows(
            partition, PendingDerivativesRow.incarnation == partition._incarnation
        )
        == 1
    )


@pytest.mark.asyncio
async def test_pending_derivatives_roundtrip_through_a_claim(
    store: SQLAlchemyEventMemoryStore,
) -> None:
    """A claim returns the pending derivatives as added, timestamp offsets and the event's properties included."""
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
            source_id="user",
            block=TextBlock(text="first"),
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
    pending_derivatives = PendingDerivatives(
        uuid=uuid4(),
        event_uuid=event_uuid,
        properties={
            "color": "red",
            "score": 42,
            "ratio": 0.5,
            "flag": True,
            "seen": datetime(2024, 2, 1, tzinfo=timezone(timedelta(hours=-8))),
        },
        derivatives=derivatives,
    )
    await partition.add_events({event_uuid: segments}, [pending_derivatives])

    claim = await store.claim_pending_derivatives(min_age_seconds=0, lease_seconds=60)

    assert claim is not None
    assert claim.partition_key == PARTITION_KEY
    assert (claim.generation, claim.attempt) == (1, 1)
    claimed = claim.pending_derivatives
    assert (claimed.uuid, claimed.event_uuid) == (pending_derivatives.uuid, event_uuid)
    assert claimed.properties == pending_derivatives.properties
    by_uuid = operator.attrgetter("uuid")
    assert sorted(claimed.derivatives, key=by_uuid) == sorted(derivatives, key=by_uuid)
    # Aware-datetime equality compares instants; the offsets come back too.
    assert {
        derivative.uuid: derivative.timestamp.utcoffset()
        for derivative in claimed.derivatives
    } == {
        derivative.uuid: derivative.timestamp.utcoffset() for derivative in derivatives
    }


# ===================================================================
# delete_pending_derivatives
# ===================================================================


@pytest.mark.asyncio
async def test_delete_pending_derivatives_removes_only_the_pending_record(
    store: SQLAlchemyEventMemoryStore,
) -> None:
    """The record's derivatives stay linked and other records stay pending. Idempotent."""
    partition = await store.open_or_create_partition(
        PARTITION_KEY, _plaintext_partition_config()
    )
    finished, unfinished = _seg(), _seg()
    segments_by_event, pending_derivatives = _events(
        {finished: [uuid4(), uuid4()], unfinished: [uuid4()]}
    )
    await partition.add_events(segments_by_event, pending_derivatives)
    finished_pending, unfinished_pending = pending_derivatives

    for _ in range(2):
        await partition.delete_pending_derivatives([finished_pending.uuid])

    derivatives = [
        derivative
        for pending in pending_derivatives
        for derivative in pending.derivatives
    ]
    assert await partition.get_segment_uuids_by_derivative_uuids(
        derivative.uuid for derivative in derivatives
    ) == {derivative.uuid: derivative.segment_uuid for derivative in derivatives}
    assert (
        await _count_rows(
            partition, PendingDerivativeRow.pending_uuid == finished_pending.uuid
        )
        == 0
    )
    claim = await store.claim_pending_derivatives(min_age_seconds=0, lease_seconds=60)
    assert claim is not None
    assert claim.pending_derivatives.uuid == unfinished_pending.uuid
    assert (
        await store.claim_pending_derivatives(min_age_seconds=0, lease_seconds=60)
        is None
    )


# ===================================================================
# get_segments and get_segment_neighborhoods
# ===================================================================


@pytest.mark.asyncio
async def test_contexts_empty_seeds(
    partition: SQLAlchemyEventMemoryStorePartition,
) -> None:
    result = await partition.get_segments([])
    assert result == {}


@pytest.mark.asyncio
async def test_contexts_unknown_seed(
    partition: SQLAlchemyEventMemoryStorePartition,
) -> None:
    """An unknown uuid has no entry, in the lookup and in the walk."""
    await _add_segments(partition, _links(_seg()))

    assert await partition.get_segments([uuid4()]) == {}
    assert await partition.get_segment_neighborhoods([uuid4()], before=2, after=2) == {}


@pytest.mark.asyncio
async def test_contexts_seed_only(
    partition: SQLAlchemyEventMemoryStorePartition,
) -> None:
    """With before=0 and after=0 the walk adds nothing around the seed."""
    seg = _seg()
    await _add_segments(partition, _links(seg))

    result = await partition.get_segments([seg.uuid])
    assert result[seg.uuid].uuid == seg.uuid
    windows = await _windows(partition, [seg])
    assert [s.uuid for s in windows[seg.uuid]] == [seg.uuid]


@pytest.mark.asyncio
async def test_contexts_backward(
    partition: SQLAlchemyEventMemoryStorePartition,
) -> None:
    ep = uuid4()
    segs = [_seg(event_uuid=ep, offset=i, ts_offset_seconds=i) for i in range(5)]
    await _add_segments(partition, _links(*segs))

    seed = segs[3]
    result = await _windows(partition, [seed], before=2)
    ctx = result[seed.uuid]
    # backward(2) + seed = 3 segments
    assert len(ctx) == 3
    uuids = [s.uuid for s in ctx]
    assert uuids == [segs[1].uuid, segs[2].uuid, seed.uuid]


@pytest.mark.asyncio
async def test_contexts_forward(
    partition: SQLAlchemyEventMemoryStorePartition,
) -> None:
    ep = uuid4()
    segs = [_seg(event_uuid=ep, offset=i, ts_offset_seconds=i) for i in range(5)]
    await _add_segments(partition, _links(*segs))

    seed = segs[1]
    result = await _windows(partition, [seed], after=2)
    ctx = result[seed.uuid]
    # seed + forward(2) = 3 segments
    assert len(ctx) == 3
    uuids = [s.uuid for s in ctx]
    assert uuids == [seed.uuid, segs[2].uuid, segs[3].uuid]


@pytest.mark.asyncio
async def test_contexts_backward_and_forward(
    partition: SQLAlchemyEventMemoryStorePartition,
) -> None:
    ep = uuid4()
    segs = [_seg(event_uuid=ep, offset=i, ts_offset_seconds=i) for i in range(7)]
    await _add_segments(partition, _links(*segs))

    seed = segs[3]
    result = await _windows(partition, [seed], before=2, after=2)
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
    partition: SQLAlchemyEventMemoryStorePartition,
) -> None:
    """Requesting more context than available returns what exists."""
    ep = uuid4()
    segs = [_seg(event_uuid=ep, offset=i, ts_offset_seconds=i) for i in range(3)]
    await _add_segments(partition, _links(*segs))

    seed = segs[0]
    result = await _windows(partition, [seed], before=10, after=10)
    ctx = result[seed.uuid]
    assert len(ctx) == 3
    uuids = [s.uuid for s in ctx]
    assert uuids == [segs[0].uuid, segs[1].uuid, segs[2].uuid]


@pytest.mark.asyncio
async def test_contexts_multiple_seeds(
    partition: SQLAlchemyEventMemoryStorePartition,
) -> None:
    ep = uuid4()
    segs = [_seg(event_uuid=ep, offset=i, ts_offset_seconds=i) for i in range(10)]
    await _add_segments(partition, _links(*segs))

    seed_a, seed_b = segs[2], segs[7]
    result = await _windows(
        partition,
        [seed_a, seed_b],
        before=1,
        after=1,
    )
    assert seed_a.uuid in result
    assert seed_b.uuid in result
    ctx_a = [s.uuid for s in result[seed_a.uuid]]
    ctx_b = [s.uuid for s in result[seed_b.uuid]]
    assert ctx_a == [segs[1].uuid, seed_a.uuid, segs[3].uuid]
    assert ctx_b == [segs[6].uuid, seed_b.uuid, segs[8].uuid]


@pytest.mark.asyncio
async def test_contexts_with_properties(
    partition: SQLAlchemyEventMemoryStorePartition,
) -> None:
    """Properties are loaded for seed and context segments."""
    ep = uuid4()
    s0 = _seg(event_uuid=ep, offset=0, ts_offset_seconds=0, properties={"k": "v0"})
    s1 = _seg(event_uuid=ep, offset=1, ts_offset_seconds=1, properties={"k": "v1"})
    s2 = _seg(event_uuid=ep, offset=2, ts_offset_seconds=2, properties={"k": "v2"})
    await _add_segments(partition, _links(s0, s1, s2))

    result = await _windows(partition, [s1], before=1, after=1)
    ctx = result[s1.uuid]
    assert len(ctx) == 3
    assert ctx[0].properties == {"k": "v0"}
    assert ctx[1].properties == {"k": "v1"}
    assert ctx[2].properties == {"k": "v2"}


@pytest.mark.asyncio
async def test_contexts_property_filter(
    partition: SQLAlchemyEventMemoryStorePartition,
) -> None:
    """Property filter excludes context rows that don't match."""
    ep = uuid4()
    s0 = _seg(event_uuid=ep, offset=0, ts_offset_seconds=0, properties={"tag": "a"})
    s1 = _seg(event_uuid=ep, offset=1, ts_offset_seconds=1, properties={"tag": "b"})
    s2 = _seg(event_uuid=ep, offset=2, ts_offset_seconds=2, properties={"tag": "a"})
    s3 = _seg(event_uuid=ep, offset=3, ts_offset_seconds=3, properties={"tag": "a"})
    await _add_segments(partition, _links(s0, s1, s2, s3))

    filt = Comparison(field="m.tag", op="=", value="a")
    result = await _windows(
        partition,
        [s2],
        before=5,
        after=5,
        property_filter=filt,
    )
    ctx = result[s2.uuid]
    uuids = [s.uuid for s in ctx]
    # s1 excluded (tag=b); s0 backward, s2 seed, s3 forward
    assert uuids == [s0.uuid, s2.uuid, s3.uuid]


@pytest.mark.asyncio
async def test_context_comes_from_the_segments_nearest_the_seed(
    partition: SQLAlchemyEventMemoryStorePartition,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A context read takes its context from a bounded number of segments on each side.

    With the bound at 4 segments, a match 5 or more segments from the seed
    is not context, however few matches lie within the bound, and an
    unfiltered read returns no segment beyond it either.
    """
    monkeypatch.setattr(sqlalchemy_event_memory_store, "_MAX_CONTEXT_DISTANCE", 4)
    tags = {1: "a", 2: "b", 3: "b", 4: "a", 5: "a", 6: "a"}
    seed = _seg(ts_offset_seconds=0, properties={"tag": "a"})
    backward = {
        d: _seg(ts_offset_seconds=-d, properties={"tag": t}) for d, t in tags.items()
    }
    forward = {
        d: _seg(ts_offset_seconds=d, properties={"tag": t}) for d, t in tags.items()
    }
    await _add_segments(partition, _links(seed, *backward.values(), *forward.values()))

    filtered = await _windows(
        partition,
        [seed],
        before=5,
        after=5,
        property_filter=Comparison(field="m.tag", op="=", value="a"),
    )
    assert [segment.uuid for segment in filtered[seed.uuid]] == [
        backward[4].uuid,
        backward[1].uuid,
        seed.uuid,
        forward[1].uuid,
        forward[4].uuid,
    ]

    unfiltered = await _windows(partition, [seed], before=6, after=6)
    assert [segment.uuid for segment in unfiltered[seed.uuid]] == [
        *(backward[distance].uuid for distance in (4, 3, 2, 1)),
        seed.uuid,
        *(forward[distance].uuid for distance in (1, 2, 3, 4)),
    ]


@pytest.mark.asyncio
async def test_contexts_filter_by_context_producer(
    partition: SQLAlchemyEventMemoryStorePartition,
) -> None:
    """`context.producer` is not a stored property; the filter just matches nothing.

    Bare names that aren't `timestamp` are looked up as `_<name>` in the JSON
    properties (matching EventMemoryPartition's `_to_vector_record_property` convention
    for system fields). `context.producer` becomes `_context.producer`, which
    isn't a stored key on any segment, so the filter returns no contexts.
    """
    ep = uuid4()
    s0 = _seg(
        event_uuid=ep,
        offset=0,
        ts_offset_seconds=0,
        context=_author("Alice"),
    )
    s1 = _seg(
        event_uuid=ep,
        offset=1,
        ts_offset_seconds=1,
        context=_author("Bob"),
    )
    s2 = _seg(
        event_uuid=ep,
        offset=2,
        ts_offset_seconds=2,
        context=_author("Alice"),
    )
    await _add_segments(partition, _links(s0, s1, s2))

    filt = Comparison(field="context.producer", op="=", value="Alice")
    assert await partition.get_segments([s0.uuid], property_filter=filt) == {}
    neighborhoods = await partition.get_segment_neighborhoods(
        [s0.uuid], before=5, after=5, property_filter=filt
    )
    assert neighborhoods == {s0.uuid: Neighborhood(before=[], after=[])}


@pytest.mark.asyncio
async def test_contexts_filter_by_context_type(
    partition: SQLAlchemyEventMemoryStorePartition,
) -> None:
    """`context.context_type` is not a stored property; filter matches nothing."""
    ep = uuid4()
    s0 = _seg(
        event_uuid=ep,
        offset=0,
        ts_offset_seconds=0,
        context=_author("Alice"),
    )
    s1 = _seg(
        event_uuid=ep,
        offset=1,
        ts_offset_seconds=1,
        context=NullContext(),
    )
    s2 = _seg(
        event_uuid=ep,
        offset=2,
        ts_offset_seconds=2,
        context=_author("Bob"),
    )
    await _add_segments(partition, _links(s0, s1, s2))

    filt = Comparison(field="context.context_type", op="=", value="producer")
    assert await partition.get_segments([s0.uuid], property_filter=filt) == {}
    neighborhoods = await partition.get_segment_neighborhoods(
        [s0.uuid], before=5, after=5, property_filter=filt
    )
    assert neighborhoods == {s0.uuid: Neighborhood(before=[], after=[])}


@pytest.mark.asyncio
async def test_contexts_session_isolation(store: SQLAlchemyEventMemoryStore) -> None:
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

    result = await _windows(partition, [s_seed], before=5, after=5)
    ctx = result[s_seed.uuid]
    uuids = [s.uuid for s in ctx]
    assert s_other.uuid not in uuids
    assert uuids == [s_seed.uuid, s_after.uuid]


@pytest.mark.asyncio
async def test_contexts_chronological_order(
    partition: SQLAlchemyEventMemoryStorePartition,
) -> None:
    """Context segments are returned in chronological order."""
    ep = uuid4()
    segs = [_seg(event_uuid=ep, offset=i, ts_offset_seconds=i) for i in range(5)]
    await _add_segments(partition, _links(*segs))

    result = await _windows(partition, [segs[2]], before=10, after=10)
    ctx = result[segs[2].uuid]
    timestamps = [s.timestamp for s in ctx]
    assert timestamps == sorted(timestamps)


@pytest.mark.asyncio
async def test_context_preserved_in_segment_windows(
    partition: SQLAlchemyEventMemoryStorePartition,
) -> None:
    """Context is preserved when retrieving segment contexts (backward/forward)."""
    ep = uuid4()
    ctx_user = _author("User")
    ctx_assistant = _author("Assistant")
    s0 = _seg(event_uuid=ep, offset=0, ts_offset_seconds=0, context=ctx_user)
    s1 = _seg(event_uuid=ep, offset=1, ts_offset_seconds=1, context=ctx_assistant)
    s2 = _seg(event_uuid=ep, offset=2, ts_offset_seconds=2, context=ctx_user)
    await _add_segments(partition, _links(s0, s1, s2))

    result = await _windows(partition, [s1], before=1, after=1)
    ctx = result[s1.uuid]
    assert len(ctx) == 3
    assert ctx[0].context == ctx_user
    assert ctx[1].context == ctx_assistant
    assert ctx[2].context == ctx_user


@pytest.mark.asyncio
@pytest.mark.parametrize("random_seed", [1, 2, 3])
async def test_random_context_reads_agree_with_a_model(
    store: SQLAlchemyEventMemoryStore,
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
        before = rng.randint(0, 4)
        after = rng.randint(0, 4)
        property_filter = rng.choice(filters)

        result = await _windows(
            partition,
            seeds,
            before=before,
            after=after,
            property_filter=property_filter,
        )
        assert {
            seed_uuid: [segment.uuid for segment in window]
            for seed_uuid, window in result.items()
        } == _model_window_uuids(
            timeline,
            seeds,
            before=before,
            after=after,
            property_filter=property_filter,
        ), (property_filter, before, after)


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


def _model_window_uuids(
    timeline: list[Segment],
    seeds: list[Segment],
    *,
    before: int,
    after: int,
    property_filter: FilterExpr | None,
) -> dict[UUID, list[UUID]]:
    """Each seed inside its neighborhood as get_segment_neighborhoods returns it, as UUIDs.

    `timeline` is the partition's segments in chronological order. The
    seed is an address, present whether or not it matches; the filter
    selects the neighbors.
    """

    def matches(segment: Segment) -> bool:
        return property_filter is None or _sql_truth(segment, property_filter) is True

    windows: dict[UUID, list[UUID]] = {}
    for seed in seeds:
        position = timeline.index(seed)
        earlier = [segment for segment in timeline[:position] if matches(segment)]
        later = [segment for segment in timeline[position + 1 :] if matches(segment)]
        window = [
            *(earlier[-before:] if before else []),
            seed,
            *later[:after],
        ]
        windows[seed.uuid] = [segment.uuid for segment in window]
    return windows


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
    partition: SQLAlchemyEventMemoryStorePartition,
) -> None:
    seg = _seg()
    d1, d2 = uuid4(), uuid4()
    await _add_segments(partition, {seg: [d1, d2]})

    result = await partition.get_segment_uuids_by_derivative_uuids([d1, d2])
    assert result == {d1: seg.uuid, d2: seg.uuid}


@pytest.mark.asyncio
async def test_get_segment_uuids_by_derivative_uuids_empty(
    partition: SQLAlchemyEventMemoryStorePartition,
) -> None:
    result = await partition.get_segment_uuids_by_derivative_uuids([])
    assert result == {}


@pytest.mark.asyncio
async def test_get_segment_uuids_by_derivative_uuids_omits_unknown(
    partition: SQLAlchemyEventMemoryStorePartition,
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
    store: SQLAlchemyEventMemoryStore,
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
async def test_get_segment_uuids_by_derivative_uuids_after_forget_events(
    partition: SQLAlchemyEventMemoryStorePartition,
) -> None:
    """Forgetting an event takes its derivatives' mapping with it."""
    kept, dropped = _seg(offset=0), _seg(offset=1)
    kept_derivative, dropped_derivative = uuid4(), uuid4()
    await _add_segments(
        partition, {kept: [kept_derivative], dropped: [dropped_derivative]}
    )

    await partition.forget_events([dropped.event_uuid])

    result = await partition.get_segment_uuids_by_derivative_uuids(
        [kept_derivative, dropped_derivative]
    )
    assert result == {kept_derivative: kept.uuid}


# ===================================================================
# forget_events
# ===================================================================


@pytest.mark.asyncio
async def test_forget_events_removes_segments_and_links(
    partition: SQLAlchemyEventMemoryStorePartition,
) -> None:
    seg = _seg()
    derivative = uuid4()
    await _add_segments(partition, {seg: [derivative]})

    await partition.forget_events([seg.event_uuid])

    assert await partition.get_segments([seg.uuid]) == {}
    assert await partition.get_segment_uuids_by_derivative_uuids([derivative]) == {}
    assert await _row_counts(partition, partition._incarnation) == (0, 0)
    assert (
        await _count_rows(partition, EventRow.incarnation == partition._incarnation)
        == 0
    )


@pytest.mark.asyncio
async def test_forget_events_leaves_other_events_intact(
    partition: SQLAlchemyEventMemoryStorePartition,
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

    await partition.forget_events([forgotten_event])

    result = await partition.get_segments(
        segment.uuid for segment in (*forgotten, *kept)
    )
    assert set(result) == {segment.uuid for segment in kept}


@pytest.mark.asyncio
async def test_forget_events_returns_a_tombstone_naming_the_unlinked_derivatives(
    partition: SQLAlchemyEventMemoryStorePartition,
) -> None:
    """The tombstone names the derivatives of every forgotten event's segments, and no others."""
    forgotten = [uuid4(), uuid4()]
    links = {
        _seg(event_uuid=forgotten[0], index=0): [uuid4(), uuid4()],
        _seg(event_uuid=forgotten[0], index=1): [uuid4()],
        _seg(event_uuid=forgotten[0], index=2): [],
        _seg(event_uuid=forgotten[1]): [uuid4()],
        _seg(): [uuid4()],
    }
    await _add_segments(partition, links)

    tombstone = await partition.forget_events([*forgotten, uuid4()])

    assert tombstone is not None
    assert sorted(tombstone.derivative_uuids) == sorted(
        derivative_uuid
        for segment, derivative_uuids in links.items()
        if segment.event_uuid in forgotten
        for derivative_uuid in derivative_uuids
    )


@pytest.mark.asyncio
async def test_forget_events_deletes_pending_derivatives(
    store: SQLAlchemyEventMemoryStore,
) -> None:
    partition = await store.open_or_create_partition(
        PARTITION_KEY, _plaintext_partition_config()
    )
    segment = _seg()
    segments_by_event, pending_derivatives = _events({segment: [uuid4(), uuid4()]})
    await partition.add_events(segments_by_event, pending_derivatives)
    [pending] = pending_derivatives

    tombstone = await partition.forget_events([segment.event_uuid])

    assert tombstone is not None
    assert sorted(tombstone.derivative_uuids) == sorted(
        derivative.uuid for derivative in pending.derivatives
    )
    assert await _count_rows(partition, PendingDerivativesRow.uuid == pending.uuid) == 0
    assert (
        await _count_rows(partition, PendingDerivativeRow.pending_uuid == pending.uuid)
        == 0
    )
    assert (
        await store.claim_pending_derivatives(min_age_seconds=0, lease_seconds=60)
        is None
    )


@pytest.mark.asyncio
async def test_forget_events_without_derivatives_returns_none(
    partition: SQLAlchemyEventMemoryStorePartition,
) -> None:
    """Unknown events and events without derivatives leave no tombstone; the latter are still forgotten."""
    bare = _seg()
    await _add_segments(partition, {bare: []})

    assert await partition.forget_events([]) is None
    assert await partition.forget_events([uuid4()]) is None
    assert await partition.forget_events([bare.event_uuid]) is None

    assert await partition.get_segments([bare.uuid]) == {}
    assert (
        await _count_rows(partition, EventRow.incarnation == partition._incarnation)
        == 0
    )
    assert (
        await _count_rows(partition, TombstoneRow.incarnation == partition._incarnation)
        == 0
    )


@pytest.mark.asyncio
async def test_forget_events_is_idempotent(
    partition: SQLAlchemyEventMemoryStorePartition,
) -> None:
    segment = _seg()
    await _add_segments(partition, _links(segment))

    assert await partition.forget_events([segment.event_uuid]) is not None
    assert await partition.forget_events([segment.event_uuid]) is None
    assert (
        await _count_rows(partition, TombstoneRow.incarnation == partition._incarnation)
        == 1
    )


@pytest.mark.asyncio
async def test_forgotten_event_can_be_added_again(
    partition: SQLAlchemyEventMemoryStorePartition,
) -> None:
    """A changed event is forgotten and added again under its UUID."""
    event_uuid = uuid4()
    await _add_segments(partition, _links(_seg(event_uuid=event_uuid)))
    await partition.forget_events([event_uuid])
    changed = _seg(event_uuid=event_uuid, text="changed")

    await _add_segments(partition, _links(changed))

    segments = await partition.get_segments([changed.uuid])
    assert segments[changed.uuid].block == TextBlock(text="changed")


@pytest.mark.asyncio
async def test_claim_on_pending_derivatives_of_a_forgotten_event_ends_without_effect(
    store: SQLAlchemyEventMemoryStore,
) -> None:
    """The claim's record went with its event, and the tombstone names the claimed derivatives."""
    partition = await store.open_or_create_partition(
        PARTITION_KEY, _plaintext_partition_config()
    )
    segment = _seg()
    segments_by_event, pending_derivatives = _events(_links(segment))
    await partition.add_events(segments_by_event, pending_derivatives)
    claim = await store.claim_pending_derivatives(min_age_seconds=0, lease_seconds=60)
    assert claim is not None

    await partition.forget_events([segment.event_uuid])
    await store.finish_pending_derivatives_claim(claim)
    await store.fail_pending_derivatives_claim(claim, RuntimeError("index unavailable"))

    assert (
        await _count_rows(
            partition, PendingDerivativesRow.incarnation == partition._incarnation
        )
        == 0
    )
    tombstone_claim = await store.claim_tombstone()
    assert tombstone_claim is not None
    assert tombstone_claim.tombstone.derivative_uuids == [
        derivative.uuid for derivative in claim.pending_derivatives.derivatives
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize("operation", ["add_events", "forget_events"])
async def test_event_write_whose_commit_fails_changes_nothing(
    store: SQLAlchemyEventMemoryStore,
    operation: str,
) -> None:
    """An event write whose commit fails leaves the partition as it was."""
    partition = await store.open_or_create_partition(
        "uncommitted", _plaintext_partition_config()
    )
    kept = _seg()
    await _add_segments(partition, _links(kept))
    write = (
        partition.add_events(*_events(_links(_seg(ts_offset_seconds=1))))
        if operation == "add_events"
        else partition.forget_events([kept.event_uuid])
    )

    with (
        _failing_next_commit(store._engine),
        pytest.raises(RuntimeError, match="injected commit failure"),
    ):
        await write

    assert await _row_counts(store, partition._incarnation) == (1, 1)
    for table, count in (
        (EventRow, 1),
        (PendingDerivativesRow, 0),
        (TombstoneRow, 0),
    ):
        assert (
            await _count_rows(store, table.incarnation == partition._incarnation)
            == count
        )
    assert kept.uuid in await partition.get_segments([kept.uuid])


# ===================================================================
# Claims
# ===================================================================

_CLAIMED_TABLES = pytest.mark.parametrize(
    "table",
    [PendingDerivativesRow, TombstoneRow],
    ids=["pending_derivatives", "tombstone"],
)


@pytest.mark.asyncio
async def test_pending_derivatives_are_claimable_once_min_age_has_passed(
    store: SQLAlchemyEventMemoryStore,
) -> None:
    partition = await store.open_or_create_partition(
        PARTITION_KEY, _plaintext_partition_config()
    )
    pending_uuid = await _add_claimable_record(partition, PendingDerivativesRow)

    for age in (timedelta(0), timedelta(minutes=50)):
        await _backdate(store, PendingDerivativesRow.created_at, age)
        assert (
            await store.claim_pending_derivatives(
                min_age_seconds=3600, lease_seconds=60
            )
            is None
        )
    await _backdate(store, PendingDerivativesRow.created_at, timedelta(minutes=70))
    claim = await store.claim_pending_derivatives(
        min_age_seconds=3600, lease_seconds=60
    )
    assert _claimed_record_uuid(claim) == pending_uuid


@pytest.mark.asyncio
@_CLAIMED_TABLES
async def test_claims_take_the_oldest_record_first(
    store: SQLAlchemyEventMemoryStore,
    table: type[PendingDerivativesRow | TombstoneRow],
) -> None:
    partition = await store.open_or_create_partition(
        PARTITION_KEY, _plaintext_partition_config()
    )
    newer = await _add_claimable_record(partition, table)
    older = await _add_claimable_record(partition, table)
    age = (
        PendingDerivativesRow.created_at
        if table is PendingDerivativesRow
        else TombstoneRow.enqueued_at
    )
    await _backdate(store, age, timedelta(minutes=5), table.uuid == older)

    claimed = [_claimed_record_uuid(await _claim(store, table)) for _ in range(2)]
    assert claimed == [older, newer]


@pytest.mark.asyncio
@_CLAIMED_TABLES
async def test_claim_holds_its_record_for_the_lease(
    store: SQLAlchemyEventMemoryStore,
    table: type[PendingDerivativesRow | TombstoneRow],
) -> None:
    """A record is claimed again only once its claim's lease is over, by a later generation."""
    partition = await store.open_or_create_partition(
        PARTITION_KEY, _plaintext_partition_config()
    )
    record_uuid = await _add_claimable_record(partition, table)

    first = await _claim(store, table)
    assert first is not None
    assert (_claimed_record_uuid(first), first.generation, first.attempt) == (
        record_uuid,
        1,
        1,
    )
    assert await _claim(store, table) is None

    await _backdate(store, table.claimed_until, timedelta(days=1))
    second = await _claim(store, table)
    assert second is not None
    assert (_claimed_record_uuid(second), second.generation, second.attempt) == (
        record_uuid,
        2,
        2,
    )


@pytest.mark.asyncio
async def test_finishing_a_pending_derivatives_claim_deletes_the_record_only_for_the_latest_claim(
    store: SQLAlchemyEventMemoryStore,
) -> None:
    """A claim that outlasted its lease leaves the record to the later claim."""
    partition = await store.open_or_create_partition(
        PARTITION_KEY, _plaintext_partition_config()
    )
    pending_uuid = await _add_claimable_record(partition, PendingDerivativesRow)
    superseded = await store.claim_pending_derivatives(
        min_age_seconds=0, lease_seconds=60
    )
    await _backdate(store, PendingDerivativesRow.claimed_until, timedelta(days=1))
    latest = await store.claim_pending_derivatives(min_age_seconds=0, lease_seconds=60)
    assert superseded is not None
    assert latest is not None

    await store.finish_pending_derivatives_claim(superseded)
    assert await _count_rows(store, PendingDerivativesRow.uuid == pending_uuid) == 1

    await store.finish_pending_derivatives_claim(latest)
    assert await _count_rows(store, PendingDerivativesRow.uuid == pending_uuid) == 0
    assert (
        await _count_rows(store, PendingDerivativeRow.pending_uuid == pending_uuid) == 0
    )


@pytest.mark.asyncio
@_CLAIMED_TABLES
async def test_failing_a_superseded_claim_changes_nothing(
    store: SQLAlchemyEventMemoryStore,
    table: type[PendingDerivativesRow | TombstoneRow],
) -> None:
    """It neither ends the later claim nor starts a backoff."""
    partition = await store.open_or_create_partition(
        PARTITION_KEY, _plaintext_partition_config()
    )
    record_uuid = await _add_claimable_record(partition, table)
    superseded = await _claim(store, table)
    await _backdate(store, table.claimed_until, timedelta(days=1))
    assert await _claim(store, table) is not None
    assert superseded is not None

    await _fail_claim(store, superseded)

    async with store._create_session() as session:
        claimed_until, last_failed_at = (
            await session.execute(
                select(table.claimed_until, table.last_failed_at).where(
                    table.uuid == record_uuid
                )
            )
        ).one()
    assert claimed_until is not None
    assert last_failed_at is None


@pytest.mark.asyncio
@_CLAIMED_TABLES
async def test_failed_claim_backs_off_doubling_up_to_the_cap(
    store: SQLAlchemyEventMemoryStore,
    monkeypatch: pytest.MonkeyPatch,
    table: type[PendingDerivativesRow | TombstoneRow],
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
        await _backdate(
            store, table.last_failed_at, timedelta(seconds=backoff_seconds - 15)
        )
        assert await _claim(store, table) is None
        await _backdate(
            store, table.last_failed_at, timedelta(seconds=backoff_seconds + 15)
        )
    claim = await _claim(store, table)
    assert claim is not None
    assert claim.attempt == 4


@pytest.mark.asyncio
@_CLAIMED_TABLES
async def test_record_is_no_longer_claimed_after_the_maximum_failed_claims(
    store: SQLAlchemyEventMemoryStore,
    caplog: pytest.LogCaptureFixture,
    table: type[PendingDerivativesRow | TombstoneRow],
) -> None:
    """The store reports the record once its claims failed the maximum number of times in a row."""
    partition = await store.open_or_create_partition(
        PARTITION_KEY, _plaintext_partition_config()
    )
    await _add_claimable_record(partition, table)
    maximum = sqlalchemy_event_memory_store._MAX_CLAIM_ATTEMPTS_WITHOUT_PROGRESS

    with caplog.at_level(logging.ERROR, logger=sqlalchemy_event_memory_store.__name__):
        for attempt in range(1, maximum + 1):
            claim = await _claim(store, table)
            assert claim is not None
            assert claim.attempt == attempt
            await _fail_claim(store, claim)
            await _backdate(store, table.last_failed_at, timedelta(days=1))

    assert await _claim(store, table) is None
    reports = [
        record for record in caplog.records if "no longer claimed" in record.message
    ]
    assert [record.levelno for record in reports] == [logging.ERROR]


@pytest.mark.asyncio
@_CLAIMED_TABLES
async def test_claims_skip_deleted_partitions(
    store: SQLAlchemyEventMemoryStore,
    table: type[PendingDerivativesRow | TombstoneRow],
) -> None:
    """A deleted partition's records are passed over, however old."""
    config = _plaintext_partition_config()
    doomed = await store.open_or_create_partition("doomed_claims", config)
    live = await store.open_or_create_partition("live_claims", config)
    live_uuid = await _add_claimable_record(live, table)
    await _add_claimable_record(doomed, table)
    age = (
        PendingDerivativesRow.created_at
        if table is PendingDerivativesRow
        else TombstoneRow.enqueued_at
    )
    await _backdate(
        store, age, timedelta(minutes=5), table.incarnation == doomed._incarnation
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
async def test_pending_derivatives_claim_partition_is_bound_to_the_records_incarnation(
    store: SQLAlchemyEventMemoryStore,
) -> None:
    config = _plaintext_partition_config()
    partition = await store.open_or_create_partition("claimed_incarnation", config)
    segment = _seg()
    derivative_uuid = uuid4()
    segments_by_event, pending_derivatives = _events({segment: [derivative_uuid]})
    await partition.add_events(segments_by_event, pending_derivatives)
    claim = await store.claim_pending_derivatives(min_age_seconds=0, lease_seconds=60)
    assert claim is not None

    await claim.partition.require_current()
    assert await claim.partition.get_segment_uuids_by_derivative_uuids(
        [derivative_uuid]
    ) == {derivative_uuid: segment.uuid}

    await store.delete_partition("claimed_incarnation")
    await store.open_or_create_partition("claimed_incarnation", config)
    with pytest.raises(EventMemoryStorePartitionHandleStaleError):
        await claim.partition.require_current()


# ===================================================================
# Tombstones
# ===================================================================


@pytest.mark.asyncio
async def test_new_tombstone_is_claimable_at_once(
    store: SQLAlchemyEventMemoryStore,
) -> None:
    partition = await store.open_or_create_partition(
        PARTITION_KEY, _plaintext_partition_config()
    )
    segment = _seg()
    derivative_uuids = [uuid4(), uuid4()]
    await _add_segments(partition, {segment: derivative_uuids})
    tombstone = await partition.forget_events([segment.event_uuid])
    assert tombstone is not None

    claim = await store.claim_tombstone()

    assert claim is not None
    assert claim.partition_key == PARTITION_KEY
    assert claim.tombstone.uuid == tombstone.uuid
    assert sorted(claim.tombstone.derivative_uuids) == sorted(derivative_uuids)
    assert claim.second_delete is False
    assert (claim.generation, claim.attempt) == (1, 1)


@pytest.mark.asyncio
@pytest.mark.parametrize("recorded_by", ["partition", "claim"])
async def test_tombstone_awaits_the_retention_after_its_first_delete(
    store: SQLAlchemyEventMemoryStore,
    monkeypatch: pytest.MonkeyPatch,
    recorded_by: str,
) -> None:
    """Its second delete falls due once the retention has passed since the forget.

    Recording the first delete again, once the second is claimed, changes
    nothing.
    """
    monkeypatch.setattr(store, "_tombstone_retention_seconds", 3600)
    partition = await store.open_or_create_partition(
        PARTITION_KEY, _plaintext_partition_config()
    )
    tombstone_uuid = await _add_claimable_record(partition, TombstoneRow)
    if recorded_by == "partition":
        await partition.record_tombstone_first_delete(tombstone_uuid)
    else:
        first = await store.claim_tombstone()
        assert first is not None
        await store.finish_tombstone_claim(first)

    for age in (timedelta(0), timedelta(minutes=50)):
        await _backdate(store, TombstoneRow.enqueued_at, age)
        assert await store.claim_tombstone() is None
    await _backdate(store, TombstoneRow.enqueued_at, timedelta(minutes=70))
    second = await store.claim_tombstone()
    assert second is not None
    assert (second.tombstone.uuid, second.second_delete, second.attempt) == (
        tombstone_uuid,
        True,
        1,
    )

    await partition.record_tombstone_first_delete(tombstone_uuid)
    assert await store.claim_tombstone() is None


@pytest.mark.asyncio
async def test_finishing_a_second_delete_claim_removes_the_tombstone(
    store: SQLAlchemyEventMemoryStore,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(store, "_tombstone_retention_seconds", 3600)
    partition = await store.open_or_create_partition(
        PARTITION_KEY, _plaintext_partition_config()
    )
    tombstone_uuid = await _add_claimable_record(partition, TombstoneRow)
    await partition.record_tombstone_first_delete(tombstone_uuid)
    await _backdate(store, TombstoneRow.enqueued_at, timedelta(hours=2))
    claim = await store.claim_tombstone()
    assert claim is not None
    assert claim.second_delete is True

    await store.finish_tombstone_claim(claim)

    assert await _count_rows(store, TombstoneRow.uuid == tombstone_uuid) == 0
    assert (
        await _count_rows(
            store, TombstoneDerivativeRow.tombstone_uuid == tombstone_uuid
        )
        == 0
    )
    assert await store.claim_tombstone() is None


@pytest.mark.asyncio
async def test_claim_timings_come_from_params(
    sqlalchemy_sqlite_engine: AsyncEngine,
) -> None:
    """The configured retry backoff and tombstone retention govern claims."""
    store = SQLAlchemyEventMemoryStore(
        SQLAlchemyEventMemoryStoreParams(
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
        failed = await store.claim_tombstone()
        assert failed is not None
        await _fail_claim(store, failed)
        await _backdate(store, TombstoneRow.last_failed_at, timedelta(seconds=20))
        first = await store.claim_tombstone()
        assert first is not None
        await store.finish_tombstone_claim(first)

        await _backdate(store, TombstoneRow.enqueued_at, timedelta(minutes=2))
        second = await store.claim_tombstone()
        assert second is not None
        assert second.second_delete is True
    finally:
        async with sqlalchemy_sqlite_engine.begin() as conn:
            await conn.run_sync(BaseEventMemoryStore.metadata.drop_all)


async def _add_claimable_record(
    partition: SQLAlchemyEventMemoryStorePartition,
    table: type[PendingDerivativesRow | TombstoneRow],
) -> UUID:
    """Leave a claimable record of `table` in the partition, and return its UUID.

    Pending derivatives are an added event's, left pending; a tombstone is
    a forgotten event's.
    """
    segment = _seg()
    segments_by_event, pending_derivatives = _events(_links(segment))
    await partition.add_events(segments_by_event, pending_derivatives)
    if table is PendingDerivativesRow:
        return pending_derivatives[0].uuid
    tombstone = await partition.forget_events([segment.event_uuid])
    assert tombstone is not None
    return tombstone.uuid


async def _claim(
    store: SQLAlchemyEventMemoryStore,
    table: type[PendingDerivativesRow | TombstoneRow],
) -> PendingDerivativesClaim | TombstoneClaim | None:
    """Claim a record of `table`, taking pending derivatives at any age."""
    if table is PendingDerivativesRow:
        return await store.claim_pending_derivatives(
            min_age_seconds=0, lease_seconds=60
        )
    return await store.claim_tombstone()


def _claimed_record_uuid(
    claim: PendingDerivativesClaim | TombstoneClaim | None,
) -> UUID | None:
    """The UUID of the record `claim` holds; None for no claim."""
    if isinstance(claim, PendingDerivativesClaim):
        return claim.pending_derivatives.uuid
    if isinstance(claim, TombstoneClaim):
        return claim.tombstone.uuid
    return None


async def _fail_claim(
    store: SQLAlchemyEventMemoryStore,
    claim: PendingDerivativesClaim | TombstoneClaim,
) -> None:
    """End `claim` as failed."""
    error = RuntimeError("index unavailable")
    if isinstance(claim, PendingDerivativesClaim):
        await store.fail_pending_derivatives_claim(claim, error)
    else:
        await store.fail_tombstone_claim(claim, error)


async def _backdate(
    store: SQLAlchemyEventMemoryStore,
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


async def _get_partition(engine: AsyncEngine) -> SQLAlchemyEventMemoryStorePartition:
    """Create a partition handle that shares the engine."""
    store = SQLAlchemyEventMemoryStore(SQLAlchemyEventMemoryStoreParams(engine=engine))
    return await store.open_or_create_partition(
        PARTITION_KEY,
        _plaintext_partition_config(),
    )


@pytest.mark.asyncio
async def test_concurrent_add_disjoint(
    store: SQLAlchemyEventMemoryStore,
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
    store: SQLAlchemyEventMemoryStore,
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
            result = await _windows(part, [segs[5]], before=5, after=5)
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
    store: SQLAlchemyEventMemoryStore,
) -> None:
    """A walk must not crash if events are forgotten concurrently."""
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
                await _windows(part, [seg], before=2, after=2)
            except Exception as e:
                errors.append(e)
            await asyncio.sleep(0.01)

    async def event_forgetter() -> None:
        part = await _get_partition(engine)
        for ep in events[1:3]:
            await part.forget_events([ep])
            await asyncio.sleep(0.02)

    await asyncio.gather(context_reader(), event_forgetter())

    assert errors == []


# ===================================================================
# Partition lifecycle
# ===================================================================


@pytest.mark.asyncio
async def test_open_or_create_partition_defaults_to_plaintext_config(
    store: SQLAlchemyEventMemoryStore,
) -> None:
    partition = await store.open_or_create_partition(
        "plaintext_default",
        _plaintext_partition_config(),
    )
    assert partition.config.payload_codec_config == PlaintextPayloadCodecConfig()


@pytest.mark.asyncio
async def test_create_partition(store: SQLAlchemyEventMemoryStore) -> None:
    await store.create_partition("new_partition", _plaintext_partition_config())
    partition = await store.open_partition("new_partition")
    assert partition is not None


@pytest.mark.asyncio
async def test_create_partition_already_exists(
    store: SQLAlchemyEventMemoryStore,
) -> None:
    await store.create_partition("dup_partition", _plaintext_partition_config())
    with pytest.raises(EventMemoryStorePartitionAlreadyExistsError):
        await store.create_partition("dup_partition", _plaintext_partition_config())


@pytest.mark.asyncio
async def test_open_partition_nonexistent(store: SQLAlchemyEventMemoryStore) -> None:
    result = await store.open_partition("nonexistent")
    assert result is None


@pytest.mark.asyncio
async def test_open_partition_existing(store: SQLAlchemyEventMemoryStore) -> None:
    await store.create_partition("existing", _plaintext_partition_config())
    partition = await store.open_partition("existing")
    assert partition is not None


@pytest.mark.asyncio
async def test_open_or_create_partition_creates(
    store: SQLAlchemyEventMemoryStore,
) -> None:
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
    store: SQLAlchemyEventMemoryStore,
) -> None:
    await store.create_partition("idem", _plaintext_partition_config())
    partition = await store.open_or_create_partition(
        "idem",
        _plaintext_partition_config(),
    )
    assert partition is not None


@pytest.mark.asyncio
async def test_delete_partition_removes_data(store: SQLAlchemyEventMemoryStore) -> None:
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
    pg_store: SQLAlchemyEventMemoryStore,
    sqlalchemy_pg_engine: AsyncEngine,
) -> None:
    """Deleting one partition must not drop the derivative-link foreign key."""
    keeper = await pg_store.open_or_create_partition(
        "keeper_fk", _plaintext_partition_config()
    )
    await pg_store.open_or_create_partition("doomed_fk", _plaintext_partition_config())

    await pg_store.delete_partition("doomed_fk")

    # A derivative link pointing at no segment must still be rejected in the
    # partition that survived.
    with pytest.raises(IntegrityError):
        async with sqlalchemy_pg_engine.begin() as connection:
            await connection.execute(
                text(
                    "INSERT INTO event_memory_store_dv_ln"
                    " (incarnation, uuid, segment_uuid)"
                    " VALUES (:incarnation, :uuid, :segment_uuid)"
                ),
                {
                    "incarnation": keeper._incarnation,
                    "uuid": uuid4(),
                    "segment_uuid": uuid4(),
                },
            )


@pytest.mark.asyncio
async def test_delete_partition_keeps_other_partitions_cascading(
    store: SQLAlchemyEventMemoryStore,
) -> None:
    """Deleting one partition must not disable the derivative-link cascade."""
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

    # The foreign keys from segments to events and from derivative links to
    # segments must still cascade for the partitions that were not deleted.
    await keeper.forget_events([segment.event_uuid])
    assert await _row_counts(keeper, keeper._incarnation) == (0, 0)


@pytest.mark.asyncio
async def test_delete_partition_cascades_segments(
    store: SQLAlchemyEventMemoryStore,
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
    result = await new_partition.get_segments([seg.uuid])
    assert result == {}
    deriv_result = await new_partition.get_segment_uuids_by_derivative_uuids([d1])
    assert deriv_result == {}


@pytest.mark.asyncio
async def test_delete_partition_idempotent(store: SQLAlchemyEventMemoryStore) -> None:
    await store.delete_partition("never_existed")


@pytest.mark.asyncio
async def test_partition_key_validation_invalid_chars(
    store: SQLAlchemyEventMemoryStore,
) -> None:
    with pytest.raises(ValueError, match="invalid characters"):
        await store.create_partition("UPPER", _plaintext_partition_config())
    with pytest.raises(ValueError, match="invalid characters"):
        await store.create_partition("has-hyphen", _plaintext_partition_config())
    with pytest.raises(ValueError, match="invalid characters"):
        await store.create_partition("has space", _plaintext_partition_config())


@pytest.mark.asyncio
async def test_partition_key_validation_too_long(
    store: SQLAlchemyEventMemoryStore,
) -> None:
    with pytest.raises(ValueError, match="too long"):
        await store.create_partition("a" * 33, _plaintext_partition_config())


@pytest.mark.integration
@pytest.mark.asyncio
async def test_pg_context_preserved_via_lateral_join(
    pg_store: SQLAlchemyEventMemoryStore,
) -> None:
    """Context is preserved when retrieved via the LATERAL join path (multiple seeds)."""
    partition = await pg_store.open_or_create_partition(
        PARTITION_KEY,
        _plaintext_partition_config(),
    )
    ep = uuid4()
    ctx_user = _author("User")
    ctx_assistant = _author("Assistant")
    s0 = _seg(event_uuid=ep, offset=0, ts_offset_seconds=0, context=ctx_user)
    s1 = _seg(event_uuid=ep, offset=1, ts_offset_seconds=1, context=ctx_assistant)
    s2 = _seg(event_uuid=ep, offset=2, ts_offset_seconds=2, context=ctx_user)
    s3 = _seg(event_uuid=ep, offset=3, ts_offset_seconds=3, context=ctx_assistant)
    s4 = _seg(event_uuid=ep, offset=4, ts_offset_seconds=4, context=ctx_user)
    all_segs = [s0, s1, s2, s3, s4]
    await _add_segments(partition, _links(*all_segs))

    # Two seeds exercises the LATERAL join code path.
    result = await _windows(partition, [s1, s3], before=1, after=1)

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
    pg_store: SQLAlchemyEventMemoryStore,
) -> None:
    """Different context types (producer, None) round-trip correctly on PG."""
    partition = await pg_store.open_or_create_partition(
        PARTITION_KEY,
        _plaintext_partition_config(),
    )
    ctx_msg = _author("User")

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

    result = await partition.get_segments([s_msg.uuid])
    assert result[s_msg.uuid].context == ctx_msg

    result = await partition.get_segments([s_none.uuid])
    assert result[s_none.uuid].context == NullContext()


# ===================================================================
# Incarnation fencing and O(1) deletion
# ===================================================================


@pytest.mark.asyncio
async def test_stale_handle_raises_after_delete(
    store: SQLAlchemyEventMemoryStore,
) -> None:
    """A handle held across deletion must fail loudly, not act."""
    partition = await store.open_or_create_partition(
        "fenced", _plaintext_partition_config()
    )
    seg = _seg()
    await _add_segments(partition, _links(seg))
    await partition.require_current()

    await store.delete_partition("fenced")

    # Empty input may skip the handle check (the ABC-permitted shortcut).
    await partition.add_events({}, [])
    assert await partition.forget_events([]) is None

    with pytest.raises(EventMemoryStorePartitionHandleStaleError):
        await partition.require_current()
    with pytest.raises(EventMemoryStorePartitionHandleStaleError):
        await partition.add_events(*_events(_links(_seg())))
    with pytest.raises(EventMemoryStorePartitionHandleStaleError):
        await partition.delete_pending_derivatives([uuid4()])
    with pytest.raises(EventMemoryStorePartitionHandleStaleError):
        await partition.get_segments([seg.uuid])
    with pytest.raises(EventMemoryStorePartitionHandleStaleError):
        await partition.get_segment_uuids_by_derivative_uuids([uuid4()])
    with pytest.raises(EventMemoryStorePartitionHandleStaleError):
        await partition.forget_events([seg.event_uuid])
    with pytest.raises(EventMemoryStorePartitionHandleStaleError):
        await partition.record_tombstone_first_delete(uuid4())
    # The stale handle deleted nothing; the rows wait for the purge.
    assert await _row_counts(store, partition._incarnation) == (1, 1)


@pytest.mark.integration
@pytest.mark.asyncio
async def test_reads_check_liveness_inside_the_data_statement(
    pg_store: SQLAlchemyEventMemoryStore,
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

    await partition.get_segments([seg.uuid])
    await partition.get_segment_uuids_by_derivative_uuids(derivative_uuids)
    assert len(recorded_statements) == 2
    assert all(
        "EXISTS (SELECT" in s and "event_memory_store_pt" in s
        for s in recorded_statements
    )

    recorded_statements.clear()
    assert await partition.get_segments([uuid4()]) == {}
    assert len(recorded_statements) == 2


@pytest.mark.asyncio
async def test_stale_handle_raises_after_recreate(
    store: SQLAlchemyEventMemoryStore,
) -> None:
    """Re-creating the key must not let an old handle act on the successor."""
    old_handle = await store.open_or_create_partition(
        "reborn", _plaintext_partition_config()
    )
    await store.delete_partition("reborn")
    new_handle = await store.open_or_create_partition(
        "reborn", _plaintext_partition_config()
    )

    with pytest.raises(EventMemoryStorePartitionHandleStaleError):
        await _add_segments(old_handle, _links(_seg()))
    await _add_segments(new_handle, _links(_seg()))  # the live handle works


@pytest.mark.asyncio
async def test_recreated_partition_is_isolated_from_old_rows(
    store: SQLAlchemyEventMemoryStore,
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

    assert await successor.get_segments([seg.uuid]) == {}
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
    store: SQLAlchemyEventMemoryStore,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Purging erases queued incarnations and leaves live partitions alone."""
    # Small default bound, so draining takes the caller's loop.
    monkeypatch.setattr(store, "_purge_max_segments", 2)

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
    assert (await live.get_segments([live_seg.uuid]))[live_seg.uuid]


@pytest.mark.asyncio
async def test_concurrent_purges_reclaim_everything(
    store: SQLAlchemyEventMemoryStore,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Racing purgers neither error nor leave garbage behind.

    Queue-entry claiming partitions the work: whatever the interleaving,
    every dead incarnation is reclaimed exactly once and the drain loops
    all terminate cleanly.
    """
    monkeypatch.setattr(store, "_purge_max_segments", 2)
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
    store: SQLAlchemyEventMemoryStore,
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
    monkeypatch.setattr(store, "_purge_max_segments", 2)

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
        elif held and self is not held[0] and "event_memory_store_sg" in str(statement):
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
    store: SQLAlchemyEventMemoryStore,
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
    monkeypatch.setattr(store, "_purge_max_segments", 2)

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
    store: SQLAlchemyEventMemoryStore,
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
    monkeypatch.setattr(store, "_purge_max_segments", bound)

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
    store: SQLAlchemyEventMemoryStore,
) -> None:
    """Events, segments, links, pending derivatives, and tombstones all go; a live partition keeps its own."""
    config = _plaintext_partition_config()
    doomed = await store.open_or_create_partition("doomed_rows", config)
    live = await store.open_or_create_partition("live_rows", config)
    records: dict[UUID, tuple[UUID, UUID]] = {}
    for partition in (doomed, live):
        kept, pending_segment, forgotten = _seg(), _seg(), _seg()
        await _add_segments(partition, _links(kept, forgotten))
        segments_by_event, pending_derivatives = _events(_links(pending_segment))
        await partition.add_events(segments_by_event, pending_derivatives)
        tombstone = await partition.forget_events([forgotten.event_uuid])
        assert tombstone is not None
        records[partition._incarnation] = (pending_derivatives[0].uuid, tombstone.uuid)
    await store.delete_partition("doomed_rows")

    while await store.purge_deleted_partitions():
        pass

    async def row_counts(incarnation: UUID) -> list[int]:
        pending_uuid, tombstone_uuid = records[incarnation]
        return [
            await _count_rows(store, condition)
            for condition in (
                EventRow.incarnation == incarnation,
                SegmentRow.incarnation == incarnation,
                DerivativeLinkRow.incarnation == incarnation,
                PendingDerivativesRow.incarnation == incarnation,
                PendingDerivativeRow.pending_uuid == pending_uuid,
                TombstoneRow.incarnation == incarnation,
                TombstoneDerivativeRow.tombstone_uuid == tombstone_uuid,
            )
        ]

    assert await row_counts(doomed._incarnation) == [0, 0, 0, 0, 0, 0, 0]
    assert await row_counts(live._incarnation) == [2, 2, 2, 1, 1, 1, 1]
    assert (
        await _count_rows(store, PurgeQueueRow.incarnation == doomed._incarnation) == 0
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("bound", [1, 3, 4, 5, 8, 10])
async def test_purge_continues_through_every_table_within_the_bound(
    store: SQLAlchemyEventMemoryStore,
    monkeypatch: pytest.MonkeyPatch,
    bound: int,
) -> None:
    """Segments, events, and tombstones share each call's bound, and each table resumes where the last call left it.

    The partition holds five segments, three events, and two tombstones, so
    the bounds end calls inside each table and exactly on each table's last
    row.
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
    await partition.add_events(*_events(_links(_seg())))
    for forgotten in (_seg(), _seg()):
        await _add_segments(partition, _links(forgotten))
        await partition.forget_events([forgotten.event_uuid])
    await store.delete_partition("tables_p")
    monkeypatch.setattr(store, "_purge_max_segments", bound)

    async def budgeted_rows() -> int:
        return sum(
            [
                await _count_rows(store, table.incarnation == incarnation)
                for table in (SegmentRow, EventRow, TombstoneRow)
            ]
        )

    left = await budgeted_rows()
    assert left == 10
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
    assert (
        await _count_rows(store, PendingDerivativesRow.incarnation == incarnation) == 0
    )
    assert await _count_rows(store, PurgeQueueRow.incarnation == incarnation) == 0


@pytest.mark.asyncio
async def test_purge_counts_a_tombstone_as_the_derivatives_it_names(
    store: SQLAlchemyEventMemoryStore,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A call purges tombstones naming up to the bound, or one tombstone alone that names more."""
    partition = await store.open_or_create_partition(
        "named_p", _plaintext_partition_config()
    )
    incarnation = partition._incarnation
    for named in (2, 5, 1, 1, 3):
        event_uuid = uuid4()
        await _add_segments(
            partition,
            _links(
                *(_seg(event_uuid=event_uuid, index=index) for index in range(named))
            ),
        )
        await partition.forget_events([event_uuid])
    await store.delete_partition("named_p")
    monkeypatch.setattr(store, "_purge_max_segments", 3)

    async def named_by_tombstone() -> dict[UUID, int]:
        async with store._create_session() as session:
            rows = await session.execute(
                select(TombstoneRow.uuid, func.count())
                .join(
                    TombstoneDerivativeRow,
                    TombstoneDerivativeRow.tombstone_uuid == TombstoneRow.uuid,
                )
                .where(TombstoneRow.incarnation == incarnation)
                .group_by(TombstoneRow.uuid)
            )
            return dict(rows.tuples().all())

    left = await named_by_tombstone()
    assert sorted(left.values()) == [1, 1, 2, 3, 5]
    for _ in range(10):
        more = await store.purge_deleted_partitions()
        now = await named_by_tombstone()
        purged = [named for uuid, named in left.items() if uuid not in now]
        assert purged or not more
        assert sum(purged) <= 3 or len(purged) == 1
        assert set(now) <= set(left)
        left = now
        if not more:
            break
    else:
        pytest.fail("the purge did not finish within 10 calls")
    assert left == {}
    assert await _count_rows(store, PurgeQueueRow.incarnation == incarnation) == 0


@pytest.mark.asyncio
async def test_purging_a_recreated_key_reclaims_only_the_dead_incarnation(
    store: SQLAlchemyEventMemoryStore,
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
    monkeypatch.setattr(store, "_purge_max_segments", 2)

    assert await store.purge_deleted_partitions() is True
    late = [_seg(ts_offset_seconds=20 + index) for index in range(2)]
    await _add_segments(successor, _links(*late))
    while await store.purge_deleted_partitions():
        pass

    assert await _row_counts(store, predecessor._incarnation) == (0, 0)
    assert await _row_counts(store, successor._incarnation) == (5, 5)
    segments = await successor.get_segments(
        [segment.uuid for segment in (*early, *late)]
    )
    assert len(segments) == 5


@pytest.mark.asyncio
@pytest.mark.parametrize("seed", [1, 2, 3])
async def test_racing_purgers_and_writers_leave_exactly_the_live_rows(
    store: SQLAlchemyEventMemoryStore,
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
    monkeypatch.setattr(store, "_purge_max_segments", rng.randint(1, 4))

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

    async def write(partition: SQLAlchemyEventMemoryStorePartition) -> None:
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
    pg_store: SQLAlchemyEventMemoryStore,
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
    pg_store: SQLAlchemyEventMemoryStore,
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

    writer = asyncio.create_task(partition.add_events(*_events(_links(_seg()))))
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

    with contextlib.suppress(EventMemoryStorePartitionHandleStaleError):
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
    pg_store: SQLAlchemyEventMemoryStore,
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
    store: SQLAlchemyEventMemoryStore,
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

    monkeypatch.setattr(sqlalchemy_event_memory_store, "uuid4", colliding_uuid4)

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
    store: SQLAlchemyEventMemoryStore,
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

    monkeypatch.setattr(sqlalchemy_event_memory_store, "uuid4", colliding_uuid4)

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
    store = SQLAlchemyEventMemoryStore(
        SQLAlchemyEventMemoryStoreParams(
            engine=sqlalchemy_sqlite_engine,
            purge_max_segments=2,
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
            _links(*(_seg(event_uuid=event_uuid, index=index) for index in range(2))),
        )
        await store.delete_partition("bound_p")

        assert await store.purge_deleted_partitions() is True
        assert await store.purge_deleted_partitions() is False
    finally:
        await _drop_schema(sqlalchemy_sqlite_engine)


@pytest.mark.integration
@pytest.mark.asyncio
async def test_mint_detects_collision_with_concurrent_deletion(
    pg_store: SQLAlchemyEventMemoryStore,
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

    monkeypatch.setattr(sqlalchemy_event_memory_store, "uuid4", colliding_uuid4)

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
    pg_store: SQLAlchemyEventMemoryStore,
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
    monkeypatch.setattr(pg_store, "_purge_max_segments", 2)
    recorded_statements.clear()

    assert await pg_store.purge_deleted_partitions() is True

    claim_statements = [
        statement
        for statement in recorded_statements
        if statement.startswith("SELECT") and "event_memory_store_gc" in statement
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
    sqlite_store: SQLAlchemyEventMemoryStore,
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

    writer = asyncio.create_task(partition.add_events(*_events(_links(_seg()))))
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
    with contextlib.suppress(EventMemoryStorePartitionHandleStaleError):
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
    store: SQLAlchemyEventMemoryStore,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A persistent constraint failure surfaces after bounded mint attempts.

    Every realistic trip through the collision-retry path beyond a few
    attempts is a persistent database error being retried, not a race;
    the mint raises EventMemoryStoreAttemptsExhaustedError instead of
    hot-looping, with the underlying error chained for diagnosis.
    """
    cause = IntegrityError("stmt", None, Exception("persistent"))
    attempts = []

    async def always_colliding(partition_key, incarnation, config) -> None:
        attempts.append(incarnation)
        raise sqlalchemy_event_memory_store._RegistryInsertRejectedError(
            str(incarnation)
        ) from cause

    monkeypatch.setattr(store, "_insert_partition_row", always_colliding)

    with pytest.raises(EventMemoryStoreAttemptsExhaustedError) as exc_info:
        await store.create_partition("mint_cap", _plaintext_partition_config())
    assert len(attempts) == sqlalchemy_event_memory_store._MAX_MINT_ATTEMPTS
    # The underlying database error stays reachable for diagnosis.
    collision = exc_info.value.__cause__
    assert collision is not None
    assert collision.__cause__ is cause

    attempts.clear()
    with pytest.raises(EventMemoryStoreAttemptsExhaustedError):
        await store.open_or_create_partition("mint_cap", _plaintext_partition_config())
    assert len(attempts) == sqlalchemy_event_memory_store._MAX_MINT_ATTEMPTS

    # The lost-race arm is bounded by the same cap: an insert that keeps
    # losing to a winner that keeps vanishing must not livelock.
    attempts.clear()

    async def always_losing(partition_key, incarnation, config) -> None:
        attempts.append(incarnation)
        raise EventMemoryStorePartitionAlreadyExistsError(partition_key)

    monkeypatch.setattr(store, "_insert_partition_row", always_losing)
    with pytest.raises(EventMemoryStoreAttemptsExhaustedError):
        await store.open_or_create_partition("mint_cap", _plaintext_partition_config())
    assert len(attempts) == sqlalchemy_event_memory_store._MAX_MINT_ATTEMPTS


@pytest.mark.asyncio
async def test_persistent_integrity_error_surfaces_with_cause(
    store: SQLAlchemyEventMemoryStore,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Any integrity rejection is retried boundedly; the cause stays chained.

    A rejected registry insert is treated as a possible incarnation
    collision and re-minted up to the attempt bound; a persistent cause
    surfaces through EventMemoryStoreAttemptsExhaustedError with the
    driver's error chained for diagnosis. A NOT NULL violation on the
    incarnation column stands in for a cause that is not a collision.
    """
    monkeypatch.setattr(sqlalchemy_event_memory_store, "uuid4", lambda: None)

    for create in (store.create_partition, store.open_or_create_partition):
        with pytest.raises(EventMemoryStoreAttemptsExhaustedError) as exc_info:
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
    store = SQLAlchemyEventMemoryStore(
        SQLAlchemyEventMemoryStoreParams(
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
        await _drop_schema(sqlalchemy_sqlite_engine)


@pytest.mark.asyncio
async def test_empty_incarnations_do_not_consume_row_budget(
    store: SQLAlchemyEventMemoryStore,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A backlog of empty entries leaves the row budget for real rows."""
    monkeypatch.setattr(store, "_purge_max_segments", 2)
    for index in range(3):
        await store.create_partition(f"noop_{index}", _plaintext_partition_config())
        await store.delete_partition(f"noop_{index}")
    rowful = await store.open_or_create_partition(
        "rowful", _plaintext_partition_config()
    )
    event_uuid = uuid4()
    await _add_segments(
        rowful,
        _links(*(_seg(event_uuid=event_uuid, index=index) for index in range(2))),
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
    store: SQLAlchemyEventMemoryStore,
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
        await _add_segments(partition, _links(_seg()))
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
    monkeypatch.setattr(store, "_purge_max_segments", 1)
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
    pg_store: SQLAlchemyEventMemoryStore,
    recorded_statements: list[str],
) -> None:
    """The FIFO key is one clock for every server: the database's."""
    await pg_store.create_partition("db_clock", _plaintext_partition_config())
    recorded_statements.clear()

    await pg_store.delete_partition("db_clock")

    enqueues = [
        s
        for s in recorded_statements
        if s.startswith("INSERT INTO event_memory_store_gc")
    ]
    assert len(enqueues) == 1
    assert "now()" in enqueues[0]


class _ForeignPayloadCodecConfig(PlaintextPayloadCodecConfig):
    """A distinct codec-config type standing in for a future variant."""


@pytest.mark.asyncio
async def test_open_or_create_with_different_config_raises_mismatch(
    store: SQLAlchemyEventMemoryStore,
) -> None:
    """Reopening a key under a different config is refused, not adapted."""
    await store.create_partition("cfg_guard", _plaintext_partition_config())

    requested = EventMemoryStorePartitionConfig(
        payload_codec_config=_ForeignPayloadCodecConfig()
    )
    with pytest.raises(EventMemoryStorePartitionConfigMismatchError) as exc_info:
        await store.open_or_create_partition("cfg_guard", requested)
    assert exc_info.value.partition_key == "cfg_guard"
    assert exc_info.value.existing_config == _plaintext_partition_config()
    assert exc_info.value.requested_config == requested

    # The partition itself is untouched and still opens.
    assert await store.open_partition("cfg_guard") is not None


@pytest.mark.asyncio
@pytest.mark.parametrize("create", ["create_partition", "open_or_create_partition"])
async def test_unloadable_codec_config_commits_no_registry_row(
    store: SQLAlchemyEventMemoryStore,
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
            await SQLAlchemyEventMemoryStore._get_partition_row(session, "codec_p")
        ) is None


@pytest.mark.asyncio
async def test_static_pool_engine_is_rejected(tmp_path) -> None:
    """StaticPool shares one connection; the params must refuse it loudly."""
    engine = create_async_engine(
        f"sqlite+aiosqlite:///{tmp_path / 'static_pool.db'}", poolclass=StaticPool
    )
    try:
        with pytest.raises(ValidationError, match="StaticPool"):
            SQLAlchemyEventMemoryStoreParams(engine=engine)
    finally:
        await engine.dispose()


@pytest.mark.asyncio
async def test_old_sqlite_runtime_is_rejected(
    sqlalchemy_sqlite_engine: AsyncEngine,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Partition deletion depends on RETURNING; refuse an older SQLite loudly.

    The store's own minimum is raised past the runtime rather than the
    stdlib's version tuple being patched process-wide, which SQLAlchemy's
    dialect also reads.
    """
    monkeypatch.setattr(sqlalchemy_event_memory_store, "_MIN_SQLITE_VERSION", (99, 0))
    with pytest.raises(ValidationError, match="RETURNING"):
        SQLAlchemyEventMemoryStoreParams(engine=sqlalchemy_sqlite_engine)


@pytest.mark.asyncio
async def test_unsupported_dialect_engine_is_rejected(
    sqlalchemy_sqlite_engine: AsyncEngine,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The store supports PostgreSQL and SQLite; refuse any other dialect loudly."""
    monkeypatch.setattr(sqlalchemy_sqlite_engine.dialect, "name", "mssql")
    with pytest.raises(ValidationError, match="mssql"):
        SQLAlchemyEventMemoryStoreParams(engine=sqlalchemy_sqlite_engine)


@pytest.mark.integration
@pytest.mark.asyncio
async def test_windowed_read_raises_when_partition_dies_between_statements(
    pg_store: SQLAlchemyEventMemoryStore,
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
    with pytest.raises(EventMemoryStorePartitionHandleStaleError):
        await _windows(partition, [segs[1]], before=1, after=1)


@pytest.mark.asyncio
async def test_partition_key_with_trailing_newline_is_rejected(
    store: SQLAlchemyEventMemoryStore,
) -> None:
    """`$` matches before a trailing newline; the validator must not."""
    with pytest.raises(ValueError, match="invalid characters"):
        await store.create_partition("bad_key\n", _plaintext_partition_config())


@pytest.mark.integration
@pytest.mark.asyncio
async def test_delete_partition_touches_only_registry_and_queue(
    pg_store: SQLAlchemyEventMemoryStore,
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

    data_tables = BaseEventMemoryStore.metadata.tables.keys() - {
        PartitionRow.__tablename__,
        PurgeQueueRow.__tablename__,
    }
    touching_data = [
        s for s in recorded_statements if any(table in s for table in data_tables)
    ]
    assert not touching_data, touching_data
    assert any("event_memory_store_gc" in s for s in recorded_statements)


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
# The ordered event and segment row locks in forget_events impose an
# acquisition order no engine guarantees on its own;
# test_overlapping_forgets_do_not_deadlock is the canary for a divergent
# order, not a proof that the locks are needed.
# ===================================================================


@pytest.mark.integration
@pytest.mark.asyncio
async def test_write_pin_blocks_partition_delete(
    pg_store: SQLAlchemyEventMemoryStore,
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

    writer = asyncio.create_task(partition.add_events(*_events(_links(_seg()))))
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


@pytest.mark.asyncio
async def test_overlapping_forgets_do_not_deadlock(
    store: SQLAlchemyEventMemoryStore,
) -> None:
    """Overlapping concurrent `forget_events` stay deadlock-free.

    Regression canary rather than a lock-necessity proof: the ordered
    pre-locks impose an acquisition order that no database documents for
    the deletes and their cascades, and this test catches the AB/BA cycle
    wherever an engine or plan produces divergent orders. Each unlinked
    derivative is named by exactly one of the racing forgets' tombstones.
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
            partition.forget_events(forward),
            partition.forget_events(backward),
        )
        assert (
            await partition.forget_events(rng.sample(event_uuids, len(event_uuids)))
            is None
        )
        assert await _row_counts(partition, partition._incarnation) == (0, 0)
        assert sorted(
            derivative_uuid
            for tombstone in tombstones
            if tombstone is not None
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
    store: SQLAlchemyEventMemoryStore,
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
                EventMemoryStorePartitionAlreadyExistsError,
                EventMemoryStorePartitionConfigMismatchError,
            ):
                pass

    await asyncio.wait_for(
        asyncio.gather(*(worker(seed) for seed in range(8))),
        120,
    )


@pytest.mark.asyncio
async def test_concurrent_partition_deletes_are_clean(
    store: SQLAlchemyEventMemoryStore,
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
    store: SQLAlchemyEventMemoryStore,
    monkeypatch: pytest.MonkeyPatch,
    seed: int,
) -> None:
    """Random sequences of every data operation agree with an in-memory model.

    Adds, forgets, partition deletes and re-creates, and bounded purge
    calls over a few keys. Each forget's tombstone names exactly the
    derivatives it drops from the model, after every step each live key
    reads back exactly the model's segments and links, and once the purge
    drains, the tables hold exactly the live rows.
    """
    rng = random.Random(seed)
    monkeypatch.setattr(store, "_purge_max_segments", 3)
    config = _plaintext_partition_config()
    keys = ["model_a", "model_b", "model_c"]
    handles: dict[str, SQLAlchemyEventMemoryStorePartition] = {}
    # Each live key's events, with each event's segments and their links.
    model: dict[str, dict[UUID, dict[UUID, set[UUID]]]] = {}
    tombstones: dict[str, int] = {}

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
        else:
            if key not in handles:
                handles[key] = await store.open_or_create_partition(key, config)
                model[key] = {}
                tombstones[key] = 0
            if operation == "add":
                segments = _random_model_events(rng, ts_offset_seconds=step * 10)
                await _add_segments(handles[key], segments)
                for segment, links in segments.items():
                    model[key].setdefault(segment.event_uuid, {})[segment.uuid] = links
            else:
                victims = rng.sample(
                    sorted(model[key]), min(len(model[key]), rng.randint(0, 3))
                )
                tombstone = await handles[key].forget_events([*victims, uuid4()])
                unlinked = [
                    link
                    for victim in victims
                    for links in model[key].pop(victim).values()
                    for link in links
                ]
                if tombstone is None:
                    assert unlinked == []
                else:
                    assert sorted(tombstone.derivative_uuids) == sorted(unlinked)
                    tombstones[key] += 1

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
                DerivativeLinkRow,
                PendingDerivativesRow,
                TombstoneRow,
            )
        ]
    assert row_counts == [
        len(events),
        sum(map(len, events)),
        sum(len(links) for event in events for links in event.values()),
        0,
        sum(tombstones.values()),
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
    partition: SQLAlchemyEventMemoryStorePartition,
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
    assert set(await partition.get_segments(segment_links)) == set(segment_links)


@pytest.mark.asyncio
@pytest.mark.parametrize("seed", [1, 2])
async def test_mixed_operation_churn_completes_and_strands_nothing(
    store: SQLAlchemyEventMemoryStore,
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
    left.
    """
    rng = random.Random(seed)
    monkeypatch.setattr(store, "_purge_max_segments", 3)
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
    finished = asyncio.Event()

    async def purger() -> None:
        while not finished.is_set():
            if not await store.purge_deleted_partitions():
                await asyncio.sleep(0.01)

    async def work() -> None:
        await asyncio.gather(
            *(
                _write_and_forget_own_events(
                    partition, written[partition._incarnation], rng
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
            DerivativeLinkRow,
            PendingDerivativesRow,
            TombstoneRow,
        )
    ]
    async with store._create_session() as session:
        queue_depth = (
            await session.execute(select(func.count()).select_from(PurgeQueueRow))
        ).scalar_one()
    assert (stranded, queue_depth) == ([0, 0, 0, 0, 0], 0)
    for incarnation, events in written.items():
        assert await _row_counts(store, incarnation) == (
            len(events),
            sum(events.values()),
        )


async def _write_and_forget_own_events(
    partition: SQLAlchemyEventMemoryStorePartition,
    mine: dict[UUID, int],
    rng: random.Random,
) -> None:
    """Add events, then forget some of this writer's own; `mine` tracks what is left."""
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
        await partition.forget_events(victims)
        for victim in victims:
            del mine[victim]


async def _churn_shared_keys(
    store: SQLAlchemyEventMemoryStore,
    config: EventMemoryStorePartitionConfig,
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
            EventMemoryStorePartitionAlreadyExistsError,
            EventMemoryStorePartitionHandleStaleError,
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
    sqlite_store: SQLAlchemyEventMemoryStore,
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

    data_tables = BaseEventMemoryStore.metadata.tables.keys() - {
        PartitionRow.__tablename__,
        PurgeQueueRow.__tablename__,
    }
    data_statements = [
        statement
        for statement in sqlite_recorded_statements
        if any(table in statement for table in data_tables)
    ]
    assert not data_statements
    assert any("event_memory_store_gc" in s for s in sqlite_recorded_statements)


@pytest.mark.asyncio
async def test_sqlite_mint_detects_collision_with_concurrent_deletion(
    sqlite_store: SQLAlchemyEventMemoryStore,
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

    monkeypatch.setattr(sqlalchemy_event_memory_store, "uuid4", colliding_uuid4)

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


# ===================================================================
# get_segment_neighborhoods -- the seed is an address
# ===================================================================


@pytest.mark.asyncio
async def test_segment_neighbors_are_two_lists_without_the_seed(
    partition: SQLAlchemyEventMemoryStorePartition,
) -> None:
    ep = uuid4()
    segs = [_seg(event_uuid=ep, offset=i, ts_offset_seconds=i) for i in range(5)]
    await _add_segments(partition, _links(*segs))

    result = await partition.get_segment_neighborhoods(
        [segs[2].uuid], before=2, after=2
    )

    neighborhood = result[segs[2].uuid]
    # Other chunks of the seed's own event are ordinary neighbors; only the
    # seed itself is withheld, and its place is between the two lists.
    assert [s.uuid for s in neighborhood.before] == [segs[0].uuid, segs[1].uuid]
    assert [s.uuid for s in neighborhood.after] == [segs[3].uuid, segs[4].uuid]
    assert segs[2].uuid not in {
        s.uuid for s in [*neighborhood.before, *neighborhood.after]
    }


@pytest.mark.asyncio
async def test_the_lookup_drops_a_failing_segment_and_the_walk_keeps_its_neighbors(
    partition: SQLAlchemyEventMemoryStorePartition,
) -> None:
    """The filter decides what a read returns: the lookup omits a segment that fails, and the walk from it returns the neighbors that pass."""
    ep = uuid4()
    s0 = _seg(event_uuid=ep, offset=0, ts_offset_seconds=0, properties={"tag": "a"})
    s1 = _seg(event_uuid=ep, offset=1, ts_offset_seconds=1, properties={"tag": "b"})
    s2 = _seg(event_uuid=ep, offset=2, ts_offset_seconds=2, properties={"tag": "a"})
    await _add_segments(partition, _links(s0, s1, s2))
    only_a = parse_filter("m.tag = 'a'")

    neighborhoods_by_seed = await partition.get_segment_neighborhoods(
        [s1.uuid], before=5, after=5, property_filter=only_a
    )
    found = await partition.get_segments([s1.uuid], property_filter=only_a)

    assert [s.uuid for s in neighborhoods_by_seed[s1.uuid].before] == [s0.uuid]
    assert [s.uuid for s in neighborhoods_by_seed[s1.uuid].after] == [s2.uuid]
    assert found == {}


@pytest.mark.asyncio
async def test_segment_neighbors_of_a_lone_seed_are_empty_lists(
    partition: SQLAlchemyEventMemoryStorePartition,
) -> None:
    seed = _seg()
    await _add_segments(partition, _links(seed))

    result = await partition.get_segment_neighborhoods([seed.uuid], before=5, after=5)

    assert result == {seed.uuid: Neighborhood(before=[], after=[])}


@pytest.mark.asyncio
async def test_segment_neighbors_with_zero_counts_still_locate_the_seed(
    partition: SQLAlchemyEventMemoryStorePartition,
) -> None:
    seed = _seg()
    await _add_segments(partition, _links(seed, _seg(ts_offset_seconds=1)))

    assert await partition.get_segment_neighborhoods([seed.uuid]) == {
        seed.uuid: Neighborhood(before=[], after=[])
    }


# ===================================================================
# Immutability
# ===================================================================


@pytest.mark.asyncio
async def test_adding_a_stored_event_again_keeps_its_segments(
    partition: SQLAlchemyEventMemoryStorePartition,
) -> None:
    """Segments are immutable: a second add of the same event is rejected, not replaced."""
    s0 = _seg(text="first")
    await _add_segments(partition, _links(s0))
    again = s0.model_copy(update={"block": TextBlock(text="second")})
    with pytest.raises(EventMemoryStoreEventAlreadyStoredError):
        await _add_segments(partition, _links(again))
    windows = await partition.get_segments([s0.uuid])
    assert windows[s0.uuid].block == TextBlock(text="first")


# ===================================================================
# since / until, source_ids
# ===================================================================


@pytest.mark.asyncio
async def test_a_naive_bound_is_rejected(
    partition: SQLAlchemyEventMemoryStorePartition,
) -> None:
    """A naive bound names no instant, so neither read accepts one."""
    seg = _seg()
    await _add_segments(partition, _links(seg))
    naive = BASE_TIME.replace(tzinfo=None)

    with pytest.raises(ValueError, match="since must be timezone-aware"):
        await partition.get_segments([seg.uuid], since=naive)
    with pytest.raises(ValueError, match="until must be timezone-aware"):
        await partition.get_segment_neighborhoods([seg.uuid], after=1, until=naive)


@pytest.mark.asyncio
async def test_a_negative_count_is_rejected(
    partition: SQLAlchemyEventMemoryStorePartition,
) -> None:
    seg = _seg()
    await _add_segments(partition, _links(seg))

    with pytest.raises(ValueError, match="before must be nonnegative"):
        await partition.get_segment_neighborhoods([seg.uuid], before=-1)
    with pytest.raises(ValueError, match="after must be nonnegative"):
        await partition.get_segment_neighborhoods([seg.uuid], after=-1)


@pytest.mark.asyncio
async def test_since_and_until_meet_without_overlap(
    partition: SQLAlchemyEventMemoryStorePartition,
) -> None:
    """`since` is inclusive and `until` exclusive on a boundary timestamp."""
    s0 = _seg(ts_offset_seconds=0)
    s1 = _seg(ts_offset_seconds=60)
    s2 = _seg(ts_offset_seconds=120)
    await _add_segments(partition, _links(s0, s1, s2))
    boundary = BASE_TIME + timedelta(seconds=60)

    from_boundary = await _windows(partition, [s1], before=5, after=5, since=boundary)
    to_boundary = await partition.get_segments([s1.uuid], until=boundary)
    around = await partition.get_segment_neighborhoods(
        [s1.uuid], before=5, after=5, since=BASE_TIME, until=boundary
    )

    assert [s.uuid for s in from_boundary[s1.uuid]] == [s1.uuid, s2.uuid]
    # The seed sits on the exclusive bound, so the lookup leaves it out...
    assert to_boundary == {}
    # ...while the walk from it still runs, and the bound admits only s0.
    assert [s.uuid for s in around[s1.uuid].before] == [s0.uuid]
    assert around[s1.uuid].after == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "bound",
    [
        "2024-01-01T00:00:30+00:00",
        "2024-01-01T08:00:30+08:00",  # the same instant, named in another zone
        "2023-12-31T16:00:30-08:00",  # and another
    ],
)
async def test_time_bounds_compare_instants_not_wall_clocks(
    partition: SQLAlchemyEventMemoryStorePartition,
    bound: str,
) -> None:
    """A bound means an instant, whatever zone it is written in.

    Timestamps are stored as the UTC instant; a bound compared with its own
    offset on SQLite would be compared on its wall-clock digits.
    """
    early = _seg(ts_offset_seconds=0)
    late = _seg(ts_offset_seconds=60)
    await _add_segments(partition, _links(early, late))
    instant = datetime.fromisoformat(bound)

    before_bound = await _windows(partition, [early], after=5, until=instant)
    from_bound = await partition.get_segment_neighborhoods(
        [early.uuid], after=5, since=instant
    )

    assert [s.uuid for s in before_bound[early.uuid]] == [early.uuid]
    assert [s.uuid for s in from_bound[early.uuid].after] == [late.uuid]


@pytest.mark.asyncio
async def test_source_ids_select_rows(
    partition: SQLAlchemyEventMemoryStorePartition,
) -> None:
    s0 = _seg(source_id="alice", ts_offset_seconds=0)
    s1 = _seg(source_id="bob", ts_offset_seconds=1)
    s2 = _seg(source_id="alice", ts_offset_seconds=2)
    s3 = _seg(source_id=None, ts_offset_seconds=3)
    await _add_segments(partition, _links(s0, s1, s2, s3))

    alice = await _windows(partition, [s0], after=5, source_ids=["alice"])
    bob_around_alice = await partition.get_segment_neighborhoods(
        [s0.uuid], after=5, source_ids=["bob"]
    )
    nobody = await partition.get_segments([s0.uuid], source_ids=[])

    assert [s.uuid for s in alice[s0.uuid]] == [s0.uuid, s2.uuid]
    assert [s.uuid for s in bob_around_alice[s0.uuid].after] == [s1.uuid]
    assert nobody == {}
    assert alice[s0.uuid][0].source_id == "alice"


@pytest.mark.asyncio
async def test_row_projections_are_derived_from_the_segment(
    partition: SQLAlchemyEventMemoryStorePartition,
) -> None:
    seg = _seg(source_id="alice")
    await _add_segments(partition, _links(seg))

    async with partition._create_session() as session:
        row = (
            await session.execute(select(SegmentRow).where(SegmentRow.uuid == seg.uuid))
        ).scalar_one()
    assert row.source_id == "alice"

    returned = (await partition.get_segments([seg.uuid]))[seg.uuid]
    assert returned.source_id == "alice"
