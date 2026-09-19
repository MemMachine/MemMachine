"""Tests for SQLAlchemySegmentStore — SQLite (unit) and PostgreSQL (integration)."""

import asyncio
import contextlib
import json
import logging
import random
from collections.abc import AsyncIterator, Iterator
from datetime import UTC, datetime, timedelta, timezone
from uuid import UUID, uuid4

import pytest
import pytest_asyncio
from pydantic import ValidationError
from sqlalchemy import delete, event, func, insert, select, text, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine
from sqlalchemy.pool import StaticPool

from memmachine_server.common.filter.filter_parser import Comparison, parse_filter
from memmachine_server.common.payload_codec.payload_codec_config import (
    PlaintextPayloadCodecConfig,
)
from memmachine_server.episodic_memory.event_memory.data_types import (
    NullContext,
    ProducerContext,
    Segment,
    TextBlock,
)
from memmachine_server.episodic_memory.event_memory.segment_store import (
    SegmentStoreAttemptsExhaustedError,
    SegmentStorePartitionAlreadyExistsError,
    SegmentStorePartitionConfig,
    SegmentStorePartitionConfigMismatchError,
    SegmentStorePartitionHandleStaleError,
    sqlalchemy_segment_store,
)
from memmachine_server.episodic_memory.event_memory.segment_store.sqlalchemy_segment_store import (
    BaseSegmentStore,
    DerivativeLinkRow,
    PartitionRow,
    PurgeQueueRow,
    SegmentRow,
    SQLAlchemySegmentStore,
    SQLAlchemySegmentStoreParams,
    SQLAlchemySegmentStorePartition,
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


def _plaintext_partition_config() -> SegmentStorePartitionConfig:
    """Return the default plaintext partition config."""
    return SegmentStorePartitionConfig()


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
# add_segments
# ===================================================================


@pytest.mark.asyncio
async def test_add_segments_and_get_contexts(
    partition: SQLAlchemySegmentStorePartition,
) -> None:
    seg = _seg(text="a")
    await partition.add_segments(_links(seg))

    result = await partition.get_segment_contexts([seg.uuid])
    assert seg.uuid in result
    assert len(result[seg.uuid]) == 1
    assert result[seg.uuid][0].uuid == seg.uuid


@pytest.mark.asyncio
async def test_add_segments_with_properties(
    partition: SQLAlchemySegmentStorePartition,
) -> None:
    seg = _seg(properties={"color": "red", "score": 42})
    await partition.add_segments(_links(seg))

    result = await partition.get_segment_contexts([seg.uuid])
    returned = result[seg.uuid][0]
    assert returned.properties == {"color": "red", "score": 42}


@pytest.mark.asyncio
async def test_add_segments_with_producer_context(
    partition: SQLAlchemySegmentStorePartition,
) -> None:
    ctx = ProducerContext(producer="User")
    seg = _seg(context=ctx)
    await partition.add_segments(_links(seg))

    result = await partition.get_segment_contexts([seg.uuid])
    assert result[seg.uuid][0].context == ctx

    async with partition._create_session() as session:
        row = (
            await session.execute(select(SegmentRow).where(SegmentRow.uuid == seg.uuid))
        ).scalar_one()
    assert json.loads(row.context) == {"context_type": "producer", "producer": "User"}
    assert json.loads(row.block) == {"block_type": "text", "text": "hello"}


@pytest.mark.asyncio
async def test_add_segments_with_no_context(
    partition: SQLAlchemySegmentStorePartition,
) -> None:
    seg = _seg()
    await partition.add_segments(_links(seg))

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
    await partition.add_segments(_links(seg))

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
    await partition.add_segments(_links(early, late))

    result = await partition.get_segment_contexts(
        [early.uuid],
        max_forward_segments=5,
        property_filter=parse_filter(f"timestamp <= date('{bound}')"),
    )
    assert [s.uuid for s in result[early.uuid]] == [early.uuid]


@pytest.mark.asyncio
async def test_add_segments_empty(
    partition: SQLAlchemySegmentStorePartition,
) -> None:
    await partition.add_segments({})


@pytest.mark.asyncio
async def test_add_multiple_derivatives_per_segment(
    partition: SQLAlchemySegmentStorePartition,
) -> None:
    seg = _seg()
    d1, d2 = uuid4(), uuid4()
    await partition.add_segments({seg: [d1, d2]})

    result = await partition.get_derivative_uuids_by_segment_uuids([seg.uuid])
    assert len(result[seg.uuid]) == 2
    assert {d1, d2} == set(result[seg.uuid])


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
    await partition.add_segments(_links(seg))

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
    await partition.add_segments(_links(*segs))

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
    await partition.add_segments(_links(*segs))

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
    await partition.add_segments(_links(*segs))

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
    await partition.add_segments(_links(*segs))

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
    await partition.add_segments(_links(*segs))

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
    await partition.add_segments(_links(s0, s1, s2))

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
    await partition.add_segments(_links(s0, s1, s2, s3))

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
async def test_contexts_filter_by_context_producer(
    partition: SQLAlchemySegmentStorePartition,
) -> None:
    """`context.producer` is not a stored property; the filter just matches nothing.

    Bare names that aren't `timestamp` are looked up as `_<name>` in the JSON
    properties (matching EventMemory's `_to_vector_record_property` convention
    for system fields). `context.producer` becomes `_context.producer`, which
    isn't a stored key on any segment, so the filter returns no contexts.
    """
    ep = uuid4()
    s0 = _seg(
        event_uuid=ep,
        offset=0,
        ts_offset_seconds=0,
        context=ProducerContext(producer="Alice"),
    )
    s1 = _seg(
        event_uuid=ep,
        offset=1,
        ts_offset_seconds=1,
        context=ProducerContext(producer="Bob"),
    )
    s2 = _seg(
        event_uuid=ep,
        offset=2,
        ts_offset_seconds=2,
        context=ProducerContext(producer="Alice"),
    )
    await partition.add_segments(_links(s0, s1, s2))

    filt = Comparison(field="context.producer", op="=", value="Alice")
    contexts = await partition.get_segment_contexts(
        [s0.uuid],
        max_backward_segments=5,
        max_forward_segments=5,
        property_filter=filt,
    )
    assert contexts == {}


@pytest.mark.asyncio
async def test_contexts_filter_by_context_type(
    partition: SQLAlchemySegmentStorePartition,
) -> None:
    """`context.context_type` is not a stored property; filter matches nothing."""
    ep = uuid4()
    s0 = _seg(
        event_uuid=ep,
        offset=0,
        ts_offset_seconds=0,
        context=ProducerContext(producer="Alice"),
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
        context=ProducerContext(producer="Bob"),
    )
    await partition.add_segments(_links(s0, s1, s2))

    filt = Comparison(field="context.context_type", op="=", value="producer")
    contexts = await partition.get_segment_contexts(
        [s0.uuid],
        max_backward_segments=5,
        max_forward_segments=5,
        property_filter=filt,
    )
    assert contexts == {}


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
    await other_partition.add_segments(_links(s_other))

    partition = await store.open_or_create_partition(
        PARTITION_KEY,
        _plaintext_partition_config(),
    )
    await partition.add_segments(_links(s_seed, s_after))

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
    await partition.add_segments(_links(*segs))

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
    await partition.add_segments(_links(s0, s1, s2))

    result = await partition.get_segment_contexts(
        [s1.uuid], max_backward_segments=1, max_forward_segments=1
    )
    ctx = result[s1.uuid]
    assert len(ctx) == 3
    assert ctx[0].context == ctx_user
    assert ctx[1].context == ctx_assistant
    assert ctx[2].context == ctx_user


# ===================================================================
# get_segment_uuids_by_event_uuids
# ===================================================================


@pytest.mark.asyncio
async def test_get_segment_uuids_by_event_uuids(
    partition: SQLAlchemySegmentStorePartition,
) -> None:
    ep = uuid4()
    segs = [_seg(event_uuid=ep, offset=i, ts_offset_seconds=i) for i in range(3)]
    await partition.add_segments(_links(*segs))

    result = await partition.get_segment_uuids_by_event_uuids([ep])
    assert ep in result
    assert set(result[ep]) == {s.uuid for s in segs}


@pytest.mark.asyncio
async def test_get_segment_uuids_by_event_uuids_empty(
    partition: SQLAlchemySegmentStorePartition,
) -> None:
    result = await partition.get_segment_uuids_by_event_uuids([])
    assert result == {}


@pytest.mark.asyncio
async def test_get_segment_uuids_by_event_uuids_unknown(
    partition: SQLAlchemySegmentStorePartition,
) -> None:
    result = await partition.get_segment_uuids_by_event_uuids([uuid4()])
    assert result == {}


# ===================================================================
# get_derivative_uuids_by_segment_uuids
# ===================================================================


@pytest.mark.asyncio
async def test_get_derivative_uuids_by_segment_uuids(
    partition: SQLAlchemySegmentStorePartition,
) -> None:
    seg = _seg()
    d1, d2 = uuid4(), uuid4()
    await partition.add_segments({seg: [d1, d2]})

    result = await partition.get_derivative_uuids_by_segment_uuids([seg.uuid])
    assert seg.uuid in result
    assert set(result[seg.uuid]) == {d1, d2}


@pytest.mark.asyncio
async def test_get_derivative_uuids_by_segment_uuids_empty(
    partition: SQLAlchemySegmentStorePartition,
) -> None:
    result = await partition.get_derivative_uuids_by_segment_uuids([])
    assert result == {}


@pytest.mark.asyncio
async def test_get_derivative_uuids_by_segment_uuids_unknown(
    partition: SQLAlchemySegmentStorePartition,
) -> None:
    result = await partition.get_derivative_uuids_by_segment_uuids([uuid4()])
    assert result == {}


# ===================================================================
# delete_segments
# ===================================================================


@pytest.mark.asyncio
async def test_delete_segments(
    partition: SQLAlchemySegmentStorePartition,
) -> None:
    ep = uuid4()
    seg = _seg(event_uuid=ep)
    await partition.add_segments(_links(seg))

    await partition.delete_segments([seg.uuid])

    # Segment gone.
    result = await partition.get_segment_contexts([seg.uuid])
    assert result == {}

    # Derivative cascaded.
    deriv_result = await partition.get_derivative_uuids_by_segment_uuids([seg.uuid])
    assert deriv_result == {}


@pytest.mark.asyncio
async def test_delete_segments_noop_unknown(
    partition: SQLAlchemySegmentStorePartition,
) -> None:
    await partition.delete_segments([uuid4()])


@pytest.mark.asyncio
async def test_delete_segments_empty(
    partition: SQLAlchemySegmentStorePartition,
) -> None:
    await partition.delete_segments([])


@pytest.mark.asyncio
async def test_delete_segments_partial(
    partition: SQLAlchemySegmentStorePartition,
) -> None:
    """Deleting one segment leaves others intact."""
    ep = uuid4()
    s1 = _seg(event_uuid=ep, offset=0, ts_offset_seconds=0)
    s2 = _seg(event_uuid=ep, offset=1, ts_offset_seconds=1)
    await partition.add_segments(_links(s1, s2))

    await partition.delete_segments([s1.uuid])

    # s1 gone, s2 still there.
    result = await partition.get_segment_contexts([s1.uuid, s2.uuid])
    assert s1.uuid not in result
    assert s2.uuid in result


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
        await part.add_segments(_links(*segs))

    await asyncio.gather(*(add_batch(i) for i in range(10)))

    # Verify all segments were added.
    part = await _get_partition(engine)
    async with part._create_session() as session:
        count = (
            await session.execute(select(func.count()).select_from(SegmentRow))
        ).scalar()
    assert count == 50


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
    await partition.add_segments(_links(*segs))

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
            await part.add_segments(_links(new_seg))
            await asyncio.sleep(0.01)

    await asyncio.gather(reader(), reader(), writer())

    assert len(read_results) > 0


@pytest.mark.asyncio
async def test_concurrent_context_reads_during_deletes(
    store: SQLAlchemySegmentStore,
) -> None:
    """get_segment_contexts should not crash if segments are deleted concurrently."""
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
    await partition.add_segments(_links(*all_segs))

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

    async def segment_deleter() -> None:
        part = await _get_partition(engine)
        for ep_idx in range(1, 3):
            ep_segs = all_segs[ep_idx * 4 : (ep_idx + 1) * 4]
            await part.delete_segments([s.uuid for s in ep_segs])
            await asyncio.sleep(0.02)

    await asyncio.gather(context_reader(), segment_deleter())

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
    await partition.add_segments(_links(seg))

    await store.delete_partition("to_delete")

    # Partition no longer exists.
    assert await store.open_partition("to_delete") is None


@pytest.mark.integration
@pytest.mark.asyncio
async def test_delete_partition_keeps_foreign_key_enforced(
    pg_store: SQLAlchemySegmentStore,
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
                    "INSERT INTO segment_store_dv_ln"
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
    store: SQLAlchemySegmentStore,
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
    await keeper.add_segments({segment: [derivative_uuid]})

    await store.delete_partition("doomed")

    # The foreign key from derivative links to segments must still cascade
    # for the partitions that were not deleted.
    await keeper.delete_segments([segment.uuid])
    assert await keeper.get_derivative_uuids_by_segment_uuids([segment.uuid]) == {}


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
    await partition.add_segments({seg: [d1]})

    await store.delete_partition("cascade_test")

    # Re-create the partition and verify data is gone.
    new_partition = await store.open_or_create_partition(
        "cascade_test",
        _plaintext_partition_config(),
    )
    result = await new_partition.get_segment_contexts([seg.uuid])
    assert result == {}
    deriv_result = await new_partition.get_derivative_uuids_by_segment_uuids([seg.uuid])
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
    await partition.add_segments(_links(*all_segs))

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
    await partition.add_segments(_links(*all_segs))

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
    await partition.add_segments(_links(seg))

    await store.delete_partition("fenced")

    # Empty input may skip the handle check (the ABC-permitted shortcut).
    await partition.add_segments({})

    with pytest.raises(SegmentStorePartitionHandleStaleError):
        await partition.add_segments(_links(_seg()))
    with pytest.raises(SegmentStorePartitionHandleStaleError):
        await partition.get_segment_contexts([seg.uuid])
    with pytest.raises(SegmentStorePartitionHandleStaleError):
        await partition.get_segment_uuids_by_event_uuids([seg.event_uuid])
    with pytest.raises(SegmentStorePartitionHandleStaleError):
        await partition.get_derivative_uuids_by_segment_uuids([seg.uuid])


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
    await partition.add_segments(_links(seg))
    recorded_statements.clear()

    await partition.get_segment_contexts([seg.uuid])
    await partition.get_segment_uuids_by_event_uuids([seg.event_uuid])
    await partition.get_derivative_uuids_by_segment_uuids([seg.uuid])
    assert len(recorded_statements) == 3
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
        await old_handle.add_segments(_links(_seg()))
    await new_handle.add_segments(_links(_seg()))  # the live handle works


@pytest.mark.asyncio
async def test_recreated_partition_is_isolated_from_old_rows(
    store: SQLAlchemySegmentStore,
) -> None:
    """Old-incarnation rows are invisible to the successor before purging."""
    partition = await store.open_or_create_partition(
        "isolated", _plaintext_partition_config()
    )
    seg = _seg()
    await partition.add_segments(_links(seg))
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
    monkeypatch.setattr(store, "_purge_max_segments", 2)

    live = await store.open_or_create_partition("live_p", _plaintext_partition_config())
    live_seg = _seg()
    await live.add_segments(_links(live_seg))

    doomed = await store.open_or_create_partition(
        "doomed_p", _plaintext_partition_config()
    )
    doomed_incarnation = doomed._incarnation
    await doomed.add_segments(_links(_seg(), _seg(), _seg()))
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
    monkeypatch.setattr(store, "_purge_max_segments", 2)
    incarnations = []
    for index in range(3):
        partition = await store.open_or_create_partition(
            f"gc_race_{index}", _plaintext_partition_config()
        )
        incarnations.append(partition._incarnation)
        await partition.add_segments(_links(_seg(), _seg(), _seg()))
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
        await partition.add_segments(_links(_seg()))
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

    writer = asyncio.create_task(partition.add_segments(_links(_seg())))
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
                    partition_key="remote_del",
                    enqueued_at=datetime.now(UTC),
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
    await doomed.add_segments(_links(_seg()))
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
            purge_max_segments=2,
        )
    )
    await store.startup()
    try:
        partition = await store.open_or_create_partition(
            "bound_p", _plaintext_partition_config()
        )
        await partition.add_segments(_links(_seg(), _seg(), _seg()))
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
    await victim.add_segments(_links(_seg()))

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
                partition_key="mint_victim",
                enqueued_at=datetime.now(UTC),
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
        await partition.add_segments(_links(_seg(), _seg(), _seg()))
        await pg_store.delete_partition(f"inc_claim_{index}")
    monkeypatch.setattr(pg_store, "_purge_max_segments", 2)
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
    """On SQLite the write fence opens the write transaction before checking.

    The driver defers BEGIN to the first data-modifying statement, so a
    SELECT-only fence would run outside the write transaction: a delete and
    a full purge could complete between the check and the insert, and the
    write would then commit rows no queue entry tracks. The fence's no-op
    registry UPDATE opens the write transaction first, so the racing
    deletion must wait for the writer and the rows stay reclaimable.
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

    writer = asyncio.create_task(partition.add_segments(_links(_seg())))
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
        "longer tracks: the SQLite fence did not open the write "
        "transaction before checking"
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
    monkeypatch.setattr(store, "_purge_max_segments", 2)
    for index in range(3):
        await store.create_partition(f"noop_{index}", _plaintext_partition_config())
        await store.delete_partition(f"noop_{index}")
    rowful = await store.open_or_create_partition(
        "rowful", _plaintext_partition_config()
    )
    await rowful.add_segments(_links(_seg(), _seg()))
    await store.delete_partition("rowful")

    # One call: three empty entries retired for free, then both rows.
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
        await partition.add_segments(_links(_seg()))
        await store.delete_partition(key)

    # Stamps come from the database clock, whose resolution need not
    # separate back-to-back deletions; set them so the LAST-inserted entry
    # is the oldest, so only the ordering key can produce the expectation.
    base = datetime(2026, 1, 1, tzinfo=UTC)
    async with partition._create_session() as session, session.begin():
        for age, key in enumerate(incarnations):
            await session.execute(
                update(PurgeQueueRow)
                .where(PurgeQueueRow.incarnation == incarnations[key])
                .values(enqueued_at=base - timedelta(seconds=age))
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


@pytest.mark.asyncio
async def test_purge_batches_links_that_escaped_integrity(
    sqlite_store: SQLAlchemySegmentStore,
    sqlalchemy_sqlite_engine: AsyncEngine,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Integrity-escaped link rows are reclaimed in bounded, warned batches.

    Staged on SQLite, where a connection without the store's
    foreign_keys pragma can insert orphan link rows; the batching under
    test is dialect-independent.
    """
    partition = await sqlite_store.open_or_create_partition(
        "leaky", _plaintext_partition_config()
    )
    incarnation = partition._incarnation
    await sqlite_store.delete_partition("leaky")

    # A second engine without the store's pragma listener enforces no
    # foreign keys, standing in for whatever once broke integrity.
    rogue_engine = create_async_engine(str(sqlalchemy_sqlite_engine.url))
    async with rogue_engine.begin() as connection:
        await connection.execute(
            insert(DerivativeLinkRow),
            [
                {
                    "incarnation": incarnation,
                    "uuid": uuid4(),
                    "segment_uuid": uuid4(),
                }
                for _ in range(5)
            ],
        )
    await rogue_engine.dispose()

    monkeypatch.setattr(sqlite_store, "_purge_max_segments", 2)
    with caplog.at_level(logging.WARNING):
        first = await sqlite_store.purge_deleted_partitions()
    # A full link batch leaves the entry claimable and warns.
    assert first is True
    assert "referential integrity" in caplog.text

    while await sqlite_store.purge_deleted_partitions():
        pass
    async with sqlite_store._create_session() as session:
        links = (
            await session.execute(
                select(func.count())
                .select_from(DerivativeLinkRow)
                .where(DerivativeLinkRow.incarnation == incarnation)
            )
        ).scalar_one()
        entries = (
            await session.execute(select(func.count()).select_from(PurgeQueueRow))
        ).scalar_one()
    assert links == 0
    assert entries == 0


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
    """Partition deletion depends on RETURNING; refuse an older SQLite loudly.

    The store's own minimum is raised past the runtime rather than the
    stdlib's version tuple being patched process-wide, which SQLAlchemy's
    dialect also reads.
    """
    monkeypatch.setattr(sqlalchemy_segment_store, "_MIN_SQLITE_VERSION", (99, 0))
    with pytest.raises(ValidationError, match="RETURNING"):
        SQLAlchemySegmentStoreParams(engine=sqlalchemy_sqlite_engine)


@pytest.mark.asyncio
async def test_windowed_read_raises_when_partition_dies_between_statements(
    store: SQLAlchemySegmentStore,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Seeds found, then deletion commits before the context statements.

    Each statement takes its own snapshot, so the context statements
    would return nothing; the read must raise rather than hand back
    seeds with silently empty context.
    """
    partition = await store.open_or_create_partition(
        "mid_read", _plaintext_partition_config()
    )
    segs = [_seg(ts_offset_seconds=i) for i in range(3)]
    await partition.add_segments(_links(*segs))

    context_method = (
        "_get_context_rows_loop"
        if partition._is_sqlite
        else "_get_context_rows_lateral"
    )
    original = getattr(partition, context_method)

    async def delete_then_read(*args, **kwargs):
        await store.delete_partition("mid_read")
        return await original(*args, **kwargs)

    monkeypatch.setattr(partition, context_method, delete_then_read)
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
    await partition.add_segments(
        _links(*(_seg(ts_offset_seconds=i) for i in range(20)))
    )
    recorded_statements.clear()

    await pg_store.delete_partition("big_delete")

    touching_data = [
        s
        for s in recorded_statements
        if "segment_store_sg" in s or "segment_store_dv_ln" in s
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
# - The writer's shared registry-row pin (_lock_partition_for_write)
#   blocks partition deletion while a write transaction is in flight:
#   test_write_pin_blocks_partition_delete fails with the pin ablated
#   in either generation and passes with it present.
# - Deletion's exclusive registry-row pin serializes racing lifecycle
#   operations: the churn and concurrent-delete tests fail with it
#   ablated. On the partitioned store those two tests failed even with
#   all locks intact (create/delete DDL lock cycles through the shared
#   parents) -- the deadlock-freedom they pin is a property of the
#   shared-table layout.
# - The ordered segment-row locks in delete_segments impose an
#   acquisition order no engine guarantees on its own; see
#   test_overlapping_segment_deletes_do_not_deadlock for why their
#   ablation is not observable on PostgreSQL today.
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

    writer = asyncio.create_task(partition.add_segments(_links(_seg())))
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
async def test_overlapping_segment_deletes_do_not_deadlock(
    store: SQLAlchemySegmentStore,
) -> None:
    """Overlapping concurrent `delete_segments` stay deadlock-free.

    Regression canary rather than a lock-necessity proof: ablating the
    ordered pre-lock does not make this fail on PostgreSQL, whose current
    executor happens to acquire row locks in consistent orders for
    identical DELETE shapes (scalar array probes are sorted; bitmap scans
    lock in TID order). That consistency is engine behavior, not a
    guarantee any database documents, so the pre-lock imposes the order
    deliberately; this test catches the AB/BA cycle wherever an engine or
    plan change ever produces divergent orders.
    """
    partition = await store.open_or_create_partition(
        "lk_row_order", _plaintext_partition_config()
    )

    async def round_trip(rng: random.Random) -> None:
        segments = [_seg(index=index) for index in range(24)]
        await partition.add_segments(_links(*segments))
        uuids = [segment.uuid for segment in segments]
        forward = uuids[:16]
        backward = [*reversed(uuids[8:])]
        await asyncio.gather(
            partition.delete_segments(forward),
            partition.delete_segments(backward),
        )
        await partition.delete_segments(rng.sample(uuids, len(uuids)))

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
    await partition.add_segments(
        _links(*(_seg(ts_offset_seconds=i) for i in range(20)))
    )
    sqlite_recorded_statements.clear()

    await sqlite_store.delete_partition("sqlite_big_delete")

    data_statements = [
        statement
        for statement in sqlite_recorded_statements
        if "segment_store_sg" in statement or "segment_store_dv_ln" in statement
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
    await victim.add_segments(_links(_seg()))

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
                partition_key="sq_mint_victim",
                enqueued_at=datetime.now(UTC),
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
