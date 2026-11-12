"""Tests for SQLAlchemySegmentStore — SQLite (unit) and PostgreSQL (integration)."""

import asyncio
import contextlib
import json
import operator
import random
from collections.abc import AsyncIterator, Generator
from datetime import UTC, datetime, timedelta, timezone
from uuid import UUID, uuid4

import pytest
import pytest_asyncio
from pydantic import ValidationError
from sqlalchemy import (
    event,
    func,
    select,
    text,
)
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine
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
    NullContext,
    ProducerContext,
    Segment,
    TextBlock,
)
from memmachine_server.episodic_memory.event_memory.segment_store import (
    SegmentStorePartitionConfig,
    SegmentStorePartitionConfigMismatchError,
    sqlalchemy_segment_store,
)
from memmachine_server.episodic_memory.event_memory.segment_store.sqlalchemy_segment_store import (
    BaseSegmentStore,
    DerivativeLinkRow,
    PartitionRow,
    SegmentRow,
    SQLAlchemySegmentStore,
    SQLAlchemySegmentStoreParams,
    SQLAlchemySegmentStorePartition,
)

PARTITION_UUID = UUID("00000000-0000-4000-8000-000000000001")
OTHER_PARTITION_UUID = UUID("00000000-0000-4000-8000-000000000002")
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


async def _open(
    store: SQLAlchemySegmentStore, partition_uuid: UUID = PARTITION_UUID
) -> SQLAlchemySegmentStorePartition:
    """Create a partition, with the store's plaintext config, and open a handle on it."""
    await store.create_partition(partition_uuid)
    partition = await store.open_partition(partition_uuid)
    assert partition is not None
    return partition


async def _row_counts(
    owner: SQLAlchemySegmentStore | SQLAlchemySegmentStorePartition,
    partition_uuid: UUID,
) -> tuple[int, int]:
    """The partition's segment and derivative-link row counts."""
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
                .select_from(DerivativeLinkRow)
                .where(DerivativeLinkRow.partition_uuid == partition_uuid)
            )
        ).scalar_one()
    return segments, links


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
    return await _open(store)


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
    await partition.add_segments(_links(seed, *backward.values(), *forward.values()))

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
    await partition.add_segments(_links(s0))

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

    other_partition = await _open(store, uuid4())
    await other_partition.add_segments(_links(s_other))

    partition = await _open(store)
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
    partition = await _open(store)
    other_partition = await _open(store, uuid4())
    event_uuids = iter(sorted(uuid4() for _ in range(45)))
    segments = [
        segment
        for _ in range(30)
        for segment in _random_event_segments(rng, next(event_uuids))
    ]
    await partition.add_segments(_links(*segments))
    await other_partition.add_segments(
        _links(
            *(
                segment
                for _ in range(15)
                for segment in _random_event_segments(rng, next(event_uuids))
            )
        )
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
# get_segment_uuids_by_derivative_uuids
# ===================================================================


@pytest.mark.asyncio
async def test_get_segment_uuids_by_derivative_uuids(
    partition: SQLAlchemySegmentStorePartition,
) -> None:
    seg = _seg()
    d1, d2 = uuid4(), uuid4()
    await partition.add_segments({seg: [d1, d2]})

    result = await partition.get_segment_uuids_by_derivative_uuids([d1, d2])
    assert result == {d1: seg.uuid, d2: seg.uuid}


@pytest.mark.asyncio
async def test_get_segment_uuids_by_derivative_uuids_agrees_with_the_forward_lookup(
    partition: SQLAlchemySegmentStorePartition,
) -> None:
    """Every derivative belongs to exactly one segment, so the two agree."""
    segments = [_seg(offset=i, ts_offset_seconds=i) for i in range(3)]
    links = {seg: [uuid4(), uuid4()] for seg in segments}
    await partition.add_segments(links)

    forward = await partition.get_derivative_uuids_by_segment_uuids(
        [seg.uuid for seg in segments]
    )
    all_derivative_uuids = [
        derivative_uuid
        for derivative_uuids in forward.values()
        for derivative_uuid in derivative_uuids
    ]
    backward = await partition.get_segment_uuids_by_derivative_uuids(
        all_derivative_uuids
    )

    assert backward == {
        derivative_uuid: segment_uuid
        for segment_uuid, derivative_uuids in forward.items()
        for derivative_uuid in derivative_uuids
    }


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
    await partition.add_segments({seg: [known]})

    unknown = uuid4()
    result = await partition.get_segment_uuids_by_derivative_uuids([known, unknown])
    assert result == {known: seg.uuid}


@pytest.mark.asyncio
async def test_get_segment_uuids_by_derivative_uuids_session_isolation(
    store: SQLAlchemySegmentStore,
) -> None:
    """A derivative another partition owns is invisible here."""
    other_partition = await _open(store, uuid4())
    other_seg = _seg()
    other_derivative = uuid4()
    await other_partition.add_segments({other_seg: [other_derivative]})

    partition = await _open(store)
    seg = _seg()
    derivative = uuid4()
    await partition.add_segments({seg: [derivative]})

    result = await partition.get_segment_uuids_by_derivative_uuids(
        [derivative, other_derivative]
    )
    assert result == {derivative: seg.uuid}


@pytest.mark.asyncio
async def test_get_segment_uuids_by_derivative_uuids_after_delete_segments(
    partition: SQLAlchemySegmentStorePartition,
) -> None:
    """Deleting a segment takes its derivatives' mapping with it."""
    kept, dropped = _seg(offset=0), _seg(offset=1)
    kept_derivative, dropped_derivative = uuid4(), uuid4()
    await partition.add_segments(
        {kept: [kept_derivative], dropped: [dropped_derivative]}
    )

    await partition.delete_segments([dropped.uuid])

    result = await partition.get_segment_uuids_by_derivative_uuids(
        [kept_derivative, dropped_derivative]
    )
    assert result == {kept_derivative: kept.uuid}


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


@pytest.mark.asyncio
async def test_add_segments_failing_partway_writes_nothing(
    partition: SQLAlchemySegmentStorePartition,
) -> None:
    """An add whose link rows are rejected keeps none of its segments either."""
    taken = uuid4()
    await partition.add_segments({_seg(): [taken]})
    rejected = [_seg(ts_offset_seconds=1), _seg(ts_offset_seconds=2)]

    with pytest.raises(IntegrityError):
        await partition.add_segments({rejected[0]: [uuid4()], rejected[1]: [taken]})

    assert await _row_counts(partition, partition._partition_uuid) == (1, 1)
    assert (
        await partition.get_segment_contexts([segment.uuid for segment in rejected])
        == {}
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("operation", ["add_segments", "delete_segments"])
async def test_segment_write_whose_commit_fails_changes_nothing(
    store: SQLAlchemySegmentStore,
    operation: str,
) -> None:
    """A segment write whose commit fails leaves the partition as it was."""
    partition = await _open(store, uuid4())
    kept = _seg()
    await partition.add_segments(_links(kept))
    write = (
        partition.add_segments(_links(_seg(ts_offset_seconds=1)))
        if operation == "add_segments"
        else partition.delete_segments([kept.uuid])
    )

    with (
        _failing_next_commit(store._engine),
        pytest.raises(RuntimeError, match="injected commit failure"),
    ):
        await write

    assert await _row_counts(store, partition._partition_uuid) == (1, 1)
    assert kept.uuid in await partition.get_segment_contexts([kept.uuid])


# ===================================================================
# Concurrency tests
# ===================================================================


async def _get_partition(engine: AsyncEngine) -> SQLAlchemySegmentStorePartition:
    """Create a partition handle that shares the engine."""
    store = SQLAlchemySegmentStore(SQLAlchemySegmentStoreParams(engine=engine))
    return await _open(store)


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
    assert await _row_counts(part, part._partition_uuid) == (50, 50)


@pytest.mark.asyncio
async def test_concurrent_reads_during_writes(
    store: SQLAlchemySegmentStore,
) -> None:
    """Reads should not fail or block indefinitely while writes are happening."""
    engine = store._engine
    partition = await _open(store)

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
    partition = await _open(store)

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


@pytest.mark.integration
@pytest.mark.asyncio
async def test_pg_context_preserved_via_lateral_join(
    pg_store: SQLAlchemySegmentStore,
) -> None:
    """Context is preserved when retrieved via the LATERAL join path (multiple seeds)."""
    partition = await _open(pg_store)
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
    partition = await _open(pg_store)
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


class _ForeignPayloadCodecConfig(PlaintextPayloadCodecConfig):
    """A distinct codec-config type standing in for a future variant."""


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
async def test_unsupported_dialect_engine_is_rejected(
    sqlalchemy_sqlite_engine: AsyncEngine,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The store supports PostgreSQL and SQLite; refuse any other dialect loudly."""
    monkeypatch.setattr(sqlalchemy_sqlite_engine.dialect, "name", "mssql")
    with pytest.raises(ValidationError, match="mssql"):
        SQLAlchemySegmentStoreParams(engine=sqlalchemy_sqlite_engine)


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
    partition = await _open(store, uuid4())

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
        assert await _row_counts(partition, partition._partition_uuid) == (0, 0)

    rng = random.Random(7)
    for _ in range(10):
        await asyncio.wait_for(round_trip(rng), 30)


# ===================================================================
# Partitions
# ===================================================================


@pytest.mark.asyncio
async def test_an_uncreated_partition_does_not_open(
    store: SQLAlchemySegmentStore,
) -> None:
    assert await store.open_partition(uuid4()) is None


@pytest.mark.asyncio
async def test_a_created_partition_opens_with_its_config(
    store: SQLAlchemySegmentStore,
) -> None:
    await store.create_partition(PARTITION_UUID)

    partition = await store.open_partition(PARTITION_UUID)
    assert partition is not None
    assert partition.config == _plaintext_partition_config()


@pytest.mark.asyncio
async def test_creating_again_with_the_same_config_changes_nothing(
    store: SQLAlchemySegmentStore,
) -> None:
    partition = await _open(store)
    segment = _seg()
    await partition.add_segments(_links(segment))

    await store.create_partition(PARTITION_UUID)
    assert await _row_counts(store, PARTITION_UUID) == (1, 1)
    assert segment.uuid in await partition.get_segment_contexts([segment.uuid])


class _ForeignPayloadCodecConfig(PlaintextPayloadCodecConfig):
    """A distinct codec-config type standing in for a future variant."""


@pytest.mark.asyncio
async def test_creating_again_with_a_different_config_raises_mismatch(
    store: SQLAlchemySegmentStore,
) -> None:
    """Creating a partition again under a different config is refused, not adapted."""
    await _open(store)

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

    # The partition itself is untouched and still opens.
    assert await store.open_partition(PARTITION_UUID) is not None


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
async def test_purge_removes_the_segments_and_their_links_and_the_drop_the_partition(
    store: SQLAlchemySegmentStore,
) -> None:
    partition = await _open(store)
    await partition.add_segments(_links(_seg(), _seg(ts_offset_seconds=1)))

    assert await store.purge_partition(PARTITION_UUID) is True
    assert await _row_counts(store, PARTITION_UUID) == (0, 0)
    assert await store.purge_partition(PARTITION_UUID) is False
    # The partition keeps its own row until it is dropped.
    assert await store.open_partition(PARTITION_UUID) is not None

    await store.drop_partition(PARTITION_UUID)
    assert await store.open_partition(PARTITION_UUID) is None
    await store.drop_partition(PARTITION_UUID)


@pytest.mark.asyncio
async def test_a_partition_that_holds_no_segments_purges_nothing_and_drops(
    store: SQLAlchemySegmentStore,
) -> None:
    await _open(store)

    assert await store.purge_partition(PARTITION_UUID) is False
    await store.drop_partition(PARTITION_UUID)
    assert await store.open_partition(PARTITION_UUID) is None


@pytest.mark.asyncio
async def test_purge_of_an_uncreated_partition_finds_nothing(
    store: SQLAlchemySegmentStore,
) -> None:
    assert await store.purge_partition(uuid4()) is False


@pytest.mark.asyncio
async def test_purge_leaves_other_partitions_whole(
    store: SQLAlchemySegmentStore,
) -> None:
    keeper = await _open(store, OTHER_PARTITION_UUID)
    segment = _seg()
    derivative_uuid = uuid4()
    await keeper.add_segments({segment: [derivative_uuid]})
    doomed = await _open(store)
    await doomed.add_segments(_links(_seg(), _seg(ts_offset_seconds=1)))

    while await store.purge_partition(PARTITION_UUID):
        pass

    assert await _row_counts(store, OTHER_PARTITION_UUID) == (1, 1)
    # The derivative-link cascade still works for the partition UUID the purge
    # left.
    await keeper.delete_segments([segment.uuid])
    assert await keeper.get_derivative_uuids_by_segment_uuids([segment.uuid]) == {}


@pytest.mark.integration
@pytest.mark.asyncio
async def test_purge_keeps_the_foreign_key_enforced(
    pg_store: SQLAlchemySegmentStore,
    sqlalchemy_pg_engine: AsyncEngine,
) -> None:
    """Purging one partition UUID must not drop the derivative-link foreign key."""
    await _open(pg_store, OTHER_PARTITION_UUID)
    await _open(pg_store)

    while await pg_store.purge_partition(PARTITION_UUID):
        pass

    # A derivative link pointing at no segment must still be rejected in the
    # partition UUID the purge left.
    with pytest.raises(IntegrityError):
        async with sqlalchemy_pg_engine.begin() as connection:
            await connection.execute(
                text(
                    "INSERT INTO segment_store_dv_ln"
                    " (partition_uuid, uuid, segment_uuid)"
                    " VALUES (:partition_uuid, :uuid, :segment_uuid)"
                ),
                {
                    "partition_uuid": OTHER_PARTITION_UUID,
                    "uuid": uuid4(),
                    "segment_uuid": uuid4(),
                },
            )


@pytest.mark.asyncio
async def test_purge_is_bounded_and_continues_after_its_cursor(
    sqlalchemy_sqlite_engine: AsyncEngine,
) -> None:
    """The configured bound governs every call, and each call picks up where the last stopped."""
    store = SQLAlchemySegmentStore(
        SQLAlchemySegmentStoreParams(
            engine=sqlalchemy_sqlite_engine,
            purge_max_segments=2,
        )
    )
    await store.startup()
    try:
        partition = await _open(store)
        segments = [_seg(ts_offset_seconds=index) for index in range(5)]
        await partition.add_segments(_links(*segments))
        ordered = sorted(segment.uuid for segment in segments)

        assert await store.purge_partition(PARTITION_UUID) is True
        assert await _row_counts(store, PARTITION_UUID) == (3, 3)
        async with store._create_session() as session:
            cursor = (
                await session.execute(
                    select(PartitionRow.purged_through).where(
                        PartitionRow.partition_uuid == PARTITION_UUID
                    )
                )
            ).scalar_one()
        assert cursor == ordered[1]

        assert await store.purge_partition(PARTITION_UUID) is True
        assert await _row_counts(store, PARTITION_UUID) == (1, 1)
        # The walk from the cursor reaches the end, and the cursor is cleared
        # for a search from the first key.
        assert await store.purge_partition(PARTITION_UUID) is True
        assert await _row_counts(store, PARTITION_UUID) == (0, 0)
        assert await _purge_cursor(store) is None
        assert await store.purge_partition(PARTITION_UUID) is False
    finally:
        async with sqlalchemy_sqlite_engine.begin() as conn:
            await conn.run_sync(BaseSegmentStore.metadata.drop_all)


@pytest.mark.asyncio
async def test_purge_finds_nothing_only_after_searching_from_the_first_key(
    sqlalchemy_sqlite_engine: AsyncEngine,
) -> None:
    """A segment written behind the cursor is purged before the purge finds nothing."""
    store = SQLAlchemySegmentStore(
        SQLAlchemySegmentStoreParams(
            engine=sqlalchemy_sqlite_engine,
            purge_max_segments=2,
        )
    )
    await store.startup()
    try:
        partition = await _open(store)
        await partition.add_segments(
            _links(*(_seg(ts_offset_seconds=index) for index in range(3)))
        )
        assert await store.purge_partition(PARTITION_UUID) is True
        assert await _purge_cursor(store) is not None

        # A late write keyed below every key the purge has passed.
        behind = _seg().model_copy(update={"uuid": UUID(int=1)})
        await partition.add_segments(_links(behind))

        while await store.purge_partition(PARTITION_UUID):
            pass
        assert await _row_counts(store, PARTITION_UUID) == (0, 0)
    finally:
        async with sqlalchemy_sqlite_engine.begin() as conn:
            await conn.run_sync(BaseSegmentStore.metadata.drop_all)


async def _purge_cursor(store: SQLAlchemySegmentStore) -> UUID | None:
    async with store._create_session() as session:
        return (
            await session.execute(
                select(PartitionRow.purged_through).where(
                    PartitionRow.partition_uuid == PARTITION_UUID
                )
            )
        ).scalar_one()


@pytest.mark.asyncio
async def test_a_failed_purge_call_leaves_nothing_half_done(
    store: SQLAlchemySegmentStore,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A purge call whose commit fails changes nothing; later calls finish.

    A call's deletions and its record of how far it got commit together or
    not at all, so no row is ever skipped after a failure.
    """
    partition = await _open(store)
    await partition.add_segments(
        _links(*(_seg(ts_offset_seconds=index) for index in range(5)))
    )
    monkeypatch.setattr(store, "_purge_max_segments", 2)

    with (
        _failing_next_commit(store._engine),
        pytest.raises(RuntimeError, match="injected commit failure"),
    ):
        await store.purge_partition(PARTITION_UUID)
    assert await _row_counts(store, PARTITION_UUID) == (5, 5)

    while await store.purge_partition(PARTITION_UUID):
        pass
    assert await _row_counts(store, PARTITION_UUID) == (0, 0)


@pytest.mark.asyncio
async def test_concurrent_purges_of_one_partition_reclaim_everything(
    store: SQLAlchemySegmentStore,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Racing purgers of one partition neither error nor leave rows behind."""
    monkeypatch.setattr(store, "_purge_max_segments", 2)
    partition = await _open(store)
    await partition.add_segments(
        _links(*(_seg(ts_offset_seconds=index) for index in range(7)))
    )

    async def drain() -> None:
        while await store.purge_partition(PARTITION_UUID):
            pass

    await asyncio.wait_for(asyncio.gather(drain(), drain(), drain()), 60)

    assert await _row_counts(store, PARTITION_UUID) == (0, 0)


@pytest.mark.asyncio
async def test_purge_reclaims_a_write_that_lands_after_the_partition_is_dropped(
    store: SQLAlchemySegmentStore,
) -> None:
    """A late write to a dropped partition is reclaimed by the next purge call."""
    partition = await _open(store)
    while await store.purge_partition(PARTITION_UUID):
        pass
    await store.drop_partition(PARTITION_UUID)

    await partition.add_segments(_links(_seg()))

    assert await store.purge_partition(PARTITION_UUID) is True
    assert await _row_counts(store, PARTITION_UUID) == (0, 0)
    assert await store.purge_partition(PARTITION_UUID) is False


@pytest.mark.asyncio
async def test_purging_while_other_partitions_are_written_leaves_exactly_their_rows(
    store: SQLAlchemySegmentStore,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Purges of some partition UUIDs and writes to others interleave cleanly."""
    monkeypatch.setattr(store, "_purge_max_segments", 3)
    retired: list[UUID] = []
    for _ in range(3):
        partition_uuid = uuid4()
        partition = await _open(store, partition_uuid)
        await partition.add_segments(
            _links(*(_seg(ts_offset_seconds=index) for index in range(5)))
        )
        retired.append(partition_uuid)
    live = [await _open(store, uuid4()) for _ in range(2)]

    async def purge(partition_uuid: UUID) -> None:
        while await store.purge_partition(partition_uuid):
            pass

    async def write(partition: SQLAlchemySegmentStorePartition, seed: int) -> int:
        rng = random.Random(seed)
        written = 0
        for step in range(20):
            count = rng.randint(1, 3)
            await partition.add_segments(
                _links(
                    *(
                        _seg(ts_offset_seconds=step * 10 + index)
                        for index in range(count)
                    )
                )
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
        assert await _row_counts(store, partition_uuid) == (0, 0)
    for partition, written in zip(live, results[: len(live)], strict=True):
        assert await _row_counts(store, partition._partition_uuid) == (written, written)


async def _assert_handles_match_model(
    handles: dict[str, SQLAlchemySegmentStorePartition],
    model: dict[str, dict[UUID, set[UUID]]],
) -> None:
    """Each handle reads back exactly its model's segments and links."""
    for slot, handle in handles.items():
        expected = model[slot]
        links = await handle.get_derivative_uuids_by_segment_uuids(expected)
        assert {
            segment_uuid: set(derivatives)
            for segment_uuid, derivatives in links.items()
        } == {
            segment_uuid: derivatives
            for segment_uuid, derivatives in expected.items()
            if derivatives
        }
        assert set(await handle.get_segment_contexts(expected)) == set(expected)


@pytest.mark.asyncio
@pytest.mark.parametrize("seed", [1, 2])
async def test_random_operation_sequences_agree_with_a_model(
    store: SQLAlchemySegmentStore,
    monkeypatch: pytest.MonkeyPatch,
    seed: int,
) -> None:
    """Random sequences of every data operation agree with an in-memory model.

    Adds and segment deletes in a few slots' current partition UUIDs, retiring
    a slot's partition UUID for a new one, and bounded purge calls on retired
    partition UUIDs. After every step each current partition UUID reads back
    exactly the model's segments and links, and once every retired
    partition UUID is purged, the tables hold exactly the current rows.
    """
    rng = random.Random(seed)
    monkeypatch.setattr(store, "_purge_max_segments", 3)
    slots = ["model_a", "model_b", "model_c"]
    handles: dict[str, SQLAlchemySegmentStorePartition] = {}
    model: dict[str, dict[UUID, set[UUID]]] = {}
    retired: list[UUID] = []

    for step in range(120):
        slot = rng.choice(slots)
        operation = rng.choice(["add", "add", "delete_segments", "retire", "purge"])
        if operation == "purge" and retired:
            await store.purge_partition(rng.choice(retired))
        elif operation == "retire" and slot in handles:
            retired.append(handles.pop(slot)._partition_uuid)
            model.pop(slot)
        elif operation in ("add", "delete_segments"):
            if slot not in handles:
                handles[slot] = await _open(store, uuid4())
                model[slot] = {}
            if operation == "add":
                segments = {
                    _seg(ts_offset_seconds=step * 10 + offset): [
                        uuid4() for _ in range(rng.randint(0, 2))
                    ]
                    for offset in range(rng.randint(1, 4))
                }
                await handles[slot].add_segments(segments)
                model[slot].update(
                    {segment.uuid: set(links) for segment, links in segments.items()}
                )
            else:
                victims = rng.sample(
                    sorted(model[slot]), min(len(model[slot]), rng.randint(0, 3))
                )
                await handles[slot].delete_segments([*victims, uuid4()])
                for victim in victims:
                    del model[slot][victim]

        await _assert_handles_match_model(handles, model)

    for partition_uuid in retired:
        while await store.purge_partition(partition_uuid):
            pass
    async with store._create_session() as session:
        segment_rows = (
            await session.execute(select(func.count()).select_from(SegmentRow))
        ).scalar_one()
        link_rows = (
            await session.execute(select(func.count()).select_from(DerivativeLinkRow))
        ).scalar_one()
    assert segment_rows == sum(map(len, model.values()))
    assert link_rows == sum(
        len(derivatives)
        for segments in model.values()
        for derivatives in segments.values()
    )
