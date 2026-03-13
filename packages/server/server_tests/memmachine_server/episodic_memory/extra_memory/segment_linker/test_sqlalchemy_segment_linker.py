"""Tests for SQLAlchemySegmentLinker — SQLite (unit) and PostgreSQL (integration)."""

from __future__ import annotations

from collections.abc import AsyncIterator
from datetime import UTC, datetime, timedelta
from uuid import UUID, uuid4

import pytest
import pytest_asyncio
from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncEngine

from memmachine_server.common.filter.filter_parser import Comparison
from memmachine_server.episodic_memory.extra_memory.data_types import (
    Segment,
    Text,
)
from memmachine_server.episodic_memory.extra_memory.segment_linker.segment_linker import (
    DerivativeNotActiveError,
)
from memmachine_server.episodic_memory.extra_memory.segment_linker.sqlalchemy_segment_linker import (
    BaseSegmentLinker,
    SegmentRow,
    SQLAlchemySegmentLinker,
    SQLAlchemySegmentLinkerParams,
)

PARTITION_KEY = "test-partition"
BASE_TIME = datetime(2024, 1, 1, tzinfo=UTC)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _seg(
    *,
    episode_uuid: UUID | None = None,
    index: int = 0,
    offset: int = 0,
    ts_offset_seconds: int = 0,
    text: str = "hello",
    properties: dict | None = None,
) -> Segment:
    return Segment(
        uuid=uuid4(),
        episode_uuid=episode_uuid or uuid4(),
        index=index,
        offset=offset,
        timestamp=BASE_TIME + timedelta(seconds=ts_offset_seconds),
        block=Text(text=text),
        properties=properties or {},
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest_asyncio.fixture
async def sqlite_linker(
    sqlalchemy_sqlite_engine: AsyncEngine,
) -> AsyncIterator[SQLAlchemySegmentLinker]:
    linker = SQLAlchemySegmentLinker(
        SQLAlchemySegmentLinkerParams(engine=sqlalchemy_sqlite_engine)
    )
    await linker.startup()
    yield linker
    async with sqlalchemy_sqlite_engine.begin() as conn:
        await conn.run_sync(BaseSegmentLinker.metadata.drop_all)


@pytest_asyncio.fixture
async def pg_linker(
    sqlalchemy_pg_engine: AsyncEngine,
) -> AsyncIterator[SQLAlchemySegmentLinker]:
    linker = SQLAlchemySegmentLinker(
        SQLAlchemySegmentLinkerParams(engine=sqlalchemy_pg_engine)
    )
    await linker.startup()
    yield linker
    async with sqlalchemy_pg_engine.begin() as conn:
        await conn.run_sync(BaseSegmentLinker.metadata.drop_all)


@pytest.fixture(
    params=[
        "sqlite_linker",
        pytest.param("pg_linker", marks=pytest.mark.integration),
    ],
)
def linker(request) -> SQLAlchemySegmentLinker:
    return request.getfixturevalue(request.param)


# ===================================================================
# register_segments
# ===================================================================


@pytest.mark.asyncio
async def test_register_and_get_by_derivatives(linker: SQLAlchemySegmentLinker) -> None:
    seg = _seg(text="a")
    deriv_uuid = uuid4()
    await linker.register_segments(PARTITION_KEY, {seg: [deriv_uuid]})

    result = await linker.get_segments_by_derivatives(PARTITION_KEY, [deriv_uuid])
    assert len(list(result[deriv_uuid])) == 1
    assert next(iter(result[deriv_uuid])).uuid == seg.uuid


@pytest.mark.asyncio
async def test_register_multiple_segments_same_derivative(
    linker: SQLAlchemySegmentLinker,
) -> None:
    ep = uuid4()
    s1 = _seg(episode_uuid=ep, offset=0, ts_offset_seconds=0)
    s2 = _seg(episode_uuid=ep, offset=1, ts_offset_seconds=1)
    deriv = uuid4()
    await linker.register_segments(PARTITION_KEY, {s1: [deriv], s2: [deriv]})

    result = await linker.get_segments_by_derivatives(PARTITION_KEY, [deriv])
    segments = list(result[deriv])
    assert len(segments) == 2
    assert {s.uuid for s in segments} == {s1.uuid, s2.uuid}


@pytest.mark.asyncio
async def test_register_with_properties(linker: SQLAlchemySegmentLinker) -> None:
    seg = _seg(properties={"color": "red", "score": 42})
    deriv = uuid4()
    await linker.register_segments(PARTITION_KEY, {seg: [deriv]})

    result = await linker.get_segments_by_derivatives(PARTITION_KEY, [deriv])
    returned = next(iter(result[deriv]))
    assert returned.properties == {"color": "red", "score": 42}


@pytest.mark.asyncio
async def test_register_active_validation(linker: SQLAlchemySegmentLinker) -> None:
    seg = _seg()
    deriv = uuid4()
    unknown = uuid4()
    with pytest.raises(DerivativeNotActiveError):
        await linker.register_segments(PARTITION_KEY, {seg: [deriv]}, active=[unknown])


@pytest.mark.asyncio
async def test_register_empty_links(linker: SQLAlchemySegmentLinker) -> None:
    await linker.register_segments(PARTITION_KEY, {})


@pytest.mark.asyncio
async def test_register_empty_links_still_validates_active(
    linker: SQLAlchemySegmentLinker,
) -> None:
    unknown = uuid4()
    with pytest.raises(DerivativeNotActiveError):
        await linker.register_segments(PARTITION_KEY, {}, active=[unknown])


@pytest.mark.asyncio
async def test_register_rejects_existing_derivative_not_declared_active(
    linker: SQLAlchemySegmentLinker,
) -> None:
    """Linking to an existing derivative without declaring it active is an error."""
    seg1 = _seg(ts_offset_seconds=0)
    deriv = uuid4()
    await linker.register_segments(PARTITION_KEY, {seg1: [deriv]})

    seg2 = _seg(ts_offset_seconds=1)
    with pytest.raises(IntegrityError):
        await linker.register_segments(PARTITION_KEY, {seg2: [deriv]})


@pytest.mark.asyncio
async def test_register_allows_existing_derivative_declared_active(
    linker: SQLAlchemySegmentLinker,
) -> None:
    """Linking to an existing derivative works when declared active."""
    seg1 = _seg(ts_offset_seconds=0)
    deriv = uuid4()
    await linker.register_segments(PARTITION_KEY, {seg1: [deriv]})

    seg2 = _seg(ts_offset_seconds=1)
    await linker.register_segments(PARTITION_KEY, {seg2: [deriv]}, active=[deriv])

    result = await linker.get_segments_by_derivatives(PARTITION_KEY, [deriv])
    assert len(list(result[deriv])) == 2


# ===================================================================
# get_segments_by_derivatives
# ===================================================================


@pytest.mark.asyncio
async def test_get_by_derivatives_empty(linker: SQLAlchemySegmentLinker) -> None:
    result = await linker.get_segments_by_derivatives(PARTITION_KEY, [])
    assert result == {}


@pytest.mark.asyncio
async def test_get_by_derivatives_unknown_uuid(linker: SQLAlchemySegmentLinker) -> None:
    result = await linker.get_segments_by_derivatives(PARTITION_KEY, [uuid4()])
    assert len(result) == 0


@pytest.mark.asyncio
async def test_get_by_derivatives_limit(linker: SQLAlchemySegmentLinker) -> None:
    ep = uuid4()
    segs = [_seg(episode_uuid=ep, offset=i, ts_offset_seconds=i) for i in range(5)]
    deriv = uuid4()
    await linker.register_segments(PARTITION_KEY, {s: [deriv] for s in segs})

    result = await linker.get_segments_by_derivatives(
        PARTITION_KEY, [deriv], limit_per_derivative=2
    )
    assert len(list(result[deriv])) == 2


@pytest.mark.asyncio
async def test_get_by_derivatives_property_filter(
    linker: SQLAlchemySegmentLinker,
) -> None:
    deriv = uuid4()
    s_red = _seg(ts_offset_seconds=0, properties={"color": "red"})
    s_blue = _seg(ts_offset_seconds=1, properties={"color": "blue"})
    await linker.register_segments(PARTITION_KEY, {s_red: [deriv], s_blue: [deriv]})

    filt = Comparison(field="color", op="=", value="red")
    result = await linker.get_segments_by_derivatives(
        PARTITION_KEY, [deriv], property_filter=filt
    )
    segments = list(result[deriv])
    assert len(segments) == 1
    assert segments[0].uuid == s_red.uuid


@pytest.mark.asyncio
async def test_get_by_derivatives_session_isolation(
    linker: SQLAlchemySegmentLinker,
) -> None:
    seg = _seg()
    deriv = uuid4()
    await linker.register_segments("session-a", {seg: [deriv]})

    result = await linker.get_segments_by_derivatives("session-b", [deriv])
    assert deriv not in result


# ===================================================================
# get_segment_contexts
# ===================================================================


@pytest.mark.asyncio
async def test_contexts_empty_seeds(linker: SQLAlchemySegmentLinker) -> None:
    result = await linker.get_segment_contexts(PARTITION_KEY, [])
    assert result == {}


@pytest.mark.asyncio
async def test_contexts_unknown_seed(linker: SQLAlchemySegmentLinker) -> None:
    result = await linker.get_segment_contexts(
        PARTITION_KEY, [uuid4()], max_backward_segments=2, max_forward_segments=2
    )
    assert result == {}


@pytest.mark.asyncio
async def test_contexts_seed_only(linker: SQLAlchemySegmentLinker) -> None:
    """When max_backward=0 and max_forward=0, return just the seed."""
    seg = _seg()
    deriv = uuid4()
    await linker.register_segments(PARTITION_KEY, {seg: [deriv]})

    result = await linker.get_segment_contexts(PARTITION_KEY, [seg.uuid])
    assert seg.uuid in result
    ctx = list(result[seg.uuid])
    assert len(ctx) == 1
    assert ctx[0].uuid == seg.uuid


@pytest.mark.asyncio
async def test_contexts_backward(linker: SQLAlchemySegmentLinker) -> None:
    ep = uuid4()
    segs = [_seg(episode_uuid=ep, offset=i, ts_offset_seconds=i) for i in range(5)]
    deriv = uuid4()
    await linker.register_segments(PARTITION_KEY, {s: [deriv] for s in segs})

    seed = segs[3]
    result = await linker.get_segment_contexts(
        PARTITION_KEY, [seed.uuid], max_backward_segments=2
    )
    ctx = list(result[seed.uuid])
    # backward(2) + seed = 3 segments
    assert len(ctx) == 3
    uuids = [s.uuid for s in ctx]
    assert uuids == [segs[1].uuid, segs[2].uuid, seed.uuid]


@pytest.mark.asyncio
async def test_contexts_forward(linker: SQLAlchemySegmentLinker) -> None:
    ep = uuid4()
    segs = [_seg(episode_uuid=ep, offset=i, ts_offset_seconds=i) for i in range(5)]
    deriv = uuid4()
    await linker.register_segments(PARTITION_KEY, {s: [deriv] for s in segs})

    seed = segs[1]
    result = await linker.get_segment_contexts(
        PARTITION_KEY, [seed.uuid], max_forward_segments=2
    )
    ctx = list(result[seed.uuid])
    # seed + forward(2) = 3 segments
    assert len(ctx) == 3
    uuids = [s.uuid for s in ctx]
    assert uuids == [seed.uuid, segs[2].uuid, segs[3].uuid]


@pytest.mark.asyncio
async def test_contexts_backward_and_forward(
    linker: SQLAlchemySegmentLinker,
) -> None:
    ep = uuid4()
    segs = [_seg(episode_uuid=ep, offset=i, ts_offset_seconds=i) for i in range(7)]
    deriv = uuid4()
    await linker.register_segments(PARTITION_KEY, {s: [deriv] for s in segs})

    seed = segs[3]
    result = await linker.get_segment_contexts(
        PARTITION_KEY, [seed.uuid], max_backward_segments=2, max_forward_segments=2
    )
    ctx = list(result[seed.uuid])
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
    linker: SQLAlchemySegmentLinker,
) -> None:
    """Requesting more context than available returns what exists."""
    ep = uuid4()
    segs = [_seg(episode_uuid=ep, offset=i, ts_offset_seconds=i) for i in range(3)]
    deriv = uuid4()
    await linker.register_segments(PARTITION_KEY, {s: [deriv] for s in segs})

    seed = segs[0]
    result = await linker.get_segment_contexts(
        PARTITION_KEY,
        [seed.uuid],
        max_backward_segments=10,
        max_forward_segments=10,
    )
    ctx = list(result[seed.uuid])
    # 0 backward (at start) + seed + 2 forward = 3
    assert len(ctx) == 3
    uuids = [s.uuid for s in ctx]
    assert uuids == [segs[0].uuid, segs[1].uuid, segs[2].uuid]


@pytest.mark.asyncio
async def test_contexts_multiple_seeds(linker: SQLAlchemySegmentLinker) -> None:
    ep = uuid4()
    segs = [_seg(episode_uuid=ep, offset=i, ts_offset_seconds=i) for i in range(10)]
    deriv = uuid4()
    await linker.register_segments(PARTITION_KEY, {s: [deriv] for s in segs})

    seed_a, seed_b = segs[2], segs[7]
    result = await linker.get_segment_contexts(
        PARTITION_KEY,
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
async def test_contexts_with_properties(linker: SQLAlchemySegmentLinker) -> None:
    """Properties are loaded for seed and context segments."""
    ep = uuid4()
    s0 = _seg(episode_uuid=ep, offset=0, ts_offset_seconds=0, properties={"k": "v0"})
    s1 = _seg(episode_uuid=ep, offset=1, ts_offset_seconds=1, properties={"k": "v1"})
    s2 = _seg(episode_uuid=ep, offset=2, ts_offset_seconds=2, properties={"k": "v2"})
    deriv = uuid4()
    await linker.register_segments(
        PARTITION_KEY, {s0: [deriv], s1: [deriv], s2: [deriv]}
    )

    result = await linker.get_segment_contexts(
        PARTITION_KEY, [s1.uuid], max_backward_segments=1, max_forward_segments=1
    )
    ctx = list(result[s1.uuid])
    assert len(ctx) == 3
    assert ctx[0].properties == {"k": "v0"}
    assert ctx[1].properties == {"k": "v1"}
    assert ctx[2].properties == {"k": "v2"}


@pytest.mark.asyncio
async def test_contexts_property_filter(linker: SQLAlchemySegmentLinker) -> None:
    """Property filter excludes context rows that don't match."""
    ep = uuid4()
    s0 = _seg(episode_uuid=ep, offset=0, ts_offset_seconds=0, properties={"tag": "a"})
    s1 = _seg(episode_uuid=ep, offset=1, ts_offset_seconds=1, properties={"tag": "b"})
    s2 = _seg(episode_uuid=ep, offset=2, ts_offset_seconds=2, properties={"tag": "a"})
    s3 = _seg(episode_uuid=ep, offset=3, ts_offset_seconds=3, properties={"tag": "a"})
    deriv = uuid4()
    await linker.register_segments(
        PARTITION_KEY, {s0: [deriv], s1: [deriv], s2: [deriv], s3: [deriv]}
    )

    filt = Comparison(field="tag", op="=", value="a")
    result = await linker.get_segment_contexts(
        PARTITION_KEY,
        [s2.uuid],
        max_backward_segments=5,
        max_forward_segments=5,
        property_filter=filt,
    )
    ctx = list(result[s2.uuid])
    uuids = [s.uuid for s in ctx]
    # s1 excluded (tag=b); s0 backward, s2 seed, s3 forward
    assert uuids == [s0.uuid, s2.uuid, s3.uuid]


@pytest.mark.asyncio
async def test_contexts_session_isolation(linker: SQLAlchemySegmentLinker) -> None:
    """Context only includes segments from the same partition_key."""
    ep = uuid4()
    s_other = _seg(episode_uuid=ep, offset=0, ts_offset_seconds=0)
    s_seed = _seg(episode_uuid=ep, offset=1, ts_offset_seconds=1)
    s_after = _seg(episode_uuid=ep, offset=2, ts_offset_seconds=2)
    deriv_other = uuid4()
    deriv = uuid4()
    await linker.register_segments("other-session", {s_other: [deriv_other]})
    await linker.register_segments(PARTITION_KEY, {s_seed: [deriv], s_after: [deriv]})

    result = await linker.get_segment_contexts(
        PARTITION_KEY,
        [s_seed.uuid],
        max_backward_segments=5,
        max_forward_segments=5,
    )
    ctx = list(result[s_seed.uuid])
    uuids = [s.uuid for s in ctx]
    assert s_other.uuid not in uuids
    assert uuids == [s_seed.uuid, s_after.uuid]


@pytest.mark.asyncio
async def test_contexts_chronological_order(linker: SQLAlchemySegmentLinker) -> None:
    """Context segments are returned in chronological order."""
    ep = uuid4()
    segs = [_seg(episode_uuid=ep, offset=i, ts_offset_seconds=i) for i in range(5)]
    deriv = uuid4()
    await linker.register_segments(PARTITION_KEY, {s: [deriv] for s in segs})

    result = await linker.get_segment_contexts(
        PARTITION_KEY,
        [segs[2].uuid],
        max_backward_segments=10,
        max_forward_segments=10,
    )
    ctx = list(result[segs[2].uuid])
    timestamps = [s.timestamp for s in ctx]
    assert timestamps == sorted(timestamps)


# ===================================================================
# delete_segments_by_episodes
# ===================================================================


@pytest.mark.asyncio
async def test_delete_by_episodes(linker: SQLAlchemySegmentLinker) -> None:
    ep = uuid4()
    seg = _seg(episode_uuid=ep)
    deriv = uuid4()
    await linker.register_segments(PARTITION_KEY, {seg: [deriv]})

    await linker.delete_segments_by_episodes(PARTITION_KEY, [ep])

    result = await linker.get_segments_by_derivatives(PARTITION_KEY, [deriv])
    assert deriv not in result


@pytest.mark.asyncio
async def test_delete_by_episodes_decrements_ref_count(
    linker: SQLAlchemySegmentLinker,
) -> None:
    ep = uuid4()
    s1 = _seg(episode_uuid=ep, offset=0)
    s2 = _seg(episode_uuid=ep, offset=1)
    deriv = uuid4()
    await linker.register_segments(PARTITION_KEY, {s1: [deriv], s2: [deriv]})

    await linker.delete_segments_by_episodes(PARTITION_KEY, [ep])

    orphans = list(await linker.get_orphaned_derivatives(PARTITION_KEY))
    assert deriv in orphans


@pytest.mark.asyncio
async def test_delete_by_episodes_noop_unknown(
    linker: SQLAlchemySegmentLinker,
) -> None:
    await linker.delete_segments_by_episodes(PARTITION_KEY, [uuid4()])


# ===================================================================
# delete_all_segments
# ===================================================================


@pytest.mark.asyncio
async def test_delete_all(linker: SQLAlchemySegmentLinker) -> None:
    deriv = uuid4()
    s1 = _seg(ts_offset_seconds=0)
    s2 = _seg(ts_offset_seconds=1)
    await linker.register_segments(PARTITION_KEY, {s1: [deriv], s2: [deriv]})

    await linker.delete_all_segments(PARTITION_KEY)

    result = await linker.get_segments_by_derivatives(PARTITION_KEY, [deriv])
    assert deriv not in result


# ===================================================================
# orphan management
# ===================================================================


@pytest.mark.asyncio
async def test_orphan_lifecycle(linker: SQLAlchemySegmentLinker) -> None:
    """Full lifecycle: register -> delete segments -> orphan -> mark -> purge."""
    ep = uuid4()
    seg = _seg(episode_uuid=ep)
    deriv = uuid4()
    await linker.register_segments(PARTITION_KEY, {seg: [deriv]})

    # No orphans yet
    assert list(await linker.get_orphaned_derivatives(PARTITION_KEY)) == []

    # Delete segment -> derivative becomes orphaned
    await linker.delete_segments_by_episodes(PARTITION_KEY, [ep])
    orphans = list(await linker.get_orphaned_derivatives(PARTITION_KEY))
    assert deriv in orphans

    # Mark for purging
    marked = list(
        await linker.mark_orphaned_derivatives_for_purging(PARTITION_KEY, [deriv])
    )
    assert deriv in marked

    # No longer shows as orphaned (state=P)
    assert list(await linker.get_orphaned_derivatives(PARTITION_KEY)) == []

    # Purge
    await linker.purge_derivatives(PARTITION_KEY, [deriv])

    # Registering against purged derivative should fail (with active check)
    seg2 = _seg()
    with pytest.raises(DerivativeNotActiveError):
        await linker.register_segments(PARTITION_KEY, {seg2: [deriv]}, active=[deriv])


@pytest.mark.asyncio
async def test_mark_orphaned_ignores_non_orphans(
    linker: SQLAlchemySegmentLinker,
) -> None:
    seg = _seg()
    deriv = uuid4()
    await linker.register_segments(PARTITION_KEY, {seg: [deriv]})

    marked = list(
        await linker.mark_orphaned_derivatives_for_purging(PARTITION_KEY, [deriv])
    )
    assert marked == []


@pytest.mark.asyncio
async def test_purge_empty(linker: SQLAlchemySegmentLinker) -> None:
    await linker.purge_derivatives(PARTITION_KEY, [])


# ===================================================================
# Concurrency tests
# ===================================================================


async def _make_linker(engine: AsyncEngine) -> SQLAlchemySegmentLinker:
    """Create a linker that shares the engine (and thus the connection pool / DB)."""
    linker = SQLAlchemySegmentLinker(SQLAlchemySegmentLinkerParams(engine=engine))
    await linker.startup()
    return linker


# --- Both backends ---


@pytest.mark.asyncio
async def test_concurrent_register_disjoint_derivatives(
    linker: SQLAlchemySegmentLinker,
) -> None:
    """Concurrent registrations with completely disjoint derivatives should not interfere."""
    import asyncio

    engine = linker._engine

    async def register_batch(batch_id: int) -> None:
        linker = await _make_linker(engine)
        segs = {
            _seg(ts_offset_seconds=batch_id * 10 + i, text=f"batch{batch_id}-{i}"): [
                uuid4()
            ]
            for i in range(5)
        }
        await linker.register_segments(PARTITION_KEY, segs)

    await asyncio.gather(*(register_batch(i) for i in range(10)))

    # Verify all segments were registered.
    linker = await _make_linker(engine)
    async with linker._create_session() as session:
        count = (
            await session.execute(select(func.count()).select_from(SegmentRow))
        ).scalar()
    assert count == 50


@pytest.mark.asyncio
async def test_concurrent_reads_during_writes(
    linker: SQLAlchemySegmentLinker,
) -> None:
    """Reads should not fail or block indefinitely while writes are happening."""
    import asyncio

    engine = linker._engine

    # Seed some data.
    ep = uuid4()
    segs = [_seg(episode_uuid=ep, offset=i, ts_offset_seconds=i) for i in range(10)]
    deriv = uuid4()
    await linker.register_segments(PARTITION_KEY, {s: [deriv] for s in segs})

    read_results: list[int] = []

    async def reader() -> None:
        linker = await _make_linker(engine)
        for _ in range(5):
            result = await linker.get_segments_by_derivatives(PARTITION_KEY, [deriv])
            if deriv in result:
                read_results.append(len(list(result[deriv])))
            await asyncio.sleep(0.01)

    async def writer() -> None:
        linker = await _make_linker(engine)
        for i in range(5):
            new_seg = _seg(ts_offset_seconds=100 + i)
            new_deriv = uuid4()
            await linker.register_segments(PARTITION_KEY, {new_seg: [new_deriv]})
            await asyncio.sleep(0.01)

    await asyncio.gather(reader(), reader(), writer())

    # Reads should have succeeded and returned at least the original 10 segments.
    assert len(read_results) > 0
    assert all(count >= 10 for count in read_results)


@pytest.mark.asyncio
async def test_concurrent_context_reads_during_deletes(
    linker: SQLAlchemySegmentLinker,
) -> None:
    """get_segment_contexts should not crash if segments are deleted concurrently."""
    import asyncio

    engine = linker._engine

    # Register segments across multiple episodes.
    episodes = [uuid4() for _ in range(5)]
    all_segs: list[Segment] = []
    deriv = uuid4()
    links: dict[Segment, list[UUID]] = {}
    for ep_idx, ep in enumerate(episodes):
        for i in range(4):
            seg = _seg(episode_uuid=ep, offset=i, ts_offset_seconds=ep_idx * 10 + i)
            all_segs.append(seg)
            links[seg] = [deriv]
    await linker.register_segments(PARTITION_KEY, links)

    errors: list[Exception] = []

    async def context_reader() -> None:
        linker = await _make_linker(engine)
        for seg in all_segs[::3]:  # Read every 3rd segment's context.
            try:
                await linker.get_segment_contexts(
                    PARTITION_KEY,
                    [seg.uuid],
                    max_backward_segments=2,
                    max_forward_segments=2,
                )
            except Exception as e:
                errors.append(e)
            await asyncio.sleep(0.01)

    async def episode_deleter() -> None:
        linker = await _make_linker(engine)
        for ep in episodes[1:3]:  # Delete 2 of 5 episodes.
            await linker.delete_segments_by_episodes(PARTITION_KEY, [ep])
            await asyncio.sleep(0.02)

    await asyncio.gather(context_reader(), episode_deleter())

    # No crashes. Some reads may return empty or partial results — that's fine.
    assert errors == []


@pytest.mark.asyncio
async def test_concurrent_delete_all_and_register(
    linker: SQLAlchemySegmentLinker,
) -> None:
    """delete_all_segments concurrent with register_segments should not deadlock."""
    import asyncio

    engine = linker._engine

    # Seed data.
    deriv = uuid4()
    segs = {_seg(ts_offset_seconds=i): [deriv] for i in range(5)}
    await linker.register_segments(PARTITION_KEY, segs)

    errors: list[Exception] = []

    async def deleter() -> None:
        try:
            linker = await _make_linker(engine)
            await linker.delete_all_segments(PARTITION_KEY)
        except Exception as e:
            errors.append(e)

    async def registerer() -> None:
        try:
            linker = await _make_linker(engine)
            new_segs = {_seg(ts_offset_seconds=100 + i): [uuid4()] for i in range(3)}
            await linker.register_segments(PARTITION_KEY, new_segs)
        except Exception as e:
            errors.append(e)

    await asyncio.gather(deleter(), registerer())
    assert errors == []


@pytest.mark.asyncio
async def test_concurrent_orphan_mark_does_not_crash(
    linker: SQLAlchemySegmentLinker,
) -> None:
    """Two concurrent mark_orphaned_derivatives_for_purging should not crash."""
    import asyncio

    engine = linker._engine

    # Register and orphan a derivative.
    ep = uuid4()
    seg = _seg(episode_uuid=ep)
    deriv = uuid4()
    await linker.register_segments(PARTITION_KEY, {seg: [deriv]})
    await linker.delete_segments_by_episodes(PARTITION_KEY, [ep])

    orphans = list(await linker.get_orphaned_derivatives(PARTITION_KEY))
    assert deriv in orphans

    errors: list[Exception] = []

    async def mark() -> list[UUID]:
        try:
            linker = await _make_linker(engine)
            return list(
                await linker.mark_orphaned_derivatives_for_purging(
                    PARTITION_KEY, [deriv]
                )
            )
        except Exception as e:
            errors.append(e)
            return []

    results = await asyncio.gather(mark(), mark())
    all_marked = [uuid for result in results for uuid in result]

    assert errors == []
    # At least one should have marked it.
    assert deriv in all_marked


@pytest.mark.asyncio
async def test_concurrent_delete_overlapping_episodes_shared_derivative(
    linker: SQLAlchemySegmentLinker,
) -> None:
    """Deleting two episodes that share a derivative concurrently should correctly decrement ref_count."""
    import asyncio

    engine = linker._engine

    ep1, ep2 = uuid4(), uuid4()
    seg1 = _seg(episode_uuid=ep1, ts_offset_seconds=0)
    seg2 = _seg(episode_uuid=ep2, ts_offset_seconds=1)
    deriv = uuid4()
    await linker.register_segments(PARTITION_KEY, {seg1: [deriv], seg2: [deriv]})

    # Both segments link to deriv (ref_count=2). Delete both episodes concurrently.
    async def delete_ep(ep: UUID) -> None:
        linker = await _make_linker(engine)
        await linker.delete_segments_by_episodes(PARTITION_KEY, [ep])

    await asyncio.gather(delete_ep(ep1), delete_ep(ep2))

    # Derivative should be orphaned (ref_count=0).
    orphans = list(await linker.get_orphaned_derivatives(PARTITION_KEY))
    assert deriv in orphans


# --- PostgreSQL-only concurrency tests ---


@pytest.mark.integration
@pytest.mark.asyncio
async def test_pg_concurrent_orphan_mark_exactly_once(
    pg_linker: SQLAlchemySegmentLinker,
) -> None:
    """On PG, FOR UPDATE ensures only one of two concurrent markers wins."""
    import asyncio

    engine = pg_linker._engine

    ep = uuid4()
    seg = _seg(episode_uuid=ep)
    deriv = uuid4()
    await pg_linker.register_segments(PARTITION_KEY, {seg: [deriv]})
    await pg_linker.delete_segments_by_episodes(PARTITION_KEY, [ep])

    orphans = list(await pg_linker.get_orphaned_derivatives(PARTITION_KEY))
    assert deriv in orphans

    async def mark() -> list[UUID]:
        linker = await _make_linker(engine)
        return list(
            await linker.mark_orphaned_derivatives_for_purging(PARTITION_KEY, [deriv])
        )

    results = await asyncio.gather(mark(), mark())
    all_marked = [uuid for result in results for uuid in result]

    # Exactly one should have marked it due to FOR UPDATE serialization.
    assert all_marked.count(deriv) == 1


@pytest.mark.integration
@pytest.mark.asyncio
async def test_pg_concurrent_register_same_derivative_declared_active(
    pg_linker: SQLAlchemySegmentLinker,
) -> None:
    """Two concurrent registrations both declaring a derivative as active should serialize correctly."""
    import asyncio

    engine = pg_linker._engine
    deriv = uuid4()

    # Initial registration.
    seg0 = _seg(ts_offset_seconds=0)
    await pg_linker.register_segments(PARTITION_KEY, {seg0: [deriv]})

    errors: list[Exception] = []

    async def register_with_active(offset: int) -> None:
        try:
            linker = await _make_linker(engine)
            seg = _seg(ts_offset_seconds=offset)
            await linker.register_segments(
                PARTITION_KEY, {seg: [deriv]}, active=[deriv]
            )
        except Exception as e:
            errors.append(e)

    await asyncio.gather(register_with_active(10), register_with_active(20))

    # Both should succeed (FOR UPDATE serializes the derivative lock).
    assert errors == []

    # Derivative should now have 3 linked segments.
    result = await pg_linker.get_segments_by_derivatives(PARTITION_KEY, [deriv])
    assert len(list(result[deriv])) == 3


@pytest.mark.integration
@pytest.mark.asyncio
async def test_pg_concurrent_register_and_delete_same_episode(
    pg_linker: SQLAlchemySegmentLinker,
) -> None:
    """Registering new segments for an episode while deleting it should not deadlock."""
    import asyncio

    engine = pg_linker._engine
    ep = uuid4()
    deriv1 = uuid4()

    # Seed.
    seg1 = _seg(episode_uuid=ep, offset=0, ts_offset_seconds=0)
    await pg_linker.register_segments(PARTITION_KEY, {seg1: [deriv1]})

    errors: list[Exception] = []

    async def deleter() -> None:
        try:
            linker = await _make_linker(engine)
            await linker.delete_segments_by_episodes(PARTITION_KEY, [ep])
        except Exception as e:
            errors.append(e)

    async def registerer() -> None:
        try:
            linker = await _make_linker(engine)
            seg2 = _seg(episode_uuid=ep, offset=1, ts_offset_seconds=1)
            deriv2 = uuid4()
            await linker.register_segments(PARTITION_KEY, {seg2: [deriv2]})
        except Exception as e:
            errors.append(e)

    await asyncio.gather(deleter(), registerer())

    # Neither should deadlock. Errors from serialization failures are acceptable
    # on PG but shouldn't crash.
    # We just verify no deadlock (the gather completed).


@pytest.mark.integration
@pytest.mark.asyncio
async def test_pg_orphan_relink_race(
    pg_linker: SQLAlchemySegmentLinker,
) -> None:
    """A derivative detected as orphaned that gets re-linked before marking should not be marked for purging."""
    import asyncio

    engine = pg_linker._engine
    ep = uuid4()
    seg = _seg(episode_uuid=ep, ts_offset_seconds=0)
    deriv = uuid4()
    await pg_linker.register_segments(PARTITION_KEY, {seg: [deriv]})

    # Orphan it.
    await pg_linker.delete_segments_by_episodes(PARTITION_KEY, [ep])
    orphans = list(await pg_linker.get_orphaned_derivatives(PARTITION_KEY))
    assert deriv in orphans

    # Now, concurrently re-link and try to mark for purging.
    relinked = asyncio.Event()

    async def relinker() -> None:
        linker = await _make_linker(engine)
        new_seg = _seg(ts_offset_seconds=10)
        await linker.register_segments(
            PARTITION_KEY, {new_seg: [deriv]}, active=[deriv]
        )
        relinked.set()

    async def marker() -> list[UUID]:
        # Wait a tiny bit to let relinker likely win, but it's a race so either outcome is valid.
        await asyncio.sleep(0.01)
        linker = await _make_linker(engine)
        return list(
            await linker.mark_orphaned_derivatives_for_purging(PARTITION_KEY, [deriv])
        )

    _, marked = await asyncio.gather(relinker(), marker())

    # If relinker won the race, deriv has ref_count > 0 and should NOT be marked.
    # If marker won the race, deriv was still orphaned and gets marked.
    # Either way, no crash.
    if not marked:
        # Relinker won — derivative should still be retrievable.
        result = await pg_linker.get_segments_by_derivatives(PARTITION_KEY, [deriv])
        assert deriv in result
    # If marked, the derivative was legitimately marked before relinking.


@pytest.mark.integration
@pytest.mark.asyncio
async def test_pg_concurrent_mass_deletes(
    pg_linker: SQLAlchemySegmentLinker,
) -> None:
    """Many concurrent episode deletes that share derivatives should not deadlock."""
    import asyncio

    engine = pg_linker._engine

    # Create 10 episodes, each with 2 segments, all sharing the same derivative.
    deriv = uuid4()
    episodes: list[UUID] = []
    links: dict[Segment, list[UUID]] = {}
    for ep_idx in range(10):
        ep = uuid4()
        episodes.append(ep)
        for i in range(2):
            seg = _seg(episode_uuid=ep, offset=i, ts_offset_seconds=ep_idx * 10 + i)
            links[seg] = [deriv]
    await pg_linker.register_segments(PARTITION_KEY, links)

    # Delete all episodes concurrently.
    async def delete_ep(ep: UUID) -> None:
        linker = await _make_linker(engine)
        await linker.delete_segments_by_episodes(PARTITION_KEY, [ep])

    await asyncio.gather(*(delete_ep(ep) for ep in episodes))

    # All segments should be gone. Derivative should be orphaned.
    result = await pg_linker.get_segments_by_derivatives(PARTITION_KEY, [deriv])
    assert deriv not in result

    orphans = list(await pg_linker.get_orphaned_derivatives(PARTITION_KEY))
    assert deriv in orphans


@pytest.mark.integration
@pytest.mark.asyncio
async def test_pg_concurrent_purge_and_register_new_derivative(
    pg_linker: SQLAlchemySegmentLinker,
) -> None:
    """Purging a derivative while registering a completely new derivative should not interfere."""
    import asyncio

    engine = pg_linker._engine

    # Create and orphan a derivative.
    ep = uuid4()
    seg = _seg(episode_uuid=ep)
    old_deriv = uuid4()
    await pg_linker.register_segments(PARTITION_KEY, {seg: [old_deriv]})
    await pg_linker.delete_segments_by_episodes(PARTITION_KEY, [ep])
    marked = list(
        await pg_linker.mark_orphaned_derivatives_for_purging(
            PARTITION_KEY, [old_deriv]
        )
    )
    assert old_deriv in marked

    errors: list[Exception] = []

    async def purger() -> None:
        try:
            linker = await _make_linker(engine)
            await linker.purge_derivatives(PARTITION_KEY, [old_deriv])
        except Exception as e:
            errors.append(e)

    async def registerer() -> None:
        try:
            linker = await _make_linker(engine)
            new_seg = _seg(ts_offset_seconds=100)
            new_deriv = uuid4()
            await linker.register_segments(PARTITION_KEY, {new_seg: [new_deriv]})
        except Exception as e:
            errors.append(e)

    await asyncio.gather(purger(), registerer())
    assert errors == []
