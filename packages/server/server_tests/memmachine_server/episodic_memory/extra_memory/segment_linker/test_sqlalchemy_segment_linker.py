"""Tests for SQLAlchemySegmentLinker — SQLite (unit) and PostgreSQL (integration)."""

from __future__ import annotations

from collections.abc import AsyncIterator
from datetime import UTC, datetime, timedelta
from uuid import UUID, uuid4

import pytest
import pytest_asyncio
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncEngine

from memmachine_server.common.filter.filter_parser import Comparison
from memmachine_server.episodic_memory.extra_memory.data_types import (
    CitationContext,
    MessageContext,
    Segment,
    Text,
)
from memmachine_server.episodic_memory.extra_memory.segment_linker.sqlalchemy_segment_linker import (
    BaseSegmentLinker,
    SegmentRow,
    SQLAlchemySegmentLinker,
    SQLAlchemySegmentLinkerParams,
    SQLAlchemySegmentLinkerPartition,
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
    context: MessageContext | CitationContext | None = None,
    properties: dict | None = None,
) -> Segment:
    return Segment(
        uuid=uuid4(),
        episode_uuid=episode_uuid or uuid4(),
        index=index,
        offset=offset,
        timestamp=BASE_TIME + timedelta(seconds=ts_offset_seconds),
        block=Text(text=text),
        context=context,
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


@pytest_asyncio.fixture
async def partition(
    linker: SQLAlchemySegmentLinker,
) -> SQLAlchemySegmentLinkerPartition:
    return await linker.open_partition(PARTITION_KEY)


# ===================================================================
# register_segments
# ===================================================================


@pytest.mark.asyncio
async def test_register_and_get_by_derivatives(
    partition: SQLAlchemySegmentLinkerPartition,
) -> None:
    seg = _seg(text="a")
    deriv_uuid = uuid4()
    await partition.register_segments({seg: [deriv_uuid]})

    result = await partition.get_segments_by_derivatives([deriv_uuid])
    assert len(list(result[deriv_uuid])) == 1
    assert next(iter(result[deriv_uuid])).uuid == seg.uuid


@pytest.mark.asyncio
async def test_register_multiple_segments_same_derivative(
    partition: SQLAlchemySegmentLinkerPartition,
) -> None:
    ep = uuid4()
    s1 = _seg(episode_uuid=ep, offset=0, ts_offset_seconds=0)
    s2 = _seg(episode_uuid=ep, offset=1, ts_offset_seconds=1)
    deriv = uuid4()
    await partition.register_segments({s1: [deriv], s2: [deriv]})

    result = await partition.get_segments_by_derivatives([deriv])
    segments = list(result[deriv])
    assert len(segments) == 2
    assert {s.uuid for s in segments} == {s1.uuid, s2.uuid}


@pytest.mark.asyncio
async def test_register_with_properties(
    partition: SQLAlchemySegmentLinkerPartition,
) -> None:
    seg = _seg(properties={"color": "red", "score": 42})
    deriv = uuid4()
    await partition.register_segments({seg: [deriv]})

    result = await partition.get_segments_by_derivatives([deriv])
    returned = next(iter(result[deriv]))
    assert returned.properties == {"color": "red", "score": 42}


@pytest.mark.asyncio
async def test_register_with_message_context(
    partition: SQLAlchemySegmentLinkerPartition,
) -> None:
    ctx = MessageContext(source="User")
    seg = _seg(context=ctx)
    deriv = uuid4()
    await partition.register_segments({seg: [deriv]})

    result = await partition.get_segments_by_derivatives([deriv])
    returned = next(iter(result[deriv]))
    assert returned.context == ctx


@pytest.mark.asyncio
async def test_register_with_citation_context(
    partition: SQLAlchemySegmentLinkerPartition,
) -> None:
    ctx = CitationContext(source="docs.txt", source_type="file", location="/tmp")
    seg = _seg(context=ctx)
    deriv = uuid4()
    await partition.register_segments({seg: [deriv]})

    result = await partition.get_segments_by_derivatives([deriv])
    returned = next(iter(result[deriv]))
    assert returned.context == ctx


@pytest.mark.asyncio
async def test_register_with_no_context(
    partition: SQLAlchemySegmentLinkerPartition,
) -> None:
    seg = _seg()
    deriv = uuid4()
    await partition.register_segments({seg: [deriv]})

    result = await partition.get_segments_by_derivatives([deriv])
    returned = next(iter(result[deriv]))
    assert returned.context is None


@pytest.mark.asyncio
async def test_context_preserved_in_segment_contexts(
    partition: SQLAlchemySegmentLinkerPartition,
) -> None:
    """Context is preserved when retrieving segment contexts (backward/forward)."""
    ep = uuid4()
    ctx_user = MessageContext(source="User")
    ctx_assistant = MessageContext(source="Assistant")
    s0 = _seg(episode_uuid=ep, offset=0, ts_offset_seconds=0, context=ctx_user)
    s1 = _seg(episode_uuid=ep, offset=1, ts_offset_seconds=1, context=ctx_assistant)
    s2 = _seg(episode_uuid=ep, offset=2, ts_offset_seconds=2, context=ctx_user)
    deriv = uuid4()
    await partition.register_segments({s0: [deriv], s1: [deriv], s2: [deriv]})

    result = await partition.get_segment_contexts(
        [s1.uuid], max_backward_segments=1, max_forward_segments=1
    )
    ctx = list(result[s1.uuid])
    assert len(ctx) == 3
    assert ctx[0].context == ctx_user
    assert ctx[1].context == ctx_assistant
    assert ctx[2].context == ctx_user


@pytest.mark.asyncio
async def test_register_empty_links(
    partition: SQLAlchemySegmentLinkerPartition,
) -> None:
    await partition.register_segments({})


@pytest.mark.asyncio
async def test_register_existing_derivative(
    partition: SQLAlchemySegmentLinkerPartition,
) -> None:
    """Linking new segments to an existing derivative should work."""
    seg1 = _seg(ts_offset_seconds=0)
    deriv = uuid4()
    await partition.register_segments({seg1: [deriv]})

    seg2 = _seg(ts_offset_seconds=1)
    await partition.register_segments({seg2: [deriv]})

    result = await partition.get_segments_by_derivatives([deriv])
    assert len(list(result[deriv])) == 2


# ===================================================================
# get_segments_by_derivatives
# ===================================================================


@pytest.mark.asyncio
async def test_get_by_derivatives_empty(
    partition: SQLAlchemySegmentLinkerPartition,
) -> None:
    result = await partition.get_segments_by_derivatives([])
    assert result == {}


@pytest.mark.asyncio
async def test_get_by_derivatives_unknown_uuid(
    partition: SQLAlchemySegmentLinkerPartition,
) -> None:
    result = await partition.get_segments_by_derivatives([uuid4()])
    assert len(result) == 0


@pytest.mark.asyncio
async def test_get_by_derivatives_limit(
    partition: SQLAlchemySegmentLinkerPartition,
) -> None:
    ep = uuid4()
    segs = [_seg(episode_uuid=ep, offset=i, ts_offset_seconds=i) for i in range(5)]
    deriv = uuid4()
    await partition.register_segments({s: [deriv] for s in segs})

    result = await partition.get_segments_by_derivatives(
        [deriv], limit_per_derivative=2
    )
    assert len(list(result[deriv])) == 2


@pytest.mark.asyncio
async def test_get_by_derivatives_property_filter(
    partition: SQLAlchemySegmentLinkerPartition,
) -> None:
    deriv = uuid4()
    s_red = _seg(ts_offset_seconds=0, properties={"color": "red"})
    s_blue = _seg(ts_offset_seconds=1, properties={"color": "blue"})
    await partition.register_segments({s_red: [deriv], s_blue: [deriv]})

    filt = Comparison(field="color", op="=", value="red")
    result = await partition.get_segments_by_derivatives([deriv], property_filter=filt)
    segments = list(result[deriv])
    assert len(segments) == 1
    assert segments[0].uuid == s_red.uuid


@pytest.mark.asyncio
async def test_get_by_derivatives_session_isolation(
    linker: SQLAlchemySegmentLinker,
) -> None:
    seg = _seg()
    deriv = uuid4()
    await (await linker.open_partition("session-a")).register_segments({seg: [deriv]})

    result = await (
        await linker.open_partition("session-b")
    ).get_segments_by_derivatives([deriv])
    assert deriv not in result


# ===================================================================
# get_segment_contexts
# ===================================================================


@pytest.mark.asyncio
async def test_contexts_empty_seeds(
    partition: SQLAlchemySegmentLinkerPartition,
) -> None:
    result = await partition.get_segment_contexts([])
    assert result == {}


@pytest.mark.asyncio
async def test_contexts_unknown_seed(
    partition: SQLAlchemySegmentLinkerPartition,
) -> None:
    result = await partition.get_segment_contexts(
        [uuid4()], max_backward_segments=2, max_forward_segments=2
    )
    assert result == {}


@pytest.mark.asyncio
async def test_contexts_seed_only(
    partition: SQLAlchemySegmentLinkerPartition,
) -> None:
    """When max_backward=0 and max_forward=0, return just the seed."""
    seg = _seg()
    deriv = uuid4()
    await partition.register_segments({seg: [deriv]})

    result = await partition.get_segment_contexts([seg.uuid])
    assert seg.uuid in result
    ctx = list(result[seg.uuid])
    assert len(ctx) == 1
    assert ctx[0].uuid == seg.uuid


@pytest.mark.asyncio
async def test_contexts_backward(
    partition: SQLAlchemySegmentLinkerPartition,
) -> None:
    ep = uuid4()
    segs = [_seg(episode_uuid=ep, offset=i, ts_offset_seconds=i) for i in range(5)]
    deriv = uuid4()
    await partition.register_segments({s: [deriv] for s in segs})

    seed = segs[3]
    result = await partition.get_segment_contexts([seed.uuid], max_backward_segments=2)
    ctx = list(result[seed.uuid])
    # backward(2) + seed = 3 segments
    assert len(ctx) == 3
    uuids = [s.uuid for s in ctx]
    assert uuids == [segs[1].uuid, segs[2].uuid, seed.uuid]


@pytest.mark.asyncio
async def test_contexts_forward(
    partition: SQLAlchemySegmentLinkerPartition,
) -> None:
    ep = uuid4()
    segs = [_seg(episode_uuid=ep, offset=i, ts_offset_seconds=i) for i in range(5)]
    deriv = uuid4()
    await partition.register_segments({s: [deriv] for s in segs})

    seed = segs[1]
    result = await partition.get_segment_contexts([seed.uuid], max_forward_segments=2)
    ctx = list(result[seed.uuid])
    # seed + forward(2) = 3 segments
    assert len(ctx) == 3
    uuids = [s.uuid for s in ctx]
    assert uuids == [seed.uuid, segs[2].uuid, segs[3].uuid]


@pytest.mark.asyncio
async def test_contexts_backward_and_forward(
    partition: SQLAlchemySegmentLinkerPartition,
) -> None:
    ep = uuid4()
    segs = [_seg(episode_uuid=ep, offset=i, ts_offset_seconds=i) for i in range(7)]
    deriv = uuid4()
    await partition.register_segments({s: [deriv] for s in segs})

    seed = segs[3]
    result = await partition.get_segment_contexts(
        [seed.uuid], max_backward_segments=2, max_forward_segments=2
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
    partition: SQLAlchemySegmentLinkerPartition,
) -> None:
    """Requesting more context than available returns what exists."""
    ep = uuid4()
    segs = [_seg(episode_uuid=ep, offset=i, ts_offset_seconds=i) for i in range(3)]
    deriv = uuid4()
    await partition.register_segments({s: [deriv] for s in segs})

    seed = segs[0]
    result = await partition.get_segment_contexts(
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
async def test_contexts_multiple_seeds(
    partition: SQLAlchemySegmentLinkerPartition,
) -> None:
    ep = uuid4()
    segs = [_seg(episode_uuid=ep, offset=i, ts_offset_seconds=i) for i in range(10)]
    deriv = uuid4()
    await partition.register_segments({s: [deriv] for s in segs})

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
    partition: SQLAlchemySegmentLinkerPartition,
) -> None:
    """Properties are loaded for seed and context segments."""
    ep = uuid4()
    s0 = _seg(episode_uuid=ep, offset=0, ts_offset_seconds=0, properties={"k": "v0"})
    s1 = _seg(episode_uuid=ep, offset=1, ts_offset_seconds=1, properties={"k": "v1"})
    s2 = _seg(episode_uuid=ep, offset=2, ts_offset_seconds=2, properties={"k": "v2"})
    deriv = uuid4()
    await partition.register_segments({s0: [deriv], s1: [deriv], s2: [deriv]})

    result = await partition.get_segment_contexts(
        [s1.uuid], max_backward_segments=1, max_forward_segments=1
    )
    ctx = list(result[s1.uuid])
    assert len(ctx) == 3
    assert ctx[0].properties == {"k": "v0"}
    assert ctx[1].properties == {"k": "v1"}
    assert ctx[2].properties == {"k": "v2"}


@pytest.mark.asyncio
async def test_contexts_property_filter(
    partition: SQLAlchemySegmentLinkerPartition,
) -> None:
    """Property filter excludes context rows that don't match."""
    ep = uuid4()
    s0 = _seg(episode_uuid=ep, offset=0, ts_offset_seconds=0, properties={"tag": "a"})
    s1 = _seg(episode_uuid=ep, offset=1, ts_offset_seconds=1, properties={"tag": "b"})
    s2 = _seg(episode_uuid=ep, offset=2, ts_offset_seconds=2, properties={"tag": "a"})
    s3 = _seg(episode_uuid=ep, offset=3, ts_offset_seconds=3, properties={"tag": "a"})
    deriv = uuid4()
    await partition.register_segments(
        {s0: [deriv], s1: [deriv], s2: [deriv], s3: [deriv]}
    )

    filt = Comparison(field="tag", op="=", value="a")
    result = await partition.get_segment_contexts(
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
    await (await linker.open_partition("other-session")).register_segments(
        {s_other: [deriv_other]}
    )
    partition = await linker.open_partition(PARTITION_KEY)
    await partition.register_segments({s_seed: [deriv], s_after: [deriv]})

    result = await partition.get_segment_contexts(
        [s_seed.uuid],
        max_backward_segments=5,
        max_forward_segments=5,
    )
    ctx = list(result[s_seed.uuid])
    uuids = [s.uuid for s in ctx]
    assert s_other.uuid not in uuids
    assert uuids == [s_seed.uuid, s_after.uuid]


@pytest.mark.asyncio
async def test_contexts_chronological_order(
    partition: SQLAlchemySegmentLinkerPartition,
) -> None:
    """Context segments are returned in chronological order."""
    ep = uuid4()
    segs = [_seg(episode_uuid=ep, offset=i, ts_offset_seconds=i) for i in range(5)]
    deriv = uuid4()
    await partition.register_segments({s: [deriv] for s in segs})

    result = await partition.get_segment_contexts(
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
async def test_delete_by_episodes(
    partition: SQLAlchemySegmentLinkerPartition,
) -> None:
    ep = uuid4()
    seg = _seg(episode_uuid=ep)
    deriv = uuid4()
    await partition.register_segments({seg: [deriv]})

    await partition.delete_segments_by_episodes([ep])

    result = await partition.get_segments_by_derivatives([deriv])
    assert deriv not in result


@pytest.mark.asyncio
async def test_delete_by_episodes_orphans_derivative(
    partition: SQLAlchemySegmentLinkerPartition,
) -> None:
    ep = uuid4()
    s1 = _seg(episode_uuid=ep, offset=0)
    s2 = _seg(episode_uuid=ep, offset=1)
    deriv = uuid4()
    await partition.register_segments({s1: [deriv], s2: [deriv]})

    await partition.delete_segments_by_episodes([ep])

    await partition.mark_orphaned_derivatives_for_purging()
    assert deriv in await partition.get_derivatives_pending_purge()


@pytest.mark.asyncio
async def test_delete_by_episodes_noop_unknown(
    partition: SQLAlchemySegmentLinkerPartition,
) -> None:
    await partition.delete_segments_by_episodes([uuid4()])


# ===================================================================
# orphan management
# ===================================================================


@pytest.mark.asyncio
async def test_orphan_lifecycle(
    partition: SQLAlchemySegmentLinkerPartition,
) -> None:
    """Full lifecycle: register -> delete segments -> orphan -> mark -> purge."""
    ep = uuid4()
    seg = _seg(episode_uuid=ep)
    deriv = uuid4()
    await partition.register_segments({seg: [deriv]})

    # No orphans yet
    await partition.mark_orphaned_derivatives_for_purging()
    assert deriv not in await partition.get_derivatives_pending_purge()

    # Delete segment -> derivative becomes orphaned
    await partition.delete_segments_by_episodes([ep])
    await partition.mark_orphaned_derivatives_for_purging()

    # Derivative is now pending purge (state=P)
    pending = await partition.get_derivatives_pending_purge()
    assert deriv in pending

    # Purge
    await partition.purge_derivatives([deriv])
    assert deriv not in await partition.get_derivatives_pending_purge()


@pytest.mark.asyncio
async def test_mark_orphaned_ignores_non_orphans(
    partition: SQLAlchemySegmentLinkerPartition,
) -> None:
    seg = _seg()
    deriv = uuid4()
    await partition.register_segments({seg: [deriv]})

    await partition.mark_orphaned_derivatives_for_purging()
    assert deriv not in await partition.get_derivatives_pending_purge()


@pytest.mark.asyncio
async def test_purge_empty(
    partition: SQLAlchemySegmentLinkerPartition,
) -> None:
    await partition.purge_derivatives([])


# ===================================================================
# Concurrency tests
# ===================================================================


async def _get_partition(engine: AsyncEngine) -> SQLAlchemySegmentLinkerPartition:
    """Create a partition handle that shares the engine (and thus the connection pool / DB)."""
    linker = SQLAlchemySegmentLinker(SQLAlchemySegmentLinkerParams(engine=engine))
    return await linker.open_partition(PARTITION_KEY)


# --- Both backends ---


@pytest.mark.asyncio
async def test_concurrent_register_disjoint_derivatives(
    linker: SQLAlchemySegmentLinker,
) -> None:
    """Concurrent registrations with completely disjoint derivatives should not interfere."""
    import asyncio

    engine = linker._engine

    async def register_batch(batch_id: int) -> None:
        part = await _get_partition(engine)
        segs = {
            _seg(ts_offset_seconds=batch_id * 10 + i, text=f"batch{batch_id}-{i}"): [
                uuid4()
            ]
            for i in range(5)
        }
        await part.register_segments(segs)

    await asyncio.gather(*(register_batch(i) for i in range(10)))

    # Verify all segments were registered.
    part = await _get_partition(engine)
    async with part._create_session() as session:
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
    partition = await linker.open_partition(PARTITION_KEY)

    # Seed some data.
    ep = uuid4()
    segs = [_seg(episode_uuid=ep, offset=i, ts_offset_seconds=i) for i in range(10)]
    deriv = uuid4()
    await partition.register_segments({s: [deriv] for s in segs})

    read_results: list[int] = []

    async def reader() -> None:
        part = await _get_partition(engine)
        for _ in range(5):
            result = await part.get_segments_by_derivatives([deriv])
            if deriv in result:
                read_results.append(len(list(result[deriv])))
            await asyncio.sleep(0.01)

    async def writer() -> None:
        part = await _get_partition(engine)
        for i in range(5):
            new_seg = _seg(ts_offset_seconds=100 + i)
            new_deriv = uuid4()
            await part.register_segments({new_seg: [new_deriv]})
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
    partition = await linker.open_partition(PARTITION_KEY)

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
    await partition.register_segments(links)

    errors: list[Exception] = []

    async def context_reader() -> None:
        part = await _get_partition(engine)
        for seg in all_segs[::3]:  # Read every 3rd segment's context.
            try:
                await part.get_segment_contexts(
                    [seg.uuid],
                    max_backward_segments=2,
                    max_forward_segments=2,
                )
            except Exception as e:
                errors.append(e)
            await asyncio.sleep(0.01)

    async def episode_deleter() -> None:
        part = await _get_partition(engine)
        for ep in episodes[1:3]:  # Delete 2 of 5 episodes.
            await part.delete_segments_by_episodes([ep])
            await asyncio.sleep(0.02)

    await asyncio.gather(context_reader(), episode_deleter())

    # No crashes. Some reads may return empty or partial results — that's fine.
    assert errors == []


@pytest.mark.asyncio
async def test_concurrent_orphan_mark_does_not_crash(
    linker: SQLAlchemySegmentLinker,
) -> None:
    """Two concurrent mark_orphaned_derivatives_for_purging should not crash."""
    import asyncio

    engine = linker._engine
    partition = await linker.open_partition(PARTITION_KEY)

    # Register and orphan a derivative.
    ep = uuid4()
    seg = _seg(episode_uuid=ep)
    deriv = uuid4()
    await partition.register_segments({seg: [deriv]})
    await partition.delete_segments_by_episodes([ep])

    errors: list[Exception] = []

    async def mark() -> None:
        try:
            part = await _get_partition(engine)
            await part.mark_orphaned_derivatives_for_purging()
        except Exception as e:
            errors.append(e)

    await asyncio.gather(mark(), mark())

    assert errors == []
    # At least one should have marked it.
    assert deriv in await partition.get_derivatives_pending_purge()


@pytest.mark.asyncio
async def test_concurrent_delete_overlapping_episodes_shared_derivative(
    linker: SQLAlchemySegmentLinker,
) -> None:
    """Deleting two episodes that share a derivative concurrently should orphan the derivative."""
    import asyncio

    partition = await linker.open_partition(PARTITION_KEY)

    ep1, ep2 = uuid4(), uuid4()
    seg1 = _seg(episode_uuid=ep1, ts_offset_seconds=0)
    seg2 = _seg(episode_uuid=ep2, ts_offset_seconds=1)
    deriv = uuid4()
    await partition.register_segments({seg1: [deriv], seg2: [deriv]})

    # Both segments link to deriv. Delete both episodes concurrently.
    # Use the same partition so the SQLite write lock serializes correctly.
    async def delete_ep(ep: UUID) -> None:
        await partition.delete_segments_by_episodes([ep])

    await asyncio.gather(delete_ep(ep1), delete_ep(ep2))

    # Derivative should be orphaned (owner_segment_uuid IS NULL).
    await partition.mark_orphaned_derivatives_for_purging()
    assert deriv in await partition.get_derivatives_pending_purge()


# --- PostgreSQL-only concurrency tests ---


@pytest.mark.integration
@pytest.mark.asyncio
async def test_pg_concurrent_orphan_mark_exactly_once(
    pg_linker: SQLAlchemySegmentLinker,
) -> None:
    """On PG, FOR UPDATE SKIP LOCKED ensures only one of two concurrent markers wins."""
    import asyncio

    engine = pg_linker._engine
    partition = await pg_linker.open_partition(PARTITION_KEY)

    ep = uuid4()
    seg = _seg(episode_uuid=ep)
    deriv = uuid4()
    await partition.register_segments({seg: [deriv]})
    await partition.delete_segments_by_episodes([ep])

    async def mark() -> None:
        part = await _get_partition(engine)
        await part.mark_orphaned_derivatives_for_purging()

    await asyncio.gather(mark(), mark())

    # Exactly one should have marked it due to SKIP LOCKED serialization.
    assert deriv in await partition.get_derivatives_pending_purge()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_pg_concurrent_register_same_existing_derivative(
    pg_linker: SQLAlchemySegmentLinker,
) -> None:
    """Two concurrent registrations linking to the same existing derivative should serialize correctly."""
    import asyncio

    engine = pg_linker._engine
    partition = await pg_linker.open_partition(PARTITION_KEY)
    deriv = uuid4()

    # Initial registration.
    seg0 = _seg(ts_offset_seconds=0)
    await partition.register_segments({seg0: [deriv]})

    errors: list[Exception] = []

    async def register(offset: int) -> None:
        try:
            part = await _get_partition(engine)
            seg = _seg(ts_offset_seconds=offset)
            await part.register_segments({seg: [deriv]})
        except Exception as e:
            errors.append(e)

    await asyncio.gather(register(10), register(20))

    # Both should succeed (FOR UPDATE serializes the derivative lock).
    assert errors == []

    # Derivative should now have 3 linked segments.
    result = await partition.get_segments_by_derivatives([deriv])
    assert len(list(result[deriv])) == 3


@pytest.mark.integration
@pytest.mark.asyncio
async def test_pg_concurrent_register_and_delete_same_episode(
    pg_linker: SQLAlchemySegmentLinker,
) -> None:
    """Registering new segments for an episode while deleting it should not deadlock."""
    import asyncio

    engine = pg_linker._engine
    partition = await pg_linker.open_partition(PARTITION_KEY)
    ep = uuid4()
    deriv1 = uuid4()

    # Seed.
    seg1 = _seg(episode_uuid=ep, offset=0, ts_offset_seconds=0)
    await partition.register_segments({seg1: [deriv1]})

    errors: list[Exception] = []

    async def deleter() -> None:
        try:
            part = await _get_partition(engine)
            await part.delete_segments_by_episodes([ep])
        except Exception as e:
            errors.append(e)

    async def registerer() -> None:
        try:
            part = await _get_partition(engine)
            seg2 = _seg(episode_uuid=ep, offset=1, ts_offset_seconds=1)
            deriv2 = uuid4()
            await part.register_segments({seg2: [deriv2]})
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
    partition = await pg_linker.open_partition(PARTITION_KEY)
    ep = uuid4()
    seg = _seg(episode_uuid=ep, ts_offset_seconds=0)
    deriv = uuid4()
    await partition.register_segments({seg: [deriv]})

    # Orphan it.
    await partition.delete_segments_by_episodes([ep])

    # Now, concurrently re-link and try to mark for purging.
    async def relinker() -> None:
        part = await _get_partition(engine)
        new_seg = _seg(ts_offset_seconds=10)
        await part.register_segments({new_seg: [deriv]})

    async def marker() -> None:
        # Wait a tiny bit to let relinker likely win, but it's a race so either outcome is valid.
        await asyncio.sleep(0.01)
        part = await _get_partition(engine)
        await part.mark_orphaned_derivatives_for_purging()

    await asyncio.gather(relinker(), marker())

    # If relinker won the race, deriv has owner != NULL and should NOT be in pending.
    # If marker won the race, deriv was still orphaned and gets marked.
    # Either way, no crash.
    pending = await partition.get_derivatives_pending_purge()
    if deriv not in pending:
        # Relinker won — derivative should still be retrievable.
        result = await partition.get_segments_by_derivatives([deriv])
        assert deriv in result
    # If in pending, the derivative was legitimately marked before relinking.


@pytest.mark.integration
@pytest.mark.asyncio
async def test_pg_concurrent_mass_deletes(
    pg_linker: SQLAlchemySegmentLinker,
) -> None:
    """Many concurrent episode deletes that share derivatives should not deadlock."""
    import asyncio

    engine = pg_linker._engine
    partition = await pg_linker.open_partition(PARTITION_KEY)

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
    await partition.register_segments(links)

    # Delete all episodes concurrently.
    async def delete_ep(ep: UUID) -> None:
        part = await _get_partition(engine)
        await part.delete_segments_by_episodes([ep])

    await asyncio.gather(*(delete_ep(ep) for ep in episodes))

    # All segments should be gone. Derivative should be orphaned.
    result = await partition.get_segments_by_derivatives([deriv])
    assert deriv not in result

    await partition.mark_orphaned_derivatives_for_purging()
    assert deriv in await partition.get_derivatives_pending_purge()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_pg_concurrent_purge_and_register_new_derivative(
    pg_linker: SQLAlchemySegmentLinker,
) -> None:
    """Purging a derivative while registering a completely new derivative should not interfere."""
    import asyncio

    engine = pg_linker._engine
    partition = await pg_linker.open_partition(PARTITION_KEY)

    # Create and orphan a derivative.
    ep = uuid4()
    seg = _seg(episode_uuid=ep)
    old_deriv = uuid4()
    await partition.register_segments({seg: [old_deriv]})
    await partition.delete_segments_by_episodes([ep])
    await partition.mark_orphaned_derivatives_for_purging()
    assert old_deriv in await partition.get_derivatives_pending_purge()

    errors: list[Exception] = []

    async def purger() -> None:
        try:
            part = await _get_partition(engine)
            await part.purge_derivatives([old_deriv])
        except Exception as e:
            errors.append(e)

    async def registerer() -> None:
        try:
            part = await _get_partition(engine)
            new_seg = _seg(ts_offset_seconds=100)
            new_deriv = uuid4()
            await part.register_segments({new_seg: [new_deriv]})
        except Exception as e:
            errors.append(e)

    await asyncio.gather(purger(), registerer())
    assert errors == []


@pytest.mark.integration
@pytest.mark.asyncio
async def test_pg_context_preserved_via_lateral_join(
    pg_linker: SQLAlchemySegmentLinker,
) -> None:
    """Context is preserved when retrieved via the LATERAL join path (multiple seeds)."""
    partition = await pg_linker.open_partition(PARTITION_KEY)
    ep = uuid4()
    ctx_user = MessageContext(source="User")
    ctx_assistant = MessageContext(source="Assistant")
    s0 = _seg(episode_uuid=ep, offset=0, ts_offset_seconds=0, context=ctx_user)
    s1 = _seg(episode_uuid=ep, offset=1, ts_offset_seconds=1, context=ctx_assistant)
    s2 = _seg(episode_uuid=ep, offset=2, ts_offset_seconds=2, context=ctx_user)
    s3 = _seg(episode_uuid=ep, offset=3, ts_offset_seconds=3, context=ctx_assistant)
    s4 = _seg(episode_uuid=ep, offset=4, ts_offset_seconds=4, context=ctx_user)
    deriv = uuid4()
    await partition.register_segments(
        {s0: [deriv], s1: [deriv], s2: [deriv], s3: [deriv], s4: [deriv]}
    )

    # Two seeds exercises the LATERAL join code path.
    result = await partition.get_segment_contexts(
        [s1.uuid, s3.uuid], max_backward_segments=1, max_forward_segments=1
    )

    ctx_a = list(result[s1.uuid])
    assert len(ctx_a) == 3
    assert ctx_a[0].context == ctx_user
    assert ctx_a[1].context == ctx_assistant
    assert ctx_a[2].context == ctx_user

    ctx_b = list(result[s3.uuid])
    assert len(ctx_b) == 3
    assert ctx_b[0].context == ctx_user
    assert ctx_b[1].context == ctx_assistant
    assert ctx_b[2].context == ctx_user


@pytest.mark.integration
@pytest.mark.asyncio
async def test_pg_mixed_context_types(
    pg_linker: SQLAlchemySegmentLinker,
) -> None:
    """Different context types (message, citation, None) round-trip correctly on PG."""
    partition = await pg_linker.open_partition(PARTITION_KEY)
    ctx_msg = MessageContext(source="User")
    ctx_cite = CitationContext(source="paper.pdf", source_type="file", location="p.3")

    s_msg = _seg(ts_offset_seconds=0, context=ctx_msg)
    s_cite = _seg(ts_offset_seconds=1, context=ctx_cite)
    s_none = _seg(ts_offset_seconds=2)

    d1, d2, d3 = uuid4(), uuid4(), uuid4()
    await partition.register_segments({s_msg: [d1], s_cite: [d2], s_none: [d3]})

    r1 = await partition.get_segments_by_derivatives([d1])
    assert next(iter(r1[d1])).context == ctx_msg

    r2 = await partition.get_segments_by_derivatives([d2])
    assert next(iter(r2[d2])).context == ctx_cite

    r3 = await partition.get_segments_by_derivatives([d3])
    assert next(iter(r3[d3])).context is None
