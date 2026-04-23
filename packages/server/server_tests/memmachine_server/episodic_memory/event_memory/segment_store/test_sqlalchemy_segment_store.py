"""Tests for SQLAlchemySegmentStore — SQLite (unit) and PostgreSQL (integration)."""

import asyncio
import json
from collections.abc import AsyncIterator
from datetime import UTC, datetime, timedelta
from typing import override
from uuid import UUID, uuid4

import pytest
import pytest_asyncio
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncEngine

from memmachine_server.common.filter.filter_parser import Comparison
from memmachine_server.common.payload_codec import (
    KMSEnvelopePayloadCodecConfig,
    KMSEnvelopePayloadCodecLoader,
    PayloadCodec,
)
from memmachine_server.common.payload_codec.payload_codec_config import (
    AESGCMPayloadCodecConfig,
    PlaintextPayloadCodecConfig,
)
from memmachine_server.episodic_memory.event_memory.data_types import (
    CitationContext,
    MessageContext,
    NullContext,
    Segment,
    Text,
)
from memmachine_server.episodic_memory.event_memory.segment_store import (
    SegmentStorePartitionAlreadyExistsError,
    SegmentStorePartitionConfig,
)
from memmachine_server.episodic_memory.event_memory.segment_store.sqlalchemy_segment_store import (
    BaseSegmentStore,
    SegmentRow,
    SQLAlchemySegmentStore,
    SQLAlchemySegmentStoreParams,
    SQLAlchemySegmentStorePartition,
)

PARTITION_KEY = "test_partition"
BASE_TIME = datetime(2024, 1, 1, tzinfo=UTC)
_NULL_CONTEXT = NullContext()


class PrefixPayloadCodec(PayloadCodec):
    """Codec that prefixes payloads so loader wiring is easy to assert."""

    def __init__(self, prefix: bytes = b"prefix:") -> None:
        self._prefix = prefix

    @override
    def encode(self, value: bytes) -> bytes:
        return self._prefix + value

    @override
    def decode(self, value: bytes) -> bytes:
        if not value.startswith(self._prefix):
            raise ValueError("encoded payload is missing the expected prefix")
        return value[len(self._prefix) :]


class PrefixPayloadCodecLoader(KMSEnvelopePayloadCodecLoader):
    """Loader that returns the prefix codec and records configs it receives."""

    def __init__(self) -> None:
        self.loaded_configs: list[KMSEnvelopePayloadCodecConfig] = []

    @override
    async def load(self, config: KMSEnvelopePayloadCodecConfig) -> PayloadCodec:
        self.loaded_configs.append(config)
        return PrefixPayloadCodec()


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
    context: MessageContext | CitationContext | NullContext = _NULL_CONTEXT,
    properties: dict | None = None,
) -> Segment:
    return Segment(
        uuid=uuid4(),
        event_uuid=event_uuid or uuid4(),
        index=index,
        offset=offset,
        timestamp=BASE_TIME + timedelta(seconds=ts_offset_seconds),
        block=Text(text=text),
        context=context,
        properties=properties or {},
    )


def _links(*segments: Segment) -> dict[Segment, list[UUID]]:
    """Build a segment-to-derivative-UUIDs mapping with one derivative per segment."""
    return {seg: [uuid4()] for seg in segments}


def _plaintext_partition_config() -> SegmentStorePartitionConfig:
    """Return the default plaintext partition config."""
    return SegmentStorePartitionConfig()


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
    return await store.open_or_create_partition(
        PARTITION_KEY,
        _plaintext_partition_config(),
    )


@pytest_asyncio.fixture
async def sqlite_store_with_loader(
    sqlalchemy_sqlite_engine: AsyncEngine,
) -> AsyncIterator[tuple[SQLAlchemySegmentStore, PrefixPayloadCodecLoader]]:
    loader = PrefixPayloadCodecLoader()
    store = SQLAlchemySegmentStore(
        SQLAlchemySegmentStoreParams(
            engine=sqlalchemy_sqlite_engine,
            payload_codec_loader=loader,
        )
    )
    await store.startup()
    yield store, loader
    async with sqlalchemy_sqlite_engine.begin() as conn:
        await conn.run_sync(BaseSegmentStore.metadata.drop_all)


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
async def test_add_segments_with_message_context(
    partition: SQLAlchemySegmentStorePartition,
) -> None:
    ctx = MessageContext(source="User")
    seg = _seg(context=ctx)
    await partition.add_segments(_links(seg))

    result = await partition.get_segment_contexts([seg.uuid])
    assert result[seg.uuid][0].context == ctx

    async with partition._create_session() as session:
        row = (
            await session.execute(select(SegmentRow).where(SegmentRow.uuid == seg.uuid))
        ).scalar_one()
    assert json.loads(row.context) == {"type": "message", "source": "User"}
    assert json.loads(row.block) == {"type": "text", "text": "hello"}


@pytest.mark.asyncio
async def test_add_segments_with_citation_context(
    partition: SQLAlchemySegmentStorePartition,
) -> None:
    ctx = CitationContext(source="docs.txt", source_type="file", location="/tmp")
    seg = _seg(context=ctx)
    await partition.add_segments(_links(seg))

    result = await partition.get_segment_contexts([seg.uuid])
    assert result[seg.uuid][0].context == ctx


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
    assert json.loads(row.context) == {"type": "null"}

    result = await partition.get_segment_contexts([seg.uuid])
    assert result[seg.uuid][0].context == NullContext()


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
async def test_contexts_filter_by_context_source(
    partition: SQLAlchemySegmentStorePartition,
) -> None:
    """Filter using ``context.source`` is not supported."""
    ep = uuid4()
    s0 = _seg(
        event_uuid=ep,
        offset=0,
        ts_offset_seconds=0,
        context=MessageContext(source="Alice"),
    )
    s1 = _seg(
        event_uuid=ep,
        offset=1,
        ts_offset_seconds=1,
        context=MessageContext(source="Bob"),
    )
    s2 = _seg(
        event_uuid=ep,
        offset=2,
        ts_offset_seconds=2,
        context=MessageContext(source="Alice"),
    )
    await partition.add_segments(_links(s0, s1, s2))

    filt = Comparison(field="context.source", op="=", value="Alice")
    with pytest.raises(ValueError, match="Unknown filter field"):
        await partition.get_segment_contexts(
            [s0.uuid],
            max_backward_segments=5,
            max_forward_segments=5,
            property_filter=filt,
        )


@pytest.mark.asyncio
async def test_contexts_filter_by_context_type(
    partition: SQLAlchemySegmentStorePartition,
) -> None:
    """Filter using ``context.type`` is not supported."""
    ep = uuid4()
    s0 = _seg(
        event_uuid=ep,
        offset=0,
        ts_offset_seconds=0,
        context=MessageContext(source="Alice"),
    )
    s1 = _seg(
        event_uuid=ep,
        offset=1,
        ts_offset_seconds=1,
        context=CitationContext(source="paper.pdf"),
    )
    s2 = _seg(
        event_uuid=ep,
        offset=2,
        ts_offset_seconds=2,
        context=MessageContext(source="Bob"),
    )
    await partition.add_segments(_links(s0, s1, s2))

    filt = Comparison(field="context.type", op="=", value="message")
    with pytest.raises(ValueError, match="Unknown filter field"):
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
    ctx_user = MessageContext(source="User")
    ctx_assistant = MessageContext(source="Assistant")
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
async def test_open_or_create_partition_uses_codec_loader_when_configured(
    sqlite_store_with_loader: tuple[
        SQLAlchemySegmentStore,
        PrefixPayloadCodecLoader,
    ],
) -> None:
    store, loader = sqlite_store_with_loader
    config = SegmentStorePartitionConfig(
        payload_codec_config=AESGCMPayloadCodecConfig(
            key_ref="partition_key",
            wrapped_dek=b"wrapped-dek",
            nonce_size=12,
            associated_data=b"partition:context",
        )
    )

    partition = await store.open_or_create_partition(
        "encrypted_partition",
        config=config,
    )
    assert partition.config == config

    segment = _seg(text="codec")
    await partition.add_segments(_links(segment))

    async with partition._create_session() as session:
        row = (
            await session.execute(
                select(SegmentRow).where(SegmentRow.uuid == segment.uuid)
            )
        ).scalar_one()

    assert row.context.startswith(b"prefix:")
    assert row.block.startswith(b"prefix:")
    assert all(
        isinstance(item, AESGCMPayloadCodecConfig) for item in loader.loaded_configs
    )

    reopened = await store.open_partition("encrypted_partition")
    assert reopened is not None
    result = await reopened.get_segment_contexts([segment.uuid])
    assert result[segment.uuid][0].uuid == segment.uuid


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


# --- PostgreSQL-only ---


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
    ctx_user = MessageContext(source="User")
    ctx_assistant = MessageContext(source="Assistant")
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
    """Different context types (message, citation, None) round-trip correctly on PG."""
    partition = await pg_store.open_or_create_partition(
        PARTITION_KEY,
        _plaintext_partition_config(),
    )
    ctx_msg = MessageContext(source="User")
    ctx_cite = CitationContext(source="paper.pdf", source_type="file", location="p.3")

    s_msg = _seg(ts_offset_seconds=0, context=ctx_msg)
    s_cite = _seg(ts_offset_seconds=1, context=ctx_cite)
    s_none = _seg(ts_offset_seconds=2)

    all_segs = [s_msg, s_cite, s_none]
    await partition.add_segments(_links(*all_segs))

    async with partition._create_session() as session:
        row = (
            await session.execute(
                select(SegmentRow).where(SegmentRow.uuid == s_none.uuid)
            )
        ).scalar_one()
    assert json.loads(row.context) == {"type": "null"}

    result = await partition.get_segment_contexts([s_msg.uuid])
    assert result[s_msg.uuid][0].context == ctx_msg

    result = await partition.get_segment_contexts([s_cite.uuid])
    assert result[s_cite.uuid][0].context == ctx_cite

    result = await partition.get_segment_contexts([s_none.uuid])
    assert result[s_none.uuid][0].context == NullContext()
