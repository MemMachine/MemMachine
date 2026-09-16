"""Tests for QdrantVectorStore."""

import asyncio
import math
from datetime import UTC, datetime, timedelta, timezone
from unittest.mock import MagicMock
from uuid import UUID, uuid4

import pytest
import pytest_asyncio
from qdrant_client import AsyncQdrantClient, models
from sqlalchemy.ext.asyncio import create_async_engine

from memmachine_server.common.data_types import PropertyValue
from memmachine_server.common.filter.filter_parser import (
    And,
    Comparison,
    In,
    IsNull,
    Not,
    Or,
)
from memmachine_server.common.metrics_factory import MetricsFactory
from memmachine_server.common.vector_store.data_types import (
    Record,
    VectorStorePartitionAlreadyExistsError,
)
from memmachine_server.common.vector_store.qdrant_vector_store import (
    _PAYLOAD_INCARNATION,
    QdrantVectorStore,
    QdrantVectorStoreParams,
    QdrantVectorStorePartition,
)
from server_tests.memmachine_server.common.vector_store.partition_lifecycle_contract import (
    PartitionLifecycleContract,
)

COLLECTION = "test_namespace"
NAME = "test_name"
VECTOR_DIM = 3

INDEXED_PROPERTIES: dict[str, type[PropertyValue]] = {
    "name": str,
    "age": int,
    "score": float,
    "active": bool,
    "created_at": datetime,
}


async def _get_or_create_partition(store, partition_key: str):
    """The partition, created if absent: what a session's creation does, for a test."""
    partition = await store.get_partition(partition_key)
    if partition is None:
        await store.create_partition(partition_key)
        partition = await store.get_partition(partition_key)
    assert partition is not None
    return partition


@pytest.fixture
def in_memory_qdrant_client():
    return AsyncQdrantClient(location=":memory:")


@pytest.fixture(
    params=[
        "in_memory_qdrant_client",
        pytest.param("qdrant_client", marks=pytest.mark.integration),
        pytest.param("qdrant_grpc_client", marks=pytest.mark.integration),
    ],
)
def any_qdrant_client(request):
    return request.getfixturevalue(request.param)


@pytest_asyncio.fixture
async def registry_engine(tmp_path):
    """The relational database holding the partition registry, one per test."""
    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'registry.db'}")
    yield engine
    await engine.dispose()


@pytest_asyncio.fixture
async def store(any_qdrant_client, registry_engine):
    params = QdrantVectorStoreParams(
        collection=COLLECTION,
        vector_dimensions=VECTOR_DIM,
        indexed_properties=INDEXED_PROPERTIES,
        client=any_qdrant_client,
        registry_engine=registry_engine,
    )
    s = QdrantVectorStore(params)
    await s.provision()
    await s.startup()
    yield s


@pytest_asyncio.fixture
async def collection(store):
    await store.create_partition(NAME)
    coll = await store.get_partition(NAME)
    assert coll is not None
    yield coll
    await store.delete_partition(NAME)


def _normalize(v: list[float]) -> list[float]:
    mag = math.sqrt(sum(x * x for x in v))
    return [x / mag for x in v]


_FETCH_LIMIT = 1000


async def _present_uuids(collection, record_uuids) -> list[UUID]:
    """Which of these UUIDs the collection still holds, in the order given.

    `query` is the only read and it answers with UUIDs and scores, so existence
    is all a test can observe about a record here. With no score threshold a
    probe vector in any direction lists the whole collection.
    """
    record_uuids = list(record_uuids)
    if not record_uuids:
        return []
    dimensions = VECTOR_DIM
    probe = [1.0] + [0.0] * (dimensions - 1)
    [result] = await collection.query(query_vectors=[probe], limit=_FETCH_LIMIT)
    present = {match.record_uuid for match in result.matches}
    return [uuid for uuid in record_uuids if uuid in present]


def _make_record(
    *,
    uuid: UUID | None = None,
    vector: list[float],
    properties: dict | None = None,
) -> Record:
    return Record(
        uuid=uuid or uuid4(),
        vector=vector,
        properties=properties or {},
    )


# ── Partition lifecycle ──


class TestCollectionLifecycle:
    @pytest.mark.asyncio
    async def test_create_get_delete(self, store):
        await store.create_partition("lifecycle")
        coll = await store.get_partition("lifecycle")
        assert isinstance(coll, QdrantVectorStorePartition)
        await store.delete_partition("lifecycle")

    @pytest.mark.asyncio
    async def test_open_collection_returns_qdrant_collection(self, store, collection):
        coll = await store.get_partition(NAME)
        assert isinstance(coll, QdrantVectorStorePartition)

    @pytest.mark.asyncio
    async def test_duplicate_name_raises(self, store, collection):
        with pytest.raises(VectorStorePartitionAlreadyExistsError):
            await store.create_partition(NAME)

    @pytest.mark.asyncio
    async def test_delete_nonexistent_is_idempotent(self, store):
        await store.delete_partition("nonexistent")

    @pytest.mark.asyncio
    async def test_open_or_create_creates_when_missing(self, store):
        coll = await _get_or_create_partition(store, "new")
        assert isinstance(coll, QdrantVectorStorePartition)
        await store.delete_partition("new")

    @pytest.mark.asyncio
    async def test_open_or_create_opens_when_exists(self, store):
        await store.create_partition("existing")
        coll = await _get_or_create_partition(store, "existing")
        assert isinstance(coll, QdrantVectorStorePartition)
        await store.delete_partition("existing")

    @pytest.mark.asyncio
    async def test_partitions_share_the_native_collection(self, store):
        """Every partition lives in the store's one native collection."""
        await store.create_partition("coll_a")
        await store.create_partition("coll_b")

        coll_a = await store.get_partition("coll_a")
        coll_b = await store.get_partition("coll_b")
        assert coll_a is not None
        assert coll_b is not None
        assert coll_a._collection_name == coll_b._collection_name

        await store.delete_partition("coll_a")
        await store.delete_partition("coll_b")


# ── Upsert + Query ──


class TestUpsertAndQuery:
    @pytest.mark.asyncio
    async def test_upsert_and_query_basic(self, collection):
        v1 = _normalize([1.0, 0.0, 0.0])
        v2 = _normalize([0.0, 1.0, 0.0])
        v3 = _normalize([1.0, 0.1, 0.0])

        r1 = _make_record(vector=v1, properties={"name": "a"})
        r2 = _make_record(vector=v2, properties={"name": "b"})
        r3 = _make_record(vector=v3, properties={"name": "c"})

        await collection.upsert(records=[r1, r2, r3])

        query_results = list(await collection.query(query_vectors=[v1], limit=3))
        matches = query_results[0].matches

        assert len(matches) == 3
        assert matches[0].record_uuid == r1.uuid
        assert matches[1].record_uuid == r3.uuid
        assert matches[2].record_uuid == r2.uuid
        assert (
            matches[0].cosine_similarity
            >= matches[1].cosine_similarity
            >= matches[2].cosine_similarity
        )

    @pytest.mark.asyncio
    async def test_query_with_similarity_threshold(self, collection):
        v1 = _normalize([1.0, 0.0, 0.0])
        v2 = _normalize([0.0, 1.0, 0.0])

        r1 = _make_record(vector=v1)
        r2 = _make_record(vector=v2)

        await collection.upsert(records=[r1, r2])

        query_results = list(
            await collection.query(
                query_vectors=[v1], limit=10, min_cosine_similarity=0.9
            )
        )
        matches = query_results[0].matches

        assert len(matches) == 1
        assert matches[0].record_uuid == r1.uuid

    @pytest.mark.asyncio
    async def test_query_with_limit(self, collection):
        vectors = [_normalize([1.0, float(i) * 0.01, 0.0]) for i in range(5)]
        records = [_make_record(vector=v) for v in vectors]
        await collection.upsert(records=records)

        query_results = list(
            await collection.query(query_vectors=[vectors[0]], limit=2)
        )
        assert len(query_results[0].matches) == 2

    @pytest.mark.asyncio
    async def test_a_match_is_a_uuid_and_a_score(self, collection):
        """A match names the record and how well it scored, and nothing else.

        Stored properties are filterable but never returned: this store is not
        the authority for a record's content, so a caller that wants its
        fields reads them from whatever owns them.
        """
        v1 = _normalize([1.0, 0.0, 0.0])
        r1 = _make_record(vector=v1, properties={"name": "test"})
        await collection.upsert(records=[r1])

        [result] = await collection.query(query_vectors=[v1], limit=10)
        assert len(result.matches) == 1
        assert result.matches[0].record_uuid == r1.uuid
        assert result.matches[0].cosine_similarity == pytest.approx(1.0, abs=0.01)
        assert not hasattr(result.matches[0], "record")

    @pytest.mark.asyncio
    async def test_query_batch_multiple_vectors(self, collection):
        v1 = _normalize([1.0, 0.0, 0.0])
        v2 = _normalize([0.0, 1.0, 0.0])

        r1 = _make_record(vector=v1, properties={"name": "a"})
        r2 = _make_record(vector=v2, properties={"name": "b"})
        await collection.upsert(records=[r1, r2])

        all_results = list(await collection.query(query_vectors=[v1, v2], limit=1))

        assert len(all_results) == 2
        assert all_results[0].matches[0].record_uuid == r1.uuid
        assert all_results[1].matches[0].record_uuid == r2.uuid

    @pytest.mark.asyncio
    async def test_query_empty_vectors(self, collection):
        all_results = list(await collection.query(query_vectors=[], limit=10))
        assert len(all_results) == 0


# ── Filters ──


class TestFilters:
    # alice=30/9.5/True, bob=25/7.0/False, carol=35/8.0/True
    async def _setup(self, collection):
        v1 = _normalize([1.0, 0.0, 0.0])
        v2 = _normalize([1.0, 0.1, 0.0])
        v3 = _normalize([1.0, 0.2, 0.0])
        r1 = _make_record(
            vector=v1,
            properties={"name": "alice", "age": 30, "score": 9.5, "active": True},
        )
        r2 = _make_record(
            vector=v2,
            properties={"name": "bob", "age": 25, "score": 7.0, "active": False},
        )
        r3 = _make_record(
            vector=v3,
            properties={"name": "carol", "age": 35, "score": 8.0, "active": True},
        )
        await collection.upsert(records=[r1, r2, r3])
        return r1, r2, r3, v1

    # scores: r0=-1.5, r1=0.0, r2=0.5, r3=1.5, r4=2.0
    async def _setup_floats(self, collection):
        vectors = [_normalize([1.0, float(i) * 0.01, 0.0]) for i in range(5)]
        scores = [-1.5, 0.0, 0.5, 1.5, 2.0]
        records = [
            _make_record(vector=v, properties={"score": s})
            for v, s in zip(vectors, scores, strict=True)
        ]
        await collection.upsert(records=records)
        return records, vectors[0]

    # dts: r0=Jan, r1=Mar, r2=Jun, r3=Sep, r4=Dec
    async def _setup_datetimes(self, collection):
        vectors = [_normalize([1.0, float(i) * 0.01, 0.0]) for i in range(5)]
        dts = [
            datetime(2024, 1, 1, tzinfo=UTC),
            datetime(2024, 3, 15, tzinfo=UTC),
            datetime(2024, 6, 1, tzinfo=UTC),
            datetime(2024, 9, 1, tzinfo=UTC),
            datetime(2024, 12, 31, tzinfo=UTC),
        ]
        records = [
            _make_record(vector=v, properties={"name": f"r{i}", "created_at": dt})
            for i, (v, dt) in enumerate(zip(vectors, dts, strict=True))
        ]
        await collection.upsert(records=records)
        return records, vectors[0], dts

    async def _query(self, collection, query_vec, field, op, value):
        all_results = list(
            await collection.query(
                query_vectors=[query_vec],
                limit=10,
                property_filter=Comparison(field=field, op=op, value=value),
            )
        )
        return {m.record_uuid for m in all_results[0].matches}

    # ── String / int ──

    @pytest.mark.asyncio
    async def test_eq_str(self, collection):
        r1, _r2, _r3, v1 = await self._setup(collection)
        query_results = list(
            await collection.query(
                query_vectors=[v1],
                limit=10,
                property_filter=Comparison(field="name", op="=", value="alice"),
            )
        )
        matches = query_results[0].matches
        assert len(matches) == 1
        assert matches[0].record_uuid == r1.uuid

    @pytest.mark.asyncio
    async def test_ne_str(self, collection):
        r1, r2, r3, v1 = await self._setup(collection)
        uuids = await self._query(collection, v1, "name", "!=", "alice")
        assert r1.uuid not in uuids
        assert r2.uuid in uuids
        assert r3.uuid in uuids

    @pytest.mark.asyncio
    async def test_gt_int(self, collection):
        _r1, _r2, r3, v1 = await self._setup(collection)
        query_results = list(
            await collection.query(
                query_vectors=[v1],
                limit=10,
                property_filter=Comparison(field="age", op=">", value=30),
            )
        )
        matches = query_results[0].matches
        assert len(matches) == 1
        assert matches[0].record_uuid == r3.uuid

    @pytest.mark.asyncio
    async def test_gte_int(self, collection):
        r1, _r2, r3, v1 = await self._setup(collection)
        uuids = await self._query(collection, v1, "age", ">=", 30)
        assert r1.uuid in uuids
        assert r3.uuid in uuids
        assert len(uuids) == 2

    @pytest.mark.asyncio
    async def test_lt_int(self, collection):
        _r1, r2, _r3, v1 = await self._setup(collection)
        query_results = list(
            await collection.query(
                query_vectors=[v1],
                limit=10,
                property_filter=Comparison(field="age", op="<", value=30),
            )
        )
        matches = query_results[0].matches
        assert len(matches) == 1
        assert matches[0].record_uuid == r2.uuid

    @pytest.mark.asyncio
    async def test_lte_int(self, collection):
        r1, r2, _r3, v1 = await self._setup(collection)
        uuids = await self._query(collection, v1, "age", "<=", 30)
        assert r1.uuid in uuids
        assert r2.uuid in uuids
        assert len(uuids) == 2

    # ── Bool ──

    @pytest.mark.asyncio
    async def test_eq_bool(self, collection):
        r1, r2, r3, v1 = await self._setup(collection)
        uuids = await self._query(collection, v1, "active", "=", True)
        assert r1.uuid in uuids
        assert r3.uuid in uuids
        assert r2.uuid not in uuids

    @pytest.mark.asyncio
    async def test_ne_bool(self, collection):
        r1, r2, r3, v1 = await self._setup(collection)
        uuids = await self._query(collection, v1, "active", "!=", True)
        assert r2.uuid in uuids
        assert r1.uuid not in uuids
        assert r3.uuid not in uuids

    # ── Float ──

    @pytest.mark.asyncio
    async def test_eq_float_fractional(self, collection):
        records, qv = await self._setup_floats(collection)
        uuids = await self._query(collection, qv, "score", "=", 0.5)
        assert records[2].uuid in uuids
        assert len(uuids) == 1

    @pytest.mark.asyncio
    async def test_eq_float_whole_number(self, collection):
        records, qv = await self._setup_floats(collection)
        uuids = await self._query(collection, qv, "score", "=", 2.0)
        assert records[4].uuid in uuids
        assert len(uuids) == 1

    @pytest.mark.asyncio
    async def test_eq_float_negative(self, collection):
        records, qv = await self._setup_floats(collection)
        uuids = await self._query(collection, qv, "score", "=", -1.5)
        assert records[0].uuid in uuids
        assert len(uuids) == 1

    @pytest.mark.asyncio
    async def test_eq_float_zero(self, collection):
        records, qv = await self._setup_floats(collection)
        uuids = await self._query(collection, qv, "score", "=", 0.0)
        assert records[1].uuid in uuids
        assert len(uuids) == 1

    @pytest.mark.asyncio
    async def test_ne_float_fractional(self, collection):
        records, qv = await self._setup_floats(collection)
        uuids = await self._query(collection, qv, "score", "!=", 0.5)
        assert records[2].uuid not in uuids
        assert len(uuids) == 4

    @pytest.mark.asyncio
    async def test_ne_float_whole_number(self, collection):
        records, qv = await self._setup_floats(collection)
        uuids = await self._query(collection, qv, "score", "!=", 2.0)
        assert records[4].uuid not in uuids
        assert len(uuids) == 4

    @pytest.mark.asyncio
    async def test_ne_float_negative(self, collection):
        records, qv = await self._setup_floats(collection)
        uuids = await self._query(collection, qv, "score", "!=", -1.5)
        assert records[0].uuid not in uuids
        assert len(uuids) == 4

    @pytest.mark.asyncio
    async def test_gt_float(self, collection):
        records, qv = await self._setup_floats(collection)
        uuids = await self._query(collection, qv, "score", ">", 0.5)
        assert records[0].uuid not in uuids  # -1.5
        assert records[1].uuid not in uuids  # 0.0
        assert records[2].uuid not in uuids  # 0.5 not strictly greater
        assert records[3].uuid in uuids  # 1.5
        assert records[4].uuid in uuids  # 2.0

    @pytest.mark.asyncio
    async def test_gte_float(self, collection):
        records, qv = await self._setup_floats(collection)
        uuids = await self._query(collection, qv, "score", ">=", 0.5)
        assert records[0].uuid not in uuids  # -1.5
        assert records[1].uuid not in uuids  # 0.0
        assert records[2].uuid in uuids  # 0.5
        assert records[3].uuid in uuids  # 1.5
        assert records[4].uuid in uuids  # 2.0

    @pytest.mark.asyncio
    async def test_lt_float(self, collection):
        records, qv = await self._setup_floats(collection)
        uuids = await self._query(collection, qv, "score", "<", 0.5)
        assert records[0].uuid in uuids  # -1.5
        assert records[1].uuid in uuids  # 0.0
        assert records[2].uuid not in uuids  # 0.5 not strictly less
        assert records[3].uuid not in uuids  # 1.5
        assert records[4].uuid not in uuids  # 2.0

    @pytest.mark.asyncio
    async def test_lte_float(self, collection):
        records, qv = await self._setup_floats(collection)
        uuids = await self._query(collection, qv, "score", "<=", 0.5)
        assert records[0].uuid in uuids  # -1.5
        assert records[1].uuid in uuids  # 0.0
        assert records[2].uuid in uuids  # 0.5
        assert records[3].uuid not in uuids  # 1.5
        assert records[4].uuid not in uuids  # 2.0

    @pytest.mark.asyncio
    async def test_gt_float_from_negative(self, collection):
        records, qv = await self._setup_floats(collection)
        uuids = await self._query(collection, qv, "score", ">", -1.5)
        assert records[0].uuid not in uuids  # -1.5 not strictly greater
        assert records[1].uuid in uuids  # 0.0
        assert records[2].uuid in uuids  # 0.5
        assert records[3].uuid in uuids  # 1.5
        assert records[4].uuid in uuids  # 2.0

    @pytest.mark.asyncio
    async def test_lt_float_zero(self, collection):
        records, qv = await self._setup_floats(collection)
        uuids = await self._query(collection, qv, "score", "<", 0.0)
        assert records[0].uuid in uuids  # -1.5
        assert records[1].uuid not in uuids  # 0.0 not strictly less
        assert records[2].uuid not in uuids
        assert records[3].uuid not in uuids
        assert records[4].uuid not in uuids

    # ── Datetime ──

    @pytest.mark.asyncio
    async def test_datetime_roundtrip(self, collection):
        v1 = _normalize([1.0, 0.0, 0.0])
        dt = datetime(2024, 6, 15, 12, 30, 0, tzinfo=UTC)
        r1 = _make_record(vector=v1, properties={"name": "test", "created_at": dt})
        await collection.upsert(records=[r1])

        # Properties are filterable but never returned, so a filter is what
        # observes them.
        [result] = await collection.query(
            query_vectors=[v1],
            limit=10,
            property_filter=Comparison(field="created_at", op="=", value=dt),
        )
        assert [m.record_uuid for m in result.matches] == [r1.uuid]

    @pytest.mark.asyncio
    async def test_datetime_microseconds_roundtrip(self, collection):
        """Microsecond precision is preserved."""
        v1 = _normalize([1.0, 0.0, 0.0])
        dt = datetime(2024, 6, 15, 12, 30, 45, 123456, tzinfo=UTC)
        r1 = _make_record(vector=v1, properties={"name": "micro", "created_at": dt})
        await collection.upsert(records=[r1])

        # Properties are filterable but never returned, so a filter is what
        # observes them.
        [result] = await collection.query(
            query_vectors=[v1],
            limit=10,
            property_filter=Comparison(field="created_at", op="=", value=dt),
        )
        assert [m.record_uuid for m in result.matches] == [r1.uuid]

    @pytest.mark.asyncio
    async def test_eq_datetime(self, collection):
        records, qv, dts = await self._setup_datetimes(collection)
        uuids = await self._query(collection, qv, "created_at", "=", dts[2])
        assert records[2].uuid in uuids
        assert len(uuids) == 1

    @pytest.mark.asyncio
    async def test_ne_datetime(self, collection):
        records, qv, dts = await self._setup_datetimes(collection)
        uuids = await self._query(collection, qv, "created_at", "!=", dts[2])
        assert records[2].uuid not in uuids
        assert len(uuids) == 4

    @pytest.mark.asyncio
    async def test_gt_datetime(self, collection):
        records, qv, dts = await self._setup_datetimes(collection)
        uuids = await self._query(collection, qv, "created_at", ">", dts[2])
        assert records[0].uuid not in uuids
        assert records[1].uuid not in uuids
        assert records[2].uuid not in uuids  # not strictly greater
        assert records[3].uuid in uuids
        assert records[4].uuid in uuids

    @pytest.mark.asyncio
    async def test_gte_datetime(self, collection):
        records, qv, dts = await self._setup_datetimes(collection)
        uuids = await self._query(collection, qv, "created_at", ">=", dts[2])
        assert records[0].uuid not in uuids
        assert records[1].uuid not in uuids
        assert records[2].uuid in uuids
        assert records[3].uuid in uuids
        assert records[4].uuid in uuids

    @pytest.mark.asyncio
    async def test_lt_datetime(self, collection):
        records, qv, dts = await self._setup_datetimes(collection)
        uuids = await self._query(collection, qv, "created_at", "<", dts[2])
        assert records[0].uuid in uuids
        assert records[1].uuid in uuids
        assert records[2].uuid not in uuids  # not strictly less
        assert records[3].uuid not in uuids
        assert records[4].uuid not in uuids

    @pytest.mark.asyncio
    async def test_lte_datetime(self, collection):
        records, qv, dts = await self._setup_datetimes(collection)
        uuids = await self._query(collection, qv, "created_at", "<=", dts[2])
        assert records[0].uuid in uuids
        assert records[1].uuid in uuids
        assert records[2].uuid in uuids
        assert records[3].uuid not in uuids
        assert records[4].uuid not in uuids

    @pytest.mark.asyncio
    async def test_datetime_microseconds_eq(self, collection):
        """Equality filter distinguishes microsecond precision."""
        v1 = _normalize([1.0, 0.0, 0.0])
        v2 = _normalize([1.0, 0.1, 0.0])
        dt1 = datetime(2024, 6, 15, 12, 30, 45, 123456, tzinfo=UTC)
        dt2 = datetime(2024, 6, 15, 12, 30, 45, 999999, tzinfo=UTC)
        r1 = _make_record(vector=v1, properties={"name": "a", "created_at": dt1})
        r2 = _make_record(vector=v2, properties={"name": "b", "created_at": dt2})
        await collection.upsert(records=[r1, r2])

        uuids = await self._query(collection, v1, "created_at", "=", dt1)
        assert r1.uuid in uuids
        assert r2.uuid not in uuids

    @pytest.mark.asyncio
    async def test_datetime_boundary_inclusive(self, collection):
        """gte and lte are both inclusive at the exact boundary."""
        v1 = _normalize([1.0, 0.0, 0.0])
        v2 = _normalize([1.0, 0.1, 0.0])
        boundary = datetime(2024, 6, 1, tzinfo=UTC)
        before = datetime(2024, 5, 31, 23, 59, 59, tzinfo=UTC)
        r1 = _make_record(vector=v1, properties={"name": "b", "created_at": boundary})
        r2 = _make_record(vector=v2, properties={"name": "a", "created_at": before})
        await collection.upsert(records=[r1, r2])

        gte_uuids = await self._query(collection, v1, "created_at", ">=", boundary)
        lte_uuids = await self._query(collection, v1, "created_at", "<=", boundary)
        assert r1.uuid in gte_uuids
        assert r2.uuid not in gte_uuids
        assert r1.uuid in lte_uuids
        assert r2.uuid in lte_uuids

    @pytest.mark.asyncio
    async def test_eq_datetime_cross_timezone(self, collection):
        """Equality matches the same instant expressed in a different timezone."""
        v1 = _normalize([1.0, 0.0, 0.0])
        v2 = _normalize([1.0, 0.1, 0.0])
        dt_utc = datetime(2024, 6, 15, 12, 0, 0, tzinfo=UTC)
        dt_other = datetime(2024, 6, 15, 18, 0, 0, tzinfo=UTC)
        r1 = _make_record(vector=v1, properties={"name": "a", "created_at": dt_utc})
        r2 = _make_record(vector=v2, properties={"name": "b", "created_at": dt_other})
        await collection.upsert(records=[r1, r2])

        plus5 = timezone(timedelta(hours=5))
        dt_filter = datetime(2024, 6, 15, 17, 0, 0, tzinfo=plus5)
        uuids = await self._query(collection, v1, "created_at", "=", dt_filter)
        assert r1.uuid in uuids
        assert len(uuids) == 1

    @pytest.mark.asyncio
    async def test_gte_datetime_cross_timezone(self, collection):
        """Range filter works correctly with a non-UTC filter value."""
        v1 = _normalize([1.0, 0.0, 0.0])
        v2 = _normalize([1.0, 0.1, 0.0])
        v3 = _normalize([1.0, 0.2, 0.0])
        dt1 = datetime(2024, 1, 1, tzinfo=UTC)
        dt2 = datetime(2024, 6, 15, 12, 0, 0, tzinfo=UTC)
        dt3 = datetime(2024, 12, 31, 23, 0, 0, tzinfo=UTC)
        r1 = _make_record(vector=v1, properties={"name": "a", "created_at": dt1})
        r2 = _make_record(vector=v2, properties={"name": "b", "created_at": dt2})
        r3 = _make_record(vector=v3, properties={"name": "c", "created_at": dt3})
        await collection.upsert(records=[r1, r2, r3])

        # 2024-06-01 00:00 PST = 2024-06-01 08:00 UTC
        pst = timezone(timedelta(hours=-8))
        cutoff = datetime(2024, 6, 1, 0, 0, 0, tzinfo=pst)
        uuids = await self._query(collection, v1, "created_at", ">=", cutoff)
        assert r1.uuid not in uuids
        assert r2.uuid in uuids
        assert r3.uuid in uuids

    @pytest.mark.asyncio
    async def test_naive_datetime_roundtrip(self, collection):
        """Naive datetimes are stored and retrieved as UTC."""
        v1 = _normalize([1.0, 0.0, 0.0])
        naive_dt = datetime(2024, 6, 15, 12, 30, 0, tzinfo=UTC).replace(tzinfo=None)
        r1 = _make_record(vector=v1, properties={"name": "n", "created_at": naive_dt})
        await collection.upsert(records=[r1])

        # Properties are filterable but never returned, so a filter is what
        # observes them.
        # A naive value is stored as the same instant in UTC.
        [result] = await collection.query(
            query_vectors=[v1],
            limit=10,
            property_filter=Comparison(
                field="created_at",
                op="=",
                value=datetime(2024, 6, 15, 12, 30, 0, tzinfo=UTC),
            ),
        )
        assert [m.record_uuid for m in result.matches] == [r1.uuid]

    @pytest.mark.asyncio
    async def test_eq_naive_datetime(self, collection):
        """Equality filter works for naive datetimes."""
        v1 = _normalize([1.0, 0.0, 0.0])
        naive_dt = datetime(2024, 6, 15, 12, 0, 0, tzinfo=UTC).replace(tzinfo=None)
        r1 = _make_record(vector=v1, properties={"name": "n", "created_at": naive_dt})
        await collection.upsert(records=[r1])

        uuids = await self._query(collection, v1, "created_at", "=", naive_dt)
        assert r1.uuid in uuids
        assert len(uuids) == 1

    @pytest.mark.asyncio
    async def test_ne_naive_datetime(self, collection):
        """Not-equal filter works for naive datetimes."""
        v1 = _normalize([1.0, 0.0, 0.0])
        v2 = _normalize([1.0, 0.1, 0.0])
        naive1 = datetime(2024, 1, 1, tzinfo=UTC).replace(tzinfo=None)
        naive2 = datetime(2024, 6, 15, tzinfo=UTC).replace(tzinfo=None)
        r1 = _make_record(vector=v1, properties={"name": "a", "created_at": naive1})
        r2 = _make_record(vector=v2, properties={"name": "b", "created_at": naive2})
        await collection.upsert(records=[r1, r2])

        uuids = await self._query(collection, v1, "created_at", "!=", naive1)
        assert r1.uuid not in uuids
        assert r2.uuid in uuids

    @pytest.mark.asyncio
    async def test_gt_naive_datetime(self, collection):
        """gt filter works for naive datetimes."""
        v1 = _normalize([1.0, 0.0, 0.0])
        v2 = _normalize([1.0, 0.1, 0.0])
        naive1 = datetime(2024, 1, 1, tzinfo=UTC).replace(tzinfo=None)
        naive2 = datetime(2024, 6, 15, 12, 0, 0, tzinfo=UTC).replace(tzinfo=None)
        r1 = _make_record(vector=v1, properties={"name": "a", "created_at": naive1})
        r2 = _make_record(vector=v2, properties={"name": "b", "created_at": naive2})
        await collection.upsert(records=[r1, r2])

        cutoff = datetime(2024, 3, 1, tzinfo=UTC).replace(tzinfo=None)
        uuids = await self._query(collection, v1, "created_at", ">", cutoff)
        assert r1.uuid not in uuids
        assert r2.uuid in uuids

    @pytest.mark.asyncio
    async def test_lt_naive_datetime(self, collection):
        """lt filter works for naive datetimes."""
        v1 = _normalize([1.0, 0.0, 0.0])
        v2 = _normalize([1.0, 0.1, 0.0])
        naive1 = datetime(2024, 1, 1, tzinfo=UTC).replace(tzinfo=None)
        naive2 = datetime(2024, 6, 15, tzinfo=UTC).replace(tzinfo=None)
        r1 = _make_record(vector=v1, properties={"name": "a", "created_at": naive1})
        r2 = _make_record(vector=v2, properties={"name": "b", "created_at": naive2})
        await collection.upsert(records=[r1, r2])

        cutoff = datetime(2024, 3, 1, tzinfo=UTC).replace(tzinfo=None)
        uuids = await self._query(collection, v1, "created_at", "<", cutoff)
        assert r1.uuid in uuids
        assert r2.uuid not in uuids

    # ── IsNull ──

    @pytest.mark.asyncio
    async def test_is_null(self, collection):
        v1 = _normalize([1.0, 0.0, 0.0])
        v2 = _normalize([1.0, 0.1, 0.0])
        v3 = _normalize([1.0, 0.2, 0.0])
        v4 = _normalize([1.0, 0.3, 0.0])
        r_has_value = _make_record(vector=v1, properties={"name": "has_name"})
        r_explicit_none = _make_record(vector=v2, properties={})
        r_key_missing = _make_record(vector=v3, properties={"age": 25})
        r_no_payload = _make_record(vector=v4, properties=None)
        await collection.upsert(
            records=[r_has_value, r_explicit_none, r_key_missing, r_no_payload],
        )

        query_results = list(
            await collection.query(
                query_vectors=[v1],
                limit=10,
                property_filter=IsNull(field="name"),
            )
        )
        uuids = {m.record_uuid for m in query_results[0].matches}
        assert r_has_value.uuid not in uuids
        assert r_explicit_none.uuid in uuids
        assert r_key_missing.uuid in uuids
        assert r_no_payload.uuid in uuids

    @pytest.mark.asyncio
    async def test_is_not_null(self, collection):
        v1 = _normalize([1.0, 0.0, 0.0])
        v2 = _normalize([1.0, 0.1, 0.0])
        v3 = _normalize([1.0, 0.2, 0.0])
        v4 = _normalize([1.0, 0.3, 0.0])
        r_has_value = _make_record(vector=v1, properties={"name": "has_name"})
        r_explicit_none = _make_record(vector=v2, properties={})
        r_key_missing = _make_record(vector=v3, properties={"age": 25})
        r_no_payload = _make_record(vector=v4, properties=None)
        await collection.upsert(
            records=[r_has_value, r_explicit_none, r_key_missing, r_no_payload],
        )

        query_results = list(
            await collection.query(
                query_vectors=[v1],
                limit=10,
                property_filter=Not(IsNull(field="name")),
            )
        )
        uuids = {m.record_uuid for m in query_results[0].matches}
        assert r_has_value.uuid in uuids
        assert r_explicit_none.uuid not in uuids
        assert r_key_missing.uuid not in uuids
        assert r_no_payload.uuid not in uuids

    # ── In / And / Or / Not ──

    @pytest.mark.asyncio
    async def test_in(self, collection):
        r1, _r2, r3, v1 = await self._setup(collection)
        query_results = list(
            await collection.query(
                query_vectors=[v1],
                limit=10,
                property_filter=In(field="name", values=["alice", "carol"]),
            )
        )
        matches = query_results[0].matches
        uuids = {m.record_uuid for m in matches}
        assert r1.uuid in uuids
        assert r3.uuid in uuids
        assert len(matches) == 2

    @pytest.mark.asyncio
    async def test_and(self, collection):
        _r1, _r2, r3, v1 = await self._setup(collection)
        query_results = list(
            await collection.query(
                query_vectors=[v1],
                limit=10,
                property_filter=And(
                    left=Comparison(field="active", op="=", value=True),
                    right=Comparison(field="age", op=">", value=30),
                ),
            )
        )
        matches = query_results[0].matches
        assert len(matches) == 1
        assert matches[0].record_uuid == r3.uuid

    @pytest.mark.asyncio
    async def test_or(self, collection):
        r1, _r2, r3, v1 = await self._setup(collection)
        query_results = list(
            await collection.query(
                query_vectors=[v1],
                limit=10,
                property_filter=Or(
                    left=Comparison(field="name", op="=", value="alice"),
                    right=Comparison(field="name", op="=", value="carol"),
                ),
            )
        )
        matches = query_results[0].matches
        uuids = {m.record_uuid for m in matches}
        assert r1.uuid in uuids
        assert r3.uuid in uuids
        assert len(matches) == 2

    @pytest.mark.asyncio
    async def test_not(self, collection):
        r1, r2, _r3, v1 = await self._setup(collection)
        query_results = list(
            await collection.query(
                query_vectors=[v1],
                limit=10,
                property_filter=Not(Comparison(field="age", op=">", value=30)),
            )
        )
        matches = query_results[0].matches
        uuids = {m.record_uuid for m in matches}
        assert r1.uuid in uuids
        assert r2.uuid in uuids
        assert len(matches) == 2


# ── Delete ──


class TestDelete:
    @pytest.mark.asyncio
    async def test_delete_records(self, collection):
        v1 = _normalize([1.0, 0.0, 0.0])
        v2 = _normalize([0.0, 1.0, 0.0])

        r1 = _make_record(vector=v1)
        r2 = _make_record(vector=v2)

        await collection.upsert(records=[r1, r2])
        await collection.delete(record_uuids=[r1.uuid])

        results = await _present_uuids(collection, [r1.uuid, r2.uuid])
        assert len(results) == 1
        assert results[0] == r2.uuid


# ── Partition isolation ──


class TestPartitionIsolation:
    @pytest.mark.asyncio
    async def test_query_only_returns_own_partition(self, store):
        await store.create_partition("tenant_a")
        await store.create_partition("tenant_b")
        coll_a = await store.get_partition("tenant_a")
        coll_b = await store.get_partition("tenant_b")
        assert coll_a is not None
        assert coll_b is not None

        v1 = _normalize([1.0, 0.0, 0.0])
        r1 = _make_record(vector=v1, properties={})
        r2 = _make_record(vector=v1, properties={})

        await coll_a.upsert(records=[r1])
        await coll_b.upsert(records=[r2])

        results_a = list(await coll_a.query(query_vectors=[v1], limit=10))
        results_b = list(await coll_b.query(query_vectors=[v1], limit=10))

        uuids_a = {m.record_uuid for m in results_a[0].matches}
        uuids_b = {m.record_uuid for m in results_b[0].matches}
        assert uuids_a == {r1.uuid}
        assert uuids_b == {r2.uuid}

        await store.delete_partition("tenant_a")
        await store.delete_partition("tenant_b")

    @pytest.mark.asyncio
    async def test_get_only_returns_own_partition(self, store):
        await store.create_partition("tenant_a")
        await store.create_partition("tenant_b")
        coll_a = await store.get_partition("tenant_a")
        coll_b = await store.get_partition("tenant_b")
        assert coll_a is not None
        assert coll_b is not None

        v1 = _normalize([1.0, 0.0, 0.0])
        r1 = _make_record(vector=v1)
        r2 = _make_record(vector=v1)

        await coll_a.upsert(records=[r1])
        await coll_b.upsert(records=[r2])

        results = await _present_uuids(coll_a, [r1.uuid, r2.uuid])
        assert len(results) == 1
        assert results[0] == r1.uuid

        await store.delete_partition("tenant_a")
        await store.delete_partition("tenant_b")

    @pytest.mark.asyncio
    async def test_delete_only_affects_own_partition(self, store):
        await store.create_partition("tenant_a")
        await store.create_partition("tenant_b")
        coll_a = await store.get_partition("tenant_a")
        coll_b = await store.get_partition("tenant_b")
        assert coll_a is not None
        assert coll_b is not None

        v1 = _normalize([1.0, 0.0, 0.0])
        r1 = _make_record(vector=v1)
        r2 = _make_record(vector=v1)

        await coll_a.upsert(records=[r1])
        await coll_b.upsert(records=[r2])

        # Attempt to delete r2 using tenant_a's collection — should not work
        await coll_a.delete(record_uuids=[r2.uuid])

        results = await _present_uuids(coll_b, [r2.uuid])
        assert len(results) == 1
        assert results[0] == r2.uuid

        await store.delete_partition("tenant_a")
        await store.delete_partition("tenant_b")


# ── Metrics ──


@pytest.mark.integration
class TestMetrics:
    @pytest.mark.asyncio
    async def test_metrics_collection(self, qdrant_client, registry_engine):
        mock_factory = MagicMock(spec=MetricsFactory)
        mock_histogram = MagicMock(spec=MetricsFactory.Histogram)
        mock_factory.get_histogram.return_value = mock_histogram

        params = QdrantVectorStoreParams(
            collection=COLLECTION,
            vector_dimensions=VECTOR_DIM,
            indexed_properties=INDEXED_PROPERTIES,
            client=qdrant_client,
            registry_engine=registry_engine,
            metrics_factory=mock_factory,
        )
        store = QdrantVectorStore(params)
        await store.provision()
        await store.startup()

        await store.create_partition("metrics_test")

        assert mock_histogram.observe.called
        call_labels = mock_histogram.observe.call_args
        assert call_labels[1]["labels"]["operation"] == "create_partition"
        assert call_labels[1]["labels"]["status"] == "ok"

        mock_histogram.reset_mock()

        coll = await store.get_partition("metrics_test")
        assert coll is not None
        v1 = _normalize([1.0, 0.0, 0.0])
        r1 = _make_record(vector=v1)
        await coll.upsert(records=[r1])

        assert mock_histogram.observe.called
        call_labels = mock_histogram.observe.call_args
        assert call_labels[1]["labels"]["operation"] == "upsert"
        assert call_labels[1]["labels"]["status"] == "ok"

        await store.delete_partition("metrics_test")


@pytest.mark.integration
class TestCollectionProvisioningAcrossWorkers:
    """Provisioning has to survive more than one provisioner.

    Run the server with MEMMACHINE_WORKERS above 1 and each worker builds its
    own store over its own client, so two workers can provision the same
    collection at the same moment, and one can find the collection already
    there without its indexes.

    These need a real server: payload indexes have no effect in local-mode
    Qdrant, so the thing under test is invisible there.
    """

    @pytest.mark.asyncio
    async def test_indexes_are_created_when_the_collection_already_exists(
        self, qdrant_client, registry_engine
    ):
        """A collection that exists without its indexes must still get them.

        The collection and its payload indexes are created in separate steps,
        each swallowing "already exists" on its own, so a provisioner that
        finds the collection there, left by another worker or by a crash
        between the two steps, still creates the indexes.

        The partition key is declared is_tenant. Verified against Qdrant 1.19:
        filtering stays correct without the index - a filtered query on an
        unindexed collection returns only the matching tenant's points - so what
        is lost is the multitenant storage layout and query speed, not
        isolation.
        """
        collection = "raced_collection"

        # Stand in for a provisioner that got as far as the collection and no further.
        await qdrant_client.create_collection(
            collection_name=collection,
            vectors_config=models.VectorParams(
                size=VECTOR_DIM, distance=models.Distance.COSINE
            ),
        )

        store = QdrantVectorStore(
            QdrantVectorStoreParams(
                collection=collection,
                vector_dimensions=VECTOR_DIM,
                indexed_properties=INDEXED_PROPERTIES,
                client=qdrant_client,
                registry_engine=registry_engine,
            )
        )
        try:
            await store.provision()
            await store.startup()
            info = await qdrant_client.get_collection(collection)
            indexed = set(info.payload_schema or {})
            assert _PAYLOAD_INCARNATION in indexed, (
                "the tenant partition index is missing: a collection that already "
                "existed never had its payload indexes created, so tenant "
                f"filtering is unindexed. present: {sorted(indexed)}"
            )
            assert "name" in indexed, (
                f"declared property index absent. present: {sorted(indexed)}"
            )
        finally:
            await qdrant_client.delete_collection(collection)

    @pytest.mark.asyncio
    async def test_two_workers_provisioning_at_once_both_succeed_and_index(
        self, qdrant_container, registry_engine
    ):
        """Two clients, one registry - the multi-worker shape, in one process.

        Both provisioners must return, and the collection they agree on must
        end up indexed; then both must be able to create partitions in it,
        and a key both create at once is created once.
        """
        client_a = qdrant_container.get_async_client()
        client_b = qdrant_container.get_async_client()
        collection = "race_two_collection"

        store_a = QdrantVectorStore(
            QdrantVectorStoreParams(
                collection=collection,
                vector_dimensions=VECTOR_DIM,
                indexed_properties=INDEXED_PROPERTIES,
                client=client_a,
                registry_engine=registry_engine,
            )
        )
        store_b = QdrantVectorStore(
            QdrantVectorStoreParams(
                collection=collection,
                vector_dimensions=VECTOR_DIM,
                indexed_properties=INDEXED_PROPERTIES,
                client=client_b,
                registry_engine=registry_engine,
            )
        )

        try:
            results = await asyncio.gather(
                store_a.provision(), store_b.provision(), return_exceptions=True
            )
            failures = [r for r in results if isinstance(r, BaseException)]
            assert not failures, f"a concurrent provisioner raised: {failures!r}"
            await store_a.startup()
            await store_b.startup()

            info = await client_a.get_collection(collection)
            indexed = set(info.payload_schema or {})
            assert _PAYLOAD_INCARNATION in indexed, (
                "two workers raced and the tenant partition index was lost: the "
                "loser skips index creation entirely. present: "
                f"{sorted(indexed)}"
            )

            # The registry's primary key arbitrates: one creator wins, the
            # other gets AlreadyExists, and both then hold the one partition.
            results = await asyncio.gather(
                store_a.create_partition("race_two_key"),
                store_b.create_partition("race_two_key"),
                return_exceptions=True,
            )
            assert sorted(type(r).__name__ for r in results) == [
                "NoneType",
                "VectorStorePartitionAlreadyExistsError",
            ], results
            partition_a = await store_a.get_partition("race_two_key")
            partition_b = await store_b.get_partition("race_two_key")
            assert partition_a is not None
            assert partition_b is not None
            assert partition_a._incarnation == partition_b._incarnation
        finally:
            await store_a.delete_partition("race_two_key")
            await client_a.delete_collection(collection)
            await client_a.close()
            await client_b.close()


@pytest.mark.integration
class TestDeclaredPayloadIndexes:
    """Local mode accepts payload indexes but reports no schema, so this needs a server."""

    @pytest.mark.asyncio
    async def test_every_declared_key_gets_a_payload_index(
        self, qdrant_client, registry_engine
    ):
        store = QdrantVectorStore(
            QdrantVectorStoreParams(
                collection=COLLECTION,
                vector_dimensions=VECTOR_DIM,
                indexed_properties=INDEXED_PROPERTIES,
                client=qdrant_client,
                registry_engine=registry_engine,
            )
        )
        await store.provision()
        await store.startup()
        await store.create_partition("declared_indexes")
        collection = await store.get_partition("declared_indexes")
        assert collection is not None
        info = await store._client.get_collection(collection._collection_name)
        indexed = set(info.payload_schema or {})
        assert set(INDEXED_PROPERTIES) <= indexed
        assert (
            info.payload_schema["created_at"].data_type
            == models.PayloadSchemaType.DATETIME
        )
        assert info.payload_schema["age"].data_type == models.PayloadSchemaType.INTEGER
        await store.delete_partition("declared_indexes")


class TestPartitionLifecycle(PartitionLifecycleContract):
    """The partition lifecycle contract, against this store."""

    @staticmethod
    async def count_stored(store) -> int:
        result = await store._client.count(collection_name=COLLECTION, exact=True)
        return result.count
