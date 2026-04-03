"""Tests for QdrantVectorStore."""

import math
from datetime import UTC, datetime, timedelta, timezone
from unittest.mock import MagicMock
from uuid import UUID, uuid4

import pytest
import pytest_asyncio
from qdrant_client import AsyncQdrantClient

from memmachine_server.common.data_types import PropertyValue, SimilarityMetric
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
    CollectionAlreadyExistsError,
    CollectionConfig,
    CollectionConfigMismatchError,
    Record,
)
from memmachine_server.common.vector_store.qdrant_vector_store import (
    QdrantCollection,
    QdrantVectorStore,
    QdrantVectorStoreParams,
)

NAMESPACE = "test_namespace"
NAME = "test_name"
VECTOR_DIM = 3


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
async def store(any_qdrant_client):
    params = QdrantVectorStoreParams(client=any_qdrant_client)
    s = QdrantVectorStore(params)
    await s.startup()
    yield s


@pytest_asyncio.fixture
async def collection(store):
    await store.create_collection(
        namespace=NAMESPACE,
        name=NAME,
        config=CollectionConfig(
            vector_dimensions=VECTOR_DIM,
            similarity_metric=SimilarityMetric.COSINE,
            properties_schema={
                "name": str,
                "age": int,
                "score": float,
                "active": bool,
                "created_at": datetime,
            },
        ),
    )
    coll = await store.open_collection(namespace=NAMESPACE, name=NAME)
    assert coll is not None
    yield coll
    await store.delete_collection(namespace=NAMESPACE, name=NAME)


def _normalize(v: list[float]) -> list[float]:
    mag = math.sqrt(sum(x * x for x in v))
    return [x / mag for x in v]


def _make_record(
    *,
    uuid: UUID | None = None,
    vector: list[float] | None = None,
    properties: dict | None = None,
) -> Record:
    return Record(
        uuid=uuid or uuid4(),
        vector=vector,
        properties=properties,
    )


# ── Collection lifecycle ──


class TestCollectionLifecycle:
    @pytest.mark.asyncio
    async def test_create_get_delete(self, store):
        await store.create_collection(
            namespace=NAMESPACE,
            name="lifecycle",
            config=CollectionConfig(vector_dimensions=VECTOR_DIM),
        )
        coll = await store.open_collection(namespace=NAMESPACE, name="lifecycle")
        assert isinstance(coll, QdrantCollection)
        await store.delete_collection(namespace=NAMESPACE, name="lifecycle")

    @pytest.mark.asyncio
    async def test_open_collection_returns_qdrant_collection(self, store, collection):
        coll = await store.open_collection(namespace=NAMESPACE, name=NAME)
        assert isinstance(coll, QdrantCollection)

    @pytest.mark.asyncio
    async def test_duplicate_name_raises(self, store, collection):
        with pytest.raises(CollectionAlreadyExistsError):
            await store.create_collection(
                namespace=NAMESPACE,
                name=NAME,
                config=CollectionConfig(
                    vector_dimensions=VECTOR_DIM,
                    similarity_metric=SimilarityMetric.COSINE,
                    properties_schema={
                        "name": str,
                        "age": int,
                        "score": float,
                        "active": bool,
                        "created_at": datetime,
                    },
                ),
            )

    @pytest.mark.asyncio
    async def test_delete_nonexistent_is_idempotent(self, store):
        await store.delete_collection(namespace=NAMESPACE, name="nonexistent")

    @pytest.mark.asyncio
    async def test_open_or_create_creates_when_missing(self, store):
        config = CollectionConfig(vector_dimensions=VECTOR_DIM)
        coll = await store.open_or_create_collection(
            namespace=NAMESPACE, name="new", config=config
        )
        assert isinstance(coll, QdrantCollection)
        await store.delete_collection(namespace=NAMESPACE, name="new")

    @pytest.mark.asyncio
    async def test_open_or_create_opens_when_exists(self, store):
        config = CollectionConfig(vector_dimensions=VECTOR_DIM)
        await store.create_collection(
            namespace=NAMESPACE, name="existing", config=config
        )
        coll = await store.open_or_create_collection(
            namespace=NAMESPACE, name="existing", config=config
        )
        assert isinstance(coll, QdrantCollection)
        await store.delete_collection(namespace=NAMESPACE, name="existing")

    @pytest.mark.asyncio
    async def test_open_or_create_raises_on_config_mismatch(self, store):
        await store.create_collection(
            namespace=NAMESPACE,
            name="mismatch",
            config=CollectionConfig(vector_dimensions=VECTOR_DIM),
        )
        with pytest.raises(CollectionConfigMismatchError):
            await store.open_or_create_collection(
                namespace=NAMESPACE,
                name="mismatch",
                config=CollectionConfig(vector_dimensions=VECTOR_DIM + 1),
            )
        await store.delete_collection(namespace=NAMESPACE, name="mismatch")

    @pytest.mark.asyncio
    async def test_same_config_shares_native_collection(self, store):
        """Two logical collections with the same config share one native collection."""
        schema: dict[str, type[PropertyValue]] = {"name": str}
        config = CollectionConfig(
            vector_dimensions=VECTOR_DIM,
            similarity_metric=SimilarityMetric.COSINE,
            properties_schema=schema,
        )
        await store.create_collection(namespace=NAMESPACE, name="coll_a", config=config)
        await store.create_collection(namespace=NAMESPACE, name="coll_b", config=config)

        coll_a = await store.open_collection(namespace=NAMESPACE, name="coll_a")
        coll_b = await store.open_collection(namespace=NAMESPACE, name="coll_b")
        assert coll_a is not None
        assert coll_b is not None
        assert coll_a._collection_name == coll_b._collection_name

        await store.delete_collection(namespace=NAMESPACE, name="coll_a")
        await store.delete_collection(namespace=NAMESPACE, name="coll_b")


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
        assert matches[0].record.uuid == r1.uuid
        assert matches[1].record.uuid == r3.uuid
        assert matches[2].record.uuid == r2.uuid
        assert matches[0].score >= matches[1].score >= matches[2].score

    @pytest.mark.asyncio
    async def test_query_with_similarity_threshold(self, collection):
        v1 = _normalize([1.0, 0.0, 0.0])
        v2 = _normalize([0.0, 1.0, 0.0])

        r1 = _make_record(vector=v1)
        r2 = _make_record(vector=v2)

        await collection.upsert(records=[r1, r2])

        query_results = list(
            await collection.query(query_vectors=[v1], score_threshold=0.9)
        )
        matches = query_results[0].matches

        assert len(matches) == 1
        assert matches[0].record.uuid == r1.uuid

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
    async def test_query_with_limit_none(self, collection):
        vectors = [_normalize([1.0, float(i) * 0.01, 0.0]) for i in range(5)]
        records = [_make_record(vector=v) for v in vectors]
        await collection.upsert(records=records)

        query_results = list(
            await collection.query(query_vectors=[vectors[0]], limit=None)
        )
        assert len(query_results[0].matches) == 5

    @pytest.mark.asyncio
    async def test_query_return_vector_false(self, collection):
        v1 = _normalize([1.0, 0.0, 0.0])
        r1 = _make_record(vector=v1, properties={"name": "test"})
        await collection.upsert(records=[r1])

        query_results = list(
            await collection.query(query_vectors=[v1], return_vector=False)
        )
        matches = query_results[0].matches
        assert len(matches) == 1
        assert matches[0].record.vector is None
        assert matches[0].record.properties is not None

    @pytest.mark.asyncio
    async def test_query_return_properties_false(self, collection):
        v1 = _normalize([1.0, 0.0, 0.0])
        r1 = _make_record(vector=v1, properties={"name": "test"})
        await collection.upsert(records=[r1])

        query_results = list(
            await collection.query(
                query_vectors=[v1],
                return_vector=True,
                return_properties=False,
            )
        )
        matches = query_results[0].matches
        assert len(matches) == 1
        assert matches[0].record.vector is not None
        assert matches[0].record.properties is None

    @pytest.mark.asyncio
    async def test_query_batch_multiple_vectors(self, collection):
        v1 = _normalize([1.0, 0.0, 0.0])
        v2 = _normalize([0.0, 1.0, 0.0])

        r1 = _make_record(vector=v1, properties={"name": "a"})
        r2 = _make_record(vector=v2, properties={"name": "b"})
        await collection.upsert(records=[r1, r2])

        all_results = list(await collection.query(query_vectors=[v1, v2], limit=1))

        assert len(all_results) == 2
        assert all_results[0].matches[0].record.uuid == r1.uuid
        assert all_results[1].matches[0].record.uuid == r2.uuid

    @pytest.mark.asyncio
    async def test_query_empty_vectors(self, collection):
        all_results = list(await collection.query(query_vectors=[]))
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
                property_filter=Comparison(field=field, op=op, value=value),
            )
        )
        return {m.record.uuid for m in all_results[0].matches}

    # ── String / int ──

    @pytest.mark.asyncio
    async def test_eq_str(self, collection):
        r1, _r2, _r3, v1 = await self._setup(collection)
        query_results = list(
            await collection.query(
                query_vectors=[v1],
                property_filter=Comparison(field="name", op="=", value="alice"),
            )
        )
        matches = query_results[0].matches
        assert len(matches) == 1
        assert matches[0].record.uuid == r1.uuid

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
                property_filter=Comparison(field="age", op=">", value=30),
            )
        )
        matches = query_results[0].matches
        assert len(matches) == 1
        assert matches[0].record.uuid == r3.uuid

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
                property_filter=Comparison(field="age", op="<", value=30),
            )
        )
        matches = query_results[0].matches
        assert len(matches) == 1
        assert matches[0].record.uuid == r2.uuid

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

        results = list(await collection.get(record_uuids=[r1.uuid]))
        assert results[0].properties["created_at"] == dt

    @pytest.mark.asyncio
    async def test_datetime_microseconds_roundtrip(self, collection):
        """Microsecond precision is preserved."""
        v1 = _normalize([1.0, 0.0, 0.0])
        dt = datetime(2024, 6, 15, 12, 30, 45, 123456, tzinfo=UTC)
        r1 = _make_record(vector=v1, properties={"name": "micro", "created_at": dt})
        await collection.upsert(records=[r1])

        results = list(await collection.get(record_uuids=[r1.uuid]))
        assert results[0].properties["created_at"] == dt

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

        results = list(await collection.get(record_uuids=[r1.uuid]))
        got = results[0].properties["created_at"]
        assert isinstance(got, datetime)
        assert got.tzinfo is not None
        assert got == datetime(2024, 6, 15, 12, 30, 0, tzinfo=UTC)

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
                property_filter=IsNull(field="name"),
            )
        )
        uuids = {m.record.uuid for m in query_results[0].matches}
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
                property_filter=Not(expr=IsNull(field="name")),
            )
        )
        uuids = {m.record.uuid for m in query_results[0].matches}
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
                property_filter=In(field="name", values=["alice", "carol"]),
            )
        )
        matches = query_results[0].matches
        uuids = {m.record.uuid for m in matches}
        assert r1.uuid in uuids
        assert r3.uuid in uuids
        assert len(matches) == 2

    @pytest.mark.asyncio
    async def test_and(self, collection):
        _r1, _r2, r3, v1 = await self._setup(collection)
        query_results = list(
            await collection.query(
                query_vectors=[v1],
                property_filter=And(
                    left=Comparison(field="active", op="=", value=True),
                    right=Comparison(field="age", op=">", value=30),
                ),
            )
        )
        matches = query_results[0].matches
        assert len(matches) == 1
        assert matches[0].record.uuid == r3.uuid

    @pytest.mark.asyncio
    async def test_or(self, collection):
        r1, _r2, r3, v1 = await self._setup(collection)
        query_results = list(
            await collection.query(
                query_vectors=[v1],
                property_filter=Or(
                    left=Comparison(field="name", op="=", value="alice"),
                    right=Comparison(field="name", op="=", value="carol"),
                ),
            )
        )
        matches = query_results[0].matches
        uuids = {m.record.uuid for m in matches}
        assert r1.uuid in uuids
        assert r3.uuid in uuids
        assert len(matches) == 2

    @pytest.mark.asyncio
    async def test_not(self, collection):
        r1, r2, _r3, v1 = await self._setup(collection)
        query_results = list(
            await collection.query(
                query_vectors=[v1],
                property_filter=Not(expr=Comparison(field="age", op=">", value=30)),
            )
        )
        matches = query_results[0].matches
        uuids = {m.record.uuid for m in matches}
        assert r1.uuid in uuids
        assert r2.uuid in uuids
        assert len(matches) == 2


# ── Get ──


class TestGet:
    @pytest.mark.asyncio
    async def test_get_by_uuids(self, collection):
        v1 = _normalize([1.0, 0.0, 0.0])
        v2 = _normalize([0.0, 1.0, 0.0])
        v3 = _normalize([0.0, 0.0, 1.0])

        r1 = _make_record(vector=v1, properties={"name": "a"})
        r2 = _make_record(vector=v2, properties={"name": "b"})
        r3 = _make_record(vector=v3, properties={"name": "c"})

        await collection.upsert(records=[r1, r2, r3])

        results = list(await collection.get(record_uuids=[r3.uuid, r1.uuid]))
        assert len(results) == 2
        assert results[0].uuid == r3.uuid
        assert results[1].uuid == r1.uuid

    @pytest.mark.asyncio
    async def test_get_missing_uuids(self, collection):
        v1 = _normalize([1.0, 0.0, 0.0])
        r1 = _make_record(vector=v1)
        await collection.upsert(records=[r1])

        missing_uuid = uuid4()
        results = list(await collection.get(record_uuids=[r1.uuid, missing_uuid]))
        assert len(results) == 1
        assert results[0].uuid == r1.uuid

    @pytest.mark.asyncio
    async def test_get_empty_list(self, collection):
        results = list(await collection.get(record_uuids=[]))
        assert len(results) == 0

    @pytest.mark.asyncio
    async def test_get_return_vector_false(self, collection):
        v1 = _normalize([1.0, 0.0, 0.0])
        r1 = _make_record(vector=v1, properties={"name": "test"})
        await collection.upsert(records=[r1])

        results = list(
            await collection.get(record_uuids=[r1.uuid], return_vector=False)
        )
        assert len(results) == 1
        assert results[0].vector is None
        assert results[0].properties is not None

    @pytest.mark.asyncio
    async def test_get_return_properties_false(self, collection):
        v1 = _normalize([1.0, 0.0, 0.0])
        r1 = _make_record(vector=v1, properties={"name": "test"})
        await collection.upsert(records=[r1])

        results = list(
            await collection.get(
                record_uuids=[r1.uuid],
                return_vector=True,
                return_properties=False,
            )
        )
        assert len(results) == 1
        assert results[0].vector is not None
        assert results[0].properties is None


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

        results = list(await collection.get(record_uuids=[r1.uuid, r2.uuid]))
        assert len(results) == 1
        assert results[0].uuid == r2.uuid


# ── Partition isolation (via separate logical collections) ──


class TestPartitionIsolation:
    @pytest.mark.asyncio
    async def test_query_only_returns_own_partition(self, store):
        await store.create_collection(
            namespace=NAMESPACE,
            name="tenant_a",
            config=CollectionConfig(vector_dimensions=VECTOR_DIM),
        )
        await store.create_collection(
            namespace=NAMESPACE,
            name="tenant_b",
            config=CollectionConfig(vector_dimensions=VECTOR_DIM),
        )
        coll_a = await store.open_collection(namespace=NAMESPACE, name="tenant_a")
        coll_b = await store.open_collection(namespace=NAMESPACE, name="tenant_b")
        assert coll_a is not None
        assert coll_b is not None

        v1 = _normalize([1.0, 0.0, 0.0])
        r1 = _make_record(vector=v1, properties={})
        r2 = _make_record(vector=v1, properties={})

        await coll_a.upsert(records=[r1])
        await coll_b.upsert(records=[r2])

        results_a = list(await coll_a.query(query_vectors=[v1], limit=10))
        results_b = list(await coll_b.query(query_vectors=[v1], limit=10))

        uuids_a = {m.record.uuid for m in results_a[0].matches}
        uuids_b = {m.record.uuid for m in results_b[0].matches}
        assert uuids_a == {r1.uuid}
        assert uuids_b == {r2.uuid}

        await store.delete_collection(namespace=NAMESPACE, name="tenant_a")
        await store.delete_collection(namespace=NAMESPACE, name="tenant_b")

    @pytest.mark.asyncio
    async def test_get_only_returns_own_partition(self, store):
        await store.create_collection(
            namespace=NAMESPACE,
            name="tenant_a",
            config=CollectionConfig(vector_dimensions=VECTOR_DIM),
        )
        await store.create_collection(
            namespace=NAMESPACE,
            name="tenant_b",
            config=CollectionConfig(vector_dimensions=VECTOR_DIM),
        )
        coll_a = await store.open_collection(namespace=NAMESPACE, name="tenant_a")
        coll_b = await store.open_collection(namespace=NAMESPACE, name="tenant_b")
        assert coll_a is not None
        assert coll_b is not None

        v1 = _normalize([1.0, 0.0, 0.0])
        r1 = _make_record(vector=v1)
        r2 = _make_record(vector=v1)

        await coll_a.upsert(records=[r1])
        await coll_b.upsert(records=[r2])

        results = list(await coll_a.get(record_uuids=[r1.uuid, r2.uuid]))
        assert len(results) == 1
        assert results[0].uuid == r1.uuid

        await store.delete_collection(namespace=NAMESPACE, name="tenant_a")
        await store.delete_collection(namespace=NAMESPACE, name="tenant_b")

    @pytest.mark.asyncio
    async def test_delete_only_affects_own_partition(self, store):
        await store.create_collection(
            namespace=NAMESPACE,
            name="tenant_a",
            config=CollectionConfig(vector_dimensions=VECTOR_DIM),
        )
        await store.create_collection(
            namespace=NAMESPACE,
            name="tenant_b",
            config=CollectionConfig(vector_dimensions=VECTOR_DIM),
        )
        coll_a = await store.open_collection(namespace=NAMESPACE, name="tenant_a")
        coll_b = await store.open_collection(namespace=NAMESPACE, name="tenant_b")
        assert coll_a is not None
        assert coll_b is not None

        v1 = _normalize([1.0, 0.0, 0.0])
        r1 = _make_record(vector=v1)
        r2 = _make_record(vector=v1)

        await coll_a.upsert(records=[r1])
        await coll_b.upsert(records=[r2])

        # Attempt to delete r2 using tenant_a's collection — should not work
        await coll_a.delete(record_uuids=[r2.uuid])

        results = list(await coll_b.get(record_uuids=[r2.uuid]))
        assert len(results) == 1
        assert results[0].uuid == r2.uuid

        await store.delete_collection(namespace=NAMESPACE, name="tenant_a")
        await store.delete_collection(namespace=NAMESPACE, name="tenant_b")


# ── Metrics ──


@pytest.mark.integration
class TestMetrics:
    @pytest.mark.asyncio
    async def test_metrics_collection(self, qdrant_client):
        mock_factory = MagicMock(spec=MetricsFactory)
        mock_histogram = MagicMock(spec=MetricsFactory.Histogram)
        mock_factory.get_histogram.return_value = mock_histogram

        params = QdrantVectorStoreParams(
            client=qdrant_client,
            metrics_factory=mock_factory,
        )
        store = QdrantVectorStore(params)
        await store.startup()

        await store.create_collection(
            namespace=NAMESPACE,
            name="metrics_test",
            config=CollectionConfig(vector_dimensions=VECTOR_DIM),
        )

        assert mock_histogram.observe.called
        call_labels = mock_histogram.observe.call_args
        assert call_labels[1]["labels"]["operation"] == "create_collection"
        assert call_labels[1]["labels"]["status"] == "ok"

        mock_histogram.reset_mock()

        coll = await store.open_collection(namespace=NAMESPACE, name="metrics_test")
        assert coll is not None
        v1 = _normalize([1.0, 0.0, 0.0])
        r1 = _make_record(vector=v1)
        await coll.upsert(records=[r1])

        assert mock_histogram.observe.called
        call_labels = mock_histogram.observe.call_args
        assert call_labels[1]["labels"]["operation"] == "upsert"
        assert call_labels[1]["labels"]["status"] == "ok"

        await store.delete_collection(namespace=NAMESPACE, name="metrics_test")


# ── Distributed sharding ──


@pytest_asyncio.fixture
async def distributed_store(distributed_qdrant_client):
    params = QdrantVectorStoreParams(
        client=distributed_qdrant_client, is_distributed=True
    )
    s = QdrantVectorStore(params)
    await s.startup()
    yield s


@pytest.mark.integration
@pytest.mark.asyncio
class TestDistributedSharding:
    """Tests for custom-sharding behaviour when is_distributed=True."""

    async def test_crud_lifecycle(self, distributed_store):
        """Full create → upsert → query → get → delete records → delete collection."""
        store = distributed_store
        ns, name = NAMESPACE, "distributed_crud"

        await store.create_collection(
            namespace=ns,
            name=name,
            config=CollectionConfig(
                vector_dimensions=VECTOR_DIM,
                similarity_metric=SimilarityMetric.COSINE,
                properties_schema={"name": str},
            ),
        )
        coll = await store.open_collection(namespace=ns, name=name)
        assert coll is not None

        v1 = _normalize([1.0, 0.0, 0.0])
        r1 = _make_record(vector=v1, properties={"name": "alice"})
        await coll.upsert(records=[r1])

        results = await coll.query(query_vectors=[v1], limit=1)
        assert len(results) == 1
        assert results[0].matches[0].record.uuid == r1.uuid

        got = await coll.get(record_uuids=[r1.uuid])
        assert len(got) == 1
        assert got[0].uuid == r1.uuid

        await coll.delete(record_uuids=[r1.uuid])
        got = await coll.get(record_uuids=[r1.uuid])
        assert len(got) == 0

        await store.delete_collection(namespace=ns, name=name)

    async def test_shard_drop_isolates_logical_collections(self, distributed_store):
        """Dropping one logical collection's shard must not affect another's data."""
        store = distributed_store
        ns = NAMESPACE
        config = CollectionConfig(
            vector_dimensions=VECTOR_DIM,
            similarity_metric=SimilarityMetric.COSINE,
        )

        await store.create_collection(namespace=ns, name="tenant_a", config=config)
        await store.create_collection(namespace=ns, name="tenant_b", config=config)

        coll_a = await store.open_collection(namespace=ns, name="tenant_a")
        coll_b = await store.open_collection(namespace=ns, name="tenant_b")
        assert coll_a is not None
        assert coll_b is not None

        v1 = _normalize([1.0, 0.0, 0.0])
        v2 = _normalize([0.0, 1.0, 0.0])
        r_a = _make_record(vector=v1)
        r_b = _make_record(vector=v2)
        await coll_a.upsert(records=[r_a])
        await coll_b.upsert(records=[r_b])

        # Delete tenant_a's shard.
        await store.delete_collection(namespace=ns, name="tenant_a")

        # tenant_b data should be untouched.
        got = await coll_b.get(record_uuids=[r_b.uuid])
        assert len(got) == 1
        assert got[0].uuid == r_b.uuid

        await store.delete_collection(namespace=ns, name="tenant_b")

    async def test_idempotent_delete(self, distributed_store):
        """Deleting an already-deleted collection should be a no-op."""
        store = distributed_store
        ns, name = NAMESPACE, "distributed_idem"
        config = CollectionConfig(vector_dimensions=VECTOR_DIM)

        await store.create_collection(namespace=ns, name=name, config=config)
        await store.delete_collection(namespace=ns, name=name)
        # Second delete should not raise.
        await store.delete_collection(namespace=ns, name=name)
