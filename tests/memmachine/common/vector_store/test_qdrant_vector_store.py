"""Tests for QdrantVectorStore."""

import math
from datetime import UTC, datetime, timedelta, timezone
from unittest.mock import MagicMock
from uuid import UUID, uuid4

import pytest
import pytest_asyncio
from qdrant_client import AsyncQdrantClient

from memmachine.common.data_types import SimilarityMetric
from memmachine.common.filter.filter_parser import And, Comparison, In, IsNull, Not, Or
from memmachine.common.metrics_factory import MetricsFactory
from memmachine.common.vector_store.data_types import Record
from memmachine.common.vector_store.qdrant_vector_store import (
    QdrantCollection,
    QdrantVectorStore,
    QdrantVectorStoreParams,
)

COLLECTION = "test_collection"
VECTOR_DIM = 3


@pytest_asyncio.fixture
async def client():
    c = AsyncQdrantClient(":memory:")
    yield c
    await c.close()


@pytest_asyncio.fixture
async def store(client):
    params = QdrantVectorStoreParams(client=client)
    s = QdrantVectorStore(params)
    await s.startup()
    yield s


@pytest_asyncio.fixture
async def collection(store):
    await store.create_collection(
        COLLECTION,
        vector_dimensions=VECTOR_DIM,
        similarity_metric=SimilarityMetric.COSINE,
        properties_schema={
            "name": str,
            "age": int,
            "score": float,
            "active": bool,
            "created_at": datetime,
        },
    )
    coll = await store.get_collection(COLLECTION)
    yield coll
    await store.delete_collection(COLLECTION)


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
            "lifecycle",
            vector_dimensions=VECTOR_DIM,
        )
        coll = await store.get_collection("lifecycle")
        assert isinstance(coll, QdrantCollection)
        await store.delete_collection("lifecycle")

    @pytest.mark.asyncio
    async def test_get_collection_returns_qdrant_collection(self, store, collection):
        coll = await store.get_collection(COLLECTION)
        assert isinstance(coll, QdrantCollection)


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

        results = list(await collection.query(query_vector=v1, limit=3))

        assert len(results) == 3
        # Most similar to v1 should be first
        assert results[0].record.uuid == r1.uuid
        # v3 is closer to v1 than v2
        assert results[1].record.uuid == r3.uuid
        assert results[2].record.uuid == r2.uuid
        # Scores should be descending
        assert results[0].score >= results[1].score >= results[2].score

    @pytest.mark.asyncio
    async def test_query_with_similarity_threshold(self, collection):
        v1 = _normalize([1.0, 0.0, 0.0])
        v2 = _normalize([0.0, 1.0, 0.0])

        r1 = _make_record(vector=v1)
        r2 = _make_record(vector=v2)

        await collection.upsert(records=[r1, r2])

        results = list(
            await collection.query(
                query_vector=v1,
                similarity_threshold=0.9,
            )
        )

        assert len(results) == 1
        assert results[0].record.uuid == r1.uuid

    @pytest.mark.asyncio
    async def test_query_with_limit(self, collection):
        vectors = [_normalize([1.0, float(i) * 0.01, 0.0]) for i in range(5)]
        records = [_make_record(vector=v) for v in vectors]
        await collection.upsert(records=records)

        results = list(
            await collection.query(
                query_vector=vectors[0],
                limit=2,
            )
        )
        assert len(results) == 2

    @pytest.mark.asyncio
    async def test_query_with_limit_none(self, collection):
        vectors = [_normalize([1.0, float(i) * 0.01, 0.0]) for i in range(5)]
        records = [_make_record(vector=v) for v in vectors]
        await collection.upsert(records=records)

        results = list(
            await collection.query(
                query_vector=vectors[0],
                limit=None,
            )
        )
        assert len(results) == 5

    @pytest.mark.asyncio
    async def test_query_return_vector_false(self, collection):
        v1 = _normalize([1.0, 0.0, 0.0])
        r1 = _make_record(vector=v1, properties={"name": "test"})
        await collection.upsert(records=[r1])

        results = list(
            await collection.query(
                query_vector=v1,
                return_vector=False,
            )
        )
        assert len(results) == 1
        assert results[0].record.vector is None
        assert results[0].record.properties is not None

    @pytest.mark.asyncio
    async def test_query_return_properties_false(self, collection):
        v1 = _normalize([1.0, 0.0, 0.0])
        r1 = _make_record(vector=v1, properties={"name": "test"})
        await collection.upsert(records=[r1])

        results = list(
            await collection.query(
                query_vector=v1,
                return_properties=False,
            )
        )
        assert len(results) == 1
        assert results[0].record.vector is not None
        assert results[0].record.properties is None


# ── Filters ──


class TestFilters:
    async def _setup_filter_data(self, collection):
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

    @pytest.mark.asyncio
    async def test_filter_eq(self, collection):
        r1, _r2, _r3, v1 = await self._setup_filter_data(collection)

        results = list(
            await collection.query(
                query_vector=v1,
                property_filter=Comparison(field="name", op="=", value="alice"),
            )
        )
        assert len(results) == 1
        assert results[0].record.uuid == r1.uuid

    @pytest.mark.asyncio
    async def test_filter_ne(self, collection):
        r1, r2, r3, v1 = await self._setup_filter_data(collection)

        results = list(
            await collection.query(
                query_vector=v1,
                property_filter=Comparison(field="name", op="!=", value="alice"),
            )
        )
        uuids = {r.record.uuid for r in results}
        assert r1.uuid not in uuids
        assert r2.uuid in uuids
        assert r3.uuid in uuids

    @pytest.mark.asyncio
    async def test_filter_gt(self, collection):
        _r1, _r2, r3, v1 = await self._setup_filter_data(collection)

        results = list(
            await collection.query(
                query_vector=v1,
                property_filter=Comparison(field="age", op=">", value=30),
            )
        )
        assert len(results) == 1
        assert results[0].record.uuid == r3.uuid

    @pytest.mark.asyncio
    async def test_filter_gte(self, collection):
        r1, _r2, r3, v1 = await self._setup_filter_data(collection)

        results = list(
            await collection.query(
                query_vector=v1,
                property_filter=Comparison(field="age", op=">=", value=30),
            )
        )
        uuids = {r.record.uuid for r in results}
        assert r1.uuid in uuids
        assert r3.uuid in uuids
        assert len(results) == 2

    @pytest.mark.asyncio
    async def test_filter_lt(self, collection):
        _r1, r2, _r3, v1 = await self._setup_filter_data(collection)

        results = list(
            await collection.query(
                query_vector=v1,
                property_filter=Comparison(field="age", op="<", value=30),
            )
        )
        assert len(results) == 1
        assert results[0].record.uuid == r2.uuid

    @pytest.mark.asyncio
    async def test_filter_lte(self, collection):
        r1, r2, _r3, v1 = await self._setup_filter_data(collection)

        results = list(
            await collection.query(
                query_vector=v1,
                property_filter=Comparison(field="age", op="<=", value=30),
            )
        )
        uuids = {r.record.uuid for r in results}
        assert r1.uuid in uuids
        assert r2.uuid in uuids
        assert len(results) == 2

    @pytest.mark.asyncio
    async def test_filter_in(self, collection):
        r1, _r2, r3, v1 = await self._setup_filter_data(collection)

        results = list(
            await collection.query(
                query_vector=v1,
                property_filter=In(field="name", values=["alice", "carol"]),
            )
        )
        uuids = {r.record.uuid for r in results}
        assert r1.uuid in uuids
        assert r3.uuid in uuids
        assert len(results) == 2

    @pytest.mark.asyncio
    async def test_filter_is_null(self, collection):
        v1 = _normalize([1.0, 0.0, 0.0])
        v2 = _normalize([1.0, 0.1, 0.0])

        r1 = _make_record(vector=v1, properties={"name": "has_name"})
        # Explicit None value so Qdrant sees a null payload field
        r2 = _make_record(vector=v2, properties={"name": None})

        await collection.upsert(records=[r1, r2])

        results = list(
            await collection.query(
                query_vector=v1,
                property_filter=IsNull(field="name"),
            )
        )
        assert len(results) == 1
        assert results[0].record.uuid == r2.uuid

    @pytest.mark.asyncio
    async def test_filter_is_not_null(self, collection):
        v1 = _normalize([1.0, 0.0, 0.0])
        v2 = _normalize([1.0, 0.1, 0.0])

        r1 = _make_record(vector=v1, properties={"name": "has_name"})
        # Explicit None value so Qdrant sees a null payload field
        r2 = _make_record(vector=v2, properties={"name": None})

        await collection.upsert(records=[r1, r2])

        results = list(
            await collection.query(
                query_vector=v1,
                property_filter=Not(expr=IsNull(field="name")),
            )
        )
        assert len(results) == 1
        assert results[0].record.uuid == r1.uuid

    @pytest.mark.asyncio
    async def test_filter_eq_float(self, collection):
        r1, _r2, _r3, v1 = await self._setup_filter_data(collection)

        results = list(
            await collection.query(
                query_vector=v1,
                property_filter=Comparison(field="score", op="=", value=9.5),
            )
        )
        assert len(results) == 1
        assert results[0].record.uuid == r1.uuid

    @pytest.mark.asyncio
    async def test_filter_eq_float_whole_number(self, collection):
        _r1, _r2, r3, v1 = await self._setup_filter_data(collection)

        results = list(
            await collection.query(
                query_vector=v1,
                property_filter=Comparison(field="score", op="=", value=8.0),
            )
        )
        assert len(results) == 1
        assert results[0].record.uuid == r3.uuid

    @pytest.mark.asyncio
    async def test_filter_ne_float(self, collection):
        r1, r2, r3, v1 = await self._setup_filter_data(collection)

        results = list(
            await collection.query(
                query_vector=v1,
                property_filter=Comparison(field="score", op="!=", value=9.5),
            )
        )
        uuids = {r.record.uuid for r in results}
        assert r1.uuid not in uuids
        assert r2.uuid in uuids
        assert r3.uuid in uuids

    @pytest.mark.asyncio
    async def test_filter_and(self, collection):
        _r1, _r2, r3, v1 = await self._setup_filter_data(collection)

        results = list(
            await collection.query(
                query_vector=v1,
                property_filter=And(
                    left=Comparison(field="active", op="=", value=True),
                    right=Comparison(field="age", op=">", value=30),
                ),
            )
        )
        assert len(results) == 1
        assert results[0].record.uuid == r3.uuid

    @pytest.mark.asyncio
    async def test_filter_or(self, collection):
        r1, _r2, r3, v1 = await self._setup_filter_data(collection)

        results = list(
            await collection.query(
                query_vector=v1,
                property_filter=Or(
                    left=Comparison(field="name", op="=", value="alice"),
                    right=Comparison(field="name", op="=", value="carol"),
                ),
            )
        )
        uuids = {r.record.uuid for r in results}
        assert r1.uuid in uuids
        assert r3.uuid in uuids
        assert len(results) == 2

    @pytest.mark.asyncio
    async def test_filter_not_comparison(self, collection):
        r1, r2, _r3, v1 = await self._setup_filter_data(collection)

        results = list(
            await collection.query(
                query_vector=v1,
                property_filter=Not(expr=Comparison(field="age", op=">", value=30)),
            )
        )
        uuids = {r.record.uuid for r in results}
        assert r1.uuid in uuids
        assert r2.uuid in uuids
        assert len(results) == 2


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

        # Request in reverse order to verify input ordering
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
        # Only the existing one should be returned
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
            await collection.get(
                record_uuids=[r1.uuid],
                return_vector=False,
            )
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


# ── Datetime round-trip ──


class TestDatetime:
    @pytest.mark.asyncio
    async def test_datetime_roundtrip(self, collection):
        v1 = _normalize([1.0, 0.0, 0.0])
        dt = datetime(2024, 6, 15, 12, 30, 0, tzinfo=UTC)

        r1 = _make_record(vector=v1, properties={"name": "test", "created_at": dt})
        await collection.upsert(records=[r1])

        results = list(await collection.get(record_uuids=[r1.uuid]))
        assert len(results) == 1
        assert results[0].properties is not None
        assert results[0].properties["created_at"] == dt

    @pytest.mark.asyncio
    async def test_datetime_filter(self, collection):
        v1 = _normalize([1.0, 0.0, 0.0])
        v2 = _normalize([1.0, 0.1, 0.0])

        dt1 = datetime(2024, 1, 1, tzinfo=UTC)
        dt2 = datetime(2024, 6, 15, tzinfo=UTC)

        r1 = _make_record(vector=v1, properties={"name": "old", "created_at": dt1})
        r2 = _make_record(vector=v2, properties={"name": "new", "created_at": dt2})

        await collection.upsert(records=[r1, r2])

        results = list(
            await collection.query(
                query_vector=v1,
                property_filter=Comparison(
                    field="created_at",
                    op=">",
                    value=datetime(2024, 3, 1, tzinfo=UTC),
                ),
            )
        )
        assert len(results) == 1
        assert results[0].record.uuid == r2.uuid

    @pytest.mark.asyncio
    async def test_datetime_filter_eq(self, collection):
        v1 = _normalize([1.0, 0.0, 0.0])
        v2 = _normalize([1.0, 0.1, 0.0])

        dt1 = datetime(2024, 1, 1, tzinfo=UTC)
        dt2 = datetime(2024, 6, 15, tzinfo=UTC)

        r1 = _make_record(vector=v1, properties={"name": "old", "created_at": dt1})
        r2 = _make_record(vector=v2, properties={"name": "new", "created_at": dt2})

        await collection.upsert(records=[r1, r2])

        results = list(
            await collection.query(
                query_vector=v1,
                property_filter=Comparison(field="created_at", op="=", value=dt1),
            )
        )
        assert len(results) == 1
        assert results[0].record.uuid == r1.uuid

    @pytest.mark.asyncio
    async def test_datetime_filter_ne(self, collection):
        v1 = _normalize([1.0, 0.0, 0.0])
        v2 = _normalize([1.0, 0.1, 0.0])

        dt1 = datetime(2024, 1, 1, tzinfo=UTC)
        dt2 = datetime(2024, 6, 15, tzinfo=UTC)

        r1 = _make_record(vector=v1, properties={"name": "old", "created_at": dt1})
        r2 = _make_record(vector=v2, properties={"name": "new", "created_at": dt2})

        await collection.upsert(records=[r1, r2])

        results = list(
            await collection.query(
                query_vector=v1,
                property_filter=Comparison(field="created_at", op="!=", value=dt1),
            )
        )
        assert len(results) == 1
        assert results[0].record.uuid == r2.uuid

    @pytest.mark.asyncio
    async def test_datetime_filter_eq_cross_timezone(self, collection):
        """Equality filter matches when the filter value is the same instant in a different timezone."""
        v1 = _normalize([1.0, 0.0, 0.0])
        v2 = _normalize([1.0, 0.1, 0.0])

        dt_utc = datetime(2024, 6, 15, 12, 0, 0, tzinfo=UTC)
        dt_other = datetime(2024, 6, 15, 18, 0, 0, tzinfo=UTC)

        r1 = _make_record(vector=v1, properties={"name": "a", "created_at": dt_utc})
        r2 = _make_record(vector=v2, properties={"name": "b", "created_at": dt_other})
        await collection.upsert(records=[r1, r2])

        # Filter using UTC+5 representation of the same instant as dt_utc
        plus5 = timezone(timedelta(hours=5))
        dt_filter = datetime(2024, 6, 15, 17, 0, 0, tzinfo=plus5)

        results = list(
            await collection.query(
                query_vector=v1,
                property_filter=Comparison(field="created_at", op="=", value=dt_filter),
            )
        )
        assert len(results) == 1
        assert results[0].record.uuid == r1.uuid

    @pytest.mark.asyncio
    async def test_datetime_filter_range_cross_timezone(self, collection):
        """Range filter works correctly when the filter value uses a different timezone."""
        v1 = _normalize([1.0, 0.0, 0.0])
        v2 = _normalize([1.0, 0.1, 0.0])
        v3 = _normalize([1.0, 0.2, 0.0])

        dt1 = datetime(2024, 1, 1, 0, 0, 0, tzinfo=UTC)
        dt2 = datetime(2024, 6, 15, 12, 0, 0, tzinfo=UTC)
        dt3 = datetime(2024, 12, 31, 23, 0, 0, tzinfo=UTC)

        r1 = _make_record(vector=v1, properties={"name": "early", "created_at": dt1})
        r2 = _make_record(vector=v2, properties={"name": "mid", "created_at": dt2})
        r3 = _make_record(vector=v3, properties={"name": "late", "created_at": dt3})
        await collection.upsert(records=[r1, r2, r3])

        # Use UTC-8 (PST) for the cutoff: 2024-06-01 00:00 PST = 2024-06-01 08:00 UTC
        pst = timezone(timedelta(hours=-8))
        cutoff = datetime(2024, 6, 1, 0, 0, 0, tzinfo=pst)

        results = list(
            await collection.query(
                query_vector=v1,
                property_filter=Comparison(field="created_at", op=">=", value=cutoff),
            )
        )
        uuids = {r.record.uuid for r in results}
        assert r1.uuid not in uuids
        assert r2.uuid in uuids
        assert r3.uuid in uuids


# ── Metrics ──


class TestMetrics:
    @pytest.mark.asyncio
    async def test_metrics_collection(self, client):
        mock_factory = MagicMock(spec=MetricsFactory)
        mock_counter = MagicMock(spec=MetricsFactory.Counter)
        mock_summary = MagicMock(spec=MetricsFactory.Summary)
        mock_factory.get_counter.return_value = mock_counter
        mock_factory.get_summary.return_value = mock_summary

        params = QdrantVectorStoreParams(
            client=client,
            metrics_factory=mock_factory,
            user_metrics_labels={"env": "test"},
        )
        store = QdrantVectorStore(params)
        await store.startup()

        await store.create_collection(
            "metrics_test",
            vector_dimensions=VECTOR_DIM,
        )

        assert mock_counter.increment.called
        assert mock_summary.observe.called

        mock_counter.reset_mock()
        mock_summary.reset_mock()

        coll = await store.get_collection("metrics_test")
        v1 = _normalize([1.0, 0.0, 0.0])
        r1 = _make_record(vector=v1)
        await coll.upsert(records=[r1])

        assert mock_counter.increment.called
        assert mock_summary.observe.called

        await store.delete_collection("metrics_test")
