"""Tests for SQLiteVectorStore."""

import asyncio
import math
from datetime import UTC, datetime, timedelta, timezone
from uuid import UUID, uuid4

import pytest
import pytest_asyncio
from sqlalchemy import func, select, update
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from memmachine_server.common.data_types import SimilarityMetric
from memmachine_server.common.filter.filter_parser import (
    And,
    Comparison,
    In,
    Not,
    Or,
)
from memmachine_server.common.vector_store.data_types import (
    Record,
    VectorStoreCollectionAlreadyExistsError,
    VectorStoreCollectionConfig,
    VectorStoreCollectionConfigMismatchError,
)
from memmachine_server.common.vector_store.sqlite_vector_store import (
    IndexLoadError,
    SQLiteVectorStore,
    SQLiteVectorStoreCollection,
    SQLiteVectorStoreParams,
    _CollectionRow,
    _PendingOperationRow,
)
from memmachine_server.common.vector_store.vector_search_engine.usearch_engine import (
    USearchVectorSearchEngine,
)
from memmachine_server.common.vector_store.vector_search_engine.vector_search_engine import (
    VectorSearchEngine,
)

NAMESPACE = "test_namespace"
NAME = "test_name"
VECTOR_DIM = 3


def _normalize(vector: list[float]) -> list[float]:
    magnitude = math.sqrt(sum(x * x for x in vector))
    return [x / magnitude for x in vector]


def _make_record(
    *,
    uuid=None,
    vector: list[float] | None = None,
    properties: dict | None = None,
) -> Record:
    return Record(
        uuid=uuid or uuid4(),
        vector=vector,
        properties=properties,
    )


_RACE_WINDOW_SECONDS = 2.0


async def _wait_for(condition) -> None:
    """Poll `condition` until it holds, and give up quietly if it never does.

    Both outcomes are expected in the concurrency tests below. The interleaving
    each one sets up is what unfixed code does while another write is parked;
    once writes are serialized the second write cannot start at all, so the
    wait simply expires. The assertions are what tell the two cases apart.
    """
    # Polling, because what is being waited on is what another task has
    # committed to the database, which no in-process event tracks. The deadline
    # sits between polls rather than in a timeout around them: cancelling a
    # query mid-flight returns its connection to the pool without resetting it,
    # and the read transaction left open there blocks the next commit.
    loop = asyncio.get_running_loop()
    deadline = loop.time() + _RACE_WINDOW_SECONDS
    while loop.time() < deadline:
        if await condition():
            return
        await asyncio.sleep(0.01)


async def _record_exists(collection, record_uuid) -> bool:
    """Whether a concurrent upsert's transaction has committed yet."""
    return bool(await collection.get(record_uuids=[record_uuid]))


@pytest_asyncio.fixture
async def store(tmp_path):
    db_path = tmp_path / "test.db"
    engine = create_async_engine(f"sqlite+aiosqlite:///{db_path}")
    params = SQLiteVectorStoreParams(
        sqlalchemy_engine=engine,
        vector_search_engine_factory=lambda ndim, metric: USearchVectorSearchEngine(
            num_dimensions=ndim, similarity_metric=metric
        ),
    )
    vector_store = SQLiteVectorStore(params)
    await vector_store.startup()
    yield vector_store
    await vector_store.shutdown()
    await engine.dispose()


@pytest_asyncio.fixture
async def collection(store):
    await store.create_collection(
        namespace=NAMESPACE,
        name=NAME,
        config=VectorStoreCollectionConfig(
            vector_dimensions=VECTOR_DIM,
            similarity_metric=SimilarityMetric.COSINE,
            indexed_properties_schema={
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


# ── Collection lifecycle ──


class TestCollectionLifecycle:
    @pytest.mark.asyncio
    async def test_create_open_delete(self, store):
        await store.create_collection(
            namespace=NAMESPACE,
            name="lifecycle",
            config=VectorStoreCollectionConfig(vector_dimensions=VECTOR_DIM),
        )
        coll = await store.open_collection(namespace=NAMESPACE, name="lifecycle")
        assert isinstance(coll, SQLiteVectorStoreCollection)
        await store.delete_collection(namespace=NAMESPACE, name="lifecycle")

    @pytest.mark.asyncio
    async def test_open_returns_correct_type(self, store, collection):
        coll = await store.open_collection(namespace=NAMESPACE, name=NAME)
        assert isinstance(coll, SQLiteVectorStoreCollection)

    @pytest.mark.asyncio
    async def test_duplicate_name_raises(self, store, collection):
        with pytest.raises(VectorStoreCollectionAlreadyExistsError):
            await store.create_collection(
                namespace=NAMESPACE,
                name=NAME,
                config=VectorStoreCollectionConfig(
                    vector_dimensions=VECTOR_DIM,
                    similarity_metric=SimilarityMetric.COSINE,
                    indexed_properties_schema={
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
        config = VectorStoreCollectionConfig(vector_dimensions=VECTOR_DIM)
        coll = await store.open_or_create_collection(
            namespace=NAMESPACE, name="new", config=config
        )
        assert isinstance(coll, SQLiteVectorStoreCollection)
        await store.delete_collection(namespace=NAMESPACE, name="new")

    @pytest.mark.asyncio
    async def test_open_or_create_opens_when_exists(self, store):
        config = VectorStoreCollectionConfig(vector_dimensions=VECTOR_DIM)
        await store.create_collection(
            namespace=NAMESPACE, name="existing", config=config
        )
        coll = await store.open_or_create_collection(
            namespace=NAMESPACE, name="existing", config=config
        )
        assert isinstance(coll, SQLiteVectorStoreCollection)
        await store.delete_collection(namespace=NAMESPACE, name="existing")

    @pytest.mark.asyncio
    async def test_open_or_create_raises_on_config_mismatch(self, store):
        await store.create_collection(
            namespace=NAMESPACE,
            name="mismatch",
            config=VectorStoreCollectionConfig(vector_dimensions=VECTOR_DIM),
        )
        with pytest.raises(VectorStoreCollectionConfigMismatchError):
            await store.open_or_create_collection(
                namespace=NAMESPACE,
                name="mismatch",
                config=VectorStoreCollectionConfig(vector_dimensions=VECTOR_DIM + 1),
            )
        await store.delete_collection(namespace=NAMESPACE, name="mismatch")

    @pytest.mark.asyncio
    async def test_open_nonexistent_returns_none(self, store):
        assert await store.open_collection(namespace=NAMESPACE, name="nope") is None

    @pytest.mark.asyncio
    async def test_unsupported_metric_raises(self, store):
        with pytest.raises(ValueError, match="does not support"):
            await store.create_collection(
                namespace=NAMESPACE,
                name="bad_metric",
                config=VectorStoreCollectionConfig(
                    vector_dimensions=VECTOR_DIM,
                    similarity_metric=SimilarityMetric.MANHATTAN,
                ),
            )

    @pytest.mark.asyncio
    async def test_invalid_namespace_raises(self, store):
        with pytest.raises(ValueError, match="Invalid namespace"):
            await store.create_collection(
                namespace="INVALID",
                name="test",
                config=VectorStoreCollectionConfig(vector_dimensions=VECTOR_DIM),
            )

    @pytest.mark.asyncio
    async def test_invalid_name_raises(self, store):
        with pytest.raises(ValueError, match="Invalid namespace"):
            await store.create_collection(
                namespace="valid",
                name="INVALID",
                config=VectorStoreCollectionConfig(vector_dimensions=VECTOR_DIM),
            )


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

        query_results = await collection.query(query_vectors=[v1], limit=3)
        matches = query_results[0].matches

        assert len(matches) == 3
        assert matches[0].record.uuid == r1.uuid
        assert matches[0].score >= matches[1].score >= matches[2].score

    @pytest.mark.asyncio
    async def test_upsert_update(self, collection):
        v1 = _normalize([1.0, 0.0, 0.0])
        record = _make_record(vector=v1, properties={"name": "original"})
        await collection.upsert(records=[record])

        updated = Record(
            uuid=record.uuid,
            vector=_normalize([0.0, 1.0, 0.0]),
            properties={"name": "updated"},
        )
        await collection.upsert(records=[updated])

        results = await collection.get(record_uuids=[record.uuid])
        assert results[0].properties["name"] == "updated"

    @pytest.mark.asyncio
    async def test_query_with_similarity_threshold(self, collection):
        v1 = _normalize([1.0, 0.0, 0.0])
        v2 = _normalize([0.0, 1.0, 0.0])

        r1 = _make_record(vector=v1)
        r2 = _make_record(vector=v2)

        await collection.upsert(records=[r1, r2])

        query_results = await collection.query(
            query_vectors=[v1], limit=10, score_threshold=0.9
        )
        matches = query_results[0].matches

        assert len(matches) == 1
        assert matches[0].record.uuid == r1.uuid

    @pytest.mark.asyncio
    async def test_query_with_limit(self, collection):
        vectors = [_normalize([1.0, float(index) * 0.01, 0.0]) for index in range(5)]
        records = [_make_record(vector=vector) for vector in vectors]
        await collection.upsert(records=records)

        query_results = await collection.query(query_vectors=[vectors[0]], limit=2)
        assert len(query_results[0].matches) == 2

    @pytest.mark.asyncio
    async def test_query_return_vector_false(self, collection):
        v1 = _normalize([1.0, 0.0, 0.0])
        r1 = _make_record(vector=v1, properties={"name": "test"})
        await collection.upsert(records=[r1])

        query_results = await collection.query(
            query_vectors=[v1], limit=10, return_vector=False
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

        query_results = await collection.query(
            query_vectors=[v1],
            limit=10,
            return_vector=True,
            return_properties=False,
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

        all_results = await collection.query(query_vectors=[v1, v2], limit=1)

        assert len(all_results) == 2
        assert all_results[0].matches[0].record.uuid == r1.uuid
        assert all_results[1].matches[0].record.uuid == r2.uuid

    @pytest.mark.asyncio
    async def test_query_empty_vectors(self, collection):
        all_results = await collection.query(query_vectors=[], limit=10)
        assert len(all_results) == 0

    @pytest.mark.asyncio
    async def test_upsert_empty_records(self, collection):
        await collection.upsert(records=[])


# ── Filters ──


class TestFilters:
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

    async def _setup_floats(self, collection):
        vectors = [_normalize([1.0, float(index) * 0.01, 0.0]) for index in range(5)]
        scores = [-1.5, 0.0, 0.5, 1.5, 2.0]
        records = [
            _make_record(vector=vector, properties={"score": score})
            for vector, score in zip(vectors, scores, strict=True)
        ]
        await collection.upsert(records=records)
        return records, vectors[0]

    async def _setup_datetimes(self, collection):
        vectors = [_normalize([1.0, float(index) * 0.01, 0.0]) for index in range(5)]
        datetimes = [
            datetime(2024, 1, 1, tzinfo=UTC),
            datetime(2024, 3, 15, tzinfo=UTC),
            datetime(2024, 6, 1, tzinfo=UTC),
            datetime(2024, 9, 1, tzinfo=UTC),
            datetime(2024, 12, 31, tzinfo=UTC),
        ]
        records = [
            _make_record(
                vector=vector,
                properties={"name": f"r{index}", "created_at": dt},
            )
            for index, (vector, dt) in enumerate(zip(vectors, datetimes, strict=True))
        ]
        await collection.upsert(records=records)
        return records, vectors[0], datetimes

    async def _query(self, collection, query_vector, field, op, value):
        all_results = await collection.query(
            query_vectors=[query_vector],
            limit=10,
            property_filter=Comparison(field=field, op=op, value=value),
        )
        return {match.record.uuid for match in all_results[0].matches}

    # ── String / int ──

    @pytest.mark.asyncio
    async def test_eq_str(self, collection):
        r1, _r2, _r3, v1 = await self._setup(collection)
        uuids = await self._query(collection, v1, "name", "=", "alice")
        assert r1.uuid in uuids
        assert len(uuids) == 1

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
        uuids = await self._query(collection, v1, "age", ">", 30)
        assert len(uuids) == 1
        assert r3.uuid in uuids

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
        uuids = await self._query(collection, v1, "age", "<", 30)
        assert len(uuids) == 1
        assert r2.uuid in uuids

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
    async def test_eq_float(self, collection):
        records, query_vector = await self._setup_floats(collection)
        uuids = await self._query(collection, query_vector, "score", "=", 0.5)
        assert records[2].uuid in uuids
        assert len(uuids) == 1

    @pytest.mark.asyncio
    async def test_gt_float(self, collection):
        records, query_vector = await self._setup_floats(collection)
        uuids = await self._query(collection, query_vector, "score", ">", 0.5)
        assert records[3].uuid in uuids
        assert records[4].uuid in uuids
        assert records[2].uuid not in uuids

    @pytest.mark.asyncio
    async def test_lt_float(self, collection):
        records, query_vector = await self._setup_floats(collection)
        uuids = await self._query(collection, query_vector, "score", "<", 0.5)
        assert records[0].uuid in uuids
        assert records[1].uuid in uuids
        assert records[2].uuid not in uuids

    # ── Datetime ──

    @pytest.mark.asyncio
    async def test_datetime_roundtrip(self, collection):
        v1 = _normalize([1.0, 0.0, 0.0])
        dt = datetime(2024, 6, 15, 12, 30, 0, tzinfo=UTC)
        r1 = _make_record(vector=v1, properties={"name": "test", "created_at": dt})
        await collection.upsert(records=[r1])

        results = await collection.get(record_uuids=[r1.uuid])
        assert results[0].properties["created_at"] == dt

    @pytest.mark.asyncio
    async def test_eq_datetime(self, collection):
        records, query_vector, datetimes = await self._setup_datetimes(collection)
        uuids = await self._query(
            collection, query_vector, "created_at", "=", datetimes[2]
        )
        assert records[2].uuid in uuids
        assert len(uuids) == 1

    @pytest.mark.asyncio
    async def test_gt_datetime(self, collection):
        records, query_vector, datetimes = await self._setup_datetimes(collection)
        uuids = await self._query(
            collection, query_vector, "created_at", ">", datetimes[2]
        )
        assert records[3].uuid in uuids
        assert records[4].uuid in uuids
        assert records[2].uuid not in uuids

    @pytest.mark.asyncio
    async def test_lt_datetime(self, collection):
        records, query_vector, datetimes = await self._setup_datetimes(collection)
        uuids = await self._query(
            collection, query_vector, "created_at", "<", datetimes[2]
        )
        assert records[0].uuid in uuids
        assert records[1].uuid in uuids
        assert records[2].uuid not in uuids

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
    async def test_datetime_timezone_roundtrip(self, collection):
        """Original timezone is preserved through storage."""
        v1 = _normalize([1.0, 0.0, 0.0])
        est = timezone(timedelta(hours=-5))
        dt = datetime(2024, 6, 15, 7, 0, 0, tzinfo=est)
        r1 = _make_record(vector=v1, properties={"name": "tz", "created_at": dt})
        await collection.upsert(records=[r1])

        results = await collection.get(record_uuids=[r1.uuid])
        got = results[0].properties["created_at"]
        assert got == dt
        assert got.utcoffset() == timedelta(hours=-5)

    # ── In / And / Or / Not ──

    @pytest.mark.asyncio
    async def test_in(self, collection):
        r1, _r2, r3, v1 = await self._setup(collection)
        query_results = await collection.query(
            query_vectors=[v1],
            limit=10,
            property_filter=In(field="name", values=["alice", "carol"]),
        )
        uuids = {match.record.uuid for match in query_results[0].matches}
        assert r1.uuid in uuids
        assert r3.uuid in uuids
        assert len(uuids) == 2

    @pytest.mark.asyncio
    async def test_and(self, collection):
        _r1, _r2, r3, v1 = await self._setup(collection)
        query_results = await collection.query(
            query_vectors=[v1],
            limit=10,
            property_filter=And(
                left=Comparison(field="active", op="=", value=True),
                right=Comparison(field="age", op=">", value=30),
            ),
        )
        matches = query_results[0].matches
        assert len(matches) == 1
        assert matches[0].record.uuid == r3.uuid

    @pytest.mark.asyncio
    async def test_or(self, collection):
        r1, _r2, r3, v1 = await self._setup(collection)
        query_results = await collection.query(
            query_vectors=[v1],
            limit=10,
            property_filter=Or(
                left=Comparison(field="name", op="=", value="alice"),
                right=Comparison(field="name", op="=", value="carol"),
            ),
        )
        uuids = {match.record.uuid for match in query_results[0].matches}
        assert r1.uuid in uuids
        assert r3.uuid in uuids
        assert len(uuids) == 2

    @pytest.mark.asyncio
    async def test_not(self, collection):
        r1, r2, _r3, v1 = await self._setup(collection)
        query_results = await collection.query(
            query_vectors=[v1],
            limit=10,
            property_filter=Not(expr=Comparison(field="age", op=">", value=30)),
        )
        uuids = {match.record.uuid for match in query_results[0].matches}
        assert r1.uuid in uuids
        assert r2.uuid in uuids
        assert len(uuids) == 2


# ── Get ──


class TestGet:
    @pytest.mark.asyncio
    async def test_get_by_uuids(self, collection):
        v1 = _normalize([1.0, 0.0, 0.0])
        v2 = _normalize([0.0, 1.0, 0.0])

        r1 = _make_record(vector=v1, properties={"name": "a"})
        r2 = _make_record(vector=v2, properties={"name": "b"})
        await collection.upsert(records=[r1, r2])

        results = await collection.get(record_uuids=[r2.uuid, r1.uuid])
        assert len(results) == 2
        assert results[0].uuid == r2.uuid
        assert results[1].uuid == r1.uuid

    @pytest.mark.asyncio
    async def test_get_missing_uuids(self, collection):
        v1 = _normalize([1.0, 0.0, 0.0])
        r1 = _make_record(vector=v1)
        await collection.upsert(records=[r1])

        missing = uuid4()
        results = await collection.get(record_uuids=[r1.uuid, missing])
        assert len(results) == 1
        assert results[0].uuid == r1.uuid

    @pytest.mark.asyncio
    async def test_get_empty_list(self, collection):
        results = await collection.get(record_uuids=[])
        assert len(results) == 0

    @pytest.mark.asyncio
    async def test_get_return_vector_false(self, collection):
        v1 = _normalize([1.0, 0.0, 0.0])
        r1 = _make_record(vector=v1, properties={"name": "test"})
        await collection.upsert(records=[r1])

        results = await collection.get(record_uuids=[r1.uuid], return_vector=False)
        assert results[0].vector is None
        assert results[0].properties is not None

    @pytest.mark.asyncio
    async def test_get_return_properties_false(self, collection):
        v1 = _normalize([1.0, 0.0, 0.0])
        r1 = _make_record(vector=v1, properties={"name": "test"})
        await collection.upsert(records=[r1])

        results = await collection.get(
            record_uuids=[r1.uuid], return_vector=True, return_properties=False
        )
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

        results = await collection.get(record_uuids=[r1.uuid, r2.uuid])
        assert len(results) == 1
        assert results[0].uuid == r2.uuid

    @pytest.mark.asyncio
    async def test_delete_empty_list(self, collection):
        await collection.delete(record_uuids=[])

    @pytest.mark.asyncio
    async def test_delete_nonexistent_uuid(self, collection):
        await collection.delete(record_uuids=[uuid4()])

    @pytest.mark.asyncio
    async def test_delete_removes_from_query(self, collection):
        v1 = _normalize([1.0, 0.0, 0.0])
        r1 = _make_record(vector=v1)
        await collection.upsert(records=[r1])
        await collection.delete(record_uuids=[r1.uuid])

        query_results = await collection.query(query_vectors=[v1], limit=10)
        assert len(query_results[0].matches) == 0


# ── Partition isolation ──


class TestPartitionIsolation:
    @pytest.mark.asyncio
    async def test_query_only_returns_own_collection(self, store):
        config = VectorStoreCollectionConfig(vector_dimensions=VECTOR_DIM)
        await store.create_collection(
            namespace=NAMESPACE, name="tenant_a", config=config
        )
        await store.create_collection(
            namespace=NAMESPACE, name="tenant_b", config=config
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

        results_a = await coll_a.query(query_vectors=[v1], limit=10)
        results_b = await coll_b.query(query_vectors=[v1], limit=10)

        uuids_a = {match.record.uuid for match in results_a[0].matches}
        uuids_b = {match.record.uuid for match in results_b[0].matches}
        assert uuids_a == {r1.uuid}
        assert uuids_b == {r2.uuid}

        await store.delete_collection(namespace=NAMESPACE, name="tenant_a")
        await store.delete_collection(namespace=NAMESPACE, name="tenant_b")

    @pytest.mark.asyncio
    async def test_get_only_returns_own_collection(self, store):
        config = VectorStoreCollectionConfig(vector_dimensions=VECTOR_DIM)
        await store.create_collection(
            namespace=NAMESPACE, name="tenant_a", config=config
        )
        await store.create_collection(
            namespace=NAMESPACE, name="tenant_b", config=config
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

        results = await coll_a.get(record_uuids=[r1.uuid, r2.uuid])
        assert len(results) == 1
        assert results[0].uuid == r1.uuid

        await store.delete_collection(namespace=NAMESPACE, name="tenant_a")
        await store.delete_collection(namespace=NAMESPACE, name="tenant_b")

    @pytest.mark.asyncio
    async def test_delete_only_affects_own_collection(self, store):
        config = VectorStoreCollectionConfig(vector_dimensions=VECTOR_DIM)
        await store.create_collection(
            namespace=NAMESPACE, name="tenant_a", config=config
        )
        await store.create_collection(
            namespace=NAMESPACE, name="tenant_b", config=config
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

        await coll_a.delete(record_uuids=[r2.uuid])

        results = await coll_b.get(record_uuids=[r2.uuid])
        assert len(results) == 1
        assert results[0].uuid == r2.uuid

        await store.delete_collection(namespace=NAMESPACE, name="tenant_a")
        await store.delete_collection(namespace=NAMESPACE, name="tenant_b")

    @pytest.mark.asyncio
    async def test_namespace_isolation(self, store):
        config = VectorStoreCollectionConfig(vector_dimensions=VECTOR_DIM)
        await store.create_collection(
            namespace="namespace_a", name="coll", config=config
        )
        await store.create_collection(
            namespace="namespace_b", name="coll", config=config
        )
        coll_a = await store.open_collection(namespace="namespace_a", name="coll")
        coll_b = await store.open_collection(namespace="namespace_b", name="coll")
        assert coll_a is not None
        assert coll_b is not None

        v1 = _normalize([1.0, 0.0, 0.0])
        r1 = _make_record(vector=v1)
        r2 = _make_record(vector=v1)

        await coll_a.upsert(records=[r1])
        await coll_b.upsert(records=[r2])

        results_a = await coll_a.query(query_vectors=[v1], limit=10)
        assert {match.record.uuid for match in results_a[0].matches} == {r1.uuid}

        await store.delete_collection(namespace="namespace_a", name="coll")
        await store.delete_collection(namespace="namespace_b", name="coll")

    @pytest.mark.asyncio
    async def test_delete_collection_does_not_affect_sibling(self, store):
        """Deleting one collection doesn't break a sibling sharing tables."""
        config = VectorStoreCollectionConfig(vector_dimensions=VECTOR_DIM)
        coll_a = await store.open_or_create_collection(
            namespace=NAMESPACE, name="sibling_a", config=config
        )
        coll_b = await store.open_or_create_collection(
            namespace=NAMESPACE, name="sibling_b", config=config
        )

        v1 = _normalize([1.0, 0.0, 0.0])
        r1 = _make_record(vector=v1)
        r2 = _make_record(vector=v1)
        await coll_a.upsert(records=[r1])
        await coll_b.upsert(records=[r2])

        await store.delete_collection(namespace=NAMESPACE, name="sibling_a")

        results = await coll_b.query(query_vectors=[v1], limit=10)
        assert len(results[0].matches) == 1
        assert results[0].matches[0].record.uuid == r2.uuid

        await store.delete_collection(namespace=NAMESPACE, name="sibling_b")


# ── Euclidean metric ──


class TestEuclideanMetric:
    @pytest.mark.asyncio
    async def test_euclidean_ordering(self, store):
        config = VectorStoreCollectionConfig(
            vector_dimensions=2,
            similarity_metric=SimilarityMetric.EUCLIDEAN,
        )
        coll = await store.open_or_create_collection(
            namespace=NAMESPACE, name="euclidean", config=config
        )
        r1 = _make_record(vector=[0.0, 0.0])
        r2 = _make_record(vector=[3.0, 4.0])
        await coll.upsert(records=[r1, r2])

        results = await coll.query(query_vectors=[[0.0, 0.0]], limit=2)
        # Euclidean: lower distance = better match; best match is first
        assert results[0].matches[0].score < results[0].matches[1].score

        await store.delete_collection(namespace=NAMESPACE, name="euclidean")


# ── No-properties collection ──


class TestNoProperties:
    @pytest.mark.asyncio
    async def test_collection_without_properties(self, store):
        config = VectorStoreCollectionConfig(vector_dimensions=2)
        coll = await store.open_or_create_collection(
            namespace=NAMESPACE, name="no_props", config=config
        )
        r1 = _make_record(vector=[1.0, 0.0])
        await coll.upsert(records=[r1])

        results = await coll.query(query_vectors=[[1.0, 0.0]], limit=1)
        assert len(results[0].matches) == 1

        await store.delete_collection(namespace=NAMESPACE, name="no_props")


# ── USearch-specific: dot product metric ──


class TestDotProductMetric:
    @pytest.mark.asyncio
    async def test_dot_product_supported(self, store):
        """Dot product is supported by USearch but not sqlite-vec."""
        config = VectorStoreCollectionConfig(
            vector_dimensions=VECTOR_DIM,
            similarity_metric=SimilarityMetric.DOT,
        )
        coll = await store.open_or_create_collection(
            namespace=NAMESPACE, name="dot", config=config
        )
        v1 = _normalize([1.0, 0.0, 0.0])
        v2 = _normalize([0.0, 1.0, 0.0])
        r1 = _make_record(vector=v1)
        r2 = _make_record(vector=v2)
        await coll.upsert(records=[r1, r2])

        results = await coll.query(query_vectors=[v1], limit=2)
        assert len(results[0].matches) == 2

        await store.delete_collection(namespace=NAMESPACE, name="dot")


# ── Input validation ──


class TestInputValidation:
    @pytest.mark.asyncio
    async def test_upsert_rejects_none_vector(self, collection):
        record = _make_record(vector=None)
        with pytest.raises(ValueError, match="vector=None"):
            await collection.upsert(records=[record])

    @pytest.mark.asyncio
    async def test_upsert_none_properties_treated_as_empty(self, collection):
        v1 = _normalize([1.0, 0.0, 0.0])
        record = _make_record(vector=v1, properties=None)
        await collection.upsert(records=[record])
        fetched = await collection.get(
            record_uuids=[record.uuid], return_properties=True
        )
        assert len(fetched) == 1
        assert fetched[0].properties == {}


# ── Score semantics ──


class TestScoreSemantics:
    @pytest.mark.asyncio
    async def test_cosine_higher_is_better(self, collection):
        v1 = _normalize([1.0, 0.0, 0.0])
        v2 = _normalize([0.0, 1.0, 0.0])
        r1 = _make_record(vector=v1)
        r2 = _make_record(vector=v2)
        await collection.upsert(records=[r1, r2])

        results = await collection.query(query_vectors=[v1], limit=2)
        scores = [m.score for m in results[0].matches]
        assert scores[0] > scores[1]

    @pytest.mark.asyncio
    async def test_euclidean_lower_is_better(self, store):
        config = VectorStoreCollectionConfig(
            vector_dimensions=2,
            similarity_metric=SimilarityMetric.EUCLIDEAN,
        )
        coll = await store.open_or_create_collection(
            namespace=NAMESPACE, name="euclidean_score", config=config
        )
        r1 = _make_record(vector=[0.0, 0.0])
        r2 = _make_record(vector=[3.0, 4.0])
        await coll.upsert(records=[r1, r2])

        results = await coll.query(query_vectors=[[0.0, 0.0]], limit=2)
        scores = [m.score for m in results[0].matches]
        assert scores[0] < scores[1]
        assert scores[0] == pytest.approx(0.0, abs=0.01)
        assert scores[1] == pytest.approx(5.0, abs=0.01)

        await store.delete_collection(namespace=NAMESPACE, name="euclidean_score")


# ── Upsert behavior ──


class TestUpsertBehavior:
    @pytest.mark.asyncio
    async def test_upsert_replaces_vector_and_properties(self, collection):
        v1 = _normalize([1.0, 0.0, 0.0])
        v2 = _normalize([0.0, 1.0, 0.0])
        record_uuid = uuid4()

        await collection.upsert(
            records=[
                _make_record(uuid=record_uuid, vector=v1, properties={"name": "alice"})
            ]
        )
        await collection.upsert(
            records=[
                _make_record(uuid=record_uuid, vector=v2, properties={"name": "bob"})
            ]
        )

        fetched = await collection.get(
            record_uuids=[record_uuid], return_vector=True, return_properties=True
        )
        assert len(fetched) == 1
        assert fetched[0].properties["name"] == "bob"

        results = await collection.query(query_vectors=[v2], limit=1)
        assert results[0].matches[0].record.uuid == record_uuid
        assert results[0].matches[0].score == pytest.approx(1.0, abs=0.01)


# ── Concurrent async behavior ──


class TestConcurrentAsync:
    @pytest.mark.asyncio
    async def test_concurrent_upserts(self, collection):
        """Multiple concurrent upserts should not error and all records should be persisted."""
        import asyncio

        all_uuids: list[UUID] = []

        async def upsert_batch(start: int) -> None:
            records = [
                _make_record(vector=_normalize([float(i), 1.0, 0.0]))
                for i in range(start, start + 10)
            ]
            all_uuids.extend(r.uuid for r in records)
            await collection.upsert(records=records)

        await asyncio.gather(
            upsert_batch(0),
            upsert_batch(10),
            upsert_batch(20),
        )

        # Verify all records were persisted (deterministic, not ANN-dependent).
        fetched = await collection.get(record_uuids=all_uuids)
        assert len(fetched) == 30

    @pytest.mark.asyncio
    async def test_concurrent_upsert_and_query(self, collection):
        """Query during upsert should not error (eventual consistency)."""
        import asyncio

        records = [
            _make_record(vector=_normalize([float(i), 1.0, 0.0])) for i in range(20)
        ]
        await collection.upsert(records=records)

        async def query_loop() -> None:
            for _ in range(5):
                await collection.query(
                    query_vectors=[_normalize([1.0, 1.0, 0.0])], limit=10
                )

        async def upsert_more() -> None:
            more_records = [
                _make_record(vector=_normalize([float(i), 0.0, 1.0]))
                for i in range(20, 30)
            ]
            await collection.upsert(records=more_records)

        await asyncio.gather(query_loop(), upsert_more())


# ── row_id reuse (issue #1468) ──


class _GatedSearchEngine(VectorSearchEngine):
    """Delegates to a real engine; the next search() waits after scoring.

    query() scores keys in the engine and then resolves the winners to record
    rows in a separate step, holding nothing in between -- readers do not
    serialize with writers. The gate parks in that gap, which is where a write
    can retire the row a returned key referred to.
    """

    def __init__(self, inner: VectorSearchEngine) -> None:
        self.inner = inner
        self.gate: asyncio.Event | None = None
        self.gate_reached = asyncio.Event()

    async def add(self, vectors):
        await self.inner.add(vectors)

    async def remove(self, keys):
        await self.inner.remove(keys)

    async def search(self, vectors, *, limit, allowed_keys=None):
        results = await self.inner.search(
            vectors, limit=limit, allowed_keys=allowed_keys
        )
        if self.gate is not None:
            gate, self.gate = self.gate, None
            self.gate_reached.set()
            await gate.wait()
        return results

    async def get_vectors(self, keys):
        return await self.inner.get_vectors(keys)

    async def save(self, path):
        await self.inner.save(path)

    async def load(self, path):
        await self.inner.load(path)


class TestRowIdReuse:
    """Regression tests for row_id reuse (issue #1468).

    Without AUTOINCREMENT, SQLite assigns max(rowid) + 1, so deleting the
    record holding the maximum row_id frees that id for the very next insert,
    and a record can inherit the id another was being served under.

    Serializing writes closes that on the write path but not on the read one:
    a query scores keys in the engine and resolves them to rows afterwards,
    holding nothing in between, so what these tests pin is the id policy
    itself.
    """

    async def _row_id_of(self, collection, record_uuid):
        async with collection._create_session() as session:
            return (
                await session.execute(
                    select(collection._records_table.c.row_id).where(
                        collection._records_table.c.uuid == record_uuid
                    )
                )
            ).scalar_one()

    @pytest.mark.asyncio
    async def test_row_ids_are_never_reused(self, collection):
        """A new record must not be assigned a previously deleted row_id."""
        record_a = _make_record(vector=_normalize([1.0, 0.0, 0.0]))
        await collection.upsert(records=[record_a])
        row_id_a = await self._row_id_of(collection, record_a.uuid)

        await collection.delete(record_uuids=[record_a.uuid])

        record_b = _make_record(vector=_normalize([0.0, 1.0, 0.0]))
        await collection.upsert(records=[record_b])
        row_id_b = await self._row_id_of(collection, record_b.uuid)

        assert row_id_b != row_id_a

    @pytest.mark.asyncio
    async def test_a_query_cannot_return_a_record_it_never_scored(self, tmp_path):
        """A key the engine scored must never resolve to a later record.

        query() scores keys and then looks the winners up by row_id. Writes
        run freely in between, so a reused row_id would let a record that was
        never scored come back wearing the score of the record that was --
        here, a match orthogonal to the query returned as a perfect hit. With
        ids never reused the stale key matches no row and is simply dropped.
        """
        db_path = tmp_path / "test.db"
        engine = create_async_engine(f"sqlite+aiosqlite:///{db_path}")
        gated_engines: list[_GatedSearchEngine] = []

        def factory(ndim, metric):
            gated = _GatedSearchEngine(
                USearchVectorSearchEngine(num_dimensions=ndim, similarity_metric=metric)
            )
            gated_engines.append(gated)
            return gated

        store = SQLiteVectorStore(
            SQLiteVectorStoreParams(
                sqlalchemy_engine=engine,
                vector_search_engine_factory=factory,
            )
        )
        await store.startup()
        try:
            collection = await store.open_or_create_collection(
                namespace=NAMESPACE,
                name=NAME,
                config=VectorStoreCollectionConfig(vector_dimensions=VECTOR_DIM),
            )
            (gated_engine,) = gated_engines

            scored = _make_record(vector=_normalize([1.0, 0.0, 0.0]))
            await collection.upsert(records=[scored])

            # The query parks holding the key it scored, which is the highest
            # row_id in the table and so the one a reused id would hand out.
            gate = asyncio.Event()
            gated_engine.gate = gate
            query_task = asyncio.create_task(
                collection.query(query_vectors=[_normalize([1.0, 0.0, 0.0])], limit=1)
            )
            await gated_engine.gate_reached.wait()

            await collection.delete(record_uuids=[scored.uuid])
            successor = _make_record(vector=_normalize([0.0, 1.0, 0.0]))
            await collection.upsert(records=[successor])

            gate.set()
            results = await query_task

            assert [match.record.uuid for match in results[0].matches] == [], (
                "a record the engine never scored was returned"
            )
        finally:
            await store.shutdown()
            await engine.dispose()


# ── Crash recovery & pending operations ──


def _engine_factory(ndim, metric):
    return USearchVectorSearchEngine(num_dimensions=ndim, similarity_metric=metric)


CONFIG = VectorStoreCollectionConfig(
    vector_dimensions=VECTOR_DIM,
    similarity_metric=SimilarityMetric.COSINE,
)


async def _fresh_store(db_path, tmp_path, *, save_threshold=1000):
    """Create a new SQLiteVectorStore against the same DB file."""
    engine = create_async_engine(f"sqlite+aiosqlite:///{db_path}")
    params = SQLiteVectorStoreParams(
        sqlalchemy_engine=engine,
        vector_search_engine_factory=_engine_factory,
        index_directory=str(tmp_path / "indexes"),
        save_threshold=save_threshold,
    )
    store = SQLiteVectorStore(params)
    await store.startup()
    return store, engine


async def _pending_operation_count(engine) -> int:
    """Count all rows in the pending operations table."""
    session_factory = async_sessionmaker(engine, expire_on_commit=False)
    async with session_factory() as session:
        return (
            await session.execute(
                select(func.count()).select_from(_PendingOperationRow)
            )
        ).scalar_one()


async def _set_all_pending_operations_unapplied(engine) -> None:
    """Mark all pending operations as unapplied (simulates crash before engine apply)."""
    session_factory = async_sessionmaker(engine, expire_on_commit=False)
    async with session_factory() as session, session.begin():
        await session.execute(update(_PendingOperationRow).values(applied=False))


class TestCrashRecovery:
    """Tests for pending operations replay on startup."""

    @pytest.mark.asyncio
    async def test_replay_upserts_after_crash(self, tmp_path):
        """Records upserted before crash are queryable after restart."""
        db_path = tmp_path / "test.db"
        store1, engine1 = await _fresh_store(db_path, tmp_path)

        coll = await store1.open_or_create_collection(
            namespace=NAMESPACE, name=NAME, config=CONFIG
        )
        records = [
            _make_record(vector=_normalize([1.0, 0.0, 0.0])),
            _make_record(vector=_normalize([0.0, 1.0, 0.0])),
        ]
        await coll.upsert(records=records)

        # Simulate crash: dispose without shutdown (pending ops remain).
        await engine1.dispose()

        # Restart with fresh in-memory engines.
        store2, engine2 = await _fresh_store(db_path, tmp_path)
        coll2 = await store2.open_collection(namespace=NAMESPACE, name=NAME)
        assert coll2 is not None

        results = await coll2.query(
            query_vectors=[_normalize([1.0, 0.0, 0.0])], limit=10
        )
        assert len(results[0].matches) == 2

        await store2.shutdown()
        await engine2.dispose()

    @pytest.mark.asyncio
    async def test_replay_unapplied_upserts(self, tmp_path):
        """Unapplied pending upserts (crash before engine apply) are replayed."""
        db_path = tmp_path / "test.db"
        store1, engine1 = await _fresh_store(db_path, tmp_path)

        coll = await store1.open_or_create_collection(
            namespace=NAMESPACE, name=NAME, config=CONFIG
        )
        records = [
            _make_record(vector=_normalize([1.0, 0.0, 0.0])),
            _make_record(vector=_normalize([0.0, 1.0, 0.0])),
        ]
        await coll.upsert(records=records)

        # Simulate crash between SQLite commit and engine apply:
        # mark all pending ops as unapplied.
        await _set_all_pending_operations_unapplied(engine1)
        await engine1.dispose()

        # Restart: replay should re-apply unapplied ops to the engine.
        store2, engine2 = await _fresh_store(db_path, tmp_path)
        coll2 = await store2.open_collection(namespace=NAMESPACE, name=NAME)
        assert coll2 is not None

        results = await coll2.query(
            query_vectors=[_normalize([1.0, 0.0, 0.0])], limit=10
        )
        assert len(results[0].matches) == 2

        await store2.shutdown()
        await engine2.dispose()

    @pytest.mark.asyncio
    async def test_replay_deletes_after_crash(self, tmp_path):
        """Pending delete operations are replayed on restart."""
        db_path = tmp_path / "test.db"
        store1, engine1 = await _fresh_store(db_path, tmp_path)

        coll = await store1.open_or_create_collection(
            namespace=NAMESPACE, name=NAME, config=CONFIG
        )
        r1 = _make_record(vector=_normalize([1.0, 0.0, 0.0]))
        r2 = _make_record(vector=_normalize([0.0, 1.0, 0.0]))
        await coll.upsert(records=[r1, r2])
        await coll.delete(record_uuids=[r1.uuid])

        # Simulate crash.
        await engine1.dispose()

        store2, engine2 = await _fresh_store(db_path, tmp_path)
        coll2 = await store2.open_collection(namespace=NAMESPACE, name=NAME)
        assert coll2 is not None

        results = await coll2.query(
            query_vectors=[_normalize([1.0, 0.0, 0.0])], limit=10
        )
        assert len(results[0].matches) == 1

        fetched = await coll2.get(record_uuids=[r1.uuid])
        assert len(fetched) == 0

        await store2.shutdown()
        await engine2.dispose()

    @pytest.mark.asyncio
    async def test_replay_mixed_upserts_and_deletes(self, tmp_path):
        """Mixed upsert and delete pending ops are replayed correctly."""
        db_path = tmp_path / "test.db"
        store1, engine1 = await _fresh_store(db_path, tmp_path)

        coll = await store1.open_or_create_collection(
            namespace=NAMESPACE, name=NAME, config=CONFIG
        )
        r1 = _make_record(vector=_normalize([1.0, 0.0, 0.0]))
        r2 = _make_record(vector=_normalize([0.0, 1.0, 0.0]))
        r3 = _make_record(vector=_normalize([0.0, 0.0, 1.0]))
        await coll.upsert(records=[r1, r2, r3])
        await coll.delete(record_uuids=[r2.uuid])

        # Simulate crash.
        await engine1.dispose()

        store2, engine2 = await _fresh_store(db_path, tmp_path)
        coll2 = await store2.open_collection(namespace=NAMESPACE, name=NAME)
        assert coll2 is not None

        results = await coll2.query(
            query_vectors=[_normalize([1.0, 0.0, 0.0])], limit=10
        )
        assert len(results[0].matches) == 2

        fetched = await coll2.get(record_uuids=[r1.uuid, r2.uuid, r3.uuid])
        fetched_uuids = {r.uuid for r in fetched}
        assert r1.uuid in fetched_uuids
        assert r2.uuid not in fetched_uuids
        assert r3.uuid in fetched_uuids

        await store2.shutdown()
        await engine2.dispose()

    @pytest.mark.asyncio
    async def test_save_threshold_clears_applied_ops(self, tmp_path):
        """Applied pending ops are deleted after save threshold is reached."""
        db_path = tmp_path / "test.db"
        store1, engine1 = await _fresh_store(db_path, tmp_path, save_threshold=2)

        coll = await store1.open_or_create_collection(
            namespace=NAMESPACE, name=NAME, config=CONFIG
        )

        # Upsert 1 record: below threshold, pending op should remain.
        r1 = _make_record(vector=_normalize([1.0, 0.0, 0.0]))
        await coll.upsert(records=[r1])
        assert await _pending_operation_count(engine1) == 1

        # Upsert 1 more: reaches threshold of 2, should trigger save + cleanup.
        r2 = _make_record(vector=_normalize([0.0, 1.0, 0.0]))
        await coll.upsert(records=[r2])
        assert await _pending_operation_count(engine1) == 0

        await store1.shutdown()
        await engine1.dispose()

    @pytest.mark.asyncio
    async def test_cascade_deletes_pending_ops(self, tmp_path):
        """Deleting a collection cascades to its pending operations."""
        db_path = tmp_path / "test.db"
        store1, engine1 = await _fresh_store(db_path, tmp_path)

        coll = await store1.open_or_create_collection(
            namespace=NAMESPACE, name=NAME, config=CONFIG
        )
        records = [
            _make_record(vector=_normalize([1.0, 0.0, 0.0])),
            _make_record(vector=_normalize([0.0, 1.0, 0.0])),
        ]
        await coll.upsert(records=records)
        assert await _pending_operation_count(engine1) == 2

        await store1.delete_collection(namespace=NAMESPACE, name=NAME)
        assert await _pending_operation_count(engine1) == 0

        await store1.shutdown()
        await engine1.dispose()

    @pytest.mark.asyncio
    async def test_require_started(self, tmp_path):
        """Store methods raise RuntimeError before startup."""
        db_path = tmp_path / "test.db"
        engine = create_async_engine(f"sqlite+aiosqlite:///{db_path}")
        store = SQLiteVectorStore(
            SQLiteVectorStoreParams(
                sqlalchemy_engine=engine,
                vector_search_engine_factory=_engine_factory,
            )
        )

        with pytest.raises(RuntimeError, match="startup"):
            await store.open_collection(namespace=NAMESPACE, name=NAME)

        with pytest.raises(RuntimeError, match="startup"):
            await store.create_collection(namespace=NAMESPACE, name=NAME, config=CONFIG)

        with pytest.raises(RuntimeError, match="startup"):
            await store.delete_collection(namespace=NAMESPACE, name=NAME)

        await engine.dispose()


# ── Concurrent write ordering (issue #1468) ──


class _GatedRemoveEngine(VectorSearchEngine):
    """Delegates to a real engine; the next remove() waits on `gate` first.

    delete() already suspends at its engine remove() (an await point after
    the SQL commit); the gate only widens that window so the interleaving
    with a concurrent upsert() is deterministic.
    """

    def __init__(self, inner: VectorSearchEngine) -> None:
        self.inner = inner
        self.gate: asyncio.Event | None = None
        self.gate_reached = asyncio.Event()

    async def add(self, vectors):
        await self.inner.add(vectors)

    async def remove(self, keys):
        if self.gate is not None:
            gate, self.gate = self.gate, None
            self.gate_reached.set()
            await gate.wait()
        await self.inner.remove(keys)

    async def search(self, vectors, *, limit, allowed_keys=None):
        return await self.inner.search(vectors, limit=limit, allowed_keys=allowed_keys)

    async def get_vectors(self, keys):
        return await self.inner.get_vectors(keys)

    async def save(self, path):
        await self.inner.save(path)

    async def load(self, path):
        await self.inner.load(path)


class _GatedSaveEngine(VectorSearchEngine):
    """Delegates to a real engine; the first save waits after writing the index.

    A save writes the index and then trims the pending operations the index now
    holds. Parking between the two widens the window in which another write can
    apply to the engine: too late to be in the file just written, early enough
    for the trim to delete the log row that was its only other copy.

    Saves after that one park before writing, so a test can reach the crash it
    is about without a later save publishing what it is trying to observe.
    """

    def __init__(self, inner: VectorSearchEngine) -> None:
        self.inner = inner
        self.gate: asyncio.Event | None = None
        self.gate_reached = asyncio.Event()
        self.saves_blocked = False
        self.blocked_save_reached = asyncio.Event()

    async def add(self, vectors):
        await self.inner.add(vectors)

    async def remove(self, keys):
        await self.inner.remove(keys)

    async def search(self, vectors, *, limit, allowed_keys=None):
        return await self.inner.search(vectors, limit=limit, allowed_keys=allowed_keys)

    async def get_vectors(self, keys):
        return await self.inner.get_vectors(keys)

    async def save(self, path):
        if self.gate is None:
            if self.saves_blocked:
                self.blocked_save_reached.set()
                await asyncio.Event().wait()
            await self.inner.save(path)
            return

        gate, self.gate = self.gate, None
        self.saves_blocked = True
        await self.inner.save(path)
        self.gate_reached.set()
        await gate.wait()

    async def load(self, path):
        await self.inner.load(path)


async def _wrapped_engine_store(db_path, tmp_path, wrap, *, save_threshold=1000):
    """Create a store whose engines are wrapped by `wrap`, and collect them."""
    engine = create_async_engine(f"sqlite+aiosqlite:///{db_path}")
    wrapped: list = []

    def factory(ndim, metric):
        gated = wrap(_engine_factory(ndim, metric))
        wrapped.append(gated)
        return gated

    store = SQLiteVectorStore(
        SQLiteVectorStoreParams(
            sqlalchemy_engine=engine,
            vector_search_engine_factory=factory,
            index_directory=str(tmp_path / "indexes"),
            save_threshold=save_threshold,
        )
    )
    await store.startup()
    return store, engine, wrapped


async def _a_delete_has_been_applied(engine) -> bool:
    """Whether a concurrent delete has reached the search engine.

    Its pending row is marked applied last, so this is true only once the
    delete has committed and its engine removal has finished -- the whole of
    the operation the upsert parked behind it has to overtake.
    """
    session_factory = async_sessionmaker(engine, expire_on_commit=False)
    async with session_factory() as session:
        applied_deletes = (
            await session.execute(
                select(func.count())
                .select_from(_PendingOperationRow)
                .where(
                    _PendingOperationRow.operation_type == "delete",
                    _PendingOperationRow.applied.is_(True),
                )
            )
        ).scalar_one()
    return applied_deletes > 0


async def _both_writes_applied(engine) -> bool:
    """Whether a second write has reached the engine behind a parked save."""
    session_factory = async_sessionmaker(engine, expire_on_commit=False)
    async with session_factory() as session:
        applied = (
            await session.execute(
                select(func.count())
                .select_from(_PendingOperationRow)
                .where(_PendingOperationRow.applied.is_(True))
            )
        ).scalar_one()
    return applied == 2


class TestConcurrentWriteOrdering:
    """The engine must see a collection's writes in the order SQLite committed.

    Every write commits its SQL transaction before applying to the search
    engine, so two writers that overlap can reach the engine in the opposite
    order to the one they committed in. Never reusing a row_id (the fix above)
    does not help here: both writes address one uuid, and an upsert of an
    existing uuid keeps its row_id by design.
    """

    @pytest.mark.asyncio
    async def test_an_upsert_survives_a_delete_of_another_record(self, tmp_path):
        """A delete must not carry an unrelated concurrent upsert with it.

        Interleaving: delete(A) commits and parks at its engine removal;
        upsert(B) lands in that window; delete(A) resumes and removes the
        row_id it was given. B has to still be searchable -- which it is
        because it was never given A's row_id, and, now, because it cannot
        run inside that window in the first place.
        """
        db_path = tmp_path / "test.db"
        store, engine, wrapped = await _wrapped_engine_store(
            db_path, tmp_path, _GatedRemoveEngine
        )
        try:
            coll = await store.open_or_create_collection(
                namespace=NAMESPACE, name=NAME, config=CONFIG
            )
            (gated_engine,) = wrapped

            deleted = _make_record(vector=_normalize([1.0, 0.0, 0.0]))
            await coll.upsert(records=[deleted])

            gate = asyncio.Event()
            gated_engine.gate = gate
            delete_task = asyncio.create_task(coll.delete(record_uuids=[deleted.uuid]))
            await gated_engine.gate_reached.wait()

            # A task, not an await: unfixed code runs this to completion
            # inside the window above, serialized writes make it wait for the
            # delete, and awaiting it here would wait forever.
            upserted = _make_record(vector=_normalize([0.0, 1.0, 0.0]))
            upsert_task = asyncio.create_task(coll.upsert(records=[upserted]))
            await _wait_for(lambda: _record_exists(coll, upserted.uuid))

            gate.set()
            await asyncio.gather(delete_task, upsert_task)

            results = await coll.query(
                query_vectors=[_normalize([0.0, 1.0, 0.0])], limit=5
            )
            assert upserted.uuid in {
                match.record.uuid for match in results[0].matches
            }, "upserted record lost from the search engine"
        finally:
            await store.shutdown()
            await engine.dispose()

    @pytest.mark.asyncio
    async def test_an_upsert_cannot_overtake_a_delete_of_the_same_uuid(self, tmp_path):
        """A vector re-added after its record is deleted belongs to nothing.

        Interleaving: upsert(U) commits and parks at its engine apply;
        delete(U) commits and applies inside that window; upsert(U) resumes and
        adds its vector back. SQLite says the record is gone, so the vector
        left in the engine resolves to no row: it is dropped from every result
        it wins -- costing exactly the slots it takes -- and the next save
        publishes it into the index for good.
        """
        db_path = tmp_path / "test.db"
        store, engine, wrapped = await _wrapped_engine_store(
            db_path, tmp_path, _GatedRemoveEngine
        )
        try:
            coll = await store.open_or_create_collection(
                namespace=NAMESPACE, name=NAME, config=CONFIG
            )
            (gated_engine,) = wrapped

            # The record the query should return once the racers are done.
            decoy = _make_record(vector=_normalize([1.0, 1.0, 0.0]))
            racer = _make_record(vector=_normalize([0.0, 1.0, 0.0]))
            await coll.upsert(records=[decoy, racer])

            # upsert(racer) commits, then parks at its engine apply.
            gate = asyncio.Event()
            gated_engine.gate = gate
            upsert_task = asyncio.create_task(
                coll.upsert(
                    records=[
                        _make_record(
                            uuid=racer.uuid, vector=_normalize([1.0, 0.0, 0.0])
                        )
                    ]
                )
            )
            await gated_engine.gate_reached.wait()

            # delete(racer) commits and applies while the upsert is parked.
            delete_task = asyncio.create_task(coll.delete(record_uuids=[racer.uuid]))
            await _wait_for(lambda: _a_delete_has_been_applied(engine))

            gate.set()
            await asyncio.gather(upsert_task, delete_task)

            # The racer's vector scores 1.0 against this query, so if it is
            # still in the engine it takes the only slot and resolves to
            # nothing, leaving no matches at all.
            results = await coll.query(
                query_vectors=[_normalize([1.0, 0.0, 0.0])], limit=1
            )
            assert [match.record.uuid for match in results[0].matches] == [decoy.uuid]
        finally:
            await store.shutdown()
            await engine.dispose()

    @pytest.mark.asyncio
    async def test_a_save_cannot_trim_a_write_it_did_not_publish(self, tmp_path):
        """A write applied behind a save must not be trimmed by that save.

        The pending log holds the only other copy of a vector, so its row may
        be deleted once the index that holds it is published. A write that
        applies to the engine after the index is written but before the trim
        satisfies neither: not in the file, and no longer in the log. It is
        live in memory and gone from disk -- a committed write lost to a
        process crash, which is the one failure this store rules out.
        """
        db_path = tmp_path / "test.db"
        store, engine, wrapped = await _wrapped_engine_store(
            db_path, tmp_path, _GatedSaveEngine, save_threshold=1
        )
        coll = await store.open_or_create_collection(
            namespace=NAMESPACE, name=NAME, config=CONFIG
        )
        (gated_engine,) = wrapped

        # This write's own save parks with the index written and the trim
        # still to come.
        gate = asyncio.Event()
        gated_engine.gate = gate
        published = _make_record(vector=_normalize([1.0, 0.0, 0.0]))
        publish_task = asyncio.create_task(coll.upsert(records=[published]))
        await gated_engine.gate_reached.wait()

        # A second write lands in that window: applied to the engine and
        # marked applied, but absent from the index just written.
        behind = _make_record(vector=_normalize([0.0, 1.0, 0.0]))
        behind_task = asyncio.create_task(coll.upsert(records=[behind]))
        await _wait_for(lambda: _both_writes_applied(engine))

        gate.set()
        await publish_task

        # The second write ends up parked in its own save, which never writes:
        # through the window above on unfixed code, by waiting its turn once
        # writes are serialized. Either way it has committed, applied, and been
        # marked applied, and nothing else can publish behind it.
        async with asyncio.timeout(_RACE_WINDOW_SECONDS):
            await gated_engine.blocked_save_reached.wait()

        # Crash.
        behind_task.cancel()
        await asyncio.gather(behind_task, return_exceptions=True)
        await engine.dispose()

        store2, engine2 = await _fresh_store(db_path, tmp_path)
        try:
            coll2 = await store2.open_collection(namespace=NAMESPACE, name=NAME)
            assert coll2 is not None

            fetched = await coll2.get(record_uuids=[behind.uuid])
            assert [record.uuid for record in fetched] == [behind.uuid]

            results = await coll2.query(
                query_vectors=[_normalize([0.0, 1.0, 0.0])], limit=5
            )
            assert behind.uuid in {match.record.uuid for match in results[0].matches}, (
                "a committed write was trimmed by a save that never published it"
            )
        finally:
            await store2.shutdown()
            await engine2.dispose()


# ── Index file durability contract ──


async def _get_index_saved(engine, namespace, name) -> bool | None:
    session_factory = async_sessionmaker(engine, expire_on_commit=False)
    async with session_factory() as session:
        return (
            await session.execute(
                select(_CollectionRow.index_saved).where(
                    _CollectionRow.namespace == namespace,
                    _CollectionRow.name == name,
                )
            )
        ).scalar_one_or_none()


class TestIndexFileDurability:
    """Once the index has been saved, the file is part of the durable contract."""

    @pytest.mark.asyncio
    async def test_saved_flag_starts_false(self, tmp_path):
        """A freshly created collection has index_saved=False."""
        db_path = tmp_path / "test.db"
        store, engine = await _fresh_store(db_path, tmp_path)
        await store.open_or_create_collection(
            namespace=NAMESPACE, name=NAME, config=CONFIG
        )

        assert await _get_index_saved(engine, NAMESPACE, NAME) is False

        await store.shutdown()
        await engine.dispose()

    @pytest.mark.asyncio
    async def test_saved_flag_flips_on_save_threshold(self, tmp_path):
        """Crossing the save threshold flips index_saved to True."""
        db_path = tmp_path / "test.db"
        store, engine = await _fresh_store(db_path, tmp_path, save_threshold=1)

        coll = await store.open_or_create_collection(
            namespace=NAMESPACE, name=NAME, config=CONFIG
        )
        await coll.upsert(records=[_make_record(vector=_normalize([1.0, 0.0, 0.0]))])

        assert await _get_index_saved(engine, NAMESPACE, NAME) is True

        await store.shutdown()
        await engine.dispose()

    @pytest.mark.asyncio
    async def test_saved_flag_flips_on_clean_shutdown(self, tmp_path):
        """Clean shutdown flips index_saved to True even below save_threshold."""
        db_path = tmp_path / "test.db"
        store, engine = await _fresh_store(db_path, tmp_path, save_threshold=1000)

        coll = await store.open_or_create_collection(
            namespace=NAMESPACE, name=NAME, config=CONFIG
        )
        await coll.upsert(records=[_make_record(vector=_normalize([1.0, 0.0, 0.0]))])

        # Below save_threshold, so _maybe_save_index has not flipped the flag yet.
        assert await _get_index_saved(engine, NAMESPACE, NAME) is False

        await store.shutdown()
        assert await _get_index_saved(engine, NAMESPACE, NAME) is True

        await engine.dispose()

    @pytest.mark.asyncio
    async def test_missing_file_when_saved_raises(self, tmp_path):
        """If the index has been saved, a missing file is loud, not silent."""
        db_path = tmp_path / "test.db"
        store1, engine1 = await _fresh_store(db_path, tmp_path)

        coll = await store1.open_or_create_collection(
            namespace=NAMESPACE, name=NAME, config=CONFIG
        )
        await coll.upsert(records=[_make_record(vector=_normalize([1.0, 0.0, 0.0]))])
        await store1.shutdown()
        await engine1.dispose()

        # Operator deletes the index file out from under us.
        index_dir = tmp_path / "indexes"
        idx_files = list(index_dir.glob("*.idx"))
        assert len(idx_files) == 1
        idx_files[0].unlink()

        # Restart: open_collection must surface the failure, not return an
        # engine silently rebuilt empty.
        store2, engine2 = await _fresh_store(db_path, tmp_path)
        with pytest.raises(IndexLoadError) as exc_info:
            await store2.open_collection(namespace=NAMESPACE, name=NAME)
        assert exc_info.value.namespace == NAMESPACE
        assert exc_info.value.name == NAME
        assert exc_info.value.__cause__ is not None

        await store2.shutdown()
        await engine2.dispose()

    @pytest.mark.asyncio
    async def test_corrupt_file_when_saved_raises(self, tmp_path):
        """If the index has been saved, a corrupt file is loud, not silent."""
        db_path = tmp_path / "test.db"
        store1, engine1 = await _fresh_store(db_path, tmp_path)

        coll = await store1.open_or_create_collection(
            namespace=NAMESPACE, name=NAME, config=CONFIG
        )
        await coll.upsert(records=[_make_record(vector=_normalize([1.0, 0.0, 0.0]))])
        await store1.shutdown()
        await engine1.dispose()

        index_dir = tmp_path / "indexes"
        idx_files = list(index_dir.glob("*.idx"))
        assert len(idx_files) == 1
        idx_files[0].write_bytes(b"not a valid index file")

        store2, engine2 = await _fresh_store(db_path, tmp_path)
        with pytest.raises(IndexLoadError) as exc_info:
            await store2.open_collection(namespace=NAMESPACE, name=NAME)
        assert exc_info.value.namespace == NAMESPACE
        assert exc_info.value.name == NAME
        assert exc_info.value.__cause__ is not None

        await store2.shutdown()
        await engine2.dispose()

    @pytest.mark.asyncio
    async def test_missing_file_before_saved_recovers(self, tmp_path):
        """Crash before the first save: empty engine + WAL replay is correct."""
        db_path = tmp_path / "test.db"
        store1, engine1 = await _fresh_store(db_path, tmp_path, save_threshold=1000)

        coll = await store1.open_or_create_collection(
            namespace=NAMESPACE, name=NAME, config=CONFIG
        )
        await coll.upsert(
            records=[
                _make_record(vector=_normalize([1.0, 0.0, 0.0])),
                _make_record(vector=_normalize([0.0, 1.0, 0.0])),
            ]
        )

        # Crash without shutdown: index_saved stays False, no file on disk.
        await engine1.dispose()
        assert not (tmp_path / "indexes").exists() or not list(
            (tmp_path / "indexes").glob("*.idx")
        )

        store2, engine2 = await _fresh_store(db_path, tmp_path)
        coll2 = await store2.open_collection(namespace=NAMESPACE, name=NAME)
        assert coll2 is not None

        results = await coll2.query(
            query_vectors=[_normalize([1.0, 0.0, 0.0])], limit=10
        )
        assert len(results[0].matches) == 2

        await store2.shutdown()
        await engine2.dispose()
