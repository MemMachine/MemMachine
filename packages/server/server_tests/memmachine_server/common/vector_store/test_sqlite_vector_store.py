"""Tests for SQLiteVectorStore."""

import asyncio
import contextlib
import math
from datetime import UTC, datetime, timedelta, timezone
from typing import Any
from uuid import UUID, uuid4

import numpy as np
import pytest
import pytest_asyncio
from pydantic import ValidationError
from sqlalchemy import func, select, update
from sqlalchemy.exc import OperationalError
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from memmachine_server.common.data_types import PropertyType
from memmachine_server.common.filter.filter_parser import (
    And,
    Comparison,
    In,
    Not,
    Or,
)
from memmachine_server.common.vector_store.data_types import (
    Record,
    VectorStorePartitionAlreadyExistsError,
    VectorStorePartitionSchemaMismatchError,
)
from memmachine_server.common.vector_store.sqlite_vector_store import (
    IndexLoadError,
    PendingOperationCorruptError,
    SQLiteVectorStore,
    SQLiteVectorStoreParams,
    SQLiteVectorStorePartition,
    _PartitionRow,
    _PendingOperationRow,
    _write_transaction,
)
from memmachine_server.common.vector_store.vector_search_engine.usearch_engine import (
    USearchVectorSearchEngine,
)
from memmachine_server.common.vector_store.vector_search_engine.vector_search_engine import (
    VectorSearchEngine,
)

VECTOR_STORE_NAME = "test_vector_store"
NAME = "test_name"
VECTOR_DIM = 3
INDEXED_PROPERTIES: dict[str, PropertyType] = {
    "name": str,
    "age": int,
    "score": float,
    "active": bool,
    "created_at": datetime,
}


def _normalize(vector: list[float]) -> list[float]:
    magnitude = math.sqrt(sum(x * x for x in vector))
    return [x / magnitude for x in vector]


def _make_record(
    *,
    uuid=None,
    vector: list[float],
    properties: dict | None = None,
) -> Record:
    return Record(
        uuid=uuid or uuid4(),
        vector=vector,
        properties=properties or {},
    )


_RACE_WINDOW_SECONDS = 2.0


async def _wait_for(condition) -> None:
    """Poll `condition` until it holds, or give up quietly at the deadline.

    Both outcomes are expected: the interleavings below are what unfixed code
    does while another write is parked, and serialized writes cannot start at
    all. The assertions tell the cases apart.
    """
    # Polling, because the condition is what another task committed to the
    # database, which no in-process event tracks. The deadline sits between
    # polls: cancelling a query mid-flight returns its connection to the pool
    # with a read transaction still open, which blocks the next commit.
    loop = asyncio.get_running_loop()
    deadline = loop.time() + _RACE_WINDOW_SECONDS
    while loop.time() < deadline:
        if await condition():
            return
        await asyncio.sleep(0.01)


_FETCH_LIMIT = 1000


async def _present_uuids(collection, record_uuids) -> list[UUID]:
    """Which of these UUIDs the partition still holds, in the order given.

    `query` is the only read and it answers with UUIDs and scores, so existence
    is all a test can observe about a record here. With no score threshold a
    probe vector in any direction lists the whole partition.
    """
    record_uuids = list(record_uuids)
    if not record_uuids:
        return []
    probe = [1.0] + [0.0] * (VECTOR_DIM - 1)
    [result] = await collection.query(query_vectors=[probe], limit=_FETCH_LIMIT)
    present = {match.record_uuid for match in result.matches}
    return [uuid for uuid in record_uuids if uuid in present]


@pytest_asyncio.fixture
async def store(tmp_path):
    db_path = tmp_path / "test.db"
    engine = create_async_engine(f"sqlite+aiosqlite:///{db_path}")
    vector_store = SQLiteVectorStore(_params(engine))
    await vector_store.provision()
    await vector_store.startup()
    yield vector_store
    await vector_store.shutdown()
    await engine.dispose()


def _engine_factory(ndim):
    return USearchVectorSearchEngine(num_dimensions=ndim)


def _params(engine, **overrides) -> SQLiteVectorStoreParams:
    params: dict[str, Any] = {
        "sqlalchemy_engine": engine,
        "vector_store_name": VECTOR_STORE_NAME,
        "vector_dimensions": VECTOR_DIM,
        "indexed_properties": INDEXED_PROPERTIES,
        "vector_search_engine_factory": _engine_factory,
    }
    params.update(overrides)
    return SQLiteVectorStoreParams(**params)


async def _store_with(store, vector_store_name: str, **overrides) -> SQLiteVectorStore:
    """Another store over the fixture's engine: its own collection, dimensions or metric."""
    other = SQLiteVectorStore(
        _params(
            store._sqlalchemy_engine, vector_store_name=vector_store_name, **overrides
        )
    )
    await other.provision()
    await other.startup()
    return other


@pytest_asyncio.fixture
async def collection(store):
    await store.create_partition(NAME)
    coll = await store.get_partition(NAME)
    assert coll is not None
    yield coll
    await store.delete_partition(NAME)


# ── Partition lifecycle ──


class TestPartitionLifecycle:
    @pytest.mark.asyncio
    async def test_create_open_delete(self, store):
        await store.create_partition("lifecycle")
        coll = await store.get_partition("lifecycle")
        assert isinstance(coll, SQLiteVectorStorePartition)
        await store.delete_partition("lifecycle")

    @pytest.mark.asyncio
    async def test_open_returns_correct_type(self, store, collection):
        coll = await store.get_partition(NAME)
        assert isinstance(coll, SQLiteVectorStorePartition)

    @pytest.mark.asyncio
    async def test_duplicate_name_raises(self, store, collection):
        with pytest.raises(VectorStorePartitionAlreadyExistsError):
            await store.create_partition(NAME)

    @pytest.mark.asyncio
    async def test_delete_nonexistent_is_idempotent(self, store):
        await store.delete_partition("nonexistent")

    @pytest.mark.asyncio
    async def test_open_or_create_creates_when_missing(self, store):
        coll = await store.open_or_create_partition("new")
        assert isinstance(coll, SQLiteVectorStorePartition)
        await store.delete_partition("new")

    @pytest.mark.asyncio
    async def test_open_or_create_opens_when_exists(self, store):
        await store.create_partition("existing")
        coll = await store.open_or_create_partition("existing")
        assert isinstance(coll, SQLiteVectorStorePartition)
        await store.delete_partition("existing")

    @pytest.mark.asyncio
    async def test_a_store_with_another_schema_cannot_open_the_partition(self, store):
        """The schema is the collection's: another store of it must declare the same."""
        await store.create_partition("mismatch")
        other_dimensions = await _store_with(
            store, VECTOR_STORE_NAME, vector_dimensions=VECTOR_DIM + 1
        )
        with pytest.raises(VectorStorePartitionSchemaMismatchError, match="mismatch"):
            await other_dimensions.open_or_create_partition("mismatch")
        await other_dimensions.shutdown()
        other_keys = await _store_with(
            store, VECTOR_STORE_NAME, indexed_properties={"name": str}
        )
        with pytest.raises(VectorStorePartitionSchemaMismatchError, match="mismatch"):
            await other_keys.get_partition("mismatch")
        await other_keys.shutdown()
        await store.delete_partition("mismatch")

    @pytest.mark.asyncio
    async def test_open_nonexistent_returns_none(self, store):
        assert await store.get_partition("nope") is None

    @pytest.mark.asyncio
    async def test_invalid_partition_key_raises(self, store):
        with pytest.raises(ValueError, match="Invalid partition key"):
            await store.create_partition("INVALID")


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
        assert matches[0].record_uuid == r1.uuid
        assert (
            matches[0].cosine_similarity
            >= matches[1].cosine_similarity
            >= matches[2].cosine_similarity
        )

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

        # Properties are filterable but never returned, so a filter is what
        # observes them.
        [result] = await collection.query(
            query_vectors=[v1],
            limit=10,
            property_filter=Comparison(field="name", op="=", value="updated"),
        )
        assert [m.record_uuid for m in result.matches] == [record.uuid]

    @pytest.mark.asyncio
    async def test_query_with_similarity_threshold(self, collection):
        v1 = _normalize([1.0, 0.0, 0.0])
        v2 = _normalize([0.0, 1.0, 0.0])

        r1 = _make_record(vector=v1)
        r2 = _make_record(vector=v2)

        await collection.upsert(records=[r1, r2])

        query_results = await collection.query(
            query_vectors=[v1], limit=10, min_cosine_similarity=0.9
        )
        matches = query_results[0].matches

        assert len(matches) == 1
        assert matches[0].record_uuid == r1.uuid

    @pytest.mark.asyncio
    async def test_query_with_limit(self, collection):
        vectors = [_normalize([1.0, float(index) * 0.01, 0.0]) for index in range(5)]
        records = [_make_record(vector=vector) for vector in vectors]
        await collection.upsert(records=records)

        query_results = await collection.query(query_vectors=[vectors[0]], limit=2)
        assert len(query_results[0].matches) == 2

    @pytest.mark.asyncio
    async def test_query_batch_multiple_vectors(self, collection):
        v1 = _normalize([1.0, 0.0, 0.0])
        v2 = _normalize([0.0, 1.0, 0.0])

        r1 = _make_record(vector=v1, properties={"name": "a"})
        r2 = _make_record(vector=v2, properties={"name": "b"})
        await collection.upsert(records=[r1, r2])

        all_results = await collection.query(query_vectors=[v1, v2], limit=1)

        assert len(all_results) == 2
        assert all_results[0].matches[0].record_uuid == r1.uuid
        assert all_results[1].matches[0].record_uuid == r2.uuid

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
        return {match.record_uuid for match in all_results[0].matches}

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

        # Properties are filterable but never returned, so a filter is what
        # observes them.
        # The instant survives the round trip whichever zone it is named in.
        [result] = await collection.query(
            query_vectors=[v1],
            limit=10,
            property_filter=Comparison(
                field="created_at", op="=", value=dt.astimezone(UTC)
            ),
        )
        assert [m.record_uuid for m in result.matches] == [r1.uuid]

    # ── In / And / Or / Not ──

    @pytest.mark.asyncio
    async def test_in(self, collection):
        r1, _r2, r3, v1 = await self._setup(collection)
        query_results = await collection.query(
            query_vectors=[v1],
            limit=10,
            property_filter=In(field="name", values=["alice", "carol"]),
        )
        uuids = {match.record_uuid for match in query_results[0].matches}
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
        assert matches[0].record_uuid == r3.uuid

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
        uuids = {match.record_uuid for match in query_results[0].matches}
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
        uuids = {match.record_uuid for match in query_results[0].matches}
        assert r1.uuid in uuids
        assert r2.uuid in uuids
        assert len(uuids) == 2


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

        results_a = await coll_a.query(query_vectors=[v1], limit=10)
        results_b = await coll_b.query(query_vectors=[v1], limit=10)

        uuids_a = {match.record_uuid for match in results_a[0].matches}
        uuids_b = {match.record_uuid for match in results_b[0].matches}
        assert uuids_a == {r1.uuid}
        assert uuids_b == {r2.uuid}

        await store.delete_partition("tenant_a")
        await store.delete_partition("tenant_b")

    @pytest.mark.asyncio
    async def test_get_only_returns_own_collection(self, store):
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
    async def test_delete_only_affects_own_collection(self, store):
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

        await coll_a.delete(record_uuids=[r2.uuid])

        results = await _present_uuids(coll_b, [r2.uuid])
        assert len(results) == 1
        assert results[0] == r2.uuid

        await store.delete_partition("tenant_a")
        await store.delete_partition("tenant_b")

    @pytest.mark.asyncio
    async def test_collections_on_one_engine_are_isolated(self, store):
        """Two stores over one engine, named differently, never see each other's rows."""
        other = await _store_with(store, "other_collection")
        await store.create_partition("coll")
        await other.create_partition("coll")
        coll_a = await store.get_partition("coll")
        coll_b = await other.get_partition("coll")
        assert coll_a is not None
        assert coll_b is not None

        v1 = _normalize([1.0, 0.0, 0.0])
        r1 = _make_record(vector=v1)
        r2 = _make_record(vector=v1)

        await coll_a.upsert(records=[r1])
        await coll_b.upsert(records=[r2])

        results_a = await coll_a.query(query_vectors=[v1], limit=10)
        assert {match.record_uuid for match in results_a[0].matches} == {r1.uuid}

        await store.delete_partition("coll")
        assert await other.get_partition("coll") is not None
        await other.delete_partition("coll")
        await other.shutdown()

    @pytest.mark.asyncio
    async def test_delete_partition_does_not_affect_sibling(self, store):
        """Deleting one collection doesn't break a sibling sharing tables."""
        coll_a = await store.open_or_create_partition("sibling_a")
        coll_b = await store.open_or_create_partition("sibling_b")

        v1 = _normalize([1.0, 0.0, 0.0])
        r1 = _make_record(vector=v1)
        r2 = _make_record(vector=v1)
        await coll_a.upsert(records=[r1])
        await coll_b.upsert(records=[r2])

        await store.delete_partition("sibling_a")

        results = await coll_b.query(query_vectors=[v1], limit=10)
        assert len(results[0].matches) == 1
        assert results[0].matches[0].record_uuid == r2.uuid

        await store.delete_partition("sibling_b")


class TestNoProperties:
    @pytest.mark.asyncio
    async def test_partition_without_properties(self, store):
        bare = await _store_with(
            store, "no_props", vector_dimensions=2, indexed_properties={}
        )
        coll = await bare.open_or_create_partition("no_props")
        r1 = _make_record(vector=[1.0, 0.0])
        await coll.upsert(records=[r1])

        results = await coll.query(query_vectors=[[1.0, 0.0]], limit=1)
        assert len(results[0].matches) == 1

        await bare.delete_partition("no_props")
        await bare.shutdown()


class TestInputValidation:
    def test_record_requires_a_vector(self):
        """The model rejects it, so no store has to re-check."""
        with pytest.raises(ValidationError):
            Record(uuid=uuid4())  # ty: ignore[missing-argument]

    def test_record_defaults_properties_to_empty(self):
        record = Record(uuid=uuid4(), vector=_normalize([1.0, 0.0, 0.0]))
        assert record.properties == {}

    @pytest.mark.asyncio
    async def test_upsert_accepts_a_record_without_properties(self, collection):
        record = Record(uuid=uuid4(), vector=_normalize([1.0, 0.0, 0.0]))
        await collection.upsert(records=[record])
        assert await _present_uuids(collection, [record.uuid]) == [record.uuid]


# ── Cosine similarity semantics ──


class TestScoreSemantics:
    @pytest.mark.asyncio
    async def test_cosine_higher_is_better(self, collection):
        v1 = _normalize([1.0, 0.0, 0.0])
        v2 = _normalize([0.0, 1.0, 0.0])
        r1 = _make_record(vector=v1)
        r2 = _make_record(vector=v2)
        await collection.upsert(records=[r1, r2])

        results = await collection.query(query_vectors=[v1], limit=2)
        scores = [m.cosine_similarity for m in results[0].matches]
        assert scores[0] > scores[1]


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

        # Properties are filterable but never returned, so a filter is what
        # observes them.
        [named_bob] = await collection.query(
            query_vectors=[v2],
            limit=10,
            property_filter=Comparison(field="name", op="=", value="bob"),
        )
        assert [m.record_uuid for m in named_bob.matches] == [record_uuid]

        results = await collection.query(query_vectors=[v2], limit=1)
        assert results[0].matches[0].record_uuid == record_uuid
        assert results[0].matches[0].cosine_similarity == pytest.approx(1.0, abs=0.01)


# ── Concurrent async behavior ──


class TestConcurrentAsync:
    @pytest.mark.asyncio
    async def test_concurrent_upserts(self, collection):
        """Multiple concurrent upserts should not error and all records should be persisted."""
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
        fetched = await _present_uuids(collection, all_uuids)
        assert len(fetched) == 30

    @pytest.mark.asyncio
    async def test_concurrent_upsert_and_query(self, collection):
        """Query during upsert should not error (eventual consistency)."""
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


# ── Engine access ──


class _GatedAfterRemoveEngine(VectorSearchEngine):
    """Delegates to a real engine; the next remove() parks after removing."""

    def __init__(self, inner: VectorSearchEngine) -> None:
        self.inner = inner
        self.gate: asyncio.Event | None = None
        self.gate_reached = asyncio.Event()

    async def add(self, vectors):
        await self.inner.add(vectors)

    async def remove(self, keys):
        await self.inner.remove(keys)
        if self.gate is not None:
            gate, self.gate = self.gate, None
            self.gate_reached.set()
            await gate.wait()

    async def search(self, vectors, *, limit, allowed_keys=None):
        return await self.inner.search(vectors, limit=limit, allowed_keys=allowed_keys)

    async def save(self, path):
        await self.inner.save(path)

    async def load(self, path):
        await self.inner.load(path)


class TestEngineAccess:
    """The store serializes engine access: searches share, mutations exclude."""

    @pytest.mark.asyncio
    async def test_a_reader_never_sees_a_rewrite_half_done(self, tmp_path):
        """A rewrite removes the old vector and adds the new one as one step.

        Engines are not safe for concurrent use, so the store holds the lock
        across both calls. A query issued while the rewrite is parked between
        them waits and sees the new vector, never neither.
        """
        db_path = tmp_path / "test.db"
        engine = create_async_engine(f"sqlite+aiosqlite:///{db_path}")
        wrapped: list[_GatedAfterRemoveEngine] = []

        def factory(ndim):
            gated = _GatedAfterRemoveEngine(_engine_factory(ndim))
            wrapped.append(gated)
            return gated

        store = SQLiteVectorStore(_params(engine, vector_search_engine_factory=factory))
        await store.provision()
        await store.startup()
        try:
            collection = await store.open_or_create_partition(NAME)
            (gated_engine,) = wrapped
            record_uuid = uuid4()
            await collection.upsert(
                records=[
                    _make_record(uuid=record_uuid, vector=_normalize([1.0, 0.0, 0.0]))
                ]
            )

            gate = asyncio.Event()
            gated_engine.gate = gate
            rewrite_task = asyncio.create_task(
                collection.upsert(
                    records=[
                        _make_record(
                            uuid=record_uuid, vector=_normalize([0.0, 1.0, 0.0])
                        )
                    ]
                )
            )
            await gated_engine.gate_reached.wait()

            # Issued while the old vector is gone and the new one not yet added.
            query_task = asyncio.create_task(
                collection.query(query_vectors=[_normalize([0.0, 1.0, 0.0])], limit=1)
            )

            gate.set()
            await rewrite_task
            [result] = await query_task
            assert [m.record_uuid for m in result.matches] == [record_uuid]
        finally:
            await store.shutdown()
            await engine.dispose()


# ── row_id reuse (issue #1468) ──


class TestRowIdReuse:
    """Regression tests for row_id reuse (issue #1468).

    Without AUTOINCREMENT, SQLite assigns max(rowid) + 1, so a deleted record's
    id can go to the next insert. Serialized writes do not cover the read path,
    which resolves scored keys to rows without a lock, so these tests pin the
    id policy itself.
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
    async def test_a_query_cannot_return_a_record_it_never_scored(
        self, collection, monkeypatch
    ):
        """A key the engine scored must never resolve to a later record.

        Writes run freely between scoring and row lookup, so a reused row_id
        would return a never-scored record with another's score: here, a
        record orthogonal to the query as a perfect hit. With ids never reused
        the stale key matches no row and is dropped.
        """
        scored = _make_record(vector=_normalize([1.0, 0.0, 0.0]))
        await collection.upsert(records=[scored])

        # The query parks holding the scored key, after the engine search and
        # before the row lookup: the highest row_id, and so the one a reused
        # id would hand out next.
        gate = asyncio.Event()
        gate_reached = asyncio.Event()
        build_matches = collection._build_matches

        async def parked_build_matches(*args, **kwargs):
            gate_reached.set()
            await gate.wait()
            return await build_matches(*args, **kwargs)

        monkeypatch.setattr(collection, "_build_matches", parked_build_matches)
        query_task = asyncio.create_task(
            collection.query(query_vectors=[_normalize([1.0, 0.0, 0.0])], limit=1)
        )
        await gate_reached.wait()

        await collection.delete(record_uuids=[scored.uuid])
        successor = _make_record(vector=_normalize([0.0, 1.0, 0.0]))
        await collection.upsert(records=[successor])

        gate.set()
        results = await query_task

        assert [match.record_uuid for match in results[0].matches] == [], (
            "a record the engine never scored was returned"
        )


# ── Crash recovery & pending operations ──


async def _fresh_store(db_path, tmp_path, *, save_threshold=1000):
    """Create a new SQLiteVectorStore against the same DB file."""
    engine = create_async_engine(f"sqlite+aiosqlite:///{db_path}")
    params = _params(
        engine,
        index_directory=str(tmp_path / "indexes"),
        save_threshold=save_threshold,
    )
    store = SQLiteVectorStore(params)
    await store.provision()
    await store.startup()
    return store, engine


async def _stored_record_uuids(engine, store) -> set[UUID]:
    """The uuids the test partition's records table holds."""
    session_factory = async_sessionmaker(engine, expire_on_commit=False)
    async with session_factory() as session:
        rows = (await session.execute(select(store._records_table(NAME).c.uuid))).all()
    return {row.uuid for row in rows}


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


class TestPendingLogStates:
    """The log's state machine, exercised across a restart."""

    @pytest.mark.asyncio
    async def test_replay_refuses_an_undecodable_vector(self, tmp_path):
        """A blob that is not a float32 vector is damage, not a shorter vector."""
        db_path = tmp_path / "test.db"
        store1, engine1 = await _fresh_store(db_path, tmp_path)
        coll = await store1.open_or_create_partition(NAME)
        await coll.upsert(records=[_make_record(vector=_normalize([1.0, 0.0, 0.0]))])

        session_factory = async_sessionmaker(engine1, expire_on_commit=False)
        async with session_factory() as session, session.begin():
            await session.execute(
                update(_PendingOperationRow).values(vector=b"\x01\x02\x03")
            )
        await engine1.dispose()

        with pytest.raises(PendingOperationCorruptError, match="cannot be decoded"):
            await _fresh_store(db_path, tmp_path)

    @pytest.mark.asyncio
    async def test_replay_refuses_a_vector_of_the_wrong_width(self, tmp_path):
        """A blob of whole float32s that is not the store's width is damage too."""
        db_path = tmp_path / "test.db"
        store1, engine1 = await _fresh_store(db_path, tmp_path)
        coll = await store1.open_or_create_partition(NAME)
        await coll.upsert(records=[_make_record(vector=_normalize([1.0, 0.0, 0.0]))])

        session_factory = async_sessionmaker(engine1, expire_on_commit=False)
        async with session_factory() as session, session.begin():
            await session.execute(
                update(_PendingOperationRow).values(
                    vector=np.zeros(VECTOR_DIM - 1, dtype=np.float32).tobytes()
                )
            )
        await engine1.dispose()

        with pytest.raises(PendingOperationCorruptError, match="dimension"):
            await _fresh_store(db_path, tmp_path)

    @pytest.mark.asyncio
    async def test_a_rewritten_uuid_replays_its_last_write(self, tmp_path):
        """Replay restores a uuid's last write.

        A second write to a uuid overwrites its log row rather than queueing
        behind it, which is only observable across a restart.
        """
        db_path = tmp_path / "test.db"
        store1, engine1 = await _fresh_store(db_path, tmp_path)
        coll = await store1.open_or_create_partition(NAME)
        first = _normalize([1.0, 0.0, 0.0])
        second = _normalize([0.0, 1.0, 0.0])
        record_uuid = uuid4()
        await coll.upsert(records=[_make_record(uuid=record_uuid, vector=first)])
        await coll.upsert(records=[_make_record(uuid=record_uuid, vector=second)])

        # Crash: dispose without shutting down, so nothing is saved or trimmed.
        await engine1.dispose()

        store2, engine2 = await _fresh_store(db_path, tmp_path)
        coll2 = await store2.get_partition(NAME)
        assert coll2 is not None

        results = await coll2.query(query_vectors=[second], limit=1)
        assert results[0].matches[0].record_uuid == record_uuid
        assert results[0].matches[0].cosine_similarity == pytest.approx(1.0, abs=0.01)

        # And the superseded vector is not still indexed under a second key.
        stale = await coll2.query(query_vectors=[first], limit=5)
        assert [m.record_uuid for m in stale[0].matches] == [record_uuid]

        await store2.shutdown()
        await engine2.dispose()

    @pytest.mark.asyncio
    async def test_an_upsert_then_delete_of_one_uuid_stays_deleted(self, tmp_path):
        """A delete staged over an untrimmed upsert wins the replay."""
        db_path = tmp_path / "test.db"
        store1, engine1 = await _fresh_store(db_path, tmp_path)
        coll = await store1.open_or_create_partition(NAME)
        vector = _normalize([1.0, 0.0, 0.0])
        record = _make_record(vector=vector)
        await coll.upsert(records=[record])
        await coll.delete(record_uuids=[record.uuid])
        await engine1.dispose()

        store2, engine2 = await _fresh_store(db_path, tmp_path)
        coll2 = await store2.get_partition(NAME)
        assert coll2 is not None
        results = await coll2.query(query_vectors=[vector], limit=5)
        assert results[0].matches == []

        await store2.shutdown()
        await engine2.dispose()

    @pytest.mark.asyncio
    async def test_a_failed_index_save_leaves_the_write_replayable(self, tmp_path):
        """A save that fails must not take the write with it.

        The log is trimmed only after the save returns.
        """
        db_path = tmp_path / "test.db"
        store1, engine1 = await _fresh_store(db_path, tmp_path, save_threshold=1)
        coll = await store1.open_or_create_partition(NAME)
        vector = _normalize([1.0, 0.0, 0.0])
        record = _make_record(vector=vector)

        search_engine = store1._search_engines[NAME]

        async def failing_save(path: str) -> None:
            raise OSError("no space left on device")

        original_save = search_engine.save
        search_engine.save = failing_save
        with pytest.raises(OSError, match="no space"):
            await coll.upsert(records=[record])
        search_engine.save = original_save

        assert await _pending_operation_count(engine1) == 1
        await engine1.dispose()

        store2, engine2 = await _fresh_store(db_path, tmp_path)
        coll2 = await store2.get_partition(NAME)
        assert coll2 is not None
        results = await coll2.query(query_vectors=[vector], limit=1)
        assert results[0].matches[0].record_uuid == record.uuid

        await store2.shutdown()
        await engine2.dispose()

    @pytest.mark.asyncio
    async def test_save_threshold_counts_log_rows_not_writes(self, tmp_path):
        """The trigger counts log rows, and one uuid only ever has one.

        Rewriting a single record never advances it, however often it is
        written.
        """
        db_path = tmp_path / "test.db"
        store, engine = await _fresh_store(db_path, tmp_path, save_threshold=3)
        coll = await store.open_or_create_partition(NAME)
        record_uuid = uuid4()
        for i in range(6):
            await coll.upsert(
                records=[
                    _make_record(
                        uuid=record_uuid, vector=_normalize([1.0, float(i) + 1.0, 0.0])
                    )
                ]
            )

        # Six writes, one row, so the threshold of three was never reached.
        assert await _pending_operation_count(engine) == 1

        # Three distinct records do reach it.
        await coll.upsert(
            records=[
                _make_record(vector=_normalize([1.0, 0.0, 0.0])),
                _make_record(vector=_normalize([0.0, 1.0, 0.0])),
                _make_record(vector=_normalize([0.0, 0.0, 1.0])),
            ]
        )
        assert await _pending_operation_count(engine) == 0

        await store.shutdown()
        await engine.dispose()


class TestBatchEdges:
    """Batches that name one record more than once."""

    @pytest.mark.asyncio
    async def test_upsert_batch_with_a_repeated_uuid_keeps_the_last(self, collection):
        record_uuid = uuid4()
        v1 = _normalize([1.0, 0.0, 0.0])
        v2 = _normalize([0.0, 1.0, 0.0])
        await collection.upsert(
            records=[
                _make_record(uuid=record_uuid, vector=v1, properties={"name": "first"}),
                _make_record(uuid=record_uuid, vector=v2, properties={"name": "last"}),
            ]
        )

        [named_last] = await collection.query(
            query_vectors=[v2],
            limit=5,
            property_filter=Comparison(field="name", op="=", value="last"),
        )
        assert [m.record_uuid for m in named_last.matches] == [record_uuid]

        results = await collection.query(query_vectors=[v2], limit=5)
        assert [m.record_uuid for m in results[0].matches] == [record_uuid]

    @pytest.mark.asyncio
    async def test_delete_tolerates_a_repeated_uuid(self, collection):
        record = _make_record(vector=_normalize([1.0, 0.0, 0.0]))
        await collection.upsert(records=[record])

        await collection.delete(record_uuids=[record.uuid, record.uuid])

        assert await _present_uuids(collection, [record.uuid]) == []

    @pytest.mark.asyncio
    async def test_tied_scores_return_distinct_records(self, collection):
        """Identical vectors must not collapse into one match or duplicate one."""
        vector = _normalize([1.0, 0.0, 0.0])
        records = [_make_record(vector=vector) for _ in range(3)]
        await collection.upsert(records=records)

        results = await collection.query(query_vectors=[vector], limit=3)
        matched = [m.record_uuid for m in results[0].matches]
        assert len(matched) == 3
        assert set(matched) == {record.uuid for record in records}


class TestOneLockPerPartition:
    """The lock belongs to the store, not to a handle.

    A handle is constructed per `get_partition` call, so a per-handle lock
    would serialize nothing across handles.
    """

    @pytest.mark.asyncio
    async def test_two_handles_on_one_partition_are_serialized(self, tmp_path):
        """The same-uuid interleaving as one handle, across two handles.

        upsert(U) on one handle parks at its engine apply; delete(U) on the
        other commits and applies in that window; the upsert resumes and
        re-adds a vector for a record SQLite says is gone.
        """
        db_path = tmp_path / "test.db"
        store, engine, wrapped = await _wrapped_engine_store(
            db_path, tmp_path, _GatedRemoveEngine
        )
        try:
            writer = await store.open_or_create_partition(NAME)
            deleter = await store.get_partition(NAME)
            assert deleter is not None
            assert deleter is not writer
            (gated_engine,) = wrapped

            decoy = _make_record(vector=_normalize([1.0, 1.0, 0.0]))
            racer = _make_record(vector=_normalize([0.0, 1.0, 0.0]))
            await writer.upsert(records=[decoy, racer])

            gate = asyncio.Event()
            gated_engine.gate = gate
            upsert_task = asyncio.create_task(
                writer.upsert(
                    records=[
                        _make_record(
                            uuid=racer.uuid, vector=_normalize([1.0, 0.0, 0.0])
                        )
                    ]
                )
            )
            await gated_engine.gate_reached.wait()

            # A task, not an await: with per-handle locks this runs to
            # completion inside the window; with the store's lock it waits.
            delete_task = asyncio.create_task(deleter.delete(record_uuids=[racer.uuid]))
            await _wait_for(lambda: _a_delete_has_been_applied(engine))

            gate.set()
            await asyncio.gather(upsert_task, delete_task)

            # The racer's vector scores 1.0 here, so if it is still in the
            # engine it takes the only slot and resolves to nothing.
            results = await writer.query(
                query_vectors=[_normalize([1.0, 0.0, 0.0])], limit=1
            )
            assert [match.record_uuid for match in results[0].matches] == [
                decoy.uuid
            ], "a vector belonging to no record is still in the engine"
        finally:
            await store.shutdown()
            await engine.dispose()


class TestConcurrentWrites:
    """Writers that do not contend for the same record still must not lose."""

    @pytest.mark.asyncio
    async def test_concurrent_upsert_and_delete_of_disjoint_sets(self, collection):
        """Deleting one set while upserting another loses neither."""
        kept = [
            _make_record(vector=_normalize([1.0, float(i) + 1.0, 0.0]))
            for i in range(5)
        ]
        doomed = [
            _make_record(vector=_normalize([0.0, 1.0, float(i) + 1.0]))
            for i in range(5)
        ]
        await collection.upsert(records=doomed)

        await asyncio.gather(
            collection.upsert(records=kept),
            collection.delete(record_uuids=[record.uuid for record in doomed]),
        )

        candidates = [record.uuid for record in kept + doomed]
        surviving = set(await _present_uuids(collection, candidates))
        assert surviving == {record.uuid for record in kept}

    @pytest.mark.asyncio
    async def test_writes_racing_a_checkpoint_are_not_lost(self, tmp_path):
        """A save runs under the same lock as a write, so none can slip past it.

        With a threshold of one every write checkpoints, driving the save and
        write paths against each other continuously.
        """
        db_path = tmp_path / "test.db"
        store, engine = await _fresh_store(db_path, tmp_path, save_threshold=1)
        coll = await store.open_or_create_partition(NAME)
        records = [
            _make_record(vector=_normalize([1.0, float(i) + 1.0, float(i)]))
            for i in range(20)
        ]
        await asyncio.gather(*(coll.upsert(records=[record]) for record in records))

        await store.shutdown()
        await engine.dispose()

        store2, engine2 = await _fresh_store(db_path, tmp_path)
        coll2 = await store2.get_partition(NAME)
        assert coll2 is not None
        found = set(await _present_uuids(coll2, [record.uuid for record in records]))
        assert found == {record.uuid for record in records}

        await store2.shutdown()
        await engine2.dispose()

    @pytest.mark.asyncio
    async def test_racing_creates_admit_exactly_one(self, store):
        """Four creates of one key: one wins, the rest are refused."""
        attempts = [store.create_partition("raced") for _ in range(4)]
        outcomes = await asyncio.gather(*attempts, return_exceptions=True)

        created = [o for o in outcomes if not isinstance(o, BaseException)]
        refused = [
            o for o in outcomes if isinstance(o, VectorStorePartitionAlreadyExistsError)
        ]
        assert len(created) == 1
        assert len(refused) == len(outcomes) - 1


class TestWriteTransactions:
    """A write transaction takes SQLite's write lock at BEGIN."""

    @pytest.mark.asyncio
    async def test_a_write_transaction_can_still_write_after_it_has_read(
        self, tmp_path
    ):
        """A write transaction's read and write are one atomic unit.

        Under a deferred BEGIN a transaction that reads first leaves a gap
        another writer can take the lock in, and the reader then loses: its
        write cannot upgrade, and SQLite reports that at once rather than
        waiting out `busy_timeout`, since waiting could only deadlock.

        The assertion is that the reading transaction completes. Without
        `BEGIN IMMEDIATE` it raises, in either journal mode.
        """
        db_path = tmp_path / "test.db"
        store, engine = await _fresh_store(db_path, tmp_path)
        await store.open_or_create_partition(NAME)

        # A second connection with a short busy timeout, so a blocked write
        # reports instead of waiting out the default five seconds.
        other = create_async_engine(
            f"sqlite+aiosqlite:///{db_path}", connect_args={"timeout": 0.2}
        )
        other_session = async_sessionmaker(other, expire_on_commit=False)

        create_session = async_sessionmaker(engine, expire_on_commit=False)
        async with _write_transaction(create_session) as session:
            # Read first, exactly as `delete` does, and write nothing yet.
            await session.execute(select(func.count()).select_from(_PartitionRow))

            # A second writer takes what it can in the gap and holds it across
            # the write below. `BEGIN IMMEDIATE` shuts it out; a deferred BEGIN
            # lets it in.
            blocked = other_session()
            try:
                with contextlib.suppress(OperationalError):
                    await blocked.execute(
                        update(_PendingOperationRow).values(applied=True)
                    )

                await session.execute(
                    update(_PartitionRow)
                    .where(_PartitionRow.vector_store_name == VECTOR_STORE_NAME)
                    .values(index_saved=False)
                )
            finally:
                await blocked.rollback()
                await blocked.close()

        await other.dispose()
        await store.shutdown()
        await engine.dispose()


class TestCrashRecovery:
    """Tests for pending operations replay on startup."""

    @pytest.mark.asyncio
    async def test_replay_refuses_an_upsert_with_no_vector(self, tmp_path):
        """A damaged log row fails the restart instead of vanishing.

        Until the next index save the log holds the only copy of the vector,
        so a skipped row is a record no search can find.
        """
        db_path = tmp_path / "test.db"
        store1, engine1 = await _fresh_store(db_path, tmp_path)
        coll = await store1.open_or_create_partition(NAME)
        await coll.upsert(records=[_make_record(vector=_normalize([1.0, 0.0, 0.0]))])

        session_factory = async_sessionmaker(engine1, expire_on_commit=False)
        async with session_factory() as session, session.begin():
            await session.execute(update(_PendingOperationRow).values(vector=None))
        await engine1.dispose()

        with pytest.raises(PendingOperationCorruptError, match="no vector"):
            await _fresh_store(db_path, tmp_path)

    @pytest.mark.asyncio
    async def test_replay_refuses_an_unknown_operation_type(self, tmp_path):
        """An operation_type replay does not understand fails the restart."""
        db_path = tmp_path / "test.db"
        store1, engine1 = await _fresh_store(db_path, tmp_path)
        coll = await store1.open_or_create_partition(NAME)
        await coll.upsert(records=[_make_record(vector=_normalize([1.0, 0.0, 0.0]))])

        session_factory = async_sessionmaker(engine1, expire_on_commit=False)
        async with session_factory() as session, session.begin():
            await session.execute(
                update(_PendingOperationRow).values(operation_type="rewrite")
            )
        await engine1.dispose()

        with pytest.raises(
            PendingOperationCorruptError, match="unknown operation_type"
        ):
            await _fresh_store(db_path, tmp_path)

    @pytest.mark.asyncio
    async def test_replay_upserts_after_crash(self, tmp_path):
        """Records upserted before crash are queryable after restart."""
        db_path = tmp_path / "test.db"
        store1, engine1 = await _fresh_store(db_path, tmp_path)

        coll = await store1.open_or_create_partition(NAME)
        records = [
            _make_record(vector=_normalize([1.0, 0.0, 0.0])),
            _make_record(vector=_normalize([0.0, 1.0, 0.0])),
        ]
        await coll.upsert(records=records)

        # Simulate crash: dispose without shutdown (pending ops remain).
        await engine1.dispose()

        # Restart with fresh in-memory engines.
        store2, engine2 = await _fresh_store(db_path, tmp_path)
        coll2 = await store2.get_partition(NAME)
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

        coll = await store1.open_or_create_partition(NAME)
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
        coll2 = await store2.get_partition(NAME)
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

        coll = await store1.open_or_create_partition(NAME)
        r1 = _make_record(vector=_normalize([1.0, 0.0, 0.0]))
        r2 = _make_record(vector=_normalize([0.0, 1.0, 0.0]))
        await coll.upsert(records=[r1, r2])
        await coll.delete(record_uuids=[r1.uuid])

        # Simulate crash.
        await engine1.dispose()

        store2, engine2 = await _fresh_store(db_path, tmp_path)
        coll2 = await store2.get_partition(NAME)
        assert coll2 is not None

        results = await coll2.query(
            query_vectors=[_normalize([1.0, 0.0, 0.0])], limit=10
        )
        assert len(results[0].matches) == 1

        fetched = await _present_uuids(coll2, [r1.uuid])
        assert len(fetched) == 0

        await store2.shutdown()
        await engine2.dispose()

    @pytest.mark.asyncio
    async def test_replay_mixed_upserts_and_deletes(self, tmp_path):
        """Mixed upsert and delete pending ops are replayed correctly."""
        db_path = tmp_path / "test.db"
        store1, engine1 = await _fresh_store(db_path, tmp_path)

        coll = await store1.open_or_create_partition(NAME)
        r1 = _make_record(vector=_normalize([1.0, 0.0, 0.0]))
        r2 = _make_record(vector=_normalize([0.0, 1.0, 0.0]))
        r3 = _make_record(vector=_normalize([0.0, 0.0, 1.0]))
        await coll.upsert(records=[r1, r2, r3])
        await coll.delete(record_uuids=[r2.uuid])

        # Simulate crash.
        await engine1.dispose()

        store2, engine2 = await _fresh_store(db_path, tmp_path)
        coll2 = await store2.get_partition(NAME)
        assert coll2 is not None

        results = await coll2.query(
            query_vectors=[_normalize([1.0, 0.0, 0.0])], limit=10
        )
        assert len(results[0].matches) == 2

        fetched = await _present_uuids(coll2, [r1.uuid, r2.uuid, r3.uuid])
        fetched_uuids = set(fetched)
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

        coll = await store1.open_or_create_partition(NAME)

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

        coll = await store1.open_or_create_partition(NAME)
        records = [
            _make_record(vector=_normalize([1.0, 0.0, 0.0])),
            _make_record(vector=_normalize([0.0, 1.0, 0.0])),
        ]
        await coll.upsert(records=records)
        assert await _pending_operation_count(engine1) == 2

        await store1.delete_partition(NAME)
        assert await _pending_operation_count(engine1) == 0

        await store1.shutdown()
        await engine1.dispose()

    @pytest.mark.asyncio
    async def test_require_started(self, tmp_path):
        """Store methods raise RuntimeError before startup."""
        db_path = tmp_path / "test.db"
        engine = create_async_engine(f"sqlite+aiosqlite:///{db_path}")
        store = SQLiteVectorStore(_params(engine))

        with pytest.raises(RuntimeError, match="startup"):
            await store.get_partition(NAME)

        with pytest.raises(RuntimeError, match="startup"):
            await store.create_partition(NAME)

        with pytest.raises(RuntimeError, match="startup"):
            await store.delete_partition(NAME)

        await engine.dispose()


# ── Concurrent write ordering (issue #1468) ──


class _GatedRemoveEngine(VectorSearchEngine):
    """Delegates to a real engine; the next remove() waits on `gate` first.

    delete() already suspends at its engine remove(), after the SQL commit;
    the gate widens that window so the interleaving is deterministic.
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

    async def save(self, path):
        await self.inner.save(path)

    async def load(self, path):
        await self.inner.load(path)


class _GatedSaveEngine(VectorSearchEngine):
    """Delegates to a real engine; the first save parks after writing the index.

    A save writes the index and then trims the log. Parking between the two
    widens the window in which another write can apply to the engine: too late
    for the file, early enough for the trim to delete its log row.

    Later saves park before writing, so a test can reach its crash without a
    save publishing what it is trying to observe.
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

    def factory(ndim):
        gated = wrap(_engine_factory(ndim))
        wrapped.append(gated)
        return gated

    store = SQLiteVectorStore(
        _params(
            engine,
            vector_search_engine_factory=factory,
            index_directory=str(tmp_path / "indexes"),
            save_threshold=save_threshold,
        )
    )
    await store.provision()
    await store.startup()
    return store, engine, wrapped


async def _record_exists(engine, store, record_uuid) -> bool:
    """Whether a concurrent upsert's transaction has committed yet."""
    return record_uuid in await _stored_record_uuids(engine, store)


async def _a_delete_has_been_applied(engine) -> bool:
    """Whether a concurrent delete has reached the search engine.

    Its pending row is marked applied last, so this is true only once the
    delete has both committed and removed from the engine.
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
    """The engine must see a partition's writes in the order SQLite committed.

    A write commits to SQLite before applying to the engine, so unserialized
    writers could reach the engine in the opposite order. Never reusing a
    row_id does not cover this: both writes address one uuid, which keeps its
    row_id across upserts.
    """

    @pytest.mark.asyncio
    async def test_an_upsert_survives_a_delete_of_another_record(self, tmp_path):
        """A delete must not carry an unrelated concurrent upsert with it.

        delete(A) commits and parks at its engine removal; upsert(B) lands in
        that window; delete(A) resumes and removes the row_id it was given. B
        stays searchable: it never had A's row_id, and now cannot run in that
        window at all.
        """
        db_path = tmp_path / "test.db"
        store, engine, wrapped = await _wrapped_engine_store(
            db_path, tmp_path, _GatedRemoveEngine
        )
        try:
            coll = await store.open_or_create_partition(NAME)
            (gated_engine,) = wrapped

            deleted = _make_record(vector=_normalize([1.0, 0.0, 0.0]))
            await coll.upsert(records=[deleted])

            gate = asyncio.Event()
            gated_engine.gate = gate
            delete_task = asyncio.create_task(coll.delete(record_uuids=[deleted.uuid]))
            await gated_engine.gate_reached.wait()

            # A task, not an await: unfixed code runs this inside the window,
            # serialized writes make it wait for the delete.
            upserted = _make_record(vector=_normalize([0.0, 1.0, 0.0]))
            upsert_task = asyncio.create_task(coll.upsert(records=[upserted]))
            await _wait_for(lambda: _record_exists(engine, store, upserted.uuid))

            gate.set()
            await asyncio.gather(delete_task, upsert_task)

            results = await coll.query(
                query_vectors=[_normalize([0.0, 1.0, 0.0])], limit=5
            )
            assert upserted.uuid in {
                match.record_uuid for match in results[0].matches
            }, "upserted record lost from the search engine"
        finally:
            await store.shutdown()
            await engine.dispose()

    @pytest.mark.asyncio
    async def test_an_upsert_cannot_overtake_a_delete_of_the_same_uuid(self, tmp_path):
        """A vector re-added after its record is deleted belongs to nothing.

        upsert(U) commits and parks at its engine apply; delete(U) commits and
        applies in that window; upsert(U) resumes and re-adds its vector. The
        vector resolves to no row, is dropped from every result it wins, and
        the next save publishes it for good.
        """
        db_path = tmp_path / "test.db"
        store, engine, wrapped = await _wrapped_engine_store(
            db_path, tmp_path, _GatedRemoveEngine
        )
        try:
            coll = await store.open_or_create_partition(NAME)
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

            # The racer's vector scores 1.0 here, so if it is still in the
            # engine it takes the only slot and resolves to nothing.
            results = await coll.query(
                query_vectors=[_normalize([1.0, 0.0, 0.0])], limit=1
            )
            assert [match.record_uuid for match in results[0].matches] == [decoy.uuid]
        finally:
            await store.shutdown()
            await engine.dispose()

    @pytest.mark.asyncio
    async def test_a_save_cannot_trim_a_write_it_did_not_publish(self, tmp_path):
        """A write applied behind a save must not be trimmed by that save.

        A write that applies to the engine after the index is written but
        before the trim is in neither the file nor the log: live in memory,
        gone from disk, lost to a process crash.
        """
        db_path = tmp_path / "test.db"
        store, engine, wrapped = await _wrapped_engine_store(
            db_path, tmp_path, _GatedSaveEngine, save_threshold=1
        )
        coll = await store.open_or_create_partition(NAME)
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

        # The second write parks in its own save, which never writes: through
        # the window on unfixed code, after its turn once serialized. Either
        # way it is committed, applied, and marked applied.
        async with asyncio.timeout(_RACE_WINDOW_SECONDS):
            await gated_engine.blocked_save_reached.wait()

        # Crash.
        behind_task.cancel()
        await asyncio.gather(behind_task, return_exceptions=True)
        await engine.dispose()

        store2, engine2 = await _fresh_store(db_path, tmp_path)
        try:
            coll2 = await store2.get_partition(NAME)
            assert coll2 is not None

            # The record is still a row.
            assert behind.uuid in await _stored_record_uuids(engine2, store2)

            results = await coll2.query(
                query_vectors=[_normalize([0.0, 1.0, 0.0])], limit=5
            )
            assert behind.uuid in {match.record_uuid for match in results[0].matches}, (
                "a committed write was trimmed by a save that never published it"
            )
        finally:
            await store2.shutdown()
            await engine2.dispose()


# ── Index file durability contract ──


async def _get_index_saved(engine, name) -> bool | None:
    session_factory = async_sessionmaker(engine, expire_on_commit=False)
    async with session_factory() as session:
        return (
            await session.execute(
                select(_PartitionRow.index_saved).where(
                    _PartitionRow.vector_store_name == VECTOR_STORE_NAME,
                    _PartitionRow.partition_key == name,
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
        await store.open_or_create_partition(NAME)

        assert await _get_index_saved(engine, NAME) is False

        await store.shutdown()
        await engine.dispose()

    @pytest.mark.asyncio
    async def test_saved_flag_flips_on_save_threshold(self, tmp_path):
        """Crossing the save threshold flips index_saved to True."""
        db_path = tmp_path / "test.db"
        store, engine = await _fresh_store(db_path, tmp_path, save_threshold=1)

        coll = await store.open_or_create_partition(NAME)
        await coll.upsert(records=[_make_record(vector=_normalize([1.0, 0.0, 0.0]))])

        assert await _get_index_saved(engine, NAME) is True

        await store.shutdown()
        await engine.dispose()

    @pytest.mark.asyncio
    async def test_saved_flag_flips_on_clean_shutdown(self, tmp_path):
        """Clean shutdown flips index_saved to True even below save_threshold."""
        db_path = tmp_path / "test.db"
        store, engine = await _fresh_store(db_path, tmp_path, save_threshold=1000)

        coll = await store.open_or_create_partition(NAME)
        await coll.upsert(records=[_make_record(vector=_normalize([1.0, 0.0, 0.0]))])

        # Below save_threshold, so _maybe_save_index has not flipped the flag yet.
        assert await _get_index_saved(engine, NAME) is False

        await store.shutdown()
        assert await _get_index_saved(engine, NAME) is True

        await engine.dispose()

    @pytest.mark.asyncio
    async def test_missing_file_when_saved_raises(self, tmp_path):
        """If the index has been saved, a missing file is loud, not silent."""
        db_path = tmp_path / "test.db"
        store1, engine1 = await _fresh_store(db_path, tmp_path)

        coll = await store1.open_or_create_partition(NAME)
        await coll.upsert(records=[_make_record(vector=_normalize([1.0, 0.0, 0.0]))])
        await store1.shutdown()
        await engine1.dispose()

        # Operator deletes the index file out from under us.
        index_dir = tmp_path / "indexes"
        idx_files = list(index_dir.glob("*.idx"))
        assert len(idx_files) == 1
        idx_files[0].unlink()

        # Restart: get_partition must surface the failure, not return an
        # engine silently rebuilt empty.
        store2, engine2 = await _fresh_store(db_path, tmp_path)
        with pytest.raises(IndexLoadError) as exc_info:
            await store2.get_partition(NAME)
        assert exc_info.value.vector_store_name == VECTOR_STORE_NAME
        assert exc_info.value.partition_key == NAME
        assert exc_info.value.__cause__ is not None

        await store2.shutdown()
        await engine2.dispose()

    @pytest.mark.asyncio
    async def test_corrupt_file_when_saved_raises(self, tmp_path):
        """If the index has been saved, a corrupt file is loud, not silent."""
        db_path = tmp_path / "test.db"
        store1, engine1 = await _fresh_store(db_path, tmp_path)

        coll = await store1.open_or_create_partition(NAME)
        await coll.upsert(records=[_make_record(vector=_normalize([1.0, 0.0, 0.0]))])
        await store1.shutdown()
        await engine1.dispose()

        index_dir = tmp_path / "indexes"
        idx_files = list(index_dir.glob("*.idx"))
        assert len(idx_files) == 1
        idx_files[0].write_bytes(b"not a valid index file")

        store2, engine2 = await _fresh_store(db_path, tmp_path)
        with pytest.raises(IndexLoadError) as exc_info:
            await store2.get_partition(NAME)
        assert exc_info.value.vector_store_name == VECTOR_STORE_NAME
        assert exc_info.value.partition_key == NAME
        assert exc_info.value.__cause__ is not None

        await store2.shutdown()
        await engine2.dispose()

    @pytest.mark.asyncio
    async def test_missing_file_before_saved_recovers(self, tmp_path):
        """Crash before the first save: empty engine + WAL replay is correct."""
        db_path = tmp_path / "test.db"
        store1, engine1 = await _fresh_store(db_path, tmp_path, save_threshold=1000)

        coll = await store1.open_or_create_partition(NAME)
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
        coll2 = await store2.get_partition(NAME)
        assert coll2 is not None

        results = await coll2.query(
            query_vectors=[_normalize([1.0, 0.0, 0.0])], limit=10
        )
        assert len(results[0].matches) == 2

        await store2.shutdown()
        await engine2.dispose()

    @pytest.mark.asyncio
    async def test_a_reverted_publication_costs_search_not_records(self, tmp_path):
        """A publication lost to power failure leaves records unsearchable.

        The swap is atomic, not durable, so a power failure can revert the last
        publication after the trim behind it has committed. Restoring the
        previous index bytes reconstructs exactly that state, deterministically
        rather than by pulling a plug, and pins the direction it fails in: the
        row survives, and only search loses the record, until it is upserted
        again. The collection contract has no read that can show the survivor --
        `query` is the only read and it goes through the index that lost it --
        so this looks at the row directly.
        """
        db_path = tmp_path / "test.db"
        store1, engine1 = await _fresh_store(db_path, tmp_path, save_threshold=1)

        coll = await store1.open_or_create_partition(NAME)
        r1 = _make_record(vector=_normalize([1.0, 0.0, 0.0]))
        await coll.upsert(records=[r1])

        index_dir = tmp_path / "indexes"
        idx_files = list(index_dir.glob("*.idx"))
        assert len(idx_files) == 1
        published_without_r2 = idx_files[0].read_bytes()

        # Publishes an index holding both, then trims r2's only other copy.
        r2 = _make_record(vector=_normalize([0.0, 1.0, 0.0]))
        await coll.upsert(records=[r2])
        assert await _pending_operation_count(engine1) == 0

        await store1.shutdown()
        await engine1.dispose()

        # Power failure: the publication that held r2 never reached the disk.
        idx_files[0].write_bytes(published_without_r2)

        store2, engine2 = await _fresh_store(db_path, tmp_path)
        coll2 = await store2.get_partition(NAME)
        assert coll2 is not None

        # The record is still a row.
        assert await _stored_record_uuids(engine2, store2) == {r1.uuid, r2.uuid}

        # The index just cannot find it any more.
        results = await coll2.query(query_vectors=[r2.vector], limit=10)
        assert [match.record_uuid for match in results[0].matches] == [r1.uuid]

        await store2.shutdown()
        await engine2.dispose()
