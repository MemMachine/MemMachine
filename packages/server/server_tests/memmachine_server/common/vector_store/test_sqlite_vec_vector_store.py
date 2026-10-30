"""Tests for SQLiteVecVectorStore."""

import asyncio
import math
import sqlite3
from datetime import UTC, datetime, timedelta, timezone
from typing import Any
from uuid import UUID, uuid4

import pytest
import pytest_asyncio
from pydantic import ValidationError
from sqlalchemy import inspect, select
from sqlalchemy.ext.asyncio import create_async_engine

from memmachine_server.common.data_types import PropertyType, SimilarityMetric
from memmachine_server.common.filter.filter_parser import (
    And,
    Comparison,
    In,
    Not,
    Or,
)
from memmachine_server.common.sql import enable_sqlite_explicit_begin
from memmachine_server.common.vector_store import sqlite_vec_vector_store
from memmachine_server.common.vector_store.data_types import (
    Record,
    VectorStoreAttemptsExhaustedError,
    VectorStorePartitionAlreadyExistsError,
    VectorStorePartitionHandleStaleError,
    VectorStoreSchema,
)
from memmachine_server.common.vector_store.sqlite_properties import (
    property_column_name,
)
from memmachine_server.common.vector_store.sqlite_vec_vector_store import (
    _STORE_SCHEMA_TABLES,
    SQLiteVecVectorStore,
    SQLiteVecVectorStoreParams,
    SQLiteVecVectorStorePartition,
    load_sqlite_vec_extension,
)
from server_tests.memmachine_server.common.vector_store.declared_schema_contract import (
    DeclaredSchemaContract,
)

pytestmark = pytest.mark.skipif(
    not hasattr(sqlite3.Connection, "enable_load_extension"),
    reason="sqlite3 built without extension loading support",
)

VECTOR_STORE_NAME = "test_vector_store"
NAME = "test_name"
VECTOR_DIM = 3
# A partition key of the longest length, with characters no identifier allows.
ANY_CHARACTERS_KEY = "Session/ID-1 ü".ljust(254, "x")
_EPOCH = datetime(1970, 1, 1, tzinfo=UTC)

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


async def _stored(partition) -> dict[UUID, dict]:
    """The records the partition's table holds, UUID to properties.

    Read past the store, from the table itself.
    """
    table = partition._records_table
    async with partition._create_session() as session:
        rows = (await session.execute(select(table))).all()
    return {
        row.uuid: _declared_values(
            row._mapping, partition.store_schema.indexed_properties
        )
        for row in rows
    }


def _declared_values(row, indexed_properties) -> dict:
    """A row's declared properties that hold a value, a datetime as its instant in UTC."""
    values = {}
    for key, property_type in indexed_properties.items():
        value = row[property_column_name(key)]
        if value is None:
            continue
        if property_type is datetime:
            value = _EPOCH + timedelta(microseconds=value)
        values[key] = value
    return values


@pytest_asyncio.fixture
async def store(tmp_path):
    db_path = tmp_path / "test.db"
    engine = create_async_engine(f"sqlite+aiosqlite:///{db_path}")
    enable_sqlite_explicit_begin(engine)
    load_sqlite_vec_extension(engine)
    vector_store = SQLiteVecVectorStore(_params(engine))
    await vector_store.migrate()
    await vector_store.startup()
    yield vector_store
    await vector_store.shutdown()
    await engine.dispose()


def _schema(**overrides) -> VectorStoreSchema:
    fields: dict[str, Any] = {
        "vector_dimensions": VECTOR_DIM,
        "similarity_metric": SimilarityMetric.COSINE,
        "indexed_properties": INDEXED_PROPERTIES,
    }
    fields.update(overrides)
    return VectorStoreSchema(**fields)


def _params(engine, **overrides) -> SQLiteVecVectorStoreParams:
    params: dict[str, Any] = {
        "engine": engine,
        "vector_store_name": VECTOR_STORE_NAME,
        "store_schema": _schema(),
    }
    params.update(overrides)
    return SQLiteVecVectorStoreParams(**params)


async def _store_with(
    store, vector_store_name: str, **overrides
) -> SQLiteVecVectorStore:
    """Another store over the fixture's engine: its own name, dimensions, or metric."""
    other = SQLiteVecVectorStore(
        _params(
            store._engine,
            vector_store_name=vector_store_name,
            store_schema=_schema(**overrides),
        )
    )
    await other.migrate()
    await other.startup()
    return other


@pytest_asyncio.fixture
async def partition(store):
    await store.create_partition(NAME)
    partition = await store.open_partition(NAME)
    assert partition is not None
    yield partition
    await store.delete_partition(NAME)


# ── Store schema record ──


@pytest.mark.asyncio
async def test_migrate_records_the_store_schema(store):
    store_table, declared_property_table = _STORE_SCHEMA_TABLES
    async with store._engine.connect() as connection:
        stores = (await connection.execute(select(store_table))).all()
        keys = (
            await connection.scalars(select(declared_property_table.c.property_key))
        ).all()

    assert [tuple(row) for row in stores] == [
        (VECTOR_STORE_NAME, VECTOR_DIM, SimilarityMetric.COSINE.value)
    ]
    assert set(keys) == set(INDEXED_PROPERTIES)


# ── Partition lifecycle ──


class TestPartitionLifecycle:
    @pytest.mark.asyncio
    async def test_create_open_delete(self, store):
        await store.create_partition("lifecycle")
        partition = await store.open_partition("lifecycle")
        assert isinstance(partition, SQLiteVecVectorStorePartition)
        await store.delete_partition("lifecycle")

    @pytest.mark.asyncio
    async def test_open_returns_correct_type(self, store, partition):
        assert isinstance(
            await store.open_partition(NAME), SQLiteVecVectorStorePartition
        )

    @pytest.mark.asyncio
    async def test_duplicate_name_raises(self, store, partition):
        with pytest.raises(VectorStorePartitionAlreadyExistsError):
            await store.create_partition(NAME)

    @pytest.mark.asyncio
    async def test_delete_nonexistent_is_idempotent(self, store):
        await store.delete_partition("nonexistent")

    @pytest.mark.asyncio
    async def test_open_or_create_creates_when_missing(self, store):
        partition = await store.open_or_create_partition("new")
        assert isinstance(partition, SQLiteVecVectorStorePartition)
        await store.delete_partition("new")

    @pytest.mark.asyncio
    async def test_open_or_create_opens_when_exists(self, store):
        await store.create_partition("existing")
        partition = await store.open_or_create_partition("existing")
        assert isinstance(partition, SQLiteVecVectorStorePartition)
        await store.delete_partition("existing")

    @pytest.mark.asyncio
    async def test_open_nonexistent_returns_none(self, store):
        assert await store.open_partition("nope") is None

    @pytest.mark.asyncio
    @pytest.mark.parametrize("create_via", ["create", "open_or_create"])
    async def test_an_incarnation_colliding_with_a_live_partitions_is_minted_again(
        self, store, monkeypatch, create_via
    ):
        live = await store.open_or_create_partition("live")
        offered: list[UUID] = []

        def colliding_uuid4() -> UUID:
            if not offered:
                offered.append(live._incarnation)
                return live._incarnation
            return uuid4()

        monkeypatch.setattr(sqlite_vec_vector_store, "uuid4", colliding_uuid4)
        if create_via == "create":
            await store.create_partition("other")
            other = await store.open_partition("other")
        else:
            other = await store.open_or_create_partition("other")
        assert other is not None
        assert other._incarnation != live._incarnation
        assert await store.open_partition("live") is not None

    @pytest.mark.asyncio
    async def test_a_persistent_insert_rejection_raises_after_bounded_attempts(
        self, store, monkeypatch
    ):
        attempts: list[UUID] = []

        async def always_rejected(partition_key: str, incarnation: UUID) -> None:
            attempts.append(incarnation)
            raise sqlite_vec_vector_store._RegistryInsertRejectedError(str(incarnation))

        monkeypatch.setattr(store, "_insert_partition", always_rejected)
        with pytest.raises(VectorStoreAttemptsExhaustedError):
            await store.create_partition("capped")
        assert len(attempts) == sqlite_vec_vector_store._MAX_MINT_ATTEMPTS

    @pytest.mark.asyncio
    async def test_unsupported_metric_raises(self, store):
        with pytest.raises(ValueError, match="sqlite-vec"):
            SQLiteVecVectorStore(
                _params(
                    store._engine,
                    vector_store_name="bad_metric",
                    store_schema=_schema(similarity_metric=SimilarityMetric.DOT),
                )
            )

    @pytest.mark.asyncio
    @pytest.mark.parametrize("partition_key", ["", "k" * 256, "ü" * 128, "with\x00nul"])
    async def test_a_partition_key_outside_the_rule_is_refused(
        self, store, partition_key
    ):
        with pytest.raises(ValueError, match="Partition key"):
            await store.create_partition(partition_key)

    @pytest.mark.parametrize(
        "vector_store_name",
        ["", "Upper", "with-hyphen", "trailing_newline\n", "x" * 33],
    )
    def test_a_vector_store_name_outside_the_rule_is_refused(
        self, tmp_path, vector_store_name
    ):
        engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'test.db'}")
        with pytest.raises(ValidationError, match="Vector store name"):
            _params(engine, vector_store_name=vector_store_name)

    @pytest.mark.asyncio
    async def test_a_partition_key_of_any_characters_and_the_longest_length_works(
        self, store
    ):
        key = ANY_CHARACTERS_KEY
        await store.create_partition(key)
        partition = await store.open_partition(key)
        assert partition is not None
        record = _make_record(vector=_normalize([1.0, 0.0, 0.0]))
        await partition.upsert(records=[record])
        [result] = await partition.query(query_vectors=[record.vector], limit=5)
        assert [match.record_uuid for match in result.matches] == [record.uuid]

        await store.delete_partition(key)
        assert await store.open_partition(key) is None

    @pytest.mark.asyncio
    async def test_a_handle_is_stale_once_its_partition_is_deleted(self, store):
        await store.create_partition("lifecycle")
        partition = await store.open_partition("lifecycle")
        assert partition is not None
        record = _make_record(vector=_normalize([1.0, 0.0, 0.0]))
        await partition.upsert(records=[record])

        await store.delete_partition("lifecycle")

        with pytest.raises(VectorStorePartitionHandleStaleError, match="lifecycle"):
            await partition.upsert(records=[record])
        with pytest.raises(VectorStorePartitionHandleStaleError, match="lifecycle"):
            await partition.query(query_vectors=[record.vector], limit=5)
        with pytest.raises(VectorStorePartitionHandleStaleError, match="lifecycle"):
            await partition.delete(record_uuids=[record.uuid])
        # So does an operation with nothing to do.
        with pytest.raises(VectorStorePartitionHandleStaleError, match="lifecycle"):
            await partition.upsert(records=[])
        with pytest.raises(VectorStorePartitionHandleStaleError, match="lifecycle"):
            await partition.query(query_vectors=[], limit=5)
        with pytest.raises(VectorStorePartitionHandleStaleError, match="lifecycle"):
            await partition.delete(record_uuids=[])

    @pytest.mark.asyncio
    async def test_a_recreated_partition_starts_empty_and_the_old_handle_stays_stale(
        self, store
    ):
        await store.create_partition("lifecycle")
        old = await store.open_partition("lifecycle")
        assert old is not None
        old_record = _make_record(vector=_normalize([1.0, 0.0, 0.0]))
        new_record = _make_record(vector=_normalize([0.0, 1.0, 0.0]))
        await old.upsert(records=[old_record])

        await store.delete_partition("lifecycle")
        await store.create_partition("lifecycle")
        new = await store.open_partition("lifecycle")
        assert new is not None
        [before] = await new.query(query_vectors=[old_record.vector], limit=5)
        assert before.matches == []
        await new.upsert(records=[new_record])

        # The old life's handle cannot reach the new life's records.
        with pytest.raises(VectorStorePartitionHandleStaleError):
            await old.upsert(records=[old_record])
        with pytest.raises(VectorStorePartitionHandleStaleError):
            await old.query(query_vectors=[new_record.vector], limit=5)
        with pytest.raises(VectorStorePartitionHandleStaleError):
            await old.delete(record_uuids=[new_record.uuid])
        assert set(await _stored(new)) == {new_record.uuid}

        await store.delete_partition("lifecycle")

    @pytest.mark.asyncio
    async def test_open_or_create_adopts_the_live_partition(self, store):
        first = await store.open_or_create_partition("lifecycle")
        second = await store.open_or_create_partition("lifecycle")
        record = _make_record(vector=_normalize([1.0, 0.0, 0.0]))
        await first.upsert(records=[record])
        [found] = await second.query(query_vectors=[record.vector], limit=5)
        assert [match.record_uuid for match in found.matches] == [record.uuid]

        await store.delete_partition("lifecycle")
        third = await store.open_or_create_partition("lifecycle")
        [empty] = await third.query(query_vectors=[record.vector], limit=5)
        assert empty.matches == []
        with pytest.raises(VectorStorePartitionHandleStaleError):
            await second.query(query_vectors=[record.vector], limit=5)

        await store.delete_partition("lifecycle")

    @pytest.mark.asyncio
    async def test_keys_differing_only_in_case_are_different_partitions(self, store):
        await store.create_partition("Key")
        await store.create_partition("key")
        upper = await store.open_partition("Key")
        lower = await store.open_partition("key")
        assert upper is not None
        assert lower is not None
        record = _make_record(vector=_normalize([1.0, 0.0, 0.0]))
        await upper.upsert(records=[record])

        [found] = await lower.query(query_vectors=[record.vector], limit=5)
        assert found.matches == []
        assert await _stored(lower) == {}

        await store.delete_partition("Key")
        assert await _stored(lower) == {}
        assert await store.open_partition("key") is not None
        await store.delete_partition("key")


# ── Upsert + Query ──


class TestUpsertAndQuery:
    @pytest.mark.asyncio
    async def test_upsert_and_query_basic(self, partition):
        v1 = _normalize([1.0, 0.0, 0.0])
        v2 = _normalize([0.0, 1.0, 0.0])
        v3 = _normalize([1.0, 0.1, 0.0])

        r1 = _make_record(vector=v1, properties={"name": "a"})
        r2 = _make_record(vector=v2, properties={"name": "b"})
        r3 = _make_record(vector=v3, properties={"name": "c"})

        await partition.upsert(records=[r1, r2, r3])

        query_results = await partition.query(query_vectors=[v1], limit=3)
        matches = query_results[0].matches

        assert len(matches) == 3
        assert matches[0].record_uuid == r1.uuid
        assert matches[0].score >= matches[1].score >= matches[2].score

    @pytest.mark.asyncio
    async def test_upsert_update(self, partition):
        v1 = _normalize([1.0, 0.0, 0.0])
        record = _make_record(vector=v1, properties={"name": "original"})
        await partition.upsert(records=[record])

        updated = Record(
            uuid=record.uuid,
            vector=_normalize([0.0, 1.0, 0.0]),
            properties={"name": "updated"},
        )
        await partition.upsert(records=[updated])

        assert (await _stored(partition))[record.uuid]["name"] == "updated"

    @pytest.mark.asyncio
    async def test_query_with_similarity_threshold(self, partition):
        v1 = _normalize([1.0, 0.0, 0.0])
        v2 = _normalize([0.0, 1.0, 0.0])

        r1 = _make_record(vector=v1)
        r2 = _make_record(vector=v2)

        await partition.upsert(records=[r1, r2])

        query_results = await partition.query(
            query_vectors=[v1], limit=10, score_threshold=0.9
        )
        matches = query_results[0].matches

        assert len(matches) == 1
        assert matches[0].record_uuid == r1.uuid

    @pytest.mark.asyncio
    async def test_query_with_limit(self, partition):
        vectors = [_normalize([1.0, float(index) * 0.01, 0.0]) for index in range(5)]
        records = [_make_record(vector=vector) for vector in vectors]
        await partition.upsert(records=records)

        query_results = await partition.query(query_vectors=[vectors[0]], limit=2)
        assert len(query_results[0].matches) == 2

    @pytest.mark.asyncio
    async def test_query_batch_multiple_vectors(self, partition):
        v1 = _normalize([1.0, 0.0, 0.0])
        v2 = _normalize([0.0, 1.0, 0.0])

        r1 = _make_record(vector=v1, properties={"name": "a"})
        r2 = _make_record(vector=v2, properties={"name": "b"})
        await partition.upsert(records=[r1, r2])

        all_results = await partition.query(query_vectors=[v1, v2], limit=1)

        assert len(all_results) == 2
        assert all_results[0].matches[0].record_uuid == r1.uuid
        assert all_results[1].matches[0].record_uuid == r2.uuid

    @pytest.mark.asyncio
    async def test_query_empty_vectors(self, partition):
        all_results = await partition.query(query_vectors=[], limit=10)
        assert len(all_results) == 0

    @pytest.mark.asyncio
    async def test_upsert_empty_records(self, partition):
        await partition.upsert(records=[])


# ── Filters ──


class TestDeclaredSchema(DeclaredSchemaContract):
    """The declared-schema contract, against this store."""

    @staticmethod
    async def settle(partition) -> None:
        # A write commits before it returns, so reads reflect it already.
        pass


class TestFilters:
    async def _setup(self, partition):
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
        await partition.upsert(records=[r1, r2, r3])
        return r1, r2, r3, v1

    async def _setup_floats(self, partition):
        vectors = [_normalize([1.0, float(index) * 0.01, 0.0]) for index in range(5)]
        scores = [-1.5, 0.0, 0.5, 1.5, 2.0]
        records = [
            _make_record(vector=vector, properties={"score": score})
            for vector, score in zip(vectors, scores, strict=True)
        ]
        await partition.upsert(records=records)
        return records, vectors[0]

    async def _setup_datetimes(self, partition):
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
        await partition.upsert(records=records)
        return records, vectors[0], datetimes

    async def _query(self, partition, query_vector, field, op, value):
        all_results = await partition.query(
            query_vectors=[query_vector],
            limit=10,
            property_filter=Comparison(field=field, op=op, value=value),
        )
        return {match.record_uuid for match in all_results[0].matches}

    # ── String / int ──

    @pytest.mark.asyncio
    async def test_eq_str(self, partition):
        r1, _r2, _r3, v1 = await self._setup(partition)
        uuids = await self._query(partition, v1, "name", "=", "alice")
        assert r1.uuid in uuids
        assert len(uuids) == 1

    @pytest.mark.asyncio
    async def test_ne_str(self, partition):
        r1, r2, r3, v1 = await self._setup(partition)
        uuids = await self._query(partition, v1, "name", "!=", "alice")
        assert r1.uuid not in uuids
        assert r2.uuid in uuids
        assert r3.uuid in uuids

    @pytest.mark.asyncio
    async def test_gt_int(self, partition):
        _r1, _r2, r3, v1 = await self._setup(partition)
        uuids = await self._query(partition, v1, "age", ">", 30)
        assert len(uuids) == 1
        assert r3.uuid in uuids

    @pytest.mark.asyncio
    async def test_gte_int(self, partition):
        r1, _r2, r3, v1 = await self._setup(partition)
        uuids = await self._query(partition, v1, "age", ">=", 30)
        assert r1.uuid in uuids
        assert r3.uuid in uuids
        assert len(uuids) == 2

    @pytest.mark.asyncio
    async def test_lt_int(self, partition):
        _r1, r2, _r3, v1 = await self._setup(partition)
        uuids = await self._query(partition, v1, "age", "<", 30)
        assert len(uuids) == 1
        assert r2.uuid in uuids

    @pytest.mark.asyncio
    async def test_lte_int(self, partition):
        r1, r2, _r3, v1 = await self._setup(partition)
        uuids = await self._query(partition, v1, "age", "<=", 30)
        assert r1.uuid in uuids
        assert r2.uuid in uuids
        assert len(uuids) == 2

    # ── Bool ──

    @pytest.mark.asyncio
    async def test_eq_bool(self, partition):
        r1, r2, r3, v1 = await self._setup(partition)
        uuids = await self._query(partition, v1, "active", "=", True)
        assert r1.uuid in uuids
        assert r3.uuid in uuids
        assert r2.uuid not in uuids

    @pytest.mark.asyncio
    async def test_ne_bool(self, partition):
        r1, r2, r3, v1 = await self._setup(partition)
        uuids = await self._query(partition, v1, "active", "!=", True)
        assert r2.uuid in uuids
        assert r1.uuid not in uuids
        assert r3.uuid not in uuids

    # ── Float ──

    @pytest.mark.asyncio
    async def test_eq_float(self, partition):
        records, query_vector = await self._setup_floats(partition)
        uuids = await self._query(partition, query_vector, "score", "=", 0.5)
        assert records[2].uuid in uuids
        assert len(uuids) == 1

    @pytest.mark.asyncio
    async def test_gt_float(self, partition):
        records, query_vector = await self._setup_floats(partition)
        uuids = await self._query(partition, query_vector, "score", ">", 0.5)
        assert records[3].uuid in uuids
        assert records[4].uuid in uuids
        assert records[2].uuid not in uuids

    @pytest.mark.asyncio
    async def test_lt_float(self, partition):
        records, query_vector = await self._setup_floats(partition)
        uuids = await self._query(partition, query_vector, "score", "<", 0.5)
        assert records[0].uuid in uuids
        assert records[1].uuid in uuids
        assert records[2].uuid not in uuids

    # ── Datetime ──

    @pytest.mark.asyncio
    async def test_datetime_roundtrip(self, partition):
        v1 = _normalize([1.0, 0.0, 0.0])
        dt = datetime(2024, 6, 15, 12, 30, 0, tzinfo=UTC)
        r1 = _make_record(vector=v1, properties={"name": "test", "created_at": dt})
        await partition.upsert(records=[r1])

        assert (await _stored(partition))[r1.uuid]["created_at"] == dt

    @pytest.mark.asyncio
    async def test_eq_datetime(self, partition):
        records, query_vector, datetimes = await self._setup_datetimes(partition)
        uuids = await self._query(
            partition, query_vector, "created_at", "=", datetimes[2]
        )
        assert records[2].uuid in uuids
        assert len(uuids) == 1

    @pytest.mark.asyncio
    async def test_gt_datetime(self, partition):
        records, query_vector, datetimes = await self._setup_datetimes(partition)
        uuids = await self._query(
            partition, query_vector, "created_at", ">", datetimes[2]
        )
        assert records[3].uuid in uuids
        assert records[4].uuid in uuids
        assert records[2].uuid not in uuids

    @pytest.mark.asyncio
    async def test_lt_datetime(self, partition):
        records, query_vector, datetimes = await self._setup_datetimes(partition)
        uuids = await self._query(
            partition, query_vector, "created_at", "<", datetimes[2]
        )
        assert records[0].uuid in uuids
        assert records[1].uuid in uuids
        assert records[2].uuid not in uuids

    @pytest.mark.asyncio
    async def test_eq_datetime_cross_timezone(self, partition):
        """Equality matches the same instant expressed in a different timezone."""
        v1 = _normalize([1.0, 0.0, 0.0])
        v2 = _normalize([1.0, 0.1, 0.0])
        dt_utc = datetime(2024, 6, 15, 12, 0, 0, tzinfo=UTC)
        dt_other = datetime(2024, 6, 15, 18, 0, 0, tzinfo=UTC)
        r1 = _make_record(vector=v1, properties={"name": "a", "created_at": dt_utc})
        r2 = _make_record(vector=v2, properties={"name": "b", "created_at": dt_other})
        await partition.upsert(records=[r1, r2])

        plus5 = timezone(timedelta(hours=5))
        dt_filter = datetime(2024, 6, 15, 17, 0, 0, tzinfo=plus5)
        uuids = await self._query(partition, v1, "created_at", "=", dt_filter)
        assert r1.uuid in uuids
        assert len(uuids) == 1

    @pytest.mark.asyncio
    async def test_a_datetime_is_stored_as_its_instant_in_utc(self, partition):
        """A datetime is stored as its instant, read back in UTC."""
        v1 = _normalize([1.0, 0.0, 0.0])
        est = timezone(timedelta(hours=-5))
        dt = datetime(2024, 6, 15, 7, 0, 0, tzinfo=est)
        r1 = _make_record(vector=v1, properties={"name": "tz", "created_at": dt})
        await partition.upsert(records=[r1])

        got = (await _stored(partition))[r1.uuid]["created_at"]
        assert got == dt
        assert got.utcoffset() == timedelta(0)

    # ── In / And / Or / Not ──

    @pytest.mark.asyncio
    async def test_in(self, partition):
        r1, _r2, r3, v1 = await self._setup(partition)
        query_results = await partition.query(
            query_vectors=[v1],
            limit=10,
            property_filter=In(field="name", values=["alice", "carol"]),
        )
        uuids = {match.record_uuid for match in query_results[0].matches}
        assert r1.uuid in uuids
        assert r3.uuid in uuids
        assert len(uuids) == 2

    @pytest.mark.asyncio
    async def test_and(self, partition):
        _r1, _r2, r3, v1 = await self._setup(partition)
        query_results = await partition.query(
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
    async def test_or(self, partition):
        r1, _r2, r3, v1 = await self._setup(partition)
        query_results = await partition.query(
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
    async def test_not(self, partition):
        r1, r2, _r3, v1 = await self._setup(partition)
        query_results = await partition.query(
            query_vectors=[v1],
            limit=10,
            property_filter=Not(expr=Comparison(field="age", op=">", value=30)),
        )
        uuids = {match.record_uuid for match in query_results[0].matches}
        assert r1.uuid in uuids
        assert r2.uuid in uuids
        assert len(uuids) == 2


# ── Delete ──


class TestDelete:
    @pytest.mark.asyncio
    async def test_delete_records(self, partition):
        v1 = _normalize([1.0, 0.0, 0.0])
        v2 = _normalize([0.0, 1.0, 0.0])

        r1 = _make_record(vector=v1)
        r2 = _make_record(vector=v2)

        await partition.upsert(records=[r1, r2])
        await partition.delete(record_uuids=[r1.uuid])

        assert set(await _stored(partition)) == {r2.uuid}

    @pytest.mark.asyncio
    async def test_delete_empty_list(self, partition):
        await partition.delete(record_uuids=[])

    @pytest.mark.asyncio
    async def test_delete_nonexistent_uuid(self, partition):
        await partition.delete(record_uuids=[uuid4()])

    @pytest.mark.asyncio
    async def test_delete_removes_from_query(self, partition):
        v1 = _normalize([1.0, 0.0, 0.0])
        r1 = _make_record(vector=v1)
        await partition.upsert(records=[r1])
        await partition.delete(record_uuids=[r1.uuid])

        query_results = await partition.query(query_vectors=[v1], limit=10)
        assert len(query_results[0].matches) == 0


# ── Partition isolation ──


class TestPartitionIsolation:
    @pytest.mark.asyncio
    async def test_query_only_returns_own_partition(self, store):
        await store.create_partition("tenant_a")
        await store.create_partition("tenant_b")
        partition_a = await store.open_partition("tenant_a")
        partition_b = await store.open_partition("tenant_b")
        assert partition_a is not None
        assert partition_b is not None

        v1 = _normalize([1.0, 0.0, 0.0])
        r1 = _make_record(vector=v1)
        r2 = _make_record(vector=v1)

        await partition_a.upsert(records=[r1])
        await partition_b.upsert(records=[r2])

        results_a = await partition_a.query(query_vectors=[v1], limit=10)
        results_b = await partition_b.query(query_vectors=[v1], limit=10)

        uuids_a = {match.record_uuid for match in results_a[0].matches}
        uuids_b = {match.record_uuid for match in results_b[0].matches}
        assert uuids_a == {r1.uuid}
        assert uuids_b == {r2.uuid}

        await store.delete_partition("tenant_a")
        await store.delete_partition("tenant_b")

    @pytest.mark.asyncio
    async def test_a_query_returns_only_its_own_partitions_records(self, store):
        await store.create_partition("tenant_a")
        await store.create_partition("tenant_b")
        partition_a = await store.open_partition("tenant_a")
        partition_b = await store.open_partition("tenant_b")
        assert partition_a is not None
        assert partition_b is not None

        v1 = _normalize([1.0, 0.0, 0.0])
        r1 = _make_record(vector=v1)
        r2 = _make_record(vector=v1)

        await partition_a.upsert(records=[r1])
        await partition_b.upsert(records=[r2])

        [result] = await partition_a.query(query_vectors=[v1], limit=10)
        assert [match.record_uuid for match in result.matches] == [r1.uuid]

        await store.delete_partition("tenant_a")
        await store.delete_partition("tenant_b")

    @pytest.mark.asyncio
    async def test_delete_only_affects_own_partition(self, store):
        await store.create_partition("tenant_a")
        await store.create_partition("tenant_b")
        partition_a = await store.open_partition("tenant_a")
        partition_b = await store.open_partition("tenant_b")
        assert partition_a is not None
        assert partition_b is not None

        v1 = _normalize([1.0, 0.0, 0.0])
        r1 = _make_record(vector=v1)
        r2 = _make_record(vector=v1)

        await partition_a.upsert(records=[r1])
        await partition_b.upsert(records=[r2])

        await partition_a.delete(record_uuids=[r2.uuid])

        assert set(await _stored(partition_b)) == {r2.uuid}

        await store.delete_partition("tenant_a")
        await store.delete_partition("tenant_b")

    @pytest.mark.asyncio
    async def test_stores_on_one_engine_are_isolated(self, store):
        """Two stores over one engine, named differently, never see each other's rows."""
        other = await _store_with(store, "other_store")
        await store.create_partition("shared_key")
        await other.create_partition("shared_key")
        partition_a = await store.open_partition("shared_key")
        partition_b = await other.open_partition("shared_key")
        assert partition_a is not None
        assert partition_b is not None

        v1 = _normalize([1.0, 0.0, 0.0])
        r1 = _make_record(vector=v1)
        r2 = _make_record(vector=v1)

        await partition_a.upsert(records=[r1])
        await partition_b.upsert(records=[r2])

        results_a = await partition_a.query(query_vectors=[v1], limit=10)
        assert {match.record_uuid for match in results_a[0].matches} == {r1.uuid}

        await store.delete_partition("shared_key")
        assert await other.open_partition("shared_key") is not None
        await other.delete_partition("shared_key")
        await other.shutdown()

    @pytest.mark.asyncio
    async def test_delete_partition_does_not_affect_sibling(self, store):
        """Deleting one partition doesn't break a sibling sharing tables."""
        partition_a = await store.open_or_create_partition("sibling_a")
        partition_b = await store.open_or_create_partition("sibling_b")

        v1 = _normalize([1.0, 0.0, 0.0])
        r1 = _make_record(vector=v1)
        r2 = _make_record(vector=v1)
        await partition_a.upsert(records=[r1])
        await partition_b.upsert(records=[r2])

        await store.delete_partition("sibling_a")

        results = await partition_b.query(query_vectors=[v1], limit=10)
        assert len(results[0].matches) == 1
        assert results[0].matches[0].record_uuid == r2.uuid

        await store.delete_partition("sibling_b")


# ── Euclidean metric ──


class TestEuclideanMetric:
    @pytest.mark.asyncio
    async def test_euclidean_ordering(self, store):
        euclidean = await _store_with(
            store,
            "euclidean",
            vector_dimensions=2,
            similarity_metric=SimilarityMetric.EUCLIDEAN,
            indexed_properties={},
        )
        partition = await euclidean.open_or_create_partition("euclidean")
        r1 = _make_record(vector=[0.0, 0.0])
        r2 = _make_record(vector=[3.0, 4.0])
        await partition.upsert(records=[r1, r2])

        results = await partition.query(query_vectors=[[0.0, 0.0]], limit=2)
        assert results[0].matches[0].score < results[0].matches[1].score

        await euclidean.delete_partition("euclidean")
        await euclidean.shutdown()


# ── No-properties store ──


class TestNoProperties:
    @pytest.mark.asyncio
    async def test_partition_without_properties(self, store):
        bare = await _store_with(
            store, "no_props", vector_dimensions=2, indexed_properties={}
        )
        partition = await bare.open_or_create_partition("no_props")
        r1 = _make_record(vector=[1.0, 0.0])
        await partition.upsert(records=[r1])

        results = await partition.query(query_vectors=[[1.0, 0.0]], limit=1)
        assert len(results[0].matches) == 1

        await bare.delete_partition("no_props")
        await bare.shutdown()


# ── Input validation ──


class TestInputValidation:
    def test_record_requires_a_vector(self):
        with pytest.raises(ValidationError):
            Record.model_validate({"uuid": uuid4()})

    def test_record_defaults_properties_to_empty(self):
        record = Record(uuid=uuid4(), vector=_normalize([1.0, 0.0, 0.0]))
        assert record.properties == {}

    @pytest.mark.asyncio
    async def test_upsert_accepts_a_record_without_properties(self, partition):
        record = Record(uuid=uuid4(), vector=_normalize([1.0, 0.0, 0.0]))
        await partition.upsert(records=[record])
        assert await _stored(partition) == {record.uuid: {}}

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("key", "value"),
        [("age", "old"), ("age", 30.5), ("age", True), ("score", 3), ("score", "high")],
    )
    async def test_a_declared_property_of_another_type_is_refused(
        self, partition, key, value
    ):
        with pytest.raises(ValueError, match=f"{key!r} is declared"):
            await partition.upsert(
                records=[
                    _make_record(
                        vector=_normalize([1.0, 0.0, 0.0]), properties={key: value}
                    )
                ]
            )

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "dimensions", [VECTOR_DIM - 1, VECTOR_DIM + 1], ids=["too_short", "too_long"]
    )
    async def test_a_vector_of_another_width_is_refused(self, partition, dimensions):
        vector = [1.0] * dimensions
        with pytest.raises(ValueError, match="dimensions"):
            await partition.upsert(records=[_make_record(vector=vector)])
        with pytest.raises(ValueError, match="dimensions"):
            await partition.query(query_vectors=[vector], limit=1)

    @pytest.mark.asyncio
    @pytest.mark.parametrize("coordinate", [math.nan, math.inf], ids=["nan", "inf"])
    async def test_a_query_vector_with_a_coordinate_that_is_not_finite_is_refused(
        self, partition, coordinate
    ):
        with pytest.raises(ValueError, match="not finite"):
            await partition.query(
                query_vectors=[[coordinate] + [1.0] * (VECTOR_DIM - 1)], limit=1
            )

    @pytest.mark.asyncio
    @pytest.mark.parametrize("threshold", [math.nan, math.inf, -math.inf])
    async def test_a_score_threshold_that_is_not_finite_is_refused(
        self, partition, threshold
    ):
        with pytest.raises(ValueError, match="not finite"):
            await partition.query(
                query_vectors=[_normalize([1.0, 0.0, 0.0])],
                limit=1,
                score_threshold=threshold,
            )

    @pytest.mark.asyncio
    @pytest.mark.parametrize("limit", [0, -1])
    async def test_a_limit_that_is_not_positive_is_refused(self, partition, limit):
        with pytest.raises(ValueError, match="not positive"):
            await partition.query(
                query_vectors=[_normalize([1.0, 0.0, 0.0])], limit=limit
            )


# ── Score semantics ──


class TestScoreSemantics:
    @pytest.mark.asyncio
    async def test_cosine_higher_is_better(self, partition):
        v1 = _normalize([1.0, 0.0, 0.0])
        v2 = _normalize([0.0, 1.0, 0.0])
        r1 = _make_record(vector=v1)
        r2 = _make_record(vector=v2)
        await partition.upsert(records=[r1, r2])

        results = await partition.query(query_vectors=[v1], limit=2)
        scores = [m.score for m in results[0].matches]
        assert scores[0] > scores[1]

    @pytest.mark.asyncio
    async def test_euclidean_lower_is_better(self, store):
        euclidean = await _store_with(
            store,
            "euclidean_score",
            vector_dimensions=2,
            similarity_metric=SimilarityMetric.EUCLIDEAN,
            indexed_properties={},
        )
        partition = await euclidean.open_or_create_partition("euclidean_score")
        r1 = _make_record(vector=[0.0, 0.0])
        r2 = _make_record(vector=[3.0, 4.0])
        await partition.upsert(records=[r1, r2])

        results = await partition.query(query_vectors=[[0.0, 0.0]], limit=2)
        scores = [m.score for m in results[0].matches]
        assert scores[0] < scores[1]
        assert scores[0] == pytest.approx(0.0, abs=0.01)
        assert scores[1] == pytest.approx(5.0, abs=0.01)

        await euclidean.delete_partition("euclidean_score")
        await euclidean.shutdown()


# A query, and vectors that each metric ranks in its own order, with no two
# tied under any metric.
_THRESHOLD_QUERY = [1.0, 0.0, 0.0]
_THRESHOLD_VECTORS = [
    [3.0, 3.0, 0.0],
    [1.0, 0.2, 0.0],
    [1.4, 0.4, 0.0],
    [1.6, 0.0, 0.0],
    [0.5, 0.05, 0.0],
]


class TestScoreThreshold:
    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "metric",
        [SimilarityMetric.COSINE, SimilarityMetric.EUCLIDEAN],
    )
    async def test_a_threshold_keeps_a_match_scoring_exactly_it(self, store, metric):
        """A score threshold equal to a match's reported score keeps the
        match; one a step better than that score drops it."""
        name = f"edge_{metric.value}"
        edged = await _store_with(
            store, name, vector_dimensions=VECTOR_DIM, similarity_metric=metric
        )
        partition = await edged.open_or_create_partition(name)
        records = [_make_record(vector=vector) for vector in _THRESHOLD_VECTORS]
        await partition.upsert(records=records)
        [ranked] = await partition.query(
            query_vectors=[_THRESHOLD_QUERY], limit=len(records)
        )
        edge = ranked.matches[2]
        better = math.inf if metric.higher_is_better else -math.inf

        [at_edge] = await partition.query(
            query_vectors=[_THRESHOLD_QUERY],
            limit=len(records),
            score_threshold=edge.score,
        )
        [past_edge] = await partition.query(
            query_vectors=[_THRESHOLD_QUERY],
            limit=len(records),
            score_threshold=math.nextafter(edge.score, better),
        )

        assert [match.record_uuid for match in at_edge.matches] == [
            match.record_uuid for match in ranked.matches[:3]
        ]
        assert [match.record_uuid for match in past_edge.matches] == [
            match.record_uuid for match in ranked.matches[:2]
        ]
        await edged.delete_partition(name)
        await edged.shutdown()


# ── Upsert behavior ──


class TestUpsertBehavior:
    @pytest.mark.asyncio
    async def test_upsert_replaces_vector_and_properties(self, partition):
        v1 = _normalize([1.0, 0.0, 0.0])
        v2 = _normalize([0.0, 1.0, 0.0])
        record_uuid = uuid4()

        await partition.upsert(
            records=[
                _make_record(uuid=record_uuid, vector=v1, properties={"name": "alice"})
            ]
        )
        await partition.upsert(
            records=[
                _make_record(uuid=record_uuid, vector=v2, properties={"name": "bob"})
            ]
        )

        assert await _stored(partition) == {record_uuid: {"name": "bob"}}

        results = await partition.query(query_vectors=[v2], limit=1)
        assert results[0].matches[0].record_uuid == record_uuid
        assert results[0].matches[0].score == pytest.approx(1.0, abs=0.01)


# ── Filter edge cases ──


class TestFilterEdgeCases:
    @pytest.mark.asyncio
    async def test_or_filter_does_not_match_all(self, partition):
        """Regression: OR filter with parenthesization must not bypass rowid constraint."""
        v1 = _normalize([1.0, 0.0, 0.0])
        r1 = _make_record(vector=v1, properties={"name": "alice"})
        r2 = _make_record(vector=v1, properties={"name": "bob"})
        r3 = _make_record(vector=v1, properties={"name": "carol"})
        await partition.upsert(records=[r1, r2, r3])

        results = await partition.query(
            query_vectors=[v1],
            limit=10,
            property_filter=Or(
                left=Comparison(field="name", op="=", value="alice"),
                right=Comparison(field="name", op="=", value="carol"),
            ),
        )
        uuids = {m.record_uuid for m in results[0].matches}
        assert r1.uuid in uuids
        assert r3.uuid in uuids
        assert r2.uuid not in uuids


# ── Partition resources ──


async def _table_names(engine) -> set[str]:
    async with engine.connect() as connection:
        return set(
            await connection.run_sync(
                lambda sync_connection: inspect(sync_connection).get_table_names()
            )
        )


class TestPartitionResources:
    """A partition's tables live and die with it."""

    @pytest.mark.asyncio
    async def test_deleting_a_partition_leaves_no_tables(self, store):
        engine = store._engine
        baseline = await _table_names(engine)
        partition = await store.open_or_create_partition(NAME)
        await partition.upsert(
            records=[_make_record(vector=_normalize([1.0, 0.0, 0.0]))]
        )
        assert await _table_names(engine) != baseline

        await store.delete_partition(NAME)

        assert await _table_names(engine) == baseline

    @pytest.mark.asyncio
    async def test_a_creation_that_loses_a_race_leaves_no_tables(self, store):
        engine = store._engine
        baseline = await _table_names(engine)
        await store.create_partition("reference")
        tables_per_partition = len(await _table_names(engine) - baseline)
        await store.delete_partition("reference")

        outcomes = await asyncio.gather(
            store.create_partition(NAME),
            store.create_partition(NAME),
            return_exceptions=True,
        )

        assert sorted(type(outcome).__name__ for outcome in outcomes) == [
            "NoneType",
            "VectorStorePartitionAlreadyExistsError",
        ]
        assert len(await _table_names(engine) - baseline) == tables_per_partition
        await store.delete_partition(NAME)
        assert await _table_names(engine) == baseline

    @pytest.mark.asyncio
    async def test_open_or_create_calls_racing_for_a_key_share_one_partition(
        self, store
    ):
        first, second = await asyncio.gather(
            store.open_or_create_partition(NAME),
            store.open_or_create_partition(NAME),
        )
        record = _make_record(vector=_normalize([1.0, 0.0, 0.0]))
        await first.upsert(records=[record])

        [found] = await second.query(query_vectors=[record.vector], limit=5)
        assert [match.record_uuid for match in found.matches] == [record.uuid]
        await store.delete_partition(NAME)
