"""Tests for SQLiteVectorStore."""

import asyncio
import math
from datetime import UTC, datetime, timedelta, timezone
from typing import Any
from uuid import UUID, uuid4

import pytest
import pytest_asyncio
from pydantic import ValidationError
from sqlalchemy import func, inspect, select, update
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from memmachine_server.common.data_types import PropertyType, SimilarityMetric
from memmachine_server.common.filter.filter_parser import (
    And,
    Comparison,
    In,
    Not,
    Or,
)
from memmachine_server.common.sql import (
    enable_sqlite_explicit_begin,
    enable_sqlite_foreign_keys,
)
from memmachine_server.common.vector_store import sqlite_vector_store
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
from memmachine_server.common.vector_store.sqlite_vector_store import (
    IndexLoadError,
    SQLiteVectorStore,
    SQLiteVectorStoreParams,
    SQLiteVectorStorePartition,
    _PartitionRow,
    _PendingOperationRow,
)
from memmachine_server.common.vector_store.vector_search_engine.usearch_engine import (
    USearchVectorSearchEngine,
)
from server_tests.memmachine_server.common.vector_store.declared_schema_contract import (
    DeclaredSchemaContract,
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

    Read past the store, from the table itself, so the answer is exact
    rather than a search engine's approximation.
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
    enable_sqlite_foreign_keys(engine)
    vector_store = SQLiteVectorStore(_params(engine))
    await vector_store.startup()
    yield vector_store
    await vector_store.shutdown()
    await engine.dispose()


def _engine_factory(ndim, metric):
    return USearchVectorSearchEngine(num_dimensions=ndim, similarity_metric=metric)


def _schema(**overrides) -> VectorStoreSchema:
    fields: dict[str, Any] = {
        "vector_dimensions": VECTOR_DIM,
        "similarity_metric": SimilarityMetric.COSINE,
        "indexed_properties": INDEXED_PROPERTIES,
    }
    fields.update(overrides)
    return VectorStoreSchema(**fields)


def _params(engine, **overrides) -> SQLiteVectorStoreParams:
    params: dict[str, Any] = {
        "sqlalchemy_engine": engine,
        "vector_store_name": VECTOR_STORE_NAME,
        "store_schema": _schema(),
        "vector_search_engine_factory": _engine_factory,
    }
    params.update(overrides)
    return SQLiteVectorStoreParams(**params)


async def _store_with(store, vector_store_name: str, **overrides) -> SQLiteVectorStore:
    """Another store over the fixture's engine: its own name, dimensions, or metric."""
    other = SQLiteVectorStore(
        _params(
            store._sqlalchemy_engine,
            vector_store_name=vector_store_name,
            store_schema=_schema(**overrides),
        )
    )
    await other.startup()
    return other


@pytest_asyncio.fixture
async def partition(store):
    await store.create_partition(NAME)
    partition = await store.open_partition(NAME)
    assert partition is not None
    yield partition
    await store.delete_partition(NAME)


# ── Partition lifecycle ──


class TestPartitionLifecycle:
    @pytest.mark.asyncio
    async def test_create_open_delete(self, store):
        await store.create_partition("lifecycle")
        partition = await store.open_partition("lifecycle")
        assert isinstance(partition, SQLiteVectorStorePartition)
        await store.delete_partition("lifecycle")

    @pytest.mark.asyncio
    async def test_open_returns_correct_type(self, store, partition):
        assert isinstance(await store.open_partition(NAME), SQLiteVectorStorePartition)

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
        assert isinstance(partition, SQLiteVectorStorePartition)
        await store.delete_partition("new")

    @pytest.mark.asyncio
    async def test_open_or_create_opens_when_exists(self, store):
        await store.create_partition("existing")
        partition = await store.open_or_create_partition("existing")
        assert isinstance(partition, SQLiteVectorStorePartition)
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

        monkeypatch.setattr(sqlite_vector_store, "uuid4", colliding_uuid4)
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
            raise sqlite_vector_store._RegistryInsertRejectedError(str(incarnation))

        monkeypatch.setattr(store, "_insert_partition", always_rejected)
        with pytest.raises(VectorStoreAttemptsExhaustedError):
            await store.create_partition("capped")
        assert len(attempts) == sqlite_vector_store._MAX_MINT_ATTEMPTS

    @pytest.mark.asyncio
    async def test_unsupported_metric_raises(self, store):
        with pytest.raises(ValueError, match="does not support"):
            await _store_with(
                store, "bad_metric", similarity_metric=SimilarityMetric.MANHATTAN
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
        # Euclidean: lower distance = better match; best match is first
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


# ── USearch-specific: dot product metric ──


class TestDotProductMetric:
    @pytest.mark.asyncio
    async def test_dot_product_supported(self, store):
        """Dot product is supported by USearch but not sqlite-vec."""
        dot = await _store_with(store, "dot", similarity_metric=SimilarityMetric.DOT)
        partition = await dot.open_or_create_partition("dot")
        v1 = _normalize([1.0, 0.0, 0.0])
        v2 = _normalize([0.0, 1.0, 0.0])
        r1 = _make_record(vector=v1)
        r2 = _make_record(vector=v2)
        await partition.upsert(records=[r1, r2])

        results = await partition.query(query_vectors=[v1], limit=2)
        assert len(results[0].matches) == 2

        await dot.delete_partition("dot")
        await dot.shutdown()


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
        [SimilarityMetric.COSINE, SimilarityMetric.DOT, SimilarityMetric.EUCLIDEAN],
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


# ── Concurrent async behavior ──


class TestConcurrentAsync:
    @pytest.mark.asyncio
    async def test_concurrent_upserts(self, partition):
        """Multiple concurrent upserts should not error and all records should be persisted."""
        import asyncio

        all_uuids: list[UUID] = []

        async def upsert_batch(start: int) -> None:
            records = [
                _make_record(vector=_normalize([float(i), 1.0, 0.0]))
                for i in range(start, start + 10)
            ]
            all_uuids.extend(r.uuid for r in records)
            await partition.upsert(records=records)

        await asyncio.gather(
            upsert_batch(0),
            upsert_batch(10),
            upsert_batch(20),
        )

        # Verify all records were persisted (deterministic, not ANN-dependent).
        assert set(await _stored(partition)) == set(all_uuids)

    @pytest.mark.asyncio
    async def test_concurrent_upsert_and_query(self, partition):
        """Query during upsert should not error (eventual consistency)."""
        import asyncio

        records = [
            _make_record(vector=_normalize([float(i), 1.0, 0.0])) for i in range(20)
        ]
        await partition.upsert(records=records)

        async def query_loop() -> None:
            for _ in range(5):
                await partition.query(
                    query_vectors=[_normalize([1.0, 1.0, 0.0])], limit=10
                )

        async def upsert_more() -> None:
            more_records = [
                _make_record(vector=_normalize([float(i), 0.0, 1.0]))
                for i in range(20, 30)
            ]
            await partition.upsert(records=more_records)

        await asyncio.gather(query_loop(), upsert_more())

    @pytest.mark.asyncio
    async def test_concurrent_first_opens_share_one_search_engine(self, tmp_path):
        db_path = tmp_path / "test.db"
        creator, creator_engine = await _fresh_store(db_path, tmp_path)
        await creator.create_partition(NAME)
        await creator.shutdown()
        await creator_engine.dispose()

        # A restarted store, whose first opens load the saved index.
        store, engine = await _fresh_store(db_path, tmp_path)
        first, second = await asyncio.gather(
            store.open_partition(NAME), store.open_partition(NAME)
        )
        assert first is not None
        assert second is not None
        record = _make_record(vector=_normalize([1.0, 0.0, 0.0]))
        await first.upsert(records=[record])
        [result] = await second.query(query_vectors=[record.vector], limit=1)
        assert [match.record_uuid for match in result.matches] == [record.uuid]

        await store.shutdown()
        await engine.dispose()

    @pytest.mark.asyncio
    @pytest.mark.parametrize("with_filter", [False, True])
    async def test_a_query_overtaken_by_its_partitions_deletion_raises_stale(
        self, store, monkeypatch, with_filter
    ):
        partition = await store.open_or_create_partition("overtaken")
        record = _make_record(
            vector=_normalize([1.0, 0.0, 0.0]), properties={"name": "alice"}
        )
        await partition.upsert(records=[record])
        engine = partition._search_engine
        search = engine.search

        async def search_overtaken_by_deletion(*args, **kwargs):
            # With a filter, the search reads the records table per
            # candidate; without one, the matches' rows are read after it.
            if with_filter:
                await store.delete_partition("overtaken")
                return await search(*args, **kwargs)
            results = await search(*args, **kwargs)
            await store.delete_partition("overtaken")
            return results

        monkeypatch.setattr(engine, "search", search_overtaken_by_deletion)
        with pytest.raises(VectorStorePartitionHandleStaleError):
            await partition.query(
                query_vectors=[record.vector],
                limit=1,
                property_filter=(
                    Comparison(field="name", op="=", value="alice")
                    if with_filter
                    else None
                ),
            )


# ── Crash recovery & pending operations ──


async def _fresh_store(db_path, tmp_path, *, save_threshold=1000):
    """Create a new SQLiteVectorStore against the same DB file."""
    engine = create_async_engine(f"sqlite+aiosqlite:///{db_path}")
    enable_sqlite_explicit_begin(engine)
    enable_sqlite_foreign_keys(engine)
    params = _params(
        engine,
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


async def _unapplied_operation_counts(engine) -> dict[str, int]:
    """Count unapplied pending operations by the vector store they belong to."""
    session_factory = async_sessionmaker(engine, expire_on_commit=False)
    async with session_factory() as session:
        rows = (
            await session.execute(
                select(_PartitionRow.vector_store_name, func.count())
                .join(
                    _PendingOperationRow,
                    _PendingOperationRow.incarnation == _PartitionRow.incarnation,
                )
                .where(_PendingOperationRow.applied.is_(False))
                .group_by(_PartitionRow.vector_store_name)
            )
        ).all()
    return {row[0]: row[1] for row in rows}


class TestCrashRecovery:
    """Tests for pending operations replay on startup."""

    @pytest.mark.asyncio
    async def test_replay_upserts_after_crash(self, tmp_path):
        """Records upserted before crash are queryable after restart."""
        db_path = tmp_path / "test.db"
        store1, engine1 = await _fresh_store(db_path, tmp_path)

        partition = await store1.open_or_create_partition(NAME)
        records = [
            _make_record(vector=_normalize([1.0, 0.0, 0.0])),
            _make_record(vector=_normalize([0.0, 1.0, 0.0])),
        ]
        await partition.upsert(records=records)

        # Simulate crash: dispose without shutdown (pending ops remain).
        await engine1.dispose()

        # Restart with fresh in-memory engines.
        store2, engine2 = await _fresh_store(db_path, tmp_path)
        partition_2 = await store2.open_partition(NAME)
        assert partition_2 is not None

        results = await partition_2.query(
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

        partition = await store1.open_or_create_partition(NAME)
        records = [
            _make_record(vector=_normalize([1.0, 0.0, 0.0])),
            _make_record(vector=_normalize([0.0, 1.0, 0.0])),
        ]
        await partition.upsert(records=records)

        # Simulate crash between SQLite commit and engine apply:
        # mark all pending ops as unapplied.
        await _set_all_pending_operations_unapplied(engine1)
        await engine1.dispose()

        # Restart: replay should re-apply unapplied ops to the engine.
        store2, engine2 = await _fresh_store(db_path, tmp_path)
        partition_2 = await store2.open_partition(NAME)
        assert partition_2 is not None

        results = await partition_2.query(
            query_vectors=[_normalize([1.0, 0.0, 0.0])], limit=10
        )
        assert len(results[0].matches) == 2

        await store2.shutdown()
        await engine2.dispose()

    @pytest.mark.asyncio
    async def test_replay_applies_only_its_own_stores_operations(self, tmp_path):
        """A store's startup replays its own partitions' operations, not another store's."""
        db_path = tmp_path / "test.db"
        store1, engine1 = await _fresh_store(db_path, tmp_path)
        other = await _store_with(store1, "other_store")
        for vector_store in (store1, other):
            partition = await vector_store.open_or_create_partition(NAME)
            await partition.upsert(
                records=[_make_record(vector=_normalize([1.0, 0.0, 0.0]))]
            )

        await _set_all_pending_operations_unapplied(engine1)
        await engine1.dispose()

        store2, engine2 = await _fresh_store(db_path, tmp_path)
        assert await _unapplied_operation_counts(engine2) == {"other_store": 1}

        await store2.shutdown()
        await engine2.dispose()

    @pytest.mark.asyncio
    async def test_replay_deletes_after_crash(self, tmp_path):
        """Pending delete operations are replayed on restart."""
        db_path = tmp_path / "test.db"
        store1, engine1 = await _fresh_store(db_path, tmp_path)

        partition = await store1.open_or_create_partition(NAME)
        r1 = _make_record(vector=_normalize([1.0, 0.0, 0.0]))
        r2 = _make_record(vector=_normalize([0.0, 1.0, 0.0]))
        await partition.upsert(records=[r1, r2])
        await partition.delete(record_uuids=[r1.uuid])

        # Simulate crash.
        await engine1.dispose()

        store2, engine2 = await _fresh_store(db_path, tmp_path)
        partition_2 = await store2.open_partition(NAME)
        assert partition_2 is not None

        results = await partition_2.query(
            query_vectors=[_normalize([1.0, 0.0, 0.0])], limit=10
        )
        assert len(results[0].matches) == 1

        assert r1.uuid not in await _stored(partition_2)

        await store2.shutdown()
        await engine2.dispose()

    @pytest.mark.asyncio
    async def test_replay_mixed_upserts_and_deletes(self, tmp_path):
        """Mixed upsert and delete pending ops are replayed correctly."""
        db_path = tmp_path / "test.db"
        store1, engine1 = await _fresh_store(db_path, tmp_path)

        partition = await store1.open_or_create_partition(NAME)
        r1 = _make_record(vector=_normalize([1.0, 0.0, 0.0]))
        r2 = _make_record(vector=_normalize([0.0, 1.0, 0.0]))
        r3 = _make_record(vector=_normalize([0.0, 0.0, 1.0]))
        await partition.upsert(records=[r1, r2, r3])
        await partition.delete(record_uuids=[r2.uuid])

        # Simulate crash.
        await engine1.dispose()

        store2, engine2 = await _fresh_store(db_path, tmp_path)
        partition_2 = await store2.open_partition(NAME)
        assert partition_2 is not None

        results = await partition_2.query(
            query_vectors=[_normalize([1.0, 0.0, 0.0])], limit=10
        )
        assert len(results[0].matches) == 2

        assert set(await _stored(partition_2)) == {r1.uuid, r3.uuid}

        await store2.shutdown()
        await engine2.dispose()

    @pytest.mark.asyncio
    async def test_save_threshold_clears_applied_ops(self, tmp_path):
        """Applied pending ops are deleted after save threshold is reached."""
        db_path = tmp_path / "test.db"
        store1, engine1 = await _fresh_store(db_path, tmp_path, save_threshold=2)

        partition = await store1.open_or_create_partition(NAME)

        # Upsert 1 record: below threshold, pending op should remain.
        r1 = _make_record(vector=_normalize([1.0, 0.0, 0.0]))
        await partition.upsert(records=[r1])
        assert await _pending_operation_count(engine1) == 1

        # Upsert 1 more: reaches threshold of 2, should trigger save + cleanup.
        r2 = _make_record(vector=_normalize([0.0, 1.0, 0.0]))
        await partition.upsert(records=[r2])
        assert await _pending_operation_count(engine1) == 0

        await store1.shutdown()
        await engine1.dispose()

    @pytest.mark.asyncio
    async def test_cascade_deletes_pending_ops(self, tmp_path):
        """Deleting a partition cascades to its pending operations."""
        db_path = tmp_path / "test.db"
        store1, engine1 = await _fresh_store(db_path, tmp_path)

        partition = await store1.open_or_create_partition(NAME)
        records = [
            _make_record(vector=_normalize([1.0, 0.0, 0.0])),
            _make_record(vector=_normalize([0.0, 1.0, 0.0])),
        ]
        await partition.upsert(records=records)
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
        enable_sqlite_explicit_begin(engine)
        enable_sqlite_foreign_keys(engine)
        store = SQLiteVectorStore(_params(engine))

        with pytest.raises(RuntimeError, match="startup"):
            await store.open_partition(NAME)

        with pytest.raises(RuntimeError, match="startup"):
            await store.create_partition(NAME)

        with pytest.raises(RuntimeError, match="startup"):
            await store.delete_partition(NAME)

        await engine.dispose()


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
        """A freshly created partition has index_saved=False."""
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

        partition = await store.open_or_create_partition(NAME)
        await partition.upsert(
            records=[_make_record(vector=_normalize([1.0, 0.0, 0.0]))]
        )

        assert await _get_index_saved(engine, NAME) is True

        await store.shutdown()
        await engine.dispose()

    @pytest.mark.asyncio
    async def test_saved_flag_flips_on_clean_shutdown(self, tmp_path):
        """Clean shutdown flips index_saved to True even below save_threshold."""
        db_path = tmp_path / "test.db"
        store, engine = await _fresh_store(db_path, tmp_path, save_threshold=1000)

        partition = await store.open_or_create_partition(NAME)
        await partition.upsert(
            records=[_make_record(vector=_normalize([1.0, 0.0, 0.0]))]
        )

        # Below save_threshold, so _maybe_save_index has not flipped the flag yet.
        assert await _get_index_saved(engine, NAME) is False

        await store.shutdown()
        assert await _get_index_saved(engine, NAME) is True

        await engine.dispose()

    @pytest.mark.asyncio
    async def test_shutdown_closes_the_connections_a_filtered_query_opened(
        self, tmp_path
    ):
        db_path = tmp_path / "test.db"
        store, engine = await _fresh_store(db_path, tmp_path)

        partition = await store.open_or_create_partition(NAME)
        vector = _normalize([1.0, 0.0, 0.0])
        await partition.upsert(
            records=[_make_record(vector=vector, properties={"name": "alice"})]
        )
        await partition.query(
            query_vectors=[vector],
            limit=1,
            property_filter=Comparison(field="name", op="=", value="alice"),
        )
        pool = store._sync_sqlalchemy_engine.pool
        assert pool.checkedin() == 1

        await store.shutdown()
        assert pool.checkedin() == 0

        await engine.dispose()

    @pytest.mark.asyncio
    async def test_missing_file_when_saved_raises(self, tmp_path):
        """If the index has been saved, a missing file is loud, not silent."""
        db_path = tmp_path / "test.db"
        store1, engine1 = await _fresh_store(db_path, tmp_path)

        partition = await store1.open_or_create_partition(NAME)
        await partition.upsert(
            records=[_make_record(vector=_normalize([1.0, 0.0, 0.0]))]
        )
        await store1.shutdown()
        await engine1.dispose()

        # Operator deletes the index file out from under us.
        index_dir = tmp_path / "indexes"
        idx_files = list(index_dir.glob("*.idx"))
        assert len(idx_files) == 1
        idx_files[0].unlink()

        # Restart: open_partition must surface the failure, not return an
        # engine silently rebuilt empty.
        store2, engine2 = await _fresh_store(db_path, tmp_path)
        with pytest.raises(IndexLoadError) as exc_info:
            await store2.open_partition(NAME)
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

        partition = await store1.open_or_create_partition(NAME)
        await partition.upsert(
            records=[_make_record(vector=_normalize([1.0, 0.0, 0.0]))]
        )
        await store1.shutdown()
        await engine1.dispose()

        index_dir = tmp_path / "indexes"
        idx_files = list(index_dir.glob("*.idx"))
        assert len(idx_files) == 1
        idx_files[0].write_bytes(b"not a valid index file")

        store2, engine2 = await _fresh_store(db_path, tmp_path)
        with pytest.raises(IndexLoadError) as exc_info:
            await store2.open_partition(NAME)
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

        partition = await store1.open_or_create_partition(NAME)
        await partition.upsert(
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
        partition_2 = await store2.open_partition(NAME)
        assert partition_2 is not None

        results = await partition_2.query(
            query_vectors=[_normalize([1.0, 0.0, 0.0])], limit=10
        )
        assert len(results[0].matches) == 2

        await store2.shutdown()
        await engine2.dispose()


# ── Partition resources ──


async def _table_names(engine) -> set[str]:
    async with engine.connect() as connection:
        return set(
            await connection.run_sync(
                lambda sync_connection: inspect(sync_connection).get_table_names()
            )
        )


class TestPartitionResources:
    """A partition's tables and index file live and die with it."""

    @pytest.mark.asyncio
    async def test_deleting_a_partition_leaves_no_tables_or_index_files(self, tmp_path):
        store, engine = await _fresh_store(
            tmp_path / "test.db", tmp_path, save_threshold=1
        )
        baseline = await _table_names(engine)
        partition = await store.open_or_create_partition(NAME)
        await partition.upsert(
            records=[_make_record(vector=_normalize([1.0, 0.0, 0.0]))]
        )
        assert len(await _table_names(engine) - baseline) == 1
        assert len(list((tmp_path / "indexes").glob("*.idx"))) == 1

        await store.delete_partition(NAME)

        assert await _table_names(engine) == baseline
        assert list((tmp_path / "indexes").glob("*.idx")) == []

        await store.shutdown()
        await engine.dispose()

    @pytest.mark.asyncio
    async def test_a_creation_that_loses_a_race_leaves_the_winners_engine(
        self, tmp_path
    ):
        """The winner's records survive a restart, saved from the engine its handle holds."""
        db_path = tmp_path / "test.db"
        store1, engine1 = await _fresh_store(db_path, tmp_path)
        outcomes = await asyncio.gather(
            store1.open_or_create_partition(NAME),
            store1.open_or_create_partition(NAME),
            return_exceptions=True,
        )
        partitions = [
            outcome
            for outcome in outcomes
            if isinstance(outcome, SQLiteVectorStorePartition)
        ]
        assert len(partitions) == 2
        record = _make_record(vector=_normalize([1.0, 0.0, 0.0]))
        await partitions[0].upsert(records=[record])
        await store1.shutdown()
        await engine1.dispose()

        store2, engine2 = await _fresh_store(db_path, tmp_path)
        partition = await store2.open_partition(NAME)
        assert partition is not None
        [found] = await partition.query(query_vectors=[record.vector], limit=5)
        assert [match.record_uuid for match in found.matches] == [record.uuid]

        await store2.shutdown()
        await engine2.dispose()

    @pytest.mark.asyncio
    async def test_a_creation_that_loses_a_race_leaves_no_tables(self, store):
        engine = store._sqlalchemy_engine
        baseline = await _table_names(engine)

        outcomes = await asyncio.gather(
            store.create_partition(NAME),
            store.create_partition(NAME),
            return_exceptions=True,
        )

        assert sorted(type(outcome).__name__ for outcome in outcomes) == [
            "NoneType",
            "VectorStorePartitionAlreadyExistsError",
        ]
        assert len(await _table_names(engine) - baseline) == 1
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

    @pytest.mark.asyncio
    async def test_a_save_that_a_deletion_overtakes_leaves_no_index_file(
        self, tmp_path
    ):
        """No file holds a deleted partition's vectors, whichever finishes first."""
        stores: list[SQLiteVectorStore] = []

        class _DeletedWhileSaving(USearchVectorSearchEngine):
            async def save(self, path: str) -> None:
                await stores[0].delete_partition(NAME)
                await super().save(path)

        engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'test.db'}")
        enable_sqlite_explicit_begin(engine)
        enable_sqlite_foreign_keys(engine)
        store = SQLiteVectorStore(
            _params(
                engine,
                index_directory=str(tmp_path / "indexes"),
                save_threshold=1,
                vector_search_engine_factory=lambda ndim, metric: _DeletedWhileSaving(
                    num_dimensions=ndim, similarity_metric=metric
                ),
            )
        )
        stores.append(store)
        await store.startup()
        partition = await store.open_or_create_partition(NAME)

        # Crossing the save threshold saves the index, and the deletion
        # commits while the file is written.
        await partition.upsert(
            records=[_make_record(vector=_normalize([1.0, 0.0, 0.0]))]
        )

        assert await store.open_partition(NAME) is None
        assert list((tmp_path / "indexes").glob("*.idx")) == []

        await store.shutdown()
        await engine.dispose()
