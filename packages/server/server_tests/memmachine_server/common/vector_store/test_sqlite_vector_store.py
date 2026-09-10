"""Tests for SQLiteVectorStore."""

import math
from datetime import UTC, datetime, timedelta, timezone
from uuid import UUID, uuid4

import pytest
import pytest_asyncio
from pydantic import ValidationError
from sqlalchemy import func, select, update
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from memmachine_server.common.data_types import PropertyValue
from memmachine_server.common.filter import (
    And,
    Equals,
    In,
    Not,
    Or,
    Ordering,
)
from memmachine_server.common.vector_store.data_types import (
    IndexedPropertiesMismatchError,
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
from server_tests.memmachine_server.common.filter.nodes import comparison
from server_tests.memmachine_server.common.vector_store.declared_schema_contract import (
    DeclaredSchemaContract,
)

NAMESPACE = "test_namespace"
NAME = "test_name"
VECTOR_DIM = 3

INDEXED_PROPERTIES: dict[str, type[PropertyValue]] = {
    "name": str,
    "age": int,
    "score": float,
    "active": bool,
    "created_at": datetime,
}


def _normalize(vector: list[float]) -> list[float]:
    magnitude = math.sqrt(sum(x * x for x in vector))
    return [x / magnitude for x in vector]


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
    dimensions = collection.config.vector_dimensions
    probe = [1.0] + [0.0] * (dimensions - 1)
    [result] = await collection.query(query_vectors=[probe], limit=_FETCH_LIMIT)
    present = {match.record_uuid for match in result.matches}
    return [uuid for uuid in record_uuids if uuid in present]


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


@pytest_asyncio.fixture
async def store(tmp_path):
    db_path = tmp_path / "test.db"
    engine = create_async_engine(f"sqlite+aiosqlite:///{db_path}")
    params = SQLiteVectorStoreParams(
        indexed_properties=INDEXED_PROPERTIES,
        sqlalchemy_engine=engine,
        vector_search_engine_factory=lambda ndim: USearchVectorSearchEngine(
            num_dimensions=ndim
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
            property_filter=Equals(field="name", value="updated"),
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


class TestDeclaredSchema(DeclaredSchemaContract):
    """The declared-schema contract, against this store."""


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
            property_filter=comparison(field, op, value),
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
            property_filter=Equals(field="created_at", value=dt),
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
            property_filter=Equals(field="created_at", value=dt.astimezone(UTC)),
        )
        assert [m.record_uuid for m in result.matches] == [r1.uuid]

    # ── In / And / Or / Not ──

    @pytest.mark.asyncio
    async def test_in(self, collection):
        r1, _r2, r3, v1 = await self._setup(collection)
        query_results = await collection.query(
            query_vectors=[v1],
            limit=10,
            property_filter=In(field="name", values=("alice", "carol")),
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
                (
                    Equals(field="active", value=True),
                    Ordering(field="age", op=">", value=30),
                )
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
                (
                    Equals(field="name", value="alice"),
                    Equals(field="name", value="carol"),
                )
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
            property_filter=Not(Ordering(field="age", op=">", value=30)),
        )
        uuids = {match.record_uuid for match in query_results[0].matches}
        assert r1.uuid in uuids
        assert r2.uuid in uuids
        assert len(uuids) == 2


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

        uuids_a = {match.record_uuid for match in results_a[0].matches}
        uuids_b = {match.record_uuid for match in results_b[0].matches}
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

        results = await _present_uuids(coll_a, [r1.uuid, r2.uuid])
        assert len(results) == 1
        assert results[0] == r1.uuid

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

        results = await _present_uuids(coll_b, [r2.uuid])
        assert len(results) == 1
        assert results[0] == r2.uuid

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
        assert {match.record_uuid for match in results_a[0].matches} == {r1.uuid}

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
        assert results[0].matches[0].record_uuid == r2.uuid

        await store.delete_collection(namespace=NAMESPACE, name="sibling_b")


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


# ── Input validation ──


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


class TestCosineSimilaritySemantics:
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
            property_filter=Equals(field="name", value="bob"),
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
        fetched = await _present_uuids(collection, all_uuids)
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


# ── Crash recovery & pending operations ──


def _engine_factory(ndim):
    return USearchVectorSearchEngine(num_dimensions=ndim)


CONFIG = VectorStoreCollectionConfig(
    vector_dimensions=VECTOR_DIM,
)


async def _fresh_store(db_path, tmp_path, *, save_threshold=1000):
    """Create a new SQLiteVectorStore against the same DB file."""
    engine = create_async_engine(f"sqlite+aiosqlite:///{db_path}")
    params = SQLiteVectorStoreParams(
        indexed_properties=INDEXED_PROPERTIES,
        sqlalchemy_engine=engine,
        vector_search_engine_factory=_engine_factory,
        index_directory=str(tmp_path / "indexes"),
        save_threshold=save_threshold,
    )
    store = SQLiteVectorStore(params)
    await store.startup()
    return store, engine


async def _stored_record_uuids(engine, store) -> set[UUID]:
    """Read a collection's record UUIDs straight out of SQLite.

    The collection contract cannot see a record whose vector the index lost, so
    a test about surviving that loss has to look past the contract at the row.
    """
    records_table = store._records_table(NAMESPACE, NAME)
    session_factory = async_sessionmaker(engine, expire_on_commit=False)
    async with session_factory() as session:
        rows = (await session.execute(select(records_table.c.uuid))).all()
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

        fetched = await _present_uuids(coll2, [r1.uuid])
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
                indexed_properties=INDEXED_PROPERTIES,
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

    @pytest.mark.asyncio
    async def test_a_reverted_publication_costs_search_not_the_record_row(
        self, tmp_path
    ):
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

        coll = await store1.open_or_create_collection(
            namespace=NAMESPACE, name=NAME, config=CONFIG
        )
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
        coll2 = await store2.open_collection(namespace=NAMESPACE, name=NAME)
        assert coll2 is not None

        # The record is still a row.
        assert await _stored_record_uuids(engine2, store2) == {r1.uuid, r2.uuid}

        # The index just cannot find it any more.
        results = await coll2.query(query_vectors=[r2.vector], limit=10)
        assert [match.record_uuid for match in results[0].matches] == [r1.uuid]

        await store2.shutdown()
        await engine2.dispose()


class TestDeclaredSchemaIsFixed:
    @pytest.mark.asyncio
    async def test_a_store_with_another_schema_cannot_open_the_collection(
        self, tmp_path
    ):
        db_path = tmp_path / "fixed.db"
        engine = create_async_engine(f"sqlite+aiosqlite:///{db_path}")
        first = SQLiteVectorStore(
            SQLiteVectorStoreParams(
                indexed_properties=INDEXED_PROPERTIES,
                sqlalchemy_engine=engine,
                vector_search_engine_factory=lambda ndim: USearchVectorSearchEngine(
                    num_dimensions=ndim
                ),
            )
        )
        await first.startup()
        await first.create_collection(
            namespace=NAMESPACE,
            name="fixed",
            config=VectorStoreCollectionConfig(vector_dimensions=VECTOR_DIM),
        )
        await first.shutdown()

        second = SQLiteVectorStore(
            SQLiteVectorStoreParams(
                indexed_properties={"name": str},
                sqlalchemy_engine=engine,
                vector_search_engine_factory=lambda ndim: USearchVectorSearchEngine(
                    num_dimensions=ndim
                ),
            )
        )
        await second.startup()
        with pytest.raises(IndexedPropertiesMismatchError, match="fixed"):
            await second.open_collection(namespace=NAMESPACE, name="fixed")
        await second.shutdown()
        await engine.dispose()
