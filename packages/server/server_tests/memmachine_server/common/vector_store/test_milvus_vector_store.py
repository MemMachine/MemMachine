"""Tests for MilvusVectorStore."""

# ruff: noqa: E402

import math
from datetime import UTC, datetime, timedelta, timezone
from typing import Any
from unittest.mock import MagicMock
from uuid import UUID, uuid4

import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import create_async_engine

from server_tests.memmachine_server.common.vector_store.partition_lifecycle_contract import (
    PartitionLifecycleContract,
)

pymilvus = pytest.importorskip("pymilvus")
DataType = pymilvus.DataType
MilvusClient = pymilvus.MilvusClient

from memmachine_server.common.data_types import PropertyType
from memmachine_server.common.filter import (
    And,
    Equals,
    In,
    IsNull,
    Not,
    Or,
    Ordering,
)
from memmachine_server.common.vector_store.data_types import (
    Record,
    VectorStorePartitionAlreadyExistsError,
    VectorStorePartitionSchemaMismatchError,
)
from memmachine_server.common.vector_store.milvus_vector_store import (
    MilvusVectorStore,
    MilvusVectorStoreParams,
    MilvusVectorStorePartition,
)
from memmachine_server.common.vector_store.partition_registry.sqlalchemy_partition_registry import (
    SQLAlchemyVectorStorePartitionRegistry,
)
from server_tests.memmachine_server.common.filter.nodes import comparison
from server_tests.memmachine_server.common.vector_store.declared_schema_contract import (
    DeclaredSchemaContract,
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
# Tombstones come due at once, so a test can purge right after deleting.
TOMBSTONE_RETENTION = timedelta(0)
REQUEST_TIMEOUT_SECONDS = 30
MAX_VARCHAR_LENGTH = 1024
PURGE_BATCH_SIZE = 10000


def _normalize(vector: list[float]) -> list[float]:
    magnitude = math.sqrt(sum(x * x for x in vector))
    return [x / magnitude for x in vector]


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


async def _params(client, registry_engine, **overrides) -> MilvusVectorStoreParams:
    """Parameters for one store: its own provisioned registry over the shared registry database."""
    params: dict[str, Any] = {
        "client": client,
        "vector_store_name": VECTOR_STORE_NAME,
        "vector_dimensions": VECTOR_DIM,
        "indexed_properties": INDEXED_PROPERTIES,
        "consistency_level": "Session",
        "request_timeout_seconds": REQUEST_TIMEOUT_SECONDS,
        "max_varchar_length": MAX_VARCHAR_LENGTH,
        "purge_batch_size": PURGE_BATCH_SIZE,
    }
    params.update(overrides)
    params["partition_registry"] = SQLAlchemyVectorStorePartitionRegistry(
        engine=registry_engine,
        vector_store_name=params["vector_store_name"],
        tombstone_retention=TOMBSTONE_RETENTION,
    )
    await params["partition_registry"].provision()
    return MilvusVectorStoreParams(**params)


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


@pytest.fixture
def server_milvus_client(milvus_container):
    client = MilvusClient(uri=milvus_container.get_connection_url())
    yield client
    client.close()


@pytest.fixture(
    params=[pytest.param("server_milvus_client", marks=pytest.mark.integration)],
)
def milvus_client(request):
    return request.getfixturevalue(request.param)


@pytest_asyncio.fixture
async def store(milvus_client, tmp_path):
    registry_engine = create_async_engine(
        f"sqlite+aiosqlite:///{tmp_path / 'registry.db'}"
    )
    vector_store = MilvusVectorStore(await _params(milvus_client, registry_engine))
    await vector_store.provision()
    await vector_store.startup()
    yield vector_store
    await vector_store.shutdown()
    await registry_engine.dispose()


# MilvusClient's constructor timeout bounds only the connection; a request
# is bounded only by the timeout passed to it, so every request must carry it.
_CLIENT_REQUESTS = (
    "has_collection",
    "create_collection",
    "query",
    "upsert",
    "search",
    "delete",
)


@pytest.mark.asyncio
async def test_every_request_carries_the_timeout(store, monkeypatch):
    spies = {}
    for name in _CLIENT_REQUESTS:
        spies[name] = MagicMock(wraps=getattr(store._client, name))
        monkeypatch.setattr(store._client, name, spies[name])

    # A collection of its own, so provisioning creates it through the spies.
    timed = MilvusVectorStore(
        await _params(
            store._client, store._partition_registry._engine, vector_store_name="timed"
        )
    )
    await timed.provision()
    await timed.startup()
    await timed.create_partition("timed")
    coll = await timed.get_partition("timed")
    assert coll is not None
    vector = _normalize([1.0, 0.0, 0.0])
    record, kept = (
        _make_record(vector=vector),
        _make_record(vector=_normalize([0.0, 1.0, 0.0])),
    )
    await coll.upsert(records=[record, kept])
    await coll.query(query_vectors=[vector], limit=1)
    await coll.delete(record_uuids=[record.uuid])
    await timed.delete_partition("timed")
    # The purge finds the record the deletion left and reclaims it.
    while await timed.purge_deleted_partitions():
        pass

    assert {name for name, spy in spies.items() if spy.call_count} >= set(
        _CLIENT_REQUESTS
    )
    for name, spy in spies.items():
        for call in spy.call_args_list:
            assert call.kwargs.get("timeout") == REQUEST_TIMEOUT_SECONDS, (name, call)


@pytest_asyncio.fixture
async def collection(store):
    await store.create_partition(NAME)
    coll = await store.get_partition(NAME)
    assert coll is not None
    yield coll
    await store.delete_partition(NAME)


class TestPartitionLifecycle:
    @pytest.mark.asyncio
    async def test_create_open_delete(self, store):
        await store.create_partition("lifecycle")
        coll = await store.get_partition("lifecycle")
        assert isinstance(coll, MilvusVectorStorePartition)
        await store.delete_partition("lifecycle")

    @pytest.mark.asyncio
    async def test_duplicate_name_raises(self, store, collection):
        with pytest.raises(VectorStorePartitionAlreadyExistsError):
            await store.create_partition(NAME)

    @pytest.mark.asyncio
    async def test_delete_nonexistent_is_idempotent(self, store):
        await store.delete_partition("nonexistent")

    @pytest.mark.asyncio
    async def test_a_store_with_another_schema_cannot_open_the_partition(self, store):
        """The schema is the collection's: another store of it must declare the same."""
        await store.create_partition("mismatch")
        other_dimensions = MilvusVectorStore(
            await _params(
                store._client,
                store._partition_registry._engine,
                vector_dimensions=VECTOR_DIM + 1,
            )
        )
        with pytest.raises(VectorStorePartitionSchemaMismatchError, match="mismatch"):
            await other_dimensions.open_or_create_partition("mismatch")
        other_keys = MilvusVectorStore(
            await _params(
                store._client,
                store._partition_registry._engine,
                indexed_properties={"name": str},
            )
        )
        with pytest.raises(VectorStorePartitionSchemaMismatchError, match="mismatch"):
            await other_keys.get_partition("mismatch")
        await store.delete_partition("mismatch")

    @pytest.mark.asyncio
    async def test_a_vector_store_name_may_begin_with_a_digit(self, store):
        """Milvus refuses a collection name beginning with a digit; the store's
        collection name begins with its prefix instead."""
        digit_first = MilvusVectorStore(
            await _params(
                store._client,
                store._partition_registry._engine,
                vector_store_name="0_digit_first",
            )
        )
        await digit_first.provision()
        await digit_first.create_partition("digit_first")
        assert await digit_first.get_partition("digit_first") is not None
        await digit_first.delete_partition("digit_first")

    @pytest.mark.asyncio
    async def test_partitions_share_the_store_collection(self, store):
        """Every partition is a partition-key value inside the store's one native collection."""
        await store.create_partition("coll_a")
        await store.create_partition("coll_b")

        coll_a = await store.get_partition("coll_a")
        coll_b = await store.get_partition("coll_b")
        assert coll_a is not None
        assert coll_b is not None
        assert coll_a._collection_name == store._collection_name
        assert coll_b._collection_name == store._collection_name

        await store.delete_partition("coll_a")
        await store.delete_partition("coll_b")

    @pytest.mark.asyncio
    async def test_native_collection_schema(self, store):
        """Each declared property is a typed, nullable, indexed field; the
        collection isolates tenants."""
        await store.create_partition("schema")
        coll = await store.get_partition("schema")
        assert coll is not None
        native = coll._collection_name

        schema = store._client.describe_collection(coll._collection_name)
        fields = {field["name"]: field for field in schema["fields"]}
        assert schema["auto_id"] is False
        assert schema["enable_dynamic_field"] is False
        assert schema["properties"]["partitionkey.isolation"] == "True"
        assert fields["id"]["is_primary"] is True
        assert fields["partition_key"]["is_partition_key"] is True
        assert fields["vector"]["type"] == DataType.FLOAT_VECTOR
        assert fields["vector"]["params"]["dim"] == VECTOR_DIM
        expected = {
            "_p_name": DataType.VARCHAR,
            "_p_age": DataType.INT64,
            "_p_score": DataType.DOUBLE,
            "_p_active": DataType.BOOL,
            "_p_created_at": DataType.TIMESTAMPTZ,
        }
        for field_name, data_type in expected.items():
            assert fields[field_name]["type"] == data_type
            assert fields[field_name]["nullable"] is True
        assert fields["_p_name"]["params"]["max_length"] == MAX_VARCHAR_LENGTH
        indexed = {
            store._client.describe_index(native, index_name)["field_name"]
            for index_name in store._client.list_indexes(native)
        }
        assert indexed == {
            "vector",
            "_p_name",
            "_p_age",
            "_p_score",
            "_p_active",
            "_p_created_at",
        }

        await store.delete_partition("schema")


class TestUpsertAndQuery:
    @pytest.mark.asyncio
    async def test_upsert_calls_native_upsert(self, collection, monkeypatch):
        captured_kwargs = None

        def tracked_upsert(**kwargs):
            nonlocal captured_kwargs
            captured_kwargs = kwargs

        def fail_insert(*args, **kwargs):
            pytest.fail("collection upsert must not call MilvusClient.insert")

        monkeypatch.setattr(collection._client, "upsert", tracked_upsert)
        monkeypatch.setattr(collection._client, "insert", fail_insert)

        record = _make_record(
            vector=_normalize([1.0, 0.0, 0.0]),
            properties={"name": "test"},
        )
        await collection.upsert(records=[record])

        assert captured_kwargs is not None
        assert captured_kwargs["collection_name"] == collection._collection_name
        assert captured_kwargs["data"] == [collection._build_entity(record)]

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
        assert matches[1].record_uuid == r3.uuid
        assert matches[2].record_uuid == r2.uuid
        assert (
            matches[0].cosine_similarity
            >= matches[1].cosine_similarity
            >= matches[2].cosine_similarity
        )
        assert matches[0].cosine_similarity == pytest.approx(1.0)
        assert matches[2].cosine_similarity == pytest.approx(0.0)

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
    async def test_upsert_removes_stale_filter_fields(self, collection):
        v1 = _normalize([1.0, 0.0, 0.0])
        record = _make_record(vector=v1, properties={"name": "old"})
        await collection.upsert(records=[record])

        await collection.upsert(
            records=[Record(uuid=record.uuid, vector=v1, properties={})]
        )

        results = await collection.query(
            query_vectors=[v1],
            limit=10,
            property_filter=IsNull(field="name"),
        )
        assert {match.record_uuid for match in results[0].matches} == {record.uuid}

    @pytest.mark.asyncio
    async def test_upsert_failure_preserves_existing_record(
        self, collection, monkeypatch
    ):
        old_vector = _normalize([1.0, 0.0, 0.0])
        new_vector = _normalize([0.0, 1.0, 0.0])
        record = _make_record(vector=old_vector, properties={"name": "old"})
        await collection.upsert(records=[record])

        def fail_upsert(*args, **kwargs):
            raise RuntimeError("upsert failed")

        monkeypatch.setattr(collection._client, "upsert", fail_upsert)

        with pytest.raises(RuntimeError, match="upsert failed"):
            await collection.upsert(
                records=[
                    Record(
                        uuid=record.uuid,
                        vector=new_vector,
                        properties={"name": "new"},
                    )
                ]
            )

        assert await _present_uuids(collection, [record.uuid]) == [record.uuid]

        # The old vector is still the indexed one.
        results = await collection.query(query_vectors=[old_vector], limit=1)
        assert results[0].matches[0].cosine_similarity == pytest.approx(1.0, abs=0.01)


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

    async def _query(self, collection, query_vec, field, op, value):
        all_results = await collection.query(
            query_vectors=[query_vec],
            limit=10,
            property_filter=comparison(field, op, value),
        )
        return {match.record_uuid for match in all_results[0].matches}

    @pytest.mark.asyncio
    async def test_scalar_filters(self, collection):
        r1, r2, r3, v1 = await self._setup(collection)

        assert await self._query(collection, v1, "name", "=", "alice") == {r1.uuid}
        assert await self._query(collection, v1, "name", "!=", "alice") == {
            r2.uuid,
            r3.uuid,
        }
        assert await self._query(collection, v1, "age", ">", 30) == {r3.uuid}
        assert await self._query(collection, v1, "age", "<=", 30) == {
            r1.uuid,
            r2.uuid,
        }
        assert await self._query(collection, v1, "score", ">=", 8.0) == {
            r1.uuid,
            r3.uuid,
        }
        assert await self._query(collection, v1, "active", "=", True) == {
            r1.uuid,
            r3.uuid,
        }

    @pytest.mark.asyncio
    async def test_datetime_filters_and_roundtrip(self, collection):
        v1 = _normalize([1.0, 0.0, 0.0])
        v2 = _normalize([1.0, 0.1, 0.0])
        dt_utc = datetime(2024, 6, 15, 12, 0, 0, tzinfo=UTC)
        dt_other = datetime(2024, 6, 15, 18, 0, 0, tzinfo=UTC)
        r1 = _make_record(vector=v1, properties={"created_at": dt_utc})
        r2 = _make_record(vector=v2, properties={"created_at": dt_other})
        await collection.upsert(records=[r1, r2])

        plus5 = timezone(timedelta(hours=5))
        dt_filter = datetime(2024, 6, 15, 17, 0, 0, tzinfo=plus5)
        uuids = await self._query(collection, v1, "created_at", "=", dt_filter)
        assert uuids == {r1.uuid}

        # Already asserted through the filter above, which is the only way
        # properties are observable.

    @pytest.mark.asyncio
    async def test_is_null_and_not_null(self, collection):
        v1 = _normalize([1.0, 0.0, 0.0])
        v2 = _normalize([1.0, 0.1, 0.0])
        r_has_value = _make_record(vector=v1, properties={"name": "has_name"})
        r_missing = _make_record(vector=v2, properties={"age": 25})
        await collection.upsert(records=[r_has_value, r_missing])

        null_results = await collection.query(
            query_vectors=[v1],
            limit=10,
            property_filter=IsNull(field="name"),
        )
        assert {m.record_uuid for m in null_results[0].matches} == {r_missing.uuid}

        not_null_results = await collection.query(
            query_vectors=[v1],
            limit=10,
            property_filter=Not(IsNull(field="name")),
        )
        assert {m.record_uuid for m in not_null_results[0].matches} == {
            r_has_value.uuid
        }

    @pytest.mark.asyncio
    async def test_in_and_or_not(self, collection):
        r1, r2, r3, v1 = await self._setup(collection)

        in_results = await collection.query(
            query_vectors=[v1],
            limit=10,
            property_filter=In(field="name", values=("alice", "carol")),
        )
        assert {m.record_uuid for m in in_results[0].matches} == {r1.uuid, r3.uuid}

        and_results = await collection.query(
            query_vectors=[v1],
            limit=10,
            property_filter=And(
                (
                    Equals(field="active", value=True),
                    Ordering(field="age", op=">", value=30),
                )
            ),
        )
        assert {m.record_uuid for m in and_results[0].matches} == {r3.uuid}

        or_results = await collection.query(
            query_vectors=[v1],
            limit=10,
            property_filter=Or(
                (Equals(field="name", value="alice"), Equals(field="name", value="bob"))
            ),
        )
        assert {m.record_uuid for m in or_results[0].matches} == {r1.uuid, r2.uuid}

    @pytest.mark.asyncio
    async def test_negation_is_the_complement_missing_values_included(self, collection):
        """A condition on a property with no value is false, so its negation,
        `!=` included, holds there, as on Qdrant."""
        r1, r2, r3, v1 = await self._setup(collection)
        bare = _make_record(vector=_normalize([1.0, 0.3, 0.0]), properties={})
        await collection.upsert(records=[bare])

        async def uuids(expr):
            [result] = await collection.query(
                query_vectors=[v1], limit=10, property_filter=expr
            )
            return {match.record_uuid for match in result.matches}

        assert await uuids(comparison("name", "!=", "alice")) == {
            r2.uuid,
            r3.uuid,
            bare.uuid,
        }
        assert await uuids(Not(Equals(field="name", value="alice"))) == {
            r2.uuid,
            r3.uuid,
            bare.uuid,
        }
        assert await uuids(Not(Ordering(field="age", op=">", value=30))) == {
            r1.uuid,
            r2.uuid,
            bare.uuid,
        }
        assert await uuids(Not(In(field="name", values=("alice", "bob")))) == {
            r3.uuid,
            bare.uuid,
        }
        assert await uuids(
            Not(
                And(
                    (
                        Equals(field="active", value=True),
                        Ordering(field="age", op=">", value=30),
                    )
                )
            )
        ) == {r1.uuid, r2.uuid, bare.uuid}
        assert await uuids(Not(Not(IsNull(field="name")))) == {bare.uuid}

    @pytest.mark.asyncio
    async def test_datetime_filters_compare_instants_across_offsets(self, collection):
        base = datetime(2024, 6, 15, 12, 0, 0, tzinfo=UTC)
        plus5 = timezone(timedelta(hours=5))
        records = [
            _make_record(
                vector=_normalize([1.0, 0.1 * index, 0.0]),
                properties={"created_at": instant},
            )
            for index, instant in enumerate(
                [
                    base,
                    (base + timedelta(microseconds=1)).astimezone(plus5),
                    base - timedelta(days=1),
                ]
            )
        ]
        await collection.upsert(records=records)
        v1 = _normalize([1.0, 0.0, 0.0])

        async def uuids(expr):
            [result] = await collection.query(
                query_vectors=[v1], limit=10, property_filter=expr
            )
            return {match.record_uuid for match in result.matches}

        same_instant = base.astimezone(plus5)
        assert await uuids(Equals(field="created_at", value=same_instant)) == {
            records[0].uuid
        }
        assert await uuids(Ordering(field="created_at", op=">", value=base)) == {
            records[1].uuid
        }
        assert await uuids(
            Ordering(field="created_at", op="<", value=same_instant)
        ) == {records[2].uuid}

    @pytest.mark.asyncio
    async def test_a_value_of_another_type_matches_nothing(self, collection):
        r1, r2, r3, v1 = await self._setup(collection)

        [matched] = await collection.query(
            query_vectors=[v1],
            limit=10,
            property_filter=Equals(field="age", value="thirty"),
        )
        assert matched.matches == []
        [complement] = await collection.query(
            query_vectors=[v1],
            limit=10,
            property_filter=Not(Equals(field="age", value="thirty")),
        )
        assert {m.record_uuid for m in complement.matches} == {
            r1.uuid,
            r2.uuid,
            r3.uuid,
        }


class TestScores:
    @pytest.mark.asyncio
    async def test_scores_are_cosine_similarities(self, collection):
        """Scores come from the server, whatever the vectors' norms."""
        await collection.upsert(records=[_make_record(vector=[1.2, 1.6, 0.0])])

        [result] = await collection.query(query_vectors=[[1.0, 0.0, 0.0]], limit=1)
        assert result.matches[0].cosine_similarity == pytest.approx(0.6, abs=1e-3)


class TestDelete:
    @pytest.mark.asyncio
    async def test_delete_records(self, collection):
        v1 = _normalize([1.0, 0.0, 0.0])
        v2 = _normalize([0.0, 1.0, 0.0])
        r1 = _make_record(vector=v1)
        r2 = _make_record(vector=v2)
        await collection.upsert(records=[r1, r2])

        await collection.delete(record_uuids=[r1.uuid])

        assert await _present_uuids(collection, [r1.uuid, r2.uuid]) == [r2.uuid]


class TestPartitionIsolation:
    @pytest.mark.asyncio
    async def test_same_uuid_can_exist_in_different_partitions(self, store):
        await store.create_partition("tenant_a")
        await store.create_partition("tenant_b")
        coll_a = await store.get_partition("tenant_a")
        coll_b = await store.get_partition("tenant_b")
        assert coll_a is not None
        assert coll_b is not None

        record_uuid = uuid4()
        v1 = _normalize([1.0, 0.0, 0.0])
        await coll_a.upsert(
            records=[Record(uuid=record_uuid, vector=v1, properties={"name": "a"})]
        )
        await coll_b.upsert(
            records=[Record(uuid=record_uuid, vector=v1, properties={"name": "b"})]
        )

        # Each collection holds its own record under the shared uuid, and
        # each one's properties are only reachable through its own filter.
        [only_a] = await coll_a.query(
            query_vectors=[v1],
            limit=10,
            property_filter=Equals(field="name", value="a"),
        )
        [only_b] = await coll_b.query(
            query_vectors=[v1],
            limit=10,
            property_filter=Equals(field="name", value="b"),
        )
        assert [m.record_uuid for m in only_a.matches] == [record_uuid]
        assert [m.record_uuid for m in only_b.matches] == [record_uuid]

        await store.delete_partition("tenant_a")
        await store.delete_partition("tenant_b")


class TestPurgeBatches:
    @pytest.mark.asyncio
    async def test_a_purge_round_reclaims_at_most_one_batch(self, store):
        store = MilvusVectorStore(
            await _params(
                store._client, store._partition_registry._engine, purge_batch_size=2
            )
        )
        await store.create_partition("batched")
        partition = await store.get_partition("batched")
        assert partition is not None
        await partition.upsert(
            records=[
                _make_record(vector=_normalize([1.0, float(i), 0.0])) for i in range(5)
            ]
        )
        incarnation = partition._incarnation
        await store.delete_partition("batched")

        def left_of_the_incarnation() -> int:
            return len(
                store._client.query(
                    collection_name=store._collection_name,
                    filter=f'partition_key == "{incarnation.hex}"',
                    output_fields=["id"],
                    limit=16384,
                )
            )

        left_after_each_round = []
        while await store.purge_deleted_partitions():
            left_after_each_round.append(left_of_the_incarnation())
        assert left_after_each_round == [3, 1, 0]


class TestLifecycleContract(PartitionLifecycleContract):
    """The partition lifecycle contract, against this store."""

    @staticmethod
    async def count_stored(store) -> int:
        rows = store._client.query(
            collection_name=store._collection_name,
            filter='id != ""',
            output_fields=["id"],
            limit=16384,
        )
        return len(list(rows))
