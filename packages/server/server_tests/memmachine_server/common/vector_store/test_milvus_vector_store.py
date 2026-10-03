"""Tests for MilvusVectorStore."""

# ruff: noqa: E402

import math
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta, timezone
from typing import Any, override
from unittest.mock import AsyncMock, MagicMock
from uuid import UUID, uuid4

import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import create_async_engine

from server_tests.memmachine_server.common.vector_store.partition_lifecycle_contract import (
    PartitionLifecycleContract,
)

pymilvus = pytest.importorskip("pymilvus")
DataType = pymilvus.DataType
AsyncMilvusClient = pymilvus.AsyncMilvusClient

from memmachine_server.common.data_types import PropertyType
from memmachine_server.common.filter.filter_parser import (
    And,
    Comparison,
    In,
    IsNull,
    Not,
    Or,
)
from memmachine_server.common.metrics_factory import OperationTracker
from memmachine_server.common.properties_json import decode_properties
from memmachine_server.common.vector_store.data_types import (
    PartitionSchema,
    Record,
    VectorStorePartitionAlreadyExistsError,
    VectorStorePartitionSchemaMismatchError,
)
from memmachine_server.common.vector_store.milvus_vector_store import (
    MilvusVectorStore,
    MilvusVectorStoreParams,
    MilvusVectorStorePartition,
)
from memmachine_server.common.vector_store.partition_registry import (
    Registration,
)
from memmachine_server.common.vector_store.partition_registry.sqlalchemy_partition_registry import (
    SQLAlchemyVectorStorePartitionRegistry,
    SQLAlchemyVectorStorePartitionRegistryParams,
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
    """Parameters for one store: its own started registry over the shared registry database."""
    params: dict[str, Any] = {
        "client": client,
        "vector_store_name": VECTOR_STORE_NAME,
        "vector_dimensions": VECTOR_DIM,
        "indexed_properties": INDEXED_PROPERTIES,
        "request_timeout_seconds": REQUEST_TIMEOUT_SECONDS,
        "max_varchar_length": MAX_VARCHAR_LENGTH,
        "purge_batch_size": PURGE_BATCH_SIZE,
    }
    params.update(overrides)
    params["partition_registry"] = SQLAlchemyVectorStorePartitionRegistry(
        SQLAlchemyVectorStorePartitionRegistryParams(
            engine=registry_engine,
            vector_store_name=params["vector_store_name"],
            # Tombstones come due at once, so a test can purge right after
            # deleting.
            tombstone_retention_seconds=0,
        )
    )
    await params["partition_registry"].startup()
    return MilvusVectorStoreParams(**params)


async def _settle(partition: MilvusVectorStorePartition) -> None:
    """Return once the store's reads reflect every write made so far.

    The store reads at Bounded, which may lag its writes. A Strong read
    returns only once the server has applied every earlier write, and with
    one replica the store's later reads start from that point.
    """
    await partition._client.query(
        collection_name=partition._collection_name,
        filter=f'partition_key == "{partition._incarnation}"',
        output_fields=["id"],
        limit=1,
        consistency_level="Strong",
        timeout=REQUEST_TIMEOUT_SECONDS,
    )


async def _stored(
    partition: MilvusVectorStorePartition, record_uuids: list[UUID]
) -> dict[UUID, dict]:
    """The entities Milvus holds under these UUIDs, read past the store at Strong.

    The store's reads may lag its writes by its consistency level; a Strong
    read reflects every write that returned before it.
    """
    rows = await partition._client.get(
        collection_name=partition._collection_name,
        ids=[partition._primary_id(record_uuid) for record_uuid in record_uuids],
        output_fields=["*"],
        consistency_level="Strong",
    )
    return {UUID(row["record_uuid"]): row for row in rows}


@pytest_asyncio.fixture
async def server_milvus_client(milvus_container):
    client = AsyncMilvusClient(uri=milvus_container.get_connection_url())
    yield client
    await client.close()


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
    await vector_store.startup()
    yield vector_store
    await vector_store.shutdown()
    await registry_engine.dispose()


# The client requests the store makes; each must carry the store's timeout.
_CLIENT_REQUESTS = (
    "has_collection",
    "create_collection",
    "list_indexes",
    "create_index",
    "load_collection",
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

    # A collection of its own, so startup creates it through the spies.
    timed = MilvusVectorStore(
        await _params(
            store._client, store._partition_registry._engine, vector_store_name="timed"
        )
    )
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
    await _settle(coll)
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
    @pytest.mark.parametrize("failing_step", ["create_index", "load_collection"])
    async def test_a_startup_that_failed_part_way_is_as_if_never_attempted(
        self, store, monkeypatch, failing_step
    ):
        """The next startup completes the native collection a failed one left behind."""
        partial = MilvusVectorStore(
            await _params(
                store._client,
                store._partition_registry._engine,
                vector_store_name=f"partial_{failing_step}",
                indexed_properties={"name": str},
            )
        )

        async def refuse(*args, **kwargs):
            raise pymilvus.MilvusException(message=f"{failing_step} refused")

        with monkeypatch.context() as patch:
            patch.setattr(store._client, failing_step, refuse)
            with pytest.raises(pymilvus.MilvusException, match="refused"):
                await partial.startup()

        await partial.startup()
        await partial.create_partition("partial")
        coll = await partial.get_partition("partial")
        assert coll is not None
        vector = _normalize([1.0, 0.0, 0.0])
        record = _make_record(vector=vector, properties={"name": "alice"})
        await coll.upsert(records=[record])
        await _settle(coll)
        [result] = await coll.query(
            query_vectors=[vector],
            limit=1,
            property_filter=Comparison(field="name", op="=", value="alice"),
        )
        assert [match.record_uuid for match in result.matches] == [record.uuid]
        assert set(await store._client.list_indexes(partial._collection_name)) == {
            "vector",
            "_p_name",
        }
        await partial.delete_partition("partial")

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
        await digit_first.startup()
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
        """Each declared property is a typed, nullable, indexed field, a
        datetime with a field for its offset; the collection isolates tenants."""
        await store.create_partition("schema")
        coll = await store.get_partition("schema")
        assert coll is not None
        native = coll._collection_name

        schema = await store._client.describe_collection(coll._collection_name)
        fields = {field["name"]: field for field in schema["fields"]}
        assert schema["auto_id"] is False
        assert schema["enable_dynamic_field"] is False
        assert schema["properties"]["partitionkey.isolation"] == "True"
        assert fields["id"]["is_primary"] is True
        assert fields["partition_key"]["is_partition_key"] is True
        assert fields["vector"]["type"] == DataType.FLOAT_VECTOR
        assert fields["vector"]["params"]["dim"] == VECTOR_DIM
        assert fields["properties"]["type"] == DataType.JSON
        expected = {
            "_p_name": DataType.VARCHAR,
            "_p_age": DataType.INT64,
            "_p_score": DataType.DOUBLE,
            "_p_active": DataType.BOOL,
            "_p_created_at": DataType.TIMESTAMPTZ,
            "_tz_created_at": DataType.INT32,
        }
        for field_name, data_type in expected.items():
            assert fields[field_name]["type"] == data_type
            assert fields[field_name]["nullable"] is True
        assert fields["_p_name"]["params"]["max_length"] == MAX_VARCHAR_LENGTH
        indexed = {
            (await store._client.describe_index(native, index_name))["field_name"]
            for index_name in await store._client.list_indexes(native)
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

        async def tracked_upsert(**kwargs):
            nonlocal captured_kwargs
            captured_kwargs = kwargs

        async def fail_insert(*args, **kwargs):
            pytest.fail("collection upsert must not call AsyncMilvusClient.insert")

        monkeypatch.setattr(collection._client, "upsert", tracked_upsert)
        monkeypatch.setattr(collection._client, "insert", fail_insert)

        record = _make_record(
            vector=_normalize([1.0, 0.0, 0.0]),
            properties={"name": "test"},
        )
        await collection.upsert(records=[record])
        await _settle(collection)

        assert captured_kwargs is not None
        assert captured_kwargs["collection_name"] == collection._collection_name
        assert captured_kwargs["data"] == [collection._build_entity(record)]

    @pytest.mark.asyncio
    async def test_a_query_may_ask_for_hundreds_of_results(self, collection):
        records = [
            _make_record(vector=_normalize([1.0, float(i), 0.0])) for i in range(3)
        ]
        await collection.upsert(records=records)
        await _settle(collection)

        [result] = await collection.query(
            query_vectors=[_normalize([1.0, 0.0, 0.0])], limit=200
        )

        assert {match.record_uuid for match in result.matches} == {
            record.uuid for record in records
        }

    @pytest.mark.asyncio
    async def test_upsert_and_query_basic(self, collection):
        v1 = _normalize([1.0, 0.0, 0.0])
        v2 = _normalize([0.0, 1.0, 0.0])
        v3 = _normalize([1.0, 0.1, 0.0])

        r1 = _make_record(vector=v1, properties={"name": "a"})
        r2 = _make_record(vector=v2, properties={"name": "b"})
        r3 = _make_record(vector=v3, properties={"name": "c"})

        await collection.upsert(records=[r1, r2, r3])
        await _settle(collection)

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
        await _settle(collection)

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
        await _settle(collection)

        all_results = await collection.query(query_vectors=[v1, v2], limit=1)

        assert len(all_results) == 2
        assert all_results[0].matches[0].record_uuid == r1.uuid
        assert all_results[1].matches[0].record_uuid == r2.uuid

    @pytest.mark.asyncio
    async def test_upsert_removes_stale_filter_fields(self, collection):
        v1 = _normalize([1.0, 0.0, 0.0])
        record = _make_record(vector=v1, properties={"name": "old"})
        await collection.upsert(records=[record])
        await _settle(collection)

        await collection.upsert(
            records=[Record(uuid=record.uuid, vector=v1, properties={})]
        )
        await _settle(collection)

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
        await _settle(collection)

        async def fail_upsert(*args, **kwargs):
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

        stored = (await _stored(collection, [record.uuid]))[record.uuid]
        assert list(stored["vector"]) == old_vector
        assert stored["_p_name"] == "old"


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
        await _settle(collection)
        return r1, r2, r3, v1

    async def _query(self, collection, query_vec, field, op, value):
        all_results = await collection.query(
            query_vectors=[query_vec],
            limit=10,
            property_filter=Comparison(field=field, op=op, value=value),
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
    async def test_datetime_filters(self, collection):
        v1 = _normalize([1.0, 0.0, 0.0])
        v2 = _normalize([1.0, 0.1, 0.0])
        dt_utc = datetime(2024, 6, 15, 12, 0, 0, tzinfo=UTC)
        dt_other = datetime(2024, 6, 15, 18, 0, 0, tzinfo=UTC)
        r1 = _make_record(vector=v1, properties={"created_at": dt_utc})
        r2 = _make_record(vector=v2, properties={"created_at": dt_other})
        await collection.upsert(records=[r1, r2])
        await _settle(collection)

        plus5 = timezone(timedelta(hours=5))
        dt_filter = datetime(2024, 6, 15, 17, 0, 0, tzinfo=plus5)
        uuids = await self._query(collection, v1, "created_at", "=", dt_filter)
        assert uuids == {r1.uuid}

    @pytest.mark.asyncio
    async def test_is_null_and_not_null(self, collection):
        v1 = _normalize([1.0, 0.0, 0.0])
        v2 = _normalize([1.0, 0.1, 0.0])
        r_has_value = _make_record(vector=v1, properties={"name": "has_name"})
        r_missing = _make_record(vector=v2, properties={"age": 25})
        await collection.upsert(records=[r_has_value, r_missing])
        await _settle(collection)

        null_results = await collection.query(
            query_vectors=[v1],
            limit=10,
            property_filter=IsNull(field="name"),
        )
        assert {m.record_uuid for m in null_results[0].matches} == {r_missing.uuid}

        not_null_results = await collection.query(
            query_vectors=[v1],
            limit=10,
            property_filter=Not(expr=IsNull(field="name")),
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
            property_filter=In(field="name", values=["alice", "carol"]),
        )
        assert {m.record_uuid for m in in_results[0].matches} == {r1.uuid, r3.uuid}

        and_results = await collection.query(
            query_vectors=[v1],
            limit=10,
            property_filter=And(
                left=Comparison(field="active", op="=", value=True),
                right=Comparison(field="age", op=">", value=30),
            ),
        )
        assert {m.record_uuid for m in and_results[0].matches} == {r3.uuid}

        or_results = await collection.query(
            query_vectors=[v1],
            limit=10,
            property_filter=Or(
                left=Comparison(field="name", op="=", value="alice"),
                right=Comparison(field="name", op="=", value="bob"),
            ),
        )
        assert {m.record_uuid for m in or_results[0].matches} == {r1.uuid, r2.uuid}

    @pytest.mark.asyncio
    async def test_negation_is_the_complement_missing_values_included(self, collection):
        """A condition on a property with no value is false, so its negation,
        `!=` included, holds there."""
        r1, r2, r3, v1 = await self._setup(collection)
        bare = _make_record(vector=_normalize([1.0, 0.3, 0.0]), properties={})
        await collection.upsert(records=[bare])
        await _settle(collection)

        async def uuids(expr):
            [result] = await collection.query(
                query_vectors=[v1], limit=10, property_filter=expr
            )
            return {match.record_uuid for match in result.matches}

        assert await uuids(Comparison(field="name", op="!=", value="alice")) == {
            r2.uuid,
            r3.uuid,
            bare.uuid,
        }
        assert await uuids(
            Not(expr=Comparison(field="name", op="=", value="alice"))
        ) == {r2.uuid, r3.uuid, bare.uuid}
        assert await uuids(Not(expr=Comparison(field="age", op=">", value=30))) == {
            r1.uuid,
            r2.uuid,
            bare.uuid,
        }
        assert await uuids(Not(expr=In(field="name", values=["alice", "bob"]))) == {
            r3.uuid,
            bare.uuid,
        }
        assert await uuids(
            Not(
                expr=And(
                    left=Comparison(field="active", op="=", value=True),
                    right=Comparison(field="age", op=">", value=30),
                )
            )
        ) == {r1.uuid, r2.uuid, bare.uuid}
        assert await uuids(Not(expr=Not(expr=IsNull(field="name")))) == {bare.uuid}

    @pytest.mark.asyncio
    async def test_filters_on_undeclared_properties(self, collection):
        """A property the schema does not declare is stored and filtered too."""
        v1 = _normalize([1.0, 0.0, 0.0])
        red = _make_record(vector=v1, properties={"color": "red", "size": 3})
        blue = _make_record(
            vector=_normalize([1.0, 0.1, 0.0]), properties={"color": "blue"}
        )
        await collection.upsert(records=[red, blue])
        await _settle(collection)

        async def uuids(expr):
            [result] = await collection.query(
                query_vectors=[v1], limit=10, property_filter=expr
            )
            return {match.record_uuid for match in result.matches}

        assert await uuids(Comparison(field="color", op="=", value="red")) == {red.uuid}
        assert await uuids(Comparison(field="size", op=">=", value=3)) == {red.uuid}
        assert await uuids(In(field="color", values=["blue", "green"])) == {blue.uuid}
        assert await uuids(IsNull(field="size")) == {blue.uuid}
        assert await uuids(Comparison(field="size", op="!=", value=3)) == {blue.uuid}
        stored = await _stored(collection, [red.uuid])
        assert decode_properties(stored[red.uuid]["properties"]) == {
            "color": "red",
            "size": 3,
        }

    @pytest.mark.asyncio
    async def test_a_declared_datetime_is_stored_with_its_offset(self, collection):
        written = datetime(
            2024, 6, 15, 17, 30, tzinfo=timezone(timedelta(hours=5, minutes=30))
        )
        record = _make_record(
            vector=_normalize([1.0, 0.0, 0.0]), properties={"created_at": written}
        )
        await collection.upsert(records=[record])

        stored = (await _stored(collection, [record.uuid]))[record.uuid]
        assert datetime.fromisoformat(stored["_p_created_at"]) == written
        assert stored["_tz_created_at"] == 5 * 3600 + 30 * 60

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
        await _settle(collection)
        v1 = _normalize([1.0, 0.0, 0.0])

        async def uuids(expr):
            [result] = await collection.query(
                query_vectors=[v1], limit=10, property_filter=expr
            )
            return {match.record_uuid for match in result.matches}

        same_instant = base.astimezone(plus5)
        assert await uuids(
            Comparison(field="created_at", op="=", value=same_instant)
        ) == {records[0].uuid}
        assert await uuids(Comparison(field="created_at", op=">", value=base)) == {
            records[1].uuid
        }
        assert await uuids(
            Comparison(field="created_at", op="<", value=same_instant)
        ) == {records[2].uuid}

    @pytest.mark.asyncio
    async def test_a_value_of_another_type_matches_nothing(self, collection):
        r1, r2, r3, v1 = await self._setup(collection)

        [matched] = await collection.query(
            query_vectors=[v1],
            limit=10,
            property_filter=Comparison(field="age", op="=", value="thirty"),
        )
        assert matched.matches == []
        [complement] = await collection.query(
            query_vectors=[v1],
            limit=10,
            property_filter=Comparison(field="age", op="!=", value="thirty"),
        )
        assert {m.record_uuid for m in complement.matches} == {
            r1.uuid,
            r2.uuid,
            r3.uuid,
        }

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("key", "value"),
        [("age", "old"), ("age", 30.5), ("age", True), ("score", 3), ("score", "high")],
    )
    async def test_a_declared_property_of_another_type_is_refused(
        self, collection, key, value
    ):
        with pytest.raises(ValueError, match=f"{key!r} is declared"):
            await collection.upsert(
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
    async def test_a_vector_of_another_width_is_refused(self, collection, dimensions):
        vector = [1.0] * dimensions
        with pytest.raises(ValueError, match="dimensions"):
            await collection.upsert(records=[_make_record(vector=vector)])
        with pytest.raises(ValueError, match="dimensions"):
            await collection.query(query_vectors=[vector], limit=1)

    @pytest.mark.asyncio
    @pytest.mark.parametrize("coordinate", [math.nan, math.inf], ids=["nan", "inf"])
    async def test_a_query_vector_with_a_coordinate_that_is_not_finite_is_refused(
        self, collection, coordinate
    ):
        with pytest.raises(ValueError, match="not finite"):
            await collection.query(
                query_vectors=[[coordinate] + [1.0] * (VECTOR_DIM - 1)], limit=1
            )

    @pytest.mark.asyncio
    @pytest.mark.parametrize("threshold", [math.nan, math.inf, -math.inf])
    async def test_a_min_cosine_similarity_that_is_not_finite_is_refused(
        self, collection, threshold
    ):
        with pytest.raises(ValueError, match="not finite"):
            await collection.query(
                query_vectors=[_normalize([1.0, 0.0, 0.0])],
                limit=1,
                min_cosine_similarity=threshold,
            )

    @pytest.mark.asyncio
    @pytest.mark.parametrize("limit", [0, -1])
    async def test_a_limit_that_is_not_positive_is_refused(self, collection, limit):
        with pytest.raises(ValueError, match="not positive"):
            await collection.query(
                query_vectors=[_normalize([1.0, 0.0, 0.0])], limit=limit
            )


class TestScores:
    @pytest.mark.asyncio
    async def test_scores_are_cosine_similarities(self, collection):
        """Scores come from the server, whatever the vectors' norms."""
        await collection.upsert(records=[_make_record(vector=[1.2, 1.6, 0.0])])
        await _settle(collection)

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
        await _settle(collection)

        await collection.delete(record_uuids=[r1.uuid])

        assert set(await _stored(collection, [r1.uuid, r2.uuid])) == {r2.uuid}

    @pytest.mark.asyncio
    async def test_deleting_records_it_does_not_hold_succeeds(self, collection):
        """Milvus accepts the delete of a primary key it does not hold."""
        record = _make_record(vector=_normalize([1.0, 0.0, 0.0]))
        await collection.upsert(records=[record])
        await _settle(collection)

        await collection.delete(record_uuids=[record.uuid, uuid4(), uuid4()])

        assert await _stored(collection, [record.uuid]) == {}


@dataclass(frozen=True)
class _CurrentRegistration(Registration):
    """A live registration whose partition is never deleted."""

    @override
    async def require_current(self) -> None:
        return None


@pytest.mark.asyncio
async def test_a_delete_milvus_does_not_accept_in_full_raises():
    """A delete Milvus accepts for fewer primary keys than the store sent raises."""
    client = MagicMock(spec=AsyncMilvusClient)
    client.delete = AsyncMock(return_value={"delete_count": 0})
    partition = MilvusVectorStorePartition(
        client=client,
        collection_name=f"sys_{VECTOR_STORE_NAME}",
        vector_store_name=VECTOR_STORE_NAME,
        registration=_CurrentRegistration(
            partition_key=NAME,
            schema=PartitionSchema(
                vector_dimensions=VECTOR_DIM,
                indexed_properties={},
            ),
            incarnation=uuid4(),
        ),
        vector_dimensions=VECTOR_DIM,
        indexed_properties={},
        tracker=OperationTracker(None, prefix="test"),
        request_timeout_seconds=REQUEST_TIMEOUT_SECONDS,
    )

    with pytest.raises(pymilvus.MilvusException, match="accepted the delete of 0 of 2"):
        await partition.delete(record_uuids=[uuid4(), uuid4()])


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
        await _settle(coll_a)
        await coll_b.upsert(
            records=[Record(uuid=record_uuid, vector=v1, properties={"name": "b"})]
        )
        await _settle(coll_b)

        stored_a = await _stored(coll_a, [record_uuid])
        stored_b = await _stored(coll_b, [record_uuid])

        # `name` is declared, so each value is in its typed field.
        assert stored_a[record_uuid]["_p_name"] == "a"
        assert stored_b[record_uuid]["_p_name"] == "b"

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
        await _settle(partition)
        incarnation = partition._incarnation
        await store.delete_partition("batched")

        async def left_of_the_incarnation() -> int:
            return len(
                await store._client.query(
                    collection_name=store._collection_name,
                    filter=f'partition_key == "{incarnation}"',
                    output_fields=["id"],
                    limit=16384,
                    consistency_level="Strong",
                )
            )

        left_after_each_round = []
        while await store.purge_deleted_partitions():
            await _settle(partition)
            left_after_each_round.append(await left_of_the_incarnation())
        # Batches of 2, then a round that finds nothing and removes the tombstone.
        assert left_after_each_round == [3, 1, 0, 0]


class TestLifecycleContract(PartitionLifecycleContract):
    """The partition lifecycle contract, against this store."""

    @staticmethod
    async def count_stored(store) -> int:
        rows = await store._client.query(
            collection_name=store._collection_name,
            filter='id != ""',
            output_fields=["id"],
            limit=16384,
            consistency_level="Strong",
        )
        return len(list(rows))

    settle = staticmethod(_settle)

    @staticmethod
    async def stored_uuids(partition) -> set[UUID]:
        rows = await partition._client.query(
            collection_name=partition._collection_name,
            filter=f'partition_key == "{partition._incarnation}"',
            output_fields=["record_uuid"],
            limit=16384,
            consistency_level="Strong",
        )
        return {UUID(row["record_uuid"]) for row in rows}
