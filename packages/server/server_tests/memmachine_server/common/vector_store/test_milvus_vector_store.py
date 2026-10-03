"""Tests for MilvusVectorStore."""

# ruff: noqa: E402

import math
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta, timezone
from typing import override
from unittest.mock import AsyncMock, MagicMock
from uuid import UUID, uuid4

import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import create_async_engine

from server_tests.memmachine_server.common.vector_store.collection_lifecycle_contract import (
    CollectionLifecycleContract,
)

pymilvus = pytest.importorskip("pymilvus")
DataType = pymilvus.DataType
AsyncMilvusClient = pymilvus.AsyncMilvusClient

from memmachine_server.common.data_types import PropertyValue, SimilarityMetric
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
from memmachine_server.common.vector_store.collection_registry import (
    Registration,
)
from memmachine_server.common.vector_store.collection_registry.sqlalchemy_collection_registry import (
    SQLAlchemyVectorStoreCollectionRegistry,
    SQLAlchemyVectorStoreCollectionRegistryParams,
)
from memmachine_server.common.vector_store.data_types import (
    Record,
    VectorStoreCollectionAlreadyExistsError,
    VectorStoreCollectionConfig,
    VectorStoreCollectionConfigMismatchError,
)
from memmachine_server.common.vector_store.milvus_vector_store import (
    MilvusVectorStore,
    MilvusVectorStoreCollection,
    MilvusVectorStoreParams,
)

NAMESPACE = "test_namespace"
NAME = "test_name"
VECTOR_DIM = 3
VECTOR_STORE_NAME = "milvus_test"
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


async def _settle(collection: MilvusVectorStoreCollection) -> None:
    """Return once the store's reads reflect every write made so far.

    The store reads at Bounded, which may lag its writes. A Strong read
    returns only once the server has applied every earlier write, and with
    one replica the store's later reads start from that point.
    """
    await collection._client.query(
        collection_name=collection._native_collection_name,
        filter=f'partition_key == "{collection._incarnation}"',
        output_fields=["id"],
        limit=1,
        consistency_level="Strong",
        timeout=REQUEST_TIMEOUT_SECONDS,
    )


async def _stored(
    collection: MilvusVectorStoreCollection, record_uuids: list[UUID]
) -> dict[UUID, dict]:
    """The entities Milvus holds under these UUIDs, read past the store at Strong.

    The store's reads may lag its writes by its consistency level; a Strong
    read reflects every write that returned before it.
    """
    rows = await collection._client.get(
        collection_name=collection._native_collection_name,
        ids=[collection._primary_id(record_uuid) for record_uuid in record_uuids],
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
    collection_registry = SQLAlchemyVectorStoreCollectionRegistry(
        SQLAlchemyVectorStoreCollectionRegistryParams(
            engine=registry_engine,
            vector_store_name=VECTOR_STORE_NAME,
            # Tombstones come due at once, so a test can purge right after
            # deleting.
            tombstone_retention_seconds=0,
        )
    )
    await collection_registry.startup()
    vector_store = MilvusVectorStore(
        MilvusVectorStoreParams(
            client=milvus_client,
            collection_registry=collection_registry,
            request_timeout_seconds=REQUEST_TIMEOUT_SECONDS,
            max_varchar_length=MAX_VARCHAR_LENGTH,
            purge_batch_size=PURGE_BATCH_SIZE,
        )
    )
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
    # Its own namespace: a native collection an earlier test left behind
    # would let create_collection skip its Milvus request.
    namespace = "timed_namespace"
    spies = {}
    for name in _CLIENT_REQUESTS:
        spies[name] = MagicMock(wraps=getattr(store._client, name))
        monkeypatch.setattr(store._client, name, spies[name])

    await store.create_collection(
        namespace=namespace,
        name="timed",
        config=VectorStoreCollectionConfig(vector_dimensions=VECTOR_DIM),
    )
    coll = await store.open_collection(namespace=namespace, name="timed")
    assert coll is not None
    record, kept = (
        _make_record(vector=_normalize([1.0, 0.0, 0.0])),
        _make_record(vector=_normalize([0.0, 1.0, 0.0])),
    )
    await coll.upsert(records=[record, kept])
    await _settle(coll)
    await coll.query(query_vectors=[record.vector], limit=1)
    await coll.delete(record_uuids=[record.uuid])
    await store.delete_collection(namespace=namespace, name="timed")
    # The purge finds the record the deletion left and reclaims it.
    while await store.purge_deleted_collections():
        pass

    assert {name for name, spy in spies.items() if spy.call_count} >= set(
        _CLIENT_REQUESTS
    )
    for name, spy in spies.items():
        for call in spy.call_args_list:
            assert call.kwargs.get("timeout") == REQUEST_TIMEOUT_SECONDS, (name, call)


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


class TestCollectionLifecycle:
    @pytest.mark.asyncio
    @pytest.mark.parametrize("failing_step", ["create_index", "load_collection"])
    async def test_a_creation_that_failed_part_way_is_as_if_never_attempted(
        self, store, monkeypatch, failing_step
    ):
        """The next creation completes the native collection a failed one left behind."""
        namespace = f"partial_{failing_step}"
        config = VectorStoreCollectionConfig(
            vector_dimensions=VECTOR_DIM, indexed_properties_schema={"name": str}
        )

        async def refuse(*args, **kwargs):
            raise pymilvus.MilvusException(message=f"{failing_step} refused")

        with monkeypatch.context() as patch:
            patch.setattr(store._client, failing_step, refuse)
            with pytest.raises(pymilvus.MilvusException, match="refused"):
                await store.create_collection(
                    namespace=namespace, name="partial", config=config
                )
        assert await store.open_collection(namespace=namespace, name="partial") is None

        await store.create_collection(
            namespace=namespace, name="partial", config=config
        )
        coll = await store.open_collection(namespace=namespace, name="partial")
        assert coll is not None
        record = _make_record(
            vector=_normalize([1.0, 0.0, 0.0]), properties={"name": "alice"}
        )
        await coll.upsert(records=[record])
        await _settle(coll)
        [result] = await coll.query(
            query_vectors=[record.vector],
            limit=1,
            property_filter=Comparison(field="name", op="=", value="alice"),
        )
        assert [match.record_uuid for match in result.matches] == [record.uuid]
        native = coll._native_collection_name
        assert set(await store._client.list_indexes(native)) == {"vector", "_p_name"}
        await store.delete_collection(namespace=namespace, name="partial")

    @pytest.mark.asyncio
    async def test_create_open_delete(self, store):
        await store.create_collection(
            namespace=NAMESPACE,
            name="lifecycle",
            config=VectorStoreCollectionConfig(vector_dimensions=VECTOR_DIM),
        )
        coll = await store.open_collection(namespace=NAMESPACE, name="lifecycle")
        assert isinstance(coll, MilvusVectorStoreCollection)
        await store.delete_collection(namespace=NAMESPACE, name="lifecycle")

    @pytest.mark.asyncio
    async def test_duplicate_name_raises(self, store, collection):
        with pytest.raises(VectorStoreCollectionAlreadyExistsError):
            await store.create_collection(
                namespace=NAMESPACE,
                name=NAME,
                config=collection.config,
            )

    @pytest.mark.asyncio
    async def test_delete_nonexistent_is_idempotent(self, store):
        await store.delete_collection(namespace=NAMESPACE, name="nonexistent")

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
    async def test_same_config_shares_native_collection(self, store):
        schema: dict[str, type[PropertyValue]] = {"name": str}
        config = VectorStoreCollectionConfig(
            vector_dimensions=VECTOR_DIM,
            similarity_metric=SimilarityMetric.COSINE,
            indexed_properties_schema=schema,
        )
        await store.create_collection(namespace=NAMESPACE, name="coll_a", config=config)
        await store.create_collection(namespace=NAMESPACE, name="coll_b", config=config)

        coll_a = await store.open_collection(namespace=NAMESPACE, name="coll_a")
        coll_b = await store.open_collection(namespace=NAMESPACE, name="coll_b")
        assert coll_a is not None
        assert coll_b is not None
        assert coll_a._native_collection_name == coll_b._native_collection_name

        await store.delete_collection(namespace=NAMESPACE, name="coll_a")
        await store.delete_collection(namespace=NAMESPACE, name="coll_b")

    @pytest.mark.asyncio
    async def test_native_collection_schema(self, store):
        """Each declared property is a typed, nullable, indexed field, a
        datetime with a field for its offset; the collection isolates tenants."""
        await store.create_collection(
            namespace=NAMESPACE,
            name="schema",
            config=VectorStoreCollectionConfig(
                vector_dimensions=VECTOR_DIM,
                indexed_properties_schema={
                    "name": str,
                    "age": int,
                    "score": float,
                    "active": bool,
                    "created_at": datetime,
                },
            ),
        )
        coll = await store.open_collection(namespace=NAMESPACE, name="schema")
        assert coll is not None
        native = coll._native_collection_name

        schema = await store._client.describe_collection(native)
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

        await store.delete_collection(namespace=NAMESPACE, name="schema")

    @pytest.mark.asyncio
    async def test_unsupported_metric_raises(self, store):
        with pytest.raises(ValueError, match="Milvus only supports"):
            await store.create_collection(
                namespace=NAMESPACE,
                name="bad_metric",
                config=VectorStoreCollectionConfig(
                    vector_dimensions=VECTOR_DIM,
                    similarity_metric=SimilarityMetric.MANHATTAN,
                ),
            )


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
        assert captured_kwargs["collection_name"] == collection._native_collection_name
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
        assert matches[0].score >= matches[1].score >= matches[2].score
        assert matches[0].score == pytest.approx(1.0)
        assert matches[2].score == pytest.approx(0.0)

    @pytest.mark.asyncio
    async def test_query_with_similarity_threshold(self, collection):
        v1 = _normalize([1.0, 0.0, 0.0])
        v2 = _normalize([0.0, 1.0, 0.0])

        r1 = _make_record(vector=v1)
        r2 = _make_record(vector=v2)
        await collection.upsert(records=[r1, r2])
        await _settle(collection)

        query_results = await collection.query(
            query_vectors=[v1], limit=10, score_threshold=0.9
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
    async def test_a_score_threshold_that_is_not_finite_is_refused(
        self, collection, threshold
    ):
        with pytest.raises(ValueError, match="not finite"):
            await collection.query(
                query_vectors=[_normalize([1.0, 0.0, 0.0])],
                limit=1,
                score_threshold=threshold,
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
    @pytest.mark.parametrize(
        ("metric", "expected"),
        [
            (SimilarityMetric.COSINE, 0.6),
            (SimilarityMetric.DOT, 1.2),
            (SimilarityMetric.EUCLIDEAN, math.sqrt(2.0 * 2.0 + 1.0 - 2.0 * 1.2)),
        ],
    )
    async def test_scores_are_the_metric_values(self, store, metric, expected):
        """Scores come from the server: cosine similarity, inner product, and
        Euclidean distance (Milvus returns it squared)."""
        name = f"scores_{metric.value}"
        await store.create_collection(
            namespace=NAMESPACE,
            name=name,
            config=VectorStoreCollectionConfig(
                vector_dimensions=VECTOR_DIM, similarity_metric=metric
            ),
        )
        collection = await store.open_collection(namespace=NAMESPACE, name=name)
        assert collection is not None
        record = _make_record(vector=[1.2, 1.6, 0.0])
        await collection.upsert(records=[record])
        await _settle(collection)

        [result] = await collection.query(query_vectors=[[1.0, 0.0, 0.0]], limit=1)
        assert result.matches[0].score == pytest.approx(expected, abs=1e-3)
        await store.delete_collection(namespace=NAMESPACE, name=name)


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
    """A registration whose collection is never deleted."""

    @override
    async def require_current(self) -> None:
        return None


@pytest.mark.asyncio
async def test_a_delete_milvus_does_not_accept_in_full_raises():
    """A delete Milvus accepts for fewer primary keys than the store sent raises."""
    client = MagicMock(spec=AsyncMilvusClient)
    client.delete = AsyncMock(return_value={"delete_count": 0})
    collection = MilvusVectorStoreCollection(
        client=client,
        native_collection_name="native",
        registration=_CurrentRegistration(
            namespace=NAMESPACE,
            name=NAME,
            config=VectorStoreCollectionConfig(vector_dimensions=VECTOR_DIM),
            incarnation=uuid4(),
        ),
        tracker=OperationTracker(None, prefix="test"),
        request_timeout_seconds=REQUEST_TIMEOUT_SECONDS,
    )

    with pytest.raises(pymilvus.MilvusException, match="accepted the delete of 0 of 2"):
        await collection.delete(record_uuids=[uuid4(), uuid4()])


class TestPartitionIsolation:
    @pytest.mark.asyncio
    async def test_same_uuid_can_exist_in_different_logical_collections(self, store):
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

        assert decode_properties(stored_a[record_uuid]["properties"]) == {"name": "a"}
        assert decode_properties(stored_b[record_uuid]["properties"]) == {"name": "b"}

        await store.delete_collection(namespace=NAMESPACE, name="tenant_a")
        await store.delete_collection(namespace=NAMESPACE, name="tenant_b")


class TestPurgeBatches:
    @pytest.mark.asyncio
    async def test_a_purge_round_reclaims_at_most_one_batch(self, store):
        store = MilvusVectorStore(
            MilvusVectorStoreParams(
                client=store._client,
                collection_registry=store._collection_registry,
                request_timeout_seconds=REQUEST_TIMEOUT_SECONDS,
                max_varchar_length=MAX_VARCHAR_LENGTH,
                purge_batch_size=2,
            )
        )
        config = VectorStoreCollectionConfig(vector_dimensions=VECTOR_DIM)
        await store.create_collection(
            namespace=NAMESPACE, name="batched", config=config
        )
        collection = await store.open_collection(namespace=NAMESPACE, name="batched")
        assert collection is not None
        await collection.upsert(
            records=[
                _make_record(vector=_normalize([1.0, float(i), 0.0])) for i in range(5)
            ]
        )
        await _settle(collection)
        incarnation = collection._incarnation
        await store.delete_collection(namespace=NAMESPACE, name="batched")
        native = MilvusVectorStore._build_native_collection_name(NAMESPACE, config)

        async def left_of_the_incarnation() -> int:
            return len(
                await store._client.query(
                    collection_name=native,
                    filter=f'partition_key == "{incarnation}"',
                    output_fields=["id"],
                    limit=16384,
                    consistency_level="Strong",
                )
            )

        left_after_each_round = []
        while await store.purge_deleted_collections():
            await _settle(collection)
            left_after_each_round.append(await left_of_the_incarnation())
        # Batches of 2, then a round that finds nothing and removes the tombstone.
        assert left_after_each_round == [3, 1, 0, 0]


class TestLifecycleContract(CollectionLifecycleContract):
    """The collection lifecycle contract, against this store."""

    @staticmethod
    async def count_stored(store, namespace: str, config) -> int:
        native = MilvusVectorStore._build_native_collection_name(namespace, config)
        rows = await store._client.query(
            collection_name=native,
            filter='id != ""',
            output_fields=["id"],
            limit=16384,
            consistency_level="Strong",
        )
        return len(list(rows))

    settle = staticmethod(_settle)

    @staticmethod
    async def stored_uuids(collection) -> set[UUID]:
        rows = await collection._client.query(
            collection_name=collection._native_collection_name,
            filter=f'partition_key == "{collection._incarnation}"',
            output_fields=["record_uuid"],
            limit=16384,
            consistency_level="Strong",
        )
        return {UUID(row["record_uuid"]) for row in rows}
