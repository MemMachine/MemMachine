"""Tests for MilvusVectorStore."""

# ruff: noqa: E402

import math
from datetime import UTC, datetime, timedelta, timezone
from uuid import UUID, uuid4

import pytest
import pytest_asyncio

pytest.importorskip("milvus_lite")
pymilvus = pytest.importorskip("pymilvus")
DataType = pymilvus.DataType
MilvusClient = pymilvus.MilvusClient

from memmachine_server.common.data_types import PropertyValue, SimilarityMetric
from memmachine_server.common.filter.filter_parser import (
    And,
    Comparison,
    In,
    IsNull,
    Not,
    Or,
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


def _normalize(vector: list[float]) -> list[float]:
    magnitude = math.sqrt(sum(x * x for x in vector))
    return [x / magnitude for x in vector]


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


@pytest_asyncio.fixture
async def store(tmp_path):
    client = MilvusClient(uri=str(tmp_path / "test_milvus.db"))
    vector_store = MilvusVectorStore(
        MilvusVectorStoreParams(client=client, consistency_level="Session")
    )
    await vector_store.startup()
    yield vector_store
    await vector_store.shutdown()
    client.close()


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
        assert coll_a._collection_name == coll_b._collection_name

        await store.delete_collection(namespace=NAMESPACE, name="coll_a")
        await store.delete_collection(namespace=NAMESPACE, name="coll_b")

    @pytest.mark.asyncio
    async def test_native_collection_schema(self, store):
        await store.create_collection(
            namespace=NAMESPACE,
            name="schema",
            config=VectorStoreCollectionConfig(vector_dimensions=VECTOR_DIM),
        )
        coll = await store.open_collection(namespace=NAMESPACE, name="schema")
        assert coll is not None

        schema = store._client.describe_collection(coll._collection_name)
        fields = {field["name"]: field for field in schema["fields"]}

        assert schema["auto_id"] is False
        assert schema["enable_dynamic_field"] is True
        assert fields["id"]["is_primary"] is True
        assert fields["partition_key"]["is_partition_key"] is True
        assert fields["vector"]["type"] == DataType.FLOAT_VECTOR
        assert fields["vector"]["params"]["dim"] == VECTOR_DIM
        assert fields["properties"]["type"] == DataType.JSON

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
        assert matches[1].record.uuid == r3.uuid
        assert matches[2].record.uuid == r2.uuid
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

        query_results = await collection.query(
            query_vectors=[v1], limit=10, score_threshold=0.9
        )
        matches = query_results[0].matches
        assert len(matches) == 1
        assert matches[0].record.uuid == r1.uuid

    @pytest.mark.asyncio
    async def test_query_return_flags(self, collection):
        v1 = _normalize([1.0, 0.0, 0.0])
        r1 = _make_record(vector=v1, properties={"name": "test"})
        await collection.upsert(records=[r1])

        no_vector = await collection.query(
            query_vectors=[v1], limit=10, return_vector=False
        )
        assert no_vector[0].matches[0].record.vector is None
        assert no_vector[0].matches[0].record.properties is not None

        no_properties = await collection.query(
            query_vectors=[v1],
            limit=10,
            return_vector=True,
            return_properties=False,
        )
        assert no_properties[0].matches[0].record.vector is not None
        assert no_properties[0].matches[0].record.properties is None

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
        assert {match.record.uuid for match in results[0].matches} == {record.uuid}


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
            property_filter=Comparison(field=field, op=op, value=value),
        )
        return {match.record.uuid for match in all_results[0].matches}

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

        results = await collection.get(record_uuids=[r1.uuid])
        assert results[0].properties["created_at"] == dt_utc

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
        assert {m.record.uuid for m in null_results[0].matches} == {r_missing.uuid}

        not_null_results = await collection.query(
            query_vectors=[v1],
            limit=10,
            property_filter=Not(expr=IsNull(field="name")),
        )
        assert {m.record.uuid for m in not_null_results[0].matches} == {
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
        assert {m.record.uuid for m in in_results[0].matches} == {r1.uuid, r3.uuid}

        and_results = await collection.query(
            query_vectors=[v1],
            limit=10,
            property_filter=And(
                left=Comparison(field="active", op="=", value=True),
                right=Comparison(field="age", op=">", value=30),
            ),
        )
        assert {m.record.uuid for m in and_results[0].matches} == {r3.uuid}

        or_results = await collection.query(
            query_vectors=[v1],
            limit=10,
            property_filter=Or(
                left=Comparison(field="name", op="=", value="alice"),
                right=Comparison(field="name", op="=", value="bob"),
            ),
        )
        assert {m.record.uuid for m in or_results[0].matches} == {r1.uuid, r2.uuid}


class TestGetAndDelete:
    @pytest.mark.asyncio
    async def test_get_by_uuids_preserves_order_and_return_flags(self, collection):
        v1 = _normalize([1.0, 0.0, 0.0])
        v2 = _normalize([0.0, 1.0, 0.0])
        r1 = _make_record(vector=v1, properties={"name": "a"})
        r2 = _make_record(vector=v2, properties={"name": "b"})
        await collection.upsert(records=[r1, r2])

        results = await collection.get(
            record_uuids=[r2.uuid, r1.uuid],
            return_vector=True,
            return_properties=False,
        )
        assert [record.uuid for record in results] == [r2.uuid, r1.uuid]
        assert results[0].vector is not None
        assert results[0].properties is None

    @pytest.mark.asyncio
    async def test_delete_records(self, collection):
        v1 = _normalize([1.0, 0.0, 0.0])
        v2 = _normalize([0.0, 1.0, 0.0])
        r1 = _make_record(vector=v1)
        r2 = _make_record(vector=v2)
        await collection.upsert(records=[r1, r2])

        await collection.delete(record_uuids=[r1.uuid])

        results = await collection.get(record_uuids=[r1.uuid, r2.uuid])
        assert [record.uuid for record in results] == [r2.uuid]


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
        await coll_b.upsert(
            records=[Record(uuid=record_uuid, vector=v1, properties={"name": "b"})]
        )

        results_a = await coll_a.get(record_uuids=[record_uuid])
        results_b = await coll_b.get(record_uuids=[record_uuid])

        assert results_a[0].properties == {"name": "a"}
        assert results_b[0].properties == {"name": "b"}

        await store.delete_collection(namespace=NAMESPACE, name="tenant_a")
        await store.delete_collection(namespace=NAMESPACE, name="tenant_b")
