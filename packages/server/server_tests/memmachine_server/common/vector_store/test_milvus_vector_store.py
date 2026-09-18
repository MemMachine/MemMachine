"""Tests for MilvusVectorStore."""

# ruff: noqa: E402

import math
from datetime import UTC, datetime, timedelta, timezone
from unittest.mock import MagicMock
from uuid import UUID, uuid4

import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import create_async_engine

from server_tests.memmachine_server.common.vector_store.partition_lifecycle_contract import (
    RemotePartitionLifecycleContract,
)

pytest.importorskip("milvus_lite")
pymilvus = pytest.importorskip("pymilvus")
DataType = pymilvus.DataType
MilvusClient = pymilvus.MilvusClient

from memmachine_server.common.data_types import PropertyValue
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
)
from memmachine_server.common.vector_store.milvus_vector_store import (
    MilvusVectorStore,
    MilvusVectorStoreParams,
    MilvusVectorStorePartition,
)
from server_tests.memmachine_server.common.filter.nodes import comparison
from server_tests.memmachine_server.common.vector_store.declared_schema_contract import (
    DeclaredSchemaContract,
)

COLLECTION = "test_namespace"
NAME = "test_name"
VECTOR_DIM = 3
REQUEST_TIMEOUT_SECONDS = 30
TOMBSTONE_RETENTION_SECONDS = 86400

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
    dimensions = VECTOR_DIM
    probe = [1.0] + [0.0] * (dimensions - 1)
    [result] = await collection.query(query_vectors=[probe], limit=_FETCH_LIMIT)
    present = {match.record_uuid for match in result.matches}
    return [uuid for uuid in record_uuids if uuid in present]


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


@pytest_asyncio.fixture
async def store(tmp_path):
    client = MilvusClient(uri=str(tmp_path / "test_milvus.db"))
    registry_engine = create_async_engine(
        f"sqlite+aiosqlite:///{tmp_path / 'registry.db'}"
    )
    vector_store = MilvusVectorStore(
        MilvusVectorStoreParams(
            collection=COLLECTION,
            vector_dimensions=VECTOR_DIM,
            indexed_properties=INDEXED_PROPERTIES,
            client=client,
            registry_engine=registry_engine,
            tombstone_retention_seconds=TOMBSTONE_RETENTION_SECONDS,
            consistency_level="Session",
            request_timeout_seconds=REQUEST_TIMEOUT_SECONDS,
        )
    )
    await vector_store.provision()
    await vector_store.startup()
    yield vector_store
    await vector_store.shutdown()
    client.close()
    await registry_engine.dispose()


# MilvusClient's constructor timeout bounds only the connection; a request
# is bounded only by the timeout passed to it, so every request must carry it.
_CLIENT_REQUESTS = (
    "has_collection",
    "create_collection",
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

    await store.create_partition("timed")
    partition = await store.get_partition("timed")
    assert partition is not None
    record = _make_record(vector=_normalize([1.0, 0.0, 0.0]))
    await partition.upsert(records=[record])
    await partition.query(query_vectors=[record.vector], limit=1)
    await partition.delete(record_uuids=[record.uuid])
    await store.delete_partition("timed")

    called = {name for name, spy in spies.items() if spy.call_count}
    assert called >= {"upsert", "search", "delete"}
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


class TestCollectionLifecycle:
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
    async def test_partitions_share_the_native_collection(self, store):
        """Every partition lives in the store's one native collection."""
        await store.create_partition("coll_a")
        await store.create_partition("coll_b")

        coll_a = await store.get_partition("coll_a")
        coll_b = await store.get_partition("coll_b")
        assert coll_a is not None
        assert coll_b is not None
        assert coll_a._collection_name == coll_b._collection_name

        await store.delete_partition("coll_a")
        await store.delete_partition("coll_b")

    @pytest.mark.asyncio
    async def test_native_collection_schema(self, store):
        await store.create_partition("schema")
        coll = await store.get_partition("schema")
        assert coll is not None

        schema = store._client.describe_collection(coll._collection_name)
        fields = {field["name"]: field for field in schema["fields"]}

        assert schema["auto_id"] is False
        assert schema["enable_dynamic_field"] is True
        assert fields["id"]["is_primary"] is True
        assert fields["partition_key"]["is_partition_key"] is True
        assert fields["vector"]["type"] == DataType.FLOAT_VECTOR
        assert fields["vector"]["params"]["dim"] == VECTOR_DIM

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


class TestPartitionIsolation:
    @pytest.mark.asyncio
    async def test_same_uuid_can_exist_in_different_logical_collections(self, store):
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


class TestPartitionLifecycle(RemotePartitionLifecycleContract):
    """The partition lifecycle contract, against this store."""

    @staticmethod
    async def count_stored(store) -> int:
        rows = store._client.query(
            collection_name=COLLECTION,
            filter='id != ""',
            output_fields=["id"],
            limit=16384,
        )
        return len(list(rows))
