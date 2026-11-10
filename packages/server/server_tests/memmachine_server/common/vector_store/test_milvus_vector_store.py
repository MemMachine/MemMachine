"""Tests for MilvusVectorStore."""

# ruff: noqa: E402

import asyncio
import math
import random
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta, timezone
from typing import Any
from unittest.mock import AsyncMock, MagicMock
from uuid import UUID, uuid4

import pytest
import pytest_asyncio

from server_tests.memmachine_server.common.vector_store.in_memory_vector_store_partition import (
    evaluate_filter,
)
from server_tests.memmachine_server.common.vector_store.partition_lifecycle_contract import (
    PartitionLifecycleContract,
)

pymilvus = pytest.importorskip("pymilvus")
DataType = pymilvus.DataType
AsyncMilvusClient = pymilvus.AsyncMilvusClient

import grpc
import grpc.aio
from pymilvus.client.types import ConsistencyLevel

from memmachine_server.common.data_types import (
    PropertyType,
    PropertyValue,
    SimilarityMetric,
)
from memmachine_server.common.filter.filter_parser import (
    And,
    Comparison,
    FilterExpr,
    In,
    IsNull,
    Not,
    Or,
)
from memmachine_server.common.metrics_factory import OperationTracker
from memmachine_server.common.vector_store.data_types import (
    Record,
    VectorStoreSchema,
)
from memmachine_server.common.vector_store.milvus_vector_store import (
    MilvusVectorStore,
    MilvusVectorStoreParams,
    MilvusVectorStorePartition,
)
from server_tests.memmachine_server.common.vector_store.declared_schema_contract import (
    DeclaredSchemaContract,
)

VECTOR_STORE_NAME = "test_vector_store"
VECTOR_DIM = 3
INDEXED_PROPERTIES: dict[str, PropertyType] = {
    "name": str,
    "age": int,
    "score": float,
    "active": bool,
    "created_at": datetime,
}
# Settings other than their defaults, so a default in their place shows.
REQUEST_TIMEOUT_SECONDS = 17
MAX_VARCHAR_LENGTH = 1024
PURGE_BATCH_SIZE = 5000
# Purge calls `_purge` makes before it fails the test: far more than the
# records a test writes need, so only a purge that keeps finding records
# reaches it.
_MAX_PURGE_CALLS = 1000


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


def _schema(**overrides) -> VectorStoreSchema:
    fields: dict[str, Any] = {
        "vector_dimensions": VECTOR_DIM,
        "similarity_metric": SimilarityMetric.COSINE,
        "indexed_properties": INDEXED_PROPERTIES,
    }
    fields.update(overrides)
    return VectorStoreSchema(**fields)


def _params(client, **overrides) -> MilvusVectorStoreParams:
    """Parameters for one store."""
    params: dict[str, Any] = {
        "client": client,
        "vector_store_name": VECTOR_STORE_NAME,
        "store_schema": _schema(),
        "request_timeout_seconds": REQUEST_TIMEOUT_SECONDS,
        "max_varchar_length": MAX_VARCHAR_LENGTH,
        "purge_batch_size": PURGE_BATCH_SIZE,
    }
    params.update(overrides)
    return MilvusVectorStoreParams(**params)


async def _settle(partition: MilvusVectorStorePartition) -> None:
    """Return once the store's reads reflect every write made so far.

    The store reads at Bounded, which may lag its writes. A Strong read
    returns only once the server has applied every earlier write, and with
    one replica the store's later reads start from that point.
    """
    await partition._client.query(
        collection_name=partition._collection_name,
        filter='id != ""',
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


async def _incarnation_uuids(
    client: AsyncMilvusClient, collection_name: str, incarnation: UUID
) -> list[UUID]:
    """The record UUIDs Milvus holds under an incarnation, once per entity, read past the store at Strong."""
    rows = await client.query(
        collection_name=collection_name,
        filter=f'partition_key == "{incarnation}"',
        output_fields=["record_uuid"],
        limit=16384,
        consistency_level="Strong",
    )
    return [UUID(row["record_uuid"]) for row in rows]


async def _open(store: MilvusVectorStore) -> MilvusVectorStorePartition:
    """A handle on a freshly prepared partition incarnation."""
    incarnation = uuid4()
    await store.prepare_partition_incarnation(incarnation)
    return await store.open_partition_incarnation(incarnation)


async def _purge(store: MilvusVectorStore, incarnation: UUID) -> None:
    """Purge an incarnation until the store finds nothing of it, failing on a purge that never ends."""
    for _ in range(_MAX_PURGE_CALLS):
        if not await store.purge_partition_incarnation(incarnation):
            return
    pytest.fail(
        f"incarnation {incarnation} still held records after "
        f"{_MAX_PURGE_CALLS} purge calls"
    )


# A property of every type, for the tests against a model.
_MODEL_DECLARED: dict[str, PropertyType] = {
    "d_bool": bool,
    "d_int": int,
    "d_float": float,
    "d_str": str,
    "d_datetime": datetime,
}
_MODEL_INTS = [-7, -1, 0, 1, 2, 3, 1 << 40]
_MODEL_FLOATS = [-2.5, -0.5, 0.0, 0.25, 1.0, 3.75]
_MODEL_STRINGS = [
    "",
    "alpha",
    "Alpha",
    "beta",
    "a b",
    'q"uote',
    "back\\slash",
    "it's",
    "\u00fcn\u00ef",
]
# Instants a microsecond and a second apart, each written at varied offsets.
_MODEL_INSTANTS = [
    datetime(1999, 12, 31, 23, 59, 59, 999999, tzinfo=UTC),
    datetime(2024, 6, 14, 12, tzinfo=UTC),
    datetime(2024, 6, 15, 12, tzinfo=UTC),
    datetime(2024, 6, 15, 12, 0, 0, 1, tzinfo=UTC),
    datetime(2024, 6, 15, 12, 0, 1, tzinfo=UTC),
]
_MODEL_OFFSETS = [
    UTC,
    timezone(timedelta(hours=5, minutes=30)),
    timezone(timedelta(hours=-8)),
    timezone(timedelta(hours=14)),
    timezone(timedelta(hours=-12)),
]


def _model_value(rng: random.Random, value_type: PropertyType) -> PropertyValue:
    if value_type is bool:
        return rng.choice([False, True])
    if value_type is int:
        return rng.choice(_MODEL_INTS)
    if value_type is float:
        return rng.choice(_MODEL_FLOATS)
    if value_type is str:
        return rng.choice(_MODEL_STRINGS)
    return rng.choice(_MODEL_INSTANTS).astimezone(rng.choice(_MODEL_OFFSETS))


def _model_properties(rng: random.Random) -> dict[str, PropertyValue]:
    """Properties of every type, each missing a quarter of the time."""
    return {
        key: _model_value(rng, value_type)
        for key, value_type in _MODEL_DECLARED.items()
        if rng.random() < 0.75
    }


def _model_filter(rng: random.Random, depth: int = 3) -> FilterExpr:
    """A random filter tree whose values have their properties' types.

    It compares with `!=` only through Not(=): on a property with no value,
    the store's `!=` holds, as the complement of `=`, where the model's does
    not.
    """
    if depth > 0 and rng.random() < 0.6:
        match rng.choice(("and", "or", "not")):
            case "not":
                return Not(expr=_model_filter(rng, depth - 1))
            case "and":
                return And(
                    left=_model_filter(rng, depth - 1),
                    right=_model_filter(rng, depth - 1),
                )
            case _:
                return Or(
                    left=_model_filter(rng, depth - 1),
                    right=_model_filter(rng, depth - 1),
                )
    key = rng.choice(list(_MODEL_DECLARED))
    value_type = _MODEL_DECLARED[key]
    match rng.choice(("is_null", "comparison", "in")):
        case "is_null":
            return IsNull(field=key)
        case "in" if value_type is int:
            return In(field=key, values=rng.sample(_MODEL_INTS, k=rng.randint(1, 3)))
        case "in" if value_type is str:
            return In(field=key, values=rng.sample(_MODEL_STRINGS, k=rng.randint(1, 3)))
        case _:
            op = "=" if value_type is bool else rng.choice(("=", "<", "<=", ">", ">="))
            return Comparison(field=key, op=op, value=_model_value(rng, value_type))


async def _model_matches(
    partition: MilvusVectorStorePartition, property_filter: FilterExpr | None
) -> set[UUID]:
    """The UUIDs of every record of the partition the filter selects."""
    [result] = await partition.query(
        query_vectors=[_normalize([1.0, 0.0, 0.0])],
        limit=1000,
        property_filter=property_filter,
    )
    return {match.record_uuid for match in result.matches}


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


async def _started_store(client: AsyncMilvusClient, **overrides) -> MilvusVectorStore:
    """A started store."""
    vector_store = MilvusVectorStore(_params(client, **overrides))
    await vector_store.startup()
    return vector_store


@pytest_asyncio.fixture
async def store(milvus_client):
    vector_store = await _started_store(milvus_client)
    yield vector_store
    await vector_store.shutdown()


@pytest_asyncio.fixture
async def other_store(milvus_container):
    """A store as another process runs it beside `store`: its own Milvus
    client, over the same Milvus."""
    client = AsyncMilvusClient(uri=milvus_container.get_connection_url())
    vector_store = await _started_store(client)
    yield vector_store
    await vector_store.shutdown()
    await client.close()


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
    timed = MilvusVectorStore(_params(store._client, vector_store_name="timed"))
    await timed.startup()
    partition = await _open(timed)
    vector = _normalize([1.0, 0.0, 0.0])
    record, kept = (
        _make_record(vector=vector),
        _make_record(vector=_normalize([0.0, 1.0, 0.0])),
    )
    await partition.upsert(records=[record, kept])
    await _settle(partition)
    await partition.query(query_vectors=[vector], limit=1)
    await partition.delete(record_uuids=[record.uuid])
    # The purge finds the record the delete left and reclaims it.
    await _purge(timed, partition._incarnation)

    assert {name for name, spy in spies.items() if spy.call_count} >= set(
        _CLIENT_REQUESTS
    )
    for name, spy in spies.items():
        for call in spy.call_args_list:
            assert call.kwargs.get("timeout") == REQUEST_TIMEOUT_SECONDS, (name, call)


@pytest.mark.parametrize(
    "setting", ["request_timeout_seconds", "max_varchar_length", "purge_batch_size"]
)
def test_a_setting_that_is_not_positive_is_refused(setting):
    with pytest.raises(ValueError, match=setting):
        MilvusVectorStoreParams(
            client=MagicMock(spec=AsyncMilvusClient),
            vector_store_name=VECTOR_STORE_NAME,
            store_schema=VectorStoreSchema(
                vector_dimensions=VECTOR_DIM, indexed_properties=INDEXED_PROPERTIES
            ),
            **{setting: 0},
        )


@pytest_asyncio.fixture
async def partition(store):
    partition = await _open(store)
    yield partition
    await _purge(store, partition._incarnation)


class TestPartitionLifecycle:
    @pytest.mark.asyncio
    @pytest.mark.parametrize("failing_step", ["create_index", "load_collection"])
    async def test_a_startup_that_failed_part_way_is_as_if_never_attempted(
        self, store, monkeypatch, failing_step
    ):
        """The next startup completes the native collection a failed one left behind."""
        partial = MilvusVectorStore(
            _params(
                store._client,
                vector_store_name=f"partial_{failing_step}",
                store_schema=_schema(indexed_properties={"name": str}),
            )
        )

        async def refuse(*args, **kwargs):
            raise pymilvus.MilvusException(message=f"{failing_step} refused")

        with monkeypatch.context() as patch:
            patch.setattr(store._client, failing_step, refuse)
            with pytest.raises(pymilvus.MilvusException, match="refused"):
                await partial.startup()

        await partial.startup()
        partition = await _open(partial)
        vector = _normalize([1.0, 0.0, 0.0])
        record = _make_record(vector=vector, properties={"name": "alice"})
        await partition.upsert(records=[record])
        await _settle(partition)
        [result] = await partition.query(
            query_vectors=[vector],
            limit=1,
            property_filter=Comparison(field="name", op="=", value="alice"),
        )
        assert [match.record_uuid for match in result.matches] == [record.uuid]
        assert set(await store._client.list_indexes(partial._collection_name)) == {
            "vector",
            "_p_name",
        }
        await _purge(partial, partition._incarnation)

    @pytest.mark.asyncio
    async def test_a_vector_store_name_may_begin_with_a_digit(self, store):
        """Milvus refuses a collection name beginning with a digit; the store's
        collection name begins with its prefix instead."""
        digit_first = MilvusVectorStore(
            _params(store._client, vector_store_name="0_digit_first")
        )
        await digit_first.startup()
        partition = await _open(digit_first)
        await _purge(digit_first, partition._incarnation)

    @pytest.mark.asyncio
    async def test_partitions_share_the_store_collection(self, store):
        """Every partition's records are entities in the store's one native collection."""
        records = {}
        for _ in range(2):
            partition = await _open(store)
            record = _make_record(vector=[1.0, 0.0, 0.0])
            await partition.upsert(records=[record])
            records[partition._incarnation] = record

        rows = await store._client.query(
            collection_name=store._collection_name,
            filter="record_uuid in ["
            + ", ".join(f'"{record.uuid}"' for record in records.values())
            + "]",
            output_fields=["record_uuid"],
            consistency_level="Strong",
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
        assert {UUID(row["record_uuid"]) for row in rows} == {
            record.uuid for record in records.values()
        }

        for incarnation in records:
            await _purge(store, incarnation)

    @pytest.mark.asyncio
    async def test_native_collection_schema(self, store):
        """Each declared property is a typed, nullable, indexed field, a
        datetime with a field for its offset; the collection isolates tenants."""
        partition = await _open(store)
        native = partition._collection_name

        schema = await store._client.describe_collection(partition._collection_name)
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
        assert set(fields) == {
            "id",
            "record_uuid",
            "partition_key",
            "vector",
            *expected,
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

        await _purge(store, partition._incarnation)

    @pytest.mark.asyncio
    async def test_the_longest_declared_key_names_a_field_milvus_accepts(self, store):
        """A declared property's field is named by its key under a prefix,
        within the server's proxy.maxNameLength."""
        key = "k" * 32  # the longest property key
        # A name of its own, so its native collection declares the key.
        longest = await _started_store(
            store._client,
            vector_store_name="longest_key",
            store_schema=_schema(indexed_properties={key: datetime}),
        )
        partition = await _open(longest)
        written = datetime(2024, 6, 15, 12, 0, tzinfo=timezone(timedelta(hours=9)))
        record = _make_record(
            vector=_normalize([1.0, 0.0, 0.0]), properties={key: written}
        )
        await partition.upsert(records=[record])
        await _settle(partition)

        stored = (await _stored(partition, [record.uuid]))[record.uuid]
        assert datetime.fromisoformat(stored[f"_p_{key}"]) == written
        [result] = await partition.query(
            query_vectors=[record.vector],
            limit=10,
            property_filter=Comparison(field=key, op="=", value=written),
        )
        assert [match.record_uuid for match in result.matches] == [record.uuid]

        await _purge(longest, partition._incarnation)

    @pytest.mark.asyncio
    async def test_unsupported_metric_raises(self, store):
        with pytest.raises(ValueError, match="Milvus only supports"):
            MilvusVectorStore(
                _params(
                    store._client,
                    vector_store_name="bad_metric",
                    store_schema=_schema(similarity_metric=SimilarityMetric.MANHATTAN),
                )
            )


async def _require_usable(partition: MilvusVectorStorePartition) -> None:
    """Write a record to the partition, and find it by a filtered query."""
    record = _make_record(
        vector=_normalize([1.0, 0.0, 0.0]), properties={"name": "usable"}
    )
    await partition.upsert(records=[record])
    await _settle(partition)
    [result] = await partition.query(
        query_vectors=[record.vector],
        limit=10,
        property_filter=Comparison(field="name", op="=", value="usable"),
    )
    assert [match.record_uuid for match in result.matches] == [record.uuid]


class TestConcurrentPreparation:
    @pytest.mark.asyncio
    async def test_a_startup_finding_the_native_collection_mid_preparation_completes_it(
        self, store, other_store, monkeypatch
    ):
        """Two stores of one name start at once, and the second finds the
        native collection the first created but has not yet indexed or
        loaded. The second's startup completes it, so its partition is usable
        before the first startup goes on, and both stores' partitions end
        usable with every index."""
        # A name of its own, so the first startup creates the native collection.
        first = MilvusVectorStore(
            _params(
                store._client,
                vector_store_name="raced_preparation",
                store_schema=_schema(indexed_properties={"name": str}),
            )
        )
        second = MilvusVectorStore(
            _params(
                other_store._client,
                vector_store_name="raced_preparation",
                store_schema=_schema(indexed_properties={"name": str}),
            )
        )
        first_created = asyncio.Event()
        second_done = asyncio.Event()
        create_native_collection = store._client.create_collection

        async def create_and_hold(*args, **kwargs):
            await create_native_collection(*args, **kwargs)
            first_created.set()
            await asyncio.wait_for(second_done.wait(), 60)

        monkeypatch.setattr(store._client, "create_collection", create_and_hold)

        async def start_second() -> None:
            try:
                await asyncio.wait_for(first_created.wait(), 60)
                await second.startup()
                await _require_usable(await _open(second))
            finally:
                second_done.set()

        await asyncio.gather(first.startup(), start_second())

        await _require_usable(await _open(first))
        assert set(await store._client.list_indexes(first._collection_name)) == {
            "vector",
            "_p_name",
        }


class TestUpsertAndQuery:
    @pytest.mark.asyncio
    async def test_an_upsert_over_the_request_size_limit_stores_every_record(
        self, store
    ):
        """Milvus refuses a request over proxy.grpc.serverMaxRecvSize (64 MiB
        unless configured), so the batch is halved until it fits."""
        # A store of its own, whose declared string holds 60,000 bytes; the
        # batch is over 64 MiB. pymilvus checks a string value a character at
        # a time while it builds a request, within the request's timeout, so
        # this store gives its requests longer than the module's timeout.
        oversized = await _started_store(
            store._client,
            vector_store_name="oversized",
            store_schema=_schema(indexed_properties={"text": str}),
            max_varchar_length=65535,
            request_timeout_seconds=120,
        )
        partition = await _open(oversized)
        text = "x" * 60_000
        records = [
            _make_record(vector=_normalize([1.0, 0.0, 0.0]), properties={"text": text})
            for _ in range(1_200)
        ]

        await partition.upsert(records=records)

        assert sorted(
            await _incarnation_uuids(
                partition._client,
                partition._collection_name,
                partition._incarnation,
            )
        ) == sorted(record.uuid for record in records)

    @pytest.mark.asyncio
    async def test_a_query_may_ask_for_hundreds_of_results(self, partition):
        records = [
            _make_record(vector=_normalize([1.0, float(i), 0.0])) for i in range(3)
        ]
        await partition.upsert(records=records)
        await _settle(partition)

        [result] = await partition.query(
            query_vectors=[_normalize([1.0, 0.0, 0.0])], limit=200
        )

        assert {match.record_uuid for match in result.matches} == {
            record.uuid for record in records
        }

    @pytest.mark.asyncio
    async def test_upsert_and_query_basic(self, partition):
        v1 = _normalize([1.0, 0.0, 0.0])
        v2 = _normalize([0.0, 1.0, 0.0])
        v3 = _normalize([1.0, 0.1, 0.0])

        r1 = _make_record(vector=v1, properties={"name": "a"})
        r2 = _make_record(vector=v2, properties={"name": "b"})
        r3 = _make_record(vector=v3, properties={"name": "c"})

        await partition.upsert(records=[r1, r2, r3])
        await _settle(partition)

        query_results = await partition.query(query_vectors=[v1], limit=3)
        matches = query_results[0].matches

        assert len(matches) == 3
        assert matches[0].record_uuid == r1.uuid
        assert matches[1].record_uuid == r3.uuid
        assert matches[2].record_uuid == r2.uuid
        assert matches[0].score >= matches[1].score >= matches[2].score
        assert matches[0].score == pytest.approx(1.0)
        assert matches[2].score == pytest.approx(0.0)

    @pytest.mark.asyncio
    async def test_query_with_similarity_threshold(self, partition):
        v1 = _normalize([1.0, 0.0, 0.0])
        v2 = _normalize([0.0, 1.0, 0.0])

        r1 = _make_record(vector=v1)
        r2 = _make_record(vector=v2)
        await partition.upsert(records=[r1, r2])
        await _settle(partition)

        query_results = await partition.query(
            query_vectors=[v1], limit=10, score_threshold=0.9
        )
        matches = query_results[0].matches
        assert len(matches) == 1
        assert matches[0].record_uuid == r1.uuid

    @pytest.mark.asyncio
    async def test_query_batch_multiple_vectors(self, partition):
        v1 = _normalize([1.0, 0.0, 0.0])
        v2 = _normalize([0.0, 1.0, 0.0])

        r1 = _make_record(vector=v1, properties={"name": "a"})
        r2 = _make_record(vector=v2, properties={"name": "b"})
        await partition.upsert(records=[r1, r2])
        await _settle(partition)

        all_results = await partition.query(query_vectors=[v1, v2], limit=1)

        assert len(all_results) == 2
        assert all_results[0].matches[0].record_uuid == r1.uuid
        assert all_results[1].matches[0].record_uuid == r2.uuid

    @pytest.mark.asyncio
    async def test_upsert_removes_stale_filter_fields(self, partition):
        v1 = _normalize([1.0, 0.0, 0.0])
        record = _make_record(vector=v1, properties={"name": "old"})
        await partition.upsert(records=[record])
        await _settle(partition)

        await partition.upsert(
            records=[Record(uuid=record.uuid, vector=v1, properties={})]
        )
        await _settle(partition)

        results = await partition.query(
            query_vectors=[v1],
            limit=10,
            property_filter=IsNull(field="name"),
        )
        assert {match.record_uuid for match in results[0].matches} == {record.uuid}
        [old] = await partition.query(
            query_vectors=[v1],
            limit=10,
            property_filter=Comparison(field="name", op="=", value="old"),
        )
        assert old.matches == []

    @pytest.mark.asyncio
    async def test_upsert_failure_preserves_existing_record(
        self, partition, monkeypatch
    ):
        old_vector = _normalize([1.0, 0.0, 0.0])
        new_vector = _normalize([0.0, 1.0, 0.0])
        record = _make_record(vector=old_vector, properties={"name": "old"})
        await partition.upsert(records=[record])
        await _settle(partition)

        async def fail_upsert(*args, **kwargs):
            raise RuntimeError("upsert failed")

        monkeypatch.setattr(partition._client, "upsert", fail_upsert)

        with pytest.raises(RuntimeError, match="upsert failed"):
            await partition.upsert(
                records=[
                    Record(
                        uuid=record.uuid,
                        vector=new_vector,
                        properties={"name": "new"},
                    )
                ]
            )

        stored = (await _stored(partition, [record.uuid]))[record.uuid]
        assert list(stored["vector"]) == old_vector
        assert stored["_p_name"] == "old"


class TestDeclaredSchema(DeclaredSchemaContract):
    """The declared-schema contract, against this store."""

    settle = staticmethod(_settle)


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
        await _settle(partition)
        return r1, r2, r3, v1

    async def _query(self, partition, query_vec, field, op, value):
        all_results = await partition.query(
            query_vectors=[query_vec],
            limit=10,
            property_filter=Comparison(field=field, op=op, value=value),
        )
        return {match.record_uuid for match in all_results[0].matches}

    @pytest.mark.asyncio
    async def test_scalar_filters(self, partition):
        r1, r2, r3, v1 = await self._setup(partition)

        assert await self._query(partition, v1, "name", "=", "alice") == {r1.uuid}
        assert await self._query(partition, v1, "name", "!=", "alice") == {
            r2.uuid,
            r3.uuid,
        }
        assert await self._query(partition, v1, "age", ">", 30) == {r3.uuid}
        assert await self._query(partition, v1, "age", "<=", 30) == {
            r1.uuid,
            r2.uuid,
        }
        assert await self._query(partition, v1, "score", ">=", 8.0) == {
            r1.uuid,
            r3.uuid,
        }
        assert await self._query(partition, v1, "active", "=", True) == {
            r1.uuid,
            r3.uuid,
        }

    @pytest.mark.asyncio
    async def test_datetime_filters(self, partition):
        v1 = _normalize([1.0, 0.0, 0.0])
        v2 = _normalize([1.0, 0.1, 0.0])
        dt_utc = datetime(2024, 6, 15, 12, 0, 0, tzinfo=UTC)
        dt_other = datetime(2024, 6, 15, 18, 0, 0, tzinfo=UTC)
        r1 = _make_record(vector=v1, properties={"created_at": dt_utc})
        r2 = _make_record(vector=v2, properties={"created_at": dt_other})
        await partition.upsert(records=[r1, r2])
        await _settle(partition)

        plus5 = timezone(timedelta(hours=5))
        dt_filter = datetime(2024, 6, 15, 17, 0, 0, tzinfo=plus5)
        uuids = await self._query(partition, v1, "created_at", "=", dt_filter)
        assert uuids == {r1.uuid}

    @pytest.mark.asyncio
    async def test_is_null_and_not_null(self, partition):
        v1 = _normalize([1.0, 0.0, 0.0])
        v2 = _normalize([1.0, 0.1, 0.0])
        r_has_value = _make_record(vector=v1, properties={"name": "has_name"})
        r_missing = _make_record(vector=v2, properties={"age": 25})
        await partition.upsert(records=[r_has_value, r_missing])
        await _settle(partition)

        null_results = await partition.query(
            query_vectors=[v1],
            limit=10,
            property_filter=IsNull(field="name"),
        )
        assert {m.record_uuid for m in null_results[0].matches} == {r_missing.uuid}

        not_null_results = await partition.query(
            query_vectors=[v1],
            limit=10,
            property_filter=Not(expr=IsNull(field="name")),
        )
        assert {m.record_uuid for m in not_null_results[0].matches} == {
            r_has_value.uuid
        }

    @pytest.mark.asyncio
    async def test_in_and_or_not(self, partition):
        r1, r2, r3, v1 = await self._setup(partition)

        in_results = await partition.query(
            query_vectors=[v1],
            limit=10,
            property_filter=In(field="name", values=["alice", "carol"]),
        )
        assert {m.record_uuid for m in in_results[0].matches} == {r1.uuid, r3.uuid}

        and_results = await partition.query(
            query_vectors=[v1],
            limit=10,
            property_filter=And(
                left=Comparison(field="active", op="=", value=True),
                right=Comparison(field="age", op=">", value=30),
            ),
        )
        assert {m.record_uuid for m in and_results[0].matches} == {r3.uuid}

        or_results = await partition.query(
            query_vectors=[v1],
            limit=10,
            property_filter=Or(
                left=Comparison(field="name", op="=", value="alice"),
                right=Comparison(field="name", op="=", value="bob"),
            ),
        )
        assert {m.record_uuid for m in or_results[0].matches} == {r1.uuid, r2.uuid}

    @pytest.mark.asyncio
    async def test_negation_is_the_complement_missing_values_included(self, partition):
        """A condition on a property with no value is false, so its negation,
        `!=` included, holds there."""
        r1, r2, r3, v1 = await self._setup(partition)
        bare = _make_record(vector=_normalize([1.0, 0.3, 0.0]), properties={})
        await partition.upsert(records=[bare])
        await _settle(partition)

        async def uuids(expr):
            [result] = await partition.query(
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
    async def test_a_declared_datetime_is_stored_as_its_instant_in_utc(self, partition):
        written = datetime(
            2024, 6, 15, 17, 30, tzinfo=timezone(timedelta(hours=5, minutes=30))
        )
        record = _make_record(
            vector=_normalize([1.0, 0.0, 0.0]), properties={"created_at": written}
        )
        await partition.upsert(records=[record])

        stored = (await _stored(partition, [record.uuid]))[record.uuid]
        kept = datetime.fromisoformat(stored["_p_created_at"])
        assert kept == written
        assert kept.utcoffset() == timedelta(0)

    @pytest.mark.asyncio
    async def test_a_declared_datetime_with_a_seconds_offset_is_stored_and_matched(
        self, partition
    ):
        """A datetime whose offset has a seconds component, as local mean time
        does, is stored with that offset and matched by its instant."""
        written = datetime(
            2024, 6, 15, 12, 0, tzinfo=timezone(timedelta(minutes=19, seconds=32))
        )
        record = _make_record(
            vector=_normalize([1.0, 0.0, 0.0]), properties={"created_at": written}
        )
        await partition.upsert(records=[record])
        await _settle(partition)

        stored = (await _stored(partition, [record.uuid]))[record.uuid]
        assert datetime.fromisoformat(stored["_p_created_at"]) == written
        [result] = await partition.query(
            query_vectors=[record.vector],
            limit=10,
            property_filter=Comparison(
                field="created_at", op="=", value=written.astimezone(UTC)
            ),
        )
        assert [match.record_uuid for match in result.matches] == [record.uuid]

    @pytest.mark.asyncio
    async def test_datetime_filters_compare_instants_across_offsets(self, partition):
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
        await partition.upsert(records=records)
        await _settle(partition)
        v1 = _normalize([1.0, 0.0, 0.0])

        async def uuids(expr):
            [result] = await partition.query(
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
    async def test_a_value_of_another_type_matches_nothing(self, partition):
        r1, r2, r3, v1 = await self._setup(partition)

        [matched] = await partition.query(
            query_vectors=[v1],
            limit=10,
            property_filter=Comparison(field="age", op="=", value="thirty"),
        )
        assert matched.matches == []
        [complement] = await partition.query(
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
    async def test_a_character_outside_the_basic_multilingual_plane_matches(
        self, partition
    ):
        """A string filter value with a character outside the Basic
        Multilingual Plane matches."""
        text = "café 😀"
        record = _make_record(
            vector=_normalize([1.0, 0.0, 0.0]), properties={"name": text}
        )
        await partition.upsert(records=[record])
        await _settle(partition)

        assert await self._query(partition, record.vector, "name", "=", text) == {
            record.uuid
        }

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


class TestFilterModel:
    @pytest.mark.asyncio
    async def test_random_filters_agree_with_the_model(self, store):
        """Seeded filter trees over properties of every type, datetimes at
        varied offsets and missing values included, select the records the
        in-memory evaluator selects."""
        rng = random.Random(1736)
        # A store of its own, declaring the model's properties.
        modeled = await _started_store(
            store._client,
            vector_store_name="filter_model",
            store_schema=_schema(indexed_properties=_MODEL_DECLARED),
        )
        partition = await _open(modeled)
        records = [
            Record(
                uuid=record_uuid,
                vector=_normalize([1.0, rng.random(), rng.random()]),
                properties=_model_properties(rng),
            )
            for record_uuid in sorted(uuid4() for _ in range(48))
        ]
        await partition.upsert(records=records)
        await _settle(partition)

        for _ in range(200):
            property_filter = _model_filter(rng)
            assert await _model_matches(partition, property_filter) == {
                record.uuid
                for record in records
                if evaluate_filter(property_filter, record.properties)
            }, property_filter

        await _purge(modeled, partition._incarnation)


@dataclass
class _ModelPartitions:
    """Partitions of one store, with the records a model says each holds."""

    store: MilvusVectorStore
    handles: dict[str, MilvusVectorStorePartition]
    records: dict[str, dict[UUID, Record]]
    dead_incarnations: list[UUID] = field(default_factory=list)

    async def upsert(self, partition_name: str, batch: list[Record]) -> list[Record]:
        """Upsert a batch, and return the records it replaced."""
        await self.handles[partition_name].upsert(records=batch)
        records = self.records[partition_name]
        replaced = [records[record.uuid] for record in batch if record.uuid in records]
        records.update({record.uuid: record for record in batch})
        return replaced

    async def delete(self, partition_name: str, record_uuids: list[UUID]) -> None:
        await self.handles[partition_name].delete(record_uuids=record_uuids)
        for record_uuid in record_uuids:
            self.records[partition_name].pop(record_uuid, None)

    async def recreate(self, partition_name: str) -> None:
        """Move the partition to a fresh incarnation, leaving its current one dead."""
        self.dead_incarnations.append(self.handles[partition_name]._incarnation)
        self.handles[partition_name] = await _open(self.store)
        self.records[partition_name] = {}

    async def purge(self) -> None:
        """Purge every dead incarnation, and check that none keeps a record."""
        for incarnation in self.dead_incarnations:
            await _purge(self.store, incarnation)
            assert (
                await _incarnation_uuids(
                    self.store._client, self.store._collection_name, incarnation
                )
                == []
            ), incarnation

    async def check(self, rng: random.Random) -> None:
        """Check what each partition holds, and what its settled queries select."""
        for partition_name, handle in self.handles.items():
            records = self.records[partition_name]
            stored = await _incarnation_uuids(
                self.store._client, handle._collection_name, handle._incarnation
            )
            assert sorted(stored) == sorted(records), partition_name
            await _settle(handle)
            assert await _model_matches(handle, None) == set(records), partition_name
            for _ in range(3):
                property_filter = _model_filter(rng)
                assert await _model_matches(handle, property_filter) == {
                    record.uuid
                    for record in records.values()
                    if evaluate_filter(property_filter, record.properties)
                }, (partition_name, property_filter)

    async def check_replaced(self, partition_name: str, replaced: list[Record]) -> None:
        """Check that a replaced record's old values select it no more."""
        for old in replaced:
            current = self.records[partition_name][old.uuid]
            for key, value in old.properties.items():
                old_value = Comparison(field=key, op="=", value=value)
                if not evaluate_filter(old_value, current.properties):
                    assert old.uuid not in await _model_matches(
                        self.handles[partition_name], old_value
                    ), old_value


class TestOperationModel:
    @pytest.mark.asyncio
    async def test_a_random_sequence_of_operations_agrees_with_a_model(self, store):
        """Seeded upserts, new and replacing, deletes of present and absent
        records, moves of a partition to a fresh incarnation, and purges of
        the dead incarnations, on two partitions of one store: after each
        step, each partition holds the model's records, its settled queries
        select what the model selects, and a replaced record's old values
        select nothing."""
        rng = random.Random(1736)
        # A store of its own, declaring the model's properties.
        modeled = await _started_store(
            store._client,
            vector_store_name="operation_model",
            store_schema=_schema(indexed_properties=_MODEL_DECLARED),
        )
        partition_names = ("alpha", "beta")
        # Few UUIDs, so each recurs in both partitions and across their lives.
        pool = sorted(uuid4() for _ in range(12))
        model = _ModelPartitions(
            store=modeled,
            handles={
                partition_name: await _open(modeled)
                for partition_name in partition_names
            },
            records={partition_name: {} for partition_name in partition_names},
        )

        for step in range(60):
            partition_name = rng.choice(partition_names)
            replaced: list[Record] = []
            action = rng.choices(
                ("upsert", "delete", "recreate", "purge"), weights=(8, 3, 1, 1)
            )[0]
            match action:
                case "upsert":
                    replaced = await model.upsert(
                        partition_name,
                        [
                            Record(
                                uuid=record_uuid,
                                vector=_normalize([1.0, rng.random(), rng.random()]),
                                properties=_model_properties(rng),
                            )
                            for record_uuid in rng.sample(pool, k=rng.randint(1, 4))
                        ],
                    )
                case "delete":
                    await model.delete(
                        partition_name, rng.sample(pool, k=rng.randint(1, 4))
                    )
                case "recreate":
                    await model.recreate(partition_name)
                case _:
                    await model.purge()
            try:
                await model.check(rng)
                await model.check_replaced(partition_name, replaced)
            except AssertionError as error:
                raise AssertionError(
                    f"after step {step}, {action} {partition_name}"
                ) from error

        for handle in model.handles.values():
            await _purge(modeled, handle._incarnation)


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
        # A store's metric is its collection's, so each metric is its own store.
        scored = MilvusVectorStore(
            _params(
                store._client,
                vector_store_name=f"scores_{metric.value}",
                store_schema=_schema(similarity_metric=metric),
            )
        )
        await scored.startup()
        partition = await _open(scored)
        record = _make_record(vector=[1.2, 1.6, 0.0])
        await partition.upsert(records=[record])
        await _settle(partition)

        [result] = await partition.query(query_vectors=[[1.0, 0.0, 0.0]], limit=1)
        assert result.matches[0].score == pytest.approx(expected, abs=1e-3)
        await _purge(scored, partition._incarnation)

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("metric", "vectors", "threshold", "expected_scores"),
        [
            # Distances 1, 2 and 5; the threshold lies between 2 and its square.
            (
                SimilarityMetric.EUCLIDEAN,
                [[2.0, 0.0, 0.0], [1.0, 2.0, 0.0], [1.0, 0.0, 5.0]],
                3.0,
                [1.0, 2.0],
            ),
            (
                SimilarityMetric.DOT,
                [[3.0, 0.0, 0.0], [2.0, 1.0, 0.0], [-1.0, 0.0, 0.0]],
                1.5,
                [3.0, 2.0],
            ),
            (
                SimilarityMetric.COSINE,
                [[1.0, 0.0, 0.0], [0.6, 0.8, 0.0], [0.0, 1.0, 0.0]],
                0.5,
                [1.0, 0.6],
            ),
        ],
        ids=["euclidean", "dot", "cosine"],
    )
    async def test_a_threshold_keeps_the_matches_within_it_best_first(
        self, store, metric, vectors, threshold, expected_scores
    ):
        """Of three records at increasing distance from the query, the two
        within the threshold match, best first, scored by the metric."""
        # A store's metric is its collection's, so each metric is its own store.
        scored = await _started_store(
            store._client,
            vector_store_name=f"threshold_{metric.value}",
            store_schema=_schema(similarity_metric=metric),
        )
        partition = await _open(scored)
        nearest, middle, farthest = (_make_record(vector=vector) for vector in vectors)
        await partition.upsert(records=[farthest, nearest, middle])
        await _settle(partition)

        [result] = await partition.query(
            query_vectors=[[1.0, 0.0, 0.0]], limit=10, score_threshold=threshold
        )

        assert [match.record_uuid for match in result.matches] == [
            nearest.uuid,
            middle.uuid,
        ]
        assert [match.score for match in result.matches] == pytest.approx(
            expected_scores, abs=1e-3
        )
        await _purge(scored, partition._incarnation)

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "metric",
        [SimilarityMetric.COSINE, SimilarityMetric.DOT, SimilarityMetric.EUCLIDEAN],
    )
    async def test_a_threshold_keeps_a_match_scoring_exactly_it(self, store, metric):
        """A score threshold equal to a match's reported score keeps the
        match; one a step better than that score drops it."""
        scored = await _started_store(
            store._client,
            vector_store_name=f"edge_{metric.value}",
            store_schema=_schema(similarity_metric=metric),
        )
        partition = await _open(scored)
        # Vectors that each metric ranks in its own order, none tied.
        records = [
            _make_record(vector=vector)
            for vector in (
                [3.0, 3.0, 0.0],
                [1.0, 0.2, 0.0],
                [1.4, 0.4, 0.0],
                [1.6, 0.0, 0.0],
                [0.5, 0.05, 0.0],
            )
        ]
        await partition.upsert(records=records)
        await _settle(partition)
        query = [1.0, 0.0, 0.0]
        [ranked] = await partition.query(query_vectors=[query], limit=len(records))
        edge = ranked.matches[2]
        better = math.inf if metric.higher_is_better else -math.inf

        [at_edge] = await partition.query(
            query_vectors=[query], limit=len(records), score_threshold=edge.score
        )
        [past_edge] = await partition.query(
            query_vectors=[query],
            limit=len(records),
            score_threshold=math.nextafter(edge.score, better),
        )

        assert [match.record_uuid for match in at_edge.matches] == [
            match.record_uuid for match in ranked.matches[:3]
        ]
        assert [match.record_uuid for match in past_edge.matches] == [
            match.record_uuid for match in ranked.matches[:2]
        ]
        await _purge(scored, partition._incarnation)


# The client's reads; each takes an optional consistency level.
_CLIENT_READS = ("search", "query", "get", "hybrid_search")
# The levels at which a read reflects every write that returned at least
# common.gracefulTime before it, given by name or by value.
_BOUNDED_OR_STRONGER = {
    "Bounded",
    "Strong",
    ConsistencyLevel.Bounded,
    ConsistencyLevel.Strong,
}


class TestConsistency:
    @pytest.mark.asyncio
    async def test_the_store_reads_at_bounded_or_stronger(self, store, monkeypatch):
        """The store creates the native collection at Bounded by name, and no
        read of the store names a weaker level, so a read lags writes by at
        most common.gracefulTime: the purge and the tombstone retention rely
        on it."""
        spies = {}
        for name in _CLIENT_READS:
            spies[name] = MagicMock(wraps=getattr(store._client, name))
            monkeypatch.setattr(store._client, name, spies[name])
        create = MagicMock(wraps=store._client.create_collection)
        monkeypatch.setattr(store._client, "create_collection", create)
        # A name of its own, so the store creates the native collection.
        bounded = await _started_store(store._client, vector_store_name="bounded")

        partition = await _open(bounded)
        record = _make_record(vector=_normalize([1.0, 0.0, 0.0]))
        await partition.upsert(records=[record])
        await _settle(partition)
        await partition.query(query_vectors=[record.vector], limit=1)
        await _purge(bounded, partition._incarnation)

        description = await store._client.describe_collection(bounded._collection_name)
        assert description["consistency_level"] == ConsistencyLevel.Bounded
        assert create.call_args.kwargs["consistency_level"] == "Bounded"
        calls = [
            (name, call) for name, spy in spies.items() for call in spy.call_args_list
        ]
        assert calls
        for name, call in calls:
            level = call.kwargs.get("consistency_level")
            assert level is None or level in _BOUNDED_OR_STRONGER, (name, call)


class TestDelete:
    @pytest.mark.asyncio
    async def test_delete_records(self, partition):
        v1 = _normalize([1.0, 0.0, 0.0])
        v2 = _normalize([0.0, 1.0, 0.0])
        r1 = _make_record(vector=v1)
        r2 = _make_record(vector=v2)
        await partition.upsert(records=[r1, r2])
        await _settle(partition)

        await partition.delete(record_uuids=[r1.uuid])

        assert set(await _stored(partition, [r1.uuid, r2.uuid])) == {r2.uuid}

    @pytest.mark.asyncio
    async def test_deleting_records_it_does_not_hold_succeeds(self, partition):
        """Milvus accepts the delete of a primary key it does not hold."""
        record = _make_record(vector=_normalize([1.0, 0.0, 0.0]))
        await partition.upsert(records=[record])
        await _settle(partition)

        await partition.delete(record_uuids=[record.uuid, uuid4(), uuid4()])

        assert await _stored(partition, [record.uuid]) == {}


def _partition_on(client: AsyncMilvusClient) -> MilvusVectorStorePartition:
    """A handle on a given client, bound to a fresh incarnation."""
    return MilvusVectorStorePartition(
        client=client,
        collection_name=f"sys_{VECTOR_STORE_NAME}",
        incarnation=uuid4(),
        store_schema=VectorStoreSchema(
            vector_dimensions=VECTOR_DIM,
            similarity_metric=SimilarityMetric.COSINE,
            indexed_properties={},
        ),
        tracker=OperationTracker(None, prefix="test"),
        request_timeout_seconds=REQUEST_TIMEOUT_SECONDS,
    )


@pytest.mark.asyncio
async def test_a_delete_milvus_does_not_accept_in_full_raises():
    """A delete Milvus accepts for fewer primary keys than the store sent raises."""
    client = MagicMock(spec=AsyncMilvusClient)
    client.delete = AsyncMock(return_value={"delete_count": 1})
    partition = _partition_on(client)

    with pytest.raises(pymilvus.MilvusException):
        await partition.delete(record_uuids=[uuid4(), uuid4()])


def _too_large() -> grpc.aio.AioRpcError:
    """The error Milvus's proxy answers a request over its receive limit with."""
    return grpc.aio.AioRpcError(
        grpc.StatusCode.RESOURCE_EXHAUSTED,
        details="grpc: received message larger than max",
    )


@pytest.mark.asyncio
async def test_a_batch_refused_as_too_large_is_halved_until_it_fits():
    """Milvus's proxy refuses a request over its gRPC receive limit with
    RESOURCE_EXHAUSTED."""
    upserted: list[list[str]] = []

    def refuse_more_than_two(
        *, collection_name: str, data: list[dict], timeout: int
    ) -> None:
        if len(data) > 2:
            raise _too_large()
        upserted.append([entity["id"] for entity in data])

    client = MagicMock(spec=AsyncMilvusClient)
    client.upsert = AsyncMock(side_effect=refuse_more_than_two)
    partition = _partition_on(client)
    records = [_make_record(vector=_normalize([1.0, 0.0, 0.0])) for _ in range(5)]

    await partition.upsert(records=records)

    assert sorted(len(batch) for batch in upserted) == [1, 2, 2]
    assert {primary_id for batch in upserted for primary_id in batch} == {
        partition._primary_id(record.uuid) for record in records
    }


@pytest.mark.asyncio
async def test_an_upsert_with_an_entity_refused_alone_raises():
    """Halving stops at a single entity: an entity refused on its own fails the
    upsert, which does not report it accepted."""
    refused = _make_record(vector=[0.0, 0.0, 1.0])

    def refuse_the_refused_entity(
        *, collection_name: str, data: list[dict], timeout: int
    ) -> None:
        if any(entity["vector"] == refused.vector for entity in data):
            raise _too_large()

    client = MagicMock(spec=AsyncMilvusClient)
    client.upsert = AsyncMock(side_effect=refuse_the_refused_entity)
    partition = _partition_on(client)
    records = [_make_record(vector=_normalize([1.0, 0.0, 0.0])) for _ in range(4)]
    records.insert(3, refused)

    with pytest.raises(grpc.aio.AioRpcError) as raised:
        await partition.upsert(records=records)
    assert raised.value.code() == grpc.StatusCode.RESOURCE_EXHAUSTED


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "error",
    [
        grpc.aio.AioRpcError(grpc.StatusCode.DEADLINE_EXCEEDED),
        pymilvus.MilvusException(code=1100, message="invalid parameter"),
    ],
    ids=["timeout", "refused_otherwise"],
)
async def test_an_upsert_that_fails_otherwise_is_not_sent_again(error: Exception):
    """A timed-out request may still be applied, so it is not resent."""
    client = MagicMock(spec=AsyncMilvusClient)
    client.upsert = AsyncMock(side_effect=error)
    partition = _partition_on(client)
    records = [_make_record(vector=_normalize([1.0, 0.0, 0.0])) for _ in range(4)]

    with pytest.raises(type(error)):
        await partition.upsert(records=records)

    client.upsert.assert_awaited_once()


@pytest.mark.asyncio
async def test_a_purge_round_milvus_does_not_accept_in_full_raises():
    """A purge round raises when Milvus accepts the delete of fewer primary
    keys than the round listed."""
    client = MagicMock(spec=AsyncMilvusClient)
    client.prepare_index_params = AsyncMilvusClient.prepare_index_params
    # The native collection exists with its index, and holds two entities of
    # the purged incarnation.
    client.has_collection = AsyncMock(return_value=True)
    client.list_indexes = AsyncMock(return_value=["vector"])
    client.load_collection = AsyncMock(return_value=None)
    client.query = AsyncMock(return_value=[{"id": "listed_a"}, {"id": "listed_b"}])
    client.delete = AsyncMock(return_value={"delete_count": 1})
    store = await _started_store(client, store_schema=_schema(indexed_properties={}))

    with pytest.raises(pymilvus.MilvusException):
        await store.purge_partition_incarnation(uuid4())


class TestPartitionIsolation:
    @pytest.mark.asyncio
    async def test_same_uuid_can_exist_in_different_partitions(self, store):
        """The same UUID in two partitions is two records, written and deleted apart."""
        partition_a = await _open(store)
        partition_b = await _open(store)

        record_uuid = uuid4()
        v1 = _normalize([1.0, 0.0, 0.0])
        await partition_a.upsert(
            records=[Record(uuid=record_uuid, vector=v1, properties={"name": "a"})]
        )
        await _settle(partition_a)
        await partition_b.upsert(
            records=[Record(uuid=record_uuid, vector=v1, properties={"name": "b"})]
        )
        await _settle(partition_b)

        stored_a = await _stored(partition_a, [record_uuid])
        stored_b = await _stored(partition_b, [record_uuid])

        # `name` is declared, so each value is in its typed field.
        assert stored_a[record_uuid]["_p_name"] == "a"
        assert stored_b[record_uuid]["_p_name"] == "b"

        await partition_a.delete(record_uuids=[record_uuid])
        await _settle(partition_b)

        assert await _stored(partition_a, [record_uuid]) == {}
        assert set(await _stored(partition_b, [record_uuid])) == {record_uuid}
        [kept] = await partition_b.query(query_vectors=[v1], limit=10)
        assert [match.record_uuid for match in kept.matches] == [record_uuid]

        await _purge(store, partition_a._incarnation)
        await _purge(store, partition_b._incarnation)

    @pytest.mark.asyncio
    async def test_a_query_returns_only_its_own_partitions_records(self, store):
        """Partitions of one store answer with their own records alone,
        filtered or not, though the other's records match too."""
        # A store of its own, declaring `color` beside the shared keys.
        isolated = await _started_store(
            store._client,
            vector_store_name="query_isolation",
            store_schema=_schema(
                indexed_properties={**INDEXED_PROPERTIES, "color": str}
            ),
        )
        vector = _normalize([1.0, 0.0, 0.0])
        own_records = []
        for _ in range(2):
            partition = await _open(isolated)
            records = [
                _make_record(
                    vector=vector, properties={"name": "shared", "color": "red"}
                )
                for _ in range(2)
            ]
            await partition.upsert(records=records)
            own_records.append((partition, {record.uuid for record in records}))

        for partition, own in own_records:
            await _settle(partition)
            for property_filter in (
                None,
                Comparison(field="name", op="=", value="shared"),
                Comparison(field="color", op="=", value="red"),
                Not(expr=IsNull(field="name")),
            ):
                [result] = await partition.query(
                    query_vectors=[vector], limit=10, property_filter=property_filter
                )
                assert {match.record_uuid for match in result.matches} == own, (
                    property_filter
                )

        for partition, _ in own_records:
            await _purge(isolated, partition._incarnation)


class TestPurgeBatches:
    @pytest.mark.asyncio
    async def test_a_purge_round_reclaims_at_most_one_batch(self, store):
        store = MilvusVectorStore(_params(store._client, purge_batch_size=2))
        partition = await _open(store)
        await partition.upsert(
            records=[
                _make_record(vector=_normalize([1.0, float(i), 0.0])) for i in range(5)
            ]
        )
        await _settle(partition)
        incarnation = partition._incarnation

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
        while await store.purge_partition_incarnation(incarnation):
            await _settle(partition)
            left_after_each_round.append(await left_of_the_incarnation())
        # Batches of 2, until a round finds nothing.
        assert left_after_each_round == [3, 1, 0]


class TestPurge:
    @pytest.mark.asyncio
    async def test_a_purge_on_a_dropped_native_collection_finds_nothing(self, store):
        """An incarnation whose native collection is gone holds nothing: its
        purge raises nothing and finds nothing."""
        # A name of its own, so dropping its native collection spares the others.
        dropped = await _started_store(
            store._client, vector_store_name="dropped_native"
        )
        partition = await _open(dropped)
        await partition.upsert(
            records=[_make_record(vector=_normalize([1.0, 0.0, 0.0]))]
        )
        await store._client.drop_collection(dropped._collection_name)

        assert (
            await dropped.purge_partition_incarnation(partition._incarnation) is False
        )

    @pytest.mark.asyncio
    async def test_purgers_on_two_stores_reclaim_the_deleted_partitions_alone(
        self, store, other_store
    ):
        """Purgers on two stores, purging the deleted partitions' incarnations
        at once in small batches, reclaim every one's records and leave the
        live partition's."""
        handles = []
        for _ in range(5):
            handle = await _open(store)
            await handle.upsert(
                records=[
                    _make_record(vector=_normalize([1.0, 0.1 * index, 0.0]))
                    for index in range(7)
                ]
            )
            handles.append(handle)
        live = handles.pop(0)
        live_uuids = await _incarnation_uuids(
            store._client, live._collection_name, live._incarnation
        )
        await _settle(live)
        purgers = [
            MilvusVectorStore(_params(purging._client, purge_batch_size=3))
            for purging in (store, other_store)
        ]

        await asyncio.gather(
            *(
                _purge(purger, handle._incarnation)
                for purger in purgers
                for handle in handles
            )
        )

        for handle in handles:
            assert (
                await _incarnation_uuids(
                    store._client, handle._collection_name, handle._incarnation
                )
                == []
            )
        assert sorted(
            await _incarnation_uuids(
                store._client, live._collection_name, live._incarnation
            )
        ) == sorted(live_uuids)
        assert len(live_uuids) == 7
        assert [
            await purger.purge_partition_incarnation(handle._incarnation)
            for purger in purgers
            for handle in handles
        ] == [False] * (len(purgers) * len(handles))
        await _purge(store, live._incarnation)


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
