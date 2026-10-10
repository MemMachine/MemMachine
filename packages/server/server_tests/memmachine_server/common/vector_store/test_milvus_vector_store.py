"""Tests for MilvusVectorStore."""

# ruff: noqa: E402

import asyncio
import math
import random
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta, timezone
from typing import Any, override
from unittest.mock import AsyncMock, MagicMock
from uuid import UUID, uuid4

import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine

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
from memmachine_server.common.properties_json import decode_properties
from memmachine_server.common.vector_store.data_types import (
    PartitionSchema,
    Record,
    VectorStoreAttemptsExhaustedError,
    VectorStorePartitionAlreadyExistsError,
    VectorStorePartitionDeletedError,
    VectorStorePartitionHandleStaleError,
    VectorStorePartitionPendingError,
    VectorStorePartitionSchemaMismatchError,
)
from memmachine_server.common.vector_store.milvus_vector_store import (
    MilvusVectorStore,
    MilvusVectorStoreParams,
    MilvusVectorStorePartition,
)
from memmachine_server.common.vector_store.partition_registry import (
    Registration,
    VectorStorePartitionRegistry,
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
# Settings other than their defaults, so a default in their place shows.
REQUEST_TIMEOUT_SECONDS = 17
MAX_VARCHAR_LENGTH = 1024
PURGE_BATCH_SIZE = 5000


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
        "similarity_metric": SimilarityMetric.COSINE,
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
    await _settle_native(partition._client, partition._collection_name)


async def _settle_native(client: AsyncMilvusClient, collection_name: str) -> None:
    """Return once the store's reads of a native collection reflect every write made so far."""
    await client.query(
        collection_name=collection_name,
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


async def _drain(store: MilvusVectorStore) -> None:
    """Run purge rounds until none is due, failing on a purge that never ends."""

    async def drain() -> None:
        while await store.purge_deleted_partitions():
            pass

    await asyncio.wait_for(drain(), 60)


def _with_purge_batch_size(
    store: MilvusVectorStore, purge_batch_size: int
) -> MilvusVectorStore:
    """A store over the same client, registry and native collection, purging in batches of this size."""
    return MilvusVectorStore(
        MilvusVectorStoreParams(
            client=store._client,
            partition_registry=store._partition_registry,
            vector_store_name=store.vector_store_name,
            vector_dimensions=store.vector_dimensions,
            similarity_metric=store.similarity_metric,
            indexed_properties=dict(store.indexed_properties),
            request_timeout_seconds=REQUEST_TIMEOUT_SECONDS,
            max_varchar_length=MAX_VARCHAR_LENGTH,
            purge_batch_size=purge_batch_size,
        )
    )


# Properties of every type, declared and not, for the tests against a model.
_MODEL_DECLARED: dict[str, PropertyType] = {
    "d_bool": bool,
    "d_int": int,
    "d_float": float,
    "d_str": str,
    "d_datetime": datetime,
}
_MODEL_PROPERTY_TYPES: dict[str, PropertyType] = {
    **_MODEL_DECLARED,
    "u_bool": bool,
    "u_int": int,
    "u_float": float,
    "u_str": str,
    "u_datetime": datetime,
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
        for key, value_type in _MODEL_PROPERTY_TYPES.items()
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
    key = rng.choice(list(_MODEL_PROPERTY_TYPES))
    value_type = _MODEL_PROPERTY_TYPES[key]
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


async def _started_store(
    client: AsyncMilvusClient, registry_engine: AsyncEngine, **overrides
) -> MilvusVectorStore:
    """A started store with its own registry object over the registry database."""
    vector_store = MilvusVectorStore(
        await _params(client, registry_engine, **overrides)
    )
    await vector_store.startup()
    return vector_store


@pytest.fixture
def registry_url(tmp_path) -> str:
    """The registry database every store of a test shares."""
    return f"sqlite+aiosqlite:///{tmp_path / 'registry.db'}"


@pytest_asyncio.fixture
async def store(milvus_client, registry_url):
    registry_engine = create_async_engine(registry_url)
    vector_store = await _started_store(milvus_client, registry_engine)
    yield vector_store
    await vector_store.shutdown()
    await registry_engine.dispose()


@pytest_asyncio.fixture
async def other_store(milvus_container, registry_url):
    """A store as another process runs it beside `store`: its own Milvus
    client and registry connections, over the same Milvus and registry
    database."""
    client = AsyncMilvusClient(uri=milvus_container.get_connection_url())
    registry_engine = create_async_engine(registry_url)
    vector_store = await _started_store(client, registry_engine)
    yield vector_store
    await vector_store.shutdown()
    await registry_engine.dispose()
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
    await _drain(timed)

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
            partition_registry=MagicMock(spec=VectorStorePartitionRegistry),
            vector_store_name=VECTOR_STORE_NAME,
            vector_dimensions=VECTOR_DIM,
            indexed_properties=INDEXED_PROPERTIES,
            **{setting: 0},
        )


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
        }
        assert set(fields) == {
            "id",
            "record_uuid",
            "partition_key",
            "vector",
            "properties",
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

        await store.delete_partition("schema")

    @pytest.mark.asyncio
    async def test_the_longest_declared_key_names_a_field_milvus_accepts(self, store):
        """A declared property's field is named by its key under a prefix,
        within the server's proxy.maxNameLength."""
        key = "k" * 32  # the longest property key
        # A name of its own, so its native collection declares the key.
        longest = await _started_store(
            store._client,
            store._partition_registry._engine,
            vector_store_name="longest_key",
            indexed_properties={key: datetime},
        )
        await longest.create_partition("longest_key")
        partition = await longest.get_partition("longest_key")
        assert partition is not None
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

        await longest.delete_partition("longest_key")

    @pytest.mark.asyncio
    async def test_unsupported_metric_raises(self, store):
        with pytest.raises(ValueError, match="Milvus only supports"):
            MilvusVectorStore(
                await _params(
                    store._client,
                    store._partition_registry._engine,
                    vector_store_name="bad_metric",
                    similarity_metric=SimilarityMetric.MANHATTAN,
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
            await _params(
                store._client,
                store._partition_registry._engine,
                vector_store_name="raced_preparation",
                indexed_properties={"name": str},
            )
        )
        second = MilvusVectorStore(
            await _params(
                other_store._client,
                other_store._partition_registry._engine,
                vector_store_name="raced_preparation",
                indexed_properties={"name": str},
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
                await second.create_partition("second")
                partition = await second.get_partition("second")
                assert partition is not None
                await _require_usable(partition)
            finally:
                second_done.set()

        await asyncio.gather(first.startup(), start_second())

        await first.create_partition("first")
        partition = await first.get_partition("first")
        assert partition is not None
        await _require_usable(partition)
        assert set(await store._client.list_indexes(first._collection_name)) == {
            "vector",
            "_p_name",
        }


class TestUpsertAndQuery:
    @pytest.mark.asyncio
    async def test_an_upsert_over_the_request_size_limit_stores_every_record(
        self, collection
    ):
        """Milvus refuses a request over proxy.grpc.serverMaxRecvSize (64 MiB
        unless configured), so the batch is halved until it fits."""
        # Each record's undeclared properties fit common.JSONMaxLength (64 KiB
        # unless configured); the batch is over 64 MiB.
        text = "x" * 60_000
        records = [
            _make_record(vector=_normalize([1.0, 0.0, 0.0]), properties={"text": text})
            for _ in range(1_200)
        ]

        await collection.upsert(records=records)

        assert sorted(
            await _incarnation_uuids(
                collection._client,
                collection._collection_name,
                collection._incarnation,
            )
        ) == sorted(record.uuid for record in records)

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
        [old] = await collection.query(
            query_vectors=[v1],
            limit=10,
            property_filter=Comparison(field="name", op="=", value="old"),
        )
        assert old.matches == []

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
    async def test_a_declared_datetime_is_stored_as_its_instant_in_utc(
        self, collection
    ):
        written = datetime(
            2024, 6, 15, 17, 30, tzinfo=timezone(timedelta(hours=5, minutes=30))
        )
        record = _make_record(
            vector=_normalize([1.0, 0.0, 0.0]), properties={"created_at": written}
        )
        await collection.upsert(records=[record])

        stored = (await _stored(collection, [record.uuid]))[record.uuid]
        kept = datetime.fromisoformat(stored["_p_created_at"])
        assert kept == written
        assert kept.utcoffset() == timedelta(0)

    @pytest.mark.asyncio
    async def test_a_declared_datetime_with_a_seconds_offset_is_stored_and_matched(
        self, collection
    ):
        """A datetime whose offset has a seconds component, as local mean time
        does, is stored with that offset and matched by its instant."""
        written = datetime(
            2024, 6, 15, 12, 0, tzinfo=timezone(timedelta(minutes=19, seconds=32))
        )
        record = _make_record(
            vector=_normalize([1.0, 0.0, 0.0]), properties={"created_at": written}
        )
        await collection.upsert(records=[record])
        await _settle(collection)

        stored = (await _stored(collection, [record.uuid]))[record.uuid]
        assert datetime.fromisoformat(stored["_p_created_at"]) == written
        [result] = await collection.query(
            query_vectors=[record.vector],
            limit=10,
            property_filter=Comparison(
                field="created_at", op="=", value=written.astimezone(UTC)
            ),
        )
        assert [match.record_uuid for match in result.matches] == [record.uuid]

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
    async def test_ints_compare_with_float_properties_and_floats_not_with_int_ones(
        self, collection
    ):
        """An int filter value compares with a declared float property by
        value; a float filter value matches no declared int property."""
        r1, _, r3, v1 = await self._setup(collection)

        assert await self._query(collection, v1, "score", ">=", 8) == {
            r1.uuid,
            r3.uuid,
        }
        assert await self._query(collection, v1, "age", "=", 30.0) == set()
        assert await self._query(collection, v1, "age", ">", 29.5) == set()

    @pytest.mark.asyncio
    async def test_an_undeclared_property_matches_only_values_of_a_comparable_type(
        self, collection
    ):
        """A string equal to an undeclared datetime's stored text matches
        nothing, and its complement keeps the record; an int still compares
        with an undeclared float."""
        seen = datetime(2024, 6, 15, 12, tzinfo=UTC)
        record = _make_record(
            vector=_normalize([1.0, 0.0, 0.0]), properties={"seen": seen, "rank": 2.0}
        )
        await collection.upsert(records=[record])
        await _settle(collection)

        assert (
            await self._query(collection, record.vector, "seen", "=", seen.isoformat())
            == set()
        )
        assert await self._query(
            collection, record.vector, "seen", "!=", seen.isoformat()
        ) == {record.uuid}
        assert await self._query(collection, record.vector, "rank", "=", 2) == {
            record.uuid
        }

    @pytest.mark.asyncio
    async def test_a_character_outside_the_basic_multilingual_plane_matches(
        self, collection
    ):
        """A string filter value with a character outside the Basic
        Multilingual Plane matches, on a declared property and an undeclared
        one."""
        text = "café 😀"
        record = _make_record(
            vector=_normalize([1.0, 0.0, 0.0]),
            properties={"name": text, "nickname": text},
        )
        await collection.upsert(records=[record])
        await _settle(collection)

        for field_name in ("name", "nickname"):
            assert await self._query(
                collection, record.vector, field_name, "=", text
            ) == {record.uuid}

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


class TestFilterModel:
    @pytest.mark.asyncio
    async def test_random_filters_agree_with_the_model(self, store):
        """Seeded filter trees over properties of every type, declared and not,
        datetimes at varied offsets and missing values included, select the
        records the in-memory evaluator selects."""
        rng = random.Random(1736)
        # A store of its own, declaring the model's properties.
        modeled = await _started_store(
            store._client,
            store._partition_registry._engine,
            vector_store_name="filter_model",
            indexed_properties=_MODEL_DECLARED,
        )
        await modeled.create_partition("model")
        partition = await modeled.get_partition("model")
        assert partition is not None
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

        await modeled.delete_partition("model")


@dataclass
class _ModelPartitions:
    """Partitions of one store, with the records a model says each holds."""

    store: MilvusVectorStore
    handles: dict[str, MilvusVectorStorePartition]
    records: dict[str, dict[UUID, Record]]
    dead_incarnations: list[UUID] = field(default_factory=list)

    async def upsert(self, partition_key: str, batch: list[Record]) -> list[Record]:
        """Upsert a batch, and return the records it replaced."""
        await self.handles[partition_key].upsert(records=batch)
        records = self.records[partition_key]
        replaced = [records[record.uuid] for record in batch if record.uuid in records]
        records.update({record.uuid: record for record in batch})
        return replaced

    async def delete(self, partition_key: str, record_uuids: list[UUID]) -> None:
        await self.handles[partition_key].delete(record_uuids=record_uuids)
        for record_uuid in record_uuids:
            self.records[partition_key].pop(record_uuid, None)

    async def recreate(self, partition_key: str) -> None:
        self.dead_incarnations.append(self.handles[partition_key]._incarnation)
        await self.store.delete_partition(partition_key)
        await self.store.create_partition(partition_key)
        handle = await self.store.get_partition(partition_key)
        assert handle is not None
        self.handles[partition_key] = handle
        self.records[partition_key] = {}

    async def purge(self) -> None:
        """Drain the purge, and check that no deleted partition's record remains."""
        await _drain(self.store)
        for incarnation in self.dead_incarnations:
            assert (
                await _incarnation_uuids(
                    self.store._client, self.store._collection_name, incarnation
                )
                == []
            ), incarnation

    async def check(self, rng: random.Random) -> None:
        """Check what each partition holds, and what its settled queries select."""
        for partition_key, handle in self.handles.items():
            records = self.records[partition_key]
            stored = await _incarnation_uuids(
                self.store._client, handle._collection_name, handle._incarnation
            )
            assert sorted(stored) == sorted(records), partition_key
            await _settle(handle)
            assert await _model_matches(handle, None) == set(records), partition_key
            for _ in range(3):
                property_filter = _model_filter(rng)
                assert await _model_matches(handle, property_filter) == {
                    record.uuid
                    for record in records.values()
                    if evaluate_filter(property_filter, record.properties)
                }, (partition_key, property_filter)

    async def check_replaced(self, partition_key: str, replaced: list[Record]) -> None:
        """Check that a replaced record's old values select it no more."""
        for old in replaced:
            current = self.records[partition_key][old.uuid]
            for key, value in old.properties.items():
                old_value = Comparison(field=key, op="=", value=value)
                if not evaluate_filter(old_value, current.properties):
                    assert old.uuid not in await _model_matches(
                        self.handles[partition_key], old_value
                    ), old_value


class TestOperationModel:
    @pytest.mark.asyncio
    async def test_a_random_sequence_of_operations_agrees_with_a_model(self, store):
        """Seeded upserts, new and replacing, deletes of present and absent
        records, deletion and recreation of a partition, and purge drains, on
        two partitions of one store: after each step, each partition holds
        the model's records, its settled queries select what the model
        selects, and a replaced record's old values select nothing."""
        rng = random.Random(1736)
        # A store of its own, declaring the model's properties.
        modeled = await _started_store(
            store._client,
            store._partition_registry._engine,
            vector_store_name="operation_model",
            indexed_properties=_MODEL_DECLARED,
        )
        partition_keys = ("alpha", "beta")
        # Few UUIDs, so each recurs in both partitions and across their lives.
        pool = sorted(uuid4() for _ in range(12))
        handles = {}
        for partition_key in partition_keys:
            await modeled.create_partition(partition_key)
            handle = await modeled.get_partition(partition_key)
            assert handle is not None
            handles[partition_key] = handle
        model = _ModelPartitions(
            store=modeled,
            handles=handles,
            records={partition_key: {} for partition_key in partition_keys},
        )

        for step in range(60):
            partition_key = rng.choice(partition_keys)
            replaced: list[Record] = []
            action = rng.choices(
                ("upsert", "delete", "recreate", "purge"), weights=(8, 3, 1, 1)
            )[0]
            match action:
                case "upsert":
                    replaced = await model.upsert(
                        partition_key,
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
                        partition_key, rng.sample(pool, k=rng.randint(1, 4))
                    )
                case "recreate":
                    await model.recreate(partition_key)
                case _:
                    await model.purge()
            try:
                await model.check(rng)
                await model.check_replaced(partition_key, replaced)
            except AssertionError as error:
                raise AssertionError(
                    f"after step {step}, {action} {partition_key}"
                ) from error

        for partition_key in partition_keys:
            await modeled.delete_partition(partition_key)


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
            await _params(
                store._client,
                store._partition_registry._engine,
                vector_store_name=f"scores_{metric.value}",
                similarity_metric=metric,
            )
        )
        await scored.startup()
        await scored.delete_partition("scores")
        await scored.create_partition("scores")
        partition = await scored.get_partition("scores")
        assert partition is not None
        record = _make_record(vector=[1.2, 1.6, 0.0])
        await partition.upsert(records=[record])
        await _settle(partition)

        [result] = await partition.query(query_vectors=[[1.0, 0.0, 0.0]], limit=1)
        assert result.matches[0].score == pytest.approx(expected, abs=1e-3)
        await scored.delete_partition("scores")

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
            store._partition_registry._engine,
            vector_store_name=f"threshold_{metric.value}",
            similarity_metric=metric,
        )
        await scored.create_partition("threshold")
        partition = await scored.get_partition("threshold")
        assert partition is not None
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
        await scored.delete_partition("threshold")

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
            store._partition_registry._engine,
            vector_store_name=f"edge_{metric.value}",
            similarity_metric=metric,
        )
        await scored.create_partition("edge")
        partition = await scored.get_partition("edge")
        assert partition is not None
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
        await scored.delete_partition("edge")


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
        bounded = await _started_store(
            store._client,
            store._partition_registry._engine,
            vector_store_name="bounded",
        )

        await bounded.create_partition("bounded")
        partition = await bounded.get_partition("bounded")
        assert partition is not None
        record = _make_record(vector=_normalize([1.0, 0.0, 0.0]))
        await partition.upsert(records=[record])
        await _settle(partition)
        await partition.query(query_vectors=[record.vector], limit=1)
        await bounded.delete_partition("bounded")
        await _drain(bounded)

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


def _partition_on(client: AsyncMilvusClient) -> MilvusVectorStorePartition:
    """A handle on a given client, bound to a live incarnation."""
    return MilvusVectorStorePartition(
        client=client,
        collection_name=f"sys_{VECTOR_STORE_NAME}",
        vector_store_name=VECTOR_STORE_NAME,
        registration=_CurrentRegistration(
            partition_key=NAME,
            schema=PartitionSchema(
                vector_dimensions=VECTOR_DIM,
                similarity_metric=SimilarityMetric.COSINE,
                indexed_properties={},
            ),
            incarnation=uuid4(),
        ),
        vector_dimensions=VECTOR_DIM,
        similarity_metric=SimilarityMetric.COSINE,
        indexed_properties={},
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
async def test_a_purge_round_milvus_does_not_accept_in_full_raises(registry_url):
    """A purge round raises when Milvus accepts the delete of fewer primary
    keys than the round listed."""
    client = MagicMock(spec=AsyncMilvusClient)
    client.prepare_index_params = AsyncMilvusClient.prepare_index_params
    # The native collection exists with its index, and holds two entities of
    # the deleted partition.
    client.has_collection = AsyncMock(return_value=True)
    client.list_indexes = AsyncMock(return_value=["vector"])
    client.load_collection = AsyncMock(return_value=None)
    client.query = AsyncMock(return_value=[{"id": "listed_a"}, {"id": "listed_b"}])
    client.delete = AsyncMock(return_value={"delete_count": 1})
    registry_engine = create_async_engine(registry_url)
    try:
        store = await _started_store(client, registry_engine, indexed_properties={})
        await store.create_partition(NAME)
        await store.delete_partition(NAME)

        with pytest.raises(pymilvus.MilvusException):
            await store.purge_deleted_partitions()
    finally:
        await registry_engine.dispose()


class TestPartitionIsolation:
    @pytest.mark.asyncio
    async def test_same_uuid_can_exist_in_different_partitions(self, store):
        """The same UUID in two partitions is two records, written and deleted apart."""
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

        await coll_a.delete(record_uuids=[record_uuid])
        await _settle(coll_b)

        assert await _stored(coll_a, [record_uuid]) == {}
        assert set(await _stored(coll_b, [record_uuid])) == {record_uuid}
        [kept] = await coll_b.query(query_vectors=[v1], limit=10)
        assert [match.record_uuid for match in kept.matches] == [record_uuid]

        await store.delete_partition("tenant_a")
        await store.delete_partition("tenant_b")

    @pytest.mark.asyncio
    async def test_a_query_returns_only_its_own_partitions_records(self, store):
        """Partitions of one store answer with their own records alone,
        filtered or not, though the other's records match too."""
        vector = _normalize([1.0, 0.0, 0.0])
        own_records = {}
        for partition_key in ("query_a", "query_b"):
            await store.create_partition(partition_key)
            partition = await store.get_partition(partition_key)
            assert partition is not None
            records = [
                _make_record(
                    vector=vector, properties={"name": "shared", "color": "red"}
                )
                for _ in range(2)
            ]
            await partition.upsert(records=records)
            own_records[partition_key] = (
                partition,
                {record.uuid for record in records},
            )

        for partition, own in own_records.values():
            await _settle(partition)
            # `name` is declared and `color` is not.
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

        for partition_key in own_records:
            await store.delete_partition(partition_key)


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


class TestPurge:
    @pytest.mark.asyncio
    async def test_a_purge_round_on_a_dropped_native_collection_retires_the_tombstone(
        self, store
    ):
        """A deleted partition whose native collection is gone holds nothing:
        its purge round raises nothing and removes the tombstone."""
        # A name of its own, so dropping its native collection spares the others.
        dropped = await _started_store(
            store._client,
            store._partition_registry._engine,
            vector_store_name="dropped_native",
        )
        await dropped.create_partition("dropped")
        partition = await dropped.get_partition("dropped")
        assert partition is not None
        await partition.upsert(
            records=[_make_record(vector=_normalize([1.0, 0.0, 0.0]))]
        )
        await dropped.delete_partition("dropped")
        await store._client.drop_collection(dropped._collection_name)

        await _drain(dropped)
        assert await dropped.purge_deleted_partitions() is False

    @pytest.mark.asyncio
    async def test_purgers_on_two_stores_reclaim_the_deleted_partitions_alone(
        self, store, other_store
    ):
        """Purgers on two stores sharing a registry, draining at once in small
        batches, reclaim every deleted partition's records and leave the live
        partition's."""
        handles = {}
        for partition_key in ("live", "dead_0", "dead_1", "dead_2", "dead_3"):
            await store.create_partition(partition_key)
            handle = await store.get_partition(partition_key)
            assert handle is not None
            await handle.upsert(
                records=[
                    _make_record(vector=_normalize([1.0, 0.1 * index, 0.0]))
                    for index in range(7)
                ]
            )
            handles[partition_key] = handle
        live = handles.pop("live")
        live_uuids = await _incarnation_uuids(
            store._client, live._collection_name, live._incarnation
        )
        await _settle(live)
        for partition_key in handles:
            await store.delete_partition(partition_key)
        purgers = [
            _with_purge_batch_size(purging, 3) for purging in (store, other_store)
        ]

        await asyncio.gather(*(_drain(purger) for purger in purgers))

        for handle in handles.values():
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
        assert [await purger.purge_deleted_partitions() for purger in purgers] == [
            False,
            False,
        ]
        await store.delete_partition("live")


_CHURN_VECTOR_STORE_NAME = "churn"
_CHURN_PARTITION_KEYS = ("churn_0", "churn_1", "churn_2")
_CHURN_DECLARED: dict[str, PropertyType] = {"owner": str, "tag": str}
_CHURN_TAGS = ("red", "green", "blue")
# The outcomes the store documents for an operation that loses a race.
_DOMAIN_ERRORS = (
    VectorStoreAttemptsExhaustedError,
    VectorStorePartitionAlreadyExistsError,
    VectorStorePartitionDeletedError,
    VectorStorePartitionHandleStaleError,
    VectorStorePartitionPendingError,
)


@dataclass(frozen=True)
class _ChurnStep:
    """A worker's step: the partition, the action, and the records it writes or deletes."""

    partition_key: str
    action: str
    records: list[Record]


@dataclass
class _ChurnLedger:
    """What the churn's operations did, by the incarnation each reached."""

    # The records each incarnation holds once every write has returned: those
    # an upsert that returned wrote, less those a delete that returned removed.
    held: dict[UUID, dict[UUID, Record]] = field(default_factory=dict)
    # The record UUIDs any upsert sent to each incarnation.
    sent: dict[UUID, set[UUID]] = field(default_factory=dict)
    # The records of upserts to an incarnation deleted before they returned.
    # Such a write can land after the incarnation's purge, which only the
    # tombstone retention prevents.
    raced_deletion: set[tuple[UUID, UUID]] = field(default_factory=set)


def _churn_script(worker: int) -> tuple[str, set[UUID], list[_ChurnStep]]:
    """A worker's owner name, its own record UUIDs, and its seeded steps."""
    rng = random.Random(worker)
    owner = f"worker_{worker}"
    own = sorted(uuid4() for _ in range(6))
    steps = []
    for _ in range(40):
        partition_key = rng.choice(_CHURN_PARTITION_KEYS)
        action = rng.choices(
            ("upsert", "delete", "query", "recreate"), weights=(4, 2, 3, 1)
        )[0]
        records = [
            Record(
                uuid=record_uuid,
                vector=_normalize([1.0, rng.random(), rng.random()]),
                properties={
                    "owner": owner,
                    "tag": rng.choice(_CHURN_TAGS),
                    "version": rng.randrange(1 << 30),
                },
            )
            for record_uuid in rng.sample(own, k=rng.randint(1, 3))
        ]
        steps.append(
            _ChurnStep(partition_key=partition_key, action=action, records=records)
        )
    return owner, set(own), steps


async def _churn_on(
    handle: MilvusVectorStorePartition,
    step: _ChurnStep,
    owner: str,
    own: set[UUID],
    ledger: _ChurnLedger,
) -> None:
    """Run an upsert, delete or query step through a handle, and record its outcome."""
    incarnation = handle._incarnation
    record_uuids = [record.uuid for record in step.records]
    match step.action:
        case "upsert":
            ledger.sent.setdefault(incarnation, set()).update(record_uuids)
            try:
                await handle.upsert(records=step.records)
            except VectorStorePartitionHandleStaleError:
                ledger.raced_deletion.update(
                    (incarnation, record_uuid) for record_uuid in record_uuids
                )
                raise
            ledger.held.setdefault(incarnation, {}).update(
                {record.uuid: record for record in step.records}
            )
        case "delete":
            await handle.delete(record_uuids=record_uuids)
            for record_uuid in record_uuids:
                ledger.held.get(incarnation, {}).pop(record_uuid, None)
        case _:
            matched = await _model_matches(
                handle, Comparison(field="owner", op="=", value=owner)
            )
            # Only records this owner sent to this incarnation.
            assert matched <= ledger.sent.get(incarnation, set()) & own


async def _churn_worker(
    store: MilvusVectorStore,
    worker: int,
    ledger: _ChurnLedger,
    deleted: asyncio.Event,
) -> None:
    owner, own, steps = _churn_script(worker)
    handles: dict[str, MilvusVectorStorePartition] = {}
    for step in steps:
        try:
            if step.action == "recreate":
                handles.pop(step.partition_key, None)
                await store.delete_partition(step.partition_key)
                deleted.set()
                await store.create_partition(step.partition_key)
                continue
            if step.partition_key not in handles:
                handles[step.partition_key] = await store.open_or_create_partition(
                    step.partition_key
                )
            await _churn_on(handles[step.partition_key], step, owner, own, ledger)
        except VectorStorePartitionHandleStaleError:
            handles.pop(step.partition_key, None)
        except _DOMAIN_ERRORS:
            pass


async def _churn(stores: tuple[MilvusVectorStore, ...], ledger: _ChurnLedger) -> None:
    """Run six workers across the stores, each store's purger draining after
    every partition deletion, until the workers finish."""
    wakes = [asyncio.Event() for _ in stores]
    deleted = asyncio.Event()
    stopping = asyncio.Event()

    async def purge(purging: MilvusVectorStore, wake: asyncio.Event) -> None:
        while not stopping.is_set():
            await wake.wait()
            wake.clear()
            await _drain(purging)

    async def wake_purgers() -> None:
        while not stopping.is_set():
            await deleted.wait()
            deleted.clear()
            for wake in wakes:
                wake.set()

    background = [
        asyncio.create_task(purge(purging, wake))
        for purging, wake in zip(stores, wakes, strict=True)
    ]
    background.append(asyncio.create_task(wake_purgers()))
    try:
        await asyncio.gather(
            *(
                _churn_worker(stores[worker % len(stores)], worker, ledger, deleted)
                for worker in range(6)
            )
        )
    finally:
        stopping.set()
        deleted.set()
        for wake in wakes:
            wake.set()
        await asyncio.gather(*background)


async def _check_live_partition(
    handle: MilvusVectorStorePartition, held: dict[UUID, Record]
) -> None:
    """Check that a partition holds exactly these records, with their values."""
    stored = await _incarnation_uuids(
        handle._client, handle._collection_name, handle._incarnation
    )
    assert sorted(stored) == sorted(held)
    rows = await _stored(handle, list(held)) if held else {}
    assert {
        record_uuid: decode_properties(row["properties"])["version"]
        for record_uuid, row in rows.items()
    } == {
        record_uuid: record.properties["version"]
        for record_uuid, record in held.items()
    }
    await _settle(handle)
    owners = {record.properties["owner"] for record in held.values()}
    for key, value in [("owner", owner) for owner in owners] + [
        ("tag", tag) for tag in _CHURN_TAGS
    ]:
        assert await _model_matches(
            handle, Comparison(field=key, op="=", value=value)
        ) == {
            record_uuid
            for record_uuid, record in held.items()
            if record.properties[key] == value
        }, (key, value)


def _settle_each_purge_round(store: MilvusVectorStore, monkeypatch) -> None:
    """Begin each of the store's purge rounds once its reads reflect every
    write that returned before the round.

    A tombstone retention longer than the store's read delay gives a round
    this; the tests' tombstones come due at once.
    """
    purge_round = store._purge_round

    async def settled_purge_round(incarnation: UUID) -> bool:
        await _settle_native(store._client, store._collection_name)
        return await purge_round(incarnation)

    monkeypatch.setattr(store, "_purge_round", settled_purge_round)


class TestChurn:
    @pytest.mark.asyncio
    async def test_churn_across_two_stores_with_purgers_keeps_every_partition_exact(
        self, store, other_store, monkeypatch
    ):
        """Workers on two stores sharing a registry upsert, delete and query
        records they alone own, and delete and recreate partitions they
        share, while each store's purger drains. Only the documented outcomes
        of a lost race are raised, and nothing hangs. Once quiet, each live
        partition holds exactly its records, with their latest values, and
        no deleted partition keeps a record but one an upsert racing its
        deletion sent."""
        # A name of its own, declaring the churn's properties, and small
        # batches, so a deleted partition takes several rounds.
        stores = tuple(
            [
                await _started_store(
                    beside._client,
                    beside._partition_registry._engine,
                    vector_store_name=_CHURN_VECTOR_STORE_NAME,
                    indexed_properties=_CHURN_DECLARED,
                    purge_batch_size=2,
                )
                for beside in (store, other_store)
            ]
        )
        for purging in stores:
            _settle_each_purge_round(purging, monkeypatch)
        ledger = _ChurnLedger()

        await asyncio.wait_for(_churn(stores, ledger), 300)
        for purging in stores:
            await _drain(purging)

        live = set()
        for partition_key in _CHURN_PARTITION_KEYS:
            handle = await stores[0].get_partition(partition_key)
            if handle is not None:
                live.add(handle._incarnation)
                await _check_live_partition(
                    handle, ledger.held.get(handle._incarnation, {})
                )
        left = await stores[0]._client.query(
            collection_name=stores[0]._collection_name,
            filter='id != ""',
            output_fields=["partition_key", "record_uuid"],
            limit=16384,
            consistency_level="Strong",
        )
        assert {
            (UUID(row["partition_key"]), UUID(row["record_uuid"]))
            for row in left
            if UUID(row["partition_key"]) not in live
        } <= ledger.raced_deletion


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
