"""Tests for QdrantVectorStore."""

import asyncio
import math
import operator
import random
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta, timezone
from typing import Any, override
from unittest.mock import AsyncMock, MagicMock
from uuid import UUID, uuid4

import httpx
import pytest
import pytest_asyncio
from qdrant_client import AsyncQdrantClient, models
from qdrant_client.http.exceptions import ResponseHandlingException, UnexpectedResponse
from sqlalchemy.engine import URL
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine

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
from memmachine_server.common.metrics_factory import MetricsFactory, OperationTracker
from memmachine_server.common.vector_store.data_types import (
    PartitionSchema,
    QueryResult,
    Record,
    VectorStoreAttemptsExhaustedError,
    VectorStorePartitionAlreadyExistsError,
    VectorStorePartitionDeletedError,
    VectorStorePartitionHandleStaleError,
    VectorStorePartitionPendingError,
    VectorStorePartitionSchemaMismatchError,
)
from memmachine_server.common.vector_store.partition_registry import (
    Registration,
)
from memmachine_server.common.vector_store.partition_registry.sqlalchemy_partition_registry import (
    SQLAlchemyVectorStorePartitionRegistry,
    SQLAlchemyVectorStorePartitionRegistryParams,
)
from memmachine_server.common.vector_store.qdrant_vector_store import (
    _PAYLOAD_INCARNATION,
    _PAYLOAD_RECORD_UUID,
    QdrantVectorStore,
    QdrantVectorStoreParams,
    QdrantVectorStorePartition,
)
from server_tests.memmachine_server.common.vector_store.partition_lifecycle_contract import (
    PartitionLifecycleContract,
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


async def _stored_uuids(partition) -> set[UUID]:
    """Record UUIDs Qdrant holds under the handle's incarnation, read past the store."""
    points, _ = await partition._client.scroll(
        collection_name=partition._vector_store_name,
        scroll_filter=models.Filter(
            must=[
                models.FieldCondition(
                    key=_PAYLOAD_INCARNATION,
                    match=models.MatchValue(value=str(partition._incarnation)),
                )
            ]
        ),
        limit=10000,
        with_payload=[_PAYLOAD_RECORD_UUID],
        with_vectors=False,
    )
    return {UUID(str((point.payload or {})[_PAYLOAD_RECORD_UUID])) for point in points}


async def _count_stored(store: QdrantVectorStore) -> int:
    """Points Qdrant holds in the store's native collection, deleted partitions' included."""
    result = await store._client.count(
        collection_name=store.vector_store_name, exact=True
    )
    return result.count


# Purge rounds a drain runs before it fails the test: far more than the
# deleted partitions a test leaves need, so only a round that keeps finding
# records reaches it.
_MAX_DRAIN_ROUNDS = 1000


async def _drain(store: QdrantVectorStore) -> None:
    """Run purge rounds until none is due."""
    for _ in range(_MAX_DRAIN_ROUNDS):
        if not await store.purge_deleted_partitions():
            return
    pytest.fail(
        f"a deleted partition was still due for purge after {_MAX_DRAIN_ROUNDS} rounds"
    )


@pytest.fixture(
    params=[
        pytest.param("qdrant_client", marks=pytest.mark.integration),
        pytest.param("qdrant_grpc_client", marks=pytest.mark.integration),
    ],
)
def any_qdrant_client(request):
    return request.getfixturevalue(request.param)


@pytest_asyncio.fixture
async def registry_engine(tmp_path):
    """The relational database holding the partition registry, one per test."""
    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'registry.db'}")
    yield engine
    await engine.dispose()


async def _params(client, registry_engine, **overrides) -> QdrantVectorStoreParams:
    """Parameters for one store: its own started registry over the shared registry database."""
    params: dict[str, Any] = {
        "client": client,
        "vector_store_name": VECTOR_STORE_NAME,
        "vector_dimensions": VECTOR_DIM,
        "similarity_metric": SimilarityMetric.COSINE,
        "indexed_properties": INDEXED_PROPERTIES,
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
    return QdrantVectorStoreParams(**params)


@pytest_asyncio.fixture
async def store(any_qdrant_client, registry_engine):
    s = QdrantVectorStore(await _params(any_qdrant_client, registry_engine))
    await s.startup()
    yield s


@pytest_asyncio.fixture
async def collection(store):
    await store.create_partition(NAME)
    coll = await store.get_partition(NAME)
    assert coll is not None
    yield coll
    await store.delete_partition(NAME)


def _normalize(v: list[float]) -> list[float]:
    mag = math.sqrt(sum(x * x for x in v))
    return [x / mag for x in v]


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


# ── Partition lifecycle ──


class TestPartitionLifecycle:
    @pytest.mark.asyncio
    async def test_create_get_delete(self, store):
        await store.create_partition("lifecycle")
        coll = await store.get_partition("lifecycle")
        assert isinstance(coll, QdrantVectorStorePartition)
        await store.delete_partition("lifecycle")

    @pytest.mark.asyncio
    async def test_get_partition_returns_qdrant_collection(self, store, collection):
        coll = await store.get_partition(NAME)
        assert isinstance(coll, QdrantVectorStorePartition)

    @pytest.mark.asyncio
    async def test_duplicate_name_raises(self, store, collection):
        with pytest.raises(VectorStorePartitionAlreadyExistsError):
            await store.create_partition(NAME)

    @pytest.mark.asyncio
    async def test_delete_nonexistent_is_idempotent(self, store):
        await store.delete_partition("nonexistent")

    @pytest.mark.asyncio
    async def test_open_or_create_creates_when_missing(self, store):
        coll = await store.open_or_create_partition("new")
        assert isinstance(coll, QdrantVectorStorePartition)
        await store.delete_partition("new")

    @pytest.mark.asyncio
    async def test_open_or_create_opens_when_exists(self, store):
        await store.create_partition("existing")
        coll = await store.open_or_create_partition("existing")
        assert isinstance(coll, QdrantVectorStorePartition)
        await store.delete_partition("existing")

    @pytest.mark.asyncio
    async def test_a_store_with_another_schema_cannot_open_the_partition(
        self, store, registry_engine
    ):
        """The schema is the collection's: another store of it must declare the same."""
        await store.create_partition("mismatch")
        other_dimensions = QdrantVectorStore(
            await _params(
                store._client, registry_engine, vector_dimensions=VECTOR_DIM + 1
            )
        )
        with pytest.raises(VectorStorePartitionSchemaMismatchError, match="mismatch"):
            await other_dimensions.open_or_create_partition("mismatch")
        other_keys = QdrantVectorStore(
            await _params(
                store._client, registry_engine, indexed_properties={"name": str}
            )
        )
        with pytest.raises(VectorStorePartitionSchemaMismatchError, match="mismatch"):
            await other_keys.get_partition("mismatch")
        await store.delete_partition("mismatch")

    @pytest.mark.asyncio
    async def test_partitions_share_the_store_collection(self, store):
        """Every partition is a payload value inside the store's one native collection."""
        await store.create_partition("coll_a")
        await store.create_partition("coll_b")

        coll_a = await store.get_partition("coll_a")
        coll_b = await store.get_partition("coll_b")
        assert coll_a is not None
        assert coll_b is not None
        assert coll_a._vector_store_name == store.vector_store_name
        assert coll_b._vector_store_name == store.vector_store_name

        await store.delete_partition("coll_a")
        await store.delete_partition("coll_b")


# ── Strict mode ──


class TestStrictMode:
    @pytest.mark.asyncio
    async def test_the_collection_overrides_the_server_strict_mode_default(
        self, any_qdrant_client, store, collection
    ):
        # The server defaults new collections to strict mode, so the store's
        # collection serves a filter on an unindexed property only by turning
        # strict mode off.
        default_collection = f"strict_mode_default_{uuid4().hex}"
        await any_qdrant_client.create_collection(
            default_collection,
            vectors_config=models.VectorParams(
                size=VECTOR_DIM, distance=models.Distance.COSINE
            ),
        )
        try:
            default_info = await any_qdrant_client.get_collection(default_collection)
        finally:
            await any_qdrant_client.delete_collection(default_collection)
        assert default_info.config.strict_mode_config is not None
        assert default_info.config.strict_mode_config.enabled is True

        info = await any_qdrant_client.get_collection(store.vector_store_name)
        assert info.config.strict_mode_config is not None
        assert info.config.strict_mode_config.enabled is False

        vector = _normalize([1.0, 0.0, 0.0])
        alpha = _make_record(vector=vector, properties={"topic": "alpha"})
        beta = _make_record(vector=vector, properties={"topic": "beta"})
        await collection.upsert(records=[alpha, beta])
        query_results = list(
            await collection.query(
                query_vectors=[vector],
                limit=10,
                property_filter=Comparison(field="topic", op="=", value="alpha"),
            )
        )
        assert [match.record_uuid for match in query_results[0].matches] == [alpha.uuid]


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

        query_results = list(await collection.query(query_vectors=[v1], limit=3))
        matches = query_results[0].matches

        assert len(matches) == 3
        assert matches[0].record_uuid == r1.uuid
        assert matches[1].record_uuid == r3.uuid
        assert matches[2].record_uuid == r2.uuid
        assert matches[0].score >= matches[1].score >= matches[2].score

    @pytest.mark.asyncio
    async def test_query_with_similarity_threshold(self, collection):
        v1 = _normalize([1.0, 0.0, 0.0])
        v2 = _normalize([0.0, 1.0, 0.0])

        r1 = _make_record(vector=v1)
        r2 = _make_record(vector=v2)

        await collection.upsert(records=[r1, r2])

        query_results = list(
            await collection.query(query_vectors=[v1], limit=10, score_threshold=0.9)
        )
        matches = query_results[0].matches

        assert len(matches) == 1
        assert matches[0].record_uuid == r1.uuid

    @pytest.mark.asyncio
    async def test_query_with_limit(self, collection):
        vectors = [_normalize([1.0, float(i) * 0.01, 0.0]) for i in range(5)]
        records = [_make_record(vector=v) for v in vectors]
        await collection.upsert(records=records)

        query_results = list(
            await collection.query(query_vectors=[vectors[0]], limit=2)
        )
        assert len(query_results[0].matches) == 2

    @pytest.mark.asyncio
    async def test_query_batch_multiple_vectors(self, collection):
        v1 = _normalize([1.0, 0.0, 0.0])
        v2 = _normalize([0.0, 1.0, 0.0])

        r1 = _make_record(vector=v1, properties={"name": "a"})
        r2 = _make_record(vector=v2, properties={"name": "b"})
        await collection.upsert(records=[r1, r2])

        all_results = list(await collection.query(query_vectors=[v1, v2], limit=1))

        assert len(all_results) == 2
        assert all_results[0].matches[0].record_uuid == r1.uuid
        assert all_results[1].matches[0].record_uuid == r2.uuid

    @pytest.mark.asyncio
    async def test_query_empty_vectors(self, collection):
        all_results = list(await collection.query(query_vectors=[], limit=10))
        assert len(all_results) == 0


# A query, and vectors that each metric ranks in its own order, with no two
# tied under any metric.
_METRIC_QUERY = [1.0, 0.0, 0.0]
_METRIC_VECTORS = [
    [3.0, 3.0, 0.0],
    [1.0, 0.2, 0.0],
    [1.4, 0.4, 0.0],
    [1.6, 0.0, 0.0],
    [0.5, 0.05, 0.0],
]


def _metric_score(
    metric: SimilarityMetric, query: list[float], vector: list[float]
) -> float:
    """The score QueryMatch defines for a vector matching a query under a metric."""
    dot = sum(q * v for q, v in zip(query, vector, strict=True))
    if metric is SimilarityMetric.COSINE:
        return dot / (math.hypot(*query) * math.hypot(*vector))
    if metric is SimilarityMetric.DOT:
        return dot
    if metric is SimilarityMetric.EUCLIDEAN:
        return math.dist(query, vector)
    return sum(abs(q - v) for q, v in zip(query, vector, strict=True))


class TestSimilarityMetrics:
    @pytest.mark.asyncio
    @pytest.mark.parametrize("metric", list(SimilarityMetric))
    async def test_matches_are_ranked_scored_and_thresholded_by_the_metric(
        self, any_qdrant_client, registry_engine, metric
    ):
        """Matches come best first, each scored as QueryMatch defines for the
        store's metric, and a score threshold keeps the matches on its
        better side: above it for a similarity, below it for a distance."""
        store = QdrantVectorStore(
            await _params(
                any_qdrant_client,
                registry_engine,
                vector_store_name=f"metric_{metric.value}",
                similarity_metric=metric,
            )
        )
        await store.startup()
        await store.create_partition(NAME)
        partition = await store.get_partition(NAME)
        assert partition is not None
        records = [_make_record(vector=vector) for vector in _METRIC_VECTORS]
        await partition.upsert(records=records)

        expected = sorted(
            (
                (_metric_score(metric, _METRIC_QUERY, record.vector), record.uuid)
                for record in records
            ),
            reverse=metric.higher_is_better,
        )
        [result] = await partition.query(
            query_vectors=[_METRIC_QUERY], limit=len(records)
        )
        assert [match.record_uuid for match in result.matches] == [
            record_uuid for _, record_uuid in expected
        ]
        assert [match.score for match in result.matches] == pytest.approx(
            [score for score, _ in expected], abs=1e-5
        )

        # Halfway between the second and third best scores.
        threshold = (expected[1][0] + expected[2][0]) / 2
        [kept] = await partition.query(
            query_vectors=[_METRIC_QUERY],
            limit=len(records),
            score_threshold=threshold,
        )
        assert [match.record_uuid for match in kept.matches] == [
            record_uuid for _, record_uuid in expected[:2]
        ]

        await store.delete_partition(NAME)

    @pytest.mark.asyncio
    @pytest.mark.parametrize("metric", list(SimilarityMetric))
    async def test_a_threshold_keeps_a_match_scoring_exactly_it(
        self, any_qdrant_client, registry_engine, metric
    ):
        """A score threshold equal to a match's reported score keeps the
        match; one a step better than that score drops it."""
        store = QdrantVectorStore(
            await _params(
                any_qdrant_client,
                registry_engine,
                vector_store_name=f"metric_{metric.value}",
                similarity_metric=metric,
            )
        )
        await store.startup()
        await store.create_partition(NAME)
        partition = await store.get_partition(NAME)
        assert partition is not None
        records = [_make_record(vector=vector) for vector in _METRIC_VECTORS]
        await partition.upsert(records=records)
        [ranked] = await partition.query(
            query_vectors=[_METRIC_QUERY], limit=len(records)
        )
        edge = ranked.matches[2]
        better = math.inf if metric.higher_is_better else -math.inf

        [at_edge] = await partition.query(
            query_vectors=[_METRIC_QUERY],
            limit=len(records),
            score_threshold=edge.score,
        )
        [past_edge] = await partition.query(
            query_vectors=[_METRIC_QUERY],
            limit=len(records),
            score_threshold=math.nextafter(edge.score, better),
        )

        assert [match.record_uuid for match in at_edge.matches] == [
            match.record_uuid for match in ranked.matches[:3]
        ]
        assert [match.record_uuid for match in past_edge.matches] == [
            match.record_uuid for match in ranked.matches[:2]
        ]
        await store.delete_partition(NAME)


@dataclass(frozen=True)
class _CurrentRegistration(Registration):
    """A live registration whose partition is never deleted."""

    @override
    async def require_current(self) -> None:
        return None


def _partition_on(client: AsyncQdrantClient) -> QdrantVectorStorePartition:
    """A handle on a given client, bound to a live incarnation."""
    return QdrantVectorStorePartition(
        client=client,
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
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("status_code", [400, 413])
async def test_a_batch_refused_as_sent_is_halved_until_it_fits(status_code: int):
    """Qdrant's REST API refuses a request over its size limit with a 400, and
    a proxy in front of it may with a 413."""
    upserted: list[list[str]] = []

    async def refuse_more_than_two(
        *, collection_name: str, points: list[models.PointStruct], wait: bool
    ) -> None:
        if len(points) > 2:
            raise UnexpectedResponse(status_code, "", b"", httpx.Headers())
        upserted.append([str(point.id) for point in points])

    client = MagicMock(spec=AsyncQdrantClient)
    client.upsert = AsyncMock(side_effect=refuse_more_than_two)
    partition = _partition_on(client)
    records = [_make_record(vector=_normalize([1.0, 0.0, 0.0])) for _ in range(5)]

    await partition.upsert(records=records)

    assert sorted(len(batch) for batch in upserted) == [1, 2, 2]
    assert {point_id for batch in upserted for point_id in batch} == {
        str(partition._point_id(record.uuid)) for record in records
    }


@pytest.mark.asyncio
@pytest.mark.parametrize("status_code", [400, 413])
async def test_an_upsert_with_a_point_refused_alone_raises(status_code: int):
    """Halving stops at a single point: a point refused on its own fails the
    upsert, which does not report it accepted."""
    refused = _make_record(vector=[0.0, 0.0, 1.0])

    async def refuse_the_refused_point(
        *, collection_name: str, points: list[models.PointStruct], wait: bool
    ) -> None:
        if any(point.vector == refused.vector for point in points):
            raise UnexpectedResponse(status_code, "", b"", httpx.Headers())

    client = MagicMock(spec=AsyncQdrantClient)
    client.upsert = AsyncMock(side_effect=refuse_the_refused_point)
    partition = _partition_on(client)
    records = [_make_record(vector=_normalize([1.0, 0.0, 0.0])) for _ in range(4)]
    records.insert(3, refused)

    with pytest.raises(UnexpectedResponse) as raised:
        await partition.upsert(records=records)
    assert raised.value.status_code == status_code


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "error",
    [
        ResponseHandlingException(TimeoutError()),
        UnexpectedResponse(500, "Internal Server Error", b"", httpx.Headers()),
    ],
    ids=["timeout", "server_error"],
)
async def test_an_upsert_that_fails_otherwise_is_not_sent_again(error: Exception):
    """A timed-out request may still be applied, so it is not resent."""
    client = MagicMock(spec=AsyncQdrantClient)
    client.upsert = AsyncMock(side_effect=error)
    partition = _partition_on(client)
    records = [_make_record(vector=_normalize([1.0, 0.0, 0.0])) for _ in range(4)]

    with pytest.raises(type(error)):
        await partition.upsert(records=records)

    client.upsert.assert_awaited_once()


# ── Filters ──


class TestFilters:
    # alice=30/9.5/True, bob=25/7.0/False, carol=35/8.0/True
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

    # scores: r0=-1.5, r1=0.0, r2=0.5, r3=1.5, r4=2.0
    async def _setup_floats(self, collection):
        vectors = [_normalize([1.0, float(i) * 0.01, 0.0]) for i in range(5)]
        scores = [-1.5, 0.0, 0.5, 1.5, 2.0]
        records = [
            _make_record(vector=v, properties={"score": s})
            for v, s in zip(vectors, scores, strict=True)
        ]
        await collection.upsert(records=records)
        return records, vectors[0]

    # dts: r0=Jan, r1=Mar, r2=Jun, r3=Sep, r4=Dec
    async def _setup_datetimes(self, collection):
        vectors = [_normalize([1.0, float(i) * 0.01, 0.0]) for i in range(5)]
        dts = [
            datetime(2024, 1, 1, tzinfo=UTC),
            datetime(2024, 3, 15, tzinfo=UTC),
            datetime(2024, 6, 1, tzinfo=UTC),
            datetime(2024, 9, 1, tzinfo=UTC),
            datetime(2024, 12, 31, tzinfo=UTC),
        ]
        records = [
            _make_record(vector=v, properties={"name": f"r{i}", "created_at": dt})
            for i, (v, dt) in enumerate(zip(vectors, dts, strict=True))
        ]
        await collection.upsert(records=records)
        return records, vectors[0], dts

    async def _query(self, collection, query_vec, field, op, value):
        all_results = list(
            await collection.query(
                query_vectors=[query_vec],
                limit=10,
                property_filter=Comparison(field=field, op=op, value=value),
            )
        )
        return {m.record_uuid for m in all_results[0].matches}

    # ── String / int ──

    @pytest.mark.asyncio
    async def test_eq_str(self, collection):
        r1, _r2, _r3, v1 = await self._setup(collection)
        query_results = list(
            await collection.query(
                query_vectors=[v1],
                limit=10,
                property_filter=Comparison(field="name", op="=", value="alice"),
            )
        )
        matches = query_results[0].matches
        assert len(matches) == 1
        assert matches[0].record_uuid == r1.uuid

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
        query_results = list(
            await collection.query(
                query_vectors=[v1],
                limit=10,
                property_filter=Comparison(field="age", op=">", value=30),
            )
        )
        matches = query_results[0].matches
        assert len(matches) == 1
        assert matches[0].record_uuid == r3.uuid

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
        query_results = list(
            await collection.query(
                query_vectors=[v1],
                limit=10,
                property_filter=Comparison(field="age", op="<", value=30),
            )
        )
        matches = query_results[0].matches
        assert len(matches) == 1
        assert matches[0].record_uuid == r2.uuid

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
    async def test_eq_float_fractional(self, collection):
        records, qv = await self._setup_floats(collection)
        uuids = await self._query(collection, qv, "score", "=", 0.5)
        assert records[2].uuid in uuids
        assert len(uuids) == 1

    @pytest.mark.asyncio
    async def test_eq_float_whole_number(self, collection):
        records, qv = await self._setup_floats(collection)
        uuids = await self._query(collection, qv, "score", "=", 2.0)
        assert records[4].uuid in uuids
        assert len(uuids) == 1

    @pytest.mark.asyncio
    async def test_eq_float_negative(self, collection):
        records, qv = await self._setup_floats(collection)
        uuids = await self._query(collection, qv, "score", "=", -1.5)
        assert records[0].uuid in uuids
        assert len(uuids) == 1

    @pytest.mark.asyncio
    async def test_eq_float_zero(self, collection):
        records, qv = await self._setup_floats(collection)
        uuids = await self._query(collection, qv, "score", "=", 0.0)
        assert records[1].uuid in uuids
        assert len(uuids) == 1

    @pytest.mark.asyncio
    async def test_ne_float_fractional(self, collection):
        records, qv = await self._setup_floats(collection)
        uuids = await self._query(collection, qv, "score", "!=", 0.5)
        assert records[2].uuid not in uuids
        assert len(uuids) == 4

    @pytest.mark.asyncio
    async def test_ne_float_whole_number(self, collection):
        records, qv = await self._setup_floats(collection)
        uuids = await self._query(collection, qv, "score", "!=", 2.0)
        assert records[4].uuid not in uuids
        assert len(uuids) == 4

    @pytest.mark.asyncio
    async def test_ne_float_negative(self, collection):
        records, qv = await self._setup_floats(collection)
        uuids = await self._query(collection, qv, "score", "!=", -1.5)
        assert records[0].uuid not in uuids
        assert len(uuids) == 4

    @pytest.mark.asyncio
    async def test_gt_float(self, collection):
        records, qv = await self._setup_floats(collection)
        uuids = await self._query(collection, qv, "score", ">", 0.5)
        assert records[0].uuid not in uuids  # -1.5
        assert records[1].uuid not in uuids  # 0.0
        assert records[2].uuid not in uuids  # 0.5 not strictly greater
        assert records[3].uuid in uuids  # 1.5
        assert records[4].uuid in uuids  # 2.0

    @pytest.mark.asyncio
    async def test_gte_float(self, collection):
        records, qv = await self._setup_floats(collection)
        uuids = await self._query(collection, qv, "score", ">=", 0.5)
        assert records[0].uuid not in uuids  # -1.5
        assert records[1].uuid not in uuids  # 0.0
        assert records[2].uuid in uuids  # 0.5
        assert records[3].uuid in uuids  # 1.5
        assert records[4].uuid in uuids  # 2.0

    @pytest.mark.asyncio
    async def test_lt_float(self, collection):
        records, qv = await self._setup_floats(collection)
        uuids = await self._query(collection, qv, "score", "<", 0.5)
        assert records[0].uuid in uuids  # -1.5
        assert records[1].uuid in uuids  # 0.0
        assert records[2].uuid not in uuids  # 0.5 not strictly less
        assert records[3].uuid not in uuids  # 1.5
        assert records[4].uuid not in uuids  # 2.0

    @pytest.mark.asyncio
    async def test_lte_float(self, collection):
        records, qv = await self._setup_floats(collection)
        uuids = await self._query(collection, qv, "score", "<=", 0.5)
        assert records[0].uuid in uuids  # -1.5
        assert records[1].uuid in uuids  # 0.0
        assert records[2].uuid in uuids  # 0.5
        assert records[3].uuid not in uuids  # 1.5
        assert records[4].uuid not in uuids  # 2.0

    @pytest.mark.asyncio
    async def test_gt_float_from_negative(self, collection):
        records, qv = await self._setup_floats(collection)
        uuids = await self._query(collection, qv, "score", ">", -1.5)
        assert records[0].uuid not in uuids  # -1.5 not strictly greater
        assert records[1].uuid in uuids  # 0.0
        assert records[2].uuid in uuids  # 0.5
        assert records[3].uuid in uuids  # 1.5
        assert records[4].uuid in uuids  # 2.0

    @pytest.mark.asyncio
    async def test_lt_float_zero(self, collection):
        records, qv = await self._setup_floats(collection)
        uuids = await self._query(collection, qv, "score", "<", 0.0)
        assert records[0].uuid in uuids  # -1.5
        assert records[1].uuid not in uuids  # 0.0 not strictly less
        assert records[2].uuid not in uuids
        assert records[3].uuid not in uuids
        assert records[4].uuid not in uuids

    # ── Datetime ──

    @pytest.mark.asyncio
    async def test_eq_datetime(self, collection):
        records, qv, dts = await self._setup_datetimes(collection)
        uuids = await self._query(collection, qv, "created_at", "=", dts[2])
        assert records[2].uuid in uuids
        assert len(uuids) == 1

    @pytest.mark.asyncio
    async def test_ne_datetime(self, collection):
        records, qv, dts = await self._setup_datetimes(collection)
        uuids = await self._query(collection, qv, "created_at", "!=", dts[2])
        assert records[2].uuid not in uuids
        assert len(uuids) == 4

    @pytest.mark.asyncio
    async def test_gt_datetime(self, collection):
        records, qv, dts = await self._setup_datetimes(collection)
        uuids = await self._query(collection, qv, "created_at", ">", dts[2])
        assert records[0].uuid not in uuids
        assert records[1].uuid not in uuids
        assert records[2].uuid not in uuids  # not strictly greater
        assert records[3].uuid in uuids
        assert records[4].uuid in uuids

    @pytest.mark.asyncio
    async def test_gte_datetime(self, collection):
        records, qv, dts = await self._setup_datetimes(collection)
        uuids = await self._query(collection, qv, "created_at", ">=", dts[2])
        assert records[0].uuid not in uuids
        assert records[1].uuid not in uuids
        assert records[2].uuid in uuids
        assert records[3].uuid in uuids
        assert records[4].uuid in uuids

    @pytest.mark.asyncio
    async def test_lt_datetime(self, collection):
        records, qv, dts = await self._setup_datetimes(collection)
        uuids = await self._query(collection, qv, "created_at", "<", dts[2])
        assert records[0].uuid in uuids
        assert records[1].uuid in uuids
        assert records[2].uuid not in uuids  # not strictly less
        assert records[3].uuid not in uuids
        assert records[4].uuid not in uuids

    @pytest.mark.asyncio
    async def test_lte_datetime(self, collection):
        records, qv, dts = await self._setup_datetimes(collection)
        uuids = await self._query(collection, qv, "created_at", "<=", dts[2])
        assert records[0].uuid in uuids
        assert records[1].uuid in uuids
        assert records[2].uuid in uuids
        assert records[3].uuid not in uuids
        assert records[4].uuid not in uuids

    @pytest.mark.asyncio
    async def test_datetime_microseconds_eq(self, collection):
        """Equality filter distinguishes microsecond precision."""
        v1 = _normalize([1.0, 0.0, 0.0])
        v2 = _normalize([1.0, 0.1, 0.0])
        dt1 = datetime(2024, 6, 15, 12, 30, 45, 123456, tzinfo=UTC)
        dt2 = datetime(2024, 6, 15, 12, 30, 45, 999999, tzinfo=UTC)
        r1 = _make_record(vector=v1, properties={"name": "a", "created_at": dt1})
        r2 = _make_record(vector=v2, properties={"name": "b", "created_at": dt2})
        await collection.upsert(records=[r1, r2])

        uuids = await self._query(collection, v1, "created_at", "=", dt1)
        assert r1.uuid in uuids
        assert r2.uuid not in uuids

    @pytest.mark.asyncio
    async def test_datetime_boundary_inclusive(self, collection):
        """gte and lte are both inclusive at the exact boundary."""
        v1 = _normalize([1.0, 0.0, 0.0])
        v2 = _normalize([1.0, 0.1, 0.0])
        boundary = datetime(2024, 6, 1, tzinfo=UTC)
        before = datetime(2024, 5, 31, 23, 59, 59, tzinfo=UTC)
        r1 = _make_record(vector=v1, properties={"name": "b", "created_at": boundary})
        r2 = _make_record(vector=v2, properties={"name": "a", "created_at": before})
        await collection.upsert(records=[r1, r2])

        gte_uuids = await self._query(collection, v1, "created_at", ">=", boundary)
        lte_uuids = await self._query(collection, v1, "created_at", "<=", boundary)
        assert r1.uuid in gte_uuids
        assert r2.uuid not in gte_uuids
        assert r1.uuid in lte_uuids
        assert r2.uuid in lte_uuids

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
    async def test_gte_datetime_cross_timezone(self, collection):
        """Range filter works correctly with a non-UTC filter value."""
        v1 = _normalize([1.0, 0.0, 0.0])
        v2 = _normalize([1.0, 0.1, 0.0])
        v3 = _normalize([1.0, 0.2, 0.0])
        dt1 = datetime(2024, 1, 1, tzinfo=UTC)
        dt2 = datetime(2024, 6, 15, 12, 0, 0, tzinfo=UTC)
        dt3 = datetime(2024, 12, 31, 23, 0, 0, tzinfo=UTC)
        r1 = _make_record(vector=v1, properties={"name": "a", "created_at": dt1})
        r2 = _make_record(vector=v2, properties={"name": "b", "created_at": dt2})
        r3 = _make_record(vector=v3, properties={"name": "c", "created_at": dt3})
        await collection.upsert(records=[r1, r2, r3])

        # 2024-06-01 00:00 PST = 2024-06-01 08:00 UTC
        pst = timezone(timedelta(hours=-8))
        cutoff = datetime(2024, 6, 1, 0, 0, 0, tzinfo=pst)
        uuids = await self._query(collection, v1, "created_at", ">=", cutoff)
        assert r1.uuid not in uuids
        assert r2.uuid in uuids
        assert r3.uuid in uuids

    @pytest.mark.asyncio
    async def test_eq_naive_datetime(self, collection):
        """Equality filter works for naive datetimes."""
        v1 = _normalize([1.0, 0.0, 0.0])
        naive_dt = datetime(2024, 6, 15, 12, 0, 0, tzinfo=UTC).replace(tzinfo=None)
        r1 = _make_record(vector=v1, properties={"name": "n", "created_at": naive_dt})
        await collection.upsert(records=[r1])

        uuids = await self._query(collection, v1, "created_at", "=", naive_dt)
        assert r1.uuid in uuids
        assert len(uuids) == 1

    @pytest.mark.asyncio
    async def test_ne_naive_datetime(self, collection):
        """Not-equal filter works for naive datetimes."""
        v1 = _normalize([1.0, 0.0, 0.0])
        v2 = _normalize([1.0, 0.1, 0.0])
        naive1 = datetime(2024, 1, 1, tzinfo=UTC).replace(tzinfo=None)
        naive2 = datetime(2024, 6, 15, tzinfo=UTC).replace(tzinfo=None)
        r1 = _make_record(vector=v1, properties={"name": "a", "created_at": naive1})
        r2 = _make_record(vector=v2, properties={"name": "b", "created_at": naive2})
        await collection.upsert(records=[r1, r2])

        uuids = await self._query(collection, v1, "created_at", "!=", naive1)
        assert r1.uuid not in uuids
        assert r2.uuid in uuids

    @pytest.mark.asyncio
    async def test_gt_naive_datetime(self, collection):
        """gt filter works for naive datetimes."""
        v1 = _normalize([1.0, 0.0, 0.0])
        v2 = _normalize([1.0, 0.1, 0.0])
        naive1 = datetime(2024, 1, 1, tzinfo=UTC).replace(tzinfo=None)
        naive2 = datetime(2024, 6, 15, 12, 0, 0, tzinfo=UTC).replace(tzinfo=None)
        r1 = _make_record(vector=v1, properties={"name": "a", "created_at": naive1})
        r2 = _make_record(vector=v2, properties={"name": "b", "created_at": naive2})
        await collection.upsert(records=[r1, r2])

        cutoff = datetime(2024, 3, 1, tzinfo=UTC).replace(tzinfo=None)
        uuids = await self._query(collection, v1, "created_at", ">", cutoff)
        assert r1.uuid not in uuids
        assert r2.uuid in uuids

    @pytest.mark.asyncio
    async def test_lt_naive_datetime(self, collection):
        """lt filter works for naive datetimes."""
        v1 = _normalize([1.0, 0.0, 0.0])
        v2 = _normalize([1.0, 0.1, 0.0])
        naive1 = datetime(2024, 1, 1, tzinfo=UTC).replace(tzinfo=None)
        naive2 = datetime(2024, 6, 15, tzinfo=UTC).replace(tzinfo=None)
        r1 = _make_record(vector=v1, properties={"name": "a", "created_at": naive1})
        r2 = _make_record(vector=v2, properties={"name": "b", "created_at": naive2})
        await collection.upsert(records=[r1, r2])

        cutoff = datetime(2024, 3, 1, tzinfo=UTC).replace(tzinfo=None)
        uuids = await self._query(collection, v1, "created_at", "<", cutoff)
        assert r1.uuid in uuids
        assert r2.uuid not in uuids

    # ── IsNull ──

    @pytest.mark.asyncio
    async def test_is_null(self, collection):
        v1 = _normalize([1.0, 0.0, 0.0])
        v2 = _normalize([1.0, 0.1, 0.0])
        v3 = _normalize([1.0, 0.2, 0.0])
        v4 = _normalize([1.0, 0.3, 0.0])
        r_has_value = _make_record(vector=v1, properties={"name": "has_name"})
        r_explicit_none = _make_record(vector=v2, properties={})
        r_key_missing = _make_record(vector=v3, properties={"age": 25})
        r_no_payload = _make_record(vector=v4, properties=None)
        await collection.upsert(
            records=[r_has_value, r_explicit_none, r_key_missing, r_no_payload],
        )

        query_results = list(
            await collection.query(
                query_vectors=[v1],
                limit=10,
                property_filter=IsNull(field="name"),
            )
        )
        uuids = {m.record_uuid for m in query_results[0].matches}
        assert r_has_value.uuid not in uuids
        assert r_explicit_none.uuid in uuids
        assert r_key_missing.uuid in uuids
        assert r_no_payload.uuid in uuids

    @pytest.mark.asyncio
    async def test_is_not_null(self, collection):
        v1 = _normalize([1.0, 0.0, 0.0])
        v2 = _normalize([1.0, 0.1, 0.0])
        v3 = _normalize([1.0, 0.2, 0.0])
        v4 = _normalize([1.0, 0.3, 0.0])
        r_has_value = _make_record(vector=v1, properties={"name": "has_name"})
        r_explicit_none = _make_record(vector=v2, properties={})
        r_key_missing = _make_record(vector=v3, properties={"age": 25})
        r_no_payload = _make_record(vector=v4, properties=None)
        await collection.upsert(
            records=[r_has_value, r_explicit_none, r_key_missing, r_no_payload],
        )

        query_results = list(
            await collection.query(
                query_vectors=[v1],
                limit=10,
                property_filter=Not(expr=IsNull(field="name")),
            )
        )
        uuids = {m.record_uuid for m in query_results[0].matches}
        assert r_has_value.uuid in uuids
        assert r_explicit_none.uuid not in uuids
        assert r_key_missing.uuid not in uuids
        assert r_no_payload.uuid not in uuids

    # ── In / And / Or / Not ──

    @pytest.mark.asyncio
    async def test_in(self, collection):
        r1, _r2, r3, v1 = await self._setup(collection)
        query_results = list(
            await collection.query(
                query_vectors=[v1],
                limit=10,
                property_filter=In(field="name", values=["alice", "carol"]),
            )
        )
        matches = query_results[0].matches
        uuids = {m.record_uuid for m in matches}
        assert r1.uuid in uuids
        assert r3.uuid in uuids
        assert len(matches) == 2

    @pytest.mark.asyncio
    async def test_and(self, collection):
        _r1, _r2, r3, v1 = await self._setup(collection)
        query_results = list(
            await collection.query(
                query_vectors=[v1],
                limit=10,
                property_filter=And(
                    left=Comparison(field="active", op="=", value=True),
                    right=Comparison(field="age", op=">", value=30),
                ),
            )
        )
        matches = query_results[0].matches
        assert len(matches) == 1
        assert matches[0].record_uuid == r3.uuid

    @pytest.mark.asyncio
    async def test_or(self, collection):
        r1, _r2, r3, v1 = await self._setup(collection)
        query_results = list(
            await collection.query(
                query_vectors=[v1],
                limit=10,
                property_filter=Or(
                    left=Comparison(field="name", op="=", value="alice"),
                    right=Comparison(field="name", op="=", value="carol"),
                ),
            )
        )
        matches = query_results[0].matches
        uuids = {m.record_uuid for m in matches}
        assert r1.uuid in uuids
        assert r3.uuid in uuids
        assert len(matches) == 2

    @pytest.mark.asyncio
    async def test_not(self, collection):
        r1, r2, _r3, v1 = await self._setup(collection)
        query_results = list(
            await collection.query(
                query_vectors=[v1],
                limit=10,
                property_filter=Not(expr=Comparison(field="age", op=">", value=30)),
            )
        )
        matches = query_results[0].matches
        uuids = {m.record_uuid for m in matches}
        assert r1.uuid in uuids
        assert r2.uuid in uuids
        assert len(matches) == 2

    # ── Delete ──

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


class TestDelete:
    @pytest.mark.asyncio
    async def test_delete_records(self, collection):
        v1 = _normalize([1.0, 0.0, 0.0])
        v2 = _normalize([0.0, 1.0, 0.0])

        r1 = _make_record(vector=v1)
        r2 = _make_record(vector=v2)

        await collection.upsert(records=[r1, r2])
        await collection.delete(record_uuids=[r1.uuid])

        assert await _stored_uuids(collection) == {r2.uuid}


# ── Partition isolation (via separate logical collections) ──


class TestPartitionIsolation:
    @pytest.mark.asyncio
    async def test_query_only_returns_own_partition(self, store):
        await store.create_partition("tenant_a")
        await store.create_partition("tenant_b")
        coll_a = await store.get_partition("tenant_a")
        coll_b = await store.get_partition("tenant_b")
        assert coll_a is not None
        assert coll_b is not None

        v1 = _normalize([1.0, 0.0, 0.0])
        r1 = _make_record(vector=v1, properties={})
        r2 = _make_record(vector=v1, properties={})

        await coll_a.upsert(records=[r1])
        await coll_b.upsert(records=[r2])

        results_a = list(await coll_a.query(query_vectors=[v1], limit=10))
        results_b = list(await coll_b.query(query_vectors=[v1], limit=10))

        uuids_a = {m.record_uuid for m in results_a[0].matches}
        uuids_b = {m.record_uuid for m in results_b[0].matches}
        assert uuids_a == {r1.uuid}
        assert uuids_b == {r2.uuid}

        await store.delete_partition("tenant_a")
        await store.delete_partition("tenant_b")

    @pytest.mark.asyncio
    async def test_a_query_returns_only_its_own_partitions_records(self, store):
        await store.create_partition("tenant_a")
        await store.create_partition("tenant_b")
        coll_a = await store.get_partition("tenant_a")
        coll_b = await store.get_partition("tenant_b")
        assert coll_a is not None
        assert coll_b is not None

        v1 = _normalize([1.0, 0.0, 0.0])
        r1 = _make_record(vector=v1)
        r2 = _make_record(vector=v1)

        await coll_a.upsert(records=[r1])
        await coll_b.upsert(records=[r2])

        [result] = await coll_a.query(query_vectors=[v1], limit=10)
        assert [match.record_uuid for match in result.matches] == [r1.uuid]

        await store.delete_partition("tenant_a")
        await store.delete_partition("tenant_b")

    @pytest.mark.asyncio
    async def test_a_filtered_query_returns_only_its_own_partitions_records(
        self, store
    ):
        """Two partitions share the store's native collection and hold
        records that match the same filters; a filtered query returns its
        own records."""
        await store.create_partition("tenant_a")
        await store.create_partition("tenant_b")
        coll_a = await store.get_partition("tenant_a")
        coll_b = await store.get_partition("tenant_b")
        assert coll_a is not None
        assert coll_b is not None

        vector = _normalize([1.0, 0.0, 0.0])
        # "note" is not declared in the schema.
        properties: dict[str, PropertyValue] = {
            "name": "alice",
            "age": 30,
            "note": "shared",
        }
        records_a = [_make_record(vector=vector, properties=properties)]
        records_b = [_make_record(vector=vector, properties=properties)]
        await coll_a.upsert(records=records_a)
        await coll_b.upsert(records=records_b)

        filters = [
            Comparison(field="name", op="=", value="alice"),
            Comparison(field="age", op=">=", value=30),
            In(field="name", values=["alice", "bob"]),
            Or(
                left=Comparison(field="name", op="=", value="bob"),
                right=Comparison(field="age", op="<", value=31),
            ),
            Not(expr=IsNull(field="note")),
            And(
                left=Comparison(field="note", op="=", value="shared"),
                right=Not(expr=IsNull(field="age")),
            ),
        ]
        for property_filter in filters:
            for partition, own in ((coll_a, records_a), (coll_b, records_b)):
                [result] = await partition.query(
                    query_vectors=[vector], limit=10, property_filter=property_filter
                )
                assert [match.record_uuid for match in result.matches] == [
                    record.uuid for record in own
                ], property_filter

        await store.delete_partition("tenant_a")
        await store.delete_partition("tenant_b")

    @pytest.mark.asyncio
    async def test_stores_of_two_names_keep_their_records_in_separate_storage(
        self, store, registry_engine
    ):
        """Stores of two names share no storage, even on one client and
        under one partition key and schema."""
        other = QdrantVectorStore(
            await _params(
                store._client,
                registry_engine,
                vector_store_name=f"{VECTOR_STORE_NAME}_other",
            )
        )
        await other.startup()
        await store.create_partition(NAME)
        await other.create_partition(NAME)
        writer = await store.get_partition(NAME)
        assert writer is not None
        writer_before = await _count_stored(store)
        other_before = await _count_stored(other)

        await writer.upsert(
            records=[
                _make_record(vector=_normalize([1.0, 0.1 * index, 0.0]))
                for index in range(3)
            ]
        )

        assert await _count_stored(store) == writer_before + 3
        assert await _count_stored(other) == other_before

        await store.delete_partition(NAME)
        await other.delete_partition(NAME)

    @pytest.mark.asyncio
    async def test_the_same_uuid_in_two_partitions_is_two_records(self, store):
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

        assert await _stored_uuids(coll_a) == {record_uuid}
        assert await _stored_uuids(coll_b) == {record_uuid}
        [kept_a] = await coll_a.query(
            query_vectors=[v1],
            limit=10,
            property_filter=Comparison(field="name", op="=", value="a"),
        )
        assert [match.record_uuid for match in kept_a.matches] == [record_uuid]

        await coll_a.delete(record_uuids=[record_uuid])
        assert await _stored_uuids(coll_a) == set()
        assert await _stored_uuids(coll_b) == {record_uuid}

        await store.delete_partition("tenant_a")
        await store.delete_partition("tenant_b")

    @pytest.mark.asyncio
    async def test_delete_only_affects_own_partition(self, store):
        await store.create_partition("tenant_a")
        await store.create_partition("tenant_b")
        coll_a = await store.get_partition("tenant_a")
        coll_b = await store.get_partition("tenant_b")
        assert coll_a is not None
        assert coll_b is not None

        v1 = _normalize([1.0, 0.0, 0.0])
        r1 = _make_record(vector=v1)
        r2 = _make_record(vector=v1)

        await coll_a.upsert(records=[r1])
        await coll_b.upsert(records=[r2])

        # Attempt to delete r2 using tenant_a's collection — should not work
        await coll_a.delete(record_uuids=[r2.uuid])

        assert await _stored_uuids(coll_b) == {r2.uuid}

        await store.delete_partition("tenant_a")
        await store.delete_partition("tenant_b")


# ── Purge ──


class TestPurge:
    @pytest.mark.asyncio
    async def test_a_failed_creations_tombstone_is_retired_by_one_round(
        self, store, monkeypatch
    ):
        """A creation whose partition storage preparation fails leaves a
        tombstone under which Qdrant holds no point: its purge round finds
        nothing there and retires it, raising nothing."""

        async def refused(partition_key, incarnation) -> None:
            raise RuntimeError("the backend refused")

        monkeypatch.setattr(store, "_prepare_partition_storage", refused)
        with pytest.raises(RuntimeError, match="refused"):
            await store.create_partition(NAME)
        monkeypatch.undo()

        # The cancelled creation's tombstone is due, and one round retires it.
        assert await store.purge_deleted_partitions() is True
        assert await store.purge_deleted_partitions() is False

    @pytest.mark.asyncio
    async def test_a_purge_round_on_a_dropped_native_collection_retires_the_tombstone(
        self, any_qdrant_client, registry_engine
    ):
        """A deleted partition whose native collection is gone holds nothing:
        its purge round raises nothing and removes the tombstone."""
        # A name of its own, so dropping its native collection spares the others.
        dropped = QdrantVectorStore(
            await _params(
                any_qdrant_client, registry_engine, vector_store_name="dropped_native"
            )
        )
        await dropped.startup()
        await dropped.create_partition("dropped")
        partition = await dropped.get_partition("dropped")
        assert partition is not None
        await partition.upsert(
            records=[_make_record(vector=_normalize([1.0, 0.0, 0.0]))]
        )
        await dropped.delete_partition("dropped")
        await any_qdrant_client.delete_collection(dropped.vector_store_name)

        await _drain(dropped)
        assert await dropped.purge_deleted_partitions() is False

    @pytest.mark.asyncio
    async def test_a_write_landing_after_a_purge_round_is_reclaimed_by_the_next(
        self, store
    ):
        """A write whose liveness check passed before its partition was
        deleted can land after a purge round deleted the incarnation's
        records; the tombstone stays until a round finds nothing, so the next
        round reclaims the late write."""
        key = "purged"
        await store.create_partition(key)
        partition = await store.get_partition(key)
        assert partition is not None
        await partition.upsert(
            records=[
                _make_record(vector=_normalize([1.0, 0.1 * index, 0.0]))
                for index in range(3)
            ]
        )
        await store.delete_partition(key)

        assert await store.purge_deleted_partitions() is True
        assert await _stored_uuids(partition) == set()

        # The handle's backend write, past its liveness checks, as a write in
        # flight across the deletion lands.
        late = [
            _make_record(vector=_normalize([0.1 * index, 1.0, 0.0]))
            for index in range(2)
        ]
        await partition._upsert(late)
        assert await _stored_uuids(partition) == {record.uuid for record in late}

        await _drain(store)
        assert await _stored_uuids(partition) == set()


# ── A seeded operation sequence against a model ──


_MODEL_SEED = 1735
_MODEL_STEPS = 200
_MODEL_POOL_SIZE = 12
_MODEL_STORE_NAME = "model"
_MODEL_KEYS = ("model_a", "model_b")
# "tag" is not declared in the schema.
_MODEL_INDEXED_PROPERTIES: dict[str, PropertyType] = {"color": str, "size": int}
_MODEL_COLORS = ("red", "green", "blue")
_MODEL_TAGS = ("x", "y")
_MODEL_PROBE = [1.0, 0.0, 0.0]


def _random_vector(rng: random.Random) -> list[float]:
    return [rng.uniform(0.1, 1.0) * rng.choice((-1.0, 1.0)) for _ in range(VECTOR_DIM)]


def _random_properties(rng: random.Random) -> dict[str, PropertyValue]:
    """Properties of which each is present or absent, so a re-upsert may change or drop any."""
    properties: dict[str, PropertyValue] = {}
    if rng.random() < 0.7:
        properties["color"] = rng.choice(_MODEL_COLORS)
    if rng.random() < 0.7:
        properties["size"] = rng.randrange(10)
    if rng.random() < 0.5:
        properties["tag"] = rng.choice(_MODEL_TAGS)
    return properties


def _random_filter(rng: random.Random) -> FilterExpr:
    color = rng.choice(_MODEL_COLORS)
    size = rng.randrange(10)
    colors = rng.sample(_MODEL_COLORS, 2)
    tag = rng.choice(_MODEL_TAGS)
    return rng.choice(
        [
            Comparison(field="color", op="=", value=color),
            Comparison(field="size", op=">", value=size),
            Comparison(field="size", op="<=", value=size),
            In(field="color", values=colors),
            IsNull(field="color"),
            Not(expr=IsNull(field="tag")),
            Comparison(field="tag", op="=", value=tag),
            Or(
                left=Comparison(field="color", op="=", value=color),
                right=Comparison(field="size", op="<=", value=size),
            ),
            And(
                left=Comparison(field="color", op="=", value=color),
                right=Not(expr=IsNull(field="size")),
            ),
        ]
    )


_MODEL_COMPARISONS = {"=": operator.eq, ">": operator.gt, "<=": operator.le}


def _model_matches(expr: FilterExpr, properties: dict[str, PropertyValue]) -> bool:
    """Whether properties meet a filter; an absent property meets only IsNull."""
    if isinstance(expr, Comparison):
        value = properties.get(expr.field)
        return value is not None and _MODEL_COMPARISONS[expr.op](value, expr.value)
    if isinstance(expr, In):
        return properties.get(expr.field) in expr.values
    if isinstance(expr, IsNull):
        return expr.field not in properties
    if isinstance(expr, Not):
        return not _model_matches(expr.expr, properties)
    if isinstance(expr, And):
        return _model_matches(expr.left, properties) and _model_matches(
            expr.right, properties
        )
    if isinstance(expr, Or):
        return _model_matches(expr.left, properties) or _model_matches(
            expr.right, properties
        )
    raise TypeError(expr)


def _assert_matches_model(
    result: QueryResult, expected: dict[UUID, Record], query: list[float]
) -> None:
    """The result holds each expected record once, scored by its latest vector, best first."""
    scores = {match.record_uuid: match.score for match in result.matches}
    assert len(result.matches) == len(expected)
    assert scores.keys() == expected.keys()
    for record_uuid, record in expected.items():
        assert scores[record_uuid] == pytest.approx(
            _metric_score(SimilarityMetric.COSINE, query, record.vector), abs=1e-5
        )
    ranked = [match.score for match in result.matches]
    assert ranked == sorted(ranked, reverse=True)


class _ModelRun:
    """Operations on the model test's partitions, each applied to the store and to the model."""

    def __init__(
        self, store: QdrantVectorStore, rng: random.Random, pool: list[UUID]
    ) -> None:
        self.store = store
        self.rng = rng
        self.pool = pool
        self.handles: dict[str, QdrantVectorStorePartition] = {}
        self.model: dict[str, dict[UUID, Record]] = {}
        self.baseline = 0

    async def create(self, key: str) -> None:
        await self.store.create_partition(key)
        handle = await self.store.get_partition(key)
        assert handle is not None
        self.handles[key] = handle
        self.model[key] = {}

    async def upsert(self, key: str) -> None:
        records = [
            Record(
                uuid=record_uuid,
                vector=_random_vector(self.rng),
                properties=_random_properties(self.rng),
            )
            for record_uuid in self.rng.sample(self.pool, self.rng.randint(1, 4))
        ]
        await self.handles[key].upsert(records=records)
        self.model[key].update({record.uuid: record for record in records})

    async def delete(self, key: str) -> None:
        record_uuids = self.rng.sample(self.pool, self.rng.randint(1, 4))
        await self.handles[key].delete(record_uuids=record_uuids)
        for record_uuid in record_uuids:
            self.model[key].pop(record_uuid, None)

    async def query(self, key: str) -> None:
        property_filter = _random_filter(self.rng)
        query = _random_vector(self.rng)
        [result] = await self.handles[key].query(
            query_vectors=[query], limit=len(self.pool), property_filter=property_filter
        )
        expected = {
            record_uuid: record
            for record_uuid, record in self.model[key].items()
            if _model_matches(property_filter, record.properties)
        }
        _assert_matches_model(result, expected, query)

    async def recreate(self, key: str) -> None:
        stale = self.handles[key]
        await self.store.delete_partition(key)
        with pytest.raises(VectorStorePartitionHandleStaleError):
            await stale.query(query_vectors=[_MODEL_PROBE], limit=1)
        await self.create(key)

    async def drain(self, key: str) -> None:
        await _drain(self.store)
        assert await _count_stored(self.store) == self.baseline + sum(
            len(records) for records in self.model.values()
        )

    async def check(self) -> None:
        """Each partition stores, and a query matches, what the model holds."""
        for key, handle in self.handles.items():
            assert await _stored_uuids(handle) == set(self.model[key]), key
            [result] = await handle.query(
                query_vectors=[_MODEL_PROBE], limit=len(self.pool)
            )
            _assert_matches_model(result, self.model[key], _MODEL_PROBE)


class TestAgainstAModel:
    @pytest.mark.asyncio
    async def test_a_seeded_operation_sequence_agrees_with_a_model(
        self, any_qdrant_client, registry_engine
    ):
        """Two partitions of one store take a seeded sequence of upserts
        (re-upserts change or drop properties), deletes (of their own, the
        other partition's and absent UUIDs), filtered queries, deletion and
        re-creation under the same key, and purge drains. After each step the
        records stored, a query's matches and a drain's count agree with a
        model that holds each partition's records."""
        store = QdrantVectorStore(
            await _params(
                any_qdrant_client,
                registry_engine,
                vector_store_name=_MODEL_STORE_NAME,
                indexed_properties=_MODEL_INDEXED_PROPERTIES,
            )
        )
        await store.startup()
        rng = random.Random(_MODEL_SEED)
        run = _ModelRun(store, rng, sorted(uuid4() for _ in range(_MODEL_POOL_SIZE)))
        for key in _MODEL_KEYS:
            await run.create(key)
        run.baseline = await _count_stored(store)

        operations = {
            run.upsert: 5,
            run.delete: 3,
            run.query: 3,
            run.recreate: 1,
            run.drain: 1,
        }
        for step in range(_MODEL_STEPS):
            key = rng.choice(_MODEL_KEYS)
            [operation] = rng.choices(
                list(operations), weights=list(operations.values())
            )
            try:
                await operation(key)
                await run.check()
            except AssertionError as error:
                raise AssertionError(
                    f"step {step}: {operation.__name__} on {key}"
                ) from error

        for key in _MODEL_KEYS:
            await store.delete_partition(key)
        await _drain(store)
        assert await _count_stored(store) == run.baseline


# ── Concurrent churn across stores sharing one registry ──


_CHURN_SEED = 1735
_CHURN_WORKERS = 8
_CHURN_STEPS = 50
_CHURN_OWNED = 4
_CHURN_KEYS = ("churn_a", "churn_b", "churn_c")
_CHURN_INDEXED_PROPERTIES: dict[str, PropertyType] = {"owner": int}
# Bounds the whole churn, so a deadlock fails the test instead of hanging it.
_CHURN_DEADLINE_SECONDS = 120
# The outcomes the contract documents for an operation that races another
# worker's: anything else fails the test.
_CHURN_DOMAIN_ERRORS = (
    VectorStorePartitionHandleStaleError,
    VectorStorePartitionAlreadyExistsError,
    VectorStorePartitionPendingError,
    VectorStorePartitionDeletedError,
    VectorStoreAttemptsExhaustedError,
)


async def _stored_points(store: QdrantVectorStore) -> set[tuple[UUID, UUID]]:
    """The (incarnation, record UUID) of every point Qdrant holds in the store's native collection."""
    points, _ = await store._client.scroll(
        collection_name=store.vector_store_name,
        limit=10000,
        with_payload=[_PAYLOAD_INCARNATION, _PAYLOAD_RECORD_UUID],
        with_vectors=False,
    )
    return {
        (
            UUID(str((point.payload or {})[_PAYLOAD_INCARNATION])),
            UUID(str((point.payload or {})[_PAYLOAD_RECORD_UUID])),
        )
        for point in points
    }


@dataclass
class _ChurnWorker:
    """A worker writing only the record UUIDs it owns, so its records in each partition life follow from its own operations.

    `lives` holds, per incarnation, the records it wrote there and has not
    deleted; `unconfirmed` the records whose upsert raised because the
    partition was deleted meanwhile, which may have landed.
    """

    index: int
    store: QdrantVectorStore
    owned: list[UUID]
    rng: random.Random
    deleted: asyncio.Event
    handles: dict[str, QdrantVectorStorePartition] = field(default_factory=dict)
    lives: defaultdict[UUID, dict[UUID, Record]] = field(
        default_factory=lambda: defaultdict(dict)
    )
    unconfirmed: set[tuple[UUID, UUID]] = field(default_factory=set)

    async def run(self) -> None:
        operations = {self.upsert: 4, self.delete: 2, self.query: 3, self.drop: 1}
        for _ in range(_CHURN_STEPS):
            key = self.rng.choice(_CHURN_KEYS)
            [operation] = self.rng.choices(
                list(operations), weights=list(operations.values())
            )
            try:
                await operation(key)
            except _CHURN_DOMAIN_ERRORS:
                self.handles.pop(key, None)

    async def handle(self, key: str) -> QdrantVectorStorePartition:
        if key not in self.handles:
            self.handles[key] = await self.store.open_or_create_partition(key)
        return self.handles[key]

    async def upsert(self, key: str) -> None:
        records = [
            Record(
                uuid=record_uuid,
                vector=_random_vector(self.rng),
                properties={"owner": self.index},
            )
            for record_uuid in self.rng.sample(self.owned, self.rng.randint(1, 3))
        ]
        handle = await self.handle(key)
        try:
            await handle.upsert(records=records)
        except VectorStorePartitionHandleStaleError:
            self.unconfirmed.update(
                (handle._incarnation, record.uuid) for record in records
            )
            raise
        self.lives[handle._incarnation].update(
            {record.uuid: record for record in records}
        )

    async def delete(self, key: str) -> None:
        record_uuids = self.rng.sample(self.owned, self.rng.randint(1, 3))
        handle = await self.handle(key)
        await handle.delete(record_uuids=record_uuids)
        for record_uuid in record_uuids:
            self.lives[handle._incarnation].pop(record_uuid, None)

    async def query(self, key: str) -> None:
        query = _random_vector(self.rng)
        handle = await self.handle(key)
        [result] = await handle.query(
            query_vectors=[query],
            limit=len(self.owned),
            property_filter=Comparison(field="owner", op="=", value=self.index),
        )
        found = {match.record_uuid for match in result.matches}
        own = set(self.lives[handle._incarnation])
        # Only this worker removes its records from a live partition; the
        # purge may remove them once the partition is deleted.
        assert found <= own
        # A delete of nothing raises once the partition is deleted, so past
        # it the partition was live throughout the query.
        await handle.delete(record_uuids=[])
        assert found == own

    async def drop(self, key: str) -> None:
        create_again = self.rng.random() < 0.5
        self.handles.pop(key, None)
        await self.store.delete_partition(key)
        self.deleted.set()
        if create_again:
            await self.store.create_partition(key)


@pytest_asyncio.fixture
async def sqlite_registry_engines(tmp_path):
    """Two engines on one SQLite registry file, as two processes open it."""
    url = f"sqlite+aiosqlite:///{tmp_path / 'registry.db'}"
    engines = [create_async_engine(url) for _ in range(2)]
    yield engines
    for engine in engines:
        await engine.dispose()


@pytest_asyncio.fixture
async def postgresql_registry_engines(pg_server):
    """Two engines on one PostgreSQL registry database, as two processes connect to it."""
    url = URL.create(
        "postgresql+asyncpg",
        username=pg_server["user"],
        password=pg_server["password"],
        host=pg_server["host"],
        port=pg_server["port"],
        database=pg_server["database"],
    )
    engines = [create_async_engine(url) for _ in range(2)]
    yield engines
    for engine in engines:
        await engine.dispose()


@pytest.fixture(params=["sqlite", "postgresql"])
def registry_engines(request):
    """Two engines on one registry database."""
    return request.getfixturevalue(f"{request.param}_registry_engines")


async def _stores_sharing_a_registry(
    clients: list[AsyncQdrantClient], engines: list[AsyncEngine]
) -> list[QdrantVectorStore]:
    """One store per client, each with its own registry object and engine on one registry."""
    # One registry and one native collection: one vector store name, its
    # registry on one database. The name is fresh, since the registry
    # database may outlive the test.
    vector_store_name = f"churn_{uuid4().hex[:24]}"
    stores = []
    for client, engine in zip(clients, engines, strict=True):
        store = QdrantVectorStore(
            await _params(
                client,
                engine,
                vector_store_name=vector_store_name,
                indexed_properties=_CHURN_INDEXED_PROPERTIES,
            )
        )
        await store.startup()
        stores.append(store)
    return stores


async def _churn(stores: list[QdrantVectorStore], workers: list[_ChurnWorker]) -> None:
    """Run the workers to completion beside a purger per store, then drain."""
    [deleted] = {worker.deleted for worker in workers}
    quiesced = False

    async def purge(store: QdrantVectorStore) -> None:
        while True:
            await deleted.wait()
            if quiesced:
                return
            deleted.clear()
            await _drain(store)

    async with asyncio.timeout(_CHURN_DEADLINE_SECONDS):
        async with asyncio.TaskGroup() as purgers:
            for store in stores:
                purgers.create_task(purge(store))
            async with asyncio.TaskGroup() as working:
                for worker in workers:
                    working.create_task(worker.run())
            quiesced = True
            deleted.set()
        for store in stores:
            await _drain(store)


async def _assert_churned_state(
    store: QdrantVectorStore, workers: list[_ChurnWorker]
) -> None:
    """Each live partition holds what its workers wrote and kept; a deleted one holds nothing written successfully."""
    live: set[UUID] = set()
    for key in _CHURN_KEYS:
        handle = await store.get_partition(key)
        if handle is None:
            continue
        live.add(handle._incarnation)
        expected = {
            record_uuid
            for worker in workers
            for record_uuid in worker.lives[handle._incarnation]
        }
        assert await _stored_uuids(handle) == expected, key
        [result] = await handle.query(
            query_vectors=[_MODEL_PROBE], limit=_CHURN_WORKERS * _CHURN_OWNED
        )
        assert {match.record_uuid for match in result.matches} == expected, key

    unconfirmed = set().union(*(worker.unconfirmed for worker in workers))
    dead = {point for point in await _stored_points(store) if point[0] not in live}
    assert dead <= unconfirmed


@pytest.mark.integration
class TestConcurrentChurn:
    @pytest.mark.asyncio
    async def test_two_stores_churning_one_registry_keep_every_partition_exact(
        self, qdrant_client, qdrant_grpc_client, registry_engines
    ):
        """Two stores of one name, on a REST and a gRPC client and their own
        engines on one registry database, serve workers that upsert, delete
        and query their own records and delete and create partitions of three
        keys, while each store's purger drains whenever a partition is
        deleted.

        Every operation succeeds or raises a documented domain error, and
        nothing deadlocks. A query's matches agree with its worker's records
        whenever the partition was live throughout it. Once the workers stop
        and the purge is drained, each live partition holds exactly what its
        workers wrote and did not delete, and no deleted partition's record
        remains but one whose upsert raised because the partition was
        deleted while it was in flight: such a write can land after a purge
        round found the incarnation empty, the race the tombstone retention
        closes, and the test's retention is zero.
        """
        stores = await _stores_sharing_a_registry(
            [qdrant_client, qdrant_grpc_client], registry_engines
        )
        pool = sorted(uuid4() for _ in range(_CHURN_WORKERS * _CHURN_OWNED))
        deleted = asyncio.Event()
        workers = [
            _ChurnWorker(
                index=index,
                store=stores[index % len(stores)],
                owned=pool[index * _CHURN_OWNED : (index + 1) * _CHURN_OWNED],
                rng=random.Random(_CHURN_SEED + index),
                deleted=deleted,
            )
            for index in range(_CHURN_WORKERS)
        ]

        await _churn(stores, workers)
        await _assert_churned_state(stores[0], workers)

        for key in _CHURN_KEYS:
            await stores[0].delete_partition(key)
        await _drain(stores[0])


# ── Metrics ──


@pytest.mark.integration
class TestMetrics:
    @pytest.mark.asyncio
    async def test_metrics_collection(self, qdrant_client, registry_engine):
        mock_factory = MagicMock(spec=MetricsFactory)
        mock_histogram = MagicMock(spec=MetricsFactory.Histogram)
        mock_factory.get_histogram.return_value = mock_histogram

        store = QdrantVectorStore(
            await _params(qdrant_client, registry_engine, metrics_factory=mock_factory)
        )
        await store.startup()

        await store.create_partition("metrics_test")

        assert mock_histogram.observe.called
        call_labels = mock_histogram.observe.call_args
        assert call_labels[1]["labels"]["operation"] == "create_partition"
        assert call_labels[1]["labels"]["status"] == "ok"

        mock_histogram.reset_mock()

        coll = await store.get_partition("metrics_test")
        assert coll is not None
        v1 = _normalize([1.0, 0.0, 0.0])
        r1 = _make_record(vector=v1)
        await coll.upsert(records=[r1])

        assert mock_histogram.observe.called
        call_labels = mock_histogram.observe.call_args
        assert call_labels[1]["labels"]["operation"] == "upsert"
        assert call_labels[1]["labels"]["status"] == "ok"

        await store.delete_partition("metrics_test")


@pytest.mark.integration
class TestCollectionProvisioningAcrossWorkers:
    """Startup and partition creation have to survive more than one worker.

    Stores on separate clients can share one registry database and one
    native collection, so two can start up and prepare the native collection
    at the same moment, one can find it already there without its indexes,
    and two can decide to create the same partition at once.
    """

    @pytest.mark.asyncio
    async def test_indexes_are_created_when_the_collection_already_exists(
        self, qdrant_client, registry_engine
    ):
        """A collection that exists without its indexes still gets them.

        A second store starting, another worker's or a retry after one died
        between the two calls, finds the collection there and creates the
        indexes it lacks.
        """
        native, name = "raced", "raced_name"

        # Stand in for a startup that got as far as the collection and no further.
        await qdrant_client.create_collection(
            collection_name=native,
            vectors_config=models.VectorParams(
                size=VECTOR_DIM, distance=models.Distance.COSINE
            ),
        )

        store = QdrantVectorStore(
            await _params(qdrant_client, registry_engine, vector_store_name=native)
        )
        await store.startup()
        try:
            await store.open_or_create_partition(name)
            info = await qdrant_client.get_collection(native)
            indexed = set(info.payload_schema or {})
            assert _PAYLOAD_INCARNATION in indexed, (
                "the tenant incarnation index is missing: a collection that already "
                "existed never had its payload indexes created, so tenant "
                f"filtering is unindexed. present: {sorted(indexed)}"
            )
            assert "name" in indexed, (
                f"declared property index absent. present: {sorted(indexed)}"
            )
        finally:
            await store.delete_partition(name)
            await qdrant_client.delete_collection(native)

    @staticmethod
    def _together(monkeypatch, targets: list, attribute: str) -> None:
        """Make each target's `attribute` wait until every target has called it, so the calls overlap."""
        barrier = asyncio.Barrier(len(targets))
        for target in targets:
            method = getattr(target, attribute)

            async def together(*args, method=method, **kwargs):
                await barrier.wait()
                return await method(*args, **kwargs)

            monkeypatch.setattr(target, attribute, together)

    @pytest.mark.asyncio
    async def test_two_workers_creating_at_once_agree_on_one_partition(
        self, new_qdrant_client, registry_engine, monkeypatch
    ):
        """Two clients, one registry - the multi-worker shape, in one process.

        Both workers find the key free and reserve it at once. Both
        open-or-creates return a handle on the one partition, so a record
        written through either is read through the other, and of a strict
        create both issue at once exactly one succeeds.
        """
        client_a = new_qdrant_client()
        client_b = new_qdrant_client()
        native, name = "race_two", "race_two_name"

        store_a = QdrantVectorStore(
            await _params(client_a, registry_engine, vector_store_name=native)
        )
        store_b = QdrantVectorStore(
            await _params(client_b, registry_engine, vector_store_name=native)
        )
        await store_a.startup()
        await store_b.startup()
        self._together(
            monkeypatch,
            [store_a._partition_registry, store_b._partition_registry],
            "reserve",
        )

        try:
            async with asyncio.timeout(60):
                results = await asyncio.gather(
                    store_a.open_or_create_partition(name),
                    store_b.open_or_create_partition(name),
                    return_exceptions=True,
                )
            handles = [r for r in results if isinstance(r, QdrantVectorStorePartition)]
            assert len(handles) == 2, f"a concurrent creator raised: {results!r}"
            record = _make_record(vector=_normalize([1.0, 0.0, 0.0]))
            await handles[0].upsert(records=[record])
            [result] = await handles[1].query(query_vectors=[record.vector], limit=10)
            assert [match.record_uuid for match in result.matches] == [record.uuid]

            # The registry's primary key arbitrates a strict create: one
            # creator wins, the other gets AlreadyExists.
            await store_a.delete_partition(name)
            async with asyncio.timeout(60):
                results = await asyncio.gather(
                    store_a.create_partition(name),
                    store_b.create_partition(name),
                    return_exceptions=True,
                )
            assert sorted(type(r).__name__ for r in results) == [
                "NoneType",
                "VectorStorePartitionAlreadyExistsError",
            ], results
        finally:
            await store_a.delete_partition(name)
            await client_a.delete_collection(native)
            await client_a.close()
            await client_b.close()

    @pytest.mark.asyncio
    @pytest.mark.parametrize("prefer_grpc", [False, True], ids=["rest", "grpc"])
    async def test_two_stores_preparing_their_storage_at_once_both_succeed(
        self, new_qdrant_client, registry_engine, monkeypatch, prefer_grpc
    ):
        """Two workers start stores of one name at once, so both prepare the
        native collection they share at the same moment. Both startups
        succeed, and a partition each worker creates holds its own records."""
        client_a = new_qdrant_client(prefer_grpc=prefer_grpc)
        client_b = new_qdrant_client(prefer_grpc=prefer_grpc)
        native = f"race_storage_{'grpc' if prefer_grpc else 'rest'}"

        store_a = QdrantVectorStore(
            await _params(client_a, registry_engine, vector_store_name=native)
        )
        store_b = QdrantVectorStore(
            await _params(client_b, registry_engine, vector_store_name=native)
        )
        self._together(monkeypatch, [store_a, store_b], "_prepare_storage")

        try:
            async with asyncio.timeout(60):
                await asyncio.gather(store_a.startup(), store_b.startup())
            await store_a.create_partition("first")
            await store_b.create_partition("second")
            first = await store_b.get_partition("first")
            second = await store_a.get_partition("second")
            assert first is not None
            assert second is not None
            records = {
                partition: _make_record(vector=_normalize([1.0, 0.0, 0.0]))
                for partition in (first, second)
            }
            for partition, record in records.items():
                await partition.upsert(records=[record])
            for partition, record in records.items():
                [result] = await partition.query(
                    query_vectors=[record.vector], limit=10
                )
                assert [match.record_uuid for match in result.matches] == [record.uuid]
        finally:
            await client_a.delete_collection(native)
            await client_a.close()
            await client_b.close()


class TestLifecycleContract(PartitionLifecycleContract):
    """The partition lifecycle contract, against this store."""

    count_stored = staticmethod(_count_stored)
    stored_uuids = staticmethod(_stored_uuids)

    @staticmethod
    async def settle(partition) -> None:
        # A Qdrant write returns once applied, so reads reflect it already.
        pass
