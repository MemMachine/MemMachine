"""Tests for QdrantVectorStore."""

import asyncio
import math
import operator
import random
from datetime import UTC, datetime, timedelta, timezone
from typing import Any
from unittest.mock import AsyncMock, MagicMock
from uuid import UUID, uuid4

import httpx
import pytest
import pytest_asyncio
from qdrant_client import AsyncQdrantClient, models
from qdrant_client.http.exceptions import ResponseHandlingException, UnexpectedResponse

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
    QueryResult,
    Record,
    VectorStoreSchema,
)
from memmachine_server.common.vector_store.qdrant_vector_store import (
    _PAYLOAD_PARTITION_UUID,
    _PAYLOAD_RECORD_UUID,
    QdrantVectorStore,
    QdrantVectorStoreParams,
    QdrantVectorStorePartition,
)
from server_tests.memmachine_server.common.vector_store.declared_schema_contract import (
    DeclaredSchemaContract,
)
from server_tests.memmachine_server.common.vector_store.partition_lifecycle_contract import (
    PartitionLifecycleContract,
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


async def _stored_uuids(partition) -> set[UUID]:
    """Record UUIDs Qdrant holds under the handle's partition UUID, read past the store."""
    points, _ = await partition._client.scroll(
        collection_name=partition._vector_store_name,
        scroll_filter=models.Filter(
            must=[
                models.FieldCondition(
                    key=_PAYLOAD_PARTITION_UUID,
                    match=models.MatchValue(value=str(partition._partition_uuid)),
                )
            ]
        ),
        limit=10000,
        with_payload=[_PAYLOAD_RECORD_UUID],
        with_vectors=False,
    )
    return {UUID(str((point.payload or {})[_PAYLOAD_RECORD_UUID])) for point in points}


async def _count_stored(store: QdrantVectorStore) -> int:
    """Points Qdrant holds in the store's native collection, every partition's included."""
    result = await store._client.count(
        collection_name=store._vector_store_name, exact=True
    )
    return result.count


# Purge calls `_purge` makes before it fails the test: far more than the
# records a test writes need, so only a purge that keeps finding records
# reaches it.
_MAX_PURGE_CALLS = 1000


async def _open(store: QdrantVectorStore) -> QdrantVectorStorePartition:
    """A handle on a freshly prepared partition."""
    partition_uuid = uuid4()
    await store.create_partition(partition_uuid)
    return await store.open_partition(partition_uuid)


async def _purge(store: QdrantVectorStore, partition_uuid: UUID) -> None:
    """Purge a partition UUID until the store finds nothing of it."""
    for _ in range(_MAX_PURGE_CALLS):
        if not await store.purge_partition(partition_uuid):
            return
    pytest.fail(
        f"partition UUID {partition_uuid} still held records after "
        f"{_MAX_PURGE_CALLS} purge calls"
    )


@pytest.fixture(
    params=[
        pytest.param("qdrant_client", marks=pytest.mark.integration),
        pytest.param("qdrant_grpc_client", marks=pytest.mark.integration),
    ],
)
def any_qdrant_client(request):
    return request.getfixturevalue(request.param)


def _schema(**overrides) -> VectorStoreSchema:
    fields: dict[str, Any] = {
        "vector_dimensions": VECTOR_DIM,
        "similarity_metric": SimilarityMetric.COSINE,
        "indexed_properties": INDEXED_PROPERTIES,
    }
    fields.update(overrides)
    return VectorStoreSchema(**fields)


def _params(client, **overrides) -> QdrantVectorStoreParams:
    params: dict[str, Any] = {
        "client": client,
        "vector_store_name": VECTOR_STORE_NAME,
        "store_schema": _schema(),
    }
    params.update(overrides)
    return QdrantVectorStoreParams(**params)


@pytest_asyncio.fixture
async def store(any_qdrant_client):
    s = QdrantVectorStore(_params(any_qdrant_client))
    await s.startup()
    yield s


@pytest_asyncio.fixture
async def partition(store):
    partition = await _open(store)
    yield partition
    await _purge(store, partition._partition_uuid)


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

        query_results = list(await partition.query(query_vectors=[v1], limit=3))
        matches = query_results[0].matches

        assert len(matches) == 3
        assert matches[0].record_uuid == r1.uuid
        assert matches[1].record_uuid == r3.uuid
        assert matches[2].record_uuid == r2.uuid
        assert matches[0].score >= matches[1].score >= matches[2].score

    @pytest.mark.asyncio
    async def test_query_with_similarity_threshold(self, partition):
        v1 = _normalize([1.0, 0.0, 0.0])
        v2 = _normalize([0.0, 1.0, 0.0])

        r1 = _make_record(vector=v1)
        r2 = _make_record(vector=v2)

        await partition.upsert(records=[r1, r2])

        query_results = list(
            await partition.query(query_vectors=[v1], limit=10, score_threshold=0.9)
        )
        matches = query_results[0].matches

        assert len(matches) == 1
        assert matches[0].record_uuid == r1.uuid

    @pytest.mark.asyncio
    async def test_query_with_limit(self, partition):
        vectors = [_normalize([1.0, float(i) * 0.01, 0.0]) for i in range(5)]
        records = [_make_record(vector=v) for v in vectors]
        await partition.upsert(records=records)

        query_results = list(await partition.query(query_vectors=[vectors[0]], limit=2))
        assert len(query_results[0].matches) == 2

    @pytest.mark.asyncio
    async def test_query_batch_multiple_vectors(self, partition):
        v1 = _normalize([1.0, 0.0, 0.0])
        v2 = _normalize([0.0, 1.0, 0.0])

        r1 = _make_record(vector=v1, properties={"name": "a"})
        r2 = _make_record(vector=v2, properties={"name": "b"})
        await partition.upsert(records=[r1, r2])

        all_results = list(await partition.query(query_vectors=[v1, v2], limit=1))

        assert len(all_results) == 2
        assert all_results[0].matches[0].record_uuid == r1.uuid
        assert all_results[1].matches[0].record_uuid == r2.uuid

    @pytest.mark.asyncio
    async def test_query_empty_vectors(self, partition):
        all_results = list(await partition.query(query_vectors=[], limit=10))
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
        self, any_qdrant_client, metric
    ):
        """Matches come best first, each scored as QueryMatch defines for the
        store's metric, and a score threshold keeps the matches on its
        better side: above it for a similarity, below it for a distance."""
        store = QdrantVectorStore(
            _params(
                any_qdrant_client,
                vector_store_name=f"metric_{metric.value}",
                store_schema=_schema(similarity_metric=metric),
            )
        )
        await store.startup()
        partition = await _open(store)
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

        await _purge(store, partition._partition_uuid)

    @pytest.mark.asyncio
    @pytest.mark.parametrize("metric", list(SimilarityMetric))
    async def test_a_threshold_keeps_a_match_scoring_exactly_it(
        self, any_qdrant_client, metric
    ):
        """A score threshold equal to a match's reported score keeps the
        match; one a step better than that score drops it."""
        store = QdrantVectorStore(
            _params(
                any_qdrant_client,
                vector_store_name=f"metric_{metric.value}",
                store_schema=_schema(similarity_metric=metric),
            )
        )
        await store.startup()
        partition = await _open(store)
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
        await _purge(store, partition._partition_uuid)


def _partition_on(client: AsyncQdrantClient) -> QdrantVectorStorePartition:
    """A handle on a given client, bound to a fresh partition UUID."""
    return QdrantVectorStorePartition(
        client=client,
        vector_store_name=VECTOR_STORE_NAME,
        partition_uuid=uuid4(),
        store_schema=VectorStoreSchema(
            vector_dimensions=VECTOR_DIM,
            similarity_metric=SimilarityMetric.COSINE,
            indexed_properties={},
        ),
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


class TestDeclaredSchema(DeclaredSchemaContract):
    """The declared-schema contract, against this store."""

    @staticmethod
    async def settle(partition) -> None:
        # A Qdrant write returns once applied, so reads reflect it already.
        pass


class TestFilters:
    # alice=30/9.5/True, bob=25/7.0/False, carol=35/8.0/True
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

    # scores: r0=-1.5, r1=0.0, r2=0.5, r3=1.5, r4=2.0
    async def _setup_floats(self, partition):
        vectors = [_normalize([1.0, float(i) * 0.01, 0.0]) for i in range(5)]
        scores = [-1.5, 0.0, 0.5, 1.5, 2.0]
        records = [
            _make_record(vector=v, properties={"score": s})
            for v, s in zip(vectors, scores, strict=True)
        ]
        await partition.upsert(records=records)
        return records, vectors[0]

    # dts: r0=Jan, r1=Mar, r2=Jun, r3=Sep, r4=Dec
    async def _setup_datetimes(self, partition):
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
        await partition.upsert(records=records)
        return records, vectors[0], dts

    async def _query(self, partition, query_vec, field, op, value):
        all_results = list(
            await partition.query(
                query_vectors=[query_vec],
                limit=10,
                property_filter=Comparison(field=field, op=op, value=value),
            )
        )
        return {m.record_uuid for m in all_results[0].matches}

    # ── String / int ──

    @pytest.mark.asyncio
    async def test_eq_str(self, partition):
        r1, _r2, _r3, v1 = await self._setup(partition)
        query_results = list(
            await partition.query(
                query_vectors=[v1],
                limit=10,
                property_filter=Comparison(field="name", op="=", value="alice"),
            )
        )
        matches = query_results[0].matches
        assert len(matches) == 1
        assert matches[0].record_uuid == r1.uuid

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
        query_results = list(
            await partition.query(
                query_vectors=[v1],
                limit=10,
                property_filter=Comparison(field="age", op=">", value=30),
            )
        )
        matches = query_results[0].matches
        assert len(matches) == 1
        assert matches[0].record_uuid == r3.uuid

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
        query_results = list(
            await partition.query(
                query_vectors=[v1],
                limit=10,
                property_filter=Comparison(field="age", op="<", value=30),
            )
        )
        matches = query_results[0].matches
        assert len(matches) == 1
        assert matches[0].record_uuid == r2.uuid

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
    async def test_eq_float_fractional(self, partition):
        records, qv = await self._setup_floats(partition)
        uuids = await self._query(partition, qv, "score", "=", 0.5)
        assert records[2].uuid in uuids
        assert len(uuids) == 1

    @pytest.mark.asyncio
    async def test_eq_float_whole_number(self, partition):
        records, qv = await self._setup_floats(partition)
        uuids = await self._query(partition, qv, "score", "=", 2.0)
        assert records[4].uuid in uuids
        assert len(uuids) == 1

    @pytest.mark.asyncio
    async def test_eq_float_negative(self, partition):
        records, qv = await self._setup_floats(partition)
        uuids = await self._query(partition, qv, "score", "=", -1.5)
        assert records[0].uuid in uuids
        assert len(uuids) == 1

    @pytest.mark.asyncio
    async def test_eq_float_zero(self, partition):
        records, qv = await self._setup_floats(partition)
        uuids = await self._query(partition, qv, "score", "=", 0.0)
        assert records[1].uuid in uuids
        assert len(uuids) == 1

    @pytest.mark.asyncio
    async def test_ne_float_fractional(self, partition):
        records, qv = await self._setup_floats(partition)
        uuids = await self._query(partition, qv, "score", "!=", 0.5)
        assert records[2].uuid not in uuids
        assert len(uuids) == 4

    @pytest.mark.asyncio
    async def test_ne_float_whole_number(self, partition):
        records, qv = await self._setup_floats(partition)
        uuids = await self._query(partition, qv, "score", "!=", 2.0)
        assert records[4].uuid not in uuids
        assert len(uuids) == 4

    @pytest.mark.asyncio
    async def test_ne_float_negative(self, partition):
        records, qv = await self._setup_floats(partition)
        uuids = await self._query(partition, qv, "score", "!=", -1.5)
        assert records[0].uuid not in uuids
        assert len(uuids) == 4

    @pytest.mark.asyncio
    async def test_gt_float(self, partition):
        records, qv = await self._setup_floats(partition)
        uuids = await self._query(partition, qv, "score", ">", 0.5)
        assert records[0].uuid not in uuids  # -1.5
        assert records[1].uuid not in uuids  # 0.0
        assert records[2].uuid not in uuids  # 0.5 not strictly greater
        assert records[3].uuid in uuids  # 1.5
        assert records[4].uuid in uuids  # 2.0

    @pytest.mark.asyncio
    async def test_gte_float(self, partition):
        records, qv = await self._setup_floats(partition)
        uuids = await self._query(partition, qv, "score", ">=", 0.5)
        assert records[0].uuid not in uuids  # -1.5
        assert records[1].uuid not in uuids  # 0.0
        assert records[2].uuid in uuids  # 0.5
        assert records[3].uuid in uuids  # 1.5
        assert records[4].uuid in uuids  # 2.0

    @pytest.mark.asyncio
    async def test_lt_float(self, partition):
        records, qv = await self._setup_floats(partition)
        uuids = await self._query(partition, qv, "score", "<", 0.5)
        assert records[0].uuid in uuids  # -1.5
        assert records[1].uuid in uuids  # 0.0
        assert records[2].uuid not in uuids  # 0.5 not strictly less
        assert records[3].uuid not in uuids  # 1.5
        assert records[4].uuid not in uuids  # 2.0

    @pytest.mark.asyncio
    async def test_lte_float(self, partition):
        records, qv = await self._setup_floats(partition)
        uuids = await self._query(partition, qv, "score", "<=", 0.5)
        assert records[0].uuid in uuids  # -1.5
        assert records[1].uuid in uuids  # 0.0
        assert records[2].uuid in uuids  # 0.5
        assert records[3].uuid not in uuids  # 1.5
        assert records[4].uuid not in uuids  # 2.0

    @pytest.mark.asyncio
    async def test_gt_float_from_negative(self, partition):
        records, qv = await self._setup_floats(partition)
        uuids = await self._query(partition, qv, "score", ">", -1.5)
        assert records[0].uuid not in uuids  # -1.5 not strictly greater
        assert records[1].uuid in uuids  # 0.0
        assert records[2].uuid in uuids  # 0.5
        assert records[3].uuid in uuids  # 1.5
        assert records[4].uuid in uuids  # 2.0

    @pytest.mark.asyncio
    async def test_lt_float_zero(self, partition):
        records, qv = await self._setup_floats(partition)
        uuids = await self._query(partition, qv, "score", "<", 0.0)
        assert records[0].uuid in uuids  # -1.5
        assert records[1].uuid not in uuids  # 0.0 not strictly less
        assert records[2].uuid not in uuids
        assert records[3].uuid not in uuids
        assert records[4].uuid not in uuids

    # ── Datetime ──

    @pytest.mark.asyncio
    async def test_eq_datetime(self, partition):
        records, qv, dts = await self._setup_datetimes(partition)
        uuids = await self._query(partition, qv, "created_at", "=", dts[2])
        assert records[2].uuid in uuids
        assert len(uuids) == 1

    @pytest.mark.asyncio
    async def test_ne_datetime(self, partition):
        records, qv, dts = await self._setup_datetimes(partition)
        uuids = await self._query(partition, qv, "created_at", "!=", dts[2])
        assert records[2].uuid not in uuids
        assert len(uuids) == 4

    @pytest.mark.asyncio
    async def test_gt_datetime(self, partition):
        records, qv, dts = await self._setup_datetimes(partition)
        uuids = await self._query(partition, qv, "created_at", ">", dts[2])
        assert records[0].uuid not in uuids
        assert records[1].uuid not in uuids
        assert records[2].uuid not in uuids  # not strictly greater
        assert records[3].uuid in uuids
        assert records[4].uuid in uuids

    @pytest.mark.asyncio
    async def test_gte_datetime(self, partition):
        records, qv, dts = await self._setup_datetimes(partition)
        uuids = await self._query(partition, qv, "created_at", ">=", dts[2])
        assert records[0].uuid not in uuids
        assert records[1].uuid not in uuids
        assert records[2].uuid in uuids
        assert records[3].uuid in uuids
        assert records[4].uuid in uuids

    @pytest.mark.asyncio
    async def test_lt_datetime(self, partition):
        records, qv, dts = await self._setup_datetimes(partition)
        uuids = await self._query(partition, qv, "created_at", "<", dts[2])
        assert records[0].uuid in uuids
        assert records[1].uuid in uuids
        assert records[2].uuid not in uuids  # not strictly less
        assert records[3].uuid not in uuids
        assert records[4].uuid not in uuids

    @pytest.mark.asyncio
    async def test_lte_datetime(self, partition):
        records, qv, dts = await self._setup_datetimes(partition)
        uuids = await self._query(partition, qv, "created_at", "<=", dts[2])
        assert records[0].uuid in uuids
        assert records[1].uuid in uuids
        assert records[2].uuid in uuids
        assert records[3].uuid not in uuids
        assert records[4].uuid not in uuids

    @pytest.mark.asyncio
    async def test_datetime_microseconds_eq(self, partition):
        """Equality filter distinguishes microsecond precision."""
        v1 = _normalize([1.0, 0.0, 0.0])
        v2 = _normalize([1.0, 0.1, 0.0])
        dt1 = datetime(2024, 6, 15, 12, 30, 45, 123456, tzinfo=UTC)
        dt2 = datetime(2024, 6, 15, 12, 30, 45, 999999, tzinfo=UTC)
        r1 = _make_record(vector=v1, properties={"name": "a", "created_at": dt1})
        r2 = _make_record(vector=v2, properties={"name": "b", "created_at": dt2})
        await partition.upsert(records=[r1, r2])

        uuids = await self._query(partition, v1, "created_at", "=", dt1)
        assert r1.uuid in uuids
        assert r2.uuid not in uuids

    @pytest.mark.asyncio
    async def test_datetime_boundary_inclusive(self, partition):
        """gte and lte are both inclusive at the exact boundary."""
        v1 = _normalize([1.0, 0.0, 0.0])
        v2 = _normalize([1.0, 0.1, 0.0])
        boundary = datetime(2024, 6, 1, tzinfo=UTC)
        before = datetime(2024, 5, 31, 23, 59, 59, tzinfo=UTC)
        r1 = _make_record(vector=v1, properties={"name": "b", "created_at": boundary})
        r2 = _make_record(vector=v2, properties={"name": "a", "created_at": before})
        await partition.upsert(records=[r1, r2])

        gte_uuids = await self._query(partition, v1, "created_at", ">=", boundary)
        lte_uuids = await self._query(partition, v1, "created_at", "<=", boundary)
        assert r1.uuid in gte_uuids
        assert r2.uuid not in gte_uuids
        assert r1.uuid in lte_uuids
        assert r2.uuid in lte_uuids

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
    async def test_gte_datetime_cross_timezone(self, partition):
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
        await partition.upsert(records=[r1, r2, r3])

        # 2024-06-01 00:00 PST = 2024-06-01 08:00 UTC
        pst = timezone(timedelta(hours=-8))
        cutoff = datetime(2024, 6, 1, 0, 0, 0, tzinfo=pst)
        uuids = await self._query(partition, v1, "created_at", ">=", cutoff)
        assert r1.uuid not in uuids
        assert r2.uuid in uuids
        assert r3.uuid in uuids

    @pytest.mark.asyncio
    async def test_eq_naive_datetime(self, partition):
        """Equality filter works for naive datetimes."""
        v1 = _normalize([1.0, 0.0, 0.0])
        naive_dt = datetime(2024, 6, 15, 12, 0, 0, tzinfo=UTC).replace(tzinfo=None)
        r1 = _make_record(vector=v1, properties={"name": "n", "created_at": naive_dt})
        await partition.upsert(records=[r1])

        uuids = await self._query(partition, v1, "created_at", "=", naive_dt)
        assert r1.uuid in uuids
        assert len(uuids) == 1

    @pytest.mark.asyncio
    async def test_ne_naive_datetime(self, partition):
        """Not-equal filter works for naive datetimes."""
        v1 = _normalize([1.0, 0.0, 0.0])
        v2 = _normalize([1.0, 0.1, 0.0])
        naive1 = datetime(2024, 1, 1, tzinfo=UTC).replace(tzinfo=None)
        naive2 = datetime(2024, 6, 15, tzinfo=UTC).replace(tzinfo=None)
        r1 = _make_record(vector=v1, properties={"name": "a", "created_at": naive1})
        r2 = _make_record(vector=v2, properties={"name": "b", "created_at": naive2})
        await partition.upsert(records=[r1, r2])

        uuids = await self._query(partition, v1, "created_at", "!=", naive1)
        assert r1.uuid not in uuids
        assert r2.uuid in uuids

    @pytest.mark.asyncio
    async def test_gt_naive_datetime(self, partition):
        """gt filter works for naive datetimes."""
        v1 = _normalize([1.0, 0.0, 0.0])
        v2 = _normalize([1.0, 0.1, 0.0])
        naive1 = datetime(2024, 1, 1, tzinfo=UTC).replace(tzinfo=None)
        naive2 = datetime(2024, 6, 15, 12, 0, 0, tzinfo=UTC).replace(tzinfo=None)
        r1 = _make_record(vector=v1, properties={"name": "a", "created_at": naive1})
        r2 = _make_record(vector=v2, properties={"name": "b", "created_at": naive2})
        await partition.upsert(records=[r1, r2])

        cutoff = datetime(2024, 3, 1, tzinfo=UTC).replace(tzinfo=None)
        uuids = await self._query(partition, v1, "created_at", ">", cutoff)
        assert r1.uuid not in uuids
        assert r2.uuid in uuids

    @pytest.mark.asyncio
    async def test_lt_naive_datetime(self, partition):
        """lt filter works for naive datetimes."""
        v1 = _normalize([1.0, 0.0, 0.0])
        v2 = _normalize([1.0, 0.1, 0.0])
        naive1 = datetime(2024, 1, 1, tzinfo=UTC).replace(tzinfo=None)
        naive2 = datetime(2024, 6, 15, tzinfo=UTC).replace(tzinfo=None)
        r1 = _make_record(vector=v1, properties={"name": "a", "created_at": naive1})
        r2 = _make_record(vector=v2, properties={"name": "b", "created_at": naive2})
        await partition.upsert(records=[r1, r2])

        cutoff = datetime(2024, 3, 1, tzinfo=UTC).replace(tzinfo=None)
        uuids = await self._query(partition, v1, "created_at", "<", cutoff)
        assert r1.uuid in uuids
        assert r2.uuid not in uuids

    # ── IsNull ──

    @pytest.mark.asyncio
    async def test_is_null(self, partition):
        v1 = _normalize([1.0, 0.0, 0.0])
        v2 = _normalize([1.0, 0.1, 0.0])
        v3 = _normalize([1.0, 0.2, 0.0])
        v4 = _normalize([1.0, 0.3, 0.0])
        r_has_value = _make_record(vector=v1, properties={"name": "has_name"})
        r_explicit_none = _make_record(vector=v2, properties={})
        r_key_missing = _make_record(vector=v3, properties={"age": 25})
        r_no_payload = _make_record(vector=v4, properties=None)
        await partition.upsert(
            records=[r_has_value, r_explicit_none, r_key_missing, r_no_payload],
        )

        query_results = list(
            await partition.query(
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
    async def test_is_not_null(self, partition):
        v1 = _normalize([1.0, 0.0, 0.0])
        v2 = _normalize([1.0, 0.1, 0.0])
        v3 = _normalize([1.0, 0.2, 0.0])
        v4 = _normalize([1.0, 0.3, 0.0])
        r_has_value = _make_record(vector=v1, properties={"name": "has_name"})
        r_explicit_none = _make_record(vector=v2, properties={})
        r_key_missing = _make_record(vector=v3, properties={"age": 25})
        r_no_payload = _make_record(vector=v4, properties=None)
        await partition.upsert(
            records=[r_has_value, r_explicit_none, r_key_missing, r_no_payload],
        )

        query_results = list(
            await partition.query(
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
    async def test_in(self, partition):
        r1, _r2, r3, v1 = await self._setup(partition)
        query_results = list(
            await partition.query(
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
    async def test_and(self, partition):
        _r1, _r2, r3, v1 = await self._setup(partition)
        query_results = list(
            await partition.query(
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
    async def test_or(self, partition):
        r1, _r2, r3, v1 = await self._setup(partition)
        query_results = list(
            await partition.query(
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
    async def test_not(self, partition):
        r1, r2, _r3, v1 = await self._setup(partition)
        query_results = list(
            await partition.query(
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


class TestDelete:
    @pytest.mark.asyncio
    async def test_delete_records(self, partition):
        v1 = _normalize([1.0, 0.0, 0.0])
        v2 = _normalize([0.0, 1.0, 0.0])

        r1 = _make_record(vector=v1)
        r2 = _make_record(vector=v2)

        await partition.upsert(records=[r1, r2])
        await partition.delete(record_uuids=[r1.uuid])

        assert await _stored_uuids(partition) == {r2.uuid}


# ── Partition isolation ──


class TestPartitionIsolation:
    @pytest.mark.asyncio
    async def test_partitions_share_the_store_collection(self, store):
        """Every partition's records are points in the store's one native collection."""
        stored_before = await _count_stored(store)
        partitions = [await _open(store) for _ in range(2)]
        for partition in partitions:
            await partition.upsert(
                records=[_make_record(vector=_normalize([1.0, 0.0, 0.0]))]
            )

        assert await _count_stored(store) == stored_before + 2

        for partition in partitions:
            await _purge(store, partition._partition_uuid)

    @pytest.mark.asyncio
    async def test_query_only_returns_own_partition(self, store):
        partition_a = await _open(store)
        partition_b = await _open(store)

        v1 = _normalize([1.0, 0.0, 0.0])
        r1 = _make_record(vector=v1, properties={})
        r2 = _make_record(vector=v1, properties={})

        await partition_a.upsert(records=[r1])
        await partition_b.upsert(records=[r2])

        results_a = list(await partition_a.query(query_vectors=[v1], limit=10))
        results_b = list(await partition_b.query(query_vectors=[v1], limit=10))

        uuids_a = {m.record_uuid for m in results_a[0].matches}
        uuids_b = {m.record_uuid for m in results_b[0].matches}
        assert uuids_a == {r1.uuid}
        assert uuids_b == {r2.uuid}

        await _purge(store, partition_a._partition_uuid)
        await _purge(store, partition_b._partition_uuid)

    @pytest.mark.asyncio
    async def test_a_query_returns_only_its_own_partitions_records(self, store):
        partition_a = await _open(store)
        partition_b = await _open(store)

        v1 = _normalize([1.0, 0.0, 0.0])
        r1 = _make_record(vector=v1)
        r2 = _make_record(vector=v1)

        await partition_a.upsert(records=[r1])
        await partition_b.upsert(records=[r2])

        [result] = await partition_a.query(query_vectors=[v1], limit=10)
        assert [match.record_uuid for match in result.matches] == [r1.uuid]

        await _purge(store, partition_a._partition_uuid)
        await _purge(store, partition_b._partition_uuid)

    @pytest.mark.asyncio
    async def test_a_filtered_query_returns_only_its_own_partitions_records(
        self, any_qdrant_client
    ):
        """Two partitions share the store's native collection and hold
        records that match the same filters; a filtered query returns its
        own records."""
        # A name of its own, declaring `note` beside the shared keys.
        store = QdrantVectorStore(
            _params(
                any_qdrant_client,
                vector_store_name="filtered_isolation",
                store_schema=_schema(
                    indexed_properties={**INDEXED_PROPERTIES, "note": str}
                ),
            )
        )
        await store.startup()
        partition_a = await _open(store)
        partition_b = await _open(store)

        vector = _normalize([1.0, 0.0, 0.0])
        properties: dict[str, PropertyValue] = {
            "name": "alice",
            "age": 30,
            "note": "shared",
        }
        records_a = [_make_record(vector=vector, properties=properties)]
        records_b = [_make_record(vector=vector, properties=properties)]
        await partition_a.upsert(records=records_a)
        await partition_b.upsert(records=records_b)

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
            for partition, own in ((partition_a, records_a), (partition_b, records_b)):
                [result] = await partition.query(
                    query_vectors=[vector], limit=10, property_filter=property_filter
                )
                assert [match.record_uuid for match in result.matches] == [
                    record.uuid for record in own
                ], property_filter

        await _purge(store, partition_a._partition_uuid)
        await _purge(store, partition_b._partition_uuid)

    @pytest.mark.asyncio
    async def test_stores_of_two_names_keep_their_records_in_separate_storage(
        self, store
    ):
        """Stores of two names share no storage, even on one client and
        under one partition and schema."""
        other = QdrantVectorStore(
            _params(store._client, vector_store_name=f"{VECTOR_STORE_NAME}_other")
        )
        await other.startup()
        partition_uuid = uuid4()
        await store.create_partition(partition_uuid)
        await other.create_partition(partition_uuid)
        writer = await store.open_partition(partition_uuid)
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

        await _purge(store, partition_uuid)
        await _purge(other, partition_uuid)

    @pytest.mark.asyncio
    async def test_the_same_uuid_in_two_partitions_is_two_records(self, store):
        partition_a = await _open(store)
        partition_b = await _open(store)

        record_uuid = uuid4()
        v1 = _normalize([1.0, 0.0, 0.0])
        await partition_a.upsert(
            records=[Record(uuid=record_uuid, vector=v1, properties={"name": "a"})]
        )
        await partition_b.upsert(
            records=[Record(uuid=record_uuid, vector=v1, properties={"name": "b"})]
        )

        assert await _stored_uuids(partition_a) == {record_uuid}
        assert await _stored_uuids(partition_b) == {record_uuid}
        [kept_a] = await partition_a.query(
            query_vectors=[v1],
            limit=10,
            property_filter=Comparison(field="name", op="=", value="a"),
        )
        assert [match.record_uuid for match in kept_a.matches] == [record_uuid]

        await partition_a.delete(record_uuids=[record_uuid])
        assert await _stored_uuids(partition_a) == set()
        assert await _stored_uuids(partition_b) == {record_uuid}

        await _purge(store, partition_a._partition_uuid)
        await _purge(store, partition_b._partition_uuid)

    @pytest.mark.asyncio
    async def test_delete_only_affects_own_partition(self, store):
        partition_a = await _open(store)
        partition_b = await _open(store)

        v1 = _normalize([1.0, 0.0, 0.0])
        r1 = _make_record(vector=v1)
        r2 = _make_record(vector=v1)

        await partition_a.upsert(records=[r1])
        await partition_b.upsert(records=[r2])

        # Attempt to delete r2 using tenant_a's partition — should not work
        await partition_a.delete(record_uuids=[r2.uuid])

        assert await _stored_uuids(partition_b) == {r2.uuid}

        await _purge(store, partition_a._partition_uuid)
        await _purge(store, partition_b._partition_uuid)


# ── Purge ──


class TestPurge:
    @pytest.mark.asyncio
    async def test_a_purge_on_a_dropped_native_collection_finds_nothing(
        self, any_qdrant_client
    ):
        """A partition UUID whose native collection is gone holds nothing: its
        purge raises nothing and finds nothing."""
        # A name of its own, so dropping its native collection spares the others.
        dropped = QdrantVectorStore(
            _params(any_qdrant_client, vector_store_name="dropped_native")
        )
        await dropped.startup()
        partition = await _open(dropped)
        await partition.upsert(
            records=[_make_record(vector=_normalize([1.0, 0.0, 0.0]))]
        )
        await any_qdrant_client.delete_collection(dropped._vector_store_name)

        assert await dropped.purge_partition(partition._partition_uuid) is False


# ── A seeded operation sequence against a model ──


_MODEL_SEED = 1735
_MODEL_STEPS = 200
_MODEL_POOL_SIZE = 12
_MODEL_STORE_NAME = "model"
_MODEL_NAMES = ("model_a", "model_b")
_MODEL_INDEXED_PROPERTIES: dict[str, PropertyType] = {
    "color": str,
    "size": int,
    "tag": str,
}
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
    """Operations on the model test's partitions, each applied to the store and to the model.

    Each key's partition is one partition UUID at a time; a partition UUID
    replaced under its key waits in `retired` until a drain purges it.
    """

    def __init__(
        self, store: QdrantVectorStore, rng: random.Random, pool: list[UUID]
    ) -> None:
        self.store = store
        self.rng = rng
        self.pool = pool
        self.handles: dict[str, QdrantVectorStorePartition] = {}
        self.model: dict[str, dict[UUID, Record]] = {}
        self.retired: list[UUID] = []
        self.baseline = 0

    async def create(self, key: str) -> None:
        self.handles[key] = await _open(self.store)
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
        self.retired.append(self.handles[key]._partition_uuid)
        await self.create(key)

    async def drain(self, key: str) -> None:
        for partition_uuid in self.retired:
            await _purge(self.store, partition_uuid)
        self.retired.clear()
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
        self, any_qdrant_client
    ):
        """Two partitions of one store take a seeded sequence of upserts
        (re-upserts change or drop properties), deletes (of their own, the
        other partition's and absent UUIDs), filtered queries, replacement by
        a fresh partition UUID under the same key, and drains that purge the
        replaced partition UUIDs. After each step the records stored, a query's
        matches and a drain's count agree with a model that holds each
        partition's records."""
        store = QdrantVectorStore(
            _params(
                any_qdrant_client,
                vector_store_name=_MODEL_STORE_NAME,
                store_schema=_schema(indexed_properties=_MODEL_INDEXED_PROPERTIES),
            )
        )
        await store.startup()
        rng = random.Random(_MODEL_SEED)
        run = _ModelRun(store, rng, sorted(uuid4() for _ in range(_MODEL_POOL_SIZE)))
        for key in _MODEL_NAMES:
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
            key = rng.choice(_MODEL_NAMES)
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

        for partition_uuid in [
            *run.retired,
            *(handle._partition_uuid for handle in run.handles.values()),
        ]:
            await _purge(store, partition_uuid)
        assert await _count_stored(store) == run.baseline


# ── Metrics ──


@pytest.mark.integration
class TestMetrics:
    @pytest.mark.asyncio
    async def test_metrics_collection(self, qdrant_client):
        mock_factory = MagicMock(spec=MetricsFactory)
        mock_histogram = MagicMock(spec=MetricsFactory.Histogram)
        mock_factory.get_histogram.return_value = mock_histogram

        store = QdrantVectorStore(_params(qdrant_client, metrics_factory=mock_factory))
        await store.startup()

        partition_uuid = uuid4()
        await store.create_partition(partition_uuid)
        partition = await store.open_partition(partition_uuid)
        mock_histogram.reset_mock()

        v1 = _normalize([1.0, 0.0, 0.0])
        r1 = _make_record(vector=v1)
        await partition.upsert(records=[r1])

        assert mock_histogram.observe.called
        call_labels = mock_histogram.observe.call_args
        assert call_labels[1]["labels"]["operation"] == "upsert"
        assert call_labels[1]["labels"]["status"] == "ok"

        mock_histogram.reset_mock()

        await _purge(store, partition_uuid)

        assert mock_histogram.observe.called
        call_labels = mock_histogram.observe.call_args
        assert call_labels[1]["labels"]["operation"] == "purge_partition"
        assert call_labels[1]["labels"]["status"] == "ok"


@pytest.mark.integration
class TestCollectionProvisioningAcrossWorkers:
    """Startup has to survive more than one worker.

    Stores of one name on separate clients share one native collection, so
    two can start up and prepare it at the same moment, and one can find it
    already there without its indexes.
    """

    @pytest.mark.asyncio
    async def test_indexes_are_created_when_the_collection_already_exists(
        self, qdrant_client
    ):
        """A collection that exists without its indexes still gets them.

        A second store starting, another worker's or a retry after one died
        between the two calls, finds the collection there and creates the
        indexes it lacks.
        """
        native = "raced"

        # Stand in for a startup that got as far as the collection and no further.
        await qdrant_client.create_collection(
            collection_name=native,
            vectors_config=models.VectorParams(
                size=VECTOR_DIM, distance=models.Distance.COSINE
            ),
        )

        store = QdrantVectorStore(_params(qdrant_client, vector_store_name=native))
        await store.startup()
        try:
            info = await qdrant_client.get_collection(native)
            indexed = set(info.payload_schema or {})
            assert _PAYLOAD_PARTITION_UUID in indexed, (
                "the tenant partition UUID index is missing: a collection that already "
                "existed never had its payload indexes created, so tenant "
                f"filtering is unindexed. present: {sorted(indexed)}"
            )
            assert "name" in indexed, (
                f"declared property index absent. present: {sorted(indexed)}"
            )
        finally:
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
    @pytest.mark.parametrize("prefer_grpc", [False, True], ids=["rest", "grpc"])
    async def test_two_stores_preparing_their_storage_at_once_both_succeed(
        self, new_qdrant_client, monkeypatch, prefer_grpc
    ):
        """Two workers start stores of one name at once, so both prepare the
        native collection they share at the same moment. Both startups
        succeed, and a partition each worker opens holds its own
        records."""
        client_a = new_qdrant_client(prefer_grpc=prefer_grpc)
        client_b = new_qdrant_client(prefer_grpc=prefer_grpc)
        native = f"race_storage_{'grpc' if prefer_grpc else 'rest'}"

        store_a = QdrantVectorStore(_params(client_a, vector_store_name=native))
        store_b = QdrantVectorStore(_params(client_b, vector_store_name=native))
        self._together(monkeypatch, [store_a, store_b], "_prepare_storage")

        try:
            async with asyncio.timeout(60):
                await asyncio.gather(store_a.startup(), store_b.startup())
            first = await _open(store_a)
            second = await _open(store_b)
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


@pytest.mark.integration
class TestStrictMode:
    """The collection is created in strict mode: a filter on an unindexed key is refused."""

    @pytest.mark.asyncio
    async def test_the_collection_is_strict(self, qdrant_client):
        store = QdrantVectorStore(_params(qdrant_client))
        await store.startup()
        info = await qdrant_client.get_collection(VECTOR_STORE_NAME)
        assert info.config.strict_mode_config is not None
        assert info.config.strict_mode_config.enabled is True
        assert info.config.strict_mode_config.unindexed_filtering_retrieve is False
        # The server enforces it: a filter on a key the store never indexed
        # is refused rather than scanned for.
        with pytest.raises(UnexpectedResponse, match=r"(?i)strict mode|index"):
            await qdrant_client.scroll(
                collection_name=VECTOR_STORE_NAME,
                scroll_filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="never_indexed", match=models.MatchValue(value="x")
                        )
                    ]
                ),
                limit=1,
            )

    @pytest.mark.asyncio
    async def test_a_predicate_of_another_type_matches_nothing(self, qdrant_client):
        """The store answers a mistyped leaf itself; the server would refuse it."""
        store = QdrantVectorStore(_params(qdrant_client))
        await store.startup()
        partition = await _open(store)
        held = _make_record(vector=_normalize([1.0, 0.0, 0.0]), properties={"age": 5})
        await partition.upsert(records=[held])

        for property_filter in (
            Comparison(field="age", op="=", value="5"),
            Comparison(field="age", op=">", value=1.5),
            In(field="age", values=["5"]),
            Comparison(field="name", op="=", value=5),
        ):
            [result] = await partition.query(
                query_vectors=[held.vector], limit=5, property_filter=property_filter
            )
            assert result.matches == [], property_filter
        [result] = await partition.query(
            query_vectors=[held.vector],
            limit=5,
            property_filter=Comparison(field="age", op="=", value=5),
        )
        assert [match.record_uuid for match in result.matches] == [held.uuid]
        await _purge(store, partition._partition_uuid)


class TestStrictModeIsRequested:
    """Local mode does not record strict mode, so the request itself is checked."""

    @pytest.mark.asyncio
    async def test_startup_creates_the_collection_strict(self, monkeypatch):
        client = AsyncQdrantClient(location=":memory:")
        requested: dict[str, models.StrictModeConfig | None] = {}
        original = client.create_collection

        async def recording_create_collection(*args, **kwargs):
            requested[kwargs["collection_name"]] = kwargs.get("strict_mode_config")
            return await original(*args, **kwargs)

        monkeypatch.setattr(client, "create_collection", recording_create_collection)
        store = QdrantVectorStore(_params(client))
        await store.startup()

        assert set(requested) == {VECTOR_STORE_NAME}
        config = requested[VECTOR_STORE_NAME]
        assert config is not None
        assert config.enabled is True
        assert config.unindexed_filtering_retrieve is False
        assert config.unindexed_filtering_update is False


class TestLifecycleContract(PartitionLifecycleContract):
    """The partition lifecycle contract, against this store."""

    count_stored = staticmethod(_count_stored)
    stored_uuids = staticmethod(_stored_uuids)

    @staticmethod
    async def settle(partition) -> None:
        # A Qdrant write returns once applied, so reads reflect it already.
        pass
