"""Tests for ChromaVectorStore and ChromaCollection."""

import json
from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import chromadb
import pytest
import pytest_asyncio
from chromadb.api import AsyncClientAPI
from chromadb.errors import NotFoundError
from testcontainers.chroma import ChromaContainer

from memmachine_server.common.data_types import SimilarityMetric
from memmachine_server.common.filter.filter_parser import (
    And as FilterAnd,
)
from memmachine_server.common.filter.filter_parser import (
    Comparison as FilterComparison,
)
from memmachine_server.common.filter.filter_parser import (
    In as FilterIn,
)
from memmachine_server.common.filter.filter_parser import (
    IsNull as FilterIsNull,
)
from memmachine_server.common.filter.filter_parser import (
    Not as FilterNot,
)
from memmachine_server.common.filter.filter_parser import (
    Or as FilterOr,
)
from memmachine_server.common.metrics_factory import MetricsFactory
from memmachine_server.common.vector_store.chroma_vector_store import (
    ChromaCollection,
    ChromaVectorStore,
    ChromaVectorStoreParams,
)
from memmachine_server.common.vector_store.data_types import Record
from server_tests.memmachine_server.conftest import is_docker_available

# ---------------------------------------------------------------------------
# Unit tests (no Chroma interaction)
# ---------------------------------------------------------------------------


class TestMetrics:
    @pytest.mark.asyncio
    async def test_collected_with_factory(self):
        mock_client = MagicMock(spec=AsyncClientAPI)
        mock_client.delete_collection = AsyncMock()
        factory = MagicMock(spec=MetricsFactory)
        histogram = MagicMock()
        factory.get_histogram = MagicMock(return_value=histogram)
        params = ChromaVectorStoreParams(
            client=mock_client,
            metrics_factory=factory,
        )
        store = ChromaVectorStore(params)
        await store.delete_collection("test_collection")
        histogram.observe.assert_called_once_with(
            value=pytest.approx(0.0, abs=1.0),
            labels={"operation": "delete_collection", "status": "ok"},
        )

    @pytest.mark.asyncio
    async def test_error_status_on_exception(self):
        mock_client = MagicMock(spec=AsyncClientAPI)
        mock_client.delete_collection = AsyncMock(side_effect=RuntimeError("boom"))
        factory = MagicMock(spec=MetricsFactory)
        histogram = MagicMock()
        factory.get_histogram = MagicMock(return_value=histogram)
        params = ChromaVectorStoreParams(
            client=mock_client,
            metrics_factory=factory,
        )
        store = ChromaVectorStore(params)
        with pytest.raises(RuntimeError):
            await store.delete_collection("test_collection")
        histogram.observe.assert_called_once_with(
            value=pytest.approx(0.0, abs=1.0),
            labels={"operation": "delete_collection", "status": "error"},
        )

    @pytest.mark.asyncio
    async def test_no_errors_without_factory(self):
        mock_client = MagicMock(spec=AsyncClientAPI)
        mock_client.delete_collection = AsyncMock()
        params = ChromaVectorStoreParams(client=mock_client)
        store = ChromaVectorStore(params)
        await store.delete_collection("test_collection")

    def test_init_with_metrics_registers_one_histogram(self):
        mock_client = MagicMock(spec=AsyncClientAPI)
        factory = MagicMock(spec=MetricsFactory)
        factory.get_histogram = MagicMock(return_value=MagicMock())
        params = ChromaVectorStoreParams(
            client=mock_client,
            metrics_factory=factory,
        )
        ChromaVectorStore(params)
        assert factory.get_histogram.call_count == 1


# ---------------------------------------------------------------------------
# Integration tests (require Docker + ChromaDB container)
# ---------------------------------------------------------------------------

VECTOR_DIM = 3


@pytest.fixture(scope="module")
def chroma_container():
    if not is_docker_available():
        pytest.skip("Docker is not available")
    with ChromaContainer() as container:
        yield container.get_config()


@pytest_asyncio.fixture(scope="module")
async def chroma_client(chroma_container):
    client = await chromadb.AsyncHttpClient(
        host=chroma_container["host"],
        port=chroma_container["port"],
    )
    yield client


@pytest.fixture(scope="module")
def store(chroma_client):
    params = ChromaVectorStoreParams(client=chroma_client)
    return ChromaVectorStore(params)


@pytest.mark.integration
class TestCreateCollection:
    @pytest.mark.asyncio
    async def test_cosine(self, store, chroma_client):
        await store.create_collection(
            "coll_cosine",
            vector_dimensions=VECTOR_DIM,
            similarity_metric=SimilarityMetric.COSINE,
        )
        raw_coll = await chroma_client.get_collection("coll_cosine")
        assert raw_coll.metadata["distance_metric"] == "cosine"

    @pytest.mark.asyncio
    async def test_dot(self, store, chroma_client):
        await store.create_collection(
            "coll_dot",
            vector_dimensions=VECTOR_DIM,
            similarity_metric=SimilarityMetric.DOT,
        )
        raw_coll = await chroma_client.get_collection("coll_dot")
        assert raw_coll.metadata["distance_metric"] == "ip"

    @pytest.mark.asyncio
    async def test_euclidean(self, store, chroma_client):
        await store.create_collection(
            "coll_euclidean",
            vector_dimensions=VECTOR_DIM,
            similarity_metric=SimilarityMetric.EUCLIDEAN,
        )
        raw_coll = await chroma_client.get_collection("coll_euclidean")
        assert raw_coll.metadata["distance_metric"] == "l2"

    @pytest.mark.asyncio
    async def test_manhattan_rejected(self, store):
        with pytest.raises(ValueError, match="manhattan"):
            await store.create_collection(
                "coll_manhattan",
                vector_dimensions=VECTOR_DIM,
                similarity_metric=SimilarityMetric.MANHATTAN,
            )

    @pytest.mark.asyncio
    async def test_without_schema_stores_empty(self, store, chroma_client):
        await store.create_collection("coll_no_schema", vector_dimensions=VECTOR_DIM)
        raw_coll = await chroma_client.get_collection("coll_no_schema")
        # No schema.* keys in metadata
        assert not any(k.startswith("schema.") for k in raw_coll.metadata)

    @pytest.mark.asyncio
    async def test_with_schema(self, store, chroma_client):
        schema = {"name": str, "count": int}
        await store.create_collection(
            "coll_schema",
            vector_dimensions=VECTOR_DIM,
            properties_schema=schema,
        )
        raw_coll = await chroma_client.get_collection("coll_schema")
        assert json.loads(raw_coll.metadata["schema.str"]) == ["name"]
        assert json.loads(raw_coll.metadata["schema.int"]) == ["count"]

    @pytest.mark.asyncio
    async def test_unsupported_schema_type(self, store):
        with pytest.raises(TypeError, match="Unsupported property type"):
            await store.create_collection(
                "coll_bad_schema",
                vector_dimensions=VECTOR_DIM,
                properties_schema={"x": list},
            )


@pytest.mark.integration
class TestGetCollection:
    @pytest.mark.asyncio
    async def test_creates_handle(self, store):
        await store.create_collection("test_get_coll", vector_dimensions=VECTOR_DIM)
        coll = await store.get_collection("test_get_coll")
        assert isinstance(coll, ChromaCollection)


@pytest.mark.integration
class TestDeleteCollection:
    @pytest.mark.asyncio
    async def test_create_then_delete(self, store, chroma_client):
        await store.create_collection("to_delete", vector_dimensions=VECTOR_DIM)
        await store.delete_collection("to_delete")
        # Verify the collection is gone via the raw client
        with pytest.raises(NotFoundError):
            await chroma_client.get_collection("to_delete")


@pytest.mark.integration
class TestUpsert:
    @pytest.mark.asyncio
    async def test_single_record(self, store):
        await store.create_collection("upsert_single", vector_dimensions=VECTOR_DIM)
        coll = await store.get_collection("upsert_single")
        uid = uuid4()
        rec = Record(uuid=uid, vector=[1.0, 2.0, 3.0], properties={"name": "alice"})
        await coll.upsert(records=[rec])
        results = list(await coll.get(record_uuids=[uid]))
        assert len(results) == 1
        assert results[0].uuid == uid
        assert results[0].properties == {"name": "alice"}
        assert results[0].vector == [1.0, 2.0, 3.0]

    @pytest.mark.asyncio
    async def test_multiple_records(self, store):
        await store.create_collection("upsert_multi", vector_dimensions=VECTOR_DIM)
        coll = await store.get_collection("upsert_multi")
        u1, u2 = uuid4(), uuid4()
        recs = [
            Record(uuid=u1, vector=[1.0, 0.0, 0.0], properties={"x": 1}),
            Record(uuid=u2, vector=[0.0, 1.0, 0.0], properties={"x": 2}),
        ]
        await coll.upsert(records=recs)
        results = list(await coll.get(record_uuids=[u1, u2]))
        assert len(results) == 2

    @pytest.mark.asyncio
    async def test_empty_iterable(self, store):
        await store.create_collection("upsert_empty", vector_dimensions=VECTOR_DIM)
        coll = await store.get_collection("upsert_empty")
        await coll.upsert(records=[])  # should not raise

    @pytest.mark.asyncio
    async def test_datetime_roundtrip(self, store):
        await store.create_collection(
            "upsert_dt",
            vector_dimensions=VECTOR_DIM,
            properties_schema={"created": datetime},
        )
        coll = await store.get_collection("upsert_dt")
        uid = uuid4()
        dt = datetime(2024, 6, 15, 12, 0, 0, tzinfo=UTC)
        rec = Record(uuid=uid, vector=[1.0, 0.0, 0.0], properties={"created": dt})
        await coll.upsert(records=[rec])
        results = list(await coll.get(record_uuids=[uid]))
        assert results[0].properties["created"] == dt

    @pytest.mark.asyncio
    async def test_null_sentinel_escape_roundtrip(self, store):
        await store.create_collection("upsert_null_esc", vector_dimensions=VECTOR_DIM)
        coll = await store.get_collection("upsert_null_esc")
        uid = uuid4()
        rec = Record(uuid=uid, vector=[1.0, 0.0, 0.0], properties={"tag": "__null__"})
        await coll.upsert(records=[rec])
        results = list(await coll.get(record_uuids=[uid]))
        assert results[0].properties["tag"] == "__null__"

    @pytest.mark.asyncio
    async def test_datetime_prefix_escape_roundtrip(self, store):
        await store.create_collection("upsert_dt_esc", vector_dimensions=VECTOR_DIM)
        coll = await store.get_collection("upsert_dt_esc")
        uid = uuid4()
        val = "__dt__:2024-01-15T10:30:00+00:00"
        rec = Record(uuid=uid, vector=[1.0, 0.0, 0.0], properties={"tag": val})
        await coll.upsert(records=[rec])
        results = list(await coll.get(record_uuids=[uid]))
        assert results[0].properties["tag"] == val

    @pytest.mark.asyncio
    async def test_double_escape_roundtrip(self, store):
        await store.create_collection("upsert_dbl_esc", vector_dimensions=VECTOR_DIM)
        coll = await store.get_collection("upsert_dbl_esc")
        uid = uuid4()
        val = "__esc__:something"
        rec = Record(uuid=uid, vector=[1.0, 0.0, 0.0], properties={"tag": val})
        await coll.upsert(records=[rec])
        results = list(await coll.get(record_uuids=[uid]))
        assert results[0].properties["tag"] == val

    @pytest.mark.asyncio
    async def test_schema_null_fill(self, store):
        await store.create_collection(
            "upsert_schema_null",
            vector_dimensions=VECTOR_DIM,
            properties_schema={"name": str, "age": int},
        )
        coll = await store.get_collection("upsert_schema_null")
        uid = uuid4()
        rec = Record(uuid=uid, vector=[1.0, 0.0, 0.0], properties={"name": "bob"})
        await coll.upsert(records=[rec])
        results = list(await coll.get(record_uuids=[uid]))
        assert results[0].properties["name"] == "bob"
        # age was not provided, so it should be filtered out (null sentinel removed)
        assert "age" not in results[0].properties

    @pytest.mark.asyncio
    async def test_schema_null_fill_non_string_types(self, store):
        await store.create_collection(
            "upsert_schema_nonstr",
            vector_dimensions=VECTOR_DIM,
            properties_schema={"active": bool, "score": float, "count": int},
        )
        coll = await store.get_collection("upsert_schema_nonstr")
        uid = uuid4()
        rec = Record(uuid=uid, vector=[1.0, 0.0, 0.0], properties={"active": True})
        await coll.upsert(records=[rec])
        results = list(await coll.get(record_uuids=[uid]))
        assert results[0].properties["active"] is True
        assert "score" not in results[0].properties
        assert "count" not in results[0].properties

    @pytest.mark.asyncio
    async def test_upsert_overwrites(self, store):
        await store.create_collection("upsert_overwrite", vector_dimensions=VECTOR_DIM)
        coll = await store.get_collection("upsert_overwrite")
        uid = uuid4()
        rec1 = Record(uuid=uid, vector=[1.0, 0.0, 0.0], properties={"name": "alice"})
        await coll.upsert(records=[rec1])
        rec2 = Record(uuid=uid, vector=[0.0, 1.0, 0.0], properties={"name": "bob"})
        await coll.upsert(records=[rec2])
        results = list(await coll.get(record_uuids=[uid]))
        assert results[0].properties["name"] == "bob"
        assert results[0].vector == [0.0, 1.0, 0.0]

    @pytest.mark.asyncio
    async def test_all_schema_fields_present(self, store):
        await store.create_collection(
            "upsert_all_fields",
            vector_dimensions=VECTOR_DIM,
            properties_schema={"name": str, "age": int},
        )
        coll = await store.get_collection("upsert_all_fields")
        uid = uuid4()
        rec = Record(
            uuid=uid,
            vector=[1.0, 0.0, 0.0],
            properties={"name": "alice", "age": 30},
        )
        await coll.upsert(records=[rec])
        results = list(await coll.get(record_uuids=[uid]))
        assert results[0].properties == {"name": "alice", "age": 30}

    @pytest.mark.asyncio
    async def test_no_properties(self, store):
        await store.create_collection(
            "upsert_no_props",
            vector_dimensions=VECTOR_DIM,
            properties_schema={"name": str, "age": int},
        )
        coll = await store.get_collection("upsert_no_props")
        uid = uuid4()
        rec = Record(uuid=uid, vector=[1.0, 0.0, 0.0])
        await coll.upsert(records=[rec])
        results = list(await coll.get(record_uuids=[uid]))
        # All schema fields become null sentinel -> filtered out
        assert "name" not in results[0].properties
        assert "age" not in results[0].properties


@pytest.mark.integration
class TestQuery:
    @pytest.mark.asyncio
    async def test_basic(self, store):
        await store.create_collection("query_basic", vector_dimensions=VECTOR_DIM)
        coll = await store.get_collection("query_basic")
        uid = uuid4()
        rec = Record(uuid=uid, vector=[1.0, 0.0, 0.0], properties={"k": "v"})
        await coll.upsert(records=[rec])
        results = list(await coll.query(query_vector=[1.0, 0.0, 0.0], limit=10))
        assert len(results) == 1
        assert results[0].record.uuid == uid
        assert results[0].record.properties == {"k": "v"}
        assert results[0].record.vector == [1.0, 0.0, 0.0]
        assert results[0].score > 0

    @pytest.mark.asyncio
    async def test_with_limit(self, store):
        await store.create_collection("query_limit", vector_dimensions=VECTOR_DIM)
        coll = await store.get_collection("query_limit")
        for _ in range(5):
            await coll.upsert(
                records=[
                    Record(uuid=uuid4(), vector=[1.0, 0.0, 0.0], properties={"x": 1})
                ]
            )
        results = list(await coll.query(query_vector=[1.0, 0.0, 0.0], limit=3))
        assert len(results) == 3

    @pytest.mark.asyncio
    async def test_without_limit_returns_all(self, store):
        await store.create_collection("query_no_limit", vector_dimensions=VECTOR_DIM)
        coll = await store.get_collection("query_no_limit")
        uids = []
        for _ in range(4):
            uid = uuid4()
            uids.append(uid)
            await coll.upsert(
                records=[Record(uuid=uid, vector=[1.0, 0.0, 0.0], properties={"x": 1})]
            )
        results = list(await coll.query(query_vector=[1.0, 0.0, 0.0]))
        assert len(results) == 4

    @pytest.mark.asyncio
    async def test_empty_collection(self, store):
        await store.create_collection("query_empty", vector_dimensions=VECTOR_DIM)
        coll = await store.get_collection("query_empty")
        results = list(await coll.query(query_vector=[1.0, 0.0, 0.0]))
        assert results == []

    @pytest.mark.asyncio
    async def test_similarity_threshold_filtering(self, store):
        await store.create_collection(
            "query_threshold",
            vector_dimensions=VECTOR_DIM,
            similarity_metric=SimilarityMetric.COSINE,
        )
        coll = await store.get_collection("query_threshold")
        # Very similar vector
        u1 = uuid4()
        await coll.upsert(
            records=[Record(uuid=u1, vector=[1.0, 0.0, 0.0], properties={"x": 1})]
        )
        # Orthogonal vector (low similarity)
        u2 = uuid4()
        await coll.upsert(
            records=[Record(uuid=u2, vector=[0.0, 1.0, 0.0], properties={"x": 2})]
        )
        results = list(
            await coll.query(
                query_vector=[1.0, 0.0, 0.0],
                limit=10,
                similarity_threshold=0.9,
            )
        )
        assert len(results) == 1
        assert results[0].record.uuid == u1

    @pytest.mark.asyncio
    async def test_property_filter(self, store):
        await store.create_collection(
            "query_filter",
            vector_dimensions=VECTOR_DIM,
            properties_schema={"color": str},
        )
        coll = await store.get_collection("query_filter")
        u1, u2 = uuid4(), uuid4()
        await coll.upsert(
            records=[
                Record(uuid=u1, vector=[1.0, 0.0, 0.0], properties={"color": "red"}),
                Record(uuid=u2, vector=[0.9, 0.1, 0.0], properties={"color": "blue"}),
            ]
        )
        filt = FilterComparison(field="color", op="=", value="red")
        results = list(
            await coll.query(
                query_vector=[1.0, 0.0, 0.0], limit=10, property_filter=filt
            )
        )
        assert len(results) == 1
        assert results[0].record.uuid == u1

    @pytest.mark.asyncio
    async def test_return_vector_false(self, store):
        await store.create_collection("query_no_vec", vector_dimensions=VECTOR_DIM)
        coll = await store.get_collection("query_no_vec")
        uid = uuid4()
        await coll.upsert(
            records=[Record(uuid=uid, vector=[1.0, 0.0, 0.0], properties={"k": "v"})]
        )
        results = list(
            await coll.query(query_vector=[1.0, 0.0, 0.0], limit=5, return_vector=False)
        )
        assert results[0].record.vector is None

    @pytest.mark.asyncio
    async def test_return_properties_false(self, store):
        await store.create_collection("query_no_props", vector_dimensions=VECTOR_DIM)
        coll = await store.get_collection("query_no_props")
        uid = uuid4()
        await coll.upsert(
            records=[Record(uuid=uid, vector=[1.0, 0.0, 0.0], properties={"k": "v"})]
        )
        results = list(
            await coll.query(
                query_vector=[1.0, 0.0, 0.0], limit=5, return_properties=False
            )
        )
        assert results[0].record.properties is None

    @pytest.mark.asyncio
    async def test_datetime_deserialization(self, store):
        await store.create_collection(
            "query_dt",
            vector_dimensions=VECTOR_DIM,
            properties_schema={"created": datetime},
        )
        coll = await store.get_collection("query_dt")
        dt = datetime(2024, 6, 15, tzinfo=UTC)
        uid = uuid4()
        await coll.upsert(
            records=[
                Record(uuid=uid, vector=[1.0, 0.0, 0.0], properties={"created": dt})
            ]
        )
        results = list(await coll.query(query_vector=[1.0, 0.0, 0.0], limit=5))
        assert results[0].record.properties["created"] == dt

    @pytest.mark.asyncio
    async def test_null_sentinel_filtered(self, store):
        await store.create_collection(
            "query_null_sent",
            vector_dimensions=VECTOR_DIM,
            properties_schema={"name": str, "age": int},
        )
        coll = await store.get_collection("query_null_sent")
        uid = uuid4()
        await coll.upsert(
            records=[
                Record(uuid=uid, vector=[1.0, 0.0, 0.0], properties={"name": "alice"})
            ]
        )
        results = list(await coll.query(query_vector=[1.0, 0.0, 0.0], limit=5))
        props = results[0].record.properties
        assert "name" in props
        assert "age" not in props

    @pytest.mark.asyncio
    async def test_distance_cosine(self, store):
        await store.create_collection(
            "query_dist_cos",
            vector_dimensions=VECTOR_DIM,
            similarity_metric=SimilarityMetric.COSINE,
        )
        coll = await store.get_collection("query_dist_cos")
        uid = uuid4()
        await coll.upsert(
            records=[Record(uuid=uid, vector=[1.0, 0.0, 0.0], properties={"x": 1})]
        )
        results = list(await coll.query(query_vector=[1.0, 0.0, 0.0], limit=5))
        # Identical vector should have similarity close to 1.0
        assert results[0].score == pytest.approx(1.0, abs=0.01)

    @pytest.mark.asyncio
    async def test_distance_dot(self, store):
        await store.create_collection(
            "query_dist_dot",
            vector_dimensions=VECTOR_DIM,
            similarity_metric=SimilarityMetric.DOT,
        )
        coll = await store.get_collection("query_dist_dot")
        uid = uuid4()
        await coll.upsert(
            records=[Record(uuid=uid, vector=[1.0, 0.0, 0.0], properties={"x": 1})]
        )
        results = list(await coll.query(query_vector=[1.0, 0.0, 0.0], limit=5))
        # Score should be positive for identical vectors
        assert results[0].score > 0

    @pytest.mark.asyncio
    async def test_distance_euclidean(self, store):
        await store.create_collection(
            "query_dist_l2",
            vector_dimensions=VECTOR_DIM,
            similarity_metric=SimilarityMetric.EUCLIDEAN,
        )
        coll = await store.get_collection("query_dist_l2")
        uid = uuid4()
        await coll.upsert(
            records=[Record(uuid=uid, vector=[1.0, 0.0, 0.0], properties={"x": 1})]
        )
        results = list(await coll.query(query_vector=[1.0, 0.0, 0.0], limit=5))
        # Euclidean distance 0 => similarity = 1/(1+0) = 1.0
        assert results[0].score == pytest.approx(1.0, abs=0.01)


@pytest.mark.integration
class TestQueryFilters:
    """Test all filter types through the query API against a real Chroma instance."""

    @pytest_asyncio.fixture(scope="class")
    async def coll(self, store):
        """Shared collection with pre-loaded data for all filter tests."""
        schema = {"color": str, "count": int, "score": float, "active": bool}
        await store.create_collection(
            "query_filters",
            vector_dimensions=VECTOR_DIM,
            properties_schema=schema,
            similarity_metric=SimilarityMetric.COSINE,
        )
        coll = await store.get_collection("query_filters")
        # Insert test records with varying properties
        records = [
            Record(
                uuid=uuid4(),
                vector=[1.0, 0.0, 0.0],
                properties={"color": "red", "count": 10, "score": 0.9, "active": True},
            ),
            Record(
                uuid=uuid4(),
                vector=[0.9, 0.1, 0.0],
                properties={"color": "blue", "count": 20, "score": 0.5, "active": True},
            ),
            Record(
                uuid=uuid4(),
                vector=[0.8, 0.2, 0.0],
                properties={
                    "color": "green",
                    "count": 30,
                    "score": 0.1,
                    "active": False,
                },
            ),
            Record(
                uuid=uuid4(),
                vector=[0.7, 0.3, 0.0],
                properties={"color": "red", "count": 40, "score": 0.7},
            ),
        ]
        await coll.upsert(records=records)
        return coll

    async def _query_colors(self, coll, filt):
        results = list(
            await coll.query(
                query_vector=[1.0, 0.0, 0.0], limit=10, property_filter=filt
            )
        )
        return sorted(r.record.properties["color"] for r in results)

    # --- Comparison operators ---

    @pytest.mark.asyncio
    async def test_eq(self, coll):
        filt = FilterComparison(field="color", op="=", value="red")
        colors = await self._query_colors(coll, filt)
        assert colors == ["red", "red"]

    @pytest.mark.asyncio
    async def test_ne(self, coll):
        filt = FilterComparison(field="color", op="!=", value="red")
        colors = await self._query_colors(coll, filt)
        assert colors == ["blue", "green"]

    @pytest.mark.asyncio
    async def test_gt(self, coll):
        filt = FilterComparison(field="count", op=">", value=20)
        results = list(
            await coll.query(
                query_vector=[1.0, 0.0, 0.0], limit=10, property_filter=filt
            )
        )
        counts = sorted(r.record.properties["count"] for r in results)
        assert counts == [30, 40]

    @pytest.mark.asyncio
    async def test_gte(self, coll):
        filt = FilterComparison(field="count", op=">=", value=20)
        results = list(
            await coll.query(
                query_vector=[1.0, 0.0, 0.0], limit=10, property_filter=filt
            )
        )
        counts = sorted(r.record.properties["count"] for r in results)
        assert counts == [20, 30, 40]

    @pytest.mark.asyncio
    async def test_lt(self, coll):
        filt = FilterComparison(field="count", op="<", value=20)
        results = list(
            await coll.query(
                query_vector=[1.0, 0.0, 0.0], limit=10, property_filter=filt
            )
        )
        counts = [r.record.properties["count"] for r in results]
        assert counts == [10]

    @pytest.mark.asyncio
    async def test_lte(self, coll):
        filt = FilterComparison(field="count", op="<=", value=20)
        results = list(
            await coll.query(
                query_vector=[1.0, 0.0, 0.0], limit=10, property_filter=filt
            )
        )
        counts = sorted(r.record.properties["count"] for r in results)
        assert counts == [10, 20]

    @pytest.mark.asyncio
    async def test_float_comparison(self, coll):
        filt = FilterComparison(field="score", op=">", value=0.6)
        results = list(
            await coll.query(
                query_vector=[1.0, 0.0, 0.0], limit=10, property_filter=filt
            )
        )
        scores = sorted(r.record.properties["score"] for r in results)
        assert scores == [0.7, 0.9]

    @pytest.mark.asyncio
    async def test_bool_comparison(self, coll):
        filt = FilterComparison(field="active", op="=", value=True)
        colors = await self._query_colors(coll, filt)
        assert colors == ["blue", "red"]

    # --- In filter ---

    @pytest.mark.asyncio
    async def test_in_strings(self, coll):
        filt = FilterIn(field="color", values=["red", "green"])
        colors = await self._query_colors(coll, filt)
        assert colors == ["green", "red", "red"]

    @pytest.mark.asyncio
    async def test_in_ints(self, coll):
        filt = FilterIn(field="count", values=[10, 30])
        results = list(
            await coll.query(
                query_vector=[1.0, 0.0, 0.0], limit=10, property_filter=filt
            )
        )
        counts = sorted(r.record.properties["count"] for r in results)
        assert counts == [10, 30]

    # --- IsNull filter ---

    @pytest.mark.asyncio
    async def test_is_null(self, coll):
        # Record 4 has no "active" property -> null sentinel
        filt = FilterIsNull(field="active")
        results = list(
            await coll.query(
                query_vector=[1.0, 0.0, 0.0], limit=10, property_filter=filt
            )
        )
        assert len(results) == 1
        assert results[0].record.properties["color"] == "red"
        assert results[0].record.properties["count"] == 40

    # --- And filter ---

    @pytest.mark.asyncio
    async def test_and(self, coll):
        filt = FilterAnd(
            left=FilterComparison(field="color", op="=", value="red"),
            right=FilterComparison(field="count", op=">", value=10),
        )
        results = list(
            await coll.query(
                query_vector=[1.0, 0.0, 0.0], limit=10, property_filter=filt
            )
        )
        assert len(results) == 1
        assert results[0].record.properties["count"] == 40

    # --- Or filter ---

    @pytest.mark.asyncio
    async def test_or(self, coll):
        filt = FilterOr(
            left=FilterComparison(field="color", op="=", value="blue"),
            right=FilterComparison(field="color", op="=", value="green"),
        )
        colors = await self._query_colors(coll, filt)
        assert colors == ["blue", "green"]

    # --- Not filter ---

    @pytest.mark.asyncio
    async def test_not_comparison(self, coll):
        # NOT color = "red" -> blue, green
        filt = FilterNot(expr=FilterComparison(field="color", op="=", value="red"))
        colors = await self._query_colors(coll, filt)
        assert colors == ["blue", "green"]

    @pytest.mark.asyncio
    async def test_not_in(self, coll):
        # NOT IN ["red", "blue"] -> green
        filt = FilterNot(expr=FilterIn(field="color", values=["red", "blue"]))
        colors = await self._query_colors(coll, filt)
        assert colors == ["green"]

    @pytest.mark.asyncio
    async def test_not_is_null(self, coll):
        # NOT IS NULL active -> records with active set (3 records)
        filt = FilterNot(expr=FilterIsNull(field="active"))
        results = list(
            await coll.query(
                query_vector=[1.0, 0.0, 0.0], limit=10, property_filter=filt
            )
        )
        assert len(results) == 3

    @pytest.mark.asyncio
    async def test_double_not(self, coll):
        # NOT NOT color = "red" -> same as color = "red"
        filt = FilterNot(
            expr=FilterNot(
                expr=FilterComparison(field="color", op="=", value="red"),
            ),
        )
        colors = await self._query_colors(coll, filt)
        assert colors == ["red", "red"]

    # --- De Morgan's law combinations ---

    @pytest.mark.asyncio
    async def test_not_and_becomes_or(self, coll):
        # NOT (color = "red" AND count = 10) -> everything except red/10
        filt = FilterNot(
            expr=FilterAnd(
                left=FilterComparison(field="color", op="=", value="red"),
                right=FilterComparison(field="count", op="=", value=10),
            ),
        )
        results = list(
            await coll.query(
                query_vector=[1.0, 0.0, 0.0], limit=10, property_filter=filt
            )
        )
        assert len(results) == 3
        counts = sorted(r.record.properties["count"] for r in results)
        assert counts == [20, 30, 40]

    @pytest.mark.asyncio
    async def test_not_or_becomes_and(self, coll):
        # NOT (color = "red" OR color = "blue") -> green only
        filt = FilterNot(
            expr=FilterOr(
                left=FilterComparison(field="color", op="=", value="red"),
                right=FilterComparison(field="color", op="=", value="blue"),
            ),
        )
        colors = await self._query_colors(coll, filt)
        assert colors == ["green"]

    # --- Complex nested filters ---

    @pytest.mark.asyncio
    async def test_and_or_combined(self, coll):
        # (color = "red" AND count > 10) OR color = "blue"
        filt = FilterOr(
            left=FilterAnd(
                left=FilterComparison(field="color", op="=", value="red"),
                right=FilterComparison(field="count", op=">", value=10),
            ),
            right=FilterComparison(field="color", op="=", value="blue"),
        )
        colors = await self._query_colors(coll, filt)
        assert colors == ["blue", "red"]

    @pytest.mark.asyncio
    async def test_nested_and(self, coll):
        # color = "red" AND count >= 10 AND score > 0.5
        filt = FilterAnd(
            left=FilterComparison(field="color", op="=", value="red"),
            right=FilterAnd(
                left=FilterComparison(field="count", op=">=", value=10),
                right=FilterComparison(field="score", op=">", value=0.5),
            ),
        )
        results = list(
            await coll.query(
                query_vector=[1.0, 0.0, 0.0], limit=10, property_filter=filt
            )
        )
        assert len(results) == 2
        scores = sorted(r.record.properties["score"] for r in results)
        assert scores == [0.7, 0.9]

    # --- IsNull on non-string schema types ---

    @pytest.mark.asyncio
    async def test_is_null_int(self, coll):
        # Record 4 omits "count" is not true — it has count=40.
        # But "active" (bool) is already tested. Test that IsNull on score
        # (float) also works: record 4 omits "active" but has "score".
        # We need a record that omits an int field. Record 4 has all of
        # color, count, score — but omits "active".
        # So let's verify IsNull on "score" returns 0 (all records have it).
        filt = FilterIsNull(field="score")
        results = list(
            await coll.query(
                query_vector=[1.0, 0.0, 0.0], limit=10, property_filter=filt
            )
        )
        assert len(results) == 0

    @pytest.mark.asyncio
    async def test_is_null_on_int_field(self, store):
        """IsNull works on int-typed schema fields."""
        schema = {"name": str, "age": int}
        await store.create_collection(
            "filt_null_int",
            vector_dimensions=VECTOR_DIM,
            properties_schema=schema,
        )
        coll = await store.get_collection("filt_null_int")
        u1, u2 = uuid4(), uuid4()
        await coll.upsert(
            records=[
                Record(
                    uuid=u1,
                    vector=[1.0, 0.0, 0.0],
                    properties={"name": "alice", "age": 30},
                ),
                Record(
                    uuid=u2,
                    vector=[0.9, 0.1, 0.0],
                    properties={"name": "bob"},
                ),
            ]
        )
        # IsNull(age) should match bob only
        filt = FilterIsNull(field="age")
        results = list(
            await coll.query(
                query_vector=[1.0, 0.0, 0.0], limit=10, property_filter=filt
            )
        )
        assert len(results) == 1
        assert results[0].record.properties["name"] == "bob"

    @pytest.mark.asyncio
    async def test_is_null_on_float_field(self, store):
        """IsNull works on float-typed schema fields."""
        schema = {"label": str, "weight": float}
        await store.create_collection(
            "filt_null_float",
            vector_dimensions=VECTOR_DIM,
            properties_schema=schema,
        )
        coll = await store.get_collection("filt_null_float")
        u1, u2 = uuid4(), uuid4()
        await coll.upsert(
            records=[
                Record(
                    uuid=u1,
                    vector=[1.0, 0.0, 0.0],
                    properties={"label": "heavy", "weight": 99.5},
                ),
                Record(
                    uuid=u2,
                    vector=[0.9, 0.1, 0.0],
                    properties={"label": "unknown"},
                ),
            ]
        )
        filt = FilterIsNull(field="weight")
        results = list(
            await coll.query(
                query_vector=[1.0, 0.0, 0.0], limit=10, property_filter=filt
            )
        )
        assert len(results) == 1
        assert results[0].record.properties["label"] == "unknown"

    @pytest.mark.asyncio
    async def test_not_is_null_on_int_field(self, store):
        """NOT IsNull on int-typed schema field returns records that have the field."""
        schema = {"name": str, "age": int}
        await store.create_collection(
            "filt_not_null_int",
            vector_dimensions=VECTOR_DIM,
            properties_schema=schema,
        )
        coll = await store.get_collection("filt_not_null_int")
        u1, u2 = uuid4(), uuid4()
        await coll.upsert(
            records=[
                Record(
                    uuid=u1,
                    vector=[1.0, 0.0, 0.0],
                    properties={"name": "alice", "age": 30},
                ),
                Record(
                    uuid=u2,
                    vector=[0.9, 0.1, 0.0],
                    properties={"name": "bob"},
                ),
            ]
        )
        filt = FilterNot(expr=FilterIsNull(field="age"))
        results = list(
            await coll.query(
                query_vector=[1.0, 0.0, 0.0], limit=10, property_filter=filt
            )
        )
        assert len(results) == 1
        assert results[0].record.properties["name"] == "alice"
        assert results[0].record.properties["age"] == 30

    # --- Datetime filters ---

    @pytest.mark.asyncio
    async def test_datetime_eq(self, store):
        """Filter by exact datetime equality."""
        await store.create_collection(
            "filt_dt_eq",
            vector_dimensions=VECTOR_DIM,
            properties_schema={"created": datetime},
        )
        coll = await store.get_collection("filt_dt_eq")
        dt1 = datetime(2024, 1, 1, tzinfo=UTC)
        dt2 = datetime(2024, 6, 15, tzinfo=UTC)
        u1, u2 = uuid4(), uuid4()
        await coll.upsert(
            records=[
                Record(uuid=u1, vector=[1.0, 0.0, 0.0], properties={"created": dt1}),
                Record(uuid=u2, vector=[0.9, 0.1, 0.0], properties={"created": dt2}),
            ]
        )
        filt = FilterComparison(field="created", op="=", value=dt1)
        results = list(
            await coll.query(
                query_vector=[1.0, 0.0, 0.0], limit=10, property_filter=filt
            )
        )
        assert len(results) == 1
        assert results[0].record.properties["created"] == dt1

    @pytest.mark.asyncio
    async def test_datetime_ne(self, store):
        """Filter by datetime not-equal."""
        await store.create_collection(
            "filt_dt_ne",
            vector_dimensions=VECTOR_DIM,
            properties_schema={"created": datetime},
        )
        coll = await store.get_collection("filt_dt_ne")
        dt1 = datetime(2024, 1, 1, tzinfo=UTC)
        dt2 = datetime(2024, 6, 15, tzinfo=UTC)
        u1, u2 = uuid4(), uuid4()
        await coll.upsert(
            records=[
                Record(uuid=u1, vector=[1.0, 0.0, 0.0], properties={"created": dt1}),
                Record(uuid=u2, vector=[0.9, 0.1, 0.0], properties={"created": dt2}),
            ]
        )
        filt = FilterComparison(field="created", op="!=", value=dt1)
        results = list(
            await coll.query(
                query_vector=[1.0, 0.0, 0.0], limit=10, property_filter=filt
            )
        )
        assert len(results) == 1
        assert results[0].record.properties["created"] == dt2

    @pytest.mark.asyncio
    async def test_datetime_gt(self, store):
        """Filter by datetime greater-than."""
        schema = {"ts": datetime}
        await store.create_collection(
            "filt_dt_gt",
            vector_dimensions=VECTOR_DIM,
            properties_schema=schema,
        )
        coll = await store.get_collection("filt_dt_gt")
        dt1 = datetime(2024, 1, 1, tzinfo=UTC)
        dt2 = datetime(2024, 6, 15, tzinfo=UTC)
        dt3 = datetime(2025, 1, 1, tzinfo=UTC)
        u1, u2, u3 = uuid4(), uuid4(), uuid4()
        await coll.upsert(
            records=[
                Record(uuid=u1, vector=[1.0, 0.0, 0.0], properties={"ts": dt1}),
                Record(uuid=u2, vector=[0.9, 0.1, 0.0], properties={"ts": dt2}),
                Record(uuid=u3, vector=[0.8, 0.2, 0.0], properties={"ts": dt3}),
            ]
        )
        filt = FilterComparison(field="ts", op=">", value=dt2)
        results = list(
            await coll.query(
                query_vector=[1.0, 0.0, 0.0], limit=10, property_filter=filt
            )
        )
        assert len(results) == 1
        assert results[0].record.properties["ts"] == dt3

    @pytest.mark.asyncio
    async def test_datetime_lte(self, store):
        """Filter by datetime less-than-or-equal."""
        schema = {"ts": datetime}
        await store.create_collection(
            "filt_dt_lte",
            vector_dimensions=VECTOR_DIM,
            properties_schema=schema,
        )
        coll = await store.get_collection("filt_dt_lte")
        dt1 = datetime(2024, 1, 1, tzinfo=UTC)
        dt2 = datetime(2024, 6, 15, tzinfo=UTC)
        dt3 = datetime(2025, 1, 1, tzinfo=UTC)
        u1, u2, u3 = uuid4(), uuid4(), uuid4()
        await coll.upsert(
            records=[
                Record(uuid=u1, vector=[1.0, 0.0, 0.0], properties={"ts": dt1}),
                Record(uuid=u2, vector=[0.9, 0.1, 0.0], properties={"ts": dt2}),
                Record(uuid=u3, vector=[0.8, 0.2, 0.0], properties={"ts": dt3}),
            ]
        )
        filt = FilterComparison(field="ts", op="<=", value=dt2)
        results = list(
            await coll.query(
                query_vector=[1.0, 0.0, 0.0], limit=10, property_filter=filt
            )
        )
        assert len(results) == 2
        timestamps = sorted(r.record.properties["ts"] for r in results)
        assert timestamps == [dt1, dt2]

    @pytest.mark.asyncio
    async def test_datetime_range_with_and(self, store):
        """Filter datetime within a range using AND."""
        schema = {"ts": datetime}
        await store.create_collection(
            "filt_dt_range",
            vector_dimensions=VECTOR_DIM,
            properties_schema=schema,
        )
        coll = await store.get_collection("filt_dt_range")
        dt1 = datetime(2024, 1, 1, tzinfo=UTC)
        dt2 = datetime(2024, 6, 15, tzinfo=UTC)
        dt3 = datetime(2025, 1, 1, tzinfo=UTC)
        u1, u2, u3 = uuid4(), uuid4(), uuid4()
        await coll.upsert(
            records=[
                Record(uuid=u1, vector=[1.0, 0.0, 0.0], properties={"ts": dt1}),
                Record(uuid=u2, vector=[0.9, 0.1, 0.0], properties={"ts": dt2}),
                Record(uuid=u3, vector=[0.8, 0.2, 0.0], properties={"ts": dt3}),
            ]
        )
        filt = FilterAnd(
            left=FilterComparison(field="ts", op=">=", value=dt1),
            right=FilterComparison(field="ts", op="<", value=dt3),
        )
        results = list(
            await coll.query(
                query_vector=[1.0, 0.0, 0.0], limit=10, property_filter=filt
            )
        )
        assert len(results) == 2
        timestamps = sorted(r.record.properties["ts"] for r in results)
        assert timestamps == [dt1, dt2]


@pytest.mark.integration
class TestGet:
    @pytest.mark.asyncio
    async def test_single_record(self, store):
        await store.create_collection("get_single", vector_dimensions=VECTOR_DIM)
        coll = await store.get_collection("get_single")
        uid = uuid4()
        await coll.upsert(
            records=[Record(uuid=uid, vector=[1.0, 2.0, 3.0], properties={"k": "v"})]
        )
        results = list(await coll.get(record_uuids=[uid]))
        assert len(results) == 1
        assert results[0].uuid == uid
        assert results[0].properties == {"k": "v"}
        assert results[0].vector == [1.0, 2.0, 3.0]

    @pytest.mark.asyncio
    async def test_multiple_records(self, store):
        await store.create_collection("get_multi", vector_dimensions=VECTOR_DIM)
        coll = await store.get_collection("get_multi")
        u1, u2 = uuid4(), uuid4()
        await coll.upsert(
            records=[
                Record(uuid=u1, vector=[1.0, 0.0, 0.0], properties={"a": 1}),
                Record(uuid=u2, vector=[0.0, 1.0, 0.0], properties={"b": 2}),
            ]
        )
        results = list(await coll.get(record_uuids=[u1, u2]))
        assert len(results) == 2

    @pytest.mark.asyncio
    async def test_preserves_input_order(self, store):
        await store.create_collection("get_order", vector_dimensions=VECTOR_DIM)
        coll = await store.get_collection("get_order")
        u1, u2 = uuid4(), uuid4()
        await coll.upsert(
            records=[
                Record(uuid=u1, vector=[1.0, 0.0, 0.0], properties={"x": 1}),
                Record(uuid=u2, vector=[0.0, 1.0, 0.0], properties={"x": 2}),
            ]
        )
        # Request in order [u1, u2]
        results = list(await coll.get(record_uuids=[u1, u2]))
        assert results[0].uuid == u1
        assert results[1].uuid == u2
        # Request in reverse order [u2, u1]
        results = list(await coll.get(record_uuids=[u2, u1]))
        assert results[0].uuid == u2
        assert results[1].uuid == u1

    @pytest.mark.asyncio
    async def test_empty_uuids(self, store):
        await store.create_collection("get_empty", vector_dimensions=VECTOR_DIM)
        coll = await store.get_collection("get_empty")
        results = list(await coll.get(record_uuids=[]))
        assert results == []

    @pytest.mark.asyncio
    async def test_return_vector_false(self, store):
        await store.create_collection("get_no_vec", vector_dimensions=VECTOR_DIM)
        coll = await store.get_collection("get_no_vec")
        uid = uuid4()
        await coll.upsert(
            records=[Record(uuid=uid, vector=[1.0, 2.0, 3.0], properties={"k": "v"})]
        )
        results = list(await coll.get(record_uuids=[uid], return_vector=False))
        assert results[0].vector is None
        assert results[0].properties == {"k": "v"}

    @pytest.mark.asyncio
    async def test_return_properties_false(self, store):
        await store.create_collection("get_no_props", vector_dimensions=VECTOR_DIM)
        coll = await store.get_collection("get_no_props")
        uid = uuid4()
        await coll.upsert(
            records=[Record(uuid=uid, vector=[1.0, 2.0, 3.0], properties={"k": "v"})]
        )
        results = list(await coll.get(record_uuids=[uid], return_properties=False))
        assert results[0].properties is None
        assert results[0].vector == [1.0, 2.0, 3.0]

    @pytest.mark.asyncio
    async def test_datetime_and_escape_deserialization(self, store):
        await store.create_collection(
            "get_deser",
            vector_dimensions=VECTOR_DIM,
            properties_schema={"created": datetime, "tag": str},
        )
        coll = await store.get_collection("get_deser")
        uid = uuid4()
        dt = datetime(2024, 1, 1, tzinfo=UTC)
        await coll.upsert(
            records=[
                Record(
                    uuid=uid,
                    vector=[1.0, 0.0, 0.0],
                    properties={"created": dt, "tag": "__null__"},
                )
            ]
        )
        results = list(await coll.get(record_uuids=[uid]))
        assert results[0].properties["created"] == dt
        assert results[0].properties["tag"] == "__null__"

    @pytest.mark.asyncio
    async def test_null_sentinel_filtered(self, store):
        await store.create_collection(
            "get_null_sent",
            vector_dimensions=VECTOR_DIM,
            properties_schema={"name": str, "count": int},
        )
        coll = await store.get_collection("get_null_sent")
        uid = uuid4()
        await coll.upsert(
            records=[Record(uuid=uid, vector=[1.0, 0.0, 0.0], properties={"name": "x"})]
        )
        results = list(await coll.get(record_uuids=[uid]))
        assert "name" in results[0].properties
        assert "count" not in results[0].properties

    @pytest.mark.asyncio
    async def test_missing_record(self, store):
        await store.create_collection("get_missing", vector_dimensions=VECTOR_DIM)
        coll = await store.get_collection("get_missing")
        uid = uuid4()
        results = list(await coll.get(record_uuids=[uid]))
        assert results == []

    @pytest.mark.asyncio
    async def test_after_upsert_with_none_properties(self, store):
        await store.create_collection("get_none_props", vector_dimensions=VECTOR_DIM)
        coll = await store.get_collection("get_none_props")
        uid = uuid4()
        await coll.upsert(records=[Record(uuid=uid, vector=[1.0, 0.0, 0.0])])
        results = list(await coll.get(record_uuids=[uid]))
        assert len(results) == 1
        # No metadata was stored, so properties is None
        assert results[0].properties is None

    @pytest.mark.asyncio
    async def test_after_upsert_with_all_null_schema_fields(self, store):
        await store.create_collection(
            "get_all_null",
            vector_dimensions=VECTOR_DIM,
            properties_schema={"name": str, "age": int},
        )
        coll = await store.get_collection("get_all_null")
        uid = uuid4()
        # Upsert with no properties — all schema fields become null sentinel
        await coll.upsert(records=[Record(uuid=uid, vector=[1.0, 0.0, 0.0])])
        results = list(await coll.get(record_uuids=[uid]))
        assert len(results) == 1
        # Both fields should be filtered out as null sentinels
        assert "name" not in results[0].properties
        assert "age" not in results[0].properties


@pytest.mark.integration
class TestDelete:
    @pytest.mark.asyncio
    async def test_basic(self, store):
        await store.create_collection("del_basic", vector_dimensions=VECTOR_DIM)
        coll = await store.get_collection("del_basic")
        uid = uuid4()
        await coll.upsert(
            records=[Record(uuid=uid, vector=[1.0, 0.0, 0.0], properties={"x": 1})]
        )
        await coll.delete(record_uuids=[uid])
        results = list(await coll.get(record_uuids=[uid]))
        assert results == []

    @pytest.mark.asyncio
    async def test_empty_uuids(self, store):
        await store.create_collection("del_empty", vector_dimensions=VECTOR_DIM)
        coll = await store.get_collection("del_empty")
        await coll.delete(record_uuids=[])  # should not raise

    @pytest.mark.asyncio
    async def test_deleted_records_gone(self, store):
        await store.create_collection("del_verify", vector_dimensions=VECTOR_DIM)
        coll = await store.get_collection("del_verify")
        u1, u2 = uuid4(), uuid4()
        await coll.upsert(
            records=[
                Record(uuid=u1, vector=[1.0, 0.0, 0.0], properties={"x": 1}),
                Record(uuid=u2, vector=[0.0, 1.0, 0.0], properties={"x": 2}),
            ]
        )
        await coll.delete(record_uuids=[u1])
        results = list(await coll.get(record_uuids=[u1, u2]))
        assert len(results) == 1
        assert results[0].uuid == u2
