"""Integration tests for ``AgeVectorGraphStore``.

Requires Docker. The fixtures launch an ``apache/age`` container with the
``age`` and ``vector`` extensions preinstalled and exercise the store end-to-
end against a real PostgreSQL instance. Tests are marked ``integration`` so
the default ``pytest`` invocation skips them.
"""

import socket
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path
from uuid import uuid4
from zoneinfo import ZoneInfo

import pytest
import pytest_asyncio
from sqlalchemy import event
from sqlalchemy.engine import URL
from sqlalchemy.ext.asyncio import create_async_engine
from testcontainers.core.container import DockerContainer
from testcontainers.core.image import DockerImage
from testcontainers.core.waiting_utils import wait_for_logs

from memmachine_server.common.age_utils import setup_age_sync_connection
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
from memmachine_server.common.metrics_factory.prometheus_metrics_factory import (
    PrometheusMetricsFactory,
)
from memmachine_server.common.vector_graph_store.age_vector_graph_store import (
    DEFAULT_GRAPH_NAME,
    AgeVectorGraphStore,
    AgeVectorGraphStoreParams,
)
from memmachine_server.common.vector_graph_store.data_types import (
    Edge,
    Node,
)
from server_tests.memmachine_server.conftest import is_docker_available

pytestmark = pytest.mark.integration


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def metrics_factory():
    return PrometheusMetricsFactory()


_POSTGRES_AGE_DOCKERFILE_DIR = (
    Path(__file__).resolve().parents[6] / "deployments" / "docker" / "postgres-age"
)


@pytest.fixture(scope="module")
def age_connection_info():
    if not is_docker_available():
        pytest.skip("Docker is not available")

    # Build the AGE + pgvector image from the same Dockerfile the production
    # docker-compose / helm stack uses, so the test environment matches prod
    # and we avoid a per-run ``apt-get install postgresql-16-pgvector`` inside
    # a stock apache/age container.
    if not _POSTGRES_AGE_DOCKERFILE_DIR.is_dir():
        pytest.skip(
            f"postgres-age Dockerfile dir not found at {_POSTGRES_AGE_DOCKERFILE_DIR}"
        )

    user = "postgres"
    password = "postgres"
    dbname = "memmachine_test"

    with DockerImage(
        path=str(_POSTGRES_AGE_DOCKERFILE_DIR),
        tag="memmachine-test/postgres-age:pg16-1.6.0",
    ) as image:
        container = (
            DockerContainer(str(image))
            .with_exposed_ports(5432)
            .with_env("POSTGRES_USER", user)
            .with_env("POSTGRES_PASSWORD", password)
            .with_env("POSTGRES_DB", dbname)
        )
        container.start()
        try:
            wait_for_logs(
                container,
                "database system is ready to accept connections",
                timeout=60,
            )

            host = container.get_container_host_ip()
            port = int(container.get_exposed_port(5432))

            # Postgres typically restarts once during init (pg_ctl reload
            # after template setup). Poll the exposed port until it accepts
            # TCP connections so tests don't race the re-listen.
            deadline = time.monotonic() + 30.0
            last_err: OSError | None = None
            while time.monotonic() < deadline:
                try:
                    with socket.create_connection((host, port), timeout=1):
                        break
                except OSError as err:
                    last_err = err
                    time.sleep(0.25)
            else:
                raise RuntimeError(
                    f"Postgres/AGE container never became reachable on {host}:{port}"
                ) from last_err

            yield {
                "host": host,
                "port": port,
                "user": user,
                "password": password,
                "db_name": dbname,
            }
        finally:
            container.stop()


@pytest_asyncio.fixture(scope="module")
async def age_engine(age_connection_info):
    url = URL.create(
        "postgresql+asyncpg",
        username=age_connection_info["user"],
        password=age_connection_info["password"],
        host=age_connection_info["host"],
        port=age_connection_info["port"],
        database=age_connection_info["db_name"],
    )
    engine = create_async_engine(url, pool_pre_ping=True)

    @event.listens_for(engine.sync_engine, "connect")
    def _on_connect(dbapi_connection, _connection_record):
        setup_age_sync_connection(dbapi_connection)

    yield engine
    await engine.dispose()


@pytest.fixture(scope="module")
def vector_graph_store(age_engine, metrics_factory):
    return AgeVectorGraphStore(
        AgeVectorGraphStoreParams(
            engine=age_engine,
            graph_name=DEFAULT_GRAPH_NAME,
            force_exact_similarity_search=True,
            metrics_factory=metrics_factory,
        ),
    )


@pytest.fixture(scope="module")
def vector_graph_store_ann(age_engine):
    return AgeVectorGraphStore(
        AgeVectorGraphStoreParams(
            engine=age_engine,
            graph_name=DEFAULT_GRAPH_NAME,
            force_exact_similarity_search=False,
            filtered_similarity_search_fudge_factor=2,
            exact_similarity_search_fallback_threshold=0.5,
            range_index_creation_threshold=0,
            vector_index_creation_threshold=0,
        ),
    )


@pytest_asyncio.fixture(autouse=True)
async def db_cleanup(age_engine, vector_graph_store):
    # Drop and recreate the AGE graph between tests to clear all labels,
    # vertices, edges, and side tables tied to the previous run.
    await vector_graph_store.delete_all_data()
    async with age_engine.begin() as conn:
        # Drop every pgvector side table belonging to the test graph. Side
        # tables live in ``public`` and are prefixed with the graph name.
        result = await conn.exec_driver_sql(
            "SELECT tablename FROM pg_tables "
            "WHERE schemaname = 'public' AND tablename LIKE $1 ESCAPE '\\'",
            (f"{DEFAULT_GRAPH_NAME}\\_emb\\_%",),
        )
        side_tables = [row[0] for row in result]
        for table in side_tables:
            await conn.exec_driver_sql(f'DROP TABLE IF EXISTS public."{table}"')
    # Reset in-memory caches so counts and index-state don't bleed across tests.
    vector_graph_store._reset_in_memory_caches()
    yield


async def _count_vertices(engine) -> int:
    async with engine.connect() as conn:
        result = await conn.exec_driver_sql(
            (
                f"SELECT * FROM cypher('{DEFAULT_GRAPH_NAME}', "
                "$cypher_body$MATCH (n) RETURN count(n)$cypher_body$"
                ") AS (c agtype)"
            ),
        )
        row = result.first()
    if row is None:
        return 0
    return int(str(row[0]))


async def _count_edges(engine) -> int:
    async with engine.connect() as conn:
        result = await conn.exec_driver_sql(
            (
                f"SELECT * FROM cypher('{DEFAULT_GRAPH_NAME}', "
                "$cypher_body$MATCH ()-[r]->() "
                "RETURN count(r)$cypher_body$"
                ") AS (c agtype)"
            ),
        )
        row = result.first()
    if row is None:
        return 0
    return int(str(row[0]))


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_add_nodes(age_engine, vector_graph_store):
    assert await _count_vertices(age_engine) == 0

    await vector_graph_store.add_nodes(collection="Entity", nodes=[])
    assert await _count_vertices(age_engine) == 0

    nodes = [
        Node(uid=str(uuid4()), properties={"name": "Node1"}),
        Node(uid=str(uuid4()), properties={"name": "Node2"}),
        Node(
            uid=str(uuid4()),
            properties={
                "name": "Node3",
                "time": datetime.now(tz=UTC),
                "none_value": None,
            },
            embeddings={
                "embedding_name": (
                    [0.1, 0.2, 0.3],
                    SimilarityMetric.COSINE,
                ),
            },
        ),
    ]

    await vector_graph_store.add_nodes(collection="Entity", nodes=nodes)
    assert await _count_vertices(age_engine) == len(nodes)


@pytest.mark.asyncio
async def test_add_edges(age_engine, vector_graph_store):
    node1_uid = str(uuid4())
    node2_uid = str(uuid4())
    node3_uid = str(uuid4())

    nodes = [
        Node(uid=node1_uid, properties={"name": "Node1"}),
        Node(uid=node2_uid, properties={"name": "Node2"}),
        Node(
            uid=node3_uid,
            properties={"name": "Node3"},
            embeddings={
                "embedding_name": (
                    [0.1, 0.2, 0.3],
                    SimilarityMetric.COSINE,
                ),
            },
        ),
    ]
    await vector_graph_store.add_nodes(collection="Entity", nodes=nodes)
    assert await _count_edges(age_engine) == 0

    await vector_graph_store.add_edges(
        relation="RELATED_TO",
        source_collection="Entity",
        target_collection="Entity",
        edges=[],
    )
    assert await _count_edges(age_engine) == 0

    related = [
        Edge(
            uid=str(uuid4()),
            source_uid=node1_uid,
            target_uid=node2_uid,
            properties={"description": "Node1 to Node2"},
        ),
        Edge(
            uid=str(uuid4()),
            source_uid=node2_uid,
            target_uid=node1_uid,
            properties={"description": "Node2 to Node1"},
        ),
        Edge(
            uid=str(uuid4()),
            source_uid=node1_uid,
            target_uid=node3_uid,
            properties={"description": "Node1 to Node3"},
            embeddings={
                "embedding_name": (
                    [0.4, 0.5, 0.6],
                    SimilarityMetric.DOT,
                ),
            },
        ),
    ]

    is_edges = [
        Edge(
            uid=str(uuid4()),
            source_uid=node1_uid,
            target_uid=node1_uid,
            properties={"description": "Node1 loop"},
        ),
        Edge(
            uid=str(uuid4()),
            source_uid=node2_uid,
            target_uid=node2_uid,
            properties={"description": "Node2 loop"},
        ),
    ]

    await vector_graph_store.add_edges(
        relation="RELATED_TO",
        source_collection="Entity",
        target_collection="Entity",
        edges=related,
    )
    await vector_graph_store.add_edges(
        relation="IS",
        source_collection="Entity",
        target_collection="Entity",
        edges=is_edges,
    )

    assert await _count_edges(age_engine) == 5


@pytest.mark.asyncio
async def test_search_similar_nodes(vector_graph_store, vector_graph_store_ann):
    nodes = [
        Node(
            uid=str(uuid4()),
            properties={"name": "Node1"},
            embeddings={
                "embedding1": ([1000.0, 0.0], SimilarityMetric.COSINE),
                "embedding2": ([1000.0, 0.0], SimilarityMetric.EUCLIDEAN),
            },
        ),
        Node(
            uid=str(uuid4()),
            properties={"name": "Node2", "include?": "yes"},
            embeddings={
                "embedding1": ([10.0, 10.0], SimilarityMetric.COSINE),
                "embedding2": ([10.0, 10.0], SimilarityMetric.EUCLIDEAN),
            },
        ),
        Node(
            uid=str(uuid4()),
            properties={"name": "Node3", "include?": "no"},
            embeddings={
                "embedding1": ([-100.0, 0.0], SimilarityMetric.COSINE),
                "embedding2": ([-100.0, 0.0], SimilarityMetric.EUCLIDEAN),
            },
        ),
        Node(
            uid=str(uuid4()),
            properties={"name": "Node4", "include?": "no"},
            embeddings={
                "embedding1": ([-100.0, -1.0], SimilarityMetric.COSINE),
                "embedding2": ([-100.0, -1.0], SimilarityMetric.EUCLIDEAN),
            },
        ),
    ]
    await vector_graph_store.add_nodes(collection="Entity", nodes=nodes)

    # Exact search (force_exact_similarity_search=True)
    results = await vector_graph_store.search_similar_nodes(
        collection="Entity",
        query_embedding=[1.0, 0.0],
        embedding_name="embedding1",
        similarity_metric=SimilarityMetric.COSINE,
        limit=3,
    )
    assert len(results) == 3
    assert results[0].properties["name"] == "Node1"

    # Exact search with property filter should isolate Node2.
    results = await vector_graph_store.search_similar_nodes(
        collection="Entity",
        query_embedding=[1.0, 0.0],
        embedding_name="embedding1",
        similarity_metric=SimilarityMetric.COSINE,
        limit=3,
        property_filter=FilterComparison(
            field="include?",
            op="=",
            value="yes",
        ),
    )
    assert len(results) == 1
    assert results[0].properties["name"] == "Node2"

    # Filter with OR + IS NULL pulls in Node1.
    results = await vector_graph_store.search_similar_nodes(
        collection="Entity",
        query_embedding=[1.0, 0.0],
        embedding_name="embedding1",
        similarity_metric=SimilarityMetric.COSINE,
        limit=3,
        property_filter=FilterOr(
            left=FilterComparison(field="include?", op="=", value="yes"),
            right=FilterIsNull(field="include?"),
        ),
    )
    assert len(results) == 2
    assert results[0].properties["name"] == "Node1"

    # EUCLIDEAN picks Node2 (closest to [1.0, 0.0]).
    results = await vector_graph_store.search_similar_nodes(
        collection="Entity",
        query_embedding=[1.0, 0.0],
        embedding_name="embedding2",
        similarity_metric=SimilarityMetric.EUCLIDEAN,
        limit=3,
    )
    assert results[0].properties["name"] == "Node2"

    # ANN path returns something; index may still be warming.
    results = await vector_graph_store_ann.search_similar_nodes(
        collection="Entity",
        query_embedding=[1.0, 0.0],
        embedding_name="embedding1",
        similarity_metric=SimilarityMetric.COSINE,
        limit=3,
    )
    assert 0 < len(results) <= 3


@pytest.mark.asyncio
async def test_search_related_nodes(vector_graph_store):
    node1_uid = str(uuid4())
    node2_uid = str(uuid4())
    node3_uid = str(uuid4())
    node4_uid = str(uuid4())

    nodes = [
        Node(uid=node1_uid, properties={"name": "Node1"}),
        Node(
            uid=node2_uid,
            properties={"name": "Node2", "extra!": "something"},
        ),
        Node(uid=node3_uid, properties={"name": "Node3", "marker?": "A"}),
        Node(uid=node4_uid, properties={"name": "Node4", "marker?": "B"}),
    ]
    await vector_graph_store.add_nodes(collection="Entity", nodes=nodes)

    related_edges = [
        Edge(
            uid=str(uuid4()),
            source_uid=node1_uid,
            target_uid=node2_uid,
            properties={"description": "Node1 to Node2"},
        ),
        Edge(
            uid=str(uuid4()),
            source_uid=node2_uid,
            target_uid=node1_uid,
            properties={"description": "Node2 to Node1"},
        ),
        Edge(
            uid=str(uuid4()),
            source_uid=node3_uid,
            target_uid=node2_uid,
            properties={"description": "Node3 to Node2", "extra": 1},
        ),
        Edge(
            uid=str(uuid4()),
            source_uid=node3_uid,
            target_uid=node4_uid,
            properties={"description": "Node3 to Node4", "extra": 2},
        ),
    ]
    await vector_graph_store.add_edges(
        relation="RELATED_TO",
        source_collection="Entity",
        target_collection="Entity",
        edges=related_edges,
    )

    # Self-loops so the undirected match from node1/node2/node3 also returns
    # the queried node itself — mirrors the Neo4j backend's test fixture so
    # assertions stay aligned across backends.
    self_loops = [
        Edge(
            uid=str(uuid4()),
            source_uid=node1_uid,
            target_uid=node1_uid,
            properties={"description": "Node1 loop"},
        ),
        Edge(
            uid=str(uuid4()),
            source_uid=node2_uid,
            target_uid=node2_uid,
            properties={"description": "Node2 loop"},
        ),
        Edge(
            uid=str(uuid4()),
            source_uid=node3_uid,
            target_uid=node3_uid,
            properties={"description": "Node3 loop"},
        ),
    ]
    await vector_graph_store.add_edges(
        relation="RELATED_TO",
        source_collection="Entity",
        target_collection="Entity",
        edges=self_loops,
    )

    results = await vector_graph_store.search_related_nodes(
        relation="RELATED_TO",
        other_collection="Entity",
        this_collection="Entity",
        this_node_uid=node1_uid,
    )
    names = {r.properties["name"] for r in results}
    assert names == {"Node1", "Node2"}

    results = await vector_graph_store.search_related_nodes(
        relation="RELATED_TO",
        other_collection="Entity",
        this_collection="Entity",
        this_node_uid=node1_uid,
        node_property_filter=FilterComparison(
            field="extra!", op="=", value="something"
        ),
    )
    assert len(results) == 1
    assert results[0].properties["name"] == "Node2"

    results = await vector_graph_store.search_related_nodes(
        relation="RELATED_TO",
        other_collection="Entity",
        this_collection="Entity",
        this_node_uid=node3_uid,
        edge_property_filter=FilterComparison(field="extra", op="=", value=1),
    )
    assert len(results) == 1


@pytest.mark.asyncio
async def test_search_directional_nodes(vector_graph_store):
    now = datetime.now(tz=UTC)
    delta = timedelta(days=1)

    nodes = [
        Node(
            uid=str(uuid4()),
            properties={"name": "Event1", "timestamp": now},
        ),
        Node(
            uid=str(uuid4()),
            properties={
                "name": "Event2",
                "timestamp": now + delta,
                "include?": "yes",
            },
        ),
        Node(
            uid=str(uuid4()),
            properties={"name": "Event3", "timestamp": now + 2 * delta},
        ),
        Node(
            uid=str(uuid4()),
            properties={
                "name": "Event4",
                "timestamp": now + 3 * delta,
                "include?": "yes",
            },
        ),
    ]
    await vector_graph_store.add_nodes(collection="Event", nodes=nodes)

    results = await vector_graph_store.search_directional_nodes(
        collection="Event",
        by_properties=["timestamp"],
        starting_at=[now + delta],
        order_ascending=[True],
        include_equal_start=True,
        limit=2,
    )
    assert len(results) == 2
    assert results[0].properties["name"] == "Event2"
    assert results[1].properties["name"] == "Event3"

    # TZ-aware comparison across zones.
    result_ts = (
        results[0].properties["timestamp"].astimezone(ZoneInfo("America/Los_Angeles"))
    )
    results = await vector_graph_store.search_directional_nodes(
        collection="Event",
        by_properties=["timestamp"],
        starting_at=[result_ts],
        order_ascending=[False],
        include_equal_start=False,
        limit=1,
    )
    assert len(results) == 1
    assert results[0].properties["name"] != "Event2"

    # Filter-augmented directional search.
    results = await vector_graph_store.search_directional_nodes(
        collection="Event",
        by_properties=["timestamp"],
        starting_at=[now + delta],
        order_ascending=[True],
        include_equal_start=True,
        limit=2,
        property_filter=FilterComparison(field="include?", op="=", value="yes"),
    )
    assert len(results) == 2
    assert results[0].properties["name"] == "Event2"
    assert results[1].properties["name"] == "Event4"

    # Null starting value behaves as "no lower bound".
    results = await vector_graph_store.search_directional_nodes(
        collection="Event",
        by_properties=["timestamp"],
        starting_at=[None],
        order_ascending=[False],
        limit=2,
    )
    assert len(results) == 2
    assert results[0].properties["name"] == "Event4"
    assert results[1].properties["name"] == "Event3"


@pytest.mark.asyncio
async def test_search_matching_nodes(vector_graph_store):
    people = [
        Node(
            uid=str(uuid4()),
            properties={
                "name": "Alice",
                "age!with$pecialchars": 30,
                "city": "San Francisco",
                "title": "Engineer",
            },
        ),
        Node(
            uid=str(uuid4()),
            properties={
                "name": "Bob",
                "age!with$pecialchars": 25,
                "city": "Los Angeles",
                "title": "Designer",
            },
        ),
        Node(
            uid=str(uuid4()),
            properties={"name": "Charlie", "city": "New York"},
        ),
        Node(
            uid=str(uuid4()),
            properties={
                "name": "David",
                "age!with$pecialchars": 30,
                "city": "New York",
                "none_value": None,
            },
        ),
    ]
    robots = [
        Node(uid=str(uuid4()), properties={"name": "Eve", "city": "Axiom"}),
    ]
    await vector_graph_store.add_nodes(collection="Person", nodes=people)
    await vector_graph_store.add_nodes(collection="Robot", nodes=robots)

    assert len(await vector_graph_store.search_matching_nodes(collection="Person")) == 4
    assert len(await vector_graph_store.search_matching_nodes(collection="Robot")) == 1

    # IS NULL on a field nobody has -> everyone matches.
    results = await vector_graph_store.search_matching_nodes(
        collection="Robot",
        property_filter=FilterIsNull(field="none_value"),
    )
    assert len(results) == 1

    results = await vector_graph_store.search_matching_nodes(
        collection="Person",
        property_filter=FilterComparison(field="city", op="=", value="New York"),
    )
    assert len(results) == 2

    results = await vector_graph_store.search_matching_nodes(
        collection="Person",
        property_filter=FilterAnd(
            left=FilterComparison(field="city", op="=", value="New York"),
            right=FilterComparison(field="age!with$pecialchars", op="=", value=30),
        ),
    )
    assert len(results) == 1

    # OR with IS NULL includes those missing the property.
    results = await vector_graph_store.search_matching_nodes(
        collection="Person",
        property_filter=FilterOr(
            left=FilterComparison(field="title", op="=", value="Engineer"),
            right=FilterIsNull(field="title"),
        ),
    )
    assert len(results) == 3  # Alice + Charlie + David

    # Inequality comparisons.
    results = await vector_graph_store.search_matching_nodes(
        collection="Person",
        property_filter=FilterComparison(
            field="age!with$pecialchars", op=">", value=25
        ),
    )
    assert len(results) == 2

    # IN list.
    results = await vector_graph_store.search_matching_nodes(
        collection="Person",
        property_filter=FilterIn(field="city", values=["San Francisco", "Los Angeles"]),
    )
    assert len(results) == 2

    # NOT.
    results = await vector_graph_store.search_matching_nodes(
        collection="Person",
        property_filter=FilterNot(
            expr=FilterComparison(field="city", op="=", value="New York"),
        ),
    )
    assert len(results) == 2


@pytest.mark.asyncio
async def test_get_nodes(vector_graph_store):
    nodes = [
        Node(
            uid=str(uuid4()),
            properties={"name": "Node1", "time": datetime.now(tz=UTC)},
        ),
        Node(uid=str(uuid4()), properties={"name": "Node2"}),
        Node(uid=str(uuid4()), properties={"name": "Node3"}),
    ]
    await vector_graph_store.add_nodes(collection="Entity", nodes=nodes)

    fetched = await vector_graph_store.get_nodes(
        collection="Entity",
        node_uids=[n.uid for n in nodes],
    )
    assert len(fetched) == 3
    assert {n.uid for n in fetched} == {n.uid for n in nodes}

    # Partial match — requesting one real + one nonexistent uid.
    fetched = await vector_graph_store.get_nodes(
        collection="Entity",
        node_uids=[nodes[0].uid, str(uuid4())],
    )
    assert len(fetched) == 1
    assert fetched[0].uid == nodes[0].uid


@pytest.mark.asyncio
async def test_delete_nodes(age_engine, vector_graph_store):
    nodes = [Node(uid=str(uuid4())) for _ in range(6)]
    await vector_graph_store.add_nodes(collection="Entity", nodes=nodes)
    assert await _count_vertices(age_engine) == 6

    # Deleting from a non-existent collection is a no-op.
    await vector_graph_store.delete_nodes(
        collection="Bad", node_uids=[n.uid for n in nodes[:3]]
    )
    assert await _count_vertices(age_engine) == 6

    await vector_graph_store.delete_nodes(
        collection="Entity", node_uids=[n.uid for n in nodes[:3]]
    )
    assert await _count_vertices(age_engine) == 3


@pytest.mark.asyncio
async def test_delete_all_data(age_engine, vector_graph_store):
    nodes = [Node(uid=str(uuid4())) for _ in range(6)]
    await vector_graph_store.add_nodes(collection="Entity", nodes=nodes)
    assert await _count_vertices(age_engine) == 6

    await vector_graph_store.delete_all_data()
    assert await _count_vertices(age_engine) == 0


@pytest.mark.asyncio
async def test_roundtrip_preserves_embeddings(vector_graph_store):
    original = Node(
        uid=str(uuid4()),
        properties={"name": "Roundtrip"},
        embeddings={
            "default": ([0.1, 0.2, 0.3], SimilarityMetric.COSINE),
        },
    )
    await vector_graph_store.add_nodes(collection="Entity", nodes=[original])
    [fetched] = await vector_graph_store.get_nodes(
        collection="Entity",
        node_uids=[original.uid],
    )
    assert fetched.uid == original.uid
    assert fetched.properties == original.properties
    assert set(fetched.embeddings.keys()) == {"default"}
    stored_embedding, stored_metric = fetched.embeddings["default"]
    assert stored_metric is SimilarityMetric.COSINE
    assert len(stored_embedding) == 3
    # pgvector may renormalize float precision; accept small epsilon.
    for actual, expected in zip(stored_embedding, [0.1, 0.2, 0.3], strict=True):
        assert abs(actual - expected) < 1e-5


@pytest.mark.asyncio
async def test_edge_embedding_persisted_in_side_table(age_engine, vector_graph_store):
    """Edges with embeddings populate their own pgvector side table.

    The public API surface doesn't expose per-edge vector search, but the
    side-table plumbing still runs and is what ``delete_all_data`` /
    ``delete_nodes`` rely on, so we verify the rows are written.
    """
    source_uid = str(uuid4())
    target_uid = str(uuid4())
    edge_uid = str(uuid4())
    await vector_graph_store.add_nodes(
        collection="Entity",
        nodes=[
            Node(uid=source_uid, properties={"name": "Source"}),
            Node(uid=target_uid, properties={"name": "Target"}),
        ],
    )
    await vector_graph_store.add_edges(
        relation="RELATED_TO",
        source_collection="Entity",
        target_collection="Entity",
        edges=[
            Edge(
                uid=edge_uid,
                source_uid=source_uid,
                target_uid=target_uid,
                properties={"kind": "witness"},
                embeddings={
                    "default": (
                        [0.4, 0.5, 0.6],
                        SimilarityMetric.EUCLIDEAN,
                    ),
                },
            ),
        ],
    )

    # Confirm the edge side table exists and contains our one row. Side-table
    # names are hashed to stay within Postgres' 63-byte identifier limit, so
    # we resolve them through the registry rather than scanning pg_tables by
    # prefix.
    async with age_engine.begin() as conn:
        result = await conn.exec_driver_sql(
            "SELECT table_name FROM public.\"{registry}\" WHERE kind = 'edge'".format(
                registry=f"{DEFAULT_GRAPH_NAME}_emb_registry"
            ),
        )
        edge_tables = [row[0] for row in result]
        assert len(edge_tables) == 1, (
            f"expected exactly one edge side table, got {edge_tables}"
        )
        [table] = edge_tables
        count_result = await conn.exec_driver_sql(
            f'SELECT COUNT(*) FROM public."{table}" WHERE edge_uid = $1',
            (edge_uid,),
        )
        count_row = count_result.first()
    assert count_row is not None
    assert count_row[0] == 1
