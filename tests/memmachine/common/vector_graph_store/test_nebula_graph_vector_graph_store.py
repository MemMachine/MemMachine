import os
from datetime import UTC, datetime, timedelta
from uuid import uuid4

import pytest
import pytest_asyncio

from memmachine.common.data_types import SimilarityMetric
from memmachine.common.filter.filter_parser import (
    And as FilterAnd,
)
from memmachine.common.filter.filter_parser import (
    Comparison as FilterComparison,
)
from memmachine.common.filter.filter_parser import (
    Or as FilterOr,
)
from memmachine.common.metrics_factory.prometheus_metrics_factory import (
    PrometheusMetricsFactory,
)
from memmachine.common.vector_graph_store.data_types import Edge, Node
from memmachine.common.vector_graph_store.nebula_graph_vector_graph_store import (
    NebulaGraphVectorGraphStore,
    NebulaGraphVectorGraphStoreParams,
)

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def metrics_factory():
    return PrometheusMetricsFactory()


@pytest.fixture(scope="module")
def nebula_connection_info():
    """NebulaGraph connection info from environment variables.

    Set these environment variables to test with NebulaGraph:
    - NEBULA_HOST (default: 127.0.0.1:9669)
    - NEBULA_USER (default: root)
    - NEBULA_PASSWORD (default: nebula)

    Note: Requires NebulaGraph Enterprise >= 5.0 for vector support.
    """
    nebula_host = os.environ.get("NEBULA_HOST", "127.0.0.1:9669")
    nebula_user = os.environ.get("NEBULA_USER", "root")
    nebula_password = os.environ.get("NEBULA_PASSWORD", "nebula")

    return {
        "hosts": [nebula_host],
        "username": nebula_user,
        "password": nebula_password,
        "schema_name": "/test_schema",
        "graph_type_name": "memmachine_type",
        "graph_name": "test_graph",
    }


@pytest_asyncio.fixture(scope="module")
async def nebula_client(nebula_connection_info):
    """Create NebulaGraph async client for testing, skip if unavailable."""
    try:
        from nebulagraph_python.client import NebulaAsyncClient, SessionConfig
    except ImportError:
        pytest.skip("nebulagraph_python not installed")

    try:
        # Connect with empty SessionConfig (schema/graph don't exist yet)
        client = await NebulaAsyncClient.connect(
            hosts=nebula_connection_info["hosts"],
            username=nebula_connection_info["username"],
            password=nebula_connection_info["password"],
            session_config=SessionConfig(),
        )

        # Initialize schema, graph type, and graph
        await client.execute(
            f"CREATE SCHEMA IF NOT EXISTS {nebula_connection_info['schema_name']}"
        )
        await client.execute(
            f"SESSION SET SCHEMA {nebula_connection_info['schema_name']}"
        )

        # Create empty graph type
        await client.execute(
            f"CREATE GRAPH TYPE IF NOT EXISTS {nebula_connection_info['graph_type_name']} AS {{}}"
        )

        # Create graph
        await client.execute(
            f"CREATE GRAPH IF NOT EXISTS {nebula_connection_info['graph_name']} TYPED {nebula_connection_info['graph_type_name']}"
        )
        await client.execute(
            f"SESSION SET GRAPH {nebula_connection_info['graph_name']}"
        )

        yield client

        # Cleanup after tests - must drop in order: graphs -> graph types -> schema
        await client.execute(
            f"SESSION SET SCHEMA {nebula_connection_info['schema_name']}"
        )
        await client.execute(
            f"DROP GRAPH IF EXISTS {nebula_connection_info['graph_name']}"
        )
        await client.execute(
            f"DROP GRAPH TYPE IF EXISTS {nebula_connection_info['graph_type_name']}"
        )
        await client.execute(
            f"DROP SCHEMA IF EXISTS {nebula_connection_info['schema_name']}"
        )
        await client.close()

    except Exception as e:
        # Skip tests if NebulaGraph is not available (e.g., in CI without deployment)
        pytest.skip(f"NebulaGraph not available: {e}")


@pytest.fixture(scope="module")
def vector_graph_store(nebula_client, nebula_connection_info, metrics_factory):
    """Vector graph store with exact search (KNN)."""
    return NebulaGraphVectorGraphStore(
        NebulaGraphVectorGraphStoreParams(
            client=nebula_client,
            schema_name=nebula_connection_info["schema_name"],
            graph_type_name=nebula_connection_info["graph_type_name"],
            graph_name=nebula_connection_info["graph_name"],
            force_exact_similarity_search=True,
            metrics_factory=metrics_factory,
        ),
    )


@pytest.fixture(scope="module")
def vector_graph_store_ann(nebula_client, nebula_connection_info):
    """Vector graph store with ANN search enabled."""
    return NebulaGraphVectorGraphStore(
        NebulaGraphVectorGraphStoreParams(
            client=nebula_client,
            schema_name=nebula_connection_info["schema_name"],
            graph_type_name=nebula_connection_info["graph_type_name"],
            graph_name=nebula_connection_info["graph_name"],
            force_exact_similarity_search=False,
            filtered_similarity_search_fudge_factor=2,
            exact_similarity_search_fallback_threshold=0.5,
            range_index_creation_threshold=0,
            vector_index_creation_threshold=0,  # Create index immediately
        ),
    )


@pytest_asyncio.fixture(autouse=True)
async def db_cleanup(vector_graph_store):
    """Clean up database before each test."""
    await vector_graph_store.delete_all_data()
    yield
    # Cleanup after test as well
    await vector_graph_store.delete_all_data()


@pytest.mark.asyncio
async def test_add_nodes(nebula_client, vector_graph_store):
    """Test adding nodes to a collection."""
    collection = "test_collection"

    nodes = [
        Node(
            uid=str(uuid4()),
            properties={"name": "Alice", "age": 30},
            embeddings={"vec": ([1.0, 2.0, 3.0], SimilarityMetric.COSINE)},
        ),
        Node(
            uid=str(uuid4()),
            properties={"name": "Bob", "age": 25},
            embeddings={"vec": ([4.0, 5.0, 6.0], SimilarityMetric.COSINE)},
        ),
    ]

    await vector_graph_store.add_nodes(collection=collection, nodes=nodes)

    # Verify nodes were added
    retrieved = await vector_graph_store.get_nodes(
        collection=collection,
        node_uids=[n.uid for n in nodes],
    )

    assert len(retrieved) == 2
    assert {n.uid for n in retrieved} == {n.uid for n in nodes}

    # Verify properties and embeddings
    retrieved_by_uid = {n.uid: n for n in retrieved}
    for original in nodes:
        retrieved_node = retrieved_by_uid[original.uid]
        assert retrieved_node.properties["name"] == original.properties["name"]
        assert retrieved_node.properties["age"] == original.properties["age"]

        assert retrieved_node.embeddings.keys() == original.embeddings.keys()
        for emb_name, (original_vec, original_metric) in original.embeddings.items():
            retrieved_vec, retrieved_metric = retrieved_node.embeddings[emb_name]
            assert retrieved_metric == original_metric
            assert retrieved_vec == pytest.approx(original_vec, abs=1e-5)


@pytest.mark.asyncio
async def test_add_edges(nebula_client, vector_graph_store):
    """Test adding edges between nodes, including embedding round-trip for both nodes and edges."""
    source_collection = "person"
    target_collection = "company"
    relation = "works_at"

    # Create source and target nodes with embeddings
    person = Node(
        uid=str(uuid4()),
        properties={"name": "Alice"},
        embeddings={"profile": ([1.0, 0.0, 0.0], SimilarityMetric.EUCLIDEAN)},
    )
    company = Node(
        uid=str(uuid4()),
        properties={"name": "Acme Corp"},
        embeddings={"profile": ([0.0, 1.0, 0.0], SimilarityMetric.EUCLIDEAN)},
    )

    await vector_graph_store.add_nodes(collection=source_collection, nodes=[person])
    await vector_graph_store.add_nodes(collection=target_collection, nodes=[company])

    # Add edge with embeddings
    edge = Edge(
        uid=str(uuid4()),
        source_uid=person.uid,
        target_uid=company.uid,
        properties={"since": 2020, "role": "Engineer"},
        embeddings={"relation_vec": ([0.5, 0.5, 0.0], SimilarityMetric.EUCLIDEAN)},
    )

    await vector_graph_store.add_edges(
        relation=relation,
        source_collection=source_collection,
        target_collection=target_collection,
        edges=[edge],
    )

    # Verify edge connectivity and node embeddings via search_related_nodes
    related = await vector_graph_store.search_related_nodes(
        relation=relation,
        other_collection=target_collection,
        this_collection=source_collection,
        this_node_uid=person.uid,
        find_targets=True,
        find_sources=False,
    )

    assert len(related) == 1
    assert related[0].uid == company.uid
    assert related[0].properties["name"] == "Acme Corp"

    # Verify the returned company node's embeddings are correctly round-tripped
    assert related[0].embeddings.keys() == company.embeddings.keys()
    for emb_name, (original_vec, original_metric) in company.embeddings.items():
        retrieved_vec, retrieved_metric = related[0].embeddings[emb_name]
        assert retrieved_metric == original_metric
        assert retrieved_vec == pytest.approx(original_vec, abs=1e-5)


@pytest.mark.asyncio
async def test_search_similar_nodes(vector_graph_store):
    """Test vector similarity search (KNN / exact search)."""
    collection = "documents"

    nodes = [
        Node(
            uid=str(uuid4()),
            properties={"title": "Doc 1"},
            embeddings={"content": ([1.0, 0.0, 0.0], SimilarityMetric.EUCLIDEAN)},
        ),
        Node(
            uid=str(uuid4()),
            properties={"title": "Doc 2"},
            embeddings={"content": ([0.0, 1.0, 0.0], SimilarityMetric.EUCLIDEAN)},
        ),
        Node(
            uid=str(uuid4()),
            properties={"title": "Doc 3"},
            embeddings={"content": ([0.9, 0.1, 0.0], SimilarityMetric.EUCLIDEAN)},
        ),
    ]

    await vector_graph_store.add_nodes(collection=collection, nodes=nodes)

    # Query closest to Doc 1 ([1,0,0]).
    # Euclidean distances: Doc1=0, Doc3≈0.141, Doc2≈1.414 → ranked order: Doc1, Doc3.
    query_vec = [1.0, 0.0, 0.0]
    results = await vector_graph_store.search_similar_nodes(
        collection=collection,
        embedding_name="content",
        query_embedding=query_vec,
        similarity_metric=SimilarityMetric.EUCLIDEAN,
        limit=2,
    )

    assert len(results) == 2
    assert results[0].properties["title"] == "Doc 1"
    assert results[1].properties["title"] == "Doc 3"


@pytest.mark.asyncio
async def test_search_similar_nodes_with_filter(vector_graph_store):
    """Test vector similarity search with property filter."""
    collection = "documents"

    nodes = [
        Node(
            uid=str(uuid4()),
            properties={"title": "Doc 1", "category": "tech"},
            embeddings={"content": ([1.0, 0.0, 0.0], SimilarityMetric.EUCLIDEAN)},
        ),
        Node(
            uid=str(uuid4()),
            properties={"title": "Doc 2", "category": "business"},
            embeddings={"content": ([0.9, 0.1, 0.0], SimilarityMetric.EUCLIDEAN)},
        ),
        Node(
            uid=str(uuid4()),
            properties={"title": "Doc 3", "category": "tech"},
            embeddings={"content": ([0.8, 0.2, 0.0], SimilarityMetric.EUCLIDEAN)},
        ),
    ]

    await vector_graph_store.add_nodes(collection=collection, nodes=nodes)

    # Search with filter
    query_vec = [1.0, 0.0, 0.0]
    filter_expr = FilterComparison(field="category", op="==", value="tech")

    results = await vector_graph_store.search_similar_nodes(
        collection=collection,
        embedding_name="content",
        query_embedding=query_vec,
        similarity_metric=SimilarityMetric.EUCLIDEAN,
        limit=10,
        property_filter=filter_expr,
    )

    assert len(results) == 2
    for node in results:
        assert node.properties["category"] == "tech"


@pytest.mark.asyncio
async def test_search_related_nodes(vector_graph_store):
    """Test searching for related nodes via edges: directions, node filter, edge filter."""
    person_collection = "person"
    company_collection = "company"
    relation = "works_at"

    # Create nodes
    alice = Node(
        uid=str(uuid4()),
        properties={"name": "Alice"},
        embeddings={},
    )
    bob = Node(
        uid=str(uuid4()),
        properties={"name": "Bob"},
        embeddings={},
    )
    acme = Node(
        uid=str(uuid4()),
        properties={"name": "Acme", "industry": "tech"},
        embeddings={},
    )
    techcorp = Node(
        uid=str(uuid4()),
        properties={"name": "TechCorp"},
        embeddings={},
    )

    await vector_graph_store.add_nodes(collection=person_collection, nodes=[alice, bob])
    await vector_graph_store.add_nodes(
        collection=company_collection, nodes=[acme, techcorp]
    )

    # Create edges: alice→acme (Engineer), bob→acme (Manager), bob→techcorp (CTO)
    edges = [
        Edge(
            uid=str(uuid4()),
            source_uid=alice.uid,
            target_uid=acme.uid,
            properties={"role": "Engineer", "seniority": 1},
        ),
        Edge(
            uid=str(uuid4()),
            source_uid=bob.uid,
            target_uid=acme.uid,
            properties={"role": "Manager", "seniority": 2},
        ),
        Edge(
            uid=str(uuid4()),
            source_uid=bob.uid,
            target_uid=techcorp.uid,
            properties={"role": "CTO"},
        ),
    ]

    await vector_graph_store.add_edges(
        relation=relation,
        source_collection=person_collection,
        target_collection=company_collection,
        edges=edges,
    )

    # find_targets=True, find_sources=False → companies where alice works
    results = await vector_graph_store.search_related_nodes(
        relation=relation,
        other_collection=company_collection,
        this_collection=person_collection,
        this_node_uid=alice.uid,
        find_targets=True,
        find_sources=False,
    )
    assert len(results) == 1
    assert results[0].uid == acme.uid

    # find_sources=True, find_targets=False → people who work at acme (reverse direction)
    results = await vector_graph_store.search_related_nodes(
        relation=relation,
        other_collection=person_collection,
        this_collection=company_collection,
        this_node_uid=acme.uid,
        find_sources=True,
        find_targets=False,
    )
    assert len(results) == 2
    names = {r.properties["name"] for r in results}
    assert names == {"Alice", "Bob"}

    # Both directions (default) — bob is both a source (→ acme, techcorp) and not a target
    results = await vector_graph_store.search_related_nodes(
        relation=relation,
        other_collection=company_collection,
        this_collection=person_collection,
        this_node_uid=bob.uid,
        find_sources=True,
        find_targets=True,
    )
    assert len(results) == 2
    names = {r.properties["name"] for r in results}
    assert names == {"Acme", "TechCorp"}

    # node_property_filter: only companies with industry == "tech"
    results = await vector_graph_store.search_related_nodes(
        relation=relation,
        other_collection=company_collection,
        this_collection=person_collection,
        this_node_uid=bob.uid,
        find_targets=True,
        find_sources=False,
        node_property_filter=FilterComparison(field="industry", op="==", value="tech"),
    )
    assert len(results) == 1
    assert results[0].properties["name"] == "Acme"

    # edge_property_filter: only edges where seniority == 2
    results = await vector_graph_store.search_related_nodes(
        relation=relation,
        other_collection=company_collection,
        this_collection=person_collection,
        this_node_uid=bob.uid,
        find_targets=True,
        find_sources=False,
        edge_property_filter=FilterComparison(field="seniority", op="==", value=2),
    )
    assert len(results) == 1
    assert results[0].properties["name"] == "Acme"


@pytest.mark.asyncio
async def test_search_directional_nodes(vector_graph_store):
    """Test searching nodes with directional ordering: directions, include_equal_start, filter, None start."""
    collection = "events"

    now = datetime.now(UTC)
    # priority values double as a rank marker
    nodes = [
        Node(
            uid=str(uuid4()),
            properties={"timestamp": now - timedelta(hours=3), "priority": 1, "tagged": "no"},
            embeddings={},
        ),
        Node(
            uid=str(uuid4()),
            properties={"timestamp": now - timedelta(hours=2), "priority": 2, "tagged": "yes"},
            embeddings={},
        ),
        Node(
            uid=str(uuid4()),
            properties={"timestamp": now - timedelta(hours=1), "priority": 3, "tagged": "yes"},
            embeddings={},
        ),
    ]

    await vector_graph_store.add_nodes(collection=collection, nodes=nodes)

    start_time = now - timedelta(hours=2, minutes=30)

    # Ascending, exclude equal start → events with priority 2 and 3
    results = await vector_graph_store.search_directional_nodes(
        collection=collection,
        by_properties=["timestamp"],
        starting_at=[start_time],
        order_ascending=[True],
        include_equal_start=False,
        limit=10,
    )
    assert len(results) == 2
    assert results[0].properties["priority"] == 2
    assert results[1].properties["priority"] == 3

    # Ascending, include equal start — boundary is exactly at priority-2 timestamp
    results = await vector_graph_store.search_directional_nodes(
        collection=collection,
        by_properties=["timestamp"],
        starting_at=[now - timedelta(hours=2)],
        order_ascending=[True],
        include_equal_start=True,
        limit=10,
    )
    assert len(results) == 2
    assert results[0].properties["priority"] == 2
    assert results[1].properties["priority"] == 3

    # Ascending, exclude equal start with same boundary → only priority 3
    results = await vector_graph_store.search_directional_nodes(
        collection=collection,
        by_properties=["timestamp"],
        starting_at=[now - timedelta(hours=2)],
        order_ascending=[True],
        include_equal_start=False,
        limit=10,
    )
    assert len(results) == 1
    assert results[0].properties["priority"] == 3

    # Descending, exclude equal start → only priority 1
    results = await vector_graph_store.search_directional_nodes(
        collection=collection,
        by_properties=["timestamp"],
        starting_at=[now - timedelta(hours=2)],
        order_ascending=[False],
        include_equal_start=False,
        limit=10,
    )
    assert len(results) == 1
    assert results[0].properties["priority"] == 1

    # Descending, include equal start → priority 2 and 1 in descending order
    results = await vector_graph_store.search_directional_nodes(
        collection=collection,
        by_properties=["timestamp"],
        starting_at=[now - timedelta(hours=2)],
        order_ascending=[False],
        include_equal_start=True,
        limit=10,
    )
    assert len(results) == 2
    assert results[0].properties["priority"] == 2
    assert results[1].properties["priority"] == 1

    # property_filter applied alongside directional ordering
    results = await vector_graph_store.search_directional_nodes(
        collection=collection,
        by_properties=["timestamp"],
        starting_at=[start_time],
        order_ascending=[True],
        include_equal_start=False,
        limit=10,
        property_filter=FilterComparison(field="tagged", op="==", value="yes"),
    )
    assert len(results) == 2
    assert all(r.properties["tagged"] == "yes" for r in results)

    # starting_at=[None] → no lower bound, descending → all nodes newest first
    results = await vector_graph_store.search_directional_nodes(
        collection=collection,
        by_properties=["timestamp"],
        starting_at=[None],
        order_ascending=[False],
        limit=2,
    )
    assert len(results) == 2
    assert results[0].properties["priority"] == 3
    assert results[1].properties["priority"] == 2


@pytest.mark.asyncio
async def test_search_matching_nodes(vector_graph_store):
    """Test searching nodes with property filters: no filter, is_null, AND."""
    collection = "products"

    nodes = [
        Node(
            uid=str(uuid4()),
            properties={"name": "Laptop", "price": 1000, "category": "electronics"},
            embeddings={},
        ),
        Node(
            uid=str(uuid4()),
            properties={"name": "Mouse", "price": 25, "category": "electronics"},
            embeddings={},
        ),
        Node(
            uid=str(uuid4()),
            # No "price" property — used for is_null test
            properties={"name": "Desk", "category": "furniture"},
            embeddings={},
        ),
    ]

    await vector_graph_store.add_nodes(collection=collection, nodes=nodes)

    # No filter → all nodes
    results = await vector_graph_store.search_matching_nodes(
        collection=collection,
        limit=10,
    )
    assert len(results) == 3

    # is_null filter: nodes with no "price" property
    results = await vector_graph_store.search_matching_nodes(
        collection=collection,
        property_filter=FilterComparison(field="price", op="is_null", value=None),
        limit=10,
    )
    assert len(results) == 1
    assert results[0].properties["name"] == "Desk"

    # AND filter: electronics AND price < 100
    filter_expr = FilterAnd(
        left=FilterComparison(field="category", op="==", value="electronics"),
        right=FilterComparison(field="price", op="<", value=100),
    )
    results = await vector_graph_store.search_matching_nodes(
        collection=collection,
        property_filter=filter_expr,
        limit=10,
    )
    assert len(results) == 1
    assert results[0].properties["name"] == "Mouse"


@pytest.mark.asyncio
async def test_get_nodes(vector_graph_store):
    """Test getting nodes by UIDs."""
    collection = "users"

    nodes = [
        Node(uid=str(uuid4()), properties={"name": "Alice"}, embeddings={}),
        Node(uid=str(uuid4()), properties={"name": "Bob"}, embeddings={}),
        Node(uid=str(uuid4()), properties={"name": "Charlie"}, embeddings={}),
    ]

    await vector_graph_store.add_nodes(collection=collection, nodes=nodes)

    # Get specific nodes
    uids_to_get = [nodes[0].uid, nodes[2].uid]
    results = await vector_graph_store.get_nodes(
        collection=collection,
        node_uids=uids_to_get,
    )

    assert len(results) == 2
    assert {n.uid for n in results} == set(uids_to_get)


@pytest.mark.asyncio
async def test_delete_nodes(nebula_client, vector_graph_store):
    """Test deleting nodes."""
    collection = "temp_data"

    nodes = [
        Node(uid=str(uuid4()), properties={"value": 1}, embeddings={}),
        Node(uid=str(uuid4()), properties={"value": 2}, embeddings={}),
    ]

    await vector_graph_store.add_nodes(collection=collection, nodes=nodes)

    # Delete first node
    await vector_graph_store.delete_nodes(
        collection=collection,
        node_uids=[nodes[0].uid],
    )

    # Verify deletion
    remaining = await vector_graph_store.get_nodes(
        collection=collection,
        node_uids=[n.uid for n in nodes],
    )

    assert len(remaining) == 1
    assert remaining[0].uid == nodes[1].uid


@pytest.mark.asyncio
async def test_delete_all_data(nebula_client, vector_graph_store):
    """Test deleting all data from the graph."""
    collection = "test_data"

    nodes = [
        Node(uid=str(uuid4()), properties={"value": i}, embeddings={}) for i in range(5)
    ]

    await vector_graph_store.add_nodes(collection=collection, nodes=nodes)

    # Delete all data
    await vector_graph_store.delete_all_data()

    # Verify all data is gone
    results = await vector_graph_store.get_nodes(
        collection=collection,
        node_uids=[n.uid for n in nodes],
    )

    assert len(results) == 0


@pytest.mark.asyncio
async def test_sanitize_name():
    """Test name sanitization for GQL identifiers."""
    from memmachine.common.vector_graph_store.nebula_graph_vector_graph_store import (
        NebulaGraphVectorGraphStore,
    )

    # Test special characters
    assert (
        NebulaGraphVectorGraphStore._sanitize_name("my-collection")
        == "SANITIZED_my_u2d_collection"
    )
    assert (
        NebulaGraphVectorGraphStore._sanitize_name("my.field")
        == "SANITIZED_my_u2e_field"
    )
    assert (
        NebulaGraphVectorGraphStore._sanitize_name("my collection")
        == "SANITIZED_my_u20_collection"
    )

    # Test desanitization
    sanitized = NebulaGraphVectorGraphStore._sanitize_name("my-collection")
    desanitized = NebulaGraphVectorGraphStore._desanitize_name(sanitized)
    assert desanitized == "my-collection"


@pytest.mark.asyncio
async def test_complex_filters(vector_graph_store):
    """Test complex filter expressions with AND and OR."""
    collection = "products"

    nodes = [
        Node(
            uid=str(uuid4()),
            properties={
                "name": "Laptop",
                "price": 1000,
                "stock": 5,
                "category": "electronics",
            },
            embeddings={},
        ),
        Node(
            uid=str(uuid4()),
            properties={
                "name": "Mouse",
                "price": 25,
                "stock": 50,
                "category": "electronics",
            },
            embeddings={},
        ),
        Node(
            uid=str(uuid4()),
            properties={
                "name": "Premium Mouse",
                "price": 80,
                "stock": 20,
                "category": "electronics",
            },
            embeddings={},
        ),
        Node(
            uid=str(uuid4()),
            properties={
                "name": "Desk",
                "price": 300,
                "stock": 10,
                "category": "furniture",
            },
            embeddings={},
        ),
    ]

    await vector_graph_store.add_nodes(collection=collection, nodes=nodes)

    # Complex filter: (category == electronics AND price < 100) OR (category == furniture)
    filter_expr = FilterOr(
        left=FilterAnd(
            left=FilterComparison(field="category", op="==", value="electronics"),
            right=FilterComparison(field="price", op="<", value=100),
        ),
        right=FilterComparison(field="category", op="==", value="furniture"),
    )

    results = await vector_graph_store.search_matching_nodes(
        collection=collection,
        property_filter=filter_expr,
        limit=10,
    )

    # Should match: Mouse (electronics, <100), Premium Mouse (electronics, <100), Desk (furniture)
    assert len(results) == 3
    names = {n.properties["name"] for n in results}
    assert names == {"Mouse", "Premium Mouse", "Desk"}


# ---------------------------------------------------------------------------
# Similarity metric table coverage
# ---------------------------------------------------------------------------


def test_similarity_metric_mappings():
    """Unit test covering every row of the metric support table.

    | Metric    | _similarity_metric_to_nebula | _get_distance_func_and_order | ANN possible? |
    |-----------|------------------------------|------------------------------|---------------|
    | EUCLIDEAN | "L2"                         | euclidean() ASC              | yes           |
    | DOT       | "IP"                         | inner_product() DESC         | yes           |
    | COSINE    | None (no index)              | cosine() DESC                | no — KNN only |
    | MANHATTAN | None (no index)              | raises ValueError            | no            |
    """
    # _similarity_metric_to_nebula
    assert NebulaGraphVectorGraphStore._similarity_metric_to_nebula(SimilarityMetric.EUCLIDEAN) == "L2"
    assert NebulaGraphVectorGraphStore._similarity_metric_to_nebula(SimilarityMetric.DOT) == "IP"
    assert NebulaGraphVectorGraphStore._similarity_metric_to_nebula(SimilarityMetric.COSINE) is None
    assert NebulaGraphVectorGraphStore._similarity_metric_to_nebula(SimilarityMetric.MANHATTAN) is None

    # _get_distance_func_and_order
    assert NebulaGraphVectorGraphStore._get_distance_func_and_order(SimilarityMetric.EUCLIDEAN) == ("euclidean", "ASC")
    assert NebulaGraphVectorGraphStore._get_distance_func_and_order(SimilarityMetric.DOT) == ("inner_product", "DESC")
    assert NebulaGraphVectorGraphStore._get_distance_func_and_order(SimilarityMetric.COSINE) == ("cosine", "DESC")
    with pytest.raises(ValueError, match="manhattan"):
        NebulaGraphVectorGraphStore._get_distance_func_and_order(SimilarityMetric.MANHATTAN)


@pytest.mark.asyncio
async def test_search_similar_nodes_dot_metric(vector_graph_store):
    """DOT metric: inner_product() DESC, KNN search."""
    collection = "dot_docs"

    nodes = [
        Node(
            uid=str(uuid4()),
            properties={"title": "Doc 1"},
            embeddings={"content": ([1.0, 0.0, 0.0], SimilarityMetric.DOT)},
        ),
        Node(
            uid=str(uuid4()),
            properties={"title": "Doc 2"},
            embeddings={"content": ([0.0, 1.0, 0.0], SimilarityMetric.DOT)},
        ),
        Node(
            uid=str(uuid4()),
            properties={"title": "Doc 3"},
            embeddings={"content": ([0.9, 0.1, 0.0], SimilarityMetric.DOT)},
        ),
    ]

    await vector_graph_store.add_nodes(collection=collection, nodes=nodes)

    # Inner products with query [1,0,0]: Doc1=1.0, Doc3=0.9, Doc2=0.0 → ranked order: Doc1, Doc3
    query_vec = [1.0, 0.0, 0.0]
    results = await vector_graph_store.search_similar_nodes(
        collection=collection,
        embedding_name="content",
        query_embedding=query_vec,
        similarity_metric=SimilarityMetric.DOT,
        limit=2,
    )

    assert len(results) == 2
    assert results[0].properties["title"] == "Doc 1"
    assert results[1].properties["title"] == "Doc 3"


@pytest.mark.asyncio
async def test_search_similar_nodes_cosine_metric(vector_graph_store):
    """COSINE metric: cosine() DESC, always KNN (no ANN index)."""
    collection = "cosine_docs"

    nodes = [
        Node(
            uid=str(uuid4()),
            properties={"title": "Doc 1"},
            embeddings={"content": ([1.0, 0.0, 0.0], SimilarityMetric.COSINE)},
        ),
        Node(
            uid=str(uuid4()),
            properties={"title": "Doc 2"},
            embeddings={"content": ([0.0, 1.0, 0.0], SimilarityMetric.COSINE)},
        ),
        Node(
            uid=str(uuid4()),
            properties={"title": "Doc 3"},
            # Slightly off-axis: cosine ≈ 0.993 with [1,0,0]
            embeddings={"content": ([0.9, 0.1, 0.0], SimilarityMetric.COSINE)},
        ),
    ]

    await vector_graph_store.add_nodes(collection=collection, nodes=nodes)

    # Cosine similarities with [1,0,0]: Doc1=1.0, Doc3≈0.993, Doc2=0.0 → ranked: Doc1, Doc3
    query_vec = [1.0, 0.0, 0.0]
    results = await vector_graph_store.search_similar_nodes(
        collection=collection,
        embedding_name="content",
        query_embedding=query_vec,
        similarity_metric=SimilarityMetric.COSINE,
        limit=2,
    )

    assert len(results) == 2
    assert results[0].properties["title"] == "Doc 1"
    assert results[1].properties["title"] == "Doc 3"


@pytest.mark.asyncio
async def test_search_similar_nodes_manhattan_raises(vector_graph_store):
    """MANHATTAN metric: not supported by NebulaGraph, raises ValueError."""
    collection = "manhattan_docs"

    nodes = [
        Node(
            uid=str(uuid4()),
            properties={"title": "Doc 1"},
            embeddings={"content": ([1.0, 0.0, 0.0], SimilarityMetric.EUCLIDEAN)},
        ),
    ]
    await vector_graph_store.add_nodes(collection=collection, nodes=nodes)

    with pytest.raises(ValueError, match="manhattan"):
        await vector_graph_store.search_similar_nodes(
            collection=collection,
            embedding_name="content",
            query_embedding=[1.0, 0.0, 0.0],
            similarity_metric=SimilarityMetric.MANHATTAN,
            limit=1,
        )


# ---------------------------------------------------------------------------
# Multi-property directional search
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_search_directional_nodes_multiple_by_properties(vector_graph_store):
    """Test search_directional_nodes with multiple sort properties (timestamp + sequence)."""
    collection = "seq_events"

    now = datetime.now(UTC)
    delta = timedelta(hours=1)

    # Two timestamps × two sequence values = 4 nodes
    nodes = [
        Node(
            uid=str(uuid4()),
            properties={"name": "T1S1", "timestamp": now, "sequence": 1},
            embeddings={},
        ),
        Node(
            uid=str(uuid4()),
            properties={"name": "T1S2", "timestamp": now, "sequence": 2},
            embeddings={},
        ),
        Node(
            uid=str(uuid4()),
            properties={"name": "T2S1", "timestamp": now + delta, "sequence": 1},
            embeddings={},
        ),
        Node(
            uid=str(uuid4()),
            properties={"name": "T2S2", "timestamp": now + delta, "sequence": 2},
            embeddings={},
        ),
    ]

    await vector_graph_store.add_nodes(collection=collection, nodes=nodes)

    # Start at (T1, S2) inclusive, both ascending → T1S2, T2S1, T2S2
    results = await vector_graph_store.search_directional_nodes(
        collection=collection,
        by_properties=["timestamp", "sequence"],
        starting_at=[now, 2],
        order_ascending=[True, True],
        include_equal_start=True,
        limit=None,
    )
    assert len(results) == 3
    assert results[0].properties["name"] == "T1S2"
    assert results[1].properties["name"] == "T2S1"
    assert results[2].properties["name"] == "T2S2"

    # Start at (T2, S1) inclusive, first ascending second descending → T2S1, T2S2 reversed
    # (same first key, second key descending from S1 means S1 then... S2 > S1 so excluded)
    # Actually: ascending timestamp, descending sequence from S1 inclusive:
    # At T2: include S1 (equal, inclusive), nothing below S1 for descending → just T2S1
    results = await vector_graph_store.search_directional_nodes(
        collection=collection,
        by_properties=["timestamp", "sequence"],
        starting_at=[now + delta, 1],
        order_ascending=[True, False],
        include_equal_start=True,
        limit=None,
    )
    assert len(results) == 1
    assert results[0].properties["name"] == "T2S1"


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_add_nodes_empty_list(vector_graph_store):
    """Adding an empty list of nodes is a no-op."""
    collection = "empty_test"

    await vector_graph_store.add_nodes(collection=collection, nodes=[])

    results = await vector_graph_store.search_matching_nodes(
        collection=collection,
        limit=10,
    )
    assert len(results) == 0


@pytest.mark.asyncio
async def test_add_edges_empty_list(vector_graph_store):
    """Adding an empty list of edges is a no-op."""
    node = Node(uid=str(uuid4()), properties={"name": "Solo"}, embeddings={})
    await vector_graph_store.add_nodes(collection="solo", nodes=[node])

    # Should not raise
    await vector_graph_store.add_edges(
        relation="knows",
        source_collection="solo",
        target_collection="solo",
        edges=[],
    )

    # Node still exists, no edges created
    results = await vector_graph_store.get_nodes(
        collection="solo", node_uids=[node.uid]
    )
    assert len(results) == 1


@pytest.mark.asyncio
async def test_add_nodes_with_none_property(vector_graph_store):
    """Nodes with None property values are stored and retrieved without error."""
    collection = "nullable_test"

    node = Node(
        uid=str(uuid4()),
        properties={"name": "Alice", "optional_field": None},
        embeddings={},
    )
    await vector_graph_store.add_nodes(collection=collection, nodes=[node])

    results = await vector_graph_store.get_nodes(
        collection=collection, node_uids=[node.uid]
    )
    assert len(results) == 1
    # None properties may be omitted on retrieval (same behaviour as Neo4j)
    assert results[0].properties.get("name") == "Alice"


@pytest.mark.asyncio
async def test_get_nodes_with_nonexistent_uids(vector_graph_store):
    """get_nodes ignores UIDs that do not exist — returns only found nodes."""
    collection = "partial_get"

    node = Node(uid=str(uuid4()), properties={"name": "Real"}, embeddings={})
    await vector_graph_store.add_nodes(collection=collection, nodes=[node])

    fake_uid = str(uuid4())
    results = await vector_graph_store.get_nodes(
        collection=collection,
        node_uids=[node.uid, fake_uid],
    )
    assert len(results) == 1
    assert results[0].uid == node.uid


@pytest.mark.asyncio
async def test_delete_nodes_wrong_collection(vector_graph_store):
    """Deleting from a non-matching collection leaves nodes untouched."""
    collection = "real_collection"
    wrong_collection = "wrong_collection"

    node = Node(uid=str(uuid4()), properties={"name": "Keep"}, embeddings={})
    await vector_graph_store.add_nodes(collection=collection, nodes=[node])

    # Attempt to delete from the wrong collection
    await vector_graph_store.delete_nodes(
        collection=wrong_collection, node_uids=[node.uid]
    )

    results = await vector_graph_store.get_nodes(
        collection=collection, node_uids=[node.uid]
    )
    assert len(results) == 1


# ---------------------------------------------------------------------------
# ANN mode
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_search_similar_nodes_ann(vector_graph_store_ann):
    """ANN mode: vector index is created immediately (threshold=0) and results are returned."""
    collection = "ann_docs"

    nodes = [
        Node(
            uid=str(uuid4()),
            properties={"title": "Doc 1"},
            embeddings={"content": ([1.0, 0.0, 0.0], SimilarityMetric.EUCLIDEAN)},
        ),
        Node(
            uid=str(uuid4()),
            properties={"title": "Doc 2"},
            embeddings={"content": ([0.0, 1.0, 0.0], SimilarityMetric.EUCLIDEAN)},
        ),
        Node(
            uid=str(uuid4()),
            properties={"title": "Doc 3"},
            embeddings={"content": ([0.9, 0.1, 0.0], SimilarityMetric.EUCLIDEAN)},
        ),
    ]

    await vector_graph_store_ann.add_nodes(collection=collection, nodes=nodes)

    results = await vector_graph_store_ann.search_similar_nodes(
        collection=collection,
        embedding_name="content",
        query_embedding=[1.0, 0.0, 0.0],
        similarity_metric=SimilarityMetric.EUCLIDEAN,
        limit=3,
    )

    # ANN may return approximate results but at minimum should return some nodes
    assert 0 < len(results) <= 3
    # The closest node (Doc 1) should still appear first in a reasonable ANN implementation
    assert results[0].properties["title"] == "Doc 1"


# ---------------------------------------------------------------------------
# Extended sanitize/desanitize coverage
# ---------------------------------------------------------------------------


def test_sanitize_name_extended():
    """Comprehensive sanitize/desanitize round-trip for edge-case inputs."""
    names = [
        "normal_name",
        "123",                  # starts with digits
        ")(*&^%$#@!",          # all special chars
        "my-collection",        # hyphen
        "my.field",             # dot
        "my collection",        # space
        "age!with$pecialchars", # mixed special
        "😀",                   # emoji (multi-byte)
        "",                     # empty string
    ]

    for name in names:
        sanitized = NebulaGraphVectorGraphStore._sanitize_name(name)

        # Sanitized name must be a non-empty valid identifier
        assert len(sanitized) > 0
        assert sanitized[0].isalpha(), f"First char not alpha for input {name!r}: {sanitized!r}"
        assert all(
            c.isalnum() or c == "_" for c in sanitized
        ), f"Invalid chars in sanitized name for input {name!r}: {sanitized!r}"

        # Round-trip must be lossless
        assert NebulaGraphVectorGraphStore._desanitize_name(sanitized) == name, (
            f"Round-trip failed for {name!r}: got {NebulaGraphVectorGraphStore._desanitize_name(sanitized)!r}"
        )

    # All sanitized forms must be distinct (no collision between inputs)
    sanitized_names = [NebulaGraphVectorGraphStore._sanitize_name(n) for n in names]
    assert len(sanitized_names) == len(set(sanitized_names)), "Sanitized names are not unique"
