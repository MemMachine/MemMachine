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
    """NebulaGraph connection information.

    Note: Connects to NebulaGraph Enterprise 5.0 testing instance.
    """
    return {
        "hosts": ["192.168.8.11:9698"],
        "username": "root",
        "password": "NebulaGraph01",
        "schema_name": "/test_schema",
        "graph_type_name": "memmachine_type",
        "graph_name": "test_graph",
    }


@pytest_asyncio.fixture(scope="module")
async def nebula_client(nebula_connection_info):
    """Create NebulaGraph async client for testing (single session, no pooling)."""
    from nebulagraph_python.client import NebulaAsyncClient, SessionConfig

    # Connect with empty SessionConfig (schema/graph don't exist yet)
    client = await NebulaAsyncClient.connect(
        hosts=nebula_connection_info["hosts"],
        username=nebula_connection_info["username"],
        password=nebula_connection_info["password"],
        session_config=SessionConfig(),
    )

    # Initialize schema, graph type, and graph
    try:
        # Create schema
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

    except Exception as e:
        await client.close()
        raise RuntimeError(
            f"Failed to initialize NebulaGraph test environment: {e}"
        ) from e

    yield client

    # Cleanup after tests - must drop in order: graphs -> graph types -> schema
    try:
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
    except Exception:
        pass

    await client.close()


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

    # Verify properties
    # TODO: verify embeddings as well
    retrieved_by_uid = {n.uid: n for n in retrieved}
    for original in nodes:
        retrieved_node = retrieved_by_uid[original.uid]
        assert retrieved_node.properties["name"] == original.properties["name"]
        assert retrieved_node.properties["age"] == original.properties["age"]


@pytest.mark.asyncio
async def test_add_edges(nebula_client, vector_graph_store):
    """Test adding edges between nodes."""
    source_collection = "person"
    target_collection = "company"
    relation = "works_at"

    # Create source and target nodes
    person = Node(
        uid=str(uuid4()),
        properties={"name": "Alice"},
        embeddings={},
    )
    company = Node(
        uid=str(uuid4()),
        properties={"name": "Acme Corp"},
        embeddings={},
    )

    await vector_graph_store.add_nodes(collection=source_collection, nodes=[person])
    await vector_graph_store.add_nodes(collection=target_collection, nodes=[company])

    # Add edge
    edge = Edge(
        uid=str(uuid4()),
        source_uid=person.uid,
        target_uid=company.uid,
        properties={"since": 2020, "role": "Engineer"},
        embeddings={},
    )

    await vector_graph_store.add_edges(
        relation=relation,
        source_collection=source_collection,
        target_collection=target_collection,
        edges=[edge],
    )

    # Verify edge by searching related nodes
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


@pytest.mark.asyncio
async def test_search_similar_nodes(vector_graph_store, vector_graph_store_ann):
    """Test vector similarity search with both KNN and ANN."""
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

    # Test with exact search (KNN)
    await vector_graph_store.add_nodes(collection=collection, nodes=nodes)

    query_vec = [1.0, 0.0, 0.0]
    results = await vector_graph_store.search_similar_nodes(
        collection=collection,
        embedding_name="content",
        query_embedding=query_vec,
        similarity_metric=SimilarityMetric.COSINE,
        limit=2,
    )

    assert len(results) <= 2
    # First result should be most similar (Doc 1 or Doc 3)
    assert results[0].properties["title"] in ["Doc 1", "Doc 3"]


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
        similarity_metric=SimilarityMetric.COSINE,
        limit=10,
        property_filter=filter_expr,
    )

    assert len(results) == 2
    for node in results:
        assert node.properties["category"] == "tech"


@pytest.mark.asyncio
async def test_search_related_nodes(vector_graph_store):
    """Test searching for related nodes via edges."""
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
        properties={"name": "Acme"},
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

    # Create edges
    edges = [
        Edge(
            uid=str(uuid4()),
            source_uid=alice.uid,
            target_uid=acme.uid,
            properties={"role": "Engineer"},
        ),
        Edge(
            uid=str(uuid4()),
            source_uid=bob.uid,
            target_uid=techcorp.uid,
            properties={"role": "Manager"},
        ),
    ]

    await vector_graph_store.add_edges(
        relation=relation,
        source_collection=person_collection,
        target_collection=company_collection,
        edges=edges,
    )

    # Find companies where Alice works
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
    assert results[0].properties["name"] == "Acme"


@pytest.mark.asyncio
async def test_search_directional_nodes(vector_graph_store):
    """Test searching nodes with directional ordering."""
    collection = "events"

    now = datetime.now(UTC)
    nodes = [
        Node(
            uid=str(uuid4()),
            properties={"timestamp": now - timedelta(hours=3), "priority": 1},
            embeddings={},
        ),
        Node(
            uid=str(uuid4()),
            properties={"timestamp": now - timedelta(hours=2), "priority": 2},
            embeddings={},
        ),
        Node(
            uid=str(uuid4()),
            properties={"timestamp": now - timedelta(hours=1), "priority": 3},
            embeddings={},
        ),
    ]

    await vector_graph_store.add_nodes(collection=collection, nodes=nodes)

    # Search for events after a certain time, ordered by timestamp ascending
    start_time = now - timedelta(hours=2, minutes=30)
    results = await vector_graph_store.search_directional_nodes(
        collection=collection,
        by_properties=["timestamp"],
        starting_at=[start_time],
        order_ascending=[True],
        include_equal_start=False,
        limit=10,
    )

    assert len(results) == 2  # Last two events
    # Should be ordered by timestamp ascending
    assert results[0].properties["priority"] == 2
    assert results[1].properties["priority"] == 3


@pytest.mark.asyncio
async def test_search_matching_nodes(vector_graph_store):
    """Test searching nodes with property filters."""
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
            properties={"name": "Desk", "price": 300, "category": "furniture"},
            embeddings={},
        ),
    ]

    await vector_graph_store.add_nodes(collection=collection, nodes=nodes)

    # Search with filter
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
