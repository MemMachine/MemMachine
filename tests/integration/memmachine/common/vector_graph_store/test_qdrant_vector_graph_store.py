"""
Integration tests for QdrantVectorGraphStore.

These tests require a running Qdrant instance.
To run these tests, start Qdrant locally:
    docker run -p 6333:6333 qdrant/qdrant

"""

import asyncio
from datetime import datetime, timedelta
from uuid import uuid4

import pytest
import pytest_asyncio

from memmachine.common.embedder import SimilarityMetric
from memmachine.common.vector_graph_store import Edge, Node
from memmachine.common.vector_graph_store.qdrant_vector_graph_store import (
    QdrantVectorGraphStore,
    QdrantVectorGraphStoreConfig,
)

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def qdrant_config():
    return {
        "url": "http://localhost:6333",
        "api_key": None,  # No API key for local testing
        "timeout": 60.0,
        "vector_dimension": 3,
    }


@pytest_asyncio.fixture(scope="module")
async def vector_graph_store(qdrant_config):
    config = QdrantVectorGraphStoreConfig(**qdrant_config)
    store = QdrantVectorGraphStore(config)
    yield store
    await store.close()


@pytest_asyncio.fixture(autouse=True)
async def cleanup(vector_graph_store):
    await vector_graph_store.clear_data()
    yield
    await vector_graph_store.clear_data()


@pytest.mark.asyncio
async def test_add_nodes(vector_graph_store):
    nodes = [
        Node(
            uuid=uuid4(),
            labels=["Entity"],
            properties={"name": "Node1"},
        ),
        Node(
            uuid=uuid4(),
            labels=["Entity"],
            properties={"name": "Node2"},
        ),
    ]

    await vector_graph_store.add_nodes(nodes)

    results = await vector_graph_store.search_matching_nodes(required_labels=["Entity"])
    assert len(results) == 2


@pytest.mark.asyncio
async def test_add_edges(vector_graph_store):
    node1_uuid = uuid4()
    node2_uuid = uuid4()

    nodes = [
        Node(
            uuid=node1_uuid,
            labels=["Entity"],
            properties={"name": "Node1"},
        ),
        Node(
            uuid=node2_uuid,
            labels=["Entity"],
            properties={"name": "Node2"},
        ),
    ]

    await vector_graph_store.add_nodes(nodes)

    edges = [
        Edge(
            uuid=uuid4(),
            source_uuid=node1_uuid,
            target_uuid=node2_uuid,
            relation="RELATED_TO",
            properties={"description": "Node1 to Node2"},
        ),
    ]

    await vector_graph_store.add_edges(edges)

    results = await vector_graph_store.search_related_nodes(
        node_uuid=node1_uuid,
        find_targets=True,
        find_sources=False,
    )
    assert len(results) == 1
    assert results[0].uuid == node2_uuid


@pytest.mark.asyncio
async def test_search_similar_nodes(vector_graph_store):
    nodes = [
        Node(
            uuid=uuid4(),
            labels=["Entity"],
            properties={
                "name": "Node1",
                "embedding": [1.0, 0.0, 0.0],
            },
        ),
        Node(
            uuid=uuid4(),
            labels=["Entity"],
            properties={
                "name": "Node2",
                "embedding": [0.0, 1.0, 0.0],
            },
        ),
        Node(
            uuid=uuid4(),
            labels=["Entity"],
            properties={
                "name": "Node3",
                "embedding": [0.0, 0.0, 1.0],
            },
        ),
    ]

    await vector_graph_store.add_nodes(nodes)

    results = await vector_graph_store.search_similar_nodes(
        query_embedding=[1.0, 0.0, 0.0],
        embedding_property_name="embedding",
        similarity_metric=SimilarityMetric.COSINE,
        limit=2,
        required_labels=["Entity"],
    )

    assert len(results) >= 1
    assert results[0].properties["name"] == "Node1"


@pytest.mark.asyncio
async def test_search_matching_nodes(vector_graph_store):
    nodes = [
        Node(
            uuid=uuid4(),
            labels=["Person"],
            properties={"name": "Alice", "age": 30},
        ),
        Node(
            uuid=uuid4(),
            labels=["Person"],
            properties={"name": "Bob", "age": 25},
        ),
        Node(
            uuid=uuid4(),
            labels=["Robot"],
            properties={"name": "Eve", "age": 0},
        ),
    ]

    await vector_graph_store.add_nodes(nodes)

    results = await vector_graph_store.search_matching_nodes(required_labels=["Person"])
    assert len(results) == 2

    results = await vector_graph_store.search_matching_nodes(
        required_properties={"age": 30}
    )
    assert len(results) == 1
    assert results[0].properties["name"] == "Alice"

    results = await vector_graph_store.search_matching_nodes(
        required_labels=["Person"], required_properties={"age": 25}
    )
    assert len(results) == 1
    assert results[0].properties["name"] == "Bob"


@pytest.mark.asyncio
async def test_search_directional_nodes(vector_graph_store):
    time = datetime.now()
    delta = timedelta(days=1)

    nodes = [
        Node(
            uuid=uuid4(),
            labels=["Event"],
            properties={
                "name": "Event1",
                "timestamp": time,
            },
        ),
        Node(
            uuid=uuid4(),
            labels=["Event"],
            properties={
                "name": "Event2",
                "timestamp": time + delta,
            },
        ),
        Node(
            uuid=uuid4(),
            labels=["Event"],
            properties={
                "name": "Event3",
                "timestamp": time + 2 * delta,
            },
        ),
    ]

    await vector_graph_store.add_nodes(nodes)

    results = await vector_graph_store.search_directional_nodes(
        by_property="timestamp",
        start_at_value=time + delta,
        include_equal_start_at_value=True,
        order_ascending=True,
        limit=2,
    )

    assert len(results) == 2
    assert results[0].properties["name"] == "Event2"
    assert results[1].properties["name"] == "Event3"


@pytest.mark.asyncio
async def test_delete_nodes(vector_graph_store):
    node1_uuid = uuid4()
    node2_uuid = uuid4()

    nodes = [
        Node(
            uuid=node1_uuid,
            labels=["Entity"],
            properties={"name": "Node1"},
        ),
        Node(
            uuid=node2_uuid,
            labels=["Entity"],
            properties={"name": "Node2"},
        ),
    ]

    await vector_graph_store.add_nodes(nodes)

    results = await vector_graph_store.search_matching_nodes()
    assert len(results) == 2

    await vector_graph_store.delete_nodes([node1_uuid])

    results = await vector_graph_store.search_matching_nodes()
    assert len(results) == 1
    assert results[0].uuid == node2_uuid


@pytest.mark.asyncio
async def test_clear_data(vector_graph_store):
    nodes = [
        Node(
            uuid=uuid4(),
            labels=["Entity"],
            properties={"name": "Node1"},
        ),
        Node(
            uuid=uuid4(),
            labels=["Entity"],
            properties={"name": "Node2"},
        ),
    ]

    await vector_graph_store.add_nodes(nodes)

    results = await vector_graph_store.search_matching_nodes()
    assert len(results) == 2

    await vector_graph_store.clear_data()

    results = await vector_graph_store.search_matching_nodes()
    assert len(results) == 0
