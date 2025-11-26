import asyncio
from datetime import datetime, timedelta
from uuid import uuid4

import pytest
import pytest_asyncio
from neo4j import AsyncGraphDatabase
from testcontainers.neo4j import Neo4jContainer

from memmachine.common.embedder import SimilarityMetric
from memmachine.common.vector_graph_store import Edge, Node
from memmachine.common.vector_graph_store.neo4j_vector_graph_store import (
    Neo4jVectorGraphStore,
    Neo4jVectorGraphStoreParams,
)

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def neo4j_connection_info():
    neo4j_username = "neo4j"
    neo4j_password = "password"

    with Neo4jContainer(
        image="neo4j:latest",
        username=neo4j_username,
        password=neo4j_password,
    ) as neo4j:
        yield {
            "uri": neo4j.get_connection_url(),
            "username": neo4j_username,
            "password": neo4j_password,
        }


@pytest_asyncio.fixture(scope="module")
async def neo4j_driver(neo4j_connection_info):
    driver = AsyncGraphDatabase.driver(
        neo4j_connection_info["uri"],
        auth=(
            neo4j_connection_info["username"],
            neo4j_connection_info["password"],
        ),
    )
    yield driver
    await driver.close()


@pytest.fixture(scope="module")
def vector_graph_store(neo4j_driver):
    return Neo4jVectorGraphStore(
        Neo4jVectorGraphStoreParams(
            driver=neo4j_driver,
            force_exact_similarity_search=True,
        )
    )


@pytest.fixture(scope="module")
def vector_graph_store_ann(neo4j_driver):
    return Neo4jVectorGraphStore(
        Neo4jVectorGraphStoreParams(
            driver=neo4j_driver,
            force_exact_similarity_search=False,
        )
    )


@pytest_asyncio.fixture(autouse=True)
async def db_cleanup(neo4j_driver):
    # Delete all nodes and relationships.
    await neo4j_driver.execute_query("MATCH (n) DETACH DELETE n")
    # Drop all vector indexes.
    records, _, _ = await neo4j_driver.execute_query(
        "SHOW VECTOR INDEXES YIELD name RETURN name"
    )
    drop_vector_index_tasks = [
        neo4j_driver.execute_query(f"DROP INDEX {record['name']} IF EXISTS")
        for record in records
    ]
    await asyncio.gather(*drop_vector_index_tasks)
    yield


@pytest.mark.asyncio
async def test_add_nodes(neo4j_driver, vector_graph_store):
    records, _, _ = await neo4j_driver.execute_query("MATCH (n) RETURN n")
    assert len(records) == 0

    nodes = []
    await vector_graph_store.add_nodes(nodes)

    records, _, _ = await neo4j_driver.execute_query("MATCH (n) RETURN n")
    assert len(records) == 0

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

    records, _, _ = await neo4j_driver.execute_query("MATCH (n) RETURN n")
    assert len(records) == 2


@pytest.mark.asyncio
async def test_add_edges(neo4j_driver, vector_graph_store):
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

    records, _, _ = await neo4j_driver.execute_query("MATCH ()-[r]->() RETURN r")
    assert len(records) == 0

    edges = []
    await vector_graph_store.add_edges(edges)

    records, _, _ = await neo4j_driver.execute_query("MATCH ()-[r]->() RETURN r")
    assert len(records) == 0

    edges = [
        Edge(
            uuid=uuid4(),
            source_uuid=node1_uuid,
            target_uuid=node1_uuid,
            relation="IS",
            properties={"description": "Node1 loop"},
        ),
        Edge(
            uuid=uuid4(),
            source_uuid=node1_uuid,
            target_uuid=node2_uuid,
            relation="RELATED_TO",
            properties={"description": "Node1 to Node2"},
        ),
        Edge(
            uuid=uuid4(),
            source_uuid=node2_uuid,
            target_uuid=node1_uuid,
            relation="RELATED_TO",
            properties={"description": "Node2 to Node1"},
        ),
        Edge(
            uuid=uuid4(),
            source_uuid=node2_uuid,
            target_uuid=node2_uuid,
            relation="IS",
            properties={"description": "Node2 loop"},
        ),
    ]

    await vector_graph_store.add_edges(edges)

    records, _, _ = await neo4j_driver.execute_query("MATCH ()-[r]->() RETURN r")
    assert len(records) == 4


@pytest.mark.asyncio
async def test_search_similar_nodes(vector_graph_store, vector_graph_store_ann):
    nodes = [
        Node(
            uuid=uuid4(),
            labels=["Entity"],
            properties={
                "name": "Node1",
                "embedding1": [1000.0, 0.0],
                "embedding2": [1000.0, 0.0],
            },
        ),
        Node(
            uuid=uuid4(),
            labels=["Entity"],
            properties={
                "name": "Node2",
                "embedding1": [10.0, 10.0],
                "embedding2": [10.0, 10.0],
                "include?": "yes",
            },
        ),
        Node(
            uuid=uuid4(),
            labels=["Entity"],
            properties={
                "name": "Node3",
                "embedding1": [-100.0, 0.0],
                "embedding2": [-100.0, 0.0],
                "include?": "no",
            },
        ),
        Node(
            uuid=uuid4(),
            labels=["Entity"],
            properties={
                "name": "Node4",
                "embedding1": [-100.0, -1.0],
                "embedding2": [-100.0, -1.0],
                "include?": "no",
            },
        ),
        Node(
            uuid=uuid4(),
            labels=["Entity"],
            properties={
                "name": "Node5",
                "embedding1": [-100.0, -2.0],
                "embedding2": [-100.0, -2.0],
                "include?": "no",
            },
        ),
        Node(
            uuid=uuid4(),
            labels=["Entity"],
            properties={
                "name": "Node6",
                "embedding1": [-100.0, -3.0],
                "embedding2": [-100.0, -3.0],
                "include?": "no",
            },
        ),
    ]

    await vector_graph_store.add_nodes(nodes)

    results = await vector_graph_store_ann.search_similar_nodes(
        query_embedding=[1.0, 0.0],
        embedding_property_name="embedding1",
        similarity_metric=SimilarityMetric.COSINE,
        limit=5,
        required_labels=["Entity"],
    )
    assert 0 < len(results) <= 5

    results = await vector_graph_store.search_similar_nodes(
        query_embedding=[1.0, 0.0],
        embedding_property_name="embedding1",
        similarity_metric=SimilarityMetric.COSINE,
        limit=5,
        required_labels=["Entity"],
    )
    assert len(results) == 5
    assert results[0].properties["name"] == "Node1"

    results = await vector_graph_store.search_similar_nodes(
        query_embedding=[1.0, 0.0],
        embedding_property_name="embedding1",
        similarity_metric=SimilarityMetric.COSINE,
        limit=5,
        required_labels=["Entity"],
        required_properties={"include?": "yes"},
        include_missing_properties=False,
    )
    assert len(results) == 1
    assert results[0].properties["name"] == "Node2"

    results = await vector_graph_store.search_similar_nodes(
        query_embedding=[1.0, 0.0],
        embedding_property_name="embedding1",
        similarity_metric=SimilarityMetric.COSINE,
        limit=5,
        required_labels=["Entity"],
        required_properties={"include?": "yes"},
        include_missing_properties=True,
    )
    assert len(results) == 2
    assert results[0].properties["name"] == "Node1"

    results = await vector_graph_store.search_similar_nodes(
        query_embedding=[1.0, 0.0],
        embedding_property_name="embedding2",
        similarity_metric=SimilarityMetric.EUCLIDEAN,
        limit=5,
        required_labels=["Entity"],
    )
    assert len(results) == 5
    assert results[0].properties["name"] == "Node2"

    results = await vector_graph_store.search_similar_nodes(
        query_embedding=[1.0, 0.0],
        embedding_property_name="embedding2",
        similarity_metric=SimilarityMetric.EUCLIDEAN,
        limit=5,
        required_labels=["Entity"],
        required_properties={"include?": "yes"},
    )
    assert len(results) == 1
    assert results[0].properties["name"] == "Node2"

    results = await vector_graph_store_ann.search_similar_nodes(
        query_embedding=[1.0, 0.0],
        embedding_property_name="embedding1",
        similarity_metric=SimilarityMetric.COSINE,
        limit=5,
        required_labels=["Entity"],
    )
    assert 0 < len(results) <= 5

    results = await vector_graph_store_ann.search_similar_nodes(
        query_embedding=[1.0, 0.0],
        embedding_property_name="embedding2",
        similarity_metric=SimilarityMetric.EUCLIDEAN,
        limit=5,
        required_labels=["Entity"],
    )
    assert 0 < len(results) <= 5


@pytest.mark.asyncio
async def test_search_related_nodes(vector_graph_store):
    node1_uuid = uuid4()
    node2_uuid = uuid4()
    node3_uuid = uuid4()
    node4_uuid = uuid4()

    nodes = [
        Node(
            uuid=node1_uuid,
            labels=["Entity"],
            properties={"name": "Node1"},
        ),
        Node(
            uuid=node2_uuid,
            labels=["Entity"],
            properties={"name": "Node2", "extra!": "something"},
        ),
        Node(
            uuid=node3_uuid,
            labels=["Entity"],
            properties={"name": "Node3", "marker?": "A"},
        ),
        Node(
            uuid=node4_uuid,
            labels=["Entity"],
            properties={"name": "Node4", "marker?": "B"},
        ),
    ]

    edges = [
        Edge(
            uuid=uuid4(),
            source_uuid=node1_uuid,
            target_uuid=node1_uuid,
            relation="IS",
            properties={"description": "Node1 loop"},
        ),
        Edge(
            uuid=uuid4(),
            source_uuid=node1_uuid,
            target_uuid=node2_uuid,
            relation="RELATED_TO",
            properties={"description": "Node1 to Node2"},
        ),
        Edge(
            uuid=uuid4(),
            source_uuid=node2_uuid,
            target_uuid=node1_uuid,
            relation="RELATED_TO",
            properties={"description": "Node2 to Node1"},
        ),
        Edge(
            uuid=uuid4(),
            source_uuid=node2_uuid,
            target_uuid=node2_uuid,
            relation="IS",
            properties={"description": "Node2 loop"},
        ),
        Edge(
            uuid=uuid4(),
            source_uuid=node3_uuid,
            target_uuid=node2_uuid,
            relation="RELATED_TO",
            properties={"description": "Node3 to Node2"},
        ),
        Edge(
            uuid=uuid4(),
            source_uuid=node3_uuid,
            target_uuid=node4_uuid,
            relation="RELATED_TO",
            properties={"description": "Node3 to Node4"},
        ),
        Edge(
            uuid=uuid4(),
            source_uuid=node3_uuid,
            target_uuid=node3_uuid,
            relation="IS",
            properties={"description": "Node3 loop"},
        ),
    ]

    await vector_graph_store.add_nodes(nodes)
    await vector_graph_store.add_edges(edges)

    results = await vector_graph_store.search_related_nodes(
        node_uuid=node1_uuid,
    )
    assert len(results) == 2
    assert results[0].properties["name"] != results[1].properties["name"]
    assert results[0].properties["name"] in ("Node1", "Node2")
    assert results[1].properties["name"] in ("Node1", "Node2")

    results = await vector_graph_store.search_related_nodes(
        node_uuid=node1_uuid,
        required_properties={"extra!": "something"},
    )
    assert len(results) == 1
    assert results[0].properties["name"] == "Node2"

    results = await vector_graph_store.search_related_nodes(
        node_uuid=node1_uuid,
        allowed_relations=["RELATED_TO"],
    )
    assert len(results) == 1
    assert results[0].properties["name"] == "Node2"

    results = await vector_graph_store.search_related_nodes(
        node_uuid=node2_uuid,
        find_sources=False,
    )
    assert len(results) == 2
    assert results[0].properties["name"] != results[1].properties["name"]
    assert results[0].properties["name"] in ("Node1", "Node2")
    assert results[1].properties["name"] in ("Node1", "Node2")

    results = await vector_graph_store.search_related_nodes(
        node_uuid=node3_uuid,
        find_targets=False,
    )
    assert len(results) == 1
    assert results[0].properties["name"] == "Node3"

    results = await vector_graph_store.search_related_nodes(
        node_uuid=node3_uuid,
        required_properties={"marker?": "A"},
    )
    assert len(results) == 1
    assert results[0].properties["name"] == "Node3"

    results = await vector_graph_store.search_related_nodes(
        node_uuid=node3_uuid,
        required_properties={"marker?": "A"},
        include_missing_properties=True,
    )
    assert len(results) == 2


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
                "include?": "yes",
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
        Node(
            uuid=uuid4(),
            labels=["Event"],
            properties={
                "name": "Event4",
                "timestamp": time + 3 * delta,
                "include?": "yes",
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

    results = await vector_graph_store.search_directional_nodes(
        by_property="timestamp",
        start_at_value=time + delta,
        include_equal_start_at_value=True,
        order_ascending=True,
        limit=2,
        required_properties={"include?": "yes"},
    )
    assert len(results) == 2
    assert results[0].properties["name"] == "Event2"
    assert results[1].properties["name"] == "Event4"

    results = await vector_graph_store.search_directional_nodes(
        by_property="timestamp",
        start_at_value=time + delta,
        include_equal_start_at_value=True,
        order_ascending=False,
        limit=2,
    )
    assert len(results) == 2
    assert results[0].properties["name"] == "Event2"
    assert results[1].properties["name"] == "Event1"

    results = await vector_graph_store.search_directional_nodes(
        by_property="timestamp",
        start_at_value=time + delta,
        include_equal_start_at_value=False,
        order_ascending=True,
        limit=2,
    )
    assert len(results) == 2
    assert results[0].properties["name"] == "Event3"
    assert results[1].properties["name"] == "Event4"

    results = await vector_graph_store.search_directional_nodes(
        by_property="timestamp",
        start_at_value=time + delta,
        include_equal_start_at_value=False,
        order_ascending=False,
        limit=2,
    )
    assert len(results) == 1
    assert results[0].properties["name"] == "Event1"

    results = await vector_graph_store.search_directional_nodes(
        by_property="timestamp",
        start_at_value=None,
        order_ascending=False,
        limit=2,
    )
    assert len(results) == 2
    assert results[0].properties["name"] == "Event4"
    assert results[1].properties["name"] == "Event3"


@pytest.mark.asyncio
async def test_search_matching_nodes(vector_graph_store):
    nodes = [
        Node(
            uuid=uuid4(),
            labels=["Person"],
            properties={
                "name": "Alice",
                "age!with$pecialchars": 30,
                "city": "San Francisco",
                "title": "Engineer",
            },
        ),
        Node(
            uuid=uuid4(),
            labels=["Person"],
            properties={
                "name": "Bob",
                "age!with$pecialchars": 25,
                "city": "Los Angeles",
                "title": "Designer",
            },
        ),
        Node(
            uuid=uuid4(),
            labels=["Person"],
            properties={
                "name": "Charlie",
                "age!with$pecialchars": 35,
                "city": "New York",
            },
        ),
        Node(
            uuid=uuid4(),
            labels=["Person"],
            properties={
                "name": "David",
                "age!with$pecialchars": 30,
                "city": "New York",
            },
        ),
        Node(
            uuid=uuid4(),
            labels=["Robot"],
            properties={"name": "Eve", "city": "Axiom"},
        ),
    ]

    await vector_graph_store.add_nodes(nodes)

    results = await vector_graph_store.search_matching_nodes(
        required_labels=["Person"],
    )
    assert len(results) == 4

    results = await vector_graph_store.search_matching_nodes(
        required_labels=["Robot"],
    )
    assert len(results) == 1

    results = await vector_graph_store.search_matching_nodes(
        required_properties={
            "city": "New York",
        },
    )
    assert len(results) == 2

    results = await vector_graph_store.search_matching_nodes(
        required_properties={
            "city": "San Francisco",
            "age!with$pecialchars": 20,
        },
    )
    assert len(results) == 0

    results = await vector_graph_store.search_matching_nodes(
        required_properties={
            "city": "New York",
            "age!with$pecialchars": 30,
        },
    )
    assert len(results) == 1

    results = await vector_graph_store.search_matching_nodes(
        required_properties={
            "age!with$pecialchars": 30,
        },
    )
    assert len(results) == 2

    results = await vector_graph_store.search_matching_nodes(
        required_properties={
            "age!with$pecialchars": 30,
        },
        include_missing_properties=True,
    )
    assert len(results) == 3

    # Should only include Alice.
    results = await vector_graph_store.search_matching_nodes(
        required_properties={
            "title": "Engineer",
        }
    )
    assert len(results) == 1

    # Should include Alice and all nodes without the "title" property.
    results = await vector_graph_store.search_matching_nodes(
        required_properties={
            "title": "Engineer",
        },
        include_missing_properties=True,
    )
    assert len(results) == 4


@pytest.mark.asyncio
async def test_delete_nodes(neo4j_driver, vector_graph_store):
    nodes = [
        Node(
            uuid=uuid4(),
        ),
        Node(
            uuid=uuid4(),
        ),
        Node(
            uuid=uuid4(),
        ),
        Node(
            uuid=uuid4(),
        ),
        Node(
            uuid=uuid4(),
        ),
        Node(
            uuid=uuid4(),
        ),
    ]

    await vector_graph_store.add_nodes(nodes)
    records, _, _ = await neo4j_driver.execute_query("MATCH (n) RETURN n")
    assert len(records) == 6

    await vector_graph_store.delete_nodes([node.uuid for node in nodes[:-3]])
    records, _, _ = await neo4j_driver.execute_query("MATCH (n) RETURN n")
    assert len(records) == 3
