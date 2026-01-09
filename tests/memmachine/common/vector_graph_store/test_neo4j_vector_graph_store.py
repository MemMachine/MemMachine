import asyncio
from datetime import UTC, datetime, timedelta
from uuid import uuid4
from zoneinfo import ZoneInfo

import pytest
import pytest_asyncio
from neo4j import AsyncGraphDatabase
from testcontainers.neo4j import Neo4jContainer

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
from memmachine.common.vector_graph_store.data_types import Edge, EntityType, Node
from memmachine.common.vector_graph_store.neo4j_vector_graph_store import (
    Neo4jVectorGraphStore,
    Neo4jVectorGraphStoreParams,
)

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def metrics_factory():
    return PrometheusMetricsFactory()


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
def vector_graph_store(neo4j_driver, metrics_factory):
    return Neo4jVectorGraphStore(
        Neo4jVectorGraphStoreParams(
            driver=neo4j_driver,
            force_exact_similarity_search=True,
            metrics_factory=metrics_factory,
        ),
    )


@pytest.fixture(scope="module")
def vector_graph_store_ann(neo4j_driver):
    return Neo4jVectorGraphStore(
        Neo4jVectorGraphStoreParams(
            driver=neo4j_driver,
            force_exact_similarity_search=False,
            filtered_similarity_search_fudge_factor=2,
            exact_similarity_search_fallback_threshold=0.5,
            range_index_creation_threshold=0,
            vector_index_creation_threshold=0,
        ),
    )


@pytest_asyncio.fixture(autouse=True)
async def db_cleanup(neo4j_driver):
    # Delete all nodes and relationships.
    await neo4j_driver.execute_query("MATCH (n) DETACH DELETE n")

    # Drop all range indexes.
    records, _, _ = await neo4j_driver.execute_query(
        "SHOW RANGE INDEXES YIELD name RETURN name",
    )
    drop_range_index_tasks = [
        neo4j_driver.execute_query(f"DROP INDEX {record['name']} IF EXISTS")
        for record in records
    ]

    # Drop all vector indexes.
    records, _, _ = await neo4j_driver.execute_query(
        "SHOW VECTOR INDEXES YIELD name RETURN name",
    )
    drop_vector_index_tasks = [
        neo4j_driver.execute_query(f"DROP INDEX {record['name']} IF EXISTS")
        for record in records
    ]

    await asyncio.gather(*drop_range_index_tasks)
    await asyncio.gather(*drop_vector_index_tasks)
    yield


@pytest.mark.asyncio
async def test_add_nodes(neo4j_driver, vector_graph_store):
    records, _, _ = await neo4j_driver.execute_query("MATCH (n) RETURN n")
    assert len(records) == 0

    nodes = []
    await vector_graph_store.add_nodes(collection="Entity", nodes=nodes)

    records, _, _ = await neo4j_driver.execute_query("MATCH (n) RETURN n")
    assert len(records) == 0

    nodes = [
        Node(
            uid=str(uuid4()),
            properties={"name": "Node1"},
        ),
        Node(
            uid=str(uuid4()),
            properties={"name": "Node2"},
        ),
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

    records, _, _ = await neo4j_driver.execute_query("MATCH (n) RETURN n")
    assert len(records) == len(nodes)


@pytest.mark.asyncio
async def test_add_edges(neo4j_driver, vector_graph_store):
    node1_uid = str(uuid4())
    node2_uid = str(uuid4())
    node3_uid = str(uuid4())

    nodes = [
        Node(
            uid=node1_uid,
            properties={"name": "Node1"},
        ),
        Node(
            uid=node2_uid,
            properties={"name": "Node2"},
        ),
        Node(
            uid=node3_uid,
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

    records, _, _ = await neo4j_driver.execute_query("MATCH ()-[r]->() RETURN r")
    assert len(records) == 0

    edges = []
    await vector_graph_store.add_edges(
        relation="RELATED_TO",
        source_collection="Entity",
        target_collection="Entity",
        edges=edges,
    )

    records, _, _ = await neo4j_driver.execute_query("MATCH ()-[r]->() RETURN r")
    assert len(records) == 0

    related_to_edges = [
        Edge(
            uid=str(uuid4()),
            source_uid=node1_uid,
            target_uid=node2_uid,
            properties={"description": "Node1 to Node2", "time": datetime.now(tz=UTC)},
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
        edges=related_to_edges,
    )
    await vector_graph_store.add_edges(
        relation="IS",
        source_collection="Entity",
        target_collection="Entity",
        edges=is_edges,
    )

    records, _, _ = await neo4j_driver.execute_query("MATCH ()-[r]->() RETURN r")
    assert len(records) == 5


@pytest.mark.asyncio
async def test_search_similar_nodes(vector_graph_store, vector_graph_store_ann):
    nodes = [
        Node(
            uid=str(uuid4()),
            properties={
                "name": "Node1",
            },
            embeddings={
                "embedding1": (
                    [1000.0, 0.0],
                    SimilarityMetric.COSINE,
                ),
                "embedding2": (
                    [1000.0, 0.0],
                    SimilarityMetric.EUCLIDEAN,
                ),
            },
        ),
        Node(
            uid=str(uuid4()),
            properties={
                "name": "Node2",
                "include?": "yes",
            },
            embeddings={
                "embedding1": (
                    [10.0, 10.0],
                    SimilarityMetric.COSINE,
                ),
                "embedding2": (
                    [10.0, 10.0],
                    SimilarityMetric.EUCLIDEAN,
                ),
            },
        ),
        Node(
            uid=str(uuid4()),
            properties={
                "name": "Node3",
                "include?": "no",
            },
            embeddings={
                "embedding1": (
                    [-100.0, 0.0],
                    SimilarityMetric.COSINE,
                ),
                "embedding2": (
                    [-100.0, 0.0],
                    SimilarityMetric.EUCLIDEAN,
                ),
            },
        ),
        Node(
            uid=str(uuid4()),
            properties={
                "name": "Node4",
                "include?": "no",
            },
            embeddings={
                "embedding1": (
                    [-100.0, -1.0],
                    SimilarityMetric.COSINE,
                ),
                "embedding2": (
                    [-100.0, -1.0],
                    SimilarityMetric.EUCLIDEAN,
                ),
            },
        ),
        Node(
            uid=str(uuid4()),
            properties={
                "name": "Node5",
                "include?": "no",
            },
            embeddings={
                "embedding1": (
                    [-100.0, -2.0],
                    SimilarityMetric.COSINE,
                ),
                "embedding2": (
                    [-100.0, -2.0],
                    SimilarityMetric.EUCLIDEAN,
                ),
            },
        ),
        Node(
            uid=str(uuid4()),
            properties={
                "name": "Node6",
                "include?": "no",
            },
            embeddings={
                "embedding1": (
                    [-100.0, -3.0],
                    SimilarityMetric.COSINE,
                ),
                "embedding2": (
                    [-100.0, -3.0],
                    SimilarityMetric.EUCLIDEAN,
                ),
            },
        ),
    ]

    await vector_graph_store.add_nodes(collection="Entity", nodes=nodes)

    results = await vector_graph_store_ann.search_similar_nodes(
        collection="Entity",
        query_embedding=[1.0, 0.0],
        embedding_name="embedding1",
        similarity_metric=SimilarityMetric.COSINE,
        limit=5,
    )
    assert 0 < len(results) <= 5

    results = await vector_graph_store.search_similar_nodes(
        collection="Entity",
        query_embedding=[1.0, 0.0],
        embedding_name="embedding1",
        similarity_metric=SimilarityMetric.COSINE,
        limit=5,
    )
    assert len(results) == 5
    assert results[0].properties["name"] == "Node1"

    results = await vector_graph_store.search_similar_nodes(
        collection="Entity",
        query_embedding=[1.0, 0.0],
        embedding_name="embedding1",
        similarity_metric=SimilarityMetric.COSINE,
        limit=5,
        property_filter=FilterComparison(
            field="include?",
            op="=",
            value="yes",
        ),
    )
    assert len(results) == 1
    assert results[0].properties["name"] == "Node2"

    results = await vector_graph_store.search_similar_nodes(
        collection="Entity",
        query_embedding=[1.0, 0.0],
        embedding_name="embedding1",
        similarity_metric=SimilarityMetric.COSINE,
        limit=5,
        property_filter=FilterOr(
            left=FilterComparison(
                field="include?",
                op="=",
                value="yes",
            ),
            right=FilterComparison(
                field="include?",
                op="is_null",
                value=None,
            ),
        ),
    )
    assert len(results) == 2
    assert results[0].properties["name"] == "Node1"

    results = await vector_graph_store.search_similar_nodes(
        collection="Entity",
        query_embedding=[1.0, 0.0],
        embedding_name="embedding2",
        similarity_metric=SimilarityMetric.EUCLIDEAN,
        limit=5,
    )
    assert len(results) == 5
    assert results[0].properties["name"] == "Node2"

    results = await vector_graph_store.search_similar_nodes(
        collection="Entity",
        query_embedding=[1.0, 0.0],
        embedding_name="embedding2",
        similarity_metric=SimilarityMetric.EUCLIDEAN,
        limit=5,
        property_filter=FilterComparison(
            field="include?",
            op="=",
            value="yes",
        ),
    )
    assert len(results) == 1
    assert results[0].properties["name"] == "Node2"

    results = await vector_graph_store_ann.search_similar_nodes(
        collection="Entity",
        query_embedding=[1.0, 0.0],
        embedding_name="embedding1",
        similarity_metric=SimilarityMetric.COSINE,
        limit=5,
    )
    assert 0 < len(results) <= 5

    results = await vector_graph_store_ann.search_similar_nodes(
        collection="Entity",
        query_embedding=[1.0, 0.0],
        embedding_name="embedding2",
        similarity_metric=SimilarityMetric.EUCLIDEAN,
        limit=5,
    )
    assert 0 < len(results) <= 5


@pytest.mark.asyncio
async def test_search_related_nodes(vector_graph_store):
    node1_uid = str(uuid4())
    node2_uid = str(uuid4())
    node3_uid = str(uuid4())
    node4_uid = str(uuid4())

    nodes = [
        Node(
            uid=node1_uid,
            properties={"name": "Node1"},
        ),
        Node(
            uid=node2_uid,
            properties={"name": "Node2", "extra!": "something"},
        ),
        Node(
            uid=node3_uid,
            properties={"name": "Node3", "marker?": "A"},
        ),
        Node(
            uid=node4_uid,
            properties={"name": "Node4", "marker?": "B"},
        ),
    ]

    related_to_edges = [
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
            properties={
                "description": "Node3 to Node2",
                "extra": 1,
            },
        ),
        Edge(
            uid=str(uuid4()),
            source_uid=node3_uid,
            target_uid=node4_uid,
            properties={
                "description": "Node3 to Node4",
                "extra": 2,
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
        Edge(
            uid=str(uuid4()),
            source_uid=node3_uid,
            target_uid=node3_uid,
            properties={"description": "Node3 loop"},
        ),
    ]

    await vector_graph_store.add_nodes(collection="Entity", nodes=nodes)
    await vector_graph_store.add_edges(
        relation="RELATED_TO",
        source_collection="Entity",
        target_collection="Entity",
        edges=related_to_edges,
    )
    await vector_graph_store.add_edges(
        relation="RELATED_TO",
        source_collection="Entity",
        target_collection="Entity",
        edges=is_edges,
    )

    results = await vector_graph_store.search_related_nodes(
        relation="RELATED_TO",
        other_collection="Entity",
        this_collection="Entity",
        this_node_uid=node1_uid,
    )
    assert len(results) == 2
    assert results[0].properties["name"] != results[1].properties["name"]
    assert results[0].properties["name"] in ("Node1", "Node2")
    assert results[1].properties["name"] in ("Node1", "Node2")

    results = await vector_graph_store.search_related_nodes(
        relation="RELATED_TO",
        other_collection="Entity",
        this_collection="Entity",
        this_node_uid=node1_uid,
        node_property_filter=FilterComparison(
            field="extra!",
            op="=",
            value="something",
        ),
    )
    assert len(results) == 1
    assert results[0].properties["name"] == "Node2"

    results = await vector_graph_store.search_related_nodes(
        relation="RELATED_TO",
        other_collection="Entity",
        this_collection="Entity",
        this_node_uid=node2_uid,
        find_sources=False,
    )
    assert len(results) == 2
    assert results[0].properties["name"] != results[1].properties["name"]
    assert results[0].properties["name"] in ("Node1", "Node2")
    assert results[1].properties["name"] in ("Node1", "Node2")

    results = await vector_graph_store.search_related_nodes(
        relation="RELATED_TO",
        other_collection="Entity",
        this_collection="Entity",
        this_node_uid=node3_uid,
        find_targets=False,
    )
    assert len(results) == 1
    assert results[0].properties["name"] == "Node3"

    results = await vector_graph_store.search_related_nodes(
        relation="RELATED_TO",
        other_collection="Entity",
        this_collection="Entity",
        this_node_uid=node3_uid,
        node_property_filter=FilterComparison(
            field="marker?",
            op="=",
            value="A",
        ),
    )
    assert len(results) == 1
    assert results[0].properties["name"] == "Node3"

    results = await vector_graph_store.search_related_nodes(
        relation="RELATED_TO",
        other_collection="Entity",
        this_collection="Entity",
        this_node_uid=node3_uid,
        node_property_filter=FilterOr(
            left=FilterComparison(
                field="marker?",
                op="=",
                value="A",
            ),
            right=FilterComparison(
                field="marker?",
                op="is_null",
                value=None,
            ),
        ),
    )
    assert len(results) == 2

    results = await vector_graph_store.search_related_nodes(
        relation="RELATED_TO",
        other_collection="Entity",
        this_collection="Entity",
        this_node_uid=node3_uid,
        edge_property_filter=FilterComparison(
            field="extra",
            op="=",
            value=1,
        ),
    )
    assert len(results) == 1

    results = await vector_graph_store.search_related_nodes(
        relation="RELATED_TO",
        other_collection="Entity",
        this_collection="Entity",
        this_node_uid=node3_uid,
        edge_property_filter=FilterOr(
            left=FilterComparison(
                field="extra",
                op="=",
                value=1,
            ),
            right=FilterComparison(
                field="extra",
                op="is_null",
                value=None,
            ),
        ),
    )
    assert len(results) == 2


@pytest.mark.asyncio
async def test_search_directional_nodes(vector_graph_store):
    time = datetime.now(tz=UTC)
    delta = timedelta(days=1)

    nodes = [
        Node(
            uid=str(uuid4()),
            properties={
                "name": "Event1",
                "timestamp": time,
            },
        ),
        Node(
            uid=str(uuid4()),
            properties={
                "name": "Event2",
                "timestamp": time + delta,
                "include?": "yes",
            },
        ),
        Node(
            uid=str(uuid4()),
            properties={
                "name": "Event3",
                "timestamp": time + 2 * delta,
            },
        ),
        Node(
            uid=str(uuid4()),
            properties={
                "name": "Event4",
                "timestamp": time + 3 * delta,
                "include?": "yes",
            },
        ),
    ]

    await vector_graph_store.add_nodes(collection="Event", nodes=nodes)

    results = await vector_graph_store.search_directional_nodes(
        collection="Event",
        by_properties=["timestamp"],
        starting_at=[time + delta],
        order_ascending=[True],
        include_equal_start=True,
        limit=2,
    )
    assert len(results) == 2
    assert results[0].properties["name"] == "Event2"
    assert results[1].properties["name"] == "Event3"

    result_timestamp = (
        results[0].properties["timestamp"].astimezone(ZoneInfo("America/Los_Angeles"))
    )

    results = await vector_graph_store.search_directional_nodes(
        collection="Event",
        by_properties=["timestamp"],
        starting_at=[result_timestamp],
        order_ascending=[False],
        include_equal_start=False,
        limit=1,
    )
    assert len(results) == 1
    assert results[0].properties["name"] != "Event2"

    results = await vector_graph_store.search_directional_nodes(
        collection="Event",
        by_properties=["timestamp"],
        starting_at=[result_timestamp],
        order_ascending=[True],
        include_equal_start=False,
        limit=1,
    )
    assert len(results) == 1
    assert results[0].properties["name"] != "Event2"

    results = await vector_graph_store.search_directional_nodes(
        collection="Event",
        by_properties=["timestamp"],
        starting_at=[time + delta],
        order_ascending=[True],
        include_equal_start=True,
        limit=2,
        property_filter=FilterComparison(
            field="include?",
            op="=",
            value="yes",
        ),
    )
    assert len(results) == 2
    assert results[0].properties["name"] == "Event2"
    assert results[1].properties["name"] == "Event4"

    results = await vector_graph_store.search_directional_nodes(
        collection="Event",
        by_properties=["timestamp"],
        starting_at=[time + delta],
        order_ascending=[False],
        include_equal_start=True,
        limit=2,
    )
    assert len(results) == 2
    assert results[0].properties["name"] == "Event2"
    assert results[1].properties["name"] == "Event1"

    results = await vector_graph_store.search_directional_nodes(
        collection="Event",
        by_properties=["timestamp"],
        starting_at=[time + delta],
        order_ascending=[True],
        include_equal_start=False,
        limit=2,
    )
    assert len(results) == 2
    assert results[0].properties["name"] == "Event3"
    assert results[1].properties["name"] == "Event4"

    results = await vector_graph_store.search_directional_nodes(
        collection="Event",
        by_properties=["timestamp"],
        starting_at=[time + delta],
        order_ascending=[False],
        include_equal_start=False,
        limit=2,
    )
    assert len(results) == 1
    assert results[0].properties["name"] == "Event1"

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
async def test_search_directional_nodes_multiple_by_properties(
    neo4j_driver,
    vector_graph_store,
):
    time = datetime.now(tz=UTC)
    delta = timedelta(days=1)

    nodes = [
        Node(
            uid=str(uuid4()),
            properties={
                "name": "Event1",
                "timestamp": time,
                "pair": 1,
                "sequence": 1,
            },
        ),
        Node(
            uid=str(uuid4()),
            properties={
                "name": "Event2",
                "timestamp": time,
                "pair": 1,
                "sequence": 2,
            },
        ),
        Node(
            uid=str(uuid4()),
            properties={
                "name": "Event3",
                "timestamp": time,
                "pair": 2,
                "sequence": 1,
            },
        ),
        Node(
            uid=str(uuid4()),
            properties={
                "name": "Event4",
                "timestamp": time,
                "pair": 2,
                "sequence": 2,
            },
        ),
        Node(
            uid=str(uuid4()),
            properties={
                "name": "Event5",
                "timestamp": time,
                "pair": 3,
                "sequence": 1,
            },
        ),
        Node(
            uid=str(uuid4()),
            properties={
                "name": "Event6",
                "timestamp": time,
                "pair": 3,
                "sequence": 2,
            },
        ),
        Node(
            uid=str(uuid4()),
            properties={
                "name": "Event7",
                "timestamp": time + delta,
                "pair": 1,
                "sequence": 1,
            },
        ),
        Node(
            uid=str(uuid4()),
            properties={
                "name": "Event8",
                "timestamp": time + delta,
                "pair": 1,
                "sequence": 2,
            },
        ),
        Node(
            uid=str(uuid4()),
            properties={
                "name": "Event9",
                "timestamp": time + delta,
                "pair": 2,
                "sequence": 1,
            },
        ),
        Node(
            uid=str(uuid4()),
            properties={
                "name": "Event10",
                "timestamp": time + delta,
                "pair": 2,
                "sequence": 2,
            },
        ),
        Node(
            uid=str(uuid4()),
            properties={
                "name": "Event11",
                "timestamp": time + delta,
                "pair": 3,
                "sequence": 1,
            },
        ),
        Node(
            uid=str(uuid4()),
            properties={
                "name": "Event12",
                "timestamp": time + delta,
                "pair": 3,
                "sequence": 2,
            },
        ),
    ]

    await vector_graph_store.add_nodes(collection="Event", nodes=nodes)
    records, _, _ = await neo4j_driver.execute_query("MATCH (n) RETURN n")
    assert len(records) == 12

    results = await vector_graph_store.search_directional_nodes(
        collection="Event",
        by_properties=["timestamp", "pair", "sequence"],
        starting_at=[time + delta, 1, 2],
        order_ascending=[True, True, True],
        include_equal_start=True,
        limit=None,
    )
    assert len(results) == 5
    assert results[0].properties["name"] == "Event8"
    assert results[1].properties["name"] == "Event9"
    assert results[2].properties["name"] == "Event10"
    assert results[3].properties["name"] == "Event11"
    assert results[4].properties["name"] == "Event12"

    results = await vector_graph_store.search_directional_nodes(
        collection="Event",
        by_properties=["timestamp", "pair", "sequence"],
        starting_at=[time + delta, 1, 2],
        order_ascending=[True, True, False],
        include_equal_start=True,
        limit=None,
    )
    assert len(results) == 6
    assert results[0].properties["name"] == "Event8"
    assert results[1].properties["name"] == "Event7"
    assert results[2].properties["name"] == "Event10"
    assert results[3].properties["name"] == "Event9"
    assert results[4].properties["name"] == "Event12"
    assert results[5].properties["name"] == "Event11"

    results = await vector_graph_store.search_directional_nodes(
        collection="Event",
        by_properties=["timestamp", "pair", "sequence"],
        starting_at=[time + delta, 1, 2],
        order_ascending=[True, False, True],
        include_equal_start=True,
        limit=None,
    )
    assert len(results) == 1
    assert results[0].properties["name"] == "Event8"

    results = await vector_graph_store.search_directional_nodes(
        collection="Event",
        by_properties=["timestamp", "pair", "sequence"],
        starting_at=[time + delta, 1, 2],
        order_ascending=[True, False, False],
        include_equal_start=True,
        limit=None,
    )
    assert len(results) == 2
    assert results[0].properties["name"] == "Event8"
    assert results[1].properties["name"] == "Event7"

    results = await vector_graph_store.search_directional_nodes(
        collection="Event",
        by_properties=["timestamp", "pair", "sequence"],
        starting_at=[time + delta, 1, 2],
        order_ascending=[False, True, True],
        include_equal_start=True,
        limit=None,
    )
    assert len(results) == 11
    assert results[0].properties["name"] == "Event8"
    assert results[1].properties["name"] == "Event9"
    assert results[2].properties["name"] == "Event10"
    assert results[3].properties["name"] == "Event11"
    assert results[4].properties["name"] == "Event12"
    assert results[5].properties["name"] == "Event1"
    assert results[6].properties["name"] == "Event2"
    assert results[7].properties["name"] == "Event3"
    assert results[8].properties["name"] == "Event4"
    assert results[9].properties["name"] == "Event5"
    assert results[10].properties["name"] == "Event6"

    results = await vector_graph_store.search_directional_nodes(
        collection="Event",
        by_properties=["timestamp", "pair", "sequence"],
        starting_at=[time + delta, 1, 2],
        order_ascending=[False, True, False],
        include_equal_start=True,
        limit=None,
    )
    assert len(results) == 12
    assert results[0].properties["name"] == "Event8"
    assert results[1].properties["name"] == "Event7"
    assert results[2].properties["name"] == "Event10"
    assert results[3].properties["name"] == "Event9"
    assert results[4].properties["name"] == "Event12"
    assert results[5].properties["name"] == "Event11"
    assert results[6].properties["name"] == "Event2"
    assert results[7].properties["name"] == "Event1"
    assert results[8].properties["name"] == "Event4"
    assert results[9].properties["name"] == "Event3"
    assert results[10].properties["name"] == "Event6"
    assert results[11].properties["name"] == "Event5"

    results = await vector_graph_store.search_directional_nodes(
        collection="Event",
        by_properties=["timestamp", "pair", "sequence"],
        starting_at=[time + delta, 1, 2],
        order_ascending=[False, False, True],
        include_equal_start=True,
        limit=None,
    )
    assert len(results) == 7
    assert results[0].properties["name"] == "Event8"
    assert results[1].properties["name"] == "Event5"
    assert results[2].properties["name"] == "Event6"
    assert results[3].properties["name"] == "Event3"
    assert results[4].properties["name"] == "Event4"
    assert results[5].properties["name"] == "Event1"
    assert results[6].properties["name"] == "Event2"

    results = await vector_graph_store.search_directional_nodes(
        collection="Event",
        by_properties=["timestamp", "pair", "sequence"],
        starting_at=[time + delta, 1, 2],
        order_ascending=[False, False, False],
        include_equal_start=True,
        limit=None,
    )
    assert len(results) == 8
    assert results[0].properties["name"] == "Event8"
    assert results[1].properties["name"] == "Event7"
    assert results[2].properties["name"] == "Event6"
    assert results[3].properties["name"] == "Event5"
    assert results[4].properties["name"] == "Event4"
    assert results[5].properties["name"] == "Event3"
    assert results[6].properties["name"] == "Event2"
    assert results[7].properties["name"] == "Event1"


@pytest.mark.asyncio
async def test_search_matching_nodes(vector_graph_store):
    person_nodes = [
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
            properties={
                "name": "Charlie",
                "city": "New York",
            },
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

    robot_nodes = [
        Node(
            uid=str(uuid4()),
            properties={"name": "Eve", "city": "Axiom"},
        ),
    ]

    await vector_graph_store.add_nodes(collection="Person", nodes=person_nodes)
    await vector_graph_store.add_nodes(collection="Robot", nodes=robot_nodes)

    results = await vector_graph_store.search_matching_nodes(
        collection="Person",
    )
    assert len(results) == 4

    results = await vector_graph_store.search_matching_nodes(
        collection="Robot",
    )
    assert len(results) == 1

    results = await vector_graph_store.search_matching_nodes(
        collection="Robot",
        property_filter=FilterComparison(
            field="none_value",
            op="is_null",
            value=None,
        ),
    )
    assert len(results) == 1

    results = await vector_graph_store.search_matching_nodes(
        collection="Robot",
        property_filter=FilterComparison(
            field="none_value",
            op="=",
            value="something",
        ),
    )
    assert len(results) == 0

    results = await vector_graph_store.search_matching_nodes(
        collection="Person",
        property_filter=FilterComparison(
            field="city",
            op="=",
            value="New York",
        ),
    )
    assert len(results) == 2

    results = await vector_graph_store.search_matching_nodes(
        collection="Person",
        property_filter=FilterAnd(
            left=FilterComparison(
                field="city",
                op="=",
                value="San Francisco",
            ),
            right=FilterComparison(
                field="age!with$pecialchars",
                op="=",
                value=20,
            ),
        ),
    )
    assert len(results) == 0

    results = await vector_graph_store.search_matching_nodes(
        collection="Person",
        property_filter=FilterAnd(
            left=FilterComparison(
                field="city",
                op="=",
                value="New York",
            ),
            right=FilterComparison(
                field="age!with$pecialchars",
                op="=",
                value=30,
            ),
        ),
    )
    assert len(results) == 1

    results = await vector_graph_store.search_matching_nodes(
        collection="Person",
        property_filter=FilterComparison(
            field="age!with$pecialchars",
            op="=",
            value=30,
        ),
    )
    assert len(results) == 2

    results = await vector_graph_store.search_matching_nodes(
        collection="Person",
        property_filter=FilterOr(
            left=FilterComparison(
                field="age!with$pecialchars",
                op="=",
                value=30,
            ),
            right=FilterComparison(
                field="age!with$pecialchars",
                op="is_null",
                value=None,
            ),
        ),
    )
    assert len(results) == 3

    # Should only include Alice.
    results = await vector_graph_store.search_matching_nodes(
        collection="Person",
        property_filter=FilterComparison(
            field="title",
            op="=",
            value="Engineer",
        ),
    )
    assert len(results) == 1

    # Should include Alice and all Person nodes without the "title" property.
    results = await vector_graph_store.search_matching_nodes(
        collection="Person",
        property_filter=FilterOr(
            left=FilterComparison(
                field="title",
                op="=",
                value="Engineer",
            ),
            right=FilterComparison(
                field="title",
                op="is_null",
                value=None,
            ),
        ),
    )
    assert len(results) == 3


@pytest.mark.asyncio
async def test_get_nodes(vector_graph_store):
    nodes = [
        Node(
            uid=str(uuid4()),
            properties={"name": "Node1", "time": datetime.now(tz=UTC)},
        ),
        Node(
            uid=str(uuid4()),
            properties={"name": "Node2"},
        ),
        Node(
            uid=str(uuid4()),
            properties={"name": "Node3"},
        ),
    ]

    await vector_graph_store.add_nodes(collection="Entity", nodes=nodes)

    fetched_nodes = await vector_graph_store.get_nodes(
        collection="Entity",
        node_uids=[node.uid for node in nodes],
    )
    assert len(fetched_nodes) == 3

    for fetched_node in fetched_nodes:
        assert fetched_node.uid in {node.uid for node in nodes}

    fetched_nodes = await vector_graph_store.get_nodes(
        collection="Entity",
        node_uids=[nodes[0].uid, uuid4()],
    )
    assert len(fetched_nodes) == 1
    assert fetched_nodes[0] == nodes[0]


@pytest.mark.asyncio
async def test_delete_nodes(neo4j_driver, vector_graph_store):
    nodes = [
        Node(
            uid=str(uuid4()),
        ),
        Node(
            uid=str(uuid4()),
        ),
        Node(
            uid=str(uuid4()),
        ),
        Node(
            uid=str(uuid4()),
        ),
        Node(
            uid=str(uuid4()),
        ),
        Node(
            uid=str(uuid4()),
        ),
    ]

    await vector_graph_store.add_nodes(collection="Entity", nodes=nodes)
    records, _, _ = await neo4j_driver.execute_query("MATCH (n) RETURN n")
    assert len(records) == 6

    await vector_graph_store.delete_nodes(
        collection="Bad", node_uids=[node.uid for node in nodes[:-3]]
    )
    records, _, _ = await neo4j_driver.execute_query("MATCH (n) RETURN n")
    assert len(records) == 6

    await vector_graph_store.delete_nodes(
        collection="Entity", node_uids=[node.uid for node in nodes[:-3]]
    )
    records, _, _ = await neo4j_driver.execute_query("MATCH (n) RETURN n")
    assert len(records) == 3


@pytest.mark.asyncio
async def test_delete_all_data(neo4j_driver, vector_graph_store):
    nodes = [
        Node(
            uid=str(uuid4()),
        ),
        Node(
            uid=str(uuid4()),
        ),
        Node(
            uid=str(uuid4()),
        ),
        Node(
            uid=str(uuid4()),
        ),
        Node(
            uid=str(uuid4()),
        ),
        Node(
            uid=str(uuid4()),
        ),
    ]

    await vector_graph_store.add_nodes(collection="Entity", nodes=nodes)
    records, _, _ = await neo4j_driver.execute_query("MATCH (n) RETURN n")
    assert len(records) == 6

    await vector_graph_store.delete_all_data()
    records, _, _ = await neo4j_driver.execute_query("MATCH (n) RETURN n")
    assert len(records) == 0


@pytest.fixture
def vector_graph_store_indexing(neo4j_driver):
    return Neo4jVectorGraphStore(
        Neo4jVectorGraphStoreParams(
            driver=neo4j_driver,
            force_exact_similarity_search=False,
            filtered_similarity_search_fudge_factor=2,
            exact_similarity_search_fallback_threshold=0.5,
            range_index_creation_threshold=10,
            vector_index_creation_threshold=10,
        ),
    )


@pytest.mark.asyncio
async def test__create_range_index_if_not_exists(
    neo4j_driver,
    vector_graph_store_indexing,
):
    records, _, _ = await neo4j_driver.execute_query(
        "SHOW RANGE INDEXES YIELD name RETURN name",
    )
    existing_indexes = {record["name"] for record in records}

    collection_or_relation_name = "SomeCollectionOrRelation"
    other_collection_or_relation_name = "OtherCollectionOrRelation"
    property_name = "some_property"
    other_property_name = "other_property"

    create_range_index_task_lists = [
        [
            vector_graph_store_indexing._create_range_index_if_not_exists(
                EntityType.NODE,
                Neo4jVectorGraphStore._sanitize_name(collection_or_relation_name),
                Neo4jVectorGraphStore._sanitize_name(property_name),
            ),
            vector_graph_store_indexing._create_range_index_if_not_exists(
                EntityType.NODE,
                Neo4jVectorGraphStore._sanitize_name(collection_or_relation_name),
                Neo4jVectorGraphStore._sanitize_name(other_property_name),
            ),
            vector_graph_store_indexing._create_range_index_if_not_exists(
                EntityType.NODE,
                Neo4jVectorGraphStore._sanitize_name(other_collection_or_relation_name),
                Neo4jVectorGraphStore._sanitize_name(property_name),
            ),
            vector_graph_store_indexing._create_range_index_if_not_exists(
                EntityType.NODE,
                Neo4jVectorGraphStore._sanitize_name(other_collection_or_relation_name),
                Neo4jVectorGraphStore._sanitize_name(other_property_name),
            ),
            vector_graph_store_indexing._create_range_index_if_not_exists(
                EntityType.EDGE,
                Neo4jVectorGraphStore._sanitize_name(collection_or_relation_name),
                Neo4jVectorGraphStore._sanitize_name(property_name),
            ),
            vector_graph_store_indexing._create_range_index_if_not_exists(
                EntityType.EDGE,
                Neo4jVectorGraphStore._sanitize_name(collection_or_relation_name),
                Neo4jVectorGraphStore._sanitize_name(other_property_name),
            ),
            vector_graph_store_indexing._create_range_index_if_not_exists(
                EntityType.EDGE,
                Neo4jVectorGraphStore._sanitize_name(other_collection_or_relation_name),
                Neo4jVectorGraphStore._sanitize_name(property_name),
            ),
            vector_graph_store_indexing._create_range_index_if_not_exists(
                EntityType.EDGE,
                Neo4jVectorGraphStore._sanitize_name(other_collection_or_relation_name),
                Neo4jVectorGraphStore._sanitize_name(other_property_name),
            ),
            vector_graph_store_indexing._create_range_index_if_not_exists(
                EntityType.NODE,
                Neo4jVectorGraphStore._sanitize_name(collection_or_relation_name),
                [
                    Neo4jVectorGraphStore._sanitize_name(property_name),
                    Neo4jVectorGraphStore._sanitize_name(other_property_name),
                ],
            ),
            vector_graph_store_indexing._create_range_index_if_not_exists(
                EntityType.NODE,
                Neo4jVectorGraphStore._sanitize_name(collection_or_relation_name),
                [
                    Neo4jVectorGraphStore._sanitize_name(other_property_name),
                    Neo4jVectorGraphStore._sanitize_name(property_name),
                ],
            ),
            vector_graph_store_indexing._create_range_index_if_not_exists(
                EntityType.EDGE,
                Neo4jVectorGraphStore._sanitize_name(collection_or_relation_name),
                [
                    Neo4jVectorGraphStore._sanitize_name(property_name),
                    Neo4jVectorGraphStore._sanitize_name(other_property_name),
                ],
            ),
            vector_graph_store_indexing._create_range_index_if_not_exists(
                EntityType.EDGE,
                Neo4jVectorGraphStore._sanitize_name(collection_or_relation_name),
                [
                    Neo4jVectorGraphStore._sanitize_name(other_property_name),
                    Neo4jVectorGraphStore._sanitize_name(property_name),
                ],
            ),
        ]
        for _ in range(10000)
    ]

    create_range_index_tasks = [
        task for task_list in create_range_index_task_lists for task in task_list
    ]

    await asyncio.gather(*create_range_index_tasks)

    records, _, _ = await neo4j_driver.execute_query(
        "SHOW RANGE INDEXES YIELD name RETURN name",
    )

    updated_indexes = {record["name"] for record in records}

    assert len(updated_indexes) == len(existing_indexes) + 12


@pytest.mark.asyncio
async def test__create_vector_index_if_not_exists(
    neo4j_driver,
    vector_graph_store_indexing,
):
    records, _, _ = await neo4j_driver.execute_query(
        "SHOW VECTOR INDEXES YIELD name RETURN name",
    )
    existing_indexes = {record["name"] for record in records}

    collection_or_relation_name = "SomeCollectionOrRelation"
    other_collection_or_relation_name = "OtherCollectionOrRelation"
    property_name = "some_property"
    other_property_name = "other_property"

    create_vector_index_task_lists = [
        [
            vector_graph_store_indexing._create_vector_index_if_not_exists(
                EntityType.NODE,
                Neo4jVectorGraphStore._sanitize_name(collection_or_relation_name),
                Neo4jVectorGraphStore._sanitize_name(property_name),
                dimensions=dimensions,
                similarity_metric=similarity_metric,
            ),
            vector_graph_store_indexing._create_vector_index_if_not_exists(
                EntityType.NODE,
                Neo4jVectorGraphStore._sanitize_name(collection_or_relation_name),
                Neo4jVectorGraphStore._sanitize_name(other_property_name),
                dimensions=dimensions,
                similarity_metric=similarity_metric,
            ),
            vector_graph_store_indexing._create_vector_index_if_not_exists(
                EntityType.NODE,
                Neo4jVectorGraphStore._sanitize_name(other_collection_or_relation_name),
                Neo4jVectorGraphStore._sanitize_name(property_name),
                dimensions=dimensions,
                similarity_metric=similarity_metric,
            ),
            vector_graph_store_indexing._create_vector_index_if_not_exists(
                EntityType.NODE,
                Neo4jVectorGraphStore._sanitize_name(other_collection_or_relation_name),
                Neo4jVectorGraphStore._sanitize_name(other_property_name),
                dimensions=dimensions,
                similarity_metric=similarity_metric,
            ),
            vector_graph_store_indexing._create_vector_index_if_not_exists(
                EntityType.EDGE,
                Neo4jVectorGraphStore._sanitize_name(collection_or_relation_name),
                Neo4jVectorGraphStore._sanitize_name(property_name),
                dimensions=dimensions,
                similarity_metric=similarity_metric,
            ),
            vector_graph_store_indexing._create_vector_index_if_not_exists(
                EntityType.EDGE,
                Neo4jVectorGraphStore._sanitize_name(collection_or_relation_name),
                Neo4jVectorGraphStore._sanitize_name(other_property_name),
                dimensions=dimensions,
                similarity_metric=similarity_metric,
            ),
            vector_graph_store_indexing._create_vector_index_if_not_exists(
                EntityType.EDGE,
                Neo4jVectorGraphStore._sanitize_name(other_collection_or_relation_name),
                Neo4jVectorGraphStore._sanitize_name(property_name),
                dimensions=dimensions,
                similarity_metric=similarity_metric,
            ),
            vector_graph_store_indexing._create_vector_index_if_not_exists(
                EntityType.EDGE,
                Neo4jVectorGraphStore._sanitize_name(other_collection_or_relation_name),
                Neo4jVectorGraphStore._sanitize_name(other_property_name),
                dimensions=dimensions,
                similarity_metric=similarity_metric,
            ),
        ]
        for dimensions in [1, 10]
        for similarity_metric in [SimilarityMetric.COSINE, SimilarityMetric.EUCLIDEAN]
        for _ in range(2000)
    ]

    create_vector_index_tasks = [
        task for task_list in create_vector_index_task_lists for task in task_list
    ]

    await asyncio.gather(*create_vector_index_tasks)

    records, _, _ = await neo4j_driver.execute_query(
        "SHOW VECTOR INDEXES YIELD name RETURN name",
    )

    updated_indexes = {record["name"] for record in records}

    assert len(updated_indexes) == len(existing_indexes) + 8


def test__sanitize_desanitize_name():
    names = [
        "normal_name",
        "123",
        ")(*&^%$#@!",
        "😀",
        "𰻝",
        " \t\n",
        "",
    ]

    sanitized_names = [Neo4jVectorGraphStore._sanitize_name(name) for name in names]

    for original, sanitized in zip(names, sanitized_names, strict=False):
        assert len(sanitized) > 0
        assert sanitized[0].isalpha()
        assert all(c.isalnum() or c == "_" for c in sanitized)

        assert original == Neo4jVectorGraphStore._desanitize_name(sanitized)


def test__index_name():
    entity_types = [
        EntityType.NODE,
        EntityType.EDGE,
    ]
    collection_or_relation_names = [
        "normal_name",
        "123",
        ")(*&^%$#@!",
        "😀",
        "𰻝",
        " \t\n",
        "",
    ]
    property_names_list = [
        "normal_name",
        "123",
        ")(*&^%$#@!",
        "😀",
        "𰻝",
        " \t\n",
        "",
        ["normal_name", "123", ")(*&^%$#@!"],
        [
            "😀",
            "𰻝",
            " \t\n",
            "",
        ],
    ]

    index_names = set()
    for entity_type in entity_types:
        for collection_or_relation_name in collection_or_relation_names:
            sanitized_collection_or_relation = Neo4jVectorGraphStore._sanitize_name(
                collection_or_relation_name,
            )
            for property_names in property_names_list:
                if isinstance(property_names, str):
                    sanitized_property_names = Neo4jVectorGraphStore._sanitize_name(
                        property_names,
                    )
                else:
                    sanitized_property_names = [
                        Neo4jVectorGraphStore._sanitize_name(property_name)
                        for property_name in property_names
                    ]

                index_name = Neo4jVectorGraphStore._index_name(
                    entity_type,
                    sanitized_collection_or_relation,
                    sanitized_property_names,
                )

                assert len(index_name) > 0
                assert index_name not in index_names
                index_names.add(index_name)


@pytest.mark.asyncio
async def test__nodes_from_neo4j_nodes(neo4j_driver, vector_graph_store):
    records, _, _ = await neo4j_driver.execute_query("MATCH (n) RETURN n")
    assert len(records) == 0

    nodes = [
        Node(
            uid=str(uuid4()),
            properties={"name": "Node1"},
        ),
        Node(
            uid=str(uuid4()),
            properties={"name": "Node2"},
        ),
        Node(
            uid=str(uuid4()),
            properties={"name": "Node3", "time": datetime.now(tz=UTC)},
            embeddings={
                "embedding_name": (
                    [0.1, 0.2, 0.3],
                    SimilarityMetric.COSINE,
                ),
            },
        ),
    ]

    await vector_graph_store.add_nodes(collection="Entity", nodes=nodes)

    records, _, _ = await neo4j_driver.execute_query("MATCH (n) RETURN n")
    assert len(records) == 3

    neo4j_nodes = [record["n"] for record in records]
    fetched_nodes = vector_graph_store._nodes_from_neo4j_nodes(neo4j_nodes)
    assert len(fetched_nodes) == 3

    assert all(fetched_node in nodes for fetched_node in fetched_nodes)
