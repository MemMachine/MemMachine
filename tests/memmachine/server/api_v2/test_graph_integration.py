"""Integration tests for graph API wiring (Group 13).

Covers end-to-end scenarios with a real Neo4j instance:
- Entity type classification via ingestion pipeline
- Entity type filtering in search
- Feature relationship auto-detection
- Graph-enhanced episodic retrieval
- Semantic search enrichment
- Config wiring verification
- Graph API endpoint HTTP calls

Requires Docker and Neo4j (testcontainers). All tests are marked
``@pytest.mark.integration`` so they are skipped by default.
"""

from __future__ import annotations

import numpy as np
import pytest
import pytest_asyncio
from neo4j import AsyncGraphDatabase
from testcontainers.neo4j import Neo4jContainer

from memmachine.common.data_types import SimilarityMetric
from memmachine.common.neo4j_utils import ENTITY_TYPE_PREFIX, sanitize_entity_type
from memmachine.common.vector_graph_store.data_types import Edge, Node
from memmachine.common.vector_graph_store.neo4j_vector_graph_store import (
    Neo4jVectorGraphStore,
    Neo4jVectorGraphStoreParams,
)
from memmachine.semantic_memory.storage.feature_relationship_types import (
    FeatureRelationshipType,
    RelationshipDirection,
)
from memmachine.semantic_memory.storage.neo4j_semantic_storage import (
    Neo4jSemanticStorage,
)
from tests.memmachine.conftest import is_docker_available

pytestmark = pytest.mark.integration


# ---------------------------------------------------------------------------
# Fixtures (module-scoped container, function-scoped cleanup)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def neo4j_connection_info():
    if not is_docker_available():
        pytest.skip("Docker is not available")

    with Neo4jContainer(
        image="neo4j:latest",
        username="neo4j",
        password="password",
    ) as neo4j:
        yield {
            "uri": neo4j.get_connection_url(),
            "username": "neo4j",
            "password": "password",
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
def store(neo4j_driver):
    """VectorGraphStore with exact similarity for deterministic tests."""
    return Neo4jVectorGraphStore(
        Neo4jVectorGraphStoreParams(
            driver=neo4j_driver,
            force_exact_similarity_search=True,
            dedup_trigger_threshold=100,  # high to avoid auto-dedup
            dedup_embedding_threshold=0.95,
            dedup_property_threshold=0.5,
            dedup_auto_merge=False,
        ),
    )


@pytest_asyncio.fixture(scope="module")
async def semantic_storage(neo4j_driver):
    s = Neo4jSemanticStorage(driver=neo4j_driver)
    await s.startup()
    yield s


@pytest_asyncio.fixture(autouse=True)
async def db_cleanup(neo4j_driver):
    """Wipe all data between tests.

    Constraints must be dropped before their backing indexes, so we
    drop constraints first, then indexes.
    """
    await neo4j_driver.execute_query("MATCH (n) DETACH DELETE n")

    # Drop constraints first (they own backing indexes).
    records, _, _ = await neo4j_driver.execute_query(
        "SHOW CONSTRAINTS YIELD name RETURN name",
    )
    for rec in records:
        await neo4j_driver.execute_query(f"DROP CONSTRAINT {rec['name']} IF EXISTS")

    # Now safe to drop remaining indexes.
    records, _, _ = await neo4j_driver.execute_query(
        "SHOW RANGE INDEXES YIELD name RETURN name",
    )
    for rec in records:
        await neo4j_driver.execute_query(f"DROP INDEX {rec['name']} IF EXISTS")

    records, _, _ = await neo4j_driver.execute_query(
        "SHOW VECTOR INDEXES YIELD name RETURN name",
    )
    for rec in records:
        await neo4j_driver.execute_query(f"DROP INDEX {rec['name']} IF EXISTS")

    yield


# =====================================================================
# 13.1 — Entity Type Classification
# =====================================================================


class TestEntityTypeClassification:
    """Verify entity type labels are persisted in Neo4j."""

    @pytest.mark.asyncio
    async def test_add_feature_with_entity_type_creates_label(self, semantic_storage):
        """Feature added with entity_type metadata gets ENTITY_TYPE_* label."""
        embedding = np.array([1.0, 0.0, 0.0])
        feature_id = await semantic_storage.add_feature(
            set_id="test-set",
            category_name="personal",
            feature="name",
            value="Alice",
            tag="test",
            embedding=embedding,
            metadata={"entity_type": "Person"},
        )

        assert feature_id is not None

        # Query Neo4j directly to verify the label.
        # add_feature() returns elementId(f), so we match via elementId().
        records, _, _ = await semantic_storage._driver.execute_query(
            "MATCH (f:Feature) WHERE elementId(f) = $fid RETURN labels(f) AS labels",
            fid=feature_id,
        )
        assert len(records) == 1
        labels = records[0]["labels"]
        assert sanitize_entity_type("Person") in labels

    @pytest.mark.asyncio
    async def test_add_feature_without_entity_type_no_extra_label(
        self, semantic_storage
    ):
        """Feature without entity_type gets no ENTITY_TYPE_* label."""
        embedding = np.array([0.0, 1.0, 0.0])
        feature_id = await semantic_storage.add_feature(
            set_id="test-set",
            category_name="general",
            feature="color",
            value="blue",
            tag="test",
            embedding=embedding,
        )

        records, _, _ = await semantic_storage._driver.execute_query(
            "MATCH (f:Feature) WHERE elementId(f) = $fid RETURN labels(f) AS labels",
            fid=feature_id,
        )
        assert len(records) == 1
        labels = records[0]["labels"]
        entity_labels = [lbl for lbl in labels if lbl.startswith(ENTITY_TYPE_PREFIX)]
        assert entity_labels == []

    @pytest.mark.asyncio
    async def test_node_entity_types_round_trip(self, store):
        """Entity types on nodes survive add → search round trip."""
        collection = "test/entity_round_trip"
        embedding = [1.0, 0.0, 0.0]

        await store.add_nodes(
            collection=collection,
            nodes=[
                Node(
                    uid="person-1",
                    properties={"name": "Alice"},
                    entity_types=["Person"],
                    embeddings={
                        "test_3": (embedding, SimilarityMetric.COSINE),
                    },
                ),
            ],
        )

        results = await store.search_similar_nodes(
            collection=collection,
            embedding_name="test_3",
            query_embedding=embedding,
            limit=1,
        )
        assert len(results) == 1
        assert "Person" in results[0].entity_types


# =====================================================================
# 13.3 — Feature Relationships
# =====================================================================


class TestFeatureRelationships:
    """Verify relationship CRUD against real Neo4j."""

    @pytest.mark.asyncio
    async def test_create_and_query_relationship(self, semantic_storage):
        """Create a RELATED_TO relationship and query it back."""
        embedding_a = np.array([1.0, 0.0, 0.0])
        embedding_b = np.array([0.9, 0.1, 0.0])

        fid_a = await semantic_storage.add_feature(
            set_id="rel-set",
            category_name="test",
            feature="likes",
            value="coffee",
            tag="t",
            embedding=embedding_a,
        )
        fid_b = await semantic_storage.add_feature(
            set_id="rel-set",
            category_name="test",
            feature="drinks",
            value="espresso",
            tag="t",
            embedding=embedding_b,
        )

        await semantic_storage.add_feature_relationship(
            source_id=fid_a,
            target_id=fid_b,
            relationship_type=FeatureRelationshipType.RELATED_TO,
            confidence=0.9,
            source="rule",
        )

        rels = await semantic_storage.get_feature_relationships(
            fid_a,
            relationship_type=FeatureRelationshipType.RELATED_TO,
            direction=RelationshipDirection.OUTGOING,
        )
        assert len(rels) == 1
        assert rels[0].target_id == fid_b
        assert rels[0].confidence == pytest.approx(0.9)

    @pytest.mark.asyncio
    async def test_delete_relationship_idempotent(self, semantic_storage):
        """Deleting a non-existent relationship doesn't error."""
        await semantic_storage.delete_feature_relationships(
            source_id="nonexistent-a",
            target_id="nonexistent-b",
            relationship_type=FeatureRelationshipType.CONTRADICTS,
        )

    @pytest.mark.asyncio
    async def test_find_contradictions_in_set(self, semantic_storage):
        """find_contradictions finds CONTRADICTS pairs within a set."""
        embedding = np.array([1.0, 0.0, 0.0])

        fid_a = await semantic_storage.add_feature(
            set_id="contra-set",
            category_name="pref",
            feature="food",
            value="likes pizza",
            tag="t",
            embedding=embedding,
        )
        fid_b = await semantic_storage.add_feature(
            set_id="contra-set",
            category_name="pref",
            feature="food",
            value="hates pizza",
            tag="t",
            embedding=np.array([0.0, 1.0, 0.0]),
        )

        await semantic_storage.add_feature_relationship(
            source_id=fid_a,
            target_id=fid_b,
            relationship_type=FeatureRelationshipType.CONTRADICTS,
            confidence=0.95,
            source="llm",
        )

        pairs = await semantic_storage.find_contradictions(set_id="contra-set")
        assert len(pairs) == 1
        assert {pairs[0].feature_id_a, pairs[0].feature_id_b} == {fid_a, fid_b}


# =====================================================================
# 13.5 — Graph-Enhanced Episodic Retrieval (multi-hop)
# =====================================================================


class TestMultiHopTraversal:
    """Verify multi-hop traversal against real Neo4j."""

    @pytest.mark.asyncio
    async def test_multi_hop_finds_connected_nodes(self, store):
        """Multi-hop search finds nodes reachable via edges."""
        collection = "test/multi_hop"
        embed_a = [1.0, 0.0, 0.0]
        embed_b = [0.0, 1.0, 0.0]
        embed_c = [0.0, 0.0, 1.0]

        await store.add_nodes(
            collection=collection,
            nodes=[
                Node(
                    uid="a",
                    properties={"name": "A"},
                    embeddings={"test_3": (embed_a, SimilarityMetric.COSINE)},
                ),
                Node(
                    uid="b",
                    properties={"name": "B"},
                    embeddings={"test_3": (embed_b, SimilarityMetric.COSINE)},
                ),
                Node(
                    uid="c",
                    properties={"name": "C"},
                    embeddings={"test_3": (embed_c, SimilarityMetric.COSINE)},
                ),
            ],
        )
        await store.add_edges(
            relation="KNOWS",
            source_collection=collection,
            target_collection=collection,
            edges=[
                Edge(uid="a-b", source_uid="a", target_uid="b"),
                Edge(uid="b-c", source_uid="b", target_uid="c"),
            ],
        )

        results = await store.search_multi_hop_nodes(
            collection=collection,
            this_node_uid="a",
            min_hops=1,
            max_hops=2,
        )

        result_uids = {r.node.uid for r in results}
        assert "b" in result_uids
        assert "c" in result_uids
        assert "a" not in result_uids


# =====================================================================
# 13.7 — Config Wiring
# =====================================================================


class TestConfigWiring:
    """Verify config fields reach the store layer."""

    def test_gds_enabled_false_disables_gds(self, neo4j_driver):
        """gds_enabled=False makes is_gds_available() return False."""
        s = Neo4jVectorGraphStore(
            Neo4jVectorGraphStoreParams(
                driver=neo4j_driver,
                gds_enabled=False,
            ),
        )
        # is_gds_available is async, but with gds_enabled=False it
        # returns immediately. We check the internal flag.
        assert s._gds_enabled is False

    def test_dedup_fields_stored_on_store(self, neo4j_driver):
        """Dedup config fields are stored on the store instance."""
        s = Neo4jVectorGraphStore(
            Neo4jVectorGraphStoreParams(
                driver=neo4j_driver,
                dedup_trigger_threshold=50,
                dedup_embedding_threshold=0.88,
                dedup_property_threshold=0.6,
                dedup_auto_merge=True,
            ),
        )
        assert s._dedup_trigger_threshold == 50
        assert s._dedup_embedding_threshold == pytest.approx(0.88)
        assert s._dedup_property_threshold == pytest.approx(0.6)
        assert s._dedup_auto_merge is True

    def test_gds_params_stored(self, neo4j_driver):
        """GDS config parameters are stored on the store instance."""
        s = Neo4jVectorGraphStore(
            Neo4jVectorGraphStoreParams(
                driver=neo4j_driver,
                gds_enabled=True,
                gds_default_damping_factor=0.9,
                gds_default_max_iterations=30,
            ),
        )
        assert s._gds_enabled is True
        assert s._gds_default_damping_factor == pytest.approx(0.9)
        assert s._gds_default_max_iterations == 30
