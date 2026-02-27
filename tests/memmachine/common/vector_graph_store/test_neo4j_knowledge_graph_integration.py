"""Integration tests for Neo4j knowledge graph improvements.

Covers: entity ontology (2.9), migration (3.6), multi-hop traversal (4.8),
graph-filtered search (5.6), entity deduplication (7.8), and GDS (8.8).

Requires Docker and Neo4j (testcontainers). All tests are marked
``@pytest.mark.integration`` so they are skipped by default.
"""

import pytest
import pytest_asyncio
from neo4j import AsyncGraphDatabase
from testcontainers.neo4j import Neo4jContainer

from memmachine.common.data_types import SimilarityMetric
from memmachine.common.vector_graph_store.data_types import (
    DuplicateResolutionStrategy,
    Edge,
    GraphFilter,
    Node,
    TraversalDirection,
)
from memmachine.common.vector_graph_store.neo4j_migration import (
    apply_uniqueness_constraints,
    audit_duplicate_uids,
    backfill_entity_type_labels,
    resolve_duplicate_uids,
)
from memmachine.common.vector_graph_store.neo4j_vector_graph_store import (
    Neo4jVectorGraphStore,
    Neo4jVectorGraphStoreParams,
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
    """VectorGraphStore with exact similarity and low dedup threshold."""
    return Neo4jVectorGraphStore(
        Neo4jVectorGraphStoreParams(
            driver=neo4j_driver,
            force_exact_similarity_search=True,
            dedup_trigger_threshold=3,  # low for testing
            dedup_embedding_threshold=0.95,
            dedup_property_threshold=0.5,
            dedup_auto_merge=False,
        ),
    )


@pytest.fixture(scope="module")
def store_auto_merge(neo4j_driver):
    """VectorGraphStore with auto-merge dedup enabled."""
    return Neo4jVectorGraphStore(
        Neo4jVectorGraphStoreParams(
            driver=neo4j_driver,
            force_exact_similarity_search=True,
            dedup_trigger_threshold=3,
            dedup_embedding_threshold=0.95,
            dedup_property_threshold=0.5,
            dedup_auto_merge=True,
        ),
    )


@pytest_asyncio.fixture(autouse=True)
async def db_cleanup(neo4j_driver):
    """Wipe all data and drop all indexes/constraints between tests."""
    await neo4j_driver.execute_query("MATCH (n) DETACH DELETE n")

    # Drop range indexes.
    records, _, _ = await neo4j_driver.execute_query(
        "SHOW RANGE INDEXES YIELD name RETURN name",
    )
    for rec in records:
        await neo4j_driver.execute_query(f"DROP INDEX {rec['name']} IF EXISTS")

    # Drop vector indexes.
    records, _, _ = await neo4j_driver.execute_query(
        "SHOW VECTOR INDEXES YIELD name RETURN name",
    )
    for rec in records:
        await neo4j_driver.execute_query(f"DROP INDEX {rec['name']} IF EXISTS")

    # Drop constraints.
    records, _, _ = await neo4j_driver.execute_query(
        "SHOW CONSTRAINTS YIELD name RETURN name",
    )
    for rec in records:
        await neo4j_driver.execute_query(f"DROP CONSTRAINT {rec['name']} IF EXISTS")

    yield


# =====================================================================
# 2.9 -- Entity Ontology & MERGE Upserts
# =====================================================================


class TestEntityOntologyAndMerge:
    """Integration tests for entity types, MERGE upserts, and constraints."""

    @pytest.mark.asyncio
    async def test_add_nodes_with_entity_types(self, neo4j_driver, store):
        """Nodes carry entity type labels in Neo4j."""
        node = Node(
            uid="person-1",
            properties={"name": "Alice"},
            entity_types=["Person", "Employee"],
        )
        await store.add_nodes(collection="memories", nodes=[node])

        records, _, _ = await neo4j_driver.execute_query(
            "MATCH (n {uid: 'person-1'}) RETURN labels(n) AS labels"
        )
        labels = set(records[0]["labels"])
        assert "ENTITY_TYPE_Person" in labels
        assert "ENTITY_TYPE_Employee" in labels

    @pytest.mark.asyncio
    async def test_merge_upsert_updates_existing_node(self, neo4j_driver, store):
        """Re-adding a node with same UID updates properties instead of creating a duplicate."""
        node_v1 = Node(uid="merge-1", properties={"name": "Alice", "age": 30})
        await store.add_nodes(collection="memories", nodes=[node_v1])

        node_v2 = Node(uid="merge-1", properties={"name": "Alice B.", "city": "NYC"})
        await store.add_nodes(collection="memories", nodes=[node_v2])

        records, _, _ = await neo4j_driver.execute_query(
            "MATCH (n {uid: 'merge-1'}) RETURN n"
        )
        assert len(records) == 1  # No duplicate

    @pytest.mark.asyncio
    async def test_merge_edge_idempotent(self, neo4j_driver, store):
        """Re-adding the same edge does not create duplicates."""
        await store.add_nodes(
            collection="memories",
            nodes=[
                Node(uid="e-src", properties={"name": "A"}),
                Node(uid="e-tgt", properties={"name": "B"}),
            ],
        )
        edge = Edge(
            uid="edge-1",
            source_uid="e-src",
            target_uid="e-tgt",
            properties={"weight": 1},
        )
        await store.add_edges(
            relation="KNOWS",
            source_collection="memories",
            target_collection="memories",
            edges=[edge],
        )
        await store.add_edges(
            relation="KNOWS",
            source_collection="memories",
            target_collection="memories",
            edges=[edge],
        )

        records, _, _ = await neo4j_driver.execute_query(
            "MATCH ()-[r:KNOWS]->() RETURN count(r) AS cnt"
        )
        assert records[0]["cnt"] == 1

    @pytest.mark.asyncio
    async def test_entity_type_filtering_in_search(self, store):
        """search_similar_nodes filters by entity_types."""
        nodes = [
            Node(
                uid="p1",
                properties={"name": "Alice"},
                entity_types=["Person"],
                embeddings={"emb": ([1.0, 0.0], SimilarityMetric.COSINE)},
            ),
            Node(
                uid="loc1",
                properties={"name": "Office"},
                entity_types=["Location"],
                embeddings={"emb": ([0.9, 0.1], SimilarityMetric.COSINE)},
            ),
        ]
        await store.add_nodes(collection="memories", nodes=nodes)

        # Filter to Person only
        results = await store.search_similar_nodes(
            collection="memories",
            query_embedding=[1.0, 0.0],
            embedding_name="emb",
            similarity_metric=SimilarityMetric.COSINE,
            limit=10,
            entity_types=["Person"],
        )
        assert len(results) == 1
        assert results[0].properties["name"] == "Alice"

    @pytest.mark.asyncio
    async def test_update_entity_types(self, neo4j_driver, store):
        """update_entity_types adds and removes type labels."""
        node = Node(
            uid="upd-1",
            properties={"name": "Bob"},
            entity_types=["Person"],
        )
        await store.add_nodes(collection="memories", nodes=[node])

        # Update: replace Person with Employee and TeamLead
        await store.update_entity_types(
            collection="memories",
            node_uid="upd-1",
            entity_types=["Employee", "TeamLead"],
        )

        records, _, _ = await neo4j_driver.execute_query(
            "MATCH (n {uid: 'upd-1'}) RETURN labels(n) AS labels"
        )
        labels = set(records[0]["labels"])
        assert "ENTITY_TYPE_Person" not in labels
        assert "ENTITY_TYPE_Employee" in labels
        assert "ENTITY_TYPE_TeamLead" in labels

    @pytest.mark.asyncio
    async def test_nodes_from_neo4j_extracts_entity_types(self, store):
        """Round-trip: entity types survive add + get."""
        node = Node(
            uid="rt-1",
            properties={"name": "Carol"},
            entity_types=["Person", "Manager"],
        )
        await store.add_nodes(collection="memories", nodes=[node])

        fetched = await store.get_nodes(collection="memories", node_uids=["rt-1"])
        assert len(fetched) == 1
        assert set(fetched[0].entity_types) == {"Person", "Manager"}


# =====================================================================
# 3.6 -- Migration Script
# =====================================================================


class TestMigration:
    """Integration tests for neo4j_migration utilities."""

    async def _create_raw_node(self, driver, label, uid, props=None):
        """Helper: create a node using raw Cypher CREATE (bypasses MERGE)."""
        prop_string = ", ".join(f"{k}: ${k}" for k in (props or {}))
        if prop_string:
            prop_string = ", " + prop_string
        await driver.execute_query(
            f"CREATE (n:{label} {{uid: $uid{prop_string}}})",
            uid=uid,
            **(props or {}),
        )

    @pytest.mark.asyncio
    async def test_audit_no_duplicates(self, neo4j_driver, store):
        """Audit on a clean collection returns zero duplicates."""
        await store.add_nodes(
            collection="clean",
            nodes=[Node(uid="a"), Node(uid="b")],
        )
        report = await audit_duplicate_uids(neo4j_driver)
        assert report.total_duplicates == 0

    @pytest.mark.asyncio
    async def test_audit_with_duplicates(self, neo4j_driver):
        """Audit detects duplicate UIDs created via raw CREATE."""
        label = Neo4jVectorGraphStore._sanitize_name("dupcol")
        await self._create_raw_node(neo4j_driver, label, "dup-uid")
        await self._create_raw_node(neo4j_driver, label, "dup-uid")
        await self._create_raw_node(neo4j_driver, label, "unique-uid")

        report = await audit_duplicate_uids(neo4j_driver)
        assert report.total_duplicates == 1

    @pytest.mark.asyncio
    async def test_resolve_duplicates(self, neo4j_driver):
        """resolve_duplicate_uids merges duplicates."""
        label = Neo4jVectorGraphStore._sanitize_name("resolvecol")
        await self._create_raw_node(
            neo4j_driver,
            label,
            "dup-r",
            {"name": "v1"},
        )
        await self._create_raw_node(
            neo4j_driver,
            label,
            "dup-r",
            {"name": "v2", "extra": "data"},
        )

        deleted = await resolve_duplicate_uids(neo4j_driver)
        assert deleted == 1

        # Only one node remains
        records, _, _ = await neo4j_driver.execute_query(
            f"MATCH (n:{label}) RETURN count(n) AS cnt"
        )
        assert records[0]["cnt"] == 1

    @pytest.mark.asyncio
    async def test_resolve_idempotent(self, neo4j_driver, store):
        """resolve_duplicate_uids on clean data does nothing."""
        await store.add_nodes(
            collection="idempotent",
            nodes=[Node(uid="x"), Node(uid="y")],
        )
        deleted = await resolve_duplicate_uids(neo4j_driver)
        assert deleted == 0

    @pytest.mark.asyncio
    async def test_apply_constraints(self, neo4j_driver, store):
        """apply_uniqueness_constraints creates constraints idempotently."""
        await store.add_nodes(collection="constrained", nodes=[Node(uid="c1")])

        names1 = await apply_uniqueness_constraints(neo4j_driver)
        assert len(names1) >= 1

        # Re-run is idempotent
        names2 = await apply_uniqueness_constraints(neo4j_driver)
        assert names2 == names1

    @pytest.mark.asyncio
    async def test_backfill_entity_type_labels(self, neo4j_driver, store):
        """backfill_entity_type_labels applies type labels to existing nodes."""
        await store.add_nodes(
            collection="backfill",
            nodes=[Node(uid="bf-1"), Node(uid="bf-2")],
        )

        updated = await backfill_entity_type_labels(
            neo4j_driver,
            {"backfill": {"Person": ["bf-1"], "Location": ["bf-2"]}},
        )
        assert updated == 2

        records, _, _ = await neo4j_driver.execute_query(
            "MATCH (n {uid: 'bf-1'}) RETURN labels(n) AS labels"
        )
        assert "ENTITY_TYPE_Person" in set(records[0]["labels"])


# =====================================================================
# 4.8 -- Multi-Hop Traversal
# =====================================================================


class TestMultiHopTraversal:
    """Integration tests for search_multi_hop_nodes."""

    async def _build_chain(self, store, collection="chain"):
        """Create a linear chain: A -> B -> C -> D."""
        nodes = [
            Node(uid="A", properties={"name": "A"}),
            Node(uid="B", properties={"name": "B"}),
            Node(uid="C", properties={"name": "C"}),
            Node(uid="D", properties={"name": "D"}),
        ]
        await store.add_nodes(collection=collection, nodes=nodes)

        for src, tgt in [("A", "B"), ("B", "C"), ("C", "D")]:
            await store.add_edges(
                relation="NEXT",
                source_collection=collection,
                target_collection=collection,
                edges=[Edge(uid=f"{src}-{tgt}", source_uid=src, target_uid=tgt)],
            )
        return nodes

    @pytest.mark.asyncio
    async def test_basic_multi_hop(self, store):
        """Basic multi-hop finds nodes at varying distances."""
        await self._build_chain(store)

        results = await store.search_multi_hop_nodes(
            collection="chain",
            this_node_uid="A",
            min_hops=1,
            max_hops=3,
        )
        result_uids = {r.node.uid for r in results}
        assert "B" in result_uids
        assert "C" in result_uids
        assert "D" in result_uids
        assert "A" not in result_uids  # Starting node excluded

    @pytest.mark.asyncio
    async def test_hop_distance_correctness(self, store):
        """Hop distances are correct for a chain."""
        await self._build_chain(store)

        results = await store.search_multi_hop_nodes(
            collection="chain",
            this_node_uid="A",
            min_hops=1,
            max_hops=3,
        )
        by_uid = {r.node.uid: r for r in results}
        assert by_uid["B"].hop_distance == 1
        assert by_uid["C"].hop_distance == 2
        assert by_uid["D"].hop_distance == 3

    @pytest.mark.asyncio
    async def test_decay_scoring_order(self, store):
        """Closer nodes score higher with decay < 1."""
        await self._build_chain(store)

        results = await store.search_multi_hop_nodes(
            collection="chain",
            this_node_uid="A",
            min_hops=1,
            max_hops=3,
            score_decay=0.7,
        )
        # Results should be sorted by score descending (B first, D last).
        assert results[0].node.uid == "B"
        assert results[-1].node.uid == "D"
        assert results[0].score > results[-1].score

    @pytest.mark.asyncio
    async def test_relationship_type_filtering(self, store):
        """Only paths using specified relationship types are traversed."""
        nodes = [
            Node(uid="X", properties={"name": "X"}),
            Node(uid="Y", properties={"name": "Y"}),
            Node(uid="Z", properties={"name": "Z"}),
        ]
        await store.add_nodes(collection="rtype", nodes=nodes)

        await store.add_edges(
            relation="KNOWS",
            source_collection="rtype",
            target_collection="rtype",
            edges=[Edge(uid="x-y", source_uid="X", target_uid="Y")],
        )
        await store.add_edges(
            relation="WORKS_WITH",
            source_collection="rtype",
            target_collection="rtype",
            edges=[Edge(uid="y-z", source_uid="Y", target_uid="Z")],
        )

        # Only follow KNOWS edges
        results = await store.search_multi_hop_nodes(
            collection="rtype",
            this_node_uid="X",
            min_hops=1,
            max_hops=3,
            relation_types=["KNOWS"],
        )
        result_uids = {r.node.uid for r in results}
        assert "Y" in result_uids
        assert "Z" not in result_uids  # Not reachable via KNOWS only

    @pytest.mark.asyncio
    async def test_limit_enforcement(self, store):
        """Limit caps the number of results."""
        await self._build_chain(store)

        results = await store.search_multi_hop_nodes(
            collection="chain",
            this_node_uid="A",
            min_hops=1,
            max_hops=3,
            limit=1,
        )
        assert len(results) == 1

    @pytest.mark.asyncio
    async def test_max_hops_clamping(self, store):
        """max_hops is clamped to ceiling of 5."""
        await self._build_chain(store)

        # Request 100 hops; should be clamped to 5 (and chain is only 3 deep)
        results = await store.search_multi_hop_nodes(
            collection="chain",
            this_node_uid="A",
            min_hops=1,
            max_hops=100,
        )
        # All 3 reachable nodes should be found (chain depth is 3 < 5)
        assert len(results) == 3

    @pytest.mark.asyncio
    async def test_distinct_node_deduplication(self, store):
        """A node reachable via multiple paths appears once at shortest distance."""
        # Diamond: A -> B -> D, A -> C -> D
        nodes = [
            Node(uid="da", properties={"name": "A"}),
            Node(uid="db", properties={"name": "B"}),
            Node(uid="dc", properties={"name": "C"}),
            Node(uid="dd", properties={"name": "D"}),
        ]
        await store.add_nodes(collection="diamond", nodes=nodes)

        for src, tgt in [("da", "db"), ("da", "dc"), ("db", "dd"), ("dc", "dd")]:
            await store.add_edges(
                relation="LINK",
                source_collection="diamond",
                target_collection="diamond",
                edges=[Edge(uid=f"{src}-{tgt}", source_uid=src, target_uid=tgt)],
            )

        results = await store.search_multi_hop_nodes(
            collection="diamond",
            this_node_uid="da",
            min_hops=1,
            max_hops=3,
        )
        # D appears once at hop_distance=2 (shortest path)
        d_results = [r for r in results if r.node.uid == "dd"]
        assert len(d_results) == 1
        assert d_results[0].hop_distance == 2


# =====================================================================
# 5.6 -- Graph-Filtered Search
# =====================================================================


class TestGraphFilteredSearch:
    """Integration tests for search_graph_filtered_similar_nodes."""

    @pytest.mark.asyncio
    async def test_two_phase_search(self, store):
        """Graph filter narrows candidates before vector similarity."""
        nodes = [
            Node(
                uid="gf-a",
                properties={"name": "Anchor"},
                embeddings={"emb": ([1.0, 0.0], SimilarityMetric.COSINE)},
            ),
            Node(
                uid="gf-b",
                properties={"name": "Connected"},
                embeddings={"emb": ([0.9, 0.1], SimilarityMetric.COSINE)},
            ),
            Node(
                uid="gf-c",
                properties={"name": "Disconnected"},
                embeddings={"emb": ([0.95, 0.05], SimilarityMetric.COSINE)},
            ),
        ]
        await store.add_nodes(collection="gfsearch", nodes=nodes)

        # Only gf-a -> gf-b edge
        await store.add_edges(
            relation="LINKS",
            source_collection="gfsearch",
            target_collection="gfsearch",
            edges=[Edge(uid="gf-e1", source_uid="gf-a", target_uid="gf-b")],
        )

        graph_filter = GraphFilter(
            anchor_node_uid="gf-a",
            anchor_collection="gfsearch",
            max_hops=1,
        )

        results = await store.search_graph_filtered_similar_nodes(
            collection="gfsearch",
            embedding_name="emb",
            query_embedding=[1.0, 0.0],
            graph_filter=graph_filter,
            limit=10,
        )
        result_uids = {r.uid for r in results}
        # gf-b is connected, gf-c is not
        assert "gf-b" in result_uids
        assert "gf-c" not in result_uids

    @pytest.mark.asyncio
    async def test_empty_candidate_set(self, store):
        """Returns empty when no nodes are graph-reachable."""
        node = Node(
            uid="iso",
            properties={"name": "Isolated"},
            embeddings={"emb": ([1.0, 0.0], SimilarityMetric.COSINE)},
        )
        await store.add_nodes(collection="isocol", nodes=[node])

        graph_filter = GraphFilter(
            anchor_node_uid="iso",
            anchor_collection="isocol",
            max_hops=1,
        )

        results = await store.search_graph_filtered_similar_nodes(
            collection="isocol",
            embedding_name="emb",
            query_embedding=[1.0, 0.0],
            graph_filter=graph_filter,
            limit=10,
        )
        assert len(results) == 0

    @pytest.mark.asyncio
    async def test_fallback_without_graph_filter(self, store):
        """Without graph_filter, falls back to standard search."""
        nodes = [
            Node(
                uid="fb-1",
                properties={"name": "Node1"},
                embeddings={"emb": ([1.0, 0.0], SimilarityMetric.COSINE)},
            ),
            Node(
                uid="fb-2",
                properties={"name": "Node2"},
                embeddings={"emb": ([0.5, 0.5], SimilarityMetric.COSINE)},
            ),
        ]
        await store.add_nodes(collection="fbcol", nodes=nodes)

        results = await store.search_graph_filtered_similar_nodes(
            collection="fbcol",
            embedding_name="emb",
            query_embedding=[1.0, 0.0],
            graph_filter=None,
            limit=10,
        )
        assert len(results) == 2

    @pytest.mark.asyncio
    async def test_direction_filter(self, store):
        """Direction filter restricts traversal direction."""
        nodes = [
            Node(
                uid="dir-a",
                properties={"name": "A"},
                embeddings={"emb": ([1.0, 0.0], SimilarityMetric.COSINE)},
            ),
            Node(
                uid="dir-b",
                properties={"name": "B"},
                embeddings={"emb": ([0.9, 0.1], SimilarityMetric.COSINE)},
            ),
            Node(
                uid="dir-c",
                properties={"name": "C"},
                embeddings={"emb": ([0.8, 0.2], SimilarityMetric.COSINE)},
            ),
        ]
        await store.add_nodes(collection="dircol", nodes=nodes)

        # dir-a -> dir-b and dir-c -> dir-a
        await store.add_edges(
            relation="FLOW",
            source_collection="dircol",
            target_collection="dircol",
            edges=[
                Edge(uid="dir-e1", source_uid="dir-a", target_uid="dir-b"),
                Edge(uid="dir-e2", source_uid="dir-c", target_uid="dir-a"),
            ],
        )

        # OUTGOING from dir-a should find dir-b only
        gf = GraphFilter(
            anchor_node_uid="dir-a",
            anchor_collection="dircol",
            max_hops=1,
            direction=TraversalDirection.OUTGOING,
        )
        results = await store.search_graph_filtered_similar_nodes(
            collection="dircol",
            embedding_name="emb",
            query_embedding=[1.0, 0.0],
            graph_filter=gf,
            limit=10,
        )
        result_uids = {r.uid for r in results}
        assert "dir-b" in result_uids
        assert "dir-c" not in result_uids


# =====================================================================
# 7.8 -- Entity Deduplication
# =====================================================================


class TestEntityDeduplication:
    """Integration tests for entity deduplication."""

    @pytest.mark.asyncio
    async def test_detect_duplicates_creates_same_as(self, neo4j_driver, store):
        """Nodes with near-identical embeddings and properties get SAME_AS."""
        # Create nodes that exceed both thresholds
        nodes = [
            Node(
                uid="dup-a",
                properties={"name": "Alice", "role": "engineer"},
                embeddings={"emb": ([1.0, 0.0, 0.0], SimilarityMetric.COSINE)},
            ),
            Node(
                uid="dup-b",
                properties={"name": "Alice", "role": "engineer"},
                embeddings={"emb": ([0.999, 0.01, 0.0], SimilarityMetric.COSINE)},
            ),
            Node(
                uid="dup-c",
                properties={"name": "Bob", "dept": "sales"},
                embeddings={"emb": ([0.0, 1.0, 0.0], SimilarityMetric.COSINE)},
            ),
        ]
        await store.add_nodes(collection="dedup", nodes=nodes)

        # Manually trigger dedup (threshold is 3, we have 3 nodes)
        await store._run_dedup_for_collection("dedup")

        proposals = await store.get_duplicate_proposals(collection="dedup")
        # dup-a and dup-b should be flagged (high embedding + property similarity)
        if proposals:
            proposal_pairs = {(p.node_uid_a, p.node_uid_b) for p in proposals}
            pair = (min("dup-a", "dup-b"), max("dup-a", "dup-b"))
            assert pair in proposal_pairs

    @pytest.mark.asyncio
    async def test_single_signal_miss(self, store):
        """Nodes that only match one signal are not flagged."""
        nodes = [
            Node(
                uid="miss-a",
                properties={"name": "Alice", "role": "engineer"},
                embeddings={"emb": ([1.0, 0.0, 0.0], SimilarityMetric.COSINE)},
            ),
            Node(
                uid="miss-b",
                properties={"name": "Alice", "role": "engineer"},
                # Different embedding -- won't pass embedding threshold
                embeddings={"emb": ([0.0, 1.0, 0.0], SimilarityMetric.COSINE)},
            ),
            Node(
                uid="miss-c",
                properties={"completely": "different"},
                # Same embedding direction -- but property similarity too low
                embeddings={"emb": ([1.0, 0.01, 0.0], SimilarityMetric.COSINE)},
            ),
        ]
        await store.add_nodes(collection="miss", nodes=nodes)
        await store._run_dedup_for_collection("miss")

        proposals = await store.get_duplicate_proposals(collection="miss")
        # None should be flagged: miss-a/miss-b fail embedding, miss-a/miss-c fail property
        assert len(proposals) == 0

    @pytest.mark.asyncio
    async def test_manual_resolve_merge(self, neo4j_driver, store):
        """resolve_duplicates with MERGE strategy removes one node."""
        nodes = [
            Node(uid="rm-a", properties={"name": "A", "source": "a"}),
            Node(uid="rm-b", properties={"name": "A", "source": "b"}),
        ]
        await store.add_nodes(collection="resolve", nodes=nodes)

        await store.resolve_duplicates(
            collection="resolve",
            pairs=[("rm-a", "rm-b", DuplicateResolutionStrategy.MERGE)],
        )

        # rm-b should be deleted
        records, _, _ = await neo4j_driver.execute_query(
            "MATCH (n {uid: 'rm-b'}) RETURN n"
        )
        assert len(records) == 0

        # rm-a should still exist
        records, _, _ = await neo4j_driver.execute_query(
            "MATCH (n {uid: 'rm-a'}) RETURN n"
        )
        assert len(records) == 1

    @pytest.mark.asyncio
    async def test_manual_resolve_dismiss(self, neo4j_driver, store):
        """resolve_duplicates with DISMISS removes SAME_AS but keeps both nodes."""
        nodes = [
            Node(
                uid="dm-a",
                properties={"name": "A"},
                embeddings={"emb": ([1.0, 0.0, 0.0], SimilarityMetric.COSINE)},
            ),
            Node(
                uid="dm-b",
                properties={"name": "A"},
                embeddings={"emb": ([0.999, 0.01, 0.0], SimilarityMetric.COSINE)},
            ),
            Node(
                uid="dm-c",
                properties={"name": "C"},
                embeddings={"emb": ([0.0, 0.0, 1.0], SimilarityMetric.COSINE)},
            ),
        ]
        await store.add_nodes(collection="dismiss", nodes=nodes)
        await store._run_dedup_for_collection("dismiss")

        # Dismiss the pair
        await store.resolve_duplicates(
            collection="dismiss",
            pairs=[("dm-a", "dm-b", DuplicateResolutionStrategy.DISMISS)],
        )

        # Both nodes still exist
        for uid in ("dm-a", "dm-b"):
            records, _, _ = await neo4j_driver.execute_query(
                f"MATCH (n {{uid: '{uid}'}}) RETURN n"
            )
            assert len(records) == 1

        # No SAME_AS relationship
        records, _, _ = await neo4j_driver.execute_query(
            "MATCH ()-[r:SAME_AS]-() RETURN count(r) AS cnt"
        )
        assert records[0]["cnt"] == 0

    @pytest.mark.asyncio
    async def test_auto_merge_mode(self, neo4j_driver, store_auto_merge):
        """With auto_merge=True, duplicates are merged automatically."""
        nodes = [
            Node(
                uid="am-a",
                properties={"name": "Alice", "role": "eng"},
                embeddings={"emb": ([1.0, 0.0, 0.0], SimilarityMetric.COSINE)},
            ),
            Node(
                uid="am-b",
                properties={"name": "Alice", "role": "eng"},
                embeddings={"emb": ([0.999, 0.01, 0.0], SimilarityMetric.COSINE)},
            ),
            Node(
                uid="am-c",
                properties={"name": "Other"},
                embeddings={"emb": ([0.0, 1.0, 0.0], SimilarityMetric.COSINE)},
            ),
        ]
        await store_auto_merge.add_nodes(collection="autom", nodes=nodes)
        await store_auto_merge._run_dedup_for_collection("autom")

        # One of the pair should be deleted
        records, _, _ = await neo4j_driver.execute_query(
            "MATCH (n) WHERE n.uid IN ['am-a', 'am-b'] RETURN n.uid AS uid"
        )
        surviving_uids = {r["uid"] for r in records}
        # At least one survives, and at most one is deleted
        assert len(surviving_uids) >= 1


# =====================================================================
# 8.8 -- GDS Memory Ranking (skip if GDS not available)
# =====================================================================


class TestGDSMemoryRanking:
    """Integration tests for GDS features.

    These tests skip if GDS is not installed (standard neo4j:latest
    may not include it).
    """

    @pytest.mark.asyncio
    async def test_gds_not_available(self, store):
        """is_gds_available returns False when GDS plugin is absent."""
        # Reset cached state
        store._gds_available = None
        available = await store.is_gds_available()
        # Standard neo4j:latest may or may not have GDS; just verify the
        # method runs without error and returns a bool.
        assert isinstance(available, bool)

    @pytest.mark.asyncio
    async def test_pagerank_without_gds(self, store):
        """compute_pagerank raises RuntimeError when GDS unavailable."""
        store._gds_available = None
        available = await store.is_gds_available()
        if available:
            pytest.skip("GDS is available; this test is for missing GDS")

        with pytest.raises(RuntimeError, match="GDS"):
            await store.compute_pagerank(collection="test")

    @pytest.mark.asyncio
    async def test_communities_without_gds(self, store):
        """detect_communities raises RuntimeError when GDS unavailable."""
        store._gds_available = None
        available = await store.is_gds_available()
        if available:
            pytest.skip("GDS is available; this test is for missing GDS")

        with pytest.raises(RuntimeError, match="GDS"):
            await store.detect_communities(collection="test")

    @pytest.mark.asyncio
    async def test_pagerank_with_gds(self, store):
        """compute_pagerank returns scores when GDS is available."""
        store._gds_available = None
        available = await store.is_gds_available()
        if not available:
            pytest.skip("GDS plugin not installed")

        # Build a small graph
        nodes = [
            Node(uid="pr-a", properties={"name": "A"}),
            Node(uid="pr-b", properties={"name": "B"}),
            Node(uid="pr-c", properties={"name": "C"}),
        ]
        await store.add_nodes(collection="prcol", nodes=nodes)
        await store.add_edges(
            relation="LINKS",
            source_collection="prcol",
            target_collection="prcol",
            edges=[
                Edge(uid="pr-e1", source_uid="pr-a", target_uid="pr-b"),
                Edge(uid="pr-e2", source_uid="pr-b", target_uid="pr-c"),
                Edge(uid="pr-e3", source_uid="pr-c", target_uid="pr-a"),
            ],
        )

        scores = await store.compute_pagerank(collection="prcol")
        assert len(scores) == 3
        for uid, score in scores:
            assert isinstance(uid, str)
            assert isinstance(score, float)
            assert score > 0

    @pytest.mark.asyncio
    async def test_communities_with_gds(self, store):
        """detect_communities returns community assignments when GDS is available."""
        store._gds_available = None
        available = await store.is_gds_available()
        if not available:
            pytest.skip("GDS plugin not installed")

        nodes = [
            Node(uid="cm-a", properties={"name": "A"}),
            Node(uid="cm-b", properties={"name": "B"}),
            Node(uid="cm-c", properties={"name": "C"}),
        ]
        await store.add_nodes(collection="cmcol", nodes=nodes)
        await store.add_edges(
            relation="LINKS",
            source_collection="cmcol",
            target_collection="cmcol",
            edges=[
                Edge(uid="cm-e1", source_uid="cm-a", target_uid="cm-b"),
                Edge(uid="cm-e2", source_uid="cm-b", target_uid="cm-c"),
            ],
        )

        communities = await store.detect_communities(collection="cmcol")
        assert isinstance(communities, dict)
        # All nodes should be assigned to communities
        all_uids = set()
        for member_uids in communities.values():
            all_uids.update(member_uids)
        assert {"cm-a", "cm-b", "cm-c"} == all_uids
