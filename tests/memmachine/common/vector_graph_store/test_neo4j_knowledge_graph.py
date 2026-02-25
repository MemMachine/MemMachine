"""Unit tests for Neo4j knowledge-graph improvements.

Tests the new methods added to Neo4jVectorGraphStore and the migration module
using mocked Neo4j drivers so they run without Docker / testcontainers.
"""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock

import pytest

from memmachine.common.data_types import SimilarityMetric
from memmachine.common.vector_graph_store.data_types import (
    DuplicateProposal,
    DuplicateResolutionStrategy,
    GraphFilter,
    Node,
    TraversalDirection,
)
from memmachine.common.vector_graph_store.neo4j_vector_graph_store import (
    Neo4jVectorGraphStore,
    Neo4jVectorGraphStoreParams,
    _cosine_similarity,
    _first_embedding,
    _jaccard_similarity,
    _similarity_function_name,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_execute_query_result(records: list[dict]) -> tuple:
    """Build a (records, summary, keys) tuple for execute_query mocking."""
    result_records = [MagicMock(__getitem__=lambda self, k, r=r: r[k]) for r in records]
    for rec, data in zip(result_records, records, strict=True):
        rec.__getitem__ = lambda self, k, d=data: d[k]
        rec.keys = lambda d=data: d.keys()
    return (result_records, MagicMock(), MagicMock())


def _make_neo4j_node(
    uid: str,
    labels: frozenset[str] | None = None,
    properties: dict | None = None,
) -> MagicMock:
    """Create a mock neo4j.graph.Node."""
    props = {"uid": uid}
    if properties:
        props.update(properties)
    node = MagicMock()
    node.__getitem__ = lambda self, k, p=props: p[k]
    node.get = lambda k, default=None, p=props: p.get(k, default)
    node.items = lambda p=props: p.items()
    node.labels = labels or frozenset()
    node.element_id = f"eid_{uid}"
    return node


@pytest.fixture
def mock_driver() -> AsyncMock:
    """Create a mock Neo4j AsyncDriver with spec to satisfy Pydantic InstanceOf."""
    from neo4j import AsyncDriver

    driver = AsyncMock(spec=AsyncDriver)
    driver.execute_query = AsyncMock(return_value=([], MagicMock(), MagicMock()))
    return driver


@pytest.fixture
def store(mock_driver: AsyncMock) -> Neo4jVectorGraphStore:
    """Create a Neo4jVectorGraphStore with mocked driver."""
    return Neo4jVectorGraphStore(
        Neo4jVectorGraphStoreParams(
            driver=mock_driver,
            force_exact_similarity_search=True,
            dedup_trigger_threshold=0,  # Disable auto-dedup for most tests
        )
    )


@pytest.fixture
def store_with_dedup(mock_driver: AsyncMock) -> Neo4jVectorGraphStore:
    """Create a store with dedup enabled and low thresholds."""
    return Neo4jVectorGraphStore(
        Neo4jVectorGraphStoreParams(
            driver=mock_driver,
            force_exact_similarity_search=True,
            dedup_trigger_threshold=2,
            dedup_embedding_threshold=0.9,
            dedup_property_threshold=0.5,
            dedup_auto_merge=False,
        )
    )


# ---------------------------------------------------------------------------
# Module-level helper tests
# ---------------------------------------------------------------------------


class TestSimilarityFunctionName:
    def test_cosine(self) -> None:
        assert (
            _similarity_function_name(SimilarityMetric.COSINE)
            == "vector.similarity.cosine"
        )

    def test_euclidean(self) -> None:
        assert (
            _similarity_function_name(SimilarityMetric.EUCLIDEAN)
            == "vector.similarity.euclidean"
        )


class TestCosineSimilarity:
    def test_identical_vectors(self) -> None:
        vec = [1.0, 0.0, 0.0]
        assert _cosine_similarity(vec, vec) == pytest.approx(1.0)

    def test_orthogonal_vectors(self) -> None:
        assert _cosine_similarity([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)

    def test_opposite_vectors(self) -> None:
        assert _cosine_similarity([1.0, 0.0], [-1.0, 0.0]) == pytest.approx(-1.0)

    def test_zero_vector_returns_zero(self) -> None:
        assert _cosine_similarity([0.0, 0.0], [1.0, 0.0]) == 0.0


class TestJaccardSimilarity:
    def test_identical_sets(self) -> None:
        s = {"a", "b", "c"}
        assert _jaccard_similarity(s, s) == pytest.approx(1.0)

    def test_disjoint_sets(self) -> None:
        assert _jaccard_similarity({"a"}, {"b"}) == pytest.approx(0.0)

    def test_overlapping_sets(self) -> None:
        assert _jaccard_similarity({"a", "b"}, {"b", "c"}) == pytest.approx(1 / 3)

    def test_empty_sets(self) -> None:
        assert _jaccard_similarity(set(), set()) == 0.0


class TestFirstEmbedding:
    def test_node_with_embedding(self) -> None:
        emb = [0.1, 0.2, 0.3]
        node = Node(uid="n1", embeddings={"e": (emb, SimilarityMetric.COSINE)})
        assert _first_embedding(node) == emb

    def test_node_without_embedding(self) -> None:
        node = Node(uid="n1")
        assert _first_embedding(node) is None


# ---------------------------------------------------------------------------
# Entity type label helpers
# ---------------------------------------------------------------------------


class TestEntityTypeSanitization:
    def test_sanitize_entity_type(self) -> None:
        result = Neo4jVectorGraphStore._sanitize_entity_type("Person")
        assert result.startswith("ENTITY_TYPE_")
        assert "Person" in result

    def test_desanitize_entity_type(self) -> None:
        sanitized = Neo4jVectorGraphStore._sanitize_entity_type("Person")
        assert Neo4jVectorGraphStore._desanitize_entity_type(sanitized) == "Person"

    def test_roundtrip(self) -> None:
        for name in ["Person", "AI Agent", "concept_category"]:
            sanitized = Neo4jVectorGraphStore._sanitize_entity_type(name)
            assert Neo4jVectorGraphStore._desanitize_entity_type(sanitized) == name

    def test_entity_type_label_fragment_empty(self) -> None:
        assert Neo4jVectorGraphStore._entity_type_label_fragment(None) == ""
        assert Neo4jVectorGraphStore._entity_type_label_fragment([]) == ""

    def test_entity_type_label_fragment_single(self) -> None:
        result = Neo4jVectorGraphStore._entity_type_label_fragment(["Person"])
        assert result.startswith(":")
        assert "ENTITY_TYPE_" in result

    def test_entity_type_label_fragment_multiple(self) -> None:
        result = Neo4jVectorGraphStore._entity_type_label_fragment(["Person", "Agent"])
        # Should have two `:ENTITY_TYPE_` fragments
        assert result.count(":ENTITY_TYPE_") == 2


# ---------------------------------------------------------------------------
# _ensure_collection_constraints
# ---------------------------------------------------------------------------


class TestEnsureCollectionConstraints:
    @pytest.mark.asyncio
    async def test_creates_constraint_on_first_call(
        self, store: Neo4jVectorGraphStore, mock_driver: AsyncMock
    ) -> None:
        sanitized = Neo4jVectorGraphStore._sanitize_name("test_collection")
        await store._ensure_collection_constraints(sanitized)
        mock_driver.execute_query.assert_called_once()
        query_arg = str(mock_driver.execute_query.call_args[0][0])
        assert "CREATE CONSTRAINT" in query_arg
        assert "IF NOT EXISTS" in query_arg

    @pytest.mark.asyncio
    async def test_skips_on_second_call(
        self, store: Neo4jVectorGraphStore, mock_driver: AsyncMock
    ) -> None:
        sanitized = Neo4jVectorGraphStore._sanitize_name("test_collection")
        await store._ensure_collection_constraints(sanitized)
        mock_driver.execute_query.reset_mock()
        await store._ensure_collection_constraints(sanitized)
        mock_driver.execute_query.assert_not_called()


# ---------------------------------------------------------------------------
# add_nodes with entity types
# ---------------------------------------------------------------------------


class TestAddNodesEntityTypes:
    @pytest.mark.asyncio
    async def test_add_nodes_with_entity_types(
        self, store: Neo4jVectorGraphStore, mock_driver: AsyncMock
    ) -> None:
        """Nodes with entity types should produce MERGE + SET with extra labels."""
        # First call: _count_nodes returns 0
        mock_driver.execute_query.side_effect = [
            _make_execute_query_result([{"node_count": 0}]),  # _count_nodes
            ([], MagicMock(), MagicMock()),  # _ensure_collection_constraints
            ([], MagicMock(), MagicMock()),  # MERGE query
        ]

        node = Node(uid="n1", entity_types=["Person", "Agent"])
        await store.add_nodes(collection="test", nodes=[node])

        # The third call should be the MERGE with entity type labels.
        merge_call = mock_driver.execute_query.call_args_list[2]
        query_text = str(merge_call[0][0])
        assert "MERGE" in query_text
        assert "ENTITY_TYPE_" in query_text

    @pytest.mark.asyncio
    async def test_add_nodes_without_entity_types(
        self, store: Neo4jVectorGraphStore, mock_driver: AsyncMock
    ) -> None:
        """Nodes without entity types should still use MERGE."""
        mock_driver.execute_query.side_effect = [
            _make_execute_query_result([{"node_count": 0}]),  # _count_nodes
            ([], MagicMock(), MagicMock()),  # _ensure_collection_constraints
            ([], MagicMock(), MagicMock()),  # MERGE query
        ]

        node = Node(uid="n1")
        await store.add_nodes(collection="test", nodes=[node])

        merge_call = mock_driver.execute_query.call_args_list[2]
        query_text = str(merge_call[0][0])
        assert "MERGE" in query_text
        # No extra SET for entity types
        assert "ENTITY_TYPE_" not in query_text


# ---------------------------------------------------------------------------
# add_edges with MERGE
# ---------------------------------------------------------------------------


class TestAddEdgesMerge:
    @pytest.mark.asyncio
    async def test_add_edges_uses_merge(
        self, store: Neo4jVectorGraphStore, mock_driver: AsyncMock
    ) -> None:
        from memmachine.common.vector_graph_store.data_types import Edge

        mock_driver.execute_query.side_effect = [
            _make_execute_query_result([{"relationship_count": 0}]),  # _count_edges
            ([], MagicMock(), MagicMock()),  # MERGE query
        ]

        edge = Edge(uid="e1", source_uid="n1", target_uid="n2")
        await store.add_edges(
            relation="KNOWS",
            source_collection="people",
            target_collection="people",
            edges=[edge],
        )

        merge_call = mock_driver.execute_query.call_args_list[1]
        query_text = str(merge_call[0][0])
        assert "MERGE" in query_text


# ---------------------------------------------------------------------------
# update_entity_types
# ---------------------------------------------------------------------------


class TestUpdateEntityTypes:
    @pytest.mark.asyncio
    async def test_update_adds_and_removes_labels(
        self, store: Neo4jVectorGraphStore, mock_driver: AsyncMock
    ) -> None:
        """Should add new labels and remove stale ones."""
        current_labels = [
            Neo4jVectorGraphStore._sanitize_name("test"),
            Neo4jVectorGraphStore._sanitize_entity_type("OldType"),
        ]
        mock_driver.execute_query.side_effect = [
            _make_execute_query_result([{"labels": current_labels}]),
            ([], MagicMock(), MagicMock()),  # SET/REMOVE query
        ]

        await store.update_entity_types(
            collection="test",
            node_uid="n1",
            entity_types=["NewType"],
        )

        # Second call should SET the new and REMOVE the old
        update_call = mock_driver.execute_query.call_args_list[1]
        query_text = str(update_call[0][0])
        assert "SET" in query_text or "REMOVE" in query_text

    @pytest.mark.asyncio
    async def test_update_noop_when_labels_match(
        self, store: Neo4jVectorGraphStore, mock_driver: AsyncMock
    ) -> None:
        """No query when desired labels already match."""
        current_labels = [
            Neo4jVectorGraphStore._sanitize_name("test"),
            Neo4jVectorGraphStore._sanitize_entity_type("Person"),
        ]
        mock_driver.execute_query.side_effect = [
            _make_execute_query_result([{"labels": current_labels}]),
        ]

        await store.update_entity_types(
            collection="test",
            node_uid="n1",
            entity_types=["Person"],
        )

        # Only one call: the label fetch. No SET/REMOVE call needed.
        assert mock_driver.execute_query.call_count == 1

    @pytest.mark.asyncio
    async def test_update_warns_on_missing_node(
        self, store: Neo4jVectorGraphStore, mock_driver: AsyncMock
    ) -> None:
        """Should not error when node is not found."""
        mock_driver.execute_query.return_value = ([], MagicMock(), MagicMock())

        # Should not raise
        await store.update_entity_types(
            collection="test",
            node_uid="missing",
            entity_types=["Person"],
        )


# ---------------------------------------------------------------------------
# _nodes_from_neo4j_nodes entity type extraction
# ---------------------------------------------------------------------------


class TestNodesFromNeo4jNodesEntityTypes:
    def test_extracts_entity_type_labels(self) -> None:
        """Entity type labels should be extracted and desanitized."""
        sanitized_type = Neo4jVectorGraphStore._sanitize_entity_type("Person")
        sanitized_collection = Neo4jVectorGraphStore._sanitize_name("test")
        neo4j_node = _make_neo4j_node(
            "n1",
            labels=frozenset({sanitized_collection, sanitized_type}),
        )
        nodes = Neo4jVectorGraphStore._nodes_from_neo4j_nodes([neo4j_node])
        assert len(nodes) == 1
        assert "Person" in nodes[0].entity_types

    def test_no_entity_type_labels(self) -> None:
        sanitized_collection = Neo4jVectorGraphStore._sanitize_name("test")
        neo4j_node = _make_neo4j_node("n1", labels=frozenset({sanitized_collection}))
        nodes = Neo4jVectorGraphStore._nodes_from_neo4j_nodes([neo4j_node])
        assert nodes[0].entity_types == []


# ---------------------------------------------------------------------------
# search_multi_hop_nodes
# ---------------------------------------------------------------------------


class TestSearchMultiHopNodes:
    @pytest.mark.asyncio
    async def test_basic_multi_hop(
        self, store: Neo4jVectorGraphStore, mock_driver: AsyncMock
    ) -> None:
        """Should return MultiHopResult with decay scoring."""
        sanitized = Neo4jVectorGraphStore._sanitize_name("test")
        neo4j_node = _make_neo4j_node("n2", labels=frozenset({sanitized}))

        mock_driver.execute_query.return_value = (
            [
                MagicMock(
                    __getitem__=lambda self, k: {
                        "end_node": neo4j_node,
                        "hop_distance": 2,
                        "path_quality": 1.0,
                    }[k]
                )
            ],
            MagicMock(),
            MagicMock(),
        )

        results = await store.search_multi_hop_nodes(
            collection="test",
            this_node_uid="n1",
            max_hops=3,
            score_decay=0.5,
        )

        assert len(results) == 1
        assert results[0].hop_distance == 2
        assert results[0].path_quality == 1.0
        assert results[0].score == pytest.approx(0.25)  # 0.5 ** 2 * 1.0

    @pytest.mark.asyncio
    async def test_max_hops_clamped(
        self, store: Neo4jVectorGraphStore, mock_driver: AsyncMock
    ) -> None:
        """max_hops should be clamped to ceiling of 5."""
        mock_driver.execute_query.return_value = ([], MagicMock(), MagicMock())

        await store.search_multi_hop_nodes(
            collection="test",
            this_node_uid="n1",
            max_hops=100,
        )

        query_text = str(mock_driver.execute_query.call_args[0][0])
        # The path pattern should use ..5 not ..100
        assert "*1..5" in query_text

    @pytest.mark.asyncio
    async def test_relation_type_filtering(
        self, store: Neo4jVectorGraphStore, mock_driver: AsyncMock
    ) -> None:
        """Should include relationship type pattern in Cypher."""
        mock_driver.execute_query.return_value = ([], MagicMock(), MagicMock())

        await store.search_multi_hop_nodes(
            collection="test",
            this_node_uid="n1",
            relation_types=["KNOWS", "LIKES"],
        )

        query_text = str(mock_driver.execute_query.call_args[0][0])
        # Should have sanitized relation types joined with |
        assert "|" in query_text


# ---------------------------------------------------------------------------
# search_graph_filtered_similar_nodes
# ---------------------------------------------------------------------------


class TestSearchGraphFilteredSimilarNodes:
    @pytest.mark.asyncio
    async def test_fallback_without_graph_filter(
        self, store: Neo4jVectorGraphStore, mock_driver: AsyncMock
    ) -> None:
        """Without graph_filter, should fall back to search_similar_nodes."""
        mock_driver.execute_query.return_value = ([], MagicMock(), MagicMock())

        results = await store.search_graph_filtered_similar_nodes(
            collection="test",
            embedding_name="emb",
            query_embedding=[0.1, 0.2],
            graph_filter=None,
        )

        assert results == []

    @pytest.mark.asyncio
    async def test_two_phase_with_graph_filter(
        self, store: Neo4jVectorGraphStore, mock_driver: AsyncMock
    ) -> None:
        """With graph_filter, should run graph pre-filter then vector search."""
        # Phase 1: return candidate eids
        phase1_result = (
            [MagicMock(__getitem__=lambda self, k: "eid_123")],
            MagicMock(),
            MagicMock(),
        )
        # Phase 2: return vector search results (empty for simplicity)
        phase2_result = ([], MagicMock(), MagicMock())

        mock_driver.execute_query.side_effect = [phase1_result, phase2_result]

        gf = GraphFilter(
            anchor_node_uid="anchor1",
            anchor_collection="test",
            max_hops=2,
        )
        await store.search_graph_filtered_similar_nodes(
            collection="test",
            embedding_name="emb",
            query_embedding=[0.1, 0.2],
            graph_filter=gf,
        )

        assert mock_driver.execute_query.call_count == 2

    @pytest.mark.asyncio
    async def test_empty_candidate_set(
        self, store: Neo4jVectorGraphStore, mock_driver: AsyncMock
    ) -> None:
        """When graph pre-filter returns no candidates, should return empty list."""
        mock_driver.execute_query.return_value = ([], MagicMock(), MagicMock())

        gf = GraphFilter(
            anchor_node_uid="anchor1",
            anchor_collection="test",
            max_hops=2,
        )
        results = await store.search_graph_filtered_similar_nodes(
            collection="test",
            embedding_name="emb",
            query_embedding=[0.1, 0.2],
            graph_filter=gf,
        )

        assert results == []
        # Only the candidate query was executed, no vector search
        assert mock_driver.execute_query.call_count == 1

    @pytest.mark.asyncio
    async def test_direction_outgoing(
        self, store: Neo4jVectorGraphStore, mock_driver: AsyncMock
    ) -> None:
        """Outgoing direction should produce -> in pattern."""
        mock_driver.execute_query.return_value = ([], MagicMock(), MagicMock())

        gf = GraphFilter(
            anchor_node_uid="a1",
            anchor_collection="test",
            max_hops=1,
            direction=TraversalDirection.OUTGOING,
        )
        await store.search_graph_filtered_similar_nodes(
            collection="test",
            embedding_name="emb",
            query_embedding=[0.1],
            graph_filter=gf,
        )

        query_text = str(mock_driver.execute_query.call_args[0][0])
        assert "->" in query_text

    @pytest.mark.asyncio
    async def test_direction_incoming(
        self, store: Neo4jVectorGraphStore, mock_driver: AsyncMock
    ) -> None:
        """Incoming direction should produce <- in pattern."""
        mock_driver.execute_query.return_value = ([], MagicMock(), MagicMock())

        gf = GraphFilter(
            anchor_node_uid="a1",
            anchor_collection="test",
            max_hops=1,
            direction=TraversalDirection.INCOMING,
        )
        await store.search_graph_filtered_similar_nodes(
            collection="test",
            embedding_name="emb",
            query_embedding=[0.1],
            graph_filter=gf,
        )

        query_text = str(mock_driver.execute_query.call_args[0][0])
        assert "<-" in query_text


# ---------------------------------------------------------------------------
# Entity deduplication
# ---------------------------------------------------------------------------


class TestDetectDuplicates:
    @pytest.mark.asyncio
    async def test_no_duplicates_below_threshold(
        self, store: Neo4jVectorGraphStore, mock_driver: AsyncMock
    ) -> None:
        """Two nodes with low similarity should not produce proposals."""
        sanitized = Neo4jVectorGraphStore._sanitize_name("test")
        from memmachine.common.vector_graph_store.data_types import (
            mangle_embedding_name,
            mangle_property_name,
        )

        emb_key = Neo4jVectorGraphStore._sanitize_name(mangle_embedding_name("emb"))
        sim_key = Neo4jVectorGraphStore._sanitize_name("similarity_metric_for_emb")
        prop_key = Neo4jVectorGraphStore._sanitize_name(mangle_property_name("name"))

        _data1 = {
            "uid": "n1",
            emb_key: [1.0, 0.0],
            sim_key: "cosine",
            prop_key: "Alice",
        }
        node1 = MagicMock()
        node1.__getitem__ = lambda self, k: _data1[k]
        node1.get = lambda k, default=None: _data1.get(k, default)
        node1.items = lambda: _data1.items()
        node1.labels = frozenset({sanitized})

        _data2 = {
            "uid": "n2",
            emb_key: [0.0, 1.0],
            sim_key: "cosine",
            prop_key: "Bob",
        }
        node2 = MagicMock()
        node2.__getitem__ = lambda self, k: _data2[k]
        node2.get = lambda k, default=None: _data2.get(k, default)
        node2.items = lambda: _data2.items()
        node2.labels = frozenset({sanitized})

        mock_driver.execute_query.return_value = (
            [
                MagicMock(__getitem__=lambda self, k: node1),
                MagicMock(__getitem__=lambda self, k: node2),
            ],
            MagicMock(),
            MagicMock(),
        )

        proposals = await store._detect_duplicates("test")
        assert proposals == []

    @pytest.mark.asyncio
    async def test_single_node_returns_empty(
        self, store: Neo4jVectorGraphStore, mock_driver: AsyncMock
    ) -> None:
        """A single node cannot have duplicates."""
        mock_driver.execute_query.return_value = (
            [MagicMock()],
            MagicMock(),
            MagicMock(),
        )
        proposals = await store._detect_duplicates("test")
        assert proposals == []

    @pytest.mark.asyncio
    async def test_empty_collection(
        self, store: Neo4jVectorGraphStore, mock_driver: AsyncMock
    ) -> None:
        """Empty collection returns no proposals."""
        mock_driver.execute_query.return_value = ([], MagicMock(), MagicMock())
        proposals = await store._detect_duplicates("test")
        assert proposals == []


class TestCompareNodePair:
    def test_above_thresholds(self, store: Neo4jVectorGraphStore) -> None:
        node_a = Node(
            uid="a",
            embeddings={"e": ([1.0, 0.0, 0.0], SimilarityMetric.COSINE)},
            properties={"x": 1, "y": 2},
        )
        node_b = Node(
            uid="b",
            embeddings={"e": ([1.0, 0.0, 0.0], SimilarityMetric.COSINE)},
            properties={"x": 1, "y": 2},
        )
        now = datetime.now(UTC)
        pair_a = (node_a, [1.0, 0.0, 0.0], {"x", "y"})
        pair_b = (node_b, [1.0, 0.0, 0.0], {"x", "y"})

        proposal = store._compare_node_pair(pair_a, pair_b, now)
        assert proposal is not None
        assert proposal.embedding_similarity == pytest.approx(1.0)
        assert proposal.property_similarity == pytest.approx(1.0)

    def test_below_embedding_threshold(self, store: Neo4jVectorGraphStore) -> None:
        now = datetime.now(UTC)
        pair_a = (Node(uid="a"), [1.0, 0.0], {"x"})
        pair_b = (Node(uid="b"), [0.0, 1.0], {"x"})

        proposal = store._compare_node_pair(pair_a, pair_b, now)
        assert proposal is None

    def test_no_embeddings(self, store: Neo4jVectorGraphStore) -> None:
        now = datetime.now(UTC)
        pair_a = (Node(uid="a"), None, {"x"})
        pair_b = (Node(uid="b"), None, {"x"})

        proposal = store._compare_node_pair(pair_a, pair_b, now)
        assert proposal is None


class TestCreateSameAsRelationships:
    @pytest.mark.asyncio
    async def test_creates_merge_for_each_proposal(
        self, store: Neo4jVectorGraphStore, mock_driver: AsyncMock
    ) -> None:
        proposals = [
            DuplicateProposal(
                node_uid_a="a",
                node_uid_b="b",
                embedding_similarity=0.98,
                property_similarity=0.9,
                detected_at=datetime.now(UTC),
            ),
        ]
        mock_driver.execute_query.return_value = ([], MagicMock(), MagicMock())

        await store._create_same_as_relationships("test", proposals)

        query_text = str(mock_driver.execute_query.call_args[0][0])
        assert "SAME_AS" in query_text
        assert "MERGE" in query_text


class TestResolveDuplicates:
    @pytest.mark.asyncio
    async def test_merge_strategy(
        self, store: Neo4jVectorGraphStore, mock_driver: AsyncMock
    ) -> None:
        mock_driver.execute_query.return_value = ([], MagicMock(), MagicMock())

        await store.resolve_duplicates(
            collection="test",
            pairs=[("a", "b", DuplicateResolutionStrategy.MERGE)],
        )

        # Should have 2 calls: SET a += properties(b), then DETACH DELETE b
        assert mock_driver.execute_query.call_count == 2
        first_query = str(mock_driver.execute_query.call_args_list[0][0][0])
        second_query = str(mock_driver.execute_query.call_args_list[1][0][0])
        assert "SET" in first_query
        assert "DETACH DELETE" in second_query

    @pytest.mark.asyncio
    async def test_dismiss_strategy(
        self, store: Neo4jVectorGraphStore, mock_driver: AsyncMock
    ) -> None:
        mock_driver.execute_query.return_value = ([], MagicMock(), MagicMock())

        await store.resolve_duplicates(
            collection="test",
            pairs=[("a", "b", DuplicateResolutionStrategy.DISMISS)],
        )

        assert mock_driver.execute_query.call_count == 1
        query_text = str(mock_driver.execute_query.call_args[0][0])
        assert "DELETE r" in query_text
        assert "SAME_AS" in query_text


class TestGetDuplicateProposals:
    @pytest.mark.asyncio
    async def test_returns_proposals(
        self, store: Neo4jVectorGraphStore, mock_driver: AsyncMock
    ) -> None:
        now = datetime.now(UTC)
        record = MagicMock()
        record.__getitem__ = lambda self, k, d={
            "uid_a": "a",
            "uid_b": "b",
            "emb_sim": 0.98,
            "prop_sim": 0.85,
            "detected_at": now,
            "auto_merged": False,
        }: d[k]

        mock_driver.execute_query.return_value = ([record], MagicMock(), MagicMock())

        proposals = await store.get_duplicate_proposals(collection="test")

        assert len(proposals) == 1
        assert proposals[0].node_uid_a == "a"
        assert proposals[0].node_uid_b == "b"

    @pytest.mark.asyncio
    async def test_deduplicates_bidirectional_pairs(
        self, store: Neo4jVectorGraphStore, mock_driver: AsyncMock
    ) -> None:
        """SAME_AS is undirected, so (a,b) and (b,a) should collapse to one."""
        now = datetime.now(UTC)
        rec1 = MagicMock()
        rec1.__getitem__ = lambda self, k, d={
            "uid_a": "a",
            "uid_b": "b",
            "emb_sim": 0.98,
            "prop_sim": 0.85,
            "detected_at": now,
            "auto_merged": False,
        }: d[k]
        rec2 = MagicMock()
        rec2.__getitem__ = lambda self, k, d={
            "uid_a": "b",
            "uid_b": "a",
            "emb_sim": 0.98,
            "prop_sim": 0.85,
            "detected_at": now,
            "auto_merged": False,
        }: d[k]

        mock_driver.execute_query.return_value = (
            [rec1, rec2],
            MagicMock(),
            MagicMock(),
        )

        proposals = await store.get_duplicate_proposals(collection="test")
        assert len(proposals) == 1


# ---------------------------------------------------------------------------
# GDS Memory Ranking
# ---------------------------------------------------------------------------


class TestIsGdsAvailable:
    @pytest.mark.asyncio
    async def test_available(
        self, store: Neo4jVectorGraphStore, mock_driver: AsyncMock
    ) -> None:
        mock_driver.execute_query.return_value = (
            [MagicMock(__getitem__=lambda self, k: "2.6.0")],
            MagicMock(),
            MagicMock(),
        )
        # Reset cached state and enable GDS config
        store._gds_available = None
        store._gds_enabled = True

        result = await store.is_gds_available()
        assert result is True

    @pytest.mark.asyncio
    async def test_not_available(
        self, store: Neo4jVectorGraphStore, mock_driver: AsyncMock
    ) -> None:
        mock_driver.execute_query.side_effect = Exception("GDS not installed")
        store._gds_available = None
        store._gds_enabled = True

        result = await store.is_gds_available()
        assert result is False

    @pytest.mark.asyncio
    async def test_returns_false_when_gds_disabled(
        self, store: Neo4jVectorGraphStore, mock_driver: AsyncMock
    ) -> None:
        store._gds_enabled = False
        store._gds_available = None

        result = await store.is_gds_available()
        assert result is False
        mock_driver.execute_query.assert_not_called()

    @pytest.mark.asyncio
    async def test_cached(
        self, store: Neo4jVectorGraphStore, mock_driver: AsyncMock
    ) -> None:
        store._gds_enabled = True
        store._gds_available = True
        result = await store.is_gds_available()
        assert result is True
        mock_driver.execute_query.assert_not_called()


class TestComputePagerank:
    @pytest.mark.asyncio
    async def test_raises_when_gds_unavailable(
        self, store: Neo4jVectorGraphStore
    ) -> None:
        store._gds_enabled = True
        store._gds_available = False
        with pytest.raises(RuntimeError, match="GDS plugin is not available"):
            await store.compute_pagerank(collection="test")

    @pytest.mark.asyncio
    async def test_returns_ranked_results(
        self, store: Neo4jVectorGraphStore, mock_driver: AsyncMock
    ) -> None:
        store._gds_enabled = True
        store._gds_available = True

        record = MagicMock()
        record.__getitem__ = lambda self, k, d={"uid": "n1", "score": 0.85}: d[k]

        graph_list_rec = MagicMock()
        graph_list_rec.__getitem__ = lambda self, k, d={
            "nodeCount": 5,
            "relationshipCount": 10,
        }: d[k]

        mock_driver.execute_query.side_effect = [
            ([], MagicMock(), MagicMock()),  # gds.graph.project.cypher
            ([graph_list_rec], MagicMock(), MagicMock()),  # gds.graph.list
            ([record], MagicMock(), MagicMock()),  # gds.pageRank.stream
            ([], MagicMock(), MagicMock()),  # _drop_projection
        ]

        results = await store.compute_pagerank(collection="test")
        assert len(results) == 1
        assert results[0] == ("n1", 0.85)


class TestDetectCommunities:
    @pytest.mark.asyncio
    async def test_raises_when_gds_unavailable(
        self, store: Neo4jVectorGraphStore
    ) -> None:
        store._gds_enabled = True
        store._gds_available = False
        with pytest.raises(RuntimeError, match="GDS plugin is not available"):
            await store.detect_communities(collection="test")

    @pytest.mark.asyncio
    async def test_returns_community_mapping(
        self, store: Neo4jVectorGraphStore, mock_driver: AsyncMock
    ) -> None:
        store._gds_enabled = True
        store._gds_available = True

        rec1 = MagicMock()
        rec1.__getitem__ = lambda self, k, d={"uid": "n1", "communityId": 0}: d[k]
        rec2 = MagicMock()
        rec2.__getitem__ = lambda self, k, d={"uid": "n2", "communityId": 0}: d[k]
        rec3 = MagicMock()
        rec3.__getitem__ = lambda self, k, d={"uid": "n3", "communityId": 1}: d[k]

        graph_list_rec = MagicMock()
        graph_list_rec.__getitem__ = lambda self, k, d={
            "nodeCount": 3,
            "relationshipCount": 2,
        }: d[k]

        mock_driver.execute_query.side_effect = [
            ([], MagicMock(), MagicMock()),  # gds.graph.project.cypher
            ([graph_list_rec], MagicMock(), MagicMock()),  # gds.graph.list
            ([rec1, rec2, rec3], MagicMock(), MagicMock()),  # gds.louvain.stream
            ([], MagicMock(), MagicMock()),  # _drop_projection
        ]

        communities = await store.detect_communities(collection="test")
        assert 0 in communities
        assert len(communities[0]) == 2
        assert 1 in communities
        assert len(communities[1]) == 1


# ---------------------------------------------------------------------------
# Migration module
# ---------------------------------------------------------------------------


class TestMigrationAuditDuplicateUids:
    @pytest.mark.asyncio
    async def test_no_duplicates(self) -> None:
        from memmachine.common.vector_graph_store.neo4j_migration import (
            audit_duplicate_uids,
        )

        driver = AsyncMock()
        # First call: db.labels
        driver.execute_query.side_effect = [
            _make_execute_query_result([{"label": "SANITIZED_test"}]),
            # Second call: duplicate query for that label
            ([], MagicMock(), MagicMock()),
        ]

        report = await audit_duplicate_uids(driver)
        assert report.total_duplicates == 0

    @pytest.mark.asyncio
    async def test_with_duplicates(self) -> None:
        from memmachine.common.vector_graph_store.neo4j_migration import (
            audit_duplicate_uids,
        )

        driver = AsyncMock()
        driver.execute_query.side_effect = [
            _make_execute_query_result([{"label": "SANITIZED_test"}]),
            _make_execute_query_result([{"uid": "dup1", "cnt": 3}]),
        ]

        report = await audit_duplicate_uids(driver)
        assert report.total_duplicates == 2  # 3 - 1 = 2 surplus


class TestMigrationApplyUniquenessConstraints:
    @pytest.mark.asyncio
    async def test_creates_constraints(self) -> None:
        from memmachine.common.vector_graph_store.neo4j_migration import (
            apply_uniqueness_constraints,
        )

        driver = AsyncMock()
        driver.execute_query.side_effect = [
            _make_execute_query_result(
                [
                    {"label": "SANITIZED_coll1"},
                    {"label": "SANITIZED_coll2"},
                ]
            ),
            ([], MagicMock(), MagicMock()),  # constraint 1
            ([], MagicMock(), MagicMock()),  # constraint 2
        ]

        names = await apply_uniqueness_constraints(driver)
        assert len(names) == 2


class TestMigrationBackfillEntityTypeLabels:
    @pytest.mark.asyncio
    async def test_no_mapping(self) -> None:
        from memmachine.common.vector_graph_store.neo4j_migration import (
            backfill_entity_type_labels,
        )

        driver = AsyncMock()
        result = await backfill_entity_type_labels(driver, None)
        assert result == 0
        driver.execute_query.assert_not_called()

    @pytest.mark.asyncio
    async def test_applies_labels(self) -> None:
        from memmachine.common.vector_graph_store.neo4j_migration import (
            backfill_entity_type_labels,
        )

        driver = AsyncMock()
        # execute_query returns (records, summary, keys) where records[0]["updated"] = 5
        count_record = MagicMock()
        count_record.__getitem__ = lambda self, k: 5
        driver.execute_query.return_value = ([count_record], MagicMock(), MagicMock())

        result = await backfill_entity_type_labels(
            driver,
            {"my_collection": {"Person": ["uid1", "uid2"]}},
        )
        assert result == 5


class TestMigrationHelpers:
    def test_pick_canonical_node(self) -> None:
        from memmachine.common.vector_graph_store.neo4j_migration import (
            _pick_canonical_node,
        )

        node_few = MagicMock()
        node_few.items = lambda: {"a": 1}.items()

        node_many = MagicMock()
        node_many.items = lambda: {"a": 1, "b": 2, "c": 3}.items()

        result = _pick_canonical_node([node_few, node_many])
        assert result is node_many

    def test_pick_canonical_node_single(self) -> None:
        from memmachine.common.vector_graph_store.neo4j_migration import (
            _pick_canonical_node,
        )

        node = MagicMock()
        node.items = lambda: {"a": 1}.items()
        assert _pick_canonical_node([node]) is node
