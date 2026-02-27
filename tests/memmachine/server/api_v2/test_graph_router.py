"""Tests for the graph REST API router endpoints (Groups 10-12)."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock, PropertyMock

import pytest
from fastapi.testclient import TestClient

from memmachine.common.vector_graph_store.graph_traversal_store import (
    GraphTraversalStore,
)
from memmachine.common.vector_graph_store.neo4j_vector_graph_store import (
    Neo4jVectorGraphStore,
)
from memmachine.semantic_memory.storage.semantic_relationship_storage import (
    SemanticRelationshipStorage,
)
from memmachine.server.api_v2.service import get_memmachine
from memmachine.server.app import MemMachineAPI

# ---------------------------------------------------------------------------
# Helpers — lightweight stand-ins for store-layer dataclasses
# ---------------------------------------------------------------------------


@dataclass(kw_only=True)
class _FakeNode:
    uid: str
    properties: dict = field(default_factory=dict)
    entity_types: list[str] = field(default_factory=list)


@dataclass(kw_only=True)
class _FakeMultiHopResult:
    node: _FakeNode
    hop_distance: int
    score: float


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_memmachine():
    mm = AsyncMock()
    # resource_manager needs to be a real-ish object for attribute access
    rm = MagicMock()
    type(mm).resource_manager = PropertyMock(return_value=rm)
    rm.config.episodic_memory.long_term_memory.vector_graph_store = "my_storage_id"
    rm.config.semantic_memory.embedding_model = "ollama_embedder"
    return mm


@pytest.fixture
def client(mock_memmachine):
    app = MemMachineAPI()
    app.dependency_overrides[get_memmachine] = lambda: mock_memmachine
    with TestClient(app) as c:
        yield c
    app.dependency_overrides = {}


# ---------------------------------------------------------------------------
# 10.4 — Multi-hop search endpoint tests
# ---------------------------------------------------------------------------


def test_multi_hop_search_success(client, mock_memmachine):
    """Multi-hop search returns traversal results when store supports it."""
    fake_results = [
        _FakeMultiHopResult(
            node=_FakeNode(
                uid="n1",
                properties={"name": "Alice"},
                entity_types=["Person"],
            ),
            hop_distance=1,
            score=0.7,
        ),
        _FakeMultiHopResult(
            node=_FakeNode(uid="n2", properties={}, entity_types=[]),
            hop_distance=2,
            score=0.49,
        ),
    ]

    mock_store = AsyncMock(spec=GraphTraversalStore)
    mock_store.search_multi_hop_nodes = AsyncMock(return_value=fake_results)
    mock_memmachine.resource_manager.get_vector_graph_store = AsyncMock(
        return_value=mock_store,
    )

    resp = client.post(
        "/api/v2/memories/graph/search/multi-hop",
        json={
            "collection": "universal/universal",
            "node_uid": "anchor-1",
            "max_hops": 3,
        },
    )

    assert resp.status_code == 200
    body = resp.json()
    assert len(body["results"]) == 2
    assert body["results"][0]["uid"] == "n1"
    assert body["results"][0]["hop_distance"] == 1
    assert body["results"][0]["entity_types"] == ["Person"]
    assert body["results"][1]["uid"] == "n2"


def test_multi_hop_search_501_when_unsupported(client, mock_memmachine):
    """Multi-hop search returns 501 when store doesn't support traversal."""
    # Return a plain mock that does NOT satisfy GraphTraversalStore isinstance.
    mock_store = AsyncMock()
    mock_memmachine.resource_manager.get_vector_graph_store = AsyncMock(
        return_value=mock_store,
    )

    resp = client.post(
        "/api/v2/memories/graph/search/multi-hop",
        json={
            "collection": "universal/universal",
            "node_uid": "anchor-1",
        },
    )

    assert resp.status_code == 501
    assert "not supported" in resp.json()["detail"].lower()


def test_multi_hop_search_passes_target_collections(client, mock_memmachine):
    """target_collections from request body is forwarded to the store method."""
    mock_store = AsyncMock(spec=GraphTraversalStore)
    mock_store.search_multi_hop_nodes = AsyncMock(return_value=[])
    mock_memmachine.resource_manager.get_vector_graph_store = AsyncMock(
        return_value=mock_store,
    )

    resp = client.post(
        "/api/v2/memories/graph/search/multi-hop",
        json={
            "collection": "universal/universal",
            "node_uid": "anchor-1",
            "max_hops": 2,
            "target_collections": ["Episode", "Feature"],
        },
    )

    assert resp.status_code == 200
    call_kwargs = mock_store.search_multi_hop_nodes.call_args[1]
    assert call_kwargs["target_collections"] == ["Episode", "Feature"]


def test_multi_hop_search_without_target_collections(client, mock_memmachine):
    """Without target_collections, None is passed to the store (backward compat)."""
    mock_store = AsyncMock(spec=GraphTraversalStore)
    mock_store.search_multi_hop_nodes = AsyncMock(return_value=[])
    mock_memmachine.resource_manager.get_vector_graph_store = AsyncMock(
        return_value=mock_store,
    )

    resp = client.post(
        "/api/v2/memories/graph/search/multi-hop",
        json={
            "collection": "universal/universal",
            "node_uid": "anchor-1",
        },
    )

    assert resp.status_code == 200
    call_kwargs = mock_store.search_multi_hop_nodes.call_args[1]
    assert call_kwargs["target_collections"] is None


# ---------------------------------------------------------------------------
# 10.5 -- Graph-filtered search endpoint tests
# ---------------------------------------------------------------------------


def test_graph_filtered_search_success(client, mock_memmachine):
    """Filtered search returns results when store supports graph filtering."""
    fake_nodes = [
        _FakeNode(
            uid="n1",
            properties={"text": "hello"},
            entity_types=["Concept"],
        ),
    ]

    mock_store = AsyncMock(spec=GraphTraversalStore)
    mock_store.search_graph_filtered_similar_nodes = AsyncMock(
        return_value=fake_nodes,
    )
    mock_memmachine.resource_manager.get_vector_graph_store = AsyncMock(
        return_value=mock_store,
    )

    mock_embedder = AsyncMock()
    mock_embedder.embed = AsyncMock(return_value=[0.1, 0.2, 0.3])
    mock_embedder.model_id = "test-model"
    mock_embedder.dimensions = 3
    mock_memmachine.resource_manager.get_embedder = AsyncMock(
        return_value=mock_embedder,
    )

    resp = client.post(
        "/api/v2/memories/graph/search/filtered",
        json={
            "collection": "universal/universal",
            "query": "hello world",
            "anchor_node_uid": "anchor-1",
        },
    )

    assert resp.status_code == 200
    body = resp.json()
    assert len(body["results"]) == 1
    assert body["results"][0]["uid"] == "n1"
    assert body["results"][0]["entity_types"] == ["Concept"]


def test_graph_filtered_search_no_anchor_fallback(client, mock_memmachine):
    """Filtered search without anchor falls back to standard similarity search."""
    fake_nodes = [
        _FakeNode(uid="n1", properties={}, entity_types=[]),
    ]

    mock_store = AsyncMock()
    mock_store.search_similar_nodes = AsyncMock(return_value=fake_nodes)
    mock_memmachine.resource_manager.get_vector_graph_store = AsyncMock(
        return_value=mock_store,
    )

    mock_embedder = AsyncMock()
    mock_embedder.embed = AsyncMock(return_value=[0.1, 0.2])
    mock_embedder.model_id = "test"
    mock_embedder.dimensions = 2
    mock_memmachine.resource_manager.get_embedder = AsyncMock(
        return_value=mock_embedder,
    )

    resp = client.post(
        "/api/v2/memories/graph/search/filtered",
        json={
            "collection": "universal/universal",
            "query": "test query",
            # No anchor_node_uid → fallback
        },
    )

    assert resp.status_code == 200
    mock_store.search_similar_nodes.assert_awaited_once()


# ---------------------------------------------------------------------------
# 10.6 — entity_types passthrough in main search endpoint
# ---------------------------------------------------------------------------


def test_entity_types_passed_to_query_search(client, mock_memmachine):
    """entity_types from SearchMemoriesSpec flows through to query_search."""
    # Set up a search result so the response can be built.
    mock_memmachine.query_search.return_value = MagicMock(
        episodic_memory=None,
        semantic_memory=None,
    )

    resp = client.post(
        "/api/v2/memories/search",
        json={
            "query": "favorite food",
            "entity_types": ["Person", "Preference"],
        },
    )

    assert resp.status_code == 200
    mock_memmachine.query_search.assert_awaited_once()
    call_kwargs = mock_memmachine.query_search.call_args[1]
    assert call_kwargs["entity_types"] == ["Person", "Preference"]


def test_entity_types_none_by_default(client, mock_memmachine):
    """entity_types defaults to None when not provided."""
    mock_memmachine.query_search.return_value = MagicMock(
        episodic_memory=None,
        semantic_memory=None,
    )

    resp = client.post(
        "/api/v2/memories/search",
        json={"query": "anything"},
    )

    assert resp.status_code == 200
    call_kwargs = mock_memmachine.query_search.call_args[1]
    assert call_kwargs["entity_types"] is None


# ---------------------------------------------------------------------------
# Helpers for relationship / dedup / analytics tests
# ---------------------------------------------------------------------------


def _wire_semantic_storage(mock_memmachine, mock_storage):
    """Wire a mock SemanticStorage through the resource_manager chain."""
    mock_sem_mgr = AsyncMock()
    mock_sem_mgr.get_semantic_storage = AsyncMock(return_value=mock_storage)
    mock_memmachine.resource_manager.get_semantic_manager = AsyncMock(
        return_value=mock_sem_mgr,
    )


@dataclass(kw_only=True)
class _FakeFeatureRelationship:
    source_id: str
    target_id: str
    relationship_type: object  # FeatureRelationshipType enum value
    confidence: float
    detected_at: datetime
    source: str


@dataclass(kw_only=True)
class _FakeContradictionPair:
    feature_id_a: str
    feature_id_b: str
    confidence: float
    detected_at: datetime
    source: str


@dataclass(kw_only=True)
class _FakeDuplicateProposal:
    node_uid_a: str
    node_uid_b: str
    embedding_similarity: float
    property_similarity: float
    detected_at: datetime
    auto_merged: bool = False


# ---------------------------------------------------------------------------
# 11.5 — Relationship endpoint tests
# ---------------------------------------------------------------------------


class TestCreateRelationship:
    """Tests for POST /memories/graph/relationships."""

    def test_create_relationship_success(self, client, mock_memmachine):
        """Create relationship returns the created relationship."""
        mock_storage = AsyncMock(spec=SemanticRelationshipStorage)
        mock_storage.add_feature_relationship = AsyncMock(return_value=None)
        _wire_semantic_storage(mock_memmachine, mock_storage)

        resp = client.post(
            "/api/v2/memories/graph/relationships",
            json={
                "source_id": "feat-1",
                "target_id": "feat-2",
                "relationship_type": "CONTRADICTS",
                "confidence": 0.95,
                "source": "llm",
            },
        )

        assert resp.status_code == 200
        body = resp.json()
        assert body["source_id"] == "feat-1"
        assert body["target_id"] == "feat-2"
        assert body["relationship_type"] == "CONTRADICTS"
        assert body["confidence"] == 0.95
        assert body["source"] == "llm"
        mock_storage.add_feature_relationship.assert_awaited_once()

    def test_create_relationship_501_when_unsupported(self, client, mock_memmachine):
        """Create relationship returns 501 when storage doesn't support it."""
        mock_storage = AsyncMock()  # plain mock → fails isinstance
        _wire_semantic_storage(mock_memmachine, mock_storage)

        resp = client.post(
            "/api/v2/memories/graph/relationships",
            json={
                "source_id": "feat-1",
                "target_id": "feat-2",
                "relationship_type": "RELATED_TO",
                "confidence": 0.8,
            },
        )

        assert resp.status_code == 501
        assert "not supported" in resp.json()["detail"].lower()


class TestGetRelationships:
    """Tests for POST /memories/graph/relationships/get."""

    def test_get_relationships_success(self, client, mock_memmachine):
        """Get relationships returns matching relationships."""
        from memmachine.semantic_memory.storage.feature_relationship_types import (
            FeatureRelationshipType,
        )

        fake_rels = [
            _FakeFeatureRelationship(
                source_id="f1",
                target_id="f2",
                relationship_type=FeatureRelationshipType.RELATED_TO,
                confidence=0.9,
                detected_at=datetime(2026, 1, 15, tzinfo=UTC),
                source="rule",
            ),
        ]

        mock_storage = AsyncMock(spec=SemanticRelationshipStorage)
        mock_storage.get_feature_relationships = AsyncMock(return_value=fake_rels)
        _wire_semantic_storage(mock_memmachine, mock_storage)

        resp = client.post(
            "/api/v2/memories/graph/relationships/get",
            json={
                "feature_id": "f1",
                "relationship_type": "RELATED_TO",
                "direction": "outgoing",
            },
        )

        assert resp.status_code == 200
        body = resp.json()
        assert len(body["relationships"]) == 1
        assert body["relationships"][0]["source_id"] == "f1"
        assert body["relationships"][0]["relationship_type"] == "RELATED_TO"

    def test_get_relationships_501_when_unsupported(self, client, mock_memmachine):
        """Get relationships returns 501 when storage doesn't support it."""
        mock_storage = AsyncMock()
        _wire_semantic_storage(mock_memmachine, mock_storage)

        resp = client.post(
            "/api/v2/memories/graph/relationships/get",
            json={"feature_id": "f1"},
        )

        assert resp.status_code == 501


class TestDeleteRelationship:
    """Tests for POST /memories/graph/relationships/delete."""

    def test_delete_relationship_success(self, client, mock_memmachine):
        """Delete relationship returns success."""
        mock_storage = AsyncMock(spec=SemanticRelationshipStorage)
        mock_storage.delete_feature_relationships = AsyncMock(return_value=None)
        _wire_semantic_storage(mock_memmachine, mock_storage)

        resp = client.post(
            "/api/v2/memories/graph/relationships/delete",
            json={
                "source_id": "f1",
                "target_id": "f2",
                "relationship_type": "SUPERSEDES",
            },
        )

        assert resp.status_code == 200
        assert resp.json()["status"] == "deleted"
        mock_storage.delete_feature_relationships.assert_awaited_once()

    def test_delete_relationship_501_when_unsupported(self, client, mock_memmachine):
        """Delete relationship returns 501 when storage doesn't support it."""
        mock_storage = AsyncMock()
        _wire_semantic_storage(mock_memmachine, mock_storage)

        resp = client.post(
            "/api/v2/memories/graph/relationships/delete",
            json={
                "source_id": "f1",
                "target_id": "f2",
                "relationship_type": "IMPLIES",
            },
        )

        assert resp.status_code == 501


class TestFindContradictions:
    """Tests for POST /memories/graph/contradictions."""

    def test_find_contradictions_success(self, client, mock_memmachine):
        """Find contradictions returns contradiction pairs."""
        fake_pairs = [
            _FakeContradictionPair(
                feature_id_a="f1",
                feature_id_b="f2",
                confidence=0.92,
                detected_at=datetime(2026, 2, 1, tzinfo=UTC),
                source="llm",
            ),
        ]

        mock_storage = AsyncMock(spec=SemanticRelationshipStorage)
        mock_storage.find_contradictions = AsyncMock(return_value=fake_pairs)
        _wire_semantic_storage(mock_memmachine, mock_storage)

        resp = client.post(
            "/api/v2/memories/graph/contradictions",
            json={"set_id": "my-set"},
        )

        assert resp.status_code == 200
        body = resp.json()
        assert len(body["contradictions"]) == 1
        assert body["contradictions"][0]["feature_id_a"] == "f1"
        assert body["contradictions"][0]["feature_id_b"] == "f2"

    def test_find_contradictions_501_when_unsupported(self, client, mock_memmachine):
        """Find contradictions returns 501 when storage doesn't support it."""
        mock_storage = AsyncMock()
        _wire_semantic_storage(mock_memmachine, mock_storage)

        resp = client.post(
            "/api/v2/memories/graph/contradictions",
            json={"set_id": "my-set"},
        )

        assert resp.status_code == 501


# ---------------------------------------------------------------------------
# 12.5 — Dedup endpoint tests
# ---------------------------------------------------------------------------


class TestDedupProposals:
    """Tests for POST /memories/graph/dedup/proposals."""

    def test_dedup_proposals_success(self, client, mock_memmachine):
        """Get dedup proposals returns proposals list."""
        fake_proposals = [
            _FakeDuplicateProposal(
                node_uid_a="a1",
                node_uid_b="b1",
                embedding_similarity=0.97,
                property_similarity=0.85,
                detected_at=datetime(2026, 2, 10, tzinfo=UTC),
                auto_merged=False,
            ),
        ]

        mock_store = AsyncMock(spec=Neo4jVectorGraphStore)
        mock_store.get_duplicate_proposals = AsyncMock(return_value=fake_proposals)
        mock_memmachine.resource_manager.get_vector_graph_store = AsyncMock(
            return_value=mock_store,
        )

        resp = client.post(
            "/api/v2/memories/graph/dedup/proposals",
            json={"collection": "universal/universal"},
        )

        assert resp.status_code == 200
        body = resp.json()
        assert len(body["proposals"]) == 1
        assert body["proposals"][0]["node_uid_a"] == "a1"
        assert body["proposals"][0]["embedding_similarity"] == 0.97

    def test_dedup_proposals_501_when_unsupported(self, client, mock_memmachine):
        """Dedup proposals returns 501 when store doesn't support it."""
        mock_store = AsyncMock()
        mock_memmachine.resource_manager.get_vector_graph_store = AsyncMock(
            return_value=mock_store,
        )

        resp = client.post(
            "/api/v2/memories/graph/dedup/proposals",
            json={"collection": "universal/universal"},
        )

        assert resp.status_code == 501


class TestDedupResolve:
    """Tests for POST /memories/graph/dedup/resolve."""

    def test_resolve_merge_success(self, client, mock_memmachine):
        """Resolve duplicates with merge strategy returns count."""
        mock_store = AsyncMock(spec=Neo4jVectorGraphStore)
        mock_store.resolve_duplicates = AsyncMock(return_value=None)
        mock_memmachine.resource_manager.get_vector_graph_store = AsyncMock(
            return_value=mock_store,
        )

        resp = client.post(
            "/api/v2/memories/graph/dedup/resolve",
            json={
                "collection": "universal/universal",
                "pairs": [
                    {
                        "node_uid_a": "a1",
                        "node_uid_b": "b1",
                        "strategy": "merge",
                    },
                    {
                        "node_uid_a": "a2",
                        "node_uid_b": "b2",
                        "strategy": "dismiss",
                    },
                ],
            },
        )

        assert resp.status_code == 200
        assert resp.json()["resolved"] == 2
        mock_store.resolve_duplicates.assert_awaited_once()

    def test_resolve_501_when_unsupported(self, client, mock_memmachine):
        """Resolve duplicates returns 501 when store doesn't support it."""
        mock_store = AsyncMock()
        mock_memmachine.resource_manager.get_vector_graph_store = AsyncMock(
            return_value=mock_store,
        )

        resp = client.post(
            "/api/v2/memories/graph/dedup/resolve",
            json={
                "collection": "universal/universal",
                "pairs": [
                    {"node_uid_a": "a1", "node_uid_b": "b1", "strategy": "merge"},
                ],
            },
        )

        assert resp.status_code == 501


# ---------------------------------------------------------------------------
# 12.6 — Analytics endpoint tests
# ---------------------------------------------------------------------------


class TestPageRank:
    """Tests for POST /memories/graph/analytics/pagerank."""

    def test_pagerank_success(self, client, mock_memmachine):
        """PageRank returns scores when GDS is available."""
        mock_store = AsyncMock(spec=Neo4jVectorGraphStore)
        mock_store.is_gds_available = AsyncMock(return_value=True)
        mock_store.compute_pagerank = AsyncMock(
            return_value=[("n1", 0.42), ("n2", 0.31)],
        )
        mock_memmachine.resource_manager.get_vector_graph_store = AsyncMock(
            return_value=mock_store,
        )

        resp = client.post(
            "/api/v2/memories/graph/analytics/pagerank",
            json={"collection": "universal/universal"},
        )

        assert resp.status_code == 200
        body = resp.json()
        assert len(body["scores"]) == 2
        assert body["scores"][0]["node_uid"] == "n1"
        assert body["scores"][0]["score"] == 0.42

    def test_pagerank_passes_relation_types_and_write_property(
        self, client, mock_memmachine
    ):
        """PageRank endpoint passes relation_types and write_property to store."""
        mock_store = AsyncMock(spec=Neo4jVectorGraphStore)
        mock_store.is_gds_available = AsyncMock(return_value=True)
        mock_store.compute_pagerank = AsyncMock(
            return_value=[("n1", 0.42)],
        )
        mock_memmachine.resource_manager.get_vector_graph_store = AsyncMock(
            return_value=mock_store,
        )

        resp = client.post(
            "/api/v2/memories/graph/analytics/pagerank",
            json={
                "collection": "universal/universal",
                "relation_types": ["RELATES_TO"],
                "write_back": True,
                "write_property": "importance_score",
            },
        )

        assert resp.status_code == 200
        mock_store.compute_pagerank.assert_awaited_once()
        call_kwargs = mock_store.compute_pagerank.call_args.kwargs
        assert call_kwargs["relation_types"] == ["RELATES_TO"]
        assert call_kwargs["write_property"] == "importance_score"

    def test_pagerank_without_new_params_uses_defaults(self, client, mock_memmachine):
        """PageRank endpoint works without relation_types/write_property (backward compat)."""
        mock_store = AsyncMock(spec=Neo4jVectorGraphStore)
        mock_store.is_gds_available = AsyncMock(return_value=True)
        mock_store.compute_pagerank = AsyncMock(
            return_value=[("n1", 0.42)],
        )
        mock_memmachine.resource_manager.get_vector_graph_store = AsyncMock(
            return_value=mock_store,
        )

        resp = client.post(
            "/api/v2/memories/graph/analytics/pagerank",
            json={"collection": "universal/universal"},
        )

        assert resp.status_code == 200
        call_kwargs = mock_store.compute_pagerank.call_args.kwargs
        assert call_kwargs["relation_types"] is None
        assert call_kwargs["write_property"] is None

    def test_pagerank_501_when_gds_unavailable(self, client, mock_memmachine):
        """PageRank returns 501 when GDS is not available."""
        mock_store = AsyncMock(spec=Neo4jVectorGraphStore)
        mock_store.is_gds_available = AsyncMock(return_value=False)
        mock_memmachine.resource_manager.get_vector_graph_store = AsyncMock(
            return_value=mock_store,
        )

        resp = client.post(
            "/api/v2/memories/graph/analytics/pagerank",
            json={"collection": "universal/universal"},
        )

        assert resp.status_code == 501
        assert "gds" in resp.json()["detail"].lower()

    def test_pagerank_501_when_store_unsupported(self, client, mock_memmachine):
        """PageRank returns 501 when store is not Neo4jVectorGraphStore."""
        mock_store = AsyncMock()
        mock_memmachine.resource_manager.get_vector_graph_store = AsyncMock(
            return_value=mock_store,
        )

        resp = client.post(
            "/api/v2/memories/graph/analytics/pagerank",
            json={"collection": "universal/universal"},
        )

        assert resp.status_code == 501


class TestCommunities:
    """Tests for POST /memories/graph/analytics/communities."""

    def test_communities_success(self, client, mock_memmachine):
        """Community detection returns groups when GDS is available."""
        mock_store = AsyncMock(spec=Neo4jVectorGraphStore)
        mock_store.is_gds_available = AsyncMock(return_value=True)
        mock_store.detect_communities = AsyncMock(
            return_value={0: ["n1", "n2"], 1: ["n3"]},
        )
        mock_memmachine.resource_manager.get_vector_graph_store = AsyncMock(
            return_value=mock_store,
        )

        resp = client.post(
            "/api/v2/memories/graph/analytics/communities",
            json={"collection": "universal/universal"},
        )

        assert resp.status_code == 200
        body = resp.json()
        assert len(body["communities"]) == 2
        # Find community with id 0
        c0 = next(c for c in body["communities"] if c["community_id"] == 0)
        assert set(c0["node_uids"]) == {"n1", "n2"}

    def test_communities_passes_relation_types_and_write_property(
        self, client, mock_memmachine
    ):
        """Communities endpoint passes relation_types and write_property to store."""
        mock_store = AsyncMock(spec=Neo4jVectorGraphStore)
        mock_store.is_gds_available = AsyncMock(return_value=True)
        mock_store.detect_communities = AsyncMock(
            return_value={0: ["n1", "n2"]},
        )
        mock_memmachine.resource_manager.get_vector_graph_store = AsyncMock(
            return_value=mock_store,
        )

        resp = client.post(
            "/api/v2/memories/graph/analytics/communities",
            json={
                "collection": "universal/universal",
                "relation_types": ["RELATES_TO", "EXTRACTED_FROM"],
                "write_back": True,
                "write_property": "topic_cluster",
            },
        )

        assert resp.status_code == 200
        mock_store.detect_communities.assert_awaited_once()
        call_kwargs = mock_store.detect_communities.call_args.kwargs
        assert call_kwargs["relation_types"] == ["RELATES_TO", "EXTRACTED_FROM"]
        assert call_kwargs["write_property"] == "topic_cluster"

    def test_communities_without_new_params_uses_defaults(
        self, client, mock_memmachine
    ):
        """Communities endpoint works without relation_types/write_property (backward compat)."""
        mock_store = AsyncMock(spec=Neo4jVectorGraphStore)
        mock_store.is_gds_available = AsyncMock(return_value=True)
        mock_store.detect_communities = AsyncMock(
            return_value={0: ["n1"]},
        )
        mock_memmachine.resource_manager.get_vector_graph_store = AsyncMock(
            return_value=mock_store,
        )

        resp = client.post(
            "/api/v2/memories/graph/analytics/communities",
            json={"collection": "universal/universal"},
        )

        assert resp.status_code == 200
        call_kwargs = mock_store.detect_communities.call_args.kwargs
        assert call_kwargs["relation_types"] is None
        assert call_kwargs["write_property"] is None

    def test_communities_501_when_gds_unavailable(self, client, mock_memmachine):
        """Community detection returns 501 when GDS is not available."""
        mock_store = AsyncMock(spec=Neo4jVectorGraphStore)
        mock_store.is_gds_available = AsyncMock(return_value=False)
        mock_memmachine.resource_manager.get_vector_graph_store = AsyncMock(
            return_value=mock_store,
        )

        resp = client.post(
            "/api/v2/memories/graph/analytics/communities",
            json={"collection": "universal/universal"},
        )

        assert resp.status_code == 501
        assert "gds" in resp.json()["detail"].lower()

    def test_communities_501_when_store_unsupported(self, client, mock_memmachine):
        """Community detection returns 501 when store is not Neo4j."""
        mock_store = AsyncMock()
        mock_memmachine.resource_manager.get_vector_graph_store = AsyncMock(
            return_value=mock_store,
        )

        resp = client.post(
            "/api/v2/memories/graph/analytics/communities",
            json={"collection": "universal/universal"},
        )

        assert resp.status_code == 501


class TestGraphStats:
    """Tests for POST /memories/graph/analytics/stats."""

    def test_graph_stats_success(self, client, mock_memmachine):
        """Stats endpoint returns node/edge counts and distributions."""
        from memmachine.common.vector_graph_store.neo4j_vector_graph_store import (
            GraphStatsResult,
        )

        mock_store = AsyncMock(spec=Neo4jVectorGraphStore)
        mock_store.graph_stats = AsyncMock(
            return_value=GraphStatsResult(
                node_count=100,
                edge_count=250,
                avg_degree=2.5,
                relationship_type_distribution={
                    "RELATES_TO": 200,
                    "EXTRACTED_FROM": 50,
                },
                entity_type_distribution={"Person": 60, "Organization": 40},
            ),
        )
        mock_memmachine.resource_manager.get_vector_graph_store = AsyncMock(
            return_value=mock_store,
        )

        resp = client.post(
            "/api/v2/memories/graph/analytics/stats",
            json={"collection": "universal/universal"},
        )

        assert resp.status_code == 200
        body = resp.json()
        assert body["node_count"] == 100
        assert body["edge_count"] == 250
        assert body["avg_degree"] == 2.5
        assert body["relationship_type_distribution"]["RELATES_TO"] == 200
        assert body["entity_type_distribution"]["Person"] == 60

    def test_graph_stats_501_when_store_unsupported(self, client, mock_memmachine):
        """Stats endpoint returns 501 when store is not Neo4j."""
        mock_store = AsyncMock()
        mock_memmachine.resource_manager.get_vector_graph_store = AsyncMock(
            return_value=mock_store,
        )

        resp = client.post(
            "/api/v2/memories/graph/analytics/stats",
            json={"collection": "universal/universal"},
        )

        assert resp.status_code == 501


# ---------------------------------------------------------------------------
# 12.6 — Shortest path endpoint tests
# ---------------------------------------------------------------------------


class TestShortestPath:
    """Tests for POST /memories/graph/search/shortest-path."""

    def test_shortest_path_success(self, client, mock_memmachine):
        """Shortest path endpoint returns path when found."""
        from memmachine.common.vector_graph_store.neo4j_vector_graph_store import (
            PathEdge,
            PathNode,
            ShortestPathResult,
        )

        mock_store = AsyncMock(spec=Neo4jVectorGraphStore)
        mock_store.shortest_path = AsyncMock(
            return_value=ShortestPathResult(
                path_length=2,
                nodes=[
                    PathNode(uid="alice", properties={"name": "Alice"}),
                    PathNode(uid="bob", properties={"name": "Bob"}),
                    PathNode(uid="charlie", properties={"name": "Charlie"}),
                ],
                edges=[
                    PathEdge(
                        source_uid="alice",
                        target_uid="bob",
                        type="KNOWS",
                        properties={},
                    ),
                    PathEdge(
                        source_uid="bob",
                        target_uid="charlie",
                        type="WORKS_WITH",
                        properties={},
                    ),
                ],
            ),
        )
        mock_memmachine.resource_manager.get_vector_graph_store = AsyncMock(
            return_value=mock_store,
        )

        resp = client.post(
            "/api/v2/memories/graph/search/shortest-path",
            json={
                "collection": "universal/universal",
                "source_uid": "alice",
                "target_uid": "charlie",
            },
        )

        assert resp.status_code == 200
        body = resp.json()
        assert body["path_length"] == 2
        assert len(body["nodes"]) == 3
        assert body["nodes"][0]["uid"] == "alice"
        assert len(body["edges"]) == 2
        assert body["edges"][0]["type"] == "KNOWS"

    def test_shortest_path_501_when_store_unsupported(self, client, mock_memmachine):
        """Shortest path endpoint returns 501 when store is not Neo4j."""
        mock_store = AsyncMock()
        mock_memmachine.resource_manager.get_vector_graph_store = AsyncMock(
            return_value=mock_store,
        )

        resp = client.post(
            "/api/v2/memories/graph/search/shortest-path",
            json={
                "collection": "universal/universal",
                "source_uid": "a",
                "target_uid": "b",
            },
        )

        assert resp.status_code == 501


# ---------------------------------------------------------------------------
# 12.7 — Degree centrality endpoint tests
# ---------------------------------------------------------------------------


class TestDegreeCentrality:
    """Tests for POST /memories/graph/analytics/degree-centrality."""

    def test_degree_centrality_success(self, client, mock_memmachine):
        """Degree centrality endpoint returns node metrics."""
        from memmachine.common.vector_graph_store.neo4j_vector_graph_store import (
            DegreeCentralityResult,
        )

        mock_store = AsyncMock(spec=Neo4jVectorGraphStore)
        mock_store.degree_centrality = AsyncMock(
            return_value=[
                DegreeCentralityResult(
                    uid="hub", in_degree=10, out_degree=5, total_degree=15
                ),
                DegreeCentralityResult(
                    uid="leaf", in_degree=1, out_degree=0, total_degree=1
                ),
            ],
        )
        mock_memmachine.resource_manager.get_vector_graph_store = AsyncMock(
            return_value=mock_store,
        )

        resp = client.post(
            "/api/v2/memories/graph/analytics/degree-centrality",
            json={"collection": "universal/universal"},
        )

        assert resp.status_code == 200
        body = resp.json()
        assert len(body["nodes"]) == 2
        assert body["nodes"][0]["uid"] == "hub"
        assert body["nodes"][0]["in_degree"] == 10
        assert body["nodes"][0]["out_degree"] == 5
        assert body["nodes"][0]["total_degree"] == 15

    def test_degree_centrality_501_when_store_unsupported(
        self, client, mock_memmachine
    ):
        """Degree centrality endpoint returns 501 when store is not Neo4j."""
        mock_store = AsyncMock()
        mock_memmachine.resource_manager.get_vector_graph_store = AsyncMock(
            return_value=mock_store,
        )

        resp = client.post(
            "/api/v2/memories/graph/analytics/degree-centrality",
            json={"collection": "universal/universal"},
        )

        assert resp.status_code == 501


# ---------------------------------------------------------------------------
# 12.8 — Subgraph extraction endpoint tests
# ---------------------------------------------------------------------------


class TestSubgraphExtraction:
    """Tests for POST /memories/graph/search/subgraph."""

    def test_subgraph_success(self, client, mock_memmachine):
        """Subgraph endpoint returns nodes and edges."""
        from memmachine.common.vector_graph_store.neo4j_vector_graph_store import (
            SubgraphEdge,
            SubgraphNode,
            SubgraphResult,
        )

        mock_store = AsyncMock(spec=Neo4jVectorGraphStore)
        mock_store.extract_subgraph = AsyncMock(
            return_value=SubgraphResult(
                nodes=[
                    SubgraphNode(uid="alice", properties={"name": "Alice"}),
                    SubgraphNode(uid="bob", properties={"name": "Bob"}),
                ],
                edges=[
                    SubgraphEdge(
                        source_uid="alice",
                        target_uid="bob",
                        type="KNOWS",
                        properties={},
                    ),
                ],
            ),
        )
        mock_memmachine.resource_manager.get_vector_graph_store = AsyncMock(
            return_value=mock_store,
        )

        resp = client.post(
            "/api/v2/memories/graph/search/subgraph",
            json={
                "collection": "universal/universal",
                "anchor_uid": "alice",
            },
        )

        assert resp.status_code == 200
        body = resp.json()
        assert len(body["nodes"]) == 2
        assert body["nodes"][0]["uid"] == "alice"
        assert len(body["edges"]) == 1
        assert body["edges"][0]["type"] == "KNOWS"

    def test_subgraph_501_when_store_unsupported(self, client, mock_memmachine):
        """Subgraph endpoint returns 501 when store is not Neo4j."""
        mock_store = AsyncMock()
        mock_memmachine.resource_manager.get_vector_graph_store = AsyncMock(
            return_value=mock_store,
        )

        resp = client.post(
            "/api/v2/memories/graph/search/subgraph",
            json={
                "collection": "universal/universal",
                "anchor_uid": "alice",
            },
        )

        assert resp.status_code == 501


# ---------------------------------------------------------------------------
# 12.9 — Betweenness centrality endpoint tests
# ---------------------------------------------------------------------------


class TestBetweennessCentrality:
    """Tests for POST /memories/graph/analytics/betweenness-centrality."""

    def test_betweenness_centrality_success(self, client, mock_memmachine):
        """Betweenness centrality endpoint returns scores."""
        mock_store = AsyncMock(spec=Neo4jVectorGraphStore)
        mock_store.is_gds_available = AsyncMock(return_value=True)
        mock_store.betweenness_centrality = AsyncMock(
            return_value=[("bridge", 15.0), ("leaf", 0.0)],
        )
        mock_memmachine.resource_manager.get_vector_graph_store = AsyncMock(
            return_value=mock_store,
        )

        resp = client.post(
            "/api/v2/memories/graph/analytics/betweenness-centrality",
            json={"collection": "universal/universal"},
        )

        assert resp.status_code == 200
        body = resp.json()
        assert len(body["scores"]) == 2
        assert body["scores"][0]["node_uid"] == "bridge"
        assert body["scores"][0]["score"] == 15.0

    def test_betweenness_501_when_store_unsupported(self, client, mock_memmachine):
        """Betweenness endpoint returns 501 when store is not Neo4j."""
        mock_store = AsyncMock()
        mock_memmachine.resource_manager.get_vector_graph_store = AsyncMock(
            return_value=mock_store,
        )

        resp = client.post(
            "/api/v2/memories/graph/analytics/betweenness-centrality",
            json={"collection": "universal/universal"},
        )

        assert resp.status_code == 501

    def test_betweenness_501_when_gds_unavailable(self, client, mock_memmachine):
        """Betweenness endpoint returns 501 when GDS is not available."""
        mock_store = AsyncMock(spec=Neo4jVectorGraphStore)
        mock_store.is_gds_available = AsyncMock(return_value=False)
        mock_memmachine.resource_manager.get_vector_graph_store = AsyncMock(
            return_value=mock_store,
        )

        resp = client.post(
            "/api/v2/memories/graph/analytics/betweenness-centrality",
            json={"collection": "universal/universal"},
        )

        assert resp.status_code == 501
