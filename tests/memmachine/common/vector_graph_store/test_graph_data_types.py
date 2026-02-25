"""Unit tests for graph traversal and knowledge-graph data types."""

from datetime import UTC, datetime

import pytest

from memmachine.common.vector_graph_store.data_types import (
    DuplicateProposal,
    DuplicateResolutionStrategy,
    GraphFilter,
    MultiHopResult,
    Node,
    TraversalDirection,
)
from memmachine.semantic_memory.storage.feature_relationship_types import (
    ContradictionPair,
    FeatureRelationship,
    FeatureRelationshipType,
    RelationshipDirection,
    SupersessionChain,
)

# ---------------------------------------------------------------------------
# Node entity_types
# ---------------------------------------------------------------------------


class TestNodeEntityTypes:
    def test_default_entity_types_empty(self) -> None:
        node = Node(uid="n1")
        assert node.entity_types == []

    def test_entity_types_set(self) -> None:
        node = Node(uid="n1", entity_types=["Person", "Concept"])
        assert node.entity_types == ["Person", "Concept"]

    def test_equality_ignores_entity_types(self) -> None:
        """entity_types is not part of __eq__ (property/embedding based)."""
        a = Node(uid="n1", entity_types=["Person"])
        b = Node(uid="n1", entity_types=["Event"])
        # Both have same uid, properties, embeddings -> equal
        assert a == b

    def test_hash_unchanged(self) -> None:
        node = Node(uid="n1", entity_types=["Person"])
        assert hash(node) == hash("n1")


# ---------------------------------------------------------------------------
# GraphFilter
# ---------------------------------------------------------------------------


class TestGraphFilter:
    def test_defaults(self) -> None:
        gf = GraphFilter(anchor_node_uid="a1", anchor_collection="col")
        assert gf.max_hops == 1
        assert gf.direction == TraversalDirection.BOTH
        assert gf.relation_types is None

    def test_custom_values(self) -> None:
        gf = GraphFilter(
            anchor_node_uid="a1",
            anchor_collection="col",
            relation_types=["RELATES_TO", "OWNS"],
            max_hops=3,
            direction=TraversalDirection.OUTGOING,
        )
        assert gf.relation_types == ["RELATES_TO", "OWNS"]
        assert gf.max_hops == 3
        assert gf.direction == TraversalDirection.OUTGOING


# ---------------------------------------------------------------------------
# TraversalDirection
# ---------------------------------------------------------------------------


class TestTraversalDirection:
    def test_values(self) -> None:
        assert TraversalDirection.OUTGOING.value == "outgoing"
        assert TraversalDirection.INCOMING.value == "incoming"
        assert TraversalDirection.BOTH.value == "both"


# ---------------------------------------------------------------------------
# MultiHopResult
# ---------------------------------------------------------------------------


class TestMultiHopResult:
    def test_construction(self) -> None:
        node = Node(uid="n1")
        result = MultiHopResult(node=node, hop_distance=2, score=0.49)
        assert result.node.uid == "n1"
        assert result.hop_distance == 2
        assert result.score == pytest.approx(0.49)


# ---------------------------------------------------------------------------
# DuplicateProposal
# ---------------------------------------------------------------------------


class TestDuplicateProposal:
    def test_construction(self) -> None:
        now = datetime.now(UTC)
        dp = DuplicateProposal(
            node_uid_a="a",
            node_uid_b="b",
            embedding_similarity=0.97,
            property_similarity=0.85,
            detected_at=now,
        )
        assert dp.node_uid_a == "a"
        assert dp.auto_merged is False

    def test_auto_merged_flag(self) -> None:
        dp = DuplicateProposal(
            node_uid_a="a",
            node_uid_b="b",
            embedding_similarity=0.97,
            property_similarity=0.85,
            detected_at=datetime.now(UTC),
            auto_merged=True,
        )
        assert dp.auto_merged is True


class TestDuplicateResolutionStrategy:
    def test_values(self) -> None:
        assert DuplicateResolutionStrategy.MERGE.value == "merge"
        assert DuplicateResolutionStrategy.DISMISS.value == "dismiss"


# ---------------------------------------------------------------------------
# FeatureRelationshipType
# ---------------------------------------------------------------------------


class TestFeatureRelationshipType:
    def test_all_values(self) -> None:
        assert FeatureRelationshipType.CONTRADICTS.value == "CONTRADICTS"
        assert FeatureRelationshipType.IMPLIES.value == "IMPLIES"
        assert FeatureRelationshipType.RELATED_TO.value == "RELATED_TO"
        assert FeatureRelationshipType.SUPERSEDES.value == "SUPERSEDES"


# ---------------------------------------------------------------------------
# RelationshipDirection
# ---------------------------------------------------------------------------


class TestRelationshipDirection:
    def test_values(self) -> None:
        assert RelationshipDirection.OUTGOING.value == "outgoing"
        assert RelationshipDirection.INCOMING.value == "incoming"
        assert RelationshipDirection.BOTH.value == "both"


# ---------------------------------------------------------------------------
# FeatureRelationship
# ---------------------------------------------------------------------------


class TestFeatureRelationship:
    def test_valid_construction(self) -> None:
        now = datetime.now(UTC)
        fr = FeatureRelationship(
            source_id="f1",
            target_id="f2",
            relationship_type=FeatureRelationshipType.CONTRADICTS,
            confidence=0.95,
            detected_at=now,
            source="llm",
        )
        assert fr.source_id == "f1"
        assert fr.target_id == "f2"
        assert fr.confidence == 0.95

    def test_confidence_below_zero_rejected(self) -> None:
        with pytest.raises(ValueError, match="confidence must be between"):
            FeatureRelationship(
                source_id="f1",
                target_id="f2",
                relationship_type=FeatureRelationshipType.IMPLIES,
                confidence=-0.1,
                detected_at=datetime.now(UTC),
                source="rule",
            )

    def test_confidence_above_one_rejected(self) -> None:
        with pytest.raises(ValueError, match="confidence must be between"):
            FeatureRelationship(
                source_id="f1",
                target_id="f2",
                relationship_type=FeatureRelationshipType.IMPLIES,
                confidence=1.5,
                detected_at=datetime.now(UTC),
                source="rule",
            )

    def test_invalid_source_rejected(self) -> None:
        with pytest.raises(ValueError, match="source must be"):
            FeatureRelationship(
                source_id="f1",
                target_id="f2",
                relationship_type=FeatureRelationshipType.RELATED_TO,
                confidence=0.5,
                detected_at=datetime.now(UTC),
                source="unknown",
            )

    def test_boundary_confidence_values(self) -> None:
        """0.0 and 1.0 are both valid."""
        now = datetime.now(UTC)
        fr_zero = FeatureRelationship(
            source_id="f1",
            target_id="f2",
            relationship_type=FeatureRelationshipType.RELATED_TO,
            confidence=0.0,
            detected_at=now,
            source="manual",
        )
        assert fr_zero.confidence == 0.0

        fr_one = FeatureRelationship(
            source_id="f1",
            target_id="f2",
            relationship_type=FeatureRelationshipType.SUPERSEDES,
            confidence=1.0,
            detected_at=now,
            source="rule",
        )
        assert fr_one.confidence == 1.0


# ---------------------------------------------------------------------------
# ContradictionPair
# ---------------------------------------------------------------------------


class TestContradictionPair:
    def test_construction(self) -> None:
        cp = ContradictionPair(
            feature_id_a="f1",
            feature_id_b="f2",
            confidence=0.9,
            detected_at=datetime.now(UTC),
            source="llm",
        )
        assert cp.feature_id_a == "f1"
        assert cp.feature_id_b == "f2"


# ---------------------------------------------------------------------------
# SupersessionChain
# ---------------------------------------------------------------------------


class TestSupersessionChain:
    def test_single_feature(self) -> None:
        sc = SupersessionChain(current="f1", chain=["f1"])
        assert sc.current == "f1"
        assert len(sc.chain) == 1

    def test_chain_ordering(self) -> None:
        sc = SupersessionChain(current="f3", chain=["f3", "f2", "f1"])
        assert sc.chain[0] == "f3"  # newest
        assert sc.chain[-1] == "f1"  # oldest
