"""Integration tests for semantic feature relationships (task 6.8).

Covers: creating each relationship type, validation rejections, querying
with filters, deletion idempotency, cascade on feature delete,
contradiction detection, and supersession chain traversal.

Requires Docker and Neo4j (testcontainers). All tests are marked
``@pytest.mark.integration`` so they are skipped by default.
"""

import numpy as np
import pytest
import pytest_asyncio
from neo4j import AsyncGraphDatabase
from testcontainers.neo4j import Neo4jContainer

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
# Fixtures
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


@pytest_asyncio.fixture(scope="module")
async def storage(neo4j_driver):
    s = Neo4jSemanticStorage(driver=neo4j_driver)
    await s.startup()
    yield s


@pytest_asyncio.fixture(autouse=True)
async def db_cleanup(neo4j_driver):
    """Wipe Feature nodes between tests."""
    await neo4j_driver.execute_query("MATCH (f:Feature) DETACH DELETE f")
    await neo4j_driver.execute_query("MATCH (h:SetHistory) DELETE h")
    await neo4j_driver.execute_query("MATCH (s:SetEmbedding) DELETE s")
    yield


async def _create_feature(storage, set_id="test-set", feature="f", value="v"):
    """Helper to create a feature and return its ID."""
    embedding = np.array([1.0, 0.0, 0.0])
    return await storage.add_feature(
        set_id=set_id,
        category_name="test",
        feature=feature,
        value=value,
        tag="test-tag",
        embedding=embedding,
    )


# =====================================================================
# Creating Each Relationship Type
# =====================================================================


class TestCreateRelationships:
    """Tests for add_feature_relationship with each type."""

    @pytest.mark.asyncio
    async def test_contradicts(self, storage):
        fid_a = await _create_feature(storage, feature="color", value="red")
        fid_b = await _create_feature(storage, feature="color", value="blue")

        await storage.add_feature_relationship(
            source_id=fid_a,
            target_id=fid_b,
            relationship_type=FeatureRelationshipType.CONTRADICTS,
            confidence=0.9,
            source="llm",
        )

        rels = await storage.get_feature_relationships(fid_a)
        assert any(
            r.relationship_type == FeatureRelationshipType.CONTRADICTS for r in rels
        )

    @pytest.mark.asyncio
    async def test_implies(self, storage):
        fid_a = await _create_feature(storage, feature="role", value="manager")
        fid_b = await _create_feature(storage, feature="access", value="admin")

        await storage.add_feature_relationship(
            source_id=fid_a,
            target_id=fid_b,
            relationship_type=FeatureRelationshipType.IMPLIES,
            confidence=0.85,
            source="rule",
        )

        rels = await storage.get_feature_relationships(fid_a)
        assert any(r.relationship_type == FeatureRelationshipType.IMPLIES for r in rels)

    @pytest.mark.asyncio
    async def test_related_to(self, storage):
        fid_a = await _create_feature(storage, feature="hobby", value="tennis")
        fid_b = await _create_feature(storage, feature="sport", value="tennis")

        await storage.add_feature_relationship(
            source_id=fid_a,
            target_id=fid_b,
            relationship_type=FeatureRelationshipType.RELATED_TO,
            confidence=0.7,
            source="manual",
        )

        rels = await storage.get_feature_relationships(fid_a)
        assert any(
            r.relationship_type == FeatureRelationshipType.RELATED_TO for r in rels
        )

    @pytest.mark.asyncio
    async def test_supersedes(self, storage):
        fid_old = await _create_feature(storage, feature="age", value="29")
        fid_new = await _create_feature(storage, feature="age", value="30")

        await storage.add_feature_relationship(
            source_id=fid_new,
            target_id=fid_old,
            relationship_type=FeatureRelationshipType.SUPERSEDES,
            confidence=1.0,
            source="llm",
        )

        rels = await storage.get_feature_relationships(fid_new)
        assert any(
            r.relationship_type == FeatureRelationshipType.SUPERSEDES for r in rels
        )


# =====================================================================
# Validation Rejections
# =====================================================================


class TestValidation:
    """Tests for validation errors on add_feature_relationship."""

    @pytest.mark.asyncio
    async def test_invalid_relationship_type(self, storage):
        fid_a = await _create_feature(storage, feature="a", value="1")
        fid_b = await _create_feature(storage, feature="b", value="2")

        with pytest.raises(TypeError, match="FeatureRelationshipType"):
            await storage.add_feature_relationship(
                source_id=fid_a,
                target_id=fid_b,
                relationship_type="NOT_A_TYPE",  # type: ignore[arg-type]
                confidence=0.5,
                source="llm",
            )

    @pytest.mark.asyncio
    async def test_confidence_out_of_range(self, storage):
        fid_a = await _create_feature(storage, feature="a", value="1")
        fid_b = await _create_feature(storage, feature="b", value="2")

        with pytest.raises(ValueError, match="confidence"):
            await storage.add_feature_relationship(
                source_id=fid_a,
                target_id=fid_b,
                relationship_type=FeatureRelationshipType.RELATED_TO,
                confidence=1.5,
                source="llm",
            )

    @pytest.mark.asyncio
    async def test_invalid_source(self, storage):
        fid_a = await _create_feature(storage, feature="a", value="1")
        fid_b = await _create_feature(storage, feature="b", value="2")

        with pytest.raises(ValueError, match="source"):
            await storage.add_feature_relationship(
                source_id=fid_a,
                target_id=fid_b,
                relationship_type=FeatureRelationshipType.RELATED_TO,
                confidence=0.5,
                source="unknown_source",
            )


# =====================================================================
# Querying With Filters
# =====================================================================


class TestQueryFilters:
    """Tests for get_feature_relationships with filters."""

    @pytest.mark.asyncio
    async def test_filter_by_type(self, storage):
        fid_a = await _create_feature(storage, feature="a", value="1")
        fid_b = await _create_feature(storage, feature="b", value="2")
        fid_c = await _create_feature(storage, feature="c", value="3")

        await storage.add_feature_relationship(
            source_id=fid_a,
            target_id=fid_b,
            relationship_type=FeatureRelationshipType.IMPLIES,
            confidence=0.9,
            source="llm",
        )
        await storage.add_feature_relationship(
            source_id=fid_a,
            target_id=fid_c,
            relationship_type=FeatureRelationshipType.RELATED_TO,
            confidence=0.5,
            source="manual",
        )

        rels = await storage.get_feature_relationships(
            fid_a,
            relationship_type=FeatureRelationshipType.IMPLIES,
        )
        assert len(rels) == 1
        assert rels[0].relationship_type == FeatureRelationshipType.IMPLIES

    @pytest.mark.asyncio
    async def test_filter_by_direction(self, storage):
        fid_a = await _create_feature(storage, feature="a", value="1")
        fid_b = await _create_feature(storage, feature="b", value="2")

        await storage.add_feature_relationship(
            source_id=fid_a,
            target_id=fid_b,
            relationship_type=FeatureRelationshipType.IMPLIES,
            confidence=0.9,
            source="llm",
        )

        # Outgoing from fid_a should find it
        rels_out = await storage.get_feature_relationships(
            fid_a,
            direction=RelationshipDirection.OUTGOING,
        )
        assert len(rels_out) == 1

        # Incoming to fid_a should not (it's an outgoing edge)
        rels_in = await storage.get_feature_relationships(
            fid_a,
            direction=RelationshipDirection.INCOMING,
        )
        assert len(rels_in) == 0

    @pytest.mark.asyncio
    async def test_filter_by_min_confidence(self, storage):
        fid_a = await _create_feature(storage, feature="a", value="1")
        fid_b = await _create_feature(storage, feature="b", value="2")
        fid_c = await _create_feature(storage, feature="c", value="3")

        await storage.add_feature_relationship(
            source_id=fid_a,
            target_id=fid_b,
            relationship_type=FeatureRelationshipType.RELATED_TO,
            confidence=0.9,
            source="llm",
        )
        await storage.add_feature_relationship(
            source_id=fid_a,
            target_id=fid_c,
            relationship_type=FeatureRelationshipType.RELATED_TO,
            confidence=0.3,
            source="llm",
        )

        rels = await storage.get_feature_relationships(
            fid_a,
            min_confidence=0.5,
        )
        assert len(rels) == 1
        assert rels[0].confidence >= 0.5


# =====================================================================
# Deletion
# =====================================================================


class TestDeletion:
    """Tests for relationship deletion and cascade."""

    @pytest.mark.asyncio
    async def test_delete_idempotent(self, storage):
        """Deleting a non-existent relationship does not error."""
        fid_a = await _create_feature(storage, feature="a", value="1")
        fid_b = await _create_feature(storage, feature="b", value="2")

        # Delete something that doesn't exist -- should not raise
        await storage.delete_feature_relationships(
            source_id=fid_a,
            target_id=fid_b,
            relationship_type=FeatureRelationshipType.CONTRADICTS,
        )

    @pytest.mark.asyncio
    async def test_delete_removes_relationship(self, storage):
        fid_a = await _create_feature(storage, feature="a", value="1")
        fid_b = await _create_feature(storage, feature="b", value="2")

        await storage.add_feature_relationship(
            source_id=fid_a,
            target_id=fid_b,
            relationship_type=FeatureRelationshipType.IMPLIES,
            confidence=0.9,
            source="llm",
        )

        await storage.delete_feature_relationships(
            source_id=fid_a,
            target_id=fid_b,
            relationship_type=FeatureRelationshipType.IMPLIES,
        )

        rels = await storage.get_feature_relationships(fid_a)
        assert len(rels) == 0

    @pytest.mark.asyncio
    async def test_cascade_on_feature_delete(self, storage, neo4j_driver):
        """Deleting a feature node removes its relationships (DETACH DELETE)."""
        fid_a = await _create_feature(storage, feature="a", value="1")
        fid_b = await _create_feature(storage, feature="b", value="2")

        await storage.add_feature_relationship(
            source_id=fid_a,
            target_id=fid_b,
            relationship_type=FeatureRelationshipType.RELATED_TO,
            confidence=0.8,
            source="manual",
        )

        # Delete feature a
        await storage.delete_features([fid_a])

        # Relationship should be gone
        rels = await storage.get_feature_relationships(fid_b)
        assert len(rels) == 0


# =====================================================================
# Contradiction Detection
# =====================================================================


class TestContradictions:
    """Tests for find_contradictions."""

    @pytest.mark.asyncio
    async def test_find_contradictions(self, storage):
        set_id = "contradiction-set"
        fid_a = await _create_feature(
            storage,
            set_id=set_id,
            feature="color",
            value="red",
        )
        fid_b = await _create_feature(
            storage,
            set_id=set_id,
            feature="color",
            value="blue",
        )
        await _create_feature(
            storage,
            set_id=set_id,
            feature="size",
            value="large",
        )

        await storage.add_feature_relationship(
            source_id=fid_a,
            target_id=fid_b,
            relationship_type=FeatureRelationshipType.CONTRADICTS,
            confidence=0.95,
            source="llm",
        )

        pairs = await storage.find_contradictions(set_id=set_id)
        assert len(pairs) == 1
        assert pairs[0].confidence == 0.95

    @pytest.mark.asyncio
    async def test_no_contradictions(self, storage):
        set_id = "no-contradiction-set"
        fid_a = await _create_feature(
            storage,
            set_id=set_id,
            feature="a",
            value="1",
        )
        fid_b = await _create_feature(
            storage,
            set_id=set_id,
            feature="b",
            value="2",
        )

        # Only IMPLIES, no CONTRADICTS
        await storage.add_feature_relationship(
            source_id=fid_a,
            target_id=fid_b,
            relationship_type=FeatureRelationshipType.IMPLIES,
            confidence=0.9,
            source="rule",
        )

        pairs = await storage.find_contradictions(set_id=set_id)
        assert len(pairs) == 0


# =====================================================================
# Supersession Chain Traversal
# =====================================================================


class TestSupersessionChain:
    """Tests for find_supersession_chain."""

    @pytest.mark.asyncio
    async def test_chain_traversal(self, storage):
        """Traverse a chain: v3 supersedes v2 supersedes v1."""
        fid_v1 = await _create_feature(storage, feature="age", value="28")
        fid_v2 = await _create_feature(storage, feature="age", value="29")
        fid_v3 = await _create_feature(storage, feature="age", value="30")

        # v2 supersedes v1
        await storage.add_feature_relationship(
            source_id=fid_v2,
            target_id=fid_v1,
            relationship_type=FeatureRelationshipType.SUPERSEDES,
            confidence=1.0,
            source="llm",
        )
        # v3 supersedes v2
        await storage.add_feature_relationship(
            source_id=fid_v3,
            target_id=fid_v2,
            relationship_type=FeatureRelationshipType.SUPERSEDES,
            confidence=1.0,
            source="llm",
        )

        # Starting from v3 (newest), should get the full chain
        chain = await storage.find_supersession_chain(fid_v3)
        assert chain.current == fid_v3
        assert len(chain.chain) >= 2  # At least v2 and v1 in the chain

    @pytest.mark.asyncio
    async def test_single_node_chain(self, storage):
        """A feature with no supersession returns itself."""
        fid = await _create_feature(storage, feature="standalone", value="val")

        chain = await storage.find_supersession_chain(fid)
        assert chain.current == fid
        assert fid in chain.chain
