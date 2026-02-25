"""Protocol for semantic storage backends that support feature relationships."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from memmachine.semantic_memory.semantic_model import FeatureIdT, SetIdT
from memmachine.semantic_memory.storage.feature_relationship_types import (
    ContradictionPair,
    FeatureRelationship,
    FeatureRelationshipType,
    RelationshipDirection,
    SupersessionChain,
)


@runtime_checkable
class SemanticRelationshipStorage(Protocol):
    """Protocol for storage backends that support typed feature relationships.

    Backends that implement this protocol can create, query, and delete
    typed relationships (CONTRADICTS, IMPLIES, RELATED_TO, SUPERSEDES)
    between semantic features.

    Callers should use ``isinstance(storage, SemanticRelationshipStorage)``
    to check whether a particular backend supports these operations.
    """

    async def add_feature_relationship(
        self,
        *,
        source_id: FeatureIdT,
        target_id: FeatureIdT,
        relationship_type: FeatureRelationshipType,
        confidence: float,
        source: str,
        similarity: float | None = None,
    ) -> None:
        """Create a typed relationship between two feature nodes.

        Args:
            source_id: ID of the source feature.
            target_id: ID of the target feature.
            relationship_type: Type of relationship to create.
            confidence: Confidence score (0.0 to 1.0).
            source: How the relationship was detected
                (``"llm"``, ``"rule"``, or ``"manual"``).
            similarity: Optional path-quality weight used by graph
                traversal scoring.  Set only for semantically
                meaningful cross-feature-name connections so that
                trivial same-name edges (quality ``NULL`` → ``0.0``)
                are excluded from path quality calculations.

        Raises:
            ValueError: If *relationship_type* is not a valid
                :class:`FeatureRelationshipType`, or *confidence* is
                outside ``[0.0, 1.0]``.

        """
        ...

    async def get_feature_relationships(
        self,
        feature_id: FeatureIdT,
        *,
        relationship_type: FeatureRelationshipType | None = None,
        direction: RelationshipDirection = RelationshipDirection.BOTH,
        min_confidence: float | None = None,
    ) -> list[FeatureRelationship]:
        """Retrieve relationships for a given feature.

        Args:
            feature_id: Feature to query relationships for.
            relationship_type: Optional filter for a specific type.
            direction: Direction filter (OUTGOING, INCOMING, or BOTH).
            min_confidence: Optional minimum confidence threshold.

        Returns:
            List of :class:`FeatureRelationship` instances.

        """
        ...

    async def delete_feature_relationships(
        self,
        *,
        source_id: FeatureIdT,
        target_id: FeatureIdT,
        relationship_type: FeatureRelationshipType,
    ) -> None:
        """Delete a specific relationship between two features.

        This operation is idempotent -- deleting a non-existent
        relationship completes without error.

        Args:
            source_id: Source feature ID.
            target_id: Target feature ID.
            relationship_type: Relationship type to delete.

        """
        ...

    async def find_contradictions(
        self,
        *,
        set_id: SetIdT,
    ) -> list[ContradictionPair]:
        """Find all CONTRADICTS relationships within a feature set.

        Args:
            set_id: The feature set to search for contradictions.

        Returns:
            List of :class:`ContradictionPair` instances.

        """
        ...

    async def find_supersession_chain(
        self,
        feature_id: FeatureIdT,
    ) -> SupersessionChain:
        """Traverse the SUPERSEDES chain from a feature.

        Returns the most current version and the full chain
        from newest to oldest.

        Args:
            feature_id: Feature to start the traversal from.

        Returns:
            A :class:`SupersessionChain` with the current (newest)
            feature and the ordered chain of all versions.

        """
        ...
