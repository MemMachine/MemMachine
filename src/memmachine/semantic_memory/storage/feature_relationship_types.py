"""Data types for relationships between semantic features."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import Enum

from memmachine.semantic_memory.semantic_model import FeatureIdT


class FeatureRelationshipType(Enum):
    """Supported typed relationships between semantic features."""

    CONTRADICTS = "CONTRADICTS"
    IMPLIES = "IMPLIES"
    RELATED_TO = "RELATED_TO"
    SUPERSEDES = "SUPERSEDES"


class RelationshipDirection(Enum):
    """Direction filter for querying feature relationships."""

    OUTGOING = "outgoing"
    INCOMING = "incoming"
    BOTH = "both"


@dataclass(kw_only=True)
class FeatureRelationship:
    """A typed, directed relationship between two semantic features."""

    source_id: FeatureIdT
    target_id: FeatureIdT
    relationship_type: FeatureRelationshipType
    confidence: float
    detected_at: datetime
    source: str  # "llm", "rule", or "manual"

    def __post_init__(self) -> None:
        """Validate fields after initialization."""
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(
                f"confidence must be between 0.0 and 1.0, got {self.confidence}"
            )
        if self.source not in {"llm", "rule", "manual"}:
            raise ValueError(
                f"source must be 'llm', 'rule', or 'manual', got {self.source!r}"
            )


@dataclass(kw_only=True)
class ContradictionPair:
    """A pair of features connected by a CONTRADICTS relationship."""

    feature_id_a: FeatureIdT
    feature_id_b: FeatureIdT
    confidence: float
    detected_at: datetime
    source: str


@dataclass(kw_only=True)
class SupersessionChain:
    """The result of traversing a SUPERSEDES chain from a feature."""

    current: FeatureIdT
    chain: list[FeatureIdT]  # ordered from newest to oldest
