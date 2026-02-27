"""Data types for nodes and edges in a vector graph store."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

from memmachine.common.data_types import SimilarityMetric

# Types that can be used as property values in nodes and edges.
PropertyValue = (
    bool
    | int
    | float
    | str
    | datetime
    | list[bool]
    | list[int]
    | list[float]
    | list[str]
    | list[datetime]
    | None
)


class EntityType(Enum):
    """Supported graph entity types."""

    NODE = "node"
    EDGE = "edge"


@dataclass(kw_only=True)
class Node:
    """Graph node representation with properties and embeddings."""

    uid: str
    properties: dict[str, PropertyValue] = field(default_factory=dict)
    embeddings: dict[str, tuple[list[float], SimilarityMetric]] = field(
        default_factory=dict,
    )
    entity_types: list[str] = field(default_factory=list)

    def __eq__(self, other: object) -> bool:
        """Compare nodes by UID, properties, and embeddings."""
        if not isinstance(other, Node):
            return False
        return (
            self.uid == other.uid
            and self.properties == other.properties
            and self.embeddings == other.embeddings
        )

    def __hash__(self) -> int:
        """Hash a node by its UID."""
        return hash(self.uid)


@dataclass(kw_only=True)
class Edge:
    """Graph edge representation with properties and embeddings."""

    uid: str
    source_uid: str
    target_uid: str
    properties: dict[str, PropertyValue] = field(default_factory=dict)
    embeddings: dict[str, tuple[list[float], SimilarityMetric]] = field(
        default_factory=dict,
    )

    def __eq__(self, other: object) -> bool:
        """Compare edges by uid, properties, and embeddings."""
        if not isinstance(other, Edge):
            return False
        return (
            self.uid == other.uid
            and self.properties == other.properties
            and self.embeddings == other.embeddings
        )

    def __hash__(self) -> int:
        """Hash an edge by its UID."""
        return hash(self.uid)


_MANGLE_PROPERTY_NAME_PREFIX = "property_"
_MANGLE_EMBEDDING_NAME_PREFIX = "embedding_"


def mangle_property_name(property_name: str) -> str:
    """Mangle a property name to avoid conflicts."""
    return _MANGLE_PROPERTY_NAME_PREFIX + property_name


def demangle_property_name(mangled_property_name: str) -> str:
    """Restore the original property name from its mangled form."""
    return mangled_property_name.removeprefix(_MANGLE_PROPERTY_NAME_PREFIX)


def is_mangled_property_name(candidate_name: str) -> bool:
    """Return True if the candidate is a mangled property name."""
    return candidate_name.startswith(_MANGLE_PROPERTY_NAME_PREFIX)


def mangle_embedding_name(embedding_name: str) -> str:
    """Mangle an embedding name to avoid conflicts."""
    return _MANGLE_EMBEDDING_NAME_PREFIX + embedding_name


def demangle_embedding_name(mangled_embedding_name: str) -> str:
    """Restore the original embedding name from its mangled form."""
    return mangled_embedding_name.removeprefix(_MANGLE_EMBEDDING_NAME_PREFIX)


def is_mangled_embedding_name(candidate_name: str) -> bool:
    """Return True if the candidate is a mangled embedding name."""
    return candidate_name.startswith(_MANGLE_EMBEDDING_NAME_PREFIX)


# ---------------------------------------------------------------------------
# Graph traversal and knowledge-graph data types
# ---------------------------------------------------------------------------


class TraversalDirection(Enum):
    """Direction constraint for graph traversal."""

    OUTGOING = "outgoing"
    INCOMING = "incoming"
    BOTH = "both"


@dataclass(kw_only=True)
class GraphFilter:
    """Structural filter for graph-aware vector search.

    Specifies a traversal pattern from an anchor node that narrows the
    candidate set before vector similarity is computed.
    """

    anchor_node_uid: str
    anchor_collection: str
    relation_types: list[str] | None = None
    max_hops: int = 1
    direction: TraversalDirection = TraversalDirection.BOTH


@dataclass(kw_only=True)
class MultiHopResult:
    """A node returned from a multi-hop traversal with distance metadata."""

    node: Node
    hop_distance: int
    score: float
    path_quality: float = 1.0
    """Minimum ``RELATED_TO`` edge similarity along the traversal path.

    Defaults to ``1.0`` when the path contains no ``RELATED_TO`` edges
    or the store implementation does not compute it.  A value of ``0.0``
    indicates the path crossed a ``RELATED_TO`` edge with no recorded
    similarity (e.g. trivial same-name matches), signalling that the
    traversal should not be trusted as a semantic discovery.
    """


class DuplicateResolutionStrategy(Enum):
    """Strategy for resolving a detected duplicate pair."""

    MERGE = "merge"
    DISMISS = "dismiss"


@dataclass(kw_only=True)
class DuplicateProposal:
    """A proposed duplicate pair detected by the deduplication process."""

    node_uid_a: str
    node_uid_b: str
    embedding_similarity: float
    property_similarity: float
    detected_at: datetime
    auto_merged: bool = False
