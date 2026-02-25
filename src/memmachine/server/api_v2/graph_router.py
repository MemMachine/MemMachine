"""API v2 router for graph-native knowledge graph endpoints.

Provides REST endpoints for multi-hop traversal, graph-filtered search,
feature relationship management, entity deduplication, and GDS analytics.
"""

from __future__ import annotations

import logging
from datetime import datetime
from enum import Enum
from typing import Annotated, Any

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field

from memmachine.common.vector_graph_store import VectorGraphStore
from memmachine.main.memmachine import MemMachine
from memmachine.semantic_memory.storage.storage_base import SemanticStorage
from memmachine.server.api_v2.exceptions import RestError
from memmachine.server.api_v2.service import get_memmachine

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Shared error message constants (SonarQube S1192 — avoid literal duplication)
# ---------------------------------------------------------------------------

_ERR_GRAPH_STORE_ACCESS = "Failed to access graph store"
_ERR_RELATIONSHIPS_NOT_SUPPORTED = (
    "Feature relationships are not supported by the configured storage."
)
_ERR_DEDUP_NOT_SUPPORTED = "Deduplication is not supported by the configured store."
_ERR_GDS_NOT_AVAILABLE = (
    "GDS plugin is not available. Enable gds_enabled in Neo4j config."
)


# ---------------------------------------------------------------------------
# Enums mirroring store-layer types for the API surface
# ---------------------------------------------------------------------------


class TraversalDirectionParam(str, Enum):
    """Direction for graph traversal."""

    OUTGOING = "outgoing"
    INCOMING = "incoming"
    BOTH = "both"


class FeatureRelationshipTypeParam(str, Enum):
    """Types of feature relationships."""

    CONTRADICTS = "CONTRADICTS"
    IMPLIES = "IMPLIES"
    RELATED_TO = "RELATED_TO"
    SUPERSEDES = "SUPERSEDES"


class RelationshipDirectionParam(str, Enum):
    """Direction for relationship queries."""

    OUTGOING = "outgoing"
    INCOMING = "incoming"
    BOTH = "both"


class DuplicateResolutionStrategyParam(str, Enum):
    """Strategy for resolving duplicate proposals."""

    MERGE = "merge"
    DISMISS = "dismiss"


# ---------------------------------------------------------------------------
# 9.2 — Multi-hop search models
# ---------------------------------------------------------------------------


class MultiHopSearchSpec(BaseModel):
    """Request model for multi-hop graph traversal search."""

    collection: Annotated[
        str,
        Field(
            ...,
            description="The collection (graph namespace) to search within.",
            examples=["universal/universal"],
        ),
    ]
    node_uid: Annotated[
        str,
        Field(
            ...,
            description="UID of the anchor node to start traversal from.",
            examples=["node-abc-123"],
        ),
    ]
    min_hops: Annotated[
        int,
        Field(
            default=1,
            ge=1,
            description="Minimum hop distance from the anchor node.",
        ),
    ]
    max_hops: Annotated[
        int,
        Field(
            default=3,
            ge=1,
            le=10,
            description="Maximum hop distance from the anchor node.",
        ),
    ]
    relation_types: Annotated[
        list[str] | None,
        Field(
            default=None,
            description=(
                "Optional list of relationship types to traverse. "
                "If None, all relationship types are followed."
            ),
        ),
    ]
    score_decay: Annotated[
        float,
        Field(
            default=0.7,
            ge=0.0,
            le=1.0,
            description="Decay factor applied per hop to the traversal score.",
        ),
    ]
    limit: Annotated[
        int | None,
        Field(
            default=None,
            ge=1,
            description="Maximum number of results to return.",
        ),
    ]
    target_collections: Annotated[
        list[str] | None,
        Field(
            default=None,
            description=(
                "Optional list of collections that end nodes must belong to. "
                "When None, traversal can reach nodes in any collection "
                "(cross-collection traversal). When provided, only end nodes "
                "belonging to at least one of these collections are returned."
            ),
        ),
    ]


class MultiHopNodeResult(BaseModel):
    """A single node returned from a multi-hop traversal."""

    uid: Annotated[str, Field(description="Unique identifier of the node.")]
    hop_distance: Annotated[
        int,
        Field(description="Number of hops from the anchor node."),
    ]
    score: Annotated[
        float,
        Field(description="Traversal score (decayed by distance)."),
    ]
    properties: Annotated[
        dict[str, Any],
        Field(
            default_factory=dict,
            description="Node properties (excluding embeddings).",
        ),
    ]
    entity_types: Annotated[
        list[str],
        Field(
            default_factory=list,
            description="Entity type labels on this node.",
        ),
    ]


class MultiHopSearchResult(BaseModel):
    """Response model for multi-hop graph traversal search."""

    results: Annotated[
        list[MultiHopNodeResult],
        Field(description="Nodes discovered via graph traversal."),
    ]


# ---------------------------------------------------------------------------
# 9.2 — Graph-filtered search models
# ---------------------------------------------------------------------------


class GraphFilteredSearchSpec(BaseModel):
    """Request model for graph-filtered similarity search."""

    collection: Annotated[
        str,
        Field(
            ...,
            description="The collection (graph namespace) to search within.",
            examples=["universal/universal"],
        ),
    ]
    query: Annotated[
        str,
        Field(
            ...,
            description="Natural language query for similarity search.",
            examples=["What is the user's favorite food?"],
        ),
    ]
    limit: Annotated[
        int,
        Field(
            default=10,
            ge=1,
            le=500,
            description="Maximum number of results to return.",
        ),
    ]
    anchor_node_uid: Annotated[
        str | None,
        Field(
            default=None,
            description=(
                "UID of the anchor node for graph filtering. "
                "When provided, candidates are narrowed via graph "
                "traversal before similarity scoring."
            ),
        ),
    ]
    anchor_collection: Annotated[
        str | None,
        Field(
            default=None,
            description="Collection of the anchor node (defaults to search collection).",
        ),
    ]
    relation_types: Annotated[
        list[str] | None,
        Field(
            default=None,
            description="Relationship types to follow from the anchor node.",
        ),
    ]
    max_hops: Annotated[
        int,
        Field(
            default=1,
            ge=1,
            le=10,
            description="Maximum hops from anchor for candidate filtering.",
        ),
    ]
    direction: Annotated[
        TraversalDirectionParam,
        Field(
            default=TraversalDirectionParam.BOTH,
            description="Traversal direction from the anchor node.",
        ),
    ]


class GraphFilteredNodeResult(BaseModel):
    """A single node from graph-filtered similarity search."""

    uid: Annotated[str, Field(description="Unique identifier of the node.")]
    properties: Annotated[
        dict[str, Any],
        Field(
            default_factory=dict,
            description="Node properties (excluding embeddings).",
        ),
    ]
    entity_types: Annotated[
        list[str],
        Field(
            default_factory=list,
            description="Entity type labels on this node.",
        ),
    ]


class GraphFilteredSearchResult(BaseModel):
    """Response model for graph-filtered similarity search."""

    results: Annotated[
        list[GraphFilteredNodeResult],
        Field(description="Nodes matching the query with optional graph filtering."),
    ]


# ---------------------------------------------------------------------------
# 9.3 — Feature relationship CRUD models
# ---------------------------------------------------------------------------


class CreateRelationshipSpec(BaseModel):
    """Request model for creating a feature relationship."""

    source_id: Annotated[
        str,
        Field(
            ...,
            description="ID of the source feature.",
        ),
    ]
    target_id: Annotated[
        str,
        Field(
            ...,
            description="ID of the target feature.",
        ),
    ]
    relationship_type: Annotated[
        FeatureRelationshipTypeParam,
        Field(
            ...,
            description="Type of relationship to create.",
        ),
    ]
    confidence: Annotated[
        float,
        Field(
            ...,
            ge=0.0,
            le=1.0,
            description="Confidence score for the relationship (0.0-1.0).",
        ),
    ]
    source: Annotated[
        str,
        Field(
            default="manual",
            description='How the relationship was created (e.g. "llm", "rule", "manual").',
        ),
    ]


class GetRelationshipsSpec(BaseModel):
    """Request model for querying feature relationships."""

    feature_id: Annotated[
        str,
        Field(
            ...,
            description="Feature ID to query relationships for.",
        ),
    ]
    relationship_type: Annotated[
        FeatureRelationshipTypeParam | None,
        Field(
            default=None,
            description="Filter by relationship type. None returns all types.",
        ),
    ]
    direction: Annotated[
        RelationshipDirectionParam,
        Field(
            default=RelationshipDirectionParam.BOTH,
            description="Direction to query (outgoing, incoming, or both).",
        ),
    ]
    min_confidence: Annotated[
        float | None,
        Field(
            default=None,
            ge=0.0,
            le=1.0,
            description="Minimum confidence threshold for returned relationships.",
        ),
    ]


class DeleteRelationshipSpec(BaseModel):
    """Request model for deleting a feature relationship."""

    source_id: Annotated[
        str,
        Field(
            ...,
            description="ID of the source feature.",
        ),
    ]
    target_id: Annotated[
        str,
        Field(
            ...,
            description="ID of the target feature.",
        ),
    ]
    relationship_type: Annotated[
        FeatureRelationshipTypeParam,
        Field(
            ...,
            description="Type of relationship to delete.",
        ),
    ]


class RelationshipResponse(BaseModel):
    """A single feature relationship."""

    source_id: Annotated[str, Field(description="ID of the source feature.")]
    target_id: Annotated[str, Field(description="ID of the target feature.")]
    relationship_type: Annotated[
        FeatureRelationshipTypeParam,
        Field(description="Type of relationship."),
    ]
    confidence: Annotated[
        float,
        Field(description="Confidence score (0.0-1.0)."),
    ]
    detected_at: Annotated[
        datetime,
        Field(description="When the relationship was detected."),
    ]
    source: Annotated[
        str,
        Field(
            description='How the relationship was created ("llm", "rule", "manual").'
        ),
    ]


class RelationshipListResponse(BaseModel):
    """Response model for listing feature relationships."""

    relationships: Annotated[
        list[RelationshipResponse],
        Field(description="Matching feature relationships."),
    ]


# ---------------------------------------------------------------------------
# 9.4 — Contradiction models
# ---------------------------------------------------------------------------


class ContradictionsSpec(BaseModel):
    """Request model for finding contradictions within a feature set."""

    set_id: Annotated[
        str,
        Field(
            ...,
            description="Semantic set ID to find contradictions within.",
        ),
    ]


class ContradictionPairResponse(BaseModel):
    """A pair of contradicting features."""

    feature_id_a: Annotated[str, Field(description="First contradicting feature ID.")]
    feature_id_b: Annotated[str, Field(description="Second contradicting feature ID.")]
    confidence: Annotated[float, Field(description="Confidence score (0.0-1.0).")]
    detected_at: Annotated[
        datetime,
        Field(description="When the contradiction was detected."),
    ]
    source: Annotated[
        str,
        Field(
            description='How the contradiction was detected ("llm", "rule", "manual").'
        ),
    ]


class ContradictionsResult(BaseModel):
    """Response model for contradiction detection."""

    contradictions: Annotated[
        list[ContradictionPairResponse],
        Field(description="Contradicting feature pairs."),
    ]


# ---------------------------------------------------------------------------
# 9.5 — Dedup models
# ---------------------------------------------------------------------------


class DedupProposalsSpec(BaseModel):
    """Request model for fetching duplicate proposals."""

    collection: Annotated[
        str,
        Field(
            ...,
            description="The collection (graph namespace) to check for duplicates.",
            examples=["universal/universal"],
        ),
    ]
    min_embedding_similarity: Annotated[
        float | None,
        Field(
            default=None,
            ge=0.0,
            le=1.0,
            description="Minimum embedding similarity to include in results.",
        ),
    ]
    include_auto_merged: Annotated[
        bool,
        Field(
            default=False,
            description="Whether to include proposals that were auto-merged.",
        ),
    ]


class DuplicateProposalResponse(BaseModel):
    """A single duplicate proposal."""

    node_uid_a: Annotated[str, Field(description="UID of the first node.")]
    node_uid_b: Annotated[str, Field(description="UID of the second node.")]
    embedding_similarity: Annotated[
        float,
        Field(description="Cosine similarity of node embeddings."),
    ]
    property_similarity: Annotated[
        float,
        Field(description="Property overlap similarity score."),
    ]
    detected_at: Annotated[
        datetime,
        Field(description="When the duplicate was detected."),
    ]
    auto_merged: Annotated[
        bool,
        Field(description="Whether this pair was auto-merged."),
    ]


class DedupProposalsResult(BaseModel):
    """Response model for duplicate proposals."""

    proposals: Annotated[
        list[DuplicateProposalResponse],
        Field(description="Duplicate node proposals."),
    ]


class DedupResolvePair(BaseModel):
    """A single pair to resolve with a strategy."""

    node_uid_a: Annotated[str, Field(description="UID of the first node.")]
    node_uid_b: Annotated[str, Field(description="UID of the second node.")]
    strategy: Annotated[
        DuplicateResolutionStrategyParam,
        Field(description="Resolution strategy: merge or dismiss."),
    ]


class DedupResolveSpec(BaseModel):
    """Request model for resolving duplicate proposals."""

    collection: Annotated[
        str,
        Field(
            ...,
            description="The collection (graph namespace) containing the duplicates.",
            examples=["universal/universal"],
        ),
    ]
    pairs: Annotated[
        list[DedupResolvePair],
        Field(
            ...,
            min_length=1,
            description="Duplicate pairs to resolve with their strategies.",
        ),
    ]


class DedupResolveResult(BaseModel):
    """Response model for duplicate resolution."""

    resolved: Annotated[
        int,
        Field(description="Number of pairs resolved."),
    ]


# ---------------------------------------------------------------------------
# 9.6 — PageRank + Community detection models
# ---------------------------------------------------------------------------


class PageRankSpec(BaseModel):
    """Request model for computing PageRank on graph nodes."""

    collection: Annotated[
        str,
        Field(
            ...,
            description="The collection (graph namespace) to compute PageRank on.",
            examples=["universal/universal"],
        ),
    ]
    relation_types: Annotated[
        list[str] | None,
        Field(
            default=None,
            description=(
                "Optional list of relationship types to include in the "
                "GDS projection. When None, all relationship types are used."
            ),
        ),
    ]
    damping_factor: Annotated[
        float,
        Field(
            default=0.85,
            ge=0.0,
            le=1.0,
            description="PageRank damping factor.",
        ),
    ]
    max_iterations: Annotated[
        int,
        Field(
            default=20,
            ge=1,
            description="Maximum iterations for the PageRank algorithm.",
        ),
    ]
    write_back: Annotated[
        bool,
        Field(
            default=False,
            description=(
                "Whether to write PageRank scores back to node properties. "
                "When False, scores are computed and returned without persisting."
            ),
        ),
    ]
    write_property: Annotated[
        str | None,
        Field(
            default=None,
            description=(
                "Property name for write-back. Defaults to 'pagerank_score' "
                "when None. Only used when write_back is True."
            ),
        ),
    ]


class PageRankNodeScore(BaseModel):
    """PageRank score for a single node."""

    node_uid: Annotated[str, Field(description="UID of the node.")]
    score: Annotated[float, Field(description="PageRank score.")]


class PageRankResult(BaseModel):
    """Response model for PageRank computation."""

    scores: Annotated[
        list[PageRankNodeScore],
        Field(description="PageRank scores per node."),
    ]


class CommunitiesSpec(BaseModel):
    """Request model for community detection."""

    collection: Annotated[
        str,
        Field(
            ...,
            description="The collection (graph namespace) to detect communities in.",
            examples=["universal/universal"],
        ),
    ]
    relation_types: Annotated[
        list[str] | None,
        Field(
            default=None,
            description=(
                "Optional list of relationship types to include in the "
                "GDS projection. When None, all relationship types are used."
            ),
        ),
    ]
    max_iterations: Annotated[
        int,
        Field(
            default=20,
            ge=1,
            description="Maximum iterations for the community detection algorithm.",
        ),
    ]
    write_back: Annotated[
        bool,
        Field(
            default=False,
            description="Whether to write community IDs back to node properties.",
        ),
    ]
    write_property: Annotated[
        str | None,
        Field(
            default=None,
            description=(
                "Property name for write-back. Defaults to 'community_id' "
                "when None. Only used when write_back is True."
            ),
        ),
    ]


class CommunityGroup(BaseModel):
    """A single detected community."""

    community_id: Annotated[int, Field(description="Numeric community identifier.")]
    node_uids: Annotated[
        list[str],
        Field(description="UIDs of nodes in this community."),
    ]


class CommunitiesResult(BaseModel):
    """Response model for community detection."""

    communities: Annotated[
        list[CommunityGroup],
        Field(description="Detected communities with their member nodes."),
    ]


class GraphStatsSpec(BaseModel):
    """Request model for graph statistics."""

    collection: Annotated[
        str,
        Field(
            ...,
            description="The collection (graph namespace) to compute statistics for.",
            examples=["universal/universal"],
        ),
    ]


class GraphStatsResponse(BaseModel):
    """Response model for graph statistics."""

    node_count: Annotated[
        int, Field(description="Total number of nodes in the collection.")
    ]
    edge_count: Annotated[
        int, Field(description="Total number of edges in the collection.")
    ]
    avg_degree: Annotated[
        float,
        Field(description="Average degree (edge_count / node_count, 0.0 if empty)."),
    ]
    relationship_type_distribution: Annotated[
        dict[str, int],
        Field(description="Mapping of relationship type name to count."),
    ]
    entity_type_distribution: Annotated[
        dict[str, int],
        Field(description="Mapping of entity type name to count (desanitized)."),
    ]


class ShortestPathSpec(BaseModel):
    """Request model for shortest-path search."""

    collection: Annotated[
        str,
        Field(
            ...,
            description="The collection (graph namespace) to search in.",
            examples=["universal/universal"],
        ),
    ]
    source_uid: Annotated[
        str,
        Field(
            ...,
            description="UID of the starting node.",
        ),
    ]
    target_uid: Annotated[
        str,
        Field(
            ...,
            description="UID of the ending node.",
        ),
    ]
    relation_types: Annotated[
        list[str] | None,
        Field(
            default=None,
            description=(
                "Optional relationship types to traverse. "
                "When None, all types are traversed."
            ),
        ),
    ]
    max_depth: Annotated[
        int,
        Field(
            default=10,
            ge=1,
            description="Maximum traversal depth for the path search.",
        ),
    ]


class PathNodeResponse(BaseModel):
    """A node along the shortest path."""

    uid: Annotated[str, Field(description="UID of the node.")]
    properties: Annotated[
        dict[str, Any],
        Field(description="Node properties (excluding uid)."),
    ]


class PathEdgeResponse(BaseModel):
    """An edge along the shortest path."""

    source_uid: Annotated[str, Field(description="UID of the source node.")]
    target_uid: Annotated[str, Field(description="UID of the target node.")]
    type: Annotated[str, Field(description="Relationship type.")]
    properties: Annotated[
        dict[str, Any],
        Field(description="Edge properties."),
    ]


class ShortestPathResponse(BaseModel):
    """Response model for shortest-path search."""

    path_length: Annotated[
        int,
        Field(description="Number of edges in the path (0 if no path found)."),
    ]
    nodes: Annotated[
        list[PathNodeResponse],
        Field(description="Nodes along the path in order."),
    ]
    edges: Annotated[
        list[PathEdgeResponse],
        Field(description="Edges along the path in order."),
    ]


class DegreeCentralitySpec(BaseModel):
    """Request model for degree centrality computation."""

    collection: Annotated[
        str,
        Field(
            ...,
            description="The collection (graph namespace) to compute degree centrality on.",
            examples=["universal/universal"],
        ),
    ]
    relation_types: Annotated[
        list[str] | None,
        Field(
            default=None,
            description=(
                "Optional relationship types to count. "
                "When None, all types are counted."
            ),
        ),
    ]
    limit: Annotated[
        int,
        Field(
            default=50,
            ge=1,
            description="Maximum number of results to return.",
        ),
    ]


class DegreeCentralityNodeResponse(BaseModel):
    """Degree centrality metrics for a single node."""

    uid: Annotated[str, Field(description="UID of the node.")]
    in_degree: Annotated[int, Field(description="Number of incoming edges.")]
    out_degree: Annotated[int, Field(description="Number of outgoing edges.")]
    total_degree: Annotated[
        int, Field(description="Total degree (in_degree + out_degree).")
    ]


class DegreeCentralityResponse(BaseModel):
    """Response model for degree centrality computation."""

    nodes: Annotated[
        list[DegreeCentralityNodeResponse],
        Field(description="Node degree metrics ordered by total degree descending."),
    ]


class SubgraphSpec(BaseModel):
    """Request model for subgraph extraction."""

    collection: Annotated[
        str,
        Field(
            ...,
            description="The collection (graph namespace) to extract from.",
            examples=["universal/universal"],
        ),
    ]
    anchor_uid: Annotated[
        str,
        Field(
            ...,
            description="UID of the anchor node for the ego-graph.",
        ),
    ]
    max_depth: Annotated[
        int,
        Field(
            default=2,
            ge=1,
            description="Maximum traversal depth from the anchor node.",
        ),
    ]
    relation_types: Annotated[
        list[str] | None,
        Field(
            default=None,
            description=(
                "Optional relationship types to traverse. "
                "When None, all types are traversed."
            ),
        ),
    ]
    node_limit: Annotated[
        int,
        Field(
            default=100,
            ge=1,
            description="Maximum number of nodes to return.",
        ),
    ]


class SubgraphNodeResponse(BaseModel):
    """A node in the extracted subgraph."""

    uid: Annotated[str, Field(description="UID of the node.")]
    properties: Annotated[
        dict[str, Any],
        Field(description="Node properties (excluding uid)."),
    ]


class SubgraphEdgeResponse(BaseModel):
    """An edge in the extracted subgraph."""

    source_uid: Annotated[str, Field(description="UID of the source node.")]
    target_uid: Annotated[str, Field(description="UID of the target node.")]
    type: Annotated[str, Field(description="Relationship type.")]
    properties: Annotated[
        dict[str, Any],
        Field(description="Edge properties."),
    ]


class SubgraphResponse(BaseModel):
    """Response model for subgraph extraction."""

    nodes: Annotated[
        list[SubgraphNodeResponse],
        Field(description="Nodes in the subgraph."),
    ]
    edges: Annotated[
        list[SubgraphEdgeResponse],
        Field(description="Edges in the subgraph."),
    ]


class BetweennessCentralitySpec(BaseModel):
    """Request model for betweenness centrality computation."""

    collection: Annotated[
        str,
        Field(
            ...,
            description="The collection (graph namespace) to compute betweenness on.",
            examples=["universal/universal"],
        ),
    ]
    relation_types: Annotated[
        list[str] | None,
        Field(
            default=None,
            description=(
                "Optional list of relationship types to include in the "
                "GDS projection. When None, all relationship types are used."
            ),
        ),
    ]
    sampling_size: Annotated[
        int | None,
        Field(
            default=None,
            ge=1,
            description=(
                "When provided, GDS uses approximate computation by "
                "sampling this many source nodes. None = exact computation."
            ),
        ),
    ]
    write_back: Annotated[
        bool,
        Field(
            default=False,
            description="Whether to write betweenness scores back to node properties.",
        ),
    ]
    write_property: Annotated[
        str | None,
        Field(
            default=None,
            description=(
                "Property name for write-back. Defaults to 'betweenness_score' "
                "when None. Only used when write_back is True."
            ),
        ),
    ]


class BetweennessCentralityResponse(BaseModel):
    """Response model for betweenness centrality computation."""

    scores: Annotated[
        list[PageRankNodeScore],
        Field(description="Betweenness centrality scores per node."),
    ]


# ---------------------------------------------------------------------------
# Router definition
# ---------------------------------------------------------------------------

graph_router = APIRouter(prefix="/memories/graph", tags=["Graph"])


# ---------------------------------------------------------------------------
# Helper: resolve the VectorGraphStore from the MemMachine instance
# ---------------------------------------------------------------------------


async def _get_vector_graph_store(
    memmachine: MemMachine,
) -> VectorGraphStore:
    """Return the VectorGraphStore for the configured Neo4j database.

    The Neo4j graph store name is taken from the episodic memory
    long-term-memory configuration (``vector_graph_store``), **not** from
    ``semantic_memory.database`` which typically points at a relational
    (Postgres/SQLite) backend.
    """
    ltm = memmachine.resource_manager.config.episodic_memory.long_term_memory
    db_name = ltm.vector_graph_store if ltm is not None else None
    if db_name is None:
        raise RestError(
            code=400,
            message="No Neo4j vector-graph-store configured in "
            "episodic_memory.long_term_memory.",
        )
    return await memmachine.resource_manager.get_vector_graph_store(db_name)


# ---------------------------------------------------------------------------
# 10.1 — Multi-hop search endpoint
# ---------------------------------------------------------------------------


@graph_router.post(
    "/search/multi-hop",
    description="Search the knowledge graph via multi-hop traversal from an anchor node.",
)
async def search_multi_hop(
    spec: MultiHopSearchSpec,
    memmachine: Annotated[MemMachine, Depends(get_memmachine)],
) -> MultiHopSearchResult:
    """Multi-hop graph traversal search."""
    from memmachine.common.vector_graph_store.graph_traversal_store import (
        GraphTraversalStore,
    )

    try:
        store = await _get_vector_graph_store(memmachine)
    except Exception as e:
        raise RestError(code=500, message=_ERR_GRAPH_STORE_ACCESS, ex=e) from e

    if not isinstance(store, GraphTraversalStore):
        raise RestError(
            code=501,
            message="Multi-hop search is not supported by the configured store.",
        )

    try:
        results = await store.search_multi_hop_nodes(
            collection=spec.collection,
            this_node_uid=spec.node_uid,
            min_hops=spec.min_hops,
            max_hops=spec.max_hops,
            relation_types=spec.relation_types,
            score_decay=spec.score_decay,
            limit=spec.limit,
            target_collections=spec.target_collections,
        )
    except Exception as e:
        raise RestError(code=500, message="Multi-hop search failed", ex=e) from e

    return MultiHopSearchResult(
        results=[
            MultiHopNodeResult(
                uid=r.node.uid,
                hop_distance=r.hop_distance,
                score=r.score,
                properties={
                    k: v for k, v in r.node.properties.items() if k != "embedding"
                },
                entity_types=r.node.entity_types,
            )
            for r in results
        ],
    )


# ---------------------------------------------------------------------------
# 10.2 — Graph-filtered similarity search endpoint
# ---------------------------------------------------------------------------


@graph_router.post(
    "/search/filtered",
    description=(
        "Search the knowledge graph using vector similarity with optional "
        "graph-based candidate filtering from an anchor node."
    ),
)
async def search_graph_filtered(
    spec: GraphFilteredSearchSpec,
    memmachine: Annotated[MemMachine, Depends(get_memmachine)],
) -> GraphFilteredSearchResult:
    """Graph-filtered similarity search."""
    from memmachine.common.vector_graph_store.data_types import (
        GraphFilter,
        TraversalDirection,
    )
    from memmachine.common.vector_graph_store.graph_traversal_store import (
        GraphTraversalStore,
    )

    try:
        store = await _get_vector_graph_store(memmachine)
    except Exception as e:
        raise RestError(code=500, message=_ERR_GRAPH_STORE_ACCESS, ex=e) from e

    # Resolve embedder and compute query embedding.
    embedder_name = memmachine.resource_manager.config.semantic_memory.embedding_model
    if embedder_name is None:
        raise RestError(code=400, message="No embedding model configured.")

    try:
        embedder = await memmachine.resource_manager.get_embedder(embedder_name)
        query_embedding = await embedder.embed(spec.query)
        embedding_name = f"{embedder.model_id}_{embedder.dimensions}"
    except Exception as e:
        raise RestError(code=500, message="Failed to compute embedding", ex=e) from e

    # Build graph filter if an anchor is provided.
    graph_filter: GraphFilter | None = None
    if spec.anchor_node_uid is not None:
        if not isinstance(store, GraphTraversalStore):
            raise RestError(
                code=501,
                message="Graph-filtered search is not supported by the configured store.",
            )
        graph_filter = GraphFilter(
            anchor_node_uid=spec.anchor_node_uid,
            anchor_collection=spec.anchor_collection or spec.collection,
            relation_types=spec.relation_types,
            max_hops=spec.max_hops,
            direction=TraversalDirection(spec.direction.value),
        )

    try:
        if graph_filter is not None and isinstance(store, GraphTraversalStore):
            nodes = await store.search_graph_filtered_similar_nodes(
                collection=spec.collection,
                embedding_name=embedding_name,
                query_embedding=query_embedding,
                limit=spec.limit,
                graph_filter=graph_filter,
            )
        else:
            nodes = await store.search_similar_nodes(
                collection=spec.collection,
                embedding_name=embedding_name,
                query_embedding=query_embedding,
                limit=spec.limit,
            )
    except Exception as e:
        raise RestError(code=500, message="Graph-filtered search failed", ex=e) from e

    return GraphFilteredSearchResult(
        results=[
            GraphFilteredNodeResult(
                uid=n.uid,
                properties={k: v for k, v in n.properties.items() if k != "embedding"},
                entity_types=n.entity_types,
            )
            for n in nodes
        ],
    )


# ---------------------------------------------------------------------------
# Helper: resolve SemanticStorage from the MemMachine instance
# ---------------------------------------------------------------------------


async def _get_semantic_storage(
    memmachine: MemMachine,
) -> SemanticStorage:
    """Return the SemanticStorage for relationship operations."""
    semantic_manager = await memmachine.resource_manager.get_semantic_manager()
    return await semantic_manager.get_semantic_storage()


# ---------------------------------------------------------------------------
# 11.1 — Create feature relationship endpoint
# ---------------------------------------------------------------------------


@graph_router.post(
    "/relationships",
    description="Create a typed relationship between two semantic features.",
)
async def create_relationship(
    spec: CreateRelationshipSpec,
    memmachine: Annotated[MemMachine, Depends(get_memmachine)],
) -> RelationshipResponse:
    """Create a feature relationship."""
    from memmachine.semantic_memory.storage.feature_relationship_types import (
        FeatureRelationshipType,
    )
    from memmachine.semantic_memory.storage.semantic_relationship_storage import (
        SemanticRelationshipStorage,
    )

    storage = await _get_semantic_storage(memmachine)

    if not isinstance(storage, SemanticRelationshipStorage):
        raise RestError(
            code=501,
            message=_ERR_RELATIONSHIPS_NOT_SUPPORTED,
        )

    rel_type = FeatureRelationshipType(spec.relationship_type.value)

    try:
        await storage.add_feature_relationship(
            source_id=spec.source_id,
            target_id=spec.target_id,
            relationship_type=rel_type,
            confidence=spec.confidence,
            source=spec.source,
        )
    except Exception as e:
        raise RestError(code=500, message="Failed to create relationship", ex=e) from e

    from datetime import UTC

    return RelationshipResponse(
        source_id=spec.source_id,
        target_id=spec.target_id,
        relationship_type=FeatureRelationshipTypeParam(rel_type.value),
        confidence=spec.confidence,
        detected_at=datetime.now(tz=UTC),
        source=spec.source,
    )


# ---------------------------------------------------------------------------
# 11.2 — Get feature relationships endpoint
# ---------------------------------------------------------------------------


@graph_router.post(
    "/relationships/get",
    description="Query relationships for a given feature with optional filters.",
)
async def get_relationships(
    spec: GetRelationshipsSpec,
    memmachine: Annotated[MemMachine, Depends(get_memmachine)],
) -> RelationshipListResponse:
    """Get feature relationships."""
    from memmachine.semantic_memory.storage.feature_relationship_types import (
        FeatureRelationshipType,
        RelationshipDirection,
    )
    from memmachine.semantic_memory.storage.semantic_relationship_storage import (
        SemanticRelationshipStorage,
    )

    storage = await _get_semantic_storage(memmachine)

    if not isinstance(storage, SemanticRelationshipStorage):
        raise RestError(
            code=501,
            message=_ERR_RELATIONSHIPS_NOT_SUPPORTED,
        )

    rel_type = (
        FeatureRelationshipType(spec.relationship_type.value)
        if spec.relationship_type is not None
        else None
    )
    direction = RelationshipDirection(spec.direction.value)

    try:
        results = await storage.get_feature_relationships(
            spec.feature_id,
            relationship_type=rel_type,
            direction=direction,
            min_confidence=spec.min_confidence,
        )
    except Exception as e:
        raise RestError(code=500, message="Failed to query relationships", ex=e) from e

    return RelationshipListResponse(
        relationships=[
            RelationshipResponse(
                source_id=r.source_id,
                target_id=r.target_id,
                relationship_type=FeatureRelationshipTypeParam(
                    r.relationship_type.value
                ),
                confidence=r.confidence,
                detected_at=r.detected_at,
                source=r.source,
            )
            for r in results
        ],
    )


# ---------------------------------------------------------------------------
# 11.3 — Delete feature relationship endpoint
# ---------------------------------------------------------------------------


@graph_router.post(
    "/relationships/delete",
    description="Delete a specific relationship between two features.",
)
async def delete_relationship(
    spec: DeleteRelationshipSpec,
    memmachine: Annotated[MemMachine, Depends(get_memmachine)],
) -> dict[str, str]:
    """Delete a feature relationship."""
    from memmachine.semantic_memory.storage.feature_relationship_types import (
        FeatureRelationshipType,
    )
    from memmachine.semantic_memory.storage.semantic_relationship_storage import (
        SemanticRelationshipStorage,
    )

    storage = await _get_semantic_storage(memmachine)

    if not isinstance(storage, SemanticRelationshipStorage):
        raise RestError(
            code=501,
            message=_ERR_RELATIONSHIPS_NOT_SUPPORTED,
        )

    rel_type = FeatureRelationshipType(spec.relationship_type.value)

    try:
        await storage.delete_feature_relationships(
            source_id=spec.source_id,
            target_id=spec.target_id,
            relationship_type=rel_type,
        )
    except Exception as e:
        raise RestError(code=500, message="Failed to delete relationship", ex=e) from e

    return {"status": "deleted"}


# ---------------------------------------------------------------------------
# 11.4 — Find contradictions endpoint
# ---------------------------------------------------------------------------


@graph_router.post(
    "/contradictions",
    description="Find all CONTRADICTS relationships within a semantic feature set.",
)
async def find_contradictions(
    spec: ContradictionsSpec,
    memmachine: Annotated[MemMachine, Depends(get_memmachine)],
) -> ContradictionsResult:
    """Find contradictions within a feature set."""
    from memmachine.semantic_memory.storage.semantic_relationship_storage import (
        SemanticRelationshipStorage,
    )

    storage = await _get_semantic_storage(memmachine)

    if not isinstance(storage, SemanticRelationshipStorage):
        raise RestError(
            code=501,
            message=_ERR_RELATIONSHIPS_NOT_SUPPORTED,
        )

    try:
        pairs = await storage.find_contradictions(set_id=spec.set_id)
    except Exception as e:
        raise RestError(code=500, message="Failed to find contradictions", ex=e) from e

    return ContradictionsResult(
        contradictions=[
            ContradictionPairResponse(
                feature_id_a=p.feature_id_a,
                feature_id_b=p.feature_id_b,
                confidence=p.confidence,
                detected_at=p.detected_at,
                source=p.source,
            )
            for p in pairs
        ],
    )


# ---------------------------------------------------------------------------
# 12.1 — Dedup proposals endpoint
# ---------------------------------------------------------------------------


@graph_router.post(
    "/dedup/proposals",
    description="List duplicate node proposals for a collection.",
)
async def get_dedup_proposals(
    spec: DedupProposalsSpec,
    memmachine: Annotated[MemMachine, Depends(get_memmachine)],
) -> DedupProposalsResult:
    """Get duplicate proposals from the graph store."""
    from memmachine.common.vector_graph_store.neo4j_vector_graph_store import (
        Neo4jVectorGraphStore,
    )

    try:
        store = await _get_vector_graph_store(memmachine)
    except Exception as e:
        raise RestError(code=500, message=_ERR_GRAPH_STORE_ACCESS, ex=e) from e

    if not isinstance(store, Neo4jVectorGraphStore):
        raise RestError(
            code=501,
            message=_ERR_DEDUP_NOT_SUPPORTED,
        )

    try:
        proposals = await store.get_duplicate_proposals(
            collection=spec.collection,
            min_embedding_similarity=spec.min_embedding_similarity,
            include_auto_merged=spec.include_auto_merged,
        )
    except Exception as e:
        raise RestError(
            code=500, message="Failed to fetch duplicate proposals", ex=e
        ) from e

    return DedupProposalsResult(
        proposals=[
            DuplicateProposalResponse(
                node_uid_a=p.node_uid_a,
                node_uid_b=p.node_uid_b,
                embedding_similarity=p.embedding_similarity,
                property_similarity=p.property_similarity,
                detected_at=p.detected_at,
                auto_merged=p.auto_merged,
            )
            for p in proposals
        ],
    )


# ---------------------------------------------------------------------------
# 12.2 — Dedup resolve endpoint
# ---------------------------------------------------------------------------


@graph_router.post(
    "/dedup/resolve",
    description="Resolve duplicate proposals by merging or dismissing pairs.",
)
async def resolve_dedup(
    spec: DedupResolveSpec,
    memmachine: Annotated[MemMachine, Depends(get_memmachine)],
) -> DedupResolveResult:
    """Resolve duplicate proposals."""
    from memmachine.common.vector_graph_store.data_types import (
        DuplicateResolutionStrategy,
    )
    from memmachine.common.vector_graph_store.neo4j_vector_graph_store import (
        Neo4jVectorGraphStore,
    )

    try:
        store = await _get_vector_graph_store(memmachine)
    except Exception as e:
        raise RestError(code=500, message=_ERR_GRAPH_STORE_ACCESS, ex=e) from e

    if not isinstance(store, Neo4jVectorGraphStore):
        raise RestError(
            code=501,
            message=_ERR_DEDUP_NOT_SUPPORTED,
        )

    pairs = [
        (
            p.node_uid_a,
            p.node_uid_b,
            DuplicateResolutionStrategy(p.strategy.value),
        )
        for p in spec.pairs
    ]

    try:
        await store.resolve_duplicates(collection=spec.collection, pairs=pairs)
    except Exception as e:
        raise RestError(code=500, message="Failed to resolve duplicates", ex=e) from e

    return DedupResolveResult(resolved=len(pairs))


# ---------------------------------------------------------------------------
# 12.3 — PageRank endpoint
# ---------------------------------------------------------------------------


@graph_router.post(
    "/analytics/pagerank",
    description="Compute PageRank scores for nodes in a collection.",
)
async def compute_pagerank(
    spec: PageRankSpec,
    memmachine: Annotated[MemMachine, Depends(get_memmachine)],
) -> PageRankResult:
    """Compute PageRank on graph nodes."""
    from memmachine.common.vector_graph_store.neo4j_vector_graph_store import (
        Neo4jVectorGraphStore,
    )

    try:
        store = await _get_vector_graph_store(memmachine)
    except Exception as e:
        raise RestError(code=500, message=_ERR_GRAPH_STORE_ACCESS, ex=e) from e

    if not isinstance(store, Neo4jVectorGraphStore):
        raise RestError(
            code=501,
            message="PageRank is not supported by the configured store.",
        )

    if not await store.is_gds_available():
        raise RestError(
            code=501,
            message=_ERR_GDS_NOT_AVAILABLE,
        )

    try:
        scores = await store.compute_pagerank(
            collection=spec.collection,
            relation_types=spec.relation_types,
            damping_factor=spec.damping_factor,
            max_iterations=spec.max_iterations,
            write_back=spec.write_back,
            write_property=spec.write_property,
        )
    except Exception as e:
        raise RestError(code=500, message="PageRank computation failed", ex=e) from e

    return PageRankResult(
        scores=[PageRankNodeScore(node_uid=uid, score=score) for uid, score in scores],
    )


# ---------------------------------------------------------------------------
# 12.4 — Community detection endpoint
# ---------------------------------------------------------------------------


@graph_router.post(
    "/analytics/communities",
    description="Detect communities in a collection using the Louvain algorithm.",
)
async def detect_communities(
    spec: CommunitiesSpec,
    memmachine: Annotated[MemMachine, Depends(get_memmachine)],
) -> CommunitiesResult:
    """Detect communities in the graph."""
    from memmachine.common.vector_graph_store.neo4j_vector_graph_store import (
        Neo4jVectorGraphStore,
    )

    try:
        store = await _get_vector_graph_store(memmachine)
    except Exception as e:
        raise RestError(code=500, message=_ERR_GRAPH_STORE_ACCESS, ex=e) from e

    if not isinstance(store, Neo4jVectorGraphStore):
        raise RestError(
            code=501,
            message="Community detection is not supported by the configured store.",
        )

    if not await store.is_gds_available():
        raise RestError(
            code=501,
            message=_ERR_GDS_NOT_AVAILABLE,
        )

    try:
        communities = await store.detect_communities(
            collection=spec.collection,
            relation_types=spec.relation_types,
            max_iterations=spec.max_iterations,
            write_back=spec.write_back,
            write_property=spec.write_property,
        )
    except Exception as e:
        raise RestError(code=500, message="Community detection failed", ex=e) from e

    return CommunitiesResult(
        communities=[
            CommunityGroup(community_id=cid, node_uids=uids)
            for cid, uids in communities.items()
        ],
    )


# ---------------------------------------------------------------------------
# 12.5 — Graph statistics endpoint
# ---------------------------------------------------------------------------


@graph_router.post(
    "/analytics/stats",
    description="Return collection-level graph statistics.",
)
async def graph_stats(
    spec: GraphStatsSpec,
    memmachine: Annotated[MemMachine, Depends(get_memmachine)],
) -> GraphStatsResponse:
    """Return graph statistics for a collection."""
    from memmachine.common.vector_graph_store.neo4j_vector_graph_store import (
        Neo4jVectorGraphStore,
    )

    try:
        store = await _get_vector_graph_store(memmachine)
    except Exception as e:
        raise RestError(code=500, message=_ERR_GRAPH_STORE_ACCESS, ex=e) from e

    if not isinstance(store, Neo4jVectorGraphStore):
        raise RestError(
            code=501,
            message="Graph statistics are not supported by the configured store.",
        )

    try:
        result = await store.graph_stats(collection=spec.collection)
    except Exception as e:
        raise RestError(code=500, message="Graph statistics query failed", ex=e) from e

    return GraphStatsResponse(
        node_count=result.node_count,
        edge_count=result.edge_count,
        avg_degree=result.avg_degree,
        relationship_type_distribution=result.relationship_type_distribution,
        entity_type_distribution=result.entity_type_distribution,
    )


# ---------------------------------------------------------------------------
# 12.6 — Shortest path endpoint
# ---------------------------------------------------------------------------


@graph_router.post(
    "/search/shortest-path",
    description="Find the shortest unweighted path between two nodes.",
)
async def shortest_path(
    spec: ShortestPathSpec,
    memmachine: Annotated[MemMachine, Depends(get_memmachine)],
) -> ShortestPathResponse:
    """Find the shortest path between two nodes."""
    from memmachine.common.vector_graph_store.neo4j_vector_graph_store import (
        Neo4jVectorGraphStore,
    )

    try:
        store = await _get_vector_graph_store(memmachine)
    except Exception as e:
        raise RestError(code=500, message=_ERR_GRAPH_STORE_ACCESS, ex=e) from e

    if not isinstance(store, Neo4jVectorGraphStore):
        raise RestError(
            code=501,
            message="Shortest path is not supported by the configured store.",
        )

    try:
        result = await store.shortest_path(
            collection=spec.collection,
            source_uid=spec.source_uid,
            target_uid=spec.target_uid,
            relation_types=spec.relation_types,
            max_depth=spec.max_depth,
        )
    except Exception as e:
        raise RestError(code=500, message="Shortest path query failed", ex=e) from e

    return ShortestPathResponse(
        path_length=result.path_length,
        nodes=[
            PathNodeResponse(uid=n.uid, properties=n.properties) for n in result.nodes
        ],
        edges=[
            PathEdgeResponse(
                source_uid=e.source_uid,
                target_uid=e.target_uid,
                type=e.type,
                properties=e.properties,
            )
            for e in result.edges
        ],
    )


# ---------------------------------------------------------------------------
# 12.7 — Degree centrality endpoint
# ---------------------------------------------------------------------------


@graph_router.post(
    "/analytics/degree-centrality",
    description="Compute degree centrality for nodes in a collection.",
)
async def degree_centrality(
    spec: DegreeCentralitySpec,
    memmachine: Annotated[MemMachine, Depends(get_memmachine)],
) -> DegreeCentralityResponse:
    """Compute degree centrality for nodes."""
    from memmachine.common.vector_graph_store.neo4j_vector_graph_store import (
        Neo4jVectorGraphStore,
    )

    try:
        store = await _get_vector_graph_store(memmachine)
    except Exception as e:
        raise RestError(code=500, message=_ERR_GRAPH_STORE_ACCESS, ex=e) from e

    if not isinstance(store, Neo4jVectorGraphStore):
        raise RestError(
            code=501,
            message="Degree centrality is not supported by the configured store.",
        )

    try:
        results = await store.degree_centrality(
            collection=spec.collection,
            relation_types=spec.relation_types,
            limit=spec.limit,
        )
    except Exception as e:
        raise RestError(
            code=500, message="Degree centrality computation failed", ex=e
        ) from e

    return DegreeCentralityResponse(
        nodes=[
            DegreeCentralityNodeResponse(
                uid=r.uid,
                in_degree=r.in_degree,
                out_degree=r.out_degree,
                total_degree=r.total_degree,
            )
            for r in results
        ],
    )


# ---------------------------------------------------------------------------
# 12.8 — Subgraph extraction endpoint
# ---------------------------------------------------------------------------


@graph_router.post(
    "/search/subgraph",
    description="Extract the ego-graph neighborhood around an anchor node.",
)
async def extract_subgraph(
    spec: SubgraphSpec,
    memmachine: Annotated[MemMachine, Depends(get_memmachine)],
) -> SubgraphResponse:
    """Extract a subgraph around an anchor node."""
    from memmachine.common.vector_graph_store.neo4j_vector_graph_store import (
        Neo4jVectorGraphStore,
    )

    try:
        store = await _get_vector_graph_store(memmachine)
    except Exception as e:
        raise RestError(code=500, message=_ERR_GRAPH_STORE_ACCESS, ex=e) from e

    if not isinstance(store, Neo4jVectorGraphStore):
        raise RestError(
            code=501,
            message="Subgraph extraction is not supported by the configured store.",
        )

    try:
        result = await store.extract_subgraph(
            collection=spec.collection,
            anchor_uid=spec.anchor_uid,
            max_depth=spec.max_depth,
            relation_types=spec.relation_types,
            node_limit=spec.node_limit,
        )
    except Exception as e:
        raise RestError(code=500, message="Subgraph extraction failed", ex=e) from e

    return SubgraphResponse(
        nodes=[
            SubgraphNodeResponse(uid=n.uid, properties=n.properties)
            for n in result.nodes
        ],
        edges=[
            SubgraphEdgeResponse(
                source_uid=e.source_uid,
                target_uid=e.target_uid,
                type=e.type,
                properties=e.properties,
            )
            for e in result.edges
        ],
    )


# ---------------------------------------------------------------------------
# 12.9 — Betweenness centrality endpoint
# ---------------------------------------------------------------------------


@graph_router.post(
    "/analytics/betweenness-centrality",
    description="Compute betweenness centrality for nodes in a collection.",
)
async def betweenness_centrality(
    spec: BetweennessCentralitySpec,
    memmachine: Annotated[MemMachine, Depends(get_memmachine)],
) -> BetweennessCentralityResponse:
    """Compute betweenness centrality for nodes."""
    from memmachine.common.vector_graph_store.neo4j_vector_graph_store import (
        Neo4jVectorGraphStore,
    )

    try:
        store = await _get_vector_graph_store(memmachine)
    except Exception as e:
        raise RestError(code=500, message=_ERR_GRAPH_STORE_ACCESS, ex=e) from e

    if not isinstance(store, Neo4jVectorGraphStore):
        raise RestError(
            code=501,
            message="Betweenness centrality is not supported by the configured store.",
        )

    if not await store.is_gds_available():
        raise RestError(
            code=501,
            message=_ERR_GDS_NOT_AVAILABLE,
        )

    try:
        scores = await store.betweenness_centrality(
            collection=spec.collection,
            relation_types=spec.relation_types,
            sampling_size=spec.sampling_size,
            write_back=spec.write_back,
            write_property=spec.write_property,
        )
    except Exception as e:
        raise RestError(
            code=500, message="Betweenness centrality computation failed", ex=e
        ) from e

    return BetweennessCentralityResponse(
        scores=[PageRankNodeScore(node_uid=uid, score=score) for uid, score in scores],
    )
