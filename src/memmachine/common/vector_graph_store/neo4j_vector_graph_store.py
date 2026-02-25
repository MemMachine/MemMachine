"""
Neo4j-based vector graph store implementation.

This module provides an asynchronous implementation
of a vector graph store using Neo4j as the backend database.
"""

import asyncio
import logging
import re
import time
from collections.abc import Awaitable, Iterable, Mapping
from datetime import datetime
from enum import Enum
from typing import Any, LiteralString, NamedTuple, cast
from uuid import uuid4

from neo4j import AsyncDriver, Query
from neo4j.graph import Node as Neo4jNode
from pydantic import BaseModel, Field, InstanceOf

from memmachine.common.data_types import FilterValue, OrderedValue, SimilarityMetric
from memmachine.common.filter.filter_parser import (
    And as FilterAnd,
)
from memmachine.common.filter.filter_parser import (
    Comparison as FilterComparison,
)
from memmachine.common.filter.filter_parser import (
    FilterExpr,
)
from memmachine.common.filter.filter_parser import (
    In as FilterIn,
)
from memmachine.common.filter.filter_parser import (
    IsNull as FilterIsNull,
)
from memmachine.common.filter.filter_parser import (
    Not as FilterNot,
)
from memmachine.common.filter.filter_parser import (
    Or as FilterOr,
)
from memmachine.common.metrics_factory import MetricsFactory
from memmachine.common.neo4j_utils import (
    ENTITY_TYPE_PREFIX as _ENTITY_TYPE_PREFIX,
    desanitize_entity_type as _desanitize_entity_type,
    render_comparison,
    sanitize_value_for_neo4j,
    value_from_neo4j,
)
from memmachine.common.neo4j_utils import (
    sanitize_entity_type as _sanitize_entity_type,
)
from memmachine.common.utils import async_locked

from .data_types import (
    DuplicateProposal,
    DuplicateResolutionStrategy,
    Edge,
    EntityType,
    GraphFilter,
    MultiHopResult,
    Node,
    PropertyValue,
    demangle_embedding_name,
    demangle_property_name,
    is_mangled_embedding_name,
    is_mangled_property_name,
    mangle_embedding_name,
    mangle_property_name,
)
from .vector_graph_store import VectorGraphStore

logger = logging.getLogger(__name__)

# Module-level constants for frequently used string literals (SonarQube S1192).
_VECTOR_SIMILARITY_COSINE = "vector.similarity.cosine"
_VECTOR_SIMILARITY_EUCLIDEAN = "vector.similarity.euclidean"
_GDS_NOT_AVAILABLE = "GDS plugin is not available"
_AND_JOINER = " AND "


def _neo4j_query(text: str) -> Query:
    return Query(cast(LiteralString, text))


def _similarity_function_name(metric: SimilarityMetric) -> str:
    """Map a SimilarityMetric to its Neo4j vector similarity function name."""
    match metric:
        case SimilarityMetric.COSINE:
            return _VECTOR_SIMILARITY_COSINE
        case SimilarityMetric.EUCLIDEAN:
            return _VECTOR_SIMILARITY_EUCLIDEAN
        case _:
            return _VECTOR_SIMILARITY_COSINE


def _cosine_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    """Compute cosine similarity between two equal-length vectors."""
    dot = sum(a * b for a, b in zip(vec_a, vec_b, strict=True))
    norm_a = sum(a * a for a in vec_a) ** 0.5
    norm_b = sum(b * b for b in vec_b) ** 0.5
    return dot / (norm_a * norm_b) if norm_a and norm_b else 0.0


def _jaccard_similarity(set_a: set[str], set_b: set[str]) -> float:
    """Compute Jaccard similarity between two sets."""
    union = set_a | set_b
    return len(set_a & set_b) / len(union) if union else 0.0


class ProjectionInfo(NamedTuple):
    """Metadata about a GDS graph projection."""

    name: str
    node_count: int
    relationship_count: int


class GraphStatsResult(NamedTuple):
    """Collection-level graph statistics."""

    node_count: int
    edge_count: int
    avg_degree: float
    relationship_type_distribution: dict[str, int]
    entity_type_distribution: dict[str, int]


class PathNode(NamedTuple):
    """A node along a shortest path."""

    uid: str
    properties: dict[str, Any]


class PathEdge(NamedTuple):
    """An edge along a shortest path."""

    source_uid: str
    target_uid: str
    type: str
    properties: dict[str, Any]


class ShortestPathResult(NamedTuple):
    """Result of a shortest-path query."""

    path_length: int
    nodes: list[PathNode]
    edges: list[PathEdge]


class DegreeCentralityResult(NamedTuple):
    """Degree centrality metrics for a single node."""

    uid: str
    in_degree: int
    out_degree: int
    total_degree: int


class SubgraphNode(NamedTuple):
    """A node in an extracted subgraph."""

    uid: str
    properties: dict[str, Any]


class SubgraphEdge(NamedTuple):
    """An edge in an extracted subgraph."""

    source_uid: str
    target_uid: str
    type: str
    properties: dict[str, Any]


class SubgraphResult(NamedTuple):
    """Result of a subgraph extraction query."""

    nodes: list[SubgraphNode]
    edges: list[SubgraphEdge]


def _first_embedding(node: "Node") -> list[float] | None:
    """Return the first available embedding vector from a node, or None."""
    for emb, _metric in node.embeddings.values():
        return emb
    return None


class Neo4jVectorGraphStoreParams(BaseModel):
    """
    Parameters for Neo4jVectorGraphStore.

    Attributes:
        driver (neo4j.AsyncDriver):
            Async Neo4j driver instance.
        force_exact_similarity_search (bool):
            Whether to force exact similarity search
            (default: False).
        filtered_similarity_search_fudge_factor (int):
            Fudge factor for filtered similarity search
            because Neo4j vector index search does not
            support pre-filtering or filtered search.
            (default: 4).
        exact_similarity_search_fallback_threshold (float):
            Threshold ratio of ANN search results to the search limit
            below which to fall back to exact similarity search
            when performing filtered similarity search
            (default: 0.5).
        range_index_hierarchies (list[list[str]]):
            List of property name hierarchies (lists)
            for which to create range indexes
            applied to all nodes and edges
            (default: []).
        range_index_creation_threshold (int):
            Threshold number of entities
            in a collection or having a relation
            at which range indexes may be created
            (default: 10,000).
        vector_index_creation_threshold (int):
            Threshold number of entities
            in a collection or having a relation
            at which vector indexes may be created
            (default: 10,000).
        metrics_factory (MetricsFactory | None):
            An instance of MetricsFactory for collecting usage metrics
            (default: None).
        user_metrics_labels (dict[str, str]):
            Labels to attach to the collected metrics
            (default: {}).

    """

    driver: InstanceOf[AsyncDriver] = Field(
        ...,
        description="Async Neo4j driver instance",
    )
    force_exact_similarity_search: bool = Field(
        False,
        description="Whether to force exact similarity search",
    )
    filtered_similarity_search_fudge_factor: int = Field(
        4,
        description=(
            "Fudge factor for filtered similarity search "
            "because Neo4j vector index search does not "
            "support pre-filtering or filtered search"
        ),
        gt=0,
    )
    exact_similarity_search_fallback_threshold: float = Field(
        0.5,
        description=(
            "Threshold ratio of ANN search results to the search limit "
            "below which to fall back to exact similarity search "
            "when performing filtered similarity search"
        ),
        ge=0.0,
        le=1.0,
    )
    range_index_hierarchies: list[list[str]] = Field(
        default_factory=list,
        description=(
            "List of property name hierarchies "
            "for which to create range indexes "
            "applied to all nodes and edges"
        ),
    )
    range_index_creation_threshold: int = Field(
        10_000,
        description=(
            "Threshold number of entities "
            "in a collection or having a relation "
            "at which range indexes may be created"
        ),
    )
    vector_index_creation_threshold: int = Field(
        10_000,
        description=(
            "Threshold number of entities "
            "in a collection or having a relation "
            "at which vector indexes may be created"
        ),
    )
    dedup_trigger_threshold: int = Field(
        1000,
        description=(
            "Number of nodes in a collection at which background "
            "duplicate detection is triggered after add_nodes()"
        ),
    )
    dedup_embedding_threshold: float = Field(
        0.95,
        description=(
            "Minimum cosine similarity between two node embeddings "
            "for them to be considered potential duplicates"
        ),
        ge=0.0,
        le=1.0,
    )
    dedup_property_threshold: float = Field(
        0.8,
        description=(
            "Minimum Jaccard similarity between two nodes' property "
            "sets for them to be considered potential duplicates"
        ),
        ge=0.0,
        le=1.0,
    )
    dedup_auto_merge: bool = Field(
        False,
        description=(
            "When True, detected duplicates are automatically merged. "
            "When False, duplicates are recorded as SAME_AS proposals only."
        ),
    )
    pagerank_auto_enabled: bool = Field(
        True,
        description=(
            "When True (and gds_enabled is also True), PageRank is "
            "automatically computed on derivative collections after "
            "add_nodes() once the node count reaches the threshold."
        ),
    )
    pagerank_trigger_threshold: int = Field(
        50,
        description=(
            "Minimum number of nodes in a derivative collection "
            "before automatic PageRank computation is triggered."
        ),
    )
    gds_enabled: bool = Field(
        False,
        description=(
            "Whether to enable Graph Data Science plugin features "
            "(PageRank, Louvain community detection). When False, "
            "is_gds_available() returns False without querying Neo4j."
        ),
    )
    gds_default_damping_factor: float = Field(
        0.85,
        description="Default damping factor for PageRank computation.",
        ge=0.0,
        le=1.0,
    )
    gds_default_max_iterations: int = Field(
        20,
        description="Default maximum iterations for GDS algorithms.",
        gt=0,
    )
    metrics_factory: InstanceOf[MetricsFactory] | None = Field(
        None,
        description="An instance of MetricsFactory for collecting usage metrics",
    )
    user_metrics_labels: dict[str, str] = Field(
        default_factory=dict,
        description="Labels to attach to the collected metrics",
    )


# https://neo4j.com/developer/kb/protecting-against-cypher-injection
# Node labels, relationship types, and property names
# cannot be parameterized.
class Neo4jVectorGraphStore(VectorGraphStore):
    """Asynchronous Neo4j-based implementation of VectorGraphStore."""

    class CacheIndexState(Enum):
        """Index state cached locally (not Neo4j authoritative)."""

        CREATING = 0
        ONLINE = 1

    def __init__(self, params: Neo4jVectorGraphStoreParams) -> None:
        """Initialize the graph store with the provided parameters."""
        super().__init__()

        self._driver: AsyncDriver = params.driver

        self._force_exact_similarity_search = params.force_exact_similarity_search
        self._filtered_similarity_search_fudge_factor = (
            params.filtered_similarity_search_fudge_factor
        )
        self._exact_similarity_search_fallback_threshold = (
            params.exact_similarity_search_fallback_threshold
        )
        self._range_index_hierarchies = params.range_index_hierarchies
        self._range_index_creation_threshold = params.range_index_creation_threshold

        self._vector_index_creation_threshold = params.vector_index_creation_threshold
        self._dedup_trigger_threshold = params.dedup_trigger_threshold
        self._dedup_embedding_threshold = params.dedup_embedding_threshold
        self._dedup_property_threshold = params.dedup_property_threshold
        self._dedup_auto_merge = params.dedup_auto_merge
        self._pagerank_auto_enabled = params.pagerank_auto_enabled
        self._pagerank_trigger_threshold = params.pagerank_trigger_threshold
        self._gds_enabled = params.gds_enabled
        self._gds_default_damping_factor = params.gds_default_damping_factor
        self._gds_default_max_iterations = params.gds_default_max_iterations

        self._index_state_cache: dict[str, Neo4jVectorGraphStore.CacheIndexState] = {}
        self._populate_index_state_cache_lock = asyncio.Lock()

        # These are only used for tracking counts approximately.
        self._collection_node_counts: dict[str, int] = {}
        self._relation_edge_counts: dict[str, int] = {}

        self._background_tasks: set[asyncio.Task] = set()
        self._constrained_collections: set[str] = set()
        self._dedup_pending_collections: set[str] = set()
        self._pagerank_pending_collections: set[str] = set()

        metrics_factory = params.metrics_factory

        self._add_nodes_calls_counter = None
        self._add_nodes_latency_summary = None
        self._add_edges_calls_counter = None
        self._add_edges_latency_summary = None
        self._search_similar_nodes_calls_counter = None
        self._search_similar_nodes_latency_summary = None
        self._search_related_nodes_calls_counter = None
        self._search_related_nodes_latency_summary = None
        self._search_directional_nodes_calls_counter = None
        self._search_directional_nodes_latency_summary = None
        self._search_matching_nodes_calls_counter = None
        self._search_matching_nodes_latency_summary = None
        self._get_nodes_calls_counter = None
        self._get_nodes_latency_summary = None
        self._delete_nodes_calls_counter = None
        self._delete_nodes_latency_summary = None
        self._count_nodes_calls_counter = None
        self._count_nodes_latency_summary = None
        self._count_edges_calls_counter = None
        self._count_edges_latency_summary = None
        self._populate_index_state_cache_calls_counter = None
        self._populate_index_state_cache_latency_summary = None
        self._create_initial_indexes_if_not_exist_calls_counter = None
        self._create_initial_indexes_if_not_exist_latency_summary = None
        self._create_range_index_if_not_exists_calls_counter = None
        self._create_range_index_if_not_exists_latency_summary = None
        self._create_vector_index_if_not_exists_calls_counter = None
        self._create_vector_index_if_not_exists_latency_summary = None
        self._search_multi_hop_nodes_calls_counter = None
        self._search_multi_hop_nodes_latency_summary = None
        self._search_graph_filtered_similar_nodes_calls_counter = None
        self._search_graph_filtered_similar_nodes_latency_summary = None
        self._compute_pagerank_calls_counter = None
        self._compute_pagerank_latency_summary = None
        self._detect_communities_calls_counter = None
        self._detect_communities_latency_summary = None
        self._graph_stats_calls_counter = None
        self._graph_stats_latency_summary = None
        self._shortest_path_calls_counter = None
        self._shortest_path_latency_summary = None
        self._degree_centrality_calls_counter = None
        self._degree_centrality_latency_summary = None
        self._extract_subgraph_calls_counter = None
        self._extract_subgraph_latency_summary = None
        self._betweenness_centrality_calls_counter = None
        self._betweenness_centrality_latency_summary = None

        self._should_collect_metrics = False
        if metrics_factory is not None:
            self._should_collect_metrics = True
            self._user_metrics_labels = params.user_metrics_labels
            label_names = self._user_metrics_labels.keys()

            self._add_nodes_calls_counter = metrics_factory.get_counter(
                "vector_graph_store_neo4j_add_nodes_calls",
                "Number of calls to add_nodes in Neo4jVectorGraphStore",
                label_names=label_names,
            )
            self._add_nodes_latency_summary = metrics_factory.get_summary(
                "vector_graph_store_neo4j_add_nodes_latency_seconds",
                "Latency in seconds for add_nodes in Neo4jVectorGraphStore",
                label_names=label_names,
            )
            self._add_edges_calls_counter = metrics_factory.get_counter(
                "vector_graph_store_neo4j_add_edges_calls",
                "Number of calls to add_edges in Neo4jVectorGraphStore",
                label_names=label_names,
            )
            self._add_edges_latency_summary = metrics_factory.get_summary(
                "vector_graph_store_neo4j_add_edges_latency_seconds",
                "Latency in seconds for add_edges in Neo4jVectorGraphStore",
                label_names=label_names,
            )

            self._search_similar_nodes_calls_counter = metrics_factory.get_counter(
                "vector_graph_store_neo4j_search_similar_nodes_calls",
                "Number of calls to search_similar_nodes in Neo4jVectorGraphStore",
                label_names=label_names,
            )
            self._search_similar_nodes_latency_summary = metrics_factory.get_summary(
                "vector_graph_store_neo4j_search_similar_nodes_latency_seconds",
                "Latency in seconds for search_similar_nodes in Neo4jVectorGraphStore",
                label_names=label_names,
            )
            self._search_related_nodes_calls_counter = metrics_factory.get_counter(
                "vector_graph_store_neo4j_search_related_nodes_calls",
                "Number of calls to search_related_nodes in Neo4jVectorGraphStore",
                label_names=label_names,
            )
            self._search_related_nodes_latency_summary = metrics_factory.get_summary(
                "vector_graph_store_neo4j_search_related_nodes_latency_seconds",
                "Latency in seconds for search_related_nodes in Neo4jVectorGraphStore",
                label_names=label_names,
            )
            self._search_directional_nodes_calls_counter = metrics_factory.get_counter(
                "vector_graph_store_neo4j_search_directional_nodes_calls",
                "Number of calls to search_directional_nodes in Neo4jVectorGraphStore",
                label_names=label_names,
            )
            self._search_directional_nodes_latency_summary = metrics_factory.get_summary(
                "vector_graph_store_neo4j_search_directional_nodes_latency_seconds",
                "Latency in seconds for search_directional_nodes in Neo4jVectorGraphStore",
                label_names=label_names,
            )
            self._search_matching_nodes_calls_counter = metrics_factory.get_counter(
                "vector_graph_store_neo4j_search_matching_nodes_calls",
                "Number of calls to search_matching_nodes in Neo4jVectorGraphStore",
                label_names=label_names,
            )
            self._search_matching_nodes_latency_summary = metrics_factory.get_summary(
                "vector_graph_store_neo4j_search_matching_nodes_latency_seconds",
                "Latency in seconds for search_matching_nodes in Neo4jVectorGraphStore",
                label_names=label_names,
            )

            self._get_nodes_calls_counter = metrics_factory.get_counter(
                "vector_graph_store_neo4j_get_nodes_calls",
                "Number of calls to get_nodes in Neo4jVectorGraphStore",
                label_names=label_names,
            )
            self._get_nodes_latency_summary = metrics_factory.get_summary(
                "vector_graph_store_neo4j_get_nodes_latency_seconds",
                "Latency in seconds for get_nodes in Neo4jVectorGraphStore",
                label_names=label_names,
            )

            self._delete_nodes_calls_counter = metrics_factory.get_counter(
                "vector_graph_store_neo4j_delete_nodes_calls",
                "Number of calls to delete_nodes in Neo4jVectorGraphStore",
                label_names=label_names,
            )
            self._delete_nodes_latency_summary = metrics_factory.get_summary(
                "vector_graph_store_neo4j_delete_nodes_latency_seconds",
                "Latency in seconds for delete_nodes in Neo4jVectorGraphStore",
                label_names=label_names,
            )

            self._count_nodes_calls_counter = metrics_factory.get_counter(
                "vector_graph_store_neo4j_count_nodes_calls",
                "Number of calls to count_nodes in Neo4jVectorGraphStore",
                label_names=label_names,
            )
            self._count_nodes_latency_summary = metrics_factory.get_summary(
                "vector_graph_store_neo4j_count_nodes_latency_seconds",
                "Latency in seconds for count_nodes in Neo4jVectorGraphStore",
                label_names=label_names,
            )
            self._count_edges_calls_counter = metrics_factory.get_counter(
                "vector_graph_store_neo4j_count_edges_calls",
                "Number of calls to count_edges in Neo4jVectorGraphStore",
                label_names=label_names,
            )
            self._count_edges_latency_summary = metrics_factory.get_summary(
                "vector_graph_store_neo4j_count_edges_latency_seconds",
                "Latency in seconds for count_edges in Neo4jVectorGraphStore",
                label_names=label_names,
            )

            self._populate_index_state_cache_calls_counter = metrics_factory.get_counter(
                "vector_graph_store_neo4j_populate_index_state_cache_calls",
                "Number of calls to _populate_index_state_cache in Neo4jVectorGraphStore",
                label_names=label_names,
            )
            self._populate_index_state_cache_latency_summary = metrics_factory.get_summary(
                "vector_graph_store_neo4j_populate_index_state_cache_latency_seconds",
                "Latency in seconds for _populate_index_state_cache in Neo4jVectorGraphStore",
                label_names=label_names,
            )

            self._create_initial_indexes_if_not_exist_calls_counter = metrics_factory.get_counter(
                "vector_graph_store_neo4j_create_initial_indexes_if_not_exist_calls",
                "Number of calls to _create_initial_indexes_if_not_exist in Neo4jVectorGraphStore",
                label_names=label_names,
            )
            self._create_initial_indexes_if_not_exist_latency_summary = metrics_factory.get_summary(
                "vector_graph_store_neo4j_create_initial_indexes_if_not_exist_latency_seconds",
                "Latency in seconds for _create_initial_indexes_if_not_exist in Neo4jVectorGraphStore",
                label_names=label_names,
            )
            self._create_range_index_if_not_exists_calls_counter = metrics_factory.get_counter(
                "vector_graph_store_neo4j_create_range_index_if_not_exists_calls",
                "Number of calls to _create_range_index_if_not_exists in Neo4jVectorGraphStore",
                label_names=label_names,
            )
            self._create_range_index_if_not_exists_latency_summary = metrics_factory.get_summary(
                "vector_graph_store_neo4j_create_range_index_if_not_exists_latency_seconds",
                "Latency in seconds for _create_range_index_if_not_exists in Neo4jVectorGraphStore",
                label_names=label_names,
            )
            self._create_vector_index_if_not_exists_calls_counter = metrics_factory.get_counter(
                "vector_graph_store_neo4j_create_vector_index_if_not_exists_calls",
                "Number of calls to _create_vector_index_if_not_exists in Neo4jVectorGraphStore",
                label_names=label_names,
            )
            self._create_vector_index_if_not_exists_latency_summary = metrics_factory.get_summary(
                "vector_graph_store_neo4j_create_vector_index_if_not_exists_latency_seconds",
                "Latency in seconds for _create_vector_index_if_not_exists in Neo4jVectorGraphStore",
                label_names=label_names,
            )

            self._search_multi_hop_nodes_calls_counter = metrics_factory.get_counter(
                "vector_graph_store_neo4j_search_multi_hop_nodes_calls",
                "Number of calls to search_multi_hop_nodes in Neo4jVectorGraphStore",
                label_names=label_names,
            )
            self._search_multi_hop_nodes_latency_summary = metrics_factory.get_summary(
                "vector_graph_store_neo4j_search_multi_hop_nodes_latency_seconds",
                "Latency in seconds for search_multi_hop_nodes in Neo4jVectorGraphStore",
                label_names=label_names,
            )

            self._search_graph_filtered_similar_nodes_calls_counter = metrics_factory.get_counter(
                "vector_graph_store_neo4j_search_graph_filtered_similar_nodes_calls",
                "Number of calls to search_graph_filtered_similar_nodes in Neo4jVectorGraphStore",
                label_names=label_names,
            )
            self._search_graph_filtered_similar_nodes_latency_summary = metrics_factory.get_summary(
                "vector_graph_store_neo4j_search_graph_filtered_similar_nodes_latency_seconds",
                "Latency in seconds for search_graph_filtered_similar_nodes in Neo4jVectorGraphStore",
                label_names=label_names,
            )

            self._compute_pagerank_calls_counter = metrics_factory.get_counter(
                "vector_graph_store_neo4j_compute_pagerank_calls",
                "Number of calls to compute_pagerank in Neo4jVectorGraphStore",
                label_names=label_names,
            )
            self._compute_pagerank_latency_summary = metrics_factory.get_summary(
                "vector_graph_store_neo4j_compute_pagerank_latency_seconds",
                "Latency in seconds for compute_pagerank in Neo4jVectorGraphStore",
                label_names=label_names,
            )

            self._detect_communities_calls_counter = metrics_factory.get_counter(
                "vector_graph_store_neo4j_detect_communities_calls",
                "Number of calls to detect_communities in Neo4jVectorGraphStore",
                label_names=label_names,
            )
            self._detect_communities_latency_summary = metrics_factory.get_summary(
                "vector_graph_store_neo4j_detect_communities_latency_seconds",
                "Latency in seconds for detect_communities in Neo4jVectorGraphStore",
                label_names=label_names,
            )

            self._gds_projection_node_count_gauge = metrics_factory.get_gauge(
                "vector_graph_store_neo4j_gds_projection_node_count",
                "Number of nodes in the last GDS graph projection",
                label_names=label_names,
            )
            self._gds_projection_relationship_count_gauge = metrics_factory.get_gauge(
                "vector_graph_store_neo4j_gds_projection_relationship_count",
                "Number of relationships in the last GDS graph projection",
                label_names=label_names,
            )

            self._graph_stats_calls_counter = metrics_factory.get_counter(
                "vector_graph_store_neo4j_graph_stats_calls",
                "Number of calls to graph_stats in Neo4jVectorGraphStore",
                label_names=label_names,
            )
            self._graph_stats_latency_summary = metrics_factory.get_summary(
                "vector_graph_store_neo4j_graph_stats_latency_seconds",
                "Latency in seconds for graph_stats in Neo4jVectorGraphStore",
                label_names=label_names,
            )

            self._shortest_path_calls_counter = metrics_factory.get_counter(
                "vector_graph_store_neo4j_shortest_path_calls",
                "Number of calls to shortest_path in Neo4jVectorGraphStore",
                label_names=label_names,
            )
            self._shortest_path_latency_summary = metrics_factory.get_summary(
                "vector_graph_store_neo4j_shortest_path_latency_seconds",
                "Latency in seconds for shortest_path in Neo4jVectorGraphStore",
                label_names=label_names,
            )

            self._degree_centrality_calls_counter = metrics_factory.get_counter(
                "vector_graph_store_neo4j_degree_centrality_calls",
                "Number of calls to degree_centrality in Neo4jVectorGraphStore",
                label_names=label_names,
            )
            self._degree_centrality_latency_summary = metrics_factory.get_summary(
                "vector_graph_store_neo4j_degree_centrality_latency_seconds",
                "Latency in seconds for degree_centrality in Neo4jVectorGraphStore",
                label_names=label_names,
            )

            self._extract_subgraph_calls_counter = metrics_factory.get_counter(
                "vector_graph_store_neo4j_extract_subgraph_calls",
                "Number of calls to extract_subgraph in Neo4jVectorGraphStore",
                label_names=label_names,
            )
            self._extract_subgraph_latency_summary = metrics_factory.get_summary(
                "vector_graph_store_neo4j_extract_subgraph_latency_seconds",
                "Latency in seconds for extract_subgraph in Neo4jVectorGraphStore",
                label_names=label_names,
            )

            self._betweenness_centrality_calls_counter = metrics_factory.get_counter(
                "vector_graph_store_neo4j_betweenness_centrality_calls",
                "Number of calls to betweenness_centrality in Neo4jVectorGraphStore",
                label_names=label_names,
            )
            self._betweenness_centrality_latency_summary = metrics_factory.get_summary(
                "vector_graph_store_neo4j_betweenness_centrality_latency_seconds",
                "Latency in seconds for betweenness_centrality in Neo4jVectorGraphStore",
                label_names=label_names,
            )

    def _track_task(self, task: asyncio.Task) -> None:
        """Keep background tasks from being garbage collected prematurely."""
        self._background_tasks.add(task)
        task.add_done_callback(self._background_tasks.discard)

    async def _ensure_collection_constraints(
        self,
        sanitized_collection: str,
    ) -> None:
        """Create a uniqueness constraint on uid for a collection if not exists.

        Uses ``CREATE CONSTRAINT IF NOT EXISTS`` so this is idempotent and safe
        to call on every first-use of a collection.
        """
        if sanitized_collection in self._constrained_collections:
            return

        constraint_name = f"unique_uid_{sanitized_collection}"
        await self._driver.execute_query(
            _neo4j_query(
                f"CREATE CONSTRAINT {constraint_name} IF NOT EXISTS "
                f"FOR (n:{sanitized_collection}) REQUIRE n.uid IS UNIQUE"
            ),
        )
        self._constrained_collections.add(sanitized_collection)

    async def add_nodes(
        self,
        *,
        collection: str,
        nodes: Iterable[Node],
    ) -> None:
        """Add nodes to a collection, creating indexes as needed."""
        start_time = time.monotonic()

        if collection not in self._collection_node_counts:
            # Not async-safe but it's not crucial if the count is off.
            self._collection_node_counts[collection] = await self._count_nodes(
                collection,
            )

        sanitized_collection = Neo4jVectorGraphStore._sanitize_name(collection)
        sanitized_embedding_names = set()
        embedding_dimensions_by_name: dict[str, int] = {}
        embedding_similarity_by_name: dict[str, SimilarityMetric] = {}

        query_nodes = []
        # Collect distinct sets of sanitized entity type labels.
        entity_type_label_sets: dict[tuple[str, ...], list[dict]] = {}

        for node in nodes:
            query_node_properties = Neo4jVectorGraphStore._sanitize_properties(
                {
                    mangle_property_name(key): value
                    for key, value in node.properties.items()
                },
            )

            for embedding_name, (
                embedding,
                similarity_metric,
            ) in node.embeddings.items():
                sanitized_embedding_name = Neo4jVectorGraphStore._sanitize_name(
                    mangle_embedding_name(embedding_name),
                )
                sanitized_similarity_metric_name = Neo4jVectorGraphStore._sanitize_name(
                    Neo4jVectorGraphStore._similarity_metric_property_name(
                        embedding_name,
                    ),
                )

                sanitized_embedding_names.add(sanitized_embedding_name)
                embedding_dimensions_by_name[sanitized_embedding_name] = len(
                    embedding,
                )
                embedding_similarity_by_name[sanitized_embedding_name] = (
                    similarity_metric
                )

                query_node_properties[sanitized_embedding_name] = embedding
                query_node_properties[sanitized_similarity_metric_name] = (
                    similarity_metric.value
                )

            query_node: dict[str, PropertyValue | dict[str, PropertyValue]] = {
                "uid": str(node.uid),
                "properties": query_node_properties,
            }
            query_nodes.append(query_node)

            # Group nodes by their entity type label set for batched Cypher.
            sanitized_types = tuple(
                sorted(
                    Neo4jVectorGraphStore._sanitize_entity_type(et)
                    for et in node.entity_types
                )
            )
            entity_type_label_sets.setdefault(sanitized_types, []).append(query_node)

        # Ensure uniqueness constraint exists for this collection.
        await self._ensure_collection_constraints(sanitized_collection)

        # Execute MERGE per distinct entity-type label set (most common case:
        # all nodes share the same set, so this is a single batch).
        for type_labels, batch_nodes in entity_type_label_sets.items():
            extra_labels = "".join(f":{lbl}" for lbl in type_labels)
            await self._driver.execute_query(
                _neo4j_query(
                    "UNWIND $nodes AS node\n"
                    f"MERGE (n:{sanitized_collection} {{uid: node.uid}})\n"
                    "SET n += node.properties"
                    + (f"\nSET n{extra_labels}" if extra_labels else "")
                ),
                nodes=batch_nodes,
            )

        self._collection_node_counts[collection] += len(query_nodes)

        self._maybe_create_node_indexes(
            collection=collection,
            sanitized_collection=sanitized_collection,
            sanitized_embedding_names=sanitized_embedding_names,
            embedding_dimensions_by_name=embedding_dimensions_by_name,
            embedding_similarity_by_name=embedding_similarity_by_name,
        )
        self._maybe_trigger_dedup(collection)
        self._maybe_trigger_pagerank(collection)

        end_time = time.monotonic()
        self._collect_metrics(
            self._add_nodes_calls_counter,
            self._add_nodes_latency_summary,
            start_time,
            end_time,
        )

    def _maybe_create_node_indexes(
        self,
        *,
        collection: str,
        sanitized_collection: str,
        sanitized_embedding_names: set[str],
        embedding_dimensions_by_name: dict[str, int],
        embedding_similarity_by_name: dict[str, SimilarityMetric],
    ) -> None:
        """Schedule background index creation tasks if thresholds are met."""
        if (
            self._collection_node_counts[collection]
            >= self._range_index_creation_threshold
        ):
            self._track_task(
                asyncio.create_task(
                    self._create_initial_indexes_if_not_exist(
                        EntityType.NODE,
                        sanitized_collection,
                    ),
                )
            )

        if (
            self._collection_node_counts[collection]
            >= self._vector_index_creation_threshold
        ):
            for sanitized_embedding_name in sanitized_embedding_names:
                if (
                    Neo4jVectorGraphStore._index_name(
                        EntityType.NODE,
                        sanitized_collection,
                        sanitized_embedding_name,
                    )
                    not in self._index_state_cache
                ):
                    self._track_task(
                        asyncio.create_task(
                            self._create_vector_index_if_not_exists(
                                entity_type=EntityType.NODE,
                                sanitized_collection_or_relation=sanitized_collection,
                                sanitized_embedding_name=sanitized_embedding_name,
                                dimensions=embedding_dimensions_by_name[
                                    sanitized_embedding_name
                                ],
                                similarity_metric=embedding_similarity_by_name[
                                    sanitized_embedding_name
                                ],
                            ),
                        )
                    )

    def _maybe_trigger_dedup(self, collection: str) -> None:
        """Schedule background dedup if the collection exceeds the threshold."""
        if (
            self._dedup_trigger_threshold > 0
            and self._collection_node_counts[collection]
            >= self._dedup_trigger_threshold
            and collection not in self._dedup_pending_collections
        ):
            self._dedup_pending_collections.add(collection)
            self._track_task(
                asyncio.create_task(
                    self._run_dedup_for_collection(collection),
                )
            )

    def _maybe_trigger_pagerank(self, collection: str) -> None:
        """Schedule background PageRank if conditions are met."""
        if (
            self._pagerank_auto_enabled
            and self._gds_enabled
            and collection.startswith("Derivative_")
            and self._pagerank_trigger_threshold > 0
            and self._collection_node_counts.get(collection, 0)
            >= self._pagerank_trigger_threshold
            and collection not in self._pagerank_pending_collections
        ):
            self._pagerank_pending_collections.add(collection)
            self._track_task(
                asyncio.create_task(
                    self._run_pagerank_for_collection(collection),
                )
            )

    async def _run_pagerank_for_collection(self, collection: str) -> None:
        """Background task: compute PageRank and write scores to nodes."""
        try:
            await self.compute_pagerank(
                collection=collection,
                damping_factor=self._gds_default_damping_factor,
                max_iterations=self._gds_default_max_iterations,
                write_back=True,
            )
        except Exception:
            logger.warning(
                "Background PageRank failed for collection %r",
                collection,
                exc_info=True,
            )
        finally:
            self._pagerank_pending_collections.discard(collection)

    async def add_edges(
        self,
        *,
        relation: str,
        source_collection: str,
        target_collection: str,
        edges: Iterable[Edge],
    ) -> None:
        """Add edges between collections, creating indexes as needed."""
        start_time = time.monotonic()

        if relation not in self._relation_edge_counts:
            # Not async-safe but it's not crucial if the count is off.
            self._relation_edge_counts[relation] = await self._count_edges(relation)

        sanitized_relation = Neo4jVectorGraphStore._sanitize_name(relation)
        sanitized_embedding_names = set()
        embedding_dimensions_by_name: dict[str, int] = {}
        embedding_similarity_by_name: dict[str, SimilarityMetric] = {}

        query_edges = []
        for edge in edges:
            query_edge_properties = Neo4jVectorGraphStore._sanitize_properties(
                {
                    mangle_property_name(key): value
                    for key, value in edge.properties.items()
                },
            )

            for embedding_name, (
                embedding,
                similarity_metric,
            ) in edge.embeddings.items():
                sanitized_embedding_name = Neo4jVectorGraphStore._sanitize_name(
                    mangle_embedding_name(embedding_name),
                )
                sanitized_similarity_metric_name = Neo4jVectorGraphStore._sanitize_name(
                    Neo4jVectorGraphStore._similarity_metric_property_name(
                        embedding_name,
                    ),
                )

                sanitized_embedding_names.add(sanitized_embedding_name)
                embedding_dimensions_by_name[sanitized_embedding_name] = len(
                    embedding,
                )
                embedding_similarity_by_name[sanitized_embedding_name] = (
                    similarity_metric
                )

                query_edge_properties[sanitized_embedding_name] = embedding
                query_edge_properties[sanitized_similarity_metric_name] = (
                    similarity_metric.value
                )

            query_edge = {
                "uid": str(edge.uid),
                "source_uid": str(edge.source_uid),
                "target_uid": str(edge.target_uid),
                "properties": query_edge_properties,
            }
            query_edges.append(query_edge)

        sanitized_source_collection = Neo4jVectorGraphStore._sanitize_name(
            source_collection,
        )
        sanitized_target_collection = Neo4jVectorGraphStore._sanitize_name(
            target_collection,
        )
        await self._driver.execute_query(
            _neo4j_query(
                "UNWIND $edges AS edge\n"
                "MATCH"
                f"    (source:{sanitized_source_collection} {{uid: edge.source_uid}}),"
                f" (target:{sanitized_target_collection} {{uid: edge.target_uid}})\n"
                "MERGE (source)"
                f"    -[r:{sanitized_relation} {{uid: edge.uid}}]->"
                "    (target)\n"
                "SET r += edge.properties"
            ),
            edges=query_edges,
        )

        self._relation_edge_counts[relation] += len(query_edges)

        if self._relation_edge_counts[relation] >= self._range_index_creation_threshold:
            self._track_task(
                asyncio.create_task(
                    self._create_initial_indexes_if_not_exist(
                        EntityType.EDGE,
                        sanitized_relation,
                    ),
                )
            )

        if (
            self._relation_edge_counts[relation]
            >= self._vector_index_creation_threshold
        ):
            for sanitized_embedding_name in sanitized_embedding_names:
                if (
                    Neo4jVectorGraphStore._index_name(
                        EntityType.EDGE,
                        sanitized_relation,
                        sanitized_embedding_name,
                    )
                    not in self._index_state_cache
                ):
                    self._track_task(
                        asyncio.create_task(
                            self._create_vector_index_if_not_exists(
                                entity_type=EntityType.EDGE,
                                sanitized_collection_or_relation=sanitized_relation,
                                sanitized_embedding_name=sanitized_embedding_name,
                                dimensions=embedding_dimensions_by_name[
                                    sanitized_embedding_name
                                ],
                                similarity_metric=embedding_similarity_by_name[
                                    sanitized_embedding_name
                                ],
                            ),
                        )
                    )

        end_time = time.monotonic()
        self._collect_metrics(
            self._add_edges_calls_counter,
            self._add_edges_latency_summary,
            start_time,
            end_time,
        )

    async def _ann_search(
        self,
        *,
        vector_index_name: str,
        query_embedding: list[float],
        limit: int,
        query_filter_string: str,
        query_filter_params: dict,
        entity_types: list[str] | None,
        property_filter: FilterExpr | None,
    ) -> tuple[list[Any], bool]:
        """Execute ANN vector search and return (records, should_fallback)."""
        entity_type_where = _AND_JOINER.join(
            f"n:{Neo4jVectorGraphStore._sanitize_entity_type(et)}"
            for et in (entity_types or [])
        )
        entity_type_ann_filter = (
            f"AND {entity_type_where}\n" if entity_type_where else ""
        )

        query = (
            "CALL db.index.vector.queryNodes(\n"
            "    $vector_index_name, $query_limit, $query_embedding\n"
            ")\n"
            "YIELD node AS n, score AS similarity\n"
            f"WHERE {query_filter_string}\n"
            f"{entity_type_ann_filter}"
            "RETURN n\n"
            "ORDER BY similarity DESC\n"
            "LIMIT $limit"
        )

        has_post_filter = property_filter is not None or bool(entity_types)
        records, _, _ = await self._driver.execute_query(
            _neo4j_query(query),
            query_embedding=query_embedding,
            query_limit=(
                limit
                if not has_post_filter
                else limit * self._filtered_similarity_search_fudge_factor
            ),
            limit=limit,
            query_filter_params=query_filter_params,
            vector_index_name=vector_index_name,
        )

        should_fallback = (
            has_post_filter
            and len(records) < limit * self._exact_similarity_search_fallback_threshold
        )
        return records, should_fallback

    async def _exact_similarity_search(
        self,
        *,
        sanitized_collection: str,
        sanitized_embedding_name: str,
        entity_type_label_fragment: str,
        similarity_metric: SimilarityMetric,
        query_embedding: list[float],
        limit: int | None,
        query_filter_string: str,
        query_filter_params: dict,
    ) -> list[Any]:
        """Execute exact (brute-force) vector similarity search."""
        vector_similarity_function = _similarity_function_name(similarity_metric)

        query = (
            f"MATCH (n:{sanitized_collection}{entity_type_label_fragment})\n"
            f"WHERE n.{sanitized_embedding_name} IS NOT NULL\n"
            f"AND {query_filter_string}\n"
            "With n,"
            f"    {vector_similarity_function}("
            f"        n.{sanitized_embedding_name}, $query_embedding"
            "    ) AS similarity\n"
            "RETURN n\n"
            "ORDER BY similarity DESC\n"
            f"{'LIMIT $limit' if limit is not None else ''}"
        )

        records, _, _ = await self._driver.execute_query(
            _neo4j_query(query),
            query_embedding=query_embedding,
            limit=limit,
            query_filter_params=query_filter_params,
        )
        return records

    async def search_similar_nodes(
        self,
        *,
        collection: str,
        embedding_name: str,
        query_embedding: list[float],
        similarity_metric: SimilarityMetric = SimilarityMetric.COSINE,
        limit: int | None = 100,
        property_filter: FilterExpr | None = None,
        entity_types: list[str] | None = None,
    ) -> list[Node]:
        """Search nodes by vector similarity with optional property filters.

        When *entity_types* is provided, only nodes that have **all** of the
        specified entity type labels are matched.  This narrows the Cypher
        ``MATCH`` pattern so Neo4j can use label-based index scans.
        """
        start_time = time.monotonic()

        sanitized_collection = Neo4jVectorGraphStore._sanitize_name(collection)
        sanitized_embedding_name = Neo4jVectorGraphStore._sanitize_name(
            mangle_embedding_name(embedding_name),
        )
        entity_type_label_fragment = Neo4jVectorGraphStore._entity_type_label_fragment(
            entity_types,
        )
        vector_index_name = Neo4jVectorGraphStore._index_name(
            EntityType.NODE,
            sanitized_collection,
            sanitized_embedding_name,
        )

        query_filter_string, query_filter_params = (
            Neo4jVectorGraphStore._build_query_filter(
                "n",
                "query_filter_params",
                property_filter,
            )
        )

        do_exact_similarity_search = self._force_exact_similarity_search

        if not do_exact_similarity_search:
            await self._populate_index_state_cache()
            if (
                self._index_state_cache.get(vector_index_name)
                != Neo4jVectorGraphStore.CacheIndexState.ONLINE
            ):
                do_exact_similarity_search = True

        records: list[Any] = []

        if not do_exact_similarity_search:
            if limit is None:
                limit = 1000
            records, should_fallback = await self._ann_search(
                vector_index_name=vector_index_name,
                query_embedding=query_embedding,
                limit=limit,
                query_filter_string=query_filter_string,
                query_filter_params=query_filter_params,
                entity_types=entity_types,
                property_filter=property_filter,
            )
            if should_fallback:
                do_exact_similarity_search = True

        if do_exact_similarity_search:
            records = await self._exact_similarity_search(
                sanitized_collection=sanitized_collection,
                sanitized_embedding_name=sanitized_embedding_name,
                entity_type_label_fragment=entity_type_label_fragment,
                similarity_metric=similarity_metric,
                query_embedding=query_embedding,
                limit=limit,
                query_filter_string=query_filter_string,
                query_filter_params=query_filter_params,
            )

        similar_neo4j_nodes = [record["n"] for record in records]
        similar_nodes = Neo4jVectorGraphStore._nodes_from_neo4j_nodes(
            similar_neo4j_nodes
        )

        end_time = time.monotonic()
        self._collect_metrics(
            self._search_similar_nodes_calls_counter,
            self._search_similar_nodes_latency_summary,
            start_time,
            end_time,
        )

        return similar_nodes

    async def search_related_nodes(
        self,
        *,
        relation: str,
        other_collection: str,
        this_collection: str,
        this_node_uid: str,
        find_sources: bool = True,
        find_targets: bool = True,
        limit: int | None = None,
        edge_property_filter: FilterExpr | None = None,
        node_property_filter: FilterExpr | None = None,
    ) -> list[Node]:
        """Search nodes connected by a relation with optional property filters."""
        start_time = time.monotonic()

        edge_query_filter_string, edge_query_filter_params = (
            Neo4jVectorGraphStore._build_query_filter(
                "r",
                "edge_query_filter_params",
                edge_property_filter,
            )
        )
        node_query_filter_string, node_query_filter_params = (
            Neo4jVectorGraphStore._build_query_filter(
                "n",
                "node_query_filter_params",
                node_property_filter,
            )
        )

        if not (find_sources or find_targets):
            end_time = time.monotonic()
            self._collect_metrics(
                self._search_related_nodes_calls_counter,
                self._search_related_nodes_latency_summary,
                start_time,
                end_time,
            )
            return []

        sanitized_this_collection = Neo4jVectorGraphStore._sanitize_name(
            this_collection,
        )
        sanitized_other_collection = Neo4jVectorGraphStore._sanitize_name(
            other_collection,
        )
        sanitized_relation = Neo4jVectorGraphStore._sanitize_name(relation)

        records, _, _ = await self._driver.execute_query(
            _neo4j_query(
                "MATCH\n"
                f"    (m:{sanitized_this_collection} {{uid: $node_uid}})"
                f"    {'-' if find_targets else '<-'}"
                f"    [r:{sanitized_relation}]"
                f"    {'-' if find_sources else '->'}"
                f"    (n:{sanitized_other_collection})"
                f"WHERE {edge_query_filter_string}\n"
                f"AND {node_query_filter_string}\n"
                "RETURN DISTINCT n\n"
                f"{'LIMIT $limit' if limit is not None else ''}"
            ),
            node_uid=str(this_node_uid),
            limit=limit,
            edge_query_filter_params=edge_query_filter_params,
            node_query_filter_params=node_query_filter_params,
        )

        related_neo4j_nodes = [record["n"] for record in records]
        related_nodes = Neo4jVectorGraphStore._nodes_from_neo4j_nodes(
            related_neo4j_nodes
        )

        end_time = time.monotonic()
        self._collect_metrics(
            self._search_related_nodes_calls_counter,
            self._search_related_nodes_latency_summary,
            start_time,
            end_time,
        )

        return related_nodes

    async def search_directional_nodes(
        self,
        *,
        collection: str,
        by_properties: Iterable[str],
        starting_at: Iterable[OrderedValue | None],
        order_ascending: Iterable[bool],
        include_equal_start: bool = False,
        limit: int | None = 1,
        property_filter: FilterExpr | None = None,
        entity_types: list[str] | None = None,
    ) -> list[Node]:
        """Find nodes ordered by property values in a chosen direction.

        When *entity_types* is provided, only nodes with **all** specified
        labels are matched in the Cypher ``MATCH`` pattern.
        """
        start_time = time.monotonic()

        by_properties = list(by_properties)
        starting_at = list(starting_at)
        order_ascending = list(order_ascending)

        if not (len(by_properties) == len(starting_at) == len(order_ascending) > 0):
            raise ValueError(
                "Lengths of "
                "by_properties, starting_at, and order_ascending "
                "must be equal and greater than 0.",
            )

        sanitized_collection = Neo4jVectorGraphStore._sanitize_name(collection)
        entity_type_label_fragment = Neo4jVectorGraphStore._entity_type_label_fragment(
            entity_types,
        )
        sanitized_by_properties = [
            Neo4jVectorGraphStore._sanitize_name(mangle_property_name(by_property))
            for by_property in by_properties
        ]

        query_relational_requirements = (
            Neo4jVectorGraphStore._query_lexicographic_relational_requirements(
                "n",
                "starting_at",
                sanitized_by_properties,
                starting_at,
                order_ascending,
            )
            + (
                (
                    " OR ("
                    + (
                        _AND_JOINER.join(
                            render_comparison(
                                f"n.{sanitized_by_property}",
                                "=",
                                f"$starting_at[{index}]",
                                starting_value,
                            )
                            for index, sanitized_by_property in enumerate(
                                sanitized_by_properties,
                            )
                            if (starting_value := starting_at[index]) is not None
                        )
                        or "TRUE"  # All starting_at values are None → no constraints
                    )
                    + ")"
                )
                if include_equal_start
                else ""
            )
        )

        query_order_by = (
            "ORDER BY "
            + ", ".join(
                f"n.{sanitized_by_property} {
                    'ASC' if order_ascending[index] else 'DESC'
                }"
                for index, sanitized_by_property in enumerate(sanitized_by_properties)
            )
            + "\n"
        )

        query_filter_string, query_filter_params = (
            Neo4jVectorGraphStore._build_query_filter(
                "n",
                "query_filter_params",
                property_filter,
            )
        )

        records, _, _ = await self._driver.execute_query(
            _neo4j_query(
                f"MATCH (n:{sanitized_collection}{entity_type_label_fragment})\n"
                f"WHERE ({query_relational_requirements})\n"
                f"AND {query_filter_string}\n"
                "RETURN n\n"
                f"{query_order_by}"
                f"{'LIMIT $limit' if limit is not None else ''}"
            ),
            starting_at=[
                sanitize_value_for_neo4j(starting_at_value)
                for starting_at_value in starting_at
            ],
            limit=limit,
            query_filter_params=query_filter_params,
        )

        directional_proximal_neo4j_nodes = [record["n"] for record in records]
        directional_proximal_nodes = Neo4jVectorGraphStore._nodes_from_neo4j_nodes(
            directional_proximal_neo4j_nodes,
        )

        end_time = time.monotonic()
        self._collect_metrics(
            self._search_directional_nodes_calls_counter,
            self._search_directional_nodes_latency_summary,
            start_time,
            end_time,
        )

        return directional_proximal_nodes

    @staticmethod
    def _query_lexicographic_relational_requirements(
        entity_query_alias: str,
        starting_at_query_parameter: str,
        sanitized_by_properties: Iterable[str],
        starting_at: Iterable[OrderedValue | None],
        order_ascending: Iterable[bool],
    ) -> str:
        sanitized_by_properties = list(sanitized_by_properties)
        starting_at = list(starting_at)
        order_ascending = list(order_ascending)

        lexicographic_relational_requirements = []
        for index, sanitized_by_property in enumerate(sanitized_by_properties):
            sanitized_equal_properties = sanitized_by_properties[:index]

            # The same points in time with different timezones are not equal in Neo4j,
            # so we use epochSeconds and nanosecond for datetime comparisons.
            # https://neo4j.com/docs/cypher-manual/current/values-and-types/ordering-equality-comparison/#ordering-spatial-temporal

            starting_value = starting_at[index]
            if starting_value is None:
                relational_requirements = [
                    f"{entity_query_alias}.{sanitized_by_property} IS NOT NULL",
                ]
            else:
                relational_requirements = [
                    render_comparison(
                        f"{entity_query_alias}.{sanitized_by_property}",
                        ">" if order_ascending[index] else "<",
                        f"${starting_at_query_parameter}[{index}]",
                        starting_value,
                    )
                ]

            for equal_index, sanitized_equal_property in enumerate(
                sanitized_equal_properties,
            ):
                starting_value = starting_at[equal_index]
                if starting_value is None:
                    relational_requirements += [
                        f"{entity_query_alias}.{sanitized_equal_property} IS NOT NULL"
                    ]
                else:
                    relational_requirements += [
                        render_comparison(
                            f"{entity_query_alias}.{sanitized_equal_property}",
                            "=",
                            f"${starting_at_query_parameter}[{equal_index}]",
                            starting_value,
                        )
                    ]

            lexicographic_relational_requirement = (
                f"({_AND_JOINER.join(relational_requirements)})"
            )

            lexicographic_relational_requirements.append(
                lexicographic_relational_requirement,
            )

        query_lexicographic_relational_requirements = (
            f"({' OR '.join(lexicographic_relational_requirements)})"
        )

        return query_lexicographic_relational_requirements

    async def search_matching_nodes(
        self,
        *,
        collection: str,
        limit: int | None = None,
        property_filter: FilterExpr | None = None,
        entity_types: list[str] | None = None,
    ) -> list[Node]:
        """Search nodes that match the provided property filters.

        When *entity_types* is provided, only nodes with **all** specified
        labels are matched in the Cypher ``MATCH`` pattern.
        """
        start_time = time.monotonic()

        sanitized_collection = Neo4jVectorGraphStore._sanitize_name(collection)
        entity_type_label_fragment = Neo4jVectorGraphStore._entity_type_label_fragment(
            entity_types,
        )

        query_filter_string, query_filter_params = (
            Neo4jVectorGraphStore._build_query_filter(
                "n",
                "query_filter_params",
                property_filter,
            )
        )

        records, _, _ = await self._driver.execute_query(
            _neo4j_query(
                f"MATCH (n:{sanitized_collection}{entity_type_label_fragment})\n"
                f"WHERE {query_filter_string}\n"
                "RETURN n\n"
                f"{'LIMIT $limit' if limit is not None else ''}"
            ),
            limit=limit,
            query_filter_params=query_filter_params,
        )

        matching_neo4j_nodes = [record["n"] for record in records]
        matching_nodes = Neo4jVectorGraphStore._nodes_from_neo4j_nodes(
            matching_neo4j_nodes
        )

        end_time = time.monotonic()
        self._collect_metrics(
            self._search_matching_nodes_calls_counter,
            self._search_matching_nodes_latency_summary,
            start_time,
            end_time,
        )

        return matching_nodes

    async def get_nodes(
        self,
        *,
        collection: str,
        node_uids: Iterable[str],
    ) -> list[Node]:
        """Retrieve nodes by uid from a specific collection."""
        start_time = time.monotonic()

        sanitized_collection = Neo4jVectorGraphStore._sanitize_name(collection)

        records, _, _ = await self._driver.execute_query(
            _neo4j_query(
                "UNWIND $node_uids AS node_uid\n"
                f"MATCH (n:{sanitized_collection} {{uid: node_uid}})\n"
                "RETURN n"
            ),
            node_uids=[str(node_uid) for node_uid in node_uids],
        )

        neo4j_nodes = [record["n"] for record in records]
        nodes = Neo4jVectorGraphStore._nodes_from_neo4j_nodes(neo4j_nodes)

        end_time = time.monotonic()
        self._collect_metrics(
            self._get_nodes_calls_counter,
            self._get_nodes_latency_summary,
            start_time,
            end_time,
        )

        return nodes

    async def update_entity_types(
        self,
        *,
        collection: str,
        node_uid: str,
        entity_types: list[str],
    ) -> None:
        """Set the entity type labels on an existing node.

        The provided *entity_types* list becomes the definitive set of entity
        type labels for the node.  Labels in the list that already exist on the
        node are kept (no-op).  Labels currently on the node that are **not**
        in the list are removed.  Labels in the list that are not yet on the
        node are added.

        The collection label and any internal (``SANITIZED_``-prefixed) labels
        are never affected.
        """
        sanitized_collection = Neo4jVectorGraphStore._sanitize_name(collection)
        desired_labels = {
            Neo4jVectorGraphStore._sanitize_entity_type(et) for et in entity_types
        }

        # Fetch the node's current labels so we can diff.
        records, _, _ = await self._driver.execute_query(
            _neo4j_query(
                f"MATCH (n:{sanitized_collection} {{uid: $uid}})\n"
                "RETURN labels(n) AS labels"
            ),
            uid=str(node_uid),
        )

        if not records:
            logger.warning(
                "update_entity_types: node %s not found in collection %s",
                node_uid,
                collection,
            )
            return

        current_labels: set[str] = set(records[0]["labels"])
        current_entity_labels = {
            lbl
            for lbl in current_labels
            if lbl.startswith(Neo4jVectorGraphStore._ENTITY_TYPE_PREFIX)
        }

        labels_to_add = desired_labels - current_entity_labels
        labels_to_remove = current_entity_labels - desired_labels

        set_clauses: list[str] = []
        if labels_to_add:
            add_fragment = "".join(f":{lbl}" for lbl in sorted(labels_to_add))
            set_clauses.append(f"SET n{add_fragment}")
        if labels_to_remove:
            set_clauses.extend(f"REMOVE n:{lbl}" for lbl in sorted(labels_to_remove))

        if not set_clauses:
            return  # Nothing to change.

        query = f"MATCH (n:{sanitized_collection} {{uid: $uid}})\n" + "\n".join(
            set_clauses
        )

        await self._driver.execute_query(
            _neo4j_query(query),
            uid=str(node_uid),
        )

    # Maximum number of hops allowed for multi-hop traversal to prevent
    # runaway queries.  The caller's ``max_hops`` is clamped to this ceiling.
    _MAX_HOPS_CEILING = 5

    async def search_multi_hop_nodes(
        self,
        *,
        collection: str,
        this_node_uid: str,
        min_hops: int = 1,
        max_hops: int = 3,
        relation_types: Iterable[str] | None = None,
        raw_relation_types: Iterable[str] | None = None,
        score_decay: float = 0.7,
        limit: int | None = None,
        edge_property_filter: FilterExpr | None = None,
        node_property_filter: FilterExpr | None = None,
        target_collections: Iterable[str] | None = None,
    ) -> list[MultiHopResult]:
        """Search for nodes reachable via multi-hop relationship traversal.

        Uses a Cypher variable-length path pattern.  Each result node appears
        at most once using the shortest path distance.  Results are scored
        with ``score_decay ** hop_distance`` and ordered by score descending.
        """
        start_time = time.monotonic()

        # Clamp max_hops to the configured ceiling.
        max_hops = min(max_hops, self._MAX_HOPS_CEILING)
        min_hops = max(1, min(min_hops, max_hops))

        sanitized_collection = Neo4jVectorGraphStore._sanitize_name(collection)

        # Build relationship type pattern.
        # Combine sanitized relation_types with raw_relation_types (already
        # in their final Neo4j form) so cross-layer traversals work.
        all_rel_types: list[str] = []

        relation_types_list: list[str] | None = (
            list(relation_types) if relation_types is not None else None
        )
        if relation_types_list:
            all_rel_types.extend(
                Neo4jVectorGraphStore._sanitize_name(rt) for rt in relation_types_list
            )

        raw_relation_types_list: list[str] | None = (
            list(raw_relation_types) if raw_relation_types is not None else None
        )
        if raw_relation_types_list:
            all_rel_types.extend(raw_relation_types_list)

        if all_rel_types:
            rel_type_pattern = "|".join(all_rel_types)
            rel_pattern = f"[r:{rel_type_pattern}*{min_hops}..{max_hops}]"
        else:
            rel_pattern = f"[*{min_hops}..{max_hops}]"

        # Build property filters for edges and nodes.
        edge_filter_string, edge_filter_params = (
            Neo4jVectorGraphStore._build_query_filter(
                "r_item",
                "edge_filter_params",
                edge_property_filter,
            )
        )
        node_filter_string, node_filter_params = (
            Neo4jVectorGraphStore._build_query_filter(
                "end_node",
                "node_filter_params",
                node_property_filter,
            )
        )

        # Cypher: find all shortest paths from the start node, annotate
        # hop distance, apply filters, deduplicate to shortest distance.
        edge_where_clause = ""
        if edge_property_filter is not None:
            edge_where_clause = (
                f"AND ALL(r_item IN relationships(path) WHERE {edge_filter_string})\n"
            )

        node_where_clause = ""
        if node_property_filter is not None:
            node_where_clause = f"AND {node_filter_string}\n"

        # Build optional target-collection constraint for end nodes.
        target_collections_clause = ""
        if target_collections is not None:
            target_list = list(target_collections)
            if target_list:
                sanitized_targets = [
                    Neo4jVectorGraphStore._sanitize_name(tc) for tc in target_list
                ]
                label_checks = " OR ".join(f"end_node:{st}" for st in sanitized_targets)
                target_collections_clause = f"AND ({label_checks})\n"
            else:
                # Empty list means no valid target -- return nothing.
                target_collections_clause = "AND false\n"

        # The Cypher query computes *path_quality* — the minimum
        # ``RELATED_TO`` edge similarity along each traversal path.
        # This differentiates semantically specific connections (e.g.
        # ``uses_tensorflow → specializes_in_tensorflow``, sim=0.81)
        # from trivial same-name bridges (``name → name``, sim=NULL).
        #
        # Paths that contain **no** ``RELATED_TO`` edges at all get
        # quality 0.0.  These are shortcuts through shared Features
        # (e.g. a Feature extracted from multiple Episodes) that
        # reach many nodes without any semantic bridging.
        #
        # Structural edges (``DERIVED_FROM``, ``EXTRACTED_FROM``)
        # are transparent (quality 1.0).  ``RELATED_TO`` edges with
        # no ``similarity`` property (same-feature-name edges)
        # receive quality 0.0 so they don't inflate the score.
        #
        # For each ``end_node`` the query keeps the path with the
        # **best** quality (then shortest hops as tiebreaker).
        query = (
            f"MATCH path = (start:{sanitized_collection} {{uid: $start_uid}})"
            f"-{rel_pattern}-"
            "(end_node)\n"
            "WHERE start <> end_node\n"
            f"{target_collections_clause}"
            f"{edge_where_clause}"
            f"{node_where_clause}"
            "WITH end_node, length(path) AS hops,\n"
            # Count RELATED_TO edges — paths without any are
            # shortcuts through shared Features and must be
            # excluded (quality 0.0).
            "     size([r IN relationships(path)"
            " WHERE type(r) = 'RELATED_TO']) AS rt_count,\n"
            "     reduce(q = 1.0, r IN relationships(path) |\n"
            "       CASE WHEN type(r) = 'RELATED_TO'\n"
            "            THEN CASE WHEN r.similarity IS NOT NULL\n"
            "                      THEN CASE WHEN r.similarity < q"
            " THEN r.similarity ELSE q END\n"
            "                      ELSE 0.0\n"
            "                 END\n"
            "            ELSE q\n"
            "       END\n"
            "     ) AS raw_quality\n"
            "WITH end_node, hops,\n"
            "     CASE WHEN rt_count > 0"
            " THEN raw_quality ELSE 0.0 END AS path_quality\n"
            "ORDER BY end_node, path_quality DESC, hops ASC\n"
            "WITH end_node,"
            " collect({hops: hops, quality: path_quality})[0] AS best\n"
            "RETURN end_node, best.hops AS hop_distance,"
            " best.quality AS path_quality\n"
            "ORDER BY path_quality DESC, hop_distance ASC\n"
            + (f"LIMIT {limit}" if limit is not None else "")
        )

        records, _, _ = await self._driver.execute_query(
            _neo4j_query(query),
            start_uid=str(this_node_uid),
            edge_filter_params=edge_filter_params,
            node_filter_params=node_filter_params,
        )

        # Build results with decay scoring.
        results: list[MultiHopResult] = []
        neo4j_nodes = [record["end_node"] for record in records]
        nodes = Neo4jVectorGraphStore._nodes_from_neo4j_nodes(neo4j_nodes)

        for node, record in zip(nodes, records, strict=True):
            hop_distance: int = record["hop_distance"]
            pq: float = float(record["path_quality"])
            score = score_decay**hop_distance * pq
            results.append(
                MultiHopResult(
                    node=node,
                    hop_distance=hop_distance,
                    score=score,
                    path_quality=pq,
                ),
            )

        # Sort by score descending (higher score = closer / more important).
        results.sort(key=lambda r: r.score, reverse=True)

        end_time = time.monotonic()
        self._collect_metrics(
            self._search_multi_hop_nodes_calls_counter,
            self._search_multi_hop_nodes_latency_summary,
            start_time,
            end_time,
        )

        return results

    async def search_graph_filtered_similar_nodes(
        self,
        *,
        collection: str,
        embedding_name: str,
        query_embedding: list[float],
        similarity_metric: SimilarityMetric = SimilarityMetric.COSINE,
        limit: int | None = 100,
        property_filter: FilterExpr | None = None,
        graph_filter: GraphFilter | None = None,
    ) -> list[Node]:
        """Search for similar nodes with optional graph-structure pre-filtering.

        When *graph_filter* is provided the implementation first narrows the
        candidate set by traversing the graph from the anchor node, then
        computes vector similarity only against those candidates.

        When *graph_filter* is ``None`` the method falls back to the standard
        ``search_similar_nodes()`` behaviour.
        """
        start_time = time.monotonic()

        if graph_filter is None:
            # Fallback to standard vector search.
            results = await self.search_similar_nodes(
                collection=collection,
                embedding_name=embedding_name,
                query_embedding=query_embedding,
                similarity_metric=similarity_metric,
                limit=limit,
                property_filter=property_filter,
            )
            end_time = time.monotonic()
            self._collect_metrics(
                self._search_graph_filtered_similar_nodes_calls_counter,
                self._search_graph_filtered_similar_nodes_latency_summary,
                start_time,
                end_time,
            )
            return results

        # Phase 1: Graph pre-filter — collect candidate element IDs.
        sanitized_anchor_collection = Neo4jVectorGraphStore._sanitize_name(
            graph_filter.anchor_collection,
        )
        max_hops = min(graph_filter.max_hops, self._MAX_HOPS_CEILING)

        if graph_filter.relation_types:
            sanitized_rel_types = [
                Neo4jVectorGraphStore._sanitize_name(rt)
                for rt in graph_filter.relation_types
            ]
            rel_type_pattern = "|".join(sanitized_rel_types)
            rel_pattern = f"[:{rel_type_pattern}*1..{max_hops}]"
        else:
            rel_pattern = f"[*1..{max_hops}]"

        # Build direction-aware pattern.
        match graph_filter.direction:
            case graph_filter.direction.OUTGOING:
                path_pattern = (
                    f"(anchor:{sanitized_anchor_collection} "
                    f"{{uid: $anchor_uid}})-{rel_pattern}->(candidate)"
                )
            case graph_filter.direction.INCOMING:
                path_pattern = (
                    f"(anchor:{sanitized_anchor_collection} "
                    f"{{uid: $anchor_uid}})<-{rel_pattern}-(candidate)"
                )
            case _:
                path_pattern = (
                    f"(anchor:{sanitized_anchor_collection} "
                    f"{{uid: $anchor_uid}})-{rel_pattern}-(candidate)"
                )

        sanitized_collection = Neo4jVectorGraphStore._sanitize_name(collection)

        candidate_query = (
            f"MATCH {path_pattern}\n"
            f"WHERE candidate:{sanitized_collection}\n"
            "RETURN DISTINCT elementId(candidate) AS eid"
        )

        candidate_records, _, _ = await self._driver.execute_query(
            _neo4j_query(candidate_query),
            anchor_uid=str(graph_filter.anchor_node_uid),
        )

        candidate_eids = [r["eid"] for r in candidate_records]

        if not candidate_eids:
            end_time = time.monotonic()
            self._collect_metrics(
                self._search_graph_filtered_similar_nodes_calls_counter,
                self._search_graph_filtered_similar_nodes_latency_summary,
                start_time,
                end_time,
            )
            return []

        # Phase 2: Vector similarity on candidates only.
        sanitized_embedding_name = Neo4jVectorGraphStore._sanitize_name(
            mangle_embedding_name(embedding_name),
        )

        query_filter_string, query_filter_params = (
            Neo4jVectorGraphStore._build_query_filter(
                "n",
                "query_filter_params",
                property_filter,
            )
        )

        vector_similarity_function = _similarity_function_name(similarity_metric)

        query = (
            f"MATCH (n:{sanitized_collection})\n"
            "WHERE elementId(n) IN $candidate_eids\n"
            f"AND n.{sanitized_embedding_name} IS NOT NULL\n"
            f"AND {query_filter_string}\n"
            "WITH n,"
            f"    {vector_similarity_function}("
            f"        n.{sanitized_embedding_name}, $query_embedding"
            "    ) AS similarity\n"
            "RETURN n\n"
            "ORDER BY similarity DESC\n"
            f"{'LIMIT $limit' if limit is not None else ''}"
        )

        records, _, _ = await self._driver.execute_query(
            _neo4j_query(query),
            candidate_eids=candidate_eids,
            query_embedding=query_embedding,
            limit=limit,
            query_filter_params=query_filter_params,
        )

        similar_neo4j_nodes = [record["n"] for record in records]
        results = Neo4jVectorGraphStore._nodes_from_neo4j_nodes(
            similar_neo4j_nodes,
        )

        end_time = time.monotonic()
        self._collect_metrics(
            self._search_graph_filtered_similar_nodes_calls_counter,
            self._search_graph_filtered_similar_nodes_latency_summary,
            start_time,
            end_time,
        )

        return results

    # ------------------------------------------------------------------
    # Entity Deduplication
    # ------------------------------------------------------------------

    async def _detect_duplicates(
        self,
        collection: str,
    ) -> list[DuplicateProposal]:
        """Detect potential duplicate nodes in a collection.

        Computes pairwise cosine similarity on embeddings and Jaccard
        similarity on property key sets.  Only pairs exceeding **both**
        configured thresholds are returned.
        """
        from datetime import UTC

        sanitized_collection = Neo4jVectorGraphStore._sanitize_name(collection)

        # Fetch all nodes with at least one embedding.
        records, _, _ = await self._driver.execute_query(
            _neo4j_query(f"MATCH (n:{sanitized_collection})\nRETURN n"),
        )

        if len(records) < 2:
            return []

        neo4j_nodes = [record["n"] for record in records]
        nodes = Neo4jVectorGraphStore._nodes_from_neo4j_nodes(neo4j_nodes)

        # Build embedding vectors and property key sets for comparison.
        node_data: list[tuple[Node, list[float] | None, set[str]]] = [
            (node, _first_embedding(node), set(node.properties.keys()))
            for node in nodes
        ]

        proposals: list[DuplicateProposal] = []
        now = datetime.now(UTC)

        for i in range(len(node_data)):
            for j in range(i + 1, len(node_data)):
                proposal = self._compare_node_pair(node_data[i], node_data[j], now)
                if proposal is not None:
                    proposals.append(proposal)

        return proposals

    def _compare_node_pair(
        self,
        pair_a: tuple[Node, list[float] | None, set[str]],
        pair_b: tuple[Node, list[float] | None, set[str]],
        detected_at: datetime,
    ) -> DuplicateProposal | None:
        """Compare two nodes and return a DuplicateProposal if they exceed thresholds."""
        node_a, emb_a, props_a = pair_a
        node_b, emb_b, props_b = pair_b

        # Cosine similarity on embeddings.
        if emb_a is not None and emb_b is not None and len(emb_a) == len(emb_b):
            emb_sim = _cosine_similarity(emb_a, emb_b)
        else:
            emb_sim = 0.0

        prop_sim = _jaccard_similarity(props_a, props_b)

        if (
            emb_sim >= self._dedup_embedding_threshold
            and prop_sim >= self._dedup_property_threshold
        ):
            return DuplicateProposal(
                node_uid_a=node_a.uid,
                node_uid_b=node_b.uid,
                embedding_similarity=emb_sim,
                property_similarity=prop_sim,
                detected_at=detected_at,
            )

        return None

    async def _create_same_as_relationships(
        self,
        collection: str,
        proposals: list[DuplicateProposal],
    ) -> None:
        """Create or update SAME_AS relationships for detected duplicates."""
        sanitized_collection = Neo4jVectorGraphStore._sanitize_name(collection)

        for proposal in proposals:
            await self._driver.execute_query(
                _neo4j_query(
                    f"MATCH (a:{sanitized_collection} {{uid: $uid_a}})\n"
                    f"MATCH (b:{sanitized_collection} {{uid: $uid_b}})\n"
                    "MERGE (a)-[r:SAME_AS]-(b)\n"
                    "SET r.embedding_similarity = $emb_sim,\n"
                    "    r.property_similarity = $prop_sim,\n"
                    "    r.detected_at = datetime(),\n"
                    "    r.auto_merged = $auto_merged"
                ),
                uid_a=proposal.node_uid_a,
                uid_b=proposal.node_uid_b,
                emb_sim=proposal.embedding_similarity,
                prop_sim=proposal.property_similarity,
                auto_merged=proposal.auto_merged,
            )

    async def _auto_merge_duplicates(
        self,
        collection: str,
        proposals: list[DuplicateProposal],
    ) -> None:
        """Auto-merge detected duplicates (newer properties win)."""
        sanitized_collection = Neo4jVectorGraphStore._sanitize_name(collection)

        for proposal in proposals:
            # Merge b into a: copy properties, repoint relationships, delete b.
            await self._driver.execute_query(
                _neo4j_query(
                    f"MATCH (a:{sanitized_collection} {{uid: $uid_a}})\n"
                    f"MATCH (b:{sanitized_collection} {{uid: $uid_b}})\n"
                    "SET a += properties(b)"
                ),
                uid_a=proposal.node_uid_a,
                uid_b=proposal.node_uid_b,
            )

            # Repoint incoming relationships from b to a.
            await self._driver.execute_query(
                _neo4j_query(
                    f"MATCH (b:{sanitized_collection} {{uid: $uid_b}})\n"
                    f"MATCH (a:{sanitized_collection} {{uid: $uid_a}})\n"
                    "MATCH (other)-[r]->(b)\n"
                    "WHERE other <> a\n"
                    "WITH other, r, a, type(r) AS rtype, "
                    "properties(r) AS rprops\n"
                    "CALL apoc.create.relationship("
                    "  other, rtype, rprops, a"
                    ") YIELD rel\n"
                    "DELETE r"
                ),
                uid_a=proposal.node_uid_a,
                uid_b=proposal.node_uid_b,
            )

            # Repoint outgoing relationships from b to a.
            await self._driver.execute_query(
                _neo4j_query(
                    f"MATCH (b:{sanitized_collection} {{uid: $uid_b}})\n"
                    f"MATCH (a:{sanitized_collection} {{uid: $uid_a}})\n"
                    "MATCH (b)-[r]->(other)\n"
                    "WHERE other <> a\n"
                    "WITH other, r, a, type(r) AS rtype, "
                    "properties(r) AS rprops\n"
                    "CALL apoc.create.relationship("
                    "  a, rtype, rprops, other"
                    ") YIELD rel\n"
                    "DELETE r"
                ),
                uid_a=proposal.node_uid_a,
                uid_b=proposal.node_uid_b,
            )

            # Delete the duplicate node.
            await self._driver.execute_query(
                _neo4j_query(
                    f"MATCH (b:{sanitized_collection} {{uid: $uid_b}})\nDETACH DELETE b"
                ),
                uid_b=proposal.node_uid_b,
            )

            proposal.auto_merged = True

    async def _run_dedup_for_collection(self, collection: str) -> None:
        """Background task: detect and handle duplicates for a collection."""
        try:
            proposals = await self._detect_duplicates(collection)
            if not proposals:
                return

            if self._dedup_auto_merge:
                await self._auto_merge_duplicates(collection, proposals)

            # Always record SAME_AS relationships (with auto_merged flag).
            await self._create_same_as_relationships(collection, proposals)
        except Exception:
            logger.exception("Background dedup failed for collection %r", collection)
        else:
            # Dedup may have merged nodes, so re-trigger PageRank to
            # recompute scores on the updated graph topology.
            self._pagerank_pending_collections.discard(collection)
            self._maybe_trigger_pagerank(collection)
        finally:
            self._dedup_pending_collections.discard(collection)

    async def get_duplicate_proposals(
        self,
        *,
        collection: str,
        min_embedding_similarity: float | None = None,
        include_auto_merged: bool = False,
    ) -> list[DuplicateProposal]:
        """Query SAME_AS relationships for duplicate proposals."""
        sanitized_collection = Neo4jVectorGraphStore._sanitize_name(collection)

        where_clauses = []
        if min_embedding_similarity is not None:
            where_clauses.append(
                f"r.embedding_similarity >= {min_embedding_similarity}"
            )
        if not include_auto_merged:
            where_clauses.append("r.auto_merged = false")

        where_string = ""
        if where_clauses:
            where_string = "WHERE " + _AND_JOINER.join(where_clauses) + "\n"

        records, _, _ = await self._driver.execute_query(
            _neo4j_query(
                f"MATCH (a:{sanitized_collection})-[r:SAME_AS]"
                f"-(b:{sanitized_collection})\n"
                f"{where_string}"
                "RETURN a.uid AS uid_a, b.uid AS uid_b, "
                "r.embedding_similarity AS emb_sim, "
                "r.property_similarity AS prop_sim, "
                "r.detected_at AS detected_at, "
                "r.auto_merged AS auto_merged"
            ),
        )

        proposals: list[DuplicateProposal] = []
        seen: set[tuple[str, str]] = set()

        for rec in records:
            uid_pair = (
                min(rec["uid_a"], rec["uid_b"]),
                max(rec["uid_a"], rec["uid_b"]),
            )
            if uid_pair in seen:
                continue
            seen.add(uid_pair)

            detected_at_raw = rec["detected_at"]
            if hasattr(detected_at_raw, "to_native"):
                detected_at: datetime = detected_at_raw.to_native()
            else:
                detected_at = detected_at_raw

            proposals.append(
                DuplicateProposal(
                    node_uid_a=uid_pair[0],
                    node_uid_b=uid_pair[1],
                    embedding_similarity=float(rec["emb_sim"]),
                    property_similarity=float(rec["prop_sim"]),
                    detected_at=detected_at,
                    auto_merged=bool(rec["auto_merged"]),
                )
            )

        return proposals

    async def resolve_duplicates(
        self,
        *,
        collection: str,
        pairs: list[tuple[str, str, DuplicateResolutionStrategy]],
    ) -> None:
        """Resolve duplicate proposals by merging or dismissing.

        Each element in *pairs* is ``(uid_a, uid_b, strategy)``.
        """
        sanitized_collection = Neo4jVectorGraphStore._sanitize_name(collection)

        for uid_a, uid_b, strategy in pairs:
            match strategy:
                case DuplicateResolutionStrategy.MERGE:
                    # Merge b into a.
                    await self._driver.execute_query(
                        _neo4j_query(
                            f"MATCH (a:{sanitized_collection} {{uid: $uid_a}})\n"
                            f"MATCH (b:{sanitized_collection} {{uid: $uid_b}})\n"
                            "SET a += properties(b)"
                        ),
                        uid_a=uid_a,
                        uid_b=uid_b,
                    )
                    await self._driver.execute_query(
                        _neo4j_query(
                            f"MATCH (b:{sanitized_collection} {{uid: $uid_b}})\n"
                            "DETACH DELETE b"
                        ),
                        uid_b=uid_b,
                    )

                case DuplicateResolutionStrategy.DISMISS:
                    # Remove the SAME_AS relationship but keep both nodes.
                    await self._driver.execute_query(
                        _neo4j_query(
                            f"MATCH (a:{sanitized_collection} {{uid: $uid_a}})"
                            f"-[r:SAME_AS]-"
                            f"(b:{sanitized_collection} {{uid: $uid_b}})\n"
                            "DELETE r"
                        ),
                        uid_a=uid_a,
                        uid_b=uid_b,
                    )

    # ------------------------------------------------------------------
    # GDS Memory Ranking
    # ------------------------------------------------------------------

    _gds_available: bool | None = None

    async def is_gds_available(self) -> bool:
        """Check if the Graph Data Science plugin is available.

        Returns False immediately if gds_enabled is False in params,
        without querying the Neo4j instance.
        """
        if not self._gds_enabled:
            return False

        if self._gds_available is not None:
            return self._gds_available

        try:
            await self._driver.execute_query(
                _neo4j_query("RETURN gds.version() AS version"),
            )
            self._gds_available = True
            logger.info("GDS plugin is available")
        except Exception:
            self._gds_available = False
            logger.warning(_GDS_NOT_AVAILABLE)

        return self._gds_available

    async def _create_scoped_projection(
        self,
        *,
        collection: str,
        projection_name: str,
        relation_types: list[str] | None = None,
    ) -> ProjectionInfo:
        """Create a GDS graph projection scoped to a collection.

        Args:
            collection: The collection to scope the projection to.
            projection_name: Unique name for the projection.
            relation_types: Optional list of relationship types to include.
                When ``None``, all relationship types are included.

        Returns:
            A :class:`ProjectionInfo` with the projection name and size.
        """
        sanitized_collection = Neo4jVectorGraphStore._sanitize_name(collection)

        if relation_types:
            escaped = ", ".join(f"'{rt}'" for rt in relation_types)
            rel_filter = f" WHERE type(r) IN [{escaped}]"
        else:
            rel_filter = ""

        await self._driver.execute_query(
            _neo4j_query(
                "CALL gds.graph.project.cypher(\n"
                "  $projection_name,\n"
                f"  'MATCH (n:{sanitized_collection}) RETURN id(n) AS id',\n"
                f"  'MATCH (n:{sanitized_collection})-[r]->"
                f"(m:{sanitized_collection}){rel_filter} "
                "RETURN id(n) AS source, id(m) AS target'\n"
                ")"
            ),
            projection_name=projection_name,
        )

        # Query projection metadata for metrics.
        records, _, _ = await self._driver.execute_query(
            _neo4j_query(
                "CALL gds.graph.list($projection_name) "
                "YIELD nodeCount, relationshipCount"
            ),
            projection_name=projection_name,
        )

        node_count = 0
        rel_count = 0
        if records:
            node_count = int(records[0]["nodeCount"])
            rel_count = int(records[0]["relationshipCount"])

        if self._should_collect_metrics:
            self._gds_projection_node_count_gauge.set(node_count)
            self._gds_projection_relationship_count_gauge.set(rel_count)

        return ProjectionInfo(
            name=projection_name,
            node_count=node_count,
            relationship_count=rel_count,
        )

    async def _drop_projection(self, projection_name: str) -> None:
        """Drop a GDS graph projection."""
        try:
            await self._driver.execute_query(
                _neo4j_query("CALL gds.graph.drop($projection_name, false)"),
                projection_name=projection_name,
            )
        except Exception:
            logger.debug(
                "Failed to drop GDS projection %r (may not exist)",
                projection_name,
            )

    async def compute_pagerank(
        self,
        *,
        collection: str,
        relation_types: list[str] | None = None,
        damping_factor: float = 0.85,
        max_iterations: int = 20,
        write_back: bool = False,
        write_property: str | None = None,
    ) -> list[tuple[str, float]]:
        """Compute PageRank for nodes in a collection.

        Args:
            collection: The collection to compute PageRank on.
            relation_types: Optional relationship types to include in the
                projection. When ``None``, all types are included.
            damping_factor: PageRank damping factor (default 0.85).
            max_iterations: Maximum iterations (default 20).
            write_back: Whether to write scores back to node properties.
            write_property: Property name for write-back. Defaults to
                ``"pagerank_score"`` when ``None``.

        Returns:
            List of ``(uid, score)`` pairs ordered by score descending.
        """
        start_time = time.monotonic()

        if not await self.is_gds_available():
            raise RuntimeError(_GDS_NOT_AVAILABLE)

        effective_write_property = write_property or "pagerank_score"
        projection_name = f"pagerank_{collection}_{uuid4().hex[:8]}"
        sanitized_collection = Neo4jVectorGraphStore._sanitize_name(collection)

        try:
            projection = await self._create_scoped_projection(
                collection=collection,
                projection_name=projection_name,
                relation_types=relation_types,
            )

            if write_back:
                await self._driver.execute_query(
                    _neo4j_query(
                        "CALL gds.pageRank.write(\n"
                        "  $projection_name,\n"
                        "  {dampingFactor: $damping, "
                        "maxIterations: $max_iter, "
                        "writeProperty: $write_prop}\n"
                        ")"
                    ),
                    projection_name=projection.name,
                    damping=damping_factor,
                    max_iter=max_iterations,
                    write_prop=effective_write_property,
                )

            records, _, _ = await self._driver.execute_query(
                _neo4j_query(
                    "CALL gds.pageRank.stream(\n"
                    "  $projection_name,\n"
                    "  {dampingFactor: $damping, "
                    "maxIterations: $max_iter}\n"
                    ")\n"
                    "YIELD nodeId, score\n"
                    f"MATCH (n:{sanitized_collection}) "
                    "WHERE id(n) = nodeId\n"
                    "RETURN n.uid AS uid, score\n"
                    "ORDER BY score DESC"
                ),
                projection_name=projection.name,
                damping=damping_factor,
                max_iter=max_iterations,
            )

            return [(rec["uid"], float(rec["score"])) for rec in records]
        finally:
            await self._drop_projection(projection_name)
            end_time = time.monotonic()
            self._collect_metrics(
                self._compute_pagerank_calls_counter,
                self._compute_pagerank_latency_summary,
                start_time,
                end_time,
            )

    async def detect_communities(
        self,
        *,
        collection: str,
        relation_types: list[str] | None = None,
        max_iterations: int = 20,
        write_back: bool = False,
        write_property: str | None = None,
    ) -> dict[int, list[str]]:
        """Detect communities using Louvain algorithm.

        Args:
            collection: The collection to detect communities in.
            relation_types: Optional relationship types to include in the
                projection. When ``None``, all types are included.
            max_iterations: Maximum iterations (default 20).
            write_back: Whether to write community IDs back to nodes.
            write_property: Property name for write-back. Defaults to
                ``"community_id"`` when ``None``.

        Returns:
            Mapping of ``{community_id: [uid, ...]}`` for each detected
            community.
        """
        start_time = time.monotonic()

        if not await self.is_gds_available():
            raise RuntimeError(_GDS_NOT_AVAILABLE)

        effective_write_property = write_property or "community_id"
        projection_name = f"louvain_{collection}_{uuid4().hex[:8]}"
        sanitized_collection = Neo4jVectorGraphStore._sanitize_name(collection)

        try:
            projection = await self._create_scoped_projection(
                collection=collection,
                projection_name=projection_name,
                relation_types=relation_types,
            )

            if write_back:
                await self._driver.execute_query(
                    _neo4j_query(
                        "CALL gds.louvain.write(\n"
                        "  $projection_name,\n"
                        "  {maxIterations: $max_iter, "
                        "writeProperty: $write_prop}\n"
                        ")"
                    ),
                    projection_name=projection.name,
                    max_iter=max_iterations,
                    write_prop=effective_write_property,
                )

            records, _, _ = await self._driver.execute_query(
                _neo4j_query(
                    "CALL gds.louvain.stream(\n"
                    "  $projection_name,\n"
                    "  {maxIterations: $max_iter}\n"
                    ")\n"
                    "YIELD nodeId, communityId\n"
                    f"MATCH (n:{sanitized_collection}) "
                    "WHERE id(n) = nodeId\n"
                    "RETURN n.uid AS uid, communityId"
                ),
                projection_name=projection.name,
                max_iter=max_iterations,
            )

            communities: dict[int, list[str]] = {}
            for rec in records:
                cid = int(rec["communityId"])
                communities.setdefault(cid, []).append(str(rec["uid"]))

            return communities
        finally:
            await self._drop_projection(projection_name)
            end_time = time.monotonic()
            self._collect_metrics(
                self._detect_communities_calls_counter,
                self._detect_communities_latency_summary,
                start_time,
                end_time,
            )

    async def graph_stats(
        self,
        *,
        collection: str,
    ) -> GraphStatsResult:
        """Return collection-level graph statistics.

        Runs three Cypher aggregation queries to compute node count,
        edge count, relationship type distribution, and entity type
        distribution.  No GDS dependency.
        """
        start_time = time.monotonic()
        sanitized_collection = Neo4jVectorGraphStore._sanitize_name(collection)

        try:
            # 1. Node count + edge count
            records, _, _ = await self._driver.execute_query(
                _neo4j_query(
                    f"MATCH (n:{sanitized_collection})\n"
                    "OPTIONAL MATCH (n)-[r]-()\n"
                    "RETURN count(DISTINCT n) AS node_count, "
                    "count(r) AS edge_count"
                ),
            )
            node_count = int(records[0]["node_count"]) if records else 0
            edge_count = int(records[0]["edge_count"]) if records else 0

            # 2. Relationship type distribution
            records, _, _ = await self._driver.execute_query(
                _neo4j_query(
                    f"MATCH (n:{sanitized_collection})-[r]-()\n"
                    "RETURN type(r) AS rel_type, count(r) AS cnt\n"
                    "ORDER BY cnt DESC"
                ),
            )
            rel_dist: dict[str, int] = {
                str(rec["rel_type"]): int(rec["cnt"]) for rec in records
            }

            # 3. Entity type distribution (desanitized)
            records, _, _ = await self._driver.execute_query(
                _neo4j_query(
                    f"MATCH (n:{sanitized_collection})\n"
                    "UNWIND labels(n) AS label\n"
                    f"WITH label WHERE label STARTS WITH '{_ENTITY_TYPE_PREFIX}'\n"
                    "RETURN label, count(*) AS cnt\n"
                    "ORDER BY cnt DESC"
                ),
            )
            entity_dist: dict[str, int] = {
                _desanitize_entity_type(str(rec["label"])): int(rec["cnt"])
                for rec in records
            }

            avg_degree = edge_count / node_count if node_count > 0 else 0.0

            return GraphStatsResult(
                node_count=node_count,
                edge_count=edge_count,
                avg_degree=avg_degree,
                relationship_type_distribution=rel_dist,
                entity_type_distribution=entity_dist,
            )
        finally:
            end_time = time.monotonic()
            self._collect_metrics(
                self._graph_stats_calls_counter,
                self._graph_stats_latency_summary,
                start_time,
                end_time,
            )

    async def shortest_path(
        self,
        *,
        collection: str,
        source_uid: str,
        target_uid: str,
        relation_types: list[str] | None = None,
        max_depth: int = 10,
    ) -> ShortestPathResult:
        """Find the shortest unweighted path between two nodes.

        Uses Cypher's built-in ``shortestPath()`` function (no GDS
        dependency).  Both source and target must belong to the given
        collection.

        Args:
            collection: The collection both nodes belong to.
            source_uid: UID of the starting node.
            target_uid: UID of the ending node.
            relation_types: Optional relationship types to traverse.
            max_depth: Maximum traversal depth (default 10).

        Returns:
            A :class:`ShortestPathResult` with the path nodes and edges,
            or an empty result if no path exists.
        """
        start_time = time.monotonic()
        sanitized_collection = Neo4jVectorGraphStore._sanitize_name(collection)

        if relation_types:
            escaped = "|".join(f"`{rt}`" for rt in relation_types)
            rel_clause = f"[:{escaped}*..{max_depth}]"
        else:
            rel_clause = f"[*..{max_depth}]"

        try:
            records, _, _ = await self._driver.execute_query(
                _neo4j_query(
                    f"MATCH (src:{sanitized_collection} {{uid: $source_uid}}),\n"
                    f"      (tgt:{sanitized_collection} {{uid: $target_uid}})\n"
                    f"MATCH p = shortestPath((src)-{rel_clause}-(tgt))\n"
                    "RETURN p"
                ),
                source_uid=source_uid,
                target_uid=target_uid,
            )

            if not records:
                return ShortestPathResult(path_length=0, nodes=[], edges=[])

            path = records[0]["p"]
            path_nodes: list[PathNode] = []
            for node in path.nodes:
                node_uid = str(node.get("uid", ""))
                props = {
                    k: value_from_neo4j(v) for k, v in dict(node).items() if k != "uid"
                }
                path_nodes.append(PathNode(uid=node_uid, properties=props))

            path_edges: list[PathEdge] = []
            for rel in path.relationships:
                src_uid = str(rel.start_node.get("uid", ""))
                tgt_uid = str(rel.end_node.get("uid", ""))
                rel_type = rel.type
                props = {k: value_from_neo4j(v) for k, v in dict(rel).items()}
                path_edges.append(
                    PathEdge(
                        source_uid=src_uid,
                        target_uid=tgt_uid,
                        type=rel_type,
                        properties=props,
                    )
                )

            return ShortestPathResult(
                path_length=len(path_edges),
                nodes=path_nodes,
                edges=path_edges,
            )
        finally:
            end_time = time.monotonic()
            self._collect_metrics(
                self._shortest_path_calls_counter,
                self._shortest_path_latency_summary,
                start_time,
                end_time,
            )

    async def degree_centrality(
        self,
        *,
        collection: str,
        relation_types: list[str] | None = None,
        limit: int = 50,
    ) -> list[DegreeCentralityResult]:
        """Compute in-degree, out-degree, and total degree for nodes.

        Uses pure Cypher aggregation (no GDS dependency).

        Args:
            collection: The collection to compute degree centrality on.
            relation_types: Optional relationship types to count.
            limit: Maximum number of results (default 50), ordered by
                total degree descending.

        Returns:
            List of :class:`DegreeCentralityResult` ordered by total
            degree descending.
        """
        start_time = time.monotonic()
        sanitized_collection = Neo4jVectorGraphStore._sanitize_name(collection)

        if relation_types:
            escaped = ", ".join(f"'{rt}'" for rt in relation_types)
            in_filter = f" WHERE type(r_in) IN [{escaped}]"
            out_filter = f" WHERE type(r_out) IN [{escaped}]"
        else:
            in_filter = ""
            out_filter = ""

        try:
            records, _, _ = await self._driver.execute_query(
                _neo4j_query(
                    f"MATCH (n:{sanitized_collection})\n"
                    f"OPTIONAL MATCH (n)<-[r_in]-(){in_filter}\n"
                    "WITH n, count(r_in) AS in_deg\n"
                    f"OPTIONAL MATCH (n)-[r_out]->()\n"
                    f"{out_filter}\n"
                    "WITH n, in_deg, count(r_out) AS out_deg\n"
                    "RETURN n.uid AS uid, in_deg, out_deg, "
                    "in_deg + out_deg AS total_deg\n"
                    "ORDER BY total_deg DESC\n"
                    "LIMIT $limit"
                ),
                limit=limit,
            )

            return [
                DegreeCentralityResult(
                    uid=str(rec["uid"]),
                    in_degree=int(rec["in_deg"]),
                    out_degree=int(rec["out_deg"]),
                    total_degree=int(rec["total_deg"]),
                )
                for rec in records
            ]
        finally:
            end_time = time.monotonic()
            self._collect_metrics(
                self._degree_centrality_calls_counter,
                self._degree_centrality_latency_summary,
                start_time,
                end_time,
            )

    async def extract_subgraph(
        self,
        *,
        collection: str,
        anchor_uid: str,
        max_depth: int = 2,
        relation_types: list[str] | None = None,
        node_limit: int = 100,
    ) -> SubgraphResult:
        """Extract the ego-graph neighborhood around an anchor node.

        Uses two Cypher queries: one for distinct nodes within *max_depth*
        hops, one for edges between those nodes.  No GDS dependency.

        Args:
            collection: The collection the anchor belongs to.
            anchor_uid: UID of the anchor node.
            max_depth: Maximum traversal depth (default 2).
            relation_types: Optional relationship types to traverse.
            node_limit: Maximum number of nodes to return (default 100).

        Returns:
            A :class:`SubgraphResult` with deduplicated nodes and edges.
        """
        start_time = time.monotonic()
        sanitized_collection = Neo4jVectorGraphStore._sanitize_name(collection)

        if relation_types:
            escaped = "|".join(f"`{rt}`" for rt in relation_types)
            rel_clause = f"[:{escaped}*0..{max_depth}]"
        else:
            rel_clause = f"[*0..{max_depth}]"

        try:
            # 1. Find distinct nodes within max_depth hops
            records, _, _ = await self._driver.execute_query(
                _neo4j_query(
                    f"MATCH (anchor:{sanitized_collection} {{uid: $anchor_uid}})\n"
                    f"MATCH (anchor)-{rel_clause}-(n:{sanitized_collection})\n"
                    "WITH DISTINCT n\n"
                    "LIMIT $node_limit\n"
                    "RETURN n"
                ),
                anchor_uid=anchor_uid,
                node_limit=node_limit,
            )

            subgraph_nodes: list[SubgraphNode] = []
            node_uids: set[str] = set()
            for rec in records:
                node = rec["n"]
                uid = str(node.get("uid", ""))
                if uid and uid not in node_uids:
                    node_uids.add(uid)
                    props = {
                        k: value_from_neo4j(v)
                        for k, v in dict(node).items()
                        if k != "uid"
                    }
                    subgraph_nodes.append(SubgraphNode(uid=uid, properties=props))

            if len(node_uids) < 2:
                return SubgraphResult(nodes=subgraph_nodes, edges=[])

            # 2. Find edges between the included nodes
            edge_rel = f"[r:{escaped}]" if relation_types else "[r]"

            records, _, _ = await self._driver.execute_query(
                _neo4j_query(
                    f"MATCH (a:{sanitized_collection})-{edge_rel}->"
                    f"(b:{sanitized_collection})\n"
                    "WHERE a.uid IN $uids AND b.uid IN $uids\n"
                    "RETURN a.uid AS src, b.uid AS tgt, "
                    "type(r) AS rel_type, properties(r) AS props"
                ),
                uids=list(node_uids),
            )

            subgraph_edges: list[SubgraphEdge] = []
            for rec in records:
                props_raw = rec["props"]
                props = (
                    {k: value_from_neo4j(v) for k, v in props_raw.items()}
                    if props_raw
                    else {}
                )
                subgraph_edges.append(
                    SubgraphEdge(
                        source_uid=str(rec["src"]),
                        target_uid=str(rec["tgt"]),
                        type=str(rec["rel_type"]),
                        properties=props,
                    )
                )

            return SubgraphResult(nodes=subgraph_nodes, edges=subgraph_edges)
        finally:
            end_time = time.monotonic()
            self._collect_metrics(
                self._extract_subgraph_calls_counter,
                self._extract_subgraph_latency_summary,
                start_time,
                end_time,
            )

    async def betweenness_centrality(
        self,
        *,
        collection: str,
        relation_types: list[str] | None = None,
        sampling_size: int | None = None,
        write_back: bool = False,
        write_property: str | None = None,
    ) -> list[tuple[str, float]]:
        """Compute betweenness centrality for nodes in a collection.

        Uses GDS ``gds.betweenness.stream()`` and optionally
        ``gds.betweenness.write()``.  Follows the same projection /
        stream / write-back / cleanup lifecycle as ``compute_pagerank()``.

        Args:
            collection: The collection to compute betweenness on.
            relation_types: Optional relationship types to include in the
                projection. When ``None``, all types are included.
            sampling_size: When provided, GDS uses approximate computation
                by sampling this many source nodes. ``None`` = exact.
            write_back: Whether to write scores back to node properties.
            write_property: Property name for write-back. Defaults to
                ``"betweenness_score"`` when ``None``.

        Returns:
            List of ``(uid, score)`` pairs ordered by score descending.
        """
        start_time = time.monotonic()

        if not await self.is_gds_available():
            raise RuntimeError(_GDS_NOT_AVAILABLE)

        effective_write_property = write_property or "betweenness_score"
        projection_name = f"betweenness_{collection}_{uuid4().hex[:8]}"
        sanitized_collection = Neo4jVectorGraphStore._sanitize_name(collection)

        # Build optional config dict for sampling
        config_parts = []
        if sampling_size is not None:
            config_parts.append("samplingSize: $sampling_size")

        try:
            projection = await self._create_scoped_projection(
                collection=collection,
                projection_name=projection_name,
                relation_types=relation_types,
            )

            if write_back:
                write_config = "writeProperty: $write_prop"
                if sampling_size is not None:
                    write_config += ", samplingSize: $sampling_size"
                await self._driver.execute_query(
                    _neo4j_query(
                        "CALL gds.betweenness.write(\n"
                        f"  $projection_name,\n"
                        f"  {{{write_config}}}\n"
                        ")"
                    ),
                    projection_name=projection.name,
                    write_prop=effective_write_property,
                    **(
                        {"sampling_size": sampling_size}
                        if sampling_size is not None
                        else {}
                    ),
                )

            stream_config = ""
            stream_params: dict[str, Any] = {
                "projection_name": projection.name,
            }
            if sampling_size is not None:
                stream_config = ", {samplingSize: $sampling_size}"
                stream_params["sampling_size"] = sampling_size

            records, _, _ = await self._driver.execute_query(
                _neo4j_query(
                    "CALL gds.betweenness.stream(\n"
                    f"  $projection_name{stream_config}\n"
                    ")\n"
                    "YIELD nodeId, score\n"
                    f"MATCH (n:{sanitized_collection}) "
                    "WHERE id(n) = nodeId\n"
                    "RETURN n.uid AS uid, score\n"
                    "ORDER BY score DESC"
                ),
                **stream_params,
            )

            return [(rec["uid"], float(rec["score"])) for rec in records]
        finally:
            await self._drop_projection(projection_name)
            end_time = time.monotonic()
            self._collect_metrics(
                self._betweenness_centrality_calls_counter,
                self._betweenness_centrality_latency_summary,
                start_time,
                end_time,
            )

    async def delete_nodes(
        self,
        *,
        collection: str,
        node_uids: Iterable[str],
    ) -> None:
        """Delete nodes by uid from a collection."""
        start_time = time.monotonic()

        sanitized_collection = Neo4jVectorGraphStore._sanitize_name(collection)

        await self._driver.execute_query(
            _neo4j_query(
                "UNWIND $node_uids AS node_uid\n"
                f"MATCH (n:{sanitized_collection} {{uid: node_uid}})\n"
                "DETACH DELETE n"
            ),
            node_uids=[str(node_uid) for node_uid in node_uids],
        )

        end_time = time.monotonic()
        self._collect_metrics(
            self._delete_nodes_calls_counter,
            self._delete_nodes_latency_summary,
            start_time,
            end_time,
        )

    async def delete_all_data(self) -> None:
        """Delete all nodes and relationships from the database."""
        await self._driver.execute_query(_neo4j_query("MATCH (n) DETACH DELETE n"))

    async def close(self) -> None:
        """Close the underlying Neo4j driver."""
        await self._driver.close()

    async def _count_nodes(self, collection: str) -> int:
        """Count the number of nodes in a collection."""
        start_time = time.monotonic()

        sanitized_collection = Neo4jVectorGraphStore._sanitize_name(collection)

        records, _, _ = await self._driver.execute_query(
            _neo4j_query(
                f"MATCH (n:{sanitized_collection})\nRETURN count(n) AS node_count"
            ),
        )

        end_time = time.monotonic()
        self._collect_metrics(
            self._count_nodes_calls_counter,
            self._count_nodes_latency_summary,
            start_time,
            end_time,
        )

        return records[0]["node_count"]

    async def _count_edges(self, relation: str) -> int:
        """Count the number of edges having a relation type."""
        start_time = time.monotonic()

        sanitized_relation = Neo4jVectorGraphStore._sanitize_name(relation)

        records, _, _ = await self._driver.execute_query(
            _neo4j_query(
                f"MATCH ()-[r:{sanitized_relation}]->()\n"
                "RETURN count(r) AS relationship_count"
            ),
        )

        end_time = time.monotonic()
        self._collect_metrics(
            self._count_edges_calls_counter,
            self._count_edges_latency_summary,
            start_time,
            end_time,
        )

        return records[0]["relationship_count"]

    async def _populate_index_state_cache(self) -> None:
        """Populate the index state cache."""
        start_time = time.monotonic()

        if self._index_state_cache:
            end_time = time.monotonic()
            self._collect_metrics(
                self._populate_index_state_cache_calls_counter,
                self._populate_index_state_cache_latency_summary,
                start_time,
                end_time,
            )
            return

        async with self._populate_index_state_cache_lock:
            if not self._index_state_cache:
                records, _, _ = await self._driver.execute_query(
                    _neo4j_query("SHOW INDEXES YIELD name RETURN name"),
                )

                # This ensures that all the indexes in records are online.
                await self._driver.execute_query(_neo4j_query("CALL db.awaitIndexes()"))

                # Synchronous code is atomic in asynchronous framework
                # so double-checked locking works here.
                self._index_state_cache.update(
                    {
                        record["name"]: Neo4jVectorGraphStore.CacheIndexState.ONLINE
                        for record in records
                    },
                )

        end_time = time.monotonic()
        self._collect_metrics(
            self._populate_index_state_cache_calls_counter,
            self._populate_index_state_cache_latency_summary,
            start_time,
            end_time,
        )

    async def _create_initial_indexes_if_not_exist(
        self,
        entity_type: EntityType,
        sanitized_collection_or_relation: str,
    ) -> None:
        """Create initial indexes if missing and wait for them to be online."""
        start_time = time.monotonic()

        tasks = [
            self._create_range_index_if_not_exists(
                entity_type=entity_type,
                sanitized_collection_or_relation=sanitized_collection_or_relation,
                sanitized_property_names="uid",
            ),
        ]
        tasks += [
            self._create_range_index_if_not_exists(
                entity_type=entity_type,
                sanitized_collection_or_relation=sanitized_collection_or_relation,
                sanitized_property_names=[
                    Neo4jVectorGraphStore._sanitize_name(
                        mangle_property_name(property_name),
                    )
                    for property_name in property_name_hierarchy
                ],
            )
            for range_index_hierarchy in self._range_index_hierarchies
            for property_name_hierarchy in [
                range_index_hierarchy[: i + 1]
                for i in range(len(range_index_hierarchy))
            ]
        ]

        end_time = time.monotonic()
        self._collect_metrics(
            self._create_initial_indexes_if_not_exist_calls_counter,
            self._create_initial_indexes_if_not_exist_latency_summary,
            start_time,
            end_time,
        )

        await asyncio.gather(*tasks)

    async def _create_range_index_if_not_exists(
        self,
        entity_type: EntityType,
        sanitized_collection_or_relation: str,
        sanitized_property_names: str | Iterable[str],
    ) -> None:
        """Create a range index if missing and wait for it to be online."""
        start_time = time.monotonic()

        if isinstance(sanitized_property_names, str):
            sanitized_property_names = [sanitized_property_names]

        sanitized_property_names = list(sanitized_property_names)
        if len(sanitized_property_names) == 0:
            raise ValueError("sanitized_property_names must be nonempty")

        await self._populate_index_state_cache()

        range_index_name = Neo4jVectorGraphStore._index_name(
            entity_type,
            sanitized_collection_or_relation,
            sanitized_property_names,
        )

        cached_index_state = self._index_state_cache.get(range_index_name)
        match cached_index_state:
            case Neo4jVectorGraphStore.CacheIndexState.CREATING:
                # Wait for the index to be online.
                await self._await_create_index_if_not_exists(
                    range_index_name,
                    asyncio.sleep(0),  # Use as a no-op.
                )
                end_time = time.monotonic()
                self._collect_metrics(
                    self._create_range_index_if_not_exists_calls_counter,
                    self._create_range_index_if_not_exists_latency_summary,
                    start_time,
                    end_time,
                )
                return
            case Neo4jVectorGraphStore.CacheIndexState.ONLINE:
                end_time = time.monotonic()
                self._collect_metrics(
                    self._create_range_index_if_not_exists_calls_counter,
                    self._create_range_index_if_not_exists_latency_summary,
                    start_time,
                    end_time,
                )
                return

        # Code is synchronous between the cache read and this write,
        # so it is effectively atomic in the asynchronous framework.
        self._index_state_cache[range_index_name] = (
            Neo4jVectorGraphStore.CacheIndexState.CREATING
        )

        match entity_type:
            case EntityType.NODE:
                query_index_for_expression = f"(e:{sanitized_collection_or_relation})"
            case EntityType.EDGE:
                query_index_for_expression = (
                    f"()-[e:{sanitized_collection_or_relation}]-()"
                )

        create_index_awaitable = self._driver.execute_query(
            _neo4j_query(
                f"CREATE RANGE INDEX {range_index_name}\n"
                "IF NOT EXISTS\n"
                f"FOR {query_index_for_expression}\n"
                f"ON ({
                    ', '.join(
                        f'e.{sanitized_property_name}'
                        for sanitized_property_name in sanitized_property_names
                    )
                })"
            ),
        )

        await self._await_create_index_if_not_exists(
            range_index_name,
            create_index_awaitable,
        )

        end_time = time.monotonic()
        self._collect_metrics(
            self._create_range_index_if_not_exists_calls_counter,
            self._create_range_index_if_not_exists_latency_summary,
            start_time,
            end_time,
        )

    async def _create_vector_index_if_not_exists(
        self,
        entity_type: EntityType,
        sanitized_collection_or_relation: str,
        sanitized_embedding_name: str,
        dimensions: int,
        similarity_metric: SimilarityMetric = SimilarityMetric.COSINE,
    ) -> None:
        """Create a vector index if missing and wait for it to be online."""
        if not (1 <= dimensions <= 4096):
            raise ValueError("dimensions must be between 1 and 4096")

        start_time = time.monotonic()

        await self._populate_index_state_cache()

        vector_index_name = Neo4jVectorGraphStore._index_name(
            entity_type,
            sanitized_collection_or_relation,
            sanitized_embedding_name,
        )

        cached_index_state = self._index_state_cache.get(vector_index_name)
        match cached_index_state:
            case Neo4jVectorGraphStore.CacheIndexState.CREATING:
                # Wait for the index to be online.
                await self._await_create_index_if_not_exists(
                    vector_index_name,
                    asyncio.sleep(0),  # Use as a no-op.
                )
                end_time = time.monotonic()
                self._collect_metrics(
                    self._create_vector_index_if_not_exists_calls_counter,
                    self._create_vector_index_if_not_exists_latency_summary,
                    start_time,
                    end_time,
                )
                return
            case Neo4jVectorGraphStore.CacheIndexState.ONLINE:
                end_time = time.monotonic()
                self._collect_metrics(
                    self._create_vector_index_if_not_exists_calls_counter,
                    self._create_vector_index_if_not_exists_latency_summary,
                    start_time,
                    end_time,
                )
                return

        # Code is synchronous between the cache read and this write,
        # so it is effectively atomic in the asynchronous framework.
        self._index_state_cache[vector_index_name] = (
            Neo4jVectorGraphStore.CacheIndexState.CREATING
        )

        match similarity_metric:
            case SimilarityMetric.COSINE:
                similarity_function = "cosine"
            case SimilarityMetric.EUCLIDEAN:
                similarity_function = "euclidean"
            case _:
                similarity_function = "cosine"

        match entity_type:
            case EntityType.NODE:
                query_index_for_expression = f"(e:{sanitized_collection_or_relation})"
            case EntityType.EDGE:
                query_index_for_expression = (
                    f"()-[e:{sanitized_collection_or_relation}]-()"
                )

        create_index_awaitable = self._driver.execute_query(
            _neo4j_query(
                f"CREATE VECTOR INDEX {vector_index_name}\n"
                "IF NOT EXISTS\n"
                f"FOR {query_index_for_expression}\n"
                f"ON e.{sanitized_embedding_name}\n"
                "OPTIONS {\n"
                "    indexConfig: {\n"
                "        `vector.dimensions`:\n"
                "            $dimensions,\n"
                "        `vector.similarity_function`:\n"
                "            $similarity_function\n"
                "    }\n"
                "}"
            ),
            dimensions=dimensions,
            similarity_function=similarity_function,
        )

        await self._await_create_index_if_not_exists(
            vector_index_name,
            create_index_awaitable,
        )

        end_time = time.monotonic()
        self._collect_metrics(
            self._create_vector_index_if_not_exists_calls_counter,
            self._create_vector_index_if_not_exists_latency_summary,
            start_time,
            end_time,
        )

    @async_locked
    async def _await_create_index_if_not_exists(
        self,
        index_name: str,
        create_index_awaitable: Awaitable,
    ) -> None:
        """Await index creation and mark it online in the cache."""
        await create_index_awaitable

        await self._driver.execute_query(
            _neo4j_query("CALL db.awaitIndex($index_name)"),
            index_name=index_name,
        )

        self._index_state_cache[index_name] = (
            Neo4jVectorGraphStore.CacheIndexState.ONLINE
        )

    _SANITIZE_NAME_PREFIX = "SANITIZED_"
    _ENTITY_TYPE_PREFIX = _ENTITY_TYPE_PREFIX

    @staticmethod
    def _sanitize_name(name: str) -> str:
        """Sanitize a name for safe Neo4j identifiers."""
        return Neo4jVectorGraphStore._SANITIZE_NAME_PREFIX + "".join(
            c if c.isalnum() else f"_u{ord(c):x}_" for c in name
        )

    @staticmethod
    def _desanitize_name(sanitized_name: str) -> str:
        """Restore a sanitized name to its original form."""
        return re.sub(
            r"_u([0-9a-fA-F]+)_",
            lambda match: chr(int(match[1], 16)),
            sanitized_name.removeprefix(Neo4jVectorGraphStore._SANITIZE_NAME_PREFIX),
        )

    @staticmethod
    def _entity_type_label_fragment(
        entity_types: list[str] | None,
    ) -> str:
        """Build a Cypher label fragment for entity type filtering.

        Returns a string like ``":ENTITY_TYPE_Person:ENTITY_TYPE_Concept"``
        that can be appended to a node alias in a ``MATCH`` pattern.  If
        *entity_types* is ``None`` or empty, returns an empty string so the
        query is unaffected.
        """
        if not entity_types:
            return ""
        return "".join(
            f":{Neo4jVectorGraphStore._sanitize_entity_type(et)}" for et in entity_types
        )

    @staticmethod
    def _sanitize_entity_type(name: str) -> str:
        """Sanitize an entity type label for safe Neo4j label use.

        Delegates to :func:`memmachine.common.neo4j_utils.sanitize_entity_type`.
        """
        return _sanitize_entity_type(name)

    @staticmethod
    def _desanitize_entity_type(sanitized_name: str) -> str:
        """Restore an entity type label from its sanitized form.

        Delegates to :func:`memmachine.common.neo4j_utils.desanitize_entity_type`.
        """
        return _desanitize_entity_type(sanitized_name)

    @staticmethod
    def _sanitize_properties(
        properties: Mapping[str, PropertyValue] | None,
    ) -> dict[str, PropertyValue]:
        """Sanitize property names in a mapping for Neo4j storage."""
        return (
            {
                Neo4jVectorGraphStore._sanitize_name(key): sanitize_value_for_neo4j(
                    value
                )
                for key, value in properties.items()
            }
            if properties is not None
            else {}
        )

    @staticmethod
    def _index_name(
        entity_type: EntityType,
        sanitized_collection_or_relation: str,
        sanitized_property_names: str | Iterable[str],
    ) -> str:
        """Generate a unique index name from entity type and properties."""
        if isinstance(sanitized_property_names, str):
            sanitized_property_names = [sanitized_property_names]

        sanitized_property_names_string = "_and_".join(
            f"{len(sanitized_property_name)}_{sanitized_property_name}"
            for sanitized_property_name in sanitized_property_names
        )

        return (
            f"{entity_type.value}_index"
            "_for_"
            f"{len(sanitized_collection_or_relation)}_"
            f"{sanitized_collection_or_relation}"
            "_on_"
            f"{sanitized_property_names_string}"
        )

    @staticmethod
    def _similarity_metric_property_name(embedding_name: str) -> str:
        """
        Get the similarity metric property name for an embedding.

        Args:
            embedding_name (str): The name of the embedding.

        Returns:
            str: The similarity metric property name.

        """
        return f"similarity_metric_for_{embedding_name}"

    @staticmethod
    def _nodes_from_neo4j_nodes(
        neo4j_nodes: Iterable[Neo4jNode],
    ) -> list[Node]:
        """
        Convert a collection of Neo4jNodes to a list of Nodes.

        Args:
            neo4j_nodes (Iterable[Neo4jNode]): Iterable of Neo4jNodes.

        Returns:
            list[Node]: List of Node objects.

        """
        nodes = []
        for neo4j_node in neo4j_nodes:
            node_properties = {}
            node_embeddings = {}

            for neo4j_property_name, neo4j_property_value in neo4j_node.items():
                desanitized_property_name = Neo4jVectorGraphStore._desanitize_name(
                    neo4j_property_name,
                )

                if is_mangled_property_name(desanitized_property_name):
                    property_name = demangle_property_name(desanitized_property_name)
                    node_properties[property_name] = value_from_neo4j(
                        neo4j_property_value,
                    )
                elif is_mangled_embedding_name(desanitized_property_name):
                    embedding_name = demangle_embedding_name(desanitized_property_name)
                    metric_key = Neo4jVectorGraphStore._sanitize_name(
                        Neo4jVectorGraphStore._similarity_metric_property_name(
                            embedding_name,
                        ),
                    )
                    # Nodes created by other storage layers (e.g.
                    # Neo4jSemanticStorage) may have properties whose
                    # desanitized names happen to start with the
                    # ``embedding_`` prefix but lack the companion
                    # similarity-metric property.  Skip them instead
                    # of crashing.
                    raw_metric = neo4j_node.get(metric_key)
                    if raw_metric is None:
                        continue

                    embedding_value = cast(
                        list[float],
                        value_from_neo4j(neo4j_property_value),
                    )
                    similarity_metric = SimilarityMetric(
                        value_from_neo4j(raw_metric),
                    )
                    node_embeddings[embedding_name] = (
                        embedding_value,
                        similarity_metric,
                    )

            # Entity type labels use the ENTITY_TYPE_ prefix to distinguish
            # them from collection labels (SANITIZED_ prefix) and any other
            # Neo4j-internal labels.
            entity_type_labels = [
                Neo4jVectorGraphStore._desanitize_entity_type(label)
                for label in neo4j_node.labels
                if label.startswith(Neo4jVectorGraphStore._ENTITY_TYPE_PREFIX)
            ]

            nodes.append(
                Node(
                    uid=neo4j_node["uid"],
                    properties=node_properties,
                    embeddings=node_embeddings,
                    entity_types=entity_type_labels,
                ),
            )

        return nodes

    def _collect_metrics(
        self,
        calls_counter: MetricsFactory.Counter | None,
        latency_summary: MetricsFactory.Summary | None,
        start_time: float,
        end_time: float,
    ) -> None:
        """Increment calls and observe latency."""
        if self._should_collect_metrics:
            cast(MetricsFactory.Counter, calls_counter).increment(
                labels=self._user_metrics_labels
            )
            cast(MetricsFactory.Summary, latency_summary).observe(
                value=end_time - start_time,
                labels=self._user_metrics_labels,
            )

    @staticmethod
    def _build_query_filter(
        entity_query_alias: str,
        query_value_parameter: str,
        property_filter: FilterExpr | None,
    ) -> tuple[str, dict[str, FilterValue]]:
        if property_filter is None:
            query_filter_string = "TRUE"
            query_filter_params: dict[str, FilterValue] = {}
        else:
            query_filter_string, query_filter_params = (
                Neo4jVectorGraphStore._render_filter_expr(
                    entity_query_alias,
                    query_value_parameter,
                    property_filter,
                )
            )

        return query_filter_string, query_filter_params

    @staticmethod
    def _render_filter_expr(
        entity_query_alias: str,
        query_value_parameter: str,
        expr: FilterExpr,
    ) -> tuple[str, dict[str, FilterValue]]:
        _render = Neo4jVectorGraphStore._render_filter_expr
        _sanitize = Neo4jVectorGraphStore._sanitize_name

        if isinstance(expr, FilterIsNull):
            field_ref = (
                f"{entity_query_alias}.{_sanitize(mangle_property_name(expr.field))}"
            )
            return f"{field_ref} IS NULL", {}

        if isinstance(expr, FilterIn):
            field_ref = (
                f"{entity_query_alias}.{_sanitize(mangle_property_name(expr.field))}"
            )
            param_name = _sanitize(f"filter_expr_param_{uuid4()}")
            condition = f"{field_ref} IN ${query_value_parameter}.{param_name}"
            params = {param_name: expr.values}
            return condition, params

        if isinstance(expr, FilterComparison):
            field_ref = (
                f"{entity_query_alias}.{_sanitize(mangle_property_name(expr.field))}"
            )
            param_name = _sanitize(f"filter_expr_param_{uuid4()}")
            condition = render_comparison(
                left=field_ref,
                op=expr.op,
                right=f"${query_value_parameter}.{param_name}",
                value=expr.value,
            )
            params = {
                param_name: cast(FilterValue, sanitize_value_for_neo4j(expr.value))
            }
            return condition, params

        if isinstance(expr, FilterAnd):
            left_cond, left_params = _render(
                entity_query_alias, query_value_parameter, expr.left
            )
            right_cond, right_params = _render(
                entity_query_alias, query_value_parameter, expr.right
            )
            return f"({left_cond}) AND ({right_cond})", left_params | right_params
        if isinstance(expr, FilterOr):
            left_cond, left_params = _render(
                entity_query_alias, query_value_parameter, expr.left
            )
            right_cond, right_params = _render(
                entity_query_alias, query_value_parameter, expr.right
            )
            return f"({left_cond}) OR ({right_cond})", left_params | right_params
        if isinstance(expr, FilterNot):
            inner_cond, inner_params = _render(
                entity_query_alias, query_value_parameter, expr.expr
            )
            return f"NOT ({inner_cond})", inner_params
        raise TypeError(f"Unsupported filter expression type: {type(expr)!r}")
