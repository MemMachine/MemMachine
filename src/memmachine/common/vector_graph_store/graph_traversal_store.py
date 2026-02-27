"""Protocol for graph-aware traversal and filtered search on a vector graph store."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Protocol, runtime_checkable

from memmachine.common.data_types import SimilarityMetric
from memmachine.common.filter.filter_parser import FilterExpr

from .data_types import GraphFilter, MultiHopResult, Node


@runtime_checkable
class GraphTraversalStore(Protocol):
    """Protocol for stores that support graph-native traversal and search.

    Backends that implement this protocol provide multi-hop relationship
    traversal and graph-structure-aware filtered vector search in addition
    to the base :class:`VectorGraphStore` interface.

    Callers should use ``isinstance(store, GraphTraversalStore)`` to check
    whether a particular backend supports these operations.
    """

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

        Args:
            collection: Collection that the starting node belongs to.
            this_node_uid: UID of the starting node.
            min_hops: Minimum number of hops (default 1).
            max_hops: Maximum number of hops (default 3, clamped to
                a configured ceiling).
            relation_types: Optional relationship type filter.  When
                provided, only paths using these relationship types are
                traversed.  These names are **sanitized** by the
                implementation before being used in the query.
            raw_relation_types: Optional relationship type names that
                are already in their final Neo4j form and should
                **not** be sanitized.  This is needed for cross-layer
                traversals where different subsystems use different
                naming conventions (e.g. episodic ``DERIVED_FROM``
                is sanitized but semantic ``EXTRACTED_FROM`` is not).
                Both ``relation_types`` and ``raw_relation_types`` are
                combined with ``|`` in the traversal pattern.
            score_decay: Decay factor applied per hop.  A result at
                distance *d* receives score ``score_decay ** d``.
                Set to ``1.0`` to disable decay scoring.
            limit: Maximum number of results to return.
            edge_property_filter: Filter applied to every edge in the path.
            node_property_filter: Filter applied to each result node.
            target_collections: Optional collection filter for end nodes.
                When ``None`` (the default), end nodes from any collection
                are returned, enabling cross-collection traversal.  When
                provided, only end nodes belonging to at least one of the
                specified collections are included.

        Returns:
            List of :class:`MultiHopResult` items sorted by score descending
            (or hop distance ascending when *score_decay* is ``1.0``).
            Each node appears at most once, using the shortest path distance.

        """
        ...

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
        computes vector similarity only on those candidates.

        When *graph_filter* is ``None`` the method behaves identically to
        :meth:`VectorGraphStore.search_similar_nodes`.

        Args:
            collection: Collection that the nodes belong to.
            embedding_name: Name of the embedding vector property.
            query_embedding: Query embedding vector.
            similarity_metric: Similarity metric to use.
            limit: Maximum number of similar nodes to return.
            property_filter: Optional property filter expression.
            graph_filter: Optional graph structure filter that narrows the
                candidate set before vector similarity is computed.

        Returns:
            List of matching :class:`Node` objects ordered by similarity.

        """
        ...
