"""
Abstract base class for a vector graph store.

Defines the interface for adding, searching,
and deleting nodes and edges.
"""

from abc import ABC, abstractmethod
from collections.abc import Iterable

from memmachine.common.data_types import SimilarityMetric
from memmachine.common.filter.filter_parser import (
    FilterExpr,
)

from .data_types import Edge, Node, OrderedPropertyValue


class VectorGraphStore(ABC):
    """Abstract base class for a vector graph store."""

    @abstractmethod
    async def add_nodes(
        self,
        *,
        collection: str,
        nodes: Iterable[Node],
    ) -> None:
        """
        Add nodes to the vector graph store.

        Args:
            collection (str):
                Collection that the nodes belong to.
            nodes (Iterable[Node]):
                Iterable of Node objects to add.

        """
        raise NotImplementedError

    @abstractmethod
    async def add_edges(
        self,
        *,
        relation: str,
        source_collection: str,
        target_collection: str,
        edges: Iterable[Edge],
    ) -> None:
        """
        Add edges to the vector graph store.

        Args:
            relation (str):
                Relation that the edges represent.
            source_collection (str):
                Collection that the source nodes belong to.
            target_collection (str):
                Collection that the target nodes belong to.
            edges (Iterable[Edge]):
                Iterable of Edge objects to add.

        """
        raise NotImplementedError

    @abstractmethod
    async def search_similar_nodes(
        self,
        *,
        collection: str,
        embedding_name: str,
        query_embedding: list[float],
        similarity_metric: SimilarityMetric = SimilarityMetric.COSINE,
        limit: int | None = 100,
        property_filter: FilterExpr | None = None,
    ) -> list[Node]:
        """
        Search for nodes with embeddings similar to the query embedding.

        Args:
            collection (str):
                Collection that the nodes belong to.
            embedding_name (str):
                The name of the embedding vector property.
            query_embedding (list[float]):
                The embedding vector to compare against.
            similarity_metric (SimilarityMetric):
                The similarity metric to use
                (default: SimilarityMetric.COSINE).
            limit (int | None):
                Maximum number of similar nodes to return.
                If None, return as many similar nodes as possible
                (default: 100).
            property_filter (FilterExpr | None):
                Filter expression tree.
                If None or empty, no property filtering is applied
                (default: None).

        Returns:
            list[Node]:
                List of Node objects
                that are similar to the query embedding.

        """
        raise NotImplementedError

    @abstractmethod
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
        """
        Search for nodes related to the specified node via edges.

        Args:
            relation (str):
                Relation that the edges represent.
            other_collection (str):
                Collection that the related nodes belong to.
            this_collection (str):
                Collection that the specified node belongs to.
            this_node_uid (str):
                UID of the node to find related nodes for.
            find_sources (bool):
                Whether to return nodes
                that are sources of edges
                pointing to the specified node
                (default: True).
            find_targets (bool):
                Whether to return nodes
                that are targets of edges
                originating from the specified node
                (default: True).
            limit (int | None):
                Maximum number of related nodes to return.
                If None, return as many related nodes as possible
                (default: None).
            edge_property_filter (FilterExpr | None):
                Filter expression tree for edge properties.
                If None or empty, no property filtering is applied
                (default: None).
            node_property_filter (FilterExpr | None):
                Filter expression tree for node properties.
                If None or empty, no property filtering is applied
                (default: None).

        Returns:
            list[Node]:
                List of Node objects
                that are related to the specified node.

        """
        raise NotImplementedError

    @abstractmethod
    async def search_directional_nodes(
        self,
        *,
        collection: str,
        by_properties: Iterable[str],
        starting_at: Iterable[OrderedPropertyValue | None],
        order_ascending: Iterable[bool],
        include_equal_start: bool = False,
        limit: int | None = 1,
        property_filter: FilterExpr | None = None,
    ) -> list[Node]:
        """
        Search for nodes ordered by a specific property.

        Args:
            collection (str):
                Collection that the nodes belong to.
            by_properties (Iterable[str]):
                Hierarchy of property names to order the nodes by.
            starting_at (Iterable[OrderedPropertyValue]):
                Values for each property to start the search from.
                If a value is None, start from the minimum or maximum
                based on order_ascending.
            order_ascending (Iterable[bool]):
                Whether to order each property ascending (True) or descending (False).
            include_equal_start (bool):
                Whether to include nodes with all property values
                equal to the starting_at values.
            limit (int | None):
                Maximum number of nodes to return.
                If None, return as many matching nodes as possible
                (default: 1).
            property_filter (FilterExpr | None):
                Filter expression tree.
                If None or empty, no property filtering is applied
                (default: None).

        Returns:
            list[Node]:
                List of Node objects ordered by the specified property.

        """
        raise NotImplementedError

    @abstractmethod
    async def search_matching_nodes(
        self,
        *,
        collection: str,
        limit: int | None = None,
        property_filter: FilterExpr | None = None,
    ) -> list[Node]:
        """
        Search for nodes matching the specified properties.

        Args:
            collection (str):
                Name of the collection to search.
            limit (int | None):
                Maximum number of nodes to return.
                If None, return as many matching nodes as possible
                (default: None).
            property_filter (FilterExpr | None):
                Filter expression tree.
                If None or empty, no property filtering is applied
                (default: None).

        Returns:
            list[Node]:
                List of Node objects matching the specified criteria.

        """
        raise NotImplementedError

    @abstractmethod
    async def get_nodes(
        self,
        *,
        collection: str,
        node_uids: Iterable[str],
    ) -> list[Node]:
        """
        Get nodes from the collection.

        Args:
            collection (str):
                Name of the collection containing the nodes.
            node_uids (Iterable[str]):
                Iterable of UIDs of the nodes to retrieve.

        Returns:
            list[Node]:
                List of Node objects with the specified UIDs.
                Order is not guaranteed.

        """
        raise NotImplementedError

    @abstractmethod
    async def delete_nodes(
        self,
        *,
        collection: str,
        node_uids: Iterable[str],
    ) -> None:
        """
        Delete nodes from the collection.

        Args:
            collection (str):
                Name of the collection containing the nodes.
            node_uids (Iterable[str]):
                Iterable of UIDs of the nodes to delete.

        """
        raise NotImplementedError

    @abstractmethod
    async def delete_all_data(self) -> None:
        """Delete all data from the vector graph store."""
        raise NotImplementedError

    @abstractmethod
    async def close(self) -> None:
        """Shut down and release resources."""
        raise NotImplementedError
