"""
Abstract base class for a vector graph store.

Defines the interface for adding, searching,
and deleting nodes and edges.
"""

from abc import ABC, abstractmethod
from collections.abc import Collection, Mapping
from typing import Any
from uuid import UUID

from memmachine.common.embedder import SimilarityMetric

from .data_types import Edge, Node, Property


class VectorGraphStore(ABC):
    """
    Abstract base class for a vector graph store.
    """

    @abstractmethod
    async def add_nodes(self, nodes: Collection[Node]):
        """
        Add nodes to the graph store.

        Args:
            nodes (Collection[Node]): Collection of Node objects to add.
        """
        raise NotImplementedError

    @abstractmethod
    async def add_edges(self, edges: Collection[Edge]):
        """
        Add edges to the graph store.

        Args:
            edges (Collection[Edge]): Collection of Edge objects to add.
        """
        raise NotImplementedError

    @abstractmethod
    async def search_similar_nodes(
        self,
        query_embedding: list[float],
        embedding_property_name: str,
        similarity_metric: SimilarityMetric = SimilarityMetric.COSINE,
        limit: int | None = 100,
        required_labels: Collection[str] | None = None,
        required_properties: Mapping[str, Property] = {},
        include_missing_properties: bool = False,
    ) -> list[Node]:
        """
        Search for nodes with embeddings similar to the query embedding.

        Args:
            query_embedding (list[float]):
                The embedding vector to compare against.
            embedding_property_name (str):
                The name of the property
                that stores the embedding vector.
            similarity_metric (SimilarityMetric, optional):
                The similarity metric to use
                (default: SimilarityMetric.COSINE).
            limit (int | None, optional):
                Maximum number of similar nodes to return.
                If None, return as many similar nodes as possible
                (default: 100).
            required_labels (Collection[str] | None, optional):
                Collection of labels that the nodes must have.
                If None, no label filtering is applied.
            required_properties (Mapping[str, Property], optional):
                Mapping of property names to their required values
                that the nodes must have.
                If empty, no property filtering is applied.
            include_missing_properties (bool, optional):
                If True, nodes missing any of the required properties
                will also be included in the results.

        Returns:
            list[Node]:
                List of Node objects
                that are similar to the query embedding.
        """
        raise NotImplementedError

    @abstractmethod
    async def search_related_nodes(
        self,
        node_uuid: UUID,
        allowed_relations: Collection[str] | None = None,
        find_sources: bool = True,
        find_targets: bool = True,
        limit: int | None = None,
        required_labels: Collection[str] | None = None,
        required_properties: Mapping[str, Property] = {},
        include_missing_properties: bool = False,
    ) -> list[Node]:
        """
        Search for nodes related to the specified node via edges.

        Args:
            node_uuid (UUID):
                UUID of the node to find related nodes for.
            allowed_relations (Collection[str] | None, optional):
                Collection of relationship types to consider.
                If None, all relationship types are considered.
            find_sources (bool, optional):
                If True, search for nodes
                that are sources of edges
                pointing to the specified node.
            find_targets (bool, optional):
                If True, search for nodes
                that are targets of edges
                originating from the specified node.
            limit (int | None, optional):
                Maximum number of related nodes to return.
                If None, return as many related nodes as possible
                (default: None).
            required_labels (Collection[str] | None, optional):
                Collection of labels that the related nodes must have.
                If None, no label filtering is applied.
            required_properties (Mapping[str, Property], optional):
                Mapping of property names to their required values
                that the nodes must have.
                If empty, no property filtering is applied.
            include_missing_properties (bool, optional):
                If True, nodes missing any of the required properties
                will also be included in the results.

        Returns:
            list[Node]:
                List of Node objects
                that are related to the specified node.
        """
        raise NotImplementedError

    @abstractmethod
    async def search_directional_nodes(
        self,
        by_property: str,
        start_at_value: Any | None = None,
        include_equal_start_at_value: bool = False,
        order_ascending: bool = True,
        limit: int | None = 1,
        required_labels: Collection[str] | None = None,
        required_properties: Mapping[str, Property] = {},
        include_missing_properties: bool = False,
    ) -> list[Node]:
        """
        Search for nodes ordered by a specific property.

        Args:
            by_property (str):
                The property name to order the nodes by.
            start_at_value (Any | None, optional):
                The value to start the search from.
                If None, start from the beginning or end
                based on order_ascending.
            include_equal_start_at_value (bool, optional):
                If True, include nodes with property value
                equal to start_at_value.
            order_ascending (bool, optional):
                If True, order nodes in ascending order.
                If False, order in descending order.
            limit (int | None, optional):
                Maximum number of nodes to return.
                If None, return as many matching nodes as possible
                (default: 1).
            required_labels (Collection[str] | None, optional):
                Collection of labels that the nodes must have.
                If None, no label filtering is applied.
            required_properties (Mapping[str, Property], optional):
                Mapping of property names to their required values
                that the nodes must have.
                If empty, no property filtering is applied.
            include_missing_properties (bool, optional):
                If True, nodes missing any of the required properties
                will also be included in the results.

        Returns:
            list[Node]:
                List of Node objects ordered by the specified property.
        """
        raise NotImplementedError

    @abstractmethod
    async def search_matching_nodes(
        self,
        limit: int | None = None,
        required_labels: Collection[str] | None = None,
        required_properties: Mapping[str, Property] = {},
        include_missing_properties: bool = False,
    ) -> list[Node]:
        """
        Search for nodes matching the specified labels and properties.

        Args:
            limit (int | None, optional):
                Maximum number of nodes to return.
                If None, return as many matching nodes as possible
                (default: None).
            required_labels (Collection[str] | None, optional):
                Collection of labels that the nodes must have.
                If None, no label filtering is applied.
            required_properties (Mapping[str, Property], optional):
                Mapping of property names to their required values
                that the nodes must have.
                If empty, no property filtering is applied.
            include_missing_properties (bool, optional):
                If True, nodes missing any of the required properties
                will also be included in the results.

        Returns:
            list[Node]:
                List of Node objects matching the specified criteria.
        """
        raise NotImplementedError

    @abstractmethod
    async def delete_nodes(
        self,
        node_uuids: Collection[UUID],
    ):
        """
        Delete nodes from the graph store.

        Args:
            node_uuids (Collection[UUID]):
                Collection of UUIDs of the nodes to delete.
        """
        raise NotImplementedError

    @abstractmethod
    async def clear_data(self):
        """
        Clear all data from the graph store.
        """
        raise NotImplementedError

    @abstractmethod
    async def close(self):
        """
        Shut down and release resources.
        """
        raise NotImplementedError
