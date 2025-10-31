"""
Abstract base class for a vector graph store.

Defines the interface for adding, searching,
and deleting nodes and edges.
"""

from abc import ABC, abstractmethod
from collections.abc import Iterable, Mapping
from typing import Any
from uuid import UUID

from memmachine.common.embedder import SimilarityMetric

from .data_types import Edge, Node, Property


class VectorGraphStore(ABC):
    """
    Abstract base class for a vector graph store.
    """

    @abstractmethod
    async def add_nodes(self, collection: str, nodes: Iterable[Node]):
        """
        Add nodes to the graph store.

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
        relation: str,
        source_collection: str,
        target_collection: str,
        edges: Iterable[Edge],
    ):
        """
        Add edges to the graph store.

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
        collection: str,
        query_embedding: list[float],
        embedding_property_name: str,
        similarity_metric: SimilarityMetric = SimilarityMetric.COSINE,
        limit: int | None = 100,
        required_properties: Mapping[str, Property] | None = None,
        include_missing_properties: bool = False,
    ) -> list[Node]:
        """
        Search for nodes with embeddings similar to the query embedding.

        Args:
            collection (str):
                Collection that the nodes belong to.
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
            required_properties (Mapping[str, Property] | None, optional):
                Mapping of property names to their required values
                that the nodes must have.
                If None or empty, no property filtering is applied
                (default: None).
            include_missing_properties (bool, optional):
                If True, nodes missing any of the required properties
                will also be included in the results
                (default: False).

        Returns:
            list[Node]:
                List of Node objects
                that are similar to the query embedding.
        """
        raise NotImplementedError

    @abstractmethod
    async def search_related_nodes(
        self,
        collection: str,
        node_uuid: UUID,
        allowed_relations: Iterable[str] | None = None,
        find_sources: bool = True,
        find_targets: bool = True,
        limit: int | None = None,
        required_properties: Mapping[str, Property] | None = None,
        include_missing_properties: bool = False,
    ) -> list[Node]:
        """
        Search for nodes related to the specified node via edges.

        Args:
            collection (str):
                Collection that the nodes belong to.
            node_uuid (UUID):
                UUID of the node to find related nodes for.
            allowed_relations(Iterable[str] | None, optional):
                Iterable of relations to consider.
                If None, all relations are considered.
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
            required_properties (Mapping[str, Property] | None, optional):
                Mapping of property names to their required values
                that the nodes must have.
                If None or empty, no property filtering is applied
                (default: None).
            include_missing_properties (bool, optional):
                If True, nodes missing any of the required properties
                will also be included in the results
                (default: False).

        Returns:
            list[Node]:
                List of Node objects
                that are related to the specified node.
        """
        raise NotImplementedError

    @abstractmethod
    async def search_directional_nodes(
        self,
        collection: str,
        by_property: str,
        start_at_value: Any | None = None,
        include_equal_start_at_value: bool = False,
        order_ascending: bool = True,
        limit: int | None = 1,
        required_properties: Mapping[str, Property] | None = None,
        include_missing_properties: bool = False,
    ) -> list[Node]:
        """
        Search for nodes ordered by a specific property.

        Args:
            collection (str):
                Collection that the nodes belong to.
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
            required_properties (Mapping[str, Property] | None, optional):
                Mapping of property names to their required values
                that the nodes must have.
                If None or empty, no property filtering is applied
                (default: None).
            include_missing_properties (bool, optional):
                If True, nodes missing any of the required properties
                will also be included in the results
                (default: False).
                will also be included in the results.

        Returns:
            list[Node]:
                List of Node objects ordered by the specified property.
        """
        raise NotImplementedError

    @abstractmethod
    async def search_matching_nodes(
        self,
        collection: str,
        limit: int | None = None,
        required_properties: Mapping[str, Property] | None = None,
        include_missing_properties: bool = False,
    ) -> list[Node]:
        """
        Search for nodes matching the specified properties.

        Args:
            limit (int | None, optional):
                Maximum number of nodes to return.
                If None, return as many matching nodes as possible
                (default: None).
            required_properties (Mapping[str, Property] | None, optional):
                Mapping of property names to their required values
                that the nodes must have.
                If None or empty, no property filtering is applied
                (default: None).
            include_missing_properties (bool, optional):
                If True, nodes missing any of the required properties
                will also be included in the results
                (default: False).

        Returns:
            list[Node]:
                List of Node objects matching the specified criteria.
        """
        raise NotImplementedError

    @abstractmethod
    async def delete_nodes(
        self,
        node_uuids: Iterable[UUID],
    ):
        """
        Delete nodes from the graph store.

        Args:
            node_uuids (Iterable[UUID]):
                Iterable of UUIDs of the nodes to delete.
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
