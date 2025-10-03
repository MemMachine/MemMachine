"""
Neo4j-based vector graph store implementation.

This module provides an asynchronous implementation
of a vector graph store using Neo4j as the backend database.
"""

import asyncio
from typing import Any
from uuid import UUID

from neo4j import AsyncGraphDatabase
from neo4j.graph import Node as Neo4jNode
from neo4j.time import DateTime as Neo4jDateTime

from memmachine.common.utils import async_with

from .data_types import Edge, Node, Property
from .vector_graph_store import VectorGraphStore


# https://neo4j.com/developer/kb/protecting-against-cypher-injection
# Node labels, relationship types, and property names
# cannot be parameterized.
class Neo4jVectorGraphStore(VectorGraphStore):
    """
    Asynchronous Neo4j-based implementation of VectorGraphStore.
    """

    def __init__(self, config: dict[str, Any]):
        """
        Initialize a Neo4jVectorGraphStore
        with the provided configuration.

        Args:
            config (dict[str, Any]):
                Configuration dictionary containing:
                - uri (str):
                    Neo4j connection URI.
                - username (str):
                    Neo4j username.
                - password (str):
                    Neo4j password.
                - max_concurrent_transactions (int, optional):
                    Maximum number of concurrent transactions
                    (default: 100).

        Raises:
            ValueError:
                If configuration argument values are missing or invalid.
            TypeError:
                If configuration argument values are of incorrect type.
        """
        super().__init__()

        uri = config.get("uri")
        if uri is None:
            raise ValueError("Neo4j URI must be provided")
        if not isinstance(uri, str):
            raise TypeError("Neo4j URI must be a string")

        username = config.get("username")
        if username is None:
            raise ValueError("Neo4j username must be provided")
        if not isinstance(username, str):
            raise TypeError("Neo4j username must be a string")

        password = config.get("password")
        if password is None:
            raise ValueError("Neo4j password must be provided")
        if not isinstance(password, str):
            raise TypeError("Neo4j password must be a string")

        max_concurrent_transactions = config.get("max_concurrent_transactions", 100)
        if not isinstance(max_concurrent_transactions, int):
            raise TypeError(
                "Maximum number of concurrent transactions must be an integer"
            )
        if max_concurrent_transactions <= 0:
            raise ValueError(
                "Maximum number of concurrent transactions must be positive"
            )

        self._semaphore = asyncio.Semaphore(max_concurrent_transactions)

        self._driver = AsyncGraphDatabase.driver(uri, auth=(username, password))

    async def add_nodes(self, nodes: list[Node]):
        labels_nodes_map: dict[tuple[str, ...], list[Node]] = {}
        for node in nodes:
            labels_nodes_map.setdefault(tuple(sorted(node.labels)), []).append(node)

        add_nodes_tasks = [
            async_with(
                self._semaphore,
                self._driver.execute_query(
                    "UNWIND $nodes AS node\n"
                    f"CREATE (n{
                        Neo4jVectorGraphStore._format_labels(set(labels))
                    } {{uuid: node.uuid}})\n"
                    "SET n += node.properties",
                    nodes=[
                        {
                            "uuid": str(node.uuid),
                            "properties": node.properties,
                        }
                        for node in nodes
                    ],
                ),
            )
            for labels, nodes in labels_nodes_map.items()
        ]

        await asyncio.gather(*add_nodes_tasks)

    async def add_edges(self, edges: list[Edge]):
        relation_edges_map: dict[str, list[Edge]] = {}
        for edge in edges:
            relation_edges_map.setdefault(edge.relation, []).append(edge)

        add_edges_tasks = [
            async_with(
                self._semaphore,
                self._driver.execute_query(
                    "UNWIND $edges AS edge\n"
                    "MATCH"
                    "    (source {uuid: edge.source_uuid}),"
                    "    (target {uuid: edge.target_uuid})\n"
                    "CREATE (source)"
                    f"    -[r:{relation} {{uuid: edge.uuid}}]->"
                    "    (target)\n"
                    "SET r += edge.properties",
                    edges=[
                        {
                            "uuid": str(edge.uuid),
                            "source_uuid": str(edge.source_uuid),
                            "target_uuid": str(edge.target_uuid),
                            "properties": edge.properties,
                        }
                        for edge in edges
                    ],
                ),
            )
            for relation, edges in relation_edges_map.items()
        ]

        await asyncio.gather(*add_edges_tasks)

    async def search_similar_nodes(
        self,
        query_embedding: list[float],
        similarity_threshold: float = 0.2,
        limit: int | None = 100,
        required_labels: set[str] | None = None,
        required_properties: dict[str, Property] = {},
        include_missing_properties: bool = False,
    ) -> list[Node]:
        async with self._semaphore:
            records, _, _ = await self._driver.execute_query(
                f"MATCH (n{Neo4jVectorGraphStore._format_labels(required_labels)})\n"
                "WHERE n.embedding IS NOT NULL\n"
                f"AND {
                    Neo4jVectorGraphStore._format_required_properties(
                        required_properties, include_missing_properties
                    )
                }\n"
                "WITH n,"
                "    vector.similarity.cosine("
                "        n.embedding, $query_embedding"
                "    ) AS similarity\n"
                "WHERE similarity > $similarity_threshold\n"
                "RETURN n\n"
                "ORDER BY similarity DESC\n"
                f"{'LIMIT $limit' if limit is not None else ''}",
                query_embedding=query_embedding,
                similarity_threshold=similarity_threshold,
                limit=limit,
                required_properties=required_properties,
            )

        similar_neo4j_nodes = [record["n"] for record in records]
        return Neo4jVectorGraphStore._nodes_from_neo4j_nodes(similar_neo4j_nodes)

    async def search_related_nodes(
        self,
        node_uuid: UUID,
        allowed_relations: set[str] | None = None,
        find_sources: bool = True,
        find_targets: bool = True,
        limit: int | None = None,
        required_labels: set[str] | None = None,
        required_properties: dict[str, Property] = {},
        include_missing_properties: bool = False,
    ) -> list[Node]:
        if not (find_sources or find_targets):
            return []

        search_related_nodes_tasks = [
            async_with(
                self._semaphore,
                self._driver.execute_query(
                    "MATCH\n"
                    "    (m {uuid: $node_uuid})"
                    f"    {'-' if find_targets else '<-'}"
                    f"    [{f':{relation}' if relation is not None else ''}]"
                    f"    {'-' if find_sources else '->'}"
                    f"    (n{Neo4jVectorGraphStore._format_labels(required_labels)})"
                    f"WHERE {
                        Neo4jVectorGraphStore._format_required_properties(
                            required_properties, include_missing_properties
                        )
                    }\n"
                    "RETURN n\n"
                    f"{'LIMIT $limit' if limit is not None else ''}",
                    node_uuid=str(node_uuid),
                    limit=limit,
                    required_properties=required_properties,
                ),
            )
            for relation in (allowed_relations or [None])
        ]

        results = await asyncio.gather(*search_related_nodes_tasks)

        related_nodes: set[Node] = set()
        for records, _, _ in results:
            related_neo4j_nodes = [record["n"] for record in records]
            related_nodes.update(
                Neo4jVectorGraphStore._nodes_from_neo4j_nodes(related_neo4j_nodes)
            )

        return list(related_nodes)[:limit]

    async def search_directional_nodes(
        self,
        by_property: str,
        start_at_value: Any | None = None,
        include_equal_start_at_value: bool = False,
        order_ascending: bool = True,
        limit: int | None = 1,
        required_labels: set[str] | None = None,
        required_properties: dict[str, Property] = {},
        include_missing_properties: bool = False,
    ) -> list[Node]:
        async with self._semaphore:
            records, _, _ = await self._driver.execute_query(
                f"MATCH (n{Neo4jVectorGraphStore._format_labels(required_labels)})\n"
                f"WHERE n.{by_property} IS NOT NULL\n"
                f"{
                    (
                        f'AND n.{by_property}'
                        + ('>' if order_ascending else '<')
                        + ('=' if include_equal_start_at_value else '')
                        + '$start_at_value'
                    )
                    if start_at_value is not None
                    else ''
                }\n"
                f"AND {
                    Neo4jVectorGraphStore._format_required_properties(
                        required_properties, include_missing_properties
                    )
                }\n"
                "RETURN n\n"
                f"ORDER BY n.{by_property} {'ASC' if order_ascending else 'DESC'}\n"
                f"{'LIMIT $limit' if limit is not None else ''}",
                start_at_value=start_at_value,
                limit=limit,
                required_properties=required_properties,
            )

        directional_proximal_neo4j_nodes = [record["n"] for record in records]
        return Neo4jVectorGraphStore._nodes_from_neo4j_nodes(
            directional_proximal_neo4j_nodes
        )

    async def search_matching_nodes(
        self,
        limit: int | None = None,
        required_labels: set[str] | None = None,
        required_properties: dict[str, Property] = {},
        include_missing_properties: bool = False,
    ) -> list[Node]:
        async with self._semaphore:
            records, _, _ = await self._driver.execute_query(
                f"MATCH (n{Neo4jVectorGraphStore._format_labels(required_labels)})\n"
                f"WHERE {
                    Neo4jVectorGraphStore._format_required_properties(
                        required_properties, include_missing_properties
                    )
                }\n"
                "RETURN n\n"
                f"{'LIMIT $limit' if limit is not None else ''}",
                limit=limit,
                required_properties=required_properties,
            )

        matching_neo4j_nodes = [record["n"] for record in records]
        return Neo4jVectorGraphStore._nodes_from_neo4j_nodes(matching_neo4j_nodes)

    async def delete_nodes(
        self,
        node_uuids: list[UUID],
    ):
        async with self._semaphore:
            await self._driver.execute_query(
                """
                UNWIND $node_uuids AS node_uuid
                MATCH (n {uuid: node_uuid})
                DETACH DELETE n
                """,
                node_uuids=[str(node_uuid) for node_uuid in node_uuids],
            )

    async def clear_data(self):
        async with self._semaphore:
            await self._driver.execute_query("MATCH (n) DETACH DELETE n")

    async def close(self):
        await self._driver.close()

    @staticmethod
    def _format_labels(labels: set[str] | None) -> str:
        """
        Format a set of labels for use in a Cypher query.

        Args:
            labels (set[str] | None):
                Set of labels to format.

        Returns:
            str:
                Formatted labels string for Cypher query.
        """
        return "".join([f":{label}" for label in labels]) if labels is not None else ""

    @staticmethod
    def _format_required_properties(
        required_properties: dict[str, Property],
        include_missing_properties: bool,
    ) -> str:
        """
        Format required properties for use in a Cypher query.

        Args:
            required_properties (dict[str, Property]):
                Dictionary of required properties.
            include_missing_properties (bool):
                Whether to include results
                with missing required properties.

        Returns:
            str:
                Formatted required properties string for Cypher query.
        """
        return (
            " AND ".join(
                [
                    f"(n.{property_name}"
                    f"    = $required_properties.{property_name}"
                    f"{
                        f' OR n.{property_name} IS NULL'
                        if include_missing_properties
                        else ''
                    })"
                    for property_name in required_properties.keys()
                ]
            )
            or "TRUE"
        )

    @staticmethod
    def _nodes_from_neo4j_nodes(
        neo4j_nodes: list[Neo4jNode],
    ) -> list[Node]:
        """
        Convert a list of Neo4j Node objects to a list of Node objects.

        Args:
            neo4j_nodes (list[Neo4jNode]): List of Neo4j Node objects.

        Returns:
            list[Node]: List of Node objects.
        """
        return [
            Node(
                uuid=UUID(neo4j_node["uuid"]),
                labels=set(neo4j_node.labels),
                properties={
                    key: Neo4jVectorGraphStore._python_value_from_neo4j_value(value)
                    for key, value in neo4j_node.items()
                    if key != "uuid"
                },
            )
            for neo4j_node in neo4j_nodes
        ]

    @staticmethod
    def _python_value_from_neo4j_value(value: Any) -> Any:
        """
        Convert a Neo4j value to a native Python value.

        Args:
            value (Any): The Neo4j value to convert.

        Returns:
            Any: The converted Python value.
        """
        if isinstance(value, Neo4jDateTime):
            return value.to_native()
        return value
