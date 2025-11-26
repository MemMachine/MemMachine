"""
Neo4j-based vector graph store implementation.

This module provides an asynchronous implementation
of a vector graph store using Neo4j as the backend database.
"""

import asyncio
import logging
import re
from collections.abc import Awaitable, Collection, Mapping
from typing import Any, cast
from uuid import UUID

from neo4j import AsyncDriver
from neo4j.graph import Node as Neo4jNode
from neo4j.time import DateTime as Neo4jDateTime
from pydantic import BaseModel, Field, InstanceOf

from memmachine.common.embedder import SimilarityMetric
from memmachine.common.utils import async_locked, async_with

from .data_types import Edge, Node, Property
from .vector_graph_store import VectorGraphStore

logger = logging.getLogger(__name__)


class Neo4jVectorGraphStoreParams(BaseModel):
    """
    Parameters for Neo4jVectorGraphStore.

    Attributes:
        driver (neo4j.AsyncDriver):
            Async Neo4j driver instance.
        max_concurrent_transactions (int):
            Maximum number of concurrent transactions
            (default: 100).
        force_exact_similarity_search (bool):
            Whether to force exact similarity search.
            (default: False).
    """

    driver: InstanceOf[AsyncDriver] = Field(
        ..., description="Async Neo4j driver instance"
    )
    max_concurrent_transactions: int = Field(
        100, description="Maximum number of concurrent transactions", gt=0
    )
    force_exact_similarity_search: bool = Field(
        False, description="Whether to force exact similarity search"
    )


# https://neo4j.com/developer/kb/protecting-against-cypher-injection
# Node labels, relationship types, and property names
# cannot be parameterized.
class Neo4jVectorGraphStore(VectorGraphStore):
    """
    Asynchronous Neo4j-based implementation of VectorGraphStore.
    """

    def __init__(self, params: Neo4jVectorGraphStoreParams):
        """
        Initialize a Neo4jVectorGraphStore
        with the provided parameters.

        Args:
            params (Neo4jVectorGraphStoreParams):
                Parameters for the Neo4jVectorGraphStore.
        """
        super().__init__()

        self._driver = params.driver

        self._semaphore = asyncio.Semaphore(params.max_concurrent_transactions)
        self._force_exact_similarity_search = params.force_exact_similarity_search

        self._vector_index_name_cache: set[str] = set()

    async def add_nodes(self, nodes: Collection[Node]):
        labels_nodes_map: dict[tuple[str, ...], list[Node]] = {}
        for node in nodes:
            labels_nodes_map.setdefault(tuple(sorted(node.labels)), []).append(node)

        add_nodes_tasks = [
            async_with(
                self._semaphore,
                self._driver.execute_query(
                    "UNWIND $nodes AS node\n"
                    f"CREATE (n{
                        Neo4jVectorGraphStore._format_labels(tuple(labels))
                    } {{uuid: node.uuid}})\n"
                    "SET n += node.properties",
                    nodes=[
                        {
                            "uuid": str(node.uuid),
                            "properties": {
                                Neo4jVectorGraphStore._sanitize_name(key): value
                                for key, value in node.properties.items()
                            },
                        }
                        for node in nodes
                    ],
                ),
            )
            for labels, nodes in labels_nodes_map.items()
        ]

        await asyncio.gather(*add_nodes_tasks)

    async def add_edges(self, edges: Collection[Edge]):
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
                    f"    -[r:{sanitized_relation} {{uuid: edge.uuid}}]->"
                    "    (target)\n"
                    "SET r += edge.properties",
                    edges=[
                        {
                            "uuid": str(edge.uuid),
                            "source_uuid": str(edge.source_uuid),
                            "target_uuid": str(edge.target_uuid),
                            "properties": {
                                Neo4jVectorGraphStore._sanitize_name(key): value
                                for key, value in edge.properties.items()
                            },
                        }
                        for edge in edges
                    ],
                ),
            )
            for sanitized_relation, edges in (
                (
                    Neo4jVectorGraphStore._sanitize_name(relation),
                    edges,
                )
                for relation, edges in relation_edges_map.items()
            )
        ]

        await asyncio.gather(*add_edges_tasks)

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
        exact_similarity_search = self._force_exact_similarity_search

        sanitized_embedding_property_name = Neo4jVectorGraphStore._sanitize_name(
            embedding_property_name
        )

        if not exact_similarity_search:
            vector_index_name = (
                Neo4jVectorGraphStore._node_vector_index_name(
                    Neo4jVectorGraphStore._sanitize_name(next(iter(required_labels))),
                    sanitized_embedding_property_name,
                )
                if required_labels is not None and len(required_labels) > 0
                else None
            )

            if vector_index_name is None:
                logger.warning(
                    "No labels specified for vector index lookup. "
                    "Falling back to exact similarity search."
                )
                exact_similarity_search = True

        # ANN search requires a finite limit.
        if limit is None and not exact_similarity_search:
            limit = 100_000

        if exact_similarity_search:
            match similarity_metric:
                case SimilarityMetric.COSINE:
                    vector_similarity_function = "vector.similarity.cosine"
                case SimilarityMetric.EUCLIDEAN:
                    vector_similarity_function = "vector.similarity.euclidean"
                case _:
                    vector_similarity_function = "vector.similarity.cosine"

            query = (
                f"MATCH (n{Neo4jVectorGraphStore._format_labels(required_labels)})\n"
                f"WHERE n.{sanitized_embedding_property_name} IS NOT NULL\n"
                f"AND {
                    Neo4jVectorGraphStore._format_required_properties(
                        'n', required_properties, include_missing_properties
                    )
                }\n"
                "WITH n,"
                f"    {vector_similarity_function}("
                f"        n.{sanitized_embedding_property_name}, $query_embedding"
                "    ) AS similarity\n"
                "RETURN n\n"
                "ORDER BY similarity DESC\n"
                f"{'LIMIT $limit' if limit is not None else ''}"
            )

            async with self._semaphore:
                records, _, _ = await self._driver.execute_query(
                    query,
                    query_embedding=query_embedding,
                    limit=limit,
                    required_properties={
                        Neo4jVectorGraphStore._sanitize_name(key): value
                        for key, value in required_properties.items()
                    },
                )

        else:
            await self._create_node_vector_index_if_not_exist(
                labels=cast(Collection[str], required_labels),
                embedding_property_name=embedding_property_name,
                dimensions=len(query_embedding),
                similarity_metric=similarity_metric,
            )

            query = (
                "CALL db.index.vector.queryNodes(\n"
                f"    $vector_index_name, $limit, $query_embedding\n"
                ")\n"
                "YIELD node AS n, score AS similarity\n"
                f"WHERE n{Neo4jVectorGraphStore._format_labels(required_labels)}\n"
                f"AND {
                    Neo4jVectorGraphStore._format_required_properties(
                        'n', required_properties, include_missing_properties
                    )
                }\n"
                "RETURN n"
            )

            async with self._semaphore:
                records, _, _ = await self._driver.execute_query(
                    query,
                    query_embedding=query_embedding,
                    limit=limit,
                    required_properties={
                        Neo4jVectorGraphStore._sanitize_name(key): value
                        for key, value in required_properties.items()
                    },
                    vector_index_name=vector_index_name,
                )

        similar_neo4j_nodes = [record["n"] for record in records]
        return Neo4jVectorGraphStore._nodes_from_neo4j_nodes(similar_neo4j_nodes)

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
        if not (find_sources or find_targets):
            return []

        query_typed_relations = (
            [
                f"[:{Neo4jVectorGraphStore._sanitize_name(relation)}]"
                for relation in allowed_relations
            ]
            if allowed_relations is not None
            else ["[]"]
        )

        search_related_nodes_tasks = [
            async_with(
                self._semaphore,
                self._driver.execute_query(
                    "MATCH\n"
                    "    (m {uuid: $node_uuid})"
                    f"    {'-' if find_targets else '<-'}"
                    f"    {query_typed_relation}"
                    f"    {'-' if find_sources else '->'}"
                    f"    (n{Neo4jVectorGraphStore._format_labels(required_labels)})"
                    f"WHERE {
                        Neo4jVectorGraphStore._format_required_properties(
                            'n',
                            required_properties,
                            include_missing_properties,
                        )
                    }\n"
                    "RETURN n\n"
                    f"{'LIMIT $limit' if limit is not None else ''}",
                    node_uuid=str(node_uuid),
                    limit=limit,
                    required_properties={
                        Neo4jVectorGraphStore._sanitize_name(key): value
                        for key, value in required_properties.items()
                    },
                ),
            )
            for query_typed_relation in query_typed_relations
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
        required_labels: Collection[str] | None = None,
        required_properties: Mapping[str, Property] = {},
        include_missing_properties: bool = False,
    ) -> list[Node]:
        sanitized_by_property = Neo4jVectorGraphStore._sanitize_name(by_property)

        async with self._semaphore:
            records, _, _ = await self._driver.execute_query(
                f"MATCH (n{Neo4jVectorGraphStore._format_labels(required_labels)})\n"
                f"WHERE n.{sanitized_by_property} IS NOT NULL\n"
                f"{
                    (
                        f'AND n.{sanitized_by_property}'
                        + ('>' if order_ascending else '<')
                        + ('=' if include_equal_start_at_value else '')
                        + '$start_at_value'
                    )
                    if start_at_value is not None
                    else ''
                }\n"
                f"AND {
                    Neo4jVectorGraphStore._format_required_properties(
                        'n', required_properties, include_missing_properties
                    )
                }\n"
                "RETURN n\n"
                f"ORDER BY n.{sanitized_by_property} {
                    'ASC' if order_ascending else 'DESC'
                }\n"
                f"{'LIMIT $limit' if limit is not None else ''}",
                start_at_value=start_at_value,
                limit=limit,
                required_properties={
                    Neo4jVectorGraphStore._sanitize_name(key): value
                    for key, value in required_properties.items()
                },
            )

        directional_proximal_neo4j_nodes = [record["n"] for record in records]
        return Neo4jVectorGraphStore._nodes_from_neo4j_nodes(
            directional_proximal_neo4j_nodes
        )

    async def search_matching_nodes(
        self,
        limit: int | None = None,
        required_labels: Collection[str] | None = None,
        required_properties: Mapping[str, Property] = {},
        include_missing_properties: bool = False,
    ) -> list[Node]:
        async with self._semaphore:
            records, _, _ = await self._driver.execute_query(
                f"MATCH (n{Neo4jVectorGraphStore._format_labels(required_labels)})\n"
                f"WHERE {
                    Neo4jVectorGraphStore._format_required_properties(
                        'n', required_properties, include_missing_properties
                    )
                }\n"
                "RETURN n\n"
                f"{'LIMIT $limit' if limit is not None else ''}",
                limit=limit,
                required_properties={
                    Neo4jVectorGraphStore._sanitize_name(key): value
                    for key, value in required_properties.items()
                },
            )

        matching_neo4j_nodes = [record["n"] for record in records]
        return Neo4jVectorGraphStore._nodes_from_neo4j_nodes(matching_neo4j_nodes)

    async def delete_nodes(
        self,
        node_uuids: Collection[UUID],
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

    async def _create_node_vector_index_if_not_exist(
        self,
        labels: Collection[str],
        embedding_property_name: str,
        dimensions: int,
        similarity_metric: SimilarityMetric = SimilarityMetric.COSINE,
    ):
        """
        Create node vector index(es) if not exist.

        Args:
            labels (Collection[str]):
                Collection of node labels to create vector indexes for.
            embedding_property_name (str):
                Name of the embedding property.
            dimensions (int):
                Dimensionality of the embedding vectors.
            similarity_metric (SimilarityMetric):
                Similarity metric to use for the vector index
                (default: SimilarityMetric.COSINE).
        """
        if not self._vector_index_name_cache:
            async with self._semaphore:
                records, _, _ = await self._driver.execute_query(
                    "SHOW VECTOR INDEXES YIELD name RETURN name"
                )

            self._vector_index_name_cache.update(record["name"] for record in records)

        sanitized_labels = [
            Neo4jVectorGraphStore._sanitize_name(label) for label in labels
        ]

        sanitized_embedding_property_name = Neo4jVectorGraphStore._sanitize_name(
            embedding_property_name
        )

        requested_vector_index_names = [
            Neo4jVectorGraphStore._node_vector_index_name(
                sanitized_label, sanitized_embedding_property_name
            )
            for sanitized_label in sanitized_labels
        ]

        info_for_vector_indexes_to_create = [
            (sanitized_label, sanitized_embedding_property_name, vector_index_name)
            for sanitized_label, vector_index_name in zip(
                sanitized_labels,
                requested_vector_index_names,
            )
            if vector_index_name not in self._vector_index_name_cache
        ]

        if len(info_for_vector_indexes_to_create) == 0:
            return

        match similarity_metric:
            case SimilarityMetric.COSINE:
                similarity_function = "cosine"
            case SimilarityMetric.EUCLIDEAN:
                similarity_function = "euclidean"
            case _:
                similarity_function = "cosine"

        create_index_tasks = [
            async_with(
                self._semaphore,
                self._driver.execute_query(
                    f"CREATE VECTOR INDEX {vector_index_name}\n"
                    "IF NOT EXISTS\n"
                    f"FOR (n:{sanitized_label})\n"
                    f"ON n.{sanitized_embedding_property_name}\n"
                    "OPTIONS {\n"
                    "    indexConfig: {\n"
                    "        `vector.dimensions`:\n"
                    "            $dimensions,\n"
                    "        `vector.similarity_function`:\n"
                    "            $similarity_function\n"
                    "    }\n"
                    "}",
                    dimensions=dimensions,
                    similarity_function=similarity_function,
                ),
            )
            for sanitized_label, sanitized_embedding_property_name, vector_index_name in info_for_vector_indexes_to_create
        ]

        await self._execute_create_node_vector_index_if_not_exist(create_index_tasks)

        self._vector_index_name_cache.update(requested_vector_index_names)

    @async_locked
    async def _execute_create_node_vector_index_if_not_exist(
        self, create_index_tasks: Collection[Awaitable]
    ):
        """
        Execute the creation of node vector indexes if not exist.
        Locked because Neo4j concurrent vector index creation
        can raise exceptions even with "IF NOT EXISTS".

        Args:
            create_index_tasks (Collection[Awaitable]):
                Collection of awaitable tasks to create vector indexes.
        """
        await asyncio.gather(*create_index_tasks)

        async with self._semaphore:
            await self._driver.execute_query("CALL db.awaitIndexes()")

    @staticmethod
    def _sanitize_name(name: str) -> str:
        """
        Sanitize a name to be used in Neo4j.
        https://neo4j.com/docs/cypher-manual/current/syntax/naming

        Args:
            name (str): The name to sanitize.

        Returns:
            str: The sanitized name.
        """
        return "".join(c if c.isalnum() else f"_u{ord(c):x}_" for c in name)

    @staticmethod
    def _desanitize_name(sanitized_name: str) -> str:
        """
        Desanitize a name from Neo4j.

        Args:
            sanitized_name (str): The sanitized name.

        Returns:
            str: The desanitized name.
        """
        return re.sub(
            r"_u([0-9a-fA-F]+)_",
            lambda match: chr(int(match[1], 16)),
            sanitized_name,
        )

    @staticmethod
    def _format_labels(labels: Collection[str] | None) -> str:
        """
        Format an iterable of labels for use in a Cypher query.

        Args:
            labels (Collection[str] | None):
                Collection of labels to format.

        Returns:
            str:
                Formatted labels string for Cypher query.
        """
        return (
            "".join(
                f":{Neo4jVectorGraphStore._sanitize_name(label)}" for label in labels
            )
            if labels is not None
            else ""
        )

    @staticmethod
    def _format_required_properties(
        entity_query_alias: str,
        required_properties: Mapping[str, Property],
        include_missing_properties: bool,
    ) -> str:
        """
        Format required properties for use in a Cypher query.

        Args:
            entity_query_alias (str):
                Alias of the node or relationship in the query
                (e.g., "n", "r").
            required_properties (Mapping[str, Property]):
                Mapping of required properties.
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
                    f"({entity_query_alias}.{sanitized_property_name}"
                    f"    = $required_properties.{sanitized_property_name}"
                    f"{
                        f' OR {entity_query_alias}.{sanitized_property_name} IS NULL'
                        if include_missing_properties
                        else ''
                    })"
                    for sanitized_property_name in (
                        Neo4jVectorGraphStore._sanitize_name(key)
                        for key in required_properties.keys()
                    )
                ]
            )
            or "TRUE"
        )

    @staticmethod
    def _node_vector_index_name(
        sanitized_label: str, sanitized_embedding_property_name: str
    ) -> str:
        """
        Generate a unique name for a node vector index
        based on the label and embedding property name.

        Args:
            sanitized_label (str):
                The sanitized node label.
            embedding_property_name (str):
                The sanitized embedding property name.

        Returns:
            str: The generated vector index name.
        """
        return (
            "node_vector_index"
            "_for_"
            f"{len(sanitized_label)}_"
            f"{sanitized_label}"
            "_on_"
            f"{len(sanitized_embedding_property_name)}_"
            f"{sanitized_embedding_property_name}"
        )

    @staticmethod
    def _nodes_from_neo4j_nodes(
        neo4j_nodes: Collection[Neo4jNode],
    ) -> list[Node]:
        """
        Convert a collection of Neo4jNodes to a list of Nodes.

        Args:
            neo4j_nodes (Collection[Neo4jNode]): Collection of Neo4jNodes.

        Returns:
            list[Node]: List of Node objects.
        """
        return [
            Node(
                uuid=UUID(neo4j_node["uuid"]),
                labels=set(neo4j_node.labels),
                properties={
                    Neo4jVectorGraphStore._desanitize_name(
                        key
                    ): Neo4jVectorGraphStore._python_value_from_neo4j_value(value)
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
