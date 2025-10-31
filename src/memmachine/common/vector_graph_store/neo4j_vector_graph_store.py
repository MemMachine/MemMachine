"""
Neo4j-based vector graph store implementation.

This module provides an asynchronous implementation
of a vector graph store using Neo4j as the backend database.
"""

import asyncio
import logging
import re
from collections.abc import Awaitable, Iterable, Mapping
from typing import Any
from uuid import UUID

from neo4j import AsyncDriver
from neo4j.graph import Node as Neo4jNode
from neo4j.time import DateTime as Neo4jDateTime
from pydantic import BaseModel, Field, InstanceOf

from memmachine.common.embedder import SimilarityMetric
from memmachine.common.utils import async_locked, async_with

from .data_types import (
    Edge,
    EntityType,
    Node,
    Property,
    demangle_property_name,
    is_mangled_property_name,
    mangle_property_name,
)
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
        exact_similarity_search (bool):
            Whether to use exact similarity search.
            (default: False).
        range_index_hierarchies (list[list[str]]):
            List of property name hierarchies (lists)
            for which to create range indexes
            applied to all nodes and edges
            (default: []).
    """

    driver: InstanceOf[AsyncDriver] = Field(
        ..., description="Async Neo4j driver instance"
    )
    max_concurrent_transactions: int = Field(
        100, description="Maximum number of concurrent transactions", gt=0
    )
    exact_similarity_search: bool = Field(
        False, description="Whether to use exact similarity search"
    )
    range_index_hierarchies: list[list[str]] = Field(
        default_factory=list,
        description=(
            "List of property name hierarchies "
            "for which to create range indexes "
            "applied to all nodes and edges"
        ),
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
        self._exact_similarity_search = params.exact_similarity_search
        self._range_index_hierarchies = params.range_index_hierarchies

        self._index_name_cache: set[str] = set()

    async def add_nodes(self, collection: str, nodes: Iterable[Node]):
        # await self._create_initial_indexes_if_not_exist(
        #     EntityType.NODE,
        #     [collection],
        # )

        sanitized_collection = Neo4jVectorGraphStore._sanitize_name(collection)

        async with self._semaphore:
            await self._driver.execute_query(
                "UNWIND $nodes AS node\n"
                f"CREATE (n:{sanitized_collection} {{uuid: node.uuid}})\n"
                "SET n += node.properties",
                nodes=[
                    {
                        "uuid": str(node.uuid),
                        "properties": Neo4jVectorGraphStore._sanitize_properties(
                            {
                                mangle_property_name(key): value
                                for key, value in node.properties.items()
                            }
                        ),
                    }
                    for node in nodes
                ],
            )

    async def add_edges(
        self,
        relation: str,
        source_collection: str,
        target_collection: str,
        edges: Iterable[Edge],
    ):
        # await self._create_initial_indexes_if_not_exist(
        #     EntityType.EDGE,
        #     [relation],
        # )

        sanitized_relation = Neo4jVectorGraphStore._sanitize_name(relation)
        sanitized_source_collection = Neo4jVectorGraphStore._sanitize_name(
            source_collection
        )
        sanitized_target_collection = Neo4jVectorGraphStore._sanitize_name(
            target_collection
        )
        async with self._semaphore:
            await self._driver.execute_query(
                "UNWIND $edges AS edge\n"
                "MATCH"
                f"    (source:{sanitized_source_collection} {{uuid: edge.source_uuid}}),"
                f"    (target:{sanitized_target_collection} {{uuid: edge.target_uuid}})\n"
                "CREATE (source)"
                f"    -[r:{sanitized_relation} {{uuid: edge.uuid}}]->"
                "    (target)\n"
                "SET r += edge.properties",
                edges=[
                    {
                        "uuid": str(edge.uuid),
                        "source_uuid": str(edge.source_uuid),
                        "target_uuid": str(edge.target_uuid),
                        "properties": Neo4jVectorGraphStore._sanitize_properties(
                            {
                                mangle_property_name(key): value
                                for key, value in edge.properties.items()
                            }
                        ),
                    }
                    for edge in edges
                ],
            )

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
        sanitized_collection = Neo4jVectorGraphStore._sanitize_name(collection)
        sanitized_embedding_property_name = Neo4jVectorGraphStore._sanitize_name(
            mangle_property_name(embedding_property_name)
        )

        if required_properties is None:
            required_properties = {}

        if self._exact_similarity_search:
            match similarity_metric:
                case SimilarityMetric.COSINE:
                    vector_similarity_function = "vector.similarity.cosine"
                case SimilarityMetric.EUCLIDEAN:
                    vector_similarity_function = "vector.similarity.euclidean"
                case _:
                    vector_similarity_function = "vector.similarity.cosine"

            query = (
                f"MATCH (n:{sanitized_collection})\n"
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
                    required_properties=Neo4jVectorGraphStore._sanitize_properties(
                        {
                            mangle_property_name(key): value
                            for key, value in required_properties.items()
                        }
                    ),
                )

        else:
            vector_index_name = Neo4jVectorGraphStore._index_name(
                EntityType.NODE,
                sanitized_collection,
                sanitized_embedding_property_name,
            )

            await self._create_vector_index_if_not_exists(
                entity_type=EntityType.NODE,
                sanitized_collection_or_relation=Neo4jVectorGraphStore._sanitize_name(
                    collection
                ),
                sanitized_embedding_property_name=Neo4jVectorGraphStore._sanitize_name(
                    mangle_property_name(embedding_property_name)
                ),
                dimensions=len(query_embedding),
                similarity_metric=similarity_metric,
            )

            # ANN search requires a finite limit.
            if limit is None:
                limit = 100_000

            query = (
                "CALL db.index.vector.queryNodes(\n"
                f"    $vector_index_name, $limit, $query_embedding\n"
                ")\n"
                "YIELD node AS n, score AS similarity\n"
                f"WHERE {
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
                    required_properties=Neo4jVectorGraphStore._sanitize_properties(
                        {
                            mangle_property_name(key): value
                            for key, value in required_properties.items()
                        }
                    ),
                    vector_index_name=vector_index_name,
                )

        similar_neo4j_nodes = [record["n"] for record in records]
        return Neo4jVectorGraphStore._nodes_from_neo4j_nodes(similar_neo4j_nodes)

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
        if not (find_sources or find_targets):
            return []

        if required_properties is None:
            required_properties = {}

        sanitized_collection = Neo4jVectorGraphStore._sanitize_name(collection)
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
                    f"    (n:{sanitized_collection})"
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
                    required_properties=Neo4jVectorGraphStore._sanitize_properties(
                        {
                            mangle_property_name(key): value
                            for key, value in required_properties.items()
                        }
                    ),
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
        collection: str,
        by_property: str,
        start_at_value: Any | None = None,
        include_equal_start_at_value: bool = False,
        order_ascending: bool = True,
        limit: int | None = 1,
        required_properties: Mapping[str, Property] | None = None,
        include_missing_properties: bool = False,
    ) -> list[Node]:
        sanitized_collection = Neo4jVectorGraphStore._sanitize_name(collection)
        sanitized_by_property = Neo4jVectorGraphStore._sanitize_name(
            mangle_property_name(by_property)
        )

        if required_properties is None:
            required_properties = {}

        async with self._semaphore:
            records, _, _ = await self._driver.execute_query(
                f"MATCH (n:{sanitized_collection})\n"
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
                required_properties=Neo4jVectorGraphStore._sanitize_properties(
                    {
                        mangle_property_name(key): value
                        for key, value in required_properties.items()
                    }
                ),
            )

        directional_proximal_neo4j_nodes = [record["n"] for record in records]
        return Neo4jVectorGraphStore._nodes_from_neo4j_nodes(
            directional_proximal_neo4j_nodes
        )

    async def search_matching_nodes(
        self,
        collection: str,
        limit: int | None = None,
        required_properties: Mapping[str, Property] | None = None,
        include_missing_properties: bool = False,
    ) -> list[Node]:
        sanitized_collection = Neo4jVectorGraphStore._sanitize_name(collection)

        if required_properties is None:
            required_properties = {}

        async with self._semaphore:
            records, _, _ = await self._driver.execute_query(
                f"MATCH (n:{sanitized_collection})\n"
                f"WHERE {
                    Neo4jVectorGraphStore._format_required_properties(
                        'n', required_properties, include_missing_properties
                    )
                }\n"
                "RETURN n\n"
                f"{'LIMIT $limit' if limit is not None else ''}",
                limit=limit,
                required_properties=Neo4jVectorGraphStore._sanitize_properties(
                    {
                        mangle_property_name(key): value
                        for key, value in required_properties.items()
                    }
                ),
            )

        matching_neo4j_nodes = [record["n"] for record in records]
        return Neo4jVectorGraphStore._nodes_from_neo4j_nodes(matching_neo4j_nodes)

    async def delete_nodes(
        self,
        node_uuids: Iterable[UUID],
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

    async def _populate_index_name_cache(self):
        """
        Populate the index name cache.
        """
        if not self._index_name_cache:
            async with self._semaphore:
                records, _, _ = await self._driver.execute_query(
                    "SHOW INDEXES YIELD name RETURN name"
                )

            self._index_name_cache.update(record["name"] for record in records)

    async def _create_initial_indexes_if_not_exist(
        self,
        entity_type: EntityType,
        collections_or_relations: Iterable[str],
    ):
        """
        Create initial indexes if not exist.

        Args:
            entity_type (EntityType):
                The type of entity the indexes are for.
            collections_or_relations (Iterable[str]):
                Collections of nodes or relation types of edges
                to create initial indexes for.
        """
        tasks = [
            self._create_unique_constraint_if_not_exists(
                entity_type=entity_type,
                sanitized_collection_or_relation=Neo4jVectorGraphStore._sanitize_name(
                    collection_or_relation
                ),
                sanitized_property_name=Neo4jVectorGraphStore._sanitize_name("uuid"),
            )
            for collection_or_relation in collections_or_relations
        ]
        tasks += [
            self._create_range_index_if_not_exists(
                entity_type=entity_type,
                sanitized_collection_or_relation=Neo4jVectorGraphStore._sanitize_name(
                    collection_or_relation
                ),
                sanitized_property_names=[
                    Neo4jVectorGraphStore._sanitize_name(
                        mangle_property_name(property_name)
                    )
                    for property_name in property_name_hierarchy
                ],
            )
            for collection_or_relation in collections_or_relations
            for range_index_hierarchy in self._range_index_hierarchies
            for property_name_hierarchy in [
                range_index_hierarchy[: i + 1]
                for i in range(len(range_index_hierarchy))
            ]
        ]
        await asyncio.gather(*tasks)

    async def _create_unique_constraint_if_not_exists(
        self,
        entity_type: EntityType,
        sanitized_collection_or_relation: str,
        sanitized_property_name: str,
    ):
        """
        Create unique constraint if not exists.

        Args:
            entity_type (EntityType):
                The type of entity the constraint is for.
            sanitized_collection_or_relation (str):
                Collection of nodes or relation type of edges
                to create unique constraint for.
            sanitized_property_name (str):
                Name of the property to create unique constraint on.
        """
        await self._populate_index_name_cache()

        unique_constraint_name = Neo4jVectorGraphStore._index_name(
            entity_type,
            sanitized_collection_or_relation,
            sanitized_property_name,
        )

        if unique_constraint_name in self._index_name_cache:
            return

        match entity_type:
            case EntityType.NODE:
                query_constraint_for_expression = (
                    f"(e:{sanitized_collection_or_relation})"
                )
            case EntityType.EDGE:
                query_constraint_for_expression = (
                    f"()-[e:{sanitized_collection_or_relation}]-()"
                )

        create_constraint_task = async_with(
            self._semaphore,
            self._driver.execute_query(
                f"CREATE CONSTRAINT {unique_constraint_name}\n"
                "IF NOT EXISTS\n"
                f"FOR {query_constraint_for_expression}\n"
                f"REQUIRE e.{sanitized_property_name} IS UNIQUE",
            ),
        )

        await self._execute_create_index_if_not_exists(create_constraint_task)
        self._index_name_cache.add(unique_constraint_name)

    async def _create_range_index_if_not_exists(
        self,
        entity_type: EntityType,
        sanitized_collection_or_relation: str,
        sanitized_property_names: Iterable[str],
    ):
        """
        Create range index if not exists.

        Args:
            entity_type (EntityType):
                The type of entity the index is for.
            sanitized_collection_or_relation (str):
                Collection of nodes or relation type of edges
                to create range index for.
            sanitized_property_names (list[str]):
                List of property names representing
                the hierarchy for the range index.
        """
        await self._populate_index_name_cache()

        range_index_name = Neo4jVectorGraphStore._index_name(
            entity_type, sanitized_collection_or_relation, sanitized_property_names
        )

        if range_index_name in self._index_name_cache:
            return

        match entity_type:
            case EntityType.NODE:
                query_index_for_expression = f"(e:{sanitized_collection_or_relation})"
            case EntityType.EDGE:
                query_index_for_expression = (
                    f"()-[e:{sanitized_collection_or_relation}]-()"
                )

        create_index_task = async_with(
            self._semaphore,
            self._driver.execute_query(
                f"CREATE RANGE INDEX {range_index_name}\n"
                "IF NOT EXISTS\n"
                f"FOR {query_index_for_expression}\n"
                f"ON ({
                    ', '.join(
                        f'e.{sanitized_property_name}'
                        for sanitized_property_name in sanitized_property_names
                    )
                })",
            ),
        )

        await self._execute_create_index_if_not_exists(create_index_task)
        self._index_name_cache.add(range_index_name)

    async def _create_vector_index_if_not_exists(
        self,
        entity_type: EntityType,
        sanitized_collection_or_relation: str,
        sanitized_embedding_property_name: str,
        dimensions: int,
        similarity_metric: SimilarityMetric = SimilarityMetric.COSINE,
    ):
        """
        Create vector index if not exists.

        Args:
            entity_type (EntityType):
                The type of entity the index is for.
            collection_or_relation (str):
                Collection of nodes or relation type of edges
                to create vector index for.
            embedding_property_name (str):
                Name of the embedding property to create vector index on.
            dimensions (int):
                Dimensionality of the embedding vectors.
            similarity_metric (SimilarityMetric):
                Similarity metric to use for the vector index
                (default: SimilarityMetric.COSINE).
        """
        await self._populate_index_name_cache()

        vector_index_name = Neo4jVectorGraphStore._index_name(
            entity_type,
            sanitized_collection_or_relation,
            sanitized_embedding_property_name,
        )

        if vector_index_name in self._index_name_cache:
            return

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

        create_index_task = async_with(
            self._semaphore,
            self._driver.execute_query(
                f"CREATE VECTOR INDEX {vector_index_name}\n"
                "IF NOT EXISTS\n"
                f"FOR {query_index_for_expression}\n"
                f"ON e.{sanitized_embedding_property_name}\n"
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

        await self._execute_create_index_if_not_exists(create_index_task)
        self._index_name_cache.add(vector_index_name)

    @async_locked
    async def _execute_create_index_if_not_exists(self, create_index_task: Awaitable):
        """
        Execute the creation of node vector index if not exists.
        Locked because Neo4j concurrent (vector) index creation
        can raise exceptions even with "IF NOT EXISTS".

        Args:
            create_index_task (Awaitable):
                Awaitable task to create vector index.
        """
        await create_index_task

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
    def _sanitize_properties(
        properties: Mapping[str, Property] | None,
    ) -> dict[str, Property]:
        """
        Sanitize property names in a properties mapping for Neo4j.

        Args:
            properties (Mapping[str, Property] | None):
                Mapping of properties or None.

        Returns:
            dict[str, Property]:
                Mapping with sanitized property names.
        """
        return (
            {
                Neo4jVectorGraphStore._sanitize_name(key): value
                for key, value in properties.items()
            }
            if properties is not None
            else {}
        )

    @staticmethod
    def _format_required_properties(
        entity_query_alias: str,
        required_properties: Mapping[str, Property] | None,
        include_missing_properties: bool,
    ) -> str:
        """
        Format required properties for use in a Cypher query.

        Args:
            entity_query_alias (str):
                Alias of the node or relationship in the query
                (e.g., "n", "r").
            required_properties (Mapping[str, Property] | None):
                Mapping of required properties or None.
            include_missing_properties (bool):
                Whether to include results
                with missing required properties.

        Returns:
            str:
                Formatted required properties string for Cypher query.
        """
        if required_properties is None:
            required_properties = {}

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
                    for sanitized_property_name in Neo4jVectorGraphStore._sanitize_properties(
                        {
                            mangle_property_name(key): value
                            for key, value in required_properties.items()
                        }
                    ).keys()
                ]
            )
            or "TRUE"
        )

    @staticmethod
    def _index_name(
        entity_type: EntityType,
        sanitized_collection_or_relation: str,
        sanitized_property_names: str | Iterable[str],
    ) -> str:
        """
        Generate a unique name for an index
        based on the entity type, collection, and property name.

        Args:
            entity_type (EntityType):
                The type of entity the index is for.
            sanitized_collection_or_relation (str):
                The sanitized node collection or edge relation type.
            sanitized_property_names (list[str]):
                The sanitized property names.

        Returns:
            str: The generated vector index name.
        """
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
        return [
            Node(
                uuid=UUID(neo4j_node["uuid"]),
                properties={
                    demangle_property_name(
                        Neo4jVectorGraphStore._desanitize_name(key)
                    ): Neo4jVectorGraphStore._python_value_from_neo4j_value(value)
                    for key, value in neo4j_node.items()
                    if is_mangled_property_name(key)
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
