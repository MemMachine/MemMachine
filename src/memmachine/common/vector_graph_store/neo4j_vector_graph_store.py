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
import unicodedata
from uuid import UUID
from collections import defaultdict

from neo4j import AsyncDriver
from neo4j.graph import Node as Neo4jNode, Relationship as Neo4jRelationship
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

    @async_locked
    async def create_fulltext_index(self):
        relation = Neo4jVectorGraphStore._sanitize_name("HAS_RELATION")
        field_name = Neo4jVectorGraphStore._sanitize_name("triple_text")
        await self._driver.execute_query(
            "CREATE FULLTEXT INDEX rel_tripletext_fts IF NOT EXISTS\n"
            f"FOR ()-[r:{relation}]-() ON EACH [r.{field_name}]\n"
            """
            OPTIONS {
            indexConfig: { `fulltext.analyzer`: 'simple' }
            };
            """,
        )

        await self._driver.execute_query(
            """
            CREATE FULLTEXT INDEX node_name_fts IF NOT EXISTS
            FOR (n:Entity) ON EACH [n.name]
            OPTIONS {
            indexConfig: { `fulltext.analyzer`: 'simple' }
            };
            """,
        )

        await self._driver.execute_query(
            # Fast lookups by uuid
            """
            CREATE CONSTRAINT entity_uuid_unique IF NOT EXISTS
            FOR (n:Entity) REQUIRE n.uuid IS UNIQUE;
            """,
        )

        await self._driver.execute_query(
            # Fast lookups by uuid
            """
            CREATE CONSTRAINT episode_uuid_unique IF NOT EXISTS
            FOR (n:Episode) REQUIRE n.uuid IS UNIQUE;
            """,
        )

        await self._driver.execute_query(
            # Fast lookups by uuid
            """
            CREATE CONSTRAINT episode_uuid_unique IF NOT EXISTS
            FOR (n:EpisodeCluster) REQUIRE n.uuid IS UNIQUE;
            """,
        )

        session_id = Neo4jVectorGraphStore._sanitize_name("session_id")
        await self._driver.execute_query(
            f"""
            CREATE INDEX episode_session_id_idx IF NOT EXISTS
            FOR (n:EpisodeCluster) ON (n.{session_id});
            """,
        )

        await self._driver.execute_query(
            f"""
            CREATE INDEX episode_session_id_idx IF NOT EXISTS
            FOR (n:Episode) ON (n.{session_id});
            """,
        )

        await self._driver.execute_query(
            f"""
            CREATE INDEX entity_session_id_idx IF NOT EXISTS
            FOR (n:Entity) ON (n.{session_id});
            """,
        )


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
        index_search_label: str = "",
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

        session_id_index = ""
        session_id = ""
        required_properties_copy = dict(required_properties)
        if "session_id" in required_properties_copy:
            session_id = required_properties_copy["session_id"]
            session_id_index = f"{{{Neo4jVectorGraphStore._sanitize_name('session_id')}: $session_id}}"
            required_properties_copy.pop("session_id")

        search_related_nodes_tasks = [
            async_with(
                self._semaphore,
                self._driver.execute_query(
                    f"MATCH (m{index_search_label} {{uuid: $node_uuid}})\n"
                    f"MATCH (n{Neo4jVectorGraphStore._format_labels(required_labels)} {session_id_index})\n"
                    f"MATCH (m)"
                    f"  {'-' if find_targets else '<-'}"
                    f"  {query_typed_relation}"
                    f"  {'-' if find_sources else '->'} "
                    f"(n{Neo4jVectorGraphStore._format_labels(required_labels)})\n"
                    f"WHERE {
                        Neo4jVectorGraphStore._format_required_properties(
                            'n',
                            required_properties_copy,
                            include_missing_properties
                        )
                    }\n"
                    "RETURN n\n"
                    f"{'LIMIT $limit' if limit is not None else ''}",
                    node_uuid=str(node_uuid),
                    limit=limit,
                    required_properties={
                        Neo4jVectorGraphStore._sanitize_name(key): value
                        for key, value in required_properties_copy.items()
                    },
                    session_id=session_id,
                    timeout=3600,
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

    async def search_related_nodes_edges_batch(
        self,
        node_uuids: list[UUID],
        allowed_relations: Collection[str] | None = None,
        find_sources: bool = True,
        find_targets: bool = True,
        limit: int | None = None,
        required_labels: Collection[str] | None = None,
        required_properties: Mapping[str, Property] = {},
        include_missing_properties: bool = False,
        index_search_label: str = "",
    ) -> tuple[list[Node], list[Edge]]:
        if not (find_sources or find_targets):
            return []

        query_typed_relations = (
            [
                f"[r:{Neo4jVectorGraphStore._sanitize_name(relation)}]"
                for relation in allowed_relations
            ]
            if allowed_relations is not None
            else ["[r]"]
        )

        session_id_index = ""
        session_id = ""
        required_properties_copy = dict(required_properties)
        if "session_id" in required_properties_copy:
            session_id = required_properties_copy["session_id"]
            session_id_index = f"{{{Neo4jVectorGraphStore._sanitize_name('session_id')}: $session_id}}"
            required_properties_copy.pop("session_id")

        search_related_nodes_tasks = [
            async_with(
                self._semaphore,
                self._driver.execute_query(
                    "UNWIND $uuids AS id\n"
                    f"MATCH (m{index_search_label} {{uuid: id}})\n"
                    f"MATCH (n{Neo4jVectorGraphStore._format_labels(required_labels)} {session_id_index})\n"
                    f"MATCH (m)"
                    f"  {'-' if find_targets else '<-'}"
                    f"  {query_typed_relation}"
                    f"  {'-' if find_sources else '->'} "
                    f"(n)\n"
                    f"WHERE {
                        Neo4jVectorGraphStore._format_required_properties(
                            'n',
                            required_properties_copy,
                            include_missing_properties
                        )
                    }\n"
                    "RETURN n, r, startNode(r) AS src, endNode(r) AS dst\n"
                    f"{'LIMIT $limit' if limit is not None else ''}",
                    uuids=[str(node_uuid) for node_uuid in node_uuids],
                    limit=limit,
                    required_properties={
                        Neo4jVectorGraphStore._sanitize_name(key): value
                        for key, value in required_properties_copy.items()
                    },
                    session_id=session_id,
                    timeout=3600,
                ),
            )
            for query_typed_relation in query_typed_relations
        ]

        results = await asyncio.gather(*search_related_nodes_tasks)

        related_nodes: set[Node] = set()
        related_edges: set[Edge] = set()
        for records, _, _ in results:
            related_neo4j_nodes = [record["n"] for record in records]
            related_nodes.update(
                Neo4jVectorGraphStore._nodes_from_neo4j_nodes(related_neo4j_nodes)
            )
            related_neo4j_relations = [record["r"] for record in records]
            related_edges.update(
                Neo4jVectorGraphStore._edges_from_neo4j_relationships(related_neo4j_relations)
            )

        return list(related_nodes), list(related_edges)

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

    def lucene_sanitize(self, query: str) -> str:
        # Escape special characters from a query before passing into Lucene
        # + - && || ! ( ) { } [ ] ^ " ~ * ? : \ /
        escape_map = str.maketrans(
            {
                '+': r'\+',
                '-': r'\-',
                '&': r'\&',
                '|': r'\|',
                '!': r'\!',
                '(': r'\(',
                ')': r'\)',
                '{': r'\{',
                '}': r'\}',
                '[': r'\[',
                ']': r'\]',
                '^': r'\^',
                '"': r'\"',
                '~': r'\~',
                '*': r'\*',
                '?': r'\?',
                ':': r'\:',
                '\\': r'\\',
                '/': r'\/',
                'O': r'\O',
                'R': r'\R',
                'N': r'\N',
                'T': r'\T',
                'A': r'\A',
                'D': r'\D',
            }
        )
        return query.translate(escape_map)

    def normalize_name(self, s: str) -> str:
        s = unicodedata.normalize("NFKC", s)
        s = s.lower()
        s = re.sub(r"\s+", " ", s).strip()
        return s

    def rrf(
        self,
        lists: list[list[Any]],
        weights: list[float] | None = None,
        k: int = 50,
    ) -> list[tuple[Any, float]]:
        """
        Reciprocal Rank Fusion (RRF) implementation.
        Args:
            lists (list[list[str]]): List of ranked lists to fuse.
            weights (list[float] | None): Weights for each ranked list. If None, equal weights are used.
            k (int): Constant to control the influence of rank. Smaller k gives more weight to higher ranks.
        Returns:
            list[tuple[str, float]]: List of tuples containing item and its fused score, sorted by score in descending order.
        """
        if not lists:
            return []

        n = len(lists)
        if weights is None:
            weights = [1.0] * n
        elif len(weights) != n:
            raise ValueError(f"RRF: weights length {len(weights)} must equal number of lists {n}")

        scores: dict[str, float] = defaultdict(float)

        for w, L in zip(weights, lists):
            for rank, _id in enumerate(L, start=1):
                if w > 0:
                    scores[_id] += w * (1.0 / (k + rank))

        return sorted(scores.items(), key=lambda x: x[1], reverse=True)

    async def search_similar_edges(
        self,
        query_text: str,
        query_embedding: list[float],
        embedding_property_name: str,
        similarity_threshold: float = 0.2,
        limit: int | None = 100,
        allowed_relations: set[str] | None = None,
        required_properties: dict[str, Property] = {},
        include_missing_properties: bool = False,
    ) -> list[Edge]:
        query_typed_relations = (
            [
                f":{Neo4jVectorGraphStore._sanitize_name(relation)}"
                for relation in allowed_relations
            ]
            if allowed_relations is not None
            else [""]
        )
        
        sanitized_embedding_property_name = Neo4jVectorGraphStore._sanitize_name(
            embedding_property_name
        )

        vector_index_name = (
            Neo4jVectorGraphStore._edge_vector_index_name(
                Neo4jVectorGraphStore._sanitize_name(next(iter(allowed_relations))),
                sanitized_embedding_property_name,
            )
            if allowed_relations is not None and len(allowed_relations) > 0
            else None
        )

        await self._create_edge_vector_index_if_not_exist(
            relations=cast(Collection[str], allowed_relations),
            embedding_property_name=embedding_property_name,
            dimensions=len(query_embedding),
            similarity_metric=SimilarityMetric.COSINE,
        )

        vector_search_tasks = [
            async_with(
                self._semaphore,
                self._driver.execute_query(
                    f"""
                    CALL {{
                        CALL db.index.vector.queryRelationships($vector_index_name, $prelimit, $query_embedding)
                        YIELD relationship AS r, score
                        WHERE score > $similarity_threshold
                            AND {
                                Neo4jVectorGraphStore._format_required_properties(
                                    "r", required_properties, include_missing_properties
                                )
                            }
                        ORDER BY score DESC
                        LIMIT $limit
                        RETURN r, score
                    }}
                    RETURN r, startNode(r) AS src, endNode(r) AS dst, score AS similarity
                    ORDER BY similarity DESC
                    """,
                    vector_index_name=vector_index_name,
                    query_embedding=query_embedding,
                    similarity_threshold=similarity_threshold,
                    limit=limit,
                    prelimit=(limit * 5) if limit is not None else 100,
                    required_properties={
                        Neo4jVectorGraphStore._sanitize_name(key): value
                        for key, value in required_properties.items()
                    },
                    include_missing_properties=include_missing_properties,
                    timeout=3600,
                ),
            )
            for relation in query_typed_relations
        ]

        results = await asyncio.gather(*vector_search_tasks)

        vector_search_edges = []
        edge_source_node_map = {}
        for records, _, _ in results:
            rels = [record["r"] for record in records]
            vector_search_edges.extend(
                Neo4jVectorGraphStore._edges_from_neo4j_relationships(rels)
            )
            neo4j_nodes = [record["src"] for record in records]
            src_nodes = Neo4jVectorGraphStore._nodes_from_neo4j_nodes(neo4j_nodes)
            edge_source_node_map.update({
                n.uuid: n for n in src_nodes
            })

        fulltext_search_tasks = [
            async_with(
                self._semaphore,
                self._driver.execute_query(
                    f"""
                    CALL {{
                        CALL db.index.fulltext.queryRelationships('rel_tripletext_fts', $q_text, {{limit: $prelimit}})
                        YIELD relationship AS r, score
                        WHERE {
                            Neo4jVectorGraphStore._format_required_properties(
                                "r", required_properties, include_missing_properties
                            )
                        }
                        ORDER BY score DESC
                        LIMIT $limit
                        RETURN r, score
                    }}
                    RETURN r, startNode(r) AS src, endNode(r) AS dst, score AS similarity
                    ORDER BY similarity DESC
                    """,
                    q_text=self.lucene_sanitize(self.normalize_name(query_text)),           # see normalization helper above
                    limit=limit,
                    prelimit=(limit * 5) if limit is not None else 100,
                    required_properties={
                        Neo4jVectorGraphStore._sanitize_name(key): value
                        for key, value in required_properties.items()
                    },
                    include_missing_properties=include_missing_properties,
                    timeout=3600,
                ),
            )
            for relation in query_typed_relations
        ]

        results = await asyncio.gather(*fulltext_search_tasks)

        fulltext_search_edges = []
        for records, _, _ in results:
            rels = [record["r"] for record in records]
            fulltext_search_edges.extend(
                Neo4jVectorGraphStore._edges_from_neo4j_relationships(rels)
            )
            neo4j_nodes = [record["src"] for record in records]
            src_nodes = Neo4jVectorGraphStore._nodes_from_neo4j_nodes(neo4j_nodes)
            edge_source_node_map.update({
                n.uuid: n for n in src_nodes
            })

        fused = self.rrf([vector_search_edges, fulltext_search_edges], k=50)
        result_edges = [edge for edge, _ in fused][:limit]
        return result_edges, [edge_source_node_map[edge.source_uuid] for edge in result_edges]

    # async def hybrid_search_edges(
    #     self,
    #     query_text: str,
    #     query_embedding: list[float],
    #     similarity_threshold: float = 0.6,
    #     limit: int | None = 100,
    #     required_properties: dict[str, Property] = {},
    #     include_missing_properties: bool = False,
    # ) -> list[Edge]:
    #     async with self._semaphore:
    #         records, _, _ = await self._driver.execute_query(
    #             f"""
    #             CALL db.index.fulltext.queryRelationships('rel_tripletext_fts', $q_text)
    #             YIELD relationship AS r, score AS bm25
    #             WITH r, bm25,
    #                 CASE WHEN r.embedding IS NOT NULL
    #                     THEN vector.similarity.cosine(r.embedding, $query_embedding)
    #                     ELSE 0.0
    #                 END AS cos
    #             WHERE cos > $similarity_threshold
    #             MATCH p = ()-[r:RELATED_TO]-()
    #             WITH p, bm25, cos, 0.6*cos + 0.4*bm25 AS hybrid
    #             RETURN p
    #             ORDER BY hybrid DESC
    #             {'LIMIT $limit' if limit is not None else ''}
    #             """,
    #             q_text=self.lucene_sanitize(self.normalize_name(query_text)),           # see normalization helper above
    #             query_embedding=query_embedding,
    #             similarity_threshold=similarity_threshold,   # e.g., 0.25
    #             limit=limit,
    #         )

    #     neo4j_paths = [record["p"] for record in records]
    #     return Neo4jVectorGraphStore._edges_from_neo4j_relationships([p.relationships[0] for p in neo4j_paths])

    async def hybrid_search_nodes(
        self,
        node_name: str,
        rrf_weights: list[float] | None,
        limit: int = 3,
        required_label: str = "Entity",
        required_properties: dict[str, Property] = {},
        include_missing_properties: bool = False,
    ) -> list[Node]:
        fulltext_nodes = []
        substr_nodes = []

        async with self._semaphore:
            # Full text search
            records, _, _ = await self._driver.execute_query(
                f"""
                CALL {{
                    CALL db.index.fulltext.queryNodes('node_name_fts', $name, {{limit: $prelimit}})
                    YIELD node AS n, score
                    WHERE {
                        Neo4jVectorGraphStore._format_required_properties(
                            "n", required_properties, include_missing_properties
                        )
                    }\n
                    RETURN n, score
                    ORDER BY score DESC
                    LIMIT $limit
                }}
                RETURN n
                ORDER BY score DESC
                """,
                name=self.lucene_sanitize(self.normalize_name(node_name)),
                limit=limit,
                prelimit=(limit * 5) if limit is not None else 100,
                labels=[required_label],
                required_properties={
                    Neo4jVectorGraphStore._sanitize_name(key): value
                    for key, value in required_properties.items()
                },
                include_missing_properties=include_missing_properties,
                timeout=3600,
            )

        neo4j_nodes = [record["n"] for record in records]
        fulltext_nodes = Neo4jVectorGraphStore._nodes_from_neo4j_nodes(neo4j_nodes)

        # TODO: This is slow, consider using vector search here instead
        substr_nodes = []
        # if rrf_weights[1] != 0:
        #     async with self._semaphore:
        #         # Substring search
        #         records, _, _ = await self._driver.execute_query(
        #             f"""
        #             MATCH (n:{required_label})
        #             WHERE n.name IS NOT NULL
        #             AND toLower(n.name) CONTAINS toLower($name)
        #             AND {
        #                 Neo4jVectorGraphStore._format_required_properties(
        #                     "n", required_properties, include_missing_properties
        #                 )
        #             }\n
        #             RETURN n
        #             {'LIMIT $limit' if limit is not None else ''}
        #             """,
        #             name=node_name,
        #             limit=limit,
        #             required_properties={
        #                 Neo4jVectorGraphStore._sanitize_name(key): value
        #                 for key, value in required_properties.items()
        #             },
        #             include_missing_properties=include_missing_properties,
        #             timeout=3600,
        #         )
        #     neo4j_nodes = [record["n"] for record in records]
        #     substr_nodes = Neo4jVectorGraphStore._nodes_from_neo4j_nodes(neo4j_nodes)

        node_map = {}
        for node in fulltext_nodes + substr_nodes:
            if node.uuid not in node_map:
                node_map[node.uuid] = node
        ft_nodes_uuids = [n.uuid for n in fulltext_nodes]
        ss_nodes_uuids = [n.uuid for n in substr_nodes]

        fused = self.rrf([ft_nodes_uuids, ss_nodes_uuids], weights=rrf_weights, k=50)
        return [node_map[uuid] for uuid, _ in fused][:limit]

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
    
    async def _create_edge_vector_index_if_not_exist(
        self,
        relations: Collection[str],
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

        sanitized_relations = [
            Neo4jVectorGraphStore._sanitize_name(relation) for relation in relations
        ]

        sanitized_embedding_property_name = Neo4jVectorGraphStore._sanitize_name(
            embedding_property_name
        )

        requested_vector_index_names = [
            Neo4jVectorGraphStore._edge_vector_index_name(
                sanitized_relation, sanitized_embedding_property_name
            )
            for sanitized_relation in sanitized_relations
        ]

        info_for_vector_indexes_to_create = [
            (sanitized_relation, sanitized_embedding_property_name, vector_index_name)
            for sanitized_relation, vector_index_name in zip(
                sanitized_relations,
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
                    f"FOR ()-[r:{sanitized_relation}]-()\n"
                    f"ON r.{sanitized_embedding_property_name}\n"
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
            for sanitized_relation, sanitized_embedding_property_name, vector_index_name in info_for_vector_indexes_to_create
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
    def _edge_vector_index_name(
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
            "edge_vector_index"
            "_for_"
            f"{len(sanitized_label)}_"
            f"{sanitized_label}"
            "_on_"
            f"{len(sanitized_embedding_property_name)}_"
            f"{sanitized_embedding_property_name}"
        )
    
    @staticmethod
    def _edges_from_neo4j_relationships(
        neo4j_relationships: list[Neo4jRelationship]
    ) -> list[Edge]:
        """
        Convert a list of Neo4j Relationships to a list of Edges.

        Args:
            neo4j_relationships (list[Neo4jRelationship]):
                List of Neo4j Relationships.

        Returns:
            list[Edge]: List of Edges.
        """
        return [
            Edge(
                uuid=UUID(neo4j_relationship["uuid"]),
                source_uuid=UUID(neo4j_relationship.start_node["uuid"]),
                target_uuid=UUID(neo4j_relationship.end_node["uuid"]),
                relation=neo4j_relationship.type,
                properties={
                    Neo4jVectorGraphStore._desanitize_name(
                        key
                    ): Neo4jVectorGraphStore._python_value_from_neo4j_value(
                        value
                    )
                    for key, value in neo4j_relationship.items()
                    if key != "uuid"
                },
            )
            for neo4j_relationship in neo4j_relationships
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
