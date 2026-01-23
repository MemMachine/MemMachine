"""
Neo4j-based vector graph store implementation.

This module provides an asynchronous implementation
of a vector graph store using Neo4j as the backend database.
"""

import asyncio
import datetime
import logging
import re
import time
from collections.abc import Awaitable, Iterable, Mapping
from enum import Enum
from typing import Any, LiteralString, cast
from uuid import uuid4

from neo4j import AsyncDriver, Query
from neo4j.graph import Node as Neo4jNode
from neo4j.time import DateTime as Neo4jDateTime
from pydantic import BaseModel, Field, InstanceOf

from memmachine.common.data_types import FilterablePropertyValue, SimilarityMetric
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
    Or as FilterOr,
)
from memmachine.common.metrics_factory import MetricsFactory
from memmachine.common.utils import async_locked

from .data_types import (
    Edge,
    EntityType,
    Node,
    OrderedPropertyValue,
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


def _neo4j_query(text: str) -> Query:
    return Query(cast(LiteralString, text))


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

        self._index_state_cache: dict[str, Neo4jVectorGraphStore.CacheIndexState] = {}
        self._populate_index_state_cache_lock = asyncio.Lock()

        # These are only used for tracking counts approximately.
        self._collection_node_counts: dict[str, int] = {}
        self._relation_edge_counts: dict[str, int] = {}

        self._background_tasks: set[asyncio.Task] = set()

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

    def _track_task(self, task: asyncio.Task) -> None:
        """Keep background tasks from being garbage collected prematurely."""
        self._background_tasks.add(task)
        task.add_done_callback(self._background_tasks.discard)

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

        await self._driver.execute_query(
            _neo4j_query(
                "UNWIND $nodes AS node\n"
                f"CREATE (n:{sanitized_collection} {{uid: node.uid}})\n"
                "SET n += node.properties"
            ),
            nodes=query_nodes,
        )

        self._collection_node_counts[collection] += len(query_nodes)

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

        end_time = time.monotonic()
        self._collect_metrics(
            self._add_nodes_calls_counter,
            self._add_nodes_latency_summary,
            start_time,
            end_time,
        )

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
                "CREATE (source)"
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
        """Search nodes by vector similarity with optional property filters."""
        start_time = time.monotonic()

        sanitized_collection = Neo4jVectorGraphStore._sanitize_name(collection)
        sanitized_embedding_name = Neo4jVectorGraphStore._sanitize_name(
            mangle_embedding_name(embedding_name),
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
        records: list[Any] = []

        if not do_exact_similarity_search:
            await self._populate_index_state_cache()

            if (
                self._index_state_cache.get(vector_index_name)
                != Neo4jVectorGraphStore.CacheIndexState.ONLINE
            ):
                do_exact_similarity_search = True

        if not do_exact_similarity_search:
            # ANN search requires a finite limit.
            if limit is None:
                limit = 1000

            query = (
                "CALL db.index.vector.queryNodes(\n"
                "    $vector_index_name, $query_limit, $query_embedding\n"
                ")\n"
                "YIELD node AS n, score AS similarity\n"
                f"WHERE {query_filter_string}\n"
                "RETURN n\n"
                "ORDER BY similarity DESC\n"
                "LIMIT $limit"
            )

            records, _, _ = await self._driver.execute_query(
                _neo4j_query(query),
                query_embedding=query_embedding,
                query_limit=(
                    limit
                    if property_filter is None
                    else limit * self._filtered_similarity_search_fudge_factor
                ),
                limit=limit,
                query_filter_params=query_filter_params,
                vector_index_name=vector_index_name,
            )

            if (
                property_filter is not None
                and len(records)
                < limit * self._exact_similarity_search_fallback_threshold
            ):
                do_exact_similarity_search = True

        if do_exact_similarity_search:
            match similarity_metric:
                case SimilarityMetric.COSINE:
                    vector_similarity_function = "vector.similarity.cosine"
                case SimilarityMetric.EUCLIDEAN:
                    vector_similarity_function = "vector.similarity.euclidean"
                case _:
                    vector_similarity_function = "vector.similarity.cosine"

            query = (
                f"MATCH (n:{sanitized_collection})\n"
                f"WHERE n.{sanitized_embedding_name} IS NOT NULL\n"
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
                query_embedding=query_embedding,
                limit=limit,
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
        starting_at: Iterable[OrderedPropertyValue | None],
        order_ascending: Iterable[bool],
        include_equal_start: bool = False,
        limit: int | None = 1,
        property_filter: FilterExpr | None = None,
    ) -> list[Node]:
        """Find nodes ordered by property values in a chosen direction."""
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
                    + " AND ".join(
                        Neo4jVectorGraphStore._render_comparison(
                            f"n.{sanitized_by_property}",
                            "=",
                            f"$starting_at[{index}]",
                            starting_at[index],
                        )
                        for index, sanitized_by_property in enumerate(
                            sanitized_by_properties,
                        )
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
                f"MATCH (n:{sanitized_collection})\n"
                f"WHERE ({query_relational_requirements})\n"
                f"AND {query_filter_string}\n"
                "RETURN n\n"
                f"{query_order_by}"
                f"{'LIMIT $limit' if limit is not None else ''}"
            ),
            starting_at=[
                Neo4jVectorGraphStore._sanitize_python_value(starting_at_value)
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
        starting_at: Iterable[OrderedPropertyValue | None],
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

            if starting_at[index] is None:
                relational_requirements = [
                    f"{entity_query_alias}.{sanitized_by_property} IS NOT NULL",
                ]
            else:
                relational_requirements = [
                    Neo4jVectorGraphStore._render_comparison(
                        f"{entity_query_alias}.{sanitized_by_property}",
                        ">" if order_ascending[index] else "<",
                        f"${starting_at_query_parameter}[{index}]",
                        starting_at[index],
                    )
                ]

            for equal_index, sanitized_equal_property in enumerate(
                sanitized_equal_properties,
            ):
                if starting_at[equal_index] is None:
                    relational_requirements += [
                        f"{entity_query_alias}.{sanitized_equal_property} IS NOT NULL"
                    ]
                else:
                    relational_requirements += [
                        Neo4jVectorGraphStore._render_comparison(
                            f"{entity_query_alias}.{sanitized_equal_property}",
                            "=",
                            f"${starting_at_query_parameter}[{equal_index}]",
                            starting_at[equal_index],
                        )
                    ]

            lexicographic_relational_requirement = (
                f"({' AND '.join(relational_requirements)})"
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
    ) -> list[Node]:
        """Search nodes that match the provided property filters."""
        start_time = time.monotonic()

        sanitized_collection = Neo4jVectorGraphStore._sanitize_name(collection)

        query_filter_string, query_filter_params = (
            Neo4jVectorGraphStore._build_query_filter(
                "n",
                "query_filter_params",
                property_filter,
            )
        )

        records, _, _ = await self._driver.execute_query(
            _neo4j_query(
                f"MATCH (n:{sanitized_collection})\n"
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
    def _sanitize_properties(
        properties: Mapping[str, PropertyValue] | None,
    ) -> dict[str, PropertyValue]:
        """Sanitize property names in a mapping for Neo4j storage."""
        return (
            {
                Neo4jVectorGraphStore._sanitize_name(
                    key
                ): Neo4jVectorGraphStore._sanitize_python_value(value)
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
                    property_value = (
                        Neo4jVectorGraphStore._python_value_from_neo4j_value(
                            neo4j_property_value,
                        )
                    )
                    node_properties[property_name] = property_value
                elif is_mangled_embedding_name(desanitized_property_name):
                    embedding_name = demangle_embedding_name(desanitized_property_name)
                    embedding_value = cast(
                        list[float],
                        Neo4jVectorGraphStore._python_value_from_neo4j_value(
                            neo4j_property_value,
                        ),
                    )
                    similarity_metric = SimilarityMetric(
                        Neo4jVectorGraphStore._python_value_from_neo4j_value(
                            neo4j_node[
                                Neo4jVectorGraphStore._sanitize_name(
                                    Neo4jVectorGraphStore._similarity_metric_property_name(
                                        embedding_name,
                                    ),
                                )
                            ],
                        ),
                    )
                    node_embeddings[embedding_name] = (
                        embedding_value,
                        similarity_metric,
                    )

            nodes.append(
                Node(
                    uid=neo4j_node["uid"],
                    properties=node_properties,
                    embeddings=node_embeddings,
                ),
            )

        return nodes

    @staticmethod
    def _sanitize_python_value(
        value: PropertyValue,
    ) -> PropertyValue:
        """
        Convert a native Python value to a sanitized value.

        Args:
            value (PropertyValue): The Python value to convert.

        Returns:
            PropertyValue: The converted Neo4j value.

        """
        if isinstance(value, datetime.datetime):
            # Other tzinfo types can lead to issues,
            # so we convert to datetime.timezone.
            utc_offset = value.utcoffset()
            tz = datetime.timezone(utc_offset) if utc_offset is not None else None
            return value.astimezone(tz=tz)
        if isinstance(value, list):
            return cast(
                PropertyValue,
                [Neo4jVectorGraphStore._sanitize_python_value(item) for item in value],
            )
        return value

    @staticmethod
    def _python_value_from_neo4j_value(
        value: PropertyValue | Neo4jDateTime,
    ) -> PropertyValue:
        """
        Convert a Neo4j value to a native Python value.

        Args:
            value (PropertyValue | Neo4jDateTime): The Neo4j value to convert.

        Returns:
            PropertyValue: The converted Python value.

        """
        if isinstance(value, Neo4jDateTime):
            return value.to_native()
        return value

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
    ) -> tuple[str, dict[str, FilterablePropertyValue | list[FilterablePropertyValue]]]:
        if property_filter is None:
            query_filter_string = "TRUE"
            query_filter_params: dict[
                str, FilterablePropertyValue | list[FilterablePropertyValue]
            ] = {}
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
    ) -> tuple[str, dict[str, FilterablePropertyValue | list[FilterablePropertyValue]]]:
        if isinstance(expr, FilterComparison):
            field_ref = f"{entity_query_alias}.{
                Neo4jVectorGraphStore._sanitize_name(mangle_property_name(expr.field))
            }"
            param_name = Neo4jVectorGraphStore._sanitize_name(
                f"filter_expr_param_{uuid4()}"
            )

            params: dict[
                str, FilterablePropertyValue | list[FilterablePropertyValue]
            ] = {}
            if expr.op in (">", "<", ">=", "<=", "=", "!=", "<>"):
                condition = Neo4jVectorGraphStore._render_comparison(
                    left=field_ref,
                    op=expr.op,
                    right=f"${query_value_parameter}.{param_name}",
                    value=expr.value,
                )
                params[param_name] = expr.value
            elif expr.op == "in":
                if not isinstance(expr.value, list):
                    raise ValueError("IN comparison requires a list of values")
                condition = f"{field_ref} IN ${query_value_parameter}.{param_name}"
                params[param_name] = [
                    cast(
                        FilterablePropertyValue,
                        Neo4jVectorGraphStore._sanitize_python_value(item),
                    )
                    for item in expr.value
                ]
            elif expr.op == "is_null":
                condition = f"{field_ref} IS NULL"
            elif expr.op == "is_not_null":
                condition = f"{field_ref} IS NOT NULL"
            else:
                raise ValueError(f"Unsupported operator: {expr.op}")
            return condition, params
        if isinstance(expr, FilterAnd):
            left_cond, left_params = Neo4jVectorGraphStore._render_filter_expr(
                entity_query_alias, query_value_parameter, expr.left
            )
            right_cond, right_params = Neo4jVectorGraphStore._render_filter_expr(
                entity_query_alias, query_value_parameter, expr.right
            )
            condition = f"({left_cond}) AND ({right_cond})"
            return condition, left_params | right_params
        if isinstance(expr, FilterOr):
            left_cond, left_params = Neo4jVectorGraphStore._render_filter_expr(
                entity_query_alias, query_value_parameter, expr.left
            )
            right_cond, right_params = Neo4jVectorGraphStore._render_filter_expr(
                entity_query_alias, query_value_parameter, expr.right
            )
            condition = f"({left_cond}) OR ({right_cond})"
            return condition, left_params | right_params
        raise TypeError(f"Unsupported filter expression type: {type(expr)!r}")

    @staticmethod
    def _render_comparison(
        left: str,
        op: str,
        right: str,
        value: FilterablePropertyValue | list[FilterablePropertyValue],
    ) -> str:
        if op == "!=":
            op = "<>"
        if isinstance(value, list):
            raise TypeError(f"'{op}' comparison cannot accept list values")
        if isinstance(value, datetime.datetime):
            if op == "=":
                return (
                    "("
                    f"{left} = {right}"
                    " OR "
                    "("
                    f"{left}.epochSeconds = {right}.epochSeconds"
                    " AND "
                    f"{left}.nanosecond = {right}.nanosecond"
                    ")"
                    ")"
                )

            if op == "<>":
                return (
                    "("
                    f"{left} <> {right}"
                    " AND "
                    "("
                    f"{left}.epochSeconds <> {right}.epochSeconds"
                    " OR "
                    f"{left}.nanosecond <> {right}.nanosecond"
                    ")"
                    ")"
                )

            return (
                "("
                f"{left} {op} {right}"
                " AND "
                "("
                f"{left}.epochSeconds {op} {right}.epochSeconds"
                " OR "
                "("
                f"{left}.epochSeconds = {right}.epochSeconds"
                " AND "
                f"{left}.nanosecond {op} {right}.nanosecond"
                ")"
                ")"
                ")"
            )

        return f"{left} {op} {right}"
