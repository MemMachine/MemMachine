"""
NebulaGraph Enterprise 5.0 implementation of VectorGraphStore interface.

This module provides a complete implementation of the VectorGraphStore abstract
interface using NebulaGraph Enterprise 5.0 as the backend. It supports:
- Graph data model with Schema + Graph
- Native VECTOR data type with IVF/HNSW indexes
- GQL (Graph Query Language - ISO/IEC 76120 standard)
- Async operations via NebulaAsyncClient
"""

import asyncio
import logging
from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import TYPE_CHECKING, Any

from nebulagraph_python.py_data_types import NVector

from memmachine.common.data_types import SimilarityMetric
from memmachine.common.filter.filter_parser import And, Comparison, FilterExpr, Or
from memmachine.common.vector_graph_store.data_types import (
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
from memmachine.common.vector_graph_store.vector_graph_store import VectorGraphStore

if TYPE_CHECKING:
    from nebulagraph_python.client import NebulaAsyncClient

logger = logging.getLogger(__name__)


@dataclass
class NebulaGraphVectorGraphStoreParams:
    """
    Parameters for configuring NebulaGraphVectorGraphStore.

    Args:
        client: Connected NebulaAsyncClient instance
        schema_name: NebulaGraph schema path (e.g., "/default_schema")
        graph_type_name: Graph type name that defines the schema structure
        graph_name: Graph instance name within the schema
        force_exact_similarity_search: If True, always use KNN exact search (no ANN)
        filtered_similarity_search_fudge_factor: Multiplier for ANN limit when filtering
        exact_similarity_search_fallback_threshold: Ratio threshold for fallback to exact search
        range_index_hierarchies: List of property hierarchies for range indexes (e.g., [["timestamp"], ["user_id", "timestamp"]])
        range_index_creation_threshold: Min entities before creating property index
        vector_index_creation_threshold: Min entities before creating vector index.
            Vector indexes enable ANN (Approximate Nearest Neighbor) search.
            Below this threshold, KNN (K-Nearest Neighbor) exact search is used,
            which does not require an index and is suitable for small datasets.
        ann_index_type: Vector index type - "IVF" or "HNSW"
        ivf_nlist: IVF clusters (higher = more accurate, slower build)
        ivf_nprobe: IVF search clusters (higher = better recall, slower query)
        hnsw_max_degree: HNSW max neighbors (higher = better recall, more memory)
        hnsw_ef_construction: HNSW build quality (higher = better index, slower build)
        hnsw_ef_search: HNSW search quality (higher = better recall, slower query)
        metrics_factory: Optional metrics factory for observability
        user_metrics_labels: Optional user-defined metric labels

    """

    client: "NebulaAsyncClient"
    schema_name: str
    graph_type_name: str
    graph_name: str
    force_exact_similarity_search: bool = False
    filtered_similarity_search_fudge_factor: int = 4
    exact_similarity_search_fallback_threshold: float = 0.5
    range_index_hierarchies: list[list[str]] = field(default_factory=list)
    range_index_creation_threshold: int = 10_000
    vector_index_creation_threshold: int = 10_000

    # Vector index tuning
    ann_index_type: str = "IVF"
    ivf_nlist: int = 256
    ivf_nprobe: int = 8
    hnsw_max_degree: int = 16
    hnsw_ef_construction: int = 200
    hnsw_ef_search: int = 40

    # Observability
    metrics_factory: Any | None = None
    user_metrics_labels: dict[str, str] = field(default_factory=dict)


class NebulaGraphVectorGraphStore(VectorGraphStore):
    """
    NebulaGraph Enterprise 5.0 implementation of VectorGraphStore.

    This implementation uses NebulaGraph Enterprise 5.0's native vector support
    with GQL (ISO Graph Query Language) instead of Cypher. Key differences:
    - Schema + Graph model
    - Native VECTOR<N, FLOAT> data type
    - Vector indexes with IVF/HNSW algorithms
    - SESSION SET SCHEMA and SESSION SET GRAPH for context
    """

    class CacheIndexState(Enum):
        """Index state tracking for local cache."""

        CREATING = auto()
        ONLINE = auto()

    def __init__(self, params: NebulaGraphVectorGraphStoreParams) -> None:
        """
        Initialize NebulaGraph vector graph store.

        Args:
            params: Configuration parameters for the store

        """
        # Client and configuration
        self._client = params.client
        self._schema_name = params.schema_name
        self._graph_type_name = params.graph_type_name
        self._graph_name = params.graph_name
        self._force_exact_similarity_search = params.force_exact_similarity_search
        self._fudge_factor = params.filtered_similarity_search_fudge_factor
        self._fallback_threshold = params.exact_similarity_search_fallback_threshold
        self._range_index_hierarchies = params.range_index_hierarchies
        self._range_index_threshold = params.range_index_creation_threshold
        self._vector_index_threshold = params.vector_index_creation_threshold

        # Vector index tuning
        self._ann_index_type = params.ann_index_type
        self._ivf_nlist = params.ivf_nlist
        self._ivf_nprobe = params.ivf_nprobe
        self._hnsw_max_degree = params.hnsw_max_degree
        self._hnsw_ef_construction = params.hnsw_ef_construction
        self._hnsw_ef_search = params.hnsw_ef_search

        # State tracking (CRITICAL: Must track schema/index state dynamically)
        # Maps collection/relation name -> {property_name: gql_type}
        self._graph_type_schemas: dict[str, dict[str, str]] = {}
        # Maps index name -> CacheIndexState
        self._index_state_cache: dict[
            str, NebulaGraphVectorGraphStore.CacheIndexState
        ] = {}
        # Maps collection name -> node count (for threshold triggers)
        self._collection_node_counts: dict[str, int] = {}
        # Maps relation name -> edge count (for threshold triggers)
        self._relation_edge_counts: dict[str, int] = {}
        # Track if we've discovered existing indexes from NebulaGraph
        self._indexes_discovered: bool = False

        # Concurrency control
        self._background_tasks: set[asyncio.Task] = set()
        self._schema_lock = asyncio.Lock()
        self._index_locks: dict[str, asyncio.Lock] = {}

        # Observability
        self._metrics_factory = params.metrics_factory
        self._user_metrics_labels = params.user_metrics_labels

    # =========================================================================
    # VectorGraphStore Interface Implementation
    # =========================================================================

    async def add_nodes(
        self,
        *,
        collection: str,
        nodes: Iterable[Node],
    ) -> None:
        """
        Add nodes to a collection.

        Creates the graph type if needed, inserts nodes with properties and embeddings,
        and creates indexes when thresholds are reached.

        Args:
            collection: Collection name (becomes node type in graph type)
            nodes: Iterable of Node objects to add

        """
        nodes_list = list(nodes)
        if not nodes_list:
            return

        await self._discover_existing_indexes()

        # Collect all properties and embeddings from all nodes to build schema
        all_properties: dict[str, PropertyValue] = {}
        all_embeddings: dict[str, tuple[list[float], SimilarityMetric]] = {}

        for node in nodes_list:
            all_properties.update(node.properties)
            # Track embeddings with their dimensions and metrics
            for emb_name, (emb_vec, metric) in node.embeddings.items():
                if emb_name not in all_embeddings:
                    all_embeddings[emb_name] = (emb_vec, metric)

        # Ensure graph type exists with required schema
        await self._ensure_graph_type_for_nodes(
            collection, all_properties, all_embeddings
        )

        sanitized_collection = self._sanitize_name(collection)

        # Insert nodes
        for node in nodes_list:
            # Build INSERT statement with embedded values (no parameterization)
            prop_assignments = []

            # Add uid
            uid_formatted = self._format_value(node.uid)
            prop_assignments.append(f"uid: {uid_formatted}")

            # Add properties
            for prop_name, prop_value in node.properties.items():
                mangled = mangle_property_name(prop_name)
                formatted_value = self._format_value(prop_value)
                prop_assignments.append(f"{mangled}: {formatted_value}")

            # Add embeddings (metric is tracked in all_embeddings for index creation)
            for emb_name, (emb_vec, _) in node.embeddings.items():
                mangled = mangle_embedding_name(emb_name)
                vec_literal = self._vector_to_gql_literal(emb_vec)
                prop_assignments.append(f"{mangled}: {vec_literal}")

            insert_stmt = f"""
            INSERT (n@{sanitized_collection} {{
                {", ".join(prop_assignments)}
            }})
            """
            await self._client.execute(insert_stmt)

        # Update node count
        current_count = self._collection_node_counts.get(collection, 0)
        new_count = current_count + len(nodes_list)
        self._collection_node_counts[collection] = new_count

        # Check if we should create indexes
        if (
            self._vector_index_threshold
            and new_count >= self._vector_index_threshold
            and current_count < self._vector_index_threshold
        ):
            # Create vector indexes
            for emb_name, (emb_vec, metric) in all_embeddings.items():
                task = asyncio.create_task(
                    self._create_vector_index_if_not_exists(
                        EntityType.NODE,
                        collection,
                        emb_name,
                        len(emb_vec),
                        metric,
                    )
                )
                self._background_tasks.add(task)
                task.add_done_callback(self._background_tasks.discard)

        # Create range indexes based on configured hierarchies
        # Note: We don't create a range index on 'uid' because it's declared as PRIMARY KEY
        # in the node type schema, and NebulaGraph automatically indexes primary keys.
        if (
            self._range_index_threshold
            and new_count >= self._range_index_threshold
            and current_count < self._range_index_threshold
        ):
            task = asyncio.create_task(
                self._create_initial_indexes_if_not_exist(
                    EntityType.NODE,
                    collection,
                )
            )
            self._background_tasks.add(task)
            task.add_done_callback(self._background_tasks.discard)

    async def add_edges(
        self,
        *,
        relation: str,
        source_collection: str,
        target_collection: str,
        edges: Iterable[Edge],
    ) -> None:
        """
        Add edges between nodes in collections.

        Creates edge type in graph type if needed, then inserts edges with properties.

        Args:
            relation: Relation name (becomes edge type in graph type)
            source_collection: Source node collection name
            target_collection: Target node collection name
            edges: Iterable of Edge objects to add

        """
        edges_list = list(edges)
        if not edges_list:
            return

        await self._discover_existing_indexes()

        # Collect all properties from all edges
        all_properties: dict[str, PropertyValue] = {}
        for edge in edges_list:
            all_properties.update(edge.properties)

        # Ensure edge type exists in graph type
        await self._ensure_graph_type_for_edges(
            relation, source_collection, target_collection, all_properties
        )

        sanitized_relation = self._sanitize_name(relation)
        sanitized_source = self._sanitize_name(source_collection)
        sanitized_target = self._sanitize_name(target_collection)

        # Insert edges
        for edge in edges_list:
            # Build property assignments with embedded values
            prop_assignments = []
            for prop_name, prop_value in edge.properties.items():
                mangled = mangle_property_name(prop_name)
                formatted_value = self._format_value(prop_value)
                prop_assignments.append(f"{mangled}: {formatted_value}")

            # Build INSERT statement with embedded values
            if prop_assignments:
                props_clause = f"{{{', '.join(prop_assignments)}}}"
            else:
                props_clause = ""

            source_uid_formatted = self._format_value(edge.source_uid)
            target_uid_formatted = self._format_value(edge.target_uid)

            insert_stmt = f"""
            MATCH (src:{sanitized_source} {{uid: {source_uid_formatted}}}),
                  (tgt:{sanitized_target} {{uid: {target_uid_formatted}}})
            INSERT (src)-[r@{sanitized_relation} {props_clause}]->(tgt)
            """
            await self._client.execute(insert_stmt)

        # Update edge count
        current_count = self._relation_edge_counts.get(relation, 0)
        new_count = current_count + len(edges_list)
        self._relation_edge_counts[relation] = new_count

        # Create range indexes based on configured hierarchies when threshold is reached
        if (
            self._range_index_threshold
            and new_count >= self._range_index_threshold
            and current_count < self._range_index_threshold
        ):
            task = asyncio.create_task(
                self._create_initial_indexes_if_not_exist(
                    EntityType.EDGE,
                    relation,
                )
            )
            self._background_tasks.add(task)
            task.add_done_callback(self._background_tasks.discard)

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
        Search for nodes with similar embeddings using ANN or KNN search.

        Automatically selects search strategy:
        - ANN (Approximate): Uses vector indexes (IVF/HNSW) when available - fast for large datasets
        - KNN (Exact): Scans all vectors without index - suitable for small datasets

        Falls back to KNN if index unavailable or ANN results insufficient.

        Args:
            collection: Collection name to search
            embedding_name: Name of the embedding vector property
            query_embedding: Query vector
            similarity_metric: COSINE or EUCLIDEAN
            limit: Maximum results to return
            property_filter: Optional property filter expression

        Returns:
            List of Node objects ordered by similarity

        """
        await self._discover_existing_indexes()

        sanitized_collection = self._sanitize_name(collection)
        mangled_embedding = mangle_embedding_name(embedding_name)

        # Check if vector index exists
        # Use mangled and sanitized embedding name to ensure valid identifier
        index_name = f"idx_{sanitized_collection}_{self._sanitize_name(mangled_embedding)}"
        has_index = (
            index_name in self._index_state_cache
            and self._index_state_cache[index_name] == self.CacheIndexState.ONLINE
        )

        # Decide on search mode
        use_ann = has_index and not self._force_exact_similarity_search

        # Build WHERE clause from property filter
        where_clause = ""
        if property_filter:
            where_clause = self._render_filter_expr("n", property_filter)

        # Build query based on search mode
        if use_ann:
            # ANN search requires a finite limit (default to 1000 like Neo4j)
            effective_limit = limit if limit is not None else 1000

            # Use fudge factor when filtering to get more candidates
            if property_filter:
                search_limit = int(effective_limit * self._fudge_factor)
            else:
                search_limit = effective_limit

            # Choose similarity function and order
            metric_name = self._similarity_metric_to_nebula(similarity_metric)
            if similarity_metric == SimilarityMetric.EUCLIDEAN:
                distance_func = "euclidean"
                order_dir = "ASC"
            else:  # COSINE
                distance_func = "inner_product"
                order_dir = "DESC"

            # Build vector literal
            vec_literal = self._vector_to_gql_literal(query_embedding)

            # Build OPTIONS clause
            if self._ann_index_type == "IVF":
                options = (
                    f"{{METRIC: {metric_name}, TYPE: IVF, NPROBE: {self._ivf_nprobe}}}"
                )
            else:  # HNSW
                options = f"{{METRIC: {metric_name}, TYPE: HNSW, EFSEARCH: {self._hnsw_ef_search}}}"

            # Build query
            query_parts = [f"MATCH (n:{sanitized_collection})"]
            if where_clause:
                query_parts.append(f"WHERE {where_clause}")
            query_parts.append(
                f"ORDER BY {distance_func}(n.{mangled_embedding}, {vec_literal}) {order_dir}"
            )
            query_parts.append("APPROXIMATE")
            query_parts.append(f"LIMIT {search_limit}")
            query_parts.append(f"OPTIONS {options}")
            query_parts.append("RETURN n")

            query = "\n".join(query_parts)

            result = await self._client.execute(query)

            # Convert results
            nodes = []
            for row in result:
                node_data = row["n"]
                nodes.append(self._nebula_result_to_node(collection, node_data))

            # Check fallback threshold
            if (
                property_filter
                and len(nodes) < (limit or 100) * self._fallback_threshold
            ):
                # Fall back to exact search
                logger.info(
                    "ANN search returned insufficient results (%s), falling back to exact search",
                    len(nodes),
                )
                return await self._exact_similarity_search(
                    collection=collection,
                    embedding_name=embedding_name,
                    query_embedding=query_embedding,
                    similarity_metric=similarity_metric,
                    limit=limit,
                    property_filter=property_filter,
                )

            # Trim to requested limit
            if limit and len(nodes) > limit:
                nodes = nodes[:limit]

            return nodes

        # Exact search (no index or forced)
        return await self._exact_similarity_search(
            collection=collection,
            embedding_name=embedding_name,
            query_embedding=query_embedding,
            similarity_metric=similarity_metric,
            limit=limit,
            property_filter=property_filter,
        )

    async def _exact_similarity_search(
        self,
        *,
        collection: str,
        embedding_name: str,
        query_embedding: list[float],
        similarity_metric: SimilarityMetric,
        limit: int | None,
        property_filter: FilterExpr | None,
    ) -> list[Node]:
        """
        Perform KNN (K-Nearest Neighbor) exact similarity search.

        This method does not use vector indexes and scans all vectors.
        Suitable for small-sized graphs and low-dimensional vectors.

        Args:
            collection: Collection name
            embedding_name: Embedding property name
            query_embedding: Query vector
            similarity_metric: COSINE or EUCLIDEAN
            limit: Maximum results
            property_filter: Optional filter

        Returns:
            List of nodes ordered by similarity

        """
        sanitized_collection = self._sanitize_name(collection)
        mangled_embedding = mangle_embedding_name(embedding_name)

        # Build WHERE clause
        where_clause = ""
        if property_filter:
            where_clause = self._render_filter_expr("n", property_filter)

        # Build vector literal
        vec_literal = self._vector_to_gql_literal(query_embedding)

        # Choose similarity function
        if similarity_metric == SimilarityMetric.EUCLIDEAN:
            distance_func = "euclidean"
            order_dir = "ASC"
        else:  # COSINE
            distance_func = "inner_product"
            order_dir = "DESC"

        # Build query (no APPROXIMATE keyword = exact search)
        query_parts = [f"MATCH (n:{sanitized_collection})"]
        if where_clause:
            query_parts.append(f"WHERE {where_clause}")
        query_parts.append("RETURN n")
        query_parts.append(
            f"ORDER BY {distance_func}(n.{mangled_embedding}, {vec_literal}) {order_dir}"
        )
        if limit:
            query_parts.append(f"LIMIT {limit}")

        query = "\n".join(query_parts)

        result = await self._client.execute(query)

        # Convert results
        nodes = []
        for row in result:
            node_data = row["n"]
            # Unwrap ValueWrapper if needed
            if hasattr(node_data, "cast_primitive"):
                node_data = node_data.cast_primitive()
                # execute returns node with structure: {id, type, labels, properties}
                if "properties" in node_data:
                    node_data = node_data["properties"]
            nodes.append(self._nebula_result_to_node(collection, node_data))

        return nodes

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
        Search for nodes connected to a specified node via edges.

        Args:
            relation: Edge type name
            other_collection: Collection of related nodes to return
            this_collection: Collection of the anchor node
            this_node_uid: UID of the anchor node
            find_sources: Include source nodes (edges pointing to anchor)
            find_targets: Include target nodes (edges from anchor)
            limit: Maximum results
            edge_property_filter: Filter on edge properties
            node_property_filter: Filter on node properties

        Returns:
            List of related Node objects

        """
        await self._discover_existing_indexes()

        sanitized_relation = self._sanitize_name(relation)
        sanitized_this = self._sanitize_name(this_collection)
        sanitized_other = self._sanitize_name(other_collection)
        node_uid_formatted = self._format_value(this_node_uid)

        # Build pattern based on direction
        if find_sources and find_targets:
            pattern = f"(m:{sanitized_this} {{uid: {node_uid_formatted}}})<-[r:{sanitized_relation}]-(n:{sanitized_other})"
            pattern2 = f"(m:{sanitized_this} {{uid: {node_uid_formatted}}})-[r:{sanitized_relation}]->(n:{sanitized_other})"
        elif find_sources:
            pattern = f"(n:{sanitized_other})-[r:{sanitized_relation}]->(m:{sanitized_this} {{uid: {node_uid_formatted}}})"
            pattern2 = None
        elif find_targets:
            pattern = f"(m:{sanitized_this} {{uid: {node_uid_formatted}}})-[r:{sanitized_relation}]->(n:{sanitized_other})"
            pattern2 = None
        else:
            return []

        # Build WHERE clauses
        where_clauses = []

        if edge_property_filter:
            edge_clause = self._render_filter_expr("r", edge_property_filter)
            if edge_clause:
                where_clauses.append(edge_clause)

        if node_property_filter:
            node_clause = self._render_filter_expr("n", node_property_filter)
            if node_clause:
                where_clauses.append(node_clause)

        # Build query
        if pattern2:
            # Need to UNION two patterns
            query_parts1 = [f"MATCH {pattern}"]
            if where_clauses:
                query_parts1.append(f"WHERE {' AND '.join(where_clauses)}")
            query_parts1.append("RETURN DISTINCT n")

            query_parts2 = [f"MATCH {pattern2}"]
            if where_clauses:
                query_parts2.append(f"WHERE {' AND '.join(where_clauses)}")
            query_parts2.append("RETURN DISTINCT n")

            query_parts = [*query_parts1, "UNION", *query_parts2]
        else:
            query_parts = [f"MATCH {pattern}"]
            if where_clauses:
                query_parts.append(f"WHERE {' AND '.join(where_clauses)}")
            query_parts.append("RETURN DISTINCT n")

        if limit is not None:
            query_parts.append(f"LIMIT {limit}")

        query = "\n".join(query_parts)

        result = await self._client.execute(query)

        # Convert results to Node objects
        nodes = []
        for row in result:
            node_data = row["n"]
            # Unwrap ValueWrapper if needed
            if hasattr(node_data, "cast_primitive"):
                node_data = node_data.cast_primitive()
                # execute returns node with structure: {id, type, labels, properties}
                if "properties" in node_data:
                    node_data = node_data["properties"]
            nodes.append(self._nebula_result_to_node(other_collection, node_data))

        return nodes

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
        Search for nodes ordered by properties with range filtering.

        Args:
            collection: Collection name to search
            by_properties: Property names to order by (hierarchy)
            starting_at: Starting values for each property
            order_ascending: Direction for each property
            include_equal_start: Include nodes equal to starting values
            limit: Maximum results
            property_filter: Additional property filter

        Returns:
            List of Node objects ordered by specified properties

        """
        await self._discover_existing_indexes()

        by_properties_list = list(by_properties)
        starting_at_list = list(starting_at)
        order_ascending_list = list(order_ascending)

        if not by_properties_list:
            raise ValueError("by_properties cannot be empty")

        if len(by_properties_list) != len(starting_at_list) or len(
            by_properties_list
        ) != len(order_ascending_list):
            raise ValueError(
                "by_properties, starting_at, and order_ascending must have same length"
            )

        sanitized_collection = self._sanitize_name(collection)

        # Build WHERE clause from property_filter
        where_clauses = []
        params = {}

        if property_filter:
            filter_clause = self._render_filter_expr("n", property_filter)
            if filter_clause:
                where_clauses.append(filter_clause)

        # Build range conditions
        for i, (prop_name, start_value, ascending) in enumerate(
            zip(
                by_properties_list, starting_at_list, order_ascending_list, strict=False
            )
        ):
            if start_value is None:
                continue

            mangled_prop = mangle_property_name(prop_name)
            param_name = f"start_{i}"
            # Build Jinja2 placeholder for execute_py (double braces for Jinja2 syntax)
            placeholder = f"{{{{{param_name}}}}}"

            # Build comparison operator
            if ascending:
                op = ">=" if include_equal_start else ">"
            else:
                op = "<=" if include_equal_start else "<"

            # Handle datetime values specially
            if isinstance(start_value, datetime):
                # Format datetime to match NebulaGraph's expected format (no microseconds, no timezone)
                params[param_name] = start_value.strftime("%Y-%m-%dT%H:%M:%S")
                where_clauses.append(
                    f"n.{mangled_prop} {op} local_datetime({placeholder})"
                )
            else:
                params[param_name] = start_value
                where_clauses.append(f"n.{mangled_prop} {op} {placeholder}")

        # Build ORDER BY clause
        order_parts = []
        for prop_name, ascending in zip(
            by_properties_list, order_ascending_list, strict=False
        ):
            mangled_prop = mangle_property_name(prop_name)
            direction = "ASC" if ascending else "DESC"
            order_parts.append(f"n.{mangled_prop} {direction}")

        # Build full query
        query_parts = [f"MATCH (n:{sanitized_collection})"]

        if where_clauses:
            query_parts.append(f"WHERE {' AND '.join(where_clauses)}")

        query_parts.append("RETURN n")
        query_parts.append(f"ORDER BY {', '.join(order_parts)}")

        if limit is not None:
            query_parts.append(f"LIMIT {limit}")

        query = "\n".join(query_parts)

        logger.debug("search_directional_nodes query: %s", query)
        logger.debug("search_directional_nodes params: %s", params)

        result = await self._client.execute_py(query, params)

        # Convert results to Node objects
        nodes = []
        for row in result:
            node_data = row["n"]
            # Check if ValueWrapper needs unwrapping
            if hasattr(node_data, "cast_primitive"):
                node_data = node_data.cast_primitive()
                if "properties" in node_data:
                    node_data = node_data["properties"]
            elif isinstance(node_data, dict) and "properties" in node_data:
                node_data = node_data["properties"]
            nodes.append(self._nebula_result_to_node(collection, node_data))

        return nodes

    async def search_matching_nodes(
        self,
        *,
        collection: str,
        limit: int | None = None,
        property_filter: FilterExpr | None = None,
    ) -> list[Node]:
        """
        Search for nodes matching property filters.

        Args:
            collection: Collection name to search
            limit: Maximum results
            property_filter: Property filter expression

        Returns:
            List of matching Node objects

        """
        await self._discover_existing_indexes()

        sanitized_collection = self._sanitize_name(collection)

        # Build WHERE clause
        where_clause = self._render_filter_expr("n", property_filter)

        # Build query
        query_parts = [f"MATCH (n:{sanitized_collection})"]

        if where_clause:
            query_parts.append(f"WHERE {where_clause}")

        query_parts.append("RETURN n")

        if limit is not None:
            query_parts.append(f"LIMIT {limit}")

        query = "\n".join(query_parts)

        result = await self._client.execute(query)

        # Convert results to Node objects
        nodes = []
        for row in result:
            node_data = row["n"]
            # Unwrap ValueWrapper if needed
            if hasattr(node_data, "cast_primitive"):
                node_data = node_data.cast_primitive()
                # execute returns node with structure: {id, type, labels, properties}
                if "properties" in node_data:
                    node_data = node_data["properties"]
            nodes.append(self._nebula_result_to_node(collection, node_data))

        return nodes

    async def get_nodes(
        self,
        *,
        collection: str,
        node_uids: Iterable[str],
    ) -> list[Node]:
        """
        Get nodes by their UIDs.

        Args:
            collection: Collection name
            node_uids: Iterable of node UIDs

        Returns:
            List of Node objects (order not guaranteed)

        """
        node_uids_list = list(node_uids)
        if not node_uids_list:
            return []

        await self._discover_existing_indexes()

        sanitized_collection = self._sanitize_name(collection)

        # Build query with uid filter using execute_py parameters
        query = f"""
        MATCH (n:{sanitized_collection})
        WHERE n.uid IN {{{{uids}}}}
        RETURN n
        """

        args = {"uids": node_uids_list}

        try:
            result = await self._client.execute_py(query, args)
        except Exception as e:
            # If node type doesn't exist (e.g., after delete_all_data), return empty list
            error_msg = str(e)
            if "not found" in error_msg.lower():
                return []
            raise

        # Convert results to Node objects
        nodes = []
        for row in result:
            # Handle ValueWrapper from execute_py results
            node_data = row["n"]
            if hasattr(node_data, "cast_primitive"):
                node_data = node_data.cast_primitive()
                # execute_py returns node with structure: {id, type, labels, properties}
                # Extract just the properties dict
                if "properties" in node_data:
                    node_data = node_data["properties"]
            nodes.append(self._nebula_result_to_node(collection, node_data))

        return nodes

    def _nebula_result_to_node(self, _collection: str, node_data: dict) -> Node:
        """
        Convert NebulaGraph result to Node object.

        Args:
            _collection: Collection name (unused, kept for API consistency)
            node_data: Node data from query result (properties dict)

        Returns:
            Node object

        """
        uid = node_data["uid"]
        properties = {}
        embeddings = {}

        for key, value in node_data.items():
            if key == "uid":
                continue
            if is_mangled_property_name(key):
                prop_name = demangle_property_name(key)
                properties[prop_name] = value
            elif is_mangled_embedding_name(key):
                emb_name = demangle_embedding_name(key)
                # Value should be a vector - extract it and wrap with metric
                # Note: We don't store the metric in DB, so default to COSINE
                vec = None
                if isinstance(value, NVector):
                    vec = value.get_values()
                elif hasattr(value, "as_list"):
                    vec = value.as_list()
                elif isinstance(value, list):
                    vec = value
                else:
                    logger.warning("Unexpected embedding value type: %s", type(value))

                if vec is not None:
                    # Return tuple with vector and default COSINE metric
                    embeddings[emb_name] = (vec, SimilarityMetric.COSINE)

        return Node(
            uid=uid,
            properties=properties,
            embeddings=embeddings,
        )

    async def delete_nodes(
        self,
        *,
        collection: str,
        node_uids: Iterable[str],
    ) -> None:
        """
        Delete nodes and their connected edges.

        Note: GQL doesn't have DETACH DELETE, so must delete edges first.

        Args:
            collection: Collection name
            node_uids: Iterable of node UIDs to delete

        """
        node_uids_list = list(node_uids)
        if not node_uids_list:
            return

        await self._discover_existing_indexes()

        sanitized_collection = self._sanitize_name(collection)

        for uid in node_uids_list:
            uid_formatted = self._format_value(uid)

            # Step 1: Delete all edges connected to this node
            delete_edges_query = f"""
            MATCH (n:{sanitized_collection} {{uid: {uid_formatted}}})-[r]-()
            DELETE r
            """
            try:
                await self._client.execute(delete_edges_query)
            except Exception as e:
                # Node may have no edges
                logger.debug("Error deleting edges for node %s: %s", uid, e)

            # Step 2: Delete the node itself
            delete_node_query = f"""
            MATCH (n:{sanitized_collection} {{uid: {uid_formatted}}})
            DELETE n
            """
            await self._client.execute(delete_node_query)

        # Update node count cache
        if collection in self._collection_node_counts:
            self._collection_node_counts[collection] = max(
                0, self._collection_node_counts[collection] - len(node_uids_list)
            )

    async def delete_all_data(self) -> None:
        """
        Delete all data from the graph.

        This removes all nodes and edges by dropping and recreating the graph instance.
        The graph type schema (node types, edge types, and their properties) is preserved.
        Indexes are dropped and will be recreated when thresholds are met again.
        """
        await self._discover_existing_indexes()

        # Drop the graph instance (clears all data and indexes)
        try:
            await self._client.execute(f"DROP GRAPH IF EXISTS {self._graph_name}")
        except Exception as e:
            logger.warning("Error dropping graph: %s", e)

        # Recreate graph from existing graph type (preserves schema)
        await self._client.execute(
            f"CREATE GRAPH {self._graph_name} TYPED {self._graph_type_name}"
        )

        # Clear index and count caches (data is gone)
        # Keep _graph_type_schemas cache (schema is preserved)
        self._index_state_cache.clear()
        self._collection_node_counts.clear()
        self._relation_edge_counts.clear()

        logger.info("Deleted all data from graph, schema preserved")

    async def close(self) -> None:
        """
        Shut down and release resources.

        Cancels background tasks and closes the client connection.
        """
        # Cancel background tasks
        for task in self._background_tasks:
            task.cancel()

        if self._background_tasks:
            await asyncio.gather(*self._background_tasks, return_exceptions=True)

        self._background_tasks.clear()

        # Client is managed externally by DatabaseManager, don't close it here
        logger.info("NebulaGraphVectorGraphStore closed")

    # =========================================================================
    # Helper Methods (to be implemented)
    # =========================================================================

    @staticmethod
    def _sanitize_name(name: str) -> str:
        """
        Sanitize identifier name for use in GQL queries.

        Converts unsafe characters to _uXX_ format where XX is hex code.
        Adds SANITIZED_ prefix to avoid conflicts.

        Args:
            name: Original identifier name

        Returns:
            Sanitized identifier safe for GQL

        """
        result = "SANITIZED_"
        for char in name:
            if char.isalnum() or char == "_":
                result += char
            else:
                result += f"_u{ord(char):x}_"
        return result

    @staticmethod
    def _desanitize_name(sanitized: str) -> str:
        """
        Restore original name from sanitized form.

        Args:
            sanitized: Sanitized identifier

        Returns:
            Original identifier name

        """
        if not sanitized.startswith("SANITIZED_"):
            return sanitized

        name = sanitized[len("SANITIZED_") :]
        result = ""
        i = 0
        while i < len(name):
            if name[i : i + 2] == "_u" and "_" in name[i + 2 :]:
                # Find the closing underscore
                end_idx = name.index("_", i + 2)
                hex_code = name[i + 2 : end_idx]
                try:
                    result += chr(int(hex_code, 16))
                    i = end_idx + 1
                except ValueError:
                    result += name[i]
                    i += 1
            else:
                result += name[i]
                i += 1
        return result

    def _infer_gql_type(self, value: PropertyValue) -> str:
        """
        Infer GQL type from Python value.

        Args:
            value: Python value to infer type from

        Returns:
            GQL type string

        """
        # Check bool before int (bool is subclass of int in Python)
        if isinstance(value, bool):
            return "BOOL"
        if isinstance(value, int):
            return "INT64"
        if isinstance(value, float):
            return "DOUBLE"
        if isinstance(value, str):
            return "STRING"
        if isinstance(value, datetime):
            return "LOCAL DATETIME"
        if isinstance(value, list):
            if not value:
                return "LIST<STRING>"
            # Check bool before int (bool is subclass of int in Python)
            if all(isinstance(x, bool) for x in value):
                return "LIST<BOOL>"
            if all(isinstance(x, int) and not isinstance(x, bool) for x in value):
                return "LIST<INT64>"
            if all(isinstance(x, float) for x in value):
                return "LIST<DOUBLE>"
            if all(isinstance(x, datetime) for x in value):
                return "LIST<LOCAL DATETIME>"
            return "LIST<STRING>"
        raise ValueError(f"Unsupported property type: {type(value)}")

    def _format_value(self, value: PropertyValue) -> str:
        """
        Format Python value for GQL query.

        Args:
            value: Python value to format

        Returns:
            GQL-formatted value string

        """
        if isinstance(value, str):
            escaped = value.replace("\\", "\\\\").replace('"', '\\"')
            return f'"{escaped}"'
        # Check bool before int/float (bool is subclass of int in Python)
        if isinstance(value, bool):
            return str(value).lower()
        if (isinstance(value, int) and not isinstance(value, bool)) or isinstance(
            value, float
        ):
            return str(value)
        if isinstance(value, datetime):
            # Use local_datetime() function for proper type conversion
            # Format: YYYY-MM-DDTHH:MM:SS (no microseconds, no timezone)
            formatted_str = value.strftime("%Y-%m-%dT%H:%M:%S")
            return f'local_datetime("{formatted_str}")'
        if isinstance(value, list):
            formatted = [self._format_value(v) for v in value]
            return f"[{', '.join(formatted)}]"
        if value is None:
            return "NULL"
        raise ValueError(f"Unsupported value type: {type(value)}")

    def _vector_to_gql_literal(self, vec: list[float]) -> str:
        """
        Convert Python list to GQL VECTOR literal.

        Args:
            vec: Vector as list of floats

        Returns:
            GQL VECTOR literal

        """
        vec_str = ", ".join(str(v) for v in vec)
        return f"VECTOR<{len(vec)}, FLOAT>([{vec_str}])"

    def _similarity_metric_to_nebula(self, metric: SimilarityMetric) -> str:
        """
        Convert SimilarityMetric to NebulaGraph metric name.

        Args:
            metric: Similarity metric

        Returns:
            NebulaGraph metric name (L2 or IP)

        """
        if metric == SimilarityMetric.EUCLIDEAN:
            return "L2"
        if metric == SimilarityMetric.COSINE:
            return "IP"
        raise ValueError(f"Unsupported similarity metric: {metric}")

    def _render_filter_expr(
        self,
        node_alias: str,
        filter_expr: FilterExpr | None,
    ) -> str:
        """
        Convert FilterExpr to NebulaGraph WHERE clause with embedded values.

        Args:
            node_alias: GQL variable name for the node
            filter_expr: Filter expression AST

        Returns:
            WHERE clause string with values embedded directly

        """
        if filter_expr is None:
            return ""

        def render(expr: FilterExpr) -> str:
            if isinstance(expr, Comparison):
                prop_name = mangle_property_name(expr.field)
                prop_ref = f"{node_alias}.{prop_name}"

                if expr.op == "==":
                    value_str = self._format_value(expr.value)
                    return f"{prop_ref} = {value_str}"
                if expr.op == "!=":
                    value_str = self._format_value(expr.value)
                    return f"{prop_ref} <> {value_str}"
                if expr.op == ">":
                    value_str = self._format_value(expr.value)
                    return f"{prop_ref} > {value_str}"
                if expr.op == ">=":
                    value_str = self._format_value(expr.value)
                    return f"{prop_ref} >= {value_str}"
                if expr.op == "<":
                    value_str = self._format_value(expr.value)
                    return f"{prop_ref} < {value_str}"
                if expr.op == "<=":
                    value_str = self._format_value(expr.value)
                    return f"{prop_ref} <= {value_str}"
                if expr.op == "in":
                    if not isinstance(expr.value, list):
                        raise ValueError("'in' operator requires list value")
                    value_strs = [self._format_value(v) for v in expr.value]
                    values_list = ", ".join(value_strs)
                    return f"{prop_ref} IN [{values_list}]"
                raise ValueError(f"Unsupported operator: {expr.op}")

            if isinstance(expr, And):
                left_clause = render(expr.left)
                right_clause = render(expr.right)
                return f"({left_clause}) AND ({right_clause})"

            if isinstance(expr, Or):
                left_clause = render(expr.left)
                right_clause = render(expr.right)
                return f"({left_clause}) OR ({right_clause})"

            raise ValueError(f"Unsupported filter expression type: {type(expr)}")

        where_clause = render(filter_expr)
        return where_clause

    async def _discover_existing_indexes(self) -> None:
        """
        Discover existing indexes in NebulaGraph and populate cache.

        Queries NebulaGraph for all indexes on the current graph and populates
        _index_state_cache with their states. This ensures that after a restart,
        existing indexes are recognized and ANN search can be used.

        Called on first operation. Session context (schema/graph) is already set
        by database_manager during client initialization.
        """
        if self._indexes_discovered:
            return

        try:
            # Query all indexes for current graph
            result = await self._client.execute("SHOW INDEXES")

            for row in result:
                index_name = row["name"]
                state = row["state"]
                index_type = row["index_type"]

                # Only track vector and normal indexes that are valid
                if state == "Valid":
                    if isinstance(index_type, str) and index_type.startswith("Vector"):
                        # Vector index is online and ready
                        self._index_state_cache[index_name] = (
                            self.CacheIndexState.ONLINE
                        )
                        logger.info("Discovered vector index: %s", index_name)
                    elif index_type == "Normal":
                        # Normal (range) index
                        self._index_state_cache[index_name] = (
                            self.CacheIndexState.ONLINE
                        )
                        logger.debug("Discovered normal index: %s", index_name)

            self._indexes_discovered = True
            logger.info(
                "Index discovery complete: found %s indexes",
                len(self._index_state_cache),
            )

        except Exception as e:
            # Don't fail if index discovery fails - allow retry on next operation
            # This handles transient failures (network issues, temporary unavailability)
            logger.warning("Failed to discover existing indexes, will retry: %s", e)

    async def _ensure_graph_type_for_nodes(
        self,
        collection: str,
        properties: dict[str, PropertyValue],
        embeddings: dict[str, tuple[list[float], SimilarityMetric]],
    ) -> None:
        """
        Ensure node type exists in graph type for this collection.

        Creates node type on first encounter. On subsequent calls, evolves schema
        by adding any new properties/embeddings not present in the existing schema.
        Existing nodes will have NULL values for newly added properties.

        Args:
            collection: Collection name (node type)
            properties: Node properties from current batch
            embeddings: Embeddings from current batch (name -> (vector, metric))

        """
        async with self._schema_lock:
            sanitized_collection = self._sanitize_name(collection)

            # Build schema from incoming data
            incoming_schema: dict[str, str] = {"uid": "STRING"}
            for prop_name, prop_value in properties.items():
                mangled = mangle_property_name(prop_name)
                incoming_schema[mangled] = self._infer_gql_type(prop_value)

            for emb_name, (vec, _metric) in embeddings.items():
                mangled = mangle_embedding_name(emb_name)
                incoming_schema[mangled] = f"VECTOR<{len(vec)}, FLOAT>"

            # Check if node type already exists in cache
            if collection in self._graph_type_schemas:
                # Node type exists - check for schema evolution
                cached_schema = self._graph_type_schemas[collection]
                new_properties = {}

                # Find properties/embeddings that aren't in the cached schema
                for prop_name, prop_type in incoming_schema.items():
                    if prop_name == "uid":
                        continue  # Skip primary key
                    if prop_name not in cached_schema:
                        new_properties[prop_name] = prop_type

                # If no new properties, we're done
                if not new_properties:
                    return

                # Evolve schema: add new properties to existing node type
                # Existing nodes will have NULL for these new properties
                prop_defs = [
                    f"{prop_name} {prop_type}"
                    for prop_name, prop_type in new_properties.items()
                ]

                alter_stmt = f"""
                ALTER GRAPH TYPE {self._graph_type_name} {{
                    ALTER NODE TYPE {sanitized_collection} ADD PROPERTIES {{{", ".join(prop_defs)}}}
                }}
                """
                await self._client.execute(alter_stmt)
                logger.info(
                    "Evolved schema for node type '%s': added %s new properties",
                    sanitized_collection,
                    len(new_properties),
                )

                # Update cached schema
                self._graph_type_schemas[collection].update(new_properties)
                return

            # First time seeing this collection - create node type
            # Build property definitions
            prop_defs = []
            for prop_name, prop_type in incoming_schema.items():
                if prop_name == "uid":
                    prop_defs.append(f"{prop_name} {prop_type} PRIMARY KEY")
                else:
                    prop_defs.append(f"{prop_name} {prop_type}")

            # Add node type to graph type using ALTER GRAPH TYPE
            alter_stmt = f"""
            ALTER GRAPH TYPE {self._graph_type_name} {{
                ADD NODE TYPE IF NOT EXISTS {sanitized_collection} (
                    LABEL {sanitized_collection} {{
                        {", ".join(prop_defs)}
                    }}
                )
            }}
            """
            await self._client.execute(alter_stmt)
            logger.info("Added node type '%s' to graph type", sanitized_collection)

            # Cache the schema
            self._graph_type_schemas[collection] = incoming_schema

    async def _ensure_graph_type_for_edges(
        self,
        relation: str,
        source_collection: str,
        target_collection: str,
        properties: dict[str, PropertyValue],
    ) -> None:
        """
        Ensure edge type exists in graph type for this relation.

        Creates edge type on first encounter. On subsequent calls, evolves schema
        by adding any new properties not present in the existing schema.
        Existing edges will have NULL values for newly added properties.

        Args:
            relation: Relation name (edge type)
            source_collection: Source node type
            target_collection: Target node type
            properties: Edge properties from current batch

        """
        async with self._schema_lock:
            edge_key = f"edge_{relation}"
            sanitized_relation = self._sanitize_name(relation)
            sanitized_source = self._sanitize_name(source_collection)
            sanitized_target = self._sanitize_name(target_collection)

            # Build schema from incoming data
            incoming_schema: dict[str, str] = {}
            for prop_name, prop_value in properties.items():
                mangled = mangle_property_name(prop_name)
                incoming_schema[mangled] = self._infer_gql_type(prop_value)

            # Check if edge type already exists in cache
            if edge_key in self._graph_type_schemas:
                # Edge type exists - check for schema evolution
                cached_schema = self._graph_type_schemas[edge_key]
                new_properties = {}

                # Find properties that aren't in the cached schema
                for prop_name, prop_type in incoming_schema.items():
                    if prop_name not in cached_schema:
                        new_properties[prop_name] = prop_type

                # If no new properties, we're done
                if not new_properties:
                    return

                # Evolve schema: add new properties to existing edge type
                # Existing edges will have NULL for these new properties
                prop_defs = [
                    f"{prop_name} {prop_type}"
                    for prop_name, prop_type in new_properties.items()
                ]

                alter_stmt = f"""
                ALTER GRAPH TYPE {self._graph_type_name} {{
                    ALTER EDGE TYPE {sanitized_relation} ADD PROPERTIES {{{", ".join(prop_defs)}}}
                }}
                """
                await self._client.execute(alter_stmt)
                logger.info(
                    "Evolved schema for edge type '%s': added %s new properties",
                    sanitized_relation,
                    len(new_properties),
                )

                # Update cached schema
                self._graph_type_schemas[edge_key].update(new_properties)
                return

            # First time seeing this relation - create edge type
            # Build property definitions
            if incoming_schema:
                prop_defs = [
                    f"{prop_name} {prop_type}"
                    for prop_name, prop_type in incoming_schema.items()
                ]
                props_clause = (
                    f"LABEL {sanitized_relation} {{ {', '.join(prop_defs)} }}"
                )
            else:
                props_clause = f"LABEL {sanitized_relation} {{}}"

            # Add edge type to graph type using ALTER GRAPH TYPE
            # Syntax: ADD EDGE TYPE name (SourceType)-[LABEL name {props}]->(TargetType)
            alter_stmt = f"""
            ALTER GRAPH TYPE {self._graph_type_name} {{
                ADD EDGE TYPE IF NOT EXISTS {sanitized_relation} ({sanitized_source})-[{props_clause}]->({sanitized_target})
            }}
            """
            logger.debug("ALTER statement for edge type: %s", alter_stmt)
            await self._client.execute(alter_stmt)
            logger.info("Added edge type '%s' to graph type", sanitized_relation)

            # Cache the schema
            self._graph_type_schemas[edge_key] = incoming_schema

    async def _create_vector_index_if_not_exists(
        self,
        entity_type: EntityType,
        node_or_edge_type: str,
        embedding_name: str,
        dimensions: int,
        similarity_metric: SimilarityMetric,
    ) -> None:
        """
        Create vector index if it doesn't exist.

        Args:
            entity_type: NODE or EDGE
            node_or_edge_type: Type name
            embedding_name: Embedding property name
            dimensions: Vector dimensions
            similarity_metric: COSINE or EUCLIDEAN

        """
        # Use mangled and sanitized embedding name to ensure valid identifier
        mangled_embedding = mangle_embedding_name(embedding_name)
        index_name = f"idx_{self._sanitize_name(node_or_edge_type)}_{self._sanitize_name(mangled_embedding)}"

        # Check cache
        if index_name in self._index_state_cache:
            return

        # Acquire lock for this index
        if index_name not in self._index_locks:
            self._index_locks[index_name] = asyncio.Lock()

        async with self._index_locks[index_name]:
            # Double-check after acquiring lock
            if index_name in self._index_state_cache:
                return

            # Mark as creating
            self._index_state_cache[index_name] = self.CacheIndexState.CREATING

            try:
                sanitized_type = self._sanitize_name(node_or_edge_type)
                metric = self._similarity_metric_to_nebula(similarity_metric)

                # Build index options based on type
                if self._ann_index_type == "IVF":
                    options = f"""{{
                        DIM: {dimensions},
                        METRIC: {metric},
                        TYPE: IVF,
                        NLIST: {self._ivf_nlist},
                        TRAINSIZE: 10000
                    }}"""
                else:  # HNSW
                    options = f"""{{
                        DIM: {dimensions},
                        METRIC: {metric},
                        TYPE: HNSW,
                        MAXDEGREE: {self._hnsw_max_degree},
                        EFCONSTRUCTION: {self._hnsw_ef_construction},
                        CAPACITY: 1000000
                    }}"""

                # Create index
                if entity_type == EntityType.NODE:
                    create_stmt = f"""
                    CREATE VECTOR INDEX IF NOT EXISTS {index_name}
                    ON NODE {sanitized_type}::{mangled_embedding}
                    OPTIONS {options}
                    """
                else:  # EDGE
                    create_stmt = f"""
                    CREATE VECTOR INDEX IF NOT EXISTS {index_name}
                    ON EDGE {sanitized_type}::{mangled_embedding}
                    OPTIONS {options}
                    """

                await self._client.execute(create_stmt)

                # Mark as online
                self._index_state_cache[index_name] = self.CacheIndexState.ONLINE
                logger.info("Created vector index: %s", index_name)

            except Exception:
                # Remove from cache on error
                self._index_state_cache.pop(index_name, None)
                logger.exception("Failed to create vector index %s", index_name)
                raise

    async def _create_range_index_if_not_exists(
        self,
        entity_type: EntityType,
        node_or_edge_type: str,
        properties: list[str],
    ) -> None:
        """
        Create normal (range) index if it doesn't exist.

        Args:
            entity_type: NODE or EDGE
            node_or_edge_type: Type name
            properties: List of property names to index

        """
        if not properties:
            return

        index_name = (
            f"idx_{self._sanitize_name(node_or_edge_type)}_{'_'.join(properties)}"
        )

        # Check cache
        if index_name in self._index_state_cache:
            return

        # Acquire lock for this index
        if index_name not in self._index_locks:
            self._index_locks[index_name] = asyncio.Lock()

        async with self._index_locks[index_name]:
            # Double-check after acquiring lock
            if index_name in self._index_state_cache:
                return

            # Mark as creating
            self._index_state_cache[index_name] = self.CacheIndexState.CREATING

            try:
                sanitized_type = self._sanitize_name(node_or_edge_type)
                mangled_props = [mangle_property_name(p) for p in properties]
                prop_list = ", ".join(f"{p} ASC" for p in mangled_props)

                # Create index on NODE or EDGE
                entity_keyword = "NODE" if entity_type == EntityType.NODE else "EDGE"
                create_stmt = f"""
                CREATE NORMAL INDEX IF NOT EXISTS {index_name}
                ON {entity_keyword} {sanitized_type} ({prop_list})
                """
                await self._client.execute(create_stmt)

                # Mark as online
                self._index_state_cache[index_name] = self.CacheIndexState.ONLINE
                logger.info("Created range index: %s", index_name)

            except Exception:
                # Remove from cache on error
                self._index_state_cache.pop(index_name, None)
                logger.exception("Failed to create range index %s", index_name)
                raise

    async def _create_initial_indexes_if_not_exist(
        self,
        entity_type: EntityType,
        node_or_edge_type: str,
    ) -> None:
        """
        Create initial range indexes based on configured hierarchies.

        Creates composite indexes for each hierarchy prefix. For example,
        [["user_id", "timestamp"]] creates indexes on:
        - (user_id)
        - (user_id, timestamp)

        Args:
            entity_type: NODE or EDGE
            node_or_edge_type: Type name

        """
        # Create indexes for each property hierarchy
        tasks = []
        for range_index_hierarchy in self._range_index_hierarchies:
            # Create index for each prefix of the hierarchy
            for i in range(len(range_index_hierarchy)):
                property_name_hierarchy = range_index_hierarchy[: i + 1]
                tasks.append(
                    self._create_range_index_if_not_exists(
                        entity_type=entity_type,
                        node_or_edge_type=node_or_edge_type,
                        properties=property_name_hierarchy,
                    )
                )

        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
