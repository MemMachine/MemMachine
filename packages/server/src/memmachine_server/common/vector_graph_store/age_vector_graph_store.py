"""Apache AGE-backed vector graph store implementation.

Apache AGE is a PostgreSQL extension that adds openCypher support on top of
relational storage. Because AGE does not provide a native vector index, this
implementation stores graph structure and scalar properties inside AGE and
delegates vector similarity search to a pgvector-indexed side table that is
keyed by vertex or edge uid. A single PostgreSQL instance therefore provides
both the graph backend and (via ``SqlAlchemyPgVectorSemanticStorage``) the
semantic backend under permissive, Apache-2.0-compatible licensing.
"""

import asyncio
import hashlib
import logging
from collections.abc import Awaitable, Iterable, Mapping
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, InstanceOf
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncConnection, AsyncEngine

from memmachine_server.common.age_utils import (
    AgeVertex,
    age_value_from_python,
    age_value_to_python,
    build_cypher_call,
    desanitize_identifier,
    encode_agtype_params,
    parse_agtype,
    render_comparison,
    sanitize_identifier,
    validate_graph_name,
)
from memmachine_server.common.data_types import (
    OrderedValue,
    SimilarityMetric,
)
from memmachine_server.common.filter.filter_parser import (
    And as FilterAnd,
)
from memmachine_server.common.filter.filter_parser import (
    Comparison as FilterComparison,
)
from memmachine_server.common.filter.filter_parser import (
    FilterExpr,
)
from memmachine_server.common.filter.filter_parser import (
    In as FilterIn,
)
from memmachine_server.common.filter.filter_parser import (
    IsNull as FilterIsNull,
)
from memmachine_server.common.filter.filter_parser import (
    Not as FilterNot,
)
from memmachine_server.common.filter.filter_parser import (
    Or as FilterOr,
)
from memmachine_server.common.metrics_factory import MetricsFactory, OperationTracker
from memmachine_server.common.utils import async_locked

from .data_types import (
    Edge,
    EntityType,
    Node,
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

# Default AGE graph name. AGE creates a schema of this name plus a pair of
# catalog tables; the side tables for vector storage are prefixed with it.
DEFAULT_GRAPH_NAME = "mem_graph"

# Upper bound for vector dimensionality. pgvector caps ``vector`` columns at
# 16000 dimensions, which we surface directly here rather than inheriting
# Neo4j's tighter 4096 cap — users who pick AGE specifically because they
# want the pgvector ceiling shouldn't be artificially limited.
_MAX_VECTOR_DIMENSIONS = 16000

# Postgres caps identifier length at 63 bytes (NAMEDATALEN - 1). Side-table
# names are derived from a sha1 digest so they always fit, but the
# (collection_label, embedding_name) mapping must still be recoverable for
# cleanup and hydration. A per-graph registry table holds that mapping.
_SIDE_TABLE_REGISTRY_SUFFIX = "_emb_registry"


class AgeVectorGraphStoreParams(BaseModel):
    """Parameters for :class:`AgeVectorGraphStore`.

    Attributes:
        engine: Async SQLAlchemy engine pointing at a PostgreSQL database with
            the ``age`` and ``vector`` extensions available.
        graph_name: AGE graph name. Used as the PostgreSQL schema name for the
            AGE catalog and as a prefix for the pgvector side tables.
        force_exact_similarity_search: Skip the pgvector HNSW index and always
            perform an exact KNN scan (useful for small collections or tests).
        filtered_similarity_search_fudge_factor: Multiplier applied to the
            search limit when a property filter is combined with ANN search, to
            compensate for the index returning unfiltered neighbors.
        exact_similarity_search_fallback_threshold: If an ANN-plus-filter
            search returns fewer than ``threshold * limit`` rows, fall back to
            an exact scan that applies the filter in SQL.
        range_index_hierarchies: List of property name hierarchies for which to
            create composite B-tree indexes on the AGE vertex/edge tables.
        range_index_creation_threshold: Minimum entity count at which range
            indexes are created automatically.
        vector_index_creation_threshold: Minimum entity count at which HNSW
            vector indexes are created automatically.
        hnsw_m: pgvector HNSW ``m`` parameter (max connections per layer).
        hnsw_ef_construction: pgvector HNSW ``ef_construction`` parameter.
        metrics_factory: Optional metrics factory for operation tracking.
    """

    engine: InstanceOf[AsyncEngine] = Field(
        ...,
        description="Async SQLAlchemy engine connected to PostgreSQL with AGE",
    )
    graph_name: str = Field(
        default=DEFAULT_GRAPH_NAME,
        description="AGE graph name (also used as the side-table prefix)",
    )
    force_exact_similarity_search: bool = Field(
        default=False,
        description="Whether to force exact similarity search",
    )
    filtered_similarity_search_fudge_factor: int = Field(
        default=4,
        description=(
            "Fudge factor for filtered similarity search. pgvector HNSW does "
            "not support pre-filtering, so we over-fetch by this multiplier "
            "and filter in SQL."
        ),
        gt=0,
    )
    exact_similarity_search_fallback_threshold: float = Field(
        default=0.5,
        description=(
            "Threshold ratio of ANN results to limit below which to fall back "
            "to exact similarity search when a property filter is applied."
        ),
        ge=0.0,
        le=1.0,
    )
    range_index_hierarchies: list[list[str]] = Field(
        default_factory=list,
        description=(
            "List of property name hierarchies for which to create composite "
            "B-tree indexes on the underlying AGE vertex/edge tables."
        ),
    )
    range_index_creation_threshold: int = Field(
        default=10_000,
        description=(
            "Threshold number of entities at which range indexes may be created."
        ),
    )
    vector_index_creation_threshold: int = Field(
        default=10_000,
        description=(
            "Threshold number of entities at which vector indexes may be created."
        ),
    )
    hnsw_m: int = Field(
        default=16,
        description="pgvector HNSW 'm' parameter (max connections per layer)",
        gt=0,
    )
    hnsw_ef_construction: int = Field(
        default=64,
        description="pgvector HNSW 'ef_construction' parameter",
        gt=0,
    )
    metrics_factory: InstanceOf[MetricsFactory] | None = Field(
        default=None,
        description="An instance of MetricsFactory for collecting usage metrics",
    )


# AGE label names cannot be parameterized and must be quoted carefully. We use
# the same sanitization scheme as the Neo4j backend so the on-wire identifiers
# only ever contain ASCII letters, digits, and underscores.
class AgeVectorGraphStore(VectorGraphStore):
    """Asynchronous vector graph store backed by PostgreSQL + Apache AGE."""

    class CacheIndexState(Enum):
        """Cached index state (not authoritative against the database)."""

        CREATING = 0
        ONLINE = 1

    def __init__(self, params: AgeVectorGraphStoreParams) -> None:
        """Initialize the graph store from ``params``."""
        super().__init__()

        self._engine: AsyncEngine = params.engine
        self._graph_name = validate_graph_name(params.graph_name)

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
        self._hnsw_m = params.hnsw_m
        self._hnsw_ef_construction = params.hnsw_ef_construction

        self._index_state_cache: dict[str, AgeVectorGraphStore.CacheIndexState] = {}
        self._populate_index_state_cache_lock = asyncio.Lock()

        # These are only used for tracking counts approximately.
        self._collection_node_counts: dict[str, int] = {}
        self._relation_edge_counts: dict[str, int] = {}

        self._ensured_labels: set[tuple[EntityType, str]] = set()
        self._ensured_labels_lock = asyncio.Lock()
        self._ensured_vector_tables: set[str] = set()
        self._ensured_vector_tables_lock = asyncio.Lock()

        self._graph_initialized = False
        self._graph_initialized_lock = asyncio.Lock()

        self._background_tasks: set[asyncio.Task] = set()

        self._tracker = OperationTracker(
            params.metrics_factory, prefix="vector_graph_store_age"
        )

    def _track_task(self, task: asyncio.Task) -> None:
        """Keep background tasks from being garbage collected prematurely."""
        self._background_tasks.add(task)
        task.add_done_callback(self._background_tasks.discard)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def _ensure_graph_initialized(self) -> None:
        """Idempotently create the AGE extension and graph namespace.

        The AGE extension and per-graph schema must exist before any Cypher
        call runs. We do this lazily on the first operation so tests and
        applications can construct the store without blocking on IO.

        ``CREATE EXTENSION`` requires elevated privileges. In environments
        where the application role lacks them, operators are expected to
        install ``age`` and ``vector`` ahead of time. We tolerate that case by
        only raising if the extension is still missing after the attempt, and
        we run each ``CREATE EXTENSION`` in its own transaction so a failure
        does not poison subsequent queries in a shared transaction.
        """
        if self._graph_initialized:
            return
        async with self._graph_initialized_lock:
            if self._graph_initialized:
                return
            await self._ensure_extension("age")
            await self._ensure_extension("vector")
            await self._assert_age_version_supported()
            async with self._engine.begin() as conn:
                await conn.exec_driver_sql("LOAD 'age'")
                await conn.exec_driver_sql(
                    'SET search_path = ag_catalog, "$user", public'
                )
                # ``create_graph`` throws if the graph already exists, so we
                # guard with a lookup against ag_catalog.ag_graph.
                exists_row = (
                    await conn.exec_driver_sql(
                        "SELECT 1 FROM ag_catalog.ag_graph WHERE name = $1",
                        (self._graph_name,),
                    )
                ).first()
                if exists_row is None:
                    await conn.exec_driver_sql(
                        "SELECT create_graph($1)",
                        (self._graph_name,),
                    )
            self._graph_initialized = True

    async def _ensure_extension(self, name: str) -> None:
        """Install a PostgreSQL extension if possible, else verify it exists.

        The CREATE and the follow-up probe each run in their own transaction.
        Postgres marks a transaction as aborted after any error, so reusing a
        single transaction would make the ``pg_extension`` lookup fail with
        "current transaction is aborted" whenever the role lacks CREATE
        privilege. Splitting them lets operators pre-install the extension
        and still have startup succeed.
        """
        try:
            async with self._engine.begin() as conn:
                await conn.exec_driver_sql(f"CREATE EXTENSION IF NOT EXISTS {name}")
        except SQLAlchemyError as exc:
            async with self._engine.connect() as conn:
                installed = (
                    await conn.exec_driver_sql(
                        "SELECT 1 FROM pg_extension WHERE extname = $1",
                        (name,),
                    )
                ).first()
            if installed is None:
                raise RuntimeError(
                    f"PostgreSQL extension '{name}' is not installed and "
                    "could not be created automatically. Ask a superuser to "
                    f"run 'CREATE EXTENSION {name}' in the target database."
                ) from exc

    async def _assert_age_version_supported(self) -> None:
        # The ``WITH $param AS p`` workaround used by add_nodes/add_edges
        # relies on cypher() parameter semantics that first shipped in AGE
        # 1.6.0. Older versions silently reject ``SET n += p`` with a
        # "SET clause expects a map" error that is opaque to operators.
        async with self._engine.connect() as conn:
            row = (
                await conn.exec_driver_sql(
                    "SELECT extversion FROM pg_extension WHERE extname = 'age'"
                )
            ).first()
        if row is None:
            return
        version_str = str(row[0])
        try:
            parts = tuple(int(p) for p in version_str.split(".")[:3])
        except ValueError:
            logger.warning("Unparseable AGE extversion %r; skipping version check", version_str)
            return
        if parts < (1, 6, 0):
            raise RuntimeError(
                f"Apache AGE {version_str} is not supported by "
                "AgeVectorGraphStore; require >= 1.6.0."
            )

    def _reset_in_memory_caches(self) -> None:
        # Test-only helper: clear every in-process cache so a fixture that
        # drops all data between cases doesn't leave stale counts or
        # ensured-table/index state behind.
        self._collection_node_counts.clear()
        self._relation_edge_counts.clear()
        self._ensured_vector_tables.clear()
        self._ensured_labels.clear()
        self._index_state_cache.clear()

    async def close(self) -> None:
        """Dispose of the underlying SQLAlchemy engine.

        Any index-creation tasks spawned by ``add_nodes`` / ``add_edges`` are
        drained first. Failures there don't block dispose — they're
        best-effort optimizations — but we log them so they don't disappear
        into the interpreter's default task finalizer.
        """
        if self._background_tasks:
            results = await asyncio.gather(
                *self._background_tasks,
                return_exceptions=True,
            )
            for result in results:
                if isinstance(result, BaseException):
                    logger.warning(
                        "Background index task raised during close: %s", result
                    )
        await self._engine.dispose()

    # ------------------------------------------------------------------
    # Writes
    # ------------------------------------------------------------------

    async def add_nodes(
        self,
        *,
        collection: str,
        nodes: Iterable[Node],
    ) -> None:
        """Add nodes to a collection, creating labels and indexes as needed."""
        async with self._tracker("add_nodes"):
            await self._ensure_graph_initialized()

            nodes = list(nodes)
            if not nodes:
                return

            sanitized_collection = sanitize_identifier(collection)
            # The label must exist before any MATCH/count on it, because AGE
            # raises "label does not exist" if create_vlabel has not been
            # called for the target label.
            await self._ensure_vertex_label(sanitized_collection)

            if collection not in self._collection_node_counts:
                self._collection_node_counts[collection] = await self._count_nodes(
                    collection,
                )

            embedding_dimensions: dict[str, int] = {}
            embedding_similarity: dict[str, SimilarityMetric] = {}
            sanitized_embedding_names: set[str] = set()

            # Create vertex rows first. We do this one call per node so we can
            # capture the generated AGE id and write the embedding rows in the
            # same transaction-scoped connection.
            async with self._engine.begin() as conn:
                for node in nodes:
                    sanitized_properties = _sanitize_properties(
                        {
                            mangle_property_name(key): value
                            for key, value in node.properties.items()
                        }
                    )
                    create_props: dict[str, Any] = {
                        "uid": str(node.uid),
                        **sanitized_properties,
                    }
                    params_json = encode_agtype_params({"props": create_props})
                    # AGE 1.6.0 rejects ``SET n += $param`` with "SET clause
                    # expects a map" even though the parameter resolves to a
                    # map when returned directly. Aliasing the parameter with
                    # ``WITH`` first sidesteps the bug without changing the
                    # semantics.
                    sql = build_cypher_call(
                        self._graph_name,
                        (
                            "WITH $props AS p "
                            f"CREATE (n:{sanitized_collection}) "
                            "SET n += p RETURN id(n) AS id"
                        ),
                        returns=("id",),
                        has_params=True,
                    )
                    result = await conn.exec_driver_sql(sql, (params_json,))
                    row = result.first()
                    if row is None:
                        raise RuntimeError(
                            f"AGE returned no id after CREATE for uid={node.uid}"
                        )
                    vertex_id = int(parse_agtype(row[0]))

                    for embedding_name, (
                        embedding,
                        similarity_metric,
                    ) in node.embeddings.items():
                        sanitized_embedding_name = sanitize_identifier(
                            mangle_embedding_name(embedding_name),
                        )
                        sanitized_embedding_names.add(sanitized_embedding_name)
                        embedding_dimensions[sanitized_embedding_name] = len(embedding)
                        embedding_similarity[sanitized_embedding_name] = (
                            similarity_metric
                        )

                        table_name = self._vector_table_name(
                            EntityType.NODE,
                            sanitized_collection,
                            sanitized_embedding_name,
                        )
                        await self._ensure_vector_table(
                            table_name,
                            entity_type=EntityType.NODE,
                            sanitized_collection_or_relation=sanitized_collection,
                            sanitized_embedding_name=sanitized_embedding_name,
                            dimensions=len(embedding),
                            uid_column="node_uid",
                            conn=conn,
                        )
                        await conn.exec_driver_sql(
                            (
                                f'INSERT INTO public."{table_name}"'
                                "  (node_uid, vertex_id, similarity_metric, embedding)"
                                " VALUES ($1, $2, $3, $4::vector)"
                                " ON CONFLICT (node_uid) DO UPDATE SET"
                                "  vertex_id = EXCLUDED.vertex_id,"
                                "  similarity_metric = EXCLUDED.similarity_metric,"
                                "  embedding = EXCLUDED.embedding"
                            ),
                            (
                                str(node.uid),
                                vertex_id,
                                similarity_metric.value,
                                _vector_literal(embedding),
                            ),
                        )

            self._collection_node_counts[collection] += len(nodes)

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
                        self._index_name(
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
                                    sanitized_collection_or_relation=(
                                        sanitized_collection
                                    ),
                                    sanitized_embedding_name=(sanitized_embedding_name),
                                    dimensions=embedding_dimensions[
                                        sanitized_embedding_name
                                    ],
                                    similarity_metric=embedding_similarity[
                                        sanitized_embedding_name
                                    ],
                                ),
                            )
                        )

    async def add_edges(
        self,
        *,
        relation: str,
        source_collection: str,
        target_collection: str,
        edges: Iterable[Edge],
    ) -> None:
        """Add edges between collections, creating labels and indexes as needed."""
        async with self._tracker("add_edges"):
            await self._ensure_graph_initialized()

            edges = list(edges)
            if not edges:
                return

            sanitized_relation = sanitize_identifier(relation)
            sanitized_source_collection = sanitize_identifier(source_collection)
            sanitized_target_collection = sanitize_identifier(target_collection)

            # Labels must exist before any MATCH/count references them.
            await self._ensure_edge_label(sanitized_relation)
            await self._ensure_vertex_label(sanitized_source_collection)
            await self._ensure_vertex_label(sanitized_target_collection)

            if relation not in self._relation_edge_counts:
                self._relation_edge_counts[relation] = await self._count_edges(
                    relation,
                )

            embedding_dimensions: dict[str, int] = {}
            embedding_similarity: dict[str, SimilarityMetric] = {}
            sanitized_embedding_names: set[str] = set()

            async with self._engine.begin() as conn:
                for edge in edges:
                    sanitized_properties = _sanitize_properties(
                        {
                            mangle_property_name(key): value
                            for key, value in edge.properties.items()
                        }
                    )
                    create_props: dict[str, Any] = {
                        "uid": str(edge.uid),
                        **sanitized_properties,
                    }
                    params_json = encode_agtype_params(
                        {
                            "source_uid": str(edge.source_uid),
                            "target_uid": str(edge.target_uid),
                            "props": create_props,
                        }
                    )
                    sql = build_cypher_call(
                        self._graph_name,
                        (
                            f"MATCH (source:{sanitized_source_collection}"
                            " {uid: $source_uid}),"
                            f" (target:{sanitized_target_collection}"
                            " {uid: $target_uid})"
                            " WITH source, target, $props AS p"
                            f" CREATE (source)-[r:{sanitized_relation}]->(target)"
                            " SET r += p RETURN id(r) AS id"
                        ),
                        returns=("id",),
                        has_params=True,
                    )
                    result = await conn.exec_driver_sql(sql, (params_json,))
                    row = result.first()
                    if row is None:
                        # Source or target vertex did not exist; silently skip
                        # to mirror the Neo4j behavior, which also yields no
                        # relationship in that case.
                        continue
                    edge_id = int(parse_agtype(row[0]))

                    for embedding_name, (
                        embedding,
                        similarity_metric,
                    ) in edge.embeddings.items():
                        sanitized_embedding_name = sanitize_identifier(
                            mangle_embedding_name(embedding_name),
                        )
                        sanitized_embedding_names.add(sanitized_embedding_name)
                        embedding_dimensions[sanitized_embedding_name] = len(embedding)
                        embedding_similarity[sanitized_embedding_name] = (
                            similarity_metric
                        )

                        table_name = self._vector_table_name(
                            EntityType.EDGE,
                            sanitized_relation,
                            sanitized_embedding_name,
                        )
                        await self._ensure_vector_table(
                            table_name,
                            entity_type=EntityType.EDGE,
                            sanitized_collection_or_relation=sanitized_relation,
                            sanitized_embedding_name=sanitized_embedding_name,
                            dimensions=len(embedding),
                            uid_column="edge_uid",
                            conn=conn,
                        )
                        await conn.exec_driver_sql(
                            (
                                f'INSERT INTO public."{table_name}"'
                                "  (edge_uid, edge_id, similarity_metric, embedding)"
                                " VALUES ($1, $2, $3, $4::vector)"
                                " ON CONFLICT (edge_uid) DO UPDATE SET"
                                "  edge_id = EXCLUDED.edge_id,"
                                "  similarity_metric = EXCLUDED.similarity_metric,"
                                "  embedding = EXCLUDED.embedding"
                            ),
                            (
                                str(edge.uid),
                                edge_id,
                                similarity_metric.value,
                                _vector_literal(embedding),
                            ),
                        )

            self._relation_edge_counts[relation] += len(edges)

            if (
                self._relation_edge_counts[relation]
                >= self._range_index_creation_threshold
            ):
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
                        self._index_name(
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
                                    sanitized_collection_or_relation=(
                                        sanitized_relation
                                    ),
                                    sanitized_embedding_name=(sanitized_embedding_name),
                                    dimensions=embedding_dimensions[
                                        sanitized_embedding_name
                                    ],
                                    similarity_metric=embedding_similarity[
                                        sanitized_embedding_name
                                    ],
                                ),
                            )
                        )

    # ------------------------------------------------------------------
    # Reads
    # ------------------------------------------------------------------

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
        """Search nodes by vector similarity with optional property filters.

        Delegates to the pgvector HNSW side table to produce a candidate list
        of vertex uids, then hydrates the matching vertices via AGE. When a
        property filter is provided, candidates are over-fetched and filtered
        by re-reading the hydrated vertices; if the post-filter set is too
        small we fall back to an exact scan that joins the side table directly.
        """
        async with self._tracker("search_similar_nodes"):
            await self._ensure_graph_initialized()

            sanitized_collection = sanitize_identifier(collection)
            sanitized_embedding_name = sanitize_identifier(
                mangle_embedding_name(embedding_name),
            )
            table_name = self._vector_table_name(
                EntityType.NODE,
                sanitized_collection,
                sanitized_embedding_name,
            )
            if not await self._vector_table_exists(table_name):
                return []

            operator = _pgvector_distance_operator(similarity_metric)
            effective_limit = limit

            uids: list[str] = []
            if not self._force_exact_similarity_search:
                candidate_limit = (
                    effective_limit
                    if property_filter is None or effective_limit is None
                    else effective_limit * self._filtered_similarity_search_fudge_factor
                )
                async with self._engine.connect() as conn:
                    sql = (
                        f'SELECT node_uid FROM public."{table_name}"'
                        f" ORDER BY embedding {operator} $1::vector"
                        + (" LIMIT $2" if candidate_limit is not None else "")
                    )
                    params: tuple[Any, ...] = (
                        (_vector_literal(query_embedding), candidate_limit)
                        if candidate_limit is not None
                        else (_vector_literal(query_embedding),)
                    )
                    result = await conn.exec_driver_sql(sql, params)
                    uids = [row[0] for row in result]

                hydrated = await self._hydrate_nodes_by_uid(
                    sanitized_collection,
                    uids,
                    property_filter=property_filter,
                    preserve_order=True,
                )
                if effective_limit is not None:
                    hydrated = hydrated[:effective_limit]

                if (
                    property_filter is not None
                    and effective_limit is not None
                    and len(hydrated)
                    < effective_limit * self._exact_similarity_search_fallback_threshold
                ):
                    hydrated = await self._exact_similar_nodes(
                        sanitized_collection=sanitized_collection,
                        table_name=table_name,
                        query_embedding=query_embedding,
                        similarity_metric=similarity_metric,
                        limit=effective_limit,
                        property_filter=property_filter,
                    )
                return hydrated

            return await self._exact_similar_nodes(
                sanitized_collection=sanitized_collection,
                table_name=table_name,
                query_embedding=query_embedding,
                similarity_metric=similarity_metric,
                limit=effective_limit,
                property_filter=property_filter,
            )

    async def _exact_similar_nodes(
        self,
        *,
        sanitized_collection: str,
        table_name: str,
        query_embedding: list[float],
        similarity_metric: SimilarityMetric,
        limit: int | None,
        property_filter: FilterExpr | None,
    ) -> list[Node]:
        operator = _pgvector_distance_operator(similarity_metric)
        async with self._engine.connect() as conn:
            sql = (
                f'SELECT node_uid FROM public."{table_name}"'
                f" ORDER BY embedding {operator} $1::vector"
            )
            result = await conn.exec_driver_sql(
                sql,
                (_vector_literal(query_embedding),),
            )
            ordered_uids = [row[0] for row in result]

        hydrated = await self._hydrate_nodes_by_uid(
            sanitized_collection,
            ordered_uids,
            property_filter=property_filter,
            preserve_order=True,
        )
        if limit is not None:
            hydrated = hydrated[:limit]
        return hydrated

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
        async with self._tracker("search_related_nodes"):
            await self._ensure_graph_initialized()

            if not (find_sources or find_targets):
                return []

            sanitized_this_collection = sanitize_identifier(this_collection)
            sanitized_other_collection = sanitize_identifier(other_collection)
            sanitized_relation = sanitize_identifier(relation)

            # If any of the required labels has never been written to, AGE
            # raises on MATCH. Short-circuit with an empty result instead.
            if not (
                await self._label_exists(sanitized_this_collection)
                and await self._label_exists(sanitized_other_collection)
                and await self._label_exists(sanitized_relation)
            ):
                return []

            edge_filter_cypher, edge_filter_params = _render_filter_expr(
                "r", edge_property_filter, prefix="edge_"
            )
            node_filter_cypher, node_filter_params = _render_filter_expr(
                "n", node_property_filter, prefix="node_"
            )

            left = "-" if find_targets else "<-"
            right = "-" if find_sources else "->"
            match_clause = (
                f"MATCH (m:{sanitized_this_collection} {{uid: $this_node_uid}})"
                f"{left}[r:{sanitized_relation}]{right}"
                f"(n:{sanitized_other_collection})"
            )

            cypher = (
                f"{match_clause} "
                f"WHERE {edge_filter_cypher} AND {node_filter_cypher} "
                "RETURN DISTINCT n" + (" LIMIT $limit" if limit is not None else "")
            )

            params_map: dict[str, Any] = {
                "this_node_uid": str(this_node_uid),
                **edge_filter_params,
                **node_filter_params,
            }
            if limit is not None:
                params_map["limit"] = limit

            sql = build_cypher_call(
                self._graph_name,
                cypher,
                returns=("n",),
                has_params=True,
            )
            params_json = encode_agtype_params(params_map)
            async with self._engine.connect() as conn:
                result = await conn.exec_driver_sql(sql, (params_json,))
                raw_values = [row[0] for row in result]

            nodes = _nodes_from_agtype_rows(raw_values)
            await self._attach_node_embeddings(sanitized_other_collection, nodes)
            return nodes

    async def search_directional_nodes(
        self,
        *,
        collection: str,
        by_properties: Iterable[str],
        starting_at: Iterable[OrderedValue | str | None],
        order_ascending: Iterable[bool],
        include_equal_start: bool = False,
        limit: int | None = 1,
        property_filter: FilterExpr | None = None,
    ) -> list[Node]:
        """Find nodes ordered by property values in a chosen direction."""
        async with self._tracker("search_directional_nodes"):
            await self._ensure_graph_initialized()

            by_properties = list(by_properties)
            starting_at = list(starting_at)
            order_ascending = list(order_ascending)

            if not (len(by_properties) == len(starting_at) == len(order_ascending) > 0):
                raise ValueError(
                    "Lengths of by_properties, starting_at, and "
                    "order_ascending must be equal and greater than 0.",
                )

            sanitized_collection = sanitize_identifier(collection)
            if not await self._label_exists(sanitized_collection):
                return []
            sanitized_by_properties = [
                sanitize_identifier(mangle_property_name(by_property))
                for by_property in by_properties
            ]

            params_map: dict[str, Any] = {}
            starting_param_names: list[str] = []
            for index, starting_value in enumerate(starting_at):
                name = f"start_{index}"
                starting_param_names.append(name)
                if starting_value is None:
                    params_map[name] = None
                else:
                    params_map[name] = age_value_from_python(starting_value)

            lexicographic = _lexicographic_relational_cypher(
                "n",
                sanitized_by_properties,
                starting_at,
                starting_param_names,
                order_ascending,
            )

            equal_clause = ""
            if include_equal_start:
                equal_parts = [
                    render_comparison(
                        f"n.{sanitized_by_property}",
                        "=",
                        f"${starting_param_names[index]}",
                        starting_value,
                    )
                    for index, sanitized_by_property in enumerate(
                        sanitized_by_properties,
                    )
                    if (starting_value := starting_at[index]) is not None
                ]
                equal_clause = " OR (" + (" AND ".join(equal_parts) or "TRUE") + ")"

            filter_cypher, filter_params = _render_filter_expr(
                "n", property_filter, prefix="pf_"
            )
            params_map.update(filter_params)

            order_by = ", ".join(
                f"n.{sanitized_by_property} {
                    'ASC' if order_ascending[index] else 'DESC'
                }"
                for index, sanitized_by_property in enumerate(sanitized_by_properties)
            )

            cypher = (
                f"MATCH (n:{sanitized_collection}) "
                f"WHERE ({lexicographic}{equal_clause}) "
                f"AND {filter_cypher} "
                f"RETURN n ORDER BY {order_by}"
                + (" LIMIT $limit" if limit is not None else "")
            )

            if limit is not None:
                params_map["limit"] = limit

            sql = build_cypher_call(
                self._graph_name,
                cypher,
                returns=("n",),
                has_params=True,
            )
            params_json = encode_agtype_params(params_map)
            async with self._engine.connect() as conn:
                result = await conn.exec_driver_sql(sql, (params_json,))
                raw_values = [row[0] for row in result]

            nodes = _nodes_from_agtype_rows(raw_values)
            await self._attach_node_embeddings(sanitized_collection, nodes)
            return nodes

    async def search_matching_nodes(
        self,
        *,
        collection: str,
        limit: int | None = None,
        property_filter: FilterExpr | None = None,
    ) -> list[Node]:
        """Search nodes that match the provided property filters."""
        async with self._tracker("search_matching_nodes"):
            await self._ensure_graph_initialized()

            sanitized_collection = sanitize_identifier(collection)
            if not await self._label_exists(sanitized_collection):
                return []
            filter_cypher, filter_params = _render_filter_expr(
                "n", property_filter, prefix="pf_"
            )

            params_map: dict[str, Any] = dict(filter_params)
            cypher = (
                f"MATCH (n:{sanitized_collection}) "
                f"WHERE {filter_cypher} "
                "RETURN n" + (" LIMIT $limit" if limit is not None else "")
            )
            if limit is not None:
                params_map["limit"] = limit

            sql = build_cypher_call(
                self._graph_name,
                cypher,
                returns=("n",),
                has_params=True,
            )
            params_json = encode_agtype_params(params_map)
            async with self._engine.connect() as conn:
                result = await conn.exec_driver_sql(sql, (params_json,))
                raw_values = [row[0] for row in result]

            nodes = _nodes_from_agtype_rows(raw_values)
            await self._attach_node_embeddings(sanitized_collection, nodes)
            return nodes

    async def get_nodes(
        self,
        *,
        collection: str,
        node_uids: Iterable[str],
    ) -> list[Node]:
        """Retrieve nodes by uid from a specific collection."""
        async with self._tracker("get_nodes"):
            await self._ensure_graph_initialized()

            uids = [str(uid) for uid in node_uids]
            if not uids:
                return []

            sanitized_collection = sanitize_identifier(collection)
            if not await self._label_exists(sanitized_collection):
                return []
            return await self._hydrate_nodes_by_uid(
                sanitized_collection,
                uids,
                property_filter=None,
                preserve_order=False,
            )

    async def delete_nodes(
        self,
        *,
        collection: str,
        node_uids: Iterable[str],
    ) -> None:
        """Delete nodes by uid from a collection."""
        async with self._tracker("delete_nodes"):
            await self._ensure_graph_initialized()

            uids = [str(uid) for uid in node_uids]
            if not uids:
                return

            sanitized_collection = sanitize_identifier(collection)
            if not await self._label_exists(sanitized_collection):
                return
            params_json = encode_agtype_params({"uids": uids})
            # AGE's documented void-cypher pattern keeps the ``AS (v agtype)``
            # column declaration but omits ``RETURN`` from the body. The
            # wrapped query simply yields zero rows after the DELETE runs.
            sql = build_cypher_call(
                self._graph_name,
                (
                    f"MATCH (n:{sanitized_collection})"
                    " WHERE n.uid IN $uids DETACH DELETE n"
                ),
                returns=("v",),
                has_params=True,
            )
            async with self._engine.begin() as conn:
                await conn.exec_driver_sql(sql, (params_json,))
                # Clean up vector side tables that key by node_uid for this
                # collection. The registry resolves hashed table names back
                # to their (collection, embedding_name) origin.
                entries = await self._list_registered_side_tables(
                    conn,
                    entity_type=EntityType.NODE,
                    sanitized_collection_or_relation=sanitized_collection,
                )
                for table, _ in entries:
                    await conn.exec_driver_sql(
                        f'DELETE FROM public."{table}"'
                        " WHERE node_uid = ANY($1::text[])",
                        (uids,),
                    )

    async def delete_all_data(self) -> None:
        """Delete all nodes, relationships, and side-table contents."""
        await self._ensure_graph_initialized()
        async with self._engine.begin() as conn:
            sql = build_cypher_call(
                self._graph_name,
                "MATCH (n) DETACH DELETE n",
                returns=("v",),
                has_params=False,
            )
            await conn.exec_driver_sql(sql)

            entries = await self._list_registered_side_tables(conn)
            for table, _ in entries:
                await conn.exec_driver_sql(f'TRUNCATE TABLE public."{table}"')

    # ------------------------------------------------------------------
    # Internal helpers: counting, hydration, label/table management
    # ------------------------------------------------------------------

    async def _hydrate_nodes_by_uid(
        self,
        sanitized_collection: str,
        uids: list[str],
        *,
        property_filter: FilterExpr | None,
        preserve_order: bool,
    ) -> list[Node]:
        if not uids:
            return []

        if not await self._label_exists(sanitized_collection):
            return []

        filter_cypher, filter_params = _render_filter_expr(
            "n", property_filter, prefix="pf_"
        )

        cypher = (
            f"MATCH (n:{sanitized_collection}) "
            "WHERE n.uid IN $uids "
            f"AND {filter_cypher} "
            "RETURN n"
        )
        params_map: dict[str, Any] = {"uids": uids, **filter_params}
        sql = build_cypher_call(
            self._graph_name,
            cypher,
            returns=("n",),
            has_params=True,
        )
        params_json = encode_agtype_params(params_map)
        async with self._engine.connect() as conn:
            result = await conn.exec_driver_sql(sql, (params_json,))
            raw_values = [row[0] for row in result]

        hydrated = _nodes_from_agtype_rows(raw_values)
        await self._attach_node_embeddings(sanitized_collection, hydrated)
        if not preserve_order:
            return hydrated
        index_by_uid = {uid: position for position, uid in enumerate(uids)}
        hydrated.sort(key=lambda node: index_by_uid.get(node.uid, len(index_by_uid)))
        return hydrated

    async def _attach_node_embeddings(
        self,
        sanitized_collection: str,
        nodes: list[Node],
    ) -> None:
        """Populate ``Node.embeddings`` from the pgvector side tables.

        Embeddings live outside the AGE vertex (in per-(collection,
        embedding_name) tables), so an extra join-by-uid is required to make
        round-tripped nodes equal to the originals supplied to ``add_nodes``.
        """
        if not nodes:
            return
        node_uids = [node.uid for node in nodes]
        nodes_by_uid = {node.uid: node for node in nodes}
        async with self._engine.connect() as conn:
            entries = await self._list_registered_side_tables(
                conn,
                entity_type=EntityType.NODE,
                sanitized_collection_or_relation=sanitized_collection,
            )
            for table, sanitized_embedding_name in entries:
                embedding_name = _demangle_embedding_from_sanitized(
                    sanitized_embedding_name
                )
                if embedding_name is None:
                    # Registry rows are always written through
                    # ``_register_side_table`` with a freshly-sanitized name,
                    # so a failed decode means the registry is corrupt. Log
                    # and skip rather than crashing hydration.
                    logger.warning(
                        "Side-table registry entry %r did not decode to an"
                        " embedding name; skipping",
                        sanitized_embedding_name,
                    )
                    continue
                result = await conn.exec_driver_sql(
                    (
                        "SELECT node_uid, similarity_metric, embedding::text"
                        f' FROM public."{table}"'
                        " WHERE node_uid = ANY($1::text[])"
                    ),
                    (node_uids,),
                )
                for row in result:
                    uid_value = row[0]
                    metric_value = row[1]
                    embedding_text = row[2]
                    # The SQL filter restricts rows to uids we just passed
                    # in, so ``nodes_by_uid`` always resolves.
                    node = nodes_by_uid[uid_value]
                    node.embeddings[embedding_name] = (
                        _parse_pgvector_text(embedding_text),
                        SimilarityMetric(metric_value),
                    )

    async def _count_nodes(self, collection: str) -> int:
        async with self._tracker("count_nodes"):
            await self._ensure_graph_initialized()
            sanitized_collection = sanitize_identifier(collection)
            cypher = f"MATCH (n:{sanitized_collection}) RETURN count(n) AS node_count"
            sql = build_cypher_call(
                self._graph_name,
                cypher,
                returns=("node_count",),
                has_params=False,
            )
            async with self._engine.connect() as conn:
                result = await conn.exec_driver_sql(sql)
                row = result.first()

            if row is None:
                return 0
            value = parse_agtype(row[0])
            return int(value) if value is not None else 0

    async def _count_edges(self, relation: str) -> int:
        async with self._tracker("count_edges"):
            await self._ensure_graph_initialized()
            sanitized_relation = sanitize_identifier(relation)
            cypher = (
                f"MATCH ()-[r:{sanitized_relation}]->() RETURN count(r) AS edge_count"
            )
            sql = build_cypher_call(
                self._graph_name,
                cypher,
                returns=("edge_count",),
                has_params=False,
            )
            async with self._engine.connect() as conn:
                result = await conn.exec_driver_sql(sql)
                row = result.first()

            if row is None:
                return 0
            value = parse_agtype(row[0])
            return int(value) if value is not None else 0

    async def _ensure_vertex_label(self, sanitized_label: str) -> None:
        await self._ensure_label(sanitized_label, kind="v")

    async def _ensure_edge_label(self, sanitized_label: str) -> None:
        await self._ensure_label(sanitized_label, kind="e")

    async def _label_exists(self, sanitized_label: str) -> bool:
        """Return True if the AGE label has a backing table in the graph schema.

        Reads use this to bail out early with empty results when a label has
        never been written to — recent AGE releases raise "label does not
        exist" on ``MATCH (n:Label)`` for unknown labels, so we cannot rely on
        Cypher returning zero rows.
        """
        await self._ensure_graph_initialized()
        async with self._engine.connect() as conn:
            result = await conn.exec_driver_sql(
                "SELECT 1 FROM pg_tables WHERE schemaname = $1 AND tablename = $2",
                (self._graph_name, sanitized_label),
            )
            return result.first() is not None

    async def _ensure_label(self, sanitized_label: str, *, kind: str) -> None:
        """Create an AGE vertex or edge label if it does not already exist.

        AGE creates each label as a table in the graph's schema, so we detect
        existence via ``pg_tables`` rather than relying on ``ag_catalog``
        internals (which vary across AGE versions). The actual creation uses
        AGE's built-in ``create_vlabel`` / ``create_elabel`` helpers.
        """
        entity_type = EntityType.NODE if kind == "v" else EntityType.EDGE
        key = (entity_type, sanitized_label)
        if key in self._ensured_labels:
            return
        async with self._ensured_labels_lock:
            if key in self._ensured_labels:
                return
            await self._ensure_graph_initialized()
            async with self._engine.begin() as conn:
                exists = (
                    await conn.exec_driver_sql(
                        "SELECT 1 FROM pg_tables "
                        "WHERE schemaname = $1 AND tablename = $2",
                        (self._graph_name, sanitized_label),
                    )
                ).first()
                if exists is None:
                    creator = "create_vlabel" if kind == "v" else "create_elabel"
                    await conn.exec_driver_sql(
                        f"SELECT {creator}($1, $2)",
                        (self._graph_name, sanitized_label),
                    )
            self._ensured_labels.add(key)

    def _registry_table_name(self) -> str:
        return f"{self._graph_name}{_SIDE_TABLE_REGISTRY_SUFFIX}"

    def _vector_table_name(
        self,
        entity_type: EntityType,
        sanitized_collection_or_relation: str,
        sanitized_embedding_name: str,
    ) -> str:
        # Postgres caps identifiers at 63 bytes, so we cannot embed the
        # full sanitized collection and embedding names directly. Hash the
        # tuple into a short digest and resolve back through the registry
        # table when we need to enumerate side tables or recover the
        # embedding name for hydration.
        kind = "node" if entity_type is EntityType.NODE else "edge"
        key = f"{kind}|{sanitized_collection_or_relation}|{sanitized_embedding_name}"
        digest = hashlib.sha1(key.encode("utf-8")).hexdigest()[:16]
        return f"{self._graph_name}_emb_{digest}"

    async def _ensure_registry_table(self, conn: AsyncConnection) -> None:
        # Called from inside ``_ensure_vector_table`` which already dedupes
        # per-table creation, so the idempotent CREATE runs at most once per
        # new side table — cheap and tolerant of out-of-band registry drops
        # (e.g. test cleanup fixtures).
        registry = self._registry_table_name()
        await conn.exec_driver_sql(
            f'CREATE TABLE IF NOT EXISTS public."{registry}" ('
            "  table_name TEXT PRIMARY KEY,"
            "  kind TEXT NOT NULL,"
            "  sanitized_collection_or_relation TEXT NOT NULL,"
            "  sanitized_embedding_name TEXT NOT NULL,"
            "  UNIQUE (kind, sanitized_collection_or_relation,"
            "          sanitized_embedding_name)"
            ")"
        )

    async def _register_side_table(
        self,
        conn: AsyncConnection,
        *,
        table_name: str,
        entity_type: EntityType,
        sanitized_collection_or_relation: str,
        sanitized_embedding_name: str,
    ) -> None:
        kind = "node" if entity_type is EntityType.NODE else "edge"
        registry = self._registry_table_name()
        await conn.exec_driver_sql(
            f'INSERT INTO public."{registry}"'
            " (table_name, kind, sanitized_collection_or_relation,"
            "  sanitized_embedding_name)"
            " VALUES ($1, $2, $3, $4)"
            " ON CONFLICT (table_name) DO NOTHING",
            (
                table_name,
                kind,
                sanitized_collection_or_relation,
                sanitized_embedding_name,
            ),
        )

    async def _list_registered_side_tables(
        self,
        conn: AsyncConnection,
        *,
        entity_type: EntityType | None = None,
        sanitized_collection_or_relation: str | None = None,
    ) -> list[tuple[str, str]]:
        """Return ``(table_name, sanitized_embedding_name)`` pairs from the registry."""
        registry = self._registry_table_name()
        if not await self._table_exists(conn, registry):
            return []
        conditions: list[str] = []
        params: list[Any] = []
        if entity_type is not None:
            kind = "node" if entity_type is EntityType.NODE else "edge"
            params.append(kind)
            conditions.append(f"kind = ${len(params)}")
        if sanitized_collection_or_relation is not None:
            params.append(sanitized_collection_or_relation)
            conditions.append(f"sanitized_collection_or_relation = ${len(params)}")
        where = (" WHERE " + " AND ".join(conditions)) if conditions else ""
        result = await conn.exec_driver_sql(
            "SELECT table_name, sanitized_embedding_name"
            f' FROM public."{registry}"{where}',
            tuple(params),
        )
        return [(row[0], row[1]) for row in result]

    async def _ensure_vector_table(
        self,
        table_name: str,
        *,
        entity_type: EntityType,
        sanitized_collection_or_relation: str,
        sanitized_embedding_name: str,
        dimensions: int,
        uid_column: str,
        conn: AsyncConnection,
    ) -> None:
        if not (1 <= dimensions <= _MAX_VECTOR_DIMENSIONS):
            raise ValueError(
                f"dimensions must be between 1 and {_MAX_VECTOR_DIMENSIONS}"
            )
        if table_name in self._ensured_vector_tables:
            return
        async with self._ensured_vector_tables_lock:
            if table_name in self._ensured_vector_tables:
                return
            await self._ensure_registry_table(conn)
            id_column = "vertex_id" if uid_column == "node_uid" else "edge_id"
            await conn.exec_driver_sql(
                f'CREATE TABLE IF NOT EXISTS public."{table_name}" ('
                f"  {uid_column} TEXT PRIMARY KEY,"
                f"  {id_column} BIGINT NOT NULL,"
                "  similarity_metric TEXT NOT NULL,"
                f"  embedding vector({dimensions}) NOT NULL"
                ")"
            )
            await self._register_side_table(
                conn,
                table_name=table_name,
                entity_type=entity_type,
                sanitized_collection_or_relation=(sanitized_collection_or_relation),
                sanitized_embedding_name=sanitized_embedding_name,
            )
            self._ensured_vector_tables.add(table_name)

    async def _table_exists(self, conn: AsyncConnection, table_name: str) -> bool:
        result = await conn.exec_driver_sql(
            "SELECT 1 FROM pg_tables WHERE schemaname = 'public' AND tablename = $1",
            (table_name,),
        )
        return result.first() is not None

    async def _vector_table_exists(self, table_name: str) -> bool:
        async with self._engine.connect() as conn:
            result = await conn.exec_driver_sql(
                "SELECT 1 FROM pg_tables "
                "WHERE schemaname = 'public' AND tablename = $1",
                (table_name,),
            )
            return result.first() is not None

    # ------------------------------------------------------------------
    # Index management
    # ------------------------------------------------------------------

    async def _populate_index_state_cache(self) -> None:
        async with self._tracker("populate_index_state_cache"):
            if self._index_state_cache:
                return
            async with self._populate_index_state_cache_lock:
                if self._index_state_cache:
                    return
                await self._ensure_graph_initialized()
                # Range indexes live on AGE's per-label tables inside the
                # graph schema; vector HNSW indexes live on the pgvector
                # side tables inside ``public``. Cover both so the cache
                # matches any index name we might look up later.
                async with self._engine.connect() as conn:
                    result = await conn.exec_driver_sql(
                        "SELECT indexname FROM pg_indexes "
                        "WHERE schemaname = ANY($1::text[])",
                        ([self._graph_name, "public"],),
                    )
                    rows = [row[0] for row in result]
                self._index_state_cache.update(
                    dict.fromkeys(rows, AgeVectorGraphStore.CacheIndexState.ONLINE)
                )

    async def _create_initial_indexes_if_not_exist(
        self,
        entity_type: EntityType,
        sanitized_collection_or_relation: str,
    ) -> None:
        async with self._tracker("create_initial_indexes_if_not_exist"):
            tasks = [
                self._create_range_index_if_not_exists(
                    entity_type=entity_type,
                    sanitized_collection_or_relation=(sanitized_collection_or_relation),
                    sanitized_property_names="uid",
                ),
            ]
            tasks += [
                self._create_range_index_if_not_exists(
                    entity_type=entity_type,
                    sanitized_collection_or_relation=(sanitized_collection_or_relation),
                    sanitized_property_names=[
                        sanitize_identifier(mangle_property_name(property_name))
                        for property_name in property_name_hierarchy
                    ],
                )
                for range_index_hierarchy in self._range_index_hierarchies
                for property_name_hierarchy in [
                    range_index_hierarchy[: i + 1]
                    for i in range(len(range_index_hierarchy))
                ]
            ]
            await asyncio.gather(*tasks)

    async def _create_range_index_if_not_exists(
        self,
        entity_type: EntityType,
        sanitized_collection_or_relation: str,
        sanitized_property_names: str | Iterable[str],
    ) -> None:
        """Create a B-tree index on the AGE-underlying table for this label."""
        async with self._tracker("create_range_index_if_not_exists"):
            if isinstance(sanitized_property_names, str):
                sanitized_property_names = [sanitized_property_names]
            sanitized_property_names = list(sanitized_property_names)
            if not sanitized_property_names:
                raise ValueError("sanitized_property_names must be nonempty")

            await self._populate_index_state_cache()
            index_name = self._index_name(
                entity_type,
                sanitized_collection_or_relation,
                sanitized_property_names,
            )
            cached_state = self._index_state_cache.get(index_name)
            if cached_state is AgeVectorGraphStore.CacheIndexState.CREATING:
                await self._await_create_index_if_not_exists(index_name, None)
                return
            if cached_state is AgeVectorGraphStore.CacheIndexState.ONLINE:
                return
            self._index_state_cache[index_name] = (
                AgeVectorGraphStore.CacheIndexState.CREATING
            )

            expressions = ", ".join(
                f"(properties->>'{name}')" for name in sanitized_property_names
            )
            sql = (
                f'CREATE INDEX IF NOT EXISTS "{index_name}" '
                f'ON "{self._graph_name}"."{sanitized_collection_or_relation}"'
                f" ({expressions})"
            )

            async def _run() -> None:
                # AGE stores properties as ``agtype`` rather than ``jsonb``;
                # property-expression indexes via ``->>`` are best-effort and
                # may not be supported on all AGE versions. Treat failure as a
                # missed optimization rather than a hard error.
                try:
                    async with self._engine.begin() as conn:
                        await conn.exec_driver_sql(sql)
                except SQLAlchemyError as exc:
                    logger.warning(
                        "Skipping range index %s for %s: %s",
                        index_name,
                        sanitized_collection_or_relation,
                        exc,
                    )

            await self._await_create_index_if_not_exists(index_name, _run())

    async def _create_vector_index_if_not_exists(
        self,
        entity_type: EntityType,
        sanitized_collection_or_relation: str,
        sanitized_embedding_name: str,
        dimensions: int,
        similarity_metric: SimilarityMetric = SimilarityMetric.COSINE,
    ) -> None:
        if not (1 <= dimensions <= _MAX_VECTOR_DIMENSIONS):
            raise ValueError(
                f"dimensions must be between 1 and {_MAX_VECTOR_DIMENSIONS}"
            )
        async with self._tracker("create_vector_index_if_not_exists"):
            await self._populate_index_state_cache()
            index_name = self._index_name(
                entity_type,
                sanitized_collection_or_relation,
                sanitized_embedding_name,
            )
            cached_state = self._index_state_cache.get(index_name)
            if cached_state is AgeVectorGraphStore.CacheIndexState.CREATING:
                await self._await_create_index_if_not_exists(index_name, None)
                return
            if cached_state is AgeVectorGraphStore.CacheIndexState.ONLINE:
                return
            self._index_state_cache[index_name] = (
                AgeVectorGraphStore.CacheIndexState.CREATING
            )

            opclass = _pgvector_hnsw_opclass(similarity_metric)
            table_name = self._vector_table_name(
                entity_type,
                sanitized_collection_or_relation,
                sanitized_embedding_name,
            )
            sql = (
                f'CREATE INDEX IF NOT EXISTS "{index_name}" '
                f'ON public."{table_name}" USING hnsw (embedding {opclass}) '
                f"WITH (m = {self._hnsw_m}, "
                f"ef_construction = {self._hnsw_ef_construction})"
            )

            async def _run() -> None:
                async with self._engine.begin() as conn:
                    await conn.exec_driver_sql(sql)

            await self._await_create_index_if_not_exists(index_name, _run())

    @async_locked
    async def _await_create_index_if_not_exists(
        self,
        index_name: str,
        create_index_awaitable: Awaitable | None,
    ) -> None:
        # ``@async_locked`` serializes calls for the same ``index_name``. A
        # None awaitable means another coroutine already owns the creation;
        # we just wait on the lock and observe the final state.
        if create_index_awaitable is not None:
            await create_index_awaitable
        self._index_state_cache[index_name] = AgeVectorGraphStore.CacheIndexState.ONLINE

    def _index_name(
        self,
        entity_type: EntityType,
        sanitized_collection_or_relation: str,
        sanitized_property_names: str | Iterable[str],
    ) -> str:
        if isinstance(sanitized_property_names, str):
            sanitized_property_names = [sanitized_property_names]
        sanitized_property_names_string = "_and_".join(
            f"{len(sanitized_property_name)}_{sanitized_property_name}"
            for sanitized_property_name in sanitized_property_names
        )
        # PostgreSQL identifiers max out at 63 bytes. We hash if too long.
        base = (
            f"{entity_type.value}_idx_"
            f"{len(sanitized_collection_or_relation)}_"
            f"{sanitized_collection_or_relation}_on_"
            f"{sanitized_property_names_string}"
        )
        if len(base) <= 63:
            return base
        digest = hashlib.sha1(base.encode("utf-8"), usedforsecurity=False).hexdigest()
        return f"mm_idx_{digest[:40]}"


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------


def _vector_literal(embedding: list[float]) -> str:
    """Render a list of floats as the pgvector text input format."""
    return "[" + ",".join(repr(float(value)) for value in embedding) + "]"


def _pgvector_distance_operator(metric: SimilarityMetric) -> str:
    if metric is SimilarityMetric.COSINE:
        return "<=>"
    if metric is SimilarityMetric.EUCLIDEAN:
        return "<->"
    if metric is SimilarityMetric.DOT:
        return "<#>"
    if metric is SimilarityMetric.MANHATTAN:
        return "<+>"
    return "<=>"


def _pgvector_hnsw_opclass(metric: SimilarityMetric) -> str:
    if metric is SimilarityMetric.COSINE:
        return "vector_cosine_ops"
    if metric is SimilarityMetric.EUCLIDEAN:
        return "vector_l2_ops"
    if metric is SimilarityMetric.DOT:
        return "vector_ip_ops"
    if metric is SimilarityMetric.MANHATTAN:
        return "vector_l1_ops"
    return "vector_cosine_ops"


def _sanitize_properties(
    properties: Mapping[str, PropertyValue] | None,
) -> dict[str, Any]:
    """Convert property keys/values into the shape stored inside AGE."""
    if properties is None:
        return {}
    return {
        sanitize_identifier(key): age_value_from_python(value)
        for key, value in properties.items()
    }


def _render_filter_expr(
    entity_alias: str,
    expr: FilterExpr | None,
    *,
    prefix: str,
) -> tuple[str, dict[str, Any]]:
    """Render a FilterExpr into Cypher and a parameter map.

    The caller chooses a unique ``prefix`` to avoid collisions when multiple
    filter expressions share the same parameter namespace (for example, edge
    and node filters in ``search_related_nodes``).
    """
    if expr is None:
        return "TRUE", {}

    params: dict[str, Any] = {}
    # Sequential counter gives deterministic, collision-free parameter names
    # within a single call; the caller's ``prefix`` keeps names distinct
    # across sibling filter expressions sharing a namespace (e.g. edge and
    # node filters in ``search_related_nodes``).
    counter = [0]

    def _next_name() -> str:
        counter[0] += 1
        return f"{prefix}{counter[0]}"

    def _walk(e: FilterExpr) -> str:
        if isinstance(e, FilterIsNull):
            field_ref = (
                f"{entity_alias}.{sanitize_identifier(mangle_property_name(e.field))}"
            )
            return f"{field_ref} IS NULL"
        if isinstance(e, FilterIn):
            field_ref = (
                f"{entity_alias}.{sanitize_identifier(mangle_property_name(e.field))}"
            )
            name = _next_name()
            params[name] = list(e.values)
            return f"{field_ref} IN ${name}"
        if isinstance(e, FilterComparison):
            field_ref = (
                f"{entity_alias}.{sanitize_identifier(mangle_property_name(e.field))}"
            )
            name = _next_name()
            params[name] = age_value_from_python(e.value)
            return render_comparison(
                left=field_ref,
                op=e.op,
                right=f"${name}",
                value=e.value,
            )
        if isinstance(e, FilterAnd):
            left = _walk(e.left)
            right = _walk(e.right)
            return f"({left}) AND ({right})"
        if isinstance(e, FilterOr):
            left = _walk(e.left)
            right = _walk(e.right)
            return f"({left}) OR ({right})"
        if isinstance(e, FilterNot):
            inner = _walk(e.expr)
            return f"NOT ({inner})"
        raise TypeError(f"Unsupported filter expression type: {type(e)!r}")

    return _walk(expr), params


def _lexicographic_relational_cypher(
    entity_alias: str,
    sanitized_by_properties: list[str],
    starting_at: list[OrderedValue | str | None],
    starting_param_names: list[str],
    order_ascending: list[bool],
) -> str:
    """Render the Cypher WHERE fragment for ``search_directional_nodes``.

    The shape matches the Neo4j backend's lexicographic comparison: for each
    tie-breaker level ``i``, require strict inequality on property ``i`` and
    equality on all previous properties, joined with OR across levels.
    """
    clauses: list[str] = []
    for index, sanitized_by_property in enumerate(sanitized_by_properties):
        starting_value = starting_at[index]
        if starting_value is None:
            parts = [
                f"{entity_alias}.{sanitized_by_property} IS NOT NULL",
            ]
        else:
            parts = [
                render_comparison(
                    f"{entity_alias}.{sanitized_by_property}",
                    ">" if order_ascending[index] else "<",
                    f"${starting_param_names[index]}",
                    starting_value,
                )
            ]
        for equal_index in range(index):
            equal_starting_value = starting_at[equal_index]
            equal_property = sanitized_by_properties[equal_index]
            if equal_starting_value is None:
                parts.append(f"{entity_alias}.{equal_property} IS NOT NULL")
            else:
                parts.append(
                    render_comparison(
                        f"{entity_alias}.{equal_property}",
                        "=",
                        f"${starting_param_names[equal_index]}",
                        equal_starting_value,
                    )
                )
        clauses.append("(" + " AND ".join(parts) + ")")
    return "(" + " OR ".join(clauses) + ")"


def _nodes_from_agtype_rows(
    raw_values: Iterable[str | bytes | None],
) -> list[Node]:
    """Decode a sequence of ``agtype`` row values into :class:`Node`s."""
    nodes: list[Node] = []
    for raw in raw_values:
        if raw is None:
            continue
        vertex = parse_agtype(raw)
        if not isinstance(vertex, AgeVertex):
            raise TypeError(f"expected AgeVertex after decoding, got {type(vertex)!r}")
        node_properties: dict[str, PropertyValue] = {}
        for sanitized_key, raw_value in vertex.properties.items():
            if sanitized_key == "uid":
                continue
            desanitized = desanitize_identifier(sanitized_key)
            if is_mangled_property_name(desanitized):
                name = demangle_property_name(desanitized)
                node_properties[name] = age_value_to_python(raw_value)
        uid = str(vertex.properties.get("uid", ""))
        nodes.append(
            Node(uid=uid, properties=node_properties),
        )
    return nodes


def _demangle_embedding_from_sanitized(sanitized_embedding_name: str) -> str | None:
    """Inverse of ``sanitize_identifier(mangle_embedding_name(name))``.

    Returns ``None`` if the input does not actually decode to a mangled
    embedding name (defensive guard against unrelated tables that happen to
    share a prefix on disk).
    """
    desanitized = desanitize_identifier(sanitized_embedding_name)
    if not is_mangled_embedding_name(desanitized):
        return None
    return demangle_embedding_name(desanitized)


def _parse_pgvector_text(text: str) -> list[float]:
    """Parse pgvector's text output ``[v1,v2,...]`` into a Python list."""
    inner = text.strip()
    if inner.startswith("[") and inner.endswith("]"):
        inner = inner[1:-1]
    if not inner:
        return []
    return [float(part) for part in inner.split(",")]


__all__ = [
    "DEFAULT_GRAPH_NAME",
    "AgeVectorGraphStore",
    "AgeVectorGraphStoreParams",
]
