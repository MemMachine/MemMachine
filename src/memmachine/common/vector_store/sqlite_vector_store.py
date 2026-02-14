"""SQLite-backed vector store implementation using sqlite-vec."""

import json
import logging
import sqlite3
import struct
import time
from collections.abc import Iterable, Mapping, Sequence
from datetime import datetime
from typing import cast
from uuid import UUID

import sqlite_vec
from pydantic import BaseModel, Field, InstanceOf
from sqlalchemy import event, text
from sqlalchemy.ext.asyncio import AsyncEngine

from memmachine.common.data_types import SimilarityMetric
from memmachine.common.filter.filter_parser import (
    And,
    Comparison,
    FilterExpr,
    In,
    IsNull,
    Not,
    Or,
)
from memmachine.common.metrics_factory import MetricsFactory

from .data_types import PropertyValue, QueryResult, Record
from .vector_store import Collection, VectorStore

logger = logging.getLogger(__name__)

_SUPPORTED_METRICS = {SimilarityMetric.COSINE, SimilarityMetric.EUCLIDEAN}

_METRIC_TO_VEC0: dict[SimilarityMetric, str] = {
    SimilarityMetric.COSINE: "cosine",
    SimilarityMetric.EUCLIDEAN: "L2",
}

_METRIC_TO_DISTANCE_FN: dict[SimilarityMetric, str] = {
    SimilarityMetric.COSINE: "vec_distance_cosine",
    SimilarityMetric.EUCLIDEAN: "vec_distance_L2",
}

_TYPE_NAME_MAP: dict[str, type[PropertyValue]] = {
    "bool": bool,
    "int": int,
    "float": float,
    "str": str,
    "datetime": datetime,
}

_TYPE_TO_NAME: dict[type[PropertyValue], str] = {
    v: k for k, v in _TYPE_NAME_MAP.items()
}

_BindParams = dict[str, str | int | float | bytes | None]


def _vec_table_name(dimensions: int, metric: SimilarityMetric) -> str:
    """Return the shared vec0 virtual table name for a (dimensions, metric) combo."""
    return f"_vs_vec_{dimensions}_{_METRIC_TO_VEC0[metric]}"


def _validate_metric(metric: SimilarityMetric) -> None:
    if metric not in _SUPPORTED_METRICS:
        msg = f"Similarity metric {metric.value!r} is not supported by sqlite-vec. Use COSINE or EUCLIDEAN."
        raise ValueError(msg)


def _serialize_vector(vector: list[float]) -> bytes:
    """Serialize a float vector to bytes for sqlite-vec."""
    return sqlite_vec.serialize_float32(vector)


def _deserialize_vector(blob: bytes) -> list[float]:
    """Deserialize bytes from sqlite-vec to a float vector."""
    n = len(blob) // 4
    return list(struct.unpack(f"{n}f", blob))


def _serialize_properties(
    properties: dict[str, PropertyValue] | None,
) -> str | None:
    """Serialize properties dict to JSON string."""
    if properties is None:
        return None

    def _convert(v: PropertyValue) -> PropertyValue | str:
        if isinstance(v, datetime):
            return v.isoformat()
        return v

    return json.dumps({k: _convert(v) for k, v in properties.items()})


def _deserialize_properties(
    raw: str | None,
    schema: dict[str, type[PropertyValue]] | None,
) -> dict[str, PropertyValue] | None:
    """Deserialize JSON string to properties dict, using schema for type coercion."""
    if raw is None:
        return None
    data: dict[str, PropertyValue] = json.loads(raw)
    if schema is None:
        return data
    result: dict[str, PropertyValue] = {}
    for k, v in data.items():
        expected_type = schema.get(k)
        if expected_type is datetime and isinstance(v, str):
            result[k] = datetime.fromisoformat(v)
        else:
            result[k] = v
    return result


def _serialize_schema(
    schema: Mapping[str, type[PropertyValue]] | None,
) -> str | None:
    """Serialize properties_schema to JSON."""
    if schema is None:
        return None
    return json.dumps({k: _TYPE_TO_NAME[v] for k, v in schema.items()})


def _deserialize_schema(
    raw: str | None,
) -> dict[str, type[PropertyValue]] | None:
    """Deserialize JSON to properties_schema."""
    if raw is None:
        return None
    data: dict[str, str] = json.loads(raw)
    return {k: _TYPE_NAME_MAP[v] for k, v in data.items()}


def _distance_to_similarity(distance: float, metric: SimilarityMetric) -> float:
    """Convert a distance value to a similarity score."""
    if metric == SimilarityMetric.COSINE:
        return 1.0 - distance
    # EUCLIDEAN
    return 1.0 / (1.0 + distance)


def _similarity_to_max_distance(
    threshold: float,
    metric: SimilarityMetric,
) -> float:
    """Convert a similarity threshold to a maximum distance for SQL filtering."""
    if metric == SimilarityMetric.COSINE:
        return 1.0 - threshold
    # EUCLIDEAN: similarity = 1/(1+d) => d = (1-t)/t
    if threshold <= 0:
        return float("inf")
    return (1.0 - threshold) / threshold


class _FilterCompiler:
    """Compiles FilterExpr trees into SQL WHERE clauses with bound parameters."""

    def __init__(self) -> None:
        self._param_counter = 0
        self.params: _BindParams = {}

    def _next_param(self) -> str:
        self._param_counter += 1
        return f"_fv{self._param_counter}"

    def compile(self, expr: FilterExpr) -> str:
        """Compile a FilterExpr into a SQL WHERE clause fragment."""
        if isinstance(expr, Comparison):
            return self._compile_comparison(expr)
        if isinstance(expr, In):
            return self._compile_in(expr)
        if isinstance(expr, IsNull):
            return self._compile_is_null(expr)
        if isinstance(expr, And):
            left = self.compile(expr.left)
            right = self.compile(expr.right)
            return f"({left}) AND ({right})"
        if isinstance(expr, Or):
            left = self.compile(expr.left)
            right = self.compile(expr.right)
            return f"({left}) OR ({right})"
        if isinstance(expr, Not):
            inner = self.compile(expr.expr)
            return f"NOT ({inner})"
        msg = f"Unsupported filter expression type: {type(expr)}"
        raise TypeError(msg)

    def _compile_comparison(self, comp: Comparison) -> str:
        json_path = f"json_extract(r.properties, '$.{comp.field}')"
        pname = self._next_param()
        self.params[pname] = self._bind_value(comp.value)
        return f"{json_path} {comp.op} :{pname}"

    def _compile_in(self, expr: In) -> str:
        json_path = f"json_extract(r.properties, '$.{expr.field}')"
        placeholders = []
        for v in expr.values:
            pname = self._next_param()
            self.params[pname] = self._bind_value(v)
            placeholders.append(f":{pname}")
        return f"{json_path} IN ({', '.join(placeholders)})"

    def _compile_is_null(self, expr: IsNull) -> str:
        json_path = f"json_extract(r.properties, '$.{expr.field}')"
        return f"{json_path} IS NULL"

    @staticmethod
    def _bind_value(
        value: PropertyValue,
    ) -> str | int | float | None:
        """Convert a filter value to a SQLite-compatible bind value."""
        if isinstance(value, bool):
            return 1 if value else 0
        if isinstance(value, datetime):
            return value.isoformat()
        if isinstance(value, (int, float, str)) or value is None:
            return value
        msg = f"Unsupported filter value type: {type(value)}"
        raise TypeError(msg)


class _CollectionConfig:
    """Holds resolved collection configuration."""

    __slots__ = ("dimensions", "metric", "schema", "vec_table")

    def __init__(
        self,
        dimensions: int,
        metric: SimilarityMetric,
        schema: dict[str, type[PropertyValue]] | None,
    ) -> None:
        self.dimensions = dimensions
        self.metric = metric
        self.schema = schema
        self.vec_table = _vec_table_name(dimensions, metric)


class SQLiteVectorStoreParams(BaseModel):
    """Parameters for SQLiteVectorStore.

    Attributes:
        engine: SQLAlchemy async engine (sqlite+aiosqlite).
        metrics_factory: Optional MetricsFactory for collecting usage metrics.
        user_metrics_labels: Labels to attach to the collected metrics.

    """

    engine: InstanceOf[AsyncEngine] = Field(
        ...,
        description="SQLAlchemy async engine (sqlite+aiosqlite)",
    )
    metrics_factory: InstanceOf[MetricsFactory] | None = Field(
        None,
        description="An instance of MetricsFactory for collecting usage metrics",
    )
    user_metrics_labels: dict[str, str] = Field(
        default_factory=dict,
        description="Labels to attach to the collected metrics",
    )


class SQLiteVectorStore(VectorStore):
    """SQLite-backed vector store using sqlite-vec for native vector search."""

    def __init__(self, params: SQLiteVectorStoreParams) -> None:
        """Initialize the vector store with the provided parameters."""
        super().__init__()
        self._engine = params.engine

        # Metrics setup
        self.upsert_calls_counter: MetricsFactory.Counter | None = None
        self.upsert_latency_summary: MetricsFactory.Summary | None = None
        self.query_calls_counter: MetricsFactory.Counter | None = None
        self.query_latency_summary: MetricsFactory.Summary | None = None
        self.get_calls_counter: MetricsFactory.Counter | None = None
        self.get_latency_summary: MetricsFactory.Summary | None = None
        self.delete_calls_counter: MetricsFactory.Counter | None = None
        self.delete_latency_summary: MetricsFactory.Summary | None = None
        self.create_collection_calls_counter: MetricsFactory.Counter | None = None
        self.create_collection_latency_summary: MetricsFactory.Summary | None = None
        self.delete_collection_calls_counter: MetricsFactory.Counter | None = None
        self.delete_collection_latency_summary: MetricsFactory.Summary | None = None

        self._should_collect_metrics = False
        metrics_factory = params.metrics_factory
        if metrics_factory is not None:
            self._should_collect_metrics = True
            self._user_metrics_labels = params.user_metrics_labels
            label_names = self._user_metrics_labels.keys()

            self.upsert_calls_counter = metrics_factory.get_counter(
                "vector_store_sqlite_upsert_calls",
                "Number of calls to upsert in SQLiteVectorStore",
                label_names=label_names,
            )
            self.upsert_latency_summary = metrics_factory.get_summary(
                "vector_store_sqlite_upsert_latency_seconds",
                "Latency in seconds for upsert in SQLiteVectorStore",
                label_names=label_names,
            )
            self.query_calls_counter = metrics_factory.get_counter(
                "vector_store_sqlite_query_calls",
                "Number of calls to query in SQLiteVectorStore",
                label_names=label_names,
            )
            self.query_latency_summary = metrics_factory.get_summary(
                "vector_store_sqlite_query_latency_seconds",
                "Latency in seconds for query in SQLiteVectorStore",
                label_names=label_names,
            )
            self.get_calls_counter = metrics_factory.get_counter(
                "vector_store_sqlite_get_calls",
                "Number of calls to get in SQLiteVectorStore",
                label_names=label_names,
            )
            self.get_latency_summary = metrics_factory.get_summary(
                "vector_store_sqlite_get_latency_seconds",
                "Latency in seconds for get in SQLiteVectorStore",
                label_names=label_names,
            )
            self.delete_calls_counter = metrics_factory.get_counter(
                "vector_store_sqlite_delete_calls",
                "Number of calls to delete in SQLiteVectorStore",
                label_names=label_names,
            )
            self.delete_latency_summary = metrics_factory.get_summary(
                "vector_store_sqlite_delete_latency_seconds",
                "Latency in seconds for delete in SQLiteVectorStore",
                label_names=label_names,
            )
            self.create_collection_calls_counter = metrics_factory.get_counter(
                "vector_store_sqlite_create_collection_calls",
                "Number of calls to create_collection in SQLiteVectorStore",
                label_names=label_names,
            )
            self.create_collection_latency_summary = metrics_factory.get_summary(
                "vector_store_sqlite_create_collection_latency_seconds",
                "Latency in seconds for create_collection in SQLiteVectorStore",
                label_names=label_names,
            )
            self.delete_collection_calls_counter = metrics_factory.get_counter(
                "vector_store_sqlite_delete_collection_calls",
                "Number of calls to delete_collection in SQLiteVectorStore",
                label_names=label_names,
            )
            self.delete_collection_latency_summary = metrics_factory.get_summary(
                "vector_store_sqlite_delete_collection_latency_seconds",
                "Latency in seconds for delete_collection in SQLiteVectorStore",
                label_names=label_names,
            )

    def collect_metrics(
        self,
        calls_counter: MetricsFactory.Counter | None,
        latency_summary: MetricsFactory.Summary | None,
        start_time: float,
        end_time: float,
    ) -> None:
        """Increment calls and observe latency."""
        if self._should_collect_metrics:
            cast(MetricsFactory.Counter, calls_counter).increment(
                labels=self._user_metrics_labels,
            )
            cast(MetricsFactory.Summary, latency_summary).observe(
                value=end_time - start_time,
                labels=self._user_metrics_labels,
            )

    async def startup(self) -> None:
        """Initialize the store: load sqlite-vec extension and create metadata tables."""

        # Register the sqlite-vec extension loader
        @event.listens_for(self._engine.sync_engine, "connect")
        def _load_vec_extension(
            dbapi_conn: sqlite3.Connection, _connection_record: object
        ) -> None:
            dbapi_conn.enable_load_extension(True)
            sqlite_vec.load(dbapi_conn)
            dbapi_conn.enable_load_extension(False)

        # Create metadata and records tables
        async with self._engine.begin() as conn:
            await conn.execute(
                text(
                    "CREATE TABLE IF NOT EXISTS _vs_collections ("
                    "  name TEXT PRIMARY KEY,"
                    "  vector_dimensions INTEGER NOT NULL,"
                    "  similarity_metric TEXT NOT NULL,"
                    "  properties_schema TEXT"
                    ")"
                )
            )
            await conn.execute(
                text(
                    "CREATE TABLE IF NOT EXISTS _vs_records ("
                    "  rowid INTEGER PRIMARY KEY AUTOINCREMENT,"
                    "  collection_name TEXT NOT NULL,"
                    "  uuid TEXT NOT NULL,"
                    "  properties TEXT,"
                    "  UNIQUE(collection_name, uuid)"
                    ")"
                )
            )
            await conn.execute(
                text(
                    "CREATE INDEX IF NOT EXISTS idx_vs_records_collection "
                    "ON _vs_records(collection_name)"
                )
            )

    async def shutdown(self) -> None:
        """Dispose of the engine."""
        await self._engine.dispose()

    async def create_collection(
        self,
        collection_name: str,
        *,
        vector_dimensions: int,
        similarity_metric: SimilarityMetric = SimilarityMetric.COSINE,
        properties_schema: Mapping[str, type[PropertyValue]] | None = None,
    ) -> None:
        """Create a collection in the vector store."""
        start_time = time.monotonic()

        _validate_metric(similarity_metric)

        async with self._engine.begin() as conn:
            # Check if collection already exists
            result = await conn.execute(
                text("SELECT name FROM _vs_collections WHERE name = :name"),
                {"name": collection_name},
            )
            if result.fetchone() is not None:
                msg = f"Collection {collection_name!r} already exists"
                raise ValueError(msg)

            # Insert collection metadata
            await conn.execute(
                text(
                    "INSERT INTO _vs_collections (name, vector_dimensions, similarity_metric, properties_schema) "
                    "VALUES (:name, :dims, :metric, :schema)"
                ),
                {
                    "name": collection_name,
                    "dims": vector_dimensions,
                    "metric": similarity_metric.value,
                    "schema": _serialize_schema(properties_schema),
                },
            )

            # Create shared vec0 table if it doesn't exist
            vtable = _vec_table_name(vector_dimensions, similarity_metric)
            vec0_metric = _METRIC_TO_VEC0[similarity_metric]
            await conn.execute(
                text(
                    f"CREATE VIRTUAL TABLE IF NOT EXISTS {vtable} USING vec0("
                    f"  embedding float[{vector_dimensions}],"
                    f"  collection_name text partition key,"
                    f'  distance_metric="{vec0_metric}"'
                    f")"
                )
            )

            # Create property indexes for filtered queries
            if properties_schema:
                safe_coll = collection_name.replace("'", "''")
                for field_name in properties_schema:
                    safe_field = field_name.replace("'", "''")
                    idx_name = f"idx_vs_prop_{collection_name}_{field_name}"
                    await conn.execute(
                        text(
                            f"CREATE INDEX IF NOT EXISTS [{idx_name}] "
                            f"ON _vs_records(json_extract(properties, '$.{safe_field}')) "
                            f"WHERE collection_name = '{safe_coll}'"
                        )
                    )

        end_time = time.monotonic()
        self.collect_metrics(
            self.create_collection_calls_counter,
            self.create_collection_latency_summary,
            start_time,
            end_time,
        )

    async def get_collection(self, collection_name: str) -> "SQLiteCollection":
        """Get a collection from the vector store."""
        return SQLiteCollection(
            engine=self._engine,
            collection_name=collection_name,
            store=self,
        )

    async def delete_collection(self, collection_name: str) -> None:
        """Delete a collection from the vector store."""
        start_time = time.monotonic()

        async with self._engine.begin() as conn:
            # Look up collection config to find the vec0 table
            result = await conn.execute(
                text(
                    "SELECT vector_dimensions, similarity_metric, properties_schema "
                    "FROM _vs_collections WHERE name = :name"
                ),
                {"name": collection_name},
            )
            row = result.fetchone()
            if row is not None:
                dims, metric_str = row[0], row[1]
                metric = SimilarityMetric(metric_str)
                vtable = _vec_table_name(dims, metric)

                # Drop property indexes
                schema = _deserialize_schema(row[2])
                if schema:
                    for field_name in schema:
                        idx_name = f"idx_vs_prop_{collection_name}_{field_name}"
                        await conn.execute(text(f"DROP INDEX IF EXISTS [{idx_name}]"))

                # Delete vectors from the shared vec0 table
                await conn.execute(
                    text(f"DELETE FROM {vtable} WHERE collection_name = :coll"),
                    {"coll": collection_name},
                )

            # Delete records
            await conn.execute(
                text("DELETE FROM _vs_records WHERE collection_name = :coll"),
                {"coll": collection_name},
            )

            # Delete metadata
            await conn.execute(
                text("DELETE FROM _vs_collections WHERE name = :name"),
                {"name": collection_name},
            )

        end_time = time.monotonic()
        self.collect_metrics(
            self.delete_collection_calls_counter,
            self.delete_collection_latency_summary,
            start_time,
            end_time,
        )

    async def get_collection_config(
        self,
        collection_name: str,
    ) -> _CollectionConfig:
        """Fetch collection configuration from metadata table.

        Returns:
            _CollectionConfig with dimensions, metric, schema, and vec_table.

        Raises:
            ValueError: If the collection does not exist.

        """
        async with self._engine.connect() as conn:
            result = await conn.execute(
                text(
                    "SELECT vector_dimensions, similarity_metric, properties_schema "
                    "FROM _vs_collections WHERE name = :name"
                ),
                {"name": collection_name},
            )
            row = result.fetchone()
            if row is None:
                msg = f"Collection {collection_name!r} does not exist"
                raise ValueError(msg)
            return _CollectionConfig(
                dimensions=row[0],
                metric=SimilarityMetric(row[1]),
                schema=_deserialize_schema(row[2]),
            )


class SQLiteCollection(Collection):
    """A collection backed by SQLite + sqlite-vec."""

    def __init__(
        self,
        *,
        engine: AsyncEngine,
        collection_name: str,
        store: SQLiteVectorStore,
    ) -> None:
        """Initialize the collection handle."""
        self._engine = engine
        self._collection_name = collection_name
        self._store = store

    async def upsert(self, *, records: Iterable[Record]) -> None:
        """Upsert records in the collection."""
        start_time = time.monotonic()

        cfg = await self._store.get_collection_config(self._collection_name)
        records_list = list(records)

        async with self._engine.begin() as conn:
            for record in records_list:
                uuid_str = str(record.uuid)
                props_json = _serialize_properties(record.properties)

                # Check if record exists
                result = await conn.execute(
                    text(
                        "SELECT rowid FROM _vs_records "
                        "WHERE collection_name = :coll AND uuid = :uuid"
                    ),
                    {"coll": self._collection_name, "uuid": uuid_str},
                )
                existing = result.fetchone()

                if existing is not None:
                    await self._upsert_existing(
                        conn,
                        existing[0],
                        props_json,
                        record.vector,
                        cfg.vec_table,
                    )
                else:
                    await self._upsert_new(
                        conn,
                        uuid_str,
                        props_json,
                        record.vector,
                        cfg.vec_table,
                    )

        end_time = time.monotonic()
        self._store.collect_metrics(
            self._store.upsert_calls_counter,
            self._store.upsert_latency_summary,
            start_time,
            end_time,
        )

    async def _upsert_existing(
        self,
        conn: object,
        row_id: int,
        props_json: str | None,
        vector: list[float] | None,
        vtable: str,
    ) -> None:
        """Update an existing record's properties and optionally its vector."""
        await conn.execute(  # type: ignore[union-attr]
            text("UPDATE _vs_records SET properties = :props WHERE rowid = :rowid"),
            {"props": props_json, "rowid": row_id},
        )
        if vector is not None:
            vec_blob = _serialize_vector(vector)
            await conn.execute(  # type: ignore[union-attr]
                text(f"UPDATE {vtable} SET embedding = :emb WHERE rowid = :rowid"),
                {"emb": vec_blob, "rowid": row_id},
            )

    async def _upsert_new(
        self,
        conn: object,
        uuid_str: str,
        props_json: str | None,
        vector: list[float] | None,
        vtable: str,
    ) -> None:
        """Insert a new record and optionally its vector."""
        result = await conn.execute(  # type: ignore[union-attr]
            text(
                "INSERT INTO _vs_records (collection_name, uuid, properties) "
                "VALUES (:coll, :uuid, :props)"
            ),
            {"coll": self._collection_name, "uuid": uuid_str, "props": props_json},
        )
        row_id = result.lastrowid

        if vector is not None:
            vec_blob = _serialize_vector(vector)
            await conn.execute(  # type: ignore[union-attr]
                text(
                    f"INSERT INTO {vtable} (rowid, embedding, collection_name) "
                    f"VALUES (:rowid, :emb, :coll)"
                ),
                {"rowid": row_id, "emb": vec_blob, "coll": self._collection_name},
            )

    async def query(
        self,
        *,
        query_vector: Sequence[float],
        similarity_threshold: float | None = None,
        limit: int | None = None,
        property_filter: FilterExpr | None = None,
        return_vector: bool = True,
        return_properties: bool = True,
    ) -> Iterable[QueryResult]:
        """Query for records matching the criteria by vector similarity."""
        start_time = time.monotonic()

        cfg = await self._store.get_collection_config(self._collection_name)
        query_blob = _serialize_vector(list(query_vector))

        if property_filter is None:
            results = await self._query_knn(
                cfg,
                query_blob,
                similarity_threshold,
                limit,
                return_vector,
                return_properties,
            )
        else:
            results = await self._query_filtered(
                cfg,
                query_blob,
                similarity_threshold,
                limit,
                property_filter,
                return_vector,
                return_properties,
            )

        end_time = time.monotonic()
        self._store.collect_metrics(
            self._store.query_calls_counter,
            self._store.query_latency_summary,
            start_time,
            end_time,
        )
        return results

    async def _query_knn(
        self,
        cfg: _CollectionConfig,
        query_blob: bytes,
        similarity_threshold: float | None,
        limit: int | None,
        return_vector: bool,
        return_properties: bool,
    ) -> list[QueryResult]:
        """Execute a pure KNN query via vec0 MATCH."""
        knn_limit = limit if limit is not None else 100

        async with self._engine.connect() as conn:
            knn_rows = await conn.execute(
                text(
                    f"SELECT knn.rowid, knn.distance "
                    f"FROM {cfg.vec_table} knn "
                    f"WHERE knn.embedding MATCH :query "
                    f"  AND knn.k = :k "
                    f"  AND knn.collection_name = :coll "
                    f"ORDER BY knn.distance"
                ),
                {"query": query_blob, "k": knn_limit, "coll": self._collection_name},
            )
            knn_results = knn_rows.fetchall()

            if not knn_results:
                return []

            rowid_to_distance: dict[int, float] = {
                row[0]: row[1] for row in knn_results
            }
            rowid_list = list(rowid_to_distance.keys())
            placeholders = ", ".join(f":rid{i}" for i in range(len(rowid_list)))
            bind_params: _BindParams = {
                f"rid{i}": rid for i, rid in enumerate(rowid_list)
            }

            # Fetch record details
            select_parts = ["r.rowid", "r.uuid"]
            if return_properties:
                select_parts.append("r.properties")

            rec_rows = await conn.execute(
                text(
                    f"SELECT {', '.join(select_parts)} "
                    f"FROM _vs_records r "
                    f"WHERE r.rowid IN ({placeholders})"
                ),
                bind_params,
            )
            rec_results = rec_rows.fetchall()

            rowid_to_rec: dict[int, tuple[str, str | None]] = {}
            for row in rec_results:
                rowid_to_rec[row[0]] = (
                    row[1],
                    row[2] if return_properties else None,
                )

            # Fetch vectors if requested
            rowid_to_vec: dict[int, list[float]] = {}
            if return_vector:
                vec_rows = await conn.execute(
                    text(
                        f"SELECT rowid, embedding FROM {cfg.vec_table} "
                        f"WHERE rowid IN ({placeholders})"
                    ),
                    bind_params,
                )
                for vrow in vec_rows.fetchall():
                    rowid_to_vec[vrow[0]] = _deserialize_vector(vrow[1])

        # Assemble results in distance order
        results: list[QueryResult] = []
        for rid in rowid_list:
            if rid not in rowid_to_rec:
                continue
            uuid_str, props_raw = rowid_to_rec[rid]
            distance = rowid_to_distance[rid]
            similarity = _distance_to_similarity(distance, cfg.metric)

            if similarity_threshold is not None and similarity < similarity_threshold:
                continue

            record = Record(
                uuid=UUID(uuid_str),
                vector=rowid_to_vec.get(rid) if return_vector else None,
                properties=_deserialize_properties(props_raw, cfg.schema)
                if return_properties
                else None,
            )
            results.append(QueryResult(score=similarity, record=record))
        return results

    async def _query_filtered(
        self,
        cfg: _CollectionConfig,
        query_blob: bytes,
        similarity_threshold: float | None,
        limit: int | None,
        property_filter: FilterExpr,
        return_vector: bool,
        return_properties: bool,
    ) -> list[QueryResult]:
        """Execute a filtered query using brute-force distance calculation."""
        distance_fn = _METRIC_TO_DISTANCE_FN[cfg.metric]
        compiler = _FilterCompiler()
        filter_clause = compiler.compile(property_filter)

        select_parts = [
            "r.uuid",
            "r.properties",
            f"{distance_fn}(v.embedding, :query) AS distance",
        ]
        if return_vector:
            select_parts.append("v.embedding")

        where_parts = [
            "r.collection_name = :coll",
            filter_clause,
        ]
        bind_params: _BindParams = {"coll": self._collection_name, "query": query_blob}
        bind_params.update(compiler.params)

        if similarity_threshold is not None:
            max_dist = _similarity_to_max_distance(similarity_threshold, cfg.metric)
            where_parts.append(f"{distance_fn}(v.embedding, :query) <= :max_dist")
            bind_params["max_dist"] = max_dist

        limit_clause = "LIMIT :lim" if limit is not None else ""
        if limit is not None:
            bind_params["lim"] = limit

        sql = (
            f"SELECT {', '.join(select_parts)} "
            f"FROM _vs_records r "
            f"JOIN {cfg.vec_table} v ON v.rowid = r.rowid "
            f"WHERE {' AND '.join(where_parts)} "
            f"ORDER BY distance ASC "
            f"{limit_clause}"
        )

        results: list[QueryResult] = []
        async with self._engine.connect() as conn:
            rows = await conn.execute(text(sql), bind_params)
            for row in rows.fetchall():
                uuid_val = UUID(row[0])
                props_raw = row[1]
                distance = row[2]
                similarity = _distance_to_similarity(distance, cfg.metric)

                vector = None
                if return_vector:
                    vector = _deserialize_vector(row[3])

                record = Record(
                    uuid=uuid_val,
                    vector=vector,
                    properties=_deserialize_properties(props_raw, cfg.schema)
                    if return_properties
                    else None,
                )
                results.append(QueryResult(score=similarity, record=record))
        return results

    async def get(
        self,
        *,
        record_uuids: Iterable[UUID],
        return_vector: bool = True,
        return_properties: bool = True,
    ) -> Iterable[Record]:
        """Get records from the collection by their UUIDs."""
        start_time = time.monotonic()

        uuid_list = list(record_uuids)
        if not uuid_list:
            end_time = time.monotonic()
            self._store.collect_metrics(
                self._store.get_calls_counter,
                self._store.get_latency_summary,
                start_time,
                end_time,
            )
            return []

        cfg = await self._store.get_collection_config(self._collection_name)
        uuid_strs = [str(u) for u in uuid_list]

        placeholders = ", ".join(f":u{i}" for i in range(len(uuid_strs)))
        bind_params: _BindParams = {f"u{i}": s for i, s in enumerate(uuid_strs)}
        bind_params["coll"] = self._collection_name

        select_parts = ["r.uuid"]
        if return_properties:
            select_parts.append("r.properties")
        if return_vector:
            select_parts.append("v.embedding")

        select_clause = ", ".join(select_parts)

        if return_vector:
            sql = (
                f"SELECT {select_clause} "
                f"FROM _vs_records r "
                f"LEFT JOIN {cfg.vec_table} v ON v.rowid = r.rowid "
                f"WHERE r.collection_name = :coll AND r.uuid IN ({placeholders})"
            )
        else:
            sql = (
                f"SELECT {select_clause} "
                f"FROM _vs_records r "
                f"WHERE r.collection_name = :coll AND r.uuid IN ({placeholders})"
            )

        async with self._engine.connect() as conn:
            rows = await conn.execute(text(sql), bind_params)
            rows_list = rows.fetchall()

        # Build a uuid->record map for ordered output
        uuid_to_record: dict[str, Record] = {}
        for row in rows_list:
            idx = 0
            uuid_val = UUID(row[idx])
            idx += 1

            props = None
            if return_properties:
                props = _deserialize_properties(row[idx], cfg.schema)
                idx += 1

            vector = None
            if return_vector:
                raw_vec = row[idx]
                if raw_vec is not None:
                    vector = _deserialize_vector(raw_vec)

            uuid_to_record[str(uuid_val)] = Record(
                uuid=uuid_val,
                vector=vector,
                properties=props,
            )

        # Return in input order
        result = [uuid_to_record[str(u)] for u in uuid_list if str(u) in uuid_to_record]

        end_time = time.monotonic()
        self._store.collect_metrics(
            self._store.get_calls_counter,
            self._store.get_latency_summary,
            start_time,
            end_time,
        )
        return result

    async def delete(self, *, record_uuids: Iterable[UUID]) -> None:
        """Delete records from the collection by their UUIDs."""
        start_time = time.monotonic()

        uuid_list = list(record_uuids)
        if not uuid_list:
            end_time = time.monotonic()
            self._store.collect_metrics(
                self._store.delete_calls_counter,
                self._store.delete_latency_summary,
                start_time,
                end_time,
            )
            return

        cfg = await self._store.get_collection_config(self._collection_name)
        uuid_strs = [str(u) for u in uuid_list]

        placeholders = ", ".join(f":u{i}" for i in range(len(uuid_strs)))
        bind_params: _BindParams = {f"u{i}": s for i, s in enumerate(uuid_strs)}
        bind_params["coll"] = self._collection_name

        async with self._engine.begin() as conn:
            # Delete vectors from vec0
            await conn.execute(
                text(
                    f"DELETE FROM {cfg.vec_table} "
                    f"WHERE rowid IN ("
                    f"  SELECT rowid FROM _vs_records "
                    f"  WHERE collection_name = :coll AND uuid IN ({placeholders})"
                    f")"
                ),
                bind_params,
            )

            # Delete records
            await conn.execute(
                text(
                    f"DELETE FROM _vs_records "
                    f"WHERE collection_name = :coll AND uuid IN ({placeholders})"
                ),
                bind_params,
            )

        end_time = time.monotonic()
        self._store.collect_metrics(
            self._store.delete_calls_counter,
            self._store.delete_latency_summary,
            start_time,
            end_time,
        )
