"""
SQLite + USearch backed vector store implementation.

SQLite stores collection metadata, record UUIDs, and properties.
USearch provides the ANN index for vector search.
Vectors are stored in both SQLite (source of truth) and USearch (derived index).

Each logical collection gets its own records table and USearch index.
Different namespaces always get separate native tables and indexes.

Crash recovery: intended USearch operations are recorded in a SQLite
``vector_store_sqlite_usearch_pending_ops`` table within the same transaction
as the data write.  After the USearch index is updated, the pending entry is
cleared.  On startup, any leftover pending ops are replayed so the index
converges with the SQLite source of truth.
"""

import asyncio
from collections import defaultdict
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import ClassVar, override
from uuid import UUID
from weakref import WeakKeyDictionary

import numpy as np
import sqlalchemy as sa
from pydantic import BaseModel, Field, InstanceOf, JsonValue
from sqlalchemy import JSON, LargeBinary, String, Text, select
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import DeclarativeBase, MappedColumn, mapped_column
from usearch.index import Index, MetricKind

from memmachine_server.common.data_types import PropertyValue, SimilarityMetric
from memmachine_server.common.filter.filter_parser import FilterExpr
from memmachine_server.common.filter.sql_filter_util import compile_sql_filter
from memmachine_server.common.properties_json import (
    decode_properties,
    encode_properties,
)

from .data_types import (
    CollectionAlreadyExistsError,
    CollectionConfig,
    CollectionConfigMismatchError,
    QueryMatch,
    QueryResult,
    Record,
)
from .utils import validate_filter, validate_identifier
from .vector_store import Collection, VectorStore


class BaseSQLiteUSearchVectorStore(DeclarativeBase):
    """Base class for SQLiteUSearchVectorStore ORM models."""


class _CollectionRow(BaseSQLiteUSearchVectorStore):
    __tablename__ = "vector_store_sqlite_usearch_collections"

    namespace: MappedColumn[str] = mapped_column(String(32), primary_key=True)
    name: MappedColumn[str] = mapped_column(String(32), primary_key=True)
    config_json: MappedColumn[dict[str, JsonValue]] = mapped_column(
        JSON, nullable=False
    )


class _PendingOpRow(BaseSQLiteUSearchVectorStore):
    """Pending USearch index operations for crash recovery.

    Written in the same SQLite transaction as the data change.
    Cleared after the USearch index is successfully updated.
    On startup, any remaining rows are replayed.
    """

    __tablename__ = "vector_store_sqlite_usearch_pending_ops"

    id: MappedColumn[int] = mapped_column(
        sa.Integer, primary_key=True, autoincrement=True
    )
    collection_prefix: MappedColumn[str] = mapped_column(Text, nullable=False)
    op: MappedColumn[str] = mapped_column(
        String(8), nullable=False
    )  # "upsert" or "remove"
    rowid: MappedColumn[int] = mapped_column(sa.Integer, nullable=False)


class SQLiteUSearchCollection(Collection):
    """A logical collection backed by SQLite + USearch HNSW.

    Each logical collection has its own records table and USearch index,
    so KNN queries search only this collection's vectors directly.
    """

    _OVERFETCH_FACTOR: ClassVar[int] = 20

    def __init__(
        self,
        *,
        create_session: async_sessionmaker[AsyncSession],
        write_lock: asyncio.Lock,
        name: str,
        config: CollectionConfig,
        records_table: sa.Table,
        index: Index,
        collection_prefix: str,
    ) -> None:
        """Initialize with session factory, lock, table, and USearch index."""
        self._create_session = create_session
        self._write_lock = write_lock
        self._name = name
        self._config = config
        self._records_table = records_table
        self._index = index
        self._metric = config.similarity_metric
        self._collection_prefix = collection_prefix

    @staticmethod
    def _distance_to_score(
        distance: float, similarity_metric: SimilarityMetric
    ) -> float:
        if similarity_metric == SimilarityMetric.COSINE:
            return 1.0 - distance
        if similarity_metric == SimilarityMetric.DOT:
            return -distance  # USearch IP returns negative inner product
        # Euclidean (L2sq returns squared distance)
        return 1.0 / (1.0 + distance)

    def _compile_property_filter(
        self, property_filter: FilterExpr
    ) -> sa.ColumnElement[bool]:
        """Compile a property filter against this collection's JSON properties column."""
        properties_column = self._records_table.c.properties
        return compile_sql_filter(
            property_filter,
            lambda field: (properties_column[field], "properties_json"),
        )

    @override
    async def upsert(self, *, records: Iterable[Record]) -> None:
        records_list = list(records)
        if not records_list:
            return

        records_table = self._records_table
        index = self._index
        collection_prefix = self._collection_prefix

        async with self._write_lock:
            async with self._create_session() as session, session.begin():
                statement = sqlite_insert(records_table).on_conflict_do_update(
                    index_elements=[records_table.c.uuid],
                    set_={
                        "properties": sqlite_insert(records_table).excluded.properties,
                        "vector": sqlite_insert(records_table).excluded.vector,
                    },
                )
                await session.execute(
                    statement,
                    [
                        {
                            "uuid": str(record.uuid),
                            "properties": encode_properties(record.properties),
                            "vector": (
                                np.array(record.vector, dtype=np.float32).tobytes()
                                if record.vector is not None
                                else None
                            ),
                        }
                        for record in records_list
                    ],
                )

                # Fetch rowids for USearch operations
                uuid_strs = [str(record.uuid) for record in records_list]
                rows = (
                    await session.execute(
                        select(records_table.c.uuid, records_table.c.rowid).where(
                            records_table.c.uuid.in_(uuid_strs),
                        )
                    )
                ).all()
                uuid_to_rowid: dict[str, int] = {row.uuid: row.rowid for row in rows}

                # Record pending ops in same transaction
                pending_rowids: list[int] = []
                for record in records_list:
                    if record.vector is not None:
                        rowid = uuid_to_rowid[str(record.uuid)]
                        pending_rowids.append(rowid)
                if pending_rowids:
                    await session.execute(
                        sa.insert(_PendingOpRow),
                        [
                            {
                                "collection_prefix": collection_prefix,
                                "op": "upsert",
                                "rowid": rowid,
                            }
                            for rowid in pending_rowids
                        ],
                    )

            # SQLite committed — update USearch index (still under write lock)
            await self._apply_index_upserts(
                records_list, uuid_to_rowid, index, collection_prefix
            )

    async def _apply_index_upserts(
        self,
        records_list: list[Record],
        uuid_to_rowid: dict[str, int],
        index: Index,
        collection_prefix: str,
    ) -> None:
        """Update USearch index after SQLite commit and clear pending ops."""
        labels: list[int] = []
        vectors: list[np.ndarray] = []
        for record in records_list:
            if record.vector is not None:
                rowid = uuid_to_rowid[str(record.uuid)]
                labels.append(rowid)
                vectors.append(np.array(record.vector, dtype=np.float32))

        if labels:
            labels_array = np.array(labels, dtype=np.int64)
            vectors_array = np.vstack(vectors)

            def _update_index() -> None:
                for label in labels_array:
                    if index.count(int(label)) > 0:
                        index.remove(int(label))
                index.add(labels_array, vectors_array)

            await asyncio.to_thread(_update_index)

            async with self._create_session() as session, session.begin():
                await session.execute(
                    sa.delete(_PendingOpRow).where(
                        _PendingOpRow.collection_prefix == collection_prefix,
                        _PendingOpRow.rowid.in_(labels),
                    )
                )

    @override
    async def query(
        self,
        *,
        query_vectors: Iterable[Sequence[float]],
        score_threshold: float | None = None,
        limit: int | None = None,
        property_filter: FilterExpr | None = None,
        return_vector: bool = False,
        return_properties: bool = True,
    ) -> list[QueryResult]:
        query_vectors_list = list(query_vectors)
        if not query_vectors_list:
            return []

        if limit is not None and limit <= 0:
            return [QueryResult(matches=[]) for _ in query_vectors_list]

        if property_filter is not None and not validate_filter(property_filter):
            raise ValueError("Filter contains invalid field names")

        results: list[QueryResult] = []
        for query_vector in query_vectors_list:
            query_array = np.array(query_vector, dtype=np.float32)

            if property_filter is not None:
                matches = await self._query_filtered(
                    query_array,
                    score_threshold,
                    limit,
                    property_filter,
                    return_vector,
                    return_properties,
                )
            else:
                matches = await self._query_knn(
                    query_array,
                    score_threshold,
                    limit,
                    return_vector,
                    return_properties,
                )
            results.append(QueryResult(matches=matches))

        return results

    async def _query_knn(
        self,
        query_array: np.ndarray,
        score_threshold: float | None,
        limit: int | None,
        return_vector: bool,
        return_properties: bool,
    ) -> list[QueryMatch]:
        """Pure KNN via USearch HNSW.

        The index contains only this collection's vectors, so no overfetch
        or post-filtering by collection is needed.
        """
        index = self._index
        if index.size == 0:
            return []

        effective_limit = min(limit if limit is not None else index.size, index.size)

        results = await asyncio.to_thread(index.search, query_array, effective_limit)

        rowid_to_distance: dict[int, float] = {
            int(key): float(distance)
            for key, distance in zip(results.keys, results.distances, strict=True)
        }

        return await self._fetch_and_score_candidates(
            rowid_to_distance,
            None,
            score_threshold,
            return_vector,
            return_properties,
        )

    async def _query_filtered(
        self,
        query_array: np.ndarray,
        score_threshold: float | None,
        limit: int | None,
        property_filter: FilterExpr,
        return_vector: bool,
        return_properties: bool,
    ) -> list[QueryMatch]:
        """Filtered query: overfetch from USearch, post-filter via SQLite."""
        index = self._index
        effective_limit = limit if limit is not None else 10_000

        if index.size == 0:
            return []

        # Overfetch to account for filter selectivity
        overfetch_k = min(effective_limit * self._OVERFETCH_FACTOR, index.size)

        results = await asyncio.to_thread(index.search, query_array, overfetch_k)

        candidate_distances = {
            int(key): float(distance)
            for key, distance in zip(results.keys, results.distances, strict=True)
        }

        if not candidate_distances:
            return []

        matches = await self._fetch_and_score_candidates(
            candidate_distances,
            property_filter,
            score_threshold,
            return_vector,
            return_properties,
        )

        if limit is not None:
            matches = matches[:limit]

        # If not enough results after filtering, retry with full index
        if limit is not None and len(matches) < limit and overfetch_k < index.size:
            return await self._query_filtered_full(
                query_array,
                score_threshold,
                limit,
                property_filter,
                return_vector,
                return_properties,
            )

        return matches

    async def _fetch_and_score_candidates(
        self,
        candidate_distances: dict[int, float],
        property_filter: FilterExpr | None,
        score_threshold: float | None,
        return_vector: bool,
        return_properties: bool,
    ) -> list[QueryMatch]:
        """Fetch records for candidate rowids, apply filter, build scored matches."""
        records_table = self._records_table

        columns = [records_table.c.uuid, records_table.c.rowid]
        if return_properties:
            columns.append(records_table.c.properties)
        if return_vector:
            columns.append(records_table.c.vector)

        statement = select(*columns).where(
            records_table.c.rowid.in_(list(candidate_distances.keys())),
        )

        if property_filter is not None:
            compiled_filter = self._compile_property_filter(property_filter)
            if compiled_filter is not None:
                statement = statement.where(compiled_filter)

        async with self._create_session() as session:
            rows = (await session.execute(statement)).all()

        matches: list[QueryMatch] = []
        for row in rows:
            distance = candidate_distances.get(row.rowid)
            if distance is None:
                continue

            score = self._distance_to_score(distance, self._metric)
            if score_threshold is not None and score < score_threshold:
                continue

            properties: dict[str, PropertyValue] | None = None
            if return_properties:
                properties = decode_properties(row.properties)

            vector: list[float] | None = None
            if return_vector and row.vector is not None:
                vector = list(np.frombuffer(row.vector, dtype=np.float32))

            matches.append(
                QueryMatch(
                    score=score,
                    record=Record(
                        uuid=UUID(row.uuid), vector=vector, properties=properties
                    ),
                )
            )

        matches.sort(key=lambda match: match.score, reverse=True)
        return matches

    async def _query_filtered_full(
        self,
        query_array: np.ndarray,
        score_threshold: float | None,
        limit: int | None,
        property_filter: FilterExpr,
        return_vector: bool,
        return_properties: bool,
    ) -> list[QueryMatch]:
        """Fallback: search entire index, post-filter."""
        index = self._index
        results = await asyncio.to_thread(index.search, query_array, index.size)

        candidate_distances = {
            int(key): float(distance)
            for key, distance in zip(results.keys, results.distances, strict=True)
        }

        matches = await self._fetch_and_score_candidates(
            candidate_distances,
            property_filter,
            score_threshold,
            return_vector,
            return_properties,
        )

        if limit is not None:
            matches = matches[:limit]
        return matches

    @override
    async def get(
        self,
        *,
        record_uuids: Iterable[UUID],
        return_vector: bool = False,
        return_properties: bool = True,
    ) -> list[Record]:
        uuid_list = list(record_uuids)
        if not uuid_list:
            return []

        records_table = self._records_table
        uuid_strs = [str(record_uuid) for record_uuid in uuid_list]

        columns = [records_table.c.uuid]
        if return_properties:
            columns.append(records_table.c.properties)
        if return_vector:
            columns.append(records_table.c.vector)

        async with self._create_session() as session:
            rows = (
                await session.execute(
                    select(*columns).where(
                        records_table.c.uuid.in_(uuid_strs),
                    )
                )
            ).all()

        record_map: dict[UUID, Record] = {}
        for row in rows:
            uuid_val = UUID(row.uuid)

            properties: dict[str, PropertyValue] | None = None
            if return_properties:
                properties = decode_properties(row.properties)

            vector: list[float] | None = None
            if return_vector and row.vector is not None:
                vector = list(np.frombuffer(row.vector, dtype=np.float32))

            record_map[uuid_val] = Record(
                uuid=uuid_val, vector=vector, properties=properties
            )

        return [
            record_map[record_uuid]
            for record_uuid in uuid_list
            if record_uuid in record_map
        ]

    @override
    async def delete(self, *, record_uuids: Iterable[UUID]) -> None:
        uuid_list = list(record_uuids)
        if not uuid_list:
            return

        records_table = self._records_table
        index = self._index
        collection_prefix = self._collection_prefix
        uuid_strs = [str(record_uuid) for record_uuid in uuid_list]

        async with self._write_lock:
            async with self._create_session() as session, session.begin():
                rows = (
                    await session.execute(
                        select(records_table.c.rowid).where(
                            records_table.c.uuid.in_(uuid_strs),
                        )
                    )
                ).all()
                if not rows:
                    return

                rowids = [row.rowid for row in rows]

                # Record pending remove ops
                await session.execute(
                    sa.insert(_PendingOpRow),
                    [
                        {
                            "collection_prefix": collection_prefix,
                            "op": "remove",
                            "rowid": rowid,
                        }
                        for rowid in rowids
                    ],
                )

                # Delete from SQLite
                await session.execute(
                    sa.delete(records_table).where(
                        records_table.c.uuid.in_(uuid_strs),
                    )
                )

            # Remove from USearch index (still under write lock)
            def _remove_from_index() -> None:
                for rowid in rowids:
                    if index.count(rowid) > 0:
                        index.remove(rowid)

            await asyncio.to_thread(_remove_from_index)

            # Clear pending ops
            async with self._create_session() as session, session.begin():
                await session.execute(
                    sa.delete(_PendingOpRow).where(
                        _PendingOpRow.collection_prefix == collection_prefix,
                        _PendingOpRow.rowid.in_(rowids),
                    )
                )


class SQLiteUSearchVectorStoreParams(BaseModel):
    """Parameters for constructing a SQLiteUSearchVectorStore.

    Attributes:
        engine: Async SQLAlchemy engine (sqlite+aiosqlite).
        index_directory: Directory for persisting USearch index files.
            If None, indexes are in-memory only.
    """

    engine: InstanceOf[AsyncEngine] = Field(
        ..., description="Async SQLAlchemy engine (sqlite+aiosqlite)"
    )
    index_directory: str | None = Field(
        None, description="Directory for persisting USearch index files"
    )


def _replay_upsert_one(
    usearch_index: Index, label: int, vector_data: np.ndarray
) -> None:
    """Replace a single vector in the USearch index (crash-recovery helper)."""
    if usearch_index.count(label) > 0:
        usearch_index.remove(label)
    usearch_index.add(label, vector_data)


def _replay_remove_one(usearch_index: Index, label: int) -> None:
    """Remove a single vector from the USearch index (crash-recovery helper)."""
    if usearch_index.count(label) > 0:
        usearch_index.remove(label)


class SQLiteUSearchVectorStore(VectorStore):
    """Vector store backed by SQLite (metadata) + USearch HNSW (ANN index).

    Each logical collection gets its own records table and USearch index.
    Vectors are stored in SQLite as source of truth; USearch is a derived
    index that can be rebuilt.
    """

    _SIMILARITY_METRIC_TO_USEARCH_METRIC: ClassVar[
        dict[SimilarityMetric, MetricKind]
    ] = {
        SimilarityMetric.COSINE: MetricKind.Cos,
        SimilarityMetric.EUCLIDEAN: MetricKind.L2sq,
        SimilarityMetric.DOT: MetricKind.IP,
    }

    # Shared across all instances so that stores using the same engine
    # serialise SQLite writes through the same lock.
    # Keyed by engine so locks are garbage-collected when the engine is.
    _write_locks: WeakKeyDictionary[AsyncEngine, asyncio.Lock] = WeakKeyDictionary()
    _name_locks_by_engine: WeakKeyDictionary[
        AsyncEngine, defaultdict[tuple[str, str], asyncio.Lock]
    ] = WeakKeyDictionary()

    def __init__(self, params: SQLiteUSearchVectorStoreParams) -> None:
        """Initialize the vector store with the provided parameters."""
        self._engine = params.engine
        self._index_dir = (
            Path(params.index_directory) if params.index_directory else None
        )
        self._create_session = async_sessionmaker(self._engine, expire_on_commit=False)
        self._records_tables: dict[str, sa.Table] = {}
        self._indexes: dict[str, Index] = {}
        self._sa_metadata = sa.MetaData()

    @property
    def _write_lock(self) -> asyncio.Lock:
        return SQLiteUSearchVectorStore._write_locks.setdefault(
            self._engine, asyncio.Lock()
        )

    @property
    def _name_locks(self) -> defaultdict[tuple[str, str], asyncio.Lock]:
        return SQLiteUSearchVectorStore._name_locks_by_engine.setdefault(
            self._engine, defaultdict(asyncio.Lock)
        )

    @staticmethod
    def _collection_prefix(namespace: str, name: str) -> str:
        """Unique prefix for a logical collection's native resources."""
        return f"vector_store_sqlite_usearch_{namespace}_{name}"

    @staticmethod
    def _validate_metric(similarity_metric: SimilarityMetric) -> None:
        if (
            similarity_metric
            not in SQLiteUSearchVectorStore._SIMILARITY_METRIC_TO_USEARCH_METRIC
        ):
            raise ValueError(
                f"Unsupported similarity metric {similarity_metric.value!r}. "
                f"Supported: {', '.join(similarity_metric.value for similarity_metric in SQLiteUSearchVectorStore._SIMILARITY_METRIC_TO_USEARCH_METRIC)}"
            )

    @staticmethod
    def _build_records_table(table_name: str, sa_metadata: sa.MetaData) -> sa.Table:
        return sa.Table(
            table_name,
            sa_metadata,
            sa.Column("rowid", sa.Integer, primary_key=True, autoincrement=True),
            sa.Column("uuid", sa.Text, nullable=False, unique=True),
            sa.Column("properties", JSON, nullable=True),
            sa.Column("vector", LargeBinary, nullable=True),
        )

    @override
    async def startup(self) -> None:
        if self._index_dir is not None:
            self._index_dir.mkdir(parents=True, exist_ok=True)
        async with self._engine.begin() as connection:
            await connection.run_sync(BaseSQLiteUSearchVectorStore.metadata.create_all)
        await self._replay_pending_ops()

    async def _replay_pending_ops(self) -> None:
        """Replay any pending USearch operations from a prior crash."""
        async with self._create_session() as session:
            pending = (
                (
                    await session.execute(
                        select(_PendingOpRow).order_by(_PendingOpRow.id)
                    )
                )
                .scalars()
                .all()
            )

        if not pending:
            return

        ops_by_prefix: dict[str, list[_PendingOpRow]] = defaultdict(list)
        for op in pending:
            ops_by_prefix[op.collection_prefix].append(op)

        for collection_prefix, ops in ops_by_prefix.items():
            await self._replay_prefix_ops(collection_prefix, ops)

    async def _resolve_prefix(
        self, collection_prefix: str
    ) -> tuple[Index, sa.Table] | None:
        """Resolve a collection prefix to its USearch index and records table.

        Returns ``None`` if no stored collection matches the prefix.
        """
        async with self._create_session() as session:
            all_collections = (
                (await session.execute(select(_CollectionRow))).scalars().all()
            )

        for collection_row in all_collections:
            if (
                self._collection_prefix(collection_row.namespace, collection_row.name)
                == collection_prefix
            ):
                config = CollectionConfig.model_validate(collection_row.config_json)
                index = self._get_or_create_index(collection_prefix, config)
                records_table = self._get_or_build_records_table(collection_prefix)
                return index, records_table

        return None

    async def _replay_prefix_ops(
        self, collection_prefix: str, ops: list[_PendingOpRow]
    ) -> None:
        """Replay pending ops for a single collection."""
        resolved = await self._resolve_prefix(collection_prefix)
        if resolved is None:
            async with self._create_session() as session, session.begin():
                await session.execute(
                    sa.delete(_PendingOpRow).where(
                        _PendingOpRow.collection_prefix == collection_prefix
                    )
                )
            return

        index, records_table = resolved

        for op in ops:
            if op.op == "upsert":
                async with self._create_session() as session:
                    row = (
                        await session.execute(
                            select(records_table.c.vector).where(
                                records_table.c.rowid == op.rowid
                            )
                        )
                    ).scalar_one_or_none()
                if row is not None:
                    vector = np.frombuffer(row, dtype=np.float32)
                    await asyncio.to_thread(_replay_upsert_one, index, op.rowid, vector)
            elif op.op == "remove":
                await asyncio.to_thread(_replay_remove_one, index, op.rowid)

        operation_ids = [op.id for op in ops]
        async with self._create_session() as session, session.begin():
            await session.execute(
                sa.delete(_PendingOpRow).where(_PendingOpRow.id.in_(operation_ids))
            )

    @override
    async def shutdown(self) -> None:
        if self._index_dir is not None:
            for collection_prefix, index in self._indexes.items():
                path = self._index_dir / f"{collection_prefix}.usearch"
                await asyncio.to_thread(index.save, str(path))
        self._indexes.clear()
        self._records_tables.clear()

    def _get_or_build_records_table(self, collection_prefix: str) -> sa.Table:
        if collection_prefix not in self._records_tables:
            table_name = f"{collection_prefix}_records"
            self._records_tables[collection_prefix] = self._build_records_table(
                table_name, self._sa_metadata
            )
        return self._records_tables[collection_prefix]

    def _get_or_create_index(
        self,
        collection_prefix: str,
        config: CollectionConfig,
    ) -> Index:
        if collection_prefix not in self._indexes:
            usearch_metric = self._SIMILARITY_METRIC_TO_USEARCH_METRIC[
                config.similarity_metric
            ]
            index = Index(
                ndim=config.vector_dimensions, metric=usearch_metric, dtype="f32"
            )

            if self._index_dir is not None:
                path = self._index_dir / f"{collection_prefix}.usearch"
                if path.exists():
                    index.load(str(path))

            self._indexes[collection_prefix] = index
        return self._indexes[collection_prefix]

    async def _get_stored_config(
        self,
        session: AsyncSession,
        namespace: str,
        name: str,
    ) -> CollectionConfig | None:
        row = (
            await session.execute(
                select(_CollectionRow.config_json).where(
                    _CollectionRow.namespace == namespace,
                    _CollectionRow.name == name,
                )
            )
        ).scalar_one_or_none()
        if row is None:
            return None
        return CollectionConfig.model_validate(row)

    async def _ensure_native_tables(
        self,
        session: AsyncSession,
        namespace: str,
        name: str,
        config: CollectionConfig,
    ) -> tuple[sa.Table, Index, str]:
        """Idempotently create per-collection native resources."""
        collection_prefix = self._collection_prefix(namespace, name)
        records_table = self._get_or_build_records_table(collection_prefix)
        index = self._get_or_create_index(collection_prefix, config)

        connection = await session.connection()
        await connection.run_sync(
            self._sa_metadata.create_all,
            tables=[records_table],
        )

        # Create property indexes
        for field_name in config.properties_schema:
            index_name = f"idx_{records_table.name}_{field_name}"
            safe_field = field_name.replace("'", "''")
            await session.execute(
                sa.text(
                    f"CREATE INDEX IF NOT EXISTS [{index_name}] "
                    f"ON [{records_table.name}]"
                    f"(json_extract(properties, '$.{safe_field}'))"
                )
            )

        return records_table, index, collection_prefix

    def _build_collection_handle(
        self,
        name: str,
        config: CollectionConfig,
        records_table: sa.Table,
        index: Index,
        collection_prefix: str,
    ) -> SQLiteUSearchCollection:
        return SQLiteUSearchCollection(
            create_session=self._create_session,
            write_lock=self._write_lock,
            name=name,
            config=config,
            records_table=records_table,
            index=index,
            collection_prefix=collection_prefix,
        )

    @override
    async def create_collection(
        self,
        *,
        namespace: str,
        name: str,
        config: CollectionConfig,
    ) -> None:
        if not validate_identifier(namespace) or not validate_identifier(name):
            raise ValueError(f"Invalid namespace {namespace!r} or name {name!r}")
        self._validate_metric(config.similarity_metric)

        lock = self._name_locks[(namespace, name)]
        async with (
            lock,
            self._write_lock,
            self._create_session() as session,
            session.begin(),
        ):
            existing = await self._get_stored_config(session, namespace, name)
            if existing is not None:
                raise CollectionAlreadyExistsError(namespace, name)

            await self._ensure_native_tables(session, namespace, name, config)
            session.add(
                _CollectionRow(
                    namespace=namespace,
                    name=name,
                    config_json=config.model_dump(mode="json"),
                )
            )

    @override
    async def open_or_create_collection(
        self,
        *,
        namespace: str,
        name: str,
        config: CollectionConfig,
    ) -> Collection:
        if not validate_identifier(namespace) or not validate_identifier(name):
            raise ValueError(f"Invalid namespace {namespace!r} or name {name!r}")
        self._validate_metric(config.similarity_metric)

        lock = self._name_locks[(namespace, name)]
        async with (
            lock,
            self._write_lock,
            self._create_session() as session,
            session.begin(),
        ):
            existing = await self._get_stored_config(session, namespace, name)
            if existing is not None:
                if existing != config:
                    raise CollectionConfigMismatchError(
                        namespace, name, existing, config
                    )
                (
                    records_table,
                    index,
                    collection_prefix,
                ) = await self._ensure_native_tables(session, namespace, name, existing)
                return self._build_collection_handle(
                    name, existing, records_table, index, collection_prefix
                )

            records_table, index, collection_prefix = await self._ensure_native_tables(
                session, namespace, name, config
            )
            session.add(
                _CollectionRow(
                    namespace=namespace,
                    name=name,
                    config_json=config.model_dump(mode="json"),
                )
            )

        return self._build_collection_handle(
            name, config, records_table, index, collection_prefix
        )

    @override
    async def open_collection(
        self,
        *,
        namespace: str,
        name: str,
    ) -> Collection | None:
        if not validate_identifier(namespace) or not validate_identifier(name):
            raise ValueError(f"Invalid namespace {namespace!r} or name {name!r}")
        async with self._create_session() as session:
            existing = await self._get_stored_config(session, namespace, name)
        if existing is None:
            return None

        collection_prefix = self._collection_prefix(namespace, name)
        records_table = self._get_or_build_records_table(collection_prefix)
        index = self._get_or_create_index(collection_prefix, existing)
        return self._build_collection_handle(
            name, existing, records_table, index, collection_prefix
        )

    @override
    async def close_collection(self, *, collection: Collection) -> None:
        pass

    @override
    async def delete_collection(self, *, namespace: str, name: str) -> None:
        if not validate_identifier(namespace) or not validate_identifier(name):
            raise ValueError(f"Invalid namespace {namespace!r} or name {name!r}")
        lock = self._name_locks[(namespace, name)]
        async with (
            lock,
            self._write_lock,
            self._create_session() as session,
            session.begin(),
        ):
            existing = await self._get_stored_config(session, namespace, name)
            if existing is None:
                return

            collection_prefix = self._collection_prefix(namespace, name)
            records_table = self._get_or_build_records_table(collection_prefix)

            # Drop the per-collection records table (cascades indexes)
            await session.execute(
                sa.text(f"DROP TABLE IF EXISTS [{records_table.name}]")
            )

            # Clear pending ops for this collection
            await session.execute(
                sa.delete(_PendingOpRow).where(
                    _PendingOpRow.collection_prefix == collection_prefix
                )
            )

            # Remove from registry
            await session.execute(
                sa.delete(_CollectionRow).where(
                    _CollectionRow.namespace == namespace,
                    _CollectionRow.name == name,
                )
            )

            # Clean up in-memory caches
            self._records_tables.pop(collection_prefix, None)
            self._indexes.pop(collection_prefix, None)
            self._sa_metadata.remove(records_table)

            # Delete index file from disk
            if self._index_dir is not None:
                index_path = self._index_dir / f"{collection_prefix}.usearch"
                if index_path.exists():
                    index_path.unlink()
