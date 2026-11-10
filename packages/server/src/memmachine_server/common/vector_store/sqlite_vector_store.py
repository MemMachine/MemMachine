"""
Vector store backed by SQLite + pluggable vector search engine.

Each life of a partition gets its own records table, vector search engine,
and index file. A pending operations table tracks search engine operations
for crash recovery: on a store's startup, its partitions' unfinalized
operations are replayed.
"""

import asyncio
from collections import defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from pathlib import Path
from typing import override
from uuid import UUID

import numpy as np
from pydantic import Field, InstanceOf, field_validator
from sqlalchemy import (
    Boolean,
    Column,
    ForeignKeyConstraint,
    Integer,
    LargeBinary,
    MetaData,
    String,
    Table,
    Uuid,
    create_engine,
    delete,
    event,
    func,
    select,
    text,
    update,
)
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.engine import Engine
from sqlalchemy.engine.interfaces import DBAPIConnection
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import DeclarativeBase, MappedColumn, Session, mapped_column
from sqlalchemy.pool import ConnectionPoolEntry
from sqlalchemy.sql.elements import ColumnElement

from memmachine_server.common.data_types import (
    SimilarityMetric,
)
from memmachine_server.common.filter.filter_parser import FilterExpr
from memmachine_server.common.sql import validate_engine_connections, write_session

from .data_types import (
    _IDENTIFIER_MAX_BYTES,
    QueryMatch,
    QueryResult,
    Record,
    VectorStoreSchema,
)
from .sqlite_properties import (
    compile_property_filter,
    property_column_values,
    property_columns,
    property_indexes,
)
from .utils import (
    validate_declared_properties,
    validate_dimensions,
    validate_limit,
    validate_query_vector,
    validate_score_threshold,
    validate_supported_filter,
)
from .vector_search_engine import VectorSearchEngine
from .vector_store import VectorStore, VectorStoreParams, VectorStorePartition


class IndexLoadError(RuntimeError):
    """Raised when a partition incarnation's on-disk index file cannot be loaded."""

    def __init__(self, vector_store_name: str, incarnation: UUID, path: Path) -> None:
        """Initialize with the vector store, the partition incarnation, and the index file path."""
        self.vector_store_name = vector_store_name
        self.incarnation = incarnation
        self.path = path
        super().__init__(
            f"Index for partition incarnation {incarnation} of vector store "
            f"{vector_store_name!r} at {path} could not be loaded"
        )


class BaseSQLiteVectorStore(DeclarativeBase):
    """Base class for SQLiteVectorStore ORM models."""


class _PartitionIncarnationRow(BaseSQLiteVectorStore):
    """One row per prepared partition incarnation, with the vector store that holds it."""

    __tablename__ = "vector_store_sqlite_pt_in"

    incarnation: MappedColumn[UUID] = mapped_column(Uuid, primary_key=True)
    vector_store_name: MappedColumn[str] = mapped_column(
        String(_IDENTIFIER_MAX_BYTES), nullable=False
    )
    # Flips to True after the first successful index save.
    # Once True, the on-disk index file is part of the durable contract:
    # missing or corrupt is treated as an error rather than silently rebuilt empty.
    index_saved: MappedColumn[bool] = mapped_column(
        Boolean, nullable=False, default=False
    )


class _PendingOperationRow(BaseSQLiteVectorStore):
    """
    Pending partition operations for crash recovery.

    One row per (partition incarnation, record). New operations replace old
    ones. Lifecycle:
    1. Inserted in the same SQLite transaction as the records table change.
    2. Marked `applied=True` after the search engine processes the operation.
    3. Applied operations are deleted after the search engine is saved to disk.
    4. On a store's startup, its partition incarnations' remaining rows
       (applied or not) are replayed.
    """

    __tablename__ = "vector_store_sqlite_pd_op"
    __table_args__ = (
        ForeignKeyConstraint(
            ["incarnation"],
            [f"{_PartitionIncarnationRow.__tablename__}.incarnation"],
            ondelete="CASCADE",
        ),
    )

    incarnation: MappedColumn[UUID] = mapped_column(Uuid, primary_key=True)
    record_row_id: MappedColumn[int] = mapped_column(Integer, primary_key=True)
    operation_type: MappedColumn[str] = mapped_column(
        String(8), nullable=False
    )  # "upsert" or "delete"
    vector: MappedColumn[bytes | None] = mapped_column(LargeBinary, nullable=True)
    applied: MappedColumn[bool] = mapped_column(Boolean, nullable=False, default=False)


def _enable_sqlite_foreign_keys(
    dbapi_connection: DBAPIConnection, _record: ConnectionPoolEntry
) -> None:
    cursor = dbapi_connection.cursor()
    cursor.execute("PRAGMA foreign_keys=ON")
    cursor.close()


async def _save_partition_index(
    *,
    create_session: async_sessionmaker[AsyncSession],
    incarnation: UUID,
    search_engine: VectorSearchEngine,
    path: str,
) -> None:
    """Save a partition incarnation's index to disk, leaving no file if the incarnation is purged."""
    await search_engine.save(path)

    async with write_session(create_session) as session:
        # The write lock holds a purge off until this commits, and the purge
        # then unlinks the file; no match means a purge committed first, so
        # the file this wrote must go.
        saved = await (await session.connection()).execute(
            update(_PartitionIncarnationRow)
            .where(_PartitionIncarnationRow.incarnation == incarnation)
            .values(index_saved=True)
        )
        if saved.rowcount == 0:
            await asyncio.to_thread(Path(path).unlink, missing_ok=True)
            return
        await session.execute(
            delete(_PendingOperationRow).where(
                _PendingOperationRow.incarnation == incarnation,
                _PendingOperationRow.applied.is_(True),
            )
        )


class SQLiteVectorStorePartition(VectorStorePartition):
    """A partition backed by SQLite + a pluggable vector search engine."""

    class _KeyFilter:
        """Per-candidate SQL filter using a sync SQLAlchemy session."""

        def __init__(
            self,
            sync_sqlalchemy_engine: Engine,
            records_table: Table,
            filter_expression: ColumnElement[bool],
        ) -> None:
            """Initialize with a sync SQLAlchemy engine, records table, and filter expression."""
            self._sync_sqlalchemy_engine = sync_sqlalchemy_engine
            self._records_table = records_table
            self._filter_expression = filter_expression

            self._cache: dict[int, bool] = {}
            self._session: Session | None = None

        def _get_session(self) -> Session:
            if self._session is None:
                self._session = Session(self._sync_sqlalchemy_engine)
            return self._session

        def __contains__(self, key: object) -> bool:
            """Return whether the key passes the SQL filter."""
            if not isinstance(key, int):
                return False
            if key in self._cache:
                return self._cache[key]

            row = (
                self._get_session()
                .execute(
                    select(self._records_table.c.row_id).where(
                        self._records_table.c.row_id == key,
                        self._filter_expression,
                    )
                )
                .scalar()
            )
            result = row is not None
            self._cache[key] = result
            return result

        def __del__(self) -> None:
            if self._session is not None:
                self._session.close()

    def __init__(
        self,
        *,
        create_session: async_sessionmaker[AsyncSession],
        sync_sqlalchemy_engine: Engine,
        records_table: Table,
        search_engine: VectorSearchEngine,
        incarnation: UUID,
        store_schema: VectorStoreSchema,
        index_path: str | None,
        save_threshold: int,
    ) -> None:
        """Initialize a handle on one partition incarnation."""
        self._create_session = create_session
        self._sync_sqlalchemy_engine = sync_sqlalchemy_engine
        self._records_table = records_table
        self._search_engine = search_engine

        self._incarnation = incarnation

        self._store_schema = store_schema

        self._index_path = index_path
        self._save_threshold = save_threshold

    async def _maybe_save_index(self) -> None:
        """Save the index to disk if applied pending operations exceed the threshold."""
        if self._index_path is None:
            return

        async with self._create_session() as session:
            count = (
                await session.execute(
                    select(func.count()).where(
                        _PendingOperationRow.incarnation == self._incarnation,
                        _PendingOperationRow.applied.is_(True),
                    )
                )
            ).scalar_one()

        if count >= self._save_threshold:
            await _save_partition_index(
                create_session=self._create_session,
                incarnation=self._incarnation,
                search_engine=self._search_engine,
                path=self._index_path,
            )

    @override
    async def upsert(self, *, records: Iterable[Record]) -> None:
        records = list(records)
        for record in records:
            validate_declared_properties(
                record.properties, self._store_schema.indexed_properties
            )
            validate_dimensions(record.vector, self._store_schema.vector_dimensions)

        async with write_session(self._create_session) as session:
            if not records:
                return
            insert_records = sqlite_insert(self._records_table)
            upsert_records = insert_records.on_conflict_do_update(
                index_elements=[self._records_table.c.uuid],
                # Every column but the row id takes the new version's value,
                # so a declared key the record no longer holds becomes NULL;
                # the uuid, rewritten to itself, keeps the SET list nonempty
                # for a store that declares no keys.
                set_={
                    column.name: insert_records.excluded[column.name]
                    for column in self._records_table.columns
                    if column.name != "row_id"
                },
            ).returning(self._records_table.c.uuid, self._records_table.c.row_id)
            rows = (
                await session.execute(
                    upsert_records,
                    [
                        {
                            "uuid": record.uuid,
                            **property_column_values(
                                record.properties, self._store_schema.indexed_properties
                            ),
                        }
                        for record in records
                    ],
                )
            ).all()
            uuid_to_row_id: dict[UUID, int] = {row.uuid: row.row_id for row in rows}

            pending_operation_values = [
                {
                    "incarnation": self._incarnation,
                    "record_row_id": uuid_to_row_id[record.uuid],
                    "operation_type": "upsert",
                    "vector": np.array(record.vector, dtype=np.float32).tobytes(),
                    "applied": False,
                }
                for record in records
            ]
            if pending_operation_values:
                upsert_pending_operation = sqlite_insert(_PendingOperationRow)
                await session.execute(
                    upsert_pending_operation.on_conflict_do_update(
                        index_elements=["incarnation", "record_row_id"],
                        set_={
                            "operation_type": upsert_pending_operation.excluded.operation_type,
                            "vector": upsert_pending_operation.excluded.vector,
                            "applied": upsert_pending_operation.excluded.applied,
                        },
                    ),
                    pending_operation_values,
                )

        await self._apply_engine_upserts(records, uuid_to_row_id)

    async def _apply_engine_upserts(
        self,
        records: Iterable[Record],
        uuid_to_row_id: Mapping[UUID, int],
    ) -> None:
        """Update search engine index after SQLite commit."""
        engine_vectors: dict[int, list[float]] = {
            uuid_to_row_id[record.uuid]: record.vector
            for record in records
            if record.vector is not None
        }

        if engine_vectors:
            await self._search_engine.remove(engine_vectors.keys())
            await self._search_engine.add(engine_vectors)

            async with write_session(self._create_session) as session:
                await session.execute(
                    update(_PendingOperationRow)
                    .where(
                        _PendingOperationRow.incarnation == self._incarnation,
                        _PendingOperationRow.record_row_id.in_(
                            list(engine_vectors.keys())
                        ),
                        _PendingOperationRow.applied.is_(False),
                    )
                    .values(applied=True)
                )

            await self._maybe_save_index()

    @override
    async def query(
        self,
        *,
        query_vectors: Iterable[Sequence[float]],
        limit: int,
        score_threshold: float | None = None,
        property_filter: FilterExpr | None = None,
    ) -> list[QueryResult]:
        validate_limit(limit)
        query_vectors = list(query_vectors)
        for query_vector in query_vectors:
            validate_query_vector(query_vector, self._store_schema.vector_dimensions)
        validate_score_threshold(score_threshold)

        if property_filter is not None:
            validate_supported_filter(
                property_filter, self._store_schema.indexed_properties
            )

        if not query_vectors:
            return []

        key_filter = self._build_key_filter(property_filter)

        search_results = await self._search_engine.search(
            query_vectors, limit=limit, allowed_keys=key_filter
        )

        results: list[QueryResult] = []
        for search_result in search_results:
            if not search_result.matches:
                results.append(QueryResult(matches=[]))
                continue
            matches = await self._build_matches(
                row_id_to_score={m.key: m.score for m in search_result.matches},
                score_threshold=score_threshold,
            )
            results.append(QueryResult(matches=matches))

        return results

    def _build_key_filter(
        self, property_filter: FilterExpr | None
    ) -> _KeyFilter | None:
        if property_filter is None:
            return None

        return SQLiteVectorStorePartition._KeyFilter(
            sync_sqlalchemy_engine=self._sync_sqlalchemy_engine,
            records_table=self._records_table,
            filter_expression=compile_property_filter(
                property_filter,
                self._records_table,
                self._store_schema.indexed_properties,
            ),
        )

    async def _build_matches(
        self,
        row_id_to_score: Mapping[int, float],
        score_threshold: float | None,
    ) -> list[QueryMatch]:
        fetch_records = select(
            self._records_table.c.uuid, self._records_table.c.row_id
        ).where(
            self._records_table.c.row_id.in_(list(row_id_to_score)),
        )

        async with self._create_session() as session:
            matched_rows = (await session.execute(fetch_records)).all()

        higher_is_better = self._store_schema.similarity_metric.higher_is_better
        matches: list[QueryMatch] = []
        for row in matched_rows:
            score = row_id_to_score.get(row.row_id)
            if score is None:
                continue

            if score_threshold is not None and (
                score < score_threshold if higher_is_better else score > score_threshold
            ):
                continue

            matches.append(QueryMatch(score=score, record_uuid=row.uuid))

        matches.sort(
            key=lambda match: match.score,
            reverse=self._store_schema.similarity_metric.higher_is_better,
        )
        return matches

    @override
    async def delete(self, *, record_uuids: Iterable[UUID]) -> None:
        record_uuids = list(record_uuids)

        async with write_session(self._create_session) as session:
            if not record_uuids:
                return
            rows = (
                await session.execute(
                    select(self._records_table.c.row_id).where(
                        self._records_table.c.uuid.in_(record_uuids),
                    )
                )
            ).all()
            if not rows:
                return

            record_row_ids = [row.row_id for row in rows]

            upsert_pending_operation = sqlite_insert(_PendingOperationRow)
            await session.execute(
                upsert_pending_operation.on_conflict_do_update(
                    index_elements=["incarnation", "record_row_id"],
                    set_={
                        "operation_type": upsert_pending_operation.excluded.operation_type,
                        "applied": upsert_pending_operation.excluded.applied,
                    },
                ),
                [
                    {
                        "incarnation": self._incarnation,
                        "record_row_id": record_row_id,
                        "operation_type": "delete",
                        "applied": False,
                    }
                    for record_row_id in record_row_ids
                ],
            )

            await session.execute(
                delete(self._records_table).where(
                    self._records_table.c.uuid.in_(record_uuids),
                )
            )

        await self._search_engine.remove(record_row_ids)
        async with write_session(self._create_session) as session:
            await session.execute(
                update(_PendingOperationRow)
                .where(
                    _PendingOperationRow.incarnation == self._incarnation,
                    _PendingOperationRow.record_row_id.in_(record_row_ids),
                    _PendingOperationRow.applied.is_(False),
                )
                .values(applied=True)
            )
        await self._maybe_save_index()


VectorSearchEngineFactory = Callable[[int, SimilarityMetric], VectorSearchEngine]
"""Callable that creates a VectorSearchEngine given (num_dimensions, similarity_metric)."""


class SQLiteVectorStoreParams(VectorStoreParams):
    """Parameters for constructing a SQLiteVectorStore.

    Stores of different names may share the engine and the index directory.

    Attributes:
        sqlalchemy_engine (AsyncEngine):
            Async SQLAlchemy engine (sqlite+aiosqlite), set up by
            `enable_sqlite_explicit_begin` and `enable_sqlite_foreign_keys`
            at its creation.
        vector_search_engine_factory (Callable[[int, SimilarityMetric], VectorSearchEngine]):
            Factory for creating :class:`VectorSearchEngine` instances.
            Receives `(ndim, metric)` and returns a search engine.
        index_directory (str | None):
            Directory for persisting index files.
            If None, indexes are in-memory only
            (default: None).
        save_threshold (int):
            Number of engine operations before auto-saving the index to disk.
            Only applies when index_directory is set
            (default: 1000).
    """

    sqlalchemy_engine: InstanceOf[AsyncEngine] = Field(
        ...,
        description=(
            "Async SQLAlchemy engine (sqlite+aiosqlite), set up by "
            "enable_sqlite_explicit_begin and enable_sqlite_foreign_keys at its "
            "creation"
        ),
    )
    vector_search_engine_factory: VectorSearchEngineFactory = Field(
        ...,
        description=(
            "Factory for creating VectorSearchEngine instances. "
            "Receives `(ndim, metric)` and returns a search engine"
        ),
    )
    index_directory: str | None = Field(
        None,
        description=(
            "Directory for persisting index files. If None, indexes are in-memory only"
        ),
    )
    save_threshold: int = Field(
        1000,
        description=(
            "Number of engine operations before auto-saving the index to disk. "
            "Only applies when index_directory is set"
        ),
    )

    @field_validator("sqlalchemy_engine")
    @classmethod
    def _validate_engine(cls, engine: AsyncEngine) -> AsyncEngine:
        validate_engine_connections(engine)
        return engine


class SQLiteVectorStore(VectorStore):
    """
    Vector store backed by SQLite + a pluggable vector search engine.

    Each life of a partition gets its own records table and engine
    instance. A declared property is a typed, indexed column of the records
    table. The engine and its index file live in the process that opened
    the partition, so one process at a time may use a partition.
    """

    def __init__(self, params: SQLiteVectorStoreParams) -> None:
        """Initialize the vector store with the provided parameters."""
        self._sqlalchemy_engine = params.sqlalchemy_engine
        self._vector_store_name = params.vector_store_name
        self._store_schema = params.store_schema
        self._vector_search_engine_factory = params.vector_search_engine_factory
        # An engine refuses a metric or dimensionality it cannot serve; one
        # built now refuses them before any partition is created.
        self._vector_search_engine_factory(
            self._store_schema.vector_dimensions,
            self._store_schema.similarity_metric,
        )

        self._index_directory = (
            Path(params.index_directory) if params.index_directory else None
        )
        self._save_threshold = params.save_threshold

        self._create_session = async_sessionmaker(
            self._sqlalchemy_engine, expire_on_commit=False
        )
        self._search_engines: dict[UUID, VectorSearchEngine] = {}
        self._sa_metadata = MetaData()

        self._sync_sqlalchemy_engine = create_engine(
            str(self._sqlalchemy_engine.url).replace("aiosqlite", "pysqlite")
        )

        event.listen(
            self._sync_sqlalchemy_engine, "connect", _enable_sqlite_foreign_keys
        )

        self._started = False

    def _require_started(self) -> None:
        if not self._started:
            raise RuntimeError(
                "VectorStore has not been started. Call startup() first."
            )

    @property
    @override
    def store_schema(self) -> VectorStoreSchema:
        return self._store_schema

    @override
    async def startup(self) -> None:
        if self._started:
            return

        if self._index_directory is not None:
            self._index_directory.mkdir(parents=True, exist_ok=True)
        async with self._sqlalchemy_engine.begin() as connection:
            await connection.run_sync(BaseSQLiteVectorStore.metadata.create_all)
        await self._replay_pending_operations()

        self._started = True

    async def _replay_pending_operations(self) -> None:
        """Replay any pending engine operations of this store's partition incarnations."""
        async with self._create_session() as session:
            operations = (
                (
                    await session.execute(
                        select(_PendingOperationRow)
                        .join(
                            _PartitionIncarnationRow,
                            _PartitionIncarnationRow.incarnation
                            == _PendingOperationRow.incarnation,
                        )
                        .where(
                            _PartitionIncarnationRow.vector_store_name
                            == self._vector_store_name
                        )
                    )
                )
                .scalars()
                .all()
            )

        operations_by_incarnation: dict[UUID, list[_PendingOperationRow]] = defaultdict(
            list
        )
        for operation in operations:
            operations_by_incarnation[operation.incarnation].append(operation)

        for incarnation, incarnation_operations in operations_by_incarnation.items():
            await self._replay_partition_operations(incarnation, incarnation_operations)

    async def _replay_partition_operations(
        self,
        incarnation: UUID,
        operations: Iterable[_PendingOperationRow],
    ) -> None:
        search_engine = await self._get_or_create_vector_search_engine(incarnation)

        upserted_vectors: dict[int, list[float]] = {}
        deleted_row_ids: list[int] = []
        for operation in operations:
            if operation.operation_type == "upsert" and operation.vector is not None:
                vector = np.frombuffer(operation.vector, dtype=np.float32)
                upserted_vectors[operation.record_row_id] = [
                    float(value) for value in vector.flat
                ]
            elif operation.operation_type == "delete":
                deleted_row_ids.append(operation.record_row_id)

        all_row_ids = list(upserted_vectors.keys()) + deleted_row_ids
        if not all_row_ids:
            return

        await search_engine.remove(all_row_ids)
        if upserted_vectors:
            await search_engine.add(upserted_vectors)

        async with write_session(self._create_session) as session:
            await session.execute(
                update(_PendingOperationRow)
                .where(
                    _PendingOperationRow.incarnation == incarnation,
                    _PendingOperationRow.record_row_id.in_(all_row_ids),
                )
                .values(applied=True)
            )

    @override
    async def shutdown(self) -> None:
        self._require_started()
        if self._index_directory is not None:
            for incarnation, search_engine in self._search_engines.items():
                path = self._index_path(incarnation)
                assert path is not None
                await _save_partition_index(
                    create_session=self._create_session,
                    incarnation=incarnation,
                    search_engine=search_engine,
                    path=str(path),
                )
        self._search_engines.clear()
        self._sync_sqlalchemy_engine.dispose()
        self._started = False

    @override
    async def prepare_partition_incarnation(self, incarnation: UUID) -> None:
        self._require_started()
        async with write_session(self._create_session) as session:
            await session.execute(
                sqlite_insert(_PartitionIncarnationRow)
                .values(
                    incarnation=incarnation,
                    vector_store_name=self._vector_store_name,
                )
                .on_conflict_do_nothing(
                    index_elements=[_PartitionIncarnationRow.incarnation]
                )
            )
            connection = await session.connection()
            await connection.run_sync(
                self._sa_metadata.create_all,
                tables=[self._records_table(incarnation)],
            )

    @override
    async def open_partition_incarnation(
        self, incarnation: UUID
    ) -> SQLiteVectorStorePartition:
        self._require_started()
        index_path = self._index_path(incarnation)
        return SQLiteVectorStorePartition(
            create_session=self._create_session,
            sync_sqlalchemy_engine=self._sync_sqlalchemy_engine,
            records_table=self._records_table(incarnation),
            search_engine=await self._get_or_create_vector_search_engine(incarnation),
            incarnation=incarnation,
            store_schema=self._store_schema,
            index_path=str(index_path) if index_path is not None else None,
            save_threshold=self._save_threshold,
        )

    @override
    async def purge_partition_incarnation(self, incarnation: UUID) -> bool:
        # The incarnation's records live in its own table and index, so one
        # call drops them all; its pending operations go with its row by the
        # foreign key's cascade.
        self._require_started()
        records_table = self._records_table(incarnation)
        async with write_session(self._create_session) as session:
            deleted = await (await session.connection()).execute(
                delete(_PartitionIncarnationRow).where(
                    _PartitionIncarnationRow.incarnation == incarnation
                )
            )
            tables = (
                await session.execute(
                    text(
                        "SELECT count(*) FROM sqlite_master "
                        "WHERE type = 'table' AND name = :name"
                    ),
                    {"name": records_table.name},
                )
            ).scalar_one()
            connection = await session.connection()
            await connection.run_sync(
                self._sa_metadata.drop_all, tables=[records_table]
            )
        self._sa_metadata.remove(records_table)
        self._search_engines.pop(incarnation, None)

        # A save this purge overtakes unlinks the file it writes.
        index_file = False
        index_path = self._index_path(incarnation)
        if index_path is not None:
            index_file = await asyncio.to_thread(index_path.exists)
            await asyncio.to_thread(index_path.unlink, missing_ok=True)
        return deleted.rowcount > 0 or tables > 0 or index_file

    # Helpers.

    def _partition_prefix(self, incarnation: UUID) -> str:
        """Unique prefix for a partition life's native resources."""
        return f"vector_store_sqlite_{incarnation.hex}"

    def _records_table(self, incarnation: UUID) -> Table:
        """The partition life's records table, built once per incarnation."""
        name = f"{self._partition_prefix(incarnation)}_rc"
        records_table = self._sa_metadata.tables.get(name)
        if records_table is not None:
            return records_table
        records_table = Table(
            name,
            self._sa_metadata,
            Column("row_id", Integer, primary_key=True, autoincrement=True),
            Column("uuid", Uuid, nullable=False, unique=True),
            *property_columns(self._store_schema.indexed_properties),
        )
        property_indexes(records_table, self._store_schema.indexed_properties)
        return records_table

    def _index_path(self, incarnation: UUID) -> Path | None:
        """Return the on-disk index path for a partition life, or None if in-memory."""
        if self._index_directory is None:
            return None
        return self._index_directory / f"{self._partition_prefix(incarnation)}.idx"

    async def _get_or_create_vector_search_engine(
        self, incarnation: UUID
    ) -> VectorSearchEngine:
        if incarnation in self._search_engines:
            return self._search_engines[incarnation]

        search_engine = self._vector_search_engine_factory(
            self._store_schema.vector_dimensions,
            self._store_schema.similarity_metric,
        )

        index_path = self._index_path(incarnation)
        if index_path is not None:
            async with self._create_session() as session:
                saved = (
                    await session.execute(
                        select(_PartitionIncarnationRow.index_saved).where(
                            _PartitionIncarnationRow.incarnation == incarnation
                        )
                    )
                ).scalar_one_or_none()

            if saved:
                # The engine just propagates whatever its backend raises.
                # Wrap any failure as IndexLoadError so callers see one type.
                try:
                    await search_engine.load(str(index_path))
                except Exception as e:
                    raise IndexLoadError(
                        self._vector_store_name, incarnation, index_path
                    ) from e

        # Every handle of an incarnation shares one engine: an open that
        # awaited past another's keeps the engine cached first.
        return self._search_engines.setdefault(incarnation, search_engine)
