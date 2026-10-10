"""
Vector store backed by SQLite + pluggable vector search engine.

The store is one collection. Each partition gets its own records table and
vector search engine, and a pending operations table shared by the
collection's partitions tracks search engine operations for crash recovery:
on startup, unfinalized operations are replayed.
"""

import logging
from collections import defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from pathlib import Path
from typing import ClassVar, override
from uuid import UUID

import numpy as np
from pydantic import BaseModel, Field, InstanceOf, JsonValue, field_validator
from sqlalchemy import (
    JSON,
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
    update,
)
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.engine import Engine
from sqlalchemy.engine.interfaces import DBAPIConnection
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import DeclarativeBase, MappedColumn, Session, mapped_column
from sqlalchemy.pool import ConnectionPoolEntry, StaticPool
from sqlalchemy.sql.elements import ColumnElement

from memmachine_server.common.data_types import (
    PropertyType,
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

from .data_types import (
    IndexedProperties,
    PartitionSchema,
    QueryMatch,
    QueryResult,
    Record,
    VectorStorePartitionAlreadyExistsError,
    VectorStorePartitionSchemaMismatchError,
    indexed_property_names,
    validate_vector_store_name,
)
from .declared_properties import require_declared_properties, require_supported_filter
from .sql_columns import (
    compile_property_filter,
    property_column_values,
    property_columns,
    property_indexes,
)
from .utils import (
    _IDENTIFIER_MAX_BYTES,
    require_dimensions,
    require_partition_key,
    require_valid_limit,
    require_valid_query_vector,
    require_valid_score_threshold,
)
from .vector_search_engine import VectorSearchEngine
from .vector_store import VectorStore, VectorStorePartition

logger = logging.getLogger(__name__)


class IndexLoadError(RuntimeError):
    """Raised when a partition's on-disk index file cannot be loaded."""

    def __init__(self, vector_store_name: str, partition_key: str, path: Path) -> None:
        """Initialize with the vector store, the partition key, and the index file path."""
        self.vector_store_name = vector_store_name
        self.partition_key = partition_key
        self.path = path
        super().__init__(
            f"Index for partition {partition_key!r} of vector store {vector_store_name!r} "
            f"at {path} could not be loaded"
        )


class BaseSQLiteVectorStore(DeclarativeBase):
    """Base class for SQLiteVectorStore ORM models."""


class _PartitionRow(BaseSQLiteVectorStore):
    """The registry: one row per partition, keyed by its vector store name and key.

    Stores of different names may share one engine, so the vector store name
    is part of the key and every read names it.
    """

    __tablename__ = "vector_store_sqlite_pt"

    vector_store_name: MappedColumn[str] = mapped_column(
        String(_IDENTIFIER_MAX_BYTES), primary_key=True
    )
    partition_key: MappedColumn[str] = mapped_column(String(255), primary_key=True)
    # The dimensions, metric, and declared schema the partition was created
    # under, so a store built with others fails loudly instead of reading
    # columns and vectors that are not there.
    schema: MappedColumn[dict[str, JsonValue]] = mapped_column(JSON, nullable=False)
    # Flips to True after the first successful index save.
    # Once True, the on-disk index file is part of the durable contract:
    # missing or corrupt is treated as an error rather than silently rebuilt empty.
    index_saved: MappedColumn[bool] = mapped_column(
        Boolean, nullable=False, default=False
    )


class _PendingOperationRow(BaseSQLiteVectorStore):
    """
    Pending partition operations for crash recovery.

    One row per (vector store, partition, record). New operations replace old
    ones. Lifecycle:
    1. Inserted in the same SQLite transaction as the records table change.
    2. Marked `applied=True` after the search engine processes the operation.
    3. Applied operations are deleted after the search engine is saved to disk.
    4. On startup, all remaining rows (applied or not) are replayed.
    """

    __tablename__ = "vector_store_sqlite_pd_op"
    __table_args__ = (
        ForeignKeyConstraint(
            ["vector_store_name", "partition_key"],
            [
                f"{_PartitionRow.__tablename__}.vector_store_name",
                f"{_PartitionRow.__tablename__}.partition_key",
            ],
            ondelete="CASCADE",
        ),
    )

    vector_store_name: MappedColumn[str] = mapped_column(
        String(_IDENTIFIER_MAX_BYTES), primary_key=True
    )
    partition_key: MappedColumn[str] = mapped_column(String(255), primary_key=True)
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
    vector_store_name: str,
    partition_key: str,
    search_engine: VectorSearchEngine,
    path: str,
) -> None:
    """Save a partition's index to disk."""
    # Write index to path.
    await search_engine.save(path)

    # Delete applied pending operations and flip index_saved to True.
    async with create_session() as session, session.begin():
        await session.execute(
            delete(_PendingOperationRow).where(
                _PendingOperationRow.vector_store_name == vector_store_name,
                _PendingOperationRow.partition_key == partition_key,
                _PendingOperationRow.applied.is_(True),
            )
        )
        await session.execute(
            update(_PartitionRow)
            .where(
                _PartitionRow.vector_store_name == vector_store_name,
                _PartitionRow.partition_key == partition_key,
                _PartitionRow.index_saved.is_(False),
            )
            .values(index_saved=True)
        )


class SQLiteVectorStorePartition(VectorStorePartition):
    """A partition backed by SQLite + a pluggable vector search engine."""

    _SUPPORTED_FILTER_NODES: ClassVar[frozenset[type]] = frozenset(
        {FilterComparison, FilterIn, FilterIsNull, FilterAnd, FilterOr, FilterNot}
    )

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
        vector_store_name: str,
        partition_key: str,
        vector_dimensions: int,
        similarity_metric: SimilarityMetric,
        indexed_properties: Mapping[str, PropertyType],
        index_path: str | None,
        save_threshold: int,
    ) -> None:
        """Initialize a partition handle."""
        self._create_session = create_session
        self._sync_sqlalchemy_engine = sync_sqlalchemy_engine
        self._records_table = records_table
        self._search_engine = search_engine

        self._vector_store_name = vector_store_name
        self._partition_key = partition_key

        self._vector_dimensions = vector_dimensions
        self._similarity_metric = similarity_metric
        self._indexed_properties = dict(indexed_properties)

        self._index_path = index_path
        self._save_threshold = save_threshold

    @property
    @override
    def partition_key(self) -> str:
        return self._partition_key

    @property
    @override
    def similarity_metric(self) -> SimilarityMetric:
        return self._similarity_metric

    @property
    @override
    def indexed_properties(self) -> Mapping[str, PropertyType]:
        return self._indexed_properties

    @property
    @override
    def supported_filter_nodes(self) -> frozenset[type]:
        return SQLiteVectorStorePartition._SUPPORTED_FILTER_NODES

    async def _maybe_save_index(self) -> None:
        """Save the index to disk if applied pending operations exceed the threshold."""
        if self._index_path is None:
            return

        async with self._create_session() as session:
            count = (
                await session.execute(
                    select(func.count()).where(
                        _PendingOperationRow.vector_store_name
                        == self._vector_store_name,
                        _PendingOperationRow.partition_key == self._partition_key,
                        _PendingOperationRow.applied.is_(True),
                    )
                )
            ).scalar_one()

        if count >= self._save_threshold:
            await _save_partition_index(
                create_session=self._create_session,
                vector_store_name=self._vector_store_name,
                partition_key=self._partition_key,
                search_engine=self._search_engine,
                path=self._index_path,
            )

    @override
    async def upsert(self, *, records: Iterable[Record]) -> None:
        records = list(records)
        if not records:
            return
        for record in records:
            require_declared_properties(record.properties, self._indexed_properties)
            require_dimensions(record.vector, self._vector_dimensions)

        async with self._create_session() as session, session.begin():
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
                                record.properties, self._indexed_properties
                            ),
                        }
                        for record in records
                    ],
                )
            ).all()
            uuid_to_row_id: dict[UUID, int] = {row.uuid: row.row_id for row in rows}

            pending_operation_values = [
                {
                    "vector_store_name": self._vector_store_name,
                    "partition_key": self._partition_key,
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
                        index_elements=[
                            "vector_store_name",
                            "partition_key",
                            "record_row_id",
                        ],
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

            async with self._create_session() as session, session.begin():
                await session.execute(
                    update(_PendingOperationRow)
                    .where(
                        _PendingOperationRow.vector_store_name
                        == self._vector_store_name,
                        _PendingOperationRow.partition_key == self._partition_key,
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
        require_valid_limit(limit)
        query_vectors = list(query_vectors)
        if not query_vectors:
            return []
        for query_vector in query_vectors:
            require_valid_query_vector(query_vector, self._vector_dimensions)
        require_valid_score_threshold(score_threshold)

        if property_filter is not None:
            require_supported_filter(
                property_filter,
                self._indexed_properties,
                SQLiteVectorStorePartition._SUPPORTED_FILTER_NODES,
            )

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
                property_filter, self._records_table, self._indexed_properties
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

        higher_is_better = self._similarity_metric.higher_is_better
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
            reverse=self._similarity_metric.higher_is_better,
        )
        return matches

    @override
    async def delete(self, *, record_uuids: Iterable[UUID]) -> None:
        uuid_list = list(record_uuids)
        if not uuid_list:
            return

        record_uuids = list(uuid_list)

        async with self._create_session() as session, session.begin():
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
                    index_elements=[
                        "vector_store_name",
                        "partition_key",
                        "record_row_id",
                    ],
                    set_={
                        "operation_type": upsert_pending_operation.excluded.operation_type,
                        "applied": upsert_pending_operation.excluded.applied,
                    },
                ),
                [
                    {
                        "vector_store_name": self._vector_store_name,
                        "partition_key": self._partition_key,
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
        async with self._create_session() as session, session.begin():
            await session.execute(
                update(_PendingOperationRow)
                .where(
                    _PendingOperationRow.vector_store_name == self._vector_store_name,
                    _PendingOperationRow.partition_key == self._partition_key,
                    _PendingOperationRow.record_row_id.in_(record_row_ids),
                    _PendingOperationRow.applied.is_(False),
                )
                .values(applied=True)
            )
        await self._maybe_save_index()


VectorSearchEngineFactory = Callable[[int, SimilarityMetric], VectorSearchEngine]
"""Callable that creates a VectorSearchEngine given (num_dimensions, similarity_metric)."""


class SQLiteVectorStoreParams(BaseModel):
    """Parameters for constructing a SQLiteVectorStore.

    Attributes:
        sqlalchemy_engine (AsyncEngine):
            Async SQLAlchemy engine (sqlite+aiosqlite).
        vector_store_name (str):
            The name of this store; names its tables and index files, so
            stores of different names may share the engine.
        vector_dimensions (int):
            Dimensionality of every vector in the store.
        similarity_metric (SimilarityMetric):
            The metric every query of the store scores by
            (default: cosine).
        indexed_properties (IndexedProperties):
            The declared schema every partition of this store carries: each
            key is a typed, indexed column of the partition's records table,
            and a record or a filter naming any other key is rejected.
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
        ..., description="Async SQLAlchemy engine (sqlite+aiosqlite)"
    )
    vector_store_name: str = Field(..., description="The name of this store")
    vector_dimensions: int = Field(
        ..., gt=0, description="Dimensionality of every vector in the store"
    )
    similarity_metric: SimilarityMetric = Field(
        SimilarityMetric.COSINE,
        description="The metric every query of the store scores by",
    )
    indexed_properties: IndexedProperties = Field(
        ...,
        description="The declared schema every partition of this store carries",
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

    @field_validator("vector_store_name")
    @classmethod
    def _validate_vector_store_name(cls, vector_store_name: str) -> str:
        validate_vector_store_name(vector_store_name)
        return vector_store_name

    @field_validator("sqlalchemy_engine")
    @classmethod
    def _validate_engine(cls, engine: AsyncEngine) -> AsyncEngine:
        assert not isinstance(engine.pool, StaticPool), (
            "Engine uses StaticPool, which shares one connection across sessions. "
            "Use a multi-connection pool instead."
        )
        db = engine.url.database
        if engine.dialect.name == "sqlite" and (db is None or db == ":memory:"):
            raise ValueError(
                "Engine uses ephemeral SQLite, where each connection gets a separate "
                "database. Use a file path instead."
            )
        return engine


class SQLiteVectorStore(VectorStore):
    """
    Vector store backed by SQLite + a pluggable vector search engine.

    The store is one collection; each partition gets its own records table
    and engine instance. The engine and its index file live in the process
    that opened the partition, so one process at a time may use a
    partition. A handle used after its partition is deleted acts on a
    partition created again under its key.
    """

    def __init__(self, params: SQLiteVectorStoreParams) -> None:
        """Initialize the vector store with the provided parameters."""
        self._sqlalchemy_engine = params.sqlalchemy_engine
        self._vector_store_name = params.vector_store_name
        self._vector_dimensions = params.vector_dimensions
        self._similarity_metric = params.similarity_metric
        self._indexed_properties = params.indexed_properties
        self._vector_search_engine_factory = params.vector_search_engine_factory

        self._index_directory = (
            Path(params.index_directory) if params.index_directory else None
        )
        self._save_threshold = params.save_threshold

        self._create_session = async_sessionmaker(
            self._sqlalchemy_engine, expire_on_commit=False
        )
        self._search_engines: dict[str, VectorSearchEngine] = {}
        self._sa_metadata = MetaData()

        self._sync_sqlalchemy_engine = create_engine(
            str(self._sqlalchemy_engine.url).replace("aiosqlite", "pysqlite")
        )

        # Stores of different collections may share the async engine; the
        # listener is registered once per engine.
        for sync_engine in (
            self._sqlalchemy_engine.sync_engine,
            self._sync_sqlalchemy_engine,
        ):
            if not event.contains(sync_engine, "connect", _enable_sqlite_foreign_keys):
                event.listen(sync_engine, "connect", _enable_sqlite_foreign_keys)

        self._started = False

    @property
    @override
    def vector_store_name(self) -> str:
        return self._vector_store_name

    @property
    @override
    def vector_dimensions(self) -> int:
        return self._vector_dimensions

    @property
    @override
    def similarity_metric(self) -> SimilarityMetric:
        return self._similarity_metric

    @property
    @override
    def indexed_properties(self) -> Mapping[str, PropertyType]:
        return self._indexed_properties

    def _require_started(self) -> None:
        if not self._started:
            raise RuntimeError(
                "VectorStore has not been started. Call startup() first."
            )

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
        """Replay any pending engine operations of this collection."""
        async with self._create_session() as session:
            pending_operations = (
                (
                    await session.execute(
                        select(_PendingOperationRow).where(
                            _PendingOperationRow.vector_store_name
                            == self._vector_store_name
                        )
                    )
                )
                .scalars()
                .all()
            )

        if not pending_operations:
            return

        operations_by_partition: dict[str, list[_PendingOperationRow]] = defaultdict(
            list
        )
        for operation in pending_operations:
            operations_by_partition[operation.partition_key].append(operation)

        for partition_key, operations in operations_by_partition.items():
            await self._replay_partition_operations(partition_key, operations)

    async def _replay_partition_operations(
        self, partition_key: str, operations: Iterable[_PendingOperationRow]
    ) -> None:
        async with self._create_session() as session:
            schema = await self._stored_schema(session, partition_key)
        if schema is None:
            return

        search_engine = await self._get_or_create_vector_search_engine(partition_key)

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

        async with self._create_session() as session, session.begin():
            await session.execute(
                update(_PendingOperationRow)
                .where(
                    _PendingOperationRow.vector_store_name == self._vector_store_name,
                    _PendingOperationRow.partition_key == partition_key,
                    _PendingOperationRow.record_row_id.in_(all_row_ids),
                )
                .values(applied=True)
            )

    @override
    async def shutdown(self) -> None:
        self._require_started()
        if self._index_directory is not None:
            for partition_key, search_engine in self._search_engines.items():
                path = self._index_path(partition_key)
                assert path is not None
                await _save_partition_index(
                    create_session=self._create_session,
                    vector_store_name=self._vector_store_name,
                    partition_key=partition_key,
                    search_engine=search_engine,
                    path=str(path),
                )
        self._search_engines.clear()
        self._started = False

    @override
    async def create_partition(self, partition_key: str) -> None:
        self._require_started()
        require_partition_key(partition_key)

        async with self._create_session() as session, session.begin():
            if await self._stored_schema(session, partition_key) is not None:
                raise VectorStorePartitionAlreadyExistsError(
                    self._vector_store_name, partition_key
                )

            self._clear_search_engine_state(partition_key)
            await self._ensure_partition_resources(session, partition_key)
            session.add(
                _PartitionRow(
                    vector_store_name=self._vector_store_name,
                    partition_key=partition_key,
                    schema=self._declared_schema().model_dump(mode="json"),
                )
            )

    @override
    async def open_or_create_partition(
        self, partition_key: str
    ) -> VectorStorePartition:
        self._require_started()
        require_partition_key(partition_key)

        async with self._create_session() as session, session.begin():
            if await self._stored_schema(session, partition_key) is None:
                self._clear_search_engine_state(partition_key)
                session.add(
                    _PartitionRow(
                        vector_store_name=self._vector_store_name,
                        partition_key=partition_key,
                        schema=self._declared_schema().model_dump(mode="json"),
                    )
                )
            records_table, search_engine = await self._ensure_partition_resources(
                session, partition_key
            )

        return self._partition_handle(partition_key, records_table, search_engine)

    @override
    async def get_partition(self, partition_key: str) -> VectorStorePartition | None:
        self._require_started()
        require_partition_key(partition_key)

        async with self._create_session() as session:
            schema = await self._stored_schema(session, partition_key)
        if schema is None:
            return None

        return self._partition_handle(
            partition_key,
            self._records_table(partition_key),
            await self._get_or_create_vector_search_engine(partition_key),
        )

    def _partition_handle(
        self,
        partition_key: str,
        records_table: Table,
        search_engine: VectorSearchEngine,
    ) -> SQLiteVectorStorePartition:
        index_path = self._index_path(partition_key)
        return SQLiteVectorStorePartition(
            create_session=self._create_session,
            sync_sqlalchemy_engine=self._sync_sqlalchemy_engine,
            records_table=records_table,
            search_engine=search_engine,
            vector_store_name=self._vector_store_name,
            partition_key=partition_key,
            vector_dimensions=self._vector_dimensions,
            similarity_metric=self._similarity_metric,
            indexed_properties=self._indexed_properties,
            index_path=str(index_path) if index_path is not None else None,
            save_threshold=self._save_threshold,
        )

    @override
    async def delete_partition(self, partition_key: str) -> None:
        self._require_started()
        require_partition_key(partition_key)

        async with self._create_session() as session:
            exists = (
                await session.execute(
                    select(_PartitionRow.partition_key).where(
                        _PartitionRow.vector_store_name == self._vector_store_name,
                        _PartitionRow.partition_key == partition_key,
                    )
                )
            ).scalar_one_or_none()
        if exists is None:
            return

        records_table = self._records_table(partition_key)
        async with self._create_session() as session, session.begin():
            connection = await session.connection()
            await connection.run_sync(
                self._sa_metadata.drop_all, tables=[records_table]
            )

            await session.execute(
                delete(_PartitionRow).where(
                    _PartitionRow.vector_store_name == self._vector_store_name,
                    _PartitionRow.partition_key == partition_key,
                )
            )

        self._sa_metadata.remove(records_table)

        # If unlink fails, the orphan is harmless.
        # _clear_search_engine_state will clean it up if a new partition with the same key is created.
        index_path = self._index_path(partition_key)
        if index_path is not None and index_path.exists():
            index_path.unlink()
        self._search_engines.pop(partition_key, None)

    @override
    async def purge_deleted_partitions(self) -> bool:
        # delete_partition drops the tables and the index file itself.
        return False

    # Helpers.

    def _partition_prefix(self, partition_key: str) -> str:
        """Unique prefix for a partition's native resources."""
        vector_store_name = self._vector_store_name
        return (
            f"vector_store_sqlite_{len(vector_store_name)}_{vector_store_name}"
            f"_{len(partition_key)}_{partition_key}"
        )

    def _records_table(self, partition_key: str) -> Table:
        """The partition's records table, built once per partition key."""
        name = f"{self._partition_prefix(partition_key)}_rc"
        records_table = self._sa_metadata.tables.get(name)
        if records_table is not None:
            return records_table
        records_table = Table(
            name,
            self._sa_metadata,
            Column("row_id", Integer, primary_key=True, autoincrement=True),
            Column("uuid", Uuid, nullable=False, unique=True),
            *property_columns(self._indexed_properties),
            extend_existing=True,
        )
        property_indexes(records_table, self._indexed_properties)
        return records_table

    def _declared_schema(self) -> PartitionSchema:
        return PartitionSchema(
            vector_dimensions=self._vector_dimensions,
            similarity_metric=self._similarity_metric,
            indexed_properties=indexed_property_names(self._indexed_properties),
        )

    def _index_path(self, partition_key: str) -> Path | None:
        """Return the on-disk index path for a partition, or None if in-memory."""
        if self._index_directory is None:
            return None
        return self._index_directory / f"{self._partition_prefix(partition_key)}.idx"

    def _clear_search_engine_state(self, partition_key: str) -> None:
        """Remove any in-memory engine and on-disk index for a partition."""
        self._search_engines.pop(partition_key, None)
        index_path = self._index_path(partition_key)
        if index_path is not None and index_path.exists():
            index_path.unlink()

    async def _stored_schema(
        self,
        session: AsyncSession,
        partition_key: str,
    ) -> PartitionSchema | None:
        """The schema the partition was created under; raises if it is not this store's."""
        stored = (
            await session.execute(
                select(_PartitionRow.schema).where(
                    _PartitionRow.vector_store_name == self._vector_store_name,
                    _PartitionRow.partition_key == partition_key,
                )
            )
        ).scalar_one_or_none()
        if stored is None:
            return None
        stored_schema = PartitionSchema.model_validate(stored)
        declared_schema = self._declared_schema()
        if stored_schema != declared_schema:
            raise VectorStorePartitionSchemaMismatchError(
                self._vector_store_name, partition_key, stored_schema, declared_schema
            )
        return stored_schema

    async def _get_or_create_vector_search_engine(
        self, partition_key: str
    ) -> VectorSearchEngine:
        if partition_key in self._search_engines:
            return self._search_engines[partition_key]

        search_engine = self._vector_search_engine_factory(
            self._vector_dimensions, self._similarity_metric
        )

        index_path = self._index_path(partition_key)
        if index_path is not None:
            async with self._create_session() as session:
                saved = (
                    await session.execute(
                        select(_PartitionRow.index_saved).where(
                            _PartitionRow.vector_store_name == self._vector_store_name,
                            _PartitionRow.partition_key == partition_key,
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
                        self._vector_store_name, partition_key, index_path
                    ) from e

        self._search_engines[partition_key] = search_engine
        return search_engine

    async def _ensure_partition_resources(
        self,
        session: AsyncSession,
        partition_key: str,
    ) -> tuple[Table, VectorSearchEngine]:
        records_table = self._records_table(partition_key)
        search_engine = await self._get_or_create_vector_search_engine(partition_key)

        connection = await session.connection()
        await connection.run_sync(
            self._sa_metadata.create_all,
            tables=[records_table],
        )

        return records_table, search_engine
