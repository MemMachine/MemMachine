"""
Vector store backed by SQLite + pluggable vector search engine.

The store is one collection. Each partition gets its own records table and
vector search engine, and a pending operations table shared by the
collection's partitions tracks search engine operations for crash recovery:
on startup, unfinalized operations are replayed.

What survives a crash
---------------------

`upsert` and `delete` commit to SQLite before they return, so a process crash
loses nothing: the pending log carries every operation the search engine has
not been checkpointed with, and startup replays it.

A power failure is weaker, and callers should size their expectations to it.
The index is published atomically but not durably (see
`vector_search_engine.index_persistence`), so a power failure can revert the
last publication while the records table -- and the trim that ran behind that
publication -- stay committed. The result is records whose vectors are missing
from the index. They are simply unfindable: `query` cannot reach them, and
nothing else reads a stored vector, so re-upserting them is the repair.
Callers that need every record searchable after a power failure must be able
to re-ingest; nothing here detects the gap for them.
"""

import asyncio
import contextlib
import logging
from collections import defaultdict
from collections.abc import AsyncIterator, Callable, Iterable, Mapping, Sequence
from pathlib import Path
from typing import override
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
from sqlalchemy.engine import Connection, Engine
from sqlalchemy.engine.interfaces import DBAPIConnection
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import DeclarativeBase, MappedColumn, Session, mapped_column
from sqlalchemy.pool import ConnectionPoolEntry, StaticPool
from sqlalchemy.sql.elements import ColumnElement

from memmachine_server.common.data_types import (
    PropertyType,
)
from memmachine_server.common.filter.filter_parser import FilterExpr
from memmachine_server.common.filter.sql_filter_util import compile_sql_filter
from memmachine_server.common.properties_json import (
    encode_properties,
)
from memmachine_server.common.rw_locks import AsyncRWLock

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
from .utils import _IDENTIFIER_MAX_BYTES, validate_filter, validate_identifier
from .vector_search_engine import VectorSearchEngine
from .vector_store import VectorStore, VectorStorePartition

logger = logging.getLogger(__name__)


class IndexLoadError(RuntimeError):
    """A partition's saved index could not be loaded.

    The cause is not classified: engines propagate whatever their backend
    raises, and backends report a truncated index and a permission denial the
    same way. `__cause__` says which. Rebuilding from content is right for a
    corrupt index and useless for an unreadable one, so read the cause before
    choosing a remedy.
    """

    def __init__(self, vector_store_name: str, partition_key: str, path: Path) -> None:
        """Initialize with the vector store, the partition key, and the index file path."""
        self.vector_store_name = vector_store_name
        self.partition_key = partition_key
        self.path = path
        super().__init__(
            f"Index for partition {partition_key!r} of vector store {vector_store_name!r} "
            f"at {path} could not be loaded"
        )


class PendingOperationCorruptError(RuntimeError):
    """A pending operation row cannot be replayed as written.

    An upsert with no vector, a vector that is not whole float32s or not the
    store's width, or an operation type this store never writes. This code
    produces none of those, so something else wrote the database.

    Do not clear the log to get past it: until the next index save it holds
    the only durable copy of those vectors. Repair the row, or accept the loss
    deliberately.
    """

    def __init__(
        self,
        vector_store_name: str,
        partition_key: str,
        record_row_id: int,
        reason: str,
    ) -> None:
        """Initialize with the partition, the row, and what is wrong with it."""
        self.vector_store_name = vector_store_name
        self.partition_key = partition_key
        self.record_row_id = record_row_id
        super().__init__(
            f"Pending operation for row {record_row_id} of partition "
            f"{partition_key!r} of vector store {vector_store_name!r} cannot be replayed: "
            f"{reason}"
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
    # The dimensions, metric and declared schema the partition was created
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


_BEGIN_IMMEDIATE_OPTION = "memmachine_sqlite_begin_immediate"
"""Execution option asking the begin hook for `BEGIN IMMEDIATE`."""


def _disable_implicit_begin(
    dbapi_connection: DBAPIConnection, _record: ConnectionPoolEntry
) -> None:
    # With the DBAPI connection in autocommit, pysqlite emits no BEGIN of its
    # own, so the begin hook chooses the mode, per transaction.
    dbapi_connection.isolation_level = None


def _begin(connection: Connection) -> None:
    """Emit `BEGIN` explicitly, and `BEGIN IMMEDIATE` where asked for.

    Reads keep the deferred `BEGIN`.
    """
    if connection.get_execution_options().get(_BEGIN_IMMEDIATE_OPTION):
        connection.exec_driver_sql("BEGIN IMMEDIATE")
    else:
        connection.exec_driver_sql("BEGIN")


@contextlib.asynccontextmanager
async def _write_transaction(
    create_session: async_sessionmaker[AsyncSession],
) -> AsyncIterator[AsyncSession]:
    """A transaction holding SQLite's write lock from `BEGIN`.

    A deferred `BEGIN` would take the lock at the first write, letting another
    writer commit between an earlier read and it. Taking it at `BEGIN` makes
    read-then-write atomic.
    """
    async with create_session() as session:
        # Set before the transaction begins, where the begin hook reads it.
        await session.connection(execution_options={_BEGIN_IMMEDIATE_OPTION: True})
        yield session
        # A body that raises is rolled back when the session closes. A rollback
        # issued here could fail and mask the body's error; one issued by the
        # pool is logged and invalidates the connection instead.
        await session.commit()


async def _save_partition_index(
    *,
    create_session: async_sessionmaker[AsyncSession],
    vector_store_name: str,
    partition_key: str,
    search_engine: VectorSearchEngine,
    engine_lock: AsyncRWLock,
    path: str,
) -> None:
    """Publish a partition's index to disk and trim the operations it holds.

    The order is the whole protocol. The pending log is the only other copy of
    these vectors -- the records table has no vector column -- so an applied row
    may be deleted only once the index that holds it has been published.
    `save` returning is that statement, and no more than that: publication is
    atomic, not durable, so a power failure can revert it after this trim has
    committed. See the module docstring for what that leaves behind.
    """
    # Write index to path.
    async with engine_lock.write_lock():
        await search_engine.save(path)

    # Delete applied pending operations and flip index_saved to True.
    async with _write_transaction(create_session) as session:
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
    """A partition backed by SQLite + a pluggable vector search engine.

    Reads run freely. Writes are serialized by `write_lock`, which the store
    shares among every handle on one partition, so the engine sees a
    partition's writes in the order SQLite committed them.
    """

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
        engine_lock: AsyncRWLock,
        write_lock: asyncio.Lock,
        vector_store_name: str,
        partition_key: str,
        indexed_properties: Mapping[str, PropertyType],
        index_path: str | None,
        save_threshold: int,
    ) -> None:
        """Initialize a partition handle."""
        self._create_session = create_session
        self._sync_sqlalchemy_engine = sync_sqlalchemy_engine
        self._records_table = records_table
        self._search_engine = search_engine
        self._engine_lock = engine_lock
        self._write_lock = write_lock

        self._vector_store_name = vector_store_name
        self._partition_key = partition_key

        self._indexed_properties = dict(indexed_properties)

        self._index_path = index_path
        self._save_threshold = save_threshold

    @property
    @override
    def partition_key(self) -> str:
        return self._partition_key

    @property
    @override
    def indexed_properties(self) -> Mapping[str, PropertyType]:
        return self._indexed_properties

    async def _maybe_save_index(self) -> None:
        """Save the index to disk if applied pending operations exceed the threshold.

        Call with the write lock held: saving trims every applied operation
        from the log, and the log is the only other copy of a vector applied
        after the index was written.
        """
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
                engine_lock=self._engine_lock,
                path=self._index_path,
            )

    @override
    async def upsert(self, *, records: Iterable[Record]) -> None:
        records = list(records)
        if not records:
            return

        async with self._write_lock:
            async with _write_transaction(self._create_session) as session:
                upsert_records = (
                    sqlite_insert(self._records_table)
                    .on_conflict_do_update(
                        index_elements=[self._records_table.c.uuid],
                        set_={
                            "properties": sqlite_insert(
                                self._records_table
                            ).excluded.properties,
                        },
                    )
                    .returning(self._records_table.c.uuid, self._records_table.c.row_id)
                )
                rows = (
                    await session.execute(
                        upsert_records,
                        [
                            {
                                "uuid": record.uuid,
                                "properties": encode_properties(record.properties),
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
            # One hold for both, so no search sees the record's old vector gone
            # and its new one not yet there.
            async with self._engine_lock.write_lock():
                await self._search_engine.remove(engine_vectors.keys())
                await self._search_engine.add(engine_vectors)

            async with _write_transaction(self._create_session) as session:
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
        min_cosine_similarity: float | None = None,
        property_filter: FilterExpr | None = None,
    ) -> list[QueryResult]:
        query_vectors = list(query_vectors)
        if not query_vectors:
            return []

        if limit <= 0:
            return [QueryResult(matches=[]) for _ in query_vectors]

        if property_filter is not None and not validate_filter(property_filter):
            raise ValueError("Filter contains invalid field names")

        key_filter = self._build_key_filter(property_filter)

        async with self._engine_lock.read_lock():
            search_results = await self._search_engine.search(
                query_vectors, limit=limit, allowed_keys=key_filter
            )

        results: list[QueryResult] = []
        for search_result in search_results:
            if not search_result.matches:
                results.append(QueryResult(matches=[]))
                continue
            matches = await self._build_matches(
                row_id_to_cosine_similarity={
                    m.key: m.cosine_similarity for m in search_result.matches
                },
                min_cosine_similarity=min_cosine_similarity,
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
            filter_expression=compile_sql_filter(
                property_filter,
                lambda field: (
                    self._records_table.c.properties[field],
                    "properties_json",
                ),
            ),
        )

    async def _build_matches(
        self,
        row_id_to_cosine_similarity: Mapping[int, float],
        min_cosine_similarity: float | None,
    ) -> list[QueryMatch]:
        matched_row_ids = list(row_id_to_cosine_similarity.keys())

        fetch_records = select(
            self._records_table.c.uuid, self._records_table.c.row_id
        ).where(
            self._records_table.c.row_id.in_(matched_row_ids),
        )

        async with self._create_session() as session:
            matched_rows = (await session.execute(fetch_records)).all()

        matches: list[QueryMatch] = []
        for row in matched_rows:
            cosine_similarity = row_id_to_cosine_similarity.get(row.row_id)
            if cosine_similarity is None:
                continue

            if (
                min_cosine_similarity is not None
                and cosine_similarity < min_cosine_similarity
            ):
                continue

            matches.append(
                QueryMatch(
                    cosine_similarity=cosine_similarity,
                    record_uuid=row.uuid,
                )
            )

        matches.sort(key=lambda match: match.cosine_similarity, reverse=True)
        return matches

    @override
    async def delete(self, *, record_uuids: Iterable[UUID]) -> None:
        uuid_list = list(record_uuids)
        if not uuid_list:
            return

        record_uuids = list(uuid_list)

        async with self._write_lock:
            async with _write_transaction(self._create_session) as session:
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

            async with self._engine_lock.write_lock():
                await self._search_engine.remove(record_row_ids)
            async with _write_transaction(self._create_session) as session:
                await session.execute(
                    update(_PendingOperationRow)
                    .where(
                        _PendingOperationRow.vector_store_name
                        == self._vector_store_name,
                        _PendingOperationRow.partition_key == self._partition_key,
                        _PendingOperationRow.record_row_id.in_(record_row_ids),
                        _PendingOperationRow.applied.is_(False),
                    )
                    .values(applied=True)
                )
            await self._maybe_save_index()


VectorSearchEngineFactory = Callable[[int], VectorSearchEngine]
"""Callable that creates a VectorSearchEngine given the number of dimensions."""


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
        indexed_properties (IndexedProperties):
            The declared schema every partition of this store carries: each
            key is indexed for filtering, and its values are typed.
        vector_search_engine_factory (Callable[[int], VectorSearchEngine]):
            Factory for creating :class:`VectorSearchEngine` instances.
            Receives the number of dimensions and returns a search engine.
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
    indexed_properties: IndexedProperties = Field(
        ...,
        description="The declared schema every partition of this store carries",
    )
    vector_search_engine_factory: VectorSearchEngineFactory = Field(
        ...,
        description=(
            "Factory for creating VectorSearchEngine instances. "
            "Receives the number of dimensions and returns a search engine"
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
    that opened the partition, so a partition is managed by at most one
    process at a time; the consumer is responsible for sharding keys across
    processes. A handle outliving its partition is not detected.
    """

    def __init__(self, params: SQLiteVectorStoreParams) -> None:
        """Initialize the vector store with the provided parameters."""
        self._sqlalchemy_engine = params.sqlalchemy_engine
        self._vector_store_name = params.vector_store_name
        self._vector_dimensions = params.vector_dimensions
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
        self._engine_locks: dict[str, AsyncRWLock] = {}
        self._write_locks: dict[str, asyncio.Lock] = {}
        self._sa_metadata = MetaData()

        self._sync_sqlalchemy_engine = create_engine(
            str(self._sqlalchemy_engine.url).replace("aiosqlite", "pysqlite")
        )

        # Stores of different collections may share the async engine; each
        # listener is registered once per engine.
        for sync_engine in (
            self._sqlalchemy_engine.sync_engine,
            self._sync_sqlalchemy_engine,
        ):
            if not event.contains(sync_engine, "connect", _enable_sqlite_foreign_keys):
                event.listen(sync_engine, "connect", _enable_sqlite_foreign_keys)
        async_sync_engine = self._sqlalchemy_engine.sync_engine
        if not event.contains(async_sync_engine, "connect", _disable_implicit_begin):
            event.listen(async_sync_engine, "connect", _disable_implicit_begin)
        if not event.contains(async_sync_engine, "begin", _begin):
            event.listen(async_sync_engine, "begin", _begin)

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
    def indexed_properties(self) -> Mapping[str, PropertyType]:
        return self._indexed_properties

    def _require_started(self) -> None:
        if not self._started:
            raise RuntimeError(
                "VectorStore has not been started. Call startup() first."
            )

    @override
    async def provision(self) -> None:
        if self._index_directory is not None:
            self._index_directory.mkdir(parents=True, exist_ok=True)

        async with self._sqlalchemy_engine.begin() as connection:
            await connection.run_sync(BaseSQLiteVectorStore.metadata.create_all)

    @override
    async def startup(self) -> None:
        if self._started:
            return

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

        # A row that cannot be replayed is damage to a durable record: until
        # the next index save the log holds the only copy of the vector, so
        # skipping the row would leave a record no search can find. Refuse,
        # and leave the log for whoever repairs it.
        upserted_vectors: dict[int, list[float]] = {}
        deleted_row_ids: list[int] = []
        for operation in operations:
            if operation.operation_type == "upsert":
                if operation.vector is None:
                    raise PendingOperationCorruptError(
                        self._vector_store_name,
                        partition_key,
                        operation.record_row_id,
                        "upsert with no vector",
                    )
                try:
                    vector = np.frombuffer(operation.vector, dtype=np.float32)
                except ValueError as error:
                    raise PendingOperationCorruptError(
                        self._vector_store_name,
                        partition_key,
                        operation.record_row_id,
                        f"vector cannot be decoded: {error}",
                    ) from error
                if vector.size != self._vector_dimensions:
                    raise PendingOperationCorruptError(
                        self._vector_store_name,
                        partition_key,
                        operation.record_row_id,
                        f"holds a {vector.size}-dimension vector in a "
                        f"{self._vector_dimensions}-dimension store",
                    )
                upserted_vectors[operation.record_row_id] = [
                    float(value) for value in vector.flat
                ]
            elif operation.operation_type == "delete":
                deleted_row_ids.append(operation.record_row_id)
            else:
                raise PendingOperationCorruptError(
                    self._vector_store_name,
                    partition_key,
                    operation.record_row_id,
                    f"unknown operation_type {operation.operation_type!r}",
                )

        all_row_ids = list(upserted_vectors.keys()) + deleted_row_ids
        if not all_row_ids:
            return

        async with self._engine_lock_for(partition_key).write_lock():
            await search_engine.remove(all_row_ids)
            if upserted_vectors:
                await search_engine.add(upserted_vectors)

        async with _write_transaction(self._create_session) as session:
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
                # Hold the partition's write lock for the whole save. A write
                # that applied after the index file was written but before the
                # trim would be absent from the file and trimmed from the log.
                async with self._write_lock_for(partition_key):
                    await _save_partition_index(
                        create_session=self._create_session,
                        vector_store_name=self._vector_store_name,
                        partition_key=partition_key,
                        search_engine=search_engine,
                        engine_lock=self._engine_lock_for(partition_key),
                        path=str(path),
                    )
        self._search_engines.clear()
        self._started = False

    @override
    async def create_partition(self, partition_key: str) -> None:
        self._require_started()
        if not validate_identifier(partition_key):
            raise ValueError(f"Invalid partition key {partition_key!r}")

        async with _write_transaction(self._create_session) as session:
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
        if not validate_identifier(partition_key):
            raise ValueError(f"Invalid partition key {partition_key!r}")

        async with _write_transaction(self._create_session) as session:
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
        if not validate_identifier(partition_key):
            raise ValueError(f"Invalid partition key {partition_key!r}")

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
            engine_lock=self._engine_lock_for(partition_key),
            write_lock=self._write_lock_for(partition_key),
            vector_store_name=self._vector_store_name,
            partition_key=partition_key,
            indexed_properties=self._indexed_properties,
            index_path=str(index_path) if index_path is not None else None,
            save_threshold=self._save_threshold,
        )

    @override
    async def close_partition(self, *, partition: VectorStorePartition) -> None:
        self._require_started()

    @override
    async def delete_partition(self, partition_key: str) -> None:
        self._require_started()
        if not validate_identifier(partition_key):
            raise ValueError(f"Invalid partition key {partition_key!r}")

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
        async with _write_transaction(self._create_session) as session:
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
        """Get or create a SQLAlchemy Table for a per-partition records table."""
        return Table(
            f"{self._partition_prefix(partition_key)}_rc",
            self._sa_metadata,
            Column("row_id", Integer, primary_key=True, autoincrement=True),
            Column("uuid", Uuid, nullable=False, unique=True),
            Column("properties", JSON, nullable=False, default=dict),
            extend_existing=True,
            # A plain rowid is reused once the highest row is deleted. query()
            # resolves scored keys to rows without a lock, so a scored key could
            # resolve to a record other than the one the engine scored.
            sqlite_autoincrement=True,
        )

    def _engine_lock_for(self, partition_key: str) -> AsyncRWLock:
        """Get or create the lock guarding a partition's search engine.

        Engines are not safe for concurrent use: searches take the read side,
        everything else the write side. Shared by every handle on the
        partition and kept for the store's lifetime: a lock replaced while
        held would guard nobody.
        """
        return self._engine_locks.setdefault(partition_key, AsyncRWLock())

    def _write_lock_for(self, partition_key: str) -> asyncio.Lock:
        """Get or create the lock serializing a partition's writes.

        Held from SQL commit through engine apply, mark-applied, and any save,
        so the engine sees writes in commit order and a save's trim cannot land
        between another write's apply and its bookkeeping. Kept for the store's
        lifetime: replacing a held lock would serialize nobody.
        """
        return self._write_locks.setdefault(partition_key, asyncio.Lock())

    def _declared_schema(self) -> PartitionSchema:
        return PartitionSchema(
            vector_dimensions=self._vector_dimensions,
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

        search_engine = self._vector_search_engine_factory(self._vector_dimensions)

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
                # Backends raise unclassified errors, so nothing narrower than
                # Exception can be caught. `IndexLoadError` says what a caller
                # can do with the cause.
                try:
                    async with self._engine_lock_for(partition_key).write_lock():
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
