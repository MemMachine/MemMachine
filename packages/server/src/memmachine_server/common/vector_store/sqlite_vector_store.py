"""
Vector store backed by SQLite + pluggable vector search engine.

The store is one collection, held in one records table named by the
collection, so stores of different collections may share one engine. Every
partition of the collection lives in that table under an incarnation the
store mints per partition life, and has its own vector search engine and
index file, named by the incarnation: a deleted-and-recreated partition
never sees its predecessor's rows, engine or file. A pending operations
table tracks search engine operations for crash recovery: on startup,
unfinalized operations of live incarnations are replayed.

Deleting a partition is a registry write: the incarnation goes onto the
purge queue and the registry row goes, so the partition is unreachable at
once; `purge_deleted_partitions` reclaims its rows, log and index file
afterward, a bounded batch per call.

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

import logging
from collections import defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from datetime import datetime
from pathlib import Path
from typing import override
from uuid import UUID, uuid4

import numpy as np
from pydantic import BaseModel, Field, InstanceOf, JsonValue, field_validator
from sqlalchemy import (
    JSON,
    Boolean,
    Column,
    DateTime,
    Index,
    Integer,
    LargeBinary,
    MetaData,
    Select,
    String,
    Table,
    UniqueConstraint,
    Uuid,
    create_engine,
    delete,
    event,
    func,
    insert,
    select,
    update,
)
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.engine import Engine
from sqlalchemy.engine.interfaces import DBAPIConnection
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import DeclarativeBase, MappedColumn, Session, mapped_column
from sqlalchemy.pool import ConnectionPoolEntry, StaticPool
from sqlalchemy.sql.elements import ColumnElement

from memmachine_server.common.data_types import PropertyType
from memmachine_server.common.filter.filter_parser import (
    FilterExpr,
)
from memmachine_server.common.filter.sql_filter_util import compile_sql_filter
from memmachine_server.common.properties_json import (
    encode_properties,
)

from .data_types import (
    COLLECTION_NAME_MAX_BYTES,
    IndexedProperties,
    PartitionSchema,
    QueryMatch,
    QueryResult,
    Record,
    VectorStoreAttemptsExhaustedError,
    VectorStorePartitionAlreadyExistsError,
    VectorStorePartitionHandleStaleError,
    VectorStorePartitionSchemaMismatchError,
    indexed_property_names,
    validate_collection_name,
)
from .utils import validate_filter, validate_identifier
from .vector_search_engine import VectorSearchEngine
from .vector_store import VectorStore, VectorStorePartition

logger = logging.getLogger(__name__)

# Consecutive failed mint attempts before the store concludes it is
# re-attempting a persistent database error rather than losing races: a
# uuid collision is a once-in-the-universe event and each race retry
# requires another actor to have changed the registry in the meantime.
_MAX_MINT_ATTEMPTS = 10


class _RegistryInsertRejectedError(Exception):
    """A registry insert was rejected; retry with a fresh incarnation."""


class IndexLoadError(RuntimeError):
    """Raised when a partition's on-disk index file cannot be loaded."""

    def __init__(self, collection: str, partition_key: str, path: Path) -> None:
        """Initialize with the collection, the partition key, and the index file path."""
        self.collection = collection
        self.partition_key = partition_key
        self.path = path
        super().__init__(
            f"Index for partition {partition_key!r} of collection {collection!r} "
            f"at {path} could not be loaded"
        )


class BaseSQLiteVectorStore(DeclarativeBase):
    """Base class for SQLiteVectorStore ORM models."""


class _PartitionRow(BaseSQLiteVectorStore):
    """The registry: one row per live partition, keyed by its collection and key.

    Stores of different collections may share one engine, so the collection
    is part of the key and every read names it. The incarnation is the
    store's own name for this life of the key; records, pending operations,
    the engine and the index file are keyed by it alone.
    """

    __tablename__ = "vector_store_sqlite_pt"

    collection: MappedColumn[str] = mapped_column(
        String(COLLECTION_NAME_MAX_BYTES), primary_key=True
    )
    partition_key: MappedColumn[str] = mapped_column(String(255), primary_key=True)
    incarnation: MappedColumn[UUID] = mapped_column(Uuid, nullable=False, unique=True)
    # The dimensions and declared schema the partition was created under, so
    # a store built with others fails loudly instead of reading columns and
    # vectors that are not there.
    schema_json: MappedColumn[dict[str, JsonValue]] = mapped_column(
        JSON, nullable=False
    )
    # Flips to True after the first successful index save.
    # Once True, the on-disk index file is part of the durable contract:
    # missing or corrupt is treated as an error rather than silently rebuilt empty.
    index_saved: MappedColumn[bool] = mapped_column(
        Boolean, nullable=False, default=False
    )


class _PurgeQueueRow(BaseSQLiteVectorStore):
    """The purge queue: one row per dead partition incarnation.

    Claimed oldest-first by the enqueue stamp. The incarnation identifies
    the rows, log entries and index file to reclaim; the collection says
    which store's table holds the rows, and the logical key is carried for
    forensics.
    """

    __tablename__ = "vector_store_sqlite_gc"

    incarnation: MappedColumn[UUID] = mapped_column(Uuid, primary_key=True)
    collection: MappedColumn[str] = mapped_column(
        String(COLLECTION_NAME_MAX_BYTES), nullable=False
    )
    partition_key: MappedColumn[str] = mapped_column(String(255), nullable=False)
    enqueued_at: MappedColumn[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )

    __table_args__ = (
        Index("vector_store_sqlite_gc__cl_ea", "collection", "enqueued_at"),
    )


class _PendingOperationRow(BaseSQLiteVectorStore):
    """
    Pending partition operations for crash recovery.

    One row per (incarnation, record). New operations replace old ones.
    Lifecycle:
    1. Inserted in the same SQLite transaction as the records table change.
    2. Marked `applied=True` after the search engine processes the operation.
    3. Applied operations are deleted after the search engine is saved to disk.
    4. On startup, all remaining rows (applied or not) of live incarnations
       are replayed; a dead incarnation's rows wait for the purge.

    No foreign key to the registry: registry rows and log rows are
    deliberately decoupled so that partition deletion is a registry write
    and the purge queue reclaims the log asynchronously.
    """

    __tablename__ = "vector_store_sqlite_pd_op"

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
    """Publish a partition's index to disk and trim the operations it holds.

    The order is the whole protocol. The pending log is the only other copy of
    these vectors -- the records table has no vector column -- so an applied row
    may be deleted only once the index that holds it has been published.
    `save` returning is that statement, and no more than that: publication is
    atomic, not durable, so a power failure can revert it after this trim has
    committed. See the module docstring for what that leaves behind.
    """
    # Write index to path.
    await search_engine.save(path)

    # Delete applied pending operations and flip index_saved to True.
    async with create_session() as session, session.begin():
        await session.execute(
            delete(_PendingOperationRow).where(
                _PendingOperationRow.incarnation == incarnation,
                _PendingOperationRow.applied.is_(True),
            )
        )
        await session.execute(
            update(_PartitionRow)
            .where(
                _PartitionRow.incarnation == incarnation,
                _PartitionRow.index_saved.is_(False),
            )
            .values(index_saved=True)
        )


class SQLiteVectorStorePartition(VectorStorePartition):
    """A partition backed by SQLite + a pluggable vector search engine."""

    class _KeyFilter:
        """Per-candidate SQL filter using a sync SQLAlchemy session."""

        def __init__(
            self,
            sync_sqlalchemy_engine: Engine,
            records_table: Table,
            incarnation: UUID,
            filter_expression: ColumnElement[bool],
        ) -> None:
            """Initialize with a sync SQLAlchemy engine, records table, and filter expression."""
            self._sync_sqlalchemy_engine = sync_sqlalchemy_engine
            self._records_table = records_table
            self._incarnation = incarnation
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
                        self._records_table.c.incarnation == self._incarnation,
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
        collection: str,
        partition_key: str,
        incarnation: UUID,
        indexed_properties: Mapping[str, PropertyType],
        index_path: str | None,
        save_threshold: int,
    ) -> None:
        """Initialize a partition handle bound to one incarnation."""
        self._create_session = create_session
        self._sync_sqlalchemy_engine = sync_sqlalchemy_engine
        self._records_table = records_table
        self._search_engine = search_engine

        self._collection = collection
        self._partition_key = partition_key
        # Rows, log entries and the engine are keyed by the incarnation
        # alone: rows of a deleted-and-recreated partition under the same
        # logical key are invisible to the new incarnation while the purge
        # queue reclaims them.
        self._incarnation = incarnation

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

    def _registry_row_query(self) -> Select[tuple[str]]:
        """This incarnation's registry row: absent once the handle is stale."""
        return select(_PartitionRow.partition_key).where(
            _PartitionRow.incarnation == self._incarnation
        )

    async def _ensure_live(self, session: AsyncSession) -> None:
        """Raise if this handle's incarnation is no longer registered.

        Writes call this inside their transaction. Reads call it when their
        data statement returned no rows: it tells an empty partition from a
        stale handle.
        """
        row = (await session.execute(self._registry_row_query())).scalar_one_or_none()
        if row is None:
            raise VectorStorePartitionHandleStaleError(
                self._collection, self._partition_key
            )

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
        if not records:
            return

        async with self._create_session() as session, session.begin():
            await self._ensure_live(session)
            upsert_records = (
                sqlite_insert(self._records_table)
                .on_conflict_do_update(
                    index_elements=[
                        self._records_table.c.incarnation,
                        self._records_table.c.uuid,
                    ],
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
                            "incarnation": self._incarnation,
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

            async with self._create_session() as session, session.begin():
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

        search_results = await self._search_engine.search(
            query_vectors, limit=limit, allowed_keys=key_filter
        )

        results: list[QueryResult] = []
        async with self._create_session() as session:
            for search_result in search_results:
                matches = await self._build_matches(
                    session,
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
            incarnation=self._incarnation,
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
        session: AsyncSession,
        *,
        row_id_to_cosine_similarity: Mapping[int, float],
        min_cosine_similarity: float | None,
    ) -> list[QueryMatch]:
        matched_row_ids = list(row_id_to_cosine_similarity.keys())

        # The rows are read only while the incarnation is registered, in
        # the same statement: a stale handle reads nothing. An empty result
        # is then either an empty partition or a stale handle, and only the
        # registry tells them apart.
        matched_rows = []
        if matched_row_ids:
            fetch_records = select(
                self._records_table.c.uuid, self._records_table.c.row_id
            ).where(
                self._records_table.c.incarnation == self._incarnation,
                self._records_table.c.row_id.in_(matched_row_ids),
                self._registry_row_query().exists(),
            )
            matched_rows = (await session.execute(fetch_records)).all()
        if not matched_rows:
            await self._ensure_live(session)

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

        async with self._create_session() as session, session.begin():
            await self._ensure_live(session)
            rows = (
                await session.execute(
                    select(self._records_table.c.row_id).where(
                        self._records_table.c.incarnation == self._incarnation,
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
                    self._records_table.c.row_id.in_(record_row_ids),
                )
            )

        await self._search_engine.remove(record_row_ids)
        async with self._create_session() as session, session.begin():
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


VectorSearchEngineFactory = Callable[[int], VectorSearchEngine]
"""Callable that creates a VectorSearchEngine given the number of dimensions."""


class SQLiteVectorStoreParams(BaseModel):
    """Parameters for constructing a SQLiteVectorStore.

    Attributes:
        sqlalchemy_engine (AsyncEngine):
            Async SQLAlchemy engine (sqlite+aiosqlite).
        collection (str):
            The collection this store is; names its tables and index files,
            so stores of different collections may share the engine.
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
        purge_max_records (int):
            Maximum number of records purged per call, each with its log
            entry (default: 10000).
        purge_max_partitions (int):
            Maximum number of queue entries a purge call processes. Entries
            cost round trips rather than row deletions, so they carry their
            own bound: a backlog of empty partitions cannot turn one
            bounded call into an unbounded transaction (default: 100).
    """

    sqlalchemy_engine: InstanceOf[AsyncEngine] = Field(
        ..., description="Async SQLAlchemy engine (sqlite+aiosqlite)"
    )
    collection: str = Field(..., description="The collection this store is")
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
    purge_max_records: int = Field(
        10_000,
        gt=0,
        description="Maximum number of records purged per call, each with its log entry",
    )
    purge_max_partitions: int = Field(
        100,
        gt=0,
        description="Maximum number of queue entries a purge call processes",
    )

    @field_validator("collection")
    @classmethod
    def _validate_collection(cls, collection: str) -> str:
        validate_collection_name(collection)
        return collection

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

    One records table per collection, shared by every partition of it under
    the partition's incarnation; one engine instance and index file per
    incarnation. The engine and its index file live in the process that
    opened the partition, so a partition is managed by at most one process
    at a time: an embedded store for one server process, not a shared one.
    """

    def __init__(self, params: SQLiteVectorStoreParams) -> None:
        """Initialize the vector store with the provided parameters."""
        self._sqlalchemy_engine = params.sqlalchemy_engine
        self._collection = params.collection
        self._vector_dimensions = params.vector_dimensions
        self._indexed_properties = params.indexed_properties
        self._vector_search_engine_factory = params.vector_search_engine_factory

        self._index_directory = (
            Path(params.index_directory) if params.index_directory else None
        )
        self._save_threshold = params.save_threshold
        self._purge_max_records = params.purge_max_records
        self._purge_max_partitions = params.purge_max_partitions

        self._create_session = async_sessionmaker(
            self._sqlalchemy_engine, expire_on_commit=False
        )
        self._search_engines: dict[UUID, VectorSearchEngine] = {}
        self._sa_metadata = MetaData()
        self._records_table = self._build_records_table()

        self._sync_sqlalchemy_engine = create_engine(
            str(self._sqlalchemy_engine.url).replace("aiosqlite", "pysqlite")
        )

        for sync_engine in (
            self._sqlalchemy_engine.sync_engine,
            self._sync_sqlalchemy_engine,
        ):
            if not event.contains(sync_engine, "connect", _enable_sqlite_foreign_keys):
                event.listen(sync_engine, "connect", _enable_sqlite_foreign_keys)

        self._started = False

    @property
    @override
    def collection(self) -> str:
        return self._collection

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
            await connection.run_sync(
                self._sa_metadata.create_all, tables=[self._records_table]
            )

    @override
    async def startup(self) -> None:
        if self._started:
            return

        await self._replay_pending_operations()

        self._started = True

    async def _replay_pending_operations(self) -> None:
        """Replay any pending engine operations of this collection's live partitions.

        A dead incarnation's rows are not replayed: nothing can read them,
        and the purge reclaims them with the rest.
        """
        async with self._create_session() as session:
            pending_operations = (
                (
                    await session.execute(
                        select(_PendingOperationRow)
                        .join(
                            _PartitionRow,
                            _PartitionRow.incarnation
                            == _PendingOperationRow.incarnation,
                        )
                        .where(_PartitionRow.collection == self._collection)
                    )
                )
                .scalars()
                .all()
            )

        if not pending_operations:
            return

        operations_by_incarnation: dict[UUID, list[_PendingOperationRow]] = defaultdict(
            list
        )
        for operation in pending_operations:
            operations_by_incarnation[operation.incarnation].append(operation)

        for incarnation, operations in operations_by_incarnation.items():
            await self._replay_partition_operations(incarnation, operations)

    async def _replay_partition_operations(
        self, incarnation: UUID, operations: Iterable[_PendingOperationRow]
    ) -> None:
        async with self._create_session() as session:
            row = await self._registry_row_by_incarnation(session, incarnation)
        if row is None:
            return
        partition_key = row.partition_key
        self._check_schema(partition_key, row.schema_json)

        search_engine = await self._get_or_create_vector_search_engine(
            incarnation, partition_key, index_saved=row.index_saved
        )

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
                await _save_partition_index(
                    create_session=self._create_session,
                    incarnation=incarnation,
                    search_engine=search_engine,
                    path=str(self._index_path(incarnation)),
                )
        self._search_engines.clear()
        self._started = False

    @override
    async def create_partition(self, partition_key: str) -> None:
        self._require_started()
        if not validate_identifier(partition_key):
            raise ValueError(f"Invalid partition key {partition_key!r}")

        attempts = 0
        while True:
            try:
                await self._insert_partition_row(partition_key, uuid4())
            except _RegistryInsertRejectedError as err:
                attempts += 1
                if attempts >= _MAX_MINT_ATTEMPTS:
                    raise VectorStoreAttemptsExhaustedError(
                        f"Creating partition {partition_key!r} of collection "
                        f"{self._collection!r} made no progress after "
                        f"{_MAX_MINT_ATTEMPTS} attempts"
                    ) from err
                continue  # Mint a fresh incarnation.
            return

    async def _insert_partition_row(
        self, partition_key: str, incarnation: UUID
    ) -> None:
        """Insert a registry row for a freshly minted incarnation.

        The registry's unique constraint rejects an incarnation colliding
        with a live one; the in-transaction queue check rejects one whose
        rows still await purge, so rows can never be adopted by, or
        reclaimed out from under, a new partition.

        Raises:
            VectorStorePartitionAlreadyExistsError:
                The partition key is taken.
            _RegistryInsertRejectedError:
                The insert cannot be kept for another reason; retry with a
                fresh incarnation.
        """
        try:
            async with self._create_session() as session, session.begin():
                await session.execute(
                    insert(_PartitionRow).values(
                        collection=self._collection,
                        partition_key=partition_key,
                        incarnation=incarnation,
                        schema_json=self._declared_schema().model_dump(mode="json"),
                    )
                )
                garbage = (
                    await session.execute(
                        select(_PurgeQueueRow.incarnation).where(
                            _PurgeQueueRow.incarnation == incarnation
                        )
                    )
                ).scalar_one_or_none()
                if garbage is not None:
                    logger.warning(
                        "Incarnation %s minted for partition %r of collection %r "
                        "collides with garbage awaiting purge; re-minting",
                        incarnation,
                        partition_key,
                        self._collection,
                    )
                    raise _RegistryInsertRejectedError(str(incarnation))
        except IntegrityError as err:
            async with self._create_session() as session:
                taken = await self._registry_row(session, partition_key)
            if taken is not None:
                raise VectorStorePartitionAlreadyExistsError(
                    self._collection, partition_key
                ) from err
            logger.warning(
                "Registry insert for partition %r of collection %r with "
                "incarnation %s failed and no row exists under the key; "
                "retrying with a fresh incarnation",
                partition_key,
                self._collection,
                incarnation,
            )
            raise _RegistryInsertRejectedError(str(incarnation)) from err

    @override
    async def get_partition(self, partition_key: str) -> VectorStorePartition | None:
        self._require_started()
        if not validate_identifier(partition_key):
            raise ValueError(f"Invalid partition key {partition_key!r}")

        async with self._create_session() as session:
            row = await self._registry_row(session, partition_key)
        if row is None:
            return None
        self._check_schema(partition_key, row.schema_json)

        incarnation = row.incarnation
        index_path = self._index_path(incarnation)
        return SQLiteVectorStorePartition(
            create_session=self._create_session,
            sync_sqlalchemy_engine=self._sync_sqlalchemy_engine,
            records_table=self._records_table,
            search_engine=await self._get_or_create_vector_search_engine(
                incarnation, partition_key, index_saved=row.index_saved
            ),
            collection=self._collection,
            partition_key=partition_key,
            incarnation=incarnation,
            indexed_properties=self._indexed_properties,
            index_path=str(index_path) if index_path is not None else None,
            save_threshold=self._save_threshold,
        )

    @override
    async def delete_partition(self, partition_key: str) -> None:
        self._require_started()
        if not validate_identifier(partition_key):
            raise ValueError(f"Invalid partition key {partition_key!r}")

        # O(1) regardless of partition size: the incarnation goes onto the
        # purge queue and the registry row is deleted. Rows, log and engine
        # become unreachable immediately: every operation resolves the
        # registry first.
        async with self._create_session() as session, session.begin():
            row = await self._registry_row(session, partition_key)
            if row is None:
                return
            incarnation = row.incarnation
            await session.execute(
                insert(_PurgeQueueRow).values(
                    incarnation=incarnation,
                    collection=self._collection,
                    partition_key=partition_key,
                    enqueued_at=func.now(),
                )
            )
            await session.execute(
                delete(_PartitionRow).where(_PartitionRow.incarnation == incarnation)
            )
        self._search_engines.pop(incarnation, None)

    @override
    async def purge_deleted_partitions(self) -> bool:
        self._require_started()
        # Reclaim dead incarnations of this collection oldest-first, within
        # the per-call bounds, in one write transaction: a raise rolls the
        # whole call back, which is what makes it safe to repeat. SQLite has
        # one writer, so concurrent purgers serialize; an entry is retired
        # only when the retirer's own DELETE found fewer rows than its
        # budget, so a doubly-claimed entry costs empty round trips, never
        # duplicated or missed reclamation. The index file goes before the
        # entry: a retired entry never leaves a file behind, and a file
        # missing on a repeated call is the expected state.
        remaining = self._purge_max_records
        entries = 0
        async with self._create_session() as session, session.begin():
            while True:
                incarnation = (
                    await session.execute(
                        select(_PurgeQueueRow.incarnation)
                        .where(_PurgeQueueRow.collection == self._collection)
                        .order_by(_PurgeQueueRow.enqueued_at)
                        .limit(1)
                    )
                ).scalar_one_or_none()
                if incarnation is None:
                    return False

                batch = (
                    select(self._records_table.c.row_id)
                    .where(self._records_table.c.incarnation == incarnation)
                    .limit(remaining)
                    .scalar_subquery()
                )
                deleted = (
                    await (await session.connection()).execute(
                        delete(self._records_table).where(
                            self._records_table.c.incarnation == incarnation,
                            self._records_table.c.row_id.in_(batch),
                        )
                    )
                ).rowcount
                if deleted == remaining:
                    # The bound was consumed exactly; this incarnation may
                    # have more rows, so leave its queue entry for the next
                    # call.
                    return True
                remaining -= deleted

                # The log is bounded by the records it describes, plus the
                # deletes of records already gone: a batch draws on the same
                # budget, count for count.
                log_batch = (
                    select(_PendingOperationRow.record_row_id)
                    .where(_PendingOperationRow.incarnation == incarnation)
                    .limit(remaining)
                    .scalar_subquery()
                )
                purged_log = (
                    await (await session.connection()).execute(
                        delete(_PendingOperationRow).where(
                            _PendingOperationRow.incarnation == incarnation,
                            _PendingOperationRow.record_row_id.in_(log_batch),
                        )
                    )
                ).rowcount
                if purged_log == remaining:
                    return True
                remaining -= purged_log

                self._search_engines.pop(incarnation, None)
                index_path = self._index_path(incarnation)
                if index_path is not None:
                    index_path.unlink(missing_ok=True)
                await session.execute(
                    delete(_PurgeQueueRow).where(
                        _PurgeQueueRow.incarnation == incarnation
                    )
                )
                entries += 1
                if entries >= self._purge_max_partitions:
                    return True

    # Helpers.

    @property
    def _table_prefix(self) -> str:
        collection = self._collection
        return f"vector_store_sqlite_{len(collection)}_{collection}"

    def _build_records_table(self) -> Table:
        """The collection's records table, shared by its partitions."""
        return Table(
            f"{self._table_prefix}_rc",
            self._sa_metadata,
            Column("row_id", Integer, primary_key=True, autoincrement=True),
            Column("incarnation", Uuid, nullable=False),
            Column("uuid", Uuid, nullable=False),
            Column("properties", JSON, nullable=False, default=dict),
            UniqueConstraint("incarnation", "uuid"),
        )

    def _declared_schema(self) -> PartitionSchema:
        return PartitionSchema(
            vector_dimensions=self._vector_dimensions,
            indexed_properties=indexed_property_names(self._indexed_properties),
        )

    def _index_path(self, incarnation: UUID) -> Path | None:
        """Return the on-disk index path for an incarnation, or None if in-memory."""
        if self._index_directory is None:
            return None
        return self._index_directory / f"{self._table_prefix}_{incarnation.hex}.idx"

    async def _registry_row(
        self, session: AsyncSession, partition_key: str
    ) -> _PartitionRow | None:
        return (
            await session.execute(
                select(_PartitionRow).where(
                    _PartitionRow.collection == self._collection,
                    _PartitionRow.partition_key == partition_key,
                )
            )
        ).scalar_one_or_none()

    @staticmethod
    async def _registry_row_by_incarnation(
        session: AsyncSession, incarnation: UUID
    ) -> _PartitionRow | None:
        return (
            await session.execute(
                select(_PartitionRow).where(_PartitionRow.incarnation == incarnation)
            )
        ).scalar_one_or_none()

    def _check_schema(self, partition_key: str, stored: dict[str, JsonValue]) -> None:
        """Raise unless the partition was created under this store's schema."""
        stored_schema = PartitionSchema.model_validate(stored)
        declared_schema = self._declared_schema()
        if stored_schema != declared_schema:
            raise VectorStorePartitionSchemaMismatchError(
                self._collection, partition_key, stored_schema, declared_schema
            )

    async def _get_or_create_vector_search_engine(
        self, incarnation: UUID, partition_key: str, *, index_saved: bool
    ) -> VectorSearchEngine:
        if incarnation in self._search_engines:
            return self._search_engines[incarnation]

        search_engine = self._vector_search_engine_factory(self._vector_dimensions)

        index_path = self._index_path(incarnation)
        if index_path is not None and index_saved:
            # The engine just propagates whatever its backend raises.
            # Wrap any failure as IndexLoadError so callers see one type.
            try:
                await search_engine.load(str(index_path))
            except Exception as e:
                raise IndexLoadError(self._collection, partition_key, index_path) from e

        self._search_engines[incarnation] = search_engine
        return search_engine
