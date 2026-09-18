"""
Vector store backed by SQLite + sqlite-vec.

The store is one collection, held in one records table and one vec0
virtual table named by the collection, so stores of different collections
may share one engine. Every partition of the collection lives in those two
tables under an incarnation the store mints per partition life: the vec0
table's partition key and the records table's row key are the incarnation,
never the caller's key, so a search reads one incarnation's chunks and a
deleted-and-recreated partition never sees its predecessor's rows.

Deleting a partition is a registry write: the incarnation goes onto the
purge queue and the registry row goes, so the partition is unreachable at
once; `purge_deleted_partitions` reclaims the rows afterward, a bounded
batch per call. The registry and the data share the one SQLite file, so
every process on the node holding it may create, use, delete and purge
partitions; a file is not shared across nodes.

The records table carries one typed, indexed column per declared property,
and a filtered query hands the KNN an allowlist of the rows the filter
admits, so the filter is evaluated during the search: vec0 ranks only the
allowed rows, and a filtered search returns fewer only when the filter
admits fewer.
"""

import logging
from collections.abc import Iterable, Mapping, Sequence
from datetime import datetime
from typing import ClassVar, override
from uuid import UUID, uuid4

import aiosqlite
import sqlite_vec
from pydantic import BaseModel, Field, InstanceOf, JsonValue, field_validator
from sqlalchemy import (
    JSON,
    Column,
    DateTime,
    Index,
    Integer,
    MetaData,
    Select,
    String,
    Table,
    UniqueConstraint,
    Uuid,
    column,
    delete,
    event,
    func,
    insert,
    select,
    text,
    update,
)
from sqlalchemy import table as sql_table
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.engine.interfaces import DBAPIConnection
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import DeclarativeBase, MappedColumn, mapped_column
from sqlalchemy.pool import ConnectionPoolEntry, StaticPool
from sqlalchemy.sql.elements import ColumnElement

from memmachine_server.common.data_types import PropertyType
from memmachine_server.common.filter import (
    And,
    Equals,
    FilterExpr,
    In,
    IsNull,
    Not,
    Or,
    Ordering,
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
from .declared_properties import require_declared_properties, require_supported_filter
from .sql_columns import (
    compile_property_filter,
    property_column_name,
    property_column_values,
    property_columns,
    property_indexes,
)
from .utils import validate_identifier
from .vector_store import VectorStore, VectorStorePartition

logger = logging.getLogger(__name__)

# Consecutive failed mint attempts before the store concludes it is
# re-attempting a persistent database error rather than losing races: a
# uuid collision is a once-in-the-universe event and each race retry
# requires another actor to have changed the registry in the meantime.
_MAX_MINT_ATTEMPTS = 10


class _RegistryInsertRejectedError(Exception):
    """A registry insert was rejected; retry with a fresh incarnation."""


class BaseSQLiteVecVectorStore(DeclarativeBase):
    """Base class for SQLiteVecVectorStore ORM models."""


class _PartitionRow(BaseSQLiteVecVectorStore):
    """The registry: one row per live partition, keyed by its collection and key.

    Stores of different collections may share one engine, so the collection
    is part of the key and every read names it. The incarnation is the
    store's own name for this life of the key; rows in the data tables are
    keyed by it alone.
    """

    __tablename__ = "vector_store_sqlite_vec_pt"

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


class _PurgeQueueRow(BaseSQLiteVecVectorStore):
    """The purge queue: one row per dead partition incarnation.

    Claimed oldest-first by the enqueue stamp. The incarnation identifies
    the rows to reclaim; the collection says which store's tables hold them,
    and the logical key is carried for forensics.
    """

    __tablename__ = "vector_store_sqlite_vec_gc"

    incarnation: MappedColumn[UUID] = mapped_column(Uuid, primary_key=True)
    collection: MappedColumn[str] = mapped_column(
        String(COLLECTION_NAME_MAX_BYTES), nullable=False
    )
    partition_key: MappedColumn[str] = mapped_column(String(255), nullable=False)
    enqueued_at: MappedColumn[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )

    __table_args__ = (
        Index("vector_store_sqlite_vec_gc__cl_ea", "collection", "enqueued_at"),
    )


class SQLiteVecVectorStorePartition(VectorStorePartition):
    """A partition backed by SQLite + sqlite-vec: one incarnation's rows in the collection's tables."""

    _SUPPORTED_FILTER_NODES: ClassVar[frozenset[type]] = frozenset(
        {Equals, Ordering, In, IsNull, And, Or, Not}
    )

    def __init__(
        self,
        *,
        create_session: async_sessionmaker[AsyncSession],
        collection: str,
        partition_key: str,
        incarnation: UUID,
        indexed_properties: Mapping[str, PropertyType],
        records_table: Table,
        vector_table_name: str,
    ) -> None:
        """Initialize with the session factory, the incarnation and the tables."""
        self._create_session = create_session
        self._collection = collection
        self._partition_key = partition_key
        self._incarnation = incarnation
        self._indexed_properties = dict(indexed_properties)
        self._records_table = records_table
        self._vector_table_name = vector_table_name

    @property
    @override
    def partition_key(self) -> str:
        return self._partition_key

    @property
    @override
    def indexed_properties(self) -> Mapping[str, PropertyType]:
        return self._indexed_properties

    @property
    @override
    def supported_filter_nodes(self) -> frozenset[type]:
        return SQLiteVecVectorStorePartition._SUPPORTED_FILTER_NODES

    @staticmethod
    def _serialize_vector(vector: Sequence[float]) -> bytes:
        return sqlite_vec.serialize_float32(list(vector))

    @staticmethod
    def _distance_to_cosine_similarity(distance: float) -> float:
        """Convert a sqlite-vec cosine distance to a cosine similarity."""
        return 1.0 - distance

    async def _fence_write(self, session: AsyncSession) -> None:
        """Take the write lock on this incarnation's registry row; raise if the handle is stale.

        SQLite's driver defers BEGIN until the first data-modifying
        statement, so a SELECT-only check would run outside the write
        transaction and fence nothing. A self-checking UPDATE of the
        registry row opens the write transaction, waits out a concurrent
        deletion (which updates the same row), and its match count is the
        staleness check.
        """
        fenced = await (await session.connection()).execute(
            update(_PartitionRow)
            .where(_PartitionRow.incarnation == self._incarnation)
            .values(incarnation=self._incarnation)
        )
        if fenced.rowcount == 0:
            raise VectorStorePartitionHandleStaleError(
                self._collection, self._partition_key
            )

    async def _ensure_live(self, session: AsyncSession) -> None:
        """Raise if this handle's incarnation is no longer registered."""
        live = (
            await session.execute(
                select(_PartitionRow.partition_key).where(
                    _PartitionRow.incarnation == self._incarnation
                )
            )
        ).scalar_one_or_none()
        if live is None:
            raise VectorStorePartitionHandleStaleError(
                self._collection, self._partition_key
            )

    @override
    async def upsert(self, *, records: Iterable[Record]) -> None:
        records = list(records)
        if not records:
            return
        for record in records:
            require_declared_properties(record.properties, self._indexed_properties)

        property_column_names = [
            property_column_name(key) for key in self._indexed_properties
        ]
        insert_records = sqlite_insert(self._records_table)
        upsert_records = insert_records.on_conflict_do_update(
            index_elements=[
                self._records_table.c.incarnation,
                self._records_table.c.uuid,
            ],
            # With no declared column the update is a no-op that still
            # returns the existing row, which `RETURNING` needs.
            set_={
                name: insert_records.excluded[name]
                for name in (property_column_names or ["uuid"])
            },
        ).returning(self._records_table.c.uuid, self._records_table.c.rowid)

        async with self._create_session() as session, session.begin():
            await self._fence_write(session)
            rows = (
                await session.execute(
                    upsert_records,
                    [
                        {
                            "incarnation": self._incarnation,
                            "uuid": record.uuid,
                            **property_column_values(
                                record.properties, self._indexed_properties
                            ),
                        }
                        for record in records
                    ],
                )
            ).all()
            uuid_to_rowid: dict[UUID, int] = {row.uuid: row.rowid for row in rows}

            vector_params = [
                {
                    "rowid": uuid_to_rowid[record.uuid],
                    "incarnation": self._incarnation.hex,
                    "vector": self._serialize_vector(record.vector),
                }
                for record in records
            ]
            await session.execute(
                text(f"DELETE FROM [{self._vector_table_name}] WHERE rowid = :rowid"),
                vector_params,
            )
            await session.execute(
                text(
                    f"INSERT INTO [{self._vector_table_name}]"
                    f"(rowid, incarnation, vector) "
                    f"VALUES (:rowid, :incarnation, :vector)"
                ),
                vector_params,
            )

    # sqlite-vec hard-caps k at 4096; larger values raise an OperationalError.
    _MAX_K: ClassVar[int] = 4096

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

        filter_expression: ColumnElement[bool] | None = None
        if property_filter is not None:
            require_supported_filter(
                property_filter,
                self._indexed_properties,
                SQLiteVecVectorStorePartition._SUPPORTED_FILTER_NODES,
            )
            filter_expression = compile_property_filter(
                property_filter, self._records_table, self._indexed_properties
            )

        k = min(limit, self._MAX_K)

        results: list[QueryResult] = []
        async with self._create_session() as session:
            for query_vector in query_vectors:
                knn_rows = (
                    await session.execute(
                        self._knn_statement(
                            self._serialize_vector(query_vector), k, filter_expression
                        )
                    )
                ).all()
                if not knn_rows:
                    # An empty partition or a stale handle; only the
                    # registry tells them apart.
                    await self._ensure_live(session)

                rowid_to_distance: dict[int, float] = {
                    row.rowid: row.distance for row in knn_rows
                }
                matches = await self._build_matches(
                    session=session,
                    rowid_to_distance=rowid_to_distance,
                    min_cosine_similarity=min_cosine_similarity,
                )
                results.append(QueryResult(matches=matches))

        return results

    def _knn_statement(
        self,
        query_blob: bytes,
        k: int,
        filter_expression: ColumnElement[bool] | None,
    ) -> Select:
        """The KNN over this incarnation's chunks, restricted to the rows a filter admits.

        The vec0 partition key narrows the search to the incarnation, and
        the registry check makes a stale handle read nothing, in the same
        statement, at no extra round trip. vec0 takes a `rowid IN (...)`
        constraint into the search itself, so the `k` nearest are the
        nearest among the admitted rows.
        """
        statement = (
            select(column("rowid"), column("distance"))
            .select_from(sql_table(self._vector_table_name))
            .where(
                text("vector MATCH :query AND k = :k").bindparams(
                    query=query_blob, k=k
                ),
                # The vec0 column holds the text the Uuid column stores: 32
                # hex digits, no hyphens.
                column("incarnation") == self._incarnation.hex,
                select(_PartitionRow.incarnation)
                .where(_PartitionRow.incarnation == self._incarnation)
                .exists(),
            )
        )
        if filter_expression is not None:
            statement = statement.where(
                column("rowid").in_(
                    select(self._records_table.c.rowid).where(
                        self._records_table.c.incarnation == self._incarnation,
                        filter_expression,
                    )
                )
            )
        return statement.order_by(column("distance"))

    async def _build_matches(
        self,
        session: AsyncSession,
        rowid_to_distance: Mapping[int, float],
        min_cosine_similarity: float | None,
    ) -> list[QueryMatch]:
        if not rowid_to_distance:
            return []

        matched_rows = (
            await session.execute(
                select(self._records_table.c.uuid, self._records_table.c.rowid).where(
                    self._records_table.c.incarnation == self._incarnation,
                    self._records_table.c.rowid.in_(list(rowid_to_distance)),
                )
            )
        ).all()

        matches: list[QueryMatch] = []
        for row in matched_rows:
            cosine_similarity = (
                SQLiteVecVectorStorePartition._distance_to_cosine_similarity(
                    rowid_to_distance[row.rowid]
                )
            )
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
        record_uuids = list(record_uuids)
        if not record_uuids:
            return

        async with self._create_session() as session, session.begin():
            await self._fence_write(session)
            rows = (
                await session.execute(
                    select(self._records_table.c.rowid).where(
                        self._records_table.c.incarnation == self._incarnation,
                        self._records_table.c.uuid.in_(record_uuids),
                    )
                )
            ).all()
            if not rows:
                return

            record_rowids = [row.rowid for row in rows]

            placeholders = ", ".join(
                f":r{index}" for index in range(len(record_rowids))
            )
            await session.execute(
                text(
                    f"DELETE FROM [{self._vector_table_name}] WHERE rowid IN ({placeholders})"
                ),
                {f"r{index}": row_id for index, row_id in enumerate(record_rowids)},
            )

            await session.execute(
                delete(self._records_table).where(
                    self._records_table.c.rowid.in_(record_rowids),
                )
            )


class SQLiteVecVectorStoreParams(BaseModel):
    """
    Parameters for constructing a SQLiteVecVectorStore.

    Attributes:
        engine (AsyncEngine): Async SQLAlchemy engine (sqlite+aiosqlite).
        collection (str):
            The collection this store is; names its tables, so stores of
            different collections may share the engine.
        vector_dimensions (int):
            Dimensionality of every vector in the store.
        indexed_properties (IndexedProperties):
            The declared schema every partition of this store carries: each
            key is a typed, indexed column of the partition's records table,
            and a record or a filter naming any other key is rejected.
        purge_max_records (int):
            Maximum number of records purged per call, each with its vector
            (default: 10000).
        purge_max_partitions (int):
            Maximum number of queue entries a purge call processes. Entries
            cost round trips rather than row deletions, so they carry their
            own bound: a backlog of empty partitions cannot turn one
            bounded call into an unbounded transaction (default: 100).
    """

    engine: InstanceOf[AsyncEngine] = Field(
        ...,
        description="Async SQLAlchemy engine (sqlite+aiosqlite)",
    )
    collection: str = Field(..., description="The collection this store is")
    vector_dimensions: int = Field(
        ..., gt=0, description="Dimensionality of every vector in the store"
    )
    indexed_properties: IndexedProperties = Field(
        ...,
        description="The declared schema every partition of this store carries",
    )
    purge_max_records: int = Field(
        10_000,
        gt=0,
        description="Maximum number of records purged per call, each with its vector",
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

    @field_validator("engine")
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


class SQLiteVecVectorStore(VectorStore):
    """
    Vector store backed by SQLite + sqlite-vec.

    One records table and one vec0 virtual table per collection, shared by
    every partition of it under the partition's incarnation.
    """

    _SQLITE_VEC_DISTANCE_METRIC: ClassVar[str] = "cosine"

    def __init__(self, params: SQLiteVecVectorStoreParams) -> None:
        """Initialize the vector store with the provided parameters."""
        self._engine = params.engine
        self._collection = params.collection
        self._vector_dimensions = params.vector_dimensions
        self._indexed_properties = params.indexed_properties
        self._purge_max_records = params.purge_max_records
        self._purge_max_partitions = params.purge_max_partitions
        self._create_session = async_sessionmaker(self._engine, expire_on_commit=False)
        self._sa_metadata = MetaData()
        self._records_table = self._build_records_table()

        @event.listens_for(self._engine.sync_engine, "connect")
        def _load_sqlite_vec(
            dbapi_connection: DBAPIConnection,
            _connection_record: ConnectionPoolEntry,
        ) -> None:
            async def _load_extension(
                aio_connection: aiosqlite.Connection,
            ) -> None:
                await aio_connection.enable_load_extension(True)
                await aio_connection.load_extension(sqlite_vec.loadable_path())
                await aio_connection.enable_load_extension(False)

            dbapi_connection.run_async(_load_extension)

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

    @override
    async def provision(self) -> None:
        async with self._engine.begin() as connection:
            await connection.run_sync(BaseSQLiteVecVectorStore.metadata.create_all)
            await connection.run_sync(
                self._sa_metadata.create_all, tables=[self._records_table]
            )
            await connection.execute(
                text(
                    f"CREATE VIRTUAL TABLE IF NOT EXISTS [{self._vector_table_name}] "
                    f"USING vec0("
                    f"incarnation text partition key, "
                    f"vector float[{self._vector_dimensions}] "
                    f"distance_metric={SQLiteVecVectorStore._SQLITE_VEC_DISTANCE_METRIC}"
                    f")"
                )
            )

    @override
    async def startup(self) -> None:
        pass

    @override
    async def shutdown(self) -> None:
        pass

    @override
    async def create_partition(self, partition_key: str) -> None:
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
        if not validate_identifier(partition_key):
            raise ValueError(f"Invalid partition key {partition_key!r}")

        async with self._create_session() as session:
            row = await self._registry_row(session, partition_key)
        if row is None:
            return None
        self._check_schema(partition_key, row.schema_json)

        return SQLiteVecVectorStorePartition(
            create_session=self._create_session,
            collection=self._collection,
            partition_key=partition_key,
            incarnation=row.incarnation,
            indexed_properties=self._indexed_properties,
            records_table=self._records_table,
            vector_table_name=self._vector_table_name,
        )

    @override
    async def delete_partition(self, partition_key: str) -> None:
        if not validate_identifier(partition_key):
            raise ValueError(f"Invalid partition key {partition_key!r}")

        # O(1) regardless of partition size: the registry row's write lock
        # waits out in-flight writers, the incarnation goes onto the purge
        # queue, and the registry row is deleted. Rows become unreachable
        # immediately: every operation resolves the registry first.
        async with self._create_session() as session, session.begin():
            # The self-checking UPDATE opens the write transaction so racing
            # deletions serialize instead of both enqueueing; no matched row
            # is the idempotent no-op case.
            pinned = await (await session.connection()).execute(
                update(_PartitionRow)
                .where(
                    _PartitionRow.collection == self._collection,
                    _PartitionRow.partition_key == partition_key,
                )
                .values(partition_key=partition_key)
                .returning(_PartitionRow.incarnation)
            )
            incarnation = pinned.scalar_one_or_none()
            if incarnation is None:
                return

            await session.execute(
                insert(_PurgeQueueRow).values(
                    incarnation=incarnation,
                    collection=self._collection,
                    partition_key=partition_key,
                    enqueued_at=func.now(),
                )
            )
            await session.execute(
                delete(_PartitionRow).where(
                    _PartitionRow.collection == self._collection,
                    _PartitionRow.partition_key == partition_key,
                )
            )

    @override
    async def purge_deleted_partitions(self) -> bool:
        # Reclaim dead incarnations of this collection oldest-first, within
        # the per-call bounds, in one transaction: a raise rolls the whole
        # call back, which is what makes it safe to repeat. SQLite has one
        # writer, so concurrent purgers serialize at the first DELETE; a
        # doubly-claimed entry costs empty round trips, never duplicated or
        # missed reclamation, since an entry is retired only when the
        # retirer's own DELETE found fewer rows than its budget.
        remaining = self._purge_max_records
        entries = 0
        async with self._engine.begin() as connection:
            while True:
                incarnation = (
                    await connection.execute(
                        select(_PurgeQueueRow.incarnation)
                        .where(_PurgeQueueRow.collection == self._collection)
                        .order_by(_PurgeQueueRow.enqueued_at)
                        .limit(1)
                    )
                ).scalar_one_or_none()
                if incarnation is None:
                    return False

                batch = [
                    row.rowid
                    for row in (
                        await connection.execute(
                            select(self._records_table.c.rowid)
                            .where(self._records_table.c.incarnation == incarnation)
                            .limit(remaining)
                        )
                    ).all()
                ]
                if batch:
                    placeholders = ", ".join(
                        f":r{index}" for index in range(len(batch))
                    )
                    await connection.execute(
                        text(
                            f"DELETE FROM [{self._vector_table_name}] "
                            f"WHERE rowid IN ({placeholders})"
                        ),
                        {f"r{index}": row_id for index, row_id in enumerate(batch)},
                    )
                    await connection.execute(
                        delete(self._records_table).where(
                            self._records_table.c.rowid.in_(batch)
                        )
                    )
                if len(batch) == remaining:
                    # The bound was consumed exactly; this incarnation may
                    # have more rows, so leave its queue entry for the
                    # next call.
                    return True
                remaining -= len(batch)

                await connection.execute(
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
        return f"vector_store_sqlite_vec_{len(collection)}_{collection}"

    @property
    def _vector_table_name(self) -> str:
        return f"{self._table_prefix}_vc"

    def _build_records_table(self) -> Table:
        records_table = Table(
            f"{self._table_prefix}_rc",
            self._sa_metadata,
            Column("rowid", Integer, primary_key=True, autoincrement=True),
            Column("incarnation", Uuid, nullable=False),
            Column("uuid", Uuid, nullable=False),
            *property_columns(self._indexed_properties),
            UniqueConstraint("incarnation", "uuid"),
        )
        property_indexes(records_table, self._indexed_properties)
        return records_table

    def _declared_schema(self) -> PartitionSchema:
        return PartitionSchema(
            vector_dimensions=self._vector_dimensions,
            indexed_properties=indexed_property_names(self._indexed_properties),
        )

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

    def _check_schema(self, partition_key: str, stored: dict[str, JsonValue]) -> None:
        """Raise unless the partition was created under this store's schema."""
        stored_schema = PartitionSchema.model_validate(stored)
        declared_schema = self._declared_schema()
        if stored_schema != declared_schema:
            raise VectorStorePartitionSchemaMismatchError(
                self._collection, partition_key, stored_schema, declared_schema
            )
