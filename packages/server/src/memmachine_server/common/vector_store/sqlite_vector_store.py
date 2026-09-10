"""
Vector store backed by SQLite + pluggable vector search engine.

Each logical collection gets its own records table and vector search engine.
A pending operations table tracks search engine operations for crash recovery:
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

import heapq
import logging
from collections import defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
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
    delete,
    event,
    func,
    select,
    update,
)
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.engine.interfaces import DBAPIConnection
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import DeclarativeBase, MappedColumn, mapped_column
from sqlalchemy.pool import ConnectionPoolEntry, StaticPool
from sqlalchemy.sql.elements import ColumnElement

from memmachine_server.common.filter.filter_parser import FilterExpr
from memmachine_server.common.filter.sql_filter_util import compile_sql_filter
from memmachine_server.common.properties_json import (
    encode_properties,
)

from .data_types import (
    QueryMatch,
    QueryResult,
    Record,
    VectorStoreCollectionAlreadyExistsError,
    VectorStoreCollectionConfig,
    VectorStoreCollectionConfigMismatchError,
)
from .utils import validate_filter, validate_identifier
from .vector_search_engine import VectorSearchEngine
from .vector_store import VectorStore, VectorStoreCollection

logger = logging.getLogger(__name__)


class IndexLoadError(RuntimeError):
    """Raised when a collection's on-disk index file cannot be loaded."""

    def __init__(self, namespace: str, name: str, path: Path) -> None:
        """Initialize with the collection namespace, name, and index file path."""
        self.namespace = namespace
        self.name = name
        self.path = path
        super().__init__(
            f"Index for collection ({namespace!r}, {name!r}) "
            f"at {path} could not be loaded"
        )


class BaseSQLiteVectorStore(DeclarativeBase):
    """Base class for SQLiteVectorStore ORM models."""


class _CollectionRow(BaseSQLiteVectorStore):
    __tablename__ = "vector_store_sqlite_cl"

    namespace: MappedColumn[str] = mapped_column(String(255), primary_key=True)
    name: MappedColumn[str] = mapped_column(String(255), primary_key=True)
    config_json: MappedColumn[dict[str, JsonValue]] = mapped_column(
        JSON, nullable=False
    )
    # Flips to True after the first successful index save.
    # Once True, the on-disk index file is part of the durable contract:
    # missing or corrupt is treated as an error rather than silently rebuilt empty.
    index_saved: MappedColumn[bool] = mapped_column(
        Boolean, nullable=False, default=False
    )


class _PendingOperationRow(BaseSQLiteVectorStore):
    """
    Pending collection operations for crash recovery.

    One row per (namespace, name, record). New operations replace old ones.
    Lifecycle:
    1. Inserted in the same SQLite transaction as the records table change.
    2. Marked `applied=True` after the search engine processes the operation.
    3. Applied operations are deleted after the search engine is saved to disk.
    4. On startup, all remaining rows (applied or not) are replayed.
    """

    __tablename__ = "vector_store_sqlite_pd_op"
    __table_args__ = (
        ForeignKeyConstraint(
            ["namespace", "name"],
            [
                f"{_CollectionRow.__tablename__}.namespace",
                f"{_CollectionRow.__tablename__}.name",
            ],
            ondelete="CASCADE",
        ),
    )

    namespace: MappedColumn[str] = mapped_column(String(255), primary_key=True)
    name: MappedColumn[str] = mapped_column(String(255), primary_key=True)
    record_row_id: MappedColumn[int] = mapped_column(Integer, primary_key=True)
    operation_type: MappedColumn[str] = mapped_column(
        String(8), nullable=False
    )  # "upsert" or "delete"
    vector: MappedColumn[bytes | None] = mapped_column(LargeBinary, nullable=True)
    applied: MappedColumn[bool] = mapped_column(Boolean, nullable=False, default=False)


async def _save_collection_index(
    *,
    create_session: async_sessionmaker[AsyncSession],
    namespace: str,
    name: str,
    search_engine: VectorSearchEngine,
    path: str,
) -> None:
    """Publish a collection's index to disk and trim the operations it holds.

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
                _PendingOperationRow.namespace == namespace,
                _PendingOperationRow.name == name,
                _PendingOperationRow.applied.is_(True),
            )
        )
        await session.execute(
            update(_CollectionRow)
            .where(
                _CollectionRow.namespace == namespace,
                _CollectionRow.name == name,
                _CollectionRow.index_saved.is_(False),
            )
            .values(index_saved=True)
        )


_OVERFETCH_BASE = 4


class SQLiteVectorStoreCollection(VectorStoreCollection):
    """A logical collection backed by SQLite + a pluggable vector search engine."""

    def __init__(
        self,
        *,
        create_session: async_sessionmaker[AsyncSession],
        records_table: Table,
        search_engine: VectorSearchEngine,
        namespace: str,
        name: str,
        config: VectorStoreCollectionConfig,
        index_path: str | None,
        save_threshold: int,
        selective_filter_limit: int,
        max_overfetch_factor: int,
    ) -> None:
        """Initialize a collection handle."""
        self._create_session = create_session
        self._records_table = records_table
        self._search_engine = search_engine

        self._namespace = namespace
        self._name = name

        self._config = config

        self._index_path = index_path
        self._save_threshold = save_threshold
        self._selective_filter_limit = selective_filter_limit
        self._max_overfetch_factor = max_overfetch_factor

    @property
    @override
    def config(self) -> VectorStoreCollectionConfig:
        return self._config

    async def _maybe_save_index(self) -> None:
        """Save the index to disk if applied pending operations exceed the threshold."""
        if self._index_path is None:
            return

        async with self._create_session() as session:
            count = (
                await session.execute(
                    select(func.count()).where(
                        _PendingOperationRow.namespace == self._namespace,
                        _PendingOperationRow.name == self._name,
                        _PendingOperationRow.applied.is_(True),
                    )
                )
            ).scalar_one()

        if count >= self._save_threshold:
            await _save_collection_index(
                create_session=self._create_session,
                namespace=self._namespace,
                name=self._name,
                search_engine=self._search_engine,
                path=self._index_path,
            )

    @override
    async def upsert(self, *, records: Iterable[Record]) -> None:
        records = list(records)
        if not records:
            return

        for record in records:
            if record.vector is None:
                raise ValueError(
                    f"Record {record.uuid} has vector=None, which is not allowed on input."
                )

        async with self._create_session() as session, session.begin():
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
                    "namespace": self._namespace,
                    "name": self._name,
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
                        index_elements=["namespace", "name", "record_row_id"],
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
                        _PendingOperationRow.namespace == self._namespace,
                        _PendingOperationRow.name == self._name,
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

        allowlist: list[int] | None = None
        if property_filter is not None:
            if not validate_filter(property_filter):
                raise ValueError("Filter contains invalid field names")

            filter_expression = compile_sql_filter(
                property_filter,
                lambda field: (
                    self._records_table.c.properties[field],
                    "properties_json",
                ),
            )
            # One row past the threshold is what separates "at most this
            # many" from "more than this many".
            matching_row_ids = await self._matching_row_ids(
                filter_expression, limit=self._selective_filter_limit + 1
            )
            if len(matching_row_ids) > self._selective_filter_limit:
                return [
                    await self._query_broad(
                        query_vector, filter_expression, limit, min_cosine_similarity
                    )
                    for query_vector in query_vectors
                ]
            allowlist = matching_row_ids

        search_results = await self._search_engine.search(
            query_vectors, limit=limit, allowlist=allowlist
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

    async def _matching_row_ids(
        self, filter_expression: ColumnElement[bool], *, limit: int
    ) -> list[int]:
        """The row_ids matching the filter, at most `limit` of them.

        Enumerating up to a bound costs no more than counting would.
        """
        async with self._create_session() as session:
            rows = (
                await session.execute(
                    select(self._records_table.c.row_id)
                    .where(filter_expression)
                    .limit(limit)
                )
            ).all()
        return [row.row_id for row in rows]

    async def _query_broad(
        self,
        query_vector: Sequence[float],
        filter_expression: ColumnElement[bool],
        limit: int,
        min_cosine_similarity: float | None,
    ) -> QueryResult:
        """
        Post-filter: search unrestricted, keep survivors, widen as needed.

        Widens the fetch until `limit` results survive the filter, the index
        is exhausted, or the fetch reaches `limit * max_overfetch_factor` --
        at the cap the query returns what survived, which may be fewer than
        `limit` results.
        """
        max_fetch = limit * self._max_overfetch_factor
        fetch_limit = min(limit * _OVERFETCH_BASE, max_fetch)
        while True:
            [search_result] = await self._search_engine.search(
                [query_vector], limit=fetch_limit
            )
            row_id_to_cosine_similarity = {
                match.key: match.cosine_similarity for match in search_result.matches
            }

            surviving_row_ids: set[int] = set()
            if row_id_to_cosine_similarity:
                async with self._create_session() as session:
                    surviving_row_ids = set(
                        (
                            await session.execute(
                                select(self._records_table.c.row_id).where(
                                    self._records_table.c.row_id.in_(
                                        row_id_to_cosine_similarity
                                    ),
                                    filter_expression,
                                )
                            )
                        ).scalars()
                    )

            exhausted = len(search_result.matches) < fetch_limit
            if len(surviving_row_ids) >= limit or exhausted or fetch_limit >= max_fetch:
                surviving = {
                    row_id: similarity
                    for row_id, similarity in row_id_to_cosine_similarity.items()
                    if row_id in surviving_row_ids
                }
                top = heapq.nlargest(limit, surviving.items(), key=lambda item: item[1])
                return QueryResult(
                    matches=await self._build_matches(
                        row_id_to_cosine_similarity=dict(top),
                        min_cosine_similarity=min_cosine_similarity,
                    )
                )

            fetch_limit = min(fetch_limit * _OVERFETCH_BASE, max_fetch)

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
                    index_elements=["namespace", "name", "record_row_id"],
                    set_={
                        "operation_type": upsert_pending_operation.excluded.operation_type,
                        "applied": upsert_pending_operation.excluded.applied,
                    },
                ),
                [
                    {
                        "namespace": self._namespace,
                        "name": self._name,
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
                    _PendingOperationRow.namespace == self._namespace,
                    _PendingOperationRow.name == self._name,
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
        vector_search_engine_factory (Callable[[int], VectorSearchEngine]):
            Factory for creating :class:`VectorSearchEngine` instances.
            Receives `(ndim)` and returns a search engine.
        index_directory (str | None):
            Directory for persisting index files.
            If None, indexes are in-memory only
            (default: None).
        save_threshold (int):
            Number of engine operations before auto-saving the index to disk.
            Only applies when index_directory is set
            (default: 1000).
        selective_filter_limit (int):
            Filters matching at most this many records are pre-filtered into
            an allowlist. Broader ones post-filter an unrestricted search
            (default: 10000).
        max_overfetch_factor (int):
            Cap on post-filter widening, as a multiple of `limit`.
            A broad query may return fewer than `limit` results
            (default: 64).
    """

    sqlalchemy_engine: InstanceOf[AsyncEngine] = Field(
        ..., description="Async SQLAlchemy engine (sqlite+aiosqlite)"
    )
    vector_search_engine_factory: VectorSearchEngineFactory = Field(
        ...,
        description=(
            "Factory for creating VectorSearchEngine instances. "
            "Receives `(ndim)` and returns a search engine"
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
    selective_filter_limit: int = Field(
        10_000,
        ge=0,
        description=(
            "Filters matching at most this many records are pre-filtered into "
            "an allowlist. Broader ones post-filter an unrestricted search"
        ),
    )
    max_overfetch_factor: int = Field(
        64,
        ge=1,
        description=(
            "Cap on post-filter widening, as a multiple of `limit`. "
            "A broad query may return fewer than `limit` results"
        ),
    )

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

    Each logical collection gets its own records table and engine instance.
    """

    def __init__(self, params: SQLiteVectorStoreParams) -> None:
        """Initialize the vector store with the provided parameters."""
        self._sqlalchemy_engine = params.sqlalchemy_engine
        self._vector_search_engine_factory = params.vector_search_engine_factory

        self._index_directory = (
            Path(params.index_directory) if params.index_directory else None
        )
        self._save_threshold = params.save_threshold
        self._selective_filter_limit = params.selective_filter_limit
        self._max_overfetch_factor = params.max_overfetch_factor

        self._create_session = async_sessionmaker(
            self._sqlalchemy_engine, expire_on_commit=False
        )
        self._search_engines: dict[tuple[str, str], VectorSearchEngine] = {}
        self._sa_metadata = MetaData()

        @event.listens_for(self._sqlalchemy_engine.sync_engine, "connect")
        def _enable_sqlite_foreign_keys(
            dbapi_connection: DBAPIConnection,
            _connection_record: ConnectionPoolEntry,
        ) -> None:
            cursor = dbapi_connection.cursor()
            cursor.execute("PRAGMA foreign_keys=ON")
            cursor.close()

        self._started = False

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
        """Replay any pending engine operations."""
        async with self._create_session() as session:
            pending_operations = (
                (await session.execute(select(_PendingOperationRow))).scalars().all()
            )

        if not pending_operations:
            return

        operations_by_collection: dict[tuple[str, str], list[_PendingOperationRow]] = (
            defaultdict(list)
        )
        for operation in pending_operations:
            operations_by_collection[(operation.namespace, operation.name)].append(
                operation
            )

        for (namespace, name), operations in operations_by_collection.items():
            await self._replay_collection_operations(namespace, name, operations)

    async def _replay_collection_operations(
        self, namespace: str, name: str, operations: Iterable[_PendingOperationRow]
    ) -> None:
        async with self._create_session() as session:
            config = await self._get_stored_config(session, namespace, name)
        if config is None:
            return

        search_engine = await self._get_or_create_vector_search_engine(
            namespace, name, config
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
                    _PendingOperationRow.namespace == namespace,
                    _PendingOperationRow.name == name,
                    _PendingOperationRow.record_row_id.in_(all_row_ids),
                )
                .values(applied=True)
            )

    @override
    async def shutdown(self) -> None:
        self._require_started()
        if self._index_directory is not None:
            for (namespace, name), search_engine in self._search_engines.items():
                path = self._index_path(namespace, name)
                assert path is not None
                await _save_collection_index(
                    create_session=self._create_session,
                    namespace=namespace,
                    name=name,
                    search_engine=search_engine,
                    path=str(path),
                )
        self._search_engines.clear()
        self._started = False

    @override
    async def create_collection(
        self,
        *,
        namespace: str,
        name: str,
        config: VectorStoreCollectionConfig,
    ) -> None:
        self._require_started()
        if not validate_identifier(namespace) or not validate_identifier(name):
            raise ValueError(f"Invalid namespace {namespace!r} or name {name!r}")

        async with self._create_session() as session, session.begin():
            existing_config = await self._get_stored_config(session, namespace, name)
            if existing_config is not None:
                raise VectorStoreCollectionAlreadyExistsError(namespace, name)

            self._clear_search_engine_state(namespace, name)
            await self._ensure_collection_resources(session, namespace, name, config)
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
        config: VectorStoreCollectionConfig,
    ) -> VectorStoreCollection:
        self._require_started()
        if not validate_identifier(namespace) or not validate_identifier(name):
            raise ValueError(f"Invalid namespace {namespace!r} or name {name!r}")

        index_path = self._index_path(namespace, name)

        async with self._create_session() as session, session.begin():
            existing_config = await self._get_stored_config(session, namespace, name)
            if existing_config is not None:
                if existing_config != config:
                    raise VectorStoreCollectionConfigMismatchError(
                        namespace, name, existing_config, config
                    )
                records_table, search_engine = await self._ensure_collection_resources(
                    session, namespace, name, existing_config
                )
                return SQLiteVectorStoreCollection(
                    create_session=self._create_session,
                    records_table=records_table,
                    search_engine=search_engine,
                    namespace=namespace,
                    name=name,
                    config=existing_config,
                    index_path=str(index_path) if index_path is not None else None,
                    save_threshold=self._save_threshold,
                    selective_filter_limit=self._selective_filter_limit,
                    max_overfetch_factor=self._max_overfetch_factor,
                )

            self._clear_search_engine_state(namespace, name)
            records_table, search_engine = await self._ensure_collection_resources(
                session, namespace, name, config
            )
            session.add(
                _CollectionRow(
                    namespace=namespace,
                    name=name,
                    config_json=config.model_dump(mode="json"),
                )
            )

        return SQLiteVectorStoreCollection(
            create_session=self._create_session,
            records_table=records_table,
            search_engine=search_engine,
            namespace=namespace,
            name=name,
            config=config,
            index_path=str(index_path) if index_path is not None else None,
            save_threshold=self._save_threshold,
            selective_filter_limit=self._selective_filter_limit,
            max_overfetch_factor=self._max_overfetch_factor,
        )

    @override
    async def open_collection(
        self,
        *,
        namespace: str,
        name: str,
    ) -> VectorStoreCollection | None:
        self._require_started()
        if not validate_identifier(namespace) or not validate_identifier(name):
            raise ValueError(f"Invalid namespace {namespace!r} or name {name!r}")

        async with self._create_session() as session:
            existing = await self._get_stored_config(session, namespace, name)
        if existing is None:
            return None

        records_table = self._records_table(namespace, name)
        search_engine = await self._get_or_create_vector_search_engine(
            namespace, name, existing
        )

        index_path = self._index_path(namespace, name)
        return SQLiteVectorStoreCollection(
            create_session=self._create_session,
            records_table=records_table,
            search_engine=search_engine,
            namespace=namespace,
            name=name,
            config=existing,
            index_path=str(index_path) if index_path is not None else None,
            save_threshold=self._save_threshold,
            selective_filter_limit=self._selective_filter_limit,
            max_overfetch_factor=self._max_overfetch_factor,
        )

    @override
    async def close_collection(self, *, collection: VectorStoreCollection) -> None:
        self._require_started()

    @override
    async def delete_collection(self, *, namespace: str, name: str) -> None:
        self._require_started()
        if not validate_identifier(namespace) or not validate_identifier(name):
            raise ValueError(f"Invalid namespace {namespace!r} or name {name!r}")

        async with self._create_session() as session:
            existing = await self._get_stored_config(session, namespace, name)
        if existing is None:
            return

        records_table = self._records_table(namespace, name)
        async with self._create_session() as session, session.begin():
            connection = await session.connection()
            await connection.run_sync(
                self._sa_metadata.drop_all, tables=[records_table]
            )

            await session.execute(
                delete(_CollectionRow).where(
                    _CollectionRow.namespace == namespace,
                    _CollectionRow.name == name,
                )
            )

        self._sa_metadata.remove(records_table)

        # If unlink fails, the orphan is harmless.
        # _clear_search_engine_state will clean it up if a new collection with the same name is created.
        index_path = self._index_path(namespace, name)
        if index_path is not None and index_path.exists():
            index_path.unlink()
        self._search_engines.pop((namespace, name), None)

    # Helpers.

    @staticmethod
    def _collection_prefix(namespace: str, name: str) -> str:
        """Unique prefix for a logical collection's native resources."""
        return f"vector_store_sqlite_{len(namespace)}_{namespace}_{len(name)}_{name}"

    def _records_table(self, namespace: str, name: str) -> Table:
        """Get or create a SQLAlchemy Table for a per-collection records table."""
        return Table(
            f"{self._collection_prefix(namespace, name)}_rc",
            self._sa_metadata,
            Column("row_id", Integer, primary_key=True, autoincrement=True),
            Column("uuid", Uuid, nullable=False, unique=True),
            Column("properties", JSON, nullable=False, default=dict),
            extend_existing=True,
        )

    def _index_path(self, namespace: str, name: str) -> Path | None:
        """Return the on-disk index path for a collection, or None if in-memory."""
        if self._index_directory is None:
            return None
        return self._index_directory / f"{self._collection_prefix(namespace, name)}.idx"

    def _clear_search_engine_state(self, namespace: str, name: str) -> None:
        """Remove any in-memory engine and on-disk index for a collection."""
        self._search_engines.pop((namespace, name), None)
        index_path = self._index_path(namespace, name)
        if index_path is not None and index_path.exists():
            index_path.unlink()

    async def _get_stored_config(
        self,
        session: AsyncSession,
        namespace: str,
        name: str,
    ) -> VectorStoreCollectionConfig | None:
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
        return VectorStoreCollectionConfig.model_validate(row)

    async def _get_or_create_vector_search_engine(
        self,
        namespace: str,
        name: str,
        config: VectorStoreCollectionConfig,
    ) -> VectorSearchEngine:
        cache_key = (namespace, name)
        if cache_key in self._search_engines:
            return self._search_engines[cache_key]

        search_engine = self._vector_search_engine_factory(config.vector_dimensions)

        index_path = self._index_path(namespace, name)
        if index_path is not None:
            async with self._create_session() as session:
                saved = (
                    await session.execute(
                        select(_CollectionRow.index_saved).where(
                            _CollectionRow.namespace == namespace,
                            _CollectionRow.name == name,
                        )
                    )
                ).scalar_one_or_none()

            if saved:
                # The engine just propagates whatever its backend raises.
                # Wrap any failure as IndexLoadError so callers see one type.
                try:
                    await search_engine.load(str(index_path))
                except Exception as e:
                    raise IndexLoadError(namespace, name, index_path) from e

        self._search_engines[cache_key] = search_engine
        return search_engine

    async def _ensure_collection_resources(
        self,
        session: AsyncSession,
        namespace: str,
        name: str,
        config: VectorStoreCollectionConfig,
    ) -> tuple[Table, VectorSearchEngine]:
        records_table = self._records_table(namespace, name)
        search_engine = await self._get_or_create_vector_search_engine(
            namespace, name, config
        )

        connection = await session.connection()
        await connection.run_sync(
            self._sa_metadata.create_all,
            tables=[records_table],
        )

        return records_table, search_engine
