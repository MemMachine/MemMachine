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

from memmachine_server.common.data_types import PropertyType
from memmachine_server.common.filter import (
    And,
    Equals,
    FilterExpr,
    In,
    IsMissing,
    Not,
    NotEquals,
    Or,
    Ordering,
)

from .data_types import (
    IndexedProperties,
    IndexedPropertiesMismatchError,
    QueryMatch,
    QueryResult,
    Record,
    VectorStoreCollectionAlreadyExistsError,
    VectorStoreCollectionConfig,
    VectorStoreCollectionConfigMismatchError,
    indexed_property_names,
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
from .vector_search_engine import VectorSearchEngine
from .vector_store import VectorStore, VectorStoreCollection

logger = logging.getLogger(__name__)

_INDEXED_PROPERTIES_KEY = "indexed_properties"


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
    # The collection config plus the declared schema it was created under,
    # so a store built with another schema fails loudly instead of reading
    # columns that are not there.
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


class SQLiteVectorStoreCollection(VectorStoreCollection):
    """A logical collection backed by SQLite + a pluggable vector search engine."""

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
        namespace: str,
        name: str,
        config: VectorStoreCollectionConfig,
        indexed_properties: Mapping[str, PropertyType],
        index_path: str | None,
        save_threshold: int,
    ) -> None:
        """Initialize a collection handle."""
        self._create_session = create_session
        self._sync_sqlalchemy_engine = sync_sqlalchemy_engine
        self._records_table = records_table
        self._search_engine = search_engine

        self._namespace = namespace
        self._name = name

        self._config = config
        self._indexed_properties = dict(indexed_properties)

        self._index_path = index_path
        self._save_threshold = save_threshold

    _SUPPORTED_FILTER_NODES: ClassVar[frozenset[type]] = frozenset(
        {Equals, NotEquals, Ordering, In, IsMissing, And, Or, Not}
    )

    @property
    @override
    def config(self) -> VectorStoreCollectionConfig:
        return self._config

    @property
    @override
    def indexed_properties(self) -> Mapping[str, PropertyType]:
        return self._indexed_properties

    @property
    @override
    def supported_filter_nodes(self) -> frozenset[type]:
        return SQLiteVectorStoreCollection._SUPPORTED_FILTER_NODES

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
            require_declared_properties(record.properties, self._indexed_properties)

        property_column_names = [
            property_column_name(key) for key in self._indexed_properties
        ]
        insert_records = sqlite_insert(self._records_table)
        upsert_records = insert_records.on_conflict_do_update(
            index_elements=[self._records_table.c.uuid],
            # With no declared column the update is a no-op that still
            # returns the existing row, which `RETURNING` needs.
            set_={
                name: insert_records.excluded[name]
                for name in (property_column_names or ["uuid"])
            },
        ).returning(self._records_table.c.uuid, self._records_table.c.row_id)

        async with self._create_session() as session, session.begin():
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

        if property_filter is not None:
            require_supported_filter(
                property_filter,
                self._indexed_properties,
                SQLiteVectorStoreCollection._SUPPORTED_FILTER_NODES,
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

        return SQLiteVectorStoreCollection._KeyFilter(
            sync_sqlalchemy_engine=self._sync_sqlalchemy_engine,
            records_table=self._records_table,
            filter_expression=compile_property_filter(
                property_filter, self._records_table, self._indexed_properties
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
        engine_factory (Callable[[int], VectorSearchEngine]):
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
        indexed_properties (IndexedProperties):
            The declared schema every collection of this store carries: each
            key is a typed, indexed column of the collection's records table,
            and a record or a filter naming any other key is rejected.
    """

    sqlalchemy_engine: InstanceOf[AsyncEngine] = Field(
        ..., description="Async SQLAlchemy engine (sqlite+aiosqlite)"
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
    indexed_properties: IndexedProperties = Field(
        ...,
        description="The declared schema every collection of this store carries",
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
        self._indexed_properties = params.indexed_properties

        self._create_session = async_sessionmaker(
            self._sqlalchemy_engine, expire_on_commit=False
        )
        self._search_engines: dict[tuple[str, str], VectorSearchEngine] = {}
        self._sa_metadata = MetaData()

        self._sync_sqlalchemy_engine = create_engine(
            str(self._sqlalchemy_engine.url).replace("aiosqlite", "pysqlite")
        )

        @event.listens_for(self._sqlalchemy_engine.sync_engine, "connect")
        @event.listens_for(self._sync_sqlalchemy_engine, "connect")
        def _enable_sqlite_foreign_keys(
            dbapi_connection: DBAPIConnection,
            _connection_record: ConnectionPoolEntry,
        ) -> None:
            cursor = dbapi_connection.cursor()
            cursor.execute("PRAGMA foreign_keys=ON")
            cursor.close()

        self._started = False

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
            session.add(self._collection_row(namespace, name, config))

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
                    sync_sqlalchemy_engine=self._sync_sqlalchemy_engine,
                    records_table=records_table,
                    search_engine=search_engine,
                    namespace=namespace,
                    name=name,
                    config=existing_config,
                    indexed_properties=self._indexed_properties,
                    index_path=str(index_path) if index_path is not None else None,
                    save_threshold=self._save_threshold,
                )

            self._clear_search_engine_state(namespace, name)
            records_table, search_engine = await self._ensure_collection_resources(
                session, namespace, name, config
            )
            session.add(self._collection_row(namespace, name, config))

        return SQLiteVectorStoreCollection(
            create_session=self._create_session,
            sync_sqlalchemy_engine=self._sync_sqlalchemy_engine,
            records_table=records_table,
            search_engine=search_engine,
            namespace=namespace,
            name=name,
            config=config,
            indexed_properties=self._indexed_properties,
            index_path=str(index_path) if index_path is not None else None,
            save_threshold=self._save_threshold,
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
            sync_sqlalchemy_engine=self._sync_sqlalchemy_engine,
            records_table=records_table,
            search_engine=search_engine,
            namespace=namespace,
            name=name,
            config=existing,
            indexed_properties=self._indexed_properties,
            index_path=str(index_path) if index_path is not None else None,
            save_threshold=self._save_threshold,
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
        records_table = Table(
            f"{self._collection_prefix(namespace, name)}_rc",
            self._sa_metadata,
            Column("row_id", Integer, primary_key=True, autoincrement=True),
            Column("uuid", Uuid, nullable=False, unique=True),
            *property_columns(self._indexed_properties),
            extend_existing=True,
        )
        if not records_table.indexes:
            property_indexes(records_table, self._indexed_properties)
        return records_table

    def _collection_row(
        self, namespace: str, name: str, config: VectorStoreCollectionConfig
    ) -> _CollectionRow:
        return _CollectionRow(
            namespace=namespace,
            name=name,
            config_json={
                **config.model_dump(mode="json"),
                _INDEXED_PROPERTIES_KEY: indexed_property_names(
                    self._indexed_properties
                ),
            },
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
        """The collection's config; raises if it was created under another schema."""
        stored = (
            await session.execute(
                select(_CollectionRow.config_json).where(
                    _CollectionRow.namespace == namespace,
                    _CollectionRow.name == name,
                )
            )
        ).scalar_one_or_none()
        if stored is None:
            return None
        stored_properties = stored.get(_INDEXED_PROPERTIES_KEY)
        declared_properties = indexed_property_names(self._indexed_properties)
        if stored_properties != declared_properties:
            raise IndexedPropertiesMismatchError(
                namespace,
                name,
                stored_properties if isinstance(stored_properties, dict) else {},
                declared_properties,
            )
        return VectorStoreCollectionConfig.model_validate(
            {
                key: value
                for key, value in stored.items()
                if key != _INDEXED_PROPERTIES_KEY
            }
        )

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
