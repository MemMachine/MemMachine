"""
Vector store backed by SQLite + sqlite-vec.

Each logical collection gets its own records table and vec0 virtual table.
Partition keys are avoided in favor of per-collection tables,
since sqlite-vec ANN indexes may not support them.

The records table carries one typed, indexed column per declared property,
and a filtered query hands the KNN an allowlist of the rows the filter
admits, so the filter is evaluated during the search: vec0 ranks only the
allowed rows, and a filtered search returns fewer only when the filter
admits fewer.
"""

from collections.abc import Iterable, Mapping, Sequence
from typing import ClassVar, override
from uuid import UUID

import aiosqlite
import sqlite_vec
from pydantic import BaseModel, Field, InstanceOf, JsonValue, field_validator
from sqlalchemy import (
    JSON,
    Column,
    Integer,
    MetaData,
    Select,
    String,
    Table,
    Uuid,
    column,
    delete,
    event,
    select,
    text,
)
from sqlalchemy import table as sql_table
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.engine.interfaces import DBAPIConnection
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
from .vector_store import VectorStore, VectorStoreCollection

_INDEXED_PROPERTIES_KEY = "indexed_properties"


class BaseSQLiteVecVectorStore(DeclarativeBase):
    """Base class for SQLiteVecVectorStore ORM models."""


class _CollectionRow(BaseSQLiteVecVectorStore):
    __tablename__ = "vector_store_sqlite_vec_cl"

    namespace: MappedColumn[str] = mapped_column(String(255), primary_key=True)
    name: MappedColumn[str] = mapped_column(String(255), primary_key=True)
    # The collection config plus the declared schema it was created under,
    # so a store built with another schema fails loudly instead of reading
    # columns that are not there.
    config_json: MappedColumn[dict[str, JsonValue]] = mapped_column(
        JSON, nullable=False
    )


class SQLiteVecVectorStoreCollection(VectorStoreCollection):
    """A logical collection backed by SQLite + sqlite-vec."""

    _SUPPORTED_FILTER_NODES: ClassVar[frozenset[type]] = frozenset(
        {Equals, NotEquals, Ordering, In, IsMissing, And, Or, Not}
    )

    def __init__(
        self,
        *,
        create_session: async_sessionmaker[AsyncSession],
        config: VectorStoreCollectionConfig,
        indexed_properties: Mapping[str, PropertyType],
        records_table: Table,
        vector_table_name: str,
    ) -> None:
        """Initialize with session factory and table references."""
        self._create_session = create_session
        self._config = config
        self._indexed_properties = dict(indexed_properties)
        self._records_table = records_table
        self._vector_table_name = vector_table_name

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
        return SQLiteVecVectorStoreCollection._SUPPORTED_FILTER_NODES

    @staticmethod
    def _serialize_vector(vector: Sequence[float]) -> bytes:
        return sqlite_vec.serialize_float32(list(vector))

    @staticmethod
    def _distance_to_cosine_similarity(distance: float) -> float:
        """Convert a sqlite-vec cosine distance to a cosine similarity."""
        return 1.0 - distance

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
        ).returning(self._records_table.c.uuid, self._records_table.c.rowid)

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
            uuid_to_rowid: dict[UUID, int] = {row.uuid: row.rowid for row in rows}

            vector_params = [
                {
                    "rowid": uuid_to_rowid[record.uuid],
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
                    f"INSERT INTO [{self._vector_table_name}](rowid, vector) "
                    f"VALUES (:rowid, :vector)"
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
                SQLiteVecVectorStoreCollection._SUPPORTED_FILTER_NODES,
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
        """The KNN over the vec0 table, restricted to the rows a filter admits.

        vec0 takes a `rowid IN (...)` constraint into the search itself, so
        the `k` nearest are the nearest among the admitted rows.
        """
        statement = (
            select(column("rowid"), column("distance"))
            .select_from(sql_table(self._vector_table_name))
            .where(
                text("vector MATCH :query AND k = :k").bindparams(query=query_blob, k=k)
            )
        )
        if filter_expression is not None:
            statement = statement.where(
                column("rowid").in_(
                    select(self._records_table.c.rowid).where(filter_expression)
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
                    self._records_table.c.rowid.in_(list(rowid_to_distance)),
                )
            )
        ).all()

        matches: list[QueryMatch] = []
        for row in matched_rows:
            cosine_similarity = (
                SQLiteVecVectorStoreCollection._distance_to_cosine_similarity(
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
            rows = (
                await session.execute(
                    select(self._records_table.c.rowid).where(
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
                    self._records_table.c.uuid.in_(record_uuids),
                )
            )


class SQLiteVecVectorStoreParams(BaseModel):
    """
    Parameters for constructing a SQLiteVecVectorStore.

    Attributes:
        engine (AsyncEngine): Async SQLAlchemy engine (sqlite+aiosqlite).
        indexed_properties (IndexedProperties):
            The declared schema every collection of this store carries: each
            key is a typed, indexed column of the collection's records table,
            and a record or a filter naming any other key is rejected.
    """

    engine: InstanceOf[AsyncEngine] = Field(
        ...,
        description="Async SQLAlchemy engine (sqlite+aiosqlite)",
    )
    indexed_properties: IndexedProperties = Field(
        ...,
        description="The declared schema every collection of this store carries",
    )

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

    Each logical collection gets its own records table and vec0 virtual table.
    """

    _SQLITE_VEC_DISTANCE_METRIC: ClassVar[str] = "cosine"

    def __init__(self, params: SQLiteVecVectorStoreParams) -> None:
        """Initialize the vector store with the provided parameters."""
        self._engine = params.engine
        self._indexed_properties = params.indexed_properties
        self._create_session = async_sessionmaker(self._engine, expire_on_commit=False)
        self._sa_metadata = MetaData()

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
    def indexed_properties(self) -> Mapping[str, PropertyType]:
        return self._indexed_properties

    @override
    async def startup(self) -> None:
        async with self._engine.begin() as connection:
            await connection.run_sync(BaseSQLiteVecVectorStore.metadata.create_all)

    @override
    async def shutdown(self) -> None:
        pass

    @override
    async def create_collection(
        self,
        *,
        namespace: str,
        name: str,
        config: VectorStoreCollectionConfig,
    ) -> None:
        if not validate_identifier(namespace) or not validate_identifier(name):
            raise ValueError(f"Invalid namespace {namespace!r} or name {name!r}")
        async with self._create_session() as session, session.begin():
            existing_config = await self._get_stored_config(session, namespace, name)
            if existing_config is not None:
                raise VectorStoreCollectionAlreadyExistsError(namespace, name)

            await self._ensure_collection_tables(session, namespace, name, config)
            session.add(self._collection_row(namespace, name, config))

    @override
    async def open_or_create_collection(
        self,
        *,
        namespace: str,
        name: str,
        config: VectorStoreCollectionConfig,
    ) -> VectorStoreCollection:
        if not validate_identifier(namespace) or not validate_identifier(name):
            raise ValueError(f"Invalid namespace {namespace!r} or name {name!r}")
        async with self._create_session() as session, session.begin():
            existing_config = await self._get_stored_config(session, namespace, name)
            if existing_config is not None:
                if existing_config != config:
                    raise VectorStoreCollectionConfigMismatchError(
                        namespace, name, existing_config, config
                    )

                records_table, vector_table_name = await self._ensure_collection_tables(
                    session, namespace, name, existing_config
                )
                return self._collection_handle(
                    existing_config, records_table, vector_table_name
                )

            records_table, vector_table_name = await self._ensure_collection_tables(
                session, namespace, name, config
            )
            session.add(self._collection_row(namespace, name, config))

        return self._collection_handle(config, records_table, vector_table_name)

    @override
    async def open_collection(
        self, *, namespace: str, name: str
    ) -> VectorStoreCollection | None:
        if not validate_identifier(namespace) or not validate_identifier(name):
            raise ValueError(f"Invalid namespace {namespace!r} or name {name!r}")

        async with self._create_session() as session:
            existing = await self._get_stored_config(session, namespace, name)
        if existing is None:
            return None

        return self._collection_handle(
            existing,
            self._records_table(namespace, name),
            self._vector_table_name(namespace, name),
        )

    @override
    async def close_collection(self, *, collection: VectorStoreCollection) -> None:
        pass  # No resources to release.

    @override
    async def delete_collection(self, *, namespace: str, name: str) -> None:
        if not validate_identifier(namespace) or not validate_identifier(name):
            raise ValueError(f"Invalid namespace {namespace!r} or name {name!r}")

        async with self._create_session() as session, session.begin():
            existing = (
                await session.execute(
                    select(_CollectionRow.config_json).where(
                        _CollectionRow.namespace == namespace,
                        _CollectionRow.name == name,
                    )
                )
            ).scalar_one_or_none()
            if existing is None:
                return

            records_table = self._records_table(namespace, name)
            vector_table_name = self._vector_table_name(namespace, name)

            await session.execute(text(f"DROP TABLE IF EXISTS [{vector_table_name}]"))
            await session.execute(text(f"DROP TABLE IF EXISTS [{records_table.name}]"))

            await session.execute(
                delete(_CollectionRow).where(
                    _CollectionRow.namespace == namespace,
                    _CollectionRow.name == name,
                )
            )

            self._sa_metadata.remove(records_table)

    # Helpers.

    @staticmethod
    def _collection_prefix(namespace: str, name: str) -> str:
        return (
            f"vector_store_sqlite_vec_{len(namespace)}_{namespace}_{len(name)}_{name}"
        )

    @staticmethod
    def _records_table_name(namespace: str, name: str) -> str:
        return f"{SQLiteVecVectorStore._collection_prefix(namespace, name)}_rc"

    @staticmethod
    def _vector_table_name(namespace: str, name: str) -> str:
        return f"{SQLiteVecVectorStore._collection_prefix(namespace, name)}_vc"

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

    def _collection_handle(
        self,
        config: VectorStoreCollectionConfig,
        records_table: Table,
        vector_table_name: str,
    ) -> SQLiteVecVectorStoreCollection:
        return SQLiteVecVectorStoreCollection(
            create_session=self._create_session,
            config=config,
            indexed_properties=self._indexed_properties,
            records_table=records_table,
            vector_table_name=vector_table_name,
        )

    async def _get_stored_config(
        self, session: AsyncSession, namespace: str, name: str
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

    def _records_table(self, namespace: str, name: str) -> Table:
        records_table = Table(
            self._records_table_name(namespace, name),
            self._sa_metadata,
            Column("rowid", Integer, primary_key=True, autoincrement=True),
            Column("uuid", Uuid, nullable=False, unique=True),
            *property_columns(self._indexed_properties),
            extend_existing=True,
        )
        if not records_table.indexes:
            property_indexes(records_table, self._indexed_properties)
        return records_table

    async def _ensure_collection_tables(
        self,
        session: AsyncSession,
        namespace: str,
        name: str,
        config: VectorStoreCollectionConfig,
    ) -> tuple[Table, str]:
        records_table = self._records_table(namespace, name)
        vector_table_name = self._vector_table_name(namespace, name)

        connection = await session.connection()
        await connection.run_sync(
            self._sa_metadata.create_all,
            tables=[records_table],
        )

        await session.execute(
            text(
                f"CREATE VIRTUAL TABLE IF NOT EXISTS [{vector_table_name}] USING vec0("
                f"vector float[{config.vector_dimensions}] "
                f"distance_metric={SQLiteVecVectorStore._SQLITE_VEC_DISTANCE_METRIC}"
                f")"
            )
        )

        return records_table, vector_table_name
