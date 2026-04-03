"""
SQLite + sqlite-vec backed vector store implementation.

SQLite stores collection metadata, record UUIDs, and properties.
sqlite-vec provides the vec0 virtual table for vector search.

Each logical collection gets its own records table and vec0 virtual table.
Different namespaces always get separate native tables, as required by the
VectorStore contract.  Per-collection vec0 tables enable future use of ANN
indexes (IVF, DiskANN) which are incompatible with vec0 partition keys.
"""

import asyncio
import json
import struct
from collections import defaultdict
from collections.abc import Iterable, Sequence
from typing import ClassVar, override
from uuid import UUID
from weakref import WeakKeyDictionary

import aiosqlite
import sqlalchemy as sa
import sqlite_vec
from pydantic import BaseModel, Field, InstanceOf, JsonValue
from sqlalchemy import JSON, String, event, select, text
from sqlalchemy.dialects import sqlite as sa_sqlite
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.engine.interfaces import DBAPIConnection
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import DeclarativeBase, MappedColumn, mapped_column
from sqlalchemy.pool import ConnectionPoolEntry

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


class BaseSQLiteVecVectorStore(DeclarativeBase):
    """Base class for SQLiteVecVectorStore ORM models."""


class _CollectionRow(BaseSQLiteVecVectorStore):
    __tablename__ = "vector_store_sqlite_vec_collections"

    namespace: MappedColumn[str] = mapped_column(String(32), primary_key=True)
    name: MappedColumn[str] = mapped_column(String(32), primary_key=True)
    config_json: MappedColumn[dict[str, JsonValue]] = mapped_column(
        JSON, nullable=False
    )


class SQLiteVecCollection(Collection):
    """A logical collection backed by SQLite + sqlite-vec.

    Each logical collection has its own records table and vec0 virtual table,
    so KNN queries search only this collection's vectors directly.
    """

    _DISTANCE_FUNCTIONS: ClassVar[dict[SimilarityMetric, str]] = {
        SimilarityMetric.COSINE: "vec_distance_cosine",
        SimilarityMetric.EUCLIDEAN: "vec_distance_L2",
    }

    def __init__(
        self,
        *,
        create_session: async_sessionmaker[AsyncSession],
        write_lock: asyncio.Lock,
        name: str,
        config: CollectionConfig,
        records_table: sa.Table,
        vec_table_name: str,
    ) -> None:
        """Initialize with session factory, lock, and table references."""
        self._create_session = create_session
        self._write_lock = write_lock
        self._name = name
        self._config = config
        self._records_table = records_table
        self._vec_table_name = vec_table_name
        self._metric = config.similarity_metric
        self._distance_function = self._DISTANCE_FUNCTIONS[config.similarity_metric]

    @staticmethod
    def _serialize_vector(vector: Sequence[float]) -> bytes:
        return sqlite_vec.serialize_float32(list(vector))

    @staticmethod
    def _deserialize_vector(data: bytes) -> list[float]:
        count = len(data) // 4
        return list(struct.unpack(f"<{count}f", data))

    @staticmethod
    def _distance_to_score(
        distance: float, similarity_metric: SimilarityMetric
    ) -> float:
        if similarity_metric == SimilarityMetric.COSINE:
            return 1.0 - distance
        return 1.0 / (1.0 + distance)

    @staticmethod
    def _score_to_max_distance(
        threshold: float, similarity_metric: SimilarityMetric
    ) -> float:
        """Convert a similarity threshold to a maximum distance for SQL filtering."""
        if similarity_metric == SimilarityMetric.COSINE:
            return 1.0 - threshold
        if threshold <= 0:
            return float("inf")
        return (1.0 - threshold) / threshold

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
        vec_table_name = self._vec_table_name

        async with self._write_lock, self._create_session() as session, session.begin():
            statement = sqlite_insert(records_table).on_conflict_do_update(
                index_elements=[records_table.c.uuid],
                set_={"properties": sqlite_insert(records_table).excluded.properties},
            )
            await session.execute(
                statement,
                [
                    {
                        "uuid": str(record.uuid),
                        "properties": encode_properties(record.properties),
                    }
                    for record in records_list
                ],
            )

            # Fetch rowids for vec0 operations
            uuid_strs = [str(record.uuid) for record in records_list]
            rows = (
                await session.execute(
                    select(records_table.c.uuid, records_table.c.rowid).where(
                        records_table.c.uuid.in_(uuid_strs),
                    )
                )
            ).all()
            uuid_to_rowid: dict[str, int] = {row.uuid: row.rowid for row in rows}

            # vec0: delete + insert for each record with a vector
            for record in records_list:
                if record.vector is not None:
                    rowid = uuid_to_rowid[str(record.uuid)]
                    await session.execute(
                        text(f"DELETE FROM [{vec_table_name}] WHERE rowid = :rid"),
                        {"rid": rowid},
                    )
                    await session.execute(
                        text(
                            f"INSERT INTO [{vec_table_name}](rowid, embedding) "
                            f"VALUES (:rid, :vec)"
                        ),
                        {
                            "rid": rowid,
                            "vec": self._serialize_vector(record.vector),
                        },
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
        async with self._create_session() as session:
            for query_vector in query_vectors_list:
                query_blob = self._serialize_vector(query_vector)
                if property_filter is None:
                    matches = await self._query_knn(
                        session,
                        query_blob,
                        score_threshold,
                        limit,
                        return_vector,
                        return_properties,
                    )
                else:
                    matches = await self._query_filtered(
                        session,
                        query_blob,
                        score_threshold,
                        limit,
                        property_filter,
                        return_vector,
                        return_properties,
                    )
                results.append(QueryResult(matches=matches))

        return results

    async def _query_knn(
        self,
        session: AsyncSession,
        query_blob: bytes,
        score_threshold: float | None,
        limit: int | None,
        return_vector: bool,
        return_properties: bool,
    ) -> list[QueryMatch]:
        """Pure KNN query via vec0 MATCH.

        The vec0 table contains only this collection's vectors, so no
        partition key filtering is needed.
        """
        vec_table_name = self._vec_table_name
        # sqlite-vec caps k at 4096
        effective_limit = min(limit, 4096) if limit is not None else 4096

        knn_rows = (
            await session.execute(
                text(
                    f"SELECT rowid, distance FROM [{vec_table_name}] "
                    f"WHERE embedding MATCH :query AND k = :k "
                    f"ORDER BY distance"
                ),
                {"query": query_blob, "k": effective_limit},
            )
        ).all()

        if not knn_rows:
            return []

        rowid_to_distance: dict[int, float] = {
            row.rowid: row.distance for row in knn_rows
        }
        return await self._build_matches_from_rowids(
            session,
            rowid_to_distance,
            score_threshold,
            return_vector,
            return_properties,
        )

    async def _query_filtered(
        self,
        session: AsyncSession,
        query_blob: bytes,
        score_threshold: float | None,
        limit: int | None,
        property_filter: FilterExpr,
        return_vector: bool,
        return_properties: bool,
    ) -> list[QueryMatch]:
        """Filtered query using brute-force distance computation (perfect recall)."""
        records_table_name = self._records_table.name
        vec_table_name = self._vec_table_name

        select_parts = [
            f"[{records_table_name}].uuid",
            f"[{records_table_name}].rowid",
        ]
        if return_properties:
            select_parts.append(f"[{records_table_name}].properties")

        distance_expression = (
            f"{self._distance_function}([{vec_table_name}].embedding, :query)"
        )
        select_parts.append(f"{distance_expression} AS distance")

        if return_vector:
            select_parts.append(f"[{vec_table_name}].embedding")

        # literal_binds is required because this filter clause is embedded in a
        # raw text() SQL query (for the vec0 distance function join).
        compiled = self._compile_property_filter(property_filter)
        filter_sql = compiled.compile(
            dialect=sa_sqlite.dialect(),
            compile_kwargs={"literal_binds": True},
        )
        filter_clause = f"AND ({filter_sql})"

        params: dict[str, bytes | str | int | float] = {}

        distance_filter = ""
        if score_threshold is not None:
            max_distance = self._score_to_max_distance(score_threshold, self._metric)
            distance_filter = f"AND {distance_expression} <= :max_dist"
            params["max_dist"] = max_distance

        limit_clause = ""
        if limit is not None:
            limit_clause = "LIMIT :lim"
            params["lim"] = limit

        sql = (
            f"SELECT {', '.join(select_parts)} "
            f"FROM [{records_table_name}] "
            f"JOIN [{vec_table_name}] ON [{vec_table_name}].rowid = [{records_table_name}].rowid "
            f"WHERE 1=1 "
            f"{filter_clause} {distance_filter} "
            f"ORDER BY distance ASC {limit_clause}"
        )

        params["query"] = query_blob

        rows = (await session.execute(text(sql), params)).all()

        return self._parse_filtered_rows(rows, return_properties, return_vector)

    def _parse_filtered_rows(
        self,
        rows: Sequence[sa.Row[tuple[str, ...]]],
        return_properties: bool,
        return_vector: bool,
    ) -> list[QueryMatch]:
        """Parse raw rows from a filtered query into QueryMatch objects."""
        matches: list[QueryMatch] = []
        for row in rows:
            column_index = 0
            uuid_val = UUID(row[column_index])
            column_index += 1
            _rowid = row[column_index]
            column_index += 1

            properties: dict[str, PropertyValue] | None = None
            if return_properties:
                raw_properties = row[column_index]
                column_index += 1
                if isinstance(raw_properties, str):
                    raw_properties = json.loads(raw_properties)
                properties = decode_properties(raw_properties)

            distance = row[column_index]
            column_index += 1

            vector: list[float] | None = None
            if return_vector:
                raw_vector = row[column_index]
                if raw_vector is not None:
                    vector = self._deserialize_vector(raw_vector)

            score = self._distance_to_score(distance, self._metric)
            matches.append(
                QueryMatch(
                    score=score,
                    record=Record(uuid=uuid_val, vector=vector, properties=properties),
                )
            )

        return matches

    async def _build_matches_from_rowids(
        self,
        session: AsyncSession,
        rowid_to_distance: dict[int, float],
        score_threshold: float | None,
        return_vector: bool,
        return_properties: bool,
    ) -> list[QueryMatch]:
        """Fetch record data for rowids from a KNN search and build matches."""
        records_table = self._records_table
        vec_table_name = self._vec_table_name

        rowid_list = list(rowid_to_distance.keys())

        columns = [records_table.c.uuid, records_table.c.rowid]
        if return_properties:
            columns.append(records_table.c.properties)
        statement = select(*columns).where(
            records_table.c.rowid.in_(rowid_list),
        )
        rows = (await session.execute(statement)).all()

        # Fetch vectors if needed
        vector_data_map: dict[int, bytes] = {}
        if return_vector:
            filtered_rowids = [row.rowid for row in rows]
            if filtered_rowids:
                placeholders = ", ".join(
                    f":r{index}" for index in range(len(filtered_rowids))
                )
                vector_rows = (
                    await session.execute(
                        text(
                            f"SELECT rowid, embedding FROM [{vec_table_name}] "
                            f"WHERE rowid IN ({placeholders})"
                        ),
                        {
                            f"r{index}": row_id
                            for index, row_id in enumerate(filtered_rowids)
                        },
                    )
                ).all()
                vector_data_map = {
                    vector_row.rowid: vector_row.embedding for vector_row in vector_rows
                }

        matches: list[QueryMatch] = []
        for row in rows:
            distance = rowid_to_distance.get(row.rowid)
            if distance is None:
                continue

            score = self._distance_to_score(distance, self._metric)
            if score_threshold is not None and score < score_threshold:
                continue

            properties: dict[str, PropertyValue] | None = None
            if return_properties:
                properties = decode_properties(row.properties)

            vector: list[float] | None = None
            if return_vector:
                raw_vector = vector_data_map.get(row.rowid)
                if raw_vector is not None:
                    vector = self._deserialize_vector(raw_vector)

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
        vec_table_name = self._vec_table_name
        dict(self._config.properties_schema)
        uuid_strs = [str(record_uuid) for record_uuid in uuid_list]

        async with self._create_session() as session:
            columns = [records_table.c.uuid, records_table.c.rowid]
            if return_properties:
                columns.append(records_table.c.properties)
            statement = select(*columns).where(
                records_table.c.uuid.in_(uuid_strs),
            )
            rows = (await session.execute(statement)).all()

            vector_data_map: dict[int, bytes] = {}
            if return_vector:
                rowids = [row.rowid for row in rows]
                if rowids:
                    placeholders = ", ".join(
                        f":r{index}" for index in range(len(rowids))
                    )
                    vector_rows = (
                        await session.execute(
                            text(
                                f"SELECT rowid, embedding FROM [{vec_table_name}] "
                                f"WHERE rowid IN ({placeholders})"
                            ),
                            {
                                f"r{index}": row_id
                                for index, row_id in enumerate(rowids)
                            },
                        )
                    ).all()
                    vector_data_map = {
                        vector_row.rowid: vector_row.embedding
                        for vector_row in vector_rows
                    }

        record_map: dict[UUID, Record] = {}
        for row in rows:
            uuid_val = UUID(row.uuid)

            properties: dict[str, PropertyValue] | None = None
            if return_properties:
                properties = decode_properties(row.properties)

            vector: list[float] | None = None
            if return_vector:
                raw_vector = vector_data_map.get(row.rowid)
                if raw_vector is not None:
                    vector = self._deserialize_vector(raw_vector)

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
        vec_table_name = self._vec_table_name
        uuid_strs = [str(record_uuid) for record_uuid in uuid_list]

        async with self._write_lock, self._create_session() as session, session.begin():
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

            # Delete from vec0
            placeholders = ", ".join(f":r{index}" for index in range(len(rowids)))
            await session.execute(
                text(f"DELETE FROM [{vec_table_name}] WHERE rowid IN ({placeholders})"),
                {f"r{index}": row_id for index, row_id in enumerate(rowids)},
            )

            # Delete from records
            await session.execute(
                sa.delete(records_table).where(
                    records_table.c.uuid.in_(uuid_strs),
                )
            )


class SQLiteVecVectorStoreParams(BaseModel):
    """Parameters for constructing a SQLiteVecVectorStore.

    Attributes:
        engine: Async SQLAlchemy engine with sqlite-vec loaded.
            Use :meth:`SQLiteVecVectorStore.create_engine` to create one.
    """

    engine: InstanceOf[AsyncEngine] = Field(
        ...,
        description="Async SQLAlchemy engine (sqlite+aiosqlite) with sqlite-vec loaded",
    )


class SQLiteVecVectorStore(VectorStore):
    """Vector store backed by SQLite + sqlite-vec.

    Each logical collection gets its own records table and vec0 virtual table.
    """

    _SIMILARITY_METRIC_TO_SQLITE_VEC_DISTANCE: ClassVar[dict[SimilarityMetric, str]] = {
        SimilarityMetric.COSINE: "cosine",
        SimilarityMetric.EUCLIDEAN: "L2",
    }

    # Shared across all instances so that stores using the same engine
    # serialise SQLite writes through the same lock.
    # Keyed by engine so locks are garbage-collected when the engine is.
    _write_locks: WeakKeyDictionary[AsyncEngine, asyncio.Lock] = WeakKeyDictionary()
    _name_locks_by_engine: WeakKeyDictionary[
        AsyncEngine, defaultdict[tuple[str, str], asyncio.Lock]
    ] = WeakKeyDictionary()

    def __init__(self, params: SQLiteVecVectorStoreParams) -> None:
        """Initialize the vector store with the provided parameters."""
        self._engine = params.engine
        self._create_session = async_sessionmaker(self._engine, expire_on_commit=False)
        self._records_tables: dict[str, sa.Table] = {}
        self._sa_metadata = sa.MetaData()

    @property
    def _write_lock(self) -> asyncio.Lock:
        return SQLiteVecVectorStore._write_locks.setdefault(
            self._engine, asyncio.Lock()
        )

    @property
    def _name_locks(self) -> defaultdict[tuple[str, str], asyncio.Lock]:
        return SQLiteVecVectorStore._name_locks_by_engine.setdefault(
            self._engine, defaultdict(asyncio.Lock)
        )

    @staticmethod
    def create_engine(url: str) -> AsyncEngine:
        """Create an ``AsyncEngine`` with sqlite-vec loaded on every connection.

        Args:
            url: SQLAlchemy database URL, e.g. ``"sqlite+aiosqlite:///path.db"``
                 or ``"sqlite+aiosqlite:///:memory:"``.
        """
        from sqlalchemy.ext.asyncio import create_async_engine

        engine = create_async_engine(url)

        @event.listens_for(engine.sync_engine, "connect")
        def _on_connect(
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

        return engine

    @staticmethod
    def _collection_prefix(namespace: str, name: str) -> str:
        """Unique prefix for a logical collection's native resources."""
        return f"vector_store_sqlite_vec_{namespace}_{name}"

    @staticmethod
    def _validate_metric(similarity_metric: SimilarityMetric) -> None:
        if (
            similarity_metric
            not in SQLiteVecVectorStore._SIMILARITY_METRIC_TO_SQLITE_VEC_DISTANCE
        ):
            supported = ", ".join(
                similarity_metric.value
                for similarity_metric in SQLiteVecVectorStore._SIMILARITY_METRIC_TO_SQLITE_VEC_DISTANCE
            )
            raise ValueError(
                f"sqlite-vec only supports {supported} similarity metrics, "
                f"got {similarity_metric.value!r}"
            )

    @staticmethod
    def _build_records_table(table_name: str, sa_metadata: sa.MetaData) -> sa.Table:
        """Build a SQLAlchemy Core Table for a per-collection records table."""
        return sa.Table(
            table_name,
            sa_metadata,
            sa.Column("rowid", sa.Integer, primary_key=True, autoincrement=True),
            sa.Column("uuid", sa.Text, nullable=False, unique=True),
            sa.Column("properties", JSON, nullable=True),
        )

    @override
    async def startup(self) -> None:
        async with self._engine.begin() as connection:
            await connection.run_sync(BaseSQLiteVecVectorStore.metadata.create_all)

    @override
    async def shutdown(self) -> None:
        self._records_tables.clear()

    def _get_or_build_records_table(self, collection_prefix: str) -> sa.Table:
        if collection_prefix not in self._records_tables:
            table_name = f"{collection_prefix}_records"
            self._records_tables[collection_prefix] = self._build_records_table(
                table_name, self._sa_metadata
            )
        return self._records_tables[collection_prefix]

    async def _get_stored_config(
        self, session: AsyncSession, namespace: str, name: str
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
    ) -> tuple[sa.Table, str]:
        """Idempotently create per-collection native tables."""
        collection_prefix = self._collection_prefix(namespace, name)
        records_table = self._get_or_build_records_table(collection_prefix)
        vec_table_name = f"{collection_prefix}_vec"
        distance_metric_value = self._SIMILARITY_METRIC_TO_SQLITE_VEC_DISTANCE[
            config.similarity_metric
        ]

        # Create records table via SQLAlchemy (idempotent)
        connection = await session.connection()
        await connection.run_sync(
            self._sa_metadata.create_all,
            tables=[records_table],
        )

        # Create vec0 virtual table (idempotent, no partition key)
        await session.execute(
            text(
                f"CREATE VIRTUAL TABLE IF NOT EXISTS [{vec_table_name}] USING vec0("
                f"embedding float[{config.vector_dimensions}] distance_metric={distance_metric_value}"
                f")"
            )
        )

        # Create property indexes
        for field_name in config.properties_schema:
            index_name = f"idx_{records_table.name}_{field_name}"
            safe_field = field_name.replace("'", "''")
            await session.execute(
                text(
                    f"CREATE INDEX IF NOT EXISTS [{index_name}] "
                    f"ON [{records_table.name}]"
                    f"(json_extract(properties, '$.{safe_field}'))"
                )
            )

        return records_table, vec_table_name

    def _build_collection_handle(
        self,
        name: str,
        config: CollectionConfig,
        records_table: sa.Table,
        vec_table_name: str,
    ) -> SQLiteVecCollection:
        return SQLiteVecCollection(
            create_session=self._create_session,
            write_lock=self._write_lock,
            name=name,
            config=config,
            records_table=records_table,
            vec_table_name=vec_table_name,
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
                records_table, vec_table_name = await self._ensure_native_tables(
                    session, namespace, name, existing
                )
                return self._build_collection_handle(
                    name, existing, records_table, vec_table_name
                )

            records_table, vec_table_name = await self._ensure_native_tables(
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
            name, config, records_table, vec_table_name
        )

    @override
    async def open_collection(self, *, namespace: str, name: str) -> Collection | None:
        if not validate_identifier(namespace) or not validate_identifier(name):
            raise ValueError(f"Invalid namespace {namespace!r} or name {name!r}")
        async with self._create_session() as session:
            existing = await self._get_stored_config(session, namespace, name)
        if existing is None:
            return None

        collection_prefix = self._collection_prefix(namespace, name)
        records_table = self._get_or_build_records_table(collection_prefix)
        vec_table_name = f"{collection_prefix}_vec"
        return self._build_collection_handle(
            name, existing, records_table, vec_table_name
        )

    @override
    async def close_collection(self, *, collection: Collection) -> None:
        pass  # No resources to release

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
            vec_table_name = f"{collection_prefix}_vec"

            # Drop per-collection tables (cascades indexes)
            await session.execute(text(f"DROP TABLE IF EXISTS [{vec_table_name}]"))
            await session.execute(text(f"DROP TABLE IF EXISTS [{records_table.name}]"))

            # Remove from registry
            await session.execute(
                sa.delete(_CollectionRow).where(
                    _CollectionRow.namespace == namespace,
                    _CollectionRow.name == name,
                )
            )

            # Clean up in-memory caches
            self._records_tables.pop(collection_prefix, None)
            self._sa_metadata.remove(records_table)
