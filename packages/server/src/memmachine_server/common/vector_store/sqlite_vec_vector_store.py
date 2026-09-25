"""
Vector store backed by SQLite + sqlite-vec.

The store is one collection. Each partition gets its own records table and
vec0 virtual table, named by the collection and the partition key, so stores
of different collections may share one engine; sqlite-vec's own partition
keys are not used, since its ANN indexes may not support them.
"""

import struct
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
    String,
    Table,
    Uuid,
    delete,
    event,
    select,
    text,
)
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.engine.interfaces import DBAPIConnection
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import DeclarativeBase, MappedColumn, mapped_column
from sqlalchemy.pool import ConnectionPoolEntry, StaticPool

from memmachine_server.common.data_types import (
    PropertyType,
    PropertyValue,
    SimilarityMetric,
)
from memmachine_server.common.filter.filter_parser import FilterExpr
from memmachine_server.common.filter.sql_filter_util import compile_sql_filter
from memmachine_server.common.properties_json import (
    decode_properties,
    encode_properties,
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
from .utils import _IDENTIFIER_MAX_BYTES, validate_filter, validate_identifier
from .vector_store import VectorStore, VectorStorePartition


class BaseSQLiteVecVectorStore(DeclarativeBase):
    """Base class for SQLiteVecVectorStore ORM models."""


class _PartitionRow(BaseSQLiteVecVectorStore):
    """The registry: one row per partition, keyed by its vector store name and key."""

    __tablename__ = "vector_store_sqlite_vec_pt"

    vector_store_name: MappedColumn[str] = mapped_column(
        String(_IDENTIFIER_MAX_BYTES), primary_key=True
    )
    partition_key: MappedColumn[str] = mapped_column(String(255), primary_key=True)
    # The dimensions, metric and declared schema the partition was created
    # under, so a store built with others fails loudly instead of reading
    # columns and vectors that are not there.
    schema: MappedColumn[dict[str, JsonValue]] = mapped_column(JSON, nullable=False)


class SQLiteVecVectorStorePartition(VectorStorePartition):
    """A partition backed by SQLite + sqlite-vec."""

    _DISTANCE_FUNCTIONS: ClassVar[dict[SimilarityMetric, str]] = {
        SimilarityMetric.COSINE: "vec_distance_cosine",
        SimilarityMetric.EUCLIDEAN: "vec_distance_L2",
    }

    def __init__(
        self,
        *,
        create_session: async_sessionmaker[AsyncSession],
        partition_key: str,
        similarity_metric: SimilarityMetric,
        indexed_properties: Mapping[str, PropertyType],
        records_table: Table,
        vector_table_name: str,
    ) -> None:
        """Initialize with session factory and table references."""
        self._create_session = create_session
        self._partition_key = partition_key
        self._similarity_metric = similarity_metric
        self._indexed_properties = dict(indexed_properties)
        self._records_table = records_table
        self._vector_table_name = vector_table_name

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

    @staticmethod
    def _serialize_vector(vector: Sequence[float]) -> bytes:
        return sqlite_vec.serialize_float32(list(vector))

    @staticmethod
    def _deserialize_vector(data: bytes) -> list[float]:
        count = len(data) // 4
        return list(struct.unpack(f"={count}f", data))

    @staticmethod
    def _distance_to_score(
        distance: float, similarity_metric: SimilarityMetric
    ) -> float:
        match similarity_metric:
            case SimilarityMetric.COSINE:
                return 1.0 - distance
            case SimilarityMetric.EUCLIDEAN:
                return distance
            case _:
                raise NotImplementedError(similarity_metric)

    @staticmethod
    def _threshold_to_max_distance(
        threshold: float, similarity_metric: SimilarityMetric
    ) -> float:
        match similarity_metric:
            case SimilarityMetric.COSINE:
                return 1.0 - threshold
            case SimilarityMetric.EUCLIDEAN:
                return threshold
            case _:
                raise NotImplementedError(similarity_metric)

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
                .returning(self._records_table.c.uuid, self._records_table.c.rowid)
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
            uuid_to_rowid: dict[UUID, int] = {row.uuid: row.rowid for row in rows}

            vector_params = []
            for record in records:
                assert record.vector is not None  # Validated above.
                vector_params.append(
                    {
                        "rowid": uuid_to_rowid[record.uuid],
                        "vector": self._serialize_vector(record.vector),
                    }
                )
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
        score_threshold: float | None = None,
        property_filter: FilterExpr | None = None,
        return_vector: bool = False,
        return_properties: bool = True,
    ) -> list[QueryResult]:
        query_vectors = list(query_vectors)
        if not query_vectors:
            return []

        if limit <= 0:
            return [QueryResult(matches=[]) for _ in query_vectors]

        if property_filter is not None and not validate_filter(property_filter):
            raise ValueError("Filter contains invalid field names")

        k = min(limit, self._MAX_K)

        results: list[QueryResult] = []
        async with self._create_session() as session:
            for query_vector in query_vectors:
                query_blob = self._serialize_vector(query_vector)

                knn_rows = (
                    await session.execute(
                        text(
                            f"SELECT rowid, distance FROM [{self._vector_table_name}] "
                            f"WHERE vector MATCH :query AND k = :k "
                            f"ORDER BY distance"
                        ),
                        {"query": query_blob, "k": k},
                    )
                ).all()

                rowid_to_distance: dict[int, float] = {
                    row.rowid: row.distance for row in knn_rows
                }
                matches = await self._build_matches(
                    session=session,
                    rowid_to_distance=rowid_to_distance,
                    score_threshold=score_threshold,
                    property_filter=property_filter,
                    return_vector=return_vector,
                    return_properties=return_properties,
                )
                results.append(QueryResult(matches=matches))

        return results

    async def _build_matches(
        self,
        session: AsyncSession,
        rowid_to_distance: Mapping[int, float],
        score_threshold: float | None,
        property_filter: FilterExpr | None,
        return_vector: bool,
        return_properties: bool,
    ) -> list[QueryMatch]:
        matched_rowids = list(rowid_to_distance.keys())

        selected_columns = [self._records_table.c.uuid, self._records_table.c.rowid]
        if return_properties:
            selected_columns.append(self._records_table.c.properties)

        fetch_records = select(*selected_columns).where(
            self._records_table.c.rowid.in_(matched_rowids),
        )
        if property_filter is not None:
            fetch_records = fetch_records.where(
                compile_sql_filter(
                    property_filter,
                    lambda field: (
                        self._records_table.c.properties[field],
                        "properties_json",
                    ),
                )
            )

        matched_rows = (await session.execute(fetch_records)).all()

        rowid_to_vector: dict[int, list[float]] = {}
        if return_vector:
            rowid_to_vector = await self._fetch_vectors(
                session, [row.rowid for row in matched_rows]
            )

        matches: list[QueryMatch] = []
        for row in matched_rows:
            distance = rowid_to_distance.get(row.rowid)
            if distance is None:
                continue

            score = self._distance_to_score(distance, self._similarity_metric)
            if score_threshold is not None and (
                score < score_threshold
                if self._similarity_metric.higher_is_better
                else score > score_threshold
            ):
                continue

            properties: dict[str, PropertyValue] | None = None
            if return_properties:
                properties = decode_properties(row.properties)

            vector: list[float] | None = None
            if return_vector:
                vector = rowid_to_vector.get(row.rowid)

            matches.append(
                QueryMatch(
                    score=score,
                    record=Record(uuid=row.uuid, vector=vector, properties=properties),
                )
            )

        matches.sort(
            key=lambda match: match.score,
            reverse=self._similarity_metric.higher_is_better,
        )
        return matches

    async def _fetch_vectors(
        self, session: AsyncSession, rowids: Iterable[int]
    ) -> dict[int, list[float]]:
        rowids = list(rowids)
        if not rowids:
            return {}

        placeholders = ", ".join(f":r{i}" for i in range(len(rowids)))
        vector_rows = (
            await session.execute(
                text(
                    f"SELECT rowid, vector FROM [{self._vector_table_name}] "
                    f"WHERE rowid IN ({placeholders})"
                ),
                {f"r{i}": rowid for i, rowid in enumerate(rowids)},
            )
        ).all()
        return {row.rowid: self._deserialize_vector(row.vector) for row in vector_rows}

    @override
    async def get(
        self,
        *,
        record_uuids: Iterable[UUID],
        return_vector: bool = False,
        return_properties: bool = True,
    ) -> list[Record]:
        record_uuids = list(record_uuids)
        if not record_uuids:
            return []

        selected_columns = [self._records_table.c.uuid, self._records_table.c.rowid]
        if return_properties:
            selected_columns.append(self._records_table.c.properties)

        async with self._create_session() as session:
            fetched_rows = (
                await session.execute(
                    select(*selected_columns).where(
                        self._records_table.c.uuid.in_(record_uuids),
                    )
                )
            ).all()

            rowid_to_vector: dict[int, list[float]] = {}
            if return_vector:
                rowid_to_vector = await self._fetch_vectors(
                    session, [row.rowid for row in fetched_rows]
                )

        record_map: dict[UUID, Record] = {}
        for row in fetched_rows:
            record_uuid = row.uuid

            properties: dict[str, PropertyValue] | None = None
            if return_properties:
                properties = decode_properties(row.properties)

            vector: list[float] | None = rowid_to_vector.get(row.rowid)

            record_map[record_uuid] = Record(
                uuid=record_uuid, vector=vector, properties=properties
            )

        return [
            record_map[record_uuid]
            for record_uuid in record_uuids
            if record_uuid in record_map
        ]

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


class SQLiteVecVectorStoreParams(BaseModel):
    """
    Parameters for constructing a SQLiteVecVectorStore.

    Attributes:
        engine (AsyncEngine): Async SQLAlchemy engine (sqlite+aiosqlite).
        vector_store_name (str):
            The name of this store; names its tables, so stores of
            different names may share the engine.
        vector_dimensions (int):
            Dimensionality of every vector in the store.
        similarity_metric (SimilarityMetric):
            The metric every query of the store scores by; cosine or
            euclidean (default: cosine).
        indexed_properties (IndexedProperties):
            The declared schema every partition of this store carries: each
            key is indexed for filtering, and its values are typed.
    """

    engine: InstanceOf[AsyncEngine] = Field(
        ...,
        description="Async SQLAlchemy engine (sqlite+aiosqlite)",
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

    @field_validator("vector_store_name")
    @classmethod
    def _validate_vector_store_name(cls, vector_store_name: str) -> str:
        validate_vector_store_name(vector_store_name)
        return vector_store_name

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

    The store is one collection; each partition gets its own records table
    and vec0 virtual table. The database file is one node's, so a partition
    is managed by the processes of one node; a handle outliving its
    partition is not detected.
    """

    _SIMILARITY_METRIC_TO_SQLITE_VEC_DISTANCE: ClassVar[dict[SimilarityMetric, str]] = {
        SimilarityMetric.COSINE: "cosine",
        SimilarityMetric.EUCLIDEAN: "L2",
    }

    def __init__(self, params: SQLiteVecVectorStoreParams) -> None:
        """Initialize the vector store with the provided parameters."""
        SQLiteVecVectorStore._validate_metric(params.similarity_metric)
        self._engine = params.engine
        self._vector_store_name = params.vector_store_name
        self._vector_dimensions = params.vector_dimensions
        self._similarity_metric = params.similarity_metric
        self._indexed_properties = params.indexed_properties
        self._create_session = async_sessionmaker(self._engine, expire_on_commit=False)
        self._sa_metadata = MetaData()

        # Stores of different collections may share the engine; the
        # extension is loaded once per connection.
        if not event.contains(self._engine.sync_engine, "connect", _load_sqlite_vec):
            event.listen(self._engine.sync_engine, "connect", _load_sqlite_vec)

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

    @override
    async def provision(self) -> None:
        async with self._engine.begin() as connection:
            await connection.run_sync(BaseSQLiteVecVectorStore.metadata.create_all)

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

        async with self._create_session() as session, session.begin():
            if await self._stored_schema(session, partition_key) is not None:
                raise VectorStorePartitionAlreadyExistsError(
                    self._vector_store_name, partition_key
                )

            await self._ensure_partition_tables(session, partition_key)
            session.add(
                _PartitionRow(
                    vector_store_name=self._vector_store_name,
                    partition_key=partition_key,
                    schema=self._declared_schema().model_dump(mode="json"),
                )
            )

    @override
    async def get_partition(self, partition_key: str) -> VectorStorePartition | None:
        if not validate_identifier(partition_key):
            raise ValueError(f"Invalid partition key {partition_key!r}")

        async with self._create_session() as session:
            schema = await self._stored_schema(session, partition_key)
        if schema is None:
            return None

        return self._partition_handle(
            partition_key,
            self._records_table(partition_key),
            self._vector_table_name(partition_key),
        )

    def _partition_handle(
        self, partition_key: str, records_table: Table, vector_table_name: str
    ) -> SQLiteVecVectorStorePartition:
        return SQLiteVecVectorStorePartition(
            create_session=self._create_session,
            partition_key=partition_key,
            similarity_metric=self._similarity_metric,
            indexed_properties=self._indexed_properties,
            records_table=records_table,
            vector_table_name=vector_table_name,
        )

    @override
    async def delete_partition(self, partition_key: str) -> None:
        if not validate_identifier(partition_key):
            raise ValueError(f"Invalid partition key {partition_key!r}")

        async with self._create_session() as session, session.begin():
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
            vector_table_name = self._vector_table_name(partition_key)

            await session.execute(text(f"DROP TABLE IF EXISTS [{vector_table_name}]"))
            await session.execute(text(f"DROP TABLE IF EXISTS [{records_table.name}]"))

            await session.execute(
                delete(_PartitionRow).where(
                    _PartitionRow.vector_store_name == self._vector_store_name,
                    _PartitionRow.partition_key == partition_key,
                )
            )

            self._sa_metadata.remove(records_table)

    @override
    async def purge_deleted_partitions(self) -> bool:
        # delete_partition drops the tables itself.
        return False

    # Helpers.

    def _partition_prefix(self, partition_key: str) -> str:
        vector_store_name = self._vector_store_name
        return (
            f"vector_store_sqlite_vec_{len(vector_store_name)}_{vector_store_name}"
            f"_{len(partition_key)}_{partition_key}"
        )

    def _records_table_name(self, partition_key: str) -> str:
        return f"{self._partition_prefix(partition_key)}_rc"

    def _vector_table_name(self, partition_key: str) -> str:
        return f"{self._partition_prefix(partition_key)}_vc"

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

    def _declared_schema(self) -> PartitionSchema:
        return PartitionSchema(
            vector_dimensions=self._vector_dimensions,
            similarity_metric=self._similarity_metric,
            indexed_properties=indexed_property_names(self._indexed_properties),
        )

    async def _stored_schema(
        self, session: AsyncSession, partition_key: str
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

    def _records_table(self, partition_key: str) -> Table:
        return Table(
            self._records_table_name(partition_key),
            self._sa_metadata,
            Column("rowid", Integer, primary_key=True, autoincrement=True),
            Column("uuid", Uuid, nullable=False, unique=True),
            Column("properties", JSON, nullable=False, default=dict),
            extend_existing=True,
        )

    async def _ensure_partition_tables(
        self,
        session: AsyncSession,
        partition_key: str,
    ) -> tuple[Table, str]:
        records_table = self._records_table(partition_key)
        vector_table_name = self._vector_table_name(partition_key)
        distance_metric_value = self._SIMILARITY_METRIC_TO_SQLITE_VEC_DISTANCE[
            self._similarity_metric
        ]

        connection = await session.connection()
        await connection.run_sync(
            self._sa_metadata.create_all,
            tables=[records_table],
        )

        await session.execute(
            text(
                f"CREATE VIRTUAL TABLE IF NOT EXISTS [{vector_table_name}] USING vec0("
                f"vector float[{self._vector_dimensions}] distance_metric={distance_metric_value}"
                f")"
            )
        )

        properties_column = Column("properties", JSON)
        for field_name in self._indexed_properties:
            value_expr = properties_column[field_name]["v"].as_string()
            compiled_expr = value_expr.compile(
                dialect=session.bind.dialect,
                compile_kwargs={"literal_binds": True},
            )
            index_name = f"{records_table.name}__{field_name}_v"
            await session.execute(
                text(
                    f"CREATE INDEX IF NOT EXISTS [{index_name}] "
                    f"ON [{records_table.name}]"
                    f"({compiled_expr})"
                )
            )

        return records_table, vector_table_name
