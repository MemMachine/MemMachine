"""
Vector store backed by SQLite + sqlite-vec.

Each life of a partition gets its own records table and vec0 virtual table,
named by the partition's incarnation, so different stores may share one
engine; sqlite-vec's own partition keys are not used, since its ANN indexes
may not support them.
"""

from collections.abc import Iterable, Mapping, Sequence
from typing import ClassVar, override
from uuid import UUID, uuid4

import aiosqlite
import sqlite_vec
from pydantic import Field, InstanceOf, field_validator
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
    SimilarityMetric,
)
from memmachine_server.common.filter.filter_parser import FilterExpr
from memmachine_server.common.filter.sql_filter_util import compile_sql_filter
from memmachine_server.common.properties_json import (
    encode_properties,
)
from memmachine_server.common.sql import write_connection, write_session

from .data_types import (
    QueryMatch,
    QueryResult,
    Record,
    VectorStorePartitionAlreadyExistsError,
    VectorStorePartitionHandleStaleError,
)
from .utils import (
    _IDENTIFIER_MAX_BYTES,
    _PARTITION_KEY_MAX_BYTES,
    is_valid_filter,
    validate_declared_types,
    validate_dimensions,
    validate_limit,
    validate_partition_key,
    validate_query_vector,
    validate_score_threshold,
)
from .vector_store import VectorStore, VectorStoreParams, VectorStorePartition


class BaseSQLiteVecVectorStore(DeclarativeBase):
    """Base class for SQLiteVecVectorStore ORM models."""


class _PartitionRow(BaseSQLiteVecVectorStore):
    """The registry: one row per partition, keyed by its vector store name and key."""

    __tablename__ = "vector_store_sqlite_vec_pt"

    vector_store_name: MappedColumn[str] = mapped_column(
        String(_IDENTIFIER_MAX_BYTES), primary_key=True
    )
    partition_key: MappedColumn[str] = mapped_column(
        String(_PARTITION_KEY_MAX_BYTES), primary_key=True
    )
    # The partition's current life. Its tables are named by it, so a
    # partition created again under the key gets tables of its own, which a
    # stale handle never reaches.
    incarnation: MappedColumn[UUID] = mapped_column(Uuid, nullable=False, unique=True)


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
        vector_store_name: str,
        partition_key: str,
        incarnation: UUID,
        vector_dimensions: int,
        similarity_metric: SimilarityMetric,
        indexed_properties: Mapping[str, PropertyType],
        records_table: Table,
        vector_table_name: str,
    ) -> None:
        """Initialize a handle on one life of a partition."""
        self._create_session = create_session
        self._vector_store_name = vector_store_name
        self._partition_key = partition_key
        self._incarnation = incarnation
        self._vector_dimensions = vector_dimensions
        self._similarity_metric = similarity_metric
        self._indexed_properties = dict(indexed_properties)
        self._records_table = records_table
        self._vector_table_name = vector_table_name

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
        for record in records:
            validate_declared_types(record.properties, self._indexed_properties)
            validate_dimensions(record.vector, self._vector_dimensions)

        async with write_session(self._create_session) as session:
            await self._require_current(session)
            if not records:
                return
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
    ) -> list[QueryResult]:
        validate_limit(limit)
        query_vectors = list(query_vectors)
        for query_vector in query_vectors:
            validate_query_vector(query_vector, self._vector_dimensions)
        validate_score_threshold(score_threshold)

        if property_filter is not None and not is_valid_filter(property_filter):
            raise ValueError("Filter contains invalid field names")

        k = min(limit, self._MAX_K)

        results: list[QueryResult] = []
        async with self._create_session() as session:
            await self._require_current(session)
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
                )
                results.append(QueryResult(matches=matches))

        return results

    async def _build_matches(
        self,
        session: AsyncSession,
        rowid_to_distance: Mapping[int, float],
        score_threshold: float | None,
        property_filter: FilterExpr | None,
    ) -> list[QueryMatch]:
        fetch_records = select(
            self._records_table.c.uuid, self._records_table.c.rowid
        ).where(
            self._records_table.c.rowid.in_(list(rowid_to_distance)),
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

            matches.append(QueryMatch(score=score, record_uuid=row.uuid))

        matches.sort(
            key=lambda match: match.score,
            reverse=self._similarity_metric.higher_is_better,
        )
        return matches

    @override
    async def delete(self, *, record_uuids: Iterable[UUID]) -> None:
        record_uuids = list(record_uuids)

        async with write_session(self._create_session) as session:
            await self._require_current(session)
            if not record_uuids:
                return
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

    async def _require_current(self, session: AsyncSession) -> None:
        """Raise unless this handle's life is still its partition's."""
        current = (
            await session.execute(
                select(_PartitionRow.incarnation).where(
                    _PartitionRow.incarnation == self._incarnation
                )
            )
        ).scalar_one_or_none()
        if current is None:
            raise VectorStorePartitionHandleStaleError(
                self._vector_store_name, self._partition_key
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


class SQLiteVecVectorStoreParams(VectorStoreParams):
    """
    Parameters for constructing a SQLiteVecVectorStore.

    Stores of different names may share the engine. The similarity metric is
    cosine or euclidean.

    Attributes:
        engine (AsyncEngine):
            Async SQLAlchemy engine (sqlite+aiosqlite), set up by
            `enable_sqlite_explicit_begin` at its creation: the store's write
            transactions take SQLite's write lock at BEGIN.
    """

    engine: InstanceOf[AsyncEngine] = Field(
        ...,
        description=(
            "Async SQLAlchemy engine (sqlite+aiosqlite), set up by "
            "enable_sqlite_explicit_begin at its creation"
        ),
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

    Each life of a partition gets its own records table and vec0 virtual
    table. The database is a local file, so the processes sharing a partition
    must run on one host. A handle used after its partition is deleted raises
    VectorStorePartitionHandleStaleError.
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

        # Different stores may share the engine; the extension is loaded
        # once per connection.
        if not event.contains(self._engine.sync_engine, "connect", _load_sqlite_vec):
            event.listen(self._engine.sync_engine, "connect", _load_sqlite_vec)

    @override
    async def startup(self) -> None:
        async with write_connection(self._engine) as connection:
            await connection.run_sync(BaseSQLiteVecVectorStore.metadata.create_all)

    @override
    async def shutdown(self) -> None:
        pass

    @override
    async def create_partition(self, partition_key: str) -> None:
        validate_partition_key(partition_key)

        _, added = await self._add_partition_if_absent(partition_key)
        if not added:
            raise VectorStorePartitionAlreadyExistsError(
                self._vector_store_name, partition_key
            )

    @override
    async def open_or_create_partition(
        self, partition_key: str
    ) -> VectorStorePartition:
        validate_partition_key(partition_key)

        async with self._create_session() as session:
            incarnation = await self._partition_incarnation(session, partition_key)
        if incarnation is None:
            incarnation, _ = await self._add_partition_if_absent(partition_key)
        return self._partition_handle(partition_key, incarnation)

    @override
    async def open_partition(self, partition_key: str) -> VectorStorePartition | None:
        validate_partition_key(partition_key)

        async with self._create_session() as session:
            incarnation = await self._partition_incarnation(session, partition_key)
        if incarnation is None:
            return None

        return self._partition_handle(partition_key, incarnation)

    def _partition_handle(
        self, partition_key: str, incarnation: UUID
    ) -> SQLiteVecVectorStorePartition:
        return SQLiteVecVectorStorePartition(
            create_session=self._create_session,
            vector_store_name=self._vector_store_name,
            partition_key=partition_key,
            incarnation=incarnation,
            vector_dimensions=self._vector_dimensions,
            similarity_metric=self._similarity_metric,
            indexed_properties=self._indexed_properties,
            records_table=self._records_table(incarnation),
            vector_table_name=self._vector_table_name(incarnation),
        )

    @override
    async def delete_partition(self, partition_key: str) -> None:
        validate_partition_key(partition_key)

        async with write_session(self._create_session) as session:
            incarnation = await self._partition_incarnation(session, partition_key)
            if incarnation is None:
                return

            records_table = self._records_table(incarnation)
            vector_table_name = self._vector_table_name(incarnation)

            await session.execute(text(f"DROP TABLE IF EXISTS [{vector_table_name}]"))
            await session.execute(text(f"DROP TABLE IF EXISTS [{records_table.name}]"))

            await session.execute(
                delete(_PartitionRow).where(_PartitionRow.incarnation == incarnation)
            )

            self._sa_metadata.remove(records_table)

    @override
    async def purge_deleted_partitions(self) -> bool:
        # delete_partition drops the tables itself.
        return False

    # Helpers.

    def _partition_prefix(self, incarnation: UUID) -> str:
        return f"vector_store_sqlite_vec_{incarnation.hex}"

    def _records_table_name(self, incarnation: UUID) -> str:
        return f"{self._partition_prefix(incarnation)}_rc"

    def _vector_table_name(self, incarnation: UUID) -> str:
        return f"{self._partition_prefix(incarnation)}_vc"

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

    async def _partition_incarnation(
        self, session: AsyncSession, partition_key: str
    ) -> UUID | None:
        """Return the incarnation of the partition under a key, or None if there is none."""
        return (
            await session.execute(
                select(_PartitionRow.incarnation).where(
                    _PartitionRow.vector_store_name == self._vector_store_name,
                    _PartitionRow.partition_key == partition_key,
                )
            )
        ).scalar_one_or_none()

    def _records_table(self, incarnation: UUID) -> Table:
        return Table(
            self._records_table_name(incarnation),
            self._sa_metadata,
            Column("rowid", Integer, primary_key=True, autoincrement=True),
            Column("uuid", Uuid, nullable=False, unique=True),
            Column("properties", JSON, nullable=False, default=dict),
            extend_existing=True,
        )

    async def _add_partition_if_absent(self, partition_key: str) -> tuple[UUID, bool]:
        """
        Return the incarnation of the partition under a key, adding the partition if there is none, and whether this call added it.

        A partition this adds gets a new incarnation and empty tables.
        """
        async with write_session(self._create_session) as session:
            incarnation = await self._partition_incarnation(session, partition_key)
            if incarnation is not None:
                return incarnation, False
            incarnation = uuid4()
            records_table = self._records_table(incarnation)
            vector_table_name = self._vector_table_name(incarnation)
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
                    f"CREATE VIRTUAL TABLE [{vector_table_name}] USING vec0("
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
                        f"CREATE INDEX [{index_name}] "
                        f"ON [{records_table.name}]"
                        f"({compiled_expr})"
                    )
                )

            session.add(
                _PartitionRow(
                    vector_store_name=self._vector_store_name,
                    partition_key=partition_key,
                    incarnation=incarnation,
                )
            )
        return incarnation, True
