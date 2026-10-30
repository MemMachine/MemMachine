"""
Vector store backed by SQLite + sqlite-vec.

Each life of a partition gets its own records table and vec0 virtual table,
since sqlite-vec's ANN indexes may not support partition keys.

A partition's records table carries one typed, indexed column per declared
property, and a filtered query hands the KNN an allowlist of the rows the
filter admits, so the filter is evaluated during the search: vec0 ranks only
the allowed rows, and a filtered search returns fewer only when the filter
admits fewer.
"""

import logging
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import ClassVar, override
from uuid import UUID, uuid4

import aiosqlite
import sqlite_vec
from pydantic import Field, InstanceOf, field_validator
from sqlalchemy import (
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
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import DeclarativeBase, MappedColumn, mapped_column
from sqlalchemy.pool import ConnectionPoolEntry, StaticPool
from sqlalchemy.sql.elements import ColumnElement

from memmachine_server.common.data_types import (
    SimilarityMetric,
)
from memmachine_server.common.filter.filter_parser import FilterExpr
from memmachine_server.common.schema_migration import (
    Lineage,
    check_lineage,
    upgrade_lineage,
)
from memmachine_server.common.sql import write_connection, write_session

from .data_types import (
    _IDENTIFIER_MAX_BYTES,
    QueryMatch,
    QueryResult,
    Record,
    VectorStoreAttemptsExhaustedError,
    VectorStorePartitionAlreadyExistsError,
    VectorStorePartitionHandleStaleError,
    VectorStoreSchema,
)
from .sql_store_schema import (
    check_store_schema,
    record_store_schema,
    store_schema_tables,
)
from .sqlite_properties import (
    compile_property_filter,
    property_column_names,
    property_column_values,
    property_columns,
    property_indexes,
)
from .utils import (
    _PARTITION_KEY_MAX_BYTES,
    validate_declared_properties,
    validate_dimensions,
    validate_limit,
    validate_partition_key,
    validate_query_vector,
    validate_score_threshold,
    validate_supported_filter,
)
from .vector_store import VectorStore, VectorStoreParams, VectorStorePartition

logger = logging.getLogger(__name__)


# Rejected inserts in a row before creating a partition gives up: each
# retry needs a minted incarnation to collide, or another creator or a
# deleter to change the key meanwhile, so this many means a lasting error.
_MAX_MINT_ATTEMPTS = 10


class _RegistryInsertRejectedError(Exception):
    """A partition's registry insert was rejected although its key is free."""


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
    incarnation: MappedColumn[UUID] = mapped_column(Uuid, nullable=False, unique=True)


_STORE_SCHEMA_TABLES = store_schema_tables(
    BaseSQLiteVecVectorStore.metadata, "vector_store_sqlite_vec"
)

SQLITE_VEC_VECTOR_STORE_LINEAGE = Lineage(
    name="sqlite_vec_vector_store",
    versions=Path(__file__).parent / "migrations" / "sqlite_vec_vector_store",
    metadata=BaseSQLiteVecVectorStore.metadata,
)


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
        store_schema: VectorStoreSchema,
        records_table: Table,
        vector_table_name: str,
    ) -> None:
        """Initialize a handle on one life of a partition."""
        self._create_session = create_session
        self._vector_store_name = vector_store_name
        self._partition_key = partition_key
        self._incarnation = incarnation
        self._store_schema = store_schema
        self._records_table = records_table
        self._vector_table_name = vector_table_name

    @property
    @override
    def store_schema(self) -> VectorStoreSchema:
        return self._store_schema

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
            validate_declared_properties(
                record.properties, self._store_schema.indexed_properties
            )
            validate_dimensions(record.vector, self._store_schema.vector_dimensions)

        column_names = property_column_names(self._store_schema.indexed_properties)
        insert_records = sqlite_insert(self._records_table)
        upsert_records = insert_records.on_conflict_do_update(
            index_elements=[self._records_table.c.uuid],
            # With no declared column the update is a no-op that still
            # returns the existing row, which `RETURNING` needs.
            set_={
                name: insert_records.excluded[name]
                for name in (column_names or ["uuid"])
            },
        ).returning(self._records_table.c.uuid, self._records_table.c.rowid)

        async with write_session(self._create_session) as session:
            await self._require_current(session)
            if not records:
                return
            rows = (
                await session.execute(
                    upsert_records,
                    [
                        {
                            "uuid": record.uuid,
                            **property_column_values(
                                record.properties, self._store_schema.indexed_properties
                            ),
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
            validate_query_vector(query_vector, self._store_schema.vector_dimensions)
        validate_score_threshold(score_threshold)

        filter_expression: ColumnElement[bool] | None = None
        if property_filter is not None:
            validate_supported_filter(
                property_filter, self._store_schema.indexed_properties
            )
            filter_expression = compile_property_filter(
                property_filter,
                self._records_table,
                self._store_schema.indexed_properties,
            )

        k = min(limit, self._MAX_K)

        results: list[QueryResult] = []
        async with self._create_session() as session:
            await self._require_current(session)
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
                    score_threshold=score_threshold,
                )
                results.append(QueryResult(matches=matches))

        return results

    def _knn_statement(
        self,
        query_blob: bytes,
        k: int,
        filter_expression: ColumnElement[bool] | None,
    ) -> Select:
        """The KNN over the partition's vectors, restricted to the rows a filter admits.

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
        score_threshold: float | None,
    ) -> list[QueryMatch]:
        matched_rowids = list(rowid_to_distance.keys())

        matched_rows = (
            await session.execute(
                select(self._records_table.c.uuid, self._records_table.c.rowid).where(
                    self._records_table.c.rowid.in_(matched_rowids),
                )
            )
        ).all()

        matches: list[QueryMatch] = []
        for row in matched_rows:
            score = self._distance_to_score(
                rowid_to_distance[row.rowid], self._store_schema.similarity_metric
            )
            if score_threshold is not None and (
                score < score_threshold
                if self._store_schema.similarity_metric.higher_is_better
                else score > score_threshold
            ):
                continue

            matches.append(QueryMatch(score=score, record_uuid=row.uuid))

        matches.sort(
            key=lambda match: match.score,
            reverse=self._store_schema.similarity_metric.higher_is_better,
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


def load_sqlite_vec_extension(engine: AsyncEngine) -> None:
    """Load the sqlite-vec extension on every connection of a SQLite engine.

    Loadable extensions are per-connection state, so the loader is
    registered at the engine's creation, before any connection is pooled.
    """

    @event.listens_for(engine.sync_engine, "connect")
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

    Stores of different names may share the engine. The schema's similarity
    metric must be cosine or euclidean.

    Attributes:
        engine (AsyncEngine):
            Async SQLAlchemy engine (sqlite+aiosqlite), set up by
            `enable_sqlite_explicit_begin` and `load_sqlite_vec_extension` at
            its creation.
    """

    engine: InstanceOf[AsyncEngine] = Field(
        ...,
        description=(
            "Async SQLAlchemy engine (sqlite+aiosqlite), set up by "
            "enable_sqlite_explicit_begin and load_sqlite_vec_extension at its "
            "creation"
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
    must run on one host.
    """

    _SIMILARITY_METRIC_TO_SQLITE_VEC_DISTANCE: ClassVar[dict[SimilarityMetric, str]] = {
        SimilarityMetric.COSINE: "cosine",
        SimilarityMetric.EUCLIDEAN: "L2",
    }

    def __init__(self, params: SQLiteVecVectorStoreParams) -> None:
        """Initialize the vector store with the provided parameters."""
        SQLiteVecVectorStore._validate_metric(params.store_schema.similarity_metric)
        self._engine = params.engine
        self._vector_store_name = params.vector_store_name
        self._store_schema = params.store_schema
        self._create_session = async_sessionmaker(self._engine, expire_on_commit=False)
        self._sa_metadata = MetaData()

    @override
    async def migrate(self) -> None:
        await upgrade_lineage(self._engine, SQLITE_VEC_VECTOR_STORE_LINEAGE)
        async with write_connection(self._engine) as connection:
            await record_store_schema(
                connection,
                _STORE_SCHEMA_TABLES,
                self._vector_store_name,
                self._store_schema,
            )
        await check_store_schema(
            self._engine,
            _STORE_SCHEMA_TABLES,
            self._vector_store_name,
            self._store_schema,
        )

    @override
    async def startup(self) -> None:
        await check_lineage(self._engine, SQLITE_VEC_VECTOR_STORE_LINEAGE)
        await check_store_schema(
            self._engine,
            _STORE_SCHEMA_TABLES,
            self._vector_store_name,
            self._store_schema,
        )

    @override
    async def shutdown(self) -> None:
        pass

    @override
    async def create_partition(self, partition_key: str) -> None:
        validate_partition_key(partition_key)

        attempts = 0
        while True:
            try:
                await self._insert_partition(partition_key, uuid4())
            except _RegistryInsertRejectedError as err:
                logger.warning(
                    "Creating partition %r was rejected: %s; minting another "
                    "incarnation",
                    partition_key,
                    err,
                )
                attempts += 1
                if attempts >= _MAX_MINT_ATTEMPTS:
                    raise VectorStoreAttemptsExhaustedError(
                        f"Creating partition {partition_key!r} made no progress "
                        f"after {_MAX_MINT_ATTEMPTS} attempts"
                    ) from err
                continue
            return

    @override
    async def open_or_create_partition(
        self, partition_key: str
    ) -> VectorStorePartition:
        validate_partition_key(partition_key)

        # Read-then-insert, retried: losing the insert means another creator
        # won (open its partition next), and finding no partition after
        # losing means a deleter removed the winner.
        attempts = 0
        while True:
            async with self._create_session() as session:
                incarnation = await self._partition_incarnation(session, partition_key)
            if incarnation is not None:
                return self._partition_handle(partition_key, incarnation)

            incarnation = uuid4()
            try:
                await self._insert_partition(partition_key, incarnation)
            except (
                VectorStorePartitionAlreadyExistsError,
                _RegistryInsertRejectedError,
            ) as err:
                if isinstance(err, _RegistryInsertRejectedError):
                    logger.warning(
                        "Creating partition %r was rejected: %s; minting another "
                        "incarnation",
                        partition_key,
                        err,
                    )
                attempts += 1
                if attempts >= _MAX_MINT_ATTEMPTS:
                    raise VectorStoreAttemptsExhaustedError(
                        f"Opening or creating partition {partition_key!r} made no "
                        f"progress after {_MAX_MINT_ATTEMPTS} attempts"
                    ) from err
                continue
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
            store_schema=self._store_schema,
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
        """The partition life's records table, built once per incarnation."""
        name = self._records_table_name(incarnation)
        records_table = self._sa_metadata.tables.get(name)
        if records_table is not None:
            return records_table
        records_table = Table(
            name,
            self._sa_metadata,
            Column("rowid", Integer, primary_key=True, autoincrement=True),
            Column("uuid", Uuid, nullable=False, unique=True),
            *property_columns(self._store_schema.indexed_properties),
        )
        property_indexes(records_table, self._store_schema.indexed_properties)
        return records_table

    async def _insert_partition(self, partition_key: str, incarnation: UUID) -> None:
        """
        Insert a partition's registry row under a minted incarnation, and create its tables.

        The row goes in first, so the registry's unique constraints decide the
        key and the incarnation, and the tables are created in its
        transaction.

        Raises:
            VectorStorePartitionAlreadyExistsError:
                If the partition key is taken.
            _RegistryInsertRejectedError:
                If the insert is rejected for a reason other than the key
                being taken.
        """
        try:
            async with write_session(self._create_session) as session:
                session.add(
                    _PartitionRow(
                        vector_store_name=self._vector_store_name,
                        partition_key=partition_key,
                        incarnation=incarnation,
                    )
                )
                await session.flush()

                records_table = self._records_table(incarnation)
                vector_table_name = self._vector_table_name(incarnation)
                distance_metric_value = self._SIMILARITY_METRIC_TO_SQLITE_VEC_DISTANCE[
                    self._store_schema.similarity_metric
                ]

                connection = await session.connection()
                await connection.run_sync(
                    self._sa_metadata.create_all,
                    tables=[records_table],
                )

                await session.execute(
                    text(
                        f"CREATE VIRTUAL TABLE [{vector_table_name}] USING vec0("
                        f"vector float[{self._store_schema.vector_dimensions}] distance_metric={distance_metric_value}"
                        f")"
                    )
                )
        except IntegrityError as err:
            async with self._create_session() as session:
                taken = await self._partition_incarnation(session, partition_key)
            if taken is not None:
                raise VectorStorePartitionAlreadyExistsError(
                    self._vector_store_name, partition_key
                ) from err
            raise _RegistryInsertRejectedError(
                f"the insert of incarnation {incarnation} failed and no "
                "partition exists under the key"
            ) from err
