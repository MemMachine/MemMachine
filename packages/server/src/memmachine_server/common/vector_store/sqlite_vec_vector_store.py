"""
Vector store backed by SQLite + sqlite-vec.

Each partition gets its own records table and vec0 virtual table,
rather than a vec0 partition key column, which sqlite-vec's ANN indexes may
not support.

A partition's records table carries one typed, indexed column per declared
property, and a filtered query hands the KNN an allowlist of the rows the
filter admits, so the filter is evaluated during the search: vec0 ranks only
the allowed rows, and a filtered search returns fewer only when the filter
admits fewer.
"""

from collections.abc import Iterable, Mapping, Sequence
from typing import ClassVar, override
from uuid import UUID

import aiosqlite
import sqlite_vec
from pydantic import Field, InstanceOf, field_validator
from sqlalchemy import (
    Column,
    Integer,
    MetaData,
    Select,
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
from sqlalchemy.pool import ConnectionPoolEntry
from sqlalchemy.sql.elements import ColumnElement

from memmachine_server.common.data_types import (
    SimilarityMetric,
)
from memmachine_server.common.filter.filter_parser import FilterExpr
from memmachine_server.common.sql import validate_engine_connections, write_session

from .data_types import (
    QueryMatch,
    QueryResult,
    Record,
    VectorStoreSchema,
)
from .sqlite_properties import (
    compile_property_filter,
    property_column_names,
    property_column_values,
    property_columns,
    property_indexes,
)
from .utils import (
    validate_declared_properties,
    validate_dimensions,
    validate_limit,
    validate_query_vector,
    validate_score_threshold,
    validate_supported_filter,
)
from .vector_store import VectorStore, VectorStoreParams, VectorStorePartition


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
        partition_uuid: UUID,
        store_schema: VectorStoreSchema,
        records_table: Table,
        vector_table_name: str,
    ) -> None:
        """Initialize a handle on one partition."""
        self._create_session = create_session
        self._partition_uuid = partition_uuid
        self._store_schema = store_schema
        self._records_table = records_table
        self._vector_table_name = vector_table_name

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
        if not records:
            return

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
        if not record_uuids:
            return

        async with write_session(self._create_session) as session:
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
        validate_engine_connections(engine)
        return engine


class SQLiteVecVectorStore(VectorStore):
    """
    Vector store backed by SQLite + sqlite-vec.

    Each partition gets its own records table and vec0 virtual
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
        self._store_schema = params.store_schema
        self._create_session = async_sessionmaker(self._engine, expire_on_commit=False)
        self._sa_metadata = MetaData()

    @property
    @override
    def store_schema(self) -> VectorStoreSchema:
        return self._store_schema

    @override
    async def startup(self) -> None:
        # Each partition's tables are created when it is
        # prepared; nothing is shared.
        pass

    @override
    async def shutdown(self) -> None:
        pass

    @override
    async def create_partition(self, partition_uuid: UUID) -> None:
        records_table = self._records_table(partition_uuid)
        vector_table_name = self._vector_table_name(partition_uuid)
        distance_metric_value = self._SIMILARITY_METRIC_TO_SQLITE_VEC_DISTANCE[
            self._store_schema.similarity_metric
        ]
        async with write_session(self._create_session) as session:
            connection = await session.connection()
            await connection.run_sync(
                self._sa_metadata.create_all,
                tables=[records_table],
            )
            await session.execute(
                text(
                    f"CREATE VIRTUAL TABLE IF NOT EXISTS [{vector_table_name}] "
                    f"USING vec0(vector float[{self._store_schema.vector_dimensions}] "
                    f"distance_metric={distance_metric_value})"
                )
            )

    @override
    async def open_partition(
        self, partition_uuid: UUID
    ) -> SQLiteVecVectorStorePartition:
        return SQLiteVecVectorStorePartition(
            create_session=self._create_session,
            partition_uuid=partition_uuid,
            store_schema=self._store_schema,
            records_table=self._records_table(partition_uuid),
            vector_table_name=self._vector_table_name(partition_uuid),
        )

    @override
    async def purge_partition(self, partition_uuid: UUID) -> bool:
        # A partition's records live in its own tables, which drop_partition
        # removes.
        return False

    @override
    async def drop_partition(self, partition_uuid: UUID) -> None:
        records_table = self._records_table(partition_uuid)
        vector_table_name = self._vector_table_name(partition_uuid)
        async with write_session(self._create_session) as session:
            await session.execute(text(f"DROP TABLE IF EXISTS [{vector_table_name}]"))
            await session.execute(text(f"DROP TABLE IF EXISTS [{records_table.name}]"))
        self._sa_metadata.remove(records_table)

    # Helpers.

    def _partition_prefix(self, partition_uuid: UUID) -> str:
        return f"vector_store_sqlite_vec_{partition_uuid.hex}"

    def _records_table_name(self, partition_uuid: UUID) -> str:
        return f"{self._partition_prefix(partition_uuid)}_rc"

    def _vector_table_name(self, partition_uuid: UUID) -> str:
        return f"{self._partition_prefix(partition_uuid)}_vc"

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

    def _records_table(self, partition_uuid: UUID) -> Table:
        """The partition's records table, built once per partition."""
        name = self._records_table_name(partition_uuid)
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
