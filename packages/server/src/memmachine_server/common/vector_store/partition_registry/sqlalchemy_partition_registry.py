"""
A partition registry in a relational database, through SQLAlchemy.

A table of registered partitions keyed by vector store and partition key,
each with its incarnation and whether it is live (its storage prepared), and
a queue of deleted incarnations claimed in the order they come due. The
primary key arbitrates registration across processes, a conditional update
confirms a reservation, unregistration is one transaction, and on
PostgreSQL a purge claim is a row lock.
"""

import logging
import sqlite3
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import override
from uuid import UUID, uuid4

from pydantic import BaseModel, Field, InstanceOf, JsonValue, field_validator
from sqlalchemy import (
    JSON,
    Boolean,
    ColumnElement,
    DateTime,
    Index,
    Integer,
    Interval,
    String,
    Uuid,
    bindparam,
    case,
    delete,
    func,
    insert,
    literal,
    or_,
    select,
    update,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncConnection, AsyncEngine
from sqlalchemy.orm import DeclarativeBase, MappedColumn, mapped_column

from memmachine_server.common.utils import ensure_tz_aware
from memmachine_server.common.vector_store.data_types import (
    PartitionSchema,
    VectorStoreAttemptsExhaustedError,
    VectorStorePartitionAlreadyExistsError,
    VectorStorePartitionDeletedError,
    VectorStorePartitionHandleStaleError,
    VectorStorePartitionPendingError,
)
from memmachine_server.common.vector_store.utils import (
    _IDENTIFIER_MAX_BYTES,
    validate_identifier,
)

from .partition_registry import (
    PurgeClaim,
    Registration,
    Reservation,
    VectorStorePartitionRegistry,
)

logger = logging.getLogger(__name__)

_MAX_MINT_ATTEMPTS = 10

# The first SQLite with RETURNING, which the registry uses.
_MIN_SQLITE_VERSION = (3, 35)

# Consecutive failed purge rounds after which a tombstone is dead-lettered.
_MAX_FAILED_PURGE_ROUNDS = 10

_JSON_AUTO = JSON().with_variant(JSONB, "postgresql")


class BasePartitionRegistry(DeclarativeBase):
    """Base class for partition registry tables."""


class PartitionRow(BasePartitionRegistry):
    """A registered partition of a vector store, pending or live."""

    __tablename__ = "partition_registry_pt"

    vector_store_name: MappedColumn[str] = mapped_column(
        String(_IDENTIFIER_MAX_BYTES), primary_key=True
    )
    partition_key: MappedColumn[str] = mapped_column(
        String(_IDENTIFIER_MAX_BYTES), primary_key=True
    )
    incarnation: MappedColumn[UUID] = mapped_column(Uuid, nullable=False, unique=True)
    # The dimensions and declared schema the partition was created under, so
    # a store built with others fails loudly instead of filtering on indexes
    # that are not there.
    schema: MappedColumn[dict[str, JsonValue]] = mapped_column(
        _JSON_AUTO, nullable=False
    )
    # Whether the partition's storage is prepared; it is pending until then.
    live: MappedColumn[bool] = mapped_column(Boolean, nullable=False)
    registered_at: MappedColumn[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )


class PurgeQueueRow(BasePartitionRegistry):
    """A deleted partition's tombstone: its incarnation awaiting purge."""

    __tablename__ = "partition_registry_gc"

    incarnation: MappedColumn[UUID] = mapped_column(Uuid, primary_key=True)
    vector_store_name: MappedColumn[str] = mapped_column(
        String(_IDENTIFIER_MAX_BYTES), nullable=False
    )
    # The partition's key, kept for inspection.
    partition_key: MappedColumn[str] = mapped_column(
        String(_IDENTIFIER_MAX_BYTES), nullable=False
    )
    enqueued_at: MappedColumn[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )
    # Consecutive purge rounds on the tombstone that raised, and when the
    # last one did, on the database clock.
    failed_rounds: MappedColumn[int] = mapped_column(Integer, nullable=False, default=0)
    last_failed_at: MappedColumn[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )

    __table_args__ = (
        Index("partition_registry_gc__vs_ea", "vector_store_name", "enqueued_at"),
    )


class _RegistryInsertRejectedError(Exception):
    """A registry insert was rejected for a reason other than the key being taken."""


class SQLAlchemyVectorStorePartitionRegistryParams(BaseModel):
    """
    Parameters for SQLAlchemyVectorStorePartitionRegistry.

    Attributes:
        engine (AsyncEngine):
            Async SQLAlchemy engine, on PostgreSQL or SQLite.
        vector_store_name (str):
            The name of the store the registry serves, under which its rows are
            kept: registry objects with the same name on the same database are
            one registry, so a name identifies one store among all the stores
            whose registries share the database. It must match [a-z0-9_]+ and
            be at most 32 bytes.
        tombstone_retention_seconds (int):
            Seconds a deleted partition's records are kept before its purge
            starts, on the database clock. It must exceed, by orders of
            magnitude, the longest a write to the backend can be in flight and
            the delay before the store's reads reflect a write.
        purge_retry_backoff_seconds (int):
            Seconds before a tombstone whose purge round raised is claimed
            again, doubled for each further consecutive failure (default: 30).
        max_purge_retry_backoff_seconds (int):
            The longest the purge retry backoff grows to (default: 3600).
    """

    engine: InstanceOf[AsyncEngine] = Field(
        ..., description="Async SQLAlchemy engine, on PostgreSQL or SQLite"
    )
    vector_store_name: str = Field(
        ...,
        description=(
            "The name of the store the registry serves, under which its rows are "
            "kept: registry objects with the same name on the same database are "
            "one registry, so a name identifies one store among all the stores "
            "whose registries share the database. It must match [a-z0-9_]+ and "
            "be at most 32 bytes"
        ),
    )
    tombstone_retention_seconds: int = Field(
        ...,
        ge=0,
        description=(
            "Seconds a deleted partition's records are kept before its purge "
            "starts, on the database clock. It must exceed, by orders of "
            "magnitude, the longest a write to the backend can be in flight and "
            "the delay before the store's reads reflect a write"
        ),
    )
    purge_retry_backoff_seconds: int = Field(
        30,
        gt=0,
        description=(
            "Seconds before a tombstone whose purge round raised is claimed "
            "again, doubled for each further consecutive failure"
        ),
    )
    max_purge_retry_backoff_seconds: int = Field(
        3600, gt=0, description="The longest the purge retry backoff grows to"
    )

    @field_validator("engine")
    @classmethod
    def _validate_engine(cls, engine: AsyncEngine) -> AsyncEngine:
        if engine.dialect.name not in ("postgresql", "sqlite"):
            raise ValueError(
                f"Engine uses the {engine.dialect.name} dialect, which the "
                "registry does not support. Use PostgreSQL or SQLite."
            )
        if (
            engine.dialect.name == "sqlite"
            and sqlite3.sqlite_version_info < _MIN_SQLITE_VERSION
        ):
            minimum = ".".join(str(part) for part in _MIN_SQLITE_VERSION)
            raise ValueError(
                f"SQLite runtime {sqlite3.sqlite_version} lacks the RETURNING "
                f"support the registry depends on. Use SQLite {minimum} or newer."
            )
        return engine

    @field_validator("vector_store_name")
    @classmethod
    def _validate_vector_store_name(cls, vector_store_name: str) -> str:
        if not validate_identifier(vector_store_name):
            raise ValueError(
                f"Vector store name {vector_store_name!r} must match [a-z0-9_]+ "
                "and be at most 32 bytes"
            )
        return vector_store_name


class SQLAlchemyVectorStorePartitionRegistry(VectorStorePartitionRegistry):
    """
    The registry of one vector store's partitions.

    Registries of different vector stores share the tables, keyed by vector
    store name. After `_MAX_FAILED_PURGE_ROUNDS` consecutive failed rounds, a
    tombstone is dead-lettered: kept, its incarnation reserved, skipped by
    claims, and reported in an error log. Setting its
    `failed_rounds` back to 0 returns it to the purge.
    """

    def __init__(self, params: SQLAlchemyVectorStorePartitionRegistryParams) -> None:
        """Initialize with the provided parameters."""
        self._engine = params.engine
        self._is_sqlite = params.engine.dialect.name == "sqlite"
        self._vector_store_name = params.vector_store_name
        self._tombstone_retention = timedelta(
            seconds=params.tombstone_retention_seconds
        )
        self._purge_retry_backoff = timedelta(
            seconds=params.purge_retry_backoff_seconds
        )
        self._max_purge_retry_backoff = timedelta(
            seconds=params.max_purge_retry_backoff_seconds
        )

    @override
    async def startup(self) -> None:
        async with self._engine.begin() as connection:
            await connection.run_sync(BasePartitionRegistry.metadata.create_all)

    @override
    async def reserve(self, partition_key: str, schema: PartitionSchema) -> Reservation:
        # The primary key arbitrates the partition key across processes.
        # An insert rejected for another reason, such as a minted incarnation
        # that is registered or awaiting purge, is tried again with a fresh one.
        attempts = 0
        while True:
            incarnation = uuid4()
            try:
                await self._insert(partition_key, incarnation, schema)
            except _RegistryInsertRejectedError as err:
                logger.warning(
                    "Reserving partition %r under incarnation %s was rejected: "
                    "%s; minting another",
                    partition_key,
                    incarnation,
                    err,
                )
                attempts += 1
                if attempts >= _MAX_MINT_ATTEMPTS:
                    raise VectorStoreAttemptsExhaustedError(
                        f"Creating partition {partition_key!r} made no progress "
                        f"after {_MAX_MINT_ATTEMPTS} attempts"
                    ) from err
                continue
            return _SQLAlchemyReservation(
                partition_key=partition_key,
                schema=schema,
                incarnation=incarnation,
                engine=self._engine,
                vector_store_name=self._vector_store_name,
            )

    async def _insert(
        self, partition_key: str, incarnation: UUID, schema: PartitionSchema
    ) -> None:
        # The queue check runs after the insert, so a concurrent deletion
        # queuing a colliding incarnation, which the insert waited on, is
        # visible to it; the locking read sees the latest committed state.
        try:
            async with self._engine.begin() as connection:
                await connection.execute(
                    insert(PartitionRow).values(
                        vector_store_name=self._vector_store_name,
                        partition_key=partition_key,
                        incarnation=incarnation,
                        schema=schema.model_dump(mode="json"),
                        live=False,
                        registered_at=func.now(),
                    )
                )
                queued = (
                    await connection.execute(
                        select(PurgeQueueRow.incarnation)
                        .where(PurgeQueueRow.incarnation == incarnation)
                        .with_for_update(read=True)
                    )
                ).scalar_one_or_none()
                if queued is not None:
                    raise _RegistryInsertRejectedError("the incarnation awaits purge")
        except IntegrityError as err:
            async with self._engine.connect() as connection:
                taken = (
                    await connection.execute(
                        select(PartitionRow.incarnation).where(
                            PartitionRow.vector_store_name == self._vector_store_name,
                            PartitionRow.partition_key == partition_key,
                        )
                    )
                ).scalar_one_or_none()
            if taken is not None:
                raise VectorStorePartitionAlreadyExistsError(
                    self._vector_store_name, partition_key
                ) from err
            raise _RegistryInsertRejectedError(
                "the insert failed and no row exists under the key"
            ) from err

    @override
    async def resolve(self, partition_key: str) -> Registration | None:
        async with self._engine.connect() as connection:
            row = (
                await connection.execute(
                    select(
                        PartitionRow.incarnation,
                        PartitionRow.schema,
                        PartitionRow.live,
                        PartitionRow.registered_at,
                    ).where(
                        PartitionRow.vector_store_name == self._vector_store_name,
                        PartitionRow.partition_key == partition_key,
                    )
                )
            ).one_or_none()
        if row is None:
            return None
        schema = PartitionSchema.model_validate(row.schema)
        if not row.live:
            raise VectorStorePartitionPendingError(
                self._vector_store_name,
                partition_key,
                ensure_tz_aware(row.registered_at),
                schema,
            )
        return _SQLAlchemyRegistration(
            partition_key=partition_key,
            schema=schema,
            incarnation=row.incarnation,
            engine=self._engine,
            vector_store_name=self._vector_store_name,
        )

    @override
    async def unregister(self, partition_key: str) -> None:
        await _unregister_where(
            self._engine,
            self._vector_store_name,
            PartitionRow.partition_key == partition_key,
        )

    @override
    @asynccontextmanager
    async def claim_purgeable_incarnation(
        self,
    ) -> AsyncGenerator[PurgeClaim | None, None]:
        # The retention is applied when a claim is decided, on the database
        # clock, so a changed retention applies to every tombstone. On
        # PostgreSQL the claim is a row lock held for the body, which a
        # concurrent purger skips; on SQLite two purgers may run the same
        # round. A round that raises rolls back, then counts against
        # the tombstone in a transaction of its own.
        claimed: UUID | None = None
        try:
            async with self._engine.begin() as connection:
                row = (
                    await connection.execute(
                        select(PurgeQueueRow.incarnation, PurgeQueueRow.failed_rounds)
                        .where(
                            PurgeQueueRow.vector_store_name == self._vector_store_name,
                            PurgeQueueRow.enqueued_at <= self._retention_cutoff(),
                            PurgeQueueRow.failed_rounds < _MAX_FAILED_PURGE_ROUNDS,
                            or_(
                                PurgeQueueRow.failed_rounds == 0,
                                PurgeQueueRow.last_failed_at <= self._backoff_cutoff(),
                            ),
                        )
                        .order_by(PurgeQueueRow.enqueued_at)
                        .limit(1)
                        .with_for_update(skip_locked=True)
                    )
                ).one_or_none()
                if row is None:
                    yield None
                    return
                claimed = row.incarnation
                claim = PurgeClaim(incarnation=row.incarnation)
                yield claim
                await self._record_round(connection, claim, row.failed_rounds)
        except Exception as error:
            if claimed is not None:
                await self._count_failed_round(claimed, error)
            raise

    async def _record_round(
        self, connection: AsyncConnection, claim: PurgeClaim, failed_rounds: int
    ) -> None:
        """Record a purge round's outcome, in the claim's transaction.

        A round that found no records removes the tombstone; one that found
        records resets its count of failed rounds.
        """
        if claim.any_records_found is None:
            raise RuntimeError(
                f"Purge round for incarnation {claim.incarnation} ended "
                "without setting any_records_found"
            )
        if not claim.any_records_found:
            await connection.execute(
                delete(PurgeQueueRow).where(
                    PurgeQueueRow.incarnation == claim.incarnation
                )
            )
        elif failed_rounds:
            await connection.execute(
                update(PurgeQueueRow)
                .where(PurgeQueueRow.incarnation == claim.incarnation)
                .values(failed_rounds=0)
            )

    async def _count_failed_round(self, incarnation: UUID, error: Exception) -> None:
        """Count a raised purge round against a tombstone, and report its dead-lettering."""
        async with self._engine.begin() as connection:
            failed_rounds = (
                await connection.execute(
                    update(PurgeQueueRow)
                    .where(PurgeQueueRow.incarnation == incarnation)
                    .values(
                        failed_rounds=PurgeQueueRow.failed_rounds + 1,
                        last_failed_at=func.now(),
                    )
                    .returning(PurgeQueueRow.failed_rounds)
                )
            ).scalar_one_or_none()
        if failed_rounds == _MAX_FAILED_PURGE_ROUNDS:
            logger.error(
                "Purge of incarnation %s failed %d rounds in a row and is "
                "dead-lettered: its records stay and it is no longer claimed. "
                "Last error: %r. Set its failed_rounds to 0 in %s to retry it.",
                incarnation,
                failed_rounds,
                error,
                PurgeQueueRow.__tablename__,
            )

    def _backoff_cutoff(self) -> ColumnElement:
        """The database clock's now, less each queue row's backoff, computed by the database."""
        # 1 << (f - 1) is 2 ** (f - 1) on both dialects. The claim evaluates
        # it for rows at 0 failures too, so the count is clamped at 0: a
        # shift by a negative count is undefined on PostgreSQL.
        doublings = literal(1, Integer).op("<<")(
            case(
                (PurgeQueueRow.failed_rounds > 0, PurgeQueueRow.failed_rounds - 1),
                else_=0,
            )
        )
        if self._is_sqlite:
            seconds = func.min(
                int(self._purge_retry_backoff.total_seconds()) * doublings,
                int(self._max_purge_retry_backoff.total_seconds()),
            )
            return func.datetime(
                "now",
                func.printf("-%d seconds", seconds),
                type_=DateTime(timezone=True),
            )
        return func.now() - func.least(
            bindparam("retry_backoff", self._purge_retry_backoff, type_=Interval)
            * doublings,
            bindparam(
                "max_retry_backoff", self._max_purge_retry_backoff, type_=Interval
            ),
        )

    def _retention_cutoff(self) -> ColumnElement:
        """The database clock's now, less the retention, computed by the database."""
        if self._is_sqlite:
            # datetime() emits the form CURRENT_TIMESTAMP stamps with, so
            # the stamps and the cutoff compare as text in time order.
            seconds = int(self._tombstone_retention.total_seconds())
            return func.datetime(
                "now",
                bindparam("retention_modifier", f"-{seconds} seconds"),
                type_=DateTime(timezone=True),
            )
        return func.now() - bindparam(
            "retention", self._tombstone_retention, type_=Interval
        )


@dataclass(frozen=True)
class _SQLAlchemyReservation(Reservation):
    """A reservation whose methods write the registry's tables directly."""

    engine: AsyncEngine = field(repr=False, compare=False)
    vector_store_name: str = field(repr=False)

    @override
    async def confirm(self) -> Registration:
        # Conditional on the incarnation and on the row being pending, so a
        # creation marks live only the partition it reserved.
        async with self.engine.begin() as connection:
            result = await connection.execute(
                update(PartitionRow)
                .where(
                    PartitionRow.vector_store_name == self.vector_store_name,
                    PartitionRow.partition_key == self.partition_key,
                    PartitionRow.incarnation == self.incarnation,
                    PartitionRow.live.is_(False),
                )
                .values(live=True)
            )
        if result.rowcount != 1:
            raise VectorStorePartitionDeletedError(
                self.vector_store_name, self.partition_key
            )
        return _SQLAlchemyRegistration(
            partition_key=self.partition_key,
            schema=self.schema,
            incarnation=self.incarnation,
            engine=self.engine,
            vector_store_name=self.vector_store_name,
        )

    @override
    async def cancel(self) -> None:
        # Conditional on the row being pending, as confirm is, so a cancel
        # after a confirmation that committed leaves the live partition.
        await _unregister_where(
            self.engine,
            self.vector_store_name,
            PartitionRow.incarnation == self.incarnation,
            PartitionRow.live.is_(False),
        )


@dataclass(frozen=True)
class _SQLAlchemyRegistration(Registration):
    """A registration whose method reads the registry's tables directly."""

    engine: AsyncEngine = field(repr=False, compare=False)
    vector_store_name: str = field(repr=False)

    @override
    async def require_current(self) -> None:
        async with self.engine.connect() as connection:
            current = (
                await connection.execute(
                    select(PartitionRow.incarnation).where(
                        PartitionRow.vector_store_name == self.vector_store_name,
                        PartitionRow.partition_key == self.partition_key,
                    )
                )
            ).scalar_one_or_none()
        if current != self.incarnation:
            raise VectorStorePartitionHandleStaleError(
                self.vector_store_name, self.partition_key
            )


async def _unregister_where(
    engine: AsyncEngine, vector_store_name: str, *conditions: ColumnElement[bool]
) -> None:
    """Delete the partition row of a vector store the conditions select, and queue its tombstone.

    One transaction, so the partition is unreachable once it commits. The
    DELETE goes first and takes the row's write lock, so racing deleters
    serialize on it and the loser finds no row and returns.
    """
    async with engine.begin() as connection:
        row = (
            await connection.execute(
                delete(PartitionRow)
                .where(PartitionRow.vector_store_name == vector_store_name, *conditions)
                .returning(PartitionRow.incarnation, PartitionRow.partition_key)
            )
        ).one_or_none()
        if row is None:
            return
        await connection.execute(
            insert(PurgeQueueRow).values(
                incarnation=row.incarnation,
                vector_store_name=vector_store_name,
                partition_key=row.partition_key,
                enqueued_at=func.now(),
            )
        )
