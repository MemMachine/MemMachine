"""
A partition registry in a relational database, through SQLAlchemy.

The registry keeps a table of registered partitions and a queue of deleted
partitions' UUIDs. A registered partition is keyed by namespace and partition
name, and holds its UUID and whether it is live; queued UUIDs are claimed in
the order they come due. The primary key arbitrates registration across
processes, a conditional update confirms a reservation, unregistration is one
transaction, and a purge claim is a lease on a tombstone, taken by one short
write and ended by another.
"""

import logging
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import NoReturn, override
from uuid import UUID, uuid4

from pydantic import BaseModel, Field, InstanceOf, field_validator
from sqlalchemy import (
    Boolean,
    ColumnElement,
    DateTime,
    Index,
    Integer,
    Interval,
    String,
    Uuid,
    case,
    delete,
    func,
    insert,
    literal,
    or_,
    select,
    update,
)
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncConnection, AsyncEngine
from sqlalchemy.orm import (
    DeclarativeBase,
    InstrumentedAttribute,
    MappedColumn,
    mapped_column,
)

from memmachine_server.common.schema_migration import (
    Lineage,
    check_lineage,
    upgrade_lineage,
)
from memmachine_server.common.sql import validate_engine_connections, write_connection
from memmachine_server.common.utils import ensure_tz_aware

from .data_types import (
    _PARTITION_NAME_MAX_BYTES,
    _PARTITION_NAMESPACE_MAX_BYTES,
    PartitionAlreadyConfirmedError,
    PartitionAlreadyExistsError,
    PartitionCreateAttemptsExhaustedError,
    PartitionDeletedError,
    PartitionPendingError,
    TombstoneClaimStaleError,
    validate_partition_namespace,
)
from .partition_registry import (
    PartitionRegistry,
    Registration,
    Reservation,
    TombstoneClaim,
)

logger = logging.getLogger(__name__)

_MAX_MINT_ATTEMPTS = 10

# The first SQLite with RETURNING, which the registry uses.
_MIN_SQLITE_VERSION = (3, 35)


class BasePartitionRegistry(DeclarativeBase):
    """Base class for partition registry tables."""


class PartitionRow(BasePartitionRegistry):
    """A registered partition in a namespace, pending or live."""

    __tablename__ = "partition_registry_pt"

    partition_namespace: MappedColumn[str] = mapped_column(
        String(_PARTITION_NAMESPACE_MAX_BYTES), primary_key=True
    )
    partition_name: MappedColumn[str] = mapped_column(
        String(_PARTITION_NAME_MAX_BYTES), primary_key=True
    )
    partition_uuid: MappedColumn[UUID] = mapped_column(
        Uuid, nullable=False, unique=True
    )
    # Whether the partition's reservation is confirmed; it is pending until then.
    live: MappedColumn[bool] = mapped_column(Boolean, nullable=False)
    registered_at: MappedColumn[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )


class PurgeQueueRow(BasePartitionRegistry):
    """
    A deleted partition's tombstone: its partition UUID awaiting purge.

    It holds only what a purge round reads.
    """

    __tablename__ = "partition_registry_gc"

    partition_uuid: MappedColumn[UUID] = mapped_column(Uuid, primary_key=True)
    partition_namespace: MappedColumn[str] = mapped_column(
        String(_PARTITION_NAMESPACE_MAX_BYTES), nullable=False
    )
    enqueued_at: MappedColumn[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )
    # Claims taken since a round that held the latest claim found records of
    # the partition, the open claim included.
    attempts_without_progress: MappedColumn[int] = mapped_column(
        Integer, nullable=False, default=0
    )
    # When a claim may next take the tombstone: after the open claim's lease
    # and then the backoff, or after the backoff from a failed round. None
    # before the first claim and after the latest claim's round found
    # records, so a claim may take the tombstone at once.
    claimable_at: MappedColumn[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    # Advanced by each claim and by each end of a claim, so an end takes
    # effect once, and only under the latest claim.
    claim_generation: MappedColumn[int] = mapped_column(
        Integer, nullable=False, default=0
    )

    __table_args__ = (
        Index("partition_registry_gc__ns_ea", "partition_namespace", "enqueued_at"),
    )


PARTITION_REGISTRY_LINEAGE = Lineage(
    name="partition_registry",
    versions=Path(__file__).parent / "migrations",
    metadata=BasePartitionRegistry.metadata,
)


class _RegistryInsertRejectedError(Exception):
    """A registry insert was rejected for a reason other than the name being taken."""


class SQLAlchemyPartitionRegistryParams(BaseModel):
    """
    Parameters for SQLAlchemyPartitionRegistry.

    Attributes:
        engine (AsyncEngine):
            Async SQLAlchemy engine, on PostgreSQL or SQLite. On SQLite, set
            up by enable_sqlite_explicit_begin at its creation: the
            registry's write transactions take SQLite's write lock at BEGIN.
        partition_namespace (str):
            The namespace the registry serves: registry objects with the same
            namespace on the same database are one registry.
        tombstone_retention_seconds (int):
            Seconds from a partition's deletion until its tombstone comes due,
            on the database clock. It must exceed, by a wide margin, the
            longest a write to a partition's records can be in flight and the
            delay before a read reflects a write.
        purge_lease_seconds (int):
            Seconds a purge round's claim holds its tombstone, on the database
            clock; it should exceed the longest round (default: 300).
        base_purge_retry_backoff_seconds (int):
            Seconds before a tombstone is claimable again after a failed purge
            round, or after the lease of a claim never ended, doubled for each
            further attempt without progress (default: 30).
        max_purge_retry_backoff_seconds (int):
            The longest the purge retry backoff grows to (default: 86400).
    """

    engine: InstanceOf[AsyncEngine] = Field(
        ...,
        description=(
            "Async SQLAlchemy engine, on PostgreSQL or SQLite. On SQLite, set "
            "up by enable_sqlite_explicit_begin at its creation: the "
            "registry's write transactions take SQLite's write lock at BEGIN"
        ),
    )
    partition_namespace: str = Field(
        ...,
        description=(
            "The namespace the registry serves: registry objects with the same "
            "namespace on the same database are one registry"
        ),
    )
    tombstone_retention_seconds: int = Field(
        ...,
        ge=0,
        description=(
            "Seconds from a partition's deletion until its tombstone comes due, "
            "on the database clock. It must exceed, by a wide margin, the "
            "longest a write to a partition's records can be in flight and the "
            "delay before a read reflects a write"
        ),
    )
    purge_lease_seconds: int = Field(
        300,
        gt=0,
        description=(
            "Seconds a purge round's claim holds its tombstone, on the database "
            "clock; it should exceed the longest round"
        ),
    )
    base_purge_retry_backoff_seconds: int = Field(
        30,
        gt=0,
        description=(
            "Seconds before a tombstone is claimable again after a failed purge "
            "round, or after the lease of a claim never ended, doubled for each "
            "further attempt without progress"
        ),
    )
    max_purge_retry_backoff_seconds: int = Field(
        86400, gt=0, description="The longest the purge retry backoff grows to"
    )

    @field_validator("engine")
    @classmethod
    def _validate_engine(cls, engine: AsyncEngine) -> AsyncEngine:
        if engine.dialect.name not in ("postgresql", "sqlite"):
            raise ValueError(
                f"Engine uses the {engine.dialect.name} dialect, which the "
                "registry does not support. Use PostgreSQL or SQLite."
            )
        # The registry's arbitration needs every transaction on its own
        # connection to one database.
        validate_engine_connections(engine)
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

    @field_validator("partition_namespace")
    @classmethod
    def _validate_partition_namespace(cls, partition_namespace: str) -> str:
        validate_partition_namespace(partition_namespace)
        return partition_namespace


class SQLAlchemyPartitionRegistry(PartitionRegistry):
    """
    The registry of one partition namespace, in a relational database.

    Registries of different namespaces share the tables, keyed by namespace.
    No transaction stays open during a purge round.

    A tombstone whose purge rounds keep failing is retried without end, at
    most `max_purge_retry_backoff_seconds` after each failure.
    """

    def __init__(self, params: SQLAlchemyPartitionRegistryParams) -> None:
        """Initialize with the provided parameters."""
        self._engine = params.engine
        self._partition_namespace = params.partition_namespace
        self._tombstone_retention_seconds = params.tombstone_retention_seconds
        self._purge_lease_seconds = params.purge_lease_seconds
        self._base_purge_retry_backoff_seconds = params.base_purge_retry_backoff_seconds
        self._max_purge_retry_backoff_seconds = params.max_purge_retry_backoff_seconds

    @override
    async def prepare(self) -> None:
        await upgrade_lineage(self._engine, PARTITION_REGISTRY_LINEAGE)

    @override
    async def startup(self) -> None:
        await check_lineage(self._engine, PARTITION_REGISTRY_LINEAGE)

    @override
    async def _reserve(self, partition_name: str) -> Reservation:
        # The primary key arbitrates the partition name across processes.
        # An insert rejected for another reason, such as a minted UUID that is
        # registered or awaiting purge, is tried again with a fresh one.
        attempts = 0
        while True:
            partition_uuid = uuid4()
            try:
                await self._insert_pending_partition(partition_name, partition_uuid)
            except _RegistryInsertRejectedError as err:
                attempts += 1
                if attempts >= _MAX_MINT_ATTEMPTS:
                    raise PartitionCreateAttemptsExhaustedError(
                        f"Creating partition {partition_name!r} made no progress "
                        f"after {_MAX_MINT_ATTEMPTS} attempts"
                    ) from err
                logger.warning(
                    "Reserving partition %r under UUID %s was rejected: "
                    "%s; minting another",
                    partition_name,
                    partition_uuid,
                    err,
                )
                continue
            return _SQLAlchemyReservation(
                partition_name=partition_name,
                partition_uuid=partition_uuid,
                engine=self._engine,
                partition_namespace=self._partition_namespace,
            )

    async def _insert_pending_partition(
        self, partition_name: str, partition_uuid: UUID
    ) -> None:
        # The queue check runs after the insert, which waits for a concurrent
        # deletion of a partition with the same UUID to commit, so the check
        # sees that deletion's tombstone; the locking read sees the latest
        # committed state.
        try:
            async with write_connection(self._engine) as connection:
                await connection.execute(
                    insert(PartitionRow).values(
                        partition_namespace=self._partition_namespace,
                        partition_name=partition_name,
                        partition_uuid=partition_uuid,
                        live=False,
                        registered_at=func.now(),
                    )
                )
                queued = (
                    await connection.execute(
                        select(PurgeQueueRow.partition_uuid)
                        .where(PurgeQueueRow.partition_uuid == partition_uuid)
                        .with_for_update(read=True)
                    )
                ).scalar_one_or_none()
                if queued is not None:
                    raise _RegistryInsertRejectedError("the UUID awaits purge")
        except IntegrityError as err:
            async with self._engine.connect() as connection:
                taken = (
                    await connection.execute(
                        select(PartitionRow.partition_uuid).where(
                            PartitionRow.partition_namespace
                            == self._partition_namespace,
                            PartitionRow.partition_name == partition_name,
                        )
                    )
                ).scalar_one_or_none()
            if taken is not None:
                raise PartitionAlreadyExistsError(
                    self._partition_namespace, partition_name
                ) from err
            raise _RegistryInsertRejectedError(
                "the insert failed and no row exists under the name"
            ) from err

    @override
    async def _resolve(self, partition_name: str) -> UUID | None:
        async with self._engine.connect() as connection:
            row = (
                await connection.execute(
                    select(
                        PartitionRow.partition_uuid,
                        PartitionRow.live,
                        PartitionRow.registered_at,
                    ).where(
                        PartitionRow.partition_namespace == self._partition_namespace,
                        PartitionRow.partition_name == partition_name,
                    )
                )
            ).one_or_none()
        if row is None:
            return None
        if not row.live:
            raise PartitionPendingError(
                self._partition_namespace,
                partition_name,
                ensure_tz_aware(row.registered_at),
            )
        return row.partition_uuid

    @override
    def _registration(self, partition_name: str, partition_uuid: UUID) -> Registration:
        return _SQLAlchemyRegistration(
            partition_name=partition_name,
            partition_uuid=partition_uuid,
            engine=self._engine,
            partition_namespace=self._partition_namespace,
        )

    @override
    async def _unregister(self, partition_name: str) -> None:
        async with write_connection(self._engine) as connection:
            await _unregister_where(
                connection,
                self._partition_namespace,
                PartitionRow.partition_name == partition_name,
            )

    @override
    async def claim_tombstone(self) -> TombstoneClaim | None:
        # One committed write claims the oldest claimable tombstone and counts
        # the attempt; the round then runs with no transaction open. The lease
        # only spreads the work, since rounds are safe to repeat. The
        # tombstone of a claim never ended is claimable again once the claim's
        # lease, and then the backoff, have passed.
        oldest_due = (
            select(PurgeQueueRow.partition_uuid)
            .where(
                PurgeQueueRow.partition_namespace == self._partition_namespace,
                _has_elapsed(
                    self._engine.dialect.name,
                    self._tombstone_retention_seconds,
                    since=PurgeQueueRow.enqueued_at,
                ),
                or_(
                    PurgeQueueRow.claimable_at.is_(None),
                    PurgeQueueRow.claimable_at <= func.now(),
                ),
            )
            .order_by(PurgeQueueRow.enqueued_at)
            .limit(1)
            # SQLite drops this; there the claim's write serializes claims.
            .with_for_update(skip_locked=True)
            .scalar_subquery()
        )
        async with write_connection(self._engine) as connection:
            row = (
                await connection.execute(
                    update(PurgeQueueRow)
                    .where(PurgeQueueRow.partition_uuid == oldest_due)
                    .values(
                        attempts_without_progress=(
                            PurgeQueueRow.attempts_without_progress + 1
                        ),
                        claimable_at=_seconds_from_now(
                            self._engine.dialect.name,
                            self._purge_lease_seconds
                            + _purge_retry_backoff_seconds(
                                self._engine.dialect.name,
                                PurgeQueueRow.attempts_without_progress + 1,
                                base_seconds=self._base_purge_retry_backoff_seconds,
                                max_seconds=self._max_purge_retry_backoff_seconds,
                            ),
                        ),
                        claim_generation=PurgeQueueRow.claim_generation + 1,
                    )
                    .returning(
                        PurgeQueueRow.partition_uuid,
                        PurgeQueueRow.claim_generation,
                        PurgeQueueRow.attempts_without_progress,
                    )
                )
            ).one_or_none()
        if row is None:
            return None
        if row.attempts_without_progress > 1:
            logger.warning(
                "Purge round on partition %s is attempt %d",
                row.partition_uuid,
                row.attempts_without_progress,
            )
        return _SQLAlchemyTombstoneClaim(
            partition_uuid=row.partition_uuid,
            claim_generation=row.claim_generation,
            attempt=row.attempts_without_progress,
            engine=self._engine,
            base_purge_retry_backoff_seconds=self._base_purge_retry_backoff_seconds,
            max_purge_retry_backoff_seconds=self._max_purge_retry_backoff_seconds,
        )


@dataclass(frozen=True)
class _SQLAlchemyReservation(Reservation):
    """A reservation whose methods write the registry's tables directly."""

    engine: AsyncEngine = field(repr=False, compare=False)
    partition_namespace: str = field(repr=False)

    # Each operation acts only on the pending row under the reservation's UUID;
    # when it matches nothing, a read in its transaction tells a confirmed
    # partition from a deleted one.

    @override
    async def confirm(self) -> None:
        async with write_connection(self.engine) as connection:
            confirmed = await connection.execute(
                update(PartitionRow)
                .where(
                    PartitionRow.partition_namespace == self.partition_namespace,
                    PartitionRow.partition_name == self.partition_name,
                    PartitionRow.partition_uuid == self.partition_uuid,
                    PartitionRow.live.is_(False),
                )
                .values(live=True)
            )
            if confirmed.rowcount == 1:
                return
            live = await self._live(connection)
        self._raise_for_state(live)

    @override
    async def _cancel(self) -> None:
        async with write_connection(self.engine) as connection:
            if await _unregister_where(
                connection,
                self.partition_namespace,
                PartitionRow.partition_uuid == self.partition_uuid,
                PartitionRow.live.is_(False),
            ):
                return
            live = await self._live(connection)
        self._raise_for_state(live)

    async def _live(self, connection: AsyncConnection) -> bool | None:
        """Whether the reservation's partition is live, or None when no partition holds its UUID."""
        return (
            await connection.execute(
                select(PartitionRow.live).where(
                    PartitionRow.partition_namespace == self.partition_namespace,
                    PartitionRow.partition_name == self.partition_name,
                    PartitionRow.partition_uuid == self.partition_uuid,
                )
            )
        ).scalar_one_or_none()

    def _raise_for_state(self, live: bool | None) -> NoReturn:
        """Raise the error for a reservation whose partition is no longer pending."""
        if live is None:
            raise PartitionDeletedError(
                self.partition_namespace, self.partition_name, self.partition_uuid
            )
        raise PartitionAlreadyConfirmedError(
            self.partition_namespace, self.partition_name, self.partition_uuid
        )


@dataclass(frozen=True)
class _SQLAlchemyRegistration(Registration):
    """A registration whose method reads the registry's tables directly."""

    engine: AsyncEngine = field(repr=False, compare=False)
    partition_namespace: str = field(repr=False)

    @override
    async def require_current(self) -> None:
        async with self.engine.connect() as connection:
            current = (
                await connection.execute(
                    select(PartitionRow.partition_uuid).where(
                        PartitionRow.partition_namespace == self.partition_namespace,
                        PartitionRow.partition_name == self.partition_name,
                    )
                )
            ).scalar_one_or_none()
        if current != self.partition_uuid:
            raise PartitionDeletedError(
                self.partition_namespace, self.partition_name, self.partition_uuid
            )


@dataclass(frozen=True)
class _SQLAlchemyTombstoneClaim(TombstoneClaim):
    """A tombstone claim whose methods write the registry's tables directly."""

    claim_generation: int
    # The claim's attempt number since the last progress, from 1.
    attempt: int
    engine: AsyncEngine = field(repr=False, compare=False)
    base_purge_retry_backoff_seconds: int = field(repr=False)
    max_purge_retry_backoff_seconds: int = field(repr=False)

    # Each end matches the claim's generation, and advances it or deletes the
    # tombstone, so it takes effect once, and only under the latest claim.

    @override
    async def complete(self) -> None:
        async with write_connection(self.engine) as connection:
            ended = await connection.execute(
                delete(PurgeQueueRow).where(
                    PurgeQueueRow.partition_uuid == self.partition_uuid,
                    PurgeQueueRow.claim_generation == self.claim_generation,
                )
            )
        if ended.rowcount != 1:
            raise TombstoneClaimStaleError(self.partition_uuid)

    @override
    async def progress(self) -> None:
        async with write_connection(self.engine) as connection:
            ended = await connection.execute(
                update(PurgeQueueRow)
                .where(
                    PurgeQueueRow.partition_uuid == self.partition_uuid,
                    PurgeQueueRow.claim_generation == self.claim_generation,
                )
                .values(
                    attempts_without_progress=0,
                    claimable_at=None,
                    claim_generation=PurgeQueueRow.claim_generation + 1,
                )
            )
        if ended.rowcount != 1:
            raise TombstoneClaimStaleError(self.partition_uuid)

    @override
    async def fail(self, error: Exception) -> None:
        error.add_note(
            f"Purge round on partition {self.partition_uuid}, attempt {self.attempt}"
        )
        async with write_connection(self.engine) as connection:
            ended = await connection.execute(
                update(PurgeQueueRow)
                .where(
                    PurgeQueueRow.partition_uuid == self.partition_uuid,
                    PurgeQueueRow.claim_generation == self.claim_generation,
                )
                .values(
                    claimable_at=_seconds_from_now(
                        self.engine.dialect.name,
                        _purge_retry_backoff_seconds(
                            self.engine.dialect.name,
                            PurgeQueueRow.attempts_without_progress,
                            base_seconds=self.base_purge_retry_backoff_seconds,
                            max_seconds=self.max_purge_retry_backoff_seconds,
                        ),
                    ),
                    claim_generation=PurgeQueueRow.claim_generation + 1,
                )
            )
        if ended.rowcount != 1:
            raise TombstoneClaimStaleError(self.partition_uuid)


async def _unregister_where(
    connection: AsyncConnection,
    partition_namespace: str,
    *conditions: ColumnElement[bool],
) -> bool:
    """
    Delete the partition row in a namespace the conditions select, and queue its tombstone, in the connection's transaction; return whether a row was deleted.

    The partition is unreachable once the transaction commits. The DELETE runs
    first and takes the row's write lock, so racing deleters serialize on it
    and the loser finds no row.
    """
    row = (
        await connection.execute(
            delete(PartitionRow)
            .where(PartitionRow.partition_namespace == partition_namespace, *conditions)
            .returning(PartitionRow.partition_uuid)
        )
    ).one_or_none()
    if row is None:
        return False
    await connection.execute(
        insert(PurgeQueueRow).values(
            partition_uuid=row.partition_uuid,
            partition_namespace=partition_namespace,
            enqueued_at=func.now(),
        )
    )
    return True


def _has_elapsed(
    dialect_name: str,
    seconds: int | ColumnElement[int],
    *,
    since: InstrumentedAttribute[datetime] | InstrumentedAttribute[datetime | None],
) -> ColumnElement[bool]:
    """Whether `seconds` have passed since `since`, on the database clock."""
    if dialect_name == "sqlite":
        # datetime() emits CURRENT_TIMESTAMP's text form, so stored stamps and
        # the cutoff compare in time order.
        cutoff = func.datetime(
            "now",
            func.printf("-%d seconds", seconds),
            type_=DateTime(timezone=True),
        )
    else:
        cutoff = func.now() - literal(timedelta(seconds=1), Interval) * seconds
    return since <= cutoff


def _seconds_from_now(
    dialect_name: str, seconds: int | ColumnElement[int]
) -> ColumnElement[datetime]:
    """The time `seconds` from now, on the database clock."""
    if dialect_name == "sqlite":
        return func.datetime(
            "now",
            func.printf("+%d seconds", seconds),
            type_=DateTime(timezone=True),
        )
    return func.now() + literal(timedelta(seconds=1), Interval) * seconds


def _purge_retry_backoff_seconds(
    dialect_name: str,
    attempts: ColumnElement[int] | InstrumentedAttribute[int],
    *,
    base_seconds: int,
    max_seconds: int,
) -> ColumnElement[int]:
    """The purge retry backoff in seconds after `attempts` attempts, computed by the database: `base_seconds` doubled for each attempt after the first, at most `max_seconds`."""
    # 1 << (a - 1) is 2 ** (a - 1) on both dialects. The shift stops at the
    # first doubling that reaches `max_seconds`, so the backoff neither
    # overflows nor wraps however many attempts a tombstone has had.
    max_shift = 0
    while base_seconds << max_shift < max_seconds:
        max_shift += 1
    doublings = literal(1, Integer).op("<<")(
        case((attempts > max_shift, max_shift), else_=attempts - 1)
    )
    smaller = func.min if dialect_name == "sqlite" else func.least
    return smaller(base_seconds * doublings, max_seconds, type_=Integer)
