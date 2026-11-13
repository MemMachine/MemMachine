"""
A partition registry in a relational database, through SQLAlchemy.

A table of registered partitions keyed by namespace and partition name,
each with its UUID and whether it is live (its storage created), and
a queue of deleted partition UUIDs claimed in the order they come due. The
primary key arbitrates registration across processes, a conditional update
confirms a reservation, unregistration is one transaction, and a purge
claim is a lease on a tombstone that one short write takes and another ends.
"""

import asyncio
import contextlib
import logging
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import override
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
    delete,
    func,
    insert,
    literal,
    or_,
    select,
    update,
)
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncEngine
from sqlalchemy.orm import (
    DeclarativeBase,
    InstrumentedAttribute,
    MappedColumn,
    mapped_column,
)

from memmachine_server.common.sql import validate_engine_connections, write_connection
from memmachine_server.common.utils import ensure_tz_aware

from .data_types import (
    _PARTITION_NAME_MAX_BYTES,
    _PARTITION_NAMESPACE_MAX_BYTES,
    PartitionAlreadyExistsError,
    PartitionCreateAttemptsExhaustedError,
    PartitionDeletedError,
    PartitionPendingError,
    validate_partition_namespace,
)
from .partition_registry import (
    PartitionRegistry,
    PurgeRound,
    Registration,
    Reservation,
)
from .partitioned_store import PartitionedStore

logger = logging.getLogger(__name__)

_MAX_MINT_ATTEMPTS = 10

# The first SQLite with RETURNING, which the registry uses.
_MIN_SQLITE_VERSION = (3, 35)

# Purge attempts without progress a tombstone gets before it is dead-lettered.
_MAX_PURGE_ATTEMPTS_WITHOUT_PROGRESS = 10


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
    # Whether the partition's storage is created; it is pending until then.
    live: MappedColumn[bool] = mapped_column(Boolean, nullable=False)
    registered_at: MappedColumn[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )


class PurgeQueueRow(BasePartitionRegistry):
    """A deleted partition's tombstone: its partition UUID awaiting purge.

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
    # Purge rounds claimed since a round last found records and deleted them,
    # the open one included; a cancelled round's attempt is taken back.
    attempts_without_progress: MappedColumn[int] = mapped_column(
        Integer, nullable=False, default=0
    )
    # When a claim may next take the tombstone: once the open claim's lease
    # and the backoff pass, or the backoff after a raise; None until a claim
    # sets it, and again after a round that found records or was cancelled.
    claimable_at: MappedColumn[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    # Incremented by each claim; a round ends only its own claim.
    claim_generation: MappedColumn[int] = mapped_column(
        Integer, nullable=False, default=0
    )

    __table_args__ = (
        Index("partition_registry_gc__ns_ea", "partition_namespace", "enqueued_at"),
    )


class _RegistryInsertRejectedError(Exception):
    """A registry insert was rejected for a reason other than the name being taken."""


@dataclass(frozen=True)
class _TombstoneClaim:
    """A purge round's claim on a tombstone."""

    partition_uuid: UUID
    claim_generation: int
    # The claim's attempt number since the last progress, from 1.
    attempt: int


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
            Seconds a deleted partition's records are kept before its purge
            starts, on the database clock. It must exceed, by orders of
            magnitude, the longest a write to the backend can be in flight and
            the delay before the store's reads reflect a write.
        purge_lease_seconds (int):
            Seconds a purge round's claim holds its tombstone, on the database
            clock; it should exceed the longest round (default: 300).
        base_purge_retry_backoff_seconds (int):
            Seconds before a tombstone is claimed again after a failed purge
            round, doubled for each further failed round in a row (default: 30).
        max_purge_retry_backoff_seconds (int):
            The longest the purge retry backoff grows to (default: 3600).
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
            "Seconds a deleted partition's records are kept before its purge "
            "starts, on the database clock. It must exceed, by orders of "
            "magnitude, the longest a write to the backend can be in flight and "
            "the delay before the store's reads reflect a write"
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
            "Seconds before a tombstone is claimed again after a failed purge "
            "round, doubled for each further failed round in a row"
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
        # The registry's arbitration rests on transactions of their own.
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


class SQLAlchemyPartitionRegistry[PartitionT](PartitionRegistry[PartitionT]):
    """
    The registry of one partition namespace, over the store that holds its partitions.

    Registries of different namespaces share the tables, keyed by namespace.
    A purge claim holds its tombstone until its round ends, or, for a round
    that never ends, until the purge lease and the backoff pass, and no
    transaction stays open during the round.
    After `_MAX_PURGE_ATTEMPTS_WITHOUT_PROGRESS` attempts without progress, a
    tombstone is dead-lettered: kept, its UUID reserved, and skipped by
    claims; a last attempt that raises is reported in an error log. Setting its
    `attempts_without_progress` back to 0 and its `claimable_at` to NULL
    returns it to the purge.
    """

    def __init__(
        self,
        params: SQLAlchemyPartitionRegistryParams,
        *,
        partitioned_store: PartitionedStore[PartitionT],
    ) -> None:
        """Initialize with the provided parameters and the store that holds the namespace's partitions."""
        super().__init__(partitioned_store)
        self._engine = params.engine
        self._is_sqlite = params.engine.dialect.name == "sqlite"
        self._partition_namespace = params.partition_namespace
        self._tombstone_retention_seconds = params.tombstone_retention_seconds
        self._purge_lease_seconds = params.purge_lease_seconds
        self._base_purge_retry_backoff_seconds = params.base_purge_retry_backoff_seconds
        self._max_purge_retry_backoff_seconds = params.max_purge_retry_backoff_seconds
        self._tombstone_claim_endings: set[asyncio.Task[None]] = set()

    @property
    @override
    def partition_namespace(self) -> str:
        return self._partition_namespace

    @override
    async def startup(self) -> None:
        async with self._engine.begin() as connection:
            await connection.run_sync(BasePartitionRegistry.metadata.create_all)

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
                logger.warning(
                    "Reserving partition %r under UUID %s was rejected: "
                    "%s; minting another",
                    partition_name,
                    partition_uuid,
                    err,
                )
                attempts += 1
                if attempts >= _MAX_MINT_ATTEMPTS:
                    raise PartitionCreateAttemptsExhaustedError(
                        f"Creating partition {partition_name!r} made no progress "
                        f"after {_MAX_MINT_ATTEMPTS} attempts"
                    ) from err
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
        # The queue check runs after the insert, so a concurrent deletion
        # queuing a colliding partition UUID, which the insert waited on, is
        # visible to it; the locking read sees the latest committed state.
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
    def _registration(
        self, partition_name: str, partition_uuid: UUID, partition: PartitionT
    ) -> Registration[PartitionT]:
        return _SQLAlchemyRegistration(
            partition_name=partition_name,
            partition_uuid=partition_uuid,
            partition=partition,
            engine=self._engine,
            partition_namespace=self._partition_namespace,
        )

    @override
    async def _unregister(self, partition_name: str) -> None:
        await _unregister_where(
            self._engine,
            self._partition_namespace,
            PartitionRow.partition_name == partition_name,
        )

    @override
    async def _run_purge_round(self, purge_round: PurgeRound) -> bool:
        claim = await self._claim_oldest_due_tombstone()
        if claim is None:
            return False
        if claim.attempt > 1:
            logger.warning(
                "Purge round on partition %s is attempt %d of %d",
                claim.partition_uuid,
                claim.attempt,
                _MAX_PURGE_ATTEMPTS_WITHOUT_PROGRESS,
            )
        try:
            any_records_found = await purge_round(claim.partition_uuid)
        except Exception as error:
            error.add_note(
                f"Purge round on partition {claim.partition_uuid}, attempt "
                f"{claim.attempt} of {_MAX_PURGE_ATTEMPTS_WITHOUT_PROGRESS}"
            )
            await self._end_tombstone_claim_after_failure(claim, error)
            raise
        except BaseException:
            await self._end_tombstone_claim_after_cancellation(claim)
            raise
        await self._record_purge_round(claim, any_records_found)
        return True

    async def _claim_oldest_due_tombstone(self) -> _TombstoneClaim | None:
        """Claim the oldest due tombstone in one committed write, counting the attempt; None when no due tombstone is claimable.

        The round then runs with no transaction open. The lease only spreads
        the work, since rounds are safe to repeat. A claim whose round never
        ended is taken again once its lease, and then the backoff, have passed.
        """
        oldest_due = (
            select(PurgeQueueRow.partition_uuid)
            .where(
                PurgeQueueRow.partition_namespace == self._partition_namespace,
                self._has_elapsed(
                    self._tombstone_retention_seconds, since=PurgeQueueRow.enqueued_at
                ),
                PurgeQueueRow.attempts_without_progress
                < _MAX_PURGE_ATTEMPTS_WITHOUT_PROGRESS,
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
                        claimable_at=self._seconds_from_now(
                            self._purge_lease_seconds
                            + self._purge_retry_backoff_seconds(
                                PurgeQueueRow.attempts_without_progress + 1
                            )
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
        return _TombstoneClaim(
            partition_uuid=row.partition_uuid,
            claim_generation=row.claim_generation,
            attempt=row.attempts_without_progress,
        )

    async def _end_tombstone_claim_after_failure(
        self, claim: _TombstoneClaim, error: Exception
    ) -> None:
        """End a raised purge round's tombstone claim, making the tombstone claimable once the backoff passes, if the claim is the latest.

        A raise on the last attempt is reported: claims skip the tombstone from
        then on.
        """
        async with write_connection(self._engine) as connection:
            ended = await connection.execute(
                update(PurgeQueueRow)
                .where(
                    PurgeQueueRow.partition_uuid == claim.partition_uuid,
                    PurgeQueueRow.claim_generation == claim.claim_generation,
                )
                .values(
                    claimable_at=self._seconds_from_now(
                        self._purge_retry_backoff_seconds(
                            PurgeQueueRow.attempts_without_progress
                        )
                    )
                )
            )
        if ended.rowcount != 1:
            self._report_purge_round_outlasting_its_claim(claim)
            return
        # Claims take only tombstones under the bound.
        if claim.attempt >= _MAX_PURGE_ATTEMPTS_WITHOUT_PROGRESS:
            logger.error(
                "Purge of partition %s is dead-lettered after %d attempts "
                "without progress: its records stay and it is no longer claimed. "
                "Last error: %r. Set its attempts_without_progress to 0 and its "
                "claimable_at to NULL in %s to retry it.",
                claim.partition_uuid,
                claim.attempt,
                error,
                PurgeQueueRow.__tablename__,
            )

    async def _end_tombstone_claim_after_cancellation(
        self, claim: _TombstoneClaim
    ) -> None:
        """End a cancelled purge round's tombstone claim without counting the round.

        Shielded, so a second cancellation cannot cut it short. The task logs
        its own failure; the tombstone is then claimable once the claim's
        lease and the backoff pass.
        """
        ending = asyncio.create_task(self._end_tombstone_claim(claim))
        self._tombstone_claim_endings.add(ending)

        def finish(task: asyncio.Task[None]) -> None:
            self._tombstone_claim_endings.discard(task)
            if not task.cancelled() and task.exception() is not None:
                logger.error(
                    "Could not end the claim of a cancelled purge round on "
                    "partition %s; its tombstone is claimable once the lease "
                    "and the backoff pass",
                    claim.partition_uuid,
                    exc_info=task.exception(),
                )

        ending.add_done_callback(finish)
        with contextlib.suppress(Exception):
            await asyncio.shield(ending)

    async def _end_tombstone_claim(self, claim: _TombstoneClaim) -> None:
        """End a tombstone claim and take back its attempt, if the claim is the latest."""
        async with write_connection(self._engine) as connection:
            await connection.execute(
                update(PurgeQueueRow)
                .where(
                    PurgeQueueRow.partition_uuid == claim.partition_uuid,
                    PurgeQueueRow.claim_generation == claim.claim_generation,
                )
                .values(
                    claimable_at=None,
                    attempts_without_progress=(
                        PurgeQueueRow.attempts_without_progress - 1
                    ),
                )
            )

    async def _record_purge_round(
        self, claim: _TombstoneClaim, any_records_found: bool
    ) -> None:
        """Record a purge round's outcome on its tombstone.

        Finding nothing removes the tombstone under any claim, since a
        partition found empty after the retention needs no more rounds.
        Finding records ends the claim and resets its attempts without
        progress, if the claim is the latest.
        """
        if not any_records_found:
            async with write_connection(self._engine) as connection:
                await connection.execute(
                    delete(PurgeQueueRow).where(
                        PurgeQueueRow.partition_uuid == claim.partition_uuid
                    )
                )
            return
        async with write_connection(self._engine) as connection:
            ended = await connection.execute(
                update(PurgeQueueRow)
                .where(
                    PurgeQueueRow.partition_uuid == claim.partition_uuid,
                    PurgeQueueRow.claim_generation == claim.claim_generation,
                )
                .values(attempts_without_progress=0, claimable_at=None)
            )
        if ended.rowcount != 1:
            self._report_purge_round_outlasting_its_claim(claim)

    def _report_purge_round_outlasting_its_claim(self, claim: _TombstoneClaim) -> None:
        """Report a purge round that ended after its tombstone was claimed again or removed."""
        logger.warning(
            "Purge round on partition %s outlasted its claim's %d s lease: the "
            "tombstone was claimed again or removed while it ran",
            claim.partition_uuid,
            self._purge_lease_seconds,
        )

    def _has_elapsed(
        self,
        seconds: int | ColumnElement[int],
        *,
        since: InstrumentedAttribute[datetime] | InstrumentedAttribute[datetime | None],
    ) -> ColumnElement[bool]:
        """Whether `seconds` have passed since `since`, on the database clock."""
        if self._is_sqlite:
            # datetime() emits CURRENT_TIMESTAMP's text form, so stored stamps
            # compare with it in time order.
            cutoff = func.datetime(
                "now",
                func.printf("-%d seconds", seconds),
                type_=DateTime(timezone=True),
            )
        else:
            cutoff = func.now() - literal(timedelta(seconds=1), Interval) * seconds
        return since <= cutoff

    def _seconds_from_now(
        self, seconds: int | ColumnElement[int]
    ) -> ColumnElement[datetime]:
        """The time `seconds` from now, on the database clock."""
        if self._is_sqlite:
            return func.datetime(
                "now",
                func.printf("+%d seconds", seconds),
                type_=DateTime(timezone=True),
            )
        return func.now() + literal(timedelta(seconds=1), Interval) * seconds

    def _purge_retry_backoff_seconds(
        self, attempts: ColumnElement[int] | InstrumentedAttribute[int]
    ) -> ColumnElement[int]:
        """The purge retry backoff in seconds after `attempts` attempts, computed by the database."""
        # 1 << (a - 1) is 2 ** (a - 1) on both dialects.
        doublings = literal(1, Integer).op("<<")(attempts - 1)
        smaller = func.min if self._is_sqlite else func.least
        return smaller(
            self._base_purge_retry_backoff_seconds * doublings,
            self._max_purge_retry_backoff_seconds,
            type_=Integer,
        )


@dataclass(frozen=True)
class _SQLAlchemyReservation(Reservation):
    """A reservation whose methods write the registry's tables directly."""

    engine: AsyncEngine = field(repr=False, compare=False)
    partition_namespace: str = field(repr=False)

    @override
    async def confirm(self) -> None:
        # Conditional on the UUID and on the row being pending, so a
        # creation marks live only the partition it reserved.
        async with write_connection(self.engine) as connection:
            result = await connection.execute(
                update(PartitionRow)
                .where(
                    PartitionRow.partition_namespace == self.partition_namespace,
                    PartitionRow.partition_name == self.partition_name,
                    PartitionRow.partition_uuid == self.partition_uuid,
                    PartitionRow.live.is_(False),
                )
                .values(live=True)
            )
        if result.rowcount != 1:
            raise PartitionDeletedError(
                self.partition_namespace, self.partition_name, self.partition_uuid
            )

    @override
    async def cancel(self) -> None:
        # Conditional on the row being pending, as confirm is, so a cancel
        # after a confirmation that committed leaves the live partition.
        await _unregister_where(
            self.engine,
            self.partition_namespace,
            PartitionRow.partition_uuid == self.partition_uuid,
            PartitionRow.live.is_(False),
        )


@dataclass(frozen=True)
class _SQLAlchemyRegistration[PartitionT](Registration[PartitionT]):
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


async def _unregister_where(
    engine: AsyncEngine, partition_namespace: str, *conditions: ColumnElement[bool]
) -> None:
    """Delete the partition row in a namespace the conditions select, and queue its tombstone.

    One transaction, so the partition is unreachable once it commits. The
    DELETE runs first and takes the row's write lock, so racing deleters
    serialize on it and the loser finds no row and returns.
    """
    async with write_connection(engine) as connection:
        row = (
            await connection.execute(
                delete(PartitionRow)
                .where(
                    PartitionRow.partition_namespace == partition_namespace, *conditions
                )
                .returning(PartitionRow.partition_uuid)
            )
        ).one_or_none()
        if row is None:
            return
        await connection.execute(
            insert(PurgeQueueRow).values(
                partition_uuid=row.partition_uuid,
                partition_namespace=partition_namespace,
                enqueued_at=func.now(),
            )
        )
