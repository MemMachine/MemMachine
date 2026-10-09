"""
A collection registry in a relational database, through SQLAlchemy.

A table of registered collections keyed by vector store, namespace, and name,
each with its incarnation and whether it is live (its storage prepared), and
a queue of deleted incarnations claimed in the order they come due. The
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
    and_,
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
from sqlalchemy.ext.asyncio import AsyncEngine
from sqlalchemy.orm import (
    DeclarativeBase,
    InstrumentedAttribute,
    MappedColumn,
    mapped_column,
)
from sqlalchemy.pool import StaticPool

from memmachine_server.common.utils import ensure_tz_aware
from memmachine_server.common.vector_store.data_types import (
    VectorStoreAttemptsExhaustedError,
    VectorStoreCollectionAlreadyExistsError,
    VectorStoreCollectionConfig,
    VectorStoreCollectionDeletedError,
    VectorStoreCollectionHandleStaleError,
    VectorStoreCollectionPendingError,
)
from memmachine_server.common.vector_store.utils import (
    _IDENTIFIER_MAX_BYTES,
)

from .collection_registry import (
    PurgeRound,
    Registration,
    Reservation,
    VectorStoreCollectionRegistry,
)

logger = logging.getLogger(__name__)

_MAX_MINT_ATTEMPTS = 10

_VECTOR_STORE_NAME_MAX_LENGTH = 255

# The first SQLite with RETURNING, which the registry uses.
_MIN_SQLITE_VERSION = (3, 35)

# Purge attempts without progress a tombstone gets before it is dead-lettered.
_MAX_PURGE_ATTEMPTS_WITHOUT_PROGRESS = 10

_JSON_AUTO = JSON().with_variant(JSONB, "postgresql")


class BaseCollectionRegistry(DeclarativeBase):
    """Base class for collection registry tables."""


class CollectionRow(BaseCollectionRegistry):
    """A registered collection of a vector store, pending or live."""

    __tablename__ = "collection_registry_ct"

    vector_store_name: MappedColumn[str] = mapped_column(
        String(_VECTOR_STORE_NAME_MAX_LENGTH), primary_key=True
    )
    namespace: MappedColumn[str] = mapped_column(
        String(_IDENTIFIER_MAX_BYTES), primary_key=True
    )
    name: MappedColumn[str] = mapped_column(
        String(_IDENTIFIER_MAX_BYTES), primary_key=True
    )
    incarnation: MappedColumn[UUID] = mapped_column(Uuid, nullable=False, unique=True)
    # The configuration the collection was created with.
    config: MappedColumn[dict[str, JsonValue]] = mapped_column(
        _JSON_AUTO, nullable=False
    )
    # Whether the collection's storage is prepared; it is pending until then.
    live: MappedColumn[bool] = mapped_column(Boolean, nullable=False)
    registered_at: MappedColumn[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )


class PurgeQueueRow(BaseCollectionRegistry):
    """A deleted collection's tombstone: its incarnation awaiting purge."""

    __tablename__ = "collection_registry_gc"

    incarnation: MappedColumn[UUID] = mapped_column(Uuid, primary_key=True)
    vector_store_name: MappedColumn[str] = mapped_column(
        String(_VECTOR_STORE_NAME_MAX_LENGTH), nullable=False
    )
    namespace: MappedColumn[str] = mapped_column(
        String(_IDENTIFIER_MAX_BYTES), nullable=False
    )
    # The collection's name, kept for inspection.
    name: MappedColumn[str] = mapped_column(
        String(_IDENTIFIER_MAX_BYTES), nullable=False
    )
    # With the namespace, the configuration locates the records in the store.
    config: MappedColumn[dict[str, JsonValue]] = mapped_column(
        _JSON_AUTO, nullable=False
    )
    enqueued_at: MappedColumn[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )
    # Purge rounds claimed since a round last found records and deleted them,
    # the open one included; a cancelled round's attempt is taken back.
    attempts_without_progress: MappedColumn[int] = mapped_column(
        Integer, nullable=False, default=0
    )
    # When a purge round last raised.
    last_failed_at: MappedColumn[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    # When the open claim was taken; None when no claim is open.
    claimed_at: MappedColumn[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    # Incremented by each claim; a round ends only its own claim.
    claim_generation: MappedColumn[int] = mapped_column(
        Integer, nullable=False, default=0
    )

    __table_args__ = (
        Index("collection_registry_gc__vs_ea", "vector_store_name", "enqueued_at"),
    )


class _RegistryInsertRejectedError(Exception):
    """A registry insert was rejected for a reason other than the name being taken."""


@dataclass(frozen=True)
class _TombstoneClaim:
    """A purge round's claim on a tombstone, and what the round needs to find the records."""

    incarnation: UUID
    namespace: str
    config: dict[str, JsonValue]
    claim_generation: int
    # The claim's attempt number since the last progress, from 1.
    attempt: int


class SQLAlchemyVectorStoreCollectionRegistryParams(BaseModel):
    """
    Parameters for SQLAlchemyVectorStoreCollectionRegistry.

    Attributes:
        engine (AsyncEngine):
            Async SQLAlchemy engine, on PostgreSQL or SQLite.
        vector_store_name (str):
            The name the registry's rows are kept under: registry objects with
            the same name on the same database are one registry.
        tombstone_retention_seconds (int):
            Seconds a deleted collection's records are kept before its purge
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
        ..., description="Async SQLAlchemy engine, on PostgreSQL or SQLite"
    )
    vector_store_name: str = Field(
        ...,
        description=(
            "The name the registry's rows are kept under: registry objects with "
            "the same name on the same database are one registry"
        ),
    )
    tombstone_retention_seconds: int = Field(
        ...,
        ge=0,
        description=(
            "Seconds a deleted collection's records are kept before its purge "
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
        engine_shares_one_connection = isinstance(engine.pool, StaticPool)
        if engine_shares_one_connection:
            raise ValueError(
                "Engine uses StaticPool, which shares one connection across "
                "sessions. Use a multi-connection pool instead."
            )
        database = engine.url.database
        if engine.dialect.name == "sqlite" and database in (None, "", ":memory:"):
            raise ValueError(
                "Engine uses in-memory SQLite, where each connection gets a "
                "separate database. Use a file path instead."
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


class SQLAlchemyVectorStoreCollectionRegistry(VectorStoreCollectionRegistry):
    """
    The registry of one vector store's collections.

    Registries of different vector stores share the tables, keyed by vector
    store name. A purge claim holds its tombstone for the purge lease, or
    until its round ends, and no transaction stays open during the round.
    After `_MAX_PURGE_ATTEMPTS_WITHOUT_PROGRESS` attempts without progress, a
    tombstone is dead-lettered: kept, its incarnation reserved, and skipped by
    claims; a last attempt that raises is reported in an error log. Setting its
    `attempts_without_progress` back to 0 returns it to the purge.
    """

    def __init__(self, params: SQLAlchemyVectorStoreCollectionRegistryParams) -> None:
        """Initialize with the provided parameters."""
        self._engine = params.engine
        self._is_sqlite = params.engine.dialect.name == "sqlite"
        self._vector_store_name = params.vector_store_name
        self._tombstone_retention_seconds = params.tombstone_retention_seconds
        self._purge_lease_seconds = params.purge_lease_seconds
        self._base_purge_retry_backoff_seconds = params.base_purge_retry_backoff_seconds
        self._max_purge_retry_backoff_seconds = params.max_purge_retry_backoff_seconds
        self._tombstone_claim_endings: set[asyncio.Task[None]] = set()

    @override
    async def startup(self) -> None:
        async with self._engine.begin() as connection:
            await connection.run_sync(BaseCollectionRegistry.metadata.create_all)

    @override
    async def reserve(
        self, namespace: str, name: str, config: VectorStoreCollectionConfig
    ) -> Reservation:
        # The primary key arbitrates the (namespace, name) across processes.
        # An insert rejected for another reason, such as a minted incarnation
        # that is registered or awaiting purge, is tried again with a fresh one.
        attempts = 0
        while True:
            incarnation = uuid4()
            try:
                await self._insert_pending_collection(
                    namespace, name, incarnation, config
                )
            except _RegistryInsertRejectedError as err:
                logger.warning(
                    "Reserving collection (%r, %r) under incarnation %s was "
                    "rejected: %s; minting another",
                    namespace,
                    name,
                    incarnation,
                    err,
                )
                attempts += 1
                if attempts >= _MAX_MINT_ATTEMPTS:
                    raise VectorStoreAttemptsExhaustedError(
                        f"Creating collection ({namespace!r}, {name!r}) made no "
                        f"progress after {_MAX_MINT_ATTEMPTS} attempts"
                    ) from err
                continue
            return _SQLAlchemyReservation(
                namespace=namespace,
                name=name,
                config=config,
                incarnation=incarnation,
                engine=self._engine,
                vector_store_name=self._vector_store_name,
            )

    async def _insert_pending_collection(
        self,
        namespace: str,
        name: str,
        incarnation: UUID,
        config: VectorStoreCollectionConfig,
    ) -> None:
        # The queue check runs after the insert, so a concurrent deletion
        # queuing a colliding incarnation, which the insert waited on, is
        # visible to it; the locking read sees the latest committed state.
        try:
            async with self._engine.begin() as connection:
                await connection.execute(
                    insert(CollectionRow).values(
                        vector_store_name=self._vector_store_name,
                        namespace=namespace,
                        name=name,
                        incarnation=incarnation,
                        config=config.model_dump(mode="json"),
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
                        select(CollectionRow.incarnation).where(
                            CollectionRow.vector_store_name == self._vector_store_name,
                            CollectionRow.namespace == namespace,
                            CollectionRow.name == name,
                        )
                    )
                ).scalar_one_or_none()
            if taken is not None:
                raise VectorStoreCollectionAlreadyExistsError(namespace, name) from err
            raise _RegistryInsertRejectedError(
                "the insert failed and no row exists under the name"
            ) from err

    @override
    async def resolve(self, namespace: str, name: str) -> Registration | None:
        async with self._engine.connect() as connection:
            row = (
                await connection.execute(
                    select(
                        CollectionRow.incarnation,
                        CollectionRow.config,
                        CollectionRow.live,
                        CollectionRow.registered_at,
                    ).where(
                        CollectionRow.vector_store_name == self._vector_store_name,
                        CollectionRow.namespace == namespace,
                        CollectionRow.name == name,
                    )
                )
            ).one_or_none()
        if row is None:
            return None
        config = VectorStoreCollectionConfig.model_validate(row.config)
        if not row.live:
            raise VectorStoreCollectionPendingError(
                namespace, name, ensure_tz_aware(row.registered_at), config
            )
        return _SQLAlchemyRegistration(
            namespace=namespace,
            name=name,
            config=config,
            incarnation=row.incarnation,
            engine=self._engine,
            vector_store_name=self._vector_store_name,
        )

    @override
    async def unregister(self, namespace: str, name: str) -> None:
        await _unregister_where(
            self._engine,
            self._vector_store_name,
            CollectionRow.namespace == namespace,
            CollectionRow.name == name,
        )

    @override
    async def run_purge_round(self, purge_round: PurgeRound) -> bool:
        claim = await self._claim_oldest_due_tombstone()
        if claim is None:
            return False
        if claim.attempt > 1:
            logger.warning(
                "Purge round on incarnation %s is attempt %d of %d",
                claim.incarnation,
                claim.attempt,
                _MAX_PURGE_ATTEMPTS_WITHOUT_PROGRESS,
            )
        try:
            any_records_found = await purge_round(
                claim.namespace,
                VectorStoreCollectionConfig.model_validate(claim.config),
                claim.incarnation,
            )
        except Exception as error:
            error.add_note(
                f"Purge round on incarnation {claim.incarnation}, attempt "
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
        """Claim the oldest due tombstone in one committed write, counting the attempt; None when none is due.

        The round then runs with no transaction open. The lease only spreads
        the work, since rounds are safe to repeat. A claim whose round never
        ended is taken again once its lease, and then the backoff, have passed.
        """
        backoff_seconds = self._purge_retry_backoff_seconds()
        oldest_due = (
            select(PurgeQueueRow.incarnation)
            .where(
                PurgeQueueRow.vector_store_name == self._vector_store_name,
                self._has_elapsed(
                    self._tombstone_retention_seconds, since=PurgeQueueRow.enqueued_at
                ),
                PurgeQueueRow.attempts_without_progress
                < _MAX_PURGE_ATTEMPTS_WITHOUT_PROGRESS,
                or_(
                    # No attempt since the last progress.
                    PurgeQueueRow.attempts_without_progress == 0,
                    # No claim is open: the backoff runs from the last raise.
                    and_(
                        PurgeQueueRow.claimed_at.is_(None),
                        or_(
                            PurgeQueueRow.last_failed_at.is_(None),
                            self._has_elapsed(
                                backoff_seconds, since=PurgeQueueRow.last_failed_at
                            ),
                        ),
                    ),
                    # The open claim's round never ended: the backoff runs
                    # from when its lease passed.
                    and_(
                        PurgeQueueRow.claimed_at.is_not(None),
                        self._has_elapsed(
                            backoff_seconds + self._purge_lease_seconds,
                            since=PurgeQueueRow.claimed_at,
                        ),
                    ),
                ),
            )
            .order_by(PurgeQueueRow.enqueued_at)
            .limit(1)
            # SQLite drops this; there the claim's write serializes claims.
            .with_for_update(skip_locked=True)
            .scalar_subquery()
        )
        async with self._engine.begin() as connection:
            row = (
                await connection.execute(
                    update(PurgeQueueRow)
                    .where(PurgeQueueRow.incarnation == oldest_due)
                    .values(
                        attempts_without_progress=(
                            PurgeQueueRow.attempts_without_progress + 1
                        ),
                        claimed_at=func.now(),
                        claim_generation=PurgeQueueRow.claim_generation + 1,
                    )
                    .returning(
                        PurgeQueueRow.incarnation,
                        PurgeQueueRow.namespace,
                        PurgeQueueRow.config,
                        PurgeQueueRow.claim_generation,
                        PurgeQueueRow.attempts_without_progress,
                    )
                )
            ).one_or_none()
        if row is None:
            return None
        return _TombstoneClaim(
            incarnation=row.incarnation,
            namespace=row.namespace,
            config=row.config,
            claim_generation=row.claim_generation,
            attempt=row.attempts_without_progress,
        )

    def _purge_retry_backoff_seconds(self) -> ColumnElement[int]:
        """Each tombstone's purge retry backoff in seconds, computed by the database."""
        # 1 << (a - 1) is 2 ** (a - 1) on both dialects. a is clamped at 0:
        # the claim evaluates this for tombstones with no attempts too, and
        # PostgreSQL leaves a negative shift undefined.
        attempts = PurgeQueueRow.attempts_without_progress
        doublings = literal(1, Integer).op("<<")(
            case((attempts > 0, attempts - 1), else_=0)
        )
        smaller = func.min if self._is_sqlite else func.least
        return smaller(
            self._base_purge_retry_backoff_seconds * doublings,
            self._max_purge_retry_backoff_seconds,
            type_=Integer,
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

    async def _end_tombstone_claim_after_failure(
        self, claim: _TombstoneClaim, error: Exception
    ) -> None:
        """End a raised purge round's tombstone claim and date its failure, if the claim is the latest.

        A raise on the last attempt is reported: claims skip the tombstone from
        then on.
        """
        async with self._engine.begin() as connection:
            ended = await connection.execute(
                update(PurgeQueueRow)
                .where(
                    PurgeQueueRow.incarnation == claim.incarnation,
                    PurgeQueueRow.claim_generation == claim.claim_generation,
                )
                .values(claimed_at=None, last_failed_at=func.now())
            )
        if ended.rowcount != 1:
            self._report_purge_round_outlasting_its_claim(claim)
            return
        # Claims take only tombstones under the bound.
        if claim.attempt >= _MAX_PURGE_ATTEMPTS_WITHOUT_PROGRESS:
            logger.error(
                "Purge of incarnation %s is dead-lettered after %d attempts "
                "without progress: its records stay and it is no longer claimed. "
                "Last error: %r. Set its attempts_without_progress to 0 in %s to "
                "retry it.",
                claim.incarnation,
                claim.attempt,
                error,
                PurgeQueueRow.__tablename__,
            )

    async def _end_tombstone_claim_after_cancellation(
        self, claim: _TombstoneClaim
    ) -> None:
        """End a cancelled purge round's tombstone claim without counting the round.

        Shielded, so a second cancellation cannot cut it short. The task logs
        its own failure; the claim then ends with its lease.
        """
        ending = asyncio.create_task(self._end_tombstone_claim(claim))
        self._tombstone_claim_endings.add(ending)

        def finish(task: asyncio.Task[None]) -> None:
            self._tombstone_claim_endings.discard(task)
            if not task.cancelled() and task.exception() is not None:
                logger.error(
                    "Could not end the claim of a cancelled purge round on "
                    "incarnation %s; it ends with its lease",
                    claim.incarnation,
                    exc_info=task.exception(),
                )

        ending.add_done_callback(finish)
        with contextlib.suppress(Exception):
            await asyncio.shield(ending)

    async def _end_tombstone_claim(self, claim: _TombstoneClaim) -> None:
        """End a tombstone claim and take back its attempt, if the claim is the latest."""
        async with self._engine.begin() as connection:
            await connection.execute(
                update(PurgeQueueRow)
                .where(
                    PurgeQueueRow.incarnation == claim.incarnation,
                    PurgeQueueRow.claim_generation == claim.claim_generation,
                )
                .values(
                    claimed_at=None,
                    attempts_without_progress=(
                        PurgeQueueRow.attempts_without_progress - 1
                    ),
                )
            )

    async def _record_purge_round(
        self, claim: _TombstoneClaim, any_records_found: bool
    ) -> None:
        """Record a purge round's outcome on its tombstone.

        Finding nothing removes the tombstone under any claim, since an
        incarnation found empty after the retention needs no more rounds.
        Finding records ends the claim and resets its attempts without
        progress, if the claim is the latest.
        """
        if not any_records_found:
            async with self._engine.begin() as connection:
                await connection.execute(
                    delete(PurgeQueueRow).where(
                        PurgeQueueRow.incarnation == claim.incarnation
                    )
                )
            return
        async with self._engine.begin() as connection:
            ended = await connection.execute(
                update(PurgeQueueRow)
                .where(
                    PurgeQueueRow.incarnation == claim.incarnation,
                    PurgeQueueRow.claim_generation == claim.claim_generation,
                )
                .values(attempts_without_progress=0, claimed_at=None)
            )
        if ended.rowcount != 1:
            self._report_purge_round_outlasting_its_claim(claim)

    def _report_purge_round_outlasting_its_claim(self, claim: _TombstoneClaim) -> None:
        """Report a purge round that ended after its tombstone was claimed again or removed."""
        logger.warning(
            "Purge round on incarnation %s outlasted its claim's %d s lease: the "
            "tombstone was claimed again or removed while it ran",
            claim.incarnation,
            self._purge_lease_seconds,
        )


@dataclass(frozen=True)
class _SQLAlchemyReservation(Reservation):
    """A reservation whose methods write the registry's tables directly."""

    engine: AsyncEngine = field(repr=False, compare=False)
    vector_store_name: str = field(repr=False)

    @override
    async def confirm(self) -> Registration:
        # Conditional on the incarnation and on the row being pending, so a
        # creation marks live only the collection it reserved.
        async with self.engine.begin() as connection:
            result = await connection.execute(
                update(CollectionRow)
                .where(
                    CollectionRow.vector_store_name == self.vector_store_name,
                    CollectionRow.namespace == self.namespace,
                    CollectionRow.name == self.name,
                    CollectionRow.incarnation == self.incarnation,
                    CollectionRow.live.is_(False),
                )
                .values(live=True)
            )
        if result.rowcount != 1:
            raise VectorStoreCollectionDeletedError(self.namespace, self.name)
        return _SQLAlchemyRegistration(
            namespace=self.namespace,
            name=self.name,
            config=self.config,
            incarnation=self.incarnation,
            engine=self.engine,
            vector_store_name=self.vector_store_name,
        )

    @override
    async def cancel(self) -> None:
        # Conditional on the row being pending, as confirm is, so a cancel
        # after a confirmation that committed leaves the live collection.
        await _unregister_where(
            self.engine,
            self.vector_store_name,
            CollectionRow.incarnation == self.incarnation,
            CollectionRow.live.is_(False),
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
                    select(CollectionRow.incarnation).where(
                        CollectionRow.vector_store_name == self.vector_store_name,
                        CollectionRow.namespace == self.namespace,
                        CollectionRow.name == self.name,
                    )
                )
            ).scalar_one_or_none()
        if current != self.incarnation:
            raise VectorStoreCollectionHandleStaleError(self.namespace, self.name)


async def _unregister_where(
    engine: AsyncEngine, vector_store_name: str, *conditions: ColumnElement[bool]
) -> None:
    """Delete the collection row of a vector store the conditions select, and queue its tombstone.

    One transaction, so the collection is unreachable once it commits. The
    DELETE goes first and takes the row's write lock, so racing deleters
    serialize on it and the loser finds no row and returns.
    """
    async with engine.begin() as connection:
        row = (
            await connection.execute(
                delete(CollectionRow)
                .where(
                    CollectionRow.vector_store_name == vector_store_name, *conditions
                )
                .returning(
                    CollectionRow.incarnation,
                    CollectionRow.namespace,
                    CollectionRow.name,
                    CollectionRow.config,
                )
            )
        ).one_or_none()
        if row is None:
            return
        await connection.execute(
            insert(PurgeQueueRow).values(
                incarnation=row.incarnation,
                vector_store_name=vector_store_name,
                namespace=row.namespace,
                name=row.name,
                config=row.config,
                enqueued_at=func.now(),
            )
        )
