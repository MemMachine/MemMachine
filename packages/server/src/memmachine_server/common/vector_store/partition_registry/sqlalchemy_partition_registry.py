"""
A partition registry in a relational database, through SQLAlchemy.

Qdrant and Milvus have no transactions, unique constraints or conditional
writes, so a catalog kept inside them cannot arbitrate two processes
creating, deleting or reclaiming the same partition. This registry lives
in the deployment's relational database instead: one table pair per
vector store, named by the store's name, with a row per live partition
keyed by its key, whose incarnation is the value every point of that life
carries, and a purge queue of dead incarnations claimed in the order they
come due. Registration is an insert the primary key arbitrates,
unregistration is one transaction, and a purge claim is a row lock the
database hands to one purger at a time.
"""

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import timedelta
from typing import override
from uuid import UUID, uuid4

from sqlalchemy import (
    JSON,
    Column,
    ColumnElement,
    DateTime,
    Index,
    Interval,
    MetaData,
    String,
    Table,
    Uuid,
    bindparam,
    delete,
    func,
    insert,
    select,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncEngine

from memmachine_server.common.vector_store.data_types import (
    PartitionSchema,
    VectorStoreAttemptsExhaustedError,
    VectorStorePartitionAlreadyExistsError,
)
from memmachine_server.common.vector_store.utils import (
    _IDENTIFIER_MAX_BYTES,
    validate_identifier,
)

from .partition_registry import (
    PurgeClaim,
    RegisteredPartition,
    VectorStorePartitionRegistry,
)

logger = logging.getLogger(__name__)

_MAX_MINT_ATTEMPTS = 10

_JSON_AUTO = JSON().with_variant(JSONB, "postgresql")


class _RegistryInsertRejectedError(Exception):
    """A registry insert was rejected; retry with a fresh incarnation."""


class SQLAlchemyVectorStorePartitionRegistry(VectorStorePartitionRegistry):
    """The registry of one vector store's partitions, in its own table pair.

    `vector_store_name` names the vector store whose partitions the
    registry holds: its tables are `partition_registry_{vector_store_name}_pt`
    and `..._gc`, so the registries of different vector stores share a
    database without sharing a row, and registry objects under one name are
    one registry. It must match `[a-z0-9_]+` and be at most 32 bytes.
    `tombstone_retention` is how long a dead incarnation's points are kept
    before its purge starts; it must exceed, by orders of magnitude, the
    longest a write to the backend can be in flight. The retention is
    measured on the database clock.
    """

    def __init__(
        self,
        *,
        engine: AsyncEngine,
        vector_store_name: str,
        tombstone_retention: timedelta,
    ) -> None:
        """Bind to the tables of the registry of the vector store `vector_store_name`."""
        if not validate_identifier(vector_store_name):
            raise ValueError(
                f"Vector store name {vector_store_name!r} must match [a-z0-9_]+ "
                "and be at most 32 bytes"
            )
        if engine.dialect.name not in ("postgresql", "sqlite"):
            raise ValueError(
                f"Engine uses the {engine.dialect.name} dialect, which the "
                "registry does not support. Use PostgreSQL or SQLite."
            )
        self._engine = engine
        self._vector_store_name = vector_store_name
        self._is_sqlite = engine.dialect.name == "sqlite"
        self._tombstone_retention = tombstone_retention
        table_prefix = f"partition_registry_{vector_store_name}"
        metadata = MetaData()
        self._partitions = Table(
            f"{table_prefix}_pt",
            metadata,
            Column("partition_key", String(_IDENTIFIER_MAX_BYTES), primary_key=True),
            Column("incarnation", Uuid, nullable=False, unique=True),
            # The dimensions, metric and declared schema the partition was
            # created under, so a store built with others fails loudly
            # instead of filtering on indexes that are not there.
            Column("schema", _JSON_AUTO, nullable=False),
        )
        self._purge_queue = Table(
            f"{table_prefix}_gc",
            metadata,
            Column("incarnation", Uuid, primary_key=True),
            Column("partition_key", String(_IDENTIFIER_MAX_BYTES), nullable=False),
            Column("enqueued_at", DateTime(timezone=True), nullable=False),
        )
        Index(f"{table_prefix}_gc__ea", self._purge_queue.c.enqueued_at)
        self._metadata = metadata

    @override
    async def provision(self) -> None:
        async with self._engine.begin() as connection:
            await connection.run_sync(self._metadata.create_all)

    @override
    async def register(self, partition_key: str, schema: PartitionSchema) -> UUID:
        # The primary key arbitrates the partition key across processes;
        # the incarnation's unique constraint and the in-transaction queue
        # check reject an incarnation that is live or still awaiting purge.
        attempts = 0
        while True:
            incarnation = uuid4()
            try:
                await self._insert(partition_key, incarnation, schema)
            except _RegistryInsertRejectedError as err:
                attempts += 1
                if attempts >= _MAX_MINT_ATTEMPTS:
                    raise VectorStoreAttemptsExhaustedError(
                        f"Creating partition {partition_key!r} made no progress "
                        f"after {_MAX_MINT_ATTEMPTS} attempts"
                    ) from err
                continue
            return incarnation

    async def _insert(
        self, partition_key: str, incarnation: UUID, schema: PartitionSchema
    ) -> None:
        # The queue check runs after the insert so that a concurrent
        # deletion moving a colliding row to the queue, which our insert
        # waited on, is already visible; after the check, no new queue
        # entry for this incarnation can appear before we commit, because
        # the only registry row carrying it is ours, uncommitted. The
        # locking read sees latest-committed state even on dialects whose
        # plain reads serve transaction-start snapshots.
        try:
            async with self._engine.begin() as connection:
                await connection.execute(
                    insert(self._partitions).values(
                        partition_key=partition_key,
                        incarnation=incarnation,
                        schema=schema.model_dump(mode="json"),
                    )
                )
                queued = (
                    await connection.execute(
                        select(self._purge_queue.c.incarnation)
                        .where(self._purge_queue.c.incarnation == incarnation)
                        .with_for_update(read=True)
                    )
                ).scalar_one_or_none()
                if queued is not None:
                    logger.warning(
                        "Incarnation %s minted for partition %r collides with "
                        "garbage awaiting purge; re-minting",
                        incarnation,
                        partition_key,
                    )
                    raise _RegistryInsertRejectedError(str(incarnation))
        except IntegrityError as err:
            if await self.get(partition_key) is not None:
                raise VectorStorePartitionAlreadyExistsError(
                    self._vector_store_name, partition_key
                ) from err
            logger.warning(
                "Registry insert for partition %r with incarnation %s failed and no "
                "row exists under the key; retrying with a fresh incarnation",
                partition_key,
                incarnation,
            )
            raise _RegistryInsertRejectedError(str(incarnation)) from err

    @override
    async def get(self, partition_key: str) -> RegisteredPartition | None:
        async with self._engine.connect() as connection:
            row = (
                await connection.execute(
                    select(
                        self._partitions.c.incarnation, self._partitions.c.schema
                    ).where(
                        self._partitions.c.partition_key == partition_key,
                    )
                )
            ).one_or_none()
        if row is None:
            return None
        return RegisteredPartition(
            incarnation=row.incarnation,
            schema=PartitionSchema.model_validate(row.schema),
        )

    @override
    async def is_live(self, incarnation: UUID) -> bool:
        async with self._engine.connect() as connection:
            row = (
                await connection.execute(
                    select(self._partitions.c.partition_key).where(
                        self._partitions.c.incarnation == incarnation
                    )
                )
            ).scalar_one_or_none()
        return row is not None

    @override
    async def unregister(self, partition_key: str) -> None:
        # One transaction: the partition is unreachable as soon as it
        # commits, and the queue row is the incarnation's tombstone. The
        # segment store pins the row before its queue insert with the
        # fence its writes use; the registry has no such fence to reuse,
        # so the DELETE goes first: it takes the row's write lock, racing
        # deleters serialize on it and the loser deletes nothing, on
        # PostgreSQL and on SQLite alike, and RETURNING resolves the
        # incarnation in the same round trip.
        async with self._engine.begin() as connection:
            incarnation = (
                await connection.execute(
                    delete(self._partitions)
                    .where(
                        self._partitions.c.partition_key == partition_key,
                    )
                    .returning(self._partitions.c.incarnation)
                )
            ).scalar_one_or_none()
            if incarnation is None:
                return
            await connection.execute(
                insert(self._purge_queue).values(
                    incarnation=incarnation,
                    partition_key=partition_key,
                    enqueued_at=func.now(),
                )
            )

    @override
    @asynccontextmanager
    async def claim_due(self) -> AsyncIterator[PurgeClaim | None]:
        # The queue stores only the deletion's time, on the database clock;
        # the retention is policy, applied by the database's own arithmetic
        # when a claim is decided, so a changed retention reaches every
        # tombstone. A tombstone is due once the retention has passed since
        # its deletion, oldest deletion first. By then every write that was
        # in flight at the deletion has landed and none can land later, so
        # a round that finds nothing proves the incarnation empty for good.
        # Tombstones of one tick of the clock are unordered among
        # themselves. The claim is a row lock held for the body: on
        # PostgreSQL a concurrent purger skips the locked entry and takes
        # the next; on SQLite a doubly claimed entry costs a repeated,
        # idempotent round.
        queue = self._purge_queue
        async with self._engine.begin() as connection:
            row = (
                await connection.execute(
                    select(queue.c.incarnation)
                    .where(queue.c.enqueued_at <= self._retention_cutoff())
                    .order_by(queue.c.enqueued_at)
                    .limit(1)
                    .with_for_update(skip_locked=True)
                )
            ).one_or_none()
            if row is None:
                yield None
                return
            claim = PurgeClaim(incarnation=row.incarnation)
            yield claim
            if claim.found is None:
                raise RuntimeError(
                    f"Purge round for incarnation {claim.incarnation} ended "
                    "without reporting what it found"
                )
            if not claim.found:
                await connection.execute(
                    delete(queue).where(queue.c.incarnation == claim.incarnation)
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
