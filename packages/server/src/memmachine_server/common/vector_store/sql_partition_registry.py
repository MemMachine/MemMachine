"""
A partition registry in SQL, for the stores whose backend holds only points.

Qdrant and Milvus have no transactions, unique constraints or conditional
writes, so a catalog kept inside them cannot arbitrate two processes
creating, deleting or reclaiming the same partition. The registry lives in
the deployment's relational database instead: one table pair per backend
kind, shared by every store on that kind of backend, with a row per live
partition keyed by collection and partition key, whose incarnation is the
name every point of that life carries, and a purge queue of dead
incarnations claimed oldest-first. Creation is an insert the primary key
arbitrates, deletion is one transaction, and a purge claim is a row lock
the database hands to one purger at a time.

A queue entry is the dead incarnation's tombstone. The backend holds the
points, and a write the registry read as live can land there after the
purge that followed the deletion, so one purge cannot be the last: the
entry stays through purge rounds until a round finds nothing, then
through a retention measured on the database clock, then through one
more round that finds nothing again. Only then is it removed, and until
then the incarnation is never re-minted.
"""

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import NamedTuple
from uuid import UUID, uuid4

from sqlalchemy import (
    JSON,
    Column,
    DateTime,
    Index,
    MetaData,
    String,
    Table,
    Uuid,
    delete,
    func,
    insert,
    or_,
    select,
    update,
)
from sqlalchemy.exc import DBAPIError, IntegrityError
from sqlalchemy.ext.asyncio import AsyncEngine

from .data_types import (
    COLLECTION_NAME_MAX_BYTES,
    PartitionSchema,
    VectorStoreAttemptsExhaustedError,
    VectorStorePartitionAlreadyExistsError,
)

logger = logging.getLogger(__name__)

_MAX_MINT_ATTEMPTS = 10


class _RegistryInsertRejectedError(Exception):
    """A registry insert was rejected; retry with a fresh incarnation."""


class RegisteredPartition(NamedTuple):
    """A live partition: the incarnation its points carry and the schema it was created under."""

    incarnation: UUID
    schema: PartitionSchema


@dataclass
class PurgeClaim:
    """A claimed tombstone: the incarnation to purge, and what the round found.

    The purger sets `found` before the claim ends: True when the backend
    still held points under the incarnation, False when it held none.
    """

    incarnation: UUID
    clean_at: datetime | None
    found: bool | None = None


class SqlPartitionRegistry:
    """The partition registry of one collection, in a table pair shared by its backend kind.

    `table_prefix` names the backend kind: every store on that kind of
    backend shares `{prefix}_pt` and `{prefix}_gc`, and the collection is
    part of every key, so stores of different collections keep apart in
    one database. `tombstone_retention` is how long a dead incarnation's
    entry outlives the first purge round that found nothing; it must
    exceed, by orders of magnitude, the longest a write to the backend
    can be in flight.
    """

    def __init__(
        self,
        *,
        engine: AsyncEngine,
        table_prefix: str,
        collection: str,
        tombstone_retention: timedelta,
    ) -> None:
        """Bind to the registry tables of one backend kind, for one collection."""
        self._engine = engine
        self._collection = collection
        self._tombstone_retention = tombstone_retention
        metadata = MetaData()
        self._partitions = Table(
            f"{table_prefix}_pt",
            metadata,
            Column("collection", String(COLLECTION_NAME_MAX_BYTES), primary_key=True),
            Column("partition_key", String(255), primary_key=True),
            Column("incarnation", Uuid, nullable=False, unique=True),
            # The dimensions and declared schema the partition was created
            # under, so a store built with others fails loudly instead of
            # filtering on indexes that are not there.
            Column("schema_json", JSON, nullable=False),
        )
        self._purge_queue = Table(
            f"{table_prefix}_gc",
            metadata,
            Column("incarnation", Uuid, primary_key=True),
            Column("collection", String(COLLECTION_NAME_MAX_BYTES), nullable=False),
            Column("partition_key", String(255), nullable=False),
            Column("enqueued_at", DateTime(timezone=True), nullable=False),
            # When a purge round last found nothing under the incarnation;
            # cleared by a round that finds something. The entry is removed
            # by a round that finds nothing a retention after this.
            Column("clean_at", DateTime(timezone=True), nullable=True),
            Index(f"{table_prefix}_gc__cl_ea", "collection", "enqueued_at"),
        )
        self._metadata = metadata

    async def provision(self) -> None:
        """Create the registry tables, idempotently.

        Two provisioners racing on an empty database can both find the
        tables absent and both issue the DDL; the loser's fails, and its
        second pass finds the winner's tables and creates nothing. Any
        other failure fails the second pass too.
        """
        try:
            await self._create_tables()
        except DBAPIError:
            await self._create_tables()

    async def _create_tables(self) -> None:
        async with self._engine.begin() as connection:
            await connection.run_sync(self._metadata.create_all)

    async def create(self, partition_key: str, schema: PartitionSchema) -> UUID:
        """Register a new partition under a freshly minted incarnation.

        The primary key arbitrates the partition key across processes; the
        incarnation's unique constraint and the in-transaction queue check
        reject an incarnation that is live or still awaiting purge, so no
        points can be adopted by, or reclaimed out from under, a new
        partition.

        Raises:
            VectorStorePartitionAlreadyExistsError: The partition key is taken.
            VectorStoreAttemptsExhaustedError:
                Every minted incarnation was rejected for another reason.
        """
        attempts = 0
        while True:
            incarnation = uuid4()
            try:
                await self._insert(partition_key, incarnation, schema)
            except _RegistryInsertRejectedError as err:
                attempts += 1
                if attempts >= _MAX_MINT_ATTEMPTS:
                    raise VectorStoreAttemptsExhaustedError(
                        f"Creating partition {partition_key!r} of collection "
                        f"{self._collection!r} made no progress after "
                        f"{_MAX_MINT_ATTEMPTS} attempts"
                    ) from err
                continue
            return incarnation

    async def _insert(
        self, partition_key: str, incarnation: UUID, schema: PartitionSchema
    ) -> None:
        try:
            async with self._engine.begin() as connection:
                await connection.execute(
                    insert(self._partitions).values(
                        collection=self._collection,
                        partition_key=partition_key,
                        incarnation=incarnation,
                        schema_json=schema.model_dump(mode="json"),
                    )
                )
                queued = (
                    await connection.execute(
                        select(self._purge_queue.c.incarnation).where(
                            self._purge_queue.c.incarnation == incarnation
                        )
                    )
                ).scalar_one_or_none()
                if queued is not None:
                    logger.warning(
                        "Incarnation %s minted for partition %r of collection %r "
                        "collides with garbage awaiting purge; re-minting",
                        incarnation,
                        partition_key,
                        self._collection,
                    )
                    raise _RegistryInsertRejectedError(str(incarnation))
        except IntegrityError as err:
            if await self.get(partition_key) is not None:
                raise VectorStorePartitionAlreadyExistsError(
                    self._collection, partition_key
                ) from err
            logger.warning(
                "Registry insert for partition %r of collection %r with "
                "incarnation %s failed and no row exists under the key; "
                "retrying with a fresh incarnation",
                partition_key,
                self._collection,
                incarnation,
            )
            raise _RegistryInsertRejectedError(str(incarnation)) from err

    async def get(self, partition_key: str) -> RegisteredPartition | None:
        """The live partition under the key, or None."""
        async with self._engine.connect() as connection:
            row = (
                await connection.execute(
                    select(
                        self._partitions.c.incarnation, self._partitions.c.schema_json
                    ).where(
                        self._partitions.c.collection == self._collection,
                        self._partitions.c.partition_key == partition_key,
                    )
                )
            ).one_or_none()
        if row is None:
            return None
        return RegisteredPartition(
            incarnation=row.incarnation,
            schema=PartitionSchema.model_validate(row.schema_json),
        )

    async def is_live(self, incarnation: UUID) -> bool:
        """Whether a partition is still registered under this incarnation."""
        async with self._engine.connect() as connection:
            row = (
                await connection.execute(
                    select(self._partitions.c.partition_key).where(
                        self._partitions.c.incarnation == incarnation
                    )
                )
            ).scalar_one_or_none()
        return row is not None

    async def delete(self, partition_key: str) -> None:
        """Unregister the partition and queue its incarnation for purge, in one transaction.

        Idempotent: no row under the key is the no-op case. The partition
        is unreachable as soon as the transaction commits; its points are
        reclaimed by the purge rounds that claim the entry, its tombstone.
        """
        # The DELETE goes first: it takes the row's write lock, so racing
        # deleters serialize on it and the loser deletes nothing, on
        # PostgreSQL and on SQLite alike.
        async with self._engine.begin() as connection:
            incarnation = (
                await connection.execute(
                    delete(self._partitions)
                    .where(
                        self._partitions.c.collection == self._collection,
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
                    collection=self._collection,
                    partition_key=partition_key,
                    enqueued_at=func.now(),
                )
            )

    @asynccontextmanager
    async def claim_oldest(self) -> AsyncIterator[PurgeClaim | None]:
        """Claim this collection's oldest tombstone due for a purge round.

        Yields None when no tombstone is due: the queue is empty, or every
        entry had a clean round less than the retention ago. The claim is a
        row lock held for the body: on PostgreSQL a concurrent purger skips
        the locked entry and takes the next; on SQLite the writers
        serialize at the end of the round, so a doubly claimed entry costs
        a repeated, idempotent round and never a missed one.

        The body purges and sets `found`. A round that found points keeps
        the entry and clears `clean_at`, so rounds continue; a round that
        found none stamps `clean_at` the first time and removes the entry
        when it is the round due after the retention. A body that raises
        leaves the entry as it was, for a later claim.
        """
        async with self._engine.begin() as connection:
            database_now = (
                await connection.execute(
                    select(func.now(type_=DateTime(timezone=True)))
                )
            ).scalar_one()
            row = (
                await connection.execute(
                    select(
                        self._purge_queue.c.incarnation, self._purge_queue.c.clean_at
                    )
                    .where(
                        self._purge_queue.c.collection == self._collection,
                        or_(
                            self._purge_queue.c.clean_at.is_(None),
                            self._purge_queue.c.clean_at
                            <= database_now - self._tombstone_retention,
                        ),
                    )
                    .order_by(self._purge_queue.c.enqueued_at)
                    .limit(1)
                    .with_for_update(skip_locked=True)
                )
            ).one_or_none()
            if row is None:
                yield None
                return
            claim = PurgeClaim(incarnation=row.incarnation, clean_at=row.clean_at)
            yield claim
            if claim.found is None:
                raise RuntimeError(
                    f"Purge round for incarnation {claim.incarnation} of collection "
                    f"{self._collection!r} ended without reporting what it found"
                )
            entry = self._purge_queue.c.incarnation == claim.incarnation
            if claim.found:
                await connection.execute(
                    update(self._purge_queue).where(entry).values(clean_at=None)
                )
            elif claim.clean_at is None:
                await connection.execute(
                    update(self._purge_queue).where(entry).values(clean_at=func.now())
                )
            else:
                await connection.execute(delete(self._purge_queue).where(entry))
