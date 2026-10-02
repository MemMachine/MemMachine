"""SQLAlchemy implementation of the EventMemoryStore interface."""

import json
import logging
import sqlite3
from collections.abc import AsyncIterator, Callable, Iterable, Mapping, Sequence
from contextlib import asynccontextmanager
from datetime import UTC, datetime, timedelta, timezone
from functools import partial
from typing import override
from uuid import UUID, uuid4

from pydantic import (
    BaseModel,
    Field,
    InstanceOf,
    JsonValue,
    field_validator,
)
from sqlalchemy import (
    JSON,
    ColumnCollection,
    ColumnElement,
    DateTime,
    ForeignKeyConstraint,
    Index,
    Integer,
    LargeBinary,
    Select,
    String,
    Tuple,
    Uuid,
    bindparam,
    delete,
    false,
    func,
    insert,
    select,
    true,
    tuple_,
    update,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.dialects.postgresql import insert as postgresql_insert
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.engine import Dialect
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
)
from sqlalchemy.orm import (
    DeclarativeBase,
    MappedColumn,
    aliased,
    mapped_column,
)
from sqlalchemy.pool import StaticPool
from sqlalchemy.types import TypeDecorator

from memmachine_server.common.filter.filter_parser import (
    FilterExpr,
    demangle_user_metadata_key,
    normalize_filter_field,
)
from memmachine_server.common.filter.sql_filter_util import (
    FieldEncoding,
    compile_sql_filter,
)
from memmachine_server.common.metrics_factory import (
    MetricsFactory,
    OperationTracker,
)
from memmachine_server.common.payload_codec import PayloadCodec
from memmachine_server.common.payload_codec.payload_codec_config import (
    PlaintextPayloadCodecConfig,
    decode_payload_codec_config,
    encode_payload_codec_config,
)
from memmachine_server.common.payload_codec.plaintext_payload_codec import (
    PlaintextPayloadCodec,
)
from memmachine_server.common.properties_json import (
    decode_properties,
    encode_properties,
)
from memmachine_server.common.utils import utc_offset_seconds
from memmachine_server.episodic_memory.event_memory.data_types import (
    Neighborhood,
    NullContext,
    Segment,
    decode_block,
    decode_context,
    encode_block,
    encode_context,
)
from memmachine_server.episodic_memory.event_memory.event_memory_store.data_types import (
    EventMemoryStoreAttemptsExhaustedError,
    EventMemoryStoreEventAlreadyStoredError,
    EventMemoryStorePartitionAlreadyExistsError,
    EventMemoryStorePartitionConfig,
    EventMemoryStorePartitionConfigMismatchError,
    EventMemoryStorePartitionHandleStaleError,
)
from memmachine_server.episodic_memory.event_memory.event_memory_store.event_memory_store import (
    EventMemoryStore,
    EventMemoryStorePartition,
    EventMemoryStorePartitionWriter,
)
from memmachine_server.episodic_memory.event_memory.event_memory_store.utils import (
    validate_partition_key,
)

logger = logging.getLogger(__name__)

_JSON_AUTO = JSON().with_variant(JSONB, "postgresql")


# Consecutive failed mint attempts before the store concludes it
# is re-attempting a persistent database error rather than losing races: a real
# UUID collision is a once-in-the-universe event and each race retry
# requires another actor to have changed the registry in a ~millisecond
# window, so consecutive failures at this depth mean the IntegrityError
# has some other, permanent cause.
_MAX_MINT_ATTEMPTS = 10

# Partition deletion depends on RETURNING, which SQLite added in 3.35.
_MIN_SQLITE_VERSION = (3, 35)

# A context read takes its context from at most this many segments on each
# side of a seed, matching or not.
_MAX_CONTEXT_DISTANCE = 1_000

# A neighborhood read's row predicates over a column collection, the
# segment table's or a window's.
_NeighborConditions = Callable[
    [ColumnCollection[str, ColumnElement]], list[ColumnElement[bool]]
]


class _RegistryInsertRejectedError(Exception):
    """A registry insert was rejected although its key is free.

    Raised when the insert fails with an integrity error but no row
    exists under the key, or when the minted incarnation still has
    garbage awaiting purge.
    """


# ORM models


class UtcInstant(TypeDecorator[datetime]):
    """A timestamp column that holds a UTC instant.

    On write, a timezone-aware datetime is converted to UTC and a naive
    one is rejected. On read, PostgreSQL returns an aware datetime, which
    is passed through; SQLite stores no zone and returns a naive one,
    which is given `tzinfo=UTC`, the zone it was written in.
    """

    impl = DateTime(timezone=True)
    cache_ok = True

    @override
    def process_bind_param(
        self, value: datetime | None, dialect: Dialect
    ) -> datetime | None:
        if value is None:
            return None
        if value.tzinfo is None:
            raise ValueError(f"a timestamp must be timezone-aware: {value!r}")
        return value.astimezone(UTC)

    @override
    def process_result_value(
        self, value: datetime | None, dialect: Dialect
    ) -> datetime | None:
        if value is None or value.tzinfo is not None:
            return value
        return value.replace(tzinfo=UTC)


class BaseEventMemoryStore(DeclarativeBase):
    """Base class for event memory store tables."""


class PartitionRow(BaseEventMemoryStore):
    """The tenant registry: one row per live partition incarnation."""

    __tablename__ = "event_memory_store_pt"

    partition_key: MappedColumn[str] = mapped_column(String(255), primary_key=True)
    incarnation: MappedColumn[UUID] = mapped_column(Uuid, nullable=False, unique=True)
    payload_codec_config: MappedColumn[dict[str, JsonValue]] = mapped_column(
        _JSON_AUTO,
        nullable=False,
    )


class EventRow(BaseEventMemoryStore):
    """One row per event the partition holds.

    The primary key is what makes an event addable once: a second
    `add_events` naming it conflicts here, before any segment is written.
    Deleting the row cascades to the event's segments and, through them,
    their derivative links.
    """

    __tablename__ = "event_memory_store_ev"

    incarnation: MappedColumn[UUID] = mapped_column(Uuid, primary_key=True)
    uuid: MappedColumn[UUID] = mapped_column(Uuid, primary_key=True)


class SegmentRow(BaseEventMemoryStore):
    """Persisted segment."""

    __tablename__ = "event_memory_store_sg"

    incarnation: MappedColumn[UUID] = mapped_column(Uuid, primary_key=True)

    uuid: MappedColumn[UUID] = mapped_column(Uuid, primary_key=True)
    event_uuid: MappedColumn[UUID] = mapped_column(Uuid, nullable=False)
    index: MappedColumn[int] = mapped_column(Integer, nullable=False)
    offset: MappedColumn[int] = mapped_column(Integer, nullable=False)
    timestamp: MappedColumn[datetime] = mapped_column(UtcInstant, nullable=False)
    timestamp_timezone_offset: MappedColumn[int] = mapped_column(
        Integer, nullable=False, default=0
    )
    source_id: MappedColumn[str | None] = mapped_column(String(255), nullable=True)
    context: MappedColumn[bytes] = mapped_column(LargeBinary, nullable=False)
    block: MappedColumn[bytes] = mapped_column(LargeBinary, nullable=False)
    properties: MappedColumn[dict[str, JsonValue]] = mapped_column(
        _JSON_AUTO, nullable=False, default=dict
    )

    # No foreign key to the registry: registry rows and data rows are
    # deliberately decoupled so that partition deletion is a registry write
    # (O(1)) and the purge queue reclaims data rows asynchronously.
    __table_args__ = (
        ForeignKeyConstraint(
            ["incarnation", "event_uuid"],
            [
                "event_memory_store_ev.incarnation",
                "event_memory_store_ev.uuid",
            ],
            ondelete="CASCADE",
        ),
        # Serves the event lookups and the cascade from the event row.
        Index(
            "event_memory_store_sg__in_ev",
            "incarnation",
            "event_uuid",
        ),
        Index(
            "event_memory_store_sg__in_ts_ev_ix_of",
            "incarnation",
            "timestamp",
            "event_uuid",
            "index",
            "offset",
        ),
    )


class DerivativeLinkRow(BaseEventMemoryStore):
    """Maps a derivative UUID to its owning segment."""

    __tablename__ = "event_memory_store_dv_ln"

    incarnation: MappedColumn[UUID] = mapped_column(Uuid, primary_key=True)

    uuid: MappedColumn[UUID] = mapped_column(Uuid, primary_key=True)
    segment_uuid: MappedColumn[UUID] = mapped_column(Uuid, nullable=False)

    __table_args__ = (
        ForeignKeyConstraint(
            ["incarnation", "segment_uuid"],
            [
                "event_memory_store_sg.incarnation",
                "event_memory_store_sg.uuid",
            ],
            ondelete="CASCADE",
        ),
        Index(
            "event_memory_store_dv_ln__in_su",
            "incarnation",
            "segment_uuid",
        ),
    )


class PurgeQueueRow(BaseEventMemoryStore):
    """The purge queue: one row per dead partition incarnation.

    Claimed oldest-first by the enqueue stamp, which is the database clock,
    so entries from every server order on one clock; entries stamped in the
    same tick are unordered among themselves. The incarnation identifies
    the rows to reclaim; the logical key is carried for forensics. Every
    segment row of the incarnation keyed at or below purged_through is
    deleted, NULL until the first full batch of segments; every event row
    keyed at or below events_purged_through is, NULL until the first full
    batch of events. Events have a cursor of their own because a batch
    that ends exactly on the last segment leaves purged_through naming a
    segment, which says nothing about how far the events got.
    """

    __tablename__ = "event_memory_store_gc"

    incarnation: MappedColumn[UUID] = mapped_column(Uuid, primary_key=True)
    partition_key: MappedColumn[str] = mapped_column(String(255), nullable=False)
    enqueued_at: MappedColumn[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )
    purged_through: MappedColumn[UUID | None] = mapped_column(Uuid, nullable=True)
    events_purged_through: MappedColumn[UUID | None] = mapped_column(
        Uuid, nullable=True
    )

    __table_args__ = (Index("event_memory_store_gc__ea", "enqueued_at"),)


class SQLAlchemyEventMemoryStorePartition(EventMemoryStorePartition):
    """SQLAlchemy-backed partition handle."""

    def __init__(
        self,
        partition_key: str,
        incarnation: UUID,
        create_session: async_sessionmaker[AsyncSession],
        is_sqlite: bool,
        config: EventMemoryStorePartitionConfig,
        payload_codec: PayloadCodec,
        tracker: OperationTracker,
    ) -> None:
        """Initialize with a partition key, its incarnation, and the store's session factory."""
        self._partition_key = partition_key
        # Data rows are keyed by the incarnation alone: rows of a
        # deleted-and-recreated partition under the same logical key are
        # invisible to the new incarnation while the purge queue reclaims
        # them, and a data query cannot even be built without resolving
        # the registry first.
        self._incarnation = incarnation
        self._config = config
        self._payload_codec = payload_codec
        self._tracker = tracker
        self._create_session = create_session
        self._is_sqlite = is_sqlite

    @override
    @property
    def config(self) -> EventMemoryStorePartitionConfig:
        return self._config

    async def _lock_partition_for_write(
        self, session: AsyncSession, *, exclusive: bool = False
    ) -> None:
        """Pin this incarnation's registry row; raise if the handle is stale.

        The shared row lock blocks concurrent deletion (which takes the
        exclusive row lock) until the write completes; the incarnation
        predicate fences a handle that outlived its partition. With
        `exclusive`, the row is taken exclusively instead, which waits
        for every shared holder, that is every write in flight, and
        blocks new ones until the transaction ends. SQLite
        drops locking clauses and its driver defers BEGIN until the first
        data-modifying statement -- a SELECT-only fence would run outside
        the write transaction and fence nothing. The proper primitive,
        BEGIN IMMEDIATE, requires taking over transaction management of
        the whole engine in SQLAlchemy (isolation_level=None plus a
        begin-event hook), which this store cannot do to a caller-owned,
        possibly shared engine -- so the fence is a self-checking
        registry-row UPDATE instead: the driver emits BEGIN before DML
        and SQLite takes the same write lock, scoped to this
        transaction, with the match count as the staleness check.
        """
        if self._is_sqlite:
            fenced = await (await session.connection()).execute(
                update(PartitionRow)
                .where(PartitionRow.incarnation == self._incarnation)
                .values(incarnation=self._incarnation)
            )
            if fenced.rowcount == 0:
                raise EventMemoryStorePartitionHandleStaleError(self._partition_key)
            return
        await self._ensure_partition_live(session, pin=True, exclusive=exclusive)

    def _registry_row_query(self) -> Select[tuple[str]]:
        """This incarnation's registry row: absent once the handle is stale.

        Reads conjoin its EXISTS to their data statements, so one statement
        (one snapshot) both checks liveness and reads: a stale handle reads
        nothing, at no extra round trip.
        """
        return select(PartitionRow.partition_key).where(
            PartitionRow.incarnation == self._incarnation
        )

    async def _ensure_partition_live(
        self, session: AsyncSession, *, pin: bool = False, exclusive: bool = False
    ) -> None:
        """Raise if this handle's incarnation is no longer registered.

        With `pin`, the row is read under a shared lock that blocks
        deletion for the rest of the transaction, or, with `exclusive`
        as well, under the exclusive lock. Reads call this without
        the lock, and only when their data statement returned no rows: it
        tells an empty partition from a stale handle.
        """
        query = self._registry_row_query()
        if pin:
            query = query.with_for_update(read=not exclusive)
        row = (await session.execute(query)).scalar_one_or_none()
        if row is None:
            raise EventMemoryStorePartitionHandleStaleError(self._partition_key)

    # Writing

    @override
    @asynccontextmanager
    async def write(
        self, *, exclusive: bool = False
    ) -> AsyncIterator[EventMemoryStorePartitionWriter]:
        async with (
            self._tracker("write"),
            self._create_session() as session,
            session.begin(),
        ):
            await self._lock_partition_for_write(session, exclusive=exclusive)
            yield SQLAlchemyEventMemoryStorePartitionWriter(
                session,
                incarnation=self._incarnation,
                is_sqlite=self._is_sqlite,
                payload_codec=self._payload_codec,
            )

    # Retrieval

    @override
    async def get_segments(
        self,
        segment_uuids: Iterable[UUID],
        *,
        since: datetime | None = None,
        until: datetime | None = None,
        source_ids: Iterable[str] | None = None,
        property_filter: FilterExpr | None = None,
    ) -> dict[UUID, Segment]:
        SQLAlchemyEventMemoryStorePartition._require_aware_bounds(since, until)
        segment_uuids = set(segment_uuids)
        if not segment_uuids:
            return {}
        conditions = SQLAlchemyEventMemoryStorePartition._row_conditions(
            SegmentRow.__table__.c,
            since=since,
            until=until,
            source_ids=source_ids,
            property_filter=property_filter,
        )

        async with (
            self._tracker("get_segments"),
            self._create_session() as session,
        ):
            rows_by_uuid = await self._segment_rows_by_uuid(
                session, segment_uuids, conditions
            )
            if not rows_by_uuid:
                # Empty may mean a stale handle; the registry read raises if so.
                await self._ensure_partition_live(session)
            return {
                segment_uuid: self._segment_from_segment_row(row)
                for segment_uuid, row in rows_by_uuid.items()
            }

    @override
    async def get_segment_neighborhoods(
        self,
        seed_uuids: Iterable[UUID],
        *,
        before: int = 0,
        after: int = 0,
        since: datetime | None = None,
        until: datetime | None = None,
        source_ids: Iterable[str] | None = None,
        property_filter: FilterExpr | None = None,
    ) -> dict[UUID, Neighborhood]:
        if before < 0:
            raise ValueError(f"before must be nonnegative: {before}")
        if after < 0:
            raise ValueError(f"after must be nonnegative: {after}")
        SQLAlchemyEventMemoryStorePartition._require_aware_bounds(since, until)
        seed_uuids = set(seed_uuids)
        if not seed_uuids:
            return {}
        filtered = any(
            value is not None for value in (since, until, source_ids, property_filter)
        )
        neighbor_conditions = (
            partial(
                SQLAlchemyEventMemoryStorePartition._row_conditions,
                since=since,
                until=until,
                source_ids=None if source_ids is None else list(source_ids),
                property_filter=property_filter,
            )
            if filtered
            else None
        )

        async with (
            self._tracker("get_segment_neighborhoods"),
            self._create_session() as session,
        ):
            # The seed is an address: it is located whatever the filters say.
            seed_rows_by_uuid = await self._segment_rows_by_uuid(
                session, seed_uuids, []
            )
            if not seed_rows_by_uuid:
                await self._ensure_partition_live(session)
                return {}

            if before <= 0 and after <= 0:
                return {
                    seed_uuid: Neighborhood(before=[], after=[])
                    for seed_uuid in seed_rows_by_uuid
                }

            if not self._is_sqlite:
                neighbor_rows_by_seed = await self._get_context_rows_lateral(
                    session, seed_rows_by_uuid, before, after, neighbor_conditions
                )
            else:
                neighbor_rows_by_seed = await self._get_context_rows_loop(
                    session, seed_rows_by_uuid, before, after, neighbor_conditions
                )

            # Each statement above took its own snapshot (READ COMMITTED):
            # a deletion committing between the seeds statement and the
            # context statements would leave the seeds with silently
            # empty context. One registry read after the last statement
            # turns that window into the contractual stale-handle error.
            await self._ensure_partition_live(session)

            return {
                seed_uuid: Neighborhood(
                    before=[
                        self._segment_from_segment_row(row)
                        for row in reversed(backward_rows)
                    ],
                    after=[self._segment_from_segment_row(row) for row in forward_rows],
                )
                for seed_uuid, (
                    backward_rows,
                    forward_rows,
                ) in neighbor_rows_by_seed.items()
            }

    async def _segment_rows_by_uuid(
        self,
        session: AsyncSession,
        segment_uuids: set[UUID],
        conditions: Sequence[ColumnElement[bool]],
    ) -> dict[UUID, SegmentRow]:
        """The rows of this partition among `segment_uuids` that satisfy `conditions`."""
        query = select(SegmentRow).where(
            SegmentRow.uuid.in_(segment_uuids),
            SegmentRow.incarnation == self._incarnation,
            self._registry_row_query().exists(),
            *conditions,
        )
        rows = (await session.execute(query)).scalars().all()
        return {row.uuid: row for row in rows}

    async def _get_context_rows_lateral(
        self,
        session: AsyncSession,
        seed_rows_by_uuid: Mapping[UUID, SegmentRow],
        before: int,
        after: int,
        neighbor_conditions: _NeighborConditions | None,
    ) -> dict[UUID, tuple[list[SegmentRow], list[SegmentRow]]]:
        """Get backward/forward context using LATERAL joins (non-SQLite)."""
        seeds_subquery = (
            select(
                SegmentRow.uuid.label("seed_uuid"),
                SegmentRow.timestamp.label("seed_timestamp"),
                SegmentRow.event_uuid.label("seed_event_uuid"),
                SegmentRow.index.label("seed_index"),
                SegmentRow.offset.label("seed_offset"),
            )
            .where(
                SegmentRow.incarnation == self._incarnation,
                SegmentRow.uuid.in_(seed_rows_by_uuid.keys()),
            )
            .subquery("seeds")
        )
        seed_ordering_columns = tuple_(
            seeds_subquery.c.seed_timestamp,
            seeds_subquery.c.seed_event_uuid,
            seeds_subquery.c.seed_index,
            seeds_subquery.c.seed_offset,
        )

        async def get_context_rows_directional(
            *,
            backward: bool,
            limit: int,
        ) -> dict[UUID, list[SegmentRow]]:
            """Get context rows per seed in the specified direction."""
            # Build a LATERAL subquery that gets context rows for each seed.
            lateral_subquery = (
                self._context_rows_query(
                    seed_ordering_columns,
                    backward=backward,
                    limit=limit,
                    neighbor_conditions=neighbor_conditions,
                )
                .subquery()
                .lateral("context")
            )

            # Join each seed to its context rows via the LATERAL subquery,
            # loaded as segment rows by the ORM.
            seed_context_join_query = select(
                seeds_subquery.c.seed_uuid, aliased(SegmentRow, lateral_subquery)
            ).select_from(seeds_subquery.join(lateral_subquery, true()))

            # Group result rows by seed UUID.
            rows_by_seed: dict[UUID, list[SegmentRow]] = {
                seed_uuid: [] for seed_uuid in seed_rows_by_uuid
            }
            for seed_uuid, segment_row in (
                await session.execute(seed_context_join_query)
            ).all():
                rows_by_seed[seed_uuid].append(segment_row)
            return rows_by_seed

        backward_rows_by_seed = (
            await get_context_rows_directional(backward=True, limit=before)
            if before > 0
            else {seed_uuid: [] for seed_uuid in seed_rows_by_uuid}
        )

        forward_rows_by_seed = (
            await get_context_rows_directional(backward=False, limit=after)
            if after > 0
            else {seed_uuid: [] for seed_uuid in seed_rows_by_uuid}
        )

        return {
            seed_uuid: (
                backward_rows_by_seed[seed_uuid],
                forward_rows_by_seed[seed_uuid],
            )
            for seed_uuid in seed_rows_by_uuid
        }

    async def _get_context_rows_loop(
        self,
        session: AsyncSession,
        seed_rows_by_uuid: Mapping[UUID, SegmentRow],
        before: int,
        after: int,
        neighbor_conditions: _NeighborConditions | None,
    ) -> dict[UUID, tuple[list[SegmentRow], list[SegmentRow]]]:
        """Get backward/forward context per seed (SQLite fallback)."""
        # Build one statement per direction and run it for each seed, binding
        # the seed's timestamp, event UUID, index and offset. Building a
        # statement costs more CPU than SQLite spends running it, so building
        # one per seed would dominate the read.
        seed_ordering_values = tuple_(
            bindparam("seed_timestamp", type_=SegmentRow.timestamp.type),
            bindparam("seed_event_uuid", type_=SegmentRow.event_uuid.type),
            bindparam("seed_index", type_=SegmentRow.index.type),
            bindparam("seed_offset", type_=SegmentRow.offset.type),
        )

        async def get_context_rows_directional(
            *,
            backward: bool,
            limit: int,
        ) -> dict[UUID, list[SegmentRow]]:
            """Get context rows per seed in the specified direction."""
            # Loaded as segment rows by the ORM, which is cheaper than building
            # them from plain rows.
            context_rows_query = select(SegmentRow).from_statement(
                self._context_rows_query(
                    seed_ordering_values,
                    backward=backward,
                    limit=limit,
                    neighbor_conditions=neighbor_conditions,
                )
            )
            rows_by_seed: dict[UUID, list[SegmentRow]] = {}
            for seed_uuid, seed_row in seed_rows_by_uuid.items():
                seed_position = {
                    "seed_timestamp": seed_row.timestamp,
                    "seed_event_uuid": seed_row.event_uuid,
                    "seed_index": seed_row.index,
                    "seed_offset": seed_row.offset,
                }
                rows_by_seed[seed_uuid] = list(
                    (await session.execute(context_rows_query, seed_position))
                    .scalars()
                    .all()
                )
            return rows_by_seed

        backward_rows_by_seed = (
            await get_context_rows_directional(backward=True, limit=before)
            if before > 0
            else {seed_uuid: [] for seed_uuid in seed_rows_by_uuid}
        )

        forward_rows_by_seed = (
            await get_context_rows_directional(backward=False, limit=after)
            if after > 0
            else {seed_uuid: [] for seed_uuid in seed_rows_by_uuid}
        )

        return {
            seed_uuid: (
                backward_rows_by_seed[seed_uuid],
                forward_rows_by_seed[seed_uuid],
            )
            for seed_uuid in seed_rows_by_uuid
        }

    def _context_rows_query(
        self,
        seed_ordering_values: Tuple,
        *,
        backward: bool,
        limit: int,
        neighbor_conditions: _NeighborConditions | None,
    ) -> Select:
        """Select a seed's context on one side, nearest the seed first.

        `seed_ordering_values` is the seed's (timestamp, event_uuid, index,
        offset): bound parameters or an enclosing statement's columns.
        Context comes from at most the next _MAX_CONTEXT_DISTANCE segments. Unfiltered, it is the first `limit`
        of them. With `neighbor_conditions`, it is the first `limit` matches
        among them, read without the conditions, which the planner cannot
        estimate, so the plan stays an ordered index scan.
        Returns nothing once the partition is deleted.
        """
        # Built from Core columns, the table's and the window's: building from
        # the mapped class adds ORM work to every column reference, and an
        # alias of it adapts each column it hands out.
        segments = SegmentRow.__table__.c
        segment_ordering_columns = tuple_(
            segments.timestamp,
            segments.event_uuid,
            segments.index,
            segments.offset,
        )
        walk = (
            select(SegmentRow.__table__)
            .where(
                segments.incarnation == self._incarnation,
                segment_ordering_columns < seed_ordering_values
                if backward
                else segment_ordering_columns > seed_ordering_values,
                self._registry_row_query().exists(),
            )
            .order_by(
                *SQLAlchemyEventMemoryStorePartition._chronological_order(
                    segments, descending=backward
                )
            )
            # Refer to an enclosing statement that supplies the seed's
            # position instead of selecting from it again.
            .correlate_except(SegmentRow.__table__)
        )
        if neighbor_conditions is None:
            return walk.limit(min(limit, _MAX_CONTEXT_DISTANCE))
        window = walk.limit(_MAX_CONTEXT_DISTANCE).subquery("window")
        return (
            select(window)
            .where(*neighbor_conditions(window.c))
            .order_by(
                *SQLAlchemyEventMemoryStorePartition._chronological_order(
                    window.c, descending=backward
                )
            )
            .limit(limit)
        )

    @staticmethod
    def _chronological_order(
        columns: ColumnCollection[str, ColumnElement],
        *,
        descending: bool,
    ) -> list[ColumnElement]:
        """Segments' chronological order over `columns`, newest first if `descending`.

        `columns` is the segment table's columns or a subquery's.
        """
        order = [columns.timestamp, columns.event_uuid, columns.index, columns.offset]
        return [column.desc() for column in order] if descending else order

    @staticmethod
    def _require_aware_bounds(since: datetime | None, until: datetime | None) -> None:
        """Reject a naive bound, which names no instant.

        Called before a session is opened, so a bad bound costs no round
        trip.
        """
        for name, bound in (("since", since), ("until", until)):
            if bound is not None and bound.tzinfo is None:
                raise ValueError(f"{name} must be timezone-aware: {bound!r}")

    @staticmethod
    def _row_conditions(
        columns: ColumnCollection[str, ColumnElement],
        *,
        since: datetime | None,
        until: datetime | None,
        source_ids: Iterable[str] | None,
        property_filter: FilterExpr | None,
    ) -> list[ColumnElement[bool]]:
        """A read's row predicates over `columns`, on the typed fields and the properties.

        `columns` is the segment table's columns or a subquery's. The
        timestamp column's type binds an aware bound as its UTC instant,
        which is what the column holds and what SQLite compares digit by
        digit. An empty id or kind list admits nothing.
        """
        conditions: list[ColumnElement[bool]] = []
        if since is not None:
            conditions.append(columns.timestamp >= since)
        if until is not None:
            conditions.append(columns.timestamp < until)
        if source_ids is not None:
            source_ids = list(source_ids)
            conditions.append(
                columns.source_id.in_(source_ids) if source_ids else false()
            )
        if property_filter is not None:
            conditions.append(
                compile_sql_filter(
                    property_filter,
                    lambda field: (
                        SQLAlchemyEventMemoryStorePartition._resolve_segment_field(
                            field, columns=columns
                        )
                    ),
                )
            )
        return conditions

    @override
    async def get_derivative_uuids_by_event_uuids(
        self,
        event_uuids: Iterable[UUID],
    ) -> dict[UUID, list[UUID]]:
        event_uuids = set(event_uuids)
        if not event_uuids:
            return {}

        async with (
            self._tracker("get_derivative_uuids_by_event_uuids"),
            self._create_session() as session,
        ):
            # Outer joins, so an event with no derivatives still answers.
            query = (
                select(EventRow.uuid, DerivativeLinkRow.uuid)
                .outerjoin(
                    SegmentRow,
                    (SegmentRow.incarnation == EventRow.incarnation)
                    & (SegmentRow.event_uuid == EventRow.uuid),
                )
                .outerjoin(
                    DerivativeLinkRow,
                    (DerivativeLinkRow.incarnation == SegmentRow.incarnation)
                    & (DerivativeLinkRow.segment_uuid == SegmentRow.uuid),
                )
                .where(
                    EventRow.incarnation == self._incarnation,
                    EventRow.uuid.in_(event_uuids),
                    self._registry_row_query().exists(),
                )
            )
            rows = (await session.execute(query)).all()
            if not rows:
                await self._ensure_partition_live(session)

        result: dict[UUID, list[UUID]] = {}
        for event_uuid, derivative_uuid in rows:
            derivative_uuids = result.setdefault(event_uuid, [])
            if derivative_uuid is not None:
                derivative_uuids.append(derivative_uuid)
        return result

    @override
    async def get_segment_uuids_by_derivative_uuids(
        self,
        derivative_uuids: Iterable[UUID],
    ) -> dict[UUID, UUID]:
        derivative_uuids = set(derivative_uuids)
        if not derivative_uuids:
            return {}

        async with (
            self._tracker("get_segment_uuids_by_derivative_uuids"),
            self._create_session() as session,
        ):
            query = _segment_uuids_by_derivative_uuids_query(
                self._incarnation, derivative_uuids
            ).where(self._registry_row_query().exists())
            rows = (await session.execute(query)).all()
            if not rows:
                await self._ensure_partition_live(session)

        return {row.uuid: row.segment_uuid for row in rows}

    # Deletion

    @override
    async def delete_events(
        self,
        event_uuids: Iterable[UUID],
    ) -> None:
        event_uuids = set(event_uuids)
        if not event_uuids:
            return

        async with (
            self._tracker("delete_events"),
            self._create_session() as session,
            session.begin(),
        ):
            await self._lock_partition_for_write(session)
            if not self._is_sqlite:
                # Lock the event rows, then their segment rows, in
                # deterministic order, so concurrent deletions with
                # overlapping sets cannot deadlock: the cascade alone would
                # take the segment locks in table order. SQLite relies on
                # write serialization by the database.
                await session.execute(
                    select(EventRow.uuid)
                    .where(
                        EventRow.incarnation == self._incarnation,
                        EventRow.uuid.in_(event_uuids),
                    )
                    .order_by(EventRow.uuid)
                    .with_for_update()
                )
                await session.execute(
                    select(SegmentRow.uuid)
                    .where(
                        SegmentRow.incarnation == self._incarnation,
                        SegmentRow.event_uuid.in_(event_uuids),
                    )
                    .order_by(SegmentRow.uuid)
                    .with_for_update()
                )

            # CASCADE deletes segments, and through them derivatives.
            await session.execute(
                delete(EventRow).where(
                    EventRow.incarnation == self._incarnation,
                    EventRow.uuid.in_(event_uuids),
                )
            )

    @override
    async def delete_segments(
        self,
        segment_uuids: Iterable[UUID],
    ) -> None:
        segment_uuids = set(segment_uuids)
        if not segment_uuids:
            return

        async with (
            self._tracker("delete_segments"),
            self._create_session() as session,
            session.begin(),
        ):
            await self._lock_partition_for_write(session)
            if not self._is_sqlite:
                # Lock rows in deterministic order to prevent deadlocks
                # from concurrent deletions with overlapping UUID sets.
                # SQLite relies on write serialization by the database.
                await session.execute(
                    select(SegmentRow.uuid)
                    .where(
                        SegmentRow.incarnation == self._incarnation,
                        SegmentRow.uuid.in_(segment_uuids),
                    )
                    .order_by(SegmentRow.uuid)
                    .with_for_update()
                )

            # CASCADE deletes derivatives via FK.
            await session.execute(
                delete(SegmentRow).where(
                    SegmentRow.incarnation == self._incarnation,
                    SegmentRow.uuid.in_(segment_uuids),
                )
            )

    @staticmethod
    def _resolve_segment_field(
        field: str,
        *,
        columns: ColumnCollection[str, ColumnElement],
    ) -> tuple[ColumnElement, FieldEncoding]:
        """Map a filter field name to one of `columns` and its encoding."""
        if field == "timestamp":
            return columns.timestamp, "column"
        internal_name, is_user_metadata = normalize_filter_field(field)
        if is_user_metadata:
            key = demangle_user_metadata_key(internal_name)
            return columns.properties[key], "properties_json"
        return columns.properties[f"_{field}"], "properties_json"

    def _segment_from_segment_row(self, row: SegmentRow) -> Segment:
        """Convert a SegmentRow into a Segment."""
        context = decode_context(json.loads(self._payload_codec.decode(row.context)))
        if context is None:
            context = NullContext()
        block = decode_block(json.loads(self._payload_codec.decode(row.block)))
        properties = decode_properties(row.properties)
        original_timezone = timezone(timedelta(seconds=row.timestamp_timezone_offset))
        timestamp = row.timestamp.astimezone(original_timezone)
        return Segment(
            uuid=row.uuid,
            event_uuid=row.event_uuid,
            index=row.index,
            offset=row.offset,
            timestamp=timestamp,
            source_id=row.source_id,
            context=context,
            block=block,
            properties=properties,
        )


class SQLAlchemyEventMemoryStorePartitionWriter(EventMemoryStorePartitionWriter):
    """The transaction `SQLAlchemyEventMemoryStorePartition.write` opens."""

    def __init__(
        self,
        session: AsyncSession,
        *,
        incarnation: UUID,
        is_sqlite: bool,
        payload_codec: PayloadCodec,
    ) -> None:
        """Bind the writer to an open session and the partition's identity."""
        self._session = session
        self._incarnation = incarnation
        self._is_sqlite = is_sqlite
        self._payload_codec = payload_codec

    @override
    async def add_events(
        self,
        events: Mapping[UUID, Mapping[Segment, Iterable[UUID]]],
    ) -> None:
        events = {
            event_uuid: {
                segment: list(derivative_uuids)
                for segment, derivative_uuids in segments.items()
            }
            for event_uuid, segments in events.items()
        }
        for event_uuid, segments in events.items():
            for segment in segments:
                if segment.event_uuid != event_uuid:
                    raise ValueError(
                        f"segment {segment.uuid} names event {segment.event_uuid}, "
                        f"listed under {event_uuid}"
                    )
        if not events:
            return

        stored = await self._insert_event_rows(events.keys())
        already_stored = events.keys() - stored
        if already_stored:
            raise EventMemoryStoreEventAlreadyStoredError(already_stored)

        segments_to_derivative_uuids = {
            segment: derivative_uuids
            for segments in events.values()
            for segment, derivative_uuids in segments.items()
        }
        await self._insert_segments(segments_to_derivative_uuids.keys())
        await self._insert_derivative_links(segments_to_derivative_uuids)

    async def _insert_event_rows(self, event_uuids: Iterable[UUID]) -> set[UUID]:
        """Insert an event row per uuid, skipping those already held.

        Returns the uuids inserted. A concurrent transaction inserting the
        same row is waited for: if it commits, its uuid is missing from
        the result; if it rolls back, the row is ours.
        """
        insert_rows = sqlite_insert if self._is_sqlite else postgresql_insert
        statement = (
            insert_rows(EventRow)
            .values(
                [
                    {"incarnation": self._incarnation, "uuid": event_uuid}
                    for event_uuid in event_uuids
                ]
            )
            .on_conflict_do_nothing(index_elements=["incarnation", "uuid"])
            .returning(EventRow.uuid)
        )
        return set((await self._session.execute(statement)).scalars())

    async def _insert_segments(self, segments: Iterable[Segment]) -> None:
        """Insert segment rows."""
        segment_row_values = [
            {
                "uuid": segment.uuid,
                "incarnation": self._incarnation,
                "event_uuid": segment.event_uuid,
                "index": segment.index,
                "offset": segment.offset,
                # Store the UTC instant; SQLite does not persist tzinfo, so the
                # original offset is recorded separately and reapplied on read.
                "timestamp": segment.timestamp,
                "timestamp_timezone_offset": utc_offset_seconds(segment.timestamp),
                "source_id": segment.source_id,
                "context": self._payload_codec.encode(
                    json.dumps(encode_context(segment.context)).encode("utf-8")
                ),
                "block": self._payload_codec.encode(
                    json.dumps(encode_block(segment.block)).encode("utf-8")
                ),
                "properties": encode_properties(segment.properties),
            }
            for segment in segments
        ]
        if segment_row_values:
            await self._session.execute(insert(SegmentRow), segment_row_values)

    async def _insert_derivative_links(
        self,
        segments_to_derivative_uuids: Mapping[Segment, Iterable[UUID]],
    ) -> None:
        """Insert derivative rows."""
        derivative_row_values = [
            {
                "uuid": derivative_uuid,
                "incarnation": self._incarnation,
                "segment_uuid": segment.uuid,
            }
            for segment, derivative_uuids in segments_to_derivative_uuids.items()
            for derivative_uuid in derivative_uuids
        ]
        if derivative_row_values:
            await self._session.execute(
                insert(DerivativeLinkRow), derivative_row_values
            )

    @override
    async def get_segment_uuids_by_derivative_uuids(
        self,
        derivative_uuids: Iterable[UUID],
    ) -> dict[UUID, UUID]:
        derivative_uuids = set(derivative_uuids)
        if not derivative_uuids:
            return {}
        rows = (
            await self._session.execute(
                _segment_uuids_by_derivative_uuids_query(
                    self._incarnation, derivative_uuids
                )
            )
        ).all()
        return {row.uuid: row.segment_uuid for row in rows}


def _segment_uuids_by_derivative_uuids_query(
    incarnation: UUID, derivative_uuids: Iterable[UUID]
) -> Select[tuple[UUID, UUID]]:
    """The link rows of the given derivatives: served by the primary key."""
    return select(DerivativeLinkRow.uuid, DerivativeLinkRow.segment_uuid).where(
        DerivativeLinkRow.incarnation == incarnation,
        DerivativeLinkRow.uuid.in_(derivative_uuids),
    )


class SQLAlchemyEventMemoryStoreParams(BaseModel):
    """
    Parameters for constructing a SQLAlchemyEventMemoryStore.

    Attributes:
        engine (AsyncEngine):
            Async SQLAlchemy engine. On SQLite it must enforce foreign
            keys on every connection, registered at engine creation
            (enable_sqlite_foreign_keys): the store relies on the
            link-table cascade and does not verify enforcement.
        metrics_factory (MetricsFactory | None):
            An instance of MetricsFactory for collecting usage metrics
            (default: None).
        purge_max_segments (int):
            Maximum number of segment rows purged per call. Their
            derivative links follow by cascade, so a call's transaction
            also scales with the deployment's links per segment; size
            the bound with that fan-out in mind (default: 10000).
        purge_max_partitions (int):
            Maximum number of queue entries a purge call processes.
            Entries cost round trips rather than row deletions --
            orders of magnitude more per unit than segment rows -- so
            they carry their own bound, sized so a full-entry call and
            a full-row call hold transactions of the same order of
            duration; a backlog of empty partitions cannot turn one
            bounded call into an unbounded transaction (default: 100).
    """

    engine: InstanceOf[AsyncEngine] = Field(
        ...,
        description=(
            "Async SQLAlchemy engine. On SQLite it must enforce foreign keys "
            "on every connection, registered at engine creation "
            "(enable_sqlite_foreign_keys): the store relies on the "
            "link-table cascade and does not verify enforcement"
        ),
    )
    metrics_factory: InstanceOf[MetricsFactory] | None = Field(
        None,
        description="An instance of MetricsFactory for collecting usage metrics",
    )
    purge_max_segments: int = Field(
        10_000,
        gt=0,
        description=(
            "Maximum number of segment rows purged per call. Their "
            "derivative links follow by cascade, so a call's transaction "
            "also scales with the deployment's links per segment; size "
            "the bound with that fan-out in mind"
        ),
    )
    purge_max_partitions: int = Field(
        100,
        gt=0,
        description=(
            "Maximum number of queue entries a purge call processes. "
            "Entries cost round trips rather than row deletions -- orders "
            "of magnitude more per unit than segment rows -- so they "
            "carry their own bound, sized so a full-entry call and a "
            "full-row call hold transactions of the same order of "
            "duration; a backlog of empty partitions cannot turn one "
            "bounded call into an unbounded transaction"
        ),
    )

    @field_validator("engine")
    @classmethod
    def _validate_engine(cls, engine: AsyncEngine) -> AsyncEngine:
        # A value judgment, not an argument-type check: pydantic turns
        # only ValueError/AssertionError into a ValidationError, so a
        # TypeError here would escape model construction raw.
        if engine.dialect.name not in ("postgresql", "sqlite"):
            raise ValueError(
                f"Engine uses the {engine.dialect.name} dialect, which the store "
                "does not support. Use PostgreSQL or SQLite."
            )
        engine_shares_one_connection = isinstance(engine.pool, StaticPool)
        if engine_shares_one_connection:
            raise ValueError(
                "Engine uses StaticPool, which shares one connection across "
                "sessions. Use a multi-connection pool instead."
            )
        db = engine.url.database
        if engine.dialect.name == "sqlite" and (db is None or db == ":memory:"):
            raise ValueError(
                "Engine uses ephemeral SQLite, where each connection gets a separate database. "
                "Use a file path instead."
            )
        if (
            engine.dialect.name == "sqlite"
            and sqlite3.sqlite_version_info < _MIN_SQLITE_VERSION
        ):
            minimum = ".".join(str(part) for part in _MIN_SQLITE_VERSION)
            raise ValueError(
                f"SQLite runtime {sqlite3.sqlite_version} lacks the RETURNING "
                f"support partition deletion depends on. Use SQLite {minimum} "
                "or newer."
            )
        return engine


class SQLAlchemyEventMemoryStore(EventMemoryStore):
    """SQLAlchemy-backed EventMemoryStore factory."""

    def __init__(self, params: SQLAlchemyEventMemoryStoreParams) -> None:
        """Initialize with an async SQLAlchemy engine."""
        self._engine = params.engine
        self._create_session = async_sessionmaker(self._engine, expire_on_commit=False)

        self._tracker = OperationTracker(
            params.metrics_factory,
            prefix="event_memory_store_sqlalchemy",
        )

        self._purge_max_segments = params.purge_max_segments
        self._purge_max_partitions = params.purge_max_partitions

        self._is_sqlite = self._engine.dialect.name == "sqlite"

    # Lifecycle

    @override
    async def startup(self) -> None:
        async with self._tracker("startup"), self._engine.begin() as connection:
            await connection.run_sync(BaseEventMemoryStore.metadata.create_all)

    @override
    async def shutdown(self) -> None:
        pass

    # Partition management

    @override
    async def create_partition(
        self,
        partition_key: str,
        config: EventMemoryStorePartitionConfig,
    ) -> None:
        validate_partition_key(partition_key)
        async with self._tracker("create_partition"):
            # Materialized before the insert so an unloadable codec config
            # fails without committing a registry row for a partition that
            # could never be opened.
            await self._load_payload_codec(config)
            attempts = 0
            while True:
                try:
                    await self._insert_partition_row(partition_key, uuid4(), config)
                except _RegistryInsertRejectedError as err:
                    logger.warning(
                        "Creating partition %r was rejected: %s; minting another "
                        "incarnation",
                        partition_key,
                        err,
                    )
                    attempts += 1
                    if attempts >= _MAX_MINT_ATTEMPTS:
                        raise EventMemoryStoreAttemptsExhaustedError(
                            f"Creating partition {partition_key!r} made no "
                            f"progress after {_MAX_MINT_ATTEMPTS} attempts"
                        ) from err
                    continue
                return

    async def _insert_partition_row(
        self,
        partition_key: str,
        incarnation: UUID,
        config: EventMemoryStorePartitionConfig,
    ) -> None:
        """Insert a registry row for a freshly minted incarnation.

        The registry's unique constraint rejects an incarnation colliding
        with a live one; the in-transaction queue re-check rejects one
        whose garbage is still awaiting purge, so data rows can never be
        adopted by (or reclaimed out from under) a new partition. The
        check runs after the insert so that a concurrent deletion moving a
        colliding row to the queue -- our insert waited on its uncommitted
        registry delete -- is already visible; after the check, no new
        queue entry for this incarnation can appear before we commit,
        because the only registry row carrying it is ours, uncommitted.
        The locking read sees latest-committed state even on dialects
        whose plain reads serve transaction-start snapshots.

        Raises:
            EventMemoryStorePartitionAlreadyExistsError:
                The partition key is taken; open or delete the existing
                partition instead.
            _RegistryInsertRejectedError:
                The insert cannot be kept, for a reason other than the
                key being taken.
        """
        try:
            async with self._create_session() as session, session.begin():
                await session.execute(
                    insert(PartitionRow).values(
                        partition_key=partition_key,
                        incarnation=incarnation,
                        payload_codec_config=encode_payload_codec_config(
                            config.payload_codec_config
                        ),
                    )
                )
                garbage_row = (
                    await session.execute(
                        select(PurgeQueueRow.incarnation)
                        .where(PurgeQueueRow.incarnation == incarnation)
                        .with_for_update(read=True)
                    )
                ).scalar_one_or_none()
                if garbage_row is not None:
                    raise _RegistryInsertRejectedError(
                        f"incarnation {incarnation} awaits purge"
                    )
        except IntegrityError as err:
            # If a committed row exists under this key, the key is taken.
            async with self._create_session() as session:
                partition_row = await SQLAlchemyEventMemoryStore._get_partition_row(
                    session, partition_key
                )
            if partition_row is not None:
                raise EventMemoryStorePartitionAlreadyExistsError(
                    partition_key
                ) from err
            raise _RegistryInsertRejectedError(
                f"the insert of incarnation {incarnation} failed and no row "
                "exists under the key"
            ) from err

    @override
    async def get_partition(
        self, partition_key: str
    ) -> SQLAlchemyEventMemoryStorePartition | None:
        validate_partition_key(partition_key)
        async with self._tracker("get_partition"):
            async with self._create_session() as session:
                partition_row = await SQLAlchemyEventMemoryStore._get_partition_row(
                    session, partition_key
                )
            if partition_row is None:
                return None

            return await self._partition_from_partition_row(partition_row)

    @override
    async def open_or_create_partition(
        self,
        partition_key: str,
        config: EventMemoryStorePartitionConfig,
    ) -> SQLAlchemyEventMemoryStorePartition:
        validate_partition_key(partition_key)
        async with self._tracker("open_or_create_partition"):
            return await self._open_or_create_partition(partition_key, config)

    async def _open_or_create_partition(
        self,
        partition_key: str,
        config: EventMemoryStorePartitionConfig,
    ) -> SQLAlchemyEventMemoryStorePartition:
        attempts = 0
        # Read-then-insert, retried: losing the insert race means a
        # concurrent creator won (reopen its row), and finding no row
        # after losing means a concurrent delete removed the winner --
        # every retry requires another actor to have changed the state
        # (or, vanishingly, a minted incarnation to have collided).
        while True:
            async with self._create_session() as session:
                partition_row = await SQLAlchemyEventMemoryStore._get_partition_row(
                    session, partition_key
                )

            if partition_row is not None:
                SQLAlchemyEventMemoryStore._raise_if_partition_config_mismatch(
                    partition_row, config
                )
                return await self._partition_from_partition_row(partition_row)

            # Materialized before the insert so an unloadable codec
            # config fails without committing a registry row for a
            # partition that could never be opened; the open path above
            # loads its codec from the committed row instead.
            payload_codec = await self._load_payload_codec(config)
            incarnation = uuid4()
            try:
                await self._insert_partition_row(partition_key, incarnation, config)
            except (
                EventMemoryStorePartitionAlreadyExistsError,
                _RegistryInsertRejectedError,
            ) as err:
                if isinstance(err, _RegistryInsertRejectedError):
                    logger.warning(
                        "Creating partition %r was rejected: %s; minting "
                        "another incarnation",
                        partition_key,
                        err,
                    )
                attempts += 1
                if attempts >= _MAX_MINT_ATTEMPTS:
                    raise EventMemoryStoreAttemptsExhaustedError(
                        f"Opening or creating partition {partition_key!r} "
                        f"made no progress after {_MAX_MINT_ATTEMPTS} "
                        f"attempts"
                    ) from err
                # Lost the creation race (reopen the winner's row next
                # iteration) or minted a colliding incarnation (mint a
                # fresh one).
                continue

            return SQLAlchemyEventMemoryStorePartition(
                partition_key=partition_key,
                incarnation=incarnation,
                create_session=self._create_session,
                is_sqlite=self._is_sqlite,
                config=config,
                payload_codec=payload_codec,
                tracker=self._tracker,
            )

    @override
    async def close_partition(
        self, event_memory_store_partition: EventMemoryStorePartition
    ) -> None:
        pass

    @override
    async def delete_partition(self, partition_key: str) -> None:
        # O(1) regardless of partition size: the exclusive row lock waits
        # out in-flight writers (which hold shared pins on the row), the
        # incarnation goes onto the purge queue, and the registry row is
        # deleted. Data rows become unreachable immediately -- every
        # operation resolves the registry first.
        validate_partition_key(partition_key)
        async with (
            self._tracker("delete_partition"),
            self._create_session() as session,
            session.begin(),
        ):
            if self._is_sqlite:
                # Same primitive as _lock_partition_for_write: the row
                # UPDATE opens the write transaction so racing deletions
                # serialize instead of both enqueueing the incarnation,
                # and no matched row is the idempotent no-op case.
                # RETURNING resolves the incarnation in the same round
                # trip; the locking select below is PostgreSQL's path
                # (SQLite drops its locking clause anyway).
                pinned = await (await session.connection()).execute(
                    update(PartitionRow)
                    .where(PartitionRow.partition_key == partition_key)
                    .values(partition_key=partition_key)
                    .returning(PartitionRow.incarnation)
                )
                incarnation = pinned.scalar_one_or_none()
            else:
                incarnation = (
                    await session.execute(
                        select(PartitionRow.incarnation)
                        .where(PartitionRow.partition_key == partition_key)
                        .with_for_update()
                    )
                ).scalar_one_or_none()
            if incarnation is None:
                return

            await session.execute(
                insert(PurgeQueueRow).values(
                    incarnation=incarnation,
                    partition_key=partition_key,
                    # The database clock: transaction start on PostgreSQL,
                    # and one deletion per transaction, so one stamp per
                    # entry.
                    enqueued_at=func.now(),
                )
            )
            await session.execute(
                delete(PartitionRow).where(PartitionRow.partition_key == partition_key)
            )

    @override
    async def purge_deleted_partitions(self) -> bool:
        # Reclaim dead incarnations oldest-first, within the per-call bounds.
        #
        # One transaction per call: reclaim up to the configured bounds and
        # commit, or nothing. That is what makes a call that fails on
        # contention safe to repeat, as the ABC promises: a raise rolls the
        # whole call back, cursor included. Entries are claimed one at a
        # time, so only the claiming call touches a dead incarnation's rows;
        # on SQLite, which drops locking clauses, the claim is a write, so
        # purgers serialize at the claim and each reads the cursor its
        # predecessor committed. Full rationale:
        # design/event_memory_store_shared_tables.md.
        remaining = self._purge_max_segments
        entries = 0
        # Pure Core DML on an engine connection: unlike Session.execute,
        # AsyncConnection.execute is typed CursorResult, whose rowcount
        # the batch loop needs.
        async with (
            self._tracker("purge_deleted_partitions"),
            self._engine.begin() as connection,
        ):
            while True:
                if self._is_sqlite:
                    # Same primitive as _lock_partition_for_write: the row
                    # UPDATE opens the write transaction, so a racing
                    # purger waits here and then reads this call's
                    # committed cursor. RETURNING resolves the entry in the
                    # same round trip.
                    oldest = (
                        select(PurgeQueueRow.incarnation)
                        .order_by(PurgeQueueRow.enqueued_at)
                        .limit(1)
                        .scalar_subquery()
                    )
                    claim = (
                        update(PurgeQueueRow)
                        .where(PurgeQueueRow.incarnation == oldest)
                        .values(incarnation=PurgeQueueRow.incarnation)
                        .returning(
                            PurgeQueueRow.incarnation,
                            PurgeQueueRow.purged_through,
                            PurgeQueueRow.events_purged_through,
                        )
                    )
                else:
                    # Skips only OTHER transactions' locks; entries this
                    # call has already claimed cannot come back, each
                    # being deleted before the next claim.
                    claim = (
                        select(
                            PurgeQueueRow.incarnation,
                            PurgeQueueRow.purged_through,
                            PurgeQueueRow.events_purged_through,
                        )
                        .order_by(PurgeQueueRow.enqueued_at)
                        .limit(1)
                        .with_for_update(skip_locked=True)
                    )
                entry = (await connection.execute(claim)).one_or_none()
                if entry is None:
                    return False
                incarnation, purged_through, events_purged_through = entry

                # The batch continues after the cursor, so it never reads
                # rows an earlier batch deleted: PostgreSQL keeps them in
                # the table and its indexes until vacuum, and a batch
                # starting from the incarnation's first key would step over
                # all of them.
                after_cursor = (
                    [] if purged_through is None else [SegmentRow.uuid > purged_through]
                )
                batch = (
                    select(SegmentRow.uuid)
                    .where(SegmentRow.incarnation == incarnation, *after_cursor)
                    .order_by(SegmentRow.uuid)
                    .limit(remaining)
                    .subquery()
                )
                last = (
                    await connection.execute(
                        select(batch.c.uuid).order_by(batch.c.uuid.desc()).limit(1)
                    )
                ).scalar_one_or_none()
                deleted = 0
                if last is not None:
                    # The link-table cascade follows the deleted segments.
                    deleted = (
                        await connection.execute(
                            delete(SegmentRow).where(
                                SegmentRow.incarnation == incarnation,
                                *after_cursor,
                                SegmentRow.uuid <= last,
                            )
                        )
                    ).rowcount
                if deleted >= remaining:
                    # The bound was consumed; this incarnation may have more
                    # rows, so record how far it got and leave its queue
                    # entry for the next call.
                    await connection.execute(
                        update(PurgeQueueRow)
                        .where(PurgeQueueRow.incarnation == incarnation)
                        .values(purged_through=last)
                    )
                    return True

                remaining -= deleted
                # Event rows are reclaimed after their segments, so the
                # cascade from an event row never runs on the budgeted path.
                # They draw on the same budget, continue after their own
                # cursor for the reason the segments do, and a full batch
                # records how far it got and leaves the entry for the next
                # call.
                after_event_cursor = (
                    []
                    if events_purged_through is None
                    else [EventRow.uuid > events_purged_through]
                )
                event_batch = (
                    select(EventRow.uuid)
                    .where(EventRow.incarnation == incarnation, *after_event_cursor)
                    .order_by(EventRow.uuid)
                    .limit(remaining)
                    .subquery()
                )
                last_event = (
                    await connection.execute(
                        select(event_batch.c.uuid)
                        .order_by(event_batch.c.uuid.desc())
                        .limit(1)
                    )
                ).scalar_one_or_none()
                purged_events = 0
                if last_event is not None:
                    purged_events = (
                        await connection.execute(
                            delete(EventRow).where(
                                EventRow.incarnation == incarnation,
                                *after_event_cursor,
                                EventRow.uuid <= last_event,
                            )
                        )
                    ).rowcount
                if purged_events >= remaining:
                    await connection.execute(
                        update(PurgeQueueRow)
                        .where(PurgeQueueRow.incarnation == incarnation)
                        .values(events_purged_through=last_event)
                    )
                    return True
                remaining -= purged_events
                await connection.execute(
                    delete(PurgeQueueRow).where(
                        PurgeQueueRow.incarnation == incarnation
                    )
                )
                # Entries cost round trips, not row deletions, so they
                # carry their own bound: a backlog of empty partitions
                # consumes no segment budget yet must not turn one
                # bounded call into an unbounded transaction.
                entries += 1
                if entries >= self._purge_max_partitions:
                    return True

    # Helpers

    async def _load_payload_codec(
        self,
        config: EventMemoryStorePartitionConfig,
    ) -> PayloadCodec:
        """Materialize a live payload codec for a partition config."""
        match config.payload_codec_config:
            case PlaintextPayloadCodecConfig():
                return PlaintextPayloadCodec()
            case _:
                raise NotImplementedError(
                    f"Unsupported payload codec config: "
                    f"{type(config.payload_codec_config).__name__}"
                )

    async def _partition_from_partition_row(
        self,
        partition_row: PartitionRow,
    ) -> SQLAlchemyEventMemoryStorePartition:
        """Materialize a partition handle from a registry row."""
        config = EventMemoryStorePartitionConfig(
            payload_codec_config=decode_payload_codec_config(
                partition_row.payload_codec_config
            )
        )
        payload_codec = await self._load_payload_codec(config)
        return SQLAlchemyEventMemoryStorePartition(
            partition_key=partition_row.partition_key,
            incarnation=partition_row.incarnation,
            create_session=self._create_session,
            is_sqlite=self._is_sqlite,
            config=config,
            payload_codec=payload_codec,
            tracker=self._tracker,
        )

    @staticmethod
    async def _get_partition_row(
        session: AsyncSession,
        partition_key: str,
    ) -> PartitionRow | None:
        """Fetch a partition row by key."""
        return (
            await session.execute(
                select(PartitionRow).where(PartitionRow.partition_key == partition_key)
            )
        ).scalar_one_or_none()

    @staticmethod
    def _raise_if_partition_config_mismatch(
        partition_row: PartitionRow,
        config: EventMemoryStorePartitionConfig,
    ) -> None:
        """Raise if an existing partition row does not match the requested config."""
        existing_config = EventMemoryStorePartitionConfig(
            payload_codec_config=decode_payload_codec_config(
                partition_row.payload_codec_config
            )
        )
        if existing_config != config:
            raise EventMemoryStorePartitionConfigMismatchError(
                partition_row.partition_key,
                existing_config,
                config,
            )
