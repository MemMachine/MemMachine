"""SQLAlchemy implementation of the EventMemoryStore interface."""

import json
import logging
import sqlite3
from collections.abc import Callable, Iterable, Mapping, Sequence
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
    Boolean,
    ColumnCollection,
    ColumnElement,
    DateTime,
    ForeignKeyConstraint,
    Index,
    Integer,
    Interval,
    LargeBinary,
    Select,
    String,
    Tuple,
    Uuid,
    and_,
    bindparam,
    case,
    delete,
    false,
    func,
    insert,
    literal,
    or_,
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
    AsyncConnection,
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
)
from sqlalchemy.orm import (
    DeclarativeBase,
    InstrumentedAttribute,
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
from memmachine_server.common.sql import write_connection, write_session
from memmachine_server.common.utils import utc_offset_seconds
from memmachine_server.episodic_memory.event_memory.data_types import (
    Derivative,
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
    PendingDerivatives,
    Tombstone,
)
from memmachine_server.episodic_memory.event_memory.event_memory_store.event_memory_store import (
    EventMemoryStore,
    EventMemoryStorePartition,
    PendingDerivativesClaim,
    TombstoneClaim,
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

# Failed claims in a row a pending derivatives record or a tombstone gets
# before it is no longer claimed.
_MAX_CLAIM_ATTEMPTS_WITHOUT_PROGRESS = 10

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
    Deleting the row cascades to the event's segments, their derivative
    links, and its pending derivatives.
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


class PendingDerivativesRow(BaseEventMemoryStore):
    """One event's derivatives awaiting their derived index.

    Written with the event; its writer deletes it once the index has the
    derivatives, and a claim takes it when the writer does not. Deleted
    with its event, so a forgotten event leaves none behind.
    """

    __tablename__ = "event_memory_store_pd"

    uuid: MappedColumn[UUID] = mapped_column(Uuid, primary_key=True)
    incarnation: MappedColumn[UUID] = mapped_column(Uuid, nullable=False)
    event_uuid: MappedColumn[UUID] = mapped_column(Uuid, nullable=False)
    # The event's properties, stored as a segment row stores its own.
    properties: MappedColumn[dict[str, JsonValue]] = mapped_column(
        _JSON_AUTO, nullable=False, default=dict
    )
    created_at: MappedColumn[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )
    # Claims since the record was last claimed without failing, the open one
    # included.
    attempts_without_progress: MappedColumn[int] = mapped_column(
        Integer, nullable=False, default=0
    )
    # When a claim on the record last failed.
    last_failed_at: MappedColumn[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    # When the open claim's lease ends; None when no claim is open.
    claimed_until: MappedColumn[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    # Incremented by each claim; a claim ends only itself.
    claim_generation: MappedColumn[int] = mapped_column(
        Integer, nullable=False, default=0
    )

    __table_args__ = (
        ForeignKeyConstraint(
            ["incarnation", "event_uuid"],
            [
                "event_memory_store_ev.incarnation",
                "event_memory_store_ev.uuid",
            ],
            ondelete="CASCADE",
        ),
        # Serves the forget's delete and the cascade from the event row.
        Index("event_memory_store_pd__in_ev", "incarnation", "event_uuid"),
        Index("event_memory_store_pd__ca", "created_at"),
    )


class PendingDerivativeRow(BaseEventMemoryStore):
    """One derivative of a pending derivatives record."""

    __tablename__ = "event_memory_store_pd_dv"

    pending_uuid: MappedColumn[UUID] = mapped_column(Uuid, primary_key=True)
    uuid: MappedColumn[UUID] = mapped_column(Uuid, primary_key=True)
    segment_uuid: MappedColumn[UUID] = mapped_column(Uuid, nullable=False)
    timestamp: MappedColumn[datetime] = mapped_column(UtcInstant, nullable=False)
    timestamp_timezone_offset: MappedColumn[int] = mapped_column(
        Integer, nullable=False, default=0
    )
    source_id: MappedColumn[str | None] = mapped_column(String(255), nullable=True)
    block: MappedColumn[bytes] = mapped_column(LargeBinary, nullable=False)

    __table_args__ = (
        ForeignKeyConstraint(
            ["pending_uuid"],
            ["event_memory_store_pd.uuid"],
            ondelete="CASCADE",
        ),
    )


class TombstoneRow(BaseEventMemoryStore):
    """A forget's tombstone: the derivatives it unlinked.

    Their derived index entries are deleted once at once and once more
    after the store's retention. It holds only what the deletes read.
    """

    __tablename__ = "event_memory_store_tb"

    uuid: MappedColumn[UUID] = mapped_column(Uuid, primary_key=True)
    incarnation: MappedColumn[UUID] = mapped_column(Uuid, nullable=False)
    enqueued_at: MappedColumn[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )
    # Whether the first delete is done, so the tombstone awaits its second.
    second_delete: MappedColumn[bool] = mapped_column(
        Boolean, nullable=False, default=False
    )
    # Claims since the tombstone last made progress, the open one included.
    attempts_without_progress: MappedColumn[int] = mapped_column(
        Integer, nullable=False, default=0
    )
    # When a claim on the tombstone last failed.
    last_failed_at: MappedColumn[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    # When the open claim's lease ends; None when no claim is open.
    claimed_until: MappedColumn[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    # Incremented by each claim; a claim ends only itself.
    claim_generation: MappedColumn[int] = mapped_column(
        Integer, nullable=False, default=0
    )

    __table_args__ = (
        Index("event_memory_store_tb__ea", "enqueued_at"),
        # Serves the purge's batches of a dead incarnation's tombstones.
        Index("event_memory_store_tb__in_uu", "incarnation", "uuid"),
    )


class TombstoneDerivativeRow(BaseEventMemoryStore):
    """One derivative a tombstone names."""

    __tablename__ = "event_memory_store_tb_dv"

    tombstone_uuid: MappedColumn[UUID] = mapped_column(Uuid, primary_key=True)
    uuid: MappedColumn[UUID] = mapped_column(Uuid, primary_key=True)

    __table_args__ = (
        ForeignKeyConstraint(
            ["tombstone_uuid"],
            ["event_memory_store_tb.uuid"],
            ondelete="CASCADE",
        ),
    )


class PurgeQueueRow(BaseEventMemoryStore):
    """The purge queue: one row per dead partition incarnation.

    Claimed oldest-first by the enqueue stamp, which is the database clock,
    so entries from every server order on one clock; entries stamped in the
    same tick are unordered among themselves. The incarnation identifies
    the rows to reclaim. The purge reclaims segments, then events, then
    tombstones, each continuing after its own cursor: every row of the
    incarnation keyed at or below the cursor is deleted, and a cursor is
    NULL until its first full batch. A batch that ends exactly on the last
    row of one table leaves that table's cursor naming it, which says
    nothing about the next table, hence a cursor per table. An entry holds
    only what the purge reads.
    """

    __tablename__ = "event_memory_store_gc"

    incarnation: MappedColumn[UUID] = mapped_column(Uuid, primary_key=True)
    enqueued_at: MappedColumn[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )
    purged_through: MappedColumn[UUID | None] = mapped_column(Uuid, nullable=True)
    events_purged_through: MappedColumn[UUID | None] = mapped_column(
        Uuid, nullable=True
    )
    tombstones_purged_through: MappedColumn[UUID | None] = mapped_column(
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
        self, session: AsyncSession, *, pin: bool = False
    ) -> None:
        """Raise if this handle's incarnation is no longer registered.

        With `pin`, the row is read under a shared lock that blocks
        deletion, which takes the exclusive row lock, for the rest of the
        transaction. SQLite drops the locking clause; there a write
        transaction holds the database's write lock from `BEGIN`, which
        deletion waits on just the same. Reads call this without the lock,
        and only when their data statement returned no rows: it tells an
        empty partition from a stale handle.
        """
        query = self._registry_row_query()
        if pin:
            query = query.with_for_update(read=True)
        row = (await session.execute(query)).scalar_one_or_none()
        if row is None:
            raise EventMemoryStorePartitionHandleStaleError(self._partition_key)

    @override
    async def require_current(self) -> None:
        async with self._create_session() as session:
            await self._ensure_partition_live(session)

    # Registration

    @override
    async def add_events(
        self,
        segments_by_event: Mapping[UUID, Iterable[Segment]],
        pending_derivatives: Iterable[PendingDerivatives],
    ) -> None:
        segments_by_event = {
            event_uuid: list(segments)
            for event_uuid, segments in segments_by_event.items()
        }
        pending_derivatives = list(pending_derivatives)
        SQLAlchemyEventMemoryStorePartition._validate_events(
            segments_by_event, pending_derivatives
        )
        if not segments_by_event:
            return

        async with (
            self._tracker("add_events"),
            write_session(self._create_session) as session,
        ):
            await self._ensure_partition_live(session, pin=True)
            stored = await self._insert_event_rows(session, segments_by_event.keys())
            already_stored = segments_by_event.keys() - stored
            if already_stored:
                raise EventMemoryStoreEventAlreadyStoredError(already_stored)
            await self._insert_segments(
                session,
                (
                    segment
                    for segments in segments_by_event.values()
                    for segment in segments
                ),
            )
            await self._insert_derivative_links(
                session,
                (
                    derivative
                    for pending in pending_derivatives
                    for derivative in pending.derivatives
                ),
            )
            await self._insert_pending_derivatives(session, pending_derivatives)

    @staticmethod
    def _validate_events(
        segments_by_event: Mapping[UUID, list[Segment]],
        pending_derivatives: list[PendingDerivatives],
    ) -> None:
        """Raise ValueError unless each segment and derivative is under its own event."""
        segment_uuids_by_event: dict[UUID, set[UUID]] = {}
        for event_uuid, segments in segments_by_event.items():
            for segment in segments:
                if segment.event_uuid != event_uuid:
                    raise ValueError(
                        f"Segment {segment.uuid} names event {segment.event_uuid}, "
                        f"listed under {event_uuid}"
                    )
            segment_uuids_by_event[event_uuid] = {segment.uuid for segment in segments}
        for pending in pending_derivatives:
            segment_uuids = segment_uuids_by_event.get(pending.event_uuid)
            if segment_uuids is None:
                raise ValueError(
                    f"Pending derivatives {pending.uuid} name event "
                    f"{pending.event_uuid}, which the batch does not add"
                )
            for derivative in pending.derivatives:
                if derivative.segment_uuid not in segment_uuids:
                    raise ValueError(
                        f"Derivative {derivative.uuid} names segment "
                        f"{derivative.segment_uuid}, which event "
                        f"{pending.event_uuid} does not have in the batch"
                    )

    async def _insert_event_rows(
        self, session: AsyncSession, event_uuids: Iterable[UUID]
    ) -> set[UUID]:
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
        return set((await session.execute(statement)).scalars())

    async def _insert_segments(
        self,
        session: AsyncSession,
        segments: Iterable[Segment],
    ) -> None:
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
            await session.execute(insert(SegmentRow), segment_row_values)

    async def _insert_derivative_links(
        self,
        session: AsyncSession,
        derivatives: Iterable[Derivative],
    ) -> None:
        """Insert a link row per derivative, to the segment it names."""
        derivative_row_values = [
            {
                "uuid": derivative.uuid,
                "incarnation": self._incarnation,
                "segment_uuid": derivative.segment_uuid,
            }
            for derivative in derivatives
        ]
        if derivative_row_values:
            await session.execute(insert(DerivativeLinkRow), derivative_row_values)

    async def _insert_pending_derivatives(
        self,
        session: AsyncSession,
        pending_derivatives: Iterable[PendingDerivatives],
    ) -> None:
        """Insert a pending derivatives row, and a row per derivative, for each record."""
        pending_derivatives = list(pending_derivatives)
        if not pending_derivatives:
            return
        # One multi-row statement, so every row carries the database clock.
        await session.execute(
            insert(PendingDerivativesRow).values(
                [
                    {
                        "uuid": pending.uuid,
                        "incarnation": self._incarnation,
                        "event_uuid": pending.event_uuid,
                        "properties": encode_properties(pending.properties),
                        "created_at": func.now(),
                    }
                    for pending in pending_derivatives
                ]
            )
        )
        derivative_row_values = [
            {
                "pending_uuid": pending.uuid,
                "uuid": derivative.uuid,
                "segment_uuid": derivative.segment_uuid,
                "timestamp": derivative.timestamp,
                "timestamp_timezone_offset": utc_offset_seconds(derivative.timestamp),
                "source_id": derivative.source_id,
                "block": self._payload_codec.encode(
                    json.dumps(encode_block(derivative.block)).encode("utf-8")
                ),
            }
            for pending in pending_derivatives
            for derivative in pending.derivatives
        ]
        if derivative_row_values:
            await session.execute(insert(PendingDerivativeRow), derivative_row_values)

    @override
    async def delete_pending_derivatives(
        self,
        pending_derivatives_uuids: Iterable[UUID],
    ) -> None:
        pending_derivatives_uuids = set(pending_derivatives_uuids)
        if not pending_derivatives_uuids:
            return

        async with (
            self._tracker("delete_pending_derivatives"),
            write_session(self._create_session) as session,
        ):
            await self._ensure_partition_live(session, pin=True)
            # The cascade follows to the records' derivatives.
            await session.execute(
                delete(PendingDerivativesRow).where(
                    PendingDerivativesRow.incarnation == self._incarnation,
                    PendingDerivativesRow.uuid.in_(pending_derivatives_uuids),
                )
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
            # Served by the primary key, which leads on the same two columns
            # this filters: the mapping is the derivative row itself.
            query = select(
                DerivativeLinkRow.uuid, DerivativeLinkRow.segment_uuid
            ).where(
                DerivativeLinkRow.incarnation == self._incarnation,
                DerivativeLinkRow.uuid.in_(derivative_uuids),
                self._registry_row_query().exists(),
            )
            rows = (await session.execute(query)).all()
            if not rows:
                await self._ensure_partition_live(session)

        return {row.uuid: row.segment_uuid for row in rows}

    # Forgetting

    @override
    async def forget_events(
        self,
        event_uuids: Iterable[UUID],
    ) -> Tombstone | None:
        event_uuids = set(event_uuids)
        if not event_uuids:
            return None

        async with (
            self._tracker("forget_events"),
            write_session(self._create_session) as session,
        ):
            await self._ensure_partition_live(session, pin=True)
            # Every later statement acts on the events this one returned:
            # each statement reads a snapshot of its own on PostgreSQL, so
            # an event committed after this one would be deleted below with
            # links nobody read. The order of the locks keeps concurrent
            # forgets with overlapping sets from deadlocking; SQLite drops
            # the locking clause and serializes write transactions instead.
            held_event_uuids = list(
                (
                    await session.execute(
                        select(EventRow.uuid)
                        .where(
                            EventRow.incarnation == self._incarnation,
                            EventRow.uuid.in_(event_uuids),
                        )
                        .order_by(EventRow.uuid)
                        .with_for_update()
                    )
                ).scalars()
            )
            if not held_event_uuids:
                return None
            if not self._is_sqlite:
                # The segments' locks too, in order: the cascade alone would
                # take them in table order.
                await session.execute(
                    select(SegmentRow.uuid)
                    .where(
                        SegmentRow.incarnation == self._incarnation,
                        SegmentRow.event_uuid.in_(held_event_uuids),
                    )
                    .order_by(SegmentRow.uuid)
                    .with_for_update()
                )

            derivative_uuids = list(
                (
                    await session.execute(
                        select(DerivativeLinkRow.uuid)
                        .join(
                            SegmentRow,
                            (SegmentRow.incarnation == DerivativeLinkRow.incarnation)
                            & (SegmentRow.uuid == DerivativeLinkRow.segment_uuid),
                        )
                        .where(
                            SegmentRow.incarnation == self._incarnation,
                            SegmentRow.event_uuid.in_(held_event_uuids),
                        )
                    )
                ).scalars()
            )

            # Before the tombstone's time is taken: a claim on these
            # pending derivatives holds its row, so this delete waits for a
            # claim in progress, and a claim either committed before the
            # time below or finds its row gone.
            await session.execute(
                delete(PendingDerivativesRow).where(
                    PendingDerivativesRow.incarnation == self._incarnation,
                    PendingDerivativesRow.event_uuid.in_(held_event_uuids),
                )
            )

            tombstone = None
            if derivative_uuids:
                tombstone = Tombstone(uuid=uuid4(), derivative_uuids=derivative_uuids)
                await session.execute(
                    insert(TombstoneRow).values(
                        uuid=tombstone.uuid,
                        incarnation=self._incarnation,
                        # The time of this statement, after the locks: the
                        # transaction's start, which `now()` gives on
                        # PostgreSQL, can precede a write the locks waited
                        # for.
                        enqueued_at=(
                            func.now()
                            if self._is_sqlite
                            else func.statement_timestamp()
                        ),
                    )
                )
                await session.execute(
                    insert(TombstoneDerivativeRow),
                    [
                        {"tombstone_uuid": tombstone.uuid, "uuid": derivative_uuid}
                        for derivative_uuid in derivative_uuids
                    ],
                )

            # The cascade follows to segments, and through them to links.
            await session.execute(
                delete(EventRow).where(
                    EventRow.incarnation == self._incarnation,
                    EventRow.uuid.in_(held_event_uuids),
                )
            )
        return tombstone

    @override
    async def record_tombstone_first_delete(self, tombstone_uuid: UUID) -> None:
        async with (
            self._tracker("record_tombstone_first_delete"),
            write_session(self._create_session) as session,
        ):
            await self._ensure_partition_live(session, pin=True)
            await session.execute(
                update(TombstoneRow)
                .where(
                    TombstoneRow.uuid == tombstone_uuid,
                    TombstoneRow.incarnation == self._incarnation,
                    TombstoneRow.second_delete.is_(False),
                )
                .values(
                    second_delete=True,
                    attempts_without_progress=0,
                    claimed_until=None,
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


class SQLAlchemyEventMemoryStoreParams(BaseModel):
    """
    Parameters for constructing a SQLAlchemyEventMemoryStore.

    Attributes:
        engine (AsyncEngine):
            Async SQLAlchemy engine. On SQLite it must enforce foreign
            keys on every connection, registered at engine creation
            (enable_sqlite_foreign_keys): the store relies on the
            link-table cascade and does not verify enforcement. It must
            also begin its transactions explicitly, set up at engine
            creation (enable_sqlite_explicit_begin): the store's write
            transactions take SQLite's write lock at BEGIN.
        metrics_factory (MetricsFactory | None):
            An instance of MetricsFactory for collecting usage metrics
            (default: None).
        purge_max_segments (int):
            Maximum number of rows purged per call: segments, events, and
            the derivatives tombstones name. Derivative links follow
            segments, and pending derivatives follow events, by cascade,
            so a call's transaction also scales with the deployment's
            derivatives per segment; size the bound with that fan-out in
            mind. A tombstone naming more derivatives than the bound is
            purged alone (default: 10000).
        purge_max_partitions (int):
            Maximum number of queue entries a purge call processes.
            Entries cost round trips rather than row deletions --
            orders of magnitude more per unit than segment rows -- so
            they carry their own bound, sized so a full-entry call and
            a full-row call hold transactions of the same order of
            duration; a backlog of empty partitions cannot turn one
            bounded call into an unbounded transaction (default: 100).
        tombstone_retention_seconds (int):
            Seconds after a forget before its tombstone's second delete is
            due, on the database clock. It must exceed, by orders of
            magnitude, the longest a write of a forgotten derivative to its
            index can land after the forget (default: 86400).
        tombstone_lease_seconds (int):
            Seconds a claim holds its tombstone, on the database clock; it
            should exceed the longest delete (default: 300).
        base_claim_retry_backoff_seconds (int):
            Seconds before a record or tombstone is claimed again after a
            failed claim, doubled for each further failed claim in a row
            (default: 30).
        max_claim_retry_backoff_seconds (int):
            The longest the claim retry backoff grows to (default: 3600).
    """

    engine: InstanceOf[AsyncEngine] = Field(
        ...,
        description=(
            "Async SQLAlchemy engine. On SQLite it must enforce foreign keys "
            "on every connection, registered at engine creation "
            "(enable_sqlite_foreign_keys): the store relies on the "
            "link-table cascade and does not verify enforcement. It must "
            "also begin its transactions explicitly, set up at engine "
            "creation (enable_sqlite_explicit_begin): the store's write "
            "transactions take SQLite's write lock at BEGIN"
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
            "Maximum number of rows purged per call: segments, events, and "
            "the derivatives tombstones name. Derivative links follow "
            "segments, and pending derivatives follow events, by cascade, "
            "so a call's transaction also scales with the deployment's "
            "derivatives per segment; size the bound with that fan-out in "
            "mind. A tombstone naming more derivatives than the bound is "
            "purged alone"
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
    tombstone_retention_seconds: int = Field(
        86_400,
        gt=0,
        description=(
            "Seconds after a forget before its tombstone's second delete is "
            "due, on the database clock. It must exceed, by orders of "
            "magnitude, the longest a write of a forgotten derivative to its "
            "index can land after the forget"
        ),
    )
    tombstone_lease_seconds: int = Field(
        300,
        gt=0,
        description=(
            "Seconds a claim holds its tombstone, on the database clock; it "
            "should exceed the longest delete"
        ),
    )
    base_claim_retry_backoff_seconds: int = Field(
        30,
        gt=0,
        description=(
            "Seconds before a record or tombstone is claimed again after a "
            "failed claim, doubled for each further failed claim in a row"
        ),
    )
    max_claim_retry_backoff_seconds: int = Field(
        3600, gt=0, description="The longest the claim retry backoff grows to"
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
        self._tombstone_retention_seconds = params.tombstone_retention_seconds
        self._tombstone_lease_seconds = params.tombstone_lease_seconds
        self._base_claim_retry_backoff_seconds = params.base_claim_retry_backoff_seconds
        self._max_claim_retry_backoff_seconds = params.max_claim_retry_backoff_seconds

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
            async with write_session(self._create_session) as session:
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
    async def open_partition(
        self, partition_key: str
    ) -> SQLAlchemyEventMemoryStorePartition | None:
        validate_partition_key(partition_key)
        async with self._tracker("open_partition"):
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
            write_session(self._create_session) as session,
        ):
            # Racing deletions serialize on the exclusive row lock, or on
            # SQLite, which drops locking clauses, on the write lock the
            # transaction holds from BEGIN; no row is the idempotent no-op
            # case.
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
        # on SQLite, which drops locking clauses, the call holds the write
        # lock from BEGIN, so purgers serialize and each reads the cursor
        # its predecessor committed. Full rationale:
        # design/event_memory_store_shared_tables.md.
        remaining = self._purge_max_segments
        entries = 0
        # Pure Core DML on an engine connection: unlike Session.execute,
        # AsyncConnection.execute is typed CursorResult, whose rowcount
        # the batch loop needs.
        async with (
            self._tracker("purge_deleted_partitions"),
            write_connection(self._engine) as connection,
        ):
            while True:
                # Skips only OTHER transactions' locks; entries this call has
                # already claimed cannot come back, each being deleted before
                # the next claim. SQLite drops the locking clause; there the
                # write lock the call holds keeps other purgers out.
                claim = (
                    select(
                        PurgeQueueRow.incarnation,
                        PurgeQueueRow.purged_through,
                        PurgeQueueRow.events_purged_through,
                        PurgeQueueRow.tombstones_purged_through,
                    )
                    .order_by(PurgeQueueRow.enqueued_at)
                    .limit(1)
                    .with_for_update(skip_locked=True)
                )
                entry = (await connection.execute(claim)).one_or_none()
                if entry is None:
                    return False
                incarnation = entry.incarnation
                cursors = {
                    "purged_through": entry.purged_through,
                    "events_purged_through": entry.events_purged_through,
                    "tombstones_purged_through": entry.tombstones_purged_through,
                }

                # Segments go first, so the cascade from an event row never
                # runs on the budgeted path, and events take their pending
                # derivatives with them. Each table draws on the same
                # budget, and a full batch records how far it got and
                # leaves the entry for the next call.
                for table, cursor in (
                    (SegmentRow, "purged_through"),
                    (EventRow, "events_purged_through"),
                    (TombstoneRow, "tombstones_purged_through"),
                ):
                    if table is TombstoneRow:
                        (
                            deleted,
                            last,
                        ) = await SQLAlchemyEventMemoryStore._purge_tombstone_batch(
                            connection,
                            incarnation,
                            after=cursors[cursor],
                            limit=remaining,
                        )
                    else:
                        deleted, last = await SQLAlchemyEventMemoryStore._purge_batch(
                            connection,
                            table,
                            incarnation,
                            after=cursors[cursor],
                            limit=remaining,
                        )
                    if deleted >= remaining:
                        await connection.execute(
                            update(PurgeQueueRow)
                            .where(PurgeQueueRow.incarnation == incarnation)
                            .values({cursor: last})
                        )
                        return True
                    remaining -= deleted

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

    @staticmethod
    async def _purge_batch(
        connection: AsyncConnection,
        table: type[SegmentRow | EventRow],
        incarnation: UUID,
        *,
        after: UUID | None,
        limit: int,
    ) -> tuple[int, UUID | None]:
        """Delete up to `limit` of a dead incarnation's rows of `table` in key order after `after`.

        Returns how many were deleted and the last key deleted. The batch
        continues after the cursor, so it never reads rows an earlier batch
        deleted: PostgreSQL keeps them in the table and its indexes until
        vacuum, and a batch starting from the incarnation's first key would
        step over all of them.
        """
        after_cursor = [] if after is None else [table.uuid > after]
        batch = (
            select(table.uuid)
            .where(table.incarnation == incarnation, *after_cursor)
            .order_by(table.uuid)
            .limit(limit)
            .subquery()
        )
        last = (
            await connection.execute(
                select(batch.c.uuid).order_by(batch.c.uuid.desc()).limit(1)
            )
        ).scalar_one_or_none()
        if last is None:
            return 0, None
        deleted = (
            await connection.execute(
                delete(table).where(
                    table.incarnation == incarnation,
                    *after_cursor,
                    table.uuid <= last,
                )
            )
        ).rowcount
        return deleted, last

    @staticmethod
    async def _purge_tombstone_batch(
        connection: AsyncConnection,
        incarnation: UUID,
        *,
        after: UUID | None,
        limit: int,
    ) -> tuple[int, UUID | None]:
        """Delete a dead incarnation's tombstones in key order after `after`, naming up to `limit` derivatives.

        A tombstone counts as the derivatives it names, which follow it by
        cascade and number as many as its forget chose. Returns how many
        derivatives the deleted tombstones named, or `limit` if a
        tombstone that did not fit remains, and the last key deleted. A
        first tombstone naming more than `limit` derivatives is deleted
        alone, in a transaction no larger than its forget's. Every
        tombstone names at least one derivative.
        """
        after_cursor = [] if after is None else [TombstoneRow.uuid > after]
        # One more than the budget shows whether the batch ends inside the
        # last tombstone it reaches.
        window = (
            select(TombstoneRow.uuid)
            .join(
                TombstoneDerivativeRow,
                TombstoneDerivativeRow.tombstone_uuid == TombstoneRow.uuid,
            )
            .where(TombstoneRow.incarnation == incarnation, *after_cursor)
            .order_by(TombstoneRow.uuid)
            .limit(limit + 1)
            .subquery()
        )
        named_counts = (
            await connection.execute(
                select(window.c.uuid, func.count())
                .group_by(window.c.uuid)
                .order_by(window.c.uuid)
            )
        ).all()
        if not named_counts:
            return 0, None
        named = sum(count for _, count in named_counts)
        if named <= limit:
            last = named_counts[-1].uuid
        else:
            # The last tombstone reached does not fit.
            named = limit
            last = named_counts[-2 if len(named_counts) > 1 else 0].uuid
        await connection.execute(
            delete(TombstoneRow).where(
                TombstoneRow.incarnation == incarnation,
                *after_cursor,
                TombstoneRow.uuid <= last,
            )
        )
        return named, last

    # Claims

    @override
    async def claim_pending_derivatives(
        self,
        *,
        min_age_seconds: int,
        lease_seconds: int,
    ) -> PendingDerivativesClaim | None:
        oldest_due = (
            select(PendingDerivativesRow.uuid)
            .where(
                self._has_elapsed(
                    min_age_seconds, since=PendingDerivativesRow.created_at
                ),
                self._is_claimable(PendingDerivativesRow),
                select(PartitionRow.partition_key)
                .where(PartitionRow.incarnation == PendingDerivativesRow.incarnation)
                .exists(),
            )
            .order_by(PendingDerivativesRow.created_at)
            .limit(1)
            # SQLite drops this; there the write transaction serializes claims.
            .with_for_update(skip_locked=True)
            .scalar_subquery()
        )
        async with (
            self._tracker("claim_pending_derivatives"),
            write_session(self._create_session) as session,
        ):
            claimed = (
                await session.execute(
                    update(PendingDerivativesRow)
                    .where(PendingDerivativesRow.uuid == oldest_due)
                    .values(
                        attempts_without_progress=(
                            PendingDerivativesRow.attempts_without_progress + 1
                        ),
                        claimed_until=self._seconds_from_now(lease_seconds),
                        claim_generation=PendingDerivativesRow.claim_generation + 1,
                    )
                    .returning(
                        PendingDerivativesRow.uuid,
                        PendingDerivativesRow.incarnation,
                        PendingDerivativesRow.event_uuid,
                        PendingDerivativesRow.properties,
                        PendingDerivativesRow.claim_generation,
                        PendingDerivativesRow.attempts_without_progress,
                    )
                )
            ).one_or_none()
            if claimed is None:
                return None
            partition_row = (
                await SQLAlchemyEventMemoryStore._get_partition_row_by_incarnation(
                    session, claimed.incarnation
                )
            )
            derivative_rows = (
                (
                    await session.execute(
                        select(PendingDerivativeRow).where(
                            PendingDerivativeRow.pending_uuid == claimed.uuid
                        )
                    )
                )
                .scalars()
                .all()
            )
        if partition_row is None:
            # Deleted since the claim's read; the claim ends with its lease,
            # and the purge reclaims the record.
            return None
        partition = await self._partition_from_partition_row(partition_row)
        payload_codec = await self._load_payload_codec(partition.config)
        return PendingDerivativesClaim(
            partition_key=partition_row.partition_key,
            partition=partition,
            pending_derivatives=PendingDerivatives(
                uuid=claimed.uuid,
                event_uuid=claimed.event_uuid,
                properties=decode_properties(claimed.properties),
                derivatives=[
                    _derivative_from_pending_derivative_row(row, payload_codec)
                    for row in derivative_rows
                ],
            ),
            generation=claimed.claim_generation,
            attempt=claimed.attempts_without_progress,
        )

    @override
    async def finish_pending_derivatives_claim(
        self, claim: PendingDerivativesClaim
    ) -> None:
        async with (
            self._tracker("finish_pending_derivatives_claim"),
            write_session(self._create_session) as session,
        ):
            # The cascade follows to the record's derivatives.
            await session.execute(
                delete(PendingDerivativesRow).where(
                    PendingDerivativesRow.uuid == claim.pending_derivatives.uuid,
                    PendingDerivativesRow.claim_generation == claim.generation,
                )
            )

    @override
    async def fail_pending_derivatives_claim(
        self, claim: PendingDerivativesClaim, error: Exception
    ) -> None:
        async with (
            self._tracker("fail_pending_derivatives_claim"),
            write_session(self._create_session) as session,
        ):
            ended = await (await session.connection()).execute(
                update(PendingDerivativesRow)
                .where(
                    PendingDerivativesRow.uuid == claim.pending_derivatives.uuid,
                    PendingDerivativesRow.claim_generation == claim.generation,
                )
                .values(claimed_until=None, last_failed_at=func.now())
            )
        if (
            ended.rowcount == 1
            and claim.attempt >= _MAX_CLAIM_ATTEMPTS_WITHOUT_PROGRESS
        ):
            logger.error(
                "Pending derivatives %s of event %s in partition %r are no longer "
                "claimed after %d failed claims in a row; last error: %r. Set "
                "their attempts_without_progress to 0 in %s to claim them again.",
                claim.pending_derivatives.uuid,
                claim.pending_derivatives.event_uuid,
                claim.partition_key,
                claim.attempt,
                error,
                PendingDerivativesRow.__tablename__,
            )

    @override
    async def claim_tombstone(self) -> TombstoneClaim | None:
        oldest_due = (
            select(TombstoneRow.uuid)
            .where(
                or_(
                    TombstoneRow.second_delete.is_(False),
                    self._has_elapsed(
                        self._tombstone_retention_seconds,
                        since=TombstoneRow.enqueued_at,
                    ),
                ),
                self._is_claimable(TombstoneRow),
                select(PartitionRow.partition_key)
                .where(PartitionRow.incarnation == TombstoneRow.incarnation)
                .exists(),
            )
            .order_by(TombstoneRow.enqueued_at)
            .limit(1)
            # SQLite drops this; there the write transaction serializes claims.
            .with_for_update(skip_locked=True)
            .scalar_subquery()
        )
        async with (
            self._tracker("claim_tombstone"),
            write_session(self._create_session) as session,
        ):
            claimed = (
                await session.execute(
                    update(TombstoneRow)
                    .where(TombstoneRow.uuid == oldest_due)
                    .values(
                        attempts_without_progress=(
                            TombstoneRow.attempts_without_progress + 1
                        ),
                        claimed_until=self._seconds_from_now(
                            self._tombstone_lease_seconds
                        ),
                        claim_generation=TombstoneRow.claim_generation + 1,
                    )
                    .returning(
                        TombstoneRow.uuid,
                        TombstoneRow.incarnation,
                        TombstoneRow.second_delete,
                        TombstoneRow.claim_generation,
                        TombstoneRow.attempts_without_progress,
                    )
                )
            ).one_or_none()
            if claimed is None:
                return None
            partition_row = (
                await SQLAlchemyEventMemoryStore._get_partition_row_by_incarnation(
                    session, claimed.incarnation
                )
            )
            derivative_uuids = list(
                (
                    await session.execute(
                        select(TombstoneDerivativeRow.uuid).where(
                            TombstoneDerivativeRow.tombstone_uuid == claimed.uuid
                        )
                    )
                ).scalars()
            )
        if partition_row is None:
            # Deleted since the claim's read: the partition's index goes
            # with it, and the purge reclaims the tombstone.
            return None
        return TombstoneClaim(
            partition_key=partition_row.partition_key,
            tombstone=Tombstone(uuid=claimed.uuid, derivative_uuids=derivative_uuids),
            second_delete=claimed.second_delete,
            generation=claimed.claim_generation,
            attempt=claimed.attempts_without_progress,
        )

    @override
    async def finish_tombstone_claim(self, claim: TombstoneClaim) -> None:
        async with (
            self._tracker("finish_tombstone_claim"),
            write_session(self._create_session) as session,
        ):
            if claim.second_delete:
                # The cascade follows to the tombstone's derivatives.
                await session.execute(
                    delete(TombstoneRow).where(
                        TombstoneRow.uuid == claim.tombstone.uuid
                    )
                )
                return
            await session.execute(
                update(TombstoneRow)
                .where(
                    TombstoneRow.uuid == claim.tombstone.uuid,
                    TombstoneRow.second_delete.is_(False),
                )
                .values(
                    second_delete=True,
                    attempts_without_progress=0,
                    claimed_until=None,
                )
            )

    @override
    async def fail_tombstone_claim(
        self, claim: TombstoneClaim, error: Exception
    ) -> None:
        async with (
            self._tracker("fail_tombstone_claim"),
            write_session(self._create_session) as session,
        ):
            ended = await (await session.connection()).execute(
                update(TombstoneRow)
                .where(
                    TombstoneRow.uuid == claim.tombstone.uuid,
                    TombstoneRow.claim_generation == claim.generation,
                )
                .values(claimed_until=None, last_failed_at=func.now())
            )
        if (
            ended.rowcount == 1
            and claim.attempt >= _MAX_CLAIM_ATTEMPTS_WITHOUT_PROGRESS
        ):
            logger.error(
                "Tombstone %s in partition %r is no longer claimed after %d failed "
                "claims in a row, so its derivatives may stay in their index; "
                "last error: %r. Set its attempts_without_progress to 0 in %s to "
                "claim it again.",
                claim.tombstone.uuid,
                claim.partition_key,
                claim.attempt,
                error,
                TombstoneRow.__tablename__,
            )

    def _is_claimable(
        self, table: type[PendingDerivativesRow | TombstoneRow]
    ) -> ColumnElement[bool]:
        """Whether a row of `table` may be claimed, aside from being due."""
        backoff_seconds = self._claim_retry_backoff_seconds(table)
        return and_(
            table.attempts_without_progress < _MAX_CLAIM_ATTEMPTS_WITHOUT_PROGRESS,
            or_(
                # No claim since the row last made progress.
                table.attempts_without_progress == 0,
                # No claim is open: the backoff runs from the last failure.
                and_(
                    table.claimed_until.is_(None),
                    or_(
                        table.last_failed_at.is_(None),
                        self._has_elapsed(backoff_seconds, since=table.last_failed_at),
                    ),
                ),
                # The open claim never ended: the backoff runs from the end
                # of its lease.
                and_(
                    table.claimed_until.is_not(None),
                    self._has_elapsed(backoff_seconds, since=table.claimed_until),
                ),
            ),
        )

    def _claim_retry_backoff_seconds(
        self, table: type[PendingDerivativesRow | TombstoneRow]
    ) -> ColumnElement[int]:
        """Each row's claim retry backoff in seconds, computed by the database."""
        # 1 << (a - 1) is 2 ** (a - 1) on both dialects. a is clamped at 0:
        # claims evaluate this for rows with no attempts too, and PostgreSQL
        # leaves a negative shift undefined.
        attempts = table.attempts_without_progress
        doublings = literal(1, Integer).op("<<")(
            case((attempts > 0, attempts - 1), else_=0)
        )
        smaller = func.min if self._is_sqlite else func.least
        return smaller(
            self._base_claim_retry_backoff_seconds * doublings,
            self._max_claim_retry_backoff_seconds,
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

    def _seconds_from_now(self, seconds: int) -> ColumnElement[datetime]:
        """The time `seconds` from now, on the database clock."""
        if self._is_sqlite:
            return func.datetime(
                "now",
                func.printf("+%d seconds", seconds),
                type_=DateTime(timezone=True),
            )
        return func.now() + literal(timedelta(seconds=seconds), Interval)

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
    async def _get_partition_row_by_incarnation(
        session: AsyncSession,
        incarnation: UUID,
    ) -> PartitionRow | None:
        """Fetch a partition row by incarnation."""
        return (
            await session.execute(
                select(PartitionRow).where(PartitionRow.incarnation == incarnation)
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


def _derivative_from_pending_derivative_row(
    row: PendingDerivativeRow, payload_codec: PayloadCodec
) -> Derivative:
    """Convert a PendingDerivativeRow into a Derivative."""
    original_timezone = timezone(timedelta(seconds=row.timestamp_timezone_offset))
    return Derivative(
        uuid=row.uuid,
        segment_uuid=row.segment_uuid,
        timestamp=row.timestamp.astimezone(original_timezone),
        source_id=row.source_id,
        block=decode_block(json.loads(payload_codec.decode(row.block))),
    )
