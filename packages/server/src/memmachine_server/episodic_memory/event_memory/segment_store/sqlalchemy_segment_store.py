"""SQLAlchemy implementation of the SegmentStore interface."""

import json
import logging
import sqlite3
from collections.abc import Iterable, Mapping
from datetime import UTC, datetime, timedelta, timezone
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
    CheckConstraint,
    ColumnCollection,
    ColumnElement,
    DateTime,
    ForeignKeyConstraint,
    Index,
    Integer,
    Interval,
    LargeBinary,
    Row,
    Select,
    String,
    Tuple,
    Uuid,
    and_,
    bindparam,
    delete,
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
from memmachine_server.common.sql import (
    validate_engine_connections,
    write_connection,
    write_session,
)
from memmachine_server.common.utils import ensure_tz_aware, utc_offset_seconds
from memmachine_server.episodic_memory.event_memory.data_types import (
    Derivative,
    NullContext,
    Segment,
    decode_block,
    decode_context,
    encode_block,
    encode_context,
)
from memmachine_server.episodic_memory.event_memory.segment_store.data_types import (
    SegmentStoreAttemptsExhaustedError,
    SegmentStoreEventAlreadyExistsError,
    SegmentStorePartitionAlreadyExistsError,
    SegmentStorePartitionConfig,
    SegmentStorePartitionConfigMismatchError,
    SegmentStorePartitionHandleStaleError,
    Tombstone,
)
from memmachine_server.episodic_memory.event_memory.segment_store.segment_store import (
    PendingWriteClaim,
    SegmentStore,
    SegmentStorePartition,
    TombstoneClaim,
)
from memmachine_server.episodic_memory.event_memory.segment_store.utils import (
    validate_partition_key,
)

logger = logging.getLogger(__name__)

_JSON_AUTO = JSON().with_variant(JSONB, "postgresql")


# Consecutive failed mint attempts before the store concludes it
# is re-attempting a persistent database error rather than losing races: a real
# uuid collision is a once-in-the-universe event and each race retry
# requires another actor to have changed the registry in a ~millisecond
# window, so consecutive failures at this depth mean the IntegrityError
# has some other, permanent cause.
_MAX_MINT_ATTEMPTS = 10

# Adding events and claiming pending writes and tombstones depend on
# RETURNING, which SQLite added in 3.35.
_MIN_SQLITE_VERSION = (3, 35)

# A context read takes its context from at most this many segments on each
# side of a seed, matching or not.
_MAX_CONTEXT_DISTANCE = 1_000

# Failed claims in a row a pending write or a tombstone gets
# before it is no longer claimed.
_MAX_CLAIM_ATTEMPTS_WITHOUT_PROGRESS = 10


class _RegistryInsertRejectedError(Exception):
    """A registry insert was rejected although its key is free.

    Raised when the insert fails with an integrity error but no row
    exists under the key, or when the minted incarnation still has
    garbage awaiting purge.
    """


# ORM models


class BaseSegmentStore(DeclarativeBase):
    """Base class for segment store tables."""


class PartitionRow(BaseSegmentStore):
    """The tenant registry: one row per live partition incarnation."""

    __tablename__ = "segment_store_pt"

    partition_key: MappedColumn[str] = mapped_column(String(255), primary_key=True)
    incarnation: MappedColumn[UUID] = mapped_column(Uuid, nullable=False, unique=True)
    payload_codec_config: MappedColumn[dict[str, JsonValue]] = mapped_column(
        _JSON_AUTO,
        nullable=False,
    )


class EventRow(BaseSegmentStore):
    """One row per event the partition holds.

    The primary key is what makes an event addable once: a second
    `add_event_segments` naming it conflicts here, before any segment is written.
    Deleting the row cascades to the event's segments, and through them to
    their pending writes.
    """

    __tablename__ = "segment_store_ev"

    incarnation: MappedColumn[UUID] = mapped_column(Uuid, primary_key=True)
    uuid: MappedColumn[UUID] = mapped_column(Uuid, primary_key=True)


class SegmentRow(BaseSegmentStore):
    """Persisted segment."""

    __tablename__ = "segment_store_sg"

    incarnation: MappedColumn[UUID] = mapped_column(Uuid, primary_key=True)

    uuid: MappedColumn[UUID] = mapped_column(Uuid, primary_key=True)
    event_uuid: MappedColumn[UUID] = mapped_column(Uuid, nullable=False)
    index: MappedColumn[int] = mapped_column(Integer, nullable=False)
    offset: MappedColumn[int] = mapped_column(Integer, nullable=False)
    timestamp: MappedColumn[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )
    timestamp_timezone_offset: MappedColumn[int] = mapped_column(
        Integer, nullable=False, default=0
    )
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
                "segment_store_ev.incarnation",
                "segment_store_ev.uuid",
            ],
            ondelete="CASCADE",
        ),
        # Serves the event lookups and the cascade from the event row.
        Index(
            "segment_store_sg__in_ev",
            "incarnation",
            "event_uuid",
        ),
        Index(
            "segment_store_sg__in_ts_ev_ix_of",
            "incarnation",
            "timestamp",
            "event_uuid",
            "index",
            "offset",
        ),
    )


class DerivativeRow(BaseSegmentStore):
    """A derivative of a segment: what its derived index is written from, and the segment a hit on it names.

    Deleted by cascade with its segment; the segment's tombstone keeps its
    UUID.
    """

    __tablename__ = "segment_store_dv"

    incarnation: MappedColumn[UUID] = mapped_column(Uuid, primary_key=True)

    uuid: MappedColumn[UUID] = mapped_column(Uuid, primary_key=True)
    segment_uuid: MappedColumn[UUID] = mapped_column(Uuid, nullable=False)
    timestamp: MappedColumn[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )
    timestamp_timezone_offset: MappedColumn[int] = mapped_column(
        Integer, nullable=False, default=0
    )
    context: MappedColumn[bytes] = mapped_column(LargeBinary, nullable=False)
    block: MappedColumn[bytes] = mapped_column(LargeBinary, nullable=False)
    properties: MappedColumn[dict[str, JsonValue]] = mapped_column(
        _JSON_AUTO, nullable=False, default=dict
    )

    __table_args__ = (
        ForeignKeyConstraint(
            ["incarnation", "segment_uuid"],
            [
                "segment_store_sg.incarnation",
                "segment_store_sg.uuid",
            ],
            ondelete="CASCADE",
        ),
        # Serves the lookups by segment and the cascade from the segment row.
        Index(
            "segment_store_dv__in_su",
            "incarnation",
            "segment_uuid",
        ),
    )


class PendingWriteRow(BaseSegmentStore):
    """A segment whose derivatives await their derived index.

    Written with the segment; its writer deletes it once the index has the
    segment's derivatives, and a claim takes it when the writer does not.
    Deleted with its segment, so a forgotten event leaves none behind.
    """

    __tablename__ = "segment_store_pw"

    incarnation: MappedColumn[UUID] = mapped_column(Uuid, primary_key=True)
    segment_uuid: MappedColumn[UUID] = mapped_column(Uuid, primary_key=True)
    # When a claim may take the record: once its writer's lease has passed,
    # once an open claim's lease and then the backoff have passed, or once
    # the backoff after a failed claim has passed.
    claimable_at: MappedColumn[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )
    # Claims since the record last made progress, the open one included.
    attempts_without_progress: MappedColumn[int] = mapped_column(
        Integer, nullable=False, default=0
    )
    # Incremented by each claim; a claim ends only itself.
    claim_generation: MappedColumn[int] = mapped_column(
        Integer, nullable=False, default=0
    )

    __table_args__ = (
        ForeignKeyConstraint(
            ["incarnation", "segment_uuid"],
            [
                "segment_store_sg.incarnation",
                "segment_store_sg.uuid",
            ],
            ondelete="CASCADE",
        ),
        Index("segment_store_pw__ca", "claimable_at"),
    )


class TombstoneRow(BaseSegmentStore):
    """A forgotten segment's tombstone, holding the UUIDs of its derivatives.

    Their derived index entries are deleted at once, and once more after
    the store's retention has passed since the first delete, which then
    deletes the tombstone and, by cascade, the UUIDs.
    """

    __tablename__ = "segment_store_tb"

    incarnation: MappedColumn[UUID] = mapped_column(Uuid, primary_key=True)
    segment_uuid: MappedColumn[UUID] = mapped_column(Uuid, primary_key=True)
    # When the first delete was recorded; None while the tombstone awaits it.
    first_deleted_at: MappedColumn[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    # When a claim may take the tombstone, once its delete is due: at once
    # from the forget, once an open claim's lease and then the backoff have
    # passed, or once the backoff after a failed claim has passed. None from
    # the first delete's record until a claim of the second sets it.
    claimable_at: MappedColumn[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    # Claims since the tombstone last made progress, the open one included.
    attempts_without_progress: MappedColumn[int] = mapped_column(
        Integer, nullable=False, default=0
    )
    # Incremented by each claim, and by the first delete's record; a claim
    # ends only itself.
    claim_generation: MappedColumn[int] = mapped_column(
        Integer, nullable=False, default=0
    )

    __table_args__ = (
        # Until its first delete, a tombstone is scheduled, or no claim
        # would find it.
        CheckConstraint(
            "first_deleted_at IS NOT NULL OR claimable_at IS NOT NULL",
            name="segment_store_tb__scheduled_until_first_delete",
        ),
        # Serves both claims: by schedule, and second deletes by their
        # first delete.
        Index("segment_store_tb__ca_fd", "claimable_at", "first_deleted_at"),
    )


class TombstoneDerivativeRow(BaseSegmentStore):
    """The UUID of a forgotten derivative, held by its segment's tombstone.

    Copied from the derivative by the forget that deletes it, and deleted
    by cascade with the tombstone.
    """

    __tablename__ = "segment_store_tb_dv"

    incarnation: MappedColumn[UUID] = mapped_column(Uuid, primary_key=True)
    uuid: MappedColumn[UUID] = mapped_column(Uuid, primary_key=True)
    segment_uuid: MappedColumn[UUID] = mapped_column(Uuid, nullable=False)

    __table_args__ = (
        ForeignKeyConstraint(
            ["incarnation", "segment_uuid"],
            [
                "segment_store_tb.incarnation",
                "segment_store_tb.segment_uuid",
            ],
            ondelete="CASCADE",
        ),
        # Serves the tombstone claims' reads and the cascade from the
        # tombstone row.
        Index(
            "segment_store_tb_dv__in_su",
            "incarnation",
            "segment_uuid",
        ),
    )


class PurgeQueueRow(BaseSegmentStore):
    """The purge queue: one row per dead partition incarnation.

    Claimed oldest-first by the enqueue stamp, which is the database clock,
    so entries from every server order on one clock; entries stamped in the
    same tick are unordered among themselves. The incarnation identifies
    the rows to reclaim. The purge reclaims derivatives, then segments with
    their pending writes, then events, then the tombstones' derivative
    UUIDs, then tombstones, each continuing after its own cursor: every row
    of the incarnation keyed at or below the cursor is deleted, and a
    cursor is NULL until its first full batch.
    A batch that ends exactly on the last row of one table leaves that
    table's cursor naming it, which says nothing about the next table,
    hence a cursor per table. An entry holds only what the purge reads.
    """

    __tablename__ = "segment_store_gc"

    incarnation: MappedColumn[UUID] = mapped_column(Uuid, primary_key=True)
    enqueued_at: MappedColumn[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )
    derivatives_purged_through: MappedColumn[UUID | None] = mapped_column(
        Uuid, nullable=True
    )
    segments_purged_through: MappedColumn[UUID | None] = mapped_column(
        Uuid, nullable=True
    )
    events_purged_through: MappedColumn[UUID | None] = mapped_column(
        Uuid, nullable=True
    )
    tombstone_derivatives_purged_through: MappedColumn[UUID | None] = mapped_column(
        Uuid, nullable=True
    )
    tombstones_purged_through: MappedColumn[UUID | None] = mapped_column(
        Uuid, nullable=True
    )

    __table_args__ = (Index("segment_store_gc__ea", "enqueued_at"),)


class SQLAlchemySegmentStorePartition(SegmentStorePartition):
    """SQLAlchemy-backed partition handle."""

    def __init__(
        self,
        partition_key: str,
        incarnation: UUID,
        create_session: async_sessionmaker[AsyncSession],
        is_sqlite: bool,
        config: SegmentStorePartitionConfig,
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
    def config(self) -> SegmentStorePartitionConfig:
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
            raise SegmentStorePartitionHandleStaleError(self._partition_key)

    # Registration

    @override
    async def add_event_segments(
        self,
        segments_by_event: Mapping[UUID, Iterable[Segment]],
        derivatives: Iterable[Derivative],
        *,
        lease_seconds: int,
    ) -> None:
        segments_by_event = {
            event_uuid: list(segments)
            for event_uuid, segments in segments_by_event.items()
        }
        derivatives = list(derivatives)
        SQLAlchemySegmentStorePartition._validate_events(segments_by_event, derivatives)
        if not segments_by_event:
            return

        async with (
            self._tracker("add_event_segments"),
            write_session(self._create_session) as session,
        ):
            await self._ensure_partition_live(session, pin=True)
            stored = await self._insert_event_rows(session, segments_by_event.keys())
            already_stored = segments_by_event.keys() - stored
            if already_stored:
                raise SegmentStoreEventAlreadyExistsError(already_stored)
            await self._insert_segments(
                session,
                (
                    segment
                    for segments in segments_by_event.values()
                    for segment in segments
                ),
            )
            await self._insert_derivatives(session, derivatives)
            await self._insert_pending_writes(
                session,
                (
                    segment.uuid
                    for segments in segments_by_event.values()
                    for segment in segments
                ),
                lease_seconds,
            )

    @staticmethod
    def _validate_events(
        segments_by_event: Mapping[UUID, list[Segment]],
        derivatives: list[Derivative],
    ) -> None:
        """Raise ValueError unless each segment is under its own event, and each derivative names a segment in the batch."""
        segment_uuids: set[UUID] = set()
        for event_uuid, segments in segments_by_event.items():
            for segment in segments:
                if segment.event_uuid != event_uuid:
                    raise ValueError(
                        f"Segment {segment.uuid} names event {segment.event_uuid}, "
                        f"listed under {event_uuid}"
                    )
                segment_uuids.add(segment.uuid)
        for derivative in derivatives:
            if derivative.segment_uuid not in segment_uuids:
                raise ValueError(
                    f"Derivative {derivative.uuid} names segment "
                    f"{derivative.segment_uuid}, which the batch does not add"
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
                "timestamp": ensure_tz_aware(segment.timestamp).astimezone(UTC),
                "timestamp_timezone_offset": utc_offset_seconds(segment.timestamp),
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

    async def _insert_derivatives(
        self,
        session: AsyncSession,
        derivatives: Iterable[Derivative],
    ) -> None:
        """Insert a row per derivative, under the segment it names."""
        derivative_row_values = [
            {
                "uuid": derivative.uuid,
                "incarnation": self._incarnation,
                "segment_uuid": derivative.segment_uuid,
                "timestamp": ensure_tz_aware(derivative.timestamp).astimezone(UTC),
                "timestamp_timezone_offset": utc_offset_seconds(derivative.timestamp),
                "context": self._payload_codec.encode(
                    json.dumps(encode_context(derivative.context)).encode("utf-8")
                ),
                "block": self._payload_codec.encode(
                    json.dumps(encode_block(derivative.block)).encode("utf-8")
                ),
                "properties": encode_properties(derivative.properties),
            }
            for derivative in derivatives
        ]
        if derivative_row_values:
            await session.execute(insert(DerivativeRow), derivative_row_values)

    async def _insert_pending_writes(
        self,
        session: AsyncSession,
        segment_uuids: Iterable[UUID],
        lease_seconds: int,
    ) -> None:
        """Insert a pending write per segment, claimable once `lease_seconds` have passed."""
        segment_uuids = list(segment_uuids)
        if not segment_uuids:
            return
        # One multi-row statement, so every row carries the database clock.
        await session.execute(
            insert(PendingWriteRow).values(
                [
                    {
                        "incarnation": self._incarnation,
                        "segment_uuid": segment_uuid,
                        "claimable_at": _seconds_from_now(
                            lease_seconds, is_sqlite=self._is_sqlite
                        ),
                    }
                    for segment_uuid in segment_uuids
                ]
            )
        )

    @override
    async def delete_pending_writes(
        self,
        segment_uuids: Iterable[UUID],
    ) -> None:
        segment_uuids = set(segment_uuids)
        if not segment_uuids:
            return

        async with (
            self._tracker("delete_pending_writes"),
            write_session(self._create_session) as session,
        ):
            await self._ensure_partition_live(session, pin=True)
            await session.execute(
                delete(PendingWriteRow).where(
                    PendingWriteRow.incarnation == self._incarnation,
                    PendingWriteRow.segment_uuid.in_(segment_uuids),
                )
            )

    # Retrieval

    @override
    async def get_segment_contexts(
        self,
        seed_segment_uuids: Iterable[UUID],
        *,
        max_backward_segments: int = 0,
        max_forward_segments: int = 0,
        property_filter: FilterExpr | None = None,
    ) -> dict[UUID, list[Segment]]:
        seed_segment_uuids = set(seed_segment_uuids)
        if not seed_segment_uuids:
            return {}

        async with (
            self._tracker("get_segment_contexts"),
            self._create_session() as session,
        ):
            seed_segments_query = select(SegmentRow).where(
                SegmentRow.uuid.in_(seed_segment_uuids),
                SegmentRow.incarnation == self._incarnation,
                self._registry_row_query().exists(),
            )
            if property_filter is not None:
                seed_segments_query = seed_segments_query.where(
                    compile_sql_filter(
                        property_filter,
                        lambda field: (
                            SQLAlchemySegmentStorePartition._resolve_segment_field(
                                field, columns=SegmentRow.__table__.c
                            )
                        ),
                    )
                )
            seed_segment_rows = (
                (await session.execute(seed_segments_query)).scalars().all()
            )

            seed_segment_rows_by_uuid: dict[UUID, SegmentRow] = {
                row.uuid: row for row in seed_segment_rows
            }
            if not seed_segment_rows_by_uuid:
                await self._ensure_partition_live(session)
                return {}

            # Short-circuit: no context needed.
            if max_backward_segments <= 0 and max_forward_segments <= 0:
                return {
                    seed_segment_uuid: [
                        self._segment_from_segment_row(
                            seed_segment_row,
                        )
                    ]
                    for seed_segment_uuid, seed_segment_row in seed_segment_rows_by_uuid.items()
                }

            # Get backward/forward context rows.
            if not self._is_sqlite:
                context_rows_by_seed = await self._get_context_rows_lateral(
                    session,
                    seed_segment_rows_by_uuid,
                    max_backward_segments,
                    max_forward_segments,
                    property_filter,
                )
            else:
                context_rows_by_seed = await self._get_context_rows_loop(
                    session,
                    seed_segment_rows_by_uuid,
                    max_backward_segments,
                    max_forward_segments,
                    property_filter,
                )

            # Each statement above took its own snapshot (READ COMMITTED):
            # a deletion committing between the seeds statement and the
            # context statements would leave the seeds with silently
            # empty context. One registry read after the last statement
            # turns that window into the contractual stale-handle error.
            await self._ensure_partition_live(session)

            # Assemble results: [backward (reversed) + seed + forward].
            segments_by_seed: dict[UUID, list[Segment]] = {}
            for seed_uuid, seed_row in seed_segment_rows_by_uuid.items():
                backward_rows, forward_rows = context_rows_by_seed.get(
                    seed_uuid, ([], [])
                )
                segments_by_seed[seed_uuid] = [
                    self._segment_from_segment_row(row)
                    for row in [*reversed(backward_rows), seed_row, *forward_rows]
                ]

            return segments_by_seed

    async def _get_context_rows_lateral(
        self,
        session: AsyncSession,
        seed_rows_by_uuid: Mapping[UUID, SegmentRow],
        max_backward_segments: int,
        max_forward_segments: int,
        property_filter: FilterExpr | None,
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
                    property_filter=property_filter,
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
            await get_context_rows_directional(
                backward=True, limit=max_backward_segments
            )
            if max_backward_segments > 0
            else {seed_uuid: [] for seed_uuid in seed_rows_by_uuid}
        )

        forward_rows_by_seed = (
            await get_context_rows_directional(
                backward=False, limit=max_forward_segments
            )
            if max_forward_segments > 0
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
        max_backward_segments: int,
        max_forward_segments: int,
        property_filter: FilterExpr | None,
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
                    property_filter=property_filter,
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
            await get_context_rows_directional(
                backward=True, limit=max_backward_segments
            )
            if max_backward_segments > 0
            else {seed_uuid: [] for seed_uuid in seed_rows_by_uuid}
        )

        forward_rows_by_seed = (
            await get_context_rows_directional(
                backward=False, limit=max_forward_segments
            )
            if max_forward_segments > 0
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
        property_filter: FilterExpr | None,
    ) -> Select:
        """Select a seed's context on one side, nearest the seed first.

        `seed_ordering_values` is the seed's (timestamp, event_uuid, index,
        offset): bound parameters or an enclosing statement's columns.
        Context comes from at most the next _MAX_CONTEXT_DISTANCE segments.
        Unfiltered, it is the first `limit` of them. With a property filter,
        it is the first `limit` matches among them, read without the filter,
        which the planner cannot estimate, so the plan stays an ordered index
        scan.
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
                *SQLAlchemySegmentStorePartition._chronological_order(
                    segments, descending=backward
                )
            )
            # Refer to an enclosing statement that supplies the seed's
            # position instead of selecting from it again.
            .correlate_except(SegmentRow.__table__)
        )
        if property_filter is None:
            return walk.limit(min(limit, _MAX_CONTEXT_DISTANCE))
        window = walk.limit(_MAX_CONTEXT_DISTANCE).subquery("window")
        return (
            select(window)
            .where(
                compile_sql_filter(
                    property_filter,
                    lambda field: (
                        SQLAlchemySegmentStorePartition._resolve_segment_field(
                            field, columns=window.c
                        )
                    ),
                )
            )
            .order_by(
                *SQLAlchemySegmentStorePartition._chronological_order(
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
            # Served by the primary key, which leads on the columns this
            # filters on.
            query = select(DerivativeRow.uuid, DerivativeRow.segment_uuid).where(
                DerivativeRow.incarnation == self._incarnation,
                DerivativeRow.uuid.in_(derivative_uuids),
                self._registry_row_query().exists(),
            )
            rows = (await session.execute(query)).all()
            if not rows:
                await self._ensure_partition_live(session)

        return {row.uuid: row.segment_uuid for row in rows}

    # Forgetting

    @override
    async def delete_event_segments(
        self,
        event_uuids: Iterable[UUID],
    ) -> list[Tombstone]:
        event_uuids = set(event_uuids)
        if not event_uuids:
            return []

        async with (
            self._tracker("delete_event_segments"),
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
                return []
            # The segments, locked in order too: the cascade alone would
            # take their locks in table order.
            segment_uuids = list(
                (
                    await session.execute(
                        select(SegmentRow.uuid)
                        .where(
                            SegmentRow.incarnation == self._incarnation,
                            SegmentRow.event_uuid.in_(held_event_uuids),
                        )
                        .order_by(SegmentRow.uuid)
                        .with_for_update()
                    )
                ).scalars()
            )

            derivative_uuids_by_segment: dict[UUID, list[UUID]] = {
                segment_uuid: [] for segment_uuid in segment_uuids
            }
            if segment_uuids:
                await session.execute(
                    insert(TombstoneRow).values(
                        [
                            {
                                "incarnation": self._incarnation,
                                "segment_uuid": segment_uuid,
                                "claimable_at": func.now(),
                            }
                            for segment_uuid in segment_uuids
                        ]
                    )
                )
                # Each tombstone copies its segment's derivative UUIDs, found
                # by the segments locked above, which the derivatives' index
                # leads on.
                copied = await session.execute(
                    insert(TombstoneDerivativeRow)
                    .from_select(
                        ["incarnation", "uuid", "segment_uuid"],
                        select(
                            DerivativeRow.incarnation,
                            DerivativeRow.uuid,
                            DerivativeRow.segment_uuid,
                        ).where(
                            DerivativeRow.incarnation == self._incarnation,
                            DerivativeRow.segment_uuid.in_(segment_uuids),
                        ),
                    )
                    .returning(
                        TombstoneDerivativeRow.uuid,
                        TombstoneDerivativeRow.segment_uuid,
                    )
                )
                for derivative_uuid, segment_uuid in copied.all():
                    derivative_uuids_by_segment[segment_uuid].append(derivative_uuid)

            # The cascade follows to the events' segments, and through them
            # to their pending writes and derivatives.
            await session.execute(
                delete(EventRow).where(
                    EventRow.incarnation == self._incarnation,
                    EventRow.uuid.in_(held_event_uuids),
                )
            )
        return [
            Tombstone(segment_uuid=segment_uuid, derivative_uuids=derivative_uuids)
            for segment_uuid, derivative_uuids in derivative_uuids_by_segment.items()
        ]

    @override
    async def record_tombstone_first_delete(
        self, segment_uuids: Iterable[UUID]
    ) -> None:
        segment_uuids = set(segment_uuids)
        if not segment_uuids:
            return
        async with (
            self._tracker("record_tombstone_first_delete"),
            write_session(self._create_session) as session,
        ):
            await self._ensure_partition_live(session, pin=True)
            await session.execute(
                update(TombstoneRow)
                .where(
                    TombstoneRow.incarnation == self._incarnation,
                    TombstoneRow.segment_uuid.in_(segment_uuids),
                    TombstoneRow.first_deleted_at.is_(None),
                )
                # Made without a claim, so it starts a new generation: an
                # open claim's late failure then changes nothing.
                .values(
                    first_deleted_at=func.now(),
                    claimable_at=None,
                    attempts_without_progress=0,
                    claim_generation=TombstoneRow.claim_generation + 1,
                )
            )

    # Helpers

    @staticmethod
    def _resolve_segment_field(
        field: str,
        *,
        columns: ColumnCollection[str, ColumnElement],
    ) -> tuple[ColumnElement, FieldEncoding]:
        """
        Map a filter field name to one of `columns` and its encoding.

        Raises ValueError for an unknown field.
        """
        if field == "timestamp":
            return columns.timestamp, "column"
        internal_name, is_user_metadata = normalize_filter_field(field)
        if not is_user_metadata:
            raise ValueError(f"Filter field {field!r} names no segment property.")
        key = demangle_user_metadata_key(internal_name)
        return columns.properties[key], "properties_json"

    def _segment_from_segment_row(self, row: SegmentRow) -> Segment:
        """Convert a SegmentRow into a Segment."""
        context = decode_context(json.loads(self._payload_codec.decode(row.context)))
        if context is None:
            context = NullContext()
        block = decode_block(json.loads(self._payload_codec.decode(row.block)))
        properties = decode_properties(row.properties)
        original_timezone = timezone(timedelta(seconds=row.timestamp_timezone_offset))
        timestamp = ensure_tz_aware(row.timestamp).astimezone(original_timezone)
        return Segment(
            uuid=row.uuid,
            event_uuid=row.event_uuid,
            index=row.index,
            offset=row.offset,
            timestamp=timestamp,
            context=context,
            block=block,
            properties=properties,
        )


class SQLAlchemySegmentStoreParams(BaseModel):
    """
    Parameters for constructing a SQLAlchemySegmentStore.

    Attributes:
        engine (AsyncEngine):
            Async SQLAlchemy engine. On SQLite it must enforce foreign
            keys on every connection, registered at engine creation
            (enable_sqlite_foreign_keys): the store relies on the
            cascade from events to segments and does not verify
            enforcement. It must also begin its transactions explicitly,
            set up at engine creation (enable_sqlite_explicit_begin): the
            store's write transactions take SQLite's write lock at BEGIN.
        metrics_factory (MetricsFactory | None):
            An instance of MetricsFactory for collecting usage metrics
            (default: None).
        purge_max_rows (int):
            Maximum number of rows purged per call: derivatives, segments,
            events, tombstones' derivative UUIDs, and tombstones. Deleting
            a segment deletes its pending write by cascade (default: 10000).
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
            "cascade from events to segments and does not verify "
            "enforcement. It must also begin its transactions explicitly, "
            "set up at engine "
            "creation (enable_sqlite_explicit_begin): the store's write "
            "transactions take SQLite's write lock at BEGIN"
        ),
    )
    metrics_factory: InstanceOf[MetricsFactory] | None = Field(
        None,
        description="An instance of MetricsFactory for collecting usage metrics",
    )
    purge_max_rows: int = Field(
        10_000,
        gt=0,
        description=(
            "Maximum number of rows purged per call: derivatives, segments, "
            "events, tombstones' derivative UUIDs, and tombstones. Deleting "
            "a segment deletes its pending write by cascade"
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
        validate_engine_connections(engine)
        if (
            engine.dialect.name == "sqlite"
            and sqlite3.sqlite_version_info < _MIN_SQLITE_VERSION
        ):
            minimum = ".".join(str(part) for part in _MIN_SQLITE_VERSION)
            raise ValueError(
                f"SQLite runtime {sqlite3.sqlite_version} lacks the RETURNING "
                "support adding events and claiming pending writes and "
                f"tombstones depend on. Use SQLite {minimum} or newer."
            )
        return engine


class SQLAlchemySegmentStore(SegmentStore):
    """SQLAlchemy-backed SegmentStore factory."""

    def __init__(self, params: SQLAlchemySegmentStoreParams) -> None:
        """Initialize with an async SQLAlchemy engine."""
        self._engine = params.engine
        self._create_session = async_sessionmaker(self._engine, expire_on_commit=False)

        self._tracker = OperationTracker(
            params.metrics_factory,
            prefix="segment_store_sqlalchemy",
        )

        self._purge_max_rows = params.purge_max_rows
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
            await connection.run_sync(BaseSegmentStore.metadata.create_all)

    @override
    async def shutdown(self) -> None:
        pass

    # Partition management

    @override
    async def create_partition(
        self,
        partition_key: str,
        config: SegmentStorePartitionConfig,
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
                        raise SegmentStoreAttemptsExhaustedError(
                            f"Creating partition {partition_key!r} made no "
                            f"progress after {_MAX_MINT_ATTEMPTS} attempts"
                        ) from err
                    continue
                return

    async def _insert_partition_row(
        self,
        partition_key: str,
        incarnation: UUID,
        config: SegmentStorePartitionConfig,
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
            SegmentStorePartitionAlreadyExistsError:
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
                partition_row = await SQLAlchemySegmentStore._get_partition_row(
                    session, partition_key
                )
            if partition_row is not None:
                raise SegmentStorePartitionAlreadyExistsError(partition_key) from err
            raise _RegistryInsertRejectedError(
                f"the insert of incarnation {incarnation} failed and no row "
                "exists under the key"
            ) from err

    @override
    async def open_partition(
        self, partition_key: str
    ) -> SQLAlchemySegmentStorePartition | None:
        validate_partition_key(partition_key)
        async with self._tracker("open_partition"):
            async with self._create_session() as session:
                partition_row = await SQLAlchemySegmentStore._get_partition_row(
                    session, partition_key
                )
            if partition_row is None:
                return None

            return await self._partition_from_partition_row(partition_row)

    @override
    async def open_or_create_partition(
        self,
        partition_key: str,
        config: SegmentStorePartitionConfig,
    ) -> SQLAlchemySegmentStorePartition:
        validate_partition_key(partition_key)
        async with self._tracker("open_or_create_partition"):
            return await self._open_or_create_partition(partition_key, config)

    async def _open_or_create_partition(
        self,
        partition_key: str,
        config: SegmentStorePartitionConfig,
    ) -> SQLAlchemySegmentStorePartition:
        attempts = 0
        # Read-then-insert, retried: losing the insert race means a
        # concurrent creator won (reopen its row), and finding no row
        # after losing means a concurrent delete removed the winner --
        # every retry requires another actor to have changed the state
        # (or, vanishingly, a minted incarnation to have collided).
        while True:
            async with self._create_session() as session:
                partition_row = await SQLAlchemySegmentStore._get_partition_row(
                    session, partition_key
                )

            if partition_row is not None:
                SQLAlchemySegmentStore._raise_if_partition_config_mismatch(
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
                SegmentStorePartitionAlreadyExistsError,
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
                    raise SegmentStoreAttemptsExhaustedError(
                        f"Opening or creating partition {partition_key!r} "
                        f"made no progress after {_MAX_MINT_ATTEMPTS} "
                        f"attempts"
                    ) from err
                # Lost the creation race (reopen the winner's row next
                # iteration) or minted a colliding incarnation (mint a
                # fresh one).
                continue

            return SQLAlchemySegmentStorePartition(
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
        self, segment_store_partition: SegmentStorePartition
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
        # design/segment_store_shared_tables.md.
        remaining = self._purge_max_rows
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
                        PurgeQueueRow.derivatives_purged_through,
                        PurgeQueueRow.segments_purged_through,
                        PurgeQueueRow.events_purged_through,
                        PurgeQueueRow.tombstone_derivatives_purged_through,
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
                    "derivatives_purged_through": entry.derivatives_purged_through,
                    "segments_purged_through": entry.segments_purged_through,
                    "events_purged_through": entry.events_purged_through,
                    "tombstone_derivatives_purged_through": (
                        entry.tombstone_derivatives_purged_through
                    ),
                    "tombstones_purged_through": entry.tombstones_purged_through,
                }

                # Each table is purged before the one its rows cascade from,
                # so no cascade runs on the budgeted path but a segment's to
                # its one pending write. Each table draws on the same budget,
                # and a full batch records how far it got and leaves the
                # entry for the next call.
                for table, key, cursor in (
                    (DerivativeRow, DerivativeRow.uuid, "derivatives_purged_through"),
                    (SegmentRow, SegmentRow.uuid, "segments_purged_through"),
                    (EventRow, EventRow.uuid, "events_purged_through"),
                    (
                        TombstoneDerivativeRow,
                        TombstoneDerivativeRow.uuid,
                        "tombstone_derivatives_purged_through",
                    ),
                    (
                        TombstoneRow,
                        TombstoneRow.segment_uuid,
                        "tombstones_purged_through",
                    ),
                ):
                    deleted, last = await SQLAlchemySegmentStore._purge_batch(
                        connection,
                        table,
                        key,
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
        table: type[
            DerivativeRow
            | SegmentRow
            | EventRow
            | TombstoneDerivativeRow
            | TombstoneRow
        ],
        key: InstrumentedAttribute[UUID],
        incarnation: UUID,
        *,
        after: UUID | None,
        limit: int,
    ) -> tuple[int, UUID | None]:
        """Delete up to `limit` of a dead incarnation's rows of `table` in `key` order after `after`.

        `key` is the column of the table's primary key after the
        incarnation. Returns how many were deleted and the last key deleted.
        The batch continues after the cursor, so it never reads rows an
        earlier batch deleted: PostgreSQL keeps them in the table and its
        indexes until vacuum, and a batch starting from the incarnation's
        first key would step over all of them.
        """
        after_cursor = [] if after is None else [key > after]
        batch = (
            select(key)
            .where(table.incarnation == incarnation, *after_cursor)
            .order_by(key)
            .limit(limit)
            .subquery()
        )
        last = (
            await connection.execute(
                select(batch.c[key.key]).order_by(batch.c[key.key].desc()).limit(1)
            )
        ).scalar_one_or_none()
        if last is None:
            return 0, None
        deleted = (
            await connection.execute(
                delete(table).where(
                    table.incarnation == incarnation,
                    *after_cursor,
                    key <= last,
                )
            )
        ).rowcount
        return deleted, last

    # Claims

    @override
    async def claim_pending_write(
        self,
        *,
        lease_seconds: int,
    ) -> PendingWriteClaim | None:
        oldest_due = (
            select(PendingWriteRow.incarnation, PendingWriteRow.segment_uuid)
            .where(PendingWriteRow.claimable_at <= func.now())
            .order_by(PendingWriteRow.claimable_at)
        )
        async with (
            self._tracker("claim_pending_write"),
            write_session(self._create_session) as session,
        ):
            claimed = await self._claim_oldest_due(
                session, PendingWriteRow, [oldest_due], lease_seconds=lease_seconds
            )
            if claimed is None:
                return None
            partition_row = (
                await SQLAlchemySegmentStore._get_partition_row_by_incarnation(
                    session, claimed.incarnation
                )
            )
            derivative_rows = (
                (
                    await session.execute(
                        select(DerivativeRow).where(
                            DerivativeRow.incarnation == claimed.incarnation,
                            DerivativeRow.segment_uuid == claimed.segment_uuid,
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
        return PendingWriteClaim(
            partition_key=partition_row.partition_key,
            incarnation=claimed.incarnation,
            segment_uuid=claimed.segment_uuid,
            derivatives=[
                _derivative_from_derivative_row(row, payload_codec)
                for row in derivative_rows
            ],
            generation=claimed.claim_generation,
            attempt=claimed.attempts_without_progress,
        )

    @override
    async def renew_pending_write_claim(
        self, claim: PendingWriteClaim, *, lease_seconds: int
    ) -> bool:
        _, claimed_record = _claimed_record(claim)
        async with (
            self._tracker("renew_pending_write_claim"),
            write_session(self._create_session) as session,
        ):
            # The update holds the record's row, as the claim did: a forget
            # deleting the record waits for this transaction, or this
            # update finds the record gone.
            renewed = (
                await session.execute(
                    update(PendingWriteRow)
                    .where(
                        claimed_record,
                        select(PartitionRow.partition_key)
                        .where(PartitionRow.incarnation == PendingWriteRow.incarnation)
                        .exists(),
                    )
                    .values(
                        claimable_at=_seconds_from_now(
                            lease_seconds
                            + self._claim_retry_backoff_seconds(
                                PendingWriteRow.attempts_without_progress
                            ),
                            is_sqlite=self._is_sqlite,
                        )
                    )
                    .returning(PendingWriteRow.segment_uuid)
                )
            ).one_or_none()
        return renewed is not None

    @override
    async def finish_pending_write_claim(self, claim: PendingWriteClaim) -> None:
        async with (
            self._tracker("finish_pending_write_claim"),
            write_session(self._create_session) as session,
        ):
            await session.execute(
                delete(PendingWriteRow).where(
                    PendingWriteRow.incarnation == claim.incarnation,
                    PendingWriteRow.segment_uuid == claim.segment_uuid,
                )
            )

    @override
    async def fail_pending_write_claim(
        self, claim: PendingWriteClaim, error: Exception
    ) -> None:
        async with (
            self._tracker("fail_pending_write_claim"),
            write_session(self._create_session) as session,
        ):
            ended = await self._schedule_after_failed_claim(session, claim)
        if ended and claim.attempt >= _MAX_CLAIM_ATTEMPTS_WITHOUT_PROGRESS:
            logger.error(
                "Pending write of segment %s in partition %r is no longer "
                "claimed after %d failed claims in a row; last error: %r. Set "
                "its attempts_without_progress to 0 in %s to claim it again.",
                claim.segment_uuid,
                claim.partition_key,
                claim.attempt,
                error,
                PendingWriteRow.__tablename__,
            )

    @override
    async def cancel_pending_write_claim(self, claim: PendingWriteClaim) -> None:
        async with (
            self._tracker("cancel_pending_write_claim"),
            write_session(self._create_session) as session,
        ):
            await self._take_back_cancelled_claim(session, claim)

    @override
    async def claim_tombstone(self) -> TombstoneClaim | None:
        # The retention is its own condition, so no schedule written to
        # claimable_at brings a second delete forward.
        retention_passed = or_(
            TombstoneRow.first_deleted_at.is_(None),
            self._has_elapsed(
                self._tombstone_retention_seconds,
                since=TombstoneRow.first_deleted_at,
            ),
        )
        # Scheduled deletes first, earliest first: first deletes and
        # retries. Then second deletes, oldest first delete first. Each is
        # one range of the (claimable_at, first_deleted_at) index.
        scheduled_due = (
            select(TombstoneRow.incarnation, TombstoneRow.segment_uuid)
            .where(TombstoneRow.claimable_at <= func.now(), retention_passed)
            .order_by(TombstoneRow.claimable_at)
        )
        second_deletes_due = (
            select(TombstoneRow.incarnation, TombstoneRow.segment_uuid)
            .where(
                TombstoneRow.claimable_at.is_(None),
                self._has_elapsed(
                    self._tombstone_retention_seconds,
                    since=TombstoneRow.first_deleted_at,
                ),
            )
            .order_by(TombstoneRow.first_deleted_at)
        )
        async with (
            self._tracker("claim_tombstone"),
            write_session(self._create_session) as session,
        ):
            claimed = await self._claim_oldest_due(
                session,
                TombstoneRow,
                [scheduled_due, second_deletes_due],
                lease_seconds=self._tombstone_lease_seconds,
            )
            if claimed is None:
                return None
            partition_row = (
                await SQLAlchemySegmentStore._get_partition_row_by_incarnation(
                    session, claimed.incarnation
                )
            )
            derivative_uuids = list(
                (
                    await session.execute(
                        select(TombstoneDerivativeRow.uuid).where(
                            TombstoneDerivativeRow.incarnation == claimed.incarnation,
                            TombstoneDerivativeRow.segment_uuid == claimed.segment_uuid,
                        )
                    )
                ).scalars()
            )
        if partition_row is None:
            # Deleted since the claim's read: the partition's deletion
            # deletes its vectors, and the purge deletes the tombstone.
            return None
        return TombstoneClaim(
            partition_key=partition_row.partition_key,
            incarnation=claimed.incarnation,
            tombstone=Tombstone(
                segment_uuid=claimed.segment_uuid, derivative_uuids=derivative_uuids
            ),
            second_delete=claimed.first_deleted_at is not None,
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
                # The cascade deletes the tombstone's derivative UUIDs.
                await session.execute(
                    delete(TombstoneRow).where(
                        TombstoneRow.incarnation == claim.incarnation,
                        TombstoneRow.segment_uuid == claim.tombstone.segment_uuid,
                    )
                )
                return
            await session.execute(
                update(TombstoneRow)
                .where(
                    TombstoneRow.incarnation == claim.incarnation,
                    TombstoneRow.segment_uuid == claim.tombstone.segment_uuid,
                    TombstoneRow.first_deleted_at.is_(None),
                )
                # Under any claim, so it starts a new generation: a later
                # claim's late failure then changes nothing.
                .values(
                    first_deleted_at=func.now(),
                    claimable_at=None,
                    attempts_without_progress=0,
                    claim_generation=TombstoneRow.claim_generation + 1,
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
            ended = await self._schedule_after_failed_claim(session, claim)
        if ended and claim.attempt >= _MAX_CLAIM_ATTEMPTS_WITHOUT_PROGRESS:
            logger.error(
                "Tombstone of segment %s in partition %r is no longer claimed "
                "after %d failed "
                "claims in a row, so its derivatives may stay in their index; "
                "last error: %r. Set its attempts_without_progress to 0 in %s to "
                "claim it again.",
                claim.tombstone.segment_uuid,
                claim.partition_key,
                claim.attempt,
                error,
                TombstoneRow.__tablename__,
            )

    @override
    async def cancel_tombstone_claim(self, claim: TombstoneClaim) -> None:
        async with (
            self._tracker("cancel_tombstone_claim"),
            write_session(self._create_session) as session,
        ):
            await self._take_back_cancelled_claim(session, claim)

    async def _claim_oldest_due(
        self,
        session: AsyncSession,
        table: type[PendingWriteRow] | type[TombstoneRow],
        due_queries: Iterable[Select[tuple[UUID, UUID]]],
        *,
        lease_seconds: int,
    ) -> Row | None:
        """Claim the first claimable record of `table` that the first of `due_queries` with one yields.

        Each query selects the keys of due records in the order they are
        claimed. A due record is claimable unless its claims failed the
        maximum number of times in a row, or its partition is deleted. The
        claim holds the record for `lease_seconds` and then the backoff, and
        returns the record's row as the claim left it.
        """
        for due_query in due_queries:
            oldest_claimable = (
                due_query.where(
                    table.attempts_without_progress
                    < _MAX_CLAIM_ATTEMPTS_WITHOUT_PROGRESS,
                    select(PartitionRow.partition_key)
                    .where(PartitionRow.incarnation == table.incarnation)
                    .exists(),
                )
                .limit(1)
                # SQLite drops this; there the write transaction serializes
                # claims.
                .with_for_update(skip_locked=True)
            )
            claimed = (
                await session.execute(
                    update(table)
                    .where(
                        tuple_(table.incarnation, table.segment_uuid).in_(
                            oldest_claimable
                        )
                    )
                    .values(
                        attempts_without_progress=table.attempts_without_progress + 1,
                        claimable_at=_seconds_from_now(
                            lease_seconds
                            + self._claim_retry_backoff_seconds(
                                table.attempts_without_progress + 1
                            ),
                            is_sqlite=self._is_sqlite,
                        ),
                        claim_generation=table.claim_generation + 1,
                    )
                    .returning(*table.__table__.columns)
                )
            ).one_or_none()
            if claimed is not None:
                return claimed
        return None

    async def _schedule_after_failed_claim(
        self, session: AsyncSession, claim: PendingWriteClaim | TombstoneClaim
    ) -> bool:
        """Make the claimed record claimable after the backoff, if the claim is the latest on it; whether it was."""
        table, claimed_record = _claimed_record(claim)
        ended = await (await session.connection()).execute(
            update(table)
            .where(claimed_record)
            .values(
                claimable_at=_seconds_from_now(
                    self._claim_retry_backoff_seconds(table.attempts_without_progress),
                    is_sqlite=self._is_sqlite,
                )
            )
        )
        return ended.rowcount == 1

    async def _take_back_cancelled_claim(
        self, session: AsyncSession, claim: PendingWriteClaim | TombstoneClaim
    ) -> None:
        """Take back the claimed record's attempt and make it claimable at once, if the claim is the latest on it."""
        table, claimed_record = _claimed_record(claim)
        await session.execute(
            update(table)
            .where(claimed_record)
            .values(
                claimable_at=func.now(),
                attempts_without_progress=table.attempts_without_progress - 1,
            )
        )

    def _claim_retry_backoff_seconds(
        self, attempts: ColumnElement[int] | InstrumentedAttribute[int]
    ) -> ColumnElement[int]:
        """The claim retry backoff in seconds after `attempts` claims, computed by the database."""
        # 1 << (a - 1) is 2 ** (a - 1) on both dialects.
        doublings = literal(1, Integer).op("<<")(attempts - 1)
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

    # Helpers

    async def _load_payload_codec(
        self,
        config: SegmentStorePartitionConfig,
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
    ) -> SQLAlchemySegmentStorePartition:
        """Materialize a partition handle from a registry row."""
        config = SegmentStorePartitionConfig(
            payload_codec_config=decode_payload_codec_config(
                partition_row.payload_codec_config
            )
        )
        payload_codec = await self._load_payload_codec(config)
        return SQLAlchemySegmentStorePartition(
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
        config: SegmentStorePartitionConfig,
    ) -> None:
        """Raise if an existing partition row does not match the requested config."""
        existing_config = SegmentStorePartitionConfig(
            payload_codec_config=decode_payload_codec_config(
                partition_row.payload_codec_config
            )
        )
        if existing_config != config:
            raise SegmentStorePartitionConfigMismatchError(
                partition_row.partition_key,
                existing_config,
                config,
            )


def _claimed_record(
    claim: PendingWriteClaim | TombstoneClaim,
) -> tuple[type[PendingWriteRow] | type[TombstoneRow], ColumnElement[bool]]:
    """The table of the record `claim` took, and that record while the claim is the latest on it."""
    if isinstance(claim, PendingWriteClaim):
        table, segment_uuid = PendingWriteRow, claim.segment_uuid
    else:
        table, segment_uuid = TombstoneRow, claim.tombstone.segment_uuid
    return table, and_(
        table.incarnation == claim.incarnation,
        table.segment_uuid == segment_uuid,
        table.claim_generation == claim.generation,
    )


def _seconds_from_now(
    seconds: int | ColumnElement[int], *, is_sqlite: bool
) -> ColumnElement[datetime]:
    """The time `seconds` from now, on the database clock."""
    if is_sqlite:
        return func.datetime(
            "now",
            func.printf("+%d seconds", seconds),
            type_=DateTime(timezone=True),
        )
    return func.now() + literal(timedelta(seconds=1), Interval) * seconds


def _derivative_from_derivative_row(
    row: DerivativeRow, payload_codec: PayloadCodec
) -> Derivative:
    """Convert a DerivativeRow into a Derivative."""
    context = decode_context(json.loads(payload_codec.decode(row.context)))
    if context is None:
        context = NullContext()
    original_timezone = timezone(timedelta(seconds=row.timestamp_timezone_offset))
    return Derivative(
        uuid=row.uuid,
        segment_uuid=row.segment_uuid,
        timestamp=ensure_tz_aware(row.timestamp).astimezone(original_timezone),
        context=context,
        block=decode_block(json.loads(payload_codec.decode(row.block))),
        properties=decode_properties(row.properties),
    )
