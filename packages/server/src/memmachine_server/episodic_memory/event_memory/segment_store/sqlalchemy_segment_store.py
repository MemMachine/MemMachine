"""SQLAlchemy implementation of the SegmentStore interface."""

import json
import logging
import sqlite3
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta, timezone
from typing import override
from uuid import UUID

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
    SQLColumnExpression,
    Tuple,
    Uuid,
    and_,
    bindparam,
    case,
    delete,
    func,
    insert,
    literal,
    literal_column,
    or_,
    select,
    true,
    tuple_,
    update,
)
from sqlalchemy.dialects.postgresql import ARRAY, JSONB
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
    SegmentStoreEventAlreadyExistsError,
    SegmentStorePartitionConfig,
    SegmentStorePartitionConfigMismatchError,
)
from memmachine_server.episodic_memory.event_memory.segment_store.segment_store import (
    IndexingJobClaim,
    SegmentStore,
    SegmentStorePartition,
    TombstoneClaim,
)

logger = logging.getLogger(__name__)

_JSON_AUTO = JSON().with_variant(JSONB, "postgresql")

# Adding events and claiming indexing jobs and tombstones depend on
# RETURNING, which SQLite added in 3.35.
_MIN_SQLITE_VERSION = (3, 35)

# A context read takes its context from at most this many segments on each
# side of a seed, matching or not.
_MAX_CONTEXT_DISTANCE = 1_000

# Failed claims in a row an indexing job or a tombstone gets
# before it is no longer claimed.
_MAX_CLAIM_ATTEMPTS_WITHOUT_PROGRESS = 10


# ORM models


class BaseSegmentStore(DeclarativeBase):
    """Base class for segment store tables."""


class PartitionRow(BaseSegmentStore):
    """One row per created partition.

    The row holds the configuration every handle on the partition uses and
    the purge's cursor on each table it walks: the segments', the events',
    and the tombstones'. The current purge walk passed the partition's rows
    of a table keyed at or below the table's cursor, deleting them. A cursor
    is NULL until the walk records one, and again once the walk reaches the
    end of every table. Dropping the partition deletes the row.
    """

    __tablename__ = "segment_store_pt"

    partition_uuid: MappedColumn[UUID] = mapped_column(Uuid, primary_key=True)
    payload_codec_config: MappedColumn[dict[str, JsonValue]] = mapped_column(
        _JSON_AUTO,
        nullable=False,
    )
    segments_purged_through: MappedColumn[UUID | None] = mapped_column(
        Uuid, nullable=True
    )
    events_purged_through: MappedColumn[UUID | None] = mapped_column(
        Uuid, nullable=True
    )
    tombstones_purged_through: MappedColumn[UUID | None] = mapped_column(
        Uuid, nullable=True
    )


class EventRow(BaseSegmentStore):
    """One row per event the partition holds.

    The primary key is what makes an event addable once: a second
    `add_event_segments` naming it conflicts here, before any segment is
    written. Deleting the row cascades to the event's segments, and through
    them to their derivatives and indexing jobs.
    """

    __tablename__ = "segment_store_ev"

    partition_uuid: MappedColumn[UUID] = mapped_column(Uuid, primary_key=True)
    uuid: MappedColumn[UUID] = mapped_column(Uuid, primary_key=True)


class SegmentRow(BaseSegmentStore):
    """Persisted segment."""

    __tablename__ = "segment_store_sg"

    partition_uuid: MappedColumn[UUID] = mapped_column(Uuid, primary_key=True)

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

    # No foreign key to the partition's row: the purge deletes a partition's
    # segments in bounded batches, and the drop deletes the row alone.
    __table_args__ = (
        ForeignKeyConstraint(
            ["partition_uuid", "event_uuid"],
            [
                "segment_store_ev.partition_uuid",
                "segment_store_ev.uuid",
            ],
            ondelete="CASCADE",
        ),
        # Serves the event lookups and the cascade from the event row.
        Index(
            "segment_store_sg__pu_ev",
            "partition_uuid",
            "event_uuid",
        ),
        Index(
            "segment_store_sg__pu_ts_ev_ix_of",
            "partition_uuid",
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

    partition_uuid: MappedColumn[UUID] = mapped_column(Uuid, primary_key=True)

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
            ["partition_uuid", "segment_uuid"],
            [
                "segment_store_sg.partition_uuid",
                "segment_store_sg.uuid",
            ],
            ondelete="CASCADE",
        ),
        # Serves the lookups by segment and the cascade from the segment row.
        Index(
            "segment_store_dv__pu_su",
            "partition_uuid",
            "segment_uuid",
        ),
    )


class IndexingJobRow(BaseSegmentStore):
    """A segment whose derivatives await their derived index.

    Written with the segment; its ingest deletes it once the index has the
    segment's derivatives, and a claim takes it when the ingest does not.
    Deleted with its segment, so a forgotten event leaves none behind.
    """

    __tablename__ = "segment_store_ij"

    partition_uuid: MappedColumn[UUID] = mapped_column(Uuid, primary_key=True)
    segment_uuid: MappedColumn[UUID] = mapped_column(Uuid, primary_key=True)
    # When a claim may take the record: once its ingest's lease has passed,
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
            ["partition_uuid", "segment_uuid"],
            [
                "segment_store_sg.partition_uuid",
                "segment_store_sg.uuid",
            ],
            ondelete="CASCADE",
        ),
        Index("segment_store_ij__ca", "claimable_at"),
    )


class TombstoneRow(BaseSegmentStore):
    """A forgotten segment's tombstone, holding the UUIDs of its derivatives.

    Their derived index entries are deleted at once, and once more after
    the store's retention has passed since the first delete, which then
    deletes the tombstone and, by cascade, the UUIDs.
    """

    __tablename__ = "segment_store_tb"

    partition_uuid: MappedColumn[UUID] = mapped_column(Uuid, primary_key=True)
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

    partition_uuid: MappedColumn[UUID] = mapped_column(Uuid, primary_key=True)
    uuid: MappedColumn[UUID] = mapped_column(Uuid, primary_key=True)
    segment_uuid: MappedColumn[UUID] = mapped_column(Uuid, nullable=False)

    __table_args__ = (
        ForeignKeyConstraint(
            ["partition_uuid", "segment_uuid"],
            [
                "segment_store_tb.partition_uuid",
                "segment_store_tb.segment_uuid",
            ],
            ondelete="CASCADE",
        ),
        # Serves the tombstone claims' reads and the cascade from the
        # tombstone row.
        Index(
            "segment_store_tb_dv__pu_su",
            "partition_uuid",
            "segment_uuid",
        ),
    )


class SQLAlchemySegmentStorePartition(SegmentStorePartition):
    """SQLAlchemy-backed partition handle."""

    def __init__(
        self,
        partition_uuid: UUID,
        create_session: async_sessionmaker[AsyncSession],
        is_sqlite: bool,
        config: SegmentStorePartitionConfig,
        payload_codec: PayloadCodec,
        tracker: OperationTracker,
    ) -> None:
        """Initialize with a partition and the store's session factory."""
        # Data rows are keyed by the partition's UUID, so the handle reaches
        # its partition's rows alone.
        self._partition_uuid = partition_uuid
        self._config = config
        self._payload_codec = payload_codec
        self._tracker = tracker
        self._create_session = create_session
        self._is_sqlite = is_sqlite

    @override
    @property
    def config(self) -> SegmentStorePartitionConfig:
        return self._config

    # Writes

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
            # A batch of several events can wait on another's insert while
            # holding its own, so batches take turns; one event waits only
            # before it holds any.
            await self._lock_partition_row(
                session, exclusive=len(segments_by_event) > 1
            )
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
            await self._insert_indexing_jobs(
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
                    {"partition_uuid": self._partition_uuid, "uuid": event_uuid}
                    for event_uuid in event_uuids
                ]
            )
            .on_conflict_do_nothing(index_elements=["partition_uuid", "uuid"])
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
                "partition_uuid": self._partition_uuid,
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
                "partition_uuid": self._partition_uuid,
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

    async def _insert_indexing_jobs(
        self,
        session: AsyncSession,
        segment_uuids: Iterable[UUID],
        lease_seconds: int,
    ) -> None:
        """Insert an indexing job per segment, claimable once `lease_seconds` have passed."""
        segment_uuids = list(segment_uuids)
        if not segment_uuids:
            return
        # One multi-row statement, so every row carries the database clock.
        await session.execute(
            insert(IndexingJobRow).values(
                [
                    {
                        "partition_uuid": self._partition_uuid,
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
    async def delete_indexing_jobs(
        self,
        segment_uuids: Iterable[UUID],
    ) -> None:
        segment_uuids = set(segment_uuids)
        if not segment_uuids:
            return

        async with (
            self._tracker("delete_indexing_jobs"),
            write_session(self._create_session) as session,
        ):
            # Skips the records another transaction holds, so this never
            # waits: a forget or the purge deletes them, and one a claim's
            # statement holds is left for the background work to write again.
            unheld_segment_uuids = (
                (
                    await session.execute(
                        self._restrict_to_keys(
                            select(IndexingJobRow.segment_uuid)
                            .where(
                                IndexingJobRow.partition_uuid == self._partition_uuid
                            )
                            .with_for_update(skip_locked=True),
                            IndexingJobRow.segment_uuid,
                            segment_uuids,
                        )
                    )
                )
                .scalars()
                .all()
            )
            if not unheld_segment_uuids:
                return
            # This transaction holds the records, so the deletes wait on no
            # other. One statement per record, each naming one key, so the
            # planner chooses its plan for that key alone; the statements run
            # on the session's connection, which executes a DELETE once per
            # parameter set.
            await (await session.connection()).execute(
                delete(IndexingJobRow).where(
                    IndexingJobRow.partition_uuid == self._partition_uuid,
                    IndexingJobRow.segment_uuid == bindparam("segment_uuid"),
                ),
                [
                    {"segment_uuid": segment_uuid}
                    for segment_uuid in unheld_segment_uuids
                ],
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
            segments = SegmentRow.__table__
            seed_segments_query = select(segments).where(
                segments.c.partition_uuid == self._partition_uuid,
            )
            if property_filter is not None:
                seed_segments_query = seed_segments_query.where(
                    compile_sql_filter(
                        property_filter,
                        lambda field: (
                            SQLAlchemySegmentStorePartition._resolve_segment_field(
                                field, columns=segments.c
                            )
                        ),
                    )
                )
            seed_segment_rows = (
                (
                    await session.execute(
                        select(SegmentRow).from_statement(
                            self._restrict_to_keys(
                                seed_segments_query,
                                segments.c.uuid,
                                seed_segment_uuids,
                            )
                        )
                    )
                )
                .scalars()
                .all()
            )

            seed_segment_rows_by_uuid: dict[UUID, SegmentRow] = {
                row.uuid: row for row in seed_segment_rows
            }
            if not seed_segment_rows_by_uuid:
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
        seeds_subquery = self._restrict_to_keys(
            select(
                SegmentRow.uuid.label("seed_uuid"),
                SegmentRow.timestamp.label("seed_timestamp"),
                SegmentRow.event_uuid.label("seed_event_uuid"),
                SegmentRow.index.label("seed_index"),
                SegmentRow.offset.label("seed_offset"),
            ).where(SegmentRow.partition_uuid == self._partition_uuid),
            SegmentRow.uuid,
            seed_rows_by_uuid.keys(),
        ).subquery("seeds")
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
                segments.partition_uuid == self._partition_uuid,
                segment_ordering_columns < seed_ordering_values
                if backward
                else segment_ordering_columns > seed_ordering_values,
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
            # Served by the primary key, which leads on the same two columns
            # this filters.
            query = self._restrict_to_keys(
                select(DerivativeRow.uuid, DerivativeRow.segment_uuid).where(
                    DerivativeRow.partition_uuid == self._partition_uuid
                ),
                DerivativeRow.uuid,
                derivative_uuids,
            )
            rows = (await session.execute(query)).all()

        return {row.uuid: row.segment_uuid for row in rows}

    # Forgetting

    @override
    async def delete_event_segments(self, event_uuids: Iterable[UUID]) -> None:
        event_uuids = set(event_uuids)
        if not event_uuids:
            return

        async with (
            self._tracker("delete_event_segments"),
            write_session(self._create_session) as session,
        ):
            # A forget of several events can wait on one event while holding
            # another, so such forgets take turns; one event waits only
            # before it holds any. A dropped partition has no row, and its
            # events are left to the purge, which has no row to lock either.
            if not await self._lock_partition_row(
                session, exclusive=len(event_uuids) > 1
            ):
                return
            # Every later statement acts on the events this one returned:
            # each statement reads a snapshot of its own on PostgreSQL, so
            # an event committed after this one would be deleted below with
            # derivatives nobody read. SQLite drops the locking clause and
            # serializes write transactions instead.
            held_event_uuids = list(
                (
                    await session.execute(
                        self._restrict_to_keys(
                            select(EventRow.uuid)
                            .where(EventRow.partition_uuid == self._partition_uuid)
                            .with_for_update(),
                            EventRow.uuid,
                            event_uuids,
                        )
                    )
                ).scalars()
            )
            if not held_event_uuids:
                return
            # Reached only through the events held above, so no other
            # transaction deletes them meanwhile.
            segment_uuids = list(
                (
                    await session.execute(
                        self._restrict_to_keys(
                            select(SegmentRow.uuid).where(
                                SegmentRow.partition_uuid == self._partition_uuid
                            ),
                            SegmentRow.event_uuid,
                            held_event_uuids,
                        )
                    )
                ).scalars()
            )
            if segment_uuids:
                await session.execute(
                    insert(TombstoneRow).values(
                        [
                            {
                                "partition_uuid": self._partition_uuid,
                                "segment_uuid": segment_uuid,
                                "claimable_at": func.now(),
                            }
                            for segment_uuid in segment_uuids
                        ]
                    )
                )
                # Each tombstone copies its segment's derivative UUIDs, found
                # by the segments read above, which the derivatives' index
                # leads on.
                await session.execute(
                    insert(TombstoneDerivativeRow).from_select(
                        ["partition_uuid", "uuid", "segment_uuid"],
                        self._restrict_to_keys(
                            select(
                                DerivativeRow.partition_uuid,
                                DerivativeRow.uuid,
                                DerivativeRow.segment_uuid,
                            ).where(
                                DerivativeRow.partition_uuid == self._partition_uuid
                            ),
                            DerivativeRow.segment_uuid,
                            segment_uuids,
                        ),
                    )
                )

            # The cascade follows to the events' segments, and through them
            # to their indexing jobs and derivatives. One statement per
            # event, each naming one key, so the planner chooses its plan for
            # that key alone; this transaction holds the events already.
            await (await session.connection()).execute(
                delete(EventRow).where(
                    EventRow.partition_uuid == self._partition_uuid,
                    EventRow.uuid == bindparam("event_uuid"),
                ),
                [{"event_uuid": event_uuid} for event_uuid in held_event_uuids],
            )

    # Helpers

    async def _lock_partition_row(
        self, session: AsyncSession, *, exclusive: bool
    ) -> bool:
        """Lock the partition's row for the rest of the transaction, before a write locks any of the partition's events; whether the row exists.

        The lock is `FOR KEY SHARE`, which waits only for the purge, under
        `FOR UPDATE`, which locks the row before any row it deletes: so a
        write and the purge take turns. With `exclusive`, it is `FOR NO KEY
        UPDATE`, which also waits for every other exclusive lock: a write
        that locks several of the partition's events takes it, so such
        writes take turns as well. SQLite drops the locking clause; there a
        write transaction holds the database's write lock from `BEGIN`.
        """
        partition_row = (
            await session.execute(
                select(PartitionRow.partition_uuid)
                .where(PartitionRow.partition_uuid == self._partition_uuid)
                .with_for_update(read=not exclusive, key_share=True)
            )
        ).one_or_none()
        return partition_row is not None

    def _restrict_to_keys(
        self,
        rows: Select,
        key_column: SQLColumnExpression[UUID],
        keys: Iterable[UUID],
    ) -> Select:
        """Select the `rows` whose `key_column` is one of `keys`.

        `rows` selects from this partition's rows of one table. `keys` holds
        each key once.

        On PostgreSQL the statement joins each key to a subquery of that
        key's rows, so the planner chooses a plan for one key at a time.
        Given the keys as one `IN` list, the planner would choose one plan
        for the whole list, and where it estimates the partition to hold
        few rows, that plan would scan all the partition's rows, however
        many the partition holds. For one key, the planner estimates the
        lookup to cost less than the scan, so the statement's cost follows
        the number of keys, unless the planner estimates the partition to
        hold one row, in which case it estimates the same cost for both and
        may scan all the partition's rows for each key.
        """
        if self._is_sqlite:
            return rows.where(key_column.in_(keys))
        key_table = (
            func.unnest(literal(list(keys), ARRAY(Uuid)))
            .table_valued("key")
            .render_derived(name="keys")
        )
        # The OFFSET keeps the planner from merging the subquery into the
        # statement, where it would choose one plan for all the keys. Its
        # zero is a literal: PostgreSQL plans a Limit node for an OFFSET
        # given as a parameter.
        rows_of_key = (
            rows.where(key_column == key_table.c.key)
            .offset(literal_column("0"))
            .lateral("rows")
        )
        return select(rows_of_key).select_from(key_table.join(rows_of_key, true()))

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
            (enable_sqlite_foreign_keys): the store relies on its tables'
            cascades and does not verify enforcement. It must also begin
            its transactions explicitly, set up at engine creation
            (enable_sqlite_explicit_begin): the store's write transactions
            take SQLite's write lock at BEGIN.
        metrics_factory (MetricsFactory | None):
            An instance of MetricsFactory for collecting usage metrics
            (default: None).
        purge_max_rows (int):
            Maximum number of segment, event, and tombstone rows purged per
            call. A segment's derivatives and indexing job, and a
            tombstone's derivative UUIDs, are deleted by cascade, so a
            call's transaction scales with the deployment's derivatives per
            segment; size the bound with that fan-out in mind (default:
            10000).
        partition_config (SegmentStorePartitionConfig):
            The configuration of each partition the store creates, recorded
            with the partition (default: plaintext payloads).
        tombstone_retention_seconds (int):
            Seconds after a forget before its tombstone's second delete is
            due, on the database clock. It must exceed, by orders of
            magnitude, the longest a write of a forgotten derivative to its
            index can land after the forget (default: 86400).
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
            "(enable_sqlite_foreign_keys): the store relies on its tables' "
            "cascades and does not verify enforcement. It must also begin "
            "its transactions explicitly, set up at engine creation "
            "(enable_sqlite_explicit_begin): the store's write transactions "
            "take SQLite's write lock at BEGIN"
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
            "Maximum number of segment, event, and tombstone rows purged per "
            "call. A segment's derivatives and indexing job, and a "
            "tombstone's derivative UUIDs, are deleted by cascade, so a "
            "call's transaction scales with the deployment's derivatives per "
            "segment; size the bound with that fan-out in mind"
        ),
    )
    partition_config: SegmentStorePartitionConfig = Field(
        default_factory=SegmentStorePartitionConfig,
        description=(
            "The configuration of each partition the store creates, recorded "
            "with the partition"
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
                "support adding events and claiming indexing jobs and "
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
        self._partition_config = params.partition_config
        self._tombstone_retention_seconds = params.tombstone_retention_seconds
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

    # Partitions

    @override
    async def create_partition(self, partition_uuid: UUID) -> None:
        config = self._partition_config
        async with self._tracker("create_partition"):
            # Materialized before the insert so an unloadable codec config
            # fails without writing a row no handle could use.
            await self._load_payload_codec(config)
            try:
                async with write_session(self._create_session) as session:
                    await session.execute(
                        insert(PartitionRow).values(
                            partition_uuid=partition_uuid,
                            payload_codec_config=encode_payload_codec_config(
                                config.payload_codec_config
                            ),
                        )
                    )
            except IntegrityError:
                # A conflicting row means a repeat creation, which succeeds
                # with the same configuration.
                partition_row = await self._get_partition_row(partition_uuid)
                if partition_row is None:
                    raise
                SQLAlchemySegmentStore._raise_if_partition_config_mismatch(
                    partition_row, config
                )

    @override
    async def get_partition(
        self, partition_uuid: UUID
    ) -> SQLAlchemySegmentStorePartition | None:
        async with self._tracker("get_partition"):
            partition_row = await self._get_partition_row(partition_uuid)
            if partition_row is None:
                return None
            config = SegmentStorePartitionConfig(
                payload_codec_config=decode_payload_codec_config(
                    partition_row.payload_codec_config
                )
            )
            return SQLAlchemySegmentStorePartition(
                partition_uuid=partition_uuid,
                create_session=self._create_session,
                is_sqlite=self._is_sqlite,
                config=config,
                payload_codec=await self._load_payload_codec(config),
                tracker=self._tracker,
            )

    @override
    async def purge_partition(self, partition_uuid: UUID) -> bool:
        # One transaction per call: the call purges up to the bound and
        # commits, or a raise rolls the whole call back, cursors included, so
        # a call that fails on contention is safe to repeat. Purgers of one
        # partition serialize on the partition row's lock, so each reads the
        # cursors its predecessor committed; on SQLite, which drops locking
        # clauses, the call holds the write lock from BEGIN.
        #
        # Segments come before events, so deleting an event cascades to no
        # segment. Deleting a segment cascades to its derivatives and indexing
        # job, and deleting a tombstone to its derivative UUIDs.
        walk = (
            (SegmentRow, SegmentRow.uuid, PartitionRow.segments_purged_through),
            (EventRow, EventRow.uuid, PartitionRow.events_purged_through),
            (
                TombstoneRow,
                TombstoneRow.segment_uuid,
                PartitionRow.tombstones_purged_through,
            ),
        )
        async with (
            self._tracker("purge_partition"),
            write_connection(self._engine) as connection,
        ):
            partition_row = (
                await connection.execute(
                    select(*(cursor for _, _, cursor in walk))
                    .where(PartitionRow.partition_uuid == partition_uuid)
                    .with_for_update()
                )
            ).one_or_none()
            cursors = (
                [None] * len(walk) if partition_row is None else list(partition_row)
            )

            remaining = self._purge_max_rows
            passed: dict[str, UUID] = {}
            for (table, key, cursor), purged_through in zip(walk, cursors, strict=True):
                found, last = await SQLAlchemySegmentStore._purge_batch(
                    connection,
                    table,
                    key,
                    partition_uuid,
                    after=purged_through,
                    limit=remaining,
                )
                if last is not None:
                    passed[cursor.key] = last
                if found >= remaining:
                    # The search found a full batch; the partition may have
                    # more rows, so record how far the walk got for the next
                    # call.
                    if partition_row is not None:
                        await connection.execute(
                            update(PartitionRow)
                            .where(PartitionRow.partition_uuid == partition_uuid)
                            .values(passed)
                        )
                    return True
                remaining -= found

            if any(purged_through is not None for purged_through in cursors):
                # The walk from the cursors reached the end of every table. The
                # next call searches from each table's first key, so a row
                # written behind a cursor is found before a call returns False.
                await connection.execute(
                    update(PartitionRow)
                    .where(PartitionRow.partition_uuid == partition_uuid)
                    .values({cursor.key: None for _, _, cursor in walk})
                )
                return True
            return bool(passed)

    @staticmethod
    async def _purge_batch(
        connection: AsyncConnection,
        table: type[SegmentRow | EventRow | TombstoneRow],
        key: InstrumentedAttribute[UUID],
        partition_uuid: UUID,
        *,
        after: UUID | None,
        limit: int,
    ) -> tuple[int, UUID | None]:
        """Delete up to `limit` of a partition's rows of `table` in `key` order after `after`.

        `key` is the column of the table's primary key after the partition's
        UUID. Returns how many rows the batch's search found and the last key
        it found: every decision rests on what the search found, since another
        transaction may delete some of the batch first. The batch continues
        after the cursor, so it reads no row an earlier batch of its walk
        deleted: PostgreSQL keeps deleted rows in the table and its indexes
        until vacuum. A walk's first batch, from the partition's first key,
        steps over the rows earlier walks deleted.
        """
        after_cursor = [] if after is None else [key > after]
        batch = (
            select(key)
            .where(table.partition_uuid == partition_uuid, *after_cursor)
            .order_by(key)
            .limit(limit)
            .subquery()
        )
        # The batch's size and its last key, in one statement.
        row = (
            await connection.execute(
                select(batch.c[key.key], func.count().over())
                .order_by(batch.c[key.key].desc())
                .limit(1)
            )
        ).one_or_none()
        if row is None:
            return 0, None
        last, found = row
        await connection.execute(
            delete(table).where(
                table.partition_uuid == partition_uuid,
                *after_cursor,
                key <= last,
            )
        )
        return found, last

    @override
    async def drop_partition(self, partition_uuid: UUID) -> None:
        async with (
            self._tracker("drop_partition"),
            write_connection(self._engine) as connection,
        ):
            await connection.execute(
                delete(PartitionRow).where(
                    PartitionRow.partition_uuid == partition_uuid
                )
            )

    # Claims

    @override
    async def claim_indexing_job(
        self,
        *,
        lease_seconds: int,
    ) -> IndexingJobClaim | None:
        oldest_due = (
            select(IndexingJobRow.partition_uuid, IndexingJobRow.segment_uuid)
            .where(IndexingJobRow.claimable_at <= func.now())
            .order_by(IndexingJobRow.claimable_at)
        )
        async with (
            self._tracker("claim_indexing_job"),
            write_session(self._create_session) as session,
        ):
            claimed = await self._claim_oldest_due(
                session, IndexingJobRow, [oldest_due], lease_seconds=lease_seconds
            )
            if claimed is None:
                return None
            payload_codec_config = (
                await session.execute(
                    select(PartitionRow.payload_codec_config).where(
                        PartitionRow.partition_uuid == claimed.partition_uuid
                    )
                )
            ).scalar_one_or_none()
            derivative_rows = (
                (
                    await session.execute(
                        select(DerivativeRow).where(
                            DerivativeRow.partition_uuid == claimed.partition_uuid,
                            DerivativeRow.segment_uuid == claimed.segment_uuid,
                        )
                    )
                )
                .scalars()
                .all()
            )
        if payload_codec_config is None:
            # Dropped: the purge before the drop deleted the partition's
            # records, so this one was written after it. The claim ends with
            # its lease.
            return None
        payload_codec = await self._load_payload_codec(
            SegmentStorePartitionConfig(
                payload_codec_config=decode_payload_codec_config(payload_codec_config)
            )
        )
        return _SQLAlchemyIndexingJobClaim(
            partition_uuid=claimed.partition_uuid,
            segment_uuid=claimed.segment_uuid,
            derivatives=[
                _derivative_from_derivative_row(row, payload_codec)
                for row in derivative_rows
            ],
            attempt=claimed.attempts_without_progress,
            claim_generation=claimed.claim_generation,
            engine=self._engine,
            tracker=self._tracker,
            base_claim_retry_backoff_seconds=self._base_claim_retry_backoff_seconds,
            max_claim_retry_backoff_seconds=self._max_claim_retry_backoff_seconds,
        )

    @override
    async def claim_tombstone(self, *, lease_seconds: int) -> TombstoneClaim | None:
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
            select(TombstoneRow.partition_uuid, TombstoneRow.segment_uuid)
            .where(TombstoneRow.claimable_at <= func.now(), retention_passed)
            .order_by(TombstoneRow.claimable_at)
        )
        second_deletes_due = (
            select(TombstoneRow.partition_uuid, TombstoneRow.segment_uuid)
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
                lease_seconds=lease_seconds,
            )
            if claimed is None:
                return None
            derivative_uuids = list(
                (
                    await session.execute(
                        select(TombstoneDerivativeRow.uuid).where(
                            TombstoneDerivativeRow.partition_uuid
                            == claimed.partition_uuid,
                            TombstoneDerivativeRow.segment_uuid == claimed.segment_uuid,
                        )
                    )
                ).scalars()
            )
        return _SQLAlchemyTombstoneClaim(
            partition_uuid=claimed.partition_uuid,
            segment_uuid=claimed.segment_uuid,
            derivative_uuids=derivative_uuids,
            second_delete=claimed.first_deleted_at is not None,
            attempt=claimed.attempts_without_progress,
            claim_generation=claimed.claim_generation,
            engine=self._engine,
            tracker=self._tracker,
            base_claim_retry_backoff_seconds=self._base_claim_retry_backoff_seconds,
            max_claim_retry_backoff_seconds=self._max_claim_retry_backoff_seconds,
        )

    async def _claim_oldest_due(
        self,
        session: AsyncSession,
        table: type[IndexingJobRow] | type[TombstoneRow],
        due_queries: Iterable[Select[tuple[UUID, UUID]]],
        *,
        lease_seconds: int,
    ) -> Row | None:
        """Claim the first claimable record of `table` that the first of `due_queries` with one yields.

        Each query selects the keys of due records in the order they are
        claimed. A due record is claimable unless its claims failed the
        maximum number of times in a row. The claim holds the record for
        `lease_seconds` and then the backoff, and returns the record's row
        as the claim left it.
        """
        for due_query in due_queries:
            # A materialized CTE is evaluated once per statement, so the
            # update takes exactly the one record it locked, and the claim
            # holds no other lock.
            oldest_claimable = (
                due_query.where(
                    table.attempts_without_progress
                    < _MAX_CLAIM_ATTEMPTS_WITHOUT_PROGRESS,
                )
                .limit(1)
                # SQLite drops this; there the write transaction serializes
                # claims.
                .with_for_update(skip_locked=True)
                .cte("oldest_claimable")
                .prefix_with("MATERIALIZED")
            )
            claimed = (
                await session.execute(
                    update(table)
                    .where(
                        table.partition_uuid == oldest_claimable.c.partition_uuid,
                        table.segment_uuid == oldest_claimable.c.segment_uuid,
                    )
                    .values(
                        attempts_without_progress=table.attempts_without_progress + 1,
                        claimable_at=_seconds_from_now(
                            lease_seconds
                            + _claim_retry_backoff_seconds(
                                table.attempts_without_progress + 1,
                                base_seconds=self._base_claim_retry_backoff_seconds,
                                max_seconds=self._max_claim_retry_backoff_seconds,
                                is_sqlite=self._is_sqlite,
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

    async def _get_partition_row(self, partition_uuid: UUID) -> PartitionRow | None:
        """Fetch a partition's row."""
        async with self._create_session() as session:
            return (
                await session.execute(
                    select(PartitionRow).where(
                        PartitionRow.partition_uuid == partition_uuid
                    )
                )
            ).scalar_one_or_none()

    @staticmethod
    def _raise_if_partition_config_mismatch(
        partition_row: PartitionRow,
        config: SegmentStorePartitionConfig,
    ) -> None:
        """Raise if a created partition's configuration differs from the requested one."""
        existing_config = SegmentStorePartitionConfig(
            payload_codec_config=decode_payload_codec_config(
                partition_row.payload_codec_config
            )
        )
        if existing_config != config:
            raise SegmentStorePartitionConfigMismatchError(
                partition_row.partition_uuid,
                existing_config,
                config,
            )


@dataclass(frozen=True)
class _SQLAlchemyIndexingJobClaim(IndexingJobClaim):
    """An indexing job claim whose methods write the store's tables directly."""

    claim_generation: int
    engine: AsyncEngine = field(repr=False, compare=False)
    tracker: OperationTracker = field(repr=False, compare=False)
    base_claim_retry_backoff_seconds: int = field(repr=False)
    max_claim_retry_backoff_seconds: int = field(repr=False)

    @override
    async def renew(self, *, lease_seconds: int) -> bool:
        async with (
            self.tracker("renew_indexing_job_claim"),
            write_connection(self.engine) as connection,
        ):
            # The update holds the record's row, as the claim did: a forget
            # deleting the record waits for this transaction, or this
            # update finds the record gone.
            renewed = (
                await connection.execute(
                    update(IndexingJobRow)
                    .where(_claimed_record(self))
                    .values(
                        claimable_at=_seconds_from_now(
                            lease_seconds
                            + _claim_retry_backoff_seconds(
                                IndexingJobRow.attempts_without_progress,
                                base_seconds=self.base_claim_retry_backoff_seconds,
                                max_seconds=self.max_claim_retry_backoff_seconds,
                                is_sqlite=self.engine.dialect.name == "sqlite",
                            ),
                            is_sqlite=self.engine.dialect.name == "sqlite",
                        )
                    )
                    .returning(IndexingJobRow.segment_uuid)
                )
            ).one_or_none()
        return renewed is not None

    @override
    async def finish(self) -> None:
        async with (
            self.tracker("finish_indexing_job_claim"),
            write_connection(self.engine) as connection,
        ):
            await connection.execute(
                delete(IndexingJobRow).where(
                    IndexingJobRow.partition_uuid == self.partition_uuid,
                    IndexingJobRow.segment_uuid == self.segment_uuid,
                )
            )

    @override
    async def fail(self, error: Exception) -> None:
        async with self.tracker("fail_indexing_job_claim"):
            ended = await _schedule_after_failed_claim(self)
        if ended and self.attempt >= _MAX_CLAIM_ATTEMPTS_WITHOUT_PROGRESS:
            logger.error(
                "Indexing job of segment %s in partition %s is no longer "
                "claimed after %d failed claims in a row; last error: %r. Set "
                "its attempts_without_progress to 0 in %s to claim it again.",
                self.segment_uuid,
                self.partition_uuid,
                self.attempt,
                error,
                IndexingJobRow.__tablename__,
            )


@dataclass(frozen=True)
class _SQLAlchemyTombstoneClaim(TombstoneClaim):
    """A tombstone claim whose methods write the store's tables directly."""

    claim_generation: int
    engine: AsyncEngine = field(repr=False, compare=False)
    tracker: OperationTracker = field(repr=False, compare=False)
    base_claim_retry_backoff_seconds: int = field(repr=False)
    max_claim_retry_backoff_seconds: int = field(repr=False)

    @override
    async def finish(self) -> None:
        async with (
            self.tracker("finish_tombstone_claim"),
            write_connection(self.engine) as connection,
        ):
            if self.second_delete:
                # The cascade deletes the tombstone's derivative UUIDs.
                await connection.execute(
                    delete(TombstoneRow).where(
                        TombstoneRow.partition_uuid == self.partition_uuid,
                        TombstoneRow.segment_uuid == self.segment_uuid,
                    )
                )
                return
            await connection.execute(
                update(TombstoneRow)
                .where(
                    TombstoneRow.partition_uuid == self.partition_uuid,
                    TombstoneRow.segment_uuid == self.segment_uuid,
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
    async def fail(self, error: Exception) -> None:
        async with self.tracker("fail_tombstone_claim"):
            ended = await _schedule_after_failed_claim(self)
        if ended and self.attempt >= _MAX_CLAIM_ATTEMPTS_WITHOUT_PROGRESS:
            logger.error(
                "Tombstone of segment %s in partition %s is no longer claimed "
                "after %d failed claims in a row, so its derivatives may stay "
                "in their index; last error: %r. Set its "
                "attempts_without_progress to 0 in %s to claim it again.",
                self.segment_uuid,
                self.partition_uuid,
                self.attempt,
                error,
                TombstoneRow.__tablename__,
            )


async def _schedule_after_failed_claim(
    claim: _SQLAlchemyIndexingJobClaim | _SQLAlchemyTombstoneClaim,
) -> bool:
    """Make the claimed record claimable after the backoff, if the claim is the latest on it; whether it was."""
    table = (
        IndexingJobRow
        if isinstance(claim, _SQLAlchemyIndexingJobClaim)
        else TombstoneRow
    )
    is_sqlite = claim.engine.dialect.name == "sqlite"
    async with write_connection(claim.engine) as connection:
        ended = await connection.execute(
            update(table)
            .where(_claimed_record(claim))
            .values(
                claimable_at=_seconds_from_now(
                    _claim_retry_backoff_seconds(
                        table.attempts_without_progress,
                        base_seconds=claim.base_claim_retry_backoff_seconds,
                        max_seconds=claim.max_claim_retry_backoff_seconds,
                        is_sqlite=is_sqlite,
                    ),
                    is_sqlite=is_sqlite,
                )
            )
        )
    return ended.rowcount == 1


def _claimed_record(
    claim: _SQLAlchemyIndexingJobClaim | _SQLAlchemyTombstoneClaim,
) -> ColumnElement[bool]:
    """The record `claim` took, while the claim is the latest on it."""
    table = (
        IndexingJobRow
        if isinstance(claim, _SQLAlchemyIndexingJobClaim)
        else TombstoneRow
    )
    return and_(
        table.partition_uuid == claim.partition_uuid,
        table.segment_uuid == claim.segment_uuid,
        table.claim_generation == claim.claim_generation,
    )


def _claim_retry_backoff_seconds(
    attempts: ColumnElement[int] | InstrumentedAttribute[int],
    *,
    base_seconds: int,
    max_seconds: int,
    is_sqlite: bool,
) -> ColumnElement[int]:
    """The claim retry backoff in seconds after `attempts` claims, computed by the database."""
    # 1 << (a - 1) is 2 ** (a - 1) on both dialects. The shift is clamped at
    # 0: an operator may reset a record's attempts while a claim on it is
    # open, and PostgreSQL leaves a negative shift undefined.
    doublings = literal(1, Integer).op("<<")(
        case((attempts > 0, attempts - 1), else_=0)
    )
    smaller = func.min if is_sqlite else func.least
    return smaller(base_seconds * doublings, max_seconds, type_=Integer)


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
