"""SQLAlchemy implementation of the SegmentStore interface."""

import json
import logging
from collections import defaultdict
from collections.abc import Iterable, Mapping
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
    ColumnCollection,
    ColumnElement,
    DateTime,
    ForeignKeyConstraint,
    Index,
    Integer,
    LargeBinary,
    Select,
    Tuple,
    Uuid,
    bindparam,
    delete,
    insert,
    select,
    true,
    tuple_,
    update,
)
from sqlalchemy.dialects.postgresql import JSONB
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
    NullContext,
    Segment,
    decode_block,
    decode_context,
    encode_block,
    encode_context,
)
from memmachine_server.episodic_memory.event_memory.segment_store.data_types import (
    SegmentStorePartitionConfig,
    SegmentStorePartitionConfigMismatchError,
)
from memmachine_server.episodic_memory.event_memory.segment_store.segment_store import (
    SegmentStore,
    SegmentStorePartition,
)

logger = logging.getLogger(__name__)

_JSON_AUTO = JSON().with_variant(JSONB, "postgresql")


# A context read takes its context from at most this many segments on each
# side of a seed, matching or not.
_MAX_CONTEXT_DISTANCE = 1_000


# ORM models


class BaseSegmentStore(DeclarativeBase):
    """Base class for segment store tables."""


class PartitionRow(BaseSegmentStore):
    """One row per created partition.

    It holds the configuration every handle on the partition uses, and,
    while the partition is purged, how far the purge got: every segment
    row of the partition keyed at or below purged_through is deleted;
    NULL until the purge's first full batch, and again once a walk from it
    reaches the end. Dropping the partition deletes the row.
    """

    __tablename__ = "segment_store_pt"

    partition_uuid: MappedColumn[UUID] = mapped_column(Uuid, primary_key=True)
    payload_codec_config: MappedColumn[dict[str, JsonValue]] = mapped_column(
        _JSON_AUTO,
        nullable=False,
    )
    purged_through: MappedColumn[UUID | None] = mapped_column(Uuid, nullable=True)


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
    # segments in bounded batches, and the drop deletes its row once the
    # purge finds none.
    __table_args__ = (
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


class DerivativeLinkRow(BaseSegmentStore):
    """Maps a derivative UUID to its owning segment."""

    __tablename__ = "segment_store_dv_ln"

    partition_uuid: MappedColumn[UUID] = mapped_column(Uuid, primary_key=True)

    uuid: MappedColumn[UUID] = mapped_column(Uuid, primary_key=True)
    segment_uuid: MappedColumn[UUID] = mapped_column(Uuid, nullable=False)

    __table_args__ = (
        ForeignKeyConstraint(
            ["partition_uuid", "segment_uuid"],
            [
                "segment_store_sg.partition_uuid",
                "segment_store_sg.uuid",
            ],
            ondelete="CASCADE",
        ),
        Index(
            "segment_store_dv_ln__pu_su",
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
        # Data rows are keyed by the partition's UUID: rows of another partition
        # of the partition are invisible to this one.
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

    # Registration

    @override
    async def add_segments(
        self,
        segments_to_derivative_uuids: Mapping[Segment, Iterable[UUID]],
    ) -> None:
        if not segments_to_derivative_uuids:
            return

        async with (
            self._tracker("add_segments"),
            write_session(self._create_session) as session,
        ):
            await self._insert_segments(session, segments_to_derivative_uuids.keys())
            await self._insert_derivative_links(session, segments_to_derivative_uuids)

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

    async def _insert_derivative_links(
        self,
        session: AsyncSession,
        segments_to_derivative_uuids: Mapping[Segment, Iterable[UUID]],
    ) -> None:
        """Insert derivative rows."""
        derivative_row_values = [
            {
                "uuid": derivative_uuid,
                "partition_uuid": self._partition_uuid,
                "segment_uuid": segment.uuid,
            }
            for segment, derivative_uuids in segments_to_derivative_uuids.items()
            for derivative_uuid in derivative_uuids
        ]
        if derivative_row_values:
            await session.execute(insert(DerivativeLinkRow), derivative_row_values)

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
                SegmentRow.partition_uuid == self._partition_uuid,
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
        seeds_subquery = (
            select(
                SegmentRow.uuid.label("seed_uuid"),
                SegmentRow.timestamp.label("seed_timestamp"),
                SegmentRow.event_uuid.label("seed_event_uuid"),
                SegmentRow.index.label("seed_index"),
                SegmentRow.offset.label("seed_offset"),
            )
            .where(
                SegmentRow.partition_uuid == self._partition_uuid,
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
    async def get_segment_uuids_by_event_uuids(
        self,
        event_uuids: Iterable[UUID],
    ) -> dict[UUID, list[UUID]]:
        event_uuids = set(event_uuids)
        if not event_uuids:
            return {}

        async with (
            self._tracker("get_segment_uuids_by_event_uuids"),
            self._create_session() as session,
        ):
            query = select(SegmentRow.event_uuid, SegmentRow.uuid).where(
                SegmentRow.partition_uuid == self._partition_uuid,
                SegmentRow.event_uuid.in_(event_uuids),
            )
            rows = (await session.execute(query)).all()

        result: defaultdict[UUID, list[UUID]] = defaultdict(list)
        for event_uuid, segment_uuid in rows:
            result[event_uuid].append(segment_uuid)
        return dict(result)

    @override
    async def get_derivative_uuids_by_segment_uuids(
        self,
        segment_uuids: Iterable[UUID],
    ) -> dict[UUID, list[UUID]]:
        segment_uuids = set(segment_uuids)
        if not segment_uuids:
            return {}

        async with (
            self._tracker("get_derivative_uuids_by_segment_uuids"),
            self._create_session() as session,
        ):
            query = select(
                DerivativeLinkRow.segment_uuid, DerivativeLinkRow.uuid
            ).where(
                DerivativeLinkRow.partition_uuid == self._partition_uuid,
                DerivativeLinkRow.segment_uuid.in_(segment_uuids),
            )
            rows = (await session.execute(query)).all()

        result: defaultdict[UUID, list[UUID]] = defaultdict(list)
        for segment_uuid, derivative_uuid in rows:
            result[segment_uuid].append(derivative_uuid)
        return dict(result)

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
                DerivativeLinkRow.partition_uuid == self._partition_uuid,
                DerivativeLinkRow.uuid.in_(derivative_uuids),
            )
            rows = (await session.execute(query)).all()

        return {row.uuid: row.segment_uuid for row in rows}

    # Deletion

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
            write_session(self._create_session) as session,
        ):
            if not self._is_sqlite:
                # Lock rows in deterministic order to prevent deadlocks
                # from concurrent deletions with overlapping UUID sets.
                # SQLite relies on write serialization by the database.
                await session.execute(
                    select(SegmentRow.uuid)
                    .where(
                        SegmentRow.partition_uuid == self._partition_uuid,
                        SegmentRow.uuid.in_(segment_uuids),
                    )
                    .order_by(SegmentRow.uuid)
                    .with_for_update()
                )

            # CASCADE deletes derivatives via FK.
            await session.execute(
                delete(SegmentRow).where(
                    SegmentRow.partition_uuid == self._partition_uuid,
                    SegmentRow.uuid.in_(segment_uuids),
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
            link-table cascade and does not verify enforcement. It must
            also begin its transactions explicitly, set up at engine
            creation (enable_sqlite_explicit_begin): the store's write
            transactions take SQLite's write lock at BEGIN.
        metrics_factory (MetricsFactory | None):
            An instance of MetricsFactory for collecting usage metrics
            (default: None).
        purge_max_segments (int):
            Maximum number of segment rows purged per call. Their
            derivative links follow by cascade, so a call's transaction
            also scales with the deployment's links per segment; size
            the bound with that fan-out in mind (default: 10000).
        partition_config (SegmentStorePartitionConfig):
            The configuration of each partition the store creates, recorded
            with the partition (default: plaintext payloads).
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
            "Maximum number of segment rows purged per call. Their "
            "derivative links follow by cascade, so a call's transaction "
            "also scales with the deployment's links per segment; size "
            "the bound with that fan-out in mind"
        ),
    )
    partition_config: SegmentStorePartitionConfig = Field(
        default_factory=SegmentStorePartitionConfig,
        description=(
            "The configuration of each partition the store creates, recorded "
            "with the partition"
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
        validate_engine_connections(engine)
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

        self._purge_max_segments = params.purge_max_segments
        self._partition_config = params.partition_config

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
                # Created before: the same configuration makes this a repeat.
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
        # One transaction per call: purge up to the bound and commit, or
        # nothing, so a call that fails on contention is safe to repeat: a
        # raise rolls the whole call back, cursor included. Purgers of one
        # partition serialize on its row's lock, so each reads the cursor
        # its predecessor committed; on SQLite, which drops locking clauses,
        # the call holds the write lock from BEGIN. Full rationale:
        # design/segment_store_shared_tables.md.
        #
        # Pure Core DML on an engine connection: unlike Session.execute,
        # AsyncConnection.execute is typed CursorResult, whose rowcount the
        # batch needs.
        async with (
            self._tracker("purge_partition"),
            write_connection(self._engine) as connection,
        ):
            partition_row = (
                await connection.execute(
                    select(PartitionRow.purged_through)
                    .where(PartitionRow.partition_uuid == partition_uuid)
                    .with_for_update()
                )
            ).one_or_none()
            purged_through = (
                None if partition_row is None else partition_row.purged_through
            )

            # The batch continues after the cursor, so it never reads rows an
            # earlier batch deleted: PostgreSQL keeps them in the table and
            # its indexes until vacuum, and a batch starting from the
            # partition's first key would step over all of them.
            after_cursor = (
                [] if purged_through is None else [SegmentRow.uuid > purged_through]
            )
            batch = (
                select(SegmentRow.uuid)
                .where(SegmentRow.partition_uuid == partition_uuid, *after_cursor)
                .order_by(SegmentRow.uuid)
                .limit(self._purge_max_segments)
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
                            SegmentRow.partition_uuid == partition_uuid,
                            *after_cursor,
                            SegmentRow.uuid <= last,
                        )
                    )
                ).rowcount

            if deleted >= self._purge_max_segments:
                # The bound was consumed; the partition may have more rows,
                # so record how far the purge got for the next call.
                if partition_row is not None:
                    await connection.execute(
                        update(PartitionRow)
                        .where(PartitionRow.partition_uuid == partition_uuid)
                        .values(purged_through=last)
                    )
                return True

            if purged_through is not None:
                # The walk from the cursor reached the end. The next call
                # searches from the partition's first key, so a write that
                # landed behind the cursor is found before the purge answers
                # that nothing remains.
                await connection.execute(
                    update(PartitionRow)
                    .where(PartitionRow.partition_uuid == partition_uuid)
                    .values(purged_through=None)
                )
                return True
            return deleted > 0

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
