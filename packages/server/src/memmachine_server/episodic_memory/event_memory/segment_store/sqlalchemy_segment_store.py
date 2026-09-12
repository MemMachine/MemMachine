"""SQLAlchemy implementation of the SegmentStore interface."""

import json
import logging
import sqlite3
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
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
    DateTime,
    ForeignKeyConstraint,
    Index,
    Integer,
    LargeBinary,
    Select,
    String,
    Text,
    Uuid,
    delete,
    false,
    func,
    insert,
    literal,
    select,
    true,
    tuple_,
    union_all,
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
    InstrumentedAttribute,
    MappedColumn,
    mapped_column,
)
from sqlalchemy.pool import StaticPool
from sqlalchemy.sql.elements import ColumnElement
from sqlalchemy.sql.selectable import Subquery

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
from memmachine_server.common.utils import ensure_tz_aware, utc_offset_seconds
from memmachine_server.episodic_memory.event_memory.data_types import (
    Neighborhood,
    Segment,
    decode_block,
    decode_context,
    encode_block,
    encode_context,
)
from memmachine_server.episodic_memory.event_memory.segment_store.data_types import (
    SegmentStoreAttemptsExhaustedError,
    SegmentStorePartitionAlreadyExistsError,
    SegmentStorePartitionConfig,
    SegmentStorePartitionConfigMismatchError,
    SegmentStorePartitionHandleStaleError,
)
from memmachine_server.episodic_memory.event_memory.segment_store.segment_store import (
    SegmentStore,
    SegmentStorePartition,
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

# Partition deletion depends on RETURNING, which SQLite added in 3.35.
_MIN_SQLITE_VERSION = (3, 35)


class _RegistryInsertRejectedError(Exception):
    """A registry insert was rejected; retry with a fresh incarnation.

    Raised when the insert fails with an integrity error but no row
    exists under the key, or when the minted incarnation still has
    garbage awaiting purge. Either way the fix is a fresh incarnation,
    retried up to `_MAX_MINT_ATTEMPTS`; a persistent failure raises
    `SegmentStoreAttemptsExhaustedError` with the database error
    chained.
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


class SegmentRow(BaseSegmentStore):
    """Persisted segment.

    `session_id`, `source_id` and `block_kind` are projections of the
    segment the row already holds in its encoded block and fields: the
    codec-encoded block is opaque to SQL, so what the store filters on is
    copied out beside it at insert. `block_kind` is derived from the block
    and never accepted as a separate input, so it cannot disagree with it.
    """

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
    session_id: MappedColumn[str] = mapped_column(Text, nullable=False)
    source_id: MappedColumn[str | None] = mapped_column(Text, nullable=True)
    context: MappedColumn[bytes] = mapped_column(LargeBinary, nullable=False)
    block_kind: MappedColumn[str] = mapped_column(Text, nullable=False)
    block: MappedColumn[bytes] = mapped_column(LargeBinary, nullable=False)
    properties: MappedColumn[dict[str, JsonValue]] = mapped_column(
        _JSON_AUTO, nullable=False, default=dict
    )

    # No foreign key to the registry: registry rows and data rows are
    # deliberately decoupled so that partition deletion is a registry write
    # (O(1)) and the purge queue reclaims data rows asynchronously.
    __table_args__ = (
        # Lookup by event, in the event's own order.
        Index(
            "segment_store_sg__in_ev_ix_of",
            "incarnation",
            "event_uuid",
            "index",
            "offset",
        ),
        # The one total order the store exposes, within a session: a walk
        # pins the session and follows it.
        Index(
            "segment_store_sg__in_se_ts_ev_ix_of",
            "incarnation",
            "session_id",
            "timestamp",
            "event_uuid",
            "index",
            "offset",
        ),
        Index(
            "segment_store_sg__in_so",
            "incarnation",
            "source_id",
        ),
    )


class DerivativeLinkRow(BaseSegmentStore):
    """Maps a derivative UUID to its owning segment."""

    __tablename__ = "segment_store_dv_ln"

    incarnation: MappedColumn[UUID] = mapped_column(Uuid, primary_key=True)

    uuid: MappedColumn[UUID] = mapped_column(Uuid, primary_key=True)
    segment_uuid: MappedColumn[UUID] = mapped_column(Uuid, nullable=False)

    __table_args__ = (
        ForeignKeyConstraint(
            ["incarnation", "segment_uuid"],
            [
                "segment_store_sg.incarnation",
                "segment_store_sg.uuid",
            ],
            ondelete="CASCADE",
        ),
        Index(
            "segment_store_dv_ln__in_su",
            "incarnation",
            "segment_uuid",
        ),
    )


class PurgeQueueRow(BaseSegmentStore):
    """The purge queue: one row per dead partition incarnation.

    Claimed oldest-first by the enqueue stamp, which is the database clock,
    so entries from every server order on one clock; entries stamped in the
    same tick are unordered among themselves. The incarnation identifies
    the rows to reclaim; the logical key is carried for forensics.
    """

    __tablename__ = "segment_store_gc"

    incarnation: MappedColumn[UUID] = mapped_column(Uuid, primary_key=True)
    partition_key: MappedColumn[str] = mapped_column(String(255), nullable=False)
    enqueued_at: MappedColumn[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )

    __table_args__ = (Index("segment_store_gc__ea", "enqueued_at"),)


@dataclass(frozen=True, slots=True)
class _SeedKey:
    """A segment's place in the store's order, and the session its walk stays in."""

    uuid: UUID
    timestamp: datetime
    event_uuid: UUID
    index: int
    offset: int
    session_id: str


def _seed_key(row: SegmentRow) -> _SeedKey:
    # The column holds the UTC instant, and SQLite compares the wall clock
    # it is given, so the key carries the instant in UTC.
    return _SeedKey(
        uuid=row.uuid,
        timestamp=ensure_tz_aware(row.timestamp).astimezone(UTC),
        event_uuid=row.event_uuid,
        index=row.index,
        offset=row.offset,
        session_id=row.session_id,
    )


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

    async def _lock_partition_for_write(self, session: AsyncSession) -> None:
        """Pin this incarnation's registry row; raise if the handle is stale.

        The shared row lock blocks concurrent deletion (which takes the
        exclusive row lock) until the write completes; the incarnation
        predicate fences a handle that outlived its partition. SQLite
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
                raise SegmentStorePartitionHandleStaleError(self._partition_key)
            return
        await self._ensure_partition_live(session, pin=True)

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
        deletion for the rest of the transaction. Reads call this without
        the lock, and only when their data statement returned no rows: it
        tells an empty partition from a stale handle.
        """
        query = self._registry_row_query()
        if pin:
            query = query.with_for_update(read=True)
        row = (await session.execute(query)).scalar_one_or_none()
        if row is None:
            raise SegmentStorePartitionHandleStaleError(self._partition_key)

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
            self._create_session() as session,
            session.begin(),
        ):
            await self._lock_partition_for_write(session)
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
                "incarnation": self._incarnation,
                "event_uuid": segment.event_uuid,
                "index": segment.index,
                "offset": segment.offset,
                # Store the UTC instant; SQLite does not persist tzinfo, so the
                # original offset is recorded separately and reapplied on read.
                "timestamp": segment.timestamp.astimezone(UTC),
                "timestamp_timezone_offset": utc_offset_seconds(segment.timestamp),
                "session_id": segment.session_id,
                "source_id": segment.source_id,
                "context": self._payload_codec.encode(
                    json.dumps(encode_context(segment.context)).encode("utf-8")
                ),
                "block_kind": segment.block.kind,
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
                "incarnation": self._incarnation,
                "segment_uuid": segment.uuid,
            }
            for segment, derivative_uuids in segments_to_derivative_uuids.items()
            for derivative_uuid in derivative_uuids
        ]
        if derivative_row_values:
            await session.execute(insert(DerivativeLinkRow), derivative_row_values)

    # Retrieval

    @override
    async def get_segments(
        self,
        segment_uuids: Iterable[UUID],
        *,
        since: datetime | None = None,
        until: datetime | None = None,
        session_ids: Iterable[str] | None = None,
        source_ids: Iterable[str] | None = None,
        block_kinds: Iterable[str] | None = None,
        property_filter: FilterExpr | None = None,
    ) -> dict[UUID, Segment]:
        segment_uuids = set(segment_uuids)
        if not segment_uuids:
            return {}
        conditions = SQLAlchemySegmentStorePartition._row_conditions(
            since, until, session_ids, source_ids, block_kinds, property_filter
        )

        async with (
            self._tracker("get_segments"),
            self._create_session() as session,
        ):
            rows_by_uuid = await self._rows_by_uuid(session, segment_uuids, conditions)
            if not rows_by_uuid:
                # The statement proves the partition live only when it
                # returns rows.
                await self._ensure_partition_live(session)
            return {
                segment_uuid: self._segment_from_segment_row(row)
                for segment_uuid, row in rows_by_uuid.items()
            }

    @override
    async def get_segment_neighborhoods(
        self,
        seed_segment_uuids: Iterable[UUID],
        *,
        before: int = 0,
        after: int = 0,
        since: datetime | None = None,
        until: datetime | None = None,
        source_ids: Iterable[str] | None = None,
        block_kinds: Iterable[str] | None = None,
        property_filter: FilterExpr | None = None,
    ) -> dict[UUID, Neighborhood]:
        seed_segment_uuids = set(seed_segment_uuids)
        if not seed_segment_uuids:
            return {}
        conditions = SQLAlchemySegmentStorePartition._row_conditions(
            since, until, None, source_ids, block_kinds, property_filter
        )

        async with (
            self._tracker("get_segment_neighborhoods"),
            self._create_session() as session,
        ):
            # The seed is an address, never part of the answer, so no
            # filter has anything to say about it.
            seed_rows_by_uuid = await self._rows_by_uuid(
                session, seed_segment_uuids, []
            )
            if not seed_rows_by_uuid:
                await self._ensure_partition_live(session)
                return {}
            seeds = [_seed_key(row) for row in seed_rows_by_uuid.values()]
            window_rows_by_seed = await self._window_rows(
                session, seeds, before, after, conditions
            )
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
                ) in window_rows_by_seed.items()
            }

    async def _rows_by_uuid(
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

    async def _window_rows(
        self,
        session: AsyncSession,
        seeds: Sequence[_SeedKey],
        before: int,
        after: int,
        conditions: Sequence[ColumnElement[bool]],
    ) -> dict[UUID, tuple[list[SegmentRow], list[SegmentRow]]]:
        """The rows before and after each seed in the store's order.

        Within the seed's session and filtered by `conditions`; strictly
        before and strictly after, so the seed is in neither list.
        """
        if before <= 0 and after <= 0:
            return {seed.uuid: ([], []) for seed in seeds}

        if not self._is_sqlite:
            window_rows_by_seed = await self._get_window_rows_lateral(
                session, seeds, before, after, conditions
            )
        else:
            window_rows_by_seed = await self._get_window_rows_loop(
                session, seeds, before, after, conditions
            )

        # Each statement above took its own snapshot (READ COMMITTED): a
        # deletion committing between them would leave a seed with
        # silently empty context. One registry read after the last
        # statement turns that window into the contractual stale-handle
        # error.
        await self._ensure_partition_live(session)
        return window_rows_by_seed

    @staticmethod
    def _row_conditions(
        since: datetime | None,
        until: datetime | None,
        session_ids: Iterable[str] | None,
        source_ids: Iterable[str] | None,
        block_kinds: Iterable[str] | None,
        property_filter: FilterExpr | None,
    ) -> list[ColumnElement[bool]]:
        """The row predicates of a read, on the typed system fields and the properties.

        Timestamp bounds are put in UTC before they are bound: the column
        holds the UTC instant, and SQLite compares the wall clock it is
        given, so a bound in another zone would be compared on its digits.
        A naive bound names no instant and is rejected. An empty id or
        kind list admits nothing.
        """
        conditions: list[ColumnElement[bool]] = []
        for name, bound in (("since", since), ("until", until)):
            if bound is not None and bound.tzinfo is None:
                raise ValueError(f"{name} must be timezone-aware: {bound!r}")
        if since is not None:
            conditions.append(SegmentRow.timestamp >= since.astimezone(UTC))
        if until is not None:
            conditions.append(SegmentRow.timestamp < until.astimezone(UTC))
        if session_ids is not None:
            conditions.append(_in_values(SegmentRow.session_id, session_ids))
        if source_ids is not None:
            conditions.append(_in_values(SegmentRow.source_id, source_ids))
        if block_kinds is not None:
            conditions.append(_in_values(SegmentRow.block_kind, block_kinds))
        if property_filter is not None:
            conditions.append(
                compile_sql_filter(
                    property_filter,
                    SQLAlchemySegmentStorePartition._resolve_segment_field,
                )
            )
        return conditions

    @staticmethod
    def _session_condition(session_id: str) -> ColumnElement[bool]:
        """The rows of a seed's session, as `=` on a value known before the statement is built, so the session-pinned ordering index serves it."""
        return SegmentRow.session_id == session_id

    async def _get_window_rows_lateral(
        self,
        session: AsyncSession,
        seeds: Sequence[_SeedKey],
        before: int,
        after: int,
        conditions: Sequence[ColumnElement[bool]],
    ) -> dict[UUID, tuple[list[SegmentRow], list[SegmentRow]]]:
        """Get backward/forward context using LATERAL joins (non-SQLite).

        The seeds are bound parameters in a row set, one pair of statements
        per distinct seed session, so the session predicate is a literal
        the session-pinned ordering index serves. Binding the seed keys, not
        rendering them, keeps the statements cacheable by shape.
        """
        rows_by_seed: dict[UUID, tuple[list[SegmentRow], list[SegmentRow]]] = {
            seed.uuid: ([], []) for seed in seeds
        }

        seeds_by_session: defaultdict[str, list[_SeedKey]] = defaultdict(list)
        for seed in seeds:
            seeds_by_session[seed.session_id].append(seed)

        chronological_order = [
            SegmentRow.timestamp,
            SegmentRow.event_uuid,
            SegmentRow.index,
            SegmentRow.offset,
        ]
        reverse_chronological_order = [col.desc() for col in chronological_order]

        for session_id, session_seeds in seeds_by_session.items():
            seeds_subquery = union_all(
                *(
                    select(
                        literal(seed.uuid, Uuid).label("seed_uuid"),
                        literal(seed.timestamp, DateTime(timezone=True)).label(
                            "seed_timestamp"
                        ),
                        literal(seed.event_uuid, Uuid).label("seed_event_uuid"),
                        literal(seed.index, Integer).label("seed_index"),
                        literal(seed.offset, Integer).label("seed_offset"),
                    )
                    for seed in session_seeds
                )
            ).subquery("seeds")
            segment_ordering_columns = tuple_(
                SegmentRow.timestamp,
                SegmentRow.event_uuid,
                SegmentRow.index,
                SegmentRow.offset,
            )
            seed_ordering_columns = tuple_(
                seeds_subquery.c.seed_timestamp,
                seeds_subquery.c.seed_event_uuid,
                seeds_subquery.c.seed_index,
                seeds_subquery.c.seed_offset,
            )
            row_conditions = [
                SQLAlchemySegmentStorePartition._session_condition(session_id),
                self._registry_row_query().exists(),
                *conditions,
            ]
            seed_uuids = [seed.uuid for seed in session_seeds]

            if before > 0:
                backward_rows_by_seed = await self._lateral_window_rows(
                    session,
                    seeds_subquery,
                    seed_uuids,
                    [
                        segment_ordering_columns < seed_ordering_columns,
                        *row_conditions,
                    ],
                    reverse_chronological_order,
                    before,
                )
                for seed_uuid, backward_rows in backward_rows_by_seed.items():
                    rows_by_seed[seed_uuid][0].extend(backward_rows)

            if after > 0:
                forward_rows_by_seed = await self._lateral_window_rows(
                    session,
                    seeds_subquery,
                    seed_uuids,
                    [
                        segment_ordering_columns > seed_ordering_columns,
                        *row_conditions,
                    ],
                    chronological_order,
                    after,
                )
                for seed_uuid, forward_rows in forward_rows_by_seed.items():
                    rows_by_seed[seed_uuid][1].extend(forward_rows)

        return rows_by_seed

    async def _lateral_window_rows(
        self,
        session: AsyncSession,
        seeds_subquery: Subquery,
        seed_uuids: Iterable[UUID],
        conditions: Sequence[ColumnElement[bool]],
        ordering: Iterable[ColumnElement | InstrumentedAttribute],
        limit: int,
    ) -> dict[UUID, list[SegmentRow]]:
        """Get context rows per seed in one direction."""
        # Build a LATERAL subquery that gets context rows for each seed.
        window_rows_query = (
            select(SegmentRow)
            .where(SegmentRow.incarnation == self._incarnation, *conditions)
            .order_by(*ordering)
            .limit(limit)
            .correlate(seeds_subquery)
        )
        lateral_subquery = window_rows_query.subquery().lateral("context")

        # Join each seed to its context rows via the LATERAL subquery.
        seed_context_join_query = select(
            seeds_subquery.c.seed_uuid,
            lateral_subquery.c.uuid,
            lateral_subquery.c.event_uuid,
            lateral_subquery.c.index,
            lateral_subquery.c.offset,
            lateral_subquery.c.timestamp,
            lateral_subquery.c.timestamp_timezone_offset,
            lateral_subquery.c.session_id,
            lateral_subquery.c.source_id,
            lateral_subquery.c.context,
            lateral_subquery.c.block_kind,
            lateral_subquery.c.block,
            lateral_subquery.c.properties,
        ).select_from(seeds_subquery.join(lateral_subquery, true()))

        # Group result rows by seed UUID.
        rows_by_seed: dict[UUID, list[SegmentRow]] = {
            seed_uuid: [] for seed_uuid in seed_uuids
        }
        for row in (await session.execute(seed_context_join_query)).all():
            rows_by_seed[row.seed_uuid].append(
                SegmentRow(
                    uuid=row.uuid,
                    incarnation=self._incarnation,
                    event_uuid=row.event_uuid,
                    index=row.index,
                    offset=row.offset,
                    timestamp=row.timestamp,
                    timestamp_timezone_offset=row.timestamp_timezone_offset,
                    session_id=row.session_id,
                    source_id=row.source_id,
                    context=row.context,
                    block_kind=row.block_kind,
                    block=row.block,
                    properties=row.properties,
                )
            )
        return rows_by_seed

    async def _get_window_rows_loop(
        self,
        session: AsyncSession,
        seeds: Sequence[_SeedKey],
        before: int,
        after: int,
        conditions: Sequence[ColumnElement[bool]],
    ) -> dict[UUID, tuple[list[SegmentRow], list[SegmentRow]]]:
        """Get backward/forward context per seed (SQLite fallback)."""
        window_rows_by_seed: dict[UUID, tuple[list[SegmentRow], list[SegmentRow]]] = {}

        segment_ordering_columns = tuple_(
            SegmentRow.timestamp,
            SegmentRow.event_uuid,
            SegmentRow.index,
            SegmentRow.offset,
        )

        for seed in seeds:
            seed_ordering_values = tuple_(
                literal(seed.timestamp),
                literal(seed.event_uuid),
                literal(seed.index),
                literal(seed.offset),
            )
            session_condition = SQLAlchemySegmentStorePartition._session_condition(
                seed.session_id
            )

            backward_rows: list[SegmentRow] = []
            if before > 0:
                backward_rows_query = (
                    select(SegmentRow)
                    .where(
                        SegmentRow.incarnation == self._incarnation,
                        session_condition,
                        segment_ordering_columns < seed_ordering_values,
                        self._registry_row_query().exists(),
                        *conditions,
                    )
                    .order_by(
                        SegmentRow.timestamp.desc(),
                        SegmentRow.event_uuid.desc(),
                        SegmentRow.index.desc(),
                        SegmentRow.offset.desc(),
                    )
                    .limit(before)
                )
                backward_rows = list(
                    (await session.execute(backward_rows_query)).scalars().all()
                )

            forward_rows: list[SegmentRow] = []
            if after > 0:
                forward_rows_query = (
                    select(SegmentRow)
                    .where(
                        SegmentRow.incarnation == self._incarnation,
                        session_condition,
                        segment_ordering_columns > seed_ordering_values,
                        self._registry_row_query().exists(),
                        *conditions,
                    )
                    .order_by(
                        SegmentRow.timestamp,
                        SegmentRow.event_uuid,
                        SegmentRow.index,
                        SegmentRow.offset,
                    )
                    .limit(after)
                )
                forward_rows = list(
                    (await session.execute(forward_rows_query)).scalars().all()
                )

            window_rows_by_seed[seed.uuid] = (backward_rows, forward_rows)

        return window_rows_by_seed

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
            query = (
                select(SegmentRow.event_uuid, SegmentRow.uuid)
                .where(
                    SegmentRow.incarnation == self._incarnation,
                    SegmentRow.event_uuid.in_(event_uuids),
                    self._registry_row_query().exists(),
                )
                .order_by(SegmentRow.event_uuid, SegmentRow.index, SegmentRow.offset)
            )
            rows = (await session.execute(query)).all()
            if not rows:
                await self._ensure_partition_live(session)

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
                DerivativeLinkRow.incarnation == self._incarnation,
                DerivativeLinkRow.segment_uuid.in_(segment_uuids),
                self._registry_row_query().exists(),
            )
            rows = (await session.execute(query)).all()
            if not rows:
                await self._ensure_partition_live(session)

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
                DerivativeLinkRow.incarnation == self._incarnation,
                DerivativeLinkRow.uuid.in_(derivative_uuids),
                self._registry_row_query().exists(),
            )
            rows = (await session.execute(query)).all()
            if not rows:
                await self._ensure_partition_live(session)

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

    @override
    async def delete_derivatives(
        self,
        derivative_uuids: Iterable[UUID],
    ) -> None:
        derivative_uuids = set(derivative_uuids)
        if not derivative_uuids:
            return

        async with (
            self._tracker("delete_derivatives"),
            self._create_session() as session,
            session.begin(),
        ):
            await self._lock_partition_for_write(session)
            if not self._is_sqlite:
                # Same deterministic lock order as delete_segments.
                await session.execute(
                    select(DerivativeLinkRow.uuid)
                    .where(
                        DerivativeLinkRow.incarnation == self._incarnation,
                        DerivativeLinkRow.uuid.in_(derivative_uuids),
                    )
                    .order_by(DerivativeLinkRow.uuid)
                    .with_for_update()
                )

            await session.execute(
                delete(DerivativeLinkRow).where(
                    DerivativeLinkRow.incarnation == self._incarnation,
                    DerivativeLinkRow.uuid.in_(derivative_uuids),
                )
            )

    # Helpers

    @staticmethod
    def _resolve_segment_field(
        field: str,
    ) -> tuple[ColumnElement, FieldEncoding]:
        """Map a filter field name to a segment column and encoding."""
        if field == "timestamp":
            return SegmentRow.timestamp.expression, "column"
        internal_name, is_user_metadata = normalize_filter_field(field)
        if is_user_metadata:
            key = demangle_user_metadata_key(internal_name)
            return SegmentRow.properties[key], "properties_json"
        return SegmentRow.properties[f"_{field}"], "properties_json"

    def _segment_from_segment_row(self, row: SegmentRow) -> Segment:
        """Convert a SegmentRow into a Segment."""
        context = decode_context(json.loads(self._payload_codec.decode(row.context)))
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
            session_id=row.session_id,
            source_id=row.source_id,
            context=context,
            block=block,
            properties=properties,
        )


def _in_values(
    column: InstrumentedAttribute[str] | InstrumentedAttribute[str | None],
    values: Iterable[str],
) -> ColumnElement[bool]:
    """`column IN values`; an empty list admits nothing."""
    values = list(values)
    if not values:
        return false()
    return column.in_(values)


class SQLAlchemySegmentStoreParams(BaseModel):
    """
    Parameters for constructing a SQLAlchemySegmentStore.

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
        self._purge_max_partitions = params.purge_max_partitions

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
                    attempts += 1
                    if attempts >= _MAX_MINT_ATTEMPTS:
                        raise SegmentStoreAttemptsExhaustedError(
                            f"Creating partition {partition_key!r} made no "
                            f"progress after {_MAX_MINT_ATTEMPTS} attempts"
                        ) from err
                    continue  # Mint a fresh incarnation.
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
                key being taken. Retry with a fresh incarnation.
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
                    logger.warning(
                        "Incarnation %s minted for partition %r collides "
                        "with garbage awaiting purge; re-minting",
                        incarnation,
                        partition_key,
                    )
                    raise _RegistryInsertRejectedError(str(incarnation))
        except IntegrityError as err:
            # If a committed row exists under this key, the key is taken.
            # Otherwise retry with a fresh incarnation.
            async with self._create_session() as session:
                partition_row = await SQLAlchemySegmentStore._get_partition_row(
                    session, partition_key
                )
            if partition_row is not None:
                raise SegmentStorePartitionAlreadyExistsError(partition_key) from err
            logger.warning(
                "Registry insert for partition %r with incarnation %s failed "
                "and no row exists under the key; retrying with a fresh "
                "incarnation",
                partition_key,
                incarnation,
            )
            raise _RegistryInsertRejectedError(str(incarnation)) from err

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
    @override
    async def purge_deleted_partitions(self) -> bool:
        # Reclaim dead incarnations oldest-first, within the per-call bounds.
        #
        # One transaction per call: reclaim up to the configured bounds and
        # commit, or nothing. That is what makes a call that fails on
        # contention safe to repeat, as the ABC promises: a raise rolls the
        # whole call back. Entries are claimed one at a time, so only
        # the claiming call touches a dead incarnation's rows. SQLite drops
        # locking clauses; there purgers serialize at the DELETE, and a
        # doubly-claimed entry costs empty round trips, never duplicated or
        # missed reclamation (an entry is retired only when the retirer's
        # own DELETEs found fewer rows than its budget). Full rationale:
        # design/segment_store_shared_tables.md.
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
                incarnation = (
                    await connection.execute(
                        # Skips only OTHER transactions' locks; entries this
                        # call has already claimed cannot come back, each
                        # being deleted before the next claim.
                        select(PurgeQueueRow.incarnation)
                        .order_by(PurgeQueueRow.enqueued_at)
                        .limit(1)
                        .with_for_update(skip_locked=True)
                    )
                ).scalar_one_or_none()
                if incarnation is None:
                    return False

                batch = (
                    select(SegmentRow.uuid)
                    .where(SegmentRow.incarnation == incarnation)
                    .limit(remaining)
                    .scalar_subquery()
                )

                # The link-table cascade follows the deleted segments.
                deleted = (
                    await connection.execute(
                        delete(SegmentRow).where(
                            SegmentRow.incarnation == incarnation,
                            SegmentRow.uuid.in_(batch),
                        )
                    )
                ).rowcount
                if deleted == remaining:
                    # The bound was consumed exactly; this incarnation may
                    # have more rows, so leave its queue entry for the
                    # next call.
                    return True

                remaining -= deleted
                # The cascade has already removed the deleted segments'
                # links; this guards retirement against rows that escaped
                # referential integrity. Normally a zero-row delete; if
                # integrity was actually broken, the leak is reclaimed in
                # batches drawing count-for-count on the same budget as
                # the segments, and a full batch leaves the entry for the
                # next call. A link row deletes cheaper than a segment
                # row (narrower, fewer indexes, no cascade), so the
                # shared budget is an upper bound on a call sized for
                # segment rows, not a guessed ratio.
                leaked_batch = (
                    select(DerivativeLinkRow.uuid)
                    .where(DerivativeLinkRow.incarnation == incarnation)
                    .limit(remaining)
                    .scalar_subquery()
                )
                leaked = (
                    await connection.execute(
                        delete(DerivativeLinkRow).where(
                            DerivativeLinkRow.incarnation == incarnation,
                            DerivativeLinkRow.uuid.in_(leaked_batch),
                        )
                    )
                ).rowcount
                if leaked:
                    logger.warning(
                        "Purged %d derivative-link rows that referential "
                        "integrity should have removed with their segments "
                        "(incarnation %s); check foreign-key enforcement",
                        leaked,
                        incarnation,
                    )
                    if leaked == remaining:
                        return True
                    remaining -= leaked
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
