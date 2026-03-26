"""SQLAlchemy implementation of the SegmentLinker interface."""

import asyncio
import logging
from collections import defaultdict
from collections.abc import Callable, Iterable, Mapping
from contextlib import AbstractAsyncContextManager, nullcontext
from datetime import UTC, datetime
from enum import StrEnum
from typing import cast, override
from uuid import UUID

from pydantic import BaseModel, Field, InstanceOf, JsonValue, TypeAdapter
from sqlalchemy import (
    JSON,
    DateTime,
    ForeignKeyConstraint,
    Index,
    Integer,
    String,
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
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import (
    DeclarativeBase,
    InstrumentedAttribute,
    MappedColumn,
    mapped_column,
)
from sqlalchemy.sql.elements import ColumnElement

from memmachine_server.common.data_types import (
    PROPERTY_TYPE_NAME_TO_PROPERTY_TYPE,
    PROPERTY_TYPE_TO_PROPERTY_TYPE_NAME,
    PropertyValue,
)
from memmachine_server.common.filter.filter_parser import (
    And,
    Comparison,
    FilterExpr,
    In,
    IsNull,
    Not,
    Or,
)
from memmachine_server.episodic_memory.extra_memory.data_types import (
    Block,
    Context,
    Segment,
)
from memmachine_server.episodic_memory.extra_memory.segment_store.segment_store import (
    SegmentLinker,
    SegmentLinkerPartition,
)

logger = logging.getLogger(__name__)

_JSON_AUTO = JSON().with_variant(JSONB, "postgresql")

_PROPERTY_TYPE_KEY = "t"
_PROPERTY_VALUE_KEY = "v"

_ContextAdapter = TypeAdapter(Context | None)
_BlockAdapter = TypeAdapter(Block)


class DerivativeState(StrEnum):
    """Lifecycle state of a derivative."""

    ACTIVE = "A"
    PURGING = "P"


# ORM models


class BaseSegmentLinker(DeclarativeBase):
    """Base class for segment linker tables."""


class SegmentRow(BaseSegmentLinker):
    """Persisted segment."""

    __tablename__ = "segment_linker_segments"

    partition_key: MappedColumn[str] = mapped_column(String, primary_key=True)

    uuid: MappedColumn[UUID] = mapped_column(Uuid, primary_key=True)
    episode_uuid: MappedColumn[UUID] = mapped_column(Uuid, nullable=False)
    index: MappedColumn[int] = mapped_column(Integer, nullable=False)
    offset: MappedColumn[int] = mapped_column(Integer, nullable=False)
    timestamp: MappedColumn[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )
    context: MappedColumn[dict[str, JsonValue] | None] = mapped_column(
        _JSON_AUTO, nullable=True
    )
    block: MappedColumn[dict[str, JsonValue]] = mapped_column(
        _JSON_AUTO, nullable=False
    )
    properties: MappedColumn[dict[str, JsonValue]] = mapped_column(
        _JSON_AUTO, nullable=False, default=dict
    )

    __table_args__ = (
        Index(
            "segment_linker_segments__pk_ep",
            "partition_key",
            "episode_uuid",
        ),
        Index(
            "segment_linker_segments__pk_ts_ep_bk_ix",
            "partition_key",
            "timestamp",
            "episode_uuid",
            "index",
            "offset",
        ),
    )


class LinkRow(BaseSegmentLinker):
    """Many-to-many join between segments and derivatives."""

    __tablename__ = "segment_linker_links"

    partition_key: MappedColumn[str] = mapped_column(String, primary_key=True)

    segment_uuid: MappedColumn[UUID] = mapped_column(Uuid, primary_key=True)
    derivative_uuid: MappedColumn[UUID] = mapped_column(Uuid, primary_key=True)

    __table_args__ = (
        ForeignKeyConstraint(
            ["partition_key", "segment_uuid"],
            [
                "segment_linker_segments.partition_key",
                "segment_linker_segments.uuid",
            ],
            ondelete="CASCADE",
        ),
        ForeignKeyConstraint(
            ["partition_key", "derivative_uuid"],
            [
                "segment_linker_derivatives.partition_key",
                "segment_linker_derivatives.uuid",
            ],
            ondelete="CASCADE",
        ),
        Index("segment_linker_links__pk_du", "partition_key", "derivative_uuid"),
    )


class DerivativeRow(BaseSegmentLinker):
    """Derivative lifecycle record."""

    __tablename__ = "segment_linker_derivatives"

    partition_key: MappedColumn[str] = mapped_column(String, primary_key=True)

    uuid: MappedColumn[UUID] = mapped_column(Uuid, primary_key=True)

    state: MappedColumn[str] = mapped_column(String(1), nullable=False)
    ref_count: MappedColumn[int] = mapped_column(Integer, nullable=False, default=0)

    __table_args__ = (
        Index(
            "segment_linker_derivatives__pk_st_rc",
            "partition_key",
            "state",
            "ref_count",
        ),
    )


class SQLAlchemySegmentLinkerPartition(SegmentLinkerPartition):
    """SQLAlchemy-backed partition handle."""

    def __init__(
        self,
        partition_key: str,
        create_session: async_sessionmaker[AsyncSession],
        write_lock: asyncio.Lock | None = None,
    ) -> None:
        """Initialize with a partition key and session maker."""
        self._partition_key = partition_key
        self._create_session = create_session
        self._write_lock = write_lock

    def _sqlite_write_lock(self) -> AbstractAsyncContextManager[None]:
        """Return the write lock if configured (SQLite), otherwise a no-op."""
        if self._write_lock is not None:
            return self._write_lock
        return nullcontext()

    # Registration

    @override
    async def register_segments(
        self,
        links: Mapping[Segment, Iterable[UUID]],
    ) -> None:
        links = {
            segment: set(derivative_uuids)
            for segment, derivative_uuids in links.items()
        }

        # Compute ref count deltas per derivative.
        deltas: dict[UUID, int] = defaultdict(int)
        for derivative_uuids in links.values():
            for derivative_uuid in derivative_uuids:
                deltas[derivative_uuid] += 1

        derivative_uuids = set(deltas.keys())

        while True:
            async with (
                self._sqlite_write_lock(),
                self._create_session() as session,
                session.begin(),
            ):
                # Insert all derivatives ON CONFLICT DO NOTHING.
                await self._insert_new_derivatives(session, derivative_uuids)

                # Lock derivatives.
                locked = await self._lock_derivatives(session, derivative_uuids)

                # Retry if any are PURGING.
                if any(row.state == DerivativeState.PURGING for row in locked.values()):
                    await session.rollback()
                else:
                    # Insert segments and links.
                    await self._insert_segments(session, links.keys())
                    await self._insert_links(session, links)

                    # Increment ref counts.
                    await self._apply_ref_count_deltas(session, deltas)
                    break

            # Session and locks are released. Sleep before retrying.
            await asyncio.sleep(0.1)

    async def _lock_derivatives(
        self,
        session: AsyncSession,
        derivative_uuids: Iterable[UUID],
    ) -> dict[UUID, DerivativeRow]:
        """Lock and return existing derivative rows, in consistent UUID order."""
        derivative_uuids = set(derivative_uuids)
        if not derivative_uuids:
            return {}

        # We rely on the following behavior to avoid deadlocks:
        # https://www.postgresql.org/docs/current/sql-select.html
        # It is possible for a SELECT command running at the READ COMMITTED transaction isolation level
        # and using ORDER BY and a locking clause to return rows out of order.
        # This is because ORDER BY is applied first.
        query = (
            select(DerivativeRow)
            .where(
                DerivativeRow.partition_key == self._partition_key,
                DerivativeRow.uuid.in_(derivative_uuids),
            )
            .order_by(DerivativeRow.uuid)
        )
        if session.bind.dialect.name != "sqlite":
            query = query.with_for_update()
        rows = (await session.execute(query)).scalars().all()
        return {row.uuid: row for row in rows}

    async def _insert_segments(
        self,
        session: AsyncSession,
        segments: Iterable[Segment],
    ) -> None:
        """Insert segment rows and their property rows."""
        segments = set(segments)

        segment_row_values = [
            {
                "uuid": segment.uuid,
                "partition_key": self._partition_key,
                "episode_uuid": segment.episode_uuid,
                "index": segment.index,
                "offset": segment.offset,
                "timestamp": segment.timestamp,
                "context": (
                    segment.context.model_dump(mode="json")
                    if segment.context is not None
                    else None
                ),
                "block": segment.block.model_dump(mode="json"),
                "properties": SQLAlchemySegmentLinkerPartition._encode_properties(
                    segment.properties
                ),
            }
            for segment in segments
        ]
        if segment_row_values:
            await session.execute(insert(SegmentRow), segment_row_values)

    async def _insert_new_derivatives(
        self,
        session: AsyncSession,
        derivative_uuids: Iterable[UUID],
    ) -> None:
        """Insert new derivative rows with ref_count=0, ON CONFLICT DO NOTHING."""
        derivative_uuids = set(derivative_uuids)
        if not derivative_uuids:
            return
        values = [
            {
                "uuid": derivative_uuid,
                "partition_key": self._partition_key,
                "state": DerivativeState.ACTIVE,
                "ref_count": 0,
            }
            for derivative_uuid in sorted(derivative_uuids)
        ]
        dialect = session.bind.dialect.name
        if dialect == "postgresql":
            insert_statement = pg_insert(DerivativeRow).on_conflict_do_nothing()
        elif dialect == "sqlite":
            insert_statement = sqlite_insert(DerivativeRow).on_conflict_do_nothing()
        else:
            raise NotImplementedError(f"Unsupported dialect: {dialect}")
        await session.execute(insert_statement, values)

    async def _insert_links(
        self,
        session: AsyncSession,
        links: Mapping[Segment, Iterable[UUID]],
    ) -> None:
        """Insert link rows between segments and derivatives."""
        link_rows = [
            {
                "partition_key": self._partition_key,
                "segment_uuid": segment.uuid,
                "derivative_uuid": derivative_uuid,
            }
            for segment, derivative_uuids in links.items()
            for derivative_uuid in derivative_uuids
        ]
        if link_rows:
            await session.execute(insert(LinkRow), link_rows)

    async def _apply_ref_count_deltas(
        self, session: AsyncSession, deltas: Mapping[UUID, int]
    ) -> None:
        """Apply ref count deltas to derivative rows."""
        if not deltas:
            return

        conn = await session.connection()
        await conn.execute(
            update(DerivativeRow)
            .where(
                DerivativeRow.partition_key == bindparam("b_partition_key"),
                DerivativeRow.uuid == bindparam("derivative_uuid"),
            )
            .values(ref_count=DerivativeRow.ref_count + bindparam("delta")),
            [
                {
                    "b_partition_key": self._partition_key,
                    "derivative_uuid": derivative_uuid,
                    "delta": delta,
                }
                for derivative_uuid, delta in deltas.items()
            ],
        )

    # Retrieval

    @override
    async def get_segments_by_derivatives(
        self,
        derivative_uuids: Iterable[UUID],
        *,
        limit_per_derivative: int | None = None,
        property_filter: FilterExpr | None = None,
    ) -> dict[UUID, list[Segment]]:
        derivative_uuids = set(derivative_uuids)
        if not derivative_uuids:
            return {}

        if limit_per_derivative is None:
            return await self._get_segments_by_derivatives_all(
                derivative_uuids, property_filter
            )

        return await self._get_segments_by_derivatives_windowed(
            derivative_uuids, limit_per_derivative, property_filter
        )

    async def _get_segments_by_derivatives_all(
        self,
        derivative_uuids: Iterable[UUID],
        property_filter: FilterExpr | None,
    ) -> dict[UUID, list[Segment]]:
        """Get all segments for each derivative."""
        derivative_uuids = set(derivative_uuids)

        segments_by_derivatives_query = (
            select(LinkRow.derivative_uuid, SegmentRow)
            .join(
                LinkRow,
                (LinkRow.partition_key == SegmentRow.partition_key)
                & (LinkRow.segment_uuid == SegmentRow.uuid),
            )
            .where(
                SegmentRow.partition_key == self._partition_key,
                LinkRow.derivative_uuid.in_(derivative_uuids),
            )
            .order_by(
                SegmentRow.timestamp,
                SegmentRow.episode_uuid,
                SegmentRow.index,
                SegmentRow.offset,
            )
        )

        if property_filter is not None:
            segments_by_derivatives_query = segments_by_derivatives_query.where(
                SQLAlchemySegmentLinkerPartition._compile_property_filter(
                    property_filter
                )
            )

        async with self._create_session() as session:
            segment_rows = (await session.execute(segments_by_derivatives_query)).all()

        segments_by_derivatives: defaultdict[UUID, list[Segment]] = defaultdict(list)
        for derivative_uuid, segment_row in segment_rows:
            segment = SQLAlchemySegmentLinkerPartition._segment_from_segment_row(
                segment_row
            )
            segments_by_derivatives[derivative_uuid].append(segment)

        return dict(segments_by_derivatives)

    async def _get_segments_by_derivatives_windowed(
        self,
        derivative_uuids: Iterable[UUID],
        limit_per_derivative: int,
        property_filter: FilterExpr | None,
    ) -> dict[UUID, list[Segment]]:
        """Get segments with limit_per_derivative applied via SQL window functions."""
        derivative_uuids = set(derivative_uuids)
        if not derivative_uuids:
            return {}

        limit_first = limit_per_derivative // 2
        limit_last = limit_per_derivative - limit_first

        row_number_ascending = (
            func.row_number()
            .over(
                partition_by=LinkRow.derivative_uuid,
                order_by=[
                    SegmentRow.timestamp,
                    SegmentRow.episode_uuid,
                    SegmentRow.index,
                    SegmentRow.offset,
                ],
            )
            .label("row_number_ascending")
        )
        row_number_descending = (
            func.row_number()
            .over(
                partition_by=LinkRow.derivative_uuid,
                order_by=[
                    SegmentRow.timestamp.desc(),
                    SegmentRow.episode_uuid.desc(),
                    SegmentRow.index.desc(),
                    SegmentRow.offset.desc(),
                ],
            )
            .label("row_number_descending")
        )

        numbered_derivative_segments_query = (
            select(
                LinkRow.derivative_uuid,
                SegmentRow.uuid,
                SegmentRow.episode_uuid,
                SegmentRow.index,
                SegmentRow.offset,
                SegmentRow.timestamp,
                SegmentRow.context,
                SegmentRow.block,
                SegmentRow.properties,
                row_number_ascending,
                row_number_descending,
            )
            .join(
                LinkRow,
                (LinkRow.partition_key == SegmentRow.partition_key)
                & (LinkRow.segment_uuid == SegmentRow.uuid),
            )
            .where(
                SegmentRow.partition_key == self._partition_key,
                LinkRow.derivative_uuid.in_(derivative_uuids),
            )
        )
        if property_filter is not None:
            numbered_derivative_segments_query = (
                numbered_derivative_segments_query.where(
                    SQLAlchemySegmentLinkerPartition._compile_property_filter(
                        property_filter
                    )
                )
            )

        async with self._create_session() as session:
            numbered_derivative_segments_subquery = (
                numbered_derivative_segments_query.subquery()
            )

            limited_derivative_segments_query = (
                select(numbered_derivative_segments_subquery)
                .where(
                    or_(
                        numbered_derivative_segments_subquery.c.row_number_ascending
                        <= limit_first,
                        numbered_derivative_segments_subquery.c.row_number_descending
                        <= limit_last,
                    )
                )
                .order_by(
                    numbered_derivative_segments_subquery.c.timestamp,
                    numbered_derivative_segments_subquery.c.episode_uuid,
                    numbered_derivative_segments_subquery.c.index,
                    numbered_derivative_segments_subquery.c.offset,
                )
            )

            limited_derivative_segment_rows = (
                await session.execute(limited_derivative_segments_query)
            ).all()

        segments_by_derivatives: defaultdict[UUID, list[Segment]] = defaultdict(list)
        for row in limited_derivative_segment_rows:
            segment_row = SegmentRow(
                uuid=row.uuid,
                episode_uuid=row.episode_uuid,
                index=row.index,
                offset=row.offset,
                timestamp=row.timestamp,
                context=row.context,
                block=row.block,
                properties=row.properties,
            )
            segment = SQLAlchemySegmentLinkerPartition._segment_from_segment_row(
                segment_row
            )
            segments_by_derivatives[row.derivative_uuid].append(segment)

        return dict(segments_by_derivatives)

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

        async with self._create_session() as session:
            seed_segments_query = select(SegmentRow).where(
                SegmentRow.uuid.in_(seed_segment_uuids),
                SegmentRow.partition_key == self._partition_key,
            )
            if property_filter is not None:
                seed_segments_query = seed_segments_query.where(
                    SQLAlchemySegmentLinkerPartition._compile_property_filter(
                        property_filter
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
                        SQLAlchemySegmentLinkerPartition._segment_from_segment_row(
                            seed_segment_row,
                        )
                    ]
                    for seed_segment_uuid, seed_segment_row in seed_segment_rows_by_uuid.items()
                }

            # Get backward/forward context rows.
            if session.bind.dialect.name != "sqlite":
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
                    SQLAlchemySegmentLinkerPartition._segment_from_segment_row(row)
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
                SegmentRow.episode_uuid.label("seed_episode_uuid"),
                SegmentRow.index.label("seed_index"),
                SegmentRow.offset.label("seed_offset"),
            )
            .where(
                SegmentRow.partition_key == self._partition_key,
                SegmentRow.uuid.in_(seed_rows_by_uuid.keys()),
            )
            .subquery("seeds")
        )

        segment_ordering_columns = tuple_(
            SegmentRow.timestamp,
            SegmentRow.episode_uuid,
            SegmentRow.index,
            SegmentRow.offset,
        )
        seed_ordering_columns = tuple_(
            seeds_subquery.c.seed_timestamp,
            seeds_subquery.c.seed_episode_uuid,
            seeds_subquery.c.seed_index,
            seeds_subquery.c.seed_offset,
        )

        partition_key = self._partition_key

        async def get_context_rows_directional(
            range_condition: ColumnElement[bool],
            ordering: Iterable[ColumnElement | InstrumentedAttribute],
            limit: int,
        ) -> dict[UUID, list[SegmentRow]]:
            """Get context rows per seed in the specified direction."""
            # Build a LATERAL subquery that gets context rows for each seed.
            context_rows_query = (
                select(SegmentRow)
                .where(SegmentRow.partition_key == partition_key, range_condition)
                .order_by(*ordering)
                .limit(limit)
                .correlate(seeds_subquery)
            )
            if property_filter is not None:
                context_rows_query = context_rows_query.where(
                    SQLAlchemySegmentLinkerPartition._compile_property_filter(
                        property_filter
                    )
                )
            lateral_subquery = context_rows_query.subquery().lateral("context")

            # Join each seed to its context rows via the LATERAL subquery.
            seed_context_join_query = select(
                seeds_subquery.c.seed_uuid,
                lateral_subquery.c.uuid,
                lateral_subquery.c.episode_uuid,
                lateral_subquery.c.index,
                lateral_subquery.c.offset,
                lateral_subquery.c.timestamp,
                lateral_subquery.c.context,
                lateral_subquery.c.block,
                lateral_subquery.c.properties,
            ).select_from(seeds_subquery.join(lateral_subquery, true()))

            # Group result rows by seed UUID.
            rows_by_seed: dict[UUID, list[SegmentRow]] = {
                seed_uuid: [] for seed_uuid in seed_rows_by_uuid
            }
            for row in (await session.execute(seed_context_join_query)).all():
                rows_by_seed[row.seed_uuid].append(
                    SegmentRow(
                        uuid=row.uuid,
                        partition_key=partition_key,
                        episode_uuid=row.episode_uuid,
                        index=row.index,
                        offset=row.offset,
                        timestamp=row.timestamp,
                        context=row.context,
                        block=row.block,
                        properties=row.properties,
                    )
                )
            return rows_by_seed

        chronological_order = [
            SegmentRow.timestamp,
            SegmentRow.episode_uuid,
            SegmentRow.index,
            SegmentRow.offset,
        ]
        reverse_chronological_order = [col.desc() for col in chronological_order]

        backward_rows_by_seed = (
            await get_context_rows_directional(
                segment_ordering_columns < seed_ordering_columns,
                reverse_chronological_order,
                max_backward_segments,
            )
            if max_backward_segments > 0
            else {seed_uuid: [] for seed_uuid in seed_rows_by_uuid}
        )

        forward_rows_by_seed = (
            await get_context_rows_directional(
                segment_ordering_columns > seed_ordering_columns,
                chronological_order,
                max_forward_segments,
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
        context_rows_by_seed: dict[UUID, tuple[list[SegmentRow], list[SegmentRow]]] = {}

        segment_ordering_columns = tuple_(
            SegmentRow.timestamp,
            SegmentRow.episode_uuid,
            SegmentRow.index,
            SegmentRow.offset,
        )

        compiled_property_filter = (
            SQLAlchemySegmentLinkerPartition._compile_property_filter(property_filter)
            if property_filter is not None
            else None
        )

        for seed_uuid, seed_row in seed_rows_by_uuid.items():
            seed_ordering_values = tuple_(
                literal(seed_row.timestamp),
                literal(seed_row.episode_uuid),
                literal(seed_row.index),
                literal(seed_row.offset),
            )

            backward_rows: list[SegmentRow] = []
            if max_backward_segments > 0:
                backward_rows_query = (
                    select(SegmentRow)
                    .where(
                        SegmentRow.partition_key == self._partition_key,
                        segment_ordering_columns < seed_ordering_values,
                    )
                    .order_by(
                        SegmentRow.timestamp.desc(),
                        SegmentRow.episode_uuid.desc(),
                        SegmentRow.index.desc(),
                        SegmentRow.offset.desc(),
                    )
                    .limit(max_backward_segments)
                )
                if compiled_property_filter is not None:
                    backward_rows_query = backward_rows_query.where(
                        compiled_property_filter
                    )
                backward_rows = list(
                    (await session.execute(backward_rows_query)).scalars().all()
                )

            forward_rows: list[SegmentRow] = []
            if max_forward_segments > 0:
                forward_rows_query = (
                    select(SegmentRow)
                    .where(
                        SegmentRow.partition_key == self._partition_key,
                        segment_ordering_columns > seed_ordering_values,
                    )
                    .order_by(
                        SegmentRow.timestamp,
                        SegmentRow.episode_uuid,
                        SegmentRow.index,
                        SegmentRow.offset,
                    )
                    .limit(max_forward_segments)
                )
                if compiled_property_filter is not None:
                    forward_rows_query = forward_rows_query.where(
                        compiled_property_filter
                    )
                forward_rows = list(
                    (await session.execute(forward_rows_query)).scalars().all()
                )

            context_rows_by_seed[seed_uuid] = (backward_rows, forward_rows)

        return context_rows_by_seed

    # Deletion

    @override
    async def delete_segments_by_episodes(
        self,
        episode_uuids: Iterable[UUID],
    ) -> None:
        episode_uuids = set(episode_uuids)
        if not episode_uuids:
            return

        async with (
            self._sqlite_write_lock(),
            self._create_session() as session,
            session.begin(),
        ):
            # Find derivative UUIDs linked to segments being deleted.
            derivative_uuids_query = (
                select(LinkRow.derivative_uuid)
                .join(
                    SegmentRow,
                    (LinkRow.partition_key == SegmentRow.partition_key)
                    & (LinkRow.segment_uuid == SegmentRow.uuid),
                )
                .where(
                    SegmentRow.partition_key == self._partition_key,
                    SegmentRow.episode_uuid.in_(episode_uuids),
                )
                .distinct()
            )
            derivative_uuids = set(
                (await session.execute(derivative_uuids_query)).scalars().all()
            )

            if derivative_uuids:
                # Lock derivatives.
                locked = await self._lock_derivatives(session, derivative_uuids)
                active_uuids = {
                    uuid
                    for uuid, row in locked.items()
                    if row.state == DerivativeState.ACTIVE
                }

                # Compute ref count deltas (negative) from links being removed.
                if active_uuids:
                    ref_count_query = (
                        select(
                            LinkRow.derivative_uuid,
                            func.count().label("ref_count"),
                        )
                        .join(
                            SegmentRow,
                            (LinkRow.partition_key == SegmentRow.partition_key)
                            & (LinkRow.segment_uuid == SegmentRow.uuid),
                        )
                        .where(
                            SegmentRow.partition_key == self._partition_key,
                            SegmentRow.episode_uuid.in_(episode_uuids),
                            LinkRow.derivative_uuid.in_(active_uuids),
                        )
                        .group_by(LinkRow.derivative_uuid)
                    )
                    ref_count_rows = (await session.execute(ref_count_query)).all()

                    deltas = {
                        derivative_uuid: -ref_count
                        for derivative_uuid, ref_count in ref_count_rows
                    }

                    # Decrement ref counts.
                    await self._apply_ref_count_deltas(session, deltas)

            # Delete segments (CASCADE deletes links via FK).
            await session.execute(
                delete(SegmentRow).where(
                    SegmentRow.partition_key == self._partition_key,
                    SegmentRow.episode_uuid.in_(episode_uuids),
                )
            )

    # Garbage collection

    @override
    async def mark_orphaned_derivatives_for_purging(self, limit: int = 1000) -> None:
        async with (
            self._sqlite_write_lock(),
            self._create_session() as session,
            session.begin(),
        ):
            # Find and lock orphaned derivatives.
            find_orphans_query = (
                select(DerivativeRow.uuid)
                .where(
                    DerivativeRow.partition_key == self._partition_key,
                    DerivativeRow.state == DerivativeState.ACTIVE,
                    DerivativeRow.ref_count == 0,
                )
                .limit(limit)
            )

            if session.bind.dialect.name != "sqlite":
                # SKIP LOCKED lets concurrent GC callers pick non-overlapping batches
                # and avoids deadlocks without ordering (never waits).
                find_orphans_query = find_orphans_query.with_for_update(
                    skip_locked=True
                )
            orphan_uuids = list(
                (await session.execute(find_orphans_query)).scalars().all()
            )

            # Mark locked orphans for purging.
            if orphan_uuids:
                await session.execute(
                    update(DerivativeRow)
                    .where(
                        DerivativeRow.partition_key == self._partition_key,
                        DerivativeRow.uuid.in_(orphan_uuids),
                    )
                    .values(state=DerivativeState.PURGING)
                )

    @override
    async def get_derivatives_pending_purge(self, limit: int = 1000) -> set[UUID]:
        async with self._create_session() as session:
            return set(
                (
                    await session.execute(
                        select(DerivativeRow.uuid)
                        .where(
                            DerivativeRow.partition_key == self._partition_key,
                            DerivativeRow.state == DerivativeState.PURGING,
                        )
                        .limit(limit)
                    )
                )
                .scalars()
                .all()
            )

    @override
    async def purge_derivatives(self, derivative_uuids: Iterable[UUID]) -> None:
        derivative_uuids = set(derivative_uuids)
        if not derivative_uuids:
            return

        async with (
            self._sqlite_write_lock(),
            self._create_session() as session,
            session.begin(),
        ):
            await session.execute(
                delete(DerivativeRow).where(
                    DerivativeRow.partition_key == self._partition_key,
                    DerivativeRow.uuid.in_(derivative_uuids),
                    DerivativeRow.state == DerivativeState.PURGING,
                )
            )

    # Helpers

    @staticmethod
    def _encode_properties(
        properties: Mapping[str, PropertyValue],
    ) -> dict[str, dict[str, bool | int | float | str]]:
        """Encode properties as type-tagged JSONB: {"key": {_PROPERTY_VALUE_KEY: value, _PROPERTY_TYPE_KEY: type_name}}."""
        encoded: dict[str, dict[str, bool | int | float | str]] = {}
        for key, value in properties.items():
            type_name = PROPERTY_TYPE_TO_PROPERTY_TYPE_NAME.get(type(value))
            if type_name is None:
                raise ValueError(f"Unsupported property value type: {type(value)!r}")
            if isinstance(value, datetime):
                utc_value = value.astimezone(UTC)
                encoded[key] = {
                    _PROPERTY_VALUE_KEY: utc_value.isoformat(),
                    _PROPERTY_TYPE_KEY: type_name,
                }
            else:
                encoded[key] = {
                    _PROPERTY_VALUE_KEY: value,
                    _PROPERTY_TYPE_KEY: type_name,
                }
        return encoded

    @staticmethod
    def _decode_properties(
        encoded: Mapping[str, JsonValue],
    ) -> dict[str, PropertyValue]:
        """Decode type-tagged JSONB properties back to Python values."""
        properties: dict[str, PropertyValue] = {}
        for key, entry in encoded.items():
            if not isinstance(entry, dict):
                raise TypeError(
                    f"Expected dict for property entry, got {type(entry)!r}"
                )
            type_name = str(entry[_PROPERTY_TYPE_KEY])
            raw_value = entry[_PROPERTY_VALUE_KEY]
            property_type = PROPERTY_TYPE_NAME_TO_PROPERTY_TYPE.get(type_name)
            if property_type is None:
                raise ValueError(f"Unknown property type name: {type_name!r}")
            if property_type is datetime:
                properties[key] = datetime.fromisoformat(str(raw_value))
            else:
                properties[key] = cast(type[bool | int | float | str], property_type)(
                    raw_value
                )
        return properties

    @staticmethod
    def _compile_property_filter(expr: FilterExpr) -> ColumnElement[bool]:
        """Compile a FilterExpr into a SQLAlchemy boolean expression against inline JSONB properties."""
        if isinstance(expr, Comparison):
            return SQLAlchemySegmentLinkerPartition._compile_comparison(
                expr.field, expr.op, expr.value
            )

        if isinstance(expr, In):
            # IN is equivalent to OR of equalities on the same field.
            if not expr.values:
                return literal(False)
            first_value = expr.values[0]
            type_name = PROPERTY_TYPE_TO_PROPERTY_TYPE_NAME[type(first_value)]
            value_element = SegmentRow.properties[expr.field][_PROPERTY_VALUE_KEY]
            type_check = (
                SegmentRow.properties[expr.field][_PROPERTY_TYPE_KEY].as_string()
                == type_name
            )
            if isinstance(first_value, int):
                return and_(type_check, value_element.as_integer().in_(expr.values))
            return and_(type_check, value_element.as_string().in_(expr.values))

        if isinstance(expr, IsNull):
            # Property is null = key does not exist in the JSONB object.
            return SegmentRow.properties[expr.field].is_(None)

        if isinstance(expr, Not):
            return ~SQLAlchemySegmentLinkerPartition._compile_property_filter(expr.expr)

        if isinstance(expr, And):
            return and_(
                SQLAlchemySegmentLinkerPartition._compile_property_filter(expr.left),
                SQLAlchemySegmentLinkerPartition._compile_property_filter(expr.right),
            )

        if isinstance(expr, Or):
            return or_(
                SQLAlchemySegmentLinkerPartition._compile_property_filter(expr.left),
                SQLAlchemySegmentLinkerPartition._compile_property_filter(expr.right),
            )

        raise TypeError(f"Unsupported filter expression type: {type(expr)!r}")

    @staticmethod
    def _compile_comparison(
        field: str, op: str, value: PropertyValue
    ) -> ColumnElement[bool]:
        """Compile a single comparison against a type-tagged JSONB property."""
        type_name = PROPERTY_TYPE_TO_PROPERTY_TYPE_NAME[type(value)]
        value_element = SegmentRow.properties[field][_PROPERTY_VALUE_KEY]
        type_check = (
            SegmentRow.properties[field][_PROPERTY_TYPE_KEY].as_string() == type_name
        )

        # Convert the value and cast the JSON element appropriately.
        if isinstance(value, bool):
            casted = value_element.as_boolean()
            cmp_value = value
        elif isinstance(value, int):
            casted = value_element.as_integer()
            cmp_value = value
        elif isinstance(value, float):
            casted = value_element.as_float()
            cmp_value = value
        elif isinstance(value, str):
            casted = value_element.as_string()
            cmp_value = value
        elif isinstance(value, datetime):
            casted = value_element.as_string()
            cmp_value = value.astimezone(UTC).isoformat()
        else:
            raise TypeError(f"Unsupported property value type: {type(value)!r}")

        ops: dict[str, Callable] = {
            "=": lambda c, v: c == v,
            "!=": lambda c, v: c != v,
            ">": lambda c, v: c > v,
            "<": lambda c, v: c < v,
            ">=": lambda c, v: c >= v,
            "<=": lambda c, v: c <= v,
        }
        return and_(type_check, ops[op](casted, cmp_value))

    @staticmethod
    def _segment_from_segment_row(
        row: SegmentRow,
    ) -> Segment:
        """Convert a SegmentRow into a Segment."""
        context = _ContextAdapter.validate_python(row.context)
        block = _BlockAdapter.validate_python(row.block)
        properties = SQLAlchemySegmentLinkerPartition._decode_properties(row.properties)
        return Segment(
            uuid=row.uuid,
            episode_uuid=row.episode_uuid,
            index=row.index,
            offset=row.offset,
            timestamp=row.timestamp,
            context=context,
            block=block,
            properties=properties,
        )


class SQLAlchemySegmentLinkerParams(BaseModel):
    """
    Parameters for constructing a SQLAlchemySegmentLinker.

    Attributes:
        engine (AsyncEngine):
            Async SQLAlchemy engine.
    """

    engine: InstanceOf[AsyncEngine] = Field(..., description="Async SQLAlchemy engine")


class SQLAlchemySegmentLinker(SegmentLinker):
    """SQLAlchemy-backed SegmentLinker factory."""

    def __init__(self, params: SQLAlchemySegmentLinkerParams) -> None:
        """Initialize with an async SQLAlchemy engine."""
        self._engine = params.engine
        self._create_session = async_sessionmaker(self._engine, expire_on_commit=False)

        # SQLite does not isolate transactions within a single connection.
        # https://sqlite.org/isolation.html
        self._use_write_lock = self._engine.dialect.name == "sqlite"
        self._write_locks: dict[str, asyncio.Lock] = {}

    @override
    async def startup(self) -> None:
        async with self._engine.begin() as conn:
            await conn.run_sync(BaseSegmentLinker.metadata.create_all)

    @override
    async def shutdown(self) -> None:
        pass

    @override
    async def open_partition(
        self, partition_key: str
    ) -> SQLAlchemySegmentLinkerPartition:
        if self._use_write_lock:
            write_lock = self._write_locks.setdefault(partition_key, asyncio.Lock())
        else:
            write_lock = None
        return SQLAlchemySegmentLinkerPartition(
            partition_key=partition_key,
            create_session=self._create_session,
            write_lock=write_lock,
        )

    @override
    async def close_partition(
        self, segment_linker_partition: SegmentLinkerPartition
    ) -> None:
        pass
