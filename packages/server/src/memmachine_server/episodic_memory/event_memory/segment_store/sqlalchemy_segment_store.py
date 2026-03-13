"""SQLAlchemy implementation of the SegmentLinker interface."""

import logging
from collections import defaultdict
from collections.abc import Callable, Iterable, Mapping
from datetime import datetime
from enum import StrEnum
from typing import ClassVar, override
from uuid import UUID

from pydantic import BaseModel, Field, InstanceOf, TypeAdapter
from sqlalchemy import (
    JSON,
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Uuid,
    and_,
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
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import (
    DeclarativeBase,
    InstrumentedAttribute,
    MappedColumn,
    mapped_column,
)
from sqlalchemy.sql.elements import ColumnElement

from memmachine_server.common.data_types import PropertyValue
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
    Segment,
)
from memmachine_server.episodic_memory.extra_memory.segment_store.segment_store import (
    DerivativeNotActiveError,
    SegmentLinker,
)

logger = logging.getLogger(__name__)

_JSON_AUTO = JSON().with_variant(JSONB, "postgresql")
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

    partition_key = mapped_column(String, nullable=False)

    uuid: MappedColumn[UUID] = mapped_column(Uuid, primary_key=True)
    episode_uuid: MappedColumn[UUID] = mapped_column(Uuid, nullable=False)
    index: MappedColumn[int] = mapped_column(Integer, nullable=False)
    offset: MappedColumn[int] = mapped_column(Integer, nullable=False)
    timestamp: MappedColumn[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )
    block: MappedColumn[Block] = mapped_column(_JSON_AUTO, nullable=False)

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


class SegmentPropertyRow(BaseSegmentLinker):
    """Property key-value pair for a segment."""

    __tablename__ = "segment_linker_segment_properties"

    segment_uuid = mapped_column(
        Uuid,
        ForeignKey("segment_linker_segments.uuid", ondelete="CASCADE"),
        primary_key=True,
    )
    key: MappedColumn[str] = mapped_column(String, primary_key=True)
    value_bool: MappedColumn[bool] = mapped_column(Boolean, nullable=True)
    value_int: MappedColumn[int] = mapped_column(Integer, nullable=True)
    value_float: MappedColumn[float] = mapped_column(Float, nullable=True)
    value_str: MappedColumn[str] = mapped_column(String, nullable=True)
    value_datetime: MappedColumn[datetime] = mapped_column(
        DateTime(timezone=True), nullable=True
    )


class LinkRow(BaseSegmentLinker):
    """Many-to-many join between segments and derivatives."""

    __tablename__ = "segment_linker_links"

    segment_uuid: MappedColumn[UUID] = mapped_column(
        Uuid,
        ForeignKey("segment_linker_segments.uuid"),
        primary_key=True,
    )
    derivative_uuid: MappedColumn[UUID] = mapped_column(
        Uuid,
        ForeignKey("segment_linker_derivatives.uuid"),
        primary_key=True,
    )

    __table_args__ = (
        Index("segment_linker_links__derivative_uuid", "derivative_uuid"),
    )


class DerivativeRow(BaseSegmentLinker):
    """Derivative lifecycle record."""

    __tablename__ = "segment_linker_derivatives"

    partition_key: MappedColumn[str] = mapped_column(String, nullable=False)

    uuid: MappedColumn[UUID] = mapped_column(Uuid, primary_key=True)
    state: MappedColumn[str] = mapped_column(String(1), nullable=False)
    ref_count: MappedColumn[int] = mapped_column(Integer, nullable=False, default=0)

    __table_args__ = (
        Index(
            "segment_linker_derivatives__pk_state_ref_count",
            "partition_key",
            "state",
            "ref_count",
        ),
    )


class SQLAlchemySegmentLinkerParams(BaseModel):
    """Parameters for constructing a SQLAlchemySegmentLinker."""

    engine: InstanceOf[AsyncEngine] = Field(..., description="Async SQLAlchemy engine")


class SQLAlchemySegmentLinker(SegmentLinker):
    """SQLAlchemy-backed SegmentLinker."""

    def __init__(self, params: SQLAlchemySegmentLinkerParams) -> None:
        """Initialize with an async SQLAlchemy engine."""
        self._engine = params.engine
        self._create_session = async_sessionmaker(self._engine, expire_on_commit=False)

    # Lifecycle

    @override
    async def startup(self) -> None:
        async with self._engine.begin() as conn:
            await conn.run_sync(BaseSegmentLinker.metadata.create_all)

    @override
    async def shutdown(self) -> None:
        pass

    # Registration

    @override
    async def register_segments(
        self,
        partition_key: str,
        links: Mapping[Segment, Iterable[UUID]],
        *,
        active: Iterable[UUID] | None = None,
    ) -> None:
        links = {
            segment: set(derivative_uuids)
            for segment, derivative_uuids in links.items()
        }
        active = set(active) if active is not None else set()

        link_counts: dict[UUID, int] = defaultdict(int)
        for derivative_uuids in links.values():
            for derivative_uuid in derivative_uuids:
                link_counts[derivative_uuid] += 1

        new_derivative_uuids = set(link_counts.keys()) - active

        async with self._create_session() as session, session.begin():
            # Only lock/fetch derivatives declared as active — skip new ones.
            if active:
                locked_rows = await self._lock_derivatives(session, active)
                self._validate_active_derivatives(partition_key, active, locked_rows)

            await self._insert_links(
                session,
                partition_key,
                links,
                link_counts,
                new_derivative_uuids,
            )

    async def _insert_links(
        self,
        session: AsyncSession,
        partition_key: str,
        links: Mapping[Segment, Iterable[UUID]],
        link_counts: Mapping[UUID, int],
        new_derivative_uuids: Iterable[UUID],
    ) -> None:
        # Insert new derivative rows with their ref counts.
        new_derivative_uuids = set(new_derivative_uuids)
        if new_derivative_uuids:
            await session.execute(
                insert(DerivativeRow),
                [
                    {
                        "uuid": derivative_uuid,
                        "partition_key": partition_key,
                        "state": DerivativeState.ACTIVE,
                        "ref_count": link_counts[derivative_uuid],
                    }
                    for derivative_uuid in new_derivative_uuids
                ],
            )

        # Increment ref counts for existing derivatives.
        existing_derivative_uuids = set(link_counts.keys()) - new_derivative_uuids
        for derivative_uuid in existing_derivative_uuids:
            await session.execute(
                update(DerivativeRow)
                .where(DerivativeRow.uuid == derivative_uuid)
                .values(
                    ref_count=DerivativeRow.ref_count + link_counts[derivative_uuid]
                )
            )

        # Insert segment rows.
        segment_row_values = [
            {
                "uuid": segment.uuid,
                "partition_key": partition_key,
                "episode_uuid": segment.episode_uuid,
                "index": segment.index,
                "offset": segment.offset,
                "timestamp": segment.timestamp,
                "block": segment.block.model_dump(),
            }
            for segment in links
        ]
        if segment_row_values:
            await session.execute(insert(SegmentRow), segment_row_values)

        # Insert segment property rows.
        segment_property_row_values: list[dict[str, PropertyValue | UUID]] = [
            {
                "segment_uuid": segment.uuid,
                "key": property_key,
                SQLAlchemySegmentLinker._property_type_column_name(
                    type(property_value)
                ): property_value,
            }
            for segment in links
            for property_key, property_value in segment.properties.items()
        ]
        if segment_property_row_values:
            await session.execute(
                insert(SegmentPropertyRow), segment_property_row_values
            )

        # Insert link rows.
        link_rows = [
            {"segment_uuid": segment.uuid, "derivative_uuid": derivative_uuid}
            for segment, derivative_uuids in links.items()
            for derivative_uuid in derivative_uuids
        ]
        if link_rows:
            await session.execute(insert(LinkRow), link_rows)

    # Retrieval

    @override
    async def get_segments_by_derivatives(
        self,
        partition_key: str,
        derivative_uuids: Iterable[UUID],
        *,
        limit_per_derivative: int | None = None,
        property_filter: FilterExpr | None = None,
    ) -> Mapping[UUID, Iterable[Segment]]:
        derivative_uuids = list(derivative_uuids)
        if not derivative_uuids:
            return {}

        get_segments_by_derivatives_statement = (
            select(LinkRow.derivative_uuid, SegmentRow)
            .join(LinkRow, LinkRow.segment_uuid == SegmentRow.uuid)
            .where(
                SegmentRow.partition_key == partition_key,
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
            get_segments_by_derivatives_statement = (
                get_segments_by_derivatives_statement.where(
                    SQLAlchemySegmentLinker._compile_property_filter(property_filter)
                )
            )

        async with self._create_session() as session:
            segment_rows = (
                await session.execute(get_segments_by_derivatives_statement)
            ).all()
            segment_uuids: set[UUID] = {
                segment_row.uuid for _, segment_row in segment_rows
            }
            properties_by_segments = await self._load_properties_by_segments(
                session, segment_uuids
            )

        # Rows arrive sorted from DB; PK (segment_uuid, derivative_uuid) guarantees no duplicates.
        segments_by_derivatives: defaultdict[UUID, list[Segment]] = defaultdict(list)
        for derivative_uuid, segment_row in segment_rows:
            segment = SQLAlchemySegmentLinker._segment_from_segment_row(
                segment_row, properties_by_segments[segment_row.uuid]
            )
            segments_by_derivatives[derivative_uuid].append(segment)

        if limit_per_derivative is not None:
            limit_first = limit_per_derivative // 2
            limit_last = limit_per_derivative - limit_first

            return {
                derivative_uuid: segments[:limit_first] + segments[-limit_last:]
                if len(segments) > limit_per_derivative
                else segments
                for derivative_uuid, segments in segments_by_derivatives.items()
            }

        return dict(segments_by_derivatives)

    @override
    async def get_segment_contexts(
        self,
        partition_key: str,
        seed_segment_uuids: Iterable[UUID],
        *,
        max_backward_segments: int = 0,
        max_forward_segments: int = 0,
        property_filter: FilterExpr | None = None,
    ) -> Mapping[UUID, Iterable[Segment]]:
        seed_segment_uuids = list(seed_segment_uuids)
        if not seed_segment_uuids:
            return {}

        async with self._create_session() as session:
            seed_statement = select(SegmentRow).where(
                SegmentRow.uuid.in_(seed_segment_uuids),
                SegmentRow.partition_key == partition_key,
            )
            if property_filter is not None:
                seed_statement = seed_statement.where(
                    SQLAlchemySegmentLinker._compile_property_filter(property_filter)
                )
            seed_segment_rows = (await session.execute(seed_statement)).scalars().all()

            seed_segment_rows_by_uuid: dict[UUID, SegmentRow] = {
                row.uuid: row for row in seed_segment_rows
            }
            if not seed_segment_rows_by_uuid:
                return {}

            # Short-circuit: no context needed.
            if max_backward_segments == 0 and max_forward_segments == 0:
                properties_by_segments = await self._load_properties_by_segments(
                    session, list(seed_segment_rows_by_uuid.keys())
                )
                return {
                    seed_segment_uuid: [
                        SQLAlchemySegmentLinker._segment_from_segment_row(
                            seed_segment_row, properties_by_segments[seed_segment_uuid]
                        )
                    ]
                    for seed_segment_uuid, seed_segment_row in seed_segment_rows_by_uuid.items()
                }

            # Fetch backward/forward context rows.
            if session.bind.dialect.name != "sqlite":
                context_rows_by_seed = await self._get_context_rows_lateral(
                    session,
                    partition_key,
                    seed_segment_rows_by_uuid,
                    max_backward_segments,
                    max_forward_segments,
                    property_filter,
                )
            else:
                context_rows_by_seed = await self._get_context_rows_loop(
                    session,
                    partition_key,
                    seed_segment_rows_by_uuid,
                    max_backward_segments,
                    max_forward_segments,
                    property_filter,
                )

            # Load properties for all segments (seeds + context).
            all_segment_uuids = set(seed_segment_rows_by_uuid.keys())
            for backward_rows, forward_rows in context_rows_by_seed.values():
                for row in backward_rows:
                    all_segment_uuids.add(row.uuid)
                for row in forward_rows:
                    all_segment_uuids.add(row.uuid)

            properties_by_segments = await self._load_properties_by_segments(
                session, all_segment_uuids
            )

            # Assemble results: [backward (reversed) + seed + forward].
            segments_by_seed: dict[UUID, list[Segment]] = {}
            for seed_uuid, seed_row in seed_segment_rows_by_uuid.items():
                backward_rows, forward_rows = context_rows_by_seed.get(
                    seed_uuid, ([], [])
                )
                segments_by_seed[seed_uuid] = [
                    SQLAlchemySegmentLinker._segment_from_segment_row(
                        row, properties_by_segments.get(row.uuid, {})
                    )
                    for row in [*reversed(backward_rows), seed_row, *forward_rows]
                ]

            return segments_by_seed

    async def _get_context_rows_lateral(
        self,
        session: AsyncSession,
        partition_key: str,
        seed_rows_by_uuid: Mapping[UUID, SegmentRow],
        max_backward_segments: int,
        max_forward_segments: int,
        property_filter: FilterExpr | None,
    ) -> dict[UUID, tuple[list[SegmentRow], list[SegmentRow]]]:
        """Fetch backward/forward context using LATERAL joins (non-SQLite)."""
        seeds_subquery = (
            select(
                SegmentRow.uuid.label("seed_uuid"),
                SegmentRow.timestamp.label("seed_timestamp"),
                SegmentRow.episode_uuid.label("seed_episode_uuid"),
                SegmentRow.index.label("seed_index"),
                SegmentRow.offset.label("seed_offset"),
            )
            .where(SegmentRow.uuid.in_(seed_rows_by_uuid.keys()))
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

        async def _lateral_query(
            range_condition: ColumnElement[bool],
            ordering: Iterable[ColumnElement | InstrumentedAttribute],
            limit: int,
        ) -> dict[UUID, list[SegmentRow]]:
            lateral_inner = (
                select(SegmentRow)
                .where(SegmentRow.partition_key == partition_key, range_condition)
                .order_by(*ordering)
                .limit(limit)
                .correlate(seeds_subquery)
            )
            if property_filter is not None:
                lateral_inner = lateral_inner.where(
                    SQLAlchemySegmentLinker._compile_property_filter(property_filter)
                )

            lateral_subquery = lateral_inner.subquery().lateral("context")

            lateral_statement = select(
                seeds_subquery.c.seed_uuid,
                lateral_subquery.c.uuid,
                lateral_subquery.c.episode_uuid,
                lateral_subquery.c.index,
                lateral_subquery.c.offset,
                lateral_subquery.c.timestamp,
                lateral_subquery.c.block,
            ).select_from(seeds_subquery.join(lateral_subquery, true()))

            rows_by_seed: dict[UUID, list[SegmentRow]] = {
                seed_uuid: [] for seed_uuid in seed_rows_by_uuid
            }
            for row in (await session.execute(lateral_statement)).all():
                rows_by_seed[row.seed_uuid].append(
                    SegmentRow(
                        uuid=row.uuid,
                        partition_key=partition_key,
                        episode_uuid=row.episode_uuid,
                        index=row.index,
                        offset=row.offset,
                        timestamp=row.timestamp,
                        block=row.block,
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
            await _lateral_query(
                segment_ordering_columns < seed_ordering_columns,
                reverse_chronological_order,
                max_backward_segments,
            )
            if max_backward_segments > 0
            else {seed_uuid: [] for seed_uuid in seed_rows_by_uuid}
        )

        forward_rows_by_seed = (
            await _lateral_query(
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
        partition_key: str,
        seed_rows_by_uuid: Mapping[UUID, SegmentRow],
        max_backward_segments: int,
        max_forward_segments: int,
        property_filter: FilterExpr | None,
    ) -> dict[UUID, tuple[list[SegmentRow], list[SegmentRow]]]:
        """Fetch backward/forward context per seed (SQLite fallback)."""
        context_rows_by_seed: dict[UUID, tuple[list[SegmentRow], list[SegmentRow]]] = {}

        segment_ordering_columns = tuple_(
            SegmentRow.timestamp,
            SegmentRow.episode_uuid,
            SegmentRow.index,
            SegmentRow.offset,
        )

        compiled_property_filter = (
            SQLAlchemySegmentLinker._compile_property_filter(property_filter)
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
                backward_statement = (
                    select(SegmentRow)
                    .where(
                        SegmentRow.partition_key == partition_key,
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
                    backward_statement = backward_statement.where(
                        compiled_property_filter
                    )
                backward_rows = list(
                    (await session.execute(backward_statement)).scalars().all()
                )

            forward_rows: list[SegmentRow] = []
            if max_forward_segments > 0:
                forward_statement = (
                    select(SegmentRow)
                    .where(
                        SegmentRow.partition_key == partition_key,
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
                    forward_statement = forward_statement.where(
                        compiled_property_filter
                    )
                forward_rows = list(
                    (await session.execute(forward_statement)).scalars().all()
                )

            context_rows_by_seed[seed_uuid] = (backward_rows, forward_rows)

        return context_rows_by_seed

    # Deletion

    @override
    async def delete_segments_by_episodes(
        self,
        partition_key: str,
        episode_uuids: Iterable[UUID],
    ) -> None:
        episode_uuids = list(episode_uuids)
        if not episode_uuids:
            return

        get_segment_uuids_statement = select(SegmentRow.uuid).where(
            SegmentRow.partition_key == partition_key,
            SegmentRow.episode_uuid.in_(episode_uuids),
        )

        async with self._create_session() as session, session.begin():
            segment_uuid_rows = (
                await session.execute(get_segment_uuids_statement)
            ).all()
            segment_uuids: list[UUID] = [
                segment_uuid for (segment_uuid,) in segment_uuid_rows
            ]
            if not segment_uuids:
                return

            await self._delete_segments(session, segment_uuids)

    @override
    async def delete_all_segments(self, partition_key: str) -> None:
        get_segment_uuids_statement = select(SegmentRow.uuid).where(
            SegmentRow.partition_key == partition_key
        )

        async with self._create_session() as session, session.begin():
            segment_uuid_rows = (
                await session.execute(get_segment_uuids_statement)
            ).all()
            segment_uuids: list[UUID] = [
                segment_uuid for (segment_uuid,) in segment_uuid_rows
            ]
            if not segment_uuids:
                return

            await self._delete_segments(session, segment_uuids)

    async def _delete_segments(
        self, session: AsyncSession, segment_uuids: Iterable[UUID]
    ) -> None:
        """Delete segments."""
        derivative_rows = (
            await session.execute(
                select(LinkRow.derivative_uuid)
                .where(LinkRow.segment_uuid.in_(segment_uuids))
                .distinct()
            )
        ).all()
        derivative_uuids: list[UUID] = [
            derivative_uuid for (derivative_uuid,) in derivative_rows
        ]

        if derivative_uuids:
            await self._lock_derivatives(session, derivative_uuids)

            # Get ref count deltas by derivatives.
            extra_ref_counts_statement = (
                select(LinkRow.derivative_uuid, func.count())
                .where(LinkRow.segment_uuid.in_(segment_uuids))
                .group_by(LinkRow.derivative_uuid)
            )
            extra_ref_counts_rows = (
                await session.execute(extra_ref_counts_statement)
            ).all()
            deltas_by_derivatives: dict[UUID, int] = {
                derivative_uuid: -extra_ref_count
                for derivative_uuid, extra_ref_count in extra_ref_counts_rows
            }

            # Delete links.
            await session.execute(
                delete(LinkRow).where(LinkRow.segment_uuid.in_(segment_uuids))
            )

            # Decrement ref_counts.
            for derivative_uuid, delta in deltas_by_derivatives.items():
                await session.execute(
                    update(DerivativeRow)
                    .where(DerivativeRow.uuid == derivative_uuid)
                    .values(ref_count=DerivativeRow.ref_count + delta)
                )

        # Delete segments (properties cascade via FK ondelete="CASCADE").
        await session.execute(
            delete(SegmentRow).where(SegmentRow.uuid.in_(segment_uuids))
        )

    # Garbage collection

    @override
    async def get_orphaned_derivatives(
        self, partition_key: str, limit: int = 1000
    ) -> Iterable[UUID]:
        get_orphans_statement = (
            select(DerivativeRow.uuid)
            .where(
                DerivativeRow.partition_key == partition_key,
                DerivativeRow.state == DerivativeState.ACTIVE,
                DerivativeRow.ref_count == 0,
            )
            .limit(limit)
        )
        async with self._create_session() as session:
            return list((await session.execute(get_orphans_statement)).scalars().all())

    @override
    async def mark_orphaned_derivatives_for_purging(
        self, partition_key: str, potential_orphan_uuids: Iterable[UUID]
    ) -> Iterable[UUID]:
        potential_orphan_uuids = sorted(set(potential_orphan_uuids))
        if not potential_orphan_uuids:
            return []

        async with self._create_session() as session, session.begin():
            # Lock candidates that are still orphaned.
            lock_statement = (
                select(DerivativeRow.uuid)
                .where(
                    DerivativeRow.partition_key == partition_key,
                    DerivativeRow.uuid.in_(potential_orphan_uuids),
                    DerivativeRow.state == DerivativeState.ACTIVE,
                    DerivativeRow.ref_count == 0,
                )
                .order_by(DerivativeRow.uuid)
                .with_for_update()
            )
            orphan_uuids = list((await session.execute(lock_statement)).scalars().all())

            if orphan_uuids:
                await session.execute(
                    update(DerivativeRow)
                    .where(DerivativeRow.uuid.in_(orphan_uuids))
                    .values(state=DerivativeState.PURGING)
                )

            return orphan_uuids

    @override
    async def purge_derivatives(
        self, partition_key: str, derivative_uuids: Iterable[UUID]
    ) -> None:
        derivative_uuids = set(derivative_uuids)
        if not derivative_uuids:
            return

        async with self._create_session() as session, session.begin():
            await session.execute(
                delete(DerivativeRow).where(
                    DerivativeRow.partition_key == partition_key,
                    DerivativeRow.uuid.in_(derivative_uuids),
                    DerivativeRow.state == DerivativeState.PURGING,
                )
            )

    # Helpers

    async def _lock_derivatives(
        self, session: AsyncSession, derivative_uuids: Iterable[UUID]
    ) -> list[DerivativeRow]:
        """Lock derivative rows in sorted UUID order. Falls back to plain SELECT on SQLite."""
        sorted_uuids = sorted(set(derivative_uuids))
        if not sorted_uuids:
            return []

        lock_statement = (
            select(DerivativeRow)
            .where(DerivativeRow.uuid.in_(sorted_uuids))
            .order_by(DerivativeRow.uuid)
        )

        # SQLite doesn't support FOR UPDATE
        dialect = session.bind.dialect.name
        if dialect != "sqlite":
            lock_statement = lock_statement.with_for_update()

        return list((await session.execute(lock_statement)).scalars().all())

    async def _load_properties_by_segments(
        self,
        session: AsyncSession,
        segment_uuids: Iterable[UUID],
    ) -> dict[UUID, dict[str, PropertyValue]]:
        """Load properties for a batch of segment UUIDs."""
        segment_uuids = set(segment_uuids)

        if not segment_uuids:
            return {}

        segment_property_rows = (
            (
                await session.execute(
                    select(SegmentPropertyRow).where(
                        SegmentPropertyRow.segment_uuid.in_(segment_uuids)
                    )
                )
            )
            .scalars()
            .all()
        )
        properties_by_segments: dict[UUID, dict[str, PropertyValue]] = {
            segment_uuid: {} for segment_uuid in segment_uuids
        }
        for segment_property_row in segment_property_rows:
            properties_by_segments[segment_property_row.segment_uuid][
                segment_property_row.key
            ] = SQLAlchemySegmentLinker._coalesce_property_value(segment_property_row)
        return properties_by_segments

    @staticmethod
    def _validate_active_derivatives(
        partition_key: str,
        active: Iterable[UUID],
        existing: Iterable[DerivativeRow],
    ) -> None:
        """Validate that every UUID in `active` exists, is in ACTIVE state, and belongs to the partition."""
        active = set(active)
        existing_map = {row.uuid: row for row in existing}
        not_active = {
            derivative_uuid
            for derivative_uuid in active
            if (row := existing_map.get(derivative_uuid)) is None
            or row.state == DerivativeState.PURGING
            or row.partition_key != partition_key
        }
        if not_active:
            raise DerivativeNotActiveError(not_active)

    _COMPARISON_OPERATORS: ClassVar[
        dict[
            str,
            Callable[
                [InstrumentedAttribute[PropertyValue], PropertyValue],
                ColumnElement[bool],
            ],
        ]
    ] = {
        "=": lambda column, value: column == value,
        "!=": lambda column, value: column != value,
        ">": lambda column, value: column > value,
        "<": lambda column, value: column < value,
        ">=": lambda column, value: column >= value,
        "<=": lambda column, value: column <= value,
    }

    @staticmethod
    def _compile_property_filter(expr: FilterExpr) -> ColumnElement[bool]:
        """Compile a FilterExpr into a SQLAlchemy boolean expression."""
        if isinstance(expr, Comparison):
            value_column = SQLAlchemySegmentLinker._property_type_column_attribute(
                type(expr.value)
            )
            comparison_op = SQLAlchemySegmentLinker._COMPARISON_OPERATORS[expr.op]
            return (
                select(1)
                .select_from(SegmentPropertyRow)
                .where(
                    SegmentPropertyRow.segment_uuid == SegmentRow.uuid,
                    SegmentPropertyRow.key == expr.field,
                    comparison_op(value_column, expr.value),
                )
                .exists()
            )

        if isinstance(expr, In):
            first_value = expr.values[0] if expr.values else None
            value_column = (
                SegmentPropertyRow.value_int
                if isinstance(first_value, int)
                else SegmentPropertyRow.value_str
            )
            return (
                select(1)
                .select_from(SegmentPropertyRow)
                .where(
                    SegmentPropertyRow.segment_uuid == SegmentRow.uuid,
                    SegmentPropertyRow.key == expr.field,
                    value_column.in_(expr.values),
                )
                .exists()
            )

        if isinstance(expr, IsNull):
            return ~(
                select(1)
                .select_from(SegmentPropertyRow)
                .where(
                    SegmentPropertyRow.segment_uuid == SegmentRow.uuid,
                    SegmentPropertyRow.key == expr.field,
                )
                .exists()
            )

        if isinstance(expr, Not):
            return ~SQLAlchemySegmentLinker._compile_property_filter(expr.expr)

        if isinstance(expr, And):
            return and_(
                SQLAlchemySegmentLinker._compile_property_filter(expr.left),
                SQLAlchemySegmentLinker._compile_property_filter(expr.right),
            )

        if isinstance(expr, Or):
            return or_(
                SQLAlchemySegmentLinker._compile_property_filter(expr.left),
                SQLAlchemySegmentLinker._compile_property_filter(expr.right),
            )

        raise TypeError(f"Unsupported filter expression type: {type(expr)!r}")

    @staticmethod
    def _property_type_column_attribute(
        property_type: type[PropertyValue],
    ) -> InstrumentedAttribute[PropertyValue]:
        """Return the SQLAlchemy column attribute for a given property type."""
        if property_type is bool:
            return SegmentPropertyRow.value_bool
        if property_type is int:
            return SegmentPropertyRow.value_int
        if property_type is float:
            return SegmentPropertyRow.value_float
        if property_type is str:
            return SegmentPropertyRow.value_str
        if property_type is datetime:
            return SegmentPropertyRow.value_datetime
        raise ValueError(f"Unsupported property value type: {property_type!r}")

    @staticmethod
    def _property_type_column_name(property_type: type[PropertyValue]) -> str:
        """Return the column name for a given property type."""
        if property_type is bool:
            return "value_bool"
        if property_type is int:
            return "value_int"
        if property_type is float:
            return "value_float"
        if property_type is str:
            return "value_str"
        if property_type is datetime:
            return "value_datetime"
        raise ValueError(f"Unsupported property value type: {property_type!r}")

    @staticmethod
    def _coalesce_property_value(row: SegmentPropertyRow) -> PropertyValue:
        """Read back whichever value_* column is non-null."""
        if row.value_bool is not None:
            return row.value_bool
        if row.value_int is not None:
            return row.value_int
        if row.value_float is not None:
            return row.value_float
        if row.value_str is not None:
            return row.value_str
        if row.value_datetime is not None:
            return row.value_datetime
        raise ValueError(f"Property row for key={row.key!r} has no value set")

    @staticmethod
    def _segment_from_segment_row(
        row: SegmentRow,
        properties: dict[str, PropertyValue],
    ) -> Segment:
        """Convert a SegmentRow and its properties into a Segment."""
        block = _BlockAdapter.validate_python(row.block)
        return Segment(
            uuid=row.uuid,
            episode_uuid=row.episode_uuid,
            index=row.index,
            offset=row.offset,
            timestamp=row.timestamp,
            block=block,
            properties=properties,
        )
