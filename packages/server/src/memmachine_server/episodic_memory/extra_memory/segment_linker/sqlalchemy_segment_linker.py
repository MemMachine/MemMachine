"""SQLAlchemy implementation of the SegmentLinker interface."""

import logging
from collections import defaultdict
from collections.abc import Callable, Iterable, Mapping
from datetime import datetime
from enum import StrEnum
from typing import ClassVar, cast, override
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
    Context,
    Segment,
)
from memmachine_server.episodic_memory.extra_memory.segment_linker.segment_linker import (
    DerivativeNotActiveError,
    SegmentLinker,
    SegmentLinkerPartition,
)

logger = logging.getLogger(__name__)

_JSON_AUTO = JSON().with_variant(JSONB, "postgresql")
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

    partition_key: MappedColumn[str] = mapped_column(String, nullable=False)

    uuid: MappedColumn[UUID] = mapped_column(Uuid, primary_key=True)
    episode_uuid: MappedColumn[UUID] = mapped_column(Uuid, nullable=False)
    index: MappedColumn[int] = mapped_column(Integer, nullable=False)
    offset: MappedColumn[int] = mapped_column(Integer, nullable=False)
    timestamp: MappedColumn[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )
    context: MappedColumn[Context | None] = mapped_column(_JSON_AUTO, nullable=True)
    block: MappedColumn[Block] = mapped_column(_JSON_AUTO, nullable=False)

    pending_delete: MappedColumn[bool] = mapped_column(
        Boolean, nullable=False, server_default=false()
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
        Index(
            "segment_linker_segments__pd",
            "pending_delete",
        ),
    )


class SegmentPropertyRow(BaseSegmentLinker):
    """Property key-value pair for a segment."""

    __tablename__ = "segment_linker_segment_properties"

    segment_uuid: MappedColumn[UUID] = mapped_column(
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
        ForeignKey("segment_linker_segments.uuid", ondelete="CASCADE"),
        primary_key=True,
    )
    derivative_uuid: MappedColumn[UUID] = mapped_column(
        Uuid,
        ForeignKey("segment_linker_derivatives.uuid", ondelete="CASCADE"),
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
    owner_segment_uuid: MappedColumn[UUID | None] = mapped_column(
        Uuid,
        ForeignKey("segment_linker_segments.uuid"),
        nullable=True,
    )

    __table_args__ = (
        Index(
            "segment_linker_derivatives__pk_st_os",
            "partition_key",
            "state",
            "owner_segment_uuid",
        ),
        Index(
            "segment_linker_derivatives__os",
            "owner_segment_uuid",
        ),
    )


class SQLAlchemySegmentLinkerPartition(SegmentLinkerPartition):
    """SQLAlchemy-backed partition handle."""

    def __init__(
        self,
        partition_key: str,
        deletion_batch_size: int,
        create_session: async_sessionmaker[AsyncSession],
    ) -> None:
        """Initialize with a partition key and session maker."""
        self._partition_key = partition_key
        self._deletion_batch_size = deletion_batch_size
        self._create_session = create_session

    # Registration

    @override
    async def register_segments(
        self,
        links: Mapping[Segment, Iterable[UUID]],
        *,
        active: Iterable[UUID] | None = None,
    ) -> None:
        links = {
            segment: set(derivative_uuids)
            for segment, derivative_uuids in links.items()
        }
        active_derivative_uuids = set(active) if active is not None else set()

        all_derivative_uuids: set[UUID] = set()
        for derivative_uuids in links.values():
            all_derivative_uuids.update(derivative_uuids)

        new_derivative_uuids = all_derivative_uuids - active_derivative_uuids

        async with self._create_session() as session, session.begin():
            orphaned_derivative_uuids: set[UUID] = set()
            if active_derivative_uuids:
                lock_statement = (
                    select(DerivativeRow)
                    .where(DerivativeRow.uuid.in_(active_derivative_uuids))
                    .order_by(DerivativeRow.uuid)
                )
                if session.bind.dialect.name != "sqlite":
                    lock_statement = lock_statement.with_for_update()
                locked_rows = list(
                    (await session.execute(lock_statement)).scalars().all()
                )
                self._validate_active_derivatives(active_derivative_uuids, locked_rows)

                orphaned_derivative_uuids = {
                    row.uuid for row in locked_rows if row.owner_segment_uuid is None
                }

            new_derivative_owner_uuids, orphaned_derivative_owner_uuids = (
                self._assign_owners(
                    links, new_derivative_uuids, orphaned_derivative_uuids
                )
            )

            await self._insert_segments(session, links.keys())
            await self._insert_new_derivatives(session, new_derivative_owner_uuids)
            await self._rescue_orphaned_derivatives(
                session, orphaned_derivative_owner_uuids
            )
            await self._insert_links(session, links)

    def _assign_owners(
        self,
        links: Mapping[Segment, Iterable[UUID]],
        new_derivative_uuids: Iterable[UUID],
        orphaned_derivative_uuids: Iterable[UUID],
    ) -> tuple[dict[UUID, UUID], dict[UUID, UUID]]:
        """Pick one owner segment per new/orphaned derivative (first seen)."""
        new_derivative_uuids = set(new_derivative_uuids)
        orphaned_derivative_uuids = set(orphaned_derivative_uuids)

        new_derivative_owner_uuids: dict[UUID, UUID] = {}
        orphaned_derivative_owner_uuids: dict[UUID, UUID] = {}

        for segment, derivative_uuids in links.items():
            for derivative_uuid in derivative_uuids:
                if derivative_uuid in new_derivative_uuids:
                    new_derivative_owner_uuids.setdefault(derivative_uuid, segment.uuid)
                elif derivative_uuid in orphaned_derivative_uuids:
                    orphaned_derivative_owner_uuids.setdefault(
                        derivative_uuid, segment.uuid
                    )

        return new_derivative_owner_uuids, orphaned_derivative_owner_uuids

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
                    segment.context.model_dump()
                    if segment.context is not None
                    else None
                ),
                "block": segment.block.model_dump(),
            }
            for segment in segments
        ]
        if segment_row_values:
            await session.execute(insert(SegmentRow), segment_row_values)

        property_row_values: list[dict[str, PropertyValue | UUID]] = [
            {
                "segment_uuid": segment.uuid,
                "key": property_key,
                SQLAlchemySegmentLinkerPartition._property_type_column_name(
                    type(property_value)
                ): property_value,
            }
            for segment in segments
            for property_key, property_value in segment.properties.items()
        ]
        if property_row_values:
            await session.execute(insert(SegmentPropertyRow), property_row_values)

    async def _insert_new_derivatives(
        self,
        session: AsyncSession,
        new_derivative_owner_uuids: Mapping[UUID, UUID],
    ) -> None:
        """Insert new derivative rows with their initial owners."""
        if new_derivative_owner_uuids:
            await session.execute(
                insert(DerivativeRow),
                [
                    {
                        "uuid": derivative_uuid,
                        "partition_key": self._partition_key,
                        "state": DerivativeState.ACTIVE,
                        "owner_segment_uuid": owner_segment_uuid,
                    }
                    for derivative_uuid, owner_segment_uuid in new_derivative_owner_uuids.items()
                ],
            )

    async def _rescue_orphaned_derivatives(
        self,
        session: AsyncSession,
        orphaned_derivative_owner_uuids: Mapping[UUID, UUID],
    ) -> None:
        """Set owner for orphaned derivatives."""
        if not orphaned_derivative_owner_uuids:
            return
        await session.execute(
            update(DerivativeRow).execution_options(synchronize_session=False),
            [
                {"uuid": derivative_uuid, "owner_segment_uuid": owner_segment_uuid}
                for derivative_uuid, owner_segment_uuid in orphaned_derivative_owner_uuids.items()
            ],
        )

    async def _insert_links(
        self,
        session: AsyncSession,
        links: Mapping[Segment, Iterable[UUID]],
    ) -> None:
        """Insert link rows between segments and derivatives."""
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
        """Fetch all segments for each derivative."""
        derivative_uuids = set(derivative_uuids)

        segments_by_derivatives_statement = (
            select(LinkRow.derivative_uuid, SegmentRow)
            .join(LinkRow, LinkRow.segment_uuid == SegmentRow.uuid)
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
            segments_by_derivatives_statement = segments_by_derivatives_statement.where(
                SQLAlchemySegmentLinkerPartition._compile_property_filter(
                    property_filter
                )
            )

        async with self._create_session() as session:
            segment_rows = (
                await session.execute(segments_by_derivatives_statement)
            ).all()
            segment_uuids: set[UUID] = {
                segment_row.uuid for _, segment_row in segment_rows
            }
            properties_by_segments = await self._load_properties_by_segments(
                session, segment_uuids
            )

        segments_by_derivatives: defaultdict[UUID, list[Segment]] = defaultdict(list)
        for derivative_uuid, segment_row in segment_rows:
            segment = SQLAlchemySegmentLinkerPartition._segment_from_segment_row(
                segment_row, properties_by_segments[segment_row.uuid]
            )
            segments_by_derivatives[derivative_uuid].append(segment)

        return dict(segments_by_derivatives)

    async def _get_segments_by_derivatives_windowed(
        self,
        derivative_uuids: Iterable[UUID],
        limit_per_derivative: int,
        property_filter: FilterExpr | None,
    ) -> dict[UUID, list[Segment]]:
        """Fetch segments with limit_per_derivative applied via SQL window functions."""
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

        numbered_derivative_segments_statement = (
            select(
                LinkRow.derivative_uuid,
                SegmentRow.uuid,
                SegmentRow.episode_uuid,
                SegmentRow.index,
                SegmentRow.offset,
                SegmentRow.timestamp,
                SegmentRow.context,
                SegmentRow.block,
                row_number_ascending,
                row_number_descending,
            )
            .join(LinkRow, LinkRow.segment_uuid == SegmentRow.uuid)
            .where(
                SegmentRow.partition_key == self._partition_key,
                LinkRow.derivative_uuid.in_(derivative_uuids),
            )
        )
        if property_filter is not None:
            numbered_derivative_segments_statement = (
                numbered_derivative_segments_statement.where(
                    SQLAlchemySegmentLinkerPartition._compile_property_filter(
                        property_filter
                    )
                )
            )

        numbered_derivative_segments_subquery = (
            numbered_derivative_segments_statement.subquery()
        )

        limited_derivative_segments_statement = (
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

        async with self._create_session() as session:
            limited_derivative_segment_rows = (
                await session.execute(limited_derivative_segments_statement)
            ).all()
            segment_uuids = {row.uuid for row in limited_derivative_segment_rows}
            properties_by_segments = await self._load_properties_by_segments(
                session, segment_uuids
            )

        segments_by_derivatives: defaultdict[UUID, list[Segment]] = defaultdict(list)
        for row in limited_derivative_segment_rows:
            segment = SQLAlchemySegmentLinkerPartition._segment_from_segment_row(
                cast(SegmentRow, row), properties_by_segments[row.uuid]
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
            seed_statement = select(SegmentRow).where(
                SegmentRow.uuid.in_(seed_segment_uuids),
                SegmentRow.partition_key == self._partition_key,
            )
            if property_filter is not None:
                seed_statement = seed_statement.where(
                    SQLAlchemySegmentLinkerPartition._compile_property_filter(
                        property_filter
                    )
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
                        SQLAlchemySegmentLinkerPartition._segment_from_segment_row(
                            seed_segment_row,
                            properties_by_segments[seed_segment_uuid],
                        )
                    ]
                    for seed_segment_uuid, seed_segment_row in seed_segment_rows_by_uuid.items()
                }

            # Fetch backward/forward context rows.
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
                    SQLAlchemySegmentLinkerPartition._segment_from_segment_row(
                        row, properties_by_segments.get(row.uuid, {})
                    )
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

        partition_key = self._partition_key

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
                    SQLAlchemySegmentLinkerPartition._compile_property_filter(
                        property_filter
                    )
                )

            lateral_subquery = lateral_inner.subquery().lateral("context")

            lateral_statement = select(
                seeds_subquery.c.seed_uuid,
                lateral_subquery.c.uuid,
                lateral_subquery.c.episode_uuid,
                lateral_subquery.c.index,
                lateral_subquery.c.offset,
                lateral_subquery.c.timestamp,
                lateral_subquery.c.context,
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
                        context=row.context,
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
                backward_statement = (
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
        episode_uuids: Iterable[UUID],
    ) -> None:
        episode_uuids = set(episode_uuids)
        if not episode_uuids:
            return
        await self._flag_and_batch_delete(
            and_(
                SegmentRow.partition_key == self._partition_key,
                SegmentRow.episode_uuid.in_(episode_uuids),
            )
        )

    @override
    async def delete_all_segments(self) -> None:
        await self._flag_and_batch_delete(
            SegmentRow.partition_key == self._partition_key
        )

    async def _flag_and_batch_delete(self, base_filter: ColumnElement[bool]) -> None:
        # Phase 1: atomically stamp the exact set to delete.
        # NOT pending_delete ensures new inserts (FALSE by default) are never touched,
        # and concurrent callers don't double-mark already-pending rows.
        async with self._create_session() as session, session.begin():
            await session.execute(
                update(SegmentRow)
                .where(base_filter, SegmentRow.pending_delete.is_(False))
                .values(pending_delete=True)
            )

        # Phase 2: batch-delete all marked rows.
        # Picks up orphaned marks from prior crashed runs — self-healing.
        while True:
            async with self._create_session() as session, session.begin():
                batch = list(
                    (
                        await session.execute(
                            select(SegmentRow.uuid)
                            .where(
                                SegmentRow.partition_key == self._partition_key,
                                SegmentRow.pending_delete.is_(True),
                            )
                            .limit(self._deletion_batch_size)
                        )
                    )
                    .scalars()
                    .all()
                )
                if not batch:
                    break
                await self._delete_segments(session, batch)

            if len(batch) < self._deletion_batch_size:
                break

    async def _delete_segments(
        self, session: AsyncSession, segment_uuids: Iterable[UUID]
    ) -> None:
        """Delete segments and reassign derivative owners."""
        segment_uuids = list(segment_uuids)

        # Step 1: Lock derivatives whose owner is being deleted, in sorted order to prevent deadlocks.
        lock_statement = (
            select(DerivativeRow.uuid)
            .where(DerivativeRow.owner_segment_uuid.in_(segment_uuids))
            .order_by(DerivativeRow.uuid)
        )
        if session.bind.dialect.name != "sqlite":
            lock_statement = lock_statement.with_for_update()
        affected_derivative_uuids = list(
            (await session.execute(lock_statement)).scalars().all()
        )

        # Step 2: Reassign owners via correlated subquery.
        # Finds a replacement from links that are not in the deleted batch
        # and not pending deletion. Sets NULL if no replacement exists.
        if affected_derivative_uuids:
            replacement_owner_subquery = (
                select(LinkRow.segment_uuid)
                .join(SegmentRow, LinkRow.segment_uuid == SegmentRow.uuid)
                .where(
                    LinkRow.derivative_uuid == DerivativeRow.uuid,
                    LinkRow.segment_uuid.not_in(segment_uuids),
                    SegmentRow.pending_delete.is_(False),
                )
                .limit(1)
                .correlate(DerivativeRow)
                .scalar_subquery()
            )
            await session.execute(
                update(DerivativeRow)
                .where(DerivativeRow.uuid.in_(affected_derivative_uuids))
                .values(owner_segment_uuid=replacement_owner_subquery)
            )

        # Step 3: Delete segment rows (links and properties cascade via FK ondelete="CASCADE").
        await session.execute(
            delete(SegmentRow).where(SegmentRow.uuid.in_(segment_uuids))
        )

    # Garbage collection

    @override
    async def mark_orphaned_derivatives_for_purging(self, limit: int = 1000) -> None:
        async with self._create_session() as session, session.begin():
            # Find and lock orphaned derivatives.
            find_orphans = (
                select(DerivativeRow.uuid)
                .where(
                    DerivativeRow.partition_key == self._partition_key,
                    DerivativeRow.state == DerivativeState.ACTIVE,
                    DerivativeRow.owner_segment_uuid.is_(None),
                )
                .limit(limit)
            )

            if session.bind.dialect.name != "sqlite":
                # SKIP LOCKED lets concurrent GC callers pick non-overlapping batches
                # and avoids deadlocks without ordering (never waits).
                find_orphans = find_orphans.with_for_update(skip_locked=True)
            orphan_uuids = list((await session.execute(find_orphans)).scalars().all())

            # Mark locked orphans for purging.
            if orphan_uuids:
                await session.execute(
                    update(DerivativeRow)
                    .where(DerivativeRow.uuid.in_(orphan_uuids))
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

        async with self._create_session() as session, session.begin():
            await session.execute(
                delete(DerivativeRow).where(
                    DerivativeRow.partition_key == self._partition_key,
                    DerivativeRow.uuid.in_(derivative_uuids),
                    DerivativeRow.state == DerivativeState.PURGING,
                )
            )

    # Helpers

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
            ] = SQLAlchemySegmentLinkerPartition._coalesce_property_value(
                segment_property_row
            )
        return properties_by_segments

    def _validate_active_derivatives(
        self,
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
            or row.partition_key != self._partition_key
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
            value_column = (
                SQLAlchemySegmentLinkerPartition._property_type_column_attribute(
                    type(expr.value)
                )
            )
            comparison_op = SQLAlchemySegmentLinkerPartition._COMPARISON_OPERATORS[
                expr.op
            ]
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
        properties: Mapping[str, PropertyValue],
    ) -> Segment:
        """Convert a SegmentRow and its properties into a Segment."""
        context = _ContextAdapter.validate_python(row.context)
        block = _BlockAdapter.validate_python(row.block)
        return Segment(
            uuid=row.uuid,
            episode_uuid=row.episode_uuid,
            index=row.index,
            offset=row.offset,
            timestamp=row.timestamp,
            context=context,
            block=block,
            properties=dict(properties),
        )


class SQLAlchemySegmentLinkerParams(BaseModel):
    """Parameters for constructing a SQLAlchemySegmentLinker."""

    engine: InstanceOf[AsyncEngine] = Field(..., description="Async SQLAlchemy engine")
    deletion_batch_size: int = Field(
        1000, description="Segments deleted per batch in batched deletion", gt=0
    )


class SQLAlchemySegmentLinker(SegmentLinker):
    """SQLAlchemy-backed SegmentLinker factory."""

    def __init__(self, params: SQLAlchemySegmentLinkerParams) -> None:
        """Initialize with an async SQLAlchemy engine."""
        self._engine = params.engine
        self._create_session = async_sessionmaker(self._engine, expire_on_commit=False)
        self._deletion_batch_size = params.deletion_batch_size

    @override
    async def startup(self) -> None:
        async with self._engine.begin() as conn:
            await conn.run_sync(BaseSegmentLinker.metadata.create_all)

    @override
    async def shutdown(self) -> None:
        pass

    @override
    def get_partition(self, partition_key: str) -> SQLAlchemySegmentLinkerPartition:
        """Get a partition-scoped handle for the given partition key."""
        return SQLAlchemySegmentLinkerPartition(
            partition_key=partition_key,
            create_session=self._create_session,
            deletion_batch_size=self._deletion_batch_size,
        )
