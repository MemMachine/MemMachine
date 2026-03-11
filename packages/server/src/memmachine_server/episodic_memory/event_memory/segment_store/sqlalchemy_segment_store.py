"""SQLAlchemy implementation of the SegmentLinker interface."""

import logging
from collections import defaultdict
from collections.abc import Iterable, Mapping
from datetime import datetime
from enum import StrEnum
from typing import override
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
    or_,
    select,
    tuple_,
    update,
)
from sqlalchemy import (
    true as sa_true,
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
    Content,
    Segment,
)
from memmachine_server.episodic_memory.extra_memory.segment_store.segment_store import (
    DerivativeNotActiveError,
    SegmentLinker,
)

logger = logging.getLogger(__name__)

_JSON_AUTO = JSON().with_variant(JSONB, "postgresql")
_ContentAdapter = TypeAdapter(Content)


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

    uuid = mapped_column(Uuid, primary_key=True)
    episode_uuid = mapped_column(Uuid, nullable=False)
    block = mapped_column(Integer, nullable=False)
    index = mapped_column(Integer, nullable=False)
    timestamp = mapped_column(DateTime(timezone=True), nullable=False)
    content = mapped_column(_JSON_AUTO, nullable=False)

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
            "block",
            "index",
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
    key = mapped_column(String, primary_key=True)
    value_bool = mapped_column(Boolean, nullable=True)
    value_int = mapped_column(Integer, nullable=True)
    value_float = mapped_column(Float, nullable=True)
    value_str = mapped_column(String, nullable=True)
    value_datetime = mapped_column(DateTime(timezone=True), nullable=True)

    __table_args__ = (
        Index(
            "segment_linker_segment_properties__key_bool",
            "key",
            "value_bool",
        ),
        Index(
            "segment_linker_segment_properties__key_int",
            "key",
            "value_int",
        ),
        Index(
            "segment_linker_segment_properties__key_float",
            "key",
            "value_float",
        ),
        Index(
            "segment_linker_segment_properties__key_str",
            "key",
            "value_str",
        ),
        Index(
            "segment_linker_segment_properties__key_datetime",
            "key",
            "value_datetime",
        ),
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

    uuid = mapped_column(Uuid, primary_key=True)
    state = mapped_column(String(1), nullable=False)
    ref_count = mapped_column(Integer, nullable=False, default=0)

    __table_args__ = (
        Index("segment_linker_derivatives__state_ref_count", "state", "ref_count"),
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

    async def shutdown(self) -> None:
        pass

    # Registration

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

        all_derivative_uuids = set(link_counts.keys()) | active

        async with self._create_session() as session, session.begin():
            locked_rows = await self._lock_derivatives(session, all_derivative_uuids)
            locked_map = {r.uuid: r for r in locked_rows}

            self._validate_derivatives(active, link_counts, locked_map)

            new_derivative_uuids = set(link_counts.keys()) - active
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
                "block": segment.block,
                "index": segment.index,
                "timestamp": segment.timestamp,
                "content": segment.content.model_dump(),
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
                SegmentRow.block,
                SegmentRow.index,
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
            segment_uuids = {segment_row.uuid for _, segment_row in segment_rows}
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
            # Step 1: Fetch seed rows
            seed_result = await session.execute(
                select(SegmentRow).where(
                    SegmentRow.uuid.in_(seed_segment_uuids),
                    SegmentRow.partition_key == partition_key,
                )
            )
            seed_rows: dict[UUID, SegmentRow] = {
                r.uuid: r for r in seed_result.scalars().all()
            }
            if not seed_rows:
                return {}

            # Short-circuit: no context needed
            if max_backward_segments == 0 and max_forward_segments == 0:
                props = await self._load_properties_by_segments(
                    session, list(seed_rows.keys())
                )
                return {
                    u: [
                        SQLAlchemySegmentLinker._segment_from_segment_row(
                            r, props.get(u, {})
                        )
                    ]
                    for u, r in seed_rows.items()
                }

            # Step 2: Fetch context rows
            bind = session.bind
            use_lateral = bind is not None and bind.dialect.name != "sqlite"

            if use_lateral:
                ctx = await self._get_contexts_lateral(
                    session,
                    partition_key,
                    seed_rows,
                    max_backward_segments,
                    max_forward_segments,
                    property_filter,
                )
            else:
                ctx = await self._get_contexts_loop(
                    session,
                    partition_key,
                    seed_rows,
                    max_backward_segments,
                    max_forward_segments,
                    property_filter,
                )

            # Step 3: Load properties for all segments
            all_uuids: set[UUID] = set(seed_rows.keys())
            for bw, fw in ctx.values():
                for r in bw:
                    all_uuids.add(r.uuid)
                for r in fw:
                    all_uuids.add(r.uuid)

            props = await self._load_properties_by_segments(session, list(all_uuids))

            # Step 4: Assemble results
            result: dict[UUID, list[Segment]] = {}
            for seed_uuid, seed_row in seed_rows.items():
                bw, fw = ctx.get(seed_uuid, ([], []))
                segments = (
                    [
                        SQLAlchemySegmentLinker._segment_from_segment_row(
                            r, props.get(r.uuid, {})
                        )
                        for r in reversed(bw)
                    ]
                    + [
                        SQLAlchemySegmentLinker._segment_from_segment_row(
                            seed_row, props.get(seed_uuid, {})
                        )
                    ]
                    + [
                        SQLAlchemySegmentLinker._segment_from_segment_row(
                            r, props.get(r.uuid, {})
                        )
                        for r in fw
                    ]
                )
                result[seed_uuid] = segments

            return result

    # Deletion

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
            segment_uuids = [segment_uuid for (segment_uuid,) in segment_uuid_rows]
            if not segment_uuids:
                return

            await self._delete_segments(session, segment_uuids)

    async def delete_all_segments(self, partition_key: str) -> None:
        get_segment_uuids_statement = select(SegmentRow.uuid).where(
            SegmentRow.partition_key == partition_key
        )

        async with self._create_session() as session, session.begin():
            segment_uuid_rows = (
                await session.execute(get_segment_uuids_statement)
            ).all()
            segment_uuids = [segment_uuid for (segment_uuid,) in segment_uuid_rows]
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
        derivative_uuids = [derivative_uuid for (derivative_uuid,) in derivative_rows]

        if derivative_uuids:
            await self._lock_derivatives(session, derivative_uuids)

            # Get ref count deltas by derivatives.
            delta_statement = (
                select(LinkRow.derivative_uuid, -func.count())
                .where(LinkRow.segment_uuid.in_(segment_uuids))
                .group_by(LinkRow.derivative_uuid)
            )

            delta_rows = (await session.execute(delta_statement)).all()
            deltas_by_derivatives = {derivative_uuid: delta for derivative_uuid, delta in delta_rows}

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

    async def get_orphaned_derivatives(self, limit: int = 1000) -> Iterable[UUID]:
        get_orphans_statement = (
            select(DerivativeRow.uuid)
            .where(
                DerivativeRow.state == DerivativeState.ACTIVE,
                DerivativeRow.ref_count == 0,
            )
            .limit(limit)
        )
        async with self._create_session() as session:
            orphan_rows = (await session.execute(get_orphans_statement)).all()
            return [orphan for (orphan,) in orphan_rows]

    async def mark_orphaned_derivatives_for_purging(
        self, potential_orphan_uuids: Iterable[UUID]
    ) -> Iterable[UUID]:
        uuid_list = sorted(potential_orphan_uuids)
        if not uuid_list:
            return []

        async with self._create_session() as session, session.begin():
            # Lock candidates that are still orphaned
            stmt = (
                select(DerivativeRow.uuid)
                .where(
                    DerivativeRow.uuid.in_(uuid_list),
                    DerivativeRow.state == DerivativeState.ACTIVE,
                    DerivativeRow.ref_count == 0,
                )
                .order_by(DerivativeRow.uuid)
                .with_for_update()
            )
            result = await session.execute(stmt)
            confirmed = [r[0] for r in result.all()]

            if confirmed:
                await session.execute(
                    update(DerivativeRow)
                    .where(DerivativeRow.uuid.in_(confirmed))
                    .values(state=DerivativeState.PURGING)
                )

            return confirmed

    async def purge_derivatives(self, derivative_uuids: Iterable[UUID]) -> None:
        derivative_uuids = list(derivative_uuids)
        if not derivative_uuids:
            return

        async with self._create_session() as session, session.begin():
            await session.execute(
                delete(DerivativeRow).where(
                    DerivativeRow.uuid.in_(derivative_uuids),
                    DerivativeRow.state == DerivativeState.PURGING,
                )
            )

    # -- context helpers -----------------------------------------------------

    async def _get_contexts_lateral(
        self,
        session: AsyncSession,
        partition_key: str,
        seed_rows: dict[UUID, SegmentRow],
        max_backward: int,
        max_forward: int,
        property_filter: FilterExpr | None,
    ) -> dict[UUID, tuple[list[SegmentRow], list[SegmentRow]]]:
        """Fetch backward/forward context using LATERAL joins (non-SQLite)."""
        seeds_sq = (
            select(
                SegmentRow.uuid.label("seed_uuid"),
                SegmentRow.timestamp.label("seed_ts"),
                SegmentRow.episode_uuid.label("seed_ep"),
                SegmentRow.block.label("seed_block"),
                SegmentRow.index.label("seed_index"),
            )
            .where(SegmentRow.uuid.in_(list(seed_rows.keys())))
            .subquery("seeds")
        )

        result: dict[UUID, tuple[list[SegmentRow], list[SegmentRow]]] = {
            u: ([], []) for u in seed_rows
        }

        seg_cols = tuple_(
            SegmentRow.timestamp,
            SegmentRow.episode_uuid,
            SegmentRow.block,
            SegmentRow.index,
        )
        seed_cols = tuple_(
            seeds_sq.c.seed_ts,
            seeds_sq.c.seed_ep,
            seeds_sq.c.seed_block,
            seeds_sq.c.seed_index,
        )

        directions: list[tuple[str, int, ColumnElement[bool], list]] = [
            (
                "B",
                max_backward,
                seg_cols < seed_cols,
                [
                    SegmentRow.timestamp.desc(),
                    SegmentRow.episode_uuid.desc(),
                    SegmentRow.block.desc(),
                    SegmentRow.index.desc(),
                ],
            ),
            (
                "F",
                max_forward,
                seg_cols > seed_cols,
                [
                    SegmentRow.timestamp,
                    SegmentRow.episode_uuid,
                    SegmentRow.block,
                    SegmentRow.index,
                ],
            ),
        ]

        for direction, max_count, cmp, ordering in directions:
            if max_count == 0:
                continue

            inner = (
                select(SegmentRow)
                .where(SegmentRow.partition_key == partition_key, cmp)
                .order_by(*ordering)
                .limit(max_count)
                .correlate(seeds_sq)
            )
            if property_filter is not None:
                inner = inner.where(
                    SQLAlchemySegmentLinker._compile_property_filter(property_filter)
                )

            lat = inner.subquery().lateral("ctx")

            stmt = select(
                seeds_sq.c.seed_uuid,
                lat.c.uuid,
                lat.c.episode_uuid,
                lat.c.block,
                lat.c.index,
                lat.c.timestamp,
                lat.c.content,
            ).select_from(seeds_sq.join(lat, sa_true()))

            rows = (await session.execute(stmt)).all()
            for row in rows:
                seed_uuid = row[0]
                seg = SegmentRow(
                    uuid=row[1],
                    partition_key=partition_key,
                    episode_uuid=row[2],
                    block=row[3],
                    index=row[4],
                    timestamp=row[5],
                    content=row[6],
                )
                idx = 0 if direction == "B" else 1
                result[seed_uuid][idx].append(seg)

        return result

    async def _get_contexts_loop(
        self,
        session: AsyncSession,
        partition_key: str,
        seed_rows: dict[UUID, SegmentRow],
        max_backward: int,
        max_forward: int,
        property_filter: FilterExpr | None,
    ) -> dict[UUID, tuple[list[SegmentRow], list[SegmentRow]]]:
        """Fetch backward/forward context per seed (SQLite fallback)."""
        result: dict[UUID, tuple[list[SegmentRow], list[SegmentRow]]] = {}

        seg_key = tuple_(
            SegmentRow.timestamp,
            SegmentRow.episode_uuid,
            SegmentRow.block,
            SegmentRow.index,
        )

        for seed_uuid, seed in seed_rows.items():
            seed_val = tuple_(
                seed.timestamp,
                seed.episode_uuid,
                seed.block,
                seed.index,
            )

            backward: list[SegmentRow] = []
            if max_backward > 0:
                bw_stmt = (
                    select(SegmentRow)
                    .where(
                        SegmentRow.partition_key == partition_key,
                        seg_key < seed_val,
                    )
                    .order_by(
                        SegmentRow.timestamp.desc(),
                        SegmentRow.episode_uuid.desc(),
                        SegmentRow.block.desc(),
                        SegmentRow.index.desc(),
                    )
                    .limit(max_backward)
                )
                if property_filter is not None:
                    bw_stmt = bw_stmt.where(
                        SQLAlchemySegmentLinker._compile_property_filter(
                            property_filter
                        )
                    )
                bw_result = await session.execute(bw_stmt)
                backward = list(bw_result.scalars().all())

            forward: list[SegmentRow] = []
            if max_forward > 0:
                fw_stmt = (
                    select(SegmentRow)
                    .where(
                        SegmentRow.partition_key == partition_key,
                        seg_key > seed_val,
                    )
                    .order_by(
                        SegmentRow.timestamp,
                        SegmentRow.episode_uuid,
                        SegmentRow.block,
                        SegmentRow.index,
                    )
                    .limit(max_forward)
                )
                if property_filter is not None:
                    fw_stmt = fw_stmt.where(
                        SQLAlchemySegmentLinker._compile_property_filter(
                            property_filter
                        )
                    )
                fw_result = await session.execute(fw_stmt)
                forward = list(fw_result.scalars().all())

            result[seed_uuid] = (backward, forward)

        return result

    # -- internal helpers ----------------------------------------------------

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

        result = await session.execute(lock_statement)
        return list(result.scalars().all())

    async def _load_properties_by_segments(
        self,
        session: AsyncSession,
        segment_uuids: Iterable[UUID],
    ) -> dict[UUID, dict[str, PropertyValue]]:
        """Load properties for a batch of segment UUIDs."""
        segment_uuids = list(segment_uuids)

        if not segment_uuids:
            return {}

        result = await session.execute(
            select(SegmentPropertyRow).where(
                SegmentPropertyRow.segment_uuid.in_(segment_uuids)
            )
        )
        props_map: dict[UUID, dict[str, PropertyValue]] = {u: {} for u in segment_uuids}
        for row in result.scalars().all():
            props_map[row.segment_uuid][row.key] = _coalesce_property_value(row)
        return props_map

    @staticmethod
    def _validate_derivatives(
        active_set: set[UUID],
        link_count: dict[UUID, int],
        locked_map: dict[UUID, DerivativeRow],
    ) -> None:
        """Validate derivative states.

        - Every UUID in ``active_set`` must exist and be active.
        - Every UUID in ``link_count`` that already exists in the DB must
          appear in ``active_set`` (caller must acknowledge existing derivatives).
        """
        safe_active: set[UUID] = active_set if active_set is not None else set()

        not_active: list[UUID] = []
        for u in safe_active:
            row = locked_map.get(u)
            if row is None or row.state == DerivativeState.PURGING:
                not_active.append(u)
        if not_active:
            raise DerivativeNotActiveError(not_active)

        for u in link_count:
            row = locked_map.get(u)
            if row is not None and u not in safe_active:
                raise DerivativeNotActiveError([u])

    @staticmethod
    def _compile_property_filter(expr: FilterExpr) -> ColumnElement[bool]:
        """Compile a FilterExpr into EXISTS subqueries against the properties table."""
        if isinstance(expr, Comparison):
            col = _value_column_element(expr.value)
            sp = SegmentPropertyRow
            cond = sp.segment_uuid == SegmentRow.uuid
            cond = and_(cond, sp.key == expr.field)
            op_map = {
                "=": lambda c, v: c == v,
                "!=": lambda c, v: c != v,
                ">": lambda c, v: c > v,
                "<": lambda c, v: c < v,
                ">=": lambda c, v: c >= v,
                "<=": lambda c, v: c <= v,
            }
            cond = and_(cond, op_map[expr.op](col, expr.value))
            return select(1).select_from(sp.__table__).where(cond).exists()

        if isinstance(expr, In):
            # Determine column by first value type
            if expr.values and isinstance(expr.values[0], int):
                col = SegmentPropertyRow.value_int
            else:
                col = SegmentPropertyRow.value_str
            sp = SegmentPropertyRow
            cond = and_(
                sp.segment_uuid == SegmentRow.uuid,
                sp.key == expr.field,
                col.in_(expr.values),
            )
            return select(1).select_from(sp.__table__).where(cond).exists()

        if isinstance(expr, IsNull):
            # Property is null = no property row exists for this key
            sp = SegmentPropertyRow
            cond = and_(sp.segment_uuid == SegmentRow.uuid, sp.key == expr.field)
            return ~select(1).select_from(sp.__table__).where(cond).exists()

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

        if isinstance(expr, Not):
            return ~SQLAlchemySegmentLinker._compile_property_filter(expr.expr)

        raise TypeError(f"Unsupported filter expression type: {type(expr)!r}")

    @staticmethod
    def _property_type_column_name(property_type: type[PropertyValue]) -> str:
        """Return the column name for the type of a given property value."""
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
    def _segment_from_segment_row(
        row: SegmentRow,
        properties: dict[str, PropertyValue],
    ) -> Segment:
        """Convert a SegmentRow and its properties into a Segment."""
        content = _ContentAdapter.validate_python(row.content)
        return Segment(
            uuid=row.uuid,
            episode_uuid=row.episode_uuid,
            block=row.block,
            index=row.index,
            timestamp=row.timestamp,
            content=content,
            properties=properties,
        )


##########################################


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


def _value_column_element(value: PropertyValue) -> InstrumentedAttribute[PropertyValue]:
    """Return the ORM column element for a property value's type."""
    if isinstance(value, bool):
        return SegmentPropertyRow.value_bool
    if isinstance(value, int):
        return SegmentPropertyRow.value_int
    if isinstance(value, float):
        return SegmentPropertyRow.value_float
    if isinstance(value, datetime):
        return SegmentPropertyRow.value_datetime
    return SegmentPropertyRow.value_str
