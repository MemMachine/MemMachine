"""SQLAlchemy implementation of the SegmentLinker interface."""

import logging
from collections import defaultdict
from collections.abc import Iterable, Mapping
from datetime import datetime
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
from sqlalchemy.orm import DeclarativeBase, InstrumentedAttribute, mapped_column
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


# ORM models


class BaseSegmentLinker(DeclarativeBase):
    """Base class for segment linker tables."""


class SegmentRow(BaseSegmentLinker):
    """Persisted segment."""

    __tablename__ = "segment_linker_segments"

    session_key = mapped_column(String, nullable=False)

    uuid = mapped_column(Uuid, primary_key=True)
    episode_uuid = mapped_column(Uuid, nullable=False)
    block = mapped_column(Integer, nullable=False)
    index = mapped_column(Integer, nullable=False)
    timestamp = mapped_column(DateTime(timezone=True), nullable=False)
    content = mapped_column(_JSON_AUTO, nullable=False)

    __table_args__ = (
        Index(
            "segment_linker_segments__sk_ep",
            "session_key",
            "episode_uuid",
        ),
        Index(
            "segment_linker_segments__sk_ts_ep_bk_ix",
            "session_key",
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
            "segment_linker_segment_properties__key_str",
            "key",
            "value_str",
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
            "segment_linker_segment_properties__key_bool",
            "key",
            "value_bool",
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

    segment_uuid = mapped_column(
        Uuid,
        ForeignKey("segment_linker_segments.uuid"),
        primary_key=True,
    )
    derivative_uuid = mapped_column(
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
    state = mapped_column(String(1), nullable=False)  # 'A' or 'P'
    refcount = mapped_column(Integer, nullable=False, default=0)

    __table_args__ = (
        Index("segment_linker_derivatives__state_refcount", "state", "refcount"),
    )


# Helpers


def _property_value_column(value: PropertyValue) -> str:
    """Return the column name for a given property value type."""
    if isinstance(value, bool):
        return "value_bool"
    if isinstance(value, int):
        return "value_int"
    if isinstance(value, float):
        return "value_float"
    if isinstance(value, datetime):
        return "value_datetime"
    return "value_str"


def _property_row(segment_uuid: UUID, key: str, value: PropertyValue) -> dict:
    col = _property_value_column(value)
    return {"segment_uuid": segment_uuid, "key": key, col: value}


def _coalesce_property_value(row: SegmentPropertyRow) -> PropertyValue:
    """Read back whichever value_* column is non-null."""
    if row.value_bool is not None:
        return row.value_bool
    if row.value_int is not None:
        return row.value_int
    if row.value_float is not None:
        return row.value_float
    if row.value_datetime is not None:
        return row.value_datetime
    if row.value_str is not None:
        return row.value_str
    raise ValueError(f"Property row for key={row.key!r} has no value set")


def _row_to_segment(
    row: SegmentRow, properties: dict[str, PropertyValue] | None
) -> Segment:
    content = _ContentAdapter.validate_python(row.content)
    return Segment(
        uuid=row.uuid,
        episode_uuid=row.episode_uuid,
        block=row.block,
        index=row.index,
        timestamp=row.timestamp,
        content=content,
        properties=properties or None,
    )


def _segment_to_values(session_key: str, segment: Segment) -> dict:
    return {
        "uuid": segment.uuid,
        "session_key": session_key,
        "episode_uuid": segment.episode_uuid,
        "block": segment.block,
        "index": segment.index,
        "timestamp": segment.timestamp,
        "content": segment.content.model_dump(),
    }


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
            _compile_property_filter(expr.left),
            _compile_property_filter(expr.right),
        )

    if isinstance(expr, Or):
        return or_(
            _compile_property_filter(expr.left),
            _compile_property_filter(expr.right),
        )

    if isinstance(expr, Not):
        return ~_compile_property_filter(expr.expr)

    raise TypeError(f"Unsupported filter expression type: {type(expr)!r}")


def _validate_derivatives(
    active_set: set[UUID] | None,
    link_count: dict[UUID, int],
    locked_map: dict[UUID, DerivativeRow],
) -> None:
    """Validate that active derivatives exist and no linked derivatives are purging."""
    if active_set is not None:
        not_active: list[UUID] = []
        for u in active_set:
            row = locked_map.get(u)
            if row is None or row.state == "P":
                not_active.append(u)
        if not_active:
            raise DerivativeNotActiveError(not_active)

    for u in link_count:
        row = locked_map.get(u)
        if row is not None and row.state == "P":
            raise DerivativeNotActiveError([u])


# Params


class SQLAlchemySegmentLinkerParams(BaseModel):
    """Parameters for constructing a SQLAlchemySegmentLinker."""

    engine: InstanceOf[AsyncEngine] = Field(..., description="Async SQLAlchemy engine")


# Implementation


class SQLAlchemySegmentLinker(SegmentLinker):
    """SQLAlchemy-backed SegmentLinker."""

    def __init__(self, params: SQLAlchemySegmentLinkerParams) -> None:
        """Initialize with an async SQLAlchemy engine."""
        self._engine = params.engine
        self._session_factory = async_sessionmaker(self._engine, expire_on_commit=False)

    def _create_session(self) -> AsyncSession:
        return self._session_factory()

    # -- lifecycle -----------------------------------------------------------

    async def startup(self) -> None:
        async with self._engine.begin() as conn:
            await conn.run_sync(BaseSegmentLinker.metadata.create_all)

    async def shutdown(self) -> None:
        pass

    # -- register ------------------------------------------------------------

    async def register_segments(
        self,
        session_key: str,
        links: Mapping[Segment, Iterable[UUID]],
        *,
        active: Iterable[UUID] | None = None,
    ) -> None:
        # Materialise inputs
        link_map: dict[Segment, list[UUID]] = {
            seg: list(derivs) for seg, derivs in links.items()
        }
        active_set = set(active) if active is not None else None

        # Collect all unique derivative UUIDs and per-derivative link counts
        link_count: dict[UUID, int] = defaultdict(int)
        for derivs in link_map.values():
            for d in derivs:
                link_count[d] += 1

        all_deriv_uuids = set(link_count.keys())
        if active_set is not None:
            all_deriv_uuids |= active_set

        if not link_map:
            return

        sorted_deriv_uuids = sorted(all_deriv_uuids)

        async with self._create_session() as session, session.begin():
            locked_rows = await self._lock_derivatives(session, sorted_deriv_uuids)
            locked_map: dict[UUID, DerivativeRow] = {r.uuid: r for r in locked_rows}

            _validate_derivatives(active_set, link_count, locked_map)

            await self._insert_registration_data(
                session, session_key, link_map, link_count, all_deriv_uuids, locked_map
            )

    # -- query ---------------------------------------------------------------

    async def get_segments_by_derivatives(
        self,
        session_key: str,
        derivative_uuids: Iterable[UUID],
        *,
        limit_per_derivative: int | None = None,
        property_filter: FilterExpr | None = None,
    ) -> Mapping[UUID, Iterable[Segment]]:
        deriv_list = list(derivative_uuids)
        if not deriv_list:
            return {}

        stmt = (
            select(SegmentRow, LinkRow.derivative_uuid)
            .join(LinkRow, LinkRow.segment_uuid == SegmentRow.uuid)
            .where(
                SegmentRow.session_key == session_key,
                LinkRow.derivative_uuid.in_(deriv_list),
            )
            .order_by(
                SegmentRow.timestamp,
                SegmentRow.episode_uuid,
                SegmentRow.block,
                SegmentRow.index,
            )
        )

        if property_filter is not None:
            stmt = stmt.where(_compile_property_filter(property_filter))

        async with self._create_session() as session:
            result = await session.execute(stmt)
            rows = result.all()

            # Collect segment UUIDs to load properties
            seg_uuids = list({r[0].uuid for r in rows})
            props_map = await self._load_properties(session, seg_uuids)

        # Group by derivative
        grouped: dict[UUID, list[Segment]] = {u: [] for u in deriv_list}
        for seg_row, deriv_uuid in rows:
            if deriv_uuid in grouped:
                segment = _row_to_segment(seg_row, props_map.get(seg_row.uuid))
                grouped[deriv_uuid].append(segment)

        # Deduplicate within each derivative list (a segment may appear multiple times
        # if it joined on different link rows, but that shouldn't happen with PK)
        # Apply limit
        if limit_per_derivative is not None:
            for key in grouped:
                grouped[key] = grouped[key][:limit_per_derivative]

        return grouped

    async def get_segment_contexts(
        self,
        session_key: str,
        seed_segment_uuids: Iterable[UUID],
        *,
        max_backward_segments: int = 0,
        max_forward_segments: int = 0,
        property_filter: FilterExpr | None = None,
    ) -> Mapping[UUID, Iterable[Segment]]:
        seed_list = list(seed_segment_uuids)
        if not seed_list:
            return {}

        async with self._create_session() as session:
            # Step 1: Fetch seed rows
            seed_result = await session.execute(
                select(SegmentRow).where(
                    SegmentRow.uuid.in_(seed_list),
                    SegmentRow.session_key == session_key,
                )
            )
            seed_rows: dict[UUID, SegmentRow] = {
                r.uuid: r for r in seed_result.scalars().all()
            }
            if not seed_rows:
                return {}

            # Short-circuit: no context needed
            if max_backward_segments == 0 and max_forward_segments == 0:
                props = await self._load_properties(session, list(seed_rows.keys()))
                return {
                    u: [_row_to_segment(r, props.get(u))] for u, r in seed_rows.items()
                }

            # Step 2: Fetch context rows
            bind = session.bind
            use_lateral = bind is not None and bind.dialect.name != "sqlite"

            if use_lateral:
                ctx = await self._get_contexts_lateral(
                    session,
                    session_key,
                    seed_rows,
                    max_backward_segments,
                    max_forward_segments,
                    property_filter,
                )
            else:
                ctx = await self._get_contexts_loop(
                    session,
                    session_key,
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

            props = await self._load_properties(session, list(all_uuids))

            # Step 4: Assemble results
            result: dict[UUID, list[Segment]] = {}
            for seed_uuid, seed_row in seed_rows.items():
                bw, fw = ctx.get(seed_uuid, ([], []))
                segments = (
                    [_row_to_segment(r, props.get(r.uuid)) for r in reversed(bw)]
                    + [_row_to_segment(seed_row, props.get(seed_uuid))]
                    + [_row_to_segment(r, props.get(r.uuid)) for r in fw]
                )
                result[seed_uuid] = segments

            return result

    # -- delete --------------------------------------------------------------

    async def delete_segments_by_episodes(
        self,
        session_key: str,
        episode_uuids: Iterable[UUID],
    ) -> None:
        ep_list = list(episode_uuids)
        if not ep_list:
            return

        async with self._create_session() as session, session.begin():
            # Find affected segment UUIDs
            seg_result = await session.execute(
                select(SegmentRow.uuid).where(
                    SegmentRow.session_key == session_key,
                    SegmentRow.episode_uuid.in_(ep_list),
                )
            )
            seg_uuids = [r[0] for r in seg_result.all()]
            if not seg_uuids:
                return

            await self._delete_segments_and_update_refcounts(session, seg_uuids)

            # Delete segments (cascades to properties via FK)
            await session.execute(
                delete(SegmentRow).where(SegmentRow.uuid.in_(seg_uuids))
            )

    async def delete_all_segments(self, session_key: str) -> None:
        async with self._create_session() as session, session.begin():
            # Find affected segment UUIDs
            seg_result = await session.execute(
                select(SegmentRow.uuid).where(
                    SegmentRow.session_key == session_key,
                )
            )
            seg_uuids = [r[0] for r in seg_result.all()]
            if not seg_uuids:
                return

            await self._delete_segments_and_update_refcounts(session, seg_uuids)

            # Delete segments
            await session.execute(
                delete(SegmentRow).where(SegmentRow.uuid.in_(seg_uuids))
            )

    # -- orphan management ---------------------------------------------------

    async def get_orphaned_derivatives(self, limit: int = 1000) -> Iterable[UUID]:
        stmt = (
            select(DerivativeRow.uuid)
            .where(DerivativeRow.state == "A", DerivativeRow.refcount == 0)
            .limit(limit)
        )
        async with self._create_session() as session:
            result = await session.execute(stmt)
            return [r[0] for r in result.all()]

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
                    DerivativeRow.state == "A",
                    DerivativeRow.refcount == 0,
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
                    .values(state="P")
                )

            return confirmed

    async def purge_derivatives(self, derivative_uuids: Iterable[UUID]) -> None:
        uuid_list = list(derivative_uuids)
        if not uuid_list:
            return

        async with self._create_session() as session, session.begin():
            await session.execute(
                delete(DerivativeRow).where(
                    DerivativeRow.uuid.in_(uuid_list),
                    DerivativeRow.state == "P",
                )
            )

    # -- context helpers -----------------------------------------------------

    async def _get_contexts_lateral(
        self,
        session: AsyncSession,
        session_key: str,
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
                .where(SegmentRow.session_key == session_key, cmp)
                .order_by(*ordering)
                .limit(max_count)
                .correlate(seeds_sq)
            )
            if property_filter is not None:
                inner = inner.where(_compile_property_filter(property_filter))

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
                    session_key=session_key,
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
        session_key: str,
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
                        SegmentRow.session_key == session_key,
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
                    bw_stmt = bw_stmt.where(_compile_property_filter(property_filter))
                bw_result = await session.execute(bw_stmt)
                backward = list(bw_result.scalars().all())

            forward: list[SegmentRow] = []
            if max_forward > 0:
                fw_stmt = (
                    select(SegmentRow)
                    .where(
                        SegmentRow.session_key == session_key,
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
                    fw_stmt = fw_stmt.where(_compile_property_filter(property_filter))
                fw_result = await session.execute(fw_stmt)
                forward = list(fw_result.scalars().all())

            result[seed_uuid] = (backward, forward)

        return result

    # -- internal helpers ----------------------------------------------------

    async def _lock_derivatives(
        self, session: AsyncSession, sorted_uuids: list[UUID]
    ) -> list[DerivativeRow]:
        """Lock derivative rows in sorted UUID order. Falls back to plain SELECT on SQLite."""
        if not sorted_uuids:
            return []

        stmt = (
            select(DerivativeRow)
            .where(DerivativeRow.uuid.in_(sorted_uuids))
            .order_by(DerivativeRow.uuid)
        )

        # SQLite doesn't support FOR UPDATE
        bind = session.bind
        dialect = bind.dialect.name if bind is not None else ""
        if dialect != "sqlite":
            stmt = stmt.with_for_update()

        result = await session.execute(stmt)
        return list(result.scalars().all())

    async def _delete_segments_and_update_refcounts(
        self, session: AsyncSession, seg_uuids: list[UUID]
    ) -> None:
        """Find affected derivatives, lock them, decrement refcounts, and delete links."""
        # Find affected derivative UUIDs
        deriv_result = await session.execute(
            select(LinkRow.derivative_uuid)
            .where(LinkRow.segment_uuid.in_(seg_uuids))
            .distinct()
        )
        deriv_uuids = [r[0] for r in deriv_result.all()]
        if not deriv_uuids:
            # No links — just delete properties
            await session.execute(
                delete(SegmentPropertyRow).where(
                    SegmentPropertyRow.segment_uuid.in_(seg_uuids)
                )
            )
            return

        # Lock derivatives in sorted order
        await self._lock_derivatives(session, sorted(deriv_uuids))

        # Count links per derivative that will be removed
        count_result = await session.execute(
            select(LinkRow.derivative_uuid, func.count())
            .where(LinkRow.segment_uuid.in_(seg_uuids))
            .group_by(LinkRow.derivative_uuid)
        )
        counts = {r[0]: r[1] for r in count_result.all()}

        # Delete links
        await session.execute(
            delete(LinkRow).where(LinkRow.segment_uuid.in_(seg_uuids))
        )

        # Delete properties
        await session.execute(
            delete(SegmentPropertyRow).where(
                SegmentPropertyRow.segment_uuid.in_(seg_uuids)
            )
        )

        # Decrement refcounts
        for deriv_uuid, count in counts.items():
            await session.execute(
                update(DerivativeRow)
                .where(DerivativeRow.uuid == deriv_uuid)
                .values(refcount=DerivativeRow.refcount - count)
            )

    async def _load_properties(
        self, session: AsyncSession, seg_uuids: list[UUID]
    ) -> dict[UUID, dict[str, PropertyValue]]:
        """Load properties for a batch of segment UUIDs."""
        if not seg_uuids:
            return {}

        result = await session.execute(
            select(SegmentPropertyRow).where(
                SegmentPropertyRow.segment_uuid.in_(seg_uuids)
            )
        )
        props_map: dict[UUID, dict[str, PropertyValue]] = defaultdict(dict)
        for row in result.scalars().all():
            props_map[row.segment_uuid][row.key] = _coalesce_property_value(row)
        return dict(props_map)

    async def _insert_registration_data(
        self,
        session: AsyncSession,
        session_key: str,
        link_map: dict[Segment, list[UUID]],
        link_count: dict[UUID, int],
        all_deriv_uuids: set[UUID],
        locked_map: dict[UUID, DerivativeRow],
    ) -> None:
        """Insert derivative, segment, property, and link rows, then update refcounts."""
        # Insert new derivative rows
        new_derivs = [
            {"uuid": u, "state": "A", "refcount": 0}
            for u in all_deriv_uuids
            if u not in locked_map
        ]
        if new_derivs:
            await session.execute(insert(DerivativeRow), new_derivs)

        # Insert segments
        seg_values = [_segment_to_values(session_key, seg) for seg in link_map]
        if seg_values:
            await session.execute(insert(SegmentRow), seg_values)

        # Insert properties
        prop_rows: list[dict[str, PropertyValue | UUID]] = []
        for seg in link_map:
            if seg.properties:
                for k, v in seg.properties.items():
                    prop_rows.append(_property_row(seg.uuid, k, v))
        if prop_rows:
            await session.execute(insert(SegmentPropertyRow), prop_rows)

        # Insert links
        link_rows = [
            {
                "segment_uuid": seg.uuid,
                "derivative_uuid": d,
            }
            for seg, derivs in link_map.items()
            for d in derivs
        ]
        if link_rows:
            await session.execute(insert(LinkRow), link_rows)

        # Update refcounts
        for u, count in link_count.items():
            await session.execute(
                update(DerivativeRow)
                .where(DerivativeRow.uuid == u)
                .values(refcount=DerivativeRow.refcount + count)
            )
