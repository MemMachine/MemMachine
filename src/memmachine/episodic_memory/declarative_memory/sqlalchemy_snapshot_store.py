from collections.abc import Iterable, Mapping
from datetime import datetime
from typing import Any
from uuid import UUID

from pydantic import BaseModel, Field
from sqlalchemy import (
    JSON,
    ColumnElement,
    DateTime,
    Enum,
    ForeignKey,
    Index,
    Integer,
    Select,
    String,
    and_,
    delete,
    desc,
    or_,
    select,
)
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

from memmachine.common.filter.filter_parser import And, Comparison, FilterExpr, Or
from memmachine.common.filter.sql_filter_util import parse_sql_filter

from .data_types import ContentType, Snapshot
from .snapshot_store import SnapshotStore


class Base(DeclarativeBase):
    pass


# --- Database Models ---


class SnapshotModel(Base):
    __tablename__ = "snapshots"

    session_key: Mapped[str] = mapped_column(String, primary_key=True)
    uuid: Mapped[UUID] = mapped_column(primary_key=True)

    episode_uuid: Mapped[UUID] = mapped_column(index=True)
    index: Mapped[int] = mapped_column(Integer)
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    context: Mapped[str] = mapped_column(String)
    content_type: Mapped[ContentType] = mapped_column(Enum(ContentType))
    content: Mapped[Any] = mapped_column(JSON)
    attributes: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)

    __table_args__ = (
        Index(
            "ix_snapshots_ordering",
            "session_key",
            "timestamp",
            "episode_uuid",
            "index",
        ),
    )


class SnapshotDerivativeModel(Base):
    __tablename__ = "snapshot_derivatives"

    session_key: Mapped[str] = mapped_column(String, primary_key=True)
    snapshot_uuid: Mapped[UUID] = mapped_column(
        ForeignKey("snapshots.uuid", ondelete="CASCADE"),
        primary_key=True,
    )
    derivative_uuid: Mapped[UUID] = mapped_column(primary_key=True)


# --- Store Implementation ---


class SQLAlchemySnapshotStoreParams(BaseModel):
    """Parameters for SQLAlchemySnapshotStore."""

    database_url: str = Field(
        ...,
        description="SQLAlchemy async database URL.",
    )
    echo: bool = Field(
        False,
        description="Whether to echo SQL statements for debugging.",
    )


class SQLAlchemySnapshotStore(SnapshotStore):
    def __init__(self, params: SQLAlchemySnapshotStoreParams) -> None:
        self._database_url = params.database_url
        self._echo = params.echo
        self._engine: AsyncEngine | None = None
        self._session_maker: async_sessionmaker[AsyncSession] | None = None

    async def startup(self) -> None:
        self._engine = create_async_engine(self._database_url, echo=self._echo)
        self._session_maker = async_sessionmaker(self._engine, expire_on_commit=False)

        async with self._engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

    async def shutdown(self) -> None:
        if self._engine:
            await self._engine.dispose()
            self._engine = None

    # --- Helpers ---

    def _snapshot_to_model(self, session_key: str, snapshot: Snapshot) -> SnapshotModel:
        return SnapshotModel(
            session_key=session_key,
            uuid=snapshot.uuid,
            episode_uuid=snapshot.episode_uuid,
            index=snapshot.index,
            timestamp=snapshot.timestamp,
            context=snapshot.context,
            content_type=snapshot.content_type,
            content=snapshot.content,
            attributes=snapshot.attributes,
        )

    def _model_to_snapshot(self, model: SnapshotModel) -> Snapshot:
        return Snapshot(
            uuid=model.uuid,
            episode_uuid=model.episode_uuid,
            index=model.index,
            timestamp=model.timestamp,
            context=model.context,
            content_type=model.content_type,
            content=model.content,
            attributes=model.attributes,
        )

    def _build_filter_clause(
        self, filter_expr: FilterExpr
    ) -> ColumnElement[bool] | None:
        """Recursively translate a FilterExpr tree into SQLAlchemy clauses."""
        if isinstance(filter_expr, Comparison):
            column = SnapshotModel.attributes[filter_expr.field].astext
            return parse_sql_filter(column, is_metadata=False, expr=filter_expr)
        if isinstance(filter_expr, And):
            left = self._build_filter_clause(filter_expr.left)
            right = self._build_filter_clause(filter_expr.right)
            parts = [c for c in (left, right) if c is not None]
            if not parts:
                return None
            return and_(*parts)
        if isinstance(filter_expr, Or):
            left = self._build_filter_clause(filter_expr.left)
            right = self._build_filter_clause(filter_expr.right)
            parts = [c for c in (left, right) if c is not None]
            if not parts:
                return None
            return or_(*parts)
        return None

    def _apply_filter(
        self, query: Select[Any], filter_expr: FilterExpr | None
    ) -> Select[Any]:
        if filter_expr is None:
            return query
        clause = self._build_filter_clause(filter_expr)
        if clause is not None:
            query = query.where(clause)
        return query

    # --- Interface Methods ---

    async def add_snapshots(
        self,
        session_key: str,
        snapshots: Mapping[Snapshot, Iterable[UUID]],
    ) -> None:
        if not self._session_maker:
            raise RuntimeError("Store not started")

        if not snapshots:
            return

        async with self._session_maker() as session, session.begin():
            for snapshot, derivative_uuids in snapshots.items():
                await session.merge(self._snapshot_to_model(session_key, snapshot))
                for deriv_uuid in derivative_uuids:
                    await session.merge(
                        SnapshotDerivativeModel(
                            session_key=session_key,
                            snapshot_uuid=snapshot.uuid,
                            derivative_uuid=deriv_uuid,
                        )
                    )

    async def get_snapshot_contexts(
        self,
        session_key: str,
        seed_snapshot_uuids: Iterable[UUID],
        *,
        max_backward_snapshots: int = 0,
        max_forward_snapshots: int = 0,
        property_filter: FilterExpr | None = None,
    ) -> Iterable[Iterable[Snapshot]]:
        if not self._session_maker:
            raise RuntimeError("Store not started")

        seed_uuids = list(seed_snapshot_uuids)
        if not seed_uuids:
            return []

        results = []

        async with self._session_maker() as session:
            stmt_seeds = select(SnapshotModel).where(
                SnapshotModel.session_key == session_key,
                SnapshotModel.uuid.in_(seed_uuids),
            )
            res_seeds = await session.execute(stmt_seeds)
            seeds_map = {s.uuid: s for s in res_seeds.scalars().all()}

            for seed_uuid in seed_uuids:
                seed = seeds_map.get(seed_uuid)
                if not seed:
                    results.append([])
                    continue

                context_items = []

                # --- Backward Context ---
                if max_backward_snapshots > 0:
                    stmt_back = select(SnapshotModel).where(
                        SnapshotModel.session_key == session_key,
                        or_(
                            SnapshotModel.timestamp < seed.timestamp,
                            and_(
                                SnapshotModel.timestamp == seed.timestamp,
                                SnapshotModel.episode_uuid < seed.episode_uuid,
                            ),
                            and_(
                                SnapshotModel.timestamp == seed.timestamp,
                                SnapshotModel.episode_uuid == seed.episode_uuid,
                                SnapshotModel.index < seed.index,
                            ),
                        ),
                    )
                    stmt_back = self._apply_filter(stmt_back, property_filter)
                    stmt_back = stmt_back.order_by(
                        desc(SnapshotModel.timestamp),
                        desc(SnapshotModel.episode_uuid),
                        desc(SnapshotModel.index),
                    ).limit(max_backward_snapshots)

                    res_back = await session.execute(stmt_back)
                    backward = res_back.scalars().all()[::-1]
                    context_items.extend(backward)

                # --- Seed ---
                context_items.append(seed)

                # --- Forward Context ---
                if max_forward_snapshots > 0:
                    stmt_fwd = select(SnapshotModel).where(
                        SnapshotModel.session_key == session_key,
                        or_(
                            SnapshotModel.timestamp > seed.timestamp,
                            and_(
                                SnapshotModel.timestamp == seed.timestamp,
                                SnapshotModel.episode_uuid > seed.episode_uuid,
                            ),
                            and_(
                                SnapshotModel.timestamp == seed.timestamp,
                                SnapshotModel.episode_uuid == seed.episode_uuid,
                                SnapshotModel.index > seed.index,
                            ),
                        ),
                    )
                    stmt_fwd = self._apply_filter(stmt_fwd, property_filter)
                    stmt_fwd = stmt_fwd.order_by(
                        SnapshotModel.timestamp,
                        SnapshotModel.episode_uuid,
                        SnapshotModel.index,
                    ).limit(max_forward_snapshots)

                    res_fwd = await session.execute(stmt_fwd)
                    forward = res_fwd.scalars().all()
                    context_items.extend(forward)

                results.append([self._model_to_snapshot(m) for m in context_items])

        return results

    async def delete_episodes_snapshots(
        self,
        session_key: str,
        episode_uuids: Iterable[UUID],
    ) -> Iterable[UUID]:
        if not self._session_maker:
            raise RuntimeError("Store not started")

        episode_uuids_list = list(episode_uuids)
        if not episode_uuids_list:
            return []

        async with self._session_maker() as session, session.begin():
            # Find snapshot UUIDs for the given episodes
            snapshot_stmt = select(SnapshotModel.uuid).where(
                SnapshotModel.session_key == session_key,
                SnapshotModel.episode_uuid.in_(episode_uuids_list),
            )
            snapshot_result = await session.execute(snapshot_stmt)
            snapshot_uuids = [row[0] for row in snapshot_result.all()]

            if not snapshot_uuids:
                return []

            # Collect embedding record UUIDs before deletion
            deriv_stmt = select(SnapshotDerivativeModel.derivative_uuid).where(
                SnapshotDerivativeModel.session_key == session_key,
                SnapshotDerivativeModel.snapshot_uuid.in_(snapshot_uuids),
            )
            deriv_result = await session.execute(deriv_stmt)
            derivative_uuids = [row[0] for row in deriv_result.all()]

            # Delete derivatives then snapshots
            await session.execute(
                delete(SnapshotDerivativeModel).where(
                    SnapshotDerivativeModel.session_key == session_key,
                    SnapshotDerivativeModel.snapshot_uuid.in_(snapshot_uuids),
                )
            )
            await session.execute(
                delete(SnapshotModel).where(
                    SnapshotModel.session_key == session_key,
                    SnapshotModel.episode_uuid.in_(episode_uuids_list),
                )
            )

            return derivative_uuids
