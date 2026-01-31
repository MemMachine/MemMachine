from collections.abc import Iterable, Mapping
from datetime import datetime
from typing import Any
from uuid import UUID

from sqlalchemy import (
    JSON,
    DateTime,
    Enum,
    ForeignKey,
    Index,
    Integer,
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

from memmachine.common.filter.filter_parser import FilterExpr
from .data_types import Snapshot, ContentType, FilterablePropertyValue
from .snapshot_store import SnapshotStore


class Base(DeclarativeBase):
    pass


# --- Database Models ---

class SnapshotModel(Base):
    __tablename__ = "snapshots"

    # Composite primary key logic:
    # We include session_key in PK to allow partitioning/sharding by session if needed later.
    session_key: Mapped[str] = mapped_column(String, primary_key=True)
    uuid: Mapped[UUID] = mapped_column(primary_key=True)

    episode_uuid: Mapped[UUID] = mapped_column(index=True)
    index: Mapped[int] = mapped_column(Integer)
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    context: Mapped[str] = mapped_column(String)

    # Store Enum as string in DB for readability/portability
    content_type: Mapped[ContentType] = mapped_column(Enum(ContentType))

    # Flexible JSON storage
    content: Mapped[Any] = mapped_column(JSON)
    filterable_properties: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)

    # Composite index specifically for get_snapshot_contexts ordering
    __table_args__ = (
        Index(
            "ix_snapshots_ordering",
            "session_key",
            "timestamp",
            "episode_uuid",
            "index"
        ),
    )


class SnapshotDerivativeModel(Base):
    __tablename__ = "snapshot_derivatives"

    session_key: Mapped[str] = mapped_column(String, primary_key=True)
    snapshot_uuid: Mapped[UUID] = mapped_column(
        ForeignKey("snapshots.uuid", ondelete="CASCADE"),
        primary_key=True
    )
    derivative_uuid: Mapped[UUID] = mapped_column(primary_key=True)


# --- Store Implementation ---

class SQLAlchemySnapshotStore(SnapshotStore):
    def __init__(self, database_url: str, echo: bool = False):
        self.database_url = database_url
        self.echo = echo
        self._engine: AsyncEngine | None = None
        self._session_maker: async_sessionmaker[AsyncSession] | None = None

    async def startup(self) -> None:
        """Initialize DB engine and create tables."""
        self._engine = create_async_engine(self.database_url, echo=self.echo)
        self._session_maker = async_sessionmaker(self._engine, expire_on_commit=False)

        async with self._engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

    async def shutdown(self) -> None:
        """Dispose of the DB engine."""
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
            filterable_properties=snapshot.filterable_properties,
        )

    def _model_to_snapshot(self, model: SnapshotModel) -> Snapshot:
        # Pydantic validation handles JSON dict -> specific types if necessary
        return Snapshot(
            uuid=model.uuid,
            episode_uuid=model.episode_uuid,
            index=model.index,
            timestamp=model.timestamp,
            context=model.context,
            content_type=model.content_type,
            content=model.content,
            filterable_properties=model.filterable_properties,
        )

    def _apply_filter(self, query, filter_expr: FilterExpr | None):
        """
        Applies property filtering to the query.

        NOTE: You must implement the specific logic to translate FilterExpr
        nodes into SQLAlchemy clauses (e.g. comparing JSON values).
        """
        if filter_expr is None:
            return query

        # Placeholder: This needs to traverse the filter_expr tree.
        # Example: query = query.where(SnapshotModel.filterable_properties['key'].astext == 'value')
        return query

    # --- Interface Methods ---

    async def add_snapshots(
        self,
        session_key: str,
        snapshots: Iterable[Snapshot],
    ) -> None:
        if not self._session_maker:
            raise RuntimeError("Store not started")

        snapshot_models = [self._snapshot_to_model(session_key, s) for s in snapshots]
        if not snapshot_models:
            return

        async with self._session_maker() as session:
            async with session.begin():
                for model in snapshot_models:
                    await session.merge(model)

    async def add_snapshot_derivative_uuids(
        self,
        session_key: str,
        snapshot_derivative_uuids: Mapping[UUID, Iterable[UUID]],
    ) -> None:
        if not self._session_maker:
            raise RuntimeError("Store not started")

        models = []
        for snap_uuid, deriv_uuids in snapshot_derivative_uuids.items():
            for deriv_uuid in deriv_uuids:
                models.append(
                    SnapshotDerivativeModel(
                        session_key=session_key,
                        snapshot_uuid=snap_uuid,
                        derivative_uuid=deriv_uuid,
                    )
                )

        if not models:
            return

        async with self._session_maker() as session:
            async with session.begin():
                for model in models:
                    await session.merge(model)

    async def get_snapshot_derivative_uuids(
        self,
        session_key: str,
        snapshot_uuids: Iterable[UUID],
    ) -> Iterable[Iterable[UUID]]:
        if not self._session_maker:
            raise RuntimeError("Store not started")

        uuids_list = list(snapshot_uuids)
        if not uuids_list:
            return []

        async with self._session_maker() as session:
            stmt = select(
                SnapshotDerivativeModel.snapshot_uuid,
                SnapshotDerivativeModel.derivative_uuid
            ).where(
                SnapshotDerivativeModel.session_key == session_key,
                SnapshotDerivativeModel.snapshot_uuid.in_(uuids_list)
            )
            result = await session.execute(stmt)
            rows = result.all()

        # Group results in memory
        deriv_map: dict[UUID, list[UUID]] = {uid: [] for uid in uuids_list}
        for s_uuid, d_uuid in rows:
            if s_uuid in deriv_map:
                deriv_map[s_uuid].append(d_uuid)

        # Return in the exact order of the input iterable
        return [deriv_map[uid] for uid in uuids_list]

    async def get_episodes_snapshots(
        self,
        session_key: str,
        episode_uuids: Iterable[UUID],
    ) -> Iterable[Snapshot]:
        if not self._session_maker:
            raise RuntimeError("Store not started")

        episode_uuids_list = list(episode_uuids)
        if not episode_uuids_list:
            return []

        async with self._session_maker() as session:
            stmt = select(SnapshotModel).where(
                SnapshotModel.session_key == session_key,
                SnapshotModel.episode_uuid.in_(episode_uuids_list)
            ).order_by(SnapshotModel.index)

            result = await session.execute(stmt)
            models = result.scalars().all()

        return [self._model_to_snapshot(m) for m in models]

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
            # 1. Fetch all seeds first to establish pivots
            stmt_seeds = select(SnapshotModel).where(
                SnapshotModel.session_key == session_key,
                SnapshotModel.uuid.in_(seed_uuids)
            )
            res_seeds = await session.execute(stmt_seeds)
            seeds_map = {s.uuid: s for s in res_seeds.scalars().all()}

            # 2. Iterate input order
            for seed_uuid in seed_uuids:
                seed = seeds_map.get(seed_uuid)
                if not seed:
                    results.append([])
                    continue

                context_items = []

                # --- Backward Context ---
                if max_backward_snapshots > 0:
                    # Logic: (time < t) OR (time=t AND ep < e) OR (time=t AND ep=e AND idx < i)
                    stmt_back = select(SnapshotModel).where(
                        SnapshotModel.session_key == session_key,
                        or_(
                            SnapshotModel.timestamp < seed.timestamp,
                            and_(
                                SnapshotModel.timestamp == seed.timestamp,
                                SnapshotModel.episode_uuid < seed.episode_uuid
                            ),
                            and_(
                                SnapshotModel.timestamp == seed.timestamp,
                                SnapshotModel.episode_uuid == seed.episode_uuid,
                                SnapshotModel.index < seed.index
                            )
                        )
                    )
                    stmt_back = self._apply_filter(stmt_back, property_filter)
                    # Order descending to find the nearest neighbors
                    stmt_back = stmt_back.order_by(
                        desc(SnapshotModel.timestamp),
                        desc(SnapshotModel.episode_uuid),
                        desc(SnapshotModel.index)
                    ).limit(max_backward_snapshots)

                    res_back = await session.execute(stmt_back)
                    # Reverse back to chronological order
                    backward = res_back.scalars().all()[::-1]
                    context_items.extend(backward)

                # --- Seed ---
                context_items.append(seed)

                # --- Forward Context ---
                if max_forward_snapshots > 0:
                    # Logic: (time > t) OR (time=t AND ep > e) OR (time=t AND ep=e AND idx > i)
                    stmt_fwd = select(SnapshotModel).where(
                        SnapshotModel.session_key == session_key,
                        or_(
                            SnapshotModel.timestamp > seed.timestamp,
                            and_(
                                SnapshotModel.timestamp == seed.timestamp,
                                SnapshotModel.episode_uuid > seed.episode_uuid
                            ),
                            and_(
                                SnapshotModel.timestamp == seed.timestamp,
                                SnapshotModel.episode_uuid == seed.episode_uuid,
                                SnapshotModel.index > seed.index
                            )
                        )
                    )
                    stmt_fwd = self._apply_filter(stmt_fwd, property_filter)
                    stmt_fwd = stmt_fwd.order_by(
                        SnapshotModel.timestamp,
                        SnapshotModel.episode_uuid,
                        SnapshotModel.index
                    ).limit(max_forward_snapshots)

                    res_fwd = await session.execute(stmt_fwd)
                    forward = res_fwd.scalars().all()
                    context_items.extend(forward)

                results.append([self._model_to_snapshot(m) for m in context_items])

        return results

    async def delete_snapshots(
        self,
        session_key: str,
        snapshot_uuids: Iterable[UUID],
    ) -> None:
        if not self._session_maker:
            raise RuntimeError("Store not started")

        uuids_list = list(snapshot_uuids)
        if not uuids_list:
            return

        async with self._session_maker() as session:
            async with session.begin():
                # Derivatives will delete automatically via CASCADE if configured in DB,
                # but explicit deletion ensures consistency at the app layer.
                await session.execute(
                    delete(SnapshotDerivativeModel).where(
                        SnapshotDerivativeModel.session_key == session_key,
                        SnapshotDerivativeModel.snapshot_uuid.in_(uuids_list)
                    )
                )

                await session.execute(
                    delete(SnapshotModel).where(
                        SnapshotModel.session_key == session_key,
                        SnapshotModel.uuid.in_(uuids_list)
                    )
                )
