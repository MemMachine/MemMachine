"""SQLAlchemy-backed implementation of cluster state storage."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, datetime

from sqlalchemy import DateTime, Integer, String, delete, select
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column
from sqlalchemy.sql.sqltypes import JSON

from memmachine.semantic_memory.cluster_manager import ClusterInfo, ClusterState
from memmachine.semantic_memory.cluster_store.cluster_store import ClusterStateStorage
from memmachine.semantic_memory.semantic_model import SetIdT


class BaseClusterStore(DeclarativeBase):
    """Declarative base for cluster state storage."""


class ClusterStateRow(BaseClusterStore):
    """Per-set cluster state metadata."""

    __tablename__ = "semantic_cluster_state"

    set_id: Mapped[str] = mapped_column(String, primary_key=True, nullable=False)
    next_cluster_id: Mapped[int] = mapped_column(Integer, nullable=False, default=0)


class ClusterRow(BaseClusterStore):
    """Cluster centroid and metadata rows."""

    __tablename__ = "semantic_cluster_entry"

    set_id: Mapped[str] = mapped_column(String, primary_key=True, nullable=False)
    cluster_id: Mapped[str] = mapped_column(String, primary_key=True, nullable=False)
    centroid: Mapped[Sequence[float]] = mapped_column(JSON, nullable=False)
    count: Mapped[int] = mapped_column(Integer, nullable=False)
    last_ts: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)


class ClusterEventRow(BaseClusterStore):
    """Mapping from event id to cluster id."""

    __tablename__ = "semantic_cluster_event"

    set_id: Mapped[str] = mapped_column(String, primary_key=True, nullable=False)
    event_id: Mapped[str] = mapped_column(String, primary_key=True, nullable=False)
    cluster_id: Mapped[str] = mapped_column(String, nullable=False)


class ClusterStateStorageSqlAlchemy(ClusterStateStorage):
    """Cluster state storage that persists data via SQLAlchemy."""

    def __init__(self, sqlalchemy_engine: AsyncEngine) -> None:
        """Store the SQLAlchemy engine and session factory."""
        self._engine = sqlalchemy_engine
        self._session_factory = async_sessionmaker(
            bind=self._engine,
            expire_on_commit=False,
        )

    def _create_session(self) -> AsyncSession:
        return self._session_factory()

    async def startup(self) -> None:
        async with self._engine.begin() as conn:
            await conn.run_sync(BaseClusterStore.metadata.create_all)

    async def delete_all(self) -> None:
        async with self._create_session() as session:
            await session.execute(delete(ClusterEventRow))
            await session.execute(delete(ClusterRow))
            await session.execute(delete(ClusterStateRow))
            await session.commit()

    async def get_state(self, *, set_id: SetIdT) -> ClusterState | None:
        set_key = str(set_id)
        async with self._create_session() as session:
            state_res = await session.execute(
                select(ClusterStateRow).where(ClusterStateRow.set_id == set_key)
            )
            state_row = state_res.scalar_one_or_none()

            cluster_res = await session.execute(
                select(ClusterRow).where(ClusterRow.set_id == set_key)
            )
            cluster_rows = cluster_res.scalars().all()

            event_res = await session.execute(
                select(ClusterEventRow).where(ClusterEventRow.set_id == set_key)
            )
            event_rows = event_res.scalars().all()

        if state_row is None and not cluster_rows and not event_rows:
            return None

        clusters = {
            row.cluster_id: ClusterInfo(
                centroid=list(row.centroid),
                count=row.count,
                last_ts=self._normalize_timestamp(row.last_ts),
            )
            for row in cluster_rows
        }
        event_to_cluster = {row.event_id: row.cluster_id for row in event_rows}
        next_cluster_id = state_row.next_cluster_id if state_row is not None else 0

        return ClusterState(
            clusters=clusters,
            event_to_cluster=event_to_cluster,
            next_cluster_id=next_cluster_id,
        )

    async def save_state(self, *, set_id: SetIdT, state: ClusterState) -> None:
        set_key = str(set_id)
        async with self._create_session() as session:
            await session.execute(
                delete(ClusterEventRow).where(ClusterEventRow.set_id == set_key)
            )
            await session.execute(
                delete(ClusterRow).where(ClusterRow.set_id == set_key)
            )
            await session.execute(
                delete(ClusterStateRow).where(ClusterStateRow.set_id == set_key)
            )

            session.add(
                ClusterStateRow(
                    set_id=set_key,
                    next_cluster_id=state.next_cluster_id,
                )
            )

            if state.clusters:
                session.add_all(
                    [
                        ClusterRow(
                            set_id=set_key,
                            cluster_id=cluster_id,
                            centroid=list(info.centroid),
                            count=info.count,
                            last_ts=info.last_ts,
                        )
                        for cluster_id, info in state.clusters.items()
                    ]
                )

            if state.event_to_cluster:
                session.add_all(
                    [
                        ClusterEventRow(
                            set_id=set_key,
                            event_id=event_id,
                            cluster_id=cluster_id,
                        )
                        for event_id, cluster_id in state.event_to_cluster.items()
                    ]
                )

            await session.commit()

    async def delete_state(self, *, set_id: SetIdT) -> None:
        set_key = str(set_id)
        async with self._create_session() as session:
            await session.execute(
                delete(ClusterEventRow).where(ClusterEventRow.set_id == set_key)
            )
            await session.execute(
                delete(ClusterRow).where(ClusterRow.set_id == set_key)
            )
            await session.execute(
                delete(ClusterStateRow).where(ClusterStateRow.set_id == set_key)
            )
            await session.commit()

    @staticmethod
    def _normalize_timestamp(value: datetime) -> datetime:
        if value.tzinfo is None:
            return value.replace(tzinfo=UTC)
        return value
