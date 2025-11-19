"""SQLAlchemy implementation of the episode storage layer."""

from datetime import UTC

from pydantic import AwareDatetime, JsonValue, validate_call
from sqlalchemy import (
    JSON,
    DateTime,
    Index,
    Integer,
    String,
    cast,
    delete,
    func,
    insert,
    select,
)
from sqlalchemy import (
    Enum as SAEnum,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import DeclarativeBase, mapped_column
from sqlalchemy.sql import Select

from memmachine.episode_store.episode_model import Episode, EpisodeType
from memmachine.episode_store.episode_storage import EpisodeIdT, EpisodeStorage


class BaseHistoryStore(DeclarativeBase):
    """Base class for SQLAlchemy history store."""


JSON_AUTO = JSON().with_variant(JSONB, "postgresql")


class History(BaseHistoryStore):
    """SQLAlchemy mapping for stored conversation messages."""

    __tablename__ = "history"
    id = mapped_column(Integer, primary_key=True)

    content = mapped_column(String, nullable=False)

    session_key = mapped_column(String, nullable=False)
    producer_id = mapped_column(String, nullable=False)
    producer_role = mapped_column(String, nullable=False)

    produced_for_id = mapped_column(String, nullable=True)
    episode_type = mapped_column(
        SAEnum(EpisodeType, name="episode_type"),
        nullable=True,
    )

    json_metadata = mapped_column(
        JSON_AUTO,
        name="metadata",
        default=dict,
        nullable=False,
    )
    created_at = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )

    __table_args__ = (
        Index("idx_session_key", "session_key"),
        Index("idx_producer_id", "producer_id"),
        Index("idx_producer_role", "producer_role"),
        Index("idx_session_key_producer_id", "session_key", "producer_id"),
        Index(
            "idx_session_key_producer_id_producer_role_produced_for_id",
            "session_key",
            "producer_id",
            "producer_role",
            "produced_for_id",
        ),
    )

    def to_typed_model(self) -> Episode:
        created_at = (
            self.created_at.replace(tzinfo=UTC)
            if self.created_at.tzinfo is None
            else self.created_at
        )
        return Episode(
            uid=EpisodeIdT(self.id),
            content=self.content,
            session_key=self.session_key,
            producer_id=self.producer_id,
            producer_role=self.producer_role,
            produced_for_id=self.produced_for_id,
            episode_type=self.episode_type,
            created_at=created_at,
            metadata=self.json_metadata or None,
        )


class SqlAlchemyEpisodeStore(EpisodeStorage):
    """SQLAlchemy episode store implementation."""

    def __init__(self, engine: AsyncEngine) -> None:
        """Initialize the store with an async SQLAlchemy engine."""
        self._engine: AsyncEngine = engine
        self._session_factory = async_sessionmaker(
            self._engine,
            expire_on_commit=False,
        )

    def _create_session(self) -> AsyncSession:
        return self._session_factory()

    @validate_call
    async def add_history(
        self,
        *,
        content: str,
        session_key: str,
        producer_id: str,
        producer_role: str,
        produced_for_id: str | None = None,
        episode_type: EpisodeType | None = None,
        metadata: dict[str, JsonValue] | None = None,
        created_at: AwareDatetime | None = None,
    ) -> EpisodeIdT:
        stmt = (
            insert(History)
            .values(
                content=content,
                session_key=session_key,
                producer_id=producer_id,
                producer_role=producer_role,
            )
            .returning(History.id)
        )

        if produced_for_id is not None:
            stmt = stmt.values(produced_for_id=produced_for_id)

        if episode_type is not None:
            stmt = stmt.values(episode_type=episode_type)

        if metadata is not None:
            stmt = stmt.values(json_metadata=metadata)

        if created_at is not None:
            stmt = stmt.values(created_at=created_at)

        async with self._create_session() as session:
            result = await session.execute(stmt)
            await session.commit()
            history_id = result.scalar_one()

        return EpisodeIdT(history_id)

    @validate_call
    async def get_history(self, history_id: EpisodeIdT) -> Episode | None:
        stmt = (
            select(History)
            .where(History.id == int(history_id))
            .order_by(History.created_at.asc())
        )

        async with self._create_session() as session:
            result = await session.execute(stmt)
            history = result.scalar_one_or_none()

        return history.to_typed_model() if history else None

    def _apply_history_filter(
        self,
        stmt: Select,
        *,
        session_keys: list[str] | None = None,
        producer_ids: list[str] | None = None,
        producer_roles: list[str] | None = None,
        produced_for_ids: list[str] | None = None,
        episode_types: list[EpisodeType] | None = None,
        start_time: AwareDatetime | None = None,
        end_time: AwareDatetime | None = None,
        metadata: dict[str, str] | None = None,
    ) -> Select:
        if session_keys is not None:
            stmt = stmt.where(History.session_key.in_(session_keys))

        if producer_ids is not None:
            stmt = stmt.where(History.producer_id.in_(producer_ids))

        if producer_roles is not None:
            stmt = stmt.where(History.producer_role.in_(producer_roles))

        if produced_for_ids is not None:
            stmt = stmt.where(History.produced_for_id.in_(produced_for_ids))

        if episode_types is not None:
            stmt = stmt.where(History.episode_type.in_(episode_types))

        if start_time is not None:
            stmt = stmt.where(History.created_at >= start_time)

        if end_time is not None:
            stmt = stmt.where(History.created_at <= end_time)

        if metadata is not None:
            if self._engine.dialect.name == "postgresql":
                stmt = stmt.where(cast(History.json_metadata, JSONB).contains(metadata))
            else:
                stmt = stmt.where(History.json_metadata.contains(metadata))

        return stmt

    @validate_call
    async def get_history_messages(
        self,
        *,
        session_keys: list[str] | None = None,
        producer_ids: list[str] | None = None,
        producer_roles: list[str] | None = None,
        produced_for_ids: list[str] | None = None,
        episode_types: list[EpisodeType] | None = None,
        start_time: AwareDatetime | None = None,
        end_time: AwareDatetime | None = None,
        metadata: dict[str, str] | None = None,
    ) -> list[Episode]:
        stmt = select(History)

        stmt = self._apply_history_filter(
            stmt,
            session_keys=session_keys,
            producer_ids=producer_ids,
            producer_roles=producer_roles,
            produced_for_ids=produced_for_ids,
            episode_types=episode_types,
            start_time=start_time,
            end_time=end_time,
            metadata=metadata,
        )

        async with self._create_session() as session:
            result = await session.execute(stmt)
            history_messages = result.scalars().all()

        return [h.to_typed_model() for h in history_messages]

    @validate_call
    async def delete_history(self, history_ids: list[EpisodeIdT]) -> None:
        int_history_ids = [int(h_id) for h_id in history_ids]

        stmt = delete(History).where(History.id.in_(int_history_ids))

        async with self._create_session() as session:
            await session.execute(stmt)
            await session.commit()

    @validate_call
    async def delete_history_messages(
        self,
        *,
        session_keys: list[str] | None = None,
        producer_ids: list[str] | None = None,
        producer_roles: list[str] | None = None,
        produced_for_ids: list[str] | None = None,
        episode_types: list[EpisodeType] | None = None,
        start_time: AwareDatetime | None = None,
        end_time: AwareDatetime | None = None,
        metadata: dict[str, JsonValue] | None = None,
    ) -> None:
        stmt = delete(History)

        stmt = self._apply_history_filter(
            stmt,
            session_keys=session_keys,
            producer_ids=producer_ids,
            producer_roles=producer_roles,
            produced_for_ids=produced_for_ids,
            episode_types=episode_types,
            start_time=start_time,
            end_time=end_time,
            metadata=metadata,
        )

        async with self._create_session() as session:
            await session.execute(stmt)
            await session.commit()
