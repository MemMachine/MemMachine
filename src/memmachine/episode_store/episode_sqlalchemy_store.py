"""SQLAlchemy implementation of the episode storage layer."""

from collections.abc import Callable
from datetime import UTC
from typing import Any, overload

from pydantic import AwareDatetime, JsonValue, validate_call
from sqlalchemy import (
    JSON,
    DateTime,
    Delete,
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
from sqlalchemy.sql.elements import ColumnElement

from memmachine.episode_store.episode_model import (
    Episode as EpisodeE,
)
from memmachine.episode_store.episode_model import EpisodeEntry, EpisodeType
from memmachine.episode_store.episode_storage import EpisodeIdT, EpisodeStorage


class BaseEpisodeStore(DeclarativeBase):
    """Base class for SQLAlchemy Episode store."""


JSON_AUTO = JSON().with_variant(JSONB, "postgresql")


class Episode(BaseEpisodeStore):
    """SQLAlchemy mapping for stored conversation messages."""

    __tablename__ = "episodestore"
    id = mapped_column(Integer, primary_key=True)

    content = mapped_column(String, nullable=False)

    session_key = mapped_column(String, nullable=False)
    producer_id = mapped_column(String, nullable=False)
    producer_role = mapped_column(String, nullable=False)

    produced_for_id = mapped_column(String, nullable=True)
    episode_type = mapped_column(
        SAEnum(EpisodeType, name="episode_type"),
        default=EpisodeType.MESSAGE,
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

    def to_typed_model(self) -> EpisodeE:
        created_at = (
            self.created_at.replace(tzinfo=UTC)
            if self.created_at.tzinfo is None
            else self.created_at
        )
        return EpisodeE(
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

    async def startup(self) -> None:
        async with self._engine.begin() as conn:
            await conn.run_sync(BaseEpisodeStore.metadata.create_all)

    @validate_call
    async def add_episode(
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
            insert(Episode)
            .values(
                content=content,
                session_key=session_key,
                producer_id=producer_id,
                producer_role=producer_role,
            )
            .returning(Episode.id)
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
            episode_id = result.scalar_one()

        return EpisodeIdT(episode_id)

    @validate_call
    async def add_episodes(
        self,
        session_key: str,
        episodes: list[EpisodeEntry],
    ) -> list[EpisodeE]:
        if not episodes:
            return []

        values_to_insert: list[dict[str, Any]] = []
        for entry in episodes:
            entry_values: dict[str, Any] = {
                "content": entry.content,
                "session_key": session_key,
                "producer_id": entry.producer_id,
                "producer_role": entry.producer_role,
            }

            if entry.produced_for_id is not None:
                entry_values["produced_for_id"] = entry.produced_for_id

            if entry.episode_type is not None:
                entry_values["episode_type"] = entry.episode_type

            if entry.metadata is not None:
                entry_values["json_metadata"] = entry.metadata

            if entry.created_at is not None:
                entry_values["created_at"] = entry.created_at

            values_to_insert.append(entry_values)

        insert_stmt = insert(Episode).returning(Episode.id)

        async with self._create_session() as session:
            result = await session.execute(insert_stmt, values_to_insert)
            inserted_ids = result.scalars().all()
            await session.commit()

        int_episode_ids = [int(episode_id) for episode_id in inserted_ids]
        if not int_episode_ids:
            return []

        select_stmt = (
            select(Episode)
            .where(Episode.id.in_(int_episode_ids))
            .order_by(Episode.id.asc())
        )

        async with self._create_session() as session:
            result = await session.execute(select_stmt)
            persisted_episodes = result.scalars().all()

        episodes_by_id = {
            episode_row.id: episode_row.to_typed_model()
            for episode_row in persisted_episodes
        }
        return [episodes_by_id[episode_id] for episode_id in int_episode_ids]

    @validate_call
    async def get_episode(self, episode_id: EpisodeIdT) -> EpisodeE | None:
        stmt = (
            select(Episode)
            .where(Episode.id == int(episode_id))
            .order_by(Episode.created_at.asc())
        )

        async with self._create_session() as session:
            result = await session.execute(stmt)
            episode = result.scalar_one_or_none()

        return episode.to_typed_model() if episode else None

    def _build_episode_filters(
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
    ) -> list[ColumnElement[bool]]:
        filters: list[ColumnElement[bool]] = []

        filter_definitions: list[
            tuple[list[str] | None, Callable[[list[str]], ColumnElement[bool]]]
        ] = [
            (session_keys, Episode.session_key.in_),
            (producer_ids, Episode.producer_id.in_),
            (producer_roles, Episode.producer_role.in_),
            (produced_for_ids, Episode.produced_for_id.in_),
        ]

        for values, builder in filter_definitions:
            if values is not None:
                filters.append(builder(values))

        if episode_types is not None:
            filters.append(Episode.episode_type.in_(episode_types))
        if start_time is not None:
            filters.append(Episode.created_at >= start_time)
        if end_time is not None:
            filters.append(Episode.created_at <= end_time)

        metadata_filter = self._build_metadata_filter(metadata)
        if metadata_filter is not None:
            filters.append(metadata_filter)

        return filters

    @overload
    def _apply_episode_filter(
        self,
        stmt: Select[Any],
        *,
        session_keys: list[str] | None = None,
        producer_ids: list[str] | None = None,
        producer_roles: list[str] | None = None,
        produced_for_ids: list[str] | None = None,
        episode_types: list[EpisodeType] | None = None,
        start_time: AwareDatetime | None = None,
        end_time: AwareDatetime | None = None,
        metadata: dict[str, JsonValue] | None = None,
    ) -> Select[Any]: ...

    @overload
    def _apply_episode_filter(
        self,
        stmt: Delete,
        *,
        session_keys: list[str] | None = None,
        producer_ids: list[str] | None = None,
        producer_roles: list[str] | None = None,
        produced_for_ids: list[str] | None = None,
        episode_types: list[EpisodeType] | None = None,
        start_time: AwareDatetime | None = None,
        end_time: AwareDatetime | None = None,
        metadata: dict[str, JsonValue] | None = None,
    ) -> Delete: ...

    def _apply_episode_filter(
        self,
        stmt: Select[Any] | Delete,
        *,
        session_keys: list[str] | None = None,
        producer_ids: list[str] | None = None,
        producer_roles: list[str] | None = None,
        produced_for_ids: list[str] | None = None,
        episode_types: list[EpisodeType] | None = None,
        start_time: AwareDatetime | None = None,
        end_time: AwareDatetime | None = None,
        metadata: dict[str, JsonValue] | None = None,
    ) -> Select[Any] | Delete:
        filters = self._build_episode_filters(
            session_keys=session_keys,
            producer_ids=producer_ids,
            producer_roles=producer_roles,
            produced_for_ids=produced_for_ids,
            episode_types=episode_types,
            start_time=start_time,
            end_time=end_time,
            metadata=metadata,
        )

        if filters:
            if isinstance(stmt, Select):
                return stmt.where(*filters)
            if isinstance(stmt, Delete):
                return stmt.where(*filters)
            raise TypeError(f"Unsupported statement type: {type(stmt)}")

        return stmt

    def _build_metadata_filter(
        self, metadata: dict[str, JsonValue] | None
    ) -> ColumnElement[bool] | None:
        if metadata is None:
            return None

        metadata_str = {key: str(metadata[key]) for key in metadata}
        if self._engine.dialect.name == "postgresql":
            return cast(Episode.json_metadata, JSONB).contains(metadata)
        return Episode.json_metadata.contains(metadata_str)

    @validate_call
    async def get_episode_messages(
        self,
        *,
        limit: int | None = None,
        session_keys: list[str] | None = None,
        producer_ids: list[str] | None = None,
        producer_roles: list[str] | None = None,
        produced_for_ids: list[str] | None = None,
        episode_types: list[EpisodeType] | None = None,
        start_time: AwareDatetime | None = None,
        end_time: AwareDatetime | None = None,
        metadata: dict[str, JsonValue] | None = None,
    ) -> list[EpisodeE]:
        stmt = select(Episode)

        stmt = self._apply_episode_filter(
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

        if limit is not None:
            stmt = stmt.limit(limit)

        async with self._create_session() as session:
            result = await session.execute(stmt)
            episode_messages = result.scalars().all()

        return [h.to_typed_model() for h in episode_messages]

    @validate_call
    async def get_episode_messages_count(
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
    ) -> int:
        stmt = select(func.count(Episode.id))

        stmt = self._apply_episode_filter(
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
            n_messages = result.scalar_one()

        return int(n_messages)

    @validate_call
    async def delete_episodes(self, episode_ids: list[EpisodeIdT]) -> None:
        int_episode_ids = [int(h_id) for h_id in episode_ids]

        stmt = delete(Episode).where(Episode.id.in_(int_episode_ids))

        async with self._create_session() as session:
            await session.execute(stmt)
            await session.commit()

    @validate_call
    async def delete_episode_messages(
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
        stmt = delete(Episode)

        stmt = self._apply_episode_filter(
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
