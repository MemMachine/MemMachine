"""SQLAlchemy implementation of the episode storage layer."""

import socket
from datetime import UTC
from typing import Any, TypeVar, overload

from pydantic import (
    AwareDatetime,
    TypeAdapter,
    ValidationError,
    validate_call,
)
from sqlalchemy import (
    JSON,
    DateTime,
    Delete,
    Index,
    Integer,
    String,
    and_,
    delete,
    func,
    insert,
    or_,
    select,
)
from sqlalchemy import Enum as SAEnum
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.exc import OperationalError
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import DeclarativeBase, mapped_column
from sqlalchemy.sql import Select
from sqlalchemy.sql.elements import ColumnElement

from memmachine.common.episode_store.episode_model import Episode as EpisodeE
from memmachine.common.episode_store.episode_model import EpisodeEntry, EpisodeType
from memmachine.common.episode_store.episode_storage import EpisodeIdT, EpisodeStorage
from memmachine.common.errors import (
    ConfigurationError,
    InvalidArgumentError,
    ResourceNotFoundError,
)
from memmachine.common.filter.filter_parser import (
    And as FilterAnd,
)
from memmachine.common.filter.filter_parser import (
    Comparison as FilterComparison,
)
from memmachine.common.filter.filter_parser import FilterExpr
from memmachine.common.filter.filter_parser import (
    Or as FilterOr,
)
from memmachine.common.filter.sql_filter_util import parse_sql_filter


class BaseEpisodeStore(DeclarativeBase):
    """Base class for SQLAlchemy Episode store."""


JSON_AUTO = JSON().with_variant(JSONB, "postgresql")

T = TypeVar("T")


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
        try:
            async with self._engine.begin() as conn:
                await conn.run_sync(BaseEpisodeStore.metadata.create_all)
        except (OperationalError, socket.gaierror) as err:
            raise ConfigurationError(
                "Failed to connect to the database during startup, please check your configuration."
            ) from err

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
        try:
            int_episode_id = int(episode_id)
        except (TypeError, ValueError) as e:
            raise ResourceNotFoundError("Invalid episode ID") from e

        stmt = (
            select(Episode)
            .where(Episode.id == int_episode_id)
            .order_by(Episode.created_at.asc())
        )

        async with self._create_session() as session:
            result = await session.execute(stmt)
            episode = result.scalar_one_or_none()

        return episode.to_typed_model() if episode else None

    @overload
    def _apply_episode_filter(
        self,
        stmt: Select[Any],
        *,
        filter_expr: FilterExpr | None = None,
        start_time: AwareDatetime | None = None,
        end_time: AwareDatetime | None = None,
    ) -> Select[Any]: ...

    @overload
    def _apply_episode_filter(
        self,
        stmt: Delete,
        *,
        filter_expr: FilterExpr | None = None,
        start_time: AwareDatetime | None = None,
        end_time: AwareDatetime | None = None,
    ) -> Delete: ...

    def _apply_episode_filter(
        self,
        stmt: Select[Any] | Delete,
        *,
        filter_expr: FilterExpr | None = None,
        start_time: AwareDatetime | None = None,
        end_time: AwareDatetime | None = None,
    ) -> Select[Any] | Delete:
        filters: list[ColumnElement[bool]] = []

        if filter_expr is not None:
            parsed_filter = self._compile_episode_filter_expr(filter_expr)
            if parsed_filter is not None:
                filters.append(parsed_filter)

        if start_time is not None:
            filters.append(Episode.created_at >= start_time)

        if end_time is not None:
            filters.append(Episode.created_at <= end_time)

        if not filters:
            return stmt

        if isinstance(stmt, Select):
            return stmt.where(*filters)
        if isinstance(stmt, Delete):
            return stmt.where(*filters)
        raise TypeError(f"Unsupported statement type: {type(stmt)}")

    def _compile_episode_comparison_expr(
        self,
        expr: FilterComparison,
    ) -> ColumnElement[bool] | None:
        column, is_metadata = self._resolve_episode_field(expr.field)

        return parse_sql_filter(
            column=column,
            is_metadata=is_metadata,
            expr=expr,
        )

    def _compile_episode_filter_expr(
        self, expr: FilterExpr
    ) -> ColumnElement[bool] | None:
        if isinstance(expr, FilterComparison):
            return self._compile_episode_comparison_expr(expr)

        if isinstance(expr, FilterAnd):
            left = self._compile_episode_filter_expr(expr.left)
            right = self._compile_episode_filter_expr(expr.right)
            return and_(left, right)

        if isinstance(expr, FilterOr):
            left = self._compile_episode_filter_expr(expr.left)
            right = self._compile_episode_filter_expr(expr.right)
            return or_(left, right)

        raise TypeError(f"Unsupported filter expression type: {type(expr)!r}")

    @staticmethod
    def _resolve_episode_field(
        field: str,
    ) -> tuple[Any, bool] | tuple[None, bool]:
        normalized = field.lower()
        field_mapping: dict[str, Any] = {
            "uid": Episode.id,
            "id": Episode.id,
            "session_key": Episode.session_key,
            "session": Episode.session_key,
            "producer_id": Episode.producer_id,
            "producer_role": Episode.producer_role,
            "produced_for_id": Episode.produced_for_id,
            "episode_type": Episode.episode_type,
            "content": Episode.content,
            "created_at": Episode.created_at,
        }

        if normalized in field_mapping:
            return field_mapping[normalized], False

        if normalized.startswith(("m.", "metadata.")):
            key = normalized.split(".", 1)[1]
            return Episode.json_metadata[key].as_string(), True
        return None, False

    async def get_episode_messages(
        self,
        *,
        page_size: int | None = None,
        page_num: int | None = None,
        filter_expr: FilterExpr | None = None,
        start_time: AwareDatetime | None = None,
        end_time: AwareDatetime | None = None,
    ) -> list[EpisodeE]:
        stmt = select(Episode)

        stmt = self._apply_episode_filter(
            stmt,
            filter_expr=filter_expr,
            start_time=start_time,
            end_time=end_time,
        )

        if page_size is not None:
            stmt = stmt.limit(page_size)
            stmt = stmt.order_by(Episode.created_at.asc())

            if page_num is not None:
                stmt = stmt.offset(page_size * page_num)

        elif page_num is not None:
            raise InvalidArgumentError("Cannot specify offset without limit")

        async with self._create_session() as session:
            result = await session.execute(stmt)
            episode_messages = result.scalars().all()

        return [h.to_typed_model() for h in episode_messages]

    async def get_episode_messages_count(
        self,
        *,
        filter_expr: FilterExpr | None = None,
        start_time: AwareDatetime | None = None,
        end_time: AwareDatetime | None = None,
    ) -> int:
        stmt = select(func.count(Episode.id))

        stmt = self._apply_episode_filter(
            stmt,
            filter_expr=filter_expr,
            start_time=start_time,
            end_time=end_time,
        )

        async with self._create_session() as session:
            result = await session.execute(stmt)
            n_messages = result.scalar_one()

        return int(n_messages)

    @validate_call
    async def delete_episodes(self, episode_ids: list[EpisodeIdT]) -> None:
        try:
            int_episode_ids = TypeAdapter(list[int]).validate_python(episode_ids)
        except ValidationError as e:
            raise ResourceNotFoundError("Invalid episode IDs") from e

        stmt = delete(Episode).where(Episode.id.in_(int_episode_ids))

        async with self._create_session() as session:
            await session.execute(stmt)
            await session.commit()

    async def delete_episode_messages(
        self,
        *,
        filter_expr: FilterExpr | None = None,
        start_time: AwareDatetime | None = None,
        end_time: AwareDatetime | None = None,
    ) -> None:
        stmt = delete(Episode)

        stmt = self._apply_episode_filter(
            stmt,
            filter_expr=filter_expr,
            start_time=start_time,
            end_time=end_time,
        )

        async with self._create_session() as session:
            await session.execute(stmt)
            await session.commit()
