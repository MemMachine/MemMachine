"""SQLAlchemy implementation of the episode storage layer."""

import contextlib
import hashlib
import logging
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
    UniqueConstraint,
    delete,
    func,
    select,
    text,
)
from sqlalchemy import Enum as SAEnum
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.exc import OperationalError
from sqlalchemy.ext.asyncio import (
    AsyncConnection,
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
)
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
    FilterExpr,
    demangle_user_metadata_key,
    normalize_filter_field,
)
from memmachine.common.filter.sql_filter_util import compile_sql_filter

logger = logging.getLogger(__name__)


def compute_content_hash(session_key: str, producer_id: str, content: str) -> str:
    """Compute a SHA-256 hash for episode deduplication.

    The hash is derived from ``session_key``, ``producer_id``, and
    ``content`` joined with null-byte separators to prevent ambiguity
    between field boundaries.
    """
    payload = f"{session_key}\0{producer_id}\0{content}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


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
    content_hash = mapped_column(String(64), nullable=True)

    __table_args__ = (
        UniqueConstraint("content_hash", name="uq_episode_content_hash"),
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

    def to_typed_model(self, *, is_new: bool = True) -> EpisodeE:
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
            is_new=is_new,
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
            await self._migrate_content_hash()
        except (OperationalError, socket.gaierror) as err:
            raise ConfigurationError(
                "Failed to connect to the database during startup, please check your configuration."
            ) from err

    async def _migrate_content_hash(self) -> None:
        """Ensure the content_hash column and unique constraint exist, then backfill."""
        dialect = self._engine.dialect.name

        async with self._engine.begin() as conn:
            # Check if column exists by inspecting the table.
            has_column = await conn.run_sync(self._check_content_hash_column)

            if not has_column:
                if dialect == "postgresql":
                    await conn.execute(
                        text(
                            "ALTER TABLE episodestore "
                            "ADD COLUMN IF NOT EXISTS content_hash VARCHAR(64)"
                        )
                    )
                else:
                    # SQLite: ALTER TABLE ADD COLUMN (no IF NOT EXISTS).
                    with contextlib.suppress(OperationalError):
                        await conn.execute(
                            text(
                                "ALTER TABLE episodestore "
                                "ADD COLUMN content_hash VARCHAR(64)"
                            )
                        )

            # Ensure the unique constraint exists.
            has_constraint = await conn.run_sync(self._check_content_hash_constraint)
            if not has_constraint:
                # Backfill before adding the constraint so NULLs don't block it.
                await self._backfill_content_hashes_in_conn(conn)
                # Remove duplicate rows, keeping the one with the lowest id.
                await self._remove_duplicate_hashes(conn, dialect)
                if dialect == "postgresql":
                    await conn.execute(
                        text(
                            "ALTER TABLE episodestore "
                            "ADD CONSTRAINT uq_episode_content_hash "
                            "UNIQUE (content_hash)"
                        )
                    )
                else:
                    with contextlib.suppress(OperationalError):
                        await conn.execute(
                            text(
                                "CREATE UNIQUE INDEX IF NOT EXISTS "
                                "uq_episode_content_hash "
                                "ON episodestore (content_hash)"
                            )
                        )
                logger.info("Created unique constraint uq_episode_content_hash")
                return  # backfill already done above

        # Backfill NULL content_hash rows (normal path when constraint exists).
        await self._backfill_content_hashes()

    @staticmethod
    def _check_content_hash_column(conn: object) -> bool:
        """Check if the content_hash column exists (sync, for run_sync)."""
        from sqlalchemy import inspect as sa_inspect

        inspector = sa_inspect(conn)  # type: ignore[arg-type]
        columns = inspector.get_columns("episodestore")
        return any(col["name"] == "content_hash" for col in columns)

    @staticmethod
    def _check_content_hash_constraint(conn: object) -> bool:
        """Check if the uq_episode_content_hash constraint exists (sync, for run_sync)."""
        from sqlalchemy import inspect as sa_inspect

        inspector = sa_inspect(conn)  # type: ignore[arg-type]
        unique_constraints = inspector.get_unique_constraints("episodestore")
        if any(uc["name"] == "uq_episode_content_hash" for uc in unique_constraints):
            return True
        # Some backends report unique constraints as indexes instead.
        indexes = inspector.get_indexes("episodestore")
        return any(
            idx.get("name") == "uq_episode_content_hash" and idx.get("unique")
            for idx in indexes
        )

    async def _backfill_content_hashes(self) -> None:
        """Compute and set content_hash for rows where it is NULL."""
        batch_size = 500
        async with self._create_session() as session:
            while True:
                result = await session.execute(
                    select(Episode)
                    .where(Episode.content_hash.is_(None))
                    .limit(batch_size)
                )
                rows = result.scalars().all()
                if not rows:
                    break
                for row in rows:
                    row.content_hash = compute_content_hash(
                        row.session_key, row.producer_id, row.content
                    )
                await session.commit()
                logger.info("Backfilled content_hash for %d episodes", len(rows))

    @staticmethod
    async def _backfill_content_hashes_in_conn(conn: AsyncConnection) -> None:
        """Backfill NULL content_hash rows using a raw connection (no ORM session)."""
        batch_size = 500
        while True:
            result = await conn.execute(
                text(
                    "SELECT id, session_key, producer_id, content "
                    "FROM episodestore WHERE content_hash IS NULL "
                    f"LIMIT {batch_size}"
                )
            )
            rows = result.fetchall()
            if not rows:
                break
            for row in rows:
                h = compute_content_hash(row.session_key, row.producer_id, row.content)
                await conn.execute(
                    text("UPDATE episodestore SET content_hash = :hash WHERE id = :id"),
                    {"hash": h, "id": row.id},
                )
            logger.info(
                "Backfilled content_hash for %d episodes (migration)", len(rows)
            )

    @staticmethod
    async def _remove_duplicate_hashes(conn: AsyncConnection, dialect: str) -> None:
        """Delete duplicate content_hash rows, keeping the row with the lowest id."""
        if dialect == "postgresql":
            result = await conn.execute(
                text(
                    "DELETE FROM episodestore "
                    "WHERE id NOT IN ("
                    "  SELECT MIN(id) FROM episodestore "
                    "  WHERE content_hash IS NOT NULL "
                    "  GROUP BY content_hash"
                    ") AND content_hash IS NOT NULL"
                )
            )
        else:
            # SQLite compatible syntax.
            result = await conn.execute(
                text(
                    "DELETE FROM episodestore "
                    "WHERE content_hash IS NOT NULL AND id NOT IN ("
                    "  SELECT MIN(id) FROM episodestore "
                    "  WHERE content_hash IS NOT NULL "
                    "  GROUP BY content_hash"
                    ")"
                )
            )
        deleted = result.rowcount
        if deleted:
            logger.info("Removed %d duplicate episode rows during migration", deleted)

    async def delete_all(self) -> None:
        async with self._create_session() as session:
            await session.execute(delete(Episode))
            await session.commit()

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
            content_hash = compute_content_hash(
                session_key, entry.producer_id, entry.content
            )
            entry_values: dict[str, Any] = {
                "content": entry.content,
                "session_key": session_key,
                "producer_id": entry.producer_id,
                "producer_role": entry.producer_role,
                "content_hash": content_hash,
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

        all_hashes = [v["content_hash"] for v in values_to_insert]
        return await self._insert_with_dedup(values_to_insert, all_hashes)

    async def _insert_with_dedup(
        self,
        values_to_insert: list[dict[str, Any]],
        all_hashes: list[str],
    ) -> list[EpisodeE]:
        """Insert episodes with ON CONFLICT dedup, return all in input order."""
        dialect = self._engine.dialect.name

        async with self._create_session() as session:
            # Snapshot which hashes already exist before the INSERT.
            pre_existing = await self._fetch_existing_hashes(session, all_hashes)

            # Insert new rows, silently skip duplicates.
            await self._do_conflict_insert(session, dialect, values_to_insert)
            await session.commit()

            # Fetch all rows (new + pre-existing) by hash.
            all_rows = await self._fetch_by_hashes(session, all_hashes)

        # Build result in original input order with is_new flag.
        row_by_hash: dict[str, Episode] = {ep.content_hash: ep for ep in all_rows}
        ordered: list[EpisodeE] = []
        for val in values_to_insert:
            ch = val["content_hash"]
            ep_orm = row_by_hash[ch]
            ordered.append(ep_orm.to_typed_model(is_new=ch not in pre_existing))
        return ordered

    @staticmethod
    async def _fetch_existing_hashes(
        session: AsyncSession,
        hashes: list[str],
    ) -> set[str]:
        """Return the subset of hashes that already exist in the table."""
        result = await session.execute(
            select(Episode.content_hash).where(Episode.content_hash.in_(hashes))
        )
        return {row[0] for row in result.all()}

    @staticmethod
    async def _do_conflict_insert(
        session: AsyncSession,
        dialect: str,
        values: list[dict[str, Any]],
    ) -> None:
        """Run dialect-appropriate INSERT ... ON CONFLICT DO NOTHING."""
        if dialect == "postgresql":
            stmt = (
                pg_insert(Episode)
                .values(values)
                .on_conflict_do_nothing(constraint="uq_episode_content_hash")
            )
            await session.execute(stmt)
        else:
            # SQLite: insert one-by-one with ON CONFLICT DO NOTHING.
            for row_values in values:
                stmt = (
                    sqlite_insert(Episode)
                    .values(**row_values)
                    .on_conflict_do_nothing(index_elements=["content_hash"])
                )
                await session.execute(stmt)

    @staticmethod
    async def _fetch_by_hashes(
        session: AsyncSession,
        hashes: list[str],
    ) -> list[Episode]:
        """Fetch episode rows by their content_hash values."""
        result = await session.execute(
            select(Episode).where(Episode.content_hash.in_(hashes))
        )
        return list(result.scalars().all())

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
            parsed_filter = compile_sql_filter(filter_expr, self._resolve_episode_field)
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

    @staticmethod
    def _resolve_episode_field(
        field: str,
    ) -> tuple[Any, bool] | tuple[None, bool]:
        internal_name, is_user_metadata = normalize_filter_field(field)
        if is_user_metadata:
            key = demangle_user_metadata_key(internal_name)
            return Episode.json_metadata[key], True

        # Check for system field mappings (case-insensitive)
        normalized = internal_name.lower()
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
