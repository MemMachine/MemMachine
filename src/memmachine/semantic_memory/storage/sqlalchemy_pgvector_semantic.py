"""SQLAlchemy-backed semantic storage implementation using pgvector."""

from pathlib import Path
from typing import Any, TypeVar

import numpy as np
from alembic import command
from alembic.config import Config
from pgvector.sqlalchemy import Vector
from pydantic import InstanceOf, TypeAdapter, validate_call
from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    String,
    Table,
    delete,
    insert,
    select,
    text,
    update,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.engine import Connection
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import DeclarativeBase, aliased, mapped_column
from sqlalchemy.sql import Delete, Select, func

from memmachine.episode_store.episode_model import EpisodeIdT
from memmachine.semantic_memory.semantic_model import SemanticFeature, SetIdT
from memmachine.semantic_memory.storage.storage_base import (
    FeatureIdT,
    SemanticStorage,
)

StmtT = TypeVar("StmtT", Select[Any], Delete)


class BaseSemanticStorage(DeclarativeBase):
    """Declarative base for semantic memory SQLAlchemy models."""


citation_association_table = Table(
    "citations",
    BaseSemanticStorage.metadata,
    Column(
        "feature_id",
        Integer,
        ForeignKey("feature.id", ondelete="CASCADE", onupdate="CASCADE"),
        primary_key=True,
    ),
    Column(
        "history_id",
        String,
        primary_key=True,
    ),
)


class Feature(BaseSemanticStorage):
    """SQLAlchemy mapping for persisted semantic features."""

    __tablename__ = "feature"
    id = mapped_column(Integer, primary_key=True)

    # Feature data
    set_id = mapped_column(String, nullable=False)
    semantic_category_id = mapped_column(String, nullable=False)
    tag_id = mapped_column(String, nullable=False)
    feature = mapped_column(String, nullable=False)
    value = mapped_column(String, nullable=False)

    # metadata
    created_at = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )
    updated_at = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )
    embedding = mapped_column(Vector)
    json_metadata = mapped_column(
        JSONB,
        name="metadata",
        server_default=text("'{}'::jsonb"),
        nullable=False,
    )

    __table_args__ = (
        Index("idx_feature_set_id", "set_id"),
        Index("idx_feature_set_id_semantic_category", "set_id", "semantic_category_id"),
        Index(
            "idx_feature_set_semantic_category_tag",
            "set_id",
            "semantic_category_id",
            "tag_id",
        ),
        Index(
            "idx_feature_set_semantic_category_tag_feature",
            "set_id",
            "semantic_category_id",
            "tag_id",
            "feature",
        ),
    )

    def to_typed_model(
        self,
        *,
        citations: list[EpisodeIdT] | None = None,
    ) -> SemanticFeature:
        return SemanticFeature(
            metadata=SemanticFeature.Metadata(
                id=FeatureIdT(self.id),
                citations=citations,
                other=self.json_metadata or None,
            ),
            set_id=self.set_id,
            category=self.semantic_category_id,
            tag=self.tag_id,
            feature_name=self.feature,
            value=self.value,
        )


class SetIngestedHistory(BaseSemanticStorage):
    """Tracks which history messages have been processed for a set."""

    __tablename__ = "set_ingested_history"
    set_id = mapped_column(String, primary_key=True)
    history_id = mapped_column(
        String,
        primary_key=True,
    )
    ingested = mapped_column(Boolean, default=False, nullable=False)


async def apply_alembic_migrations(engine: AsyncEngine) -> None:
    """Run Alembic migrations for the semantic storage tables."""
    script_location = Path(__file__).parent / "alembic_pg"
    versions_location = script_location / "versions"

    async with engine.begin() as conn:
        await conn.exec_driver_sql("CREATE EXTENSION IF NOT EXISTS vector")

        def run_migrations(sync_conn: Connection) -> None:
            config = Config()
            script_path = str(script_location.resolve())
            versions_path = str(versions_location.resolve())
            config.set_main_option("script_location", script_path)
            config.set_main_option("version_locations", versions_path)
            config.set_main_option("path_separator", "os")
            config.set_main_option("sqlalchemy.url", str(sync_conn.engine.url))
            config.attributes["connection"] = sync_conn
            command.upgrade(config, "head")

        await conn.run_sync(run_migrations)


class SqlAlchemyPgVectorSemanticStorage(SemanticStorage):
    """Concrete SemanticStorageBase backed by PostgreSQL with pgvector."""

    def __init__(self, sqlalchemy_engine: AsyncEngine) -> None:
        """Initialize the storage with an async SQLAlchemy engine."""
        self._engine = sqlalchemy_engine
        self._session_factory = async_sessionmaker(
            bind=self._engine,
            expire_on_commit=False,
        )

    def _create_session(self) -> AsyncSession:
        return self._session_factory()

    async def _initialize_db(self) -> None:
        await apply_alembic_migrations(self._engine)

    async def startup(self) -> None:
        await self._initialize_db()

    async def cleanup(self) -> None:
        await self._engine.dispose()

    @validate_call
    async def delete_all(self) -> None:
        async with self._create_session() as session:
            await session.execute(delete(citation_association_table))
            await session.execute(delete(SetIngestedHistory))
            await session.execute(delete(Feature))
            await session.commit()

    @validate_call
    async def add_feature(
        self,
        *,
        set_id: str,
        category_name: str,
        feature: str,
        value: str,
        tag: str,
        embedding: InstanceOf[np.ndarray],
        metadata: dict[str, Any] | None = None,
    ) -> FeatureIdT:
        stmt = (
            insert(Feature)
            .values(
                set_id=set_id,
                semantic_category_id=category_name,
                tag_id=tag,
                feature=feature,
                value=value,
                embedding=embedding,
                json_metadata=metadata,
            )
            .returning(Feature.id)
        )

        async with self._create_session() as session:
            result = await session.execute(stmt)
            await session.commit()
            feature_id = result.scalar_one()

        return FeatureIdT(feature_id)

    @validate_call
    async def update_feature(
        self,
        feature_id: FeatureIdT,
        *,
        set_id: SetIdT | None = None,
        category_name: str | None = None,
        feature: str | None = None,
        value: str | None = None,
        tag: str | None = None,
        embedding: InstanceOf[np.ndarray] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        stmt = update(Feature).where(Feature.id == int(feature_id))

        if set_id is not None:
            stmt = stmt.values(set_id=set_id)
        if category_name is not None:
            stmt = stmt.values(semantic_category_id=category_name)
        if feature is not None:
            stmt = stmt.values(feature=feature)
        if value is not None:
            stmt = stmt.values(value=value)
        if tag is not None:
            stmt = stmt.values(tag_id=tag)
        if embedding is not None:
            stmt = stmt.values(embedding=embedding)
        if metadata is not None:
            stmt = stmt.values(json_metadata=metadata)

        async with self._create_session() as session:
            await session.execute(stmt)
            await session.commit()

    @validate_call
    async def get_feature(
        self,
        feature_id: FeatureIdT,
        load_citations: bool = False,
    ) -> SemanticFeature | None:
        stmt = select(Feature).where(Feature.id == int(feature_id))

        async with self._create_session() as session:
            result = await session.execute(stmt)
            feature = result.scalar_one_or_none()

            citations_map: dict[int, list[EpisodeIdT]] = {}
            if feature is not None and load_citations:
                citations_map = await self._load_feature_citations(
                    session,
                    [feature.id],
                )

        if feature is None:
            return None

        return feature.to_typed_model(citations=citations_map.get(feature.id))

    @validate_call
    async def get_feature_set(
        self,
        *,
        set_ids: list[SetIdT] | None = None,
        category_names: list[str] | None = None,
        feature_names: list[str] | None = None,
        tags: list[str] | None = None,
        limit: int | None = None,
        vector_search_opts: SemanticStorage.VectorSearchOpts | None = None,
        tag_threshold: int | None = None,
        load_citations: bool = False,
    ) -> list[SemanticFeature]:
        stmt = select(Feature)

        stmt = self._apply_feature_filter(
            stmt,
            set_ids=set_ids,
            category_names=category_names,
            tags=tags,
            feature_names=feature_names,
            thresh=tag_threshold,
            k=limit,
            vector_search_opts=vector_search_opts,
        )

        async with self._create_session() as session:
            result = await session.execute(stmt)
            features = result.scalars().all()
            citations_map: dict[int, list[EpisodeIdT]] = {}
            if load_citations and features:
                citations_map = await self._load_feature_citations(
                    session,
                    [f.id for f in features if f.id is not None],
                )

        return [f.to_typed_model(citations=citations_map.get(f.id)) for f in features]

    @validate_call
    async def delete_features(self, feature_ids: list[FeatureIdT]) -> None:
        feature_ids_ints = [int(f_id) for f_id in feature_ids]

        stmt = delete(Feature).where(Feature.id.in_(feature_ids_ints))
        async with self._create_session() as session:
            await session.execute(stmt)
            await session.commit()

    @validate_call
    async def delete_feature_set(
        self,
        *,
        set_ids: list[SetIdT] | None = None,
        category_names: list[str] | None = None,
        feature_names: list[str] | None = None,
        tags: list[str] | None = None,
        thresh: int | None = None,
        limit: int | None = None,
        vector_search_opts: SemanticStorage.VectorSearchOpts | None = None,
    ) -> None:
        stmt = delete(Feature)

        stmt = self._apply_feature_filter(
            stmt,
            set_ids=set_ids,
            category_names=category_names,
            tags=tags,
            feature_names=feature_names,
            thresh=thresh,
            k=limit,
            vector_search_opts=vector_search_opts,
        )

        async with self._create_session() as session:
            await session.execute(stmt)
            await session.commit()

    @validate_call()
    async def add_citations(
        self,
        feature_id: FeatureIdT,
        history_ids: list[EpisodeIdT],
    ) -> None:
        rows = [
            {"feature_id": int(feature_id), "history_id": str(hid)}
            for hid in history_ids
        ]

        stmt = insert(citation_association_table).values(rows)

        async with self._create_session() as session:
            await session.execute(stmt)
            await session.commit()

    @validate_call
    async def get_history_messages(
        self,
        *,
        set_ids: list[str] | None = None,
        limit: int | None = None,
        is_ingested: bool | None = None,
    ) -> list[EpisodeIdT]:
        stmt = select(SetIngestedHistory.history_id).order_by(
            SetIngestedHistory.history_id.asc(),
        )

        stmt = self._apply_history_filter(
            stmt,
            set_ids=set_ids,
            is_ingested=is_ingested,
            limit=limit,
        )

        async with self._create_session() as session:
            result = await session.execute(stmt)
            history_ids = result.scalars().all()

        return TypeAdapter(list[EpisodeIdT]).validate_python(history_ids)

    @validate_call
    async def get_history_messages_count(
        self,
        *,
        set_ids: list[str] | None = None,
        is_ingested: bool | None = None,
    ) -> int:
        stmt = select(func.count(SetIngestedHistory.history_id))

        stmt = self._apply_history_filter(
            stmt,
            set_ids=set_ids,
            is_ingested=is_ingested,
        )

        async with self._create_session() as session:
            result = await session.execute(stmt)
            count = result.scalar_one()

        return count

    @validate_call
    async def mark_messages_ingested(
        self,
        set_id: str,
        history_ids: list[EpisodeIdT],
    ) -> None:
        if len(history_ids) == 0:
            raise ValueError("No ids provided")

        stmt = (
            update(SetIngestedHistory)
            .where(SetIngestedHistory.set_id == set_id)
            .where(SetIngestedHistory.history_id.in_(history_ids))
            .values(ingested=True)
        )

        async with self._create_session() as session:
            await session.execute(stmt)
            await session.commit()

    @validate_call
    async def add_history_to_set(
        self,
        set_id: str,
        history_id: EpisodeIdT,
    ) -> None:
        stmt = insert(SetIngestedHistory).values(set_id=set_id, history_id=history_id)

        async with self._create_session() as session:
            await session.execute(stmt)
            await session.commit()

    def _apply_history_filter(
        self,
        stmt: Select,
        *,
        set_ids: list[str] | None = None,
        is_ingested: bool | None = None,
        limit: int | None = None,
    ) -> Select:
        if set_ids is not None and len(set_ids) > 0:
            stmt = stmt.where(SetIngestedHistory.set_id.in_(set_ids))
        if is_ingested is not None:
            stmt = stmt.where(SetIngestedHistory.ingested == is_ingested)
        if limit is not None:
            stmt = stmt.limit(limit)

        return stmt

    def _apply_vector_search_opts(
        self,
        *,
        stmt: Select[Any],
        vector_search_opts: SemanticStorage.VectorSearchOpts,
    ) -> Select[Any]:
        if vector_search_opts.min_distance is not None:
            threshold = 1 - vector_search_opts.min_distance
            stmt = stmt.where(
                Feature.embedding.cosine_distance(
                    vector_search_opts.query_embedding,
                )
                <= threshold,
            )

        stmt = stmt.order_by(
            Feature.embedding.cosine_distance(
                vector_search_opts.query_embedding,
            ).asc(),
        )

        return stmt

    def _apply_feature_select_filter(
        self,
        stmt: StmtT,
        *,
        k: int | None = None,
        vector_search_opts: SemanticStorage.VectorSearchOpts | None = None,
    ) -> StmtT:
        if k is not None:
            if type(stmt) is not Select:
                raise RuntimeError("k is only supported for select statements")
            stmt = stmt.limit(k)

        if vector_search_opts is not None:
            if type(stmt) is not Select:
                raise RuntimeError(
                    "vector_search_opts is only supported for select statements"
                )

            stmt = self._apply_vector_search_opts(
                stmt=stmt,
                vector_search_opts=vector_search_opts,
            )

        return stmt

    def _apply_feature_filter(
        self,
        stmt: StmtT,
        *,
        set_ids: list[str] | None = None,
        category_names: list[str] | None = None,
        feature_names: list[str] | None = None,
        tags: list[str] | None = None,
        thresh: int | None = None,
        k: int | None = None,
        vector_search_opts: SemanticStorage.VectorSearchOpts | None = None,
    ) -> StmtT:
        _StmtT = TypeVar("_StmtT", Select[Any], Delete)

        def _apply_feature_id_filter(
            _stmt: _StmtT,
        ) -> _StmtT:
            if set_ids is not None and len(set_ids) > 0:
                _stmt = _stmt.where(Feature.set_id.in_(set_ids))

            if category_names is not None and len(category_names) > 0:
                _stmt = _stmt.where(Feature.semantic_category_id.in_(category_names))

            if tags is not None and len(tags) > 0:
                _stmt = _stmt.where(Feature.tag_id.in_(tags))

            if feature_names is not None and len(feature_names) > 0:
                _stmt = _stmt.where(Feature.feature.in_(feature_names))

            _stmt = self._apply_feature_select_filter(
                _stmt,
                k=k,
                vector_search_opts=vector_search_opts,
            )

            return _stmt

        stmt = _apply_feature_id_filter(stmt)

        if thresh is not None:
            subquery_stmt = self._get_tags_with_more_than_k_features(thresh)
            subquery = _apply_feature_id_filter(subquery_stmt)

            stmt = stmt.where(Feature.tag_id.in_(subquery))

        return stmt

    @staticmethod
    def _get_tags_with_more_than_k_features(k: int) -> Select[Any]:
        return select(Feature.tag_id).group_by(Feature.tag_id).having(func.count() >= k)

    async def _load_feature_citations(
        self,
        session: AsyncSession,
        feature_ids: list[int],
    ) -> dict[int, list[EpisodeIdT]]:
        if not feature_ids:
            return {}

        stmt = select(
            citation_association_table.c.feature_id,
            citation_association_table.c.history_id,
        ).where(citation_association_table.c.feature_id.in_(feature_ids))

        result = await session.execute(stmt)

        citations: dict[int, list[EpisodeIdT]] = {
            feature_id: [] for feature_id in feature_ids
        }

        for feature_id, history_id in result:
            citations.setdefault(feature_id, []).append(history_id)

        return citations

    async def get_history_set_ids(
        self,
        *,
        min_uningested_messages: int | None = None,
    ) -> list[SetIdT]:
        stmt = select(SetIngestedHistory.set_id).distinct()

        if min_uningested_messages is not None and min_uningested_messages > 0:
            inner = aliased(SetIngestedHistory)

            count_uningested = (
                select(func.count(inner.set_id))
                .where(
                    inner.set_id == SetIngestedHistory.set_id,  # correlate on set_id
                    inner.ingested.is_(False),
                )
                .scalar_subquery()
            )

            stmt = stmt.where(count_uningested >= min_uningested_messages)

        async with self._create_session() as session:
            result = await session.execute(stmt)
            set_ids = result.scalars().all()

        return TypeAdapter(list[SetIdT]).validate_python(set_ids)
