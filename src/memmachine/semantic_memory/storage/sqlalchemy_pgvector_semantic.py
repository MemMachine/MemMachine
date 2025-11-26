"""SQLAlchemy-backed semantic storage implementation using pgvector."""

from pathlib import Path
from typing import Any, overload

import numpy as np
from alembic import command
from alembic.config import Config
from pgvector.sqlalchemy import Vector
from pydantic import InstanceOf, TypeAdapter, ValidationError
from sqlalchemy import (
    Boolean,
    Column,
    ColumnElement,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    String,
    Table,
    and_,
    delete,
    insert,
    or_,
    select,
    text,
    update,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.engine import Connection
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import DeclarativeBase, aliased, mapped_column
from sqlalchemy.sql import Delete, Select, func

from memmachine.common.data_types import FilterablePropertyValue
from memmachine.common.episode_store.episode_model import EpisodeIdT
from memmachine.common.errors import InvalidArgumentError, ResourceNotFoundError
from memmachine.common.filter.filter_parser import (
    And as FilterAnd,
)
from memmachine.common.filter.filter_parser import (
    Comparison as FilterComparison,
)
from memmachine.common.filter.filter_parser import (
    FilterExpr,
)
from memmachine.common.filter.filter_parser import (
    Or as FilterOr,
)
from memmachine.semantic_memory.semantic_model import SemanticFeature, SetIdT
from memmachine.semantic_memory.storage.storage_base import (
    FeatureIdT,
    SemanticStorage,
)


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

    async def delete_all(self) -> None:
        async with self._create_session() as session:
            await session.execute(delete(citation_association_table))
            await session.execute(delete(SetIngestedHistory))
            await session.execute(delete(Feature))
            await session.commit()

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
        try:
            feature_id_int = int(feature_id)
        except (TypeError, ValueError) as e:
            raise ResourceNotFoundError(f"Invalid feature ID: {feature_id}") from e

        stmt = update(Feature).where(Feature.id == feature_id_int)

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

    async def get_feature(
        self,
        feature_id: FeatureIdT,
        load_citations: bool = False,
    ) -> SemanticFeature | None:
        try:
            feature_id_int = int(feature_id)
        except (TypeError, ValueError) as e:
            raise ResourceNotFoundError(f"Invalid feature ID: {feature_id}") from e

        stmt = select(Feature).where(Feature.id == feature_id_int)

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

    async def get_feature_set(
        self,
        *,
        page_size: int | None = None,
        page_num: int | None = None,
        vector_search_opts: SemanticStorage.VectorSearchOpts | None = None,
        tag_threshold: int | None = None,
        load_citations: bool = False,
        filter_expr: FilterExpr | None = None,
    ) -> list[SemanticFeature]:
        stmt = select(Feature)

        stmt = self._apply_feature_filter(
            stmt,
            k=None,
            vector_search_opts=vector_search_opts,
            filter_expr=filter_expr,
        )

        if vector_search_opts is None:
            stmt = stmt.order_by(Feature.created_at.asc(), Feature.id.asc())

        if page_size is not None:
            stmt = stmt.limit(page_size)
            stmt = stmt.offset(page_size * (page_num or 0))

        elif page_num is not None:
            raise InvalidArgumentError("Cannot specify offset without limit")

        async with self._create_session() as session:
            result = await session.execute(stmt)
            features = result.scalars().all()
            citations_map: dict[int, list[EpisodeIdT]] = {}
            if load_citations and features:
                citations_map = await self._load_feature_citations(
                    session,
                    [f.id for f in features if f.id is not None],
                )
        if tag_threshold is not None and tag_threshold > 0 and features:
            from collections import Counter

            counts = Counter(f.tag_id for f in features)
            features = [f for f in features if counts[f.tag_id] >= tag_threshold]

        return [f.to_typed_model(citations=citations_map.get(f.id)) for f in features]

    async def delete_features(self, feature_ids: list[FeatureIdT]) -> None:
        try:
            feature_ids_ints = TypeAdapter(list[int]).validate_python(feature_ids)
        except ValidationError as e:
            raise ResourceNotFoundError(f"Invalid feature IDs: {feature_ids}") from e

        stmt = delete(Feature).where(Feature.id.in_(feature_ids_ints))
        async with self._create_session() as session:
            await session.execute(stmt)
            await session.commit()

    async def delete_feature_set(
        self,
        *,
        filter_expr: FilterExpr | None = None,
    ) -> None:
        async with self._create_session() as session:
            stmt = delete(Feature)
            stmt = self._apply_feature_filter(
                stmt,
                filter_expr=filter_expr,
            )
            await session.execute(stmt)
            await session.commit()

    async def add_citations(
        self,
        feature_id: FeatureIdT,
        history_ids: list[EpisodeIdT],
    ) -> None:
        try:
            feature_id_int = int(feature_id)
        except (TypeError, ValueError) as e:
            raise ResourceNotFoundError(f"Invalid feature ID: {feature_id}") from e

        rows = [
            {"feature_id": feature_id_int, "history_id": str(hid)}
            for hid in history_ids
        ]

        stmt = insert(citation_association_table).values(rows)

        async with self._create_session() as session:
            await session.execute(stmt)
            await session.commit()

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

    async def add_history_to_set(
        self,
        set_id: str,
        history_id: EpisodeIdT,
    ) -> None:
        stmt = insert(SetIngestedHistory).values(set_id=set_id, history_id=history_id)

        async with self._create_session() as session:
            await session.execute(stmt)
            await session.commit()

    async def delete_history(self, history_ids: list[EpisodeIdT]) -> None:
        if not history_ids:
            return

        stmt = delete(SetIngestedHistory).where(
            SetIngestedHistory.history_id.in_(history_ids),
        )

        async with self._create_session() as session:
            await session.execute(stmt)
            await session.commit()

    async def delete_history_set(self, set_ids: list[SetIdT]) -> None:
        if not set_ids:
            return

        stmt = delete(SetIngestedHistory).where(
            SetIngestedHistory.set_id.in_(set_ids),
        )

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
        stmt: Select[Any],
        *,
        k: int | None = None,
        vector_search_opts: SemanticStorage.VectorSearchOpts | None = None,
    ) -> Select[Any]:
        if k is not None:
            stmt = stmt.limit(k)

        if vector_search_opts is not None:
            stmt = self._apply_vector_search_opts(
                stmt=stmt,
                vector_search_opts=vector_search_opts,
            )

        return stmt

    @overload
    def _apply_feature_filter(
        self,
        stmt: Select[Any],
        *,
        k: int | None = None,
        vector_search_opts: SemanticStorage.VectorSearchOpts | None = None,
        filter_expr: FilterExpr | None = None,
    ) -> Select[Any]: ...

    @overload
    def _apply_feature_filter(
        self,
        stmt: Delete,
        *,
        k: int | None = None,
        vector_search_opts: SemanticStorage.VectorSearchOpts | None = None,
        filter_expr: FilterExpr | None = None,
    ) -> Delete: ...

    def _apply_feature_filter(
        self,
        stmt: Select[Any] | Delete,
        *,
        k: int | None = None,
        vector_search_opts: SemanticStorage.VectorSearchOpts | None = None,
        filter_expr: FilterExpr | None = None,
    ) -> Select[Any] | Delete:
        if isinstance(stmt, Select):
            working_stmt: Select[Any] | Delete = self._apply_feature_select_filter(
                stmt,
                k=k,
                vector_search_opts=vector_search_opts,
            )
            if filter_expr is not None:
                clause = self._compile_feature_filter_expr(filter_expr, Feature)
                working_stmt = working_stmt.where(clause)
            return working_stmt

        if k is not None or vector_search_opts is not None:
            raise RuntimeError(
                "k and vector_search_opts are only supported for select statements"
            )

        delete_stmt = stmt
        if filter_expr is not None:
            clause = self._compile_feature_filter_expr(filter_expr, Feature)
            delete_stmt = delete_stmt.where(clause)

        return delete_stmt

    def _compile_feature_comparison_expr(
        self,
        expr: FilterComparison,
        table: type[Feature],
    ) -> ColumnElement[bool]:
        column, is_metadata = self._resolve_feature_field(table, expr.field)

        if column is None:
            raise ValueError(f"Unsupported feature filter field: {expr.field}")

        if expr.op == "=":
            value = expr.value
            if isinstance(value, list):
                raise ValueError("'=' comparison cannot accept list values")
            if is_metadata:
                value = self._normalize_metadata_value(value)
                return column == value
            return column == value

        if expr.op == "in":
            if not isinstance(expr.value, list):
                raise ValueError("IN comparison requires a list of values")

            values = expr.value
            if is_metadata:
                values = [self._normalize_metadata_value(v) for v in values]
            return column.in_(values)

        if expr.op == "is_null":
            return column.is_(None)

        if expr.op == "is_not_null":
            return column.is_not(None)

        raise ValueError(f"Unsupported operator: {expr.op}")

    def _compile_feature_filter_expr(
        self,
        expr: FilterExpr,
        table: type[Feature],
    ) -> ColumnElement[bool]:
        if isinstance(expr, FilterComparison):
            return self._compile_feature_comparison_expr(expr, table)

        if isinstance(expr, FilterAnd):
            left = self._compile_feature_filter_expr(expr.left, table)
            right = self._compile_feature_filter_expr(expr.right, table)

            return and_(left, right)

        if isinstance(expr, FilterOr):
            left = self._compile_feature_filter_expr(expr.left, table)
            right = self._compile_feature_filter_expr(expr.right, table)

            return or_(left, right)

        raise TypeError(f"Unsupported filter expression type: {type(expr)!r}")

    @staticmethod
    def _normalize_metadata_value(
        value: FilterablePropertyValue | list[FilterablePropertyValue],
    ) -> str:
        if isinstance(value, bool):
            return "true" if value else "false"
        return "" if value is None else str(value)

    @staticmethod
    def _resolve_feature_field(
        table: type[Feature],
        field: str,
    ) -> tuple[Any, bool] | tuple[None, bool]:
        normalized = field
        field_mapping = {
            "set_id": table.set_id,
            "set": table.set_id,
            "semantic_category_id": table.semantic_category_id,
            "category_name": table.semantic_category_id,
            "category": table.semantic_category_id,
            "tag_id": table.tag_id,
            "tag": table.tag_id,
            "feature": table.feature,
            "feature_name": table.feature,
            "value": table.value,
            "created_at": table.created_at,
            "updated_at": table.updated_at,
        }
        if normalized in field_mapping:
            return field_mapping[normalized], False
        if normalized.startswith(("m.", "metadata.")):
            key = normalized.split(".", 1)[1]
            return table.json_metadata[key].as_string(), True
        return None, False

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
