"""Semantic storage backed by relational metadata and a VectorStore collection."""

from __future__ import annotations

from collections.abc import AsyncIterator, Mapping, MutableMapping, Sequence
from datetime import datetime
from typing import Any, cast
from uuid import UUID, uuid5

import numpy as np
from pydantic import AwareDatetime, InstanceOf, TypeAdapter, ValidationError
from sqlalchemy import (
    JSON,
    Boolean,
    Column,
    ColumnElement,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    String,
    Table,
    delete,
    insert,
    select,
    union,
    update,
)
from sqlalchemy.engine import CursorResult
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import DeclarativeBase, aliased, mapped_column
from sqlalchemy.sql import Delete, Select, func

from memmachine_server.common.episode_store.episode_model import EpisodeIdT
from memmachine_server.common.errors import InvalidArgumentError, ResourceNotFoundError
from memmachine_server.common.filter.filter_parser import (
    FilterExpr,
    demangle_user_metadata_key,
    normalize_filter_field,
)
from memmachine_server.common.filter.sql_filter_util import (
    FieldEncoding,
    compile_sql_filter,
)
from memmachine_server.common.vector_store import Record, VectorStoreCollection
from memmachine_server.semantic_memory.semantic_model import SemanticFeature, SetIdT
from memmachine_server.semantic_memory.storage.storage_base import (
    FeatureIdT,
    SemanticStorage,
)
from memmachine_server.semantic_memory.storage.text_sanitizer import sanitize_pg_text

_FEATURE_VECTOR_NAMESPACE = UUID("f4f9f7b0-99dd-4a4a-9f24-682077ae1ebd")
_DEFAULT_VECTOR_QUERY_LIMIT = 10_000


def feature_vector_uuid(feature_id: FeatureIdT) -> UUID:
    """Return the stable vector record UUID for a semantic feature id."""
    return uuid5(_FEATURE_VECTOR_NAMESPACE, str(feature_id))


class BaseVectorSemanticStorage(DeclarativeBase):
    """Declarative base for vector-backed semantic memory tables."""


vector_citation_association_table = Table(
    "vector_semantic_citations",
    BaseVectorSemanticStorage.metadata,
    Column(
        "feature_id",
        Integer,
        ForeignKey(
            "vector_semantic_feature.id",
            ondelete="CASCADE",
            onupdate="CASCADE",
        ),
        primary_key=True,
    ),
    Column("history_id", String, primary_key=True),
)


class VectorSemanticFeature(BaseVectorSemanticStorage):
    """Relational metadata for a vector-backed semantic feature."""

    __tablename__ = "vector_semantic_feature"

    id = mapped_column(Integer, primary_key=True)
    set_id = mapped_column(String, nullable=False, index=True)
    semantic_category_id = mapped_column(String, nullable=False)
    tag_id = mapped_column(String, nullable=False)
    feature = mapped_column(String, nullable=False)
    value = mapped_column(String, nullable=False)
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
    json_metadata = mapped_column(
        JSON,
        name="metadata",
        default=dict,
        server_default="{}",
        nullable=False,
    )

    __table_args__ = (
        Index("idx_vector_semantic_feature_set_id", "set_id"),
        Index(
            "idx_vector_semantic_feature_set_category",
            "set_id",
            "semantic_category_id",
        ),
        Index(
            "idx_vector_semantic_feature_set_category_tag",
            "set_id",
            "semantic_category_id",
            "tag_id",
        ),
        Index(
            "idx_vector_semantic_feature_lookup",
            "set_id",
            "semantic_category_id",
            "tag_id",
            "feature",
        ),
    )

    def to_typed_model(
        self,
        *,
        citations: Sequence[EpisodeIdT] | None = None,
    ) -> SemanticFeature:
        return SemanticFeature(
            metadata=SemanticFeature.Metadata(
                id=FeatureIdT(str(self.id)),
                citations=citations,
                other=self.json_metadata or None,
            ),
            set_id=self.set_id,
            category=self.semantic_category_id,
            tag=self.tag_id,
            feature_name=self.feature,
            value=self.value,
        )


class VectorSemanticSetIngestedHistory(BaseVectorSemanticStorage):
    """Tracks semantic ingestion state for vector-backed storage."""

    __tablename__ = "vector_semantic_set_ingested_history"

    set_id = mapped_column(String, primary_key=True, index=True)
    history_id = mapped_column(String, primary_key=True)
    created_at = mapped_column(DateTime(timezone=True), server_default=func.now())
    ingested = mapped_column(Boolean, default=False, nullable=False)

    __table_args__ = (
        Index(
            "ix_vector_semantic_history_set_id_ingested",
            "set_id",
            "ingested",
        ),
    )


class VectorStoreSemanticStorage(SemanticStorage):
    """SemanticStorage using SQLAlchemy metadata and VectorStore embeddings."""

    backend_name = "vector_store"

    def __init__(
        self,
        sqlalchemy_engine: AsyncEngine,
        vector_collection: VectorStoreCollection,
    ) -> None:
        """Initialize storage with an async SQLAlchemy engine and vector collection."""
        self._engine = sqlalchemy_engine
        self._vector_collection = vector_collection
        self._session_factory = async_sessionmaker(
            bind=self._engine,
            expire_on_commit=False,
        )

    def _create_session(self) -> AsyncSession:
        return self._session_factory()

    async def startup(self) -> None:
        async with self._engine.begin() as conn:
            await conn.run_sync(BaseVectorSemanticStorage.metadata.create_all)

    async def cleanup(self) -> None:
        await self._engine.dispose()

    async def delete_all(self) -> None:
        feature_ids = await self._feature_ids_for_filter(None)
        async with self._create_session() as session:
            await session.execute(delete(vector_citation_association_table))
            await session.execute(delete(VectorSemanticSetIngestedHistory))
            await session.execute(delete(VectorSemanticFeature))
            await session.commit()
        await self._delete_vector_records(feature_ids)

    async def reset_set_ids(self, set_ids: Sequence[SetIdT]) -> None:
        del set_ids

    async def add_feature(
        self,
        *,
        set_id: SetIdT,
        category_name: str,
        feature: str,
        value: str,
        tag: str,
        embedding: InstanceOf[np.ndarray],
        metadata: Mapping[str, Any] | None = None,
    ) -> FeatureIdT:
        stmt = (
            insert(VectorSemanticFeature)
            .values(
                set_id=set_id,
                semantic_category_id=category_name,
                tag_id=sanitize_pg_text(tag, context="feature.tag"),
                feature=sanitize_pg_text(feature, context="feature.feature"),
                value=sanitize_pg_text(value, context="feature.value"),
                json_metadata=dict(metadata or {}),
            )
            .returning(VectorSemanticFeature.id)
        )

        async with self._create_session() as session:
            result = await session.execute(stmt)
            await session.commit()
            feature_id = FeatureIdT(str(result.scalar_one()))

        await self._upsert_vector_record(
            feature_id=feature_id,
            set_id=set_id,
            category_name=category_name,
            feature=feature,
            value=value,
            tag=tag,
            embedding=embedding,
            metadata=metadata,
        )
        return feature_id

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
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        feature_id_int = self._coerce_feature_id(feature_id)

        values: dict[str, Any] = {}
        if set_id is not None:
            values["set_id"] = set_id
        if category_name is not None:
            values["semantic_category_id"] = category_name
        if feature is not None:
            values["feature"] = sanitize_pg_text(feature, context="feature.feature")
        if value is not None:
            values["value"] = sanitize_pg_text(value, context="feature.value")
        if tag is not None:
            values["tag_id"] = sanitize_pg_text(tag, context="feature.tag")
        if metadata is not None:
            values["json_metadata"] = dict(metadata)

        async with self._create_session() as session:
            if values:
                await session.execute(
                    update(VectorSemanticFeature)
                    .where(VectorSemanticFeature.id == feature_id_int)
                    .values(**values)
                )
                await session.commit()

            row = await session.get(VectorSemanticFeature, feature_id_int)

        if row is None:
            raise ResourceNotFoundError(f"Feature ID not found: {feature_id}")

        if values or embedding is not None:
            existing_record = await self._get_existing_vector_record(feature_id)
            await self._upsert_vector_record(
                feature_id=feature_id,
                set_id=row.set_id,
                category_name=row.semantic_category_id,
                feature=row.feature,
                value=row.value,
                tag=row.tag_id,
                embedding=(
                    embedding
                    if embedding is not None
                    else np.array(existing_record.vector, dtype=float)
                ),
                metadata=row.json_metadata,
            )

    async def get_feature(
        self,
        feature_id: FeatureIdT,
        load_citations: bool = False,
    ) -> SemanticFeature | None:
        feature_id_int = self._coerce_feature_id(feature_id)
        async with self._create_session() as session:
            feature = await session.get(VectorSemanticFeature, feature_id_int)
            citations_map: Mapping[int, Sequence[EpisodeIdT]] = {}
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
        filter_expr: FilterExpr | None = None,
        page_size: int | None = None,
        page_num: int | None = None,
        vector_search_opts: SemanticStorage.VectorSearchOpts | None = None,
        tag_threshold: int | None = None,
        load_citations: bool = False,
    ) -> AsyncIterator[SemanticFeature]:
        if page_num is not None and page_size is None:
            raise InvalidArgumentError("Cannot specify offset without limit")

        if vector_search_opts is not None:
            features = await self._vector_search_features(
                filter_expr=filter_expr,
                page_size=page_size,
                page_num=page_num,
                vector_search_opts=vector_search_opts,
                load_citations=load_citations,
            )
        else:
            features = await self._query_relational_features(
                filter_expr=filter_expr,
                page_size=page_size,
                page_num=page_num,
                load_citations=load_citations,
            )

        if tag_threshold is not None and tag_threshold > 0 and features:
            from collections import Counter

            counts = Counter(feature.tag for feature in features)
            features = [
                feature for feature in features if counts[feature.tag] >= tag_threshold
            ]

        for feature in features:
            yield feature

    async def delete_features(self, feature_ids: Sequence[FeatureIdT]) -> None:
        try:
            feature_id_ints = TypeAdapter(list[int]).validate_python(feature_ids)
        except ValidationError as e:
            raise ResourceNotFoundError(f"Invalid feature IDs: {feature_ids}") from e

        async with self._create_session() as session:
            await session.execute(
                delete(VectorSemanticFeature).where(
                    VectorSemanticFeature.id.in_(feature_id_ints)
                )
            )
            await session.commit()

        await self._delete_vector_records(
            [FeatureIdT(str(fid)) for fid in feature_id_ints]
        )

    async def delete_feature_set(
        self,
        *,
        filter_expr: FilterExpr | None = None,
    ) -> None:
        feature_ids = await self._feature_ids_for_filter(filter_expr)
        stmt = delete(VectorSemanticFeature)
        stmt = self._apply_feature_filter(stmt, filter_expr=filter_expr)
        async with self._create_session() as session:
            await session.execute(stmt)
            await session.commit()
        await self._delete_vector_records(feature_ids)

    async def add_citations(
        self,
        feature_id: FeatureIdT,
        history_ids: Sequence[EpisodeIdT],
    ) -> None:
        if not history_ids:
            return
        feature_id_int = self._coerce_feature_id(feature_id)
        rows = [
            {"feature_id": feature_id_int, "history_id": str(history_id)}
            for history_id in history_ids
        ]
        async with self._create_session() as session:
            await session.execute(
                insert(vector_citation_association_table).values(rows)
            )
            await session.commit()

    async def get_history_messages(
        self,
        *,
        set_ids: Sequence[SetIdT] | None = None,
        limit: int | None = None,
        is_ingested: bool | None = None,
    ) -> AsyncIterator[EpisodeIdT]:
        stmt = select(VectorSemanticSetIngestedHistory.history_id).order_by(
            VectorSemanticSetIngestedHistory.history_id.asc()
        )
        stmt = self._apply_history_filter(
            stmt,
            set_ids=set_ids,
            is_ingested=is_ingested,
            limit=limit,
        )
        async with self._create_session() as session:
            result = await session.stream(stmt)
            async for history_id in result.scalars():
                yield EpisodeIdT(history_id)

    async def get_history_messages_count(
        self,
        *,
        set_ids: Sequence[SetIdT] | None = None,
        is_ingested: bool | None = None,
    ) -> int:
        stmt = select(func.count(VectorSemanticSetIngestedHistory.history_id))
        stmt = self._apply_history_filter(
            stmt,
            set_ids=set_ids,
            is_ingested=is_ingested,
        )
        async with self._create_session() as session:
            result = await session.execute(stmt)
            return int(result.scalar_one())

    async def add_history_to_set(self, set_id: SetIdT, history_id: EpisodeIdT) -> None:
        stmt = insert(VectorSemanticSetIngestedHistory).values(
            set_id=set_id,
            history_id=history_id,
        )
        async with self._create_session() as session:
            await session.execute(stmt)
            await session.commit()

    async def delete_history(self, history_ids: Sequence[EpisodeIdT]) -> None:
        if not history_ids:
            return
        async with self._create_session() as session:
            await session.execute(
                delete(vector_citation_association_table).where(
                    vector_citation_association_table.c.history_id.in_(history_ids)
                )
            )
            await session.execute(
                delete(VectorSemanticSetIngestedHistory).where(
                    VectorSemanticSetIngestedHistory.history_id.in_(history_ids)
                )
            )
            await session.commit()

    async def delete_history_set(self, set_ids: Sequence[SetIdT]) -> None:
        if not set_ids:
            return
        async with self._create_session() as session:
            await session.execute(
                delete(VectorSemanticSetIngestedHistory).where(
                    VectorSemanticSetIngestedHistory.set_id.in_(set_ids)
                )
            )
            await session.commit()

    async def mark_messages_ingested(
        self,
        *,
        set_id: SetIdT,
        history_ids: Sequence[EpisodeIdT],
    ) -> None:
        if len(history_ids) == 0:
            raise ValueError("No ids provided")
        stmt = (
            update(VectorSemanticSetIngestedHistory)
            .where(VectorSemanticSetIngestedHistory.set_id == set_id)
            .where(VectorSemanticSetIngestedHistory.history_id.in_(history_ids))
            .values(ingested=True)
        )
        async with self._create_session() as session:
            await session.execute(stmt)
            await session.commit()

    def get_history_set_ids(
        self,
        *,
        min_uningested_messages: int | None = None,
        older_than: AwareDatetime | None = None,
    ) -> AsyncIterator[SetIdT]:
        async def _iter() -> AsyncIterator[SetIdT]:
            subqueries: list[Select] = []
            if min_uningested_messages is not None and min_uningested_messages > 0:
                subqueries.append(
                    select(VectorSemanticSetIngestedHistory.set_id)
                    .where(VectorSemanticSetIngestedHistory.ingested.is_(False))
                    .group_by(VectorSemanticSetIngestedHistory.set_id)
                    .having(func.count() >= min_uningested_messages)
                )
            if older_than is not None:
                subqueries.append(
                    select(VectorSemanticSetIngestedHistory.set_id)
                    .where(
                        VectorSemanticSetIngestedHistory.ingested.is_(False),
                        VectorSemanticSetIngestedHistory.created_at <= older_than,
                    )
                    .distinct()
                )
            if not subqueries:
                stmt = select(VectorSemanticSetIngestedHistory.set_id).distinct()
            elif len(subqueries) == 1:
                stmt = subqueries[0]
            else:
                stmt = union(*subqueries)

            async with self._create_session() as session:
                result = await session.stream(stmt)
                async for set_id in result.scalars():
                    if set_id is not None:
                        yield SetIdT(set_id)

        return _iter()

    async def purge_ingested_rows(self, set_ids: list[SetIdT]) -> int:
        if not set_ids:
            return 0
        pending_alias = aliased(VectorSemanticSetIngestedHistory)
        pending_exists = (
            select(pending_alias.set_id)
            .where(
                pending_alias.set_id == VectorSemanticSetIngestedHistory.set_id,
                pending_alias.ingested.is_(False),
            )
            .correlate(VectorSemanticSetIngestedHistory)
            .exists()
        )
        stmt = delete(VectorSemanticSetIngestedHistory).where(
            VectorSemanticSetIngestedHistory.set_id.in_(set_ids),
            VectorSemanticSetIngestedHistory.ingested.is_(True),
            ~pending_exists,
        )
        async with self._create_session() as session:
            result = cast(CursorResult[Any], await session.execute(stmt))
            await session.commit()
            return result.rowcount

    async def get_set_ids_starts_with(self, prefix: str) -> AsyncIterator[SetIdT]:
        stmt = union(
            select(VectorSemanticSetIngestedHistory.set_id).where(
                VectorSemanticSetIngestedHistory.set_id.startswith(prefix)
            ),
            select(VectorSemanticFeature.set_id).where(
                VectorSemanticFeature.set_id.startswith(prefix)
            ),
        )
        async with self._create_session() as session:
            result = await session.stream(stmt)
            async for set_id in result.scalars():
                yield SetIdT(set_id)

    async def _upsert_vector_record(
        self,
        *,
        feature_id: FeatureIdT,
        set_id: SetIdT,
        category_name: str,
        feature: str,
        value: str,
        tag: str,
        embedding: InstanceOf[np.ndarray],
        metadata: Mapping[str, Any] | None,
    ) -> None:
        properties = self._vector_properties(
            feature_id=feature_id,
            set_id=set_id,
            category_name=category_name,
            feature=feature,
            value=value,
            tag=tag,
            metadata=metadata,
        )
        await self._vector_collection.upsert(
            records=[
                Record(
                    uuid=feature_vector_uuid(feature_id),
                    vector=[float(item) for item in embedding.tolist()],
                    properties=properties,
                )
            ]
        )

    async def _get_existing_vector_record(self, feature_id: FeatureIdT) -> Record:
        records = await self._vector_collection.get(
            record_uuids=[feature_vector_uuid(feature_id)],
            return_vector=True,
            return_properties=False,
        )
        if not records or records[0].vector is None:
            raise ResourceNotFoundError(f"Vector record not found: {feature_id}")
        return records[0]

    async def _delete_vector_records(self, feature_ids: Sequence[FeatureIdT]) -> None:
        if not feature_ids:
            return
        await self._vector_collection.delete(
            record_uuids=[feature_vector_uuid(feature_id) for feature_id in feature_ids]
        )

    async def _vector_search_features(
        self,
        *,
        filter_expr: FilterExpr | None,
        page_size: int | None,
        page_num: int | None,
        vector_search_opts: SemanticStorage.VectorSearchOpts,
        load_citations: bool,
    ) -> list[SemanticFeature]:
        offset = (page_num or 0) * (page_size or 0)
        limit = max(
            _DEFAULT_VECTOR_QUERY_LIMIT,
            offset + page_size if page_size is not None else 0,
        )
        [query_result] = await self._vector_collection.query(
            query_vectors=[vector_search_opts.query_embedding.tolist()],
            limit=limit,
            score_threshold=vector_search_opts.min_distance,
            return_vector=False,
            return_properties=True,
        )
        ordered_ids = [
            FeatureIdT(str((match.record.properties or {})["feature_id"]))
            for match in query_result.matches
            if match.record.properties and "feature_id" in match.record.properties
        ]
        features = await self._features_by_ids(
            ordered_ids,
            filter_expr=filter_expr,
            load_citations=load_citations,
        )
        if page_size is not None:
            features = features[offset : offset + page_size]
        return features

    async def _query_relational_features(
        self,
        *,
        filter_expr: FilterExpr | None,
        page_size: int | None,
        page_num: int | None,
        load_citations: bool,
    ) -> list[SemanticFeature]:
        stmt = select(VectorSemanticFeature).order_by(
            VectorSemanticFeature.created_at.asc(),
            VectorSemanticFeature.id.asc(),
        )
        stmt = cast(
            Select[Any],
            self._apply_feature_filter(stmt, filter_expr=filter_expr),
        )
        if page_size is not None:
            stmt = stmt.limit(page_size).offset(page_size * (page_num or 0))
        async with self._create_session() as session:
            result = await session.execute(stmt)
            rows = list(result.scalars())
            citations_map: Mapping[int, Sequence[EpisodeIdT]] = {}
            if load_citations and rows:
                citations_map = await self._load_feature_citations(
                    session,
                    [row.id for row in rows],
                )
        return [row.to_typed_model(citations=citations_map.get(row.id)) for row in rows]

    async def _features_by_ids(
        self,
        feature_ids: Sequence[FeatureIdT],
        *,
        filter_expr: FilterExpr | None,
        load_citations: bool,
    ) -> list[SemanticFeature]:
        if not feature_ids:
            return []
        int_ids = [self._coerce_feature_id(feature_id) for feature_id in feature_ids]
        stmt = select(VectorSemanticFeature).where(
            VectorSemanticFeature.id.in_(int_ids)
        )
        stmt = self._apply_feature_filter(stmt, filter_expr=filter_expr)
        async with self._create_session() as session:
            result = await session.execute(stmt)
            rows_by_id = {row.id: row for row in result.scalars()}
            ordered_rows = [
                rows_by_id[row_id] for row_id in int_ids if row_id in rows_by_id
            ]
            citations_map: Mapping[int, Sequence[EpisodeIdT]] = {}
            if load_citations and ordered_rows:
                citations_map = await self._load_feature_citations(
                    session,
                    [row.id for row in ordered_rows],
                )
        return [
            row.to_typed_model(citations=citations_map.get(row.id))
            for row in ordered_rows
        ]

    async def _feature_ids_for_filter(
        self,
        filter_expr: FilterExpr | None,
    ) -> list[FeatureIdT]:
        stmt = select(VectorSemanticFeature.id)
        stmt = self._apply_feature_filter(stmt, filter_expr=filter_expr)
        async with self._create_session() as session:
            result = await session.execute(stmt)
            return [FeatureIdT(str(feature_id)) for feature_id in result.scalars()]

    def _apply_history_filter(
        self,
        stmt: Select,
        *,
        set_ids: Sequence[str] | None = None,
        is_ingested: bool | None = None,
        limit: int | None = None,
    ) -> Select:
        if set_ids is not None and len(set_ids) > 0:
            stmt = stmt.where(VectorSemanticSetIngestedHistory.set_id.in_(set_ids))
        if is_ingested is not None:
            stmt = stmt.where(VectorSemanticSetIngestedHistory.ingested == is_ingested)
        if limit is not None:
            stmt = stmt.limit(limit)
        return stmt

    def _apply_feature_filter(
        self,
        stmt: Select[Any] | Delete,
        *,
        filter_expr: FilterExpr | None = None,
    ) -> Select[Any] | Delete:
        if filter_expr is None:
            return stmt
        clause = compile_sql_filter(filter_expr, self._resolve_feature_field_default)
        return stmt.where(clause)

    def _resolve_feature_field_default(
        self,
        field: str,
    ) -> tuple[ColumnElement, FieldEncoding]:
        return self._resolve_feature_field(VectorSemanticFeature, field)

    @staticmethod
    def _resolve_feature_field(
        table: type[VectorSemanticFeature],
        field: str,
    ) -> tuple[ColumnElement, FieldEncoding]:
        internal_name, is_user_property = normalize_filter_field(field)
        if is_user_property:
            key = demangle_user_metadata_key(internal_name)
            return table.json_metadata[key], "json"

        field_mapping: dict[str, ColumnElement] = {
            "set_id": table.set_id.expression,
            "set": table.set_id.expression,
            "semantic_category_id": table.semantic_category_id.expression,
            "category_name": table.semantic_category_id.expression,
            "category": table.semantic_category_id.expression,
            "tag_id": table.tag_id.expression,
            "tag": table.tag_id.expression,
            "feature": table.feature.expression,
            "feature_name": table.feature.expression,
            "value": table.value.expression,
            "created_at": table.created_at.expression,
            "updated_at": table.updated_at.expression,
        }
        if internal_name in field_mapping:
            return field_mapping[internal_name], "column"
        raise ValueError(f"Unknown filter field: {field!r}")

    async def _load_feature_citations(
        self,
        session: AsyncSession,
        feature_ids: Sequence[int],
    ) -> Mapping[int, Sequence[EpisodeIdT]]:
        if not feature_ids:
            return {}
        stmt = select(
            vector_citation_association_table.c.feature_id,
            vector_citation_association_table.c.history_id,
        ).where(vector_citation_association_table.c.feature_id.in_(feature_ids))
        result = await session.execute(stmt)
        citations: MutableMapping[int, list[EpisodeIdT]] = {
            feature_id: [] for feature_id in feature_ids
        }
        for feature_id, history_id in result:
            citations.setdefault(feature_id, []).append(EpisodeIdT(history_id))
        return citations

    @staticmethod
    def _vector_properties(
        *,
        feature_id: FeatureIdT,
        set_id: SetIdT,
        category_name: str,
        feature: str,
        value: str,
        tag: str,
        metadata: Mapping[str, Any] | None,
    ) -> dict[str, str | int | float | bool | datetime]:
        properties: dict[str, str | int | float | bool | datetime] = {
            "feature_id": feature_id,
            "set_id": set_id,
            "set": set_id,
            "semantic_category_id": category_name,
            "category_name": category_name,
            "category": category_name,
            "tag_id": tag,
            "tag": tag,
            "feature": feature,
            "feature_name": feature,
            "value": value,
        }
        properties.update(
            {
                key: item
                for key, item in (metadata or {}).items()
                if isinstance(item, bool | int | float | str | datetime)
            }
        )
        return properties

    @staticmethod
    def _coerce_feature_id(feature_id: FeatureIdT) -> int:
        try:
            return int(feature_id)
        except (TypeError, ValueError) as e:
            raise ResourceNotFoundError(f"Invalid feature ID: {feature_id}") from e
