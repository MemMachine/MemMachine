from typing import Any, Optional, Set

import numpy as np
from pgvector.sqlalchemy import Vector
from pydantic import AwareDatetime, InstanceOf, validate_call
from sqlalchemy import (
    TIMESTAMP,
    Boolean,
    Column,
    Engine,
    ForeignKey,
    Index,
    Integer,
    String,
    Table,
    delete,
    insert,
    select,
    update,
)
from sqlalchemy.orm import (
    DeclarativeBase,
    Mapped,
    joinedload,
    mapped_column,
    relationship,
    sessionmaker,
)
from sqlalchemy.sql import func

from memmachine.semantic_memory.semantic_model import HistoryMessage, SemanticFeature
from memmachine.semantic_memory.storage.storage_base import SemanticStorageBase


class BaseSemanticStorage(DeclarativeBase):
    pass


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
        Integer,
        ForeignKey("history.id", ondelete="CASCADE", onupdate="CASCADE"),
        primary_key=True,
    ),
)


class Feature(BaseSemanticStorage):
    __tablename__ = "feature"
    id = mapped_column(Integer, primary_key=True)

    # Feature data
    set_id = mapped_column(String)
    semantic_type_id = mapped_column(String)
    tag_id = mapped_column(String)
    feature = mapped_column(String)
    value = mapped_column(String)

    # metadata
    created_a = mapped_column(
        TIMESTAMP,
        server_default=func.now(),
    )
    updated_at = mapped_column(
        TIMESTAMP,
        server_default=func.now(),
        onupdate=func.now(),
    )
    embedding = mapped_column(Vector)
    json_metadata = mapped_column(String, name="metadata", server_default="{}")

    citations: Mapped[Set["History"]] = relationship(
        secondary=citation_association_table,
    )

    __table_args__ = (
        Index("idx_feature_set_id", "set_id"),
        Index("idx_feature_set_id_semantic_type", "set_id", "semantic_type_id"),
        Index(
            "idx_feature_set_semantic_type_tag", "set_id", "semantic_type_id", "tag_id"
        ),
        Index(
            "idx_feature_set_semantic_type_tag_feature",
            "set_id",
            "semantic_type_id",
            "tag_id",
            "feature",
        ),
    )

    def to_typed_model(self, with_citations: bool = False) -> SemanticFeature:
        if with_citations:
            citations = [c.to_typed_model() for c in self.citations]
        else:
            citations = None

        return SemanticFeature(
            metadata=SemanticFeature.Metadata(
                id=self.id,
                citations=citations,
            ),
            set_id=self.set_id,
            type=self.semantic_type_id,
            tag=self.tag_id,
            feature=self.feature,
            value=self.value,
        )


class History(BaseSemanticStorage):
    __tablename__ = "history"
    id = mapped_column(Integer, primary_key=True)

    content = mapped_column(String)

    json_metadata = mapped_column(String, name="metadata", server_default="{}")
    created_at = mapped_column(
        TIMESTAMP,
        server_default=func.now(),
    )

    def to_typed_model(self) -> HistoryMessage:
        return HistoryMessage(
            metadata=HistoryMessage.Metadata(
                id=self.id,
            ),
            content=self.content,
            created_at=self.created_at,
        )


class SetIngestedHistory(BaseSemanticStorage):
    __tablename__ = "set_ingested_history"
    set_id = mapped_column(String, primary_key=True)
    history_id = mapped_column(
        Integer,
        ForeignKey("history.id", ondelete="CASCADE", onupdate="CASCADE"),
        primary_key=True,
    )
    ingested = mapped_column(Boolean, default=False)


class SqlAlchemyPgVectorSemanticStorage(SemanticStorageBase):
    def __init__(self, sqlalchemy_engine: Engine):
        self._engine = sqlalchemy_engine

    def _create_session(self):
        return sessionmaker(bind=self._engine)()

    def _initialize_db(self):
        with self._engine.begin() as conn:
            conn.exec_driver_sql("CREATE EXTENSION IF NOT EXISTS vector")

        BaseSemanticStorage.metadata.create_all(self._engine)

    async def startup(self):
        self._initialize_db()

    async def cleanup(self):
        pass

    @validate_call
    async def delete_all(self):
        with self._create_session() as session:
            session.query(citation_association_table).delete()
            session.query(SetIngestedHistory).delete()
            session.query(Feature).delete()
            session.query(History).delete()
            session.commit()

    @validate_call
    async def add_feature(
        self,
        *,
        set_id: str,
        type_name: str,
        feature: str,
        value: str,
        tag: str,
        embedding: InstanceOf[np.ndarray],
        metadata: dict[str, Any] | None = None,
    ) -> int:
        stmt = (
            insert(Feature)
            .values(
                set_id=set_id,
                semantic_type_id=type_name,
                tag_id=tag,
                feature=feature,
                value=value,
                embedding=embedding,
                json_metadata=metadata,
            )
            .returning(Feature.id)
        )

        with self._create_session() as session:
            res = session.execute(stmt).scalar_one()
            session.commit()

        return res

    async def update_feature(
        self,
        feature_id: int,
        *,
        set_id: Optional[str] = None,
        type_name: Optional[str] = None,
        feature: Optional[str] = None,
        value: Optional[str] = None,
        tag: Optional[str] = None,
        embedding: Optional[InstanceOf[np.ndarray]] = None,
        metadata: dict[str, Any] | None = None,
    ):
        stmt = update(Feature).where(Feature.id == feature_id)

        if set_id is not None:
            stmt = stmt.values(set_id=set_id)
        if type_name is not None:
            stmt = stmt.values(semantic_type_id=type_name)
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

        with self._create_session() as session:
            session.execute(stmt)
            session.commit()

    @validate_call
    async def get_feature(
        self,
        feature_id: int,
        load_citations: bool = False,
    ) -> SemanticFeature | None:
        stmt = select(Feature).where(Feature.id == feature_id)

        if load_citations:
            stmt.options(joinedload(Feature.citations))

        with self._create_session() as session:
            feature = session.execute(stmt).scalar_one_or_none()

            return (
                feature.to_typed_model(with_citations=load_citations)
                if feature
                else None
            )

    @validate_call
    async def get_feature_set(
        self,
        *,
        set_ids: Optional[list[str]] = None,
        type_names: Optional[list[str]] = None,
        feature_names: Optional[list[str]] = None,
        tags: Optional[list[str]] = None,
        k: Optional[int] = None,
        vector_search_opts: Optional[SemanticStorageBase.VectorSearchOpts] = None,
        tag_threshold: Optional[int] = None,
        load_citations: bool = False,
    ) -> list[SemanticFeature]:
        stmt = select(Feature)

        stmt = self._apply_feature_filter(
            stmt,
            set_ids=set_ids,
            type_names=type_names,
            tags=tags,
            feature_names=feature_names,
            thresh=tag_threshold,
            k=k,
            vector_search_opts=vector_search_opts,
        )

        if load_citations:
            stmt.options(joinedload(Feature.citations))

        with self._create_session() as session:
            features = session.execute(stmt).scalars().all()

            return [f.to_typed_model(with_citations=load_citations) for f in features]

    @validate_call
    async def delete_features(self, feature_ids: list[int]):
        stmt = delete(Feature).where(Feature.id.in_(feature_ids))
        with self._create_session() as session:
            session.execute(stmt)
            session.commit()

    @validate_call
    async def delete_feature_set(
        self,
        *,
        set_ids: Optional[list[str]] = None,
        type_names: Optional[list[str]] = None,
        feature_names: Optional[list[str]] = None,
        tags: Optional[list[str]] = None,
        thresh: Optional[int] = None,
        k: Optional[int] = None,
        vector_search_opts: Optional[SemanticStorageBase.VectorSearchOpts] = None,
    ):
        stmt = delete(Feature)

        stmt = self._apply_feature_filter(
            stmt,
            set_ids=set_ids,
            type_names=type_names,
            tags=tags,
            feature_names=feature_names,
            thresh=thresh,
            k=k,
            vector_search_opts=vector_search_opts,
        )

        with self._create_session() as session:
            session.execute(stmt)
            session.commit()

    async def add_citations(self, feature_id: int, history_ids: list[int]):
        rows = [{"feature_id": feature_id, "history_id": hid} for hid in history_ids]

        stmt = insert(citation_association_table).values(rows)

        with self._create_session() as session:
            session.execute(stmt)
            session.commit()

    @validate_call
    async def add_history(
        self,
        content: str,
        metadata: dict[str, str] | None = None,
        created_at: Optional[AwareDatetime] = None,
    ) -> int:
        stmt = (
            insert(History)
            .values(
                content=content,
            )
            .returning(History.id)
        )

        if metadata is not None:
            # TODO: Add metadata to History
            pass

        if created_at is not None:
            stmt = stmt.values(created_at=created_at)

        with self._create_session() as session:
            res = session.execute(stmt).scalar_one()
            session.commit()

        return res

    @validate_call
    async def get_history(self, history_id: int) -> Optional[HistoryMessage]:
        stmt = select(History).where(History.id == history_id)

        with self._create_session() as session:
            history = session.execute(stmt).scalar_one_or_none()

            return history.to_typed_model() if history else None

    @validate_call
    async def delete_history(self, history_ids: list[int]):
        stmt = delete(History).where(History.id.in_(history_ids))

        with self._create_session() as session:
            session.execute(stmt)
            session.commit()

    @validate_call
    async def delete_history_messages(
        self,
        *,
        start_time: Optional[AwareDatetime] = None,
        end_time: Optional[AwareDatetime] = None,
    ):
        stmt = delete(History)

        stmt = self._apply_history_filter(
            stmt,
            start_time=start_time,
            end_time=end_time,
        )

        with self._create_session() as session:
            session.execute(stmt)
            session.commit()

    @validate_call
    async def get_history_messages(
        self,
        *,
        set_id: Optional[str] = None,
        k: Optional[int] = None,
        start_time: Optional[AwareDatetime] = None,
        end_time: Optional[AwareDatetime] = None,
        is_ingested: Optional[bool] = None,
    ) -> list[HistoryMessage]:
        stmt = select(History)

        # Order oldest to newest [ first, second, third ]
        stmt = stmt.order_by(History.created_at.asc())

        stmt = self._apply_history_filter(
            stmt,
            set_id=set_id,
            start_time=start_time,
            end_time=end_time,
            k=k,
            is_ingested=is_ingested,
        )

        with self._create_session() as session:
            history_messages = session.execute(stmt).scalars().all()

        return [h.to_typed_model() for h in history_messages]

    @validate_call
    async def get_history_messages_count(
        self,
        *,
        set_id: Optional[str] = None,
        k: Optional[int] = None,
        start_time: Optional[AwareDatetime] = None,
        end_time: Optional[AwareDatetime] = None,
        is_ingested: Optional[bool] = None,
    ) -> int:
        stmt = select(func.count(History.id))

        stmt = self._apply_history_filter(
            stmt,
            set_id=set_id,
            start_time=start_time,
            end_time=end_time,
            k=k,
            is_ingested=is_ingested,
        )

        with self._create_session() as session:
            return session.execute(stmt).scalar_one()

    @validate_call
    async def mark_messages_ingested(self, set_id: str, ids: list[int]) -> None:
        if len(ids) == 0:
            raise ValueError("No ids provided")

        stmt = (
            update(SetIngestedHistory)
            .where(SetIngestedHistory.set_id == set_id)
            .where(SetIngestedHistory.history_id.in_(ids))
            .values(ingested=True)
        )

        with self._create_session() as session:
            session.execute(stmt)
            session.commit()

    @validate_call
    async def add_history_to_set(
        self,
        set_id: str,
        history_id: int,
    ) -> None:
        stmt = insert(SetIngestedHistory).values(set_id=set_id, history_id=history_id)

        with self._create_session() as session:
            session.execute(stmt)
            session.commit()

    @staticmethod
    def _apply_history_filter(
        stmt,
        *,
        set_id: Optional[str] = None,
        k: Optional[int] = None,
        start_time: Optional[AwareDatetime] = None,
        end_time: Optional[AwareDatetime] = None,
        is_ingested: Optional[bool] = None,
    ):
        if start_time is not None:
            stmt = stmt.where(History.created_at >= start_time)
        if end_time is not None:
            stmt = stmt.where(History.created_at <= end_time)
        if k is not None:
            stmt = stmt.limit(k)

        if set_id is not None or is_ingested is not None:
            stmt = stmt.join(
                SetIngestedHistory, History.id == SetIngestedHistory.history_id
            )
            if set_id is not None:
                stmt = stmt.where(SetIngestedHistory.set_id == set_id)
            if is_ingested is not None:
                stmt = stmt.where(SetIngestedHistory.ingested == is_ingested)

        return stmt

    def _apply_feature_filter(
        self,
        stmt,
        *,
        set_ids: Optional[list[str]] = None,
        type_names: Optional[list[str]] = None,
        feature_names: Optional[list[str]] = None,
        tags: Optional[list[str]] = None,
        thresh: Optional[int] = None,
        k: Optional[int] = None,
        vector_search_opts: Optional[SemanticStorageBase.VectorSearchOpts] = None,
    ):
        def _apply_feature_id_filter(
            _stmt,
        ):
            if set_ids is not None and len(set_ids) > 0:
                _stmt = _stmt.where(Feature.set_id.in_(set_ids))

            if type_names is not None and len(type_names) > 0:
                _stmt = _stmt.where(Feature.semantic_type_id.in_(type_names))

            if tags is not None and len(tags) > 0:
                _stmt = _stmt.where(Feature.tag_id.in_(tags))

            if feature_names is not None and len(feature_names) > 0:
                _stmt = _stmt.where(Feature.feature.in_(feature_names))

            if k is not None:
                _stmt = _stmt.limit(k)

            if vector_search_opts is not None:
                if vector_search_opts.min_cos is not None:
                    threshold = 1 - vector_search_opts.min_cos
                    _stmt = _stmt.where(
                        Feature.embedding.cosine_distance(
                            vector_search_opts.query_embedding
                        )
                        <= threshold
                    )

                _stmt = _stmt.order_by(
                    Feature.embedding.cosine_distance(
                        vector_search_opts.query_embedding
                    ).asc()
                )

            return _stmt

        stmt = _apply_feature_id_filter(stmt)

        if thresh is not None:
            subquery = self._get_tags_with_more_than_k_features(thresh)
            subquery = _apply_feature_id_filter(subquery)

            stmt = stmt.where(Feature.tag_id.in_(subquery))

        return stmt

    @staticmethod
    def _get_tags_with_more_than_k_features(k: int):
        return select(Feature.tag_id).group_by(Feature.tag_id).having(func.count() >= k)
