from collections.abc import AsyncIterator, Mapping, Sequence
from datetime import datetime
from typing import Any
from unittest.mock import AsyncMock

import numpy as np
from pydantic import InstanceOf

from memmachine.common.data_types import SimilarityMetric
from memmachine.common.embedder import Embedder
from memmachine.common.episode_store import EpisodeIdT
from memmachine.common.filter.filter_parser import FilterExpr
from memmachine.semantic_memory.semantic_model import (
    FeatureIdT,
    Resources,
    SemanticFeature,
    SetIdT,
)
from memmachine.semantic_memory.storage.storage_base import SemanticStorage


class MockSemanticStorage(SemanticStorage):
    def __init__(self):
        self.get_history_messages_mock = AsyncMock()
        self.get_feature_set_mock = AsyncMock()
        self.add_feature_mock = AsyncMock()
        self.add_citations_mock = AsyncMock()
        self.delete_feature_set_mock = AsyncMock()
        self.mark_messages_ingested_mock = AsyncMock()
        self.delete_features_mock = AsyncMock()

    async def startup(self):
        raise NotImplementedError

    async def cleanup(self):
        raise NotImplementedError

    async def delete_all(self):
        raise NotImplementedError

    async def reset_set_ids(self, set_ids: Sequence[SetIdT]) -> None:
        raise NotImplementedError

    async def get_feature(
        self,
        feature_id: FeatureIdT,
        load_citations: bool = False,
    ) -> SemanticFeature | None:
        raise NotImplementedError

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
        return await self.add_feature_mock(
            set_id=set_id,
            type_name=category_name,
            feature=feature,
            value=value,
            tag=tag,
            embedding=embedding,
            metadata=metadata,
        )

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
        raise NotImplementedError

    async def delete_features(self, feature_ids: Sequence[FeatureIdT]) -> None:
        await self.delete_features_mock(feature_ids)

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
        features = await self.get_feature_set_mock(
            filter_expr=filter_expr,
            page_size=page_size,
            page_num=page_num,
            vector_search_opts=vector_search_opts,
            tag_threshold=tag_threshold,
            load_citations=load_citations,
        )
        for feature in features:
            yield feature

    async def delete_feature_set(
        self,
        *,
        filter_expr: FilterExpr | None = None,
    ) -> None:
        await self.delete_feature_set_mock(
            filter_expr=filter_expr,
        )

    async def add_citations(
        self,
        feature_id: FeatureIdT,
        history_ids: Sequence[EpisodeIdT],
    ) -> None:
        await self.add_citations_mock(feature_id, history_ids)

    async def add_history(
        self,
        content: str,
        metadata: Mapping[str, str] | None = None,
        created_at: datetime | None = None,
    ) -> EpisodeIdT:
        raise NotImplementedError

    async def get_history(self, history_id: EpisodeIdT):
        raise NotImplementedError

    async def delete_history(self, history_ids: Sequence[EpisodeIdT]) -> None:
        raise NotImplementedError

    async def delete_history_messages(
        self,
        *,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
    ) -> None:
        raise NotImplementedError

    async def get_history_messages(
        self,
        *,
        set_ids: Sequence[SetIdT] | None = None,
        limit: int | None = None,
        is_ingested: bool | None = None,
    ) -> AsyncIterator[EpisodeIdT]:
        messages = await self.get_history_messages_mock(
            set_ids=set_ids,
            limit=limit,
            is_ingested=is_ingested,
        )
        for message in messages:
            yield message

    async def get_history_messages_count(
        self,
        *,
        set_ids: Sequence[SetIdT] | None = None,
        is_ingested: bool | None = None,
    ) -> int:
        raise NotImplementedError

    async def add_history_to_set(self, set_id: SetIdT, history_id: EpisodeIdT) -> None:
        raise NotImplementedError

    async def mark_messages_ingested(
        self,
        *,
        set_id: SetIdT,
        history_ids: Sequence[EpisodeIdT],
    ) -> None:
        await self.mark_messages_ingested_mock(set_id=set_id, ids=history_ids)

    async def delete_history_set(self, set_ids: Sequence[SetIdT]) -> None:
        raise NotImplementedError

    def get_history_set_ids(
        self,
        *,
        min_uningested_messages: int | None = None,
        older_than: datetime | None = None,
    ) -> AsyncIterator[SetIdT]:
        raise NotImplementedError

    def get_set_ids_starts_with(self, prefix: str) -> AsyncIterator[SetIdT]:
        raise NotImplementedError


class MockEmbedder(Embedder):
    def __init__(self):
        self.ingest_calls: list[list[str]] = []

    async def ingest_embed(self, inputs: list[Any], max_attempts: int = 1):
        self.ingest_calls.append(list(inputs))
        return [[float(len(value)), float(len(value)) * -1] for value in inputs]

    async def search_embed(self, queries: list[Any], max_attempts: int = 1):
        raise NotImplementedError

    @property
    def model_id(self) -> str:
        return "embedder-double"

    @property
    def dimensions(self) -> int:
        return 2

    @property
    def similarity_metric(self) -> SimilarityMetric:
        return SimilarityMetric.COSINE


class MockResourceRetriever:
    def __init__(self, resources: Resources):
        self._resources = resources
        self.seen_ids: list[str] = []

    async def get_resources(self, set_id: str) -> Resources:
        self.seen_ids.append(set_id)
        return self._resources
