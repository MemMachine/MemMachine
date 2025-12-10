from datetime import datetime
from typing import Any
from unittest.mock import AsyncMock

import numpy as np

from memmachine.common.data_types import SimilarityMetric
from memmachine.common.embedder import Embedder
from memmachine.common.episode_store import Episode
from memmachine.common.filter.filter_parser import FilterExpr
from memmachine.semantic_memory.semantic_model import (
    Resources,
    SemanticFeature,
)
from memmachine.semantic_memory.semantic_session_manager import (
    IsolationType,
    SemanticSessionManager,
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

    async def get_feature(self, feature_id: int, load_citations: bool = False):
        raise NotImplementedError

    async def add_feature(
        self,
        *,
        set_id: str,
        category_name: str,
        feature: str,
        value: str,
        tag: str,
        embedding: np.ndarray,
        metadata: dict[str, Any] | None = None,
    ) -> int:
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
        feature_id: int,
        *,
        set_id: str | None = None,
        category_name: str | None = None,
        feature: str | None = None,
        value: str | None = None,
        tag: str | None = None,
        embedding: np.ndarray | None = None,
        metadata: dict[str, Any] | None = None,
    ):
        raise NotImplementedError

    async def delete_features(self, feature_ids: list[int]):
        await self.delete_features_mock(feature_ids)

    async def get_feature_set(
        self,
        *,
        filter_expr: FilterExpr | None = None,
        page_size: int | None = None,
        vector_search_opts: SemanticStorage.VectorSearchOpts | None = None,
        tag_threshold: int | None = None,
        load_citations: bool = False,
    ) -> list[SemanticFeature]:
        return await self.get_feature_set_mock(
            filter_expr=filter_expr,
            k=page_size,
            vector_search_opts=vector_search_opts,
            tag_threshold=tag_threshold,
            load_citations=load_citations,
        )

    async def delete_feature_set(
        self,
        *,
        filter_expr: FilterExpr | None = None,
    ):
        await self.delete_feature_set_mock(
            filter_expr=filter_expr,
        )

    async def add_citations(self, feature_id: int, history_ids: list[int]):
        await self.add_citations_mock(feature_id, history_ids)

    async def add_history(
        self,
        content: str,
        metadata: dict[str, str] | None = None,
        created_at: datetime | None = None,
    ) -> int:
        raise NotImplementedError

    async def get_history(self, history_id: int):
        raise NotImplementedError

    async def delete_history(self, history_ids: list[int]):
        raise NotImplementedError

    async def delete_history_messages(
        self,
        *,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
    ):
        raise NotImplementedError

    async def get_history_messages(
        self,
        *,
        set_ids: list[str] | None = None,
        limit: int | None = None,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
        is_ingested: bool | None = None,
    ) -> list[Episode]:
        return await self.get_history_messages_mock(
            set_ids=set_ids,
            k=limit,
            start_time=start_time,
            end_time=end_time,
            is_ingested=is_ingested,
        )

    async def get_history_messages_count(
        self,
        *,
        set_ids: list[str] | None = None,
        limit: int | None = None,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
        is_ingested: bool | None = None,
    ) -> int:
        raise NotImplementedError

    async def add_history_to_set(self, set_id: str, history_id: int):
        raise NotImplementedError

    async def mark_messages_ingested(
        self,
        *,
        set_id: str,
        history_ids: list[int],
    ) -> None:
        await self.mark_messages_ingested_mock(set_id=set_id, ids=history_ids)


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

    def get_resources(self, set_id: str) -> Resources:
        self.seen_ids.append(set_id)
        return self._resources


class SimpleSessionResourceRetriever:
    """Resolves the `Resources` bundle for a set_id, falling back to isolation defaults."""

    def __init__(
        self,
        default_resources: dict[IsolationType, Resources],
    ) -> None:
        """Initialize the retriever with isolation defaults and a session checker."""
        self._default_resources = default_resources

    def get_resources(self, set_id: str) -> Resources:
        set_id_type = SemanticSessionManager.set_id_isolation_type(set_id)

        return self._default_resources[set_id_type]
