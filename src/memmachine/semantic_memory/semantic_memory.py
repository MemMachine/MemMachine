"""
Core module for the Semantic Memory engine.

This module contains the `SemanticMemoryManager` class, which is the central component
for creating, managing, and searching feature sets based on their
conversation history. It integrates with language models for intelligent
information extraction and a vector database for semantic search capabilities.
"""

import asyncio
import logging
from asyncio import Task
from typing import Any

import numpy as np
from pydantic import BaseModel, InstanceOf

from memmachine.common.episode_store import EpisodeIdT, EpisodeStorage
from memmachine.common.filter.filter_parser import FilterExpr

from .semantic_ingestion import IngestionService
from .semantic_model import FeatureIdT, ResourceRetriever, SemanticFeature, SetIdT
from .storage.storage_base import SemanticStorage

logger = logging.getLogger(__name__)


def _consolidate_errors_and_raise(possible_errors: list[Any], msg: str) -> None:
    errors = [r for r in possible_errors if isinstance(r, Exception)]
    if len(errors) > 0:
        raise ExceptionGroup(msg, errors)


class SemanticService:
    """High-level coordinator for ingesting history and serving semantic features."""

    class Params(BaseModel):
        """Infrastructure dependencies and background-update configuration."""

        semantic_storage: InstanceOf[SemanticStorage]
        episode_storage: InstanceOf[EpisodeStorage]
        consolidation_threshold: int = 20

        feature_update_interval_sec: float = 2.0

        feature_update_message_limit: int = 5

        resource_retriever: InstanceOf[ResourceRetriever]

        debug_fail_loudly: bool = False

    def __init__(
        self,
        params: Params,
    ) -> None:
        """Set up semantic memory services and background ingestion tracking."""
        self._semantic_storage = params.semantic_storage
        self._episode_storage = params.episode_storage
        self._background_ingestion_interval_sec = params.feature_update_interval_sec

        self._resource_retriever: ResourceRetriever = params.resource_retriever

        self._consolidation_threshold = params.consolidation_threshold

        self._feature_update_message_limit = params.feature_update_message_limit

        self._ingestion_task: Task | None = None
        self._is_shutting_down = False
        self._debug_fail_loudly = params.debug_fail_loudly

    async def start(self) -> None:
        if self._ingestion_task is not None:
            return

        self._is_shutting_down = False
        self._ingestion_task = asyncio.create_task(self._background_ingestion_task())

    async def stop(self) -> None:
        if self._ingestion_task is None:
            return

        self._is_shutting_down = True
        await self._ingestion_task

    async def search(
        self,
        set_ids: list[SetIdT],
        query: str,
        *,
        min_distance: float | None = None,
        limit: int | None = 30,
        load_citations: bool = False,
        filter_expr: FilterExpr | None = None,
    ) -> list[SemanticFeature]:
        resources = self._resource_retriever.get_resources(set_ids[0])
        query_embedding = (await resources.embedder.search_embed([query]))[0]

        return await self._semantic_storage.get_feature_set(
            filter_expr=filter_expr,
            page_size=limit,
            vector_search_opts=SemanticStorage.VectorSearchOpts(
                query_embedding=np.array(query_embedding),
                min_distance=min_distance,
            ),
            load_citations=load_citations,
        )

    async def add_messages(self, set_id: SetIdT, history_ids: list[EpisodeIdT]) -> None:
        res = await asyncio.gather(
            *[
                self._semantic_storage.add_history_to_set(
                    set_id=set_id,
                    history_id=h_id,
                )
                for h_id in history_ids
            ],
            return_exceptions=True,
        )

        _consolidate_errors_and_raise(res, "Failed to add messages to set")

    async def add_message_to_sets(
        self,
        history_id: EpisodeIdT,
        set_ids: list[SetIdT],
    ) -> None:
        res = await asyncio.gather(
            *[
                self._semantic_storage.add_history_to_set(
                    set_id=set_id,
                    history_id=history_id,
                )
                for set_id in set_ids
            ],
            return_exceptions=True,
        )

        _consolidate_errors_and_raise(res, "Failed to add message to sets")

    async def delete_messages(self, *, set_ids: list[SetIdT]) -> None:
        await self._semantic_storage.delete_history_set(set_ids=set_ids)

    async def number_of_uningested(self, set_ids: list[SetIdT]) -> int:
        return await self._semantic_storage.get_history_messages_count(
            set_ids=set_ids,
            is_ingested=False,
        )

    async def add_new_feature(
        self,
        *,
        set_id: SetIdT,
        category_name: str,
        feature: str,
        value: str,
        tag: str,
        metadata: dict[str, str] | None = None,
        citations: list[EpisodeIdT] | None = None,
    ) -> FeatureIdT:
        resources = self._resource_retriever.get_resources(set_id)
        embedding = (await resources.embedder.ingest_embed([value]))[0]

        f_id = await self._semantic_storage.add_feature(
            set_id=set_id,
            category_name=category_name,
            feature=feature,
            value=value,
            tag=tag,
            metadata=metadata,
            embedding=np.array(embedding),
        )

        if citations is not None:
            await self._semantic_storage.add_citations(f_id, citations)

        return f_id

    async def get_feature(
        self,
        feature_id: FeatureIdT,
        load_citations: bool,
    ) -> SemanticFeature | None:
        return await self._semantic_storage.get_feature(
            feature_id,
            load_citations=load_citations,
        )

    async def get_set_features(
        self,
        *,
        filter_expr: FilterExpr | None = None,
        page_size: int | None = None,
        page_num: int | None = None,
        with_citations: bool = False,
    ) -> list[SemanticFeature]:
        return await self._semantic_storage.get_feature_set(
            filter_expr=filter_expr,
            page_size=page_size,
            page_num=page_num,
            load_citations=with_citations,
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
        metadata: dict[str, str] | None = None,
    ) -> None:
        if value is not None:
            if set_id is None:
                original_feature = await self._semantic_storage.get_feature(feature_id)
                if original_feature is None or original_feature.set_id is None:
                    raise ValueError(
                        "Unable to deduce set_id, the feature_id may be incorrect. "
                        "set_id is required to update a feature",
                    )
                set_id = original_feature.set_id

            resources = self._resource_retriever.get_resources(set_id)

            embedding = (await resources.embedder.ingest_embed([value]))[0]
        else:
            embedding = None

        await self._semantic_storage.update_feature(
            feature_id=feature_id,
            set_id=set_id,
            category_name=category_name,
            feature=feature,
            value=value,
            tag=tag,
            metadata=metadata,
            embedding=np.array(embedding),
        )

    async def delete_history(self, history_ids: list[EpisodeIdT]) -> None:
        await self._semantic_storage.delete_history(history_ids)

    async def delete_features(self, feature_ids: list[FeatureIdT]) -> None:
        await self._semantic_storage.delete_features(feature_ids)

    async def delete_feature_set(
        self,
        *,
        filter_expr: FilterExpr | None = None,
    ) -> None:
        await self._semantic_storage.delete_feature_set(
            filter_expr=filter_expr,
        )

    async def _background_ingestion_task(self) -> None:
        ingestion_service = IngestionService(
            params=IngestionService.Params(
                semantic_storage=self._semantic_storage,
                resource_retriever=self._resource_retriever,
                history_store=self._episode_storage,
            ),
        )

        while not self._is_shutting_down:
            dirty_sets = await self._semantic_storage.get_history_set_ids(
                min_uningested_messages=self._feature_update_message_limit,
            )

            if len(dirty_sets) == 0:
                await asyncio.sleep(self._background_ingestion_interval_sec)
                continue

            try:
                await ingestion_service.process_set_ids(dirty_sets)
            except Exception:
                if self._debug_fail_loudly:
                    raise
                logger.exception("background task crashed, restarting")
