"""Core module for the Semantic Memory engine.

This module contains the `SemanticMemoryManager` class, which is the central component
for creating, managing, and searching feature sets based on their
conversation history. It integrates with language models for intelligent
information extraction and a vector database for semantic search capabilities.
"""

import asyncio
import logging
from typing import Any, Optional

import numpy as np
from pydantic import BaseModel, InstanceOf, field_validator

from .semantic_ingestion import IngestionService
from .semantic_model import ResourceRetriever, SemanticFeature
from .semantic_set_config import ConfigManager
from .semantic_tracker import SemanticUpdateTrackerManager
from .storage.storage_base import SemanticStorageBase

logger = logging.getLogger(__name__)


def _consolidate_errors_and_raise(possible_errors: list[Any], msg: str) -> None:
    errors = [r for r in possible_errors if isinstance(r, Exception)]
    if len(errors) > 0:
        raise ExceptionGroup(msg, errors)


class SemanticService:
    class Params(BaseModel):
        config_manager: InstanceOf[ConfigManager]
        semantic_storage: InstanceOf[SemanticStorageBase]
        consolidation_threshold: int = 20

        feature_update_interval_sec: float = 2.0
        """ Interval in seconds for feature updates. This controls how often the
        background task checks for dirty sets and processes their
        conversation history to update features.
        """

        feature_update_message_limit: int = 5
        """ Number of messages after which a feature update is triggered.
        If a set sends this many messages, their features will be updated.
        """

        feature_update_time_limit_sec: float = 120.0
        """ Time in seconds after which a feature update is triggered.
        If a set has sent messages and this much time has passed since
        the first message, their features will be updated.
        """

        resource_retriever: InstanceOf[ResourceRetriever]

        @field_validator("resource_retriever")
        @classmethod
        def validate_resource_storage(cls, v):
            if not isinstance(v, ResourceRetriever):
                raise ValueError(
                    "resource_storage must be an instance of ResourceRetriever"
                )
            return v

    def __init__(
        self,
        params: Params,
    ):
        self._semantic_storage = params.semantic_storage
        self._background_ingestion_interval_sec = params.feature_update_interval_sec

        self._resource_retriever: ResourceRetriever = params.resource_retriever

        self._consolidation_threshold = params.consolidation_threshold

        self._dirty_sets: SemanticUpdateTrackerManager = SemanticUpdateTrackerManager(
            message_limit=params.feature_update_message_limit,
            time_limit_sec=params.feature_update_time_limit_sec,
        )

        self._ingestion_task = None
        self._is_shutting_down = False

    async def start(self):
        if self._ingestion_task is not None:
            return

        self._is_shutting_down = False
        self._ingestion_task = asyncio.create_task(self._background_ingestion_task())

    async def stop(self):
        if self._ingestion_task is None:
            return

        self._is_shutting_down = True
        await self._ingestion_task

    async def search(
        self,
        set_ids: list[str],
        query: str,
        *,
        min_cos: float = 0.7,
        type_names: Optional[list[str]] = None,
        tag_names: Optional[list[str]] = None,
        feature_names: Optional[list[str]] = None,
        k: Optional[int] = None,
        load_citations: bool = False,
    ) -> list[SemanticFeature]:
        resources = self._resource_retriever.get_resources(set_ids[0])
        query_embedding = (await resources.embedder.search_embed([query]))[0]

        return await self._semantic_storage.get_feature_set(
            set_ids=set_ids,
            vector_search_opts=SemanticStorageBase.VectorSearchOpts(
                query_embedding=np.array(query_embedding),
                min_cos=min_cos,
            ),
            type_names=type_names,
            tags=tag_names,
            feature_names=feature_names,
            k=k,
            load_citations=load_citations,
        )

    async def add_messages(self, set_id: str, history_ids: list[int]):
        res = await asyncio.gather(
            *[
                self._semantic_storage.add_history_to_set(
                    set_id=set_id, history_id=h_id
                )
                for h_id in history_ids
            ],
            return_exceptions=True,
        )

        _consolidate_errors_and_raise(res, "Failed to add messages to set")

    async def add_message_to_sets(self, history_id: int, set_ids: list[str]):
        res = await asyncio.gather(
            *[
                self._semantic_storage.add_history_to_set(
                    set_id=set_id, history_id=history_id
                )
                for set_id in set_ids
            ],
            return_exceptions=True,
        )

        _consolidate_errors_and_raise(res, "Failed to add message to sets")

    async def number_of_uningested(self, set_id: str) -> int:
        return await self._semantic_storage.get_history_messages_count(
            set_id=set_id, is_ingested=False
        )

    async def add_new_feature(
        self,
        *,
        set_id: str,
        type_id: str,
        feature: str,
        value: str,
        tag: str,
        metadata: dict[str, str] | None = None,
        citations: list[int] | None = None,
    ):
        resources = self._resource_retriever.get_resources(set_id)
        embedding = (await resources.embedder.ingest_embed([value]))[0]

        f_id = await self._semantic_storage.add_feature(
            set_id=set_id,
            type_name=type_id,
            feature=feature,
            value=value,
            tag=tag,
            metadata=metadata,
            embedding=np.array(embedding),
        )

        if citations is not None:
            await self._semantic_storage.add_citations(f_id, citations)

    async def get_features(self, feature_ids: list[int]) -> list[SemanticFeature]:
        pass

    class FeatureSearchOpts(BaseModel):
        set_ids: list[str] | None = None
        type_names: list[str] | None = None
        feature_names: list[str] | None = None
        tags: list[str] | None = None
        k: int = 100
        with_citations: bool = False

    async def get_set_features(
        self,
        opts: FeatureSearchOpts,
    ) -> list[SemanticFeature]:
        return await self._semantic_storage.get_feature_set(
            set_ids=opts.set_ids,
            type_names=opts.type_names,
            feature_names=opts.feature_names,
            tags=opts.tags,
            k=opts.k,
            load_citations=opts.with_citations,
        )

    async def update_feature(
        self,
        feature_id: int,
        *,
        set_id: Optional[str] = None,
        type_id: Optional[str] = None,
        feature: Optional[str] = None,
        value: Optional[str] = None,
        tag: Optional[str] = None,
        metadata: dict[str, str] | None = None,
    ):
        resources = self._resource_retriever.get_resources(set_id)
        embedding = (await resources.embedder.ingest_embed([value]))[0]

        await self._semantic_storage.update_feature(
            feature_id=feature_id,
            set_id=set_id,
            type_name=type_id,
            feature=feature,
            value=value,
            tag=tag,
            metadata=metadata,
            embedding=np.array(embedding),
        )

    async def delete_features(self, feature_ids: list[int]):
        await self._semantic_storage.delete_features(feature_ids)

    async def delete_feature_set(self, opts: FeatureSearchOpts):
        await self._semantic_storage.delete_feature_set(
            set_ids=opts.set_ids,
            type_names=opts.type_names,
            feature_names=opts.feature_names,
            tags=opts.tags,
        )

    async def _background_ingestion_task(self):
        ingestion_service = IngestionService(
            params=IngestionService.Params(
                semantic_storage=self._semantic_storage,
                resource_retriever=self._resource_retriever,
            )
        )

        while not self._is_shutting_down:
            dirty_sets = await self._dirty_sets.get_sets_to_update()

            if len(dirty_sets) == 0:
                await asyncio.sleep(self._background_ingestion_interval_sec)
                continue

            await ingestion_service.process_set_ids(dirty_sets)
