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
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from typing import Any, Protocol, cast, runtime_checkable

import numpy as np
from pydantic import BaseModel, InstanceOf

from memmachine.common.data_types import FilterablePropertyValue
from memmachine.common.embedder import Embedder
from memmachine.common.episode_store import EpisodeIdT, EpisodeStorage
from memmachine.common.errors import (
    CategoryNotFoundError,
    InvalidSetIdConfigurationError,
)
from memmachine.common.filter.filter_parser import And, Comparison, FilterExpr
from memmachine.common.language_model import LanguageModel

from .config_store.config_store import SemanticConfigStorage
from .semantic_ingestion import IngestionService
from .semantic_model import (
    CategoryIdT,
    FeatureIdT,
    Resources,
    SemanticCategory,
    SemanticFeature,
    SetIdT,
    TagIdT,
)
from .storage.storage_base import SemanticStorage

logger = logging.getLogger(__name__)


def _consolidate_errors_and_raise(possible_errors: list[Any], msg: str) -> None:
    errors = [r for r in possible_errors if isinstance(r, Exception)]
    if len(errors) > 0:
        raise ExceptionGroup(msg, errors)


DefaultCategoryRetriever = Callable[[SetIdT], list[SemanticCategory]]


@runtime_checkable
class ResourceManager(Protocol):
    """Resource Retriever interface for Semantic Memory."""

    async def get_embedder(self, embedder_name: str) -> Embedder: ...

    async def get_language_model(self, language_model_name: str) -> LanguageModel: ...


def _with_has_set_ids(
    set_ids: list[SetIdT],
    filter_expr: FilterExpr | None,
) -> FilterExpr:
    if len(set_ids) == 0:
        return filter_expr

    set_expr = Comparison(
        field="set_id",
        op="in",
        value=cast(list[FilterablePropertyValue], list(set_ids)),
    )

    if filter_expr is None:
        return set_expr
    return And(left=set_expr, right=filter_expr)


class SemanticService:
    """High-level coordinator for ingesting history and serving semantic features."""

    class Params(BaseModel):
        """Infrastructure dependencies and background-update configuration."""

        semantic_storage: InstanceOf[SemanticStorage]
        episode_storage: InstanceOf[EpisodeStorage]
        semantic_config_storage: InstanceOf[SemanticConfigStorage]
        consolidation_threshold: int = 20

        feature_update_interval_sec: float = 2.0

        uningested_message_limit: int = 5
        uningested_time_limit: timedelta = timedelta(minutes=5)

        resource_manager: InstanceOf[ResourceManager]

        default_embedder: InstanceOf[Embedder]
        default_embedder_name: str
        default_language_model: InstanceOf[LanguageModel]
        default_category_retriever: DefaultCategoryRetriever

        debug_fail_loudly: bool = False

    def __init__(
        self,
        params: Params,
    ) -> None:
        """Set up semantic memory services and background ingestion tracking."""
        self._semantic_storage = params.semantic_storage
        self._episode_storage = params.episode_storage
        self._semantic_config_storage: SemanticConfigStorage = (
            params.semantic_config_storage
        )
        self._background_ingestion_interval_sec = params.feature_update_interval_sec

        self._resource_manager = params.resource_manager
        self._default_embedder: Embedder = params.default_embedder
        self._default_embedder_name: str = params.default_embedder_name
        self._default_language_model: LanguageModel = params.default_language_model
        self._default_category_retriever: Callable[[SetIdT], list[SemanticCategory]] = (
            params.default_category_retriever
        )

        self._consolidation_threshold = params.consolidation_threshold

        self._feature_update_message_limit = max(
            params.uningested_message_limit,
            1,
        )
        self._feature_time_limit = params.uningested_time_limit

        self._ingestion_task: Task | None = None
        self._is_shutting_down = False
        self._debug_fail_loudly = params.debug_fail_loudly

    async def start(self) -> None:
        logger.info("Starting semantic memory services")

        if self._ingestion_task is not None:
            return

        self._is_shutting_down = False
        self._ingestion_task = asyncio.create_task(self._background_ingestion_task())

    async def stop(self) -> None:
        logger.info("Stopping semantic memory services")

        if self._ingestion_task is None:
            return

        self._is_shutting_down = True
        await self._ingestion_task
        self._ingestion_task = None

    async def _set_id_search(
        self,
        *,
        set_id: str,
        embedding: list[float],
        min_distance: float | None = None,
        limit: int | None = 30,
        load_citations: bool = False,
        filter_expr: FilterExpr | None = None,
    ) -> list[SemanticFeature]:
        filter_expr = _with_has_set_ids(
            set_ids=[set_id],
            filter_expr=filter_expr,
        )

        return await self._semantic_storage.get_feature_set(
            filter_expr=filter_expr,
            page_size=limit,
            vector_search_opts=SemanticStorage.VectorSearchOpts(
                query_embedding=np.array(embedding),
                min_distance=min_distance,
            ),
            load_citations=load_citations,
        )

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
        logger.debug("Searching for %s in set ids %s", query, set_ids)

        embeddings: dict[str, list[float]] = await self._set_ids_embed(
            set_ids=set_ids,
            query=query,
        )

        assert set(set_ids) == set(embeddings.keys())

        tasks: list[asyncio.Task[list[SemanticFeature]]] = []
        async with asyncio.TaskGroup() as tg:
            for set_id in set_ids:
                t = tg.create_task(
                    self._set_id_search(
                        set_id=set_id,
                        embedding=embeddings[set_id],
                        min_distance=min_distance,
                        limit=limit,
                        load_citations=load_citations,
                        filter_expr=filter_expr,
                    )
                )
                tasks.append(t)

        t_res = [t.result() for t in tasks]

        res: list[SemanticFeature] = []
        for t_list in t_res:
            res = res + t_list

        return res

    async def add_messages(self, set_id: SetIdT, history_ids: list[EpisodeIdT]) -> None:
        logger.info("Adding messages to set %s: %s", set_id, history_ids)

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
        assert len(set_ids) == len(set(set_ids))

        logger.info("Adding message id %s to sets %s", history_id, set_ids)

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
        logger.info("Deleting messages from sets %s", set_ids)

        await self._semantic_storage.delete_history_set(set_ids=set_ids)

    async def number_of_uningested(self, set_ids: list[SetIdT]) -> int:
        logger.debug("Getting number of uningested messages for set ids %s", set_ids)

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
        logger.info("Adding new feature %s to set %s", feature, set_id)

        resources = await self._set_id_resource(set_id=set_id)

        # Validate that the category exists
        category_names = {cat.name for cat in resources.semantic_categories}
        if category_name not in category_names:
            raise CategoryNotFoundError(set_id=set_id, category_name=category_name)

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
        logger.debug("Getting feature %s", feature_id)

        return await self._semantic_storage.get_feature(
            feature_id,
            load_citations=load_citations,
        )

    async def get_set_features(
        self,
        *,
        set_ids: list[SetIdT],
        filter_expr: FilterExpr | None = None,
        page_size: int | None = None,
        page_num: int | None = None,
        with_citations: bool = False,
    ) -> list[SemanticFeature]:
        logger.debug("Getting features for set ids %s", set_ids)

        return await self._semantic_storage.get_feature_set(
            filter_expr=_with_has_set_ids(
                set_ids=set_ids,
                filter_expr=filter_expr,
            ),
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
        logger.info("Updating feature %s", feature_id)
        embedding = None

        if category_name is not None or value is not None:
            if set_id is None:
                original_feature = await self._semantic_storage.get_feature(feature_id)
                if original_feature is None or original_feature.set_id is None:
                    raise ValueError(
                        "Unable to deduce set_id, the feature_id may be incorrect. "
                        "set_id is required to update a feature",
                    )
                str_set_id = original_feature.set_id
            else:
                str_set_id = set_id
            resources = await self._set_id_resource(set_id=str_set_id)

            if value is not None:
                e = (await resources.embedder.ingest_embed([value]))[0]
                embedding = np.array(e)

            if category_name is not None:
                category_names = {cat.name for cat in resources.semantic_categories}

                if category_name not in category_names:
                    raise CategoryNotFoundError(
                        set_id=str_set_id, category_name=category_name
                    )

        await self._semantic_storage.update_feature(
            feature_id=feature_id,
            set_id=set_id,
            category_name=category_name,
            feature=feature,
            value=value,
            tag=tag,
            metadata=metadata,
            embedding=embedding,
        )

    async def delete_history(self, history_ids: list[EpisodeIdT]) -> None:
        logger.info("Deleting history ids %s", history_ids)

        await self._semantic_storage.delete_history(history_ids)

    async def delete_features(self, feature_ids: list[FeatureIdT]) -> None:
        logger.info("Deleting features ids %s", feature_ids)

        await self._semantic_storage.delete_features(feature_ids)

    async def delete_feature_set(
        self,
        *,
        set_ids: list[SetIdT],
        filter_expr: FilterExpr | None = None,
    ) -> None:
        logger.info("Deleting filter feature set ids %s", set_ids)

        await self._semantic_storage.delete_feature_set(
            filter_expr=_with_has_set_ids(
                set_ids=set_ids,
                filter_expr=filter_expr,
            ),
        )

    async def _set_ids_embed(
        self,
        *,
        set_ids: list[SetIdT],
        query: str,
    ) -> dict[str, list[float]]:
        set_id_embedders: dict[str, str | None]
        embedders: dict[str, Embedder]
        has_default_embedder: bool

        (
            set_id_embedders,
            embedders,
            has_default_embedder,
        ) = await self._set_ids_embedders(set_ids=set_ids)

        tasks: dict[str, asyncio.Task[list[list[float]]]] = {}
        default_task: asyncio.Task[list[list[float]]] | None = None
        async with asyncio.TaskGroup() as tg:
            for e_str, e_cls in embedders.items():
                tasks[e_str] = tg.create_task(e_cls.search_embed(queries=[query]))

            if has_default_embedder:
                default_task = tg.create_task(
                    self._default_embedder.search_embed(queries=[query])
                )

        results: dict[str, list[float]] = {}
        for set_id in set_ids:
            embedder_str = set_id_embedders[set_id]

            if embedder_str is None:
                assert default_task is not None

                results[set_id] = default_task.result()[0]
            else:
                results[set_id] = tasks[embedder_str].result()[0]

        return results

    async def _set_ids_embedders(
        self,
        *,
        set_ids: list[SetIdT],
    ) -> tuple[dict[str, str | None], dict[str, Embedder], bool]:
        async def _get_embedder_name(s_id: SetIdT) -> str | None:
            cfg = await self._semantic_config_storage.get_setid_config(set_id=s_id)
            return cfg.embedder_name

        tasks: dict[str, asyncio.Task[str | None]] = {}
        async with asyncio.TaskGroup() as tg:
            for s_id in set_ids:
                tasks[s_id] = tg.create_task(_get_embedder_name(s_id))

        set_id_embedders = {s_id: task.result() for s_id, task in tasks.items()}

        embedders: dict[str, Embedder] = {}
        has_default_embedder = False
        for embedder_name in set_id_embedders.values():
            if embedder_name is None:
                has_default_embedder = True
                continue

            if embedder_name in embedders:
                continue

            embed = await self._resource_manager.get_embedder(embedder_name)
            embedders[embedder_name] = embed

        return set_id_embedders, embedders, has_default_embedder

    async def _set_id_resource(
        self,
        set_id: SetIdT,
    ) -> Resources:
        config = await self._semantic_config_storage.get_setid_config(set_id=set_id)

        embedder: Embedder
        if config.embedder_name is None:
            # When no embedder is specified, use the default embedder and store it in the set_id config.
            await self._semantic_config_storage.set_setid_config(
                set_id=set_id,
                embedder_name=self._default_embedder_name,
            )

            embedder = self._default_embedder
        else:
            embedder = await self._resource_manager.get_embedder(config.embedder_name)

        llm_model: LanguageModel
        if config.llm_name is None:
            llm_model = self._default_language_model
        else:
            llm_model = await self._resource_manager.get_language_model(config.llm_name)

        if config.disabled_categories is None:
            disabled_categories = []
        else:
            disabled_categories = config.disabled_categories

        user_categories = config.categories

        default_categories = self._default_category_retriever(set_id)
        enabled_default = [
            category
            for category in default_categories
            if category.name not in disabled_categories
        ]

        categories = user_categories + enabled_default

        return Resources(
            embedder=embedder,
            language_model=llm_model,
            semantic_categories=categories,
        )

    async def delete_set_id(self, *, set_ids: list[SetIdT]) -> None:
        logger.info("Deleting set ids %s", set_ids)

        async with asyncio.TaskGroup() as tg:
            tg.create_task(self._semantic_storage.delete_history_set(set_ids=set_ids))
            tg.create_task(
                self._semantic_storage.delete_feature_set(
                    filter_expr=_with_has_set_ids(set_ids=set_ids, filter_expr=None),
                )
            )
            tg.create_task(self._semantic_storage.reset_set_ids(set_ids=set_ids))

    async def set_set_id_config(
        self,
        *,
        set_id: SetIdT,
        embedder_name: str | None = None,
        llm_name: str | None = None,
    ) -> None:
        logger.info("Setting set id config for %s", set_id)

        current_set_id_embedder_name = (
            await self._semantic_config_storage.get_setid_config(set_id=set_id)
        ).embedder_name

        if embedder_name != current_set_id_embedder_name:
            valid_changing_embedder = (
                current_set_id_embedder_name is None
                and len(await self.get_set_features(set_ids=[set_id])) == 0
            )
            if not valid_changing_embedder:
                raise InvalidSetIdConfigurationError(set_id=set_id)

            # Reset any indexes that the set id may have referencing it.
            await self.delete_set_id(set_ids=[set_id])

        await self._semantic_config_storage.set_setid_config(
            set_id=set_id, embedder_name=embedder_name, llm_name=llm_name
        )

    async def get_set_id_config(
        self, *, set_id: SetIdT
    ) -> SemanticConfigStorage.Config:
        logger.debug("Getting set id config for %s", set_id)

        return await self._semantic_config_storage.get_setid_config(set_id=set_id)

    async def get_category(
        self,
        *,
        category_id: CategoryIdT,
    ) -> SemanticConfigStorage.Category | None:
        logger.debug("Getting category %s", category_id)

        return await self._semantic_config_storage.get_category(category_id=category_id)

    async def add_new_category_to_set_id(
        self,
        *,
        set_id: SetIdT,
        category_name: str,
        prompt: str,
        description: str | None,
    ) -> CategoryIdT:
        logger.info("Adding new category %s to set %s", category_name, set_id)

        category_id = await self._semantic_config_storage.create_category(
            set_id=set_id,
            category_name=category_name,
            prompt=prompt,
            description=description,
        )

        return category_id

    async def add_new_category_to_set_type(
        self,
        *,
        set_type_id: str,
        category_name: str,
        prompt: str,
        description: str | None,
    ) -> CategoryIdT:
        logger.info("Adding new category %s to set type %s", category_name, set_type_id)

        return await self._semantic_config_storage.create_set_type_category(
            set_type_id=set_type_id,
            category_name=category_name,
            prompt=prompt,
            description=description,
        )

    async def get_set_type_categories(
        self, *, set_type_id: str
    ) -> list[SemanticCategory]:
        logger.debug("Getting set type categories for %s", set_type_id)

        return await self._semantic_config_storage.get_set_type_categories(
            set_type_id=set_type_id
        )

    async def clone_category(
        self,
        *,
        category_id: CategoryIdT,
        new_set_id: SetIdT,
        new_category_name: str,
    ) -> CategoryIdT:
        logger.info("Cloning category %s to %s", category_id, new_category_name)

        return await self._semantic_config_storage.clone_category(
            category_id=category_id,
            new_set_id=new_set_id,
            new_name=new_category_name,
        )

    async def disable_category(self, *, set_id: SetIdT, category_name: str) -> None:
        logger.info("Disabling default category %s for set %s", category_name, set_id)

        await self._semantic_config_storage.add_disabled_category_to_setid(
            set_id=set_id,
            category_name=category_name,
        )

    async def get_category_set_ids(
        self,
        *,
        category_id: CategoryIdT,
    ) -> list[SetIdT]:
        logger.debug("Getting set_ids for category %s", category_id)

        return await self._semantic_config_storage.get_category_set_ids(
            category_id=category_id
        )

    async def delete_category(
        self,
        *,
        category_id: CategoryIdT,
    ) -> None:
        logger.info("Deleting category %s", category_id)

        async with asyncio.TaskGroup() as tg:
            set_ids_task = tg.create_task(
                self._semantic_config_storage.get_category_set_ids(
                    category_id=category_id,
                )
            )

            category_task = tg.create_task(
                self._semantic_config_storage.get_category(
                    category_id=category_id,
                )
            )

        set_ids = set_ids_task.result()
        category = category_task.result()

        if category is not None and set_ids:
            logger.debug(
                "Deleting features for category %s for set_ids: %s",
                category.name,
                set_ids,
            )

            await self.delete_feature_set(
                set_ids=set_ids,
                filter_expr=Comparison(
                    field="category_name",
                    op="=",
                    value=category.name,
                ),
            )
        else:
            logger.debug(
                "No features found for category %s, nothing to delete",
                category_id,
            )

        await self._semantic_config_storage.delete_category(category_id=category_id)

    async def add_tag(
        self,
        *,
        category_id: CategoryIdT,
        tag_name: str,
        tag_description: str,
    ) -> TagIdT:
        logger.info("Adding tag %s to category %s", tag_name, category_id)

        return await self._semantic_config_storage.add_tag(
            category_id=category_id,
            tag_name=tag_name,
            description=tag_description,
        )

    async def update_tag(
        self,
        *,
        tag_id: TagIdT,
        tag_name: str,
        tag_description: str,
    ) -> None:
        logger.info("Updating tag %s", tag_id)

        await self._semantic_config_storage.update_tag(
            tag_id=tag_id,
            tag_name=tag_name,
            tag_description=tag_description,
        )

    async def delete_tag(self, *, tag_id: TagIdT) -> None:
        logger.info("Deleting tag %s", tag_id)

        await self._semantic_config_storage.delete_tag(tag_id=tag_id)

    async def list_set_id_starts_with(self, prefix: str) -> list[SetIdT]:
        logger.debug("Listing set ids starts with %s", prefix)

        return await self._semantic_storage.get_set_ids_starts_with(prefix)

    async def _background_ingestion_task(self) -> None:
        ingestion_service = IngestionService(
            params=IngestionService.Params(
                semantic_storage=self._semantic_storage,
                resource_retriever=self._set_id_resource,
                history_store=self._episode_storage,
            ),
        )

        while not self._is_shutting_down:
            dirty_sets = await self._semantic_storage.get_history_set_ids(
                min_uningested_messages=self._feature_update_message_limit,
                older_than=datetime.now(tz=UTC) - self._feature_time_limit,
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
