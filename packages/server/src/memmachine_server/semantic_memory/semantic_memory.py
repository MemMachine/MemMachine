"""
Semantic memory for orgs and projects.

`SemanticService` groups the messages of each project into semantic sets,
extracts features from them in the background, and serves those features.
"""

import asyncio
import contextlib
import hashlib
import logging
import re
from asyncio import Task
from collections.abc import (
    AsyncIterator,
    Callable,
    Iterable,
    Iterator,
    Mapping,
    MutableMapping,
    Sequence,
)
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Protocol, runtime_checkable
from uuid import UUID

import numpy as np
from pydantic import BaseModel, Field, InstanceOf, JsonValue

from memmachine_server.common.embedder import Embedder
from memmachine_server.common.episode_store import Episode, EpisodeStorage
from memmachine_server.common.errors import (
    CategoryNotFoundError,
    InvalidSetIdConfigurationError,
    ResourceNotFoundError,
)
from memmachine_server.common.filter.filter_parser import (
    And,
    Comparison,
    FilterExpr,
    In,
)
from memmachine_server.common.language_model import LanguageModel
from memmachine_server.common.utils import merge_async_iterators

from .config_store.config_store import SemanticConfigStorage
from .semantic_ingestion import IngestionService
from .semantic_model import (
    CategoryIdT,
    FeatureIdT,
    Resources,
    SemanticCategory,
    SemanticFeature,
    SetIdT,
    SetTypeEntry,
    TagIdT,
)
from .storage.storage_base import SemanticStorage

logger = logging.getLogger(__name__)


def _consolidate_errors_and_raise(possible_errors: Sequence[Any], msg: str) -> None:
    errors = [r for r in possible_errors if isinstance(r, Exception)]
    if len(errors) > 0:
        raise ExceptionGroup(msg, errors)


DefaultCategoryRetriever = Callable[[SetIdT], Sequence[SemanticCategory]]


@runtime_checkable
class ResourceManager(Protocol):
    """Resource Retriever interface for Semantic Memory."""

    async def get_embedder(self, embedder_name: str) -> Embedder: ...

    async def get_language_model(self, language_model_name: str) -> LanguageModel: ...


def _with_has_set_ids(
    set_ids: Sequence[SetIdT],
    filter_expr: FilterExpr | None,
) -> FilterExpr:
    if len(set_ids) == 0:
        return filter_expr

    set_expr = In(
        field="set_id",
        values=list(set_ids),
    )

    if filter_expr is None:
        return set_expr
    return And(left=set_expr, right=filter_expr)


def _hash_tag_list(strings: Iterable[str]) -> str:
    strings = sorted(strings)

    h = hashlib.shake_256()
    for s in strings:
        h.update(s.encode("utf-8"))
        h.update(b"\x00")
    return h.hexdigest(6)


class SemanticService:
    """Ingests the messages of orgs and projects and serves semantic features."""

    @runtime_checkable
    class SessionData(Protocol):
        """Protocol exposing the identifiers used to derive set_ids."""

        @property
        def org_id(self) -> str: ...

        @property
        def project_id(self) -> str: ...

    class SetType(Enum):
        """Default set_id prefixes used by `SemanticService`."""

        OrgSet = "set_type"
        ProjectSet = "project_set"
        UserSet = "user_set"
        OtherSet = "other_set"

    class Params(BaseModel):
        """Infrastructure dependencies and background-update configuration."""

        semantic_storage: InstanceOf[SemanticStorage]
        episode_storage: InstanceOf[EpisodeStorage]
        semantic_config_storage: InstanceOf[SemanticConfigStorage]
        consolidation_threshold: int = 20

        feature_update_interval_sec: float = 2.0

        uningested_message_limit: int = 5
        uningested_time_limit: timedelta = timedelta(minutes=5)
        missing_episode_grace_period_sec: float = Field(default=30.0, ge=0)

        max_features_per_update: int = 50

        resource_manager: InstanceOf[ResourceManager]

        default_embedder: InstanceOf[Embedder]
        default_embedder_name: str
        default_language_model: InstanceOf[LanguageModel]
        default_category_retriever: DefaultCategoryRetriever

        debug_fail_loudly: bool = False

    class Set(BaseModel):
        """Semantic memory set, containing a set of features and resources."""

        id: SetIdT
        is_org_level: bool
        tags: Sequence[str]
        name: str | None = None
        description: str | None = None

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
        self._default_category_retriever: Callable[
            [SetIdT], Sequence[SemanticCategory]
        ] = params.default_category_retriever

        self._consolidation_threshold = params.consolidation_threshold
        self._max_features_per_update = params.max_features_per_update

        self._feature_update_message_limit = max(
            params.uningested_message_limit,
            1,
        )
        self._feature_time_limit = params.uningested_time_limit
        self._missing_episode_grace_period_sec = params.missing_episode_grace_period_sec

        self._ingestion_task: Task | None = None
        self._is_shutting_down = False
        self._shutdown_event = asyncio.Event()
        self._debug_fail_loudly = params.debug_fail_loudly

    async def start(self) -> None:
        logger.info("Starting semantic memory services")

        if self._ingestion_task is not None:
            return

        self._is_shutting_down = False
        self._shutdown_event.clear()
        self._ingestion_task = asyncio.create_task(self._background_ingestion_task())

    async def stop(self) -> None:
        logger.info("Stopping semantic memory services")

        if self._ingestion_task is None:
            return

        self._is_shutting_down = True
        self._shutdown_event.set()
        await self._ingestion_task
        self._ingestion_task = None

    async def add_message(
        self,
        episodes: Sequence[Episode],
        session_data: SessionData,
    ) -> None:
        self._assert_session_data_implements_protocol(session_data=session_data)
        if len(episodes) == 0:
            return

        episode_ids = [e.uid for e in episodes]
        assert len(episode_ids) == len(set(episode_ids)), "Episodes must be unique"

        async with asyncio.TaskGroup() as tg:
            for episode in episodes:
                tg.create_task(self._add_single_episode(episode, session_data))

    async def _add_single_episode(
        self,
        episode: Episode,
        session_data: SessionData,
    ) -> None:
        episode_metadata: MutableMapping[str, JsonValue] = (
            dict(episode.metadata) if episode.metadata is not None else {}
        )

        episode_metadata.setdefault("producer_id", episode.producer_id)

        set_ids = await self._get_set_ids_str_from_metadata(
            session_data=session_data,
            metadata=episode_metadata,
        )
        await self._add_message_to_sets(
            episode.uid,
            list(set_ids),
            created_at=episode.created_at,
        )

    async def _add_message_to_sets(
        self,
        history_id: UUID,
        set_ids: Sequence[SetIdT],
        *,
        created_at: datetime,
    ) -> None:
        assert len(set_ids) == len(set(set_ids))
        if not set_ids:
            return

        logger.debug("Adding message id %s to sets %s", history_id, set_ids)

        res = await asyncio.gather(
            *[
                self._semantic_storage.add_history_to_set(
                    set_id=set_id,
                    history_id=history_id,
                    created_at=created_at,
                )
                for set_id in set_ids
            ],
            return_exceptions=True,
        )

        _consolidate_errors_and_raise(res, "Failed to add message to sets")

    async def delete_history(self, history_ids: Sequence[UUID]) -> None:
        logger.info("Deleting history ids %s", history_ids)

        await self._semantic_storage.delete_history(history_ids)

    async def search(
        self,
        message: str,
        session_data: SessionData,
        *,
        set_metadata: Mapping[str, JsonValue] | None = None,
        min_distance: float | None = None,
        limit: int | None = None,
        load_citations: bool = False,
        search_filter: FilterExpr | None = None,
    ) -> AsyncIterator[SemanticFeature]:
        self._assert_session_data_implements_protocol(session_data=session_data)

        set_ids = await self._get_set_ids_str_from_metadata(
            session_data=session_data,
            metadata=set_metadata,
        )

        logger.debug("Searching for %s in set ids %s", message, set_ids)

        embeddings: Mapping[str, Sequence[float]] = await self._set_ids_embed(
            set_ids=list(set_ids),
            query=message,
        )

        # Create iterators for each set_id search (parallel)
        iterators = [
            self._set_id_search(
                set_id=set_id,
                embedding=embeddings[set_id],
                min_distance=min_distance,
                limit=limit,
                load_citations=load_citations,
                filter_expr=search_filter,
            )
            for set_id in set_ids
        ]

        # Merge and yield results as they become available
        async for feature in merge_async_iterators(iterators):
            yield feature

    async def _set_id_search(
        self,
        *,
        set_id: str,
        embedding: Sequence[float],
        min_distance: float | None = None,
        limit: int | None = 30,
        load_citations: bool = False,
        filter_expr: FilterExpr | None = None,
    ) -> AsyncIterator[SemanticFeature]:
        filter_expr = _with_has_set_ids(
            set_ids=[set_id],
            filter_expr=filter_expr,
        )

        async for feature in self._semantic_storage.get_feature_set(
            filter_expr=filter_expr,
            page_size=limit,
            vector_search_opts=SemanticStorage.VectorSearchOpts(
                query_embedding=np.array(embedding),
                min_distance=min_distance,
            ),
            load_citations=load_citations,
        ):
            yield feature

    async def _set_ids_embed(
        self,
        *,
        set_ids: Sequence[SetIdT],
        query: str,
    ) -> Mapping[str, Sequence[float]]:
        set_id_embedders: Mapping[str, str | None]
        embedders: Mapping[str, Embedder]
        has_default_embedder: bool

        (
            set_id_embedders,
            embedders,
            has_default_embedder,
        ) = await self._set_ids_embedders(set_ids=set_ids)

        tasks: MutableMapping[str, asyncio.Task[Sequence[Sequence[float]]]] = {}
        default_task: asyncio.Task[Sequence[Sequence[float]]] | None = None
        async with asyncio.TaskGroup() as tg:
            for e_str, e_cls in embedders.items():
                tasks[e_str] = tg.create_task(e_cls.search_embed(queries=[query]))

            if has_default_embedder:
                default_task = tg.create_task(
                    self._default_embedder.search_embed(queries=[query])
                )

        results: MutableMapping[str, Sequence[float]] = {}
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
        set_ids: Sequence[SetIdT],
    ) -> tuple[Mapping[str, str | None], Mapping[str, Embedder], bool]:
        async def _get_embedder_name(s_id: SetIdT) -> str | None:
            cfg = await self._semantic_config_storage.get_setid_config(set_id=s_id)
            return cfg.embedder_name

        tasks: MutableMapping[str, asyncio.Task[str | None]] = {}
        async with asyncio.TaskGroup() as tg:
            for s_id in set_ids:
                tasks[s_id] = tg.create_task(_get_embedder_name(s_id))

        set_id_embedders = {s_id: task.result() for s_id, task in tasks.items()}

        embedders: MutableMapping[str, Embedder] = {}
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

    async def add_feature(
        self,
        *,
        set_id: SetIdT,
        feature_metadata: Mapping[str, JsonValue] | None = None,
        category_name: str,
        feature: str,
        value: str,
        tag: str,
        citations: Sequence[UUID] | None = None,
    ) -> FeatureIdT:
        logger.debug("Adding new feature %s to set %s", feature, set_id)

        metadata = (
            {k: str(v) for k, v in feature_metadata.items()}
            if feature_metadata is not None
            else None
        )

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
        load_citations: bool = False,
    ) -> SemanticFeature | None:
        logger.debug("Getting feature %s", feature_id)

        return await self._semantic_storage.get_feature(
            feature_id,
            load_citations=load_citations,
        )

    async def update_feature(
        self,
        feature_id: FeatureIdT,
        *,
        category_name: str | None = None,
        feature: str | None = None,
        value: str | None = None,
        tag: str | None = None,
        metadata: Mapping[str, str] | None = None,
    ) -> None:
        logger.debug("Updating feature %s", feature_id)
        embedding = None

        if category_name is not None or value is not None:
            original_feature = await self._semantic_storage.get_feature(feature_id)
            if original_feature is None or original_feature.set_id is None:
                raise ValueError(
                    "Unable to deduce set_id, the feature_id may be incorrect. "
                    "set_id is required to update a feature",
                )
            set_id = original_feature.set_id
            resources = await self._set_id_resource(set_id=set_id)

            if value is not None:
                e = (await resources.embedder.ingest_embed([value]))[0]
                embedding = np.array(e)

            if category_name is not None:
                category_names = {cat.name for cat in resources.semantic_categories}

                if category_name not in category_names:
                    raise CategoryNotFoundError(
                        set_id=set_id, category_name=category_name
                    )

        await self._semantic_storage.update_feature(
            feature_id=feature_id,
            category_name=category_name,
            feature=feature,
            value=value,
            tag=tag,
            metadata=metadata,
            embedding=embedding,
        )

    async def delete_features(
        self,
        session_data: SessionData,
        feature_ids: Sequence[FeatureIdT],
    ) -> None:
        """
        Delete features in the sets the session addresses.

        Ids that name no feature are ignored.

        Raises:
            ResourceNotFoundError: A feature is in a set the session does not
                address. Nothing is deleted.

        """
        self._assert_session_data_implements_protocol(session_data=session_data)

        for feature_id in feature_ids:
            feature = await self._semantic_storage.get_feature(
                feature_id,
                load_citations=False,
            )
            if feature is not None and (
                feature.set_id is None
                or not self._session_addresses_set_id(
                    session_data=session_data,
                    set_id=feature.set_id,
                )
            ):
                raise ResourceNotFoundError(
                    f"Semantic feature '{feature_id}' was not found in this project."
                )

        logger.info("Deleting features ids %s", feature_ids)

        await self._semantic_storage.delete_features(feature_ids)

    async def get_set_features(
        self,
        session_data: SessionData,
        *,
        set_metadata: Mapping[str, JsonValue] | None = None,
        search_filter: FilterExpr | None = None,
        page_size: int | None = None,
        page_num: int | None = None,
        load_citations: bool = False,
    ) -> AsyncIterator[SemanticFeature]:
        self._assert_session_data_implements_protocol(session_data=session_data)

        set_ids = await self._get_set_ids_str_from_metadata(
            session_data=session_data,
            metadata=set_metadata,
        )

        logger.debug("Getting features for set ids %s", set_ids)

        async for feature in self._semantic_storage.get_feature_set(
            filter_expr=_with_has_set_ids(
                set_ids=list(set_ids),
                filter_expr=search_filter,
            ),
            page_size=page_size,
            page_num=page_num,
            load_citations=load_citations,
        ):
            yield feature

    async def delete_feature_set(
        self,
        session_data: SessionData,
        *,
        set_metadata: Mapping[str, JsonValue] | None = None,
        property_filter: FilterExpr | None = None,
    ) -> None:
        self._assert_session_data_implements_protocol(session_data=session_data)

        set_ids = await self._get_set_ids_str_from_metadata(
            session_data=session_data,
            metadata=set_metadata,
        )

        await self._delete_set_features(
            set_ids=list(set_ids),
            filter_expr=property_filter,
        )

    async def _get_set_ids_str_from_metadata(
        self,
        *,
        session_data: SessionData,
        metadata: Mapping[str, JsonValue] | None,
    ) -> Iterable[SetIdT]:
        normalized_metadata = self._normalize_metadata(metadata)

        deduped: MutableMapping[SetIdT, None] = {}

        async for entry in self._get_set_id_entries(
            session_data=session_data,
            metadata=metadata,
        ):
            set_id = self._generate_set_id(
                org_id=session_data.org_id,
                project_id=session_data.project_id if not entry.is_org_level else None,
                metadata=entry.tags,
            )
            if set_id not in deduped:
                deduped[set_id] = None
                await self._semantic_config_storage.register_set_id_set_type(
                    set_id=set_id,
                    set_type_id=entry.set_type_id,
                )

        for default in self._generate_default_sets(
            org_id=session_data.org_id,
            project_id=session_data.project_id,
            producer_id=normalized_metadata.get("producer_id"),
        ):
            deduped.setdefault(default.id, None)

        return tuple(deduped.keys())

    @dataclass(frozen=True, slots=True)
    class _SetIdEntry:
        """Resolved set_id qualifiers for a session message."""

        set_type_id: str
        is_org_level: bool
        tags: Mapping[str, str]

    async def _get_set_id_entries(
        self,
        *,
        session_data: SessionData,
        metadata: Mapping[str, JsonValue] | None = None,
    ) -> AsyncIterator[_SetIdEntry]:
        normalized_metadata = self._normalize_metadata(metadata)

        if "producer_id" in normalized_metadata:
            await self._ensure_user_set_type(org_id=session_data.org_id)

        set_type_ids = await self._semantic_config_storage.list_set_type_ids(
            org_id=session_data.org_id
        )

        metadata_tags = set(normalized_metadata.keys())

        yielded = False
        for set_type in set_type_ids:
            if set_type.id is None:
                continue

            if not metadata_tags.issuperset(set(set_type.tags)):
                continue

            yielded = True
            yield self._SetIdEntry(
                set_type_id=set_type.id,
                is_org_level=set_type.is_org_level,
                tags={tag: normalized_metadata[tag] for tag in set_type.tags},
            )

        if not yielded:
            logger.debug(
                "No relevant set type ids found for metadata %s",
                normalized_metadata,
            )

    @staticmethod
    def _normalize_metadata(
        metadata: Mapping[str, JsonValue] | None,
    ) -> Mapping[str, str]:
        if metadata is None:
            return {}

        metadata_dict = dict(metadata)
        return {
            key: str(value) for key, value in metadata_dict.items() if value is not None
        }

    @staticmethod
    def _org_set_id(
        *,
        org_id: str,
        project_id: str | None = None,
    ) -> SetIdT:
        org_base = f"org_{org_id}"

        if project_id is not None:
            org_project = f"{org_base}_project_{project_id}"
        else:
            org_project = org_base

        return org_project

    @classmethod
    def _generate_set_id(
        cls,
        *,
        org_id: str,
        project_id: str | None = None,
        metadata: Mapping[str, JsonValue],
    ) -> SetIdT:
        org_project = cls._org_set_id(org_id=org_id, project_id=project_id)

        string_tags = [f"{k}_{v}" for k, v in metadata.items()]

        metadata_keys = set(metadata.keys())

        if len(string_tags) == 0:
            if project_id is not None:
                def_type = SemanticService.SetType.ProjectSet
            else:
                def_type = SemanticService.SetType.OrgSet
        elif metadata_keys == {"producer_id"}:
            def_type = SemanticService.SetType.UserSet
        else:
            def_type = SemanticService.SetType.OtherSet

        return f"mem_{def_type.value}_{org_project}_{len(metadata)}_{_hash_tag_list(metadata.keys())}__{'_'.join(sorted(string_tags))}"

    @classmethod
    def _session_addresses_set_id(
        cls,
        *,
        session_data: SessionData,
        set_id: SetIdT,
    ) -> bool:
        """
        Report whether `_generate_set_id` produces `set_id` for the session.

        Those are the org-level sets of the session's org and the sets of
        its project, with any metadata.
        """
        set_types = "|".join(re.escape(set_type.value) for set_type in cls.SetType)
        org_projects = "|".join(
            re.escape(
                cls._org_set_id(org_id=session_data.org_id, project_id=project_id)
            )
            for project_id in (None, session_data.project_id)
        )
        return (
            re.fullmatch(
                rf"mem_(?:{set_types})_(?:{org_projects})_[0-9]+_[0-9a-f]+__.*",
                set_id,
                flags=re.DOTALL,
            )
            is not None
        )

    @staticmethod
    def get_default_set_id_type(
        set_id: SetIdT,
    ) -> SetType:
        for def_type in SemanticService.SetType:
            if set_id.startswith(f"mem_{def_type.value}"):
                return def_type

        raise RuntimeError(f"Invalid set_id: {set_id}")

    async def _ensure_user_set_type(self, *, org_id: str) -> None:
        set_types = await self._semantic_config_storage.list_set_type_ids(org_id=org_id)
        for set_type in set_types:
            if set_type.is_org_level and set(set_type.tags) == {"producer_id"}:
                return

        await self._semantic_config_storage.add_set_type_id(
            org_id=org_id,
            org_level_set=True,
            metadata_tags=["producer_id"],
            name="User Profile",
            description="Semantic memory scoped to producer identifiers.",
        )

    async def create_set_type(
        self,
        *,
        session_data: SessionData,
        is_org_level: bool = False,
        metadata_tags: Sequence[str],
        name: str | None = None,
        description: str | None = None,
    ) -> str:
        self._assert_session_data_implements_protocol(session_data=session_data)

        return await self._semantic_config_storage.add_set_type_id(
            org_id=session_data.org_id,
            org_level_set=is_org_level,
            metadata_tags=list(metadata_tags),
            name=name,
            description=description,
        )

    async def delete_set_type(
        self,
        *,
        set_type_id: str,
    ) -> None:
        await self._semantic_config_storage.delete_set_type_id(set_type_id=set_type_id)

    async def list_set_types(
        self,
        *,
        session_data: SessionData,
    ) -> Iterable[SetTypeEntry]:
        self._assert_session_data_implements_protocol(session_data=session_data)

        return await self._semantic_config_storage.list_set_type_ids(
            org_id=session_data.org_id
        )

    async def configure_set(
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
            has_features = False
            async for _ in self._semantic_storage.get_feature_set(
                filter_expr=_with_has_set_ids(set_ids=[set_id], filter_expr=None),
            ):
                has_features = True
                break
            valid_changing_embedder = (
                current_set_id_embedder_name is None and not has_features
            )
            if not valid_changing_embedder:
                raise InvalidSetIdConfigurationError(set_id=set_id)

            # Reset any indexes that the set id may have referencing it.
            await self._delete_set_ids(set_ids=[set_id])

        await self._semantic_config_storage.set_setid_config(
            set_id=set_id, embedder_name=embedder_name, llm_name=llm_name
        )

    async def _delete_set_ids(self, *, set_ids: Sequence[SetIdT]) -> None:
        logger.info("Deleting set ids %s", set_ids)

        async with asyncio.TaskGroup() as tg:
            tg.create_task(self._delete_set_features(set_ids=set_ids))
            tg.create_task(self._semantic_storage.reset_set_ids(set_ids=set_ids))

    async def get_set_id(
        self,
        *,
        session_data: SessionData,
        set_metadata_keys: Sequence[str],
        set_metadata: Mapping[str, JsonValue] | None = None,
        is_org_level: bool = False,
    ) -> SetIdT:
        self._assert_session_data_implements_protocol(session_data=session_data)

        metadata = (
            {key: set_metadata.get(key) for key in set_metadata_keys}
            if set_metadata is not None
            else {}
        )
        normalized_metadata = {
            key: str(value) for key, value in metadata.items() if value is not None
        }

        if "producer_id" in normalized_metadata:
            await self._ensure_user_set_type(org_id=session_data.org_id)

        set_id = self._generate_set_id(
            org_id=session_data.org_id,
            project_id=session_data.project_id if not is_org_level else None,
            metadata=normalized_metadata,
        )

        set_type_ids = await self._semantic_config_storage.list_set_type_ids(
            org_id=session_data.org_id
        )
        for set_type in set_type_ids:
            if set_type.id is None:
                continue

            if set_type.is_org_level != is_org_level:
                continue

            if set(set_type.tags) != set(set_metadata_keys):
                continue

            await self._semantic_config_storage.register_set_id_set_type(
                set_id=set_id,
                set_type_id=set_type.id,
            )
            break

        return set_id

    async def list_set_ids(
        self,
        *,
        session_data: SessionData,
        set_metadata: Mapping[str, JsonValue] | None = None,
    ) -> Iterable[Set]:
        self._assert_session_data_implements_protocol(session_data=session_data)

        normalized_metadata = self._normalize_metadata(set_metadata)

        set_type_ids = await self._semantic_config_storage.list_set_type_ids(
            org_id=session_data.org_id
        )
        set_type_map = {sid.id: sid for sid in set_type_ids if sid.id is not None}

        sets: MutableMapping[str, SemanticService.Set] = {}
        async for sid in self._get_set_id_entries(
            session_data=session_data,
            metadata=set_metadata,
        ):
            set_id = self._generate_set_id(
                org_id=session_data.org_id,
                project_id=session_data.project_id if not sid.is_org_level else None,
                metadata=sid.tags,
            )

            set_type_entry = set_type_map.get(sid.set_type_id)
            sets.setdefault(
                set_id,
                SemanticService.Set(
                    id=set_id,
                    is_org_level=sid.is_org_level,
                    tags=list(sid.tags.keys()),
                    name=set_type_entry.name if set_type_entry else None,
                    description=set_type_entry.description if set_type_entry else None,
                ),
            )

        for default in self._generate_default_sets(
            org_id=session_data.org_id,
            project_id=session_data.project_id,
            producer_id=normalized_metadata.get("producer_id"),
        ):
            sets.setdefault(default.id, default)

        return tuple(sets.values())

    @classmethod
    def _generate_default_sets(
        cls,
        *,
        org_id: str,
        project_id: str | None,
        producer_id: str | None,
    ) -> Iterator[Set]:
        yield SemanticService.Set(
            id=cls._generate_set_id(
                org_id=org_id,
                project_id=None,
                metadata={},
            ),
            is_org_level=True,
            tags=[],
            name=None,
            description=None,
        )

        if producer_id is not None:
            yield SemanticService.Set(
                id=cls._generate_set_id(
                    org_id=org_id,
                    project_id=None,
                    metadata={"producer_id": producer_id},
                ),
                is_org_level=True,
                tags=["producer_id"],
                name=None,
                description=None,
            )

        if project_id is not None:
            yield SemanticService.Set(
                id=cls._generate_set_id(
                    org_id=org_id,
                    project_id=project_id,
                    metadata={},
                ),
                is_org_level=False,
                tags=[],
            )

    async def get_category(
        self,
        *,
        category_id: CategoryIdT,
    ) -> SemanticConfigStorage.Category | None:
        logger.debug("Getting category %s", category_id)

        return await self._semantic_config_storage.get_category(category_id=category_id)

    async def get_category_set_ids(
        self,
        *,
        category_id: CategoryIdT,
    ) -> Sequence[SetIdT]:
        logger.debug("Getting set_ids for category %s", category_id)

        return await self._semantic_config_storage.get_category_set_ids(
            category_id=category_id
        )

    async def add_new_category(
        self,
        *,
        set_id: SetIdT,
        category_name: str,
        prompt: str,
        description: str | None,
    ) -> CategoryIdT:
        logger.info("Adding new category %s to set %s", category_name, set_id)

        return await self._semantic_config_storage.create_category(
            set_id=set_id,
            category_name=category_name,
            prompt=prompt,
            description=description,
        )

    async def add_category_template(
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

    async def list_category_templates(
        self, *, set_type_id: str
    ) -> Iterable[SemanticCategory]:
        logger.debug("Getting set type categories for %s", set_type_id)

        return await self._semantic_config_storage.get_set_type_categories(
            set_type_id=set_type_id
        )

    async def disable_category(
        self,
        *,
        set_id: SetIdT,
        category_name: str,
    ) -> None:
        logger.info("Disabling default category %s for set %s", category_name, set_id)

        await self._semantic_config_storage.add_disabled_category_to_setid(
            set_id=set_id,
            category_name=category_name,
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

            await self._delete_set_features(
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

    async def _delete_set_features(
        self,
        *,
        set_ids: Sequence[SetIdT],
        filter_expr: FilterExpr | None = None,
    ) -> None:
        logger.debug("Deleting filter feature set ids %s", set_ids)

        await self._semantic_storage.delete_feature_set(
            filter_expr=_with_has_set_ids(
                set_ids=set_ids,
                filter_expr=filter_expr,
            ),
        )

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

    async def delete_tag(self, *, tag_id: TagIdT) -> None:
        logger.info("Deleting tag %s", tag_id)

        await self._semantic_config_storage.delete_tag(tag_id=tag_id)

    @staticmethod
    def _assert_session_data_implements_protocol(session_data: SessionData) -> None:
        if not isinstance(session_data, SemanticService.SessionData):
            raise TypeError(
                "session_data must implement SemanticService.SessionData protocol"
            )

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

        categories = [*user_categories, *enabled_default]

        return Resources(
            embedder=embedder,
            language_model=llm_model,
            semantic_categories=categories,
        )

    async def _interruptible_sleep(self, seconds: float) -> None:
        """Sleep that wakes early when shutdown is requested."""
        with contextlib.suppress(asyncio.TimeoutError):
            await asyncio.wait_for(self._shutdown_event.wait(), timeout=seconds)

    async def _background_ingestion_task(self) -> None:
        ingestion_service = IngestionService(
            params=IngestionService.Params(
                semantic_storage=self._semantic_storage,
                resource_retriever=self._set_id_resource,
                history_store=self._episode_storage,
                max_features_per_update=self._max_features_per_update,
                missing_episode_grace_period_sec=self._missing_episode_grace_period_sec,
                debug_fail_loudly=self._debug_fail_loudly,
            ),
        )

        backoff_sec = self._background_ingestion_interval_sec

        while not self._is_shutting_down:
            dirty_sets = [
                s
                async for s in self._semantic_storage.get_history_set_ids(
                    min_uningested_messages=self._feature_update_message_limit,
                    older_than=(await self._semantic_storage.get_storage_time())
                    - self._feature_time_limit,
                )
            ]

            if len(dirty_sets) == 0:
                # Reset backoff so prior error-induced backoff doesn't carry
                # over across long idle periods.
                backoff_sec = self._background_ingestion_interval_sec
                await self._interruptible_sleep(self._background_ingestion_interval_sec)
                continue

            had_errors = False
            try:
                await ingestion_service.process_set_ids(dirty_sets)
            except Exception:
                if self._debug_fail_loudly:
                    raise
                had_errors = True
                logger.exception(
                    "background ingestion failed, backing off %.1fs", backoff_sec
                )

            # Always attempt purge — purge_ingested_rows only deletes rows
            # for fully-ingested sets, so partially-failed sets are skipped
            # and their duplicate guard is preserved.
            purged = await self._semantic_storage.purge_ingested_rows(dirty_sets)
            if purged:
                logger.info(
                    "purged %d ingested rows for %d sets", purged, len(dirty_sets)
                )

            if had_errors:
                await self._interruptible_sleep(backoff_sec)
                backoff_sec = min(backoff_sec * 2, 60.0)
            else:
                backoff_sec = self._background_ingestion_interval_sec
                await self._interruptible_sleep(self._background_ingestion_interval_sec)
