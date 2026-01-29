"""Manage semantic memory sessions and associated lifecycle hooks."""

import asyncio
import hashlib
import logging
from collections.abc import AsyncIterator, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Protocol, runtime_checkable

from pydantic import BaseModel, JsonValue

from memmachine.common.episode_store import Episode, EpisodeIdT
from memmachine.common.filter.filter_parser import FilterExpr
from memmachine.semantic_memory.config_store.config_store import (
    SemanticConfigStorage as ESemanticConfigStorage,
)
from memmachine.semantic_memory.semantic_memory import SemanticService
from memmachine.semantic_memory.semantic_model import (
    CategoryIdT,
    FeatureIdT,
    SemanticCategory,
    SemanticFeature,
    SetIdT,
    SetTypeEntry,
    TagIdT,
)

logger = logging.getLogger(__name__)


def _hash_tag_list(strings: Iterable[str]) -> str:
    strings = sorted(strings)

    h = hashlib.shake_256()
    for s in strings:
        h.update(s.encode("utf-8"))
        h.update(b"\x00")
    return h.hexdigest(6)


@runtime_checkable
class SemanticConfigStorage(Protocol):
    """Protocol for persisting and retrieving semantic memory configuration."""

    async def add_set_type_id(
        self,
        *,
        org_id: str,
        org_level_set: bool = False,
        metadata_tags: list[str],
        name: str | None = None,
        description: str | None = None,
    ) -> str: ...

    async def list_set_type_ids(self, *, org_id: str) -> list[SetTypeEntry]: ...

    async def delete_set_type_id(self, *, set_type_id: str) -> None: ...

    async def register_set_id_set_type(
        self,
        *,
        set_id: SetIdT,
        set_type_id: str,
    ) -> None: ...


class SemanticSessionManager:
    """
    Maps high-level session operations onto set_ids managed by `SemanticService`.

    The manager persists conversation history, resolves the relevant set_ids from
    `SessionData`, and dispatches calls to `SemanticService`.
    """

    @runtime_checkable
    class SessionData(Protocol):
        """Protocol exposing the identifiers used to derive set_ids."""

        @property
        def org_id(self) -> str: ...

        @property
        def project_id(self) -> str: ...

        @property
        def metadata(self) -> Mapping[str, JsonValue] | None: ...

    class SetType(Enum):
        """Default set_id prefixes used by `SemanticSessionManager`."""

        OrgSet = "set_type"
        ProjectSet = "project_set"
        UserSet = "user_set"
        OtherSet = "other_set"

    def __init__(
        self,
        semantic_service: SemanticService,
        semantic_config_storage: SemanticConfigStorage,
    ) -> None:
        """Initialize the manager with the underlying semantic service."""
        if not isinstance(semantic_service, SemanticService):
            raise TypeError("semantic_service must be a SemanticService")

        if not isinstance(semantic_config_storage, SemanticConfigStorage):
            raise TypeError("semantic_config_storage must be a SemanticConfigStorage")

        self._semantic_service: SemanticService = semantic_service
        self._semantic_config: SemanticConfigStorage = semantic_config_storage

    async def _add_single_episode(
        self, episode: Episode, session_data: SessionData
    ) -> None:
        episode_metadata: dict[str, JsonValue] = (
            dict(episode.metadata) if episode.metadata is not None else {}
        )

        episode_metadata.setdefault("producer_id", episode.producer_id)

        set_ids = await self._get_set_ids_str_from_metadata(
            session_data=session_data,
            metadata=episode_metadata,
        )
        await self._semantic_service.add_message_to_sets(episode.uid, list(set_ids))

    @staticmethod
    def _assert_session_data_implements_protocol(session_data: SessionData) -> None:
        if not isinstance(session_data, SemanticSessionManager.SessionData):
            raise TypeError(
                "session_data must implement SemanticSessionManager.SessionData protocol"
            )

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
            for e in episodes:
                tg.create_task(self._add_single_episode(e, session_data))

    async def delete_all_project_messages(
        self,
        session_data: SessionData,
    ) -> None:
        self._assert_session_data_implements_protocol(session_data=session_data)

        set_ids = await self._get_all_set_ids(
            org_id=session_data.org_id,
            project_id=session_data.project_id,
        )

        await self._semantic_service.delete_messages(set_ids=list(set_ids))

    async def delete_all_org_messages(
        self,
        session_data: SessionData,
    ) -> None:
        self._assert_session_data_implements_protocol(session_data=session_data)

        set_ids = await self._get_all_set_ids(
            org_id=session_data.org_id,
            project_id=None,
        )

        await self._semantic_service.delete_messages(set_ids=list(set_ids))

    async def search(
        self,
        message: str,
        session_data: SessionData,
        *,
        min_distance: float | None = None,
        limit: int | None = None,
        load_citations: bool = False,
        search_filter: FilterExpr | None = None,
    ) -> Iterable[SemanticFeature]:
        self._assert_session_data_implements_protocol(session_data=session_data)

        set_ids = await self._get_set_ids_str_from_metadata(
            session_data=session_data,
            metadata=session_data.metadata,
        )

        return await self._semantic_service.search(
            set_ids=list(set_ids),
            query=message,
            min_distance=min_distance,
            limit=limit,
            load_citations=load_citations,
            filter_expr=search_filter,
        )

    async def number_of_uningested_messages(
        self,
        session_data: SessionData,
    ) -> int:
        self._assert_session_data_implements_protocol(session_data=session_data)

        set_ids = await self._get_set_ids_str_from_metadata(
            session_data=session_data,
            metadata=session_data.metadata,
        )

        return await self._semantic_service.number_of_uningested(
            set_ids=list(set_ids),
        )

    async def add_feature(
        self,
        *,
        set_id: SetIdT,
        feature_metadata: dict[str, JsonValue] | None = None,
        category_name: str,
        feature: str,
        value: str,
        tag: str,
        citations: list[EpisodeIdT] | None = None,
    ) -> FeatureIdT:
        metadata = (
            {k: str(v) for k, v in feature_metadata.items()}
            if feature_metadata is not None
            else None
        )

        return await self._semantic_service.add_new_feature(
            set_id=set_id,
            category_name=category_name,
            feature=feature,
            value=value,
            tag=tag,
            metadata=metadata,
            citations=citations,
        )

    async def get_feature(
        self,
        feature_id: FeatureIdT,
        load_citations: bool = False,
    ) -> SemanticFeature | None:
        return await self._semantic_service.get_feature(
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
        metadata: dict[str, str] | None = None,
    ) -> None:
        await self._semantic_service.update_feature(
            feature_id,
            category_name=category_name,
            feature=feature,
            value=value,
            tag=tag,
            metadata=metadata,
        )

    async def delete_features(self, feature_ids: list[FeatureIdT]) -> None:
        await self._semantic_service.delete_features(feature_ids)

    async def get_set_features(
        self,
        session_data: SessionData,
        *,
        search_filter: FilterExpr | None = None,
        page_size: int | None = None,
        page_num: int | None = None,
        load_citations: bool = False,
    ) -> list[SemanticFeature]:
        self._assert_session_data_implements_protocol(session_data=session_data)

        set_ids = await self._get_set_ids_str_from_metadata(
            session_data=session_data,
            metadata=session_data.metadata,
        )

        return await self._semantic_service.get_set_features(
            set_ids=list(set_ids),
            filter_expr=search_filter,
            page_size=page_size,
            page_num=page_num,
            with_citations=load_citations,
        )

    async def delete_feature_set(
        self,
        session_data: SessionData,
        *,
        property_filter: FilterExpr | None = None,
    ) -> None:
        self._assert_session_data_implements_protocol(session_data=session_data)

        set_ids = await self._get_set_ids_str_from_metadata(
            session_data=session_data,
            metadata=session_data.metadata,
        )

        await self._semantic_service.delete_feature_set(
            set_ids=list(set_ids),
            filter_expr=property_filter,
        )

    async def _get_all_set_ids(
        self,
        *,
        org_id: str,
        project_id: str | None = None,
    ) -> Iterable[SetIdT]:
        base_set_id = self._org_set_id(
            org_id=org_id,
            project_id=project_id,
        )

        set_ids = await self._semantic_service.list_set_id_starts_with(
            prefix=base_set_id
        )

        return set_ids

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

        set_type_ids = await self._semantic_config.list_set_type_ids(
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

    async def _get_set_ids_str_from_metadata(
        self,
        *,
        session_data: SessionData,
        metadata: Mapping[str, JsonValue] | None,
    ) -> Iterable[SetIdT]:
        normalized_metadata = self._normalize_metadata(metadata)

        deduped: dict[SetIdT, None] = {}

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
                await self._semantic_config.register_set_id_set_type(
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

    @staticmethod
    def _normalize_metadata(
        metadata: Mapping[str, JsonValue] | None,
    ) -> dict[str, str]:
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
                def_type = SemanticSessionManager.SetType.ProjectSet
            else:
                def_type = SemanticSessionManager.SetType.OrgSet
        elif metadata_keys == {"producer_id"}:
            def_type = SemanticSessionManager.SetType.UserSet
        else:
            def_type = SemanticSessionManager.SetType.OtherSet

        return f"mem_{def_type.value}_{org_project}_{len(metadata)}_{_hash_tag_list(metadata.keys())}__{'_'.join(sorted(string_tags))}"

    @classmethod
    def generate_user_set_id(
        cls,
        *,
        org_id: str,
        producer_id: str,
    ) -> SetIdT:
        return cls._generate_set_id(
            org_id=org_id,
            project_id=None,
            metadata={"producer_id": producer_id},
        )

    @staticmethod
    def get_default_set_id_type(
        set_id: SetIdT,
    ) -> SetType:
        for def_type in SemanticSessionManager.SetType:
            if set_id.startswith(f"mem_{def_type.value}"):
                return def_type

        raise RuntimeError(f"Invalid set_id: {set_id}")

    async def _ensure_user_set_type(self, *, org_id: str) -> None:
        set_types = await self._semantic_config.list_set_type_ids(org_id=org_id)
        for set_type in set_types:
            if set_type.is_org_level and set(set_type.tags) == {"producer_id"}:
                return

        await self._semantic_config.add_set_type_id(
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
        metadata_tags: list[str],
        name: str | None = None,
        description: str | None = None,
    ) -> str:
        self._assert_session_data_implements_protocol(session_data=session_data)

        return await self._semantic_config.add_set_type_id(
            org_id=session_data.org_id,
            org_level_set=is_org_level,
            metadata_tags=metadata_tags,
            name=name,
            description=description,
        )

    async def delete_set_type(
        self,
        *,
        set_type_id: str,
    ) -> None:
        await self._semantic_config.delete_set_type_id(set_type_id=set_type_id)

    async def list_set_types(
        self,
        *,
        session_data: SessionData,
    ) -> Iterable[SetTypeEntry]:
        self._assert_session_data_implements_protocol(session_data=session_data)

        return await self._semantic_config.list_set_type_ids(org_id=session_data.org_id)

    async def configure_set(
        self,
        *,
        set_id: SetIdT,
        embedder_name: str | None = None,
        llm_name: str | None = None,
    ) -> None:
        await self._semantic_service.set_set_id_config(
            set_id=set_id,
            embedder_name=embedder_name,
            llm_name=llm_name,
        )

    async def get_set_id_category_names(self, *, set_id: SetIdT) -> Iterable[str]:
        return await self._semantic_service.get_set_id_category_names(set_id=set_id)

    async def get_set_id_config(
        self,
        *,
        set_id: SetIdT,
    ) -> ESemanticConfigStorage.Config | None:
        return await self._semantic_service.get_set_id_config(
            set_id=set_id,
        )

    async def get_category(
        self,
        *,
        category_id: CategoryIdT,
    ) -> ESemanticConfigStorage.Category | None:
        return await self._semantic_service.get_category(category_id=category_id)

    async def get_category_set_ids(
        self,
        *,
        category_id: CategoryIdT,
    ) -> Iterable[SetIdT]:
        return await self._semantic_service.get_category_set_ids(
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
        return await self._semantic_service.add_new_category_to_set_id(
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
        return await self._semantic_service.add_new_category_to_set_type(
            set_type_id=set_type_id,
            category_name=category_name,
            prompt=prompt,
            description=description,
        )

    async def list_category_templates(
        self, *, set_type_id: str
    ) -> Iterable[SemanticCategory]:
        return await self._semantic_service.get_set_type_categories(
            set_type_id=set_type_id
        )

    async def get_set_id(
        self,
        *,
        session_data: SessionData,
        set_metadata_keys: list[str],
        is_org_level: bool = False,
    ) -> SetIdT:
        self._assert_session_data_implements_protocol(session_data=session_data)

        metadata = (
            {key: session_data.metadata.get(key) for key in set_metadata_keys}
            if session_data.metadata is not None
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

        set_type_ids = await self._semantic_config.list_set_type_ids(
            org_id=session_data.org_id
        )
        for set_type in set_type_ids:
            if set_type.id is None:
                continue

            if set_type.is_org_level != is_org_level:
                continue

            if set(set_type.tags) != set(set_metadata_keys):
                continue

            await self._semantic_config.register_set_id_set_type(
                set_id=set_id,
                set_type_id=set_type.id,
            )
            break

        return set_id

    class Set(BaseModel):
        """Semantic memory set, containing a set of features and resources."""

        id: SetIdT
        is_org_level: bool
        tags: list[str]
        name: str | None = None
        description: str | None = None

    async def list_sets(
        self,
        *,
        session_data: SessionData,
    ) -> Iterable[Set]:
        self._assert_session_data_implements_protocol(session_data=session_data)

        normalized_metadata = self._normalize_metadata(session_data.metadata)

        set_type_ids = await self._semantic_config.list_set_type_ids(
            org_id=session_data.org_id
        )
        set_type_map = {sid.id: sid for sid in set_type_ids if sid.id is not None}

        sets: dict[str, SemanticSessionManager.Set] = {}
        async for sid in self._get_set_id_entries(
            session_data=session_data,
            metadata=session_data.metadata,
        ):
            set_id = self._generate_set_id(
                org_id=session_data.org_id,
                project_id=session_data.project_id if not sid.is_org_level else None,
                metadata=sid.tags,
            )

            set_type_entry = set_type_map.get(sid.set_type_id)
            sets.setdefault(
                set_id,
                SemanticSessionManager.Set(
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

    async def list_set_ids(
        self,
        *,
        session_data: SessionData,
    ) -> Iterable[Set]:
        """Backward compatible alias preserving list semantics."""
        return await self.list_sets(session_data=session_data)

    @classmethod
    def _generate_default_sets(
        cls,
        *,
        org_id: str,
        project_id: str | None,
        producer_id: str | None,
    ) -> Iterator[Set]:
        yield SemanticSessionManager.Set(
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
            yield SemanticSessionManager.Set(
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
            yield SemanticSessionManager.Set(
                id=cls._generate_set_id(
                    org_id=org_id,
                    project_id=project_id,
                    metadata={},
                ),
                is_org_level=False,
                tags=[],
            )

    async def disable_category(
        self,
        *,
        set_id: SetIdT,
        category_name: str,
    ) -> None:
        await self._semantic_service.disable_category(
            set_id=set_id,
            category_name=category_name,
        )

    async def delete_category(
        self,
        *,
        category_id: CategoryIdT,
    ) -> None:
        await self._semantic_service.delete_category(
            category_id=category_id,
        )

    async def add_tag(
        self,
        *,
        category_id: CategoryIdT,
        tag_name: str,
        tag_description: str,
    ) -> TagIdT:
        return await self._semantic_service.add_tag(
            category_id=category_id,
            tag_name=tag_name,
            tag_description=tag_description,
        )

    async def update_tag(
        self,
        *,
        tag_id: TagIdT,
        tag_name: str,
        tag_description: str,
    ) -> None:
        await self._semantic_service.update_tag(
            tag_id=tag_id,
            tag_name=tag_name,
            tag_description=tag_description,
        )

    async def delete_tag(self, *, tag_id: TagIdT) -> None:
        await self._semantic_service.delete_tag(tag_id=tag_id)
