"""Manage semantic memory sessions and associated lifecycle hooks."""

import asyncio
import hashlib
import logging
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from enum import Enum
from typing import Protocol, runtime_checkable

from pydantic import JsonValue

from memmachine.common.episode_store import Episode, EpisodeIdT
from memmachine.common.filter.filter_parser import FilterExpr
from memmachine.semantic_memory.semantic_memory import SemanticService
from memmachine.semantic_memory.semantic_model import (
    CategoryIdT,
    FeatureIdT,
    OrgSetIdEntry,
    SemanticFeature,
    SetIdT,
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

    async def add_org_set_id(
        self,
        *,
        org_id: str,
        org_level_set: bool = False,
        metadata_tags: list[str],
    ) -> str: ...

    async def list_org_set_ids(self, *, org_id: str) -> list[OrgSetIdEntry]: ...

    async def delete_org_set_id(self, *, org_set_id: str) -> None: ...


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

    class DefaultType(Enum):
        """Default set_id prefixes used by `SemanticSessionManager`."""

        OrgSet = "org_set"
        ProjectSet = "project_set"
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
        set_ids = await self._get_set_ids_str_from_metadata(
            session_data=session_data,
            metadata=episode.metadata,
        )
        await self._semantic_service.add_message_to_sets(episode.uid, set_ids)

    @staticmethod
    def _assert_session_data_implements_protocol(session_data: SessionData) -> None:
        if not isinstance(session_data, SemanticSessionManager.SessionData):
            raise TypeError(
                "session_data must implement SematicSessionManager.SessionData protocol"
            )

    async def add_message(
        self,
        episodes: list[Episode],
        session_data: SessionData,
    ) -> None:
        self._assert_session_data_implements_protocol(session_data=session_data)
        if len(episodes) == 0:
            return

        episode_ids = [e.uid for e in episodes]
        assert len(episode_ids) == len(set(episodes)), "Episodes must be unique"

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

        await self._semantic_service.delete_messages(set_ids=set_ids)

    async def delete_all_org_messages(
        self,
        session_data: SessionData,
    ) -> None:
        self._assert_session_data_implements_protocol(session_data=session_data)

        set_ids = await self._get_all_set_ids(
            org_id=session_data.org_id,
            project_id=None,
        )

        await self._semantic_service.delete_messages(set_ids=set_ids)

    async def search(
        self,
        message: str,
        session_data: SessionData,
        *,
        min_distance: float | None = None,
        limit: int | None = None,
        load_citations: bool = False,
        search_filter: FilterExpr | None = None,
    ) -> list[SemanticFeature]:
        self._assert_session_data_implements_protocol(session_data=session_data)

        set_ids = await self._get_set_ids_str_from_metadata(
            session_data=session_data,
            metadata=session_data.metadata,
        )

        return await self._semantic_service.search(
            set_ids=set_ids,
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
            set_ids=set_ids,
        )

    async def add_feature(
        self,
        session_data: SessionData,
        *,
        feature_metadata: dict[str, JsonValue] | None = None,
        category_name: str,
        feature: str,
        value: str,
        tag: str,
        citations: list[EpisodeIdT] | None = None,
        set_metadata_keys: list[str],
        is_org_level: bool = False,
    ) -> FeatureIdT:
        self._assert_session_data_implements_protocol(session_data=session_data)

        metadata = (
            {key: session_data.metadata.get(key) for key in set_metadata_keys}
            if session_data.metadata is not None
            else {}
        )
        set_id = self._generate_set_id(
            org_id=session_data.org_id,
            project_id=session_data.project_id if not is_org_level else None,
            metadata=metadata,
        )

        return await self._semantic_service.add_new_feature(
            set_id=set_id,
            category_name=category_name,
            feature=feature,
            value=value,
            tag=tag,
            metadata=feature_metadata,
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
            set_ids=set_ids,
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
            set_ids=set_ids,
            filter_expr=property_filter,
        )

    async def _get_all_set_ids(
        self,
        *,
        org_id: str,
        project_id: str | None = None,
    ) -> list[SetIdT]:
        base_set_id = self._org_set_id(
            org_id=org_id,
            project_id=project_id,
        )

        set_ids = await self._semantic_service.list_set_id_starts_with(
            prefix=base_set_id
        )

        return set_ids

    @dataclass(frozen=True, slots=True)
    class SetIdEntry:
        """Resolved set_id qualifiers for a session message."""

        is_org_level: bool
        tags: Mapping[str, str]

    async def _get_set_id_entries(
        self,
        *,
        session_data: SessionData,
        metadata: Mapping[str, JsonValue] | None = None,
    ) -> list[SetIdEntry]:
        org_set_ids = await self._semantic_config.list_org_set_ids(
            org_id=session_data.org_id
        )

        if metadata is None:
            metadata = {}

        metadata_tags = set(metadata.keys())
        relevant_set_ids = [
            sid for sid in org_set_ids if metadata_tags.issuperset(set(sid.tags))
        ]

        if len(relevant_set_ids) == 0:
            logger.debug("No relevant set ids found for metadata %s", metadata)
            return []

        set_id_entries = [
            SemanticSessionManager.SetIdEntry(
                is_org_level=sid.is_org_level,
                tags={t: str(metadata[t]) for t in sid.tags},
            )
            for sid in relevant_set_ids
        ]

        return set_id_entries

    async def _get_set_ids_str_from_metadata(
        self,
        *,
        session_data: SessionData,
        metadata: Mapping[str, JsonValue] | None,
    ) -> list[SetIdT]:
        set_id_entries = await self._get_set_id_entries(
            session_data=session_data,
            metadata=metadata,
        )

        set_ids = [
            self._generate_set_id(
                org_id=session_data.org_id,
                project_id=session_data.project_id if not sid.is_org_level else None,
                metadata=sid.tags,
            )
            for sid in set_id_entries
        ] + [
            self._generate_set_id(
                org_id=session_data.org_id,
                project_id=session_data.project_id,
                metadata={},
            ),
            self._generate_set_id(
                org_id=session_data.org_id,
                project_id=None,
                metadata={},
            ),
        ]

        return list(set(set_ids))

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

    @staticmethod
    def _generate_set_id(
        *,
        org_id: str,
        project_id: str | None = None,
        metadata: Mapping[str, JsonValue],
    ) -> SetIdT:
        org_project = SemanticSessionManager._org_set_id(
            org_id=org_id, project_id=project_id
        )

        string_tags = [f"{k}_{v}" for k, v in metadata.items()]

        if len(string_tags) == 0:
            if project_id is not None:
                def_type = SemanticSessionManager.DefaultType.ProjectSet
            else:
                def_type = SemanticSessionManager.DefaultType.OrgSet
        else:
            def_type = SemanticSessionManager.DefaultType.OtherSet

        return f"mem_{def_type.value}_{org_project}_{len(metadata)}_{_hash_tag_list(metadata.keys())}__{'_'.join(sorted(string_tags))}"

    @staticmethod
    def get_default_set_id_type(
        set_id: SetIdT,
    ) -> DefaultType:
        for def_type in SemanticSessionManager.DefaultType:
            if set_id.startswith(f"mem_{def_type.value}"):
                return def_type

        raise RuntimeError(f"Invalid set_id: {set_id}")

    async def create_org_set_type(
        self,
        *,
        session_data: SessionData,
        is_org_level: bool = False,
        metadata_tags: list[str],
    ) -> str:
        self._assert_session_data_implements_protocol(session_data=session_data)

        return await self._semantic_config.add_org_set_id(
            org_id=session_data.org_id,
            org_level_set=is_org_level,
            metadata_tags=metadata_tags,
        )

    async def delete_org_set_type(
        self,
        *,
        org_set_id: str,
    ) -> None:
        await self._semantic_config.delete_org_set_id(org_set_id=org_set_id)

    async def list_org_set_types(
        self,
        *,
        session_data: SessionData,
    ) -> list[OrgSetIdEntry]:
        self._assert_session_data_implements_protocol(session_data=session_data)

        return await self._semantic_config.list_org_set_ids(org_id=session_data.org_id)

    async def configure_set(
        self,
        session_data: SessionData,
        *,
        set_metadata_keys: list[str],
        is_org_level: bool = False,
        embedder_name: str | None = None,
        llm_name: str | None = None,
    ) -> None:
        self._assert_session_data_implements_protocol(session_data=session_data)

        metadata = (
            {key: session_data.metadata.get(key) for key in set_metadata_keys}
            if session_data.metadata is not None
            else {}
        )
        set_id = self._generate_set_id(
            org_id=session_data.org_id,
            project_id=session_data.project_id if not is_org_level else None,
            metadata=metadata,
        )

        await self._semantic_service.set_set_id_config(
            set_id=set_id,
            embedder_name=embedder_name,
            llm_name=llm_name,
        )

    async def add_new_category(
        self,
        *,
        session_data: SessionData,
        set_metadata_keys: list[str],
        is_org_level: bool = False,
        category_name: str,
        description: str,
    ) -> CategoryIdT:
        self._assert_session_data_implements_protocol(session_data=session_data)

        metadata = (
            {key: session_data.metadata.get(key) for key in set_metadata_keys}
            if session_data.metadata is not None
            else {}
        )
        set_id = self._generate_set_id(
            org_id=session_data.org_id,
            project_id=session_data.project_id if not is_org_level else None,
            metadata=metadata,
        )

        return await self._semantic_service.add_new_category_to_set_id(
            set_id=set_id,
            category_name=category_name,
            description=description,
        )

    async def disable_default_category(
        self,
        *,
        session_data: SessionData,
        set_metadata_keys: list[str],
        is_org_level: bool = False,
        category_name: str,
    ) -> None:
        self._assert_session_data_implements_protocol(session_data=session_data)

        metadata = (
            {key: session_data.metadata.get(key) for key in set_metadata_keys}
            if session_data.metadata is not None
            else {}
        )
        set_id = self._generate_set_id(
            org_id=session_data.org_id,
            project_id=session_data.project_id if not is_org_level else None,
            metadata=metadata,
        )

        await self._semantic_service.disable_default_category(
            set_id=set_id,
            category_name=category_name,
        )

    async def delete_category_and_its_tags(
        self,
        *,
        category_id: CategoryIdT,
    ) -> None:
        await self._semantic_service.delete_category_and_its_tags(
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
