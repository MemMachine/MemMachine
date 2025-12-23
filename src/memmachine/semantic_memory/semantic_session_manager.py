"""Manage semantic memory sessions and associated lifecycle hooks."""

import asyncio
from hashlib import sha256
from typing import Protocol, runtime_checkable

from pydantic import InstanceOf, JsonValue

from memmachine.common.episode_store import Episode, EpisodeIdT
from memmachine.common.filter.filter_parser import FilterExpr
from memmachine.semantic_memory.config_store.config_store import SemanticConfigStorage
from memmachine.semantic_memory.semantic_memory import SemanticService
from memmachine.semantic_memory.semantic_model import (
    FeatureIdT,
    OrgSetIdEntry,
    SemanticFeature,
    SetIdT,
)


def _hash_tag_list(strings: list[str]) -> str:
    strings = sorted(strings)

    h = sha256()
    for s in strings:
        h.update(s.encode("utf-8"))
        h.update(b"\x00")
    return h.hexdigest()


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

    def __init__(
        self,
        semantic_service: SemanticService,
        semantic_config_storage: SemanticConfigStorage,
    ) -> None:
        """Initialize the manager with the underlying semantic service."""
        self._semantic_service: SemanticService = semantic_service
        self._semantic_config: SemanticConfigStorage = semantic_config_storage

    async def _add_single_episode(
        self, episode: Episode, session_data: SessionData
    ) -> None:
        set_ids = await self._get_set_ids_from_metadata(
            session_data=session_data, metadata=episode.metadata
        )
        await self._semantic_service.add_message_to_sets(episode.id, set_ids)

    async def add_message(
        self,
        episodes: list[Episode],
        session_data: InstanceOf[SessionData],
    ) -> None:
        if len(episodes) == 0:
            return

        async with asyncio.TaskGroup() as tg:
            for e in episodes:
                tg.create_task(self._add_single_episode(e, session_data))

    async def delete_all_messages(
        self,
        session_data: SessionData,
    ) -> None:
        set_ids = await self._get_all_set_ids(session_data=session_data)

        await self._semantic_service.delete_messages(set_ids=set_ids)

    async def search(
        self,
        message: str,
        session_data: SessionData,
        *,
        metadata: dict[str, JsonValue] | None = None,
        min_distance: float | None = None,
        limit: int | None = None,
        load_citations: bool = False,
        search_filter: FilterExpr | None = None,
    ) -> list[SemanticFeature]:
        set_ids = await self._get_set_ids_from_metadata(
            session_data=session_data, metadata=metadata
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
        *,
        metadata: dict[str, JsonValue] | None = None,
    ) -> int:
        set_ids = await self._get_set_ids_from_metadata(
            session_data=session_data, metadata=metadata
        )

        return await self._semantic_service.number_of_uningested(
            set_ids=set_ids,
        )

    async def add_feature(
        self,
        session_data: SessionData,
        *,
        feature_metadata: dict[str, JsonValue] | None = None,
        session_metadata: dict[str, JsonValue] | None = None,
        category_name: str,
        feature: str,
        value: str,
        tag: str,
        citations: list[EpisodeIdT] | None = None,
    ) -> FeatureIdT:
        set_ids = await self._get_set_ids_from_metadata(session_data=session_data, metadata=session_metadata)

        if len(set_ids) != 1:
            raise ValueError("Invalid set_ids", set_ids)
        set_id = set_ids[0]

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
        metadata: dict[str, JsonValue] | None = None,
        page_size: int | None = None,
        page_num: int | None = None,
        load_citations: bool = False,
    ) -> list[SemanticFeature]:
        set_ids = await self._get_set_ids_from_metadata(
            session_data=session_data,
            metadata=metadata,
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
        metadata: dict[str, JsonValue] | None = None,
        property_filter: FilterExpr | None = None,
    ) -> None:
        set_ids = await self._get_set_ids_from_metadata(
            session_data=session_data,
            metadata=metadata,
        )

        await self._semantic_service.delete_feature_set(
            set_ids=set_ids,
            filter_expr=property_filter,
        )

    async def _get_all_set_ids(
        self,
        *,
        session_data: SessionData,
    ) -> list[SetIdT]:
        org_set_ids = await self._semantic_config.list_org_set_ids(
            org_id=session_data.org_id
        )

        return self._get_set_ids(session_data, org_set_ids)

    async def _get_set_ids_from_metadata(
        self,
        *,
        session_data: SessionData,
        metadata: dict[str, JsonValue] | None,
    ) -> list[SetIdT]:
        org_set_ids = await self._semantic_config.list_org_set_ids(
            org_id=session_data.org_id
        )

        metadata_tags = set(metadata.keys() if metadata else [])
        relevant_set_ids = [
            sid for sid in org_set_ids if metadata_tags.issuperset(set(sid.tags))
        ]

        return self._get_set_ids(session_data, relevant_set_ids)

    def _get_set_ids(
        self, session_data: SessionData, org_set_entries: list[OrgSetIdEntry]
    ) -> list[SetIdT]:
        org_base = f"org_{session_data.org_id}"
        org_project = f"{org_base}_project_{session_data.project_id}"

        set_ids = [
            f"mem_{(org_base if sid.is_org_level else org_project)}_{_hash_tag_list(sid.tags)}_{'_'.join(sorted(sid.tags))}"
            for sid in org_set_entries
        ]

        return set_ids
