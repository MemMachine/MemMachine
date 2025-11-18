"""Manage semantic memory sessions and associated lifecycle hooks."""

import asyncio
from typing import Any

from memmachine.episode_store.episode_model import EpisodeIdT
from memmachine.semantic_memory.semantic_memory import SemanticService
from memmachine.semantic_memory.semantic_model import FeatureIdT, SemanticFeature
from memmachine.semantic_memory.semantic_session_resource import (
    ALL_MEMORY_TYPES,
    IsolationType,
    SessionData,
)


class SemanticSessionManager:
    """
    Maps high-level session operations onto set_ids managed by `SemanticService`.

    The manager persists conversation history, resolves the relevant set_ids from
    `SessionData`, and dispatches calls to `SemanticService`.
    """

    def __init__(
        self,
        semantic_service: SemanticService,
    ) -> None:
        """Initialize the manager with the underlying semantic service."""
        self._semantic_service: SemanticService = semantic_service

    async def add_message(
        self,
        episode_ids: list[EpisodeIdT],
        session_data: SessionData,
        memory_type: list[IsolationType] = ALL_MEMORY_TYPES,
    ) -> None:
        if len(episode_ids) == 0:
            return

        set_ids = self._get_set_ids(session_data, memory_type)

        if len(episode_ids) == 1:
            await self._semantic_service.add_message_to_sets(episode_ids[0], set_ids)
            return

        tasks = [
            self._semantic_service.add_messages(s_id, episode_ids) for s_id in set_ids
        ]

        await asyncio.gather(*tasks)

    async def search(
        self,
        message: str,
        session_data: SessionData,
        *,
        memory_type: list[IsolationType] = ALL_MEMORY_TYPES,
        min_distance: float | None = None,
        category_names: list[str] | None = None,
        tag_names: list[str] | None = None,
        feature_names: list[str] | None = None,
        limit: int | None = None,
        load_citations: bool = False,
    ) -> list[SemanticFeature]:
        set_ids = self._get_set_ids(session_data, memory_type)

        optionals_default_args: dict[str, Any] = {
            "set_ids": set_ids,
            "query": message,
            "category_names": category_names,
            "tag_names": tag_names,
            "feature_names": feature_names,
            "load_citations": load_citations,
        }

        if min_distance is not None:
            optionals_default_args["min_distance"] = min_distance

        if limit is not None:
            optionals_default_args["limit"] = limit

        return await self._semantic_service.search(
            **optionals_default_args,
        )

    async def number_of_uningested_messages(
        self,
        session_data: SessionData,
        *,
        memory_type: list[IsolationType] = ALL_MEMORY_TYPES,
    ) -> int:
        set_ids = self._get_set_ids(session_data, memory_type)

        return await self._semantic_service.number_of_uningested(
            set_ids=set_ids,
        )

    async def add_feature(
        self,
        session_data: SessionData,
        *,
        memory_type: IsolationType,
        category_name: str,
        feature: str,
        value: str,
        tag: str,
        metadata: dict[str, str] | None = None,
        citations: list[EpisodeIdT] | None = None,
    ) -> FeatureIdT:
        set_ids = self._get_set_ids(session_data, [memory_type])
        if len(set_ids) != 1:
            raise ValueError("Invalid set_ids", set_ids)
        set_id = set_ids[0]

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
        memory_type: list[IsolationType] = ALL_MEMORY_TYPES,
        category_names: list[str] | None = None,
        tag_names: list[str] | None = None,
        feature_names: list[str] | None = None,
    ) -> list[SemanticFeature]:
        set_ids = self._get_set_ids(session_data, memory_type)

        return await self._semantic_service.get_set_features(
            SemanticService.FeatureSearchOpts(
                set_ids=set_ids,
                category_names=category_names,
                feature_names=feature_names,
                tags=tag_names,
            ),
        )

    async def delete_feature_set(
        self,
        session_data: SessionData,
        *,
        memory_type: list[IsolationType] = ALL_MEMORY_TYPES,
        category_names: list[str] | None = None,
        feature_names: list[str] | None = None,
        tags: list[str] | None = None,
    ) -> None:
        set_ids = self._get_set_ids(session_data, memory_type)

        search_opts: dict[str, Any] = {
            "set_ids": set_ids,
            "category_names": category_names,
            "feature_names": feature_names,
            "tags": tags,
        }

        return await self._semantic_service.delete_feature_set(
            SemanticService.FeatureSearchOpts(
                **search_opts,
            ),
        )

    @staticmethod
    def _get_set_ids(
        session_data: SessionData,
        isolation_level: list[IsolationType],
    ) -> list[str]:
        s: list[str] = []
        if IsolationType.SESSION in isolation_level:
            session_id = session_data.session_id()
            if session_id is not None:
                s.append(session_id)

        if IsolationType.USER in isolation_level:
            user_id = session_data.user_profile_id()
            if user_id is not None:
                s.append(user_id)

        if IsolationType.ROLE in isolation_level:
            role_id = session_data.role_profile_id()
            if role_id is not None:
                s.append(role_id)

        return s
