"""Manage semantic memory sessions and associated lifecycle hooks."""

import asyncio
from enum import Enum
from typing import Final, Protocol, runtime_checkable

from pydantic import InstanceOf

from memmachine.common.episode_store import EpisodeIdT
from memmachine.common.filter.filter_parser import And, Comparison, FilterExpr
from memmachine.semantic_memory.semantic_memory import SemanticService
from memmachine.semantic_memory.semantic_model import (
    FeatureIdT,
    SemanticFeature,
    SetIdT,
)

_SESSION_ID_PREFIX: Final[str] = "mem_session_"
_USER_ID_PREFIX: Final[str] = "mem_user_"
_ROLE_ID_PREFIX: Final[str] = "mem_role_"


class IsolationType(Enum):
    """Isolation scopes supported when mapping activity to semantic-memory set_ids."""

    USER = "user_profile"
    ROLE = "role_profile"
    SESSION = "session"


ALL_MEMORY_TYPES: Final[list[IsolationType]] = list(IsolationType)


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
        def user_profile_id(self) -> SetIdT | None:
            raise NotImplementedError

        @property
        def session_id(self) -> SetIdT | None:
            raise NotImplementedError

        @property
        def role_profile_id(self) -> SetIdT | None:
            raise NotImplementedError

    def _generate_session_data(
        self,
        *,
        session_data: SessionData,
    ) -> SessionData:
        class _SessionDataImpl:
            """Lightweight `SessionData` implementation backed by generated set_ids."""

            def __init__(
                self,
                *,
                _user_profile_id: str | None,
                _role_profile_id: str | None,
                _session_id: str | None,
            ) -> None:
                """Capture the identifiers used to scope memory resources."""
                self._user_id: str | None = _user_profile_id
                self._role_id: str | None = _role_profile_id
                self._session_id: str | None = _session_id

            @property
            def user_profile_id(self) -> str | None:
                return self._user_id

            @property
            def role_profile_id(self) -> str | None:
                return self._role_id

            @property
            def session_id(self) -> str | None:
                return self._session_id

        user_id = session_data.user_profile_id
        role_id = session_data.role_profile_id
        session_id = session_data.session_id

        return _SessionDataImpl(
            _user_profile_id=_USER_ID_PREFIX + user_id if user_id else None,
            _session_id=_SESSION_ID_PREFIX + session_id if session_id else None,
            _role_profile_id=_ROLE_ID_PREFIX + role_id if role_id else None,
        )

    @classmethod
    def set_id_isolation_type(cls, set_id: SetIdT) -> IsolationType:
        if cls.is_session_id(set_id):
            return IsolationType.SESSION
        if cls.is_producer_id(set_id):
            return IsolationType.USER
        if cls.is_role_id(set_id):
            return IsolationType.ROLE
        raise ValueError(f"Invalid id: {set_id}")

    @staticmethod
    def is_session_id(_id: str) -> bool:
        return _id.startswith(_SESSION_ID_PREFIX)

    @staticmethod
    def is_producer_id(_id: str) -> bool:
        return _id.startswith(_USER_ID_PREFIX)

    @staticmethod
    def is_role_id(_id: str) -> bool:
        return _id.startswith(_ROLE_ID_PREFIX)

    def __init__(
        self,
        semantic_service: SemanticService,
    ) -> None:
        """Initialize the manager with the underlying semantic service."""
        self._semantic_service: SemanticService = semantic_service

    async def add_message(
        self,
        episode_ids: list[EpisodeIdT],
        session_data: InstanceOf[SessionData],
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

    async def delete_messages(
        self,
        session_data: SessionData,
        *,
        memory_type: list[IsolationType] = ALL_MEMORY_TYPES,
    ) -> None:
        set_ids = self._get_set_ids(session_data, memory_type)

        await self._semantic_service.delete_messages(set_ids=set_ids)

    async def search(
        self,
        message: str,
        session_data: SessionData,
        *,
        memory_type: list[IsolationType] = ALL_MEMORY_TYPES,
        min_distance: float | None = None,
        limit: int | None = None,
        load_citations: bool = False,
        search_filter: FilterExpr | None = None,
    ) -> list[SemanticFeature]:
        set_ids = self._get_set_ids(session_data, memory_type)
        filter_expr = self._merge_filters(set_ids, search_filter)

        return await self._semantic_service.search(
            set_ids=set_ids,
            query=message,
            min_distance=min_distance,
            limit=limit,
            load_citations=load_citations,
            filter_expr=filter_expr,
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
        search_filter: FilterExpr | None = None,
        page_size: int | None = None,
        page_num: int | None = None,
        load_citations: bool = False,
    ) -> list[SemanticFeature]:
        set_ids = self._get_set_ids(session_data, memory_type)
        filter_expr = self._merge_filters(set_ids, search_filter)

        return await self._semantic_service.get_set_features(
            filter_expr=filter_expr,
            page_size=page_size,
            page_num=page_num,
            with_citations=load_citations,
        )

    async def delete_feature_set(
        self,
        session_data: SessionData,
        *,
        memory_type: list[IsolationType] = ALL_MEMORY_TYPES,
        property_filter: FilterExpr | None = None,
    ) -> None:
        set_ids = self._get_set_ids(session_data, memory_type)
        filter_expr = self._merge_filters(set_ids, property_filter)

        await self._semantic_service.delete_feature_set(
            filter_expr=filter_expr,
        )

    def _get_set_ids(
        self,
        session_data: SessionData,
        isolation_level: list[IsolationType],
    ) -> list[str]:
        session_data = self._generate_session_data(session_data=session_data)

        s: list[str] = []
        if IsolationType.SESSION in isolation_level:
            session_id = session_data.session_id
            if session_id is not None:
                s.append(session_id)

        if IsolationType.USER in isolation_level:
            user_id = session_data.user_profile_id
            if user_id is not None:
                s.append(user_id)

        if IsolationType.ROLE in isolation_level:
            role_id = session_data.role_profile_id
            if role_id is not None:
                s.append(role_id)

        return s

    @staticmethod
    def _merge_filters(
        set_ids: list[str],
        property_filter: FilterExpr | None,
    ) -> FilterExpr | None:
        expr = property_filter
        if set_ids:
            set_expr = Comparison(field="set_id", op="in", value=list(set_ids))
            expr = set_expr if expr is None else And(left=set_expr, right=expr)
        return expr
