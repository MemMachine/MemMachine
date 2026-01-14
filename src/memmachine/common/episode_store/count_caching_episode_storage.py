"""EpisodeStorage wrapper adding an in-memory count cache."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import cast

from pydantic import AwareDatetime

from memmachine.common.episode_store.episode_model import (
    Episode,
    EpisodeEntry,
    EpisodeIdT,
)
from memmachine.common.episode_store.episode_storage import EpisodeStorage
from memmachine.common.filter.filter_parser import Comparison, FilterExpr


@dataclass(frozen=True)
class _CacheEntry:
    count: int


def _session_key_from_filter(filter_expr: FilterExpr | None) -> str | None:
    """Return the session_key if the filter is exactly a session equality."""
    if not isinstance(filter_expr, Comparison):
        return None
    if filter_expr.op != "=":
        return None
    if filter_expr.field not in {"session_key", "session"}:
        return None
    if not isinstance(filter_expr.value, str):
        return None
    return filter_expr.value


class CountCachingEpisodeStorage(EpisodeStorage):
    """
    Incoherent count cache as an EpisodeStorage decorator.

    As an incoherent cache, changes to the underlying store are not reflected in the cache.
    Making this cache unsuited to concurrent deployment.
    """

    def __init__(self, wrapped: EpisodeStorage) -> None:
        """Initialize the decorator with a wrapped EpisodeStorage."""
        self._wrapped = wrapped
        self._count_cache: dict[str, _CacheEntry] = {}
        self._lock = asyncio.Lock()

    async def startup(self) -> None:
        await self._wrapped.startup()

    async def add_episodes(
        self,
        session_key: str,
        episodes: list[EpisodeEntry],
    ) -> list[Episode]:
        stored = await self._wrapped.add_episodes(session_key, episodes)

        async with self._lock:
            entry = self._count_cache.get(session_key)
            if entry is not None:
                self._count_cache[session_key] = _CacheEntry(
                    count=entry.count + len(episodes),
                )

        return stored

    async def get_episode(
        self,
        episode_id: EpisodeIdT,
    ) -> Episode | None:
        return await self._wrapped.get_episode(episode_id)

    async def get_episode_messages(
        self,
        *,
        page_size: int | None = None,
        page_num: int | None = None,
        filter_expr: FilterExpr | None = None,
        start_time: AwareDatetime | None = None,
        end_time: AwareDatetime | None = None,
    ) -> list[Episode]:
        return await self._wrapped.get_episode_messages(
            page_size=page_size,
            page_num=page_num,
            filter_expr=filter_expr,
            start_time=start_time,
            end_time=end_time,
        )

    async def get_episode_messages_count(
        self,
        *,
        filter_expr: FilterExpr | None = None,
        start_time: AwareDatetime | None = None,
        end_time: AwareDatetime | None = None,
    ) -> int:
        session_key = _session_key_from_filter(filter_expr)

        searching_only_session_key = (
            session_key is not None and start_time is None and end_time is None
        )

        if searching_only_session_key:
            session_key = cast(str, session_key)

            async with self._lock:
                entry = self._count_cache.get(session_key)
                if entry is not None:
                    return entry.count

        count = await self._wrapped.get_episode_messages_count(
            filter_expr=filter_expr,
            start_time=start_time,
            end_time=end_time,
        )

        if searching_only_session_key:
            session_key = cast(str, session_key)

            async with self._lock:
                self._count_cache[session_key] = _CacheEntry(
                    count=count,
                )

        return count

    async def delete_episodes(
        self,
        episode_ids: list[EpisodeIdT],
    ) -> None:
        await self._wrapped.delete_episodes(episode_ids)
        await self._clear_cache()

    async def delete_episode_messages(
        self,
        *,
        filter_expr: FilterExpr | None = None,
        start_time: AwareDatetime | None = None,
        end_time: AwareDatetime | None = None,
    ) -> None:
        await self._wrapped.delete_episode_messages(
            filter_expr=filter_expr,
            start_time=start_time,
            end_time=end_time,
        )
        await self._clear_cache()

    async def _clear_cache(self) -> None:
        async with self._lock:
            self._count_cache.clear()
