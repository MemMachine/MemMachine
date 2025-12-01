"""Abstract storage interface for episodic history."""

from abc import ABC, abstractmethod

from pydantic import AwareDatetime

from memmachine.common.episode_store.episode_model import (
    Episode,
    EpisodeEntry,
    EpisodeIdT,
)
from memmachine.common.filter.filter_parser import FilterExpr


class EpisodeStorage(ABC):
    """Abstract interface for persisting and retrieving episodic history."""

    @abstractmethod
    async def startup(self) -> None:
        raise NotImplementedError

    @abstractmethod
    async def add_episodes(
        self,
        session_key: str,
        episodes: list[EpisodeEntry],
    ) -> list[Episode]:
        raise NotImplementedError

    @abstractmethod
    async def get_episode(
        self,
        episode_id: EpisodeIdT,
    ) -> Episode | None:
        raise NotImplementedError

    @abstractmethod
    async def get_episode_messages(
        self,
        *,
        page_size: int | None = None,
        page_num: int | None = None,
        filter_expr: FilterExpr | None = None,
        start_time: AwareDatetime | None = None,
        end_time: AwareDatetime | None = None,
    ) -> list[Episode]:
        raise NotImplementedError

    @abstractmethod
    async def get_episode_messages_count(
        self,
        *,
        filter_expr: FilterExpr | None = None,
        start_time: AwareDatetime | None = None,
        end_time: AwareDatetime | None = None,
    ) -> int:
        raise NotImplementedError

    @abstractmethod
    async def delete_episodes(
        self,
        episode_ids: list[EpisodeIdT],
    ) -> None:
        raise NotImplementedError

    @abstractmethod
    async def delete_episode_messages(
        self,
        *,
        filter_expr: FilterExpr | None = None,
        start_time: AwareDatetime | None = None,
        end_time: AwareDatetime | None = None,
    ) -> None:
        raise NotImplementedError
