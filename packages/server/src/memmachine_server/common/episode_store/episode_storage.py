"""Abstract storage interface for episodic history."""

from abc import ABC, abstractmethod
from collections.abc import Iterable, Sequence
from uuid import UUID

from pydantic import AwareDatetime

from memmachine_server.common.episode_store.episode_model import (
    Episode,
    EpisodeEntry,
)
from memmachine_server.common.filter.filter_parser import FilterExpr


class EpisodeStorage(ABC):
    """Abstract interface for persisting and retrieving episodic history."""

    @abstractmethod
    async def startup(self) -> None:
        raise NotImplementedError

    @abstractmethod
    async def delete_all(self) -> None:
        raise NotImplementedError

    @abstractmethod
    async def reserve_sequence_numbers(self, count: int) -> list[int]:
        """Reserve positive, increasing numbers for a batch of episodes."""
        raise NotImplementedError

    @abstractmethod
    async def add_episodes(
        self,
        session_key: str,
        episodes: list[EpisodeEntry],
        *,
        sequence_nums: Sequence[int] | None = None,
    ) -> list[Episode]:
        raise NotImplementedError

    @abstractmethod
    async def get_episode(
        self,
        episode_id: UUID,
    ) -> Episode | None:
        raise NotImplementedError

    @abstractmethod
    async def get_episodes(
        self,
        episode_ids: Iterable[UUID],
    ) -> list[Episode]:
        """Batch fetch episodes by UID.

        Order of results is not guaranteed; missing UIDs are simply absent
        from the returned list.
        """
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
    async def get_episode_ids(
        self,
        *,
        page_size: int,
        filter_expr: FilterExpr | None = None,
    ) -> list[UUID]:
        raise NotImplementedError

    @abstractmethod
    async def delete_episodes(
        self,
        episode_ids: list[UUID],
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
