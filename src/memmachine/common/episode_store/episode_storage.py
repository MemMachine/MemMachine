"""Abstract storage interface for episodic history."""

from abc import ABC, abstractmethod

from pydantic import AwareDatetime, JsonValue

from memmachine.common.episode_store.episode_model import (
    Episode,
    EpisodeEntry,
    EpisodeIdT,
    EpisodeType,
)


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
        history_id: EpisodeIdT,
    ) -> Episode | None:
        raise NotImplementedError

    @abstractmethod
    async def get_episode_messages(
        self,
        *,
        limit: int | None = None,
        session_keys: list[str] | None = None,
        producer_ids: list[str] | None = None,
        producer_roles: list[str] | None = None,
        produced_for_ids: list[str] | None = None,
        episode_types: list[EpisodeType] | None = None,
        start_time: AwareDatetime | None = None,
        end_time: AwareDatetime | None = None,
        metadata: dict[str, JsonValue] | None = None,
    ) -> list[Episode]:
        raise NotImplementedError

    @abstractmethod
    async def get_episode_messages_count(
        self,
        *,
        session_keys: list[str] | None = None,
        producer_ids: list[str] | None = None,
        producer_roles: list[str] | None = None,
        produced_for_ids: list[str] | None = None,
        episode_types: list[EpisodeType] | None = None,
        start_time: AwareDatetime | None = None,
        end_time: AwareDatetime | None = None,
        metadata: dict[str, JsonValue] | None = None,
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
        session_keys: list[str] | None = None,
        producer_ids: list[str] | None = None,
        producer_roles: list[str] | None = None,
        produced_for_ids: list[str] | None = None,
        episode_types: list[EpisodeType] | None = None,
        start_time: AwareDatetime | None = None,
        end_time: AwareDatetime | None = None,
        metadata: dict[str, JsonValue] | None = None,
    ) -> None:
        raise NotImplementedError
