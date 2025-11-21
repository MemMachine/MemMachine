"""Abstract storage interface for episodic history."""

from abc import ABC, abstractmethod

from pydantic import AwareDatetime, JsonValue

from memmachine.episode_store.episode_model import Episode, EpisodeIdT, EpisodeType


class EpisodeStorage(ABC):
    """Abstract interface for persisting and retrieving episodic history."""

    @abstractmethod
    async def add_episode(
        self,
        *,
        content: str,
        session_key: str,
        producer_id: str,
        producer_role: str,
        produced_for_id: str | None = None,
        episode_type: EpisodeType | None = None,
        metadata: dict[str, JsonValue] | None = None,
        created_at: AwareDatetime | None = None,
        uid: str | None = None,
    ) -> EpisodeIdT:
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
    async def delete_episode(
        self,
        history_ids: list[EpisodeIdT],
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
