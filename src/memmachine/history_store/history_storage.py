from abc import ABC, abstractmethod

from pydantic import AwareDatetime

from memmachine.episodic_memory.data_types import EpisodeType
from memmachine.history_store.history_model import HistoryIdT, HistoryMessage


class HistoryStorage(ABC):
    @abstractmethod
    async def add_history(
        self,
        *,
        content: str,
        session_key: str,
        producer_id: str,
        producer_role: str,
        produced_for_id: str | None = None,
        episode_type: EpisodeType | None = None,
        metadata: dict[str, str] | None = None,
        created_at: AwareDatetime | None = None,
    ) -> HistoryIdT:
        raise NotImplementedError

    @abstractmethod
    async def get_history(
        self,
        history_id: HistoryIdT,
    ) -> HistoryMessage | None:
        raise NotImplementedError

    @abstractmethod
    async def get_history_messages(
        self,
        *,
        session_keys: list[str] | None = None,
        producer_ids: list[str] | None = None,
        producer_roles: list[str] | None = None,
        produced_for_ids: list[str] | None = None,
        episode_types: list[EpisodeType] | None = None,
        start_time: AwareDatetime | None = None,
        end_time: AwareDatetime | None = None,
        metadata: dict[str, str] | None = None,
    ) -> list[HistoryMessage]:
        raise NotImplementedError

    @abstractmethod
    async def delete_history(
        self,
        history_ids: list[HistoryIdT],
    ):
        raise NotImplementedError

    @abstractmethod
    async def delete_history_messages(
        self,
        *,
        session_keys: list[str] | None = None,
        producer_ids: list[str] | None = None,
        producer_roles: list[str] | None = None,
        produced_for_ids: list[str] | None = None,
        episode_types: list[EpisodeType] | None = None,
        start_time: AwareDatetime | None = None,
        end_time: AwareDatetime | None = None,
        metadata: dict[str, str] | None = None,
    ):
        raise NotImplementedError
