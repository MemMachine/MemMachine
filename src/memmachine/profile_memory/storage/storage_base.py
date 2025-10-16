from abc import ABC, abstractmethod
from typing import Any, Mapping

import numpy as np


class ProfileStorageBase(ABC):
    """
    The base class for profile storage
    """

    @abstractmethod
    async def startup(self):
        """
        initializations for the profile storage,
        such as creating connection to the database
        """
        raise NotImplementedError

    @abstractmethod
    async def cleanup(self):
        """
        cleanup for the profile storage
        such as closing connection to the database
        """
        raise NotImplementedError

    @abstractmethod
    async def delete_all(self):
        """
        delete all profiles in the storage
        such as truncating the database table
        """
        raise NotImplementedError

    @abstractmethod
    async def get_profile(
        self,
        user_id: str,
        isolations: dict[str, bool | int | float | str] | None = None,
    ) -> dict[str, Any]:
        """
        Get profile by id
        Return: A list of KV for eatch feature and value.
           The value is an array with: feature value, feature tag and deleted, update time, create time and delete time.
        """
        raise NotImplementedError

    @abstractmethod
    async def delete_profile(
        self,
        user_id: str,
        isolations: dict[str, bool | int | float | str] | None = None,
    ):
        """
        Delete all the profile by id
        """
        raise NotImplementedError

    @abstractmethod
    async def add_profile_feature(
        self,
        user_id: str,
        feature: str,
        value: str,
        tag: str,
        embedding: np.ndarray,
        metadata: dict[str, Any] | None = None,
        isolations: dict[str, bool | int | float | str] | None = None,
        citations: list[int] | None = None,
    ):
        """
        Add a new feature to the profile.
        """
        raise NotImplementedError

    @abstractmethod
    async def semantic_search(
        self,
        user_id: str,
        qemb: np.ndarray,
        k: int,
        min_cos: float,
        isolations: dict[str, bool | int | float | str] | None = None,
        include_citations: bool = False,
    ) -> list[dict[str, Any]]:
        raise NotImplementedError

    @abstractmethod
    async def delete_profile_feature_by_id(self, pid: int):
        raise NotImplementedError

    @abstractmethod
    async def get_all_citations_for_ids(
        self, pids: list[int]
    ) -> list[tuple[int, dict[str, bool | int | float | str]]]:
        raise NotImplementedError

    @abstractmethod
    async def delete_profile_feature(
        self,
        user_id: str,
        feature: str,
        tag: str,
        value: str | None = None,
        isolations: dict[str, bool | int | float | str] | None = None,
    ):
        """
        Delete a feature from the profile with the key from the given user
        """
        raise NotImplementedError

    @abstractmethod
    async def get_large_profile_sections(
        self,
        user_id: str,
        thresh: int,
        isolations: dict[str, bool | int | float | str] | None = None,
    ) -> list[list[dict[str, Any]]]:
        """
        get sections of profile with at least thresh entries
        """
        raise NotImplementedError

    @abstractmethod
    async def add_history(
        self,
        user_id: str,
        content: str,
        metadata: dict[str, str] | None = None,
        isolations: dict[str, bool | int | float | str] | None = None,
    ) -> Mapping[str, Any]:
        raise NotImplementedError

    @abstractmethod
    async def delete_history(
        self,
        user_id: str,
        start_time: int = 0,
        end_time: int = 0,
        isolations: dict[str, bool | int | float | str] | None = None,
    ):
        raise NotImplementedError

    @abstractmethod
    async def get_history_messages_by_ingestion_status(
        self,
        user_id: str,
        k: int = 0,
        is_ingested: bool = False,
    ) -> list[Mapping[str, Any]]:
        """
        retrieve the list of the history messages for the user
        with the ingestion status, up to k messages if k > 0
        """
        raise NotImplementedError

    @abstractmethod
    async def get_uningested_history_messages_count(self) -> int:
        """
        retrieve the count of the uningested history messages
        """
        raise NotImplementedError

    @abstractmethod
    async def mark_messages_ingested(
        self,
        ids: list[int],
    ) -> None:
        """
        mark the messages with the id as ingested
        """
        raise NotImplementedError

    @abstractmethod
    async def get_history_message(
        self,
        user_id: str,
        start_time: int = 0,
        end_time: int = 0,
        isolations: dict[str, bool | int | float | str] | None = None,
    ) -> list[str]:
        raise NotImplementedError

    @abstractmethod
    async def purge_history(
        self,
        user_id: str,
        start_time: int = 0,
        isolations: dict[str, bool | int | float | str] | None = None,
    ):
        raise NotImplementedError
