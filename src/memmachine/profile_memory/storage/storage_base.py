from abc import ABC, abstractmethod
from typing import Any

import numpy as np


## LEGACY CODE to be fixed.
class ProfileStorageBase(ABC):
    """
    The base class for profile storage
    """

    @abstractmethod
    async def startup(self):
        raise NotImplementedError

    @abstractmethod
    async def cleanup(self):
        raise NotImplementedError

    @abstractmethod
    async def delete_all(self):
        raise NotImplementedError

    @abstractmethod
    async def get_profile(self, user_id: str) -> dict[str, Any]:
        """
        Get profile by id
        Return: A list of KV for eatch feature and value.
           The value is an array with: feature value, feature tag and deleted, update time, create time and delete time.
        """
        raise NotImplementedError

    @abstractmethod
    async def delete_profile(self, user_id: str):
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
    ):
        """
        Add a new feature to the profile.
        """
        raise NotImplementedError

    @abstractmethod
    async def delete_profile_feature(
        self,
        user_id: str,
        feature: str,
        tag: str,
        value: str | None = None,
        isolations: dict[str, bool | int | float | str] = None,
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
    ):
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
    async def get_ingested_history_messages(
            self,
            user_id: str,
            k: int = 0,
            is_ingested: bool = False,
    ) -> list[dict[str, Any]]:
        raise NotImplementedError

    @abstractmethod
    async def get_uningested_history_messages_count(self) -> int:
        raise NotImplementedError


    @abstractmethod
    async def mark_messages_ingested(self, ids: list[int]) -> None:
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
        isolations: dict[str, bool | int | float | str] | None= None,
    ):
        raise NotImplementedError
