from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Mapping, Optional

import numpy as np


class SemanticStorageBase(ABC):
    """
    The base class for Semantic storage
    """

    @abstractmethod
    async def startup(self):
        """
        initializations for the semantic storage,
        such as creating connection to the database
        """
        raise NotImplementedError

    @abstractmethod
    async def cleanup(self):
        """
        cleanup for the semantic storage
        such as closing connection to the database
        """
        raise NotImplementedError

    @abstractmethod
    async def delete_all(self):
        """
        delete all semantic features in the storage
        such as truncating the database table
        """
        raise NotImplementedError

    @abstractmethod
    async def get_set_features(
        self,
        *,
        set_id: str,
        semantic_type_id: Optional[str] = None,
        tag: Optional[str] = None,
    ) -> dict[str, Any]:
        """
        Get feature set by user id
        Return: A list of KV for each feature and value.
           The value is an array with: feature value, feature tag and deleted, update time, create time and delete time.
        """
        raise NotImplementedError

    @abstractmethod
    async def delete_feature_set(
        self,
        *,
        set_id: str,
        semantic_type_id: Optional[str] = None,
        tag: Optional[str] = None,
    ):
        """
        Delete all the features by id
        """
        raise NotImplementedError

    @abstractmethod
    async def add_feature(
        self,
        *,
        set_id: str,
        semantic_type_id: str,
        feature: str,
        value: str,
        tag: str,
        embedding: np.ndarray,
        metadata: dict[str, Any] | None = None,
        citations: list[int] | None = None,
    ) -> int:
        """
        Add a new feature to the user.
        """
        raise NotImplementedError

    @abstractmethod
    async def semantic_search(
        self,
        *,
        set_id: str,
        qemb: np.ndarray,
        k: int,
        min_cos: float,
        include_citations: bool = False,
    ) -> list[dict[str, Any]]:
        raise NotImplementedError

    @abstractmethod
    async def delete_features(self, feature_ids: list[int]):
        raise NotImplementedError

    @abstractmethod
    async def get_all_citations_for_ids(self, feature_ids: list[int]) -> list[int]:
        raise NotImplementedError

    @abstractmethod
    async def delete_feature_with_filter(
        self,
        *,
        set_id: str,
        semantic_type_id: str,
        feature: str,
        tag: str,
    ):
        """
        Delete a feature from the user
        """
        raise NotImplementedError

    @abstractmethod
    async def get_large_feature_sections(
        self,
        set_id: str,
        thresh: int,
    ) -> list[list[dict[str, Any]]]:
        """
        get feature sets with at least thresh entries
        """
        raise NotImplementedError

    @abstractmethod
    async def add_history(
        self,
        set_id: str,
        content: str,
        metadata: dict[str, str] | None = None,
    ) -> Mapping[str, Any]:
        raise NotImplementedError

    @abstractmethod
    async def delete_history(
        self,
        *,
        set_id: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
    ):
        raise NotImplementedError

    @abstractmethod
    async def get_history_messages_by_ingestion_status(
        self,
        *,
        set_id: str,
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
        *,
        ids: list[int],
    ) -> None:
        """
        mark the messages with the id as ingested
        """
        raise NotImplementedError

    @abstractmethod
    async def get_history_message(
        self,
        *,
        set_id: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
    ) -> list[str]:
        raise NotImplementedError
