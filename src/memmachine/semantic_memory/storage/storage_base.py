"""Abstract interfaces for semantic storage implementations."""

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import numpy as np
from pydantic import InstanceOf

from memmachine.common.episode_store.episode_model import EpisodeIdT
from memmachine.common.filter.filter_parser import FilterExpr
from memmachine.semantic_memory.semantic_model import (
    FeatureIdT,
    SemanticFeature,
    SetIdT,
)


class SemanticStorage(ABC):
    """Base class for semantic storage backends."""

    backend_name: str | None = None

    @abstractmethod
    async def startup(self) -> None:
        """Initialize the semantic storage connection."""
        raise NotImplementedError

    @abstractmethod
    async def cleanup(self) -> None:
        """Clean up semantic storage resources."""
        raise NotImplementedError

    @abstractmethod
    async def delete_all(self) -> None:
        """Delete all semantic features in the storage."""
        raise NotImplementedError

    @abstractmethod
    async def reset_set_ids(self, set_ids: Sequence[SetIdT]) -> None:
        """Reset the set ids for the provided feature sets."""
        raise NotImplementedError

    @abstractmethod
    async def get_feature(
        self,
        feature_id: FeatureIdT,
        load_citations: bool = False,
    ) -> SemanticFeature | None:
        """Fetch a feature by id, optionally loading citation details."""
        raise NotImplementedError

    @abstractmethod
    async def add_feature(
        self,
        *,
        set_id: SetIdT,
        category_name: str,
        feature: str,
        value: str,
        tag: str,
        embedding: InstanceOf[np.ndarray],
        metadata: Mapping[str, Any] | None = None,
    ) -> FeatureIdT:
        """Add a new feature to the user."""
        raise NotImplementedError

    @abstractmethod
    async def update_feature(
        self,
        feature_id: FeatureIdT,
        *,
        set_id: SetIdT | None = None,
        category_name: str | None = None,
        feature: str | None = None,
        value: str | None = None,
        tag: str | None = None,
        embedding: InstanceOf[np.ndarray] | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        """Update an existing feature with any provided fields."""
        raise NotImplementedError

    @abstractmethod
    async def delete_features(self, feature_ids: Sequence[FeatureIdT]) -> None:
        """Delete the requested feature ids."""
        raise NotImplementedError

    @dataclass
    class VectorSearchOpts:
        """Parameters controlling vector similarity constraints for retrieval."""

        query_embedding: InstanceOf[np.ndarray]
        min_distance: float | None = None

    @abstractmethod
    def get_feature_set(
        self,
        *,
        filter_expr: FilterExpr | None = None,
        page_size: int | None = None,
        page_num: int | None = None,
        vector_search_opts: VectorSearchOpts | None = None,
        tag_threshold: int | None = None,
        load_citations: bool = False,
    ) -> AsyncIterator[SemanticFeature]:
        """
        Retrieve features matching the provided filters.

        Returns:
            An async iterator of features with their metadata and timestamps.

        """
        raise NotImplementedError

    @abstractmethod
    async def delete_feature_set(
        self,
        *,
        filter_expr: FilterExpr | None = None,
    ) -> None:
        """Delete features matching the given filters."""
        raise NotImplementedError

    @abstractmethod
    async def add_citations(
        self,
        feature_id: FeatureIdT,
        history_ids: Sequence[EpisodeIdT],
    ) -> None:
        """Associate history ids as citations for a feature."""
        raise NotImplementedError

    @abstractmethod
    def get_history_messages(
        self,
        *,
        set_ids: Sequence[SetIdT] | None = None,
        limit: int | None = None,
        is_ingested: bool | None = None,
    ) -> AsyncIterator[EpisodeIdT]:
        """Retrieve history messages with optional ingestion status."""
        raise NotImplementedError

    @abstractmethod
    async def get_history_messages_count(
        self,
        *,
        set_ids: Sequence[SetIdT] | None = None,
        is_ingested: bool | None = None,
    ) -> int:
        """Return the count of history messages."""
        raise NotImplementedError

    @abstractmethod
    async def add_history_to_set(self, set_id: SetIdT, history_id: EpisodeIdT) -> None:
        """Attach a history id to a feature set."""
        raise NotImplementedError

    @abstractmethod
    async def delete_history(self, history_ids: Sequence[EpisodeIdT]) -> None:
        raise NotImplementedError

    @abstractmethod
    async def delete_history_set(self, set_ids: Sequence[SetIdT]) -> None:
        raise NotImplementedError

    @abstractmethod
    async def mark_messages_ingested(
        self,
        *,
        set_id: SetIdT,
        history_ids: Sequence[EpisodeIdT],
    ) -> None:
        """Mark the provided history messages as ingested."""
        raise NotImplementedError

    @abstractmethod
    def get_history_set_ids(
        self,
        *,
        min_uningested_messages: int | None = None,
        older_than: datetime | None = None,
    ) -> AsyncIterator[SetIdT]:
        """Return all set id's that match the specified filters."""
        raise NotImplementedError

    @abstractmethod
    def get_set_ids_starts_with(self, prefix: str) -> AsyncIterator[SetIdT]:
        """Return all set id's that start with the specified prefix."""
        raise NotImplementedError
