from datetime import datetime
from enum import Enum
from typing import Any
from uuid import UUID

from pydantic import BaseModel, JsonValue

from memmachine.common.data_types import FilterablePropertyValue


class ContentType(Enum):
    """Types of content."""

    HYBRID = "hybrid"
    MESSAGE = "message"
    TEXT = "text"


class Episode(BaseModel):
    uuid: UUID
    timestamp: datetime
    context: str
    content_type: ContentType
    content: Any
    attributes: dict[str, FilterablePropertyValue] | None = None
    payload: dict[str, JsonValue] | None = None

    def __hash__(self) -> int:
        """Hash an episode by its UUID."""
        return hash(self.uuid)


class Snapshot(BaseModel):
    uuid: UUID
    episode_uuid: UUID
    index: int
    timestamp: datetime
    context: str
    content_type: ContentType
    content: Any
    attributes: dict[str, FilterablePropertyValue] | None = None

    def __hash__(self) -> int:
        """Hash a snapshot by its UUID."""
        return hash(self.uuid)


class Derivative(BaseModel):
    uuid: UUID
    snapshot_uuid: UUID
    episode_uuid: UUID
    timestamp: datetime
    context: str
    content_type: ContentType
    content: Any
    attributes: dict[str, FilterablePropertyValue] | None = None

    def __hash__(self) -> int:
        """Hash a derivative by its UUID."""
        return hash(self.uuid)
