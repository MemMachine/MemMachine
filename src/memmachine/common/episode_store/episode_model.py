"""Data models for representing episodes and related enumerations."""

from enum import Enum

from pydantic import AwareDatetime, BaseModel, JsonValue

from memmachine.common.api import EpisodeType
from memmachine.common.data_types import FilterablePropertyValue

EpisodeIdT = str


class ContentType(Enum):
    """Enumeration for the type of content within an Episode."""

    STRING = "string"
    # Other content types like 'vector', 'image' could be added here.


class EpisodeEntry(BaseModel):
    """Payload used when creating a new episode entry."""

    content: str

    producer_id: str
    producer_role: str

    produced_for_id: str | None = None
    episode_type: EpisodeType | None = None
    metadata: dict[str, JsonValue] | None = None
    created_at: AwareDatetime | None = None


class EpisodeResponse(EpisodeEntry):
    """Episode data returned in responses."""

    uid: EpisodeIdT


class Episode(BaseModel):
    """Conversation message stored in history together with persistence metadata."""

    uid: EpisodeIdT
    content: str
    session_key: str
    created_at: AwareDatetime

    producer_id: str
    producer_role: str
    produced_for_id: str | None = None

    sequence_num: int = 0

    episode_type: EpisodeType = EpisodeType.MESSAGE
    content_type: ContentType = ContentType.STRING
    filterable_metadata: dict[str, FilterablePropertyValue] | None = None
    metadata: dict[str, JsonValue] | None = None

    def __hash__(self) -> int:
        """Hash an episode by its UID."""
        return hash(self.uid)
