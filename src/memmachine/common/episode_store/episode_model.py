"""Data models for representing episodes and related enumerations."""

import datetime
import json
from collections.abc import Iterable
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
    score: float | None = None


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


def episodes_to_string(
    episode_response_context: Iterable[EpisodeResponse] | Iterable[Episode],
) -> str:
    """Format episode response context as a string."""
    context_string = ""

    for episode_response in episode_response_context:
        match episode_response.episode_type:
            case EpisodeType.MESSAGE:
                context_date = (
                    _format_date(
                        episode_response.created_at.date(),
                    )
                    if episode_response.created_at
                    else "Unknown Date"
                )
                context_time = (
                    _format_time(
                        episode_response.created_at.time(),
                    )
                    if episode_response.created_at
                    else "Unknown Time"
                )
                context_string += f"[{context_date} at {context_time}] {episode_response.producer_id}: {json.dumps(episode_response.content)}\n"
            case _:
                context_string += json.dumps(episode_response.content) + "\n"

    return context_string


def _format_date(date: datetime.date) -> str:
    """Format the date as a string."""
    return date.strftime("%A, %B %d, %Y")


def _format_time(time: datetime.time) -> str:
    """Format the time as a string."""
    return time.strftime("%I:%M %p")
