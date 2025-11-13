from datetime import datetime
from typing import Any

from memmachine.episodic_memory.data_types import EpisodeType
from pydantic import BaseModel

HistoryIdT = str


class HistoryMessage(BaseModel):
    """Conversation message stored in history together with persistence metadata."""

    class Metadata(BaseModel):
        """Optional storage details for a history message (id, provider-specific info)."""

        id: HistoryIdT | None = None
        other: dict[str, Any] | None = None

    content: str
    session_key: str

    created_at: datetime

    producer_id: str
    producer_role: str
    produced_for_id: str | None = None

    episode_type: EpisodeType | None = None

    metadata: Metadata = Metadata()
