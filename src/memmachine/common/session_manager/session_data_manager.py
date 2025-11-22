"""Session data manager abstraction."""

from abc import ABC, abstractmethod
from typing import Any

from pydantic import BaseModel

from memmachine.common.configuration.episodic_config import EpisodicMemoryConf


class SessionDataManager(ABC):
    """Interface for managing session data and short-term memory."""

    @abstractmethod
    async def close(self) -> None:
        """Close the database connection."""
        raise NotImplementedError

    @abstractmethod
    async def create_tables(self) -> None:
        """Create the necessary tables in the database."""
        raise NotImplementedError

    @abstractmethod
    async def drop_tables(self) -> None:
        """Drop all created tables from the database."""
        raise NotImplementedError

    @abstractmethod
    async def create_new_session(
        self,
        session_key: str,
        configuration: dict[str, object],
        param: EpisodicMemoryConf,
        description: str,
        metadata: dict[str, object],
    ) -> None:
        """Create a new session entry in the database."""
        raise NotImplementedError

    @abstractmethod
    async def delete_session(self, session_key: str) -> None:
        """Delete a session entry from the database."""
        raise NotImplementedError

    class SessionInfo(BaseModel):
        """Metadata describing a stored session."""

        configuration: dict[str, Any]
        description: str
        user_metadata: dict[str, Any]
        episode_memory_conf: EpisodicMemoryConf

    @abstractmethod
    async def get_session_info(
        self,
        session_key: str,
    ) -> SessionInfo | None:
        """Get configuration, description, metadata, and params for a session."""
        raise NotImplementedError

    @abstractmethod
    async def get_sessions(
        self,
        filters: dict[str, object] | None = None,
    ) -> list[str]:
        """Return a list of all session keys (optionally filtered)."""
        raise NotImplementedError

    @abstractmethod
    async def save_short_term_memory(
        self,
        session_key: str,
        summary: str,
        last_seq: int,
        episode_num: int,
    ) -> None:
        """Save or update short-term memory data for a session."""
        raise NotImplementedError

    @abstractmethod
    async def get_short_term_memory(self, session_key: str) -> tuple[str, int, int]:
        """Retrieve short-term memory data for a session."""
        raise NotImplementedError
