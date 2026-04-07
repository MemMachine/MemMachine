"""Session data manager abstraction."""

from abc import ABC, abstractmethod
from enum import StrEnum

from pydantic import BaseModel, JsonValue

from memmachine_server.common.configuration.episodic_config import EpisodicMemoryConf
from memmachine_server.common.configuration.event_memory_config import EventMemoryConf
from memmachine_server.common.data_types import PropertyValue


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
        configuration: dict[str, JsonValue],
        param: EpisodicMemoryConf,
        description: str,
        metadata: dict[str, JsonValue],
    ) -> None:
        """Create a new session entry in the database."""
        raise NotImplementedError

    @abstractmethod
    async def update_session_status(
        self,
        session_key: str,
        status: str,
    ) -> None:
        """Update the status of a session."""
        raise NotImplementedError

    @abstractmethod
    async def get_sessions_by_status(
        self,
        status: str,
    ) -> list[str]:
        """Get a list of session keys by status."""
        raise NotImplementedError

    @abstractmethod
    async def delete_session(self, session_key: str) -> None:
        """Delete a session entry from the database."""
        raise NotImplementedError

    class SessionStatus(StrEnum):
        """Enum for session status."""

        Active = "active"
        Deleted = "delete"

    class SessionInfo(BaseModel):
        """Metadata describing a stored session."""

        configuration: dict[str, JsonValue]
        description: str
        user_metadata: dict[str, JsonValue]
        episode_memory_conf: EpisodicMemoryConf
        event_memory_conf: EventMemoryConf | None = None
        status: str

    @abstractmethod
    async def get_session_info(
        self,
        session_key: str,
        status: SessionStatus | None = None,
    ) -> SessionInfo | None:
        """Get configuration, description, metadata, and params for a session."""
        raise NotImplementedError

    @abstractmethod
    async def get_sessions(
        self,
        filters: dict[str, PropertyValue | None] | None = None,
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

    @abstractmethod
    async def update_session_episodic_config(
        self,
        session_key: str,
        enabled: bool | None = None,
        long_term_memory_enabled: bool | None = None,
        short_term_memory_enabled: bool | None = None,
    ) -> None:
        """
        Update episodic memory configuration flags for a session.

        Only provided (non-None) values are updated.

        Args:
            session_key: The session key to update.
            enabled: Whether episodic memory is enabled overall.
            long_term_memory_enabled: Whether long-term memory is enabled.
            short_term_memory_enabled: Whether short-term memory is enabled.

        Raises:
            SessionNotFoundError: If the session does not exist.
        """
        raise NotImplementedError

    @abstractmethod
    async def get_event_memory_conf(
        self,
        session_key: str,
    ) -> EventMemoryConf | None:
        """Get the event memory configuration for a session.

        Returns None if the session has no event memory configuration.

        Raises:
            SessionNotFoundError: If the session does not exist.
        """
        raise NotImplementedError

    @abstractmethod
    async def set_event_memory_conf(
        self,
        session_key: str,
        conf: EventMemoryConf | None,
    ) -> None:
        """Set or clear the event memory configuration for a session.

        Args:
            session_key: The session key to update.
            conf: The configuration to store, or None to clear.

        Raises:
            SessionNotFoundError: If the session does not exist.
        """
        raise NotImplementedError
