"""Core-owned interface for the session-data store used by short-term memory.

Short-term memory persists and reloads its summary state through a session-data
store. The concrete implementation (``SessionDataManager``) lives in
``memmachine-server``; this module declares the minimal structural contract that
short-term memory depends on, so the core package needs no server-side import.
"""

from typing import Protocol, runtime_checkable


@runtime_checkable
class SessionDataStore(Protocol):
    """Minimal session-data persistence interface used by short-term memory."""

    async def create_tables(self) -> None:
        """Create the tables backing the session-data store."""
        raise NotImplementedError

    async def get_short_term_memory(self, session_key: str) -> tuple[str, int, int]:
        """Return the stored ``(summary, last_seq, episode_num)`` for a session."""
        raise NotImplementedError

    async def save_short_term_memory(
        self,
        session_key: str,
        summary: str,
        last_seq: int,
        episode_num: int,
    ) -> None:
        """Persist short-term memory state for a session."""
        raise NotImplementedError
