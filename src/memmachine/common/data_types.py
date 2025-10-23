"""
Common data types for MemMachine.
"""

from typing import Iterable, Protocol

class ExternalServiceAPIError(Exception):
    """
    Raised when an API error occurs for an external service.
    """

    pass

class SessionDataProtocol(Protocol):
    """Protocol for any object that contains session data fields."""

    group_id: str
    user_id: Iterable[str]
    session_id: str
    agent_id: Iterable[str]

    def generate_all_combinations(self) -> Iterable[dict[str, str]]:
        """Generate all combinations of user_id, agent_id, group_id, and session_id."""
        for user_id in self.user_id:
            for agent_id in self.agent_id:
                yield {
                    "group_id": self.group_id,
                    "session_id": self.session_id,
                    "user_id": user_id,
                    "agent_id": agent_id,
                }
