"""
Common data types for MemMachine.
"""

from dataclasses import dataclass
from typing import Iterable


class ExternalServiceAPIError(Exception):
    """
    Raised when an API error occurs for an external service.
    """

    pass


@dataclass
class SessionData:
    """Class for session data."""

    group_id: str
    user_id: list[str]
    session_id: str
    agent_id: list[str]

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
