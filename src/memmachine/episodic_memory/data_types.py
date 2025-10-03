from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any
from uuid import UUID

# Type alias for JSON-compatible data structures.
JSONValue = None | bool | int | float | str | list["JSONValue"] | dict[str, "JSONValue"]


class ContentType(Enum):
    """Enumeration for the type of content within an Episode."""

    STRING = "string"
    # Other content types like 'vector', 'image' could be added here.


@dataclass
class SessionInfo:
    """
    Represents the information about a single conversation session.
    This is typically retrieved from or stored in a session management
    database.
    """

    group_id: str
    """The identifier for a group conversation."""
    session_id: str
    """
    A unique string identifier for the session.
    """
    agent_ids: list[str]
    """A list of agent identifiers participating in the session."""
    user_ids: list[str]
    """A list of user identifiers participating in the session."""
    configuration: dict
    """A dictionary containing any custom configuration for this session."""


@dataclass
class GroupConfiguration:
    """
    Represents the configuration for a group of conversations.
    """

    group_id: str
    """The identifier for the group."""
    agent_list: list[str]
    """A list of agent identifiers in the group."""
    user_list: list[str]
    """A list of user identifiers in the group."""
    configuration: dict
    """A dictionary containing any custom configuration for the group."""


@dataclass
class MemoryContext:
    """
    Defines the unique context for a memory instance.
    It's used to isolate memories for different conversations, users,
    and agents.
    """

    group_id: str
    """The identifier for the group context."""
    agent_id: set[str]
    """A set of agent identifiers for the context."""
    user_id: set[str]
    """A set of user identifiers for the context."""
    session_id: str
    """The identifier for the session context."""

    def __eq__(self, other):
        if not isinstance(other, MemoryContext):
            return False
        return self.group_id == other.group_id and self.session_id == other.session_id

    def __hash__(self):
        return hash(
            f"""{len(self.group_id)}#{self.group_id}_
            {len(self.session_id)}#{self.session_id}"""
        )


@dataclass(kw_only=True)
class Episode:
    """
    Represents a single, atomic event or piece of data in the memory system.
    `kw_only=True` enforces that all fields must be specified as keyword
    arguments during instantiation, improving clarity.
    """

    uuid: UUID
    """A unique identifier (UUID) for the episode."""
    episode_type: str
    """
    A string indicating the type of the episode (e.g., 'message', 'thought',
    'action').
    """
    content_type: ContentType
    """The type of the data stored in the 'content' field."""
    content: Any
    """The actual data of the episode, which can be of any type."""
    timestamp: datetime
    """The date and time when the episode occurred."""
    group_id: str
    """Identifier for the group (e.g., a specific chat room or DM)."""
    session_id: str
    """Identifier for the session to which this episode belongs."""
    producer_id: str
    """The identifier of the user or agent that created this episode."""
    produced_for_id: str | None = None
    """The identifier of the intended recipient, if any."""
    user_metadata: JSONValue = None
    """
    A dictionary for any additional, user-defined metadata in a
    JSON-compatible format."""
