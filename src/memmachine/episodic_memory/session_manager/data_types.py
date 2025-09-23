from dataclasses import dataclass


@dataclass
class SessionInfo:
    """
    Represents the information about a single conversation session.
    This is typically retrieved from or stored in a session management
    database.
    """

    user_ids: list[str]
    """A list of user identifiers participating in the session."""
    session_id: str
    """
    A unique string identifier for the session, separate from the
    database ID.
    """
    group_id: str | None = None
    """The identifier for a group conversation, if applicable."""
    agent_ids: list[str] | None = None
    """A list of agent identifiers participating in the session."""
    configuration: dict | None = None
    """A dictionary containing any custom configuration for this session."""


@dataclass
class GroupConfiguration:
    group_id: str
    """The identifier for the group."""
    agent_list: list[str]
    """A list of agent identifiers in the group."""
    user_list: list[str]
    """A list of user identifiers in the group."""
    configuration: dict
    """A dictionary containing any custom configuration for the group."""
