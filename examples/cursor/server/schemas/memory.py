"""
MemMachine schemas for the MemMachine Extension MCP Server.
"""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from ..core.constants import DEFAULT_EPISODE_TYPE


class SessionData(BaseModel):
    """Session data model for MemMachine requests."""

    group_id: str = Field(..., description="Group ID for the session")
    agent_id: List[str] = Field(..., description="List of agent IDs")
    user_id: Optional[List[str]] = Field(..., description="List of user IDs")
    session_id: str = Field(..., description="Unique session identifier")


class MemoryEpisode(BaseModel):
    """Memory episode data model."""

    session: SessionData = Field(..., description="Session data for the memory")
    producer: str = Field(..., description="Who produced the memory")
    produced_for: str = Field(..., description="Who the memory is produced for")
    episode_content: str = Field(..., description="The actual memory content")
    episode_type: str = Field(
        DEFAULT_EPISODE_TYPE, description="Type of the memory episode"
    )
    metadata: Optional[Dict[str, Any]] = Field(
        None, description="Additional metadata for the memory"
    )


class SearchQuery(BaseModel):
    """Search query model."""

    session: SessionData = Field(..., description="Session data for the search")
    query: str = Field(..., description="The search query string")
    limit: Optional[int] = Field(5, description="Maximum number of results to return")
    filter: Optional[Dict[str, Any]] = Field(
        None, description="Optional filters for the search"
    )


class DeleteRequest(BaseModel):
    """Delete request model."""

    session: SessionData = Field(..., description="Session data to delete")


class SessionsResponseData(BaseModel):
    """Sessions response data."""

    sessions: List[Dict[str, Any]] = Field(default_factory=list, description="List of user sessions")
    total: int = Field(..., description="Total number of sessions")
