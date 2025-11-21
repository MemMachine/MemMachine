"""API v2 specification models for request and response structures."""

from datetime import datetime
from enum import Enum
from typing import Annotated, Any

from pydantic import AfterValidator, BaseModel, Field


def _validate_no_slash(v: str) -> str:
    if "/" in v:
        raise ValueError("ID cannot contain '/'")
    return v


SafeId = Annotated[str, AfterValidator(_validate_no_slash), Field(...)]


class MemoryType(str, Enum):
    """Enumeration of memory types."""

    EPISODIC = "episodic"
    SEMANTIC = "semantic"


class ProjectConfig(BaseModel):
    """Project configuration model."""

    reranker: Annotated[str, Field(default="default")]
    embedder: Annotated[str, Field(default="default")]


class CreateProjectSpec(BaseModel):
    """Specification model for creating a new project."""

    org_id: SafeId
    project_id: SafeId
    description: Annotated[str, Field(default="")]
    config: Annotated[ProjectConfig, Field(default_factory=ProjectConfig)]


class GetProjectSpec(BaseModel):
    """Specification model for getting a project."""

    org_id: SafeId
    project_id: SafeId


class SessionInfo(BaseModel):
    """Model representing session information."""

    org_id: SafeId
    project_id: SafeId

    model_config = {"extra": "ignore"}


class DeleteProjectSpec(BaseModel):
    """Specification model for deleting a project."""

    org_id: SafeId
    project_id: SafeId


class MemoryMessage(BaseModel):
    """Model representing a memory message."""

    content: Annotated[str, Field(...)]
    producer: Annotated[str, Field(...)]
    produced_for: Annotated[str, Field(default="")]
    timestamp: Annotated[datetime, Field(...)]
    role: Annotated[str, Field(...)]
    metadata: Annotated[dict[str, str], Field(default_factory=dict)]


class AddMemoriesSpec(BaseModel):
    """Specification model for adding memories."""

    org_id: SafeId
    project_id: SafeId
    messages: Annotated[list[MemoryMessage], Field(...)]


class SearchMemoriesSpec(BaseModel):
    """Specification model for searching memories."""

    org_id: SafeId
    project_id: SafeId
    top_k: Annotated[int, Field(default=10)]
    query: Annotated[str, Field(...)]
    filter: Annotated[str, Field(default="")]
    types: Annotated[list[MemoryType], Field(default_factory=list)]


class ListMemoriesSpec(BaseModel):
    """Specification model for listing memories."""

    org_id: SafeId
    project_id: SafeId
    limit: Annotated[int, Field(default=100)]
    offset: Annotated[int, Field(default=0)]
    filter: Annotated[str, Field(default="")]
    type: Annotated[MemoryType, Field(default="")]


class DeleteMemoriesSpec(BaseModel):
    """Specification model for deleting memories."""

    org_id: SafeId
    project_id: SafeId
    filter: Annotated[str, Field(default="")]


class DeleteEpisodicMemorySpec(BaseModel):
    """Specification model for deleting episodic memories."""

    org_id: SafeId
    project_id: SafeId
    episodic_id: Annotated[str, Field(...)]


class DeleteSemanticMemorySpec(BaseModel):
    """Specification model for deleting episodic memories."""

    org_id: SafeId
    project_id: SafeId
    semantic_id: Annotated[str, Field(...)]


class SearchResult(BaseModel):
    """Response model for memory search results."""

    status: Annotated[int, Field(default=0)]
    content: Annotated[dict[str, Any], Field(...)]
