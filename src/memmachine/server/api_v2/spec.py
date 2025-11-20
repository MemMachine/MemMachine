"""API v2 specification models for request and response structures."""

from typing import Annotated, Any

from pydantic import BaseModel, Field


class ProjectConfig(BaseModel):
    """Project configuration model."""

    reranker: Annotated[str, Field(default="default")]
    embedder: Annotated[str, Field(default="default")]


class CreateProjectSpec(BaseModel):
    """Specification model for creating a new project."""

    org_id: Annotated[str, Field(...)]
    project_id: Annotated[str, Field(...)]
    description: Annotated[str, Field(default="")]
    config: Annotated[ProjectConfig, Field(default_factory=ProjectConfig)]


class SessionInfo(BaseModel):
    """Model representing session information."""

    org_id: Annotated[str, Field(...)]
    project_id: Annotated[str, Field(...)]

    model_config = {"extra": "ignore"}


class DeleteProjectSpec(BaseModel):
    """Specification model for deleting a project."""

    org_id: Annotated[str, Field(...)]
    project_id: Annotated[str, Field(...)]


class MemoryMessage(BaseModel):
    """Model representing a memory message."""

    content: Annotated[str, Field(...)]
    producer: Annotated[str, Field(...)]
    timestamp: Annotated[str, Field(...)]
    role: Annotated[str, Field(...)]
    metadata: Annotated[dict[str, str], Field(default_factory=dict)]


class AddMemoriesSpec(BaseModel):
    """Specification model for adding memories."""

    org_id: Annotated[str, Field(...)]
    project_id: Annotated[str, Field(...)]
    messages: Annotated[list[MemoryMessage], Field(...)]


class SearchMemoriesSpec(BaseModel):
    """Specification model for searching memories."""

    org_id: Annotated[str, Field(...)]
    project_id: Annotated[str, Field(...)]
    top_k: Annotated[int, Field(default=10)]
    query: Annotated[str, Field(...)]
    filter: Annotated[str, Field(default="")]
    types: Annotated[list[str], Field(default_factory=list)]


class ListMemoriesSpec(BaseModel):
    """Specification model for listing memories."""

    org_id: Annotated[str, Field(...)]
    project_id: Annotated[str, Field(...)]
    limit: Annotated[int, Field(default=100)]
    offset: Annotated[int, Field(default=0)]
    filter: Annotated[str, Field(default="")]
    type: Annotated[str, Field(default="")]


class DeleteMemoriesSpec(BaseModel):
    """Specification model for deleting memories."""

    org_id: Annotated[str, Field(...)]
    project_id: Annotated[str, Field(...)]
    filter: Annotated[str, Field(default="")]


class DeleteEpisodicMemorySpec(BaseModel):
    """Specification model for deleting episodic memories."""

    org_id: Annotated[str, Field(...)]
    project_id: Annotated[str, Field(...)]
    episodic_id: Annotated[str, Field(...)]


class DeleteSemanticMemorySpec(BaseModel):
    """Specification model for deleting episodic memories."""

    org_id: Annotated[str, Field(...)]
    project_id: Annotated[str, Field(...)]
    semantic_id: Annotated[str, Field(...)]


class SearchResult(BaseModel):
    """Response model for memory search results."""

    status: Annotated[int, Field(default=0)]
    content: Annotated[dict[str, Any], Field(...)]
