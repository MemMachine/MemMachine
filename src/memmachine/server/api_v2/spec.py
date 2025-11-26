"""API v2 specification models for request and response structures."""

from datetime import UTC, datetime
from typing import Annotated, Any

from pydantic import AfterValidator, BaseModel, Field

from memmachine.main.memmachine import MemoryType

DEFAULT_ORG_AND_PROJECT_ID = "universal"


def _validate_no_slash(v: str) -> str:
    if "/" in v:
        raise ValueError("ID cannot contain '/'")
    return v


def _validate_int_compatible(v: str) -> str:
    try:
        int(v)
    except ValueError as e:
        raise ValueError("ID must be int-compatible") from e
    return v


IntCompatibleId = Annotated[str, AfterValidator(_validate_int_compatible), Field(...)]


SafeId = Annotated[str, AfterValidator(_validate_no_slash), Field(...)]
SafeIdWithDefault = Annotated[SafeId, Field(default=DEFAULT_ORG_AND_PROJECT_ID)]


class ProjectConfig(BaseModel):
    """Project configuration model."""

    reranker: Annotated[str, Field(default="")]
    embedder: Annotated[str, Field(default="")]


class CreateProjectSpec(BaseModel):
    """Specification model for creating a new project."""

    org_id: SafeId
    project_id: SafeId
    description: Annotated[str, Field(default="")]
    config: Annotated[ProjectConfig, Field(default_factory=ProjectConfig)]


class ProjectResponse(BaseModel):
    """Response model for project operations."""

    org_id: SafeId
    project_id: SafeId
    description: Annotated[str, Field(default="")]
    config: Annotated[ProjectConfig, Field(default_factory=ProjectConfig)]


class GetProjectSpec(BaseModel):
    """Specification model for getting a project."""

    org_id: SafeId
    project_id: SafeId


class DeleteProjectSpec(BaseModel):
    """Specification model for deleting a project."""

    org_id: SafeId
    project_id: SafeId


class MemoryMessage(BaseModel):
    """Model representing a memory message."""

    content: Annotated[str, Field(...)]
    producer: Annotated[str, Field(default="user")]
    produced_for: Annotated[str, Field(default="")]
    timestamp: Annotated[datetime, Field(default_factory=lambda: datetime.now(UTC))]
    role: Annotated[str, Field(default="")]
    metadata: Annotated[dict[str, str], Field(default_factory=dict)]


class AddMemoriesSpec(BaseModel):
    """Specification model for adding memories."""

    org_id: SafeIdWithDefault
    project_id: SafeIdWithDefault
    messages: Annotated[list[MemoryMessage], Field(min_length=1)]


class AddMemoryResult(BaseModel):
    """Response model for adding memories."""

    uid: Annotated[str, Field(...)]


class AddMemoriesResponse(BaseModel):
    """Response model for adding memories."""

    results: Annotated[list[AddMemoryResult], Field(...)]


class SearchMemoriesSpec(BaseModel):
    """Specification model for searching memories."""

    org_id: SafeIdWithDefault
    project_id: SafeIdWithDefault
    top_k: Annotated[int, Field(default=10)]
    query: Annotated[str, Field(...)]
    filter: Annotated[str, Field(default="")]
    types: Annotated[list[MemoryType], Field(default_factory=list)]


class ListMemoriesSpec(BaseModel):
    """Specification model for listing memories."""

    org_id: SafeIdWithDefault
    project_id: SafeIdWithDefault
    page_size: Annotated[int, Field(default=100)]
    page_num: Annotated[int, Field(default=0)]
    filter: Annotated[str, Field(default="")]
    type: Annotated[MemoryType, Field(default=MemoryType.Episodic)]


class DeleteEpisodicMemorySpec(BaseModel):
    """Specification model for deleting episodic memories."""

    org_id: SafeIdWithDefault
    project_id: SafeIdWithDefault
    episodic_id: SafeId


class DeleteSemanticMemorySpec(BaseModel):
    """Specification model for deleting episodic memories."""

    org_id: SafeIdWithDefault
    project_id: SafeIdWithDefault
    semantic_id: SafeId


class SearchResult(BaseModel):
    """Response model for memory search results."""

    status: Annotated[int, Field(default=0)]
    content: Annotated[dict[str, Any], Field(...)]
