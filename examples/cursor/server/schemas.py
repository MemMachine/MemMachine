"""
schemas for the Cursor MCP Server.
"""

from typing import Any, Dict, Generic, List, Optional, TypeVar

from pydantic import BaseModel, EmailStr, Field

from .config import DEFAULT_EPISODE_TYPE

DataT = TypeVar("DataT")


# Base schemas for MemMachine requests/responses
class ResponseSchema(BaseModel, Generic[DataT]):
    """Standard API response schema"""

    success: bool = Field(True, description="Request success status")
    message: str = Field("Success", description="Response message")
    data: Optional[DataT] = Field(None, description="Response data")
    error_code: Optional[int] = Field(None, description="Error code if failed")

    @classmethod
    def success_response(
        cls, data: Any = None, message: str = "Success"
    ) -> "ResponseSchema":
        return cls(success=True, message=message, data=data, error_code=0)  # type: ignore[arg-type]

    @classmethod
    def error_response(cls, message: str, error_code: int = 1) -> "ResponseSchema":
        return cls(success=False, message=message, error_code=error_code)  # type: ignore[arg-type]


class PaginationSchema(BaseModel):
    page: int = Field(1, ge=1, description="Page number (1-based)")
    size: int = Field(20, ge=1, le=100, description="Page size (max 100)")

    @property
    def offset(self) -> int:
        return (self.page - 1) * self.size


class PaginatedResponseSchema(BaseModel, Generic[DataT]):
    success: bool = Field(True, description="Request success status")
    message: str = Field("Success", description="Response message")
    data: List[DataT] = Field(default_factory=list, description="Response data")
    pagination: Dict[str, Any] = Field(..., description="Pagination information")

    @classmethod
    def create(
        cls,
        data: List[Any],
        page: int,
        size: int,
        total: int,
        message: str = "Success",
    ) -> "PaginatedResponseSchema":
        total_pages = (total + size - 1) // size
        pagination = {
            "page": page,
            "size": size,
            "total": total,
            "total_pages": total_pages,
            "has_next": page < total_pages,
            "has_prev": page > 1,
        }
        return cls(success=True, message=message, data=data, pagination=pagination)  # type: ignore[arg-type]


# Pydantic schemas for MemMachine requests/responses
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


# Pydantic schemas for REST API requests/responses
class UserRegistration(BaseModel):
    """User registration request model."""

    username: str = Field(
        ...,
        min_length=3,
        max_length=50,
        description="Username",
        json_schema_extra={"example": "john_doe"},
    )
    email: Optional[EmailStr] = Field(
        None,
        description="Email address",
        json_schema_extra={"example": "john@example.com"},
    )
    password: str = Field(
        ...,
        min_length=6,
        description="Password",
        json_schema_extra={"example": "mySecurePass123"},
    )


class UserLogin(BaseModel):
    """User login request model."""

    username: str = Field(
        ..., description="Username", json_schema_extra={"example": "john_doe"}
    )
    password: str = Field(
        ..., description="Password", json_schema_extra={"example": "mySecurePass123"}
    )


class UserResponse(BaseModel):
    """User response model (without password)."""

    id: int
    username: str
    email: Optional[str]
    is_active: bool
    created_at: str  # ISO format string

    class Config:
        from_attributes = True


# Auth response schemas
class AuthTokenData(BaseModel):
    """Authentication token data."""

    token: str = Field(..., description="Token")
    expires_at: str = Field(..., description="Token expiration timestamp")


class UserInfo(BaseModel):
    """User information for auth responses."""

    id: int = Field(..., description="User ID")
    username: str = Field(..., description="Username")
    email: Optional[str] = Field(None, description="User email")
    is_active: Optional[bool] = Field(None, description="User active status")
    created_at: str = Field(..., description="User creation timestamp (ISO format)")
    updated_at: Optional[str] = Field(None, description="User last update timestamp (ISO format)")


class RegistrationResponseData(BaseModel):
    """Registration response data."""

    user: UserInfo = Field(..., description="User information")
    token: str = Field(..., description="Token")
    expires_at: str = Field(..., description="Token expiration timestamp")


class LoginResponseData(BaseModel):
    """Login response data."""

    user: UserInfo = Field(..., description="User information")
    token: str = Field(..., description="Token")
    expires_at: str = Field(..., description="Token expiration timestamp")


class UserInfoResponseData(BaseModel):
    """Current user info response data."""

    user: UserInfo = Field(..., description="Current user information")
    token: str = Field(..., description="Token")
    expires_at: str = Field(..., description="Token expiration timestamp")


# Debug response schema
class DebugInfo(BaseModel):
    """Debug information response data."""

    server: str = Field(..., description="Server name")
    memory_backend_url: str = Field(..., description="Memory backend URL")
    port: int = Field(..., description="Server port")
    mcp_endpoint: str = Field(..., description="MCP endpoint path")
    health_endpoint: str = Field(..., description="Health check endpoint path")
    api_endpoint: str = Field(..., description="API endpoint path")


# Sessions response schema
class SessionsResponseData(BaseModel):
    """Sessions response data."""

    sessions: List[Dict[str, Any]] = Field(default_factory=list, description="List of user sessions")
    total: int = Field(..., description="Total number of sessions")
