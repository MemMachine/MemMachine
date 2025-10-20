"""
Authentication schemas for the MemMachine Extension MCP Server.
"""

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, EmailStr, Field


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
    invitationCode: str = Field(
        ...,
        description="Invitation code required for registration",
        json_schema_extra={"example": "???"},
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


class UserAuthToken(BaseModel):
    """Authentication token."""

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
