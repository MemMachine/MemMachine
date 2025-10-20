"""
Schemas package for the MemMachine Extension MCP Server.
"""

from .auth import (
    LoginResponseData,
    RegistrationResponseData,
    UserAuthToken,
    UserInfo,
    UserInfoResponseData,
    UserLogin,
    UserRegistration,
    UserResponse,
)
from .base import (
    DataT,
    DebugInfo,
    PaginatedResponseSchema,
    PaginationSchema,
    ResponseSchema,
)

# Memory schemas
from .memory import (
    DeleteRequest,
    MemoryEpisode,
    SearchQuery,
    SessionData,
    SessionsResponseData,
)

__all__ = [
    # Base schemas
    "DataT",
    "DebugInfo",
    "ResponseSchema",
    "PaginationSchema",
    "PaginatedResponseSchema",
    # Memory schemas
    "SessionData",
    "MemoryEpisode",
    "SearchQuery",
    "DeleteRequest",
    "SessionsResponseData",
    # Auth schemas
    "UserRegistration",
    "UserLogin",
    "UserResponse",
    "UserAuthToken",
    "UserInfo",
    "RegistrationResponseData",
    "LoginResponseData",
    "UserInfoResponseData",
]
