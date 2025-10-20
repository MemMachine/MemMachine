"""
Base schemas for the MemMachine Extension Server.
"""

from typing import Any, Dict, Generic, List, Optional, TypeVar

from pydantic import BaseModel, Field

DataT = TypeVar("DataT")


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
        }
        return cls(success=True, message=message, data=data, pagination=pagination)  # type: ignore[arg-type]


class DebugInfo(BaseModel):
    """Debug information response data."""

    server: str = Field(..., description="Server name")
    mm_backend_url: str = Field(..., description="MemMachine backend URL")
    port: int = Field(..., description="Server port")
    mcp_endpoint: str = Field(..., description="MCP endpoint path")
    health_endpoint: str = Field(..., description="Health check endpoint path")
    api_endpoint: str = Field(..., description="API endpoint path")
