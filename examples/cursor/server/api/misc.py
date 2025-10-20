"""
Miscellaneous API routes (health check, debug, etc.).
"""

import logging

from fastapi import APIRouter

from ..schemas import DebugInfo, ResponseSchema
from ..settings import settings

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/health")
async def health_check() -> ResponseSchema:
    """Simple health check endpoint."""
    return ResponseSchema.success_response(
        message="MemMachine Extension Server is running"
    )


@router.get("/debug")
async def debug_info() -> ResponseSchema:
    """Debug endpoint to show server configuration."""
    debug_data = DebugInfo(
        server="MemMachine Extension Server",
        mm_backend_url=settings.mm_backend_url,
        port=settings.server_port,
        mcp_endpoint="/mcp",
        health_endpoint="/health",
        api_endpoint="/api",
    )
    return ResponseSchema.success_response(
        data=debug_data.model_dump(),
        message="Debug information retrieved successfully",
    )
