"""
Memory API routes.
"""

import logging

import requests
from fastapi import APIRouter, Depends

from ..core.handlers import (
    _fetch_episodic_memory,
    _fetch_profile_memory,
    _handle_add_memory,
    _handle_delete_episodic_memory,
    _handle_search_memory,
)
from ..core.mm_client import memmachine_client
from ..models import User
from ..schemas import ResponseSchema
from ..services.user import get_current_user

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/get_profile_memory")
async def rest_get_profile_memory(
    limit: int = 10,
    current_user: User = Depends(get_current_user),
) -> ResponseSchema:
    """Get the profile memory for the current user via REST endpoint."""
    try:
        result = await _fetch_profile_memory(
            user=current_user,
            limit=limit,
            memmachine_client=memmachine_client,
        )
        return ResponseSchema.success_response(
            data=result, message="Profile memory retrieved successfully"
        )
    except requests.exceptions.RequestException as e:
        return ResponseSchema.error_response(
            message=f"Failed to get profile memory: {str(e)}"
        )
    except Exception as e:
        return ResponseSchema.error_response(message=f"Internal error: {str(e)}")


@router.get("/get_episodic_memory")
async def rest_get_episodic_memory(
    limit: int = 10,
    current_user: User = Depends(get_current_user),
) -> ResponseSchema:
    """Get the episodic memory for the current session via REST endpoint.

    This endpoint reuses the shared helper to fetch episodic memory using the
    resolved session id (explicit parameter takes precedence over headers).
    """
    try:
        result = await _fetch_episodic_memory(
            user=current_user,
            limit=limit,
            memmachine_client=memmachine_client,
        )
        return ResponseSchema.success_response(
            data=result, message="Episodic memory retrieved successfully"
        )
    except requests.exceptions.RequestException as e:
        return ResponseSchema.error_response(
            message=f"Failed to get episodic memory: {str(e)}"
        )
    except Exception as e:
        return ResponseSchema.error_response(message=f"Internal error: {str(e)}")


@router.post("/add")
async def rest_add_memory(
    content: str,
    current_user: User = Depends(get_current_user),
) -> ResponseSchema:
    """REST endpoint to add a memory episode (mirrors MCP add_memory)."""
    try:
        await _handle_add_memory(
            user=current_user,
            content=content,
            memmachine_client=memmachine_client,
        )
        return ResponseSchema.success_response(
            data=None, message="Memory added successfully"
        )
    except requests.exceptions.RequestException as e:
        return ResponseSchema.error_response(message=f"Failed to add memory: {str(e)}")
    except Exception as e:
        return ResponseSchema.error_response(message=f"Internal error: {str(e)}")


@router.get("/search")
async def rest_search_memory(
    query: str,
    limit: int = 5,
    current_user: User = Depends(get_current_user),
) -> ResponseSchema:
    """REST endpoint to search memories (mirrors MCP search_memory)."""
    try:
        result = await _handle_search_memory(
            user=current_user,
            query=query,
            limit=limit,
            memmachine_client=memmachine_client,
        )
        return ResponseSchema.success_response(
            data=result, message="Search completed successfully"
        )
    except requests.exceptions.RequestException as e:
        return ResponseSchema.error_response(
            message=f"Failed to search memory: {str(e)}"
        )
    except Exception as e:
        return ResponseSchema.error_response(message=f"Internal error: {str(e)}")


@router.delete("/episodic/delete")
async def rest_delete_session_memory(
    current_user: User = Depends(get_current_user),
) -> ResponseSchema:
    """REST endpoint to delete all memories."""
    try:
        await _handle_delete_episodic_memory(
            user=current_user,
            memmachine_client=memmachine_client,
        )
        return ResponseSchema.success_response(
            data=None, message="Memories deleted successfully"
        )
    except ValueError as e:
        return ResponseSchema.error_response(message=str(e))
    except requests.exceptions.RequestException as e:
        return ResponseSchema.error_response(
            message=f"Failed to delete memories: {str(e)}"
        )
    except Exception as e:
        return ResponseSchema.error_response(message=f"Internal error: {str(e)}")
