"""
REST endpoints for the Cursor MCP Server.
"""

import logging
from typing import Any, Dict

import requests
from fastapi import APIRouter, Depends, FastAPI, Request

from .auth import TokenStore
from .client import MemMachineClient
from .config import (
    CURSOR_MCP_PORT,
    MEMORY_BACKEND_URL,
    TOKEN_TTL_SECONDS,
)
from .handlers import (
    _fetch_episodic_memory,
    _fetch_profile_memory,
    _handle_add_memory,
    _handle_delete_session_memory,
    _handle_search_memory,
)
from .middleware import get_current_user
from .models import User
from .schemas import (
    DebugInfo,
    LoginResponseData,
    RegistrationResponseData,
    ResponseSchema,
    SessionsResponseData,
    UserInfo,
    UserInfoResponseData,
    UserLogin,
    UserRegistration,
)

logger = logging.getLogger(__name__)


def register_rest_endpoints(
    app: FastAPI, memmachine_client: MemMachineClient, token_store: TokenStore
):
    """Register all REST endpoints with the FastAPI app."""

    # Create API router for all REST endpoints
    api_router = APIRouter(prefix="/api")

    # Add health check endpoint (keep at root level for load balancers)
    @api_router.get("/health")
    async def health_check() -> ResponseSchema:
        """Simple health check endpoint."""
        return ResponseSchema.success_response(message="Cursor MCP Server is running")

    # Add debug endpoint (keep at root level for easy access)
    @api_router.get("/debug")
    async def debug_info() -> ResponseSchema:
        """Debug endpoint to show server configuration."""
        debug_data = DebugInfo(
            server="Cursor MCP Server",
            memory_backend_url=MEMORY_BACKEND_URL,
            port=CURSOR_MCP_PORT,
            mcp_endpoint="/mcp",
            health_endpoint="/health",
            api_endpoint="/api",
        )
        return ResponseSchema.success_response(
            data=debug_data.model_dump(), message="Debug information retrieved successfully"
        )

    # Auth endpoints
    @api_router.post("/auth/register")
    async def register(registration: UserRegistration) -> ResponseSchema:
        """Register a new user."""
        user = token_store.create_user(
            username=registration.username,
            password=registration.password,
            email=registration.email,
        )

        if not user:
            return ResponseSchema.error_response(
                message="User registration failed. Username or email may already exist."
            )

        # Issue token for the newly registered user
        issued = token_store.issue_token(
            subject=user.username, ttl_seconds=TOKEN_TTL_SECONDS
        )

        registration_data = RegistrationResponseData(
            user=UserInfo(
                id=user.id,
                username=user.username,
                email=user.email,
                is_active=user.is_active,
                created_at=user.created_at.isoformat(),
                updated_at=user.updated_at.isoformat(),
            ),
            token=issued["token"],
            expires_at=issued["expires_at"],
        )
        return ResponseSchema.success_response(
            data=registration_data.model_dump(), message="User registered successfully"
        )

    @api_router.post("/auth/login")
    async def login(credentials: UserLogin) -> ResponseSchema:
        """Login with username/password to get a bearer token."""
        # First try database authentication
        user = token_store.authenticate_user(credentials.username, credentials.password)

        if user:
            # Database user authenticated
            issued = token_store.issue_token(
                subject=user.username, ttl_seconds=TOKEN_TTL_SECONDS
            )
            login_data = LoginResponseData(
                user=UserInfo(
                    id=user.id,
                    username=user.username,
                    email=user.email,
                    is_active=user.is_active,
                    created_at=user.created_at.isoformat(),
                    updated_at=user.updated_at.isoformat(),
                ),
                token=issued["token"],
                expires_at=issued["expires_at"],
            )
            return ResponseSchema.success_response(
                data=login_data.model_dump(), message="Login successful"
            )

        return ResponseSchema.error_response(message="Invalid credentials")

    @api_router.post("/auth/logout")
    async def logout(request: Request) -> ResponseSchema:
        """Logout by revoking the provided token."""
        # Get token from request state (set by middleware)
        provided_token = getattr(request.state, "token", None)
        if not provided_token:
            return ResponseSchema.error_response(message="Missing token")
        revoked = token_store.revoke_token(provided_token)
        if not revoked:
            return ResponseSchema.error_response(message="Token not found or already revoked")
        return ResponseSchema.success_response(message="Logout successful")

    @api_router.get("/auth/me")
    async def get_current_user_info(
        request: Request,
        current_user: User = Depends(get_current_user),
    ) -> ResponseSchema:
        """Get current authenticated user information."""
        try:
            # Get token from request state (set by middleware)
            provided_token = getattr(request.state, "token", None)

            # Get token information including expiration
            token_info = None
            if provided_token:
                token_info = token_store.get_token_info(provided_token)

            # This should not happen since middleware validates the token
            if not token_info:
                return ResponseSchema.error_response(message="Unable to retrieve token information")

            user_info_data = UserInfoResponseData(
                user=UserInfo(
                    id=current_user.id,
                    username=current_user.username,
                    email=current_user.email,
                    is_active=current_user.is_active,
                    created_at=current_user.created_at.isoformat(),
                    updated_at=current_user.updated_at.isoformat(),
                ),
                token=token_info["token"],
                expires_at=token_info["expires_at"],
            )
            return ResponseSchema.success_response(
                data=user_info_data.model_dump(), message="User information retrieved successfully"
            )
        except Exception as e:
            return ResponseSchema.error_response(message=f"Internal error: {str(e)}")

    @api_router.get("/memory/get_profile_memory")
    async def rest_get_profile_memory(
        limit: int = 10,
        session_id: str = "",
        current_user: User = Depends(get_current_user),
    ) -> ResponseSchema:
        """Get the profile memory for the current session via REST endpoint.

        This endpoint reuses the shared helper to fetch profile memory using the
        resolved session id (explicit parameter takes precedence over headers).
        """
        try:
            result = await _fetch_profile_memory(
                user=current_user,
                session_id=session_id,
                limit=limit,
                memmachine_client=memmachine_client,
            )
            return ResponseSchema.success_response(
                data=result, message="Profile memory retrieved successfully"
            )
        except requests.exceptions.RequestException as e:
            return ResponseSchema.error_response(message=f"Failed to get profile memory: {str(e)}")
        except Exception as e:
            return ResponseSchema.error_response(message=f"Internal error: {str(e)}")

    @api_router.get("/memory/get_episodic_memory")
    async def rest_get_episodic_memory(
        limit: int = 10,
        session_id: str = "",
        current_user: User = Depends(get_current_user),
    ) -> ResponseSchema:
        """Get the episodic memory for the current session via REST endpoint.

        This endpoint reuses the shared helper to fetch episodic memory using the
        resolved session id (explicit parameter takes precedence over headers).
        """
        try:
            result = await _fetch_episodic_memory(
                user=current_user,
                session_id=session_id,
                limit=limit,
                memmachine_client=memmachine_client,
            )
            return ResponseSchema.success_response(
                data=result, message="Episodic memory retrieved successfully"
            )
        except requests.exceptions.RequestException as e:
            return ResponseSchema.error_response(message=f"Failed to get episodic memory: {str(e)}")
        except Exception as e:
            return ResponseSchema.error_response(message=f"Internal error: {str(e)}")

    # Mirror MCP tools as REST endpoints
    @api_router.post("/memory/add")
    async def rest_add_memory(
        content: str,
        session_id: str,
        current_user: User = Depends(get_current_user),
    ) -> ResponseSchema:
        """REST endpoint to add a memory episode (mirrors MCP add_memory)."""
        try:
            result = await _handle_add_memory(
                user=current_user,
                session_id=session_id,
                content=content,
                memmachine_client=memmachine_client,
            )
            return ResponseSchema.success_response(
                data=result, message="Memory added successfully"
            )
        except requests.exceptions.RequestException as e:
            return ResponseSchema.error_response(message=f"Failed to add memory: {str(e)}")
        except Exception as e:
            return ResponseSchema.error_response(message=f"Internal error: {str(e)}")

    @api_router.get("/memory/search")
    async def rest_search_memory(
        query: str,
        limit: int = 5,
        session_id: str = "",
        current_user: User = Depends(get_current_user),
    ) -> ResponseSchema:
        """REST endpoint to search memories (mirrors MCP search_memory)."""
        try:
            result = await _handle_search_memory(
                user=current_user,
                session_id=session_id,
                query=query,
                limit=limit,
                memmachine_client=memmachine_client,
            )
            return ResponseSchema.success_response(
                data=result, message="Search completed successfully"
            )
        except requests.exceptions.RequestException as e:
            return ResponseSchema.error_response(message=f"Failed to search memory: {str(e)}")
        except Exception as e:
            return ResponseSchema.error_response(message=f"Internal error: {str(e)}")

    @api_router.delete("/memory/episodic/delete")
    async def rest_delete_session_memory(
        session_id: str,
        current_user: User = Depends(get_current_user),
    ) -> ResponseSchema:
        """REST endpoint to delete all memories for the current session (mirrors MCP delete_session_memory)."""
        try:
            result = await _handle_delete_session_memory(
                user=current_user,
                session_id=session_id,
                memmachine_client=memmachine_client,
            )
            return ResponseSchema.success_response(
                data=result, message="Session deleted successfully"
            )
        except ValueError as e:
            return ResponseSchema.error_response(message=str(e))
        except requests.exceptions.RequestException as e:
            return ResponseSchema.error_response(message=f"Failed to delete session: {str(e)}")
        except Exception as e:
            return ResponseSchema.error_response(message=f"Internal error: {str(e)}")

    @api_router.get("/sessions")
    async def get_current_user_sessions(
        current_user: User = Depends(get_current_user),
    ) -> ResponseSchema:
        """Get all sessions for the current authenticated user.

        This endpoint fetches all sessions associated with the current
        authenticated user from the MemMachine backend.
        """
        try:
            # Fetch sessions from MemMachine backend
            result = memmachine_client.get_sessions_for_user(current_user.username)
            sessions_data = SessionsResponseData(
                sessions=result.get("sessions", []),
                total=len(result.get("sessions", [])),
            )
            return ResponseSchema.success_response(
                data=sessions_data.model_dump(), message="Sessions retrieved successfully"
            )
        except requests.exceptions.RequestException as e:
            return ResponseSchema.error_response(message=f"Failed to get sessions: {str(e)}")
        except Exception as e:
            return ResponseSchema.error_response(message=f"Internal error: {str(e)}")

    @api_router.get("/mcp/tools")
    async def list_mcp_tools() -> Dict[str, Any]:
        """List all registered MCP tools.

        This implementation inspects module globals for functions with the
        "mcp_" prefix, which mirrors the set of MCP-exposed tools in this
        server. It avoids relying on FastMCP private internals.
        """
        try:
            tools_info: list[dict[str, Any]] = []
            for name, obj in globals().items():
                if callable(obj) and isinstance(name, str) and name.startswith("mcp_"):
                    tools_info.append({"name": name})
            return ResponseSchema.success_response(
                "MCP tools listed successfully", tools=tools_info, total=len(tools_info)
            )
        except Exception as e:
            return ResponseSchema.error_response(message=f"Failed to list MCP tools: {str(e)}")

    # Mount the API router to the main app
    app.include_router(api_router)
