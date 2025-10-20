"""
Middleware for the Cursor MCP Server.
"""

import json
import logging
from typing import Optional

from fastapi import HTTPException, Request
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from .auth import TokenStore
from .models import User as UserModel
from .schemas import ResponseSchema

logger = logging.getLogger(__name__)


def get_current_user(request: Request) -> UserModel:
    """Dependency to get the current authenticated user from request state."""
    current_user = getattr(request.state, "current_user", None)
    if not current_user:
        raise HTTPException(status_code=401, detail="No authenticated user found")
    return current_user


class SessionMiddleware(BaseHTTPMiddleware):
    """Middleware for session management and request logging."""

    def __init__(self, app, token_store: TokenStore):
        super().__init__(app)
        self.token_store = token_store
        self.logger = logging.getLogger(f"{__name__}.SessionMiddleware")

    async def dispatch(self, request: Request, call_next):
        """Process request with session management and authentication."""
        path = request.url.path

        # Public endpoints that don't require authentication
        public_paths = {
            "/api/health",
            "/api/debug",
            "/api/auth/login",
            "/api/auth/register",
            "/docs",
            "/redoc",
            "/openapi.json",
            "/"
        }

        public_path_prefixes = {
            "/static/",  # Static files (CSS, JS, images)
        }

        # Check if this is a public endpoint (exact match or prefix match)
        is_public = path in public_paths or any(
            path.startswith(prefix) for prefix in public_path_prefixes
        )

        if is_public:
            self.logger.debug(f"Public endpoint accessed: {path}")
            return await call_next(request)

        self.logger.debug(f"Protected endpoint accessed: {path} - validating token")

        # Extract token from various header formats
        provided_token: Optional[str] = None
        auth_header = request.headers.get("authorization") or request.headers.get(
            "Authorization"
        )
        if auth_header and auth_header.startswith("Bearer "):
            provided_token = auth_header[len("Bearer ") :].strip()

        if not provided_token:
            self.logger.warning(
                f"Unauthorized request blocked (no token provided) - Path: {path}"
            )
            error_response = ResponseSchema.error_response(
                message="Missing authentication token. Please provide a valid token in the Authorization header.",
                error_code=401
            )
            return JSONResponse(
                status_code=401,
                content=error_response.model_dump(),
                headers={"WWW-Authenticate": "Bearer"},
            )

        request.state.token = provided_token
        token_valid = False

        if provided_token:
            token_valid = self.token_store.validate_token(provided_token)

        if not token_valid:
            self.logger.warning(
                f"Unauthorized request blocked (invalid token) - Path: {path}"
            )
            error_response = ResponseSchema.error_response(
                message="Invalid authentication token. Please provide a valid token.",
                error_code=401
            )
            return JSONResponse(
                status_code=401,
                content=error_response.model_dump(),
                headers={"WWW-Authenticate": "Bearer"},
            )

        # Token validation successful - get the current user and add to request state
        self.logger.debug(f"Token validation successful for path: {path}")
        try:
            current_user = self.token_store.get_user_by_token(provided_token)
            if current_user:
                request.state.current_user = current_user
                self.logger.debug(
                    f"Current user set in request state: {current_user.username}"
                )
            else:
                self.logger.warning(
                    f"Token valid but could not retrieve user - Path: {path}"
                )
                request.state.current_user = None
        except Exception as e:
            self.logger.error(f"Error retrieving user from token: {e}")
            request.state.current_user = None

        # Log request details
        self._log_request_details(request)

        # Log request body for POST requests
        if request.method == "POST":
            await self._log_request_body(request)

        try:
            response = await call_next(request)

            # response_session_id = response.headers.get("Mcp-Session-Id", None)
            # # Set session ID in response headers if available
            # if response_session_id:
            #     response.headers["Mcp-Session-Id"] = response_session_id

            return response

        except Exception as e:
            self.logger.error(f"Error in request processing: {e}")
            self.logger.error(f"Error type: {type(e).__name__}")
            import traceback

            self.logger.error(f"Traceback: {traceback.format_exc()}")

            return JSONResponse(
                status_code=500,
                content={
                    "error": "Internal server error",
                    "message": str(e),
                    "type": type(e).__name__,
                },
            )

    def _log_request_details(self, request: Request) -> None:
        """Log request details for debugging."""
        self.logger.info("=== MCP Server Request Debug ===")
        self.logger.info(f"Request URL: {request.url}")
        self.logger.info(f"Request Method: {request.method}")
        self.logger.info(f"Request Path: {request.url.path}")
        self.logger.info("All Headers:")
        for header_name, header_value in request.headers.items():
            lower_name = header_name.lower()
            if lower_name in {"authorization"}:
                masked_value = "***redacted***"
                self.logger.info(f"  {header_name}: {masked_value}")
            else:
                self.logger.info(f"  {header_name}: {header_value}")

    async def _log_request_body(self, request: Request) -> None:
        """Log request body for POST requests."""
        try:
            body = await request.body()
            if body:
                try:
                    data = json.loads(body)
                    self.logger.info(f"Request body: {json.dumps(data, indent=2)}")
                except json.JSONDecodeError:
                    self.logger.info(
                        f"Request body (raw): {body.decode('utf-8', errors='ignore')}"
                    )
            else:
                self.logger.info("Request body: (empty)")
        except Exception as e:
            self.logger.error(f"Error reading request body: {e}")