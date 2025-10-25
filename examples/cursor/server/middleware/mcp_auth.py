"""
Middleware for authentication and authorization in the MemMachine Extension MCP Server.
"""

import logging
from typing import Optional

from fastapi import Request
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from ..auth.token import validate_token
from ..services.user import get_user_by_token
from ..settings import settings

logger = logging.getLogger(__name__)

class MCPAuthMiddleware(BaseHTTPMiddleware):
    """Middleware for authentication and authorization."""

    def __init__(self, app):
        super().__init__(app)
        self.logger = logging.getLogger(f"{__name__}.MCPAuthMiddleware")

    async def dispatch(self, request: Request, call_next):
        """Process request with authentication and authorization."""

        # Extract token from various header formats
        provided_token: Optional[str] = None
        auth_header = request.headers.get("authorization")

        if auth_header and auth_header.startswith("Bearer "):
            provided_token = auth_header[len("Bearer ") :].strip()

        if not provided_token:
            self.logger.warning(
                "Unauthorized request blocked (no token provided)"
            )
            return JSONResponse(
                status_code=401,
                content={
                    "error": "Unauthorized",
                    "message": "Access token missing or invalid",
                },
                headers={"WWW-Authenticate": f"Bearer resource_metadata=\"{settings.oauth_domain}/.well-known/oauth-protected-resource\", error=\"invalid_token\", error_description=\"Access token missing or invalid\""},
            )

        if not validate_token(provided_token):
            self.logger.warning(
                "Unauthorized request blocked (invalid token)"
            )

            return JSONResponse(
                status_code=401,
                content={
                    "error": "Unauthorized",
                    "message": "Access token invalid",
                },
                headers={"WWW-Authenticate": f"Bearer resource_metadata=\"{settings.oauth_domain}/.well-known/oauth-protected-resource\", error=\"invalid_token\", error_description=\"Access token missing or invalid\""},
            )

        self.logger.debug("Token validation successful")

        try:
            current_user = get_user_by_token(provided_token)
            if not current_user:
                return JSONResponse(
                    status_code=401,
                    content={
                        "error": "Unauthorized",
                        "message": "Access token invalid",
                    },
                    headers={"WWW-Authenticate": f"Bearer resource_metadata=\"{settings.oauth_domain}/.well-known/oauth-protected-resource\", error=\"invalid_token\", error_description=\"Access token missing or invalid\""},
                )
            else:
                request.state.current_user = current_user
                request.state.token = provided_token
        except Exception as e:
            self.logger.error(f"Error retrieving user from token: {e}")
            return JSONResponse(
                status_code=401,
                content={
                    "error": "Unauthorized",
                    "message": "Access token invalid",
                },
                headers={"WWW-Authenticate": f"Bearer resource_metadata=\"{settings.oauth_domain}/.well-known/oauth-protected-resource\", error=\"invalid_token\", error_description=\"Access token missing or invalid\""},
            )

        return await call_next(request)
