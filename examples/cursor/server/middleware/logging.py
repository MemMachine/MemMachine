"""
Middleware for logging requests and responses in the MemMachine Extension MCP Server.
"""

import json
import logging

from fastapi import Request
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

logger = logging.getLogger(__name__)


class LoggingMiddleware(BaseHTTPMiddleware):
    """Middleware for request and response logging."""

    def __init__(self, app):
        super().__init__(app)
        self.logger = logging.getLogger(f"{__name__}.LoggingMiddleware")

    async def dispatch(self, request: Request, call_next):
        """Process request with logging."""
        # Log request details
        self._log_request_details(request)

        if request.method == "POST":
            await self._log_request_body(request)

        try:
            response = await call_next(request)
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
