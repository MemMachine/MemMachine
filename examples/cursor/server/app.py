#!/usr/bin/env python3
"""
Application factory for the Cursor MCP Server.

This module contains the create_custom_app function that builds and configures
the FastAPI application with all necessary middleware, tools, and endpoints.
"""

import atexit
import logging
import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.utils import get_openapi
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from fastmcp import FastMCP

from server.database import close_db, init_db

from .auth import TokenStore
from .client import MemMachineClient
from .config import MEMORY_BACKEND_URL, VERIFY_SSL
from .endpoints import register_rest_endpoints
from .mcp_tools import register_mcp_tools
from .middleware import SessionMiddleware

logger = logging.getLogger(__name__)

# =============================================================================
# Application Factory
# =============================================================================

def create_custom_app() -> FastAPI:
    """Create a custom FastAPI app with session management.

    Returns:
        Configured FastAPI application
    """
    # Initialize database before creating the app
    logger.info("Starting application...")
    try:
        init_db()
        logger.info("Database initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize database: {e}")
        raise

    # Register cleanup handler for database shutdown
    atexit.register(close_db)

    memmachine_client = MemMachineClient(MEMORY_BACKEND_URL, verify_ssl=VERIFY_SSL)
    token_store = TokenStore()

    # Create MCP app first to get its lifespan
    mcp = FastMCP("CursorMemMachine")
    register_mcp_tools(mcp, memmachine_client, token_store)
    mcp_app = mcp.http_app("/")

    app = FastAPI(
        title="Cursor MCP Server",
        description="MCP Server for Cursor integration with MemMachine",
        version="1.0.0",
        lifespan=mcp_app.lifespan
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.add_middleware(SessionMiddleware, token_store=token_store)

    register_rest_endpoints(app, memmachine_client, token_store)

    # Include MCP routes directly instead of mounting
    # This avoids the Mount object issue that causes 405 errors
    for route in mcp_app.routes:
        # Create a new route with proper HTTP methods
        if hasattr(route, 'path') and route.path == "/":
            # Add the MCP route with proper HTTP methods
            app.add_route(
                "/mcp",
                route.endpoint,
                methods=["POST", "GET", "OPTIONS"],
                name="mcp_endpoint"
            )

    # Mount static files for frontend assets (CSS, JS, images, etc.)
    if os.path.exists("static"):
        app.mount("/static", StaticFiles(directory="static"), name="static")

    # Add SPA fallback route (this should be LAST)
    @app.get("/{full_path:path}")
    async def serve_spa(full_path: str):
        # """Serve SPA for all non-API routes."""
        # # Check if it's an API route or other reserved paths
        # if full_path.startswith(("api/", "mcp/", "static/", "docs", "openapi.json")):
        #     return {"error": "Not found"}

        # Serve the main SPA file (index.html)
        spa_file = "static/index.html"
        if os.path.exists(spa_file):
            return FileResponse(spa_file)
        else:
            return {"error": "SPA not found"}

    def custom_openapi():
        if app.openapi_schema:
            return app.openapi_schema


        openapi_schema = get_openapi(
            title=app.title,
            version=app.version,
            description=app.description,
            routes=app.routes,
        )

        # Add security scheme for authentication
        openapi_schema["components"]["securitySchemes"] = {
            "Bearer": {
                "type": "http",
                "scheme": "bearer",
                "bearerFormat": "JWT",
                "description": "Enter your authentication token"
            }
        }

        # Set global security requirements
        openapi_schema["security"] = [{"Bearer": []}]

        app.openapi_schema = openapi_schema
        return app.openapi_schema

    app.openapi = custom_openapi

    logger.info("Application created successfully")

    return app

