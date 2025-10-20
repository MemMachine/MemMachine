#!/usr/bin/env python3
"""
Application factory for the MemMachine Extension MCP Server.
"""

import atexit
import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.utils import get_openapi

from .api import auth_router, memory_router, misc_router
from .core.database import close_db, init_db
from .mcp.app import mcp_app
from .middleware import LoggingMiddleware

logger = logging.getLogger(__name__)


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

    app = FastAPI(
        title="MemMachine Extension Server",
        description="MCP Server for MemMachine Extension",
        version="1.0.0",
        lifespan=mcp_app.lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Add logging middleware second (after authentication)
    app.add_middleware(LoggingMiddleware)

    # Include all the modular routers with their respective prefixes
    app.include_router(misc_router, prefix="/api", tags=["misc"])
    app.include_router(auth_router, prefix="/api/auth", tags=["auth"])
    app.include_router(memory_router, prefix="/api/memory", tags=["memory"])

    # Mount the MCP server
    app.mount("/", mcp_app)

    def custom_openapi():
        if app.openapi_schema:
            return app.openapi_schema

        openapi_schema = get_openapi(
            title=app.title,
            version=app.version,
            description=app.description,
            routes=app.routes,
        )

        app.openapi_schema = openapi_schema
        return app.openapi_schema

    app.openapi = custom_openapi

    logger.info("Application created successfully")

    return app
