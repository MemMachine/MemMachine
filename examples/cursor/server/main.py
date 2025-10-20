#!/usr/bin/env python3
"""
Entry point for the MemMachine Extension Server.

This is the main entry point that can be run directly:
    python main.py

It imports from the modular package structure using relative imports.
"""

# /// script
# requires-python = ">=3.13"
# dependencies = []
# ///

import logging

import uvicorn

from .app import create_custom_app
from .settings import settings

# =============================================================================
# Logging Configuration
# =============================================================================

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(filename)s:%(lineno)d - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# =============================================================================
# Application Entry Point
# =============================================================================

if __name__ == "__main__":
    logger.info("=" * 60)
    logger.info("Starting MemMachine Extension MCP Server")
    logger.info("=" * 60)
    logger.info(f"Port: {settings.server_port}")
    logger.info(f"MemMachine Backend URL: {settings.mm_backend_url}")
    logger.info("=" * 60)

    # Create custom app with session management
    app = create_custom_app()

    # Start the server
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=settings.server_port,
        log_level="info"
    )
