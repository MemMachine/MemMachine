"""
Configuration and constants for the Cursor MCP Server.
"""

import os
from pathlib import Path
from typing import List, Optional

from dotenv import load_dotenv

# Load environment variables from .env file
# Look for .env file in parent of server directory
env_path = Path(__file__).parent.parent / ".env"
load_dotenv(dotenv_path=env_path)

# Environment Configuration
MEMORY_BACKEND_URL: str = os.getenv("MEMORY_BACKEND_URL", "http://localhost:8080")
CURSOR_MCP_PORT: int = int(os.getenv("CURSOR_MCP_PORT", "8001"))

# SSL verification setting for self-signed certificates
VERIFY_SSL: bool = os.getenv("VERIFY_SSL", "false").lower() in (
    "true",
    "1",
    "yes",
    "on",
)

# Optional auth token. If set, all requests must include it
MCP_AUTH_TOKEN: Optional[str] = os.getenv("MCP_AUTH_TOKEN") or None
AUTH_USERNAME: Optional[str] = os.getenv("AUTH_USERNAME") or None
AUTH_PASSWORD: Optional[str] = os.getenv("AUTH_PASSWORD") or None
TOKEN_TTL_SECONDS: int = int(os.getenv("TOKEN_TTL_SECONDS", "2419200"))

# Database Configuration
DATABASE_URL: str = os.getenv("DATABASE_URL", "sqlite:///./bigmemory.db")


DEBUG: bool = os.getenv("DEBUG", "false")

# Session Configuration Constants
DEFAULT_PRODUCED_FOR: str = "cursor_assistant"
DEFAULT_EPISODE_TYPE: str = "message"

# HTTP Configuration
REQUEST_TIMEOUT: int = 30
