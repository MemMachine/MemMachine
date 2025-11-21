"""
Configuration and constants for the MemMachine Extension MCP Server.
"""

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=Path(__file__).parent.parent / ".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore"
    )

    server_port: int = Field(
        default=8001,
        description="Port for the MemVerge Extension server",
        json_schema_extra={"env": "SERVER_PORT"}
    )

    mm_backend_url: str = Field(
        default="http://localhost:8080",
        description="URL of the MemMachine backend service",
        json_schema_extra={"env": "MM_BACKEND_URL"}
    )

    oauth_domain: str = Field(
        default="https://api.example.com",
        description="Domain for OAuth resource metadata",
        json_schema_extra={"env": "OAUTH_DOMAIN"}
    )

    # SSL verification setting for self-signed certificates
    verify_ssl: bool = Field(
        default=False,
        description="Whether to verify SSL certificates",
        json_schema_extra={"env": "VERIFY_SSL"}
    )

    # HTTP Configuration
    request_timeout: int = Field(
        default=30,
        description="HTTP request timeout in seconds",
        json_schema_extra={"env": "REQUEST_TIMEOUT"}
    )

    # Database Configuration
    database_url: str = Field(
        default="sqlite:///./bigmemory.db",
        description="Database connection URL",
        json_schema_extra={"env": "DATABASE_URL"}
    )

    # Debug mode
    debug: bool = Field(
        default=False,
        description="Enable debug mode",
        json_schema_extra={"env": "DEBUG"}
    )

    # JWT Configuration
    jwt_secret_key: str = Field(
        default="your-secret-key-change-this-in-production",
        description="Secret key for JWT token signing",
        json_schema_extra={"env": "JWT_SECRET_KEY"}
    )
    jwt_algorithm: str = Field(
        default="HS256",
        description="JWT signing algorithm",
        json_schema_extra={"env": "JWT_ALGORITHM"}
    )
    jwt_access_token_expire_hours: int = Field(
        default=24,
        description="JWT access token expiration time in hours",
        json_schema_extra={"env": "JWT_ACCESS_TOKEN_EXPIRE_HOURS"}
    )

    # Token Configuration
    token_ttl_seconds: int = Field(
        default=3600 * 24 * 7, # 7 days
        description="Default token time-to-live in seconds",
        json_schema_extra={"env": "TOKEN_TTL_SECONDS"}
    )


# Create a global settings instance
settings = Settings()
