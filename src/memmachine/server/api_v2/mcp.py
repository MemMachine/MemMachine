"""MCP tool implementations for MemMachine."""

import contextvars
import logging
import os
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Any, Self, cast

from fastapi import FastAPI
from fastmcp import FastMCP
from fastmcp.server.http import StarletteWithLifespan
from pydantic import BaseModel, Field, model_validator
from starlette import status
from starlette.applications import Starlette
from starlette.types import Lifespan, Receive, Scope, Send

from memmachine.common.configuration import load_config_yml_file
from memmachine.common.resource_manager.resource_manager import ResourceManagerImpl
from memmachine.main.memmachine import ALL_MEMORY_TYPES, MemMachine
from memmachine.server.api_v2.service import (
    _add_messages_to,
    _search_target_memories,
)
from memmachine.server.api_v2.spec import (
    AddMemoriesSpec,
    MemoryMessage,
    SearchMemoriesSpec,
    SearchResult,
)

logger = logging.getLogger(__name__)


class McpStatus:
    """Status codes for MCP responses."""

    SUCCESS = 200


class McpResponse(BaseModel):
    """Error model for MCP responses."""

    status: int
    """Error status code"""
    message: str
    """Error message"""


org_id_context_var: contextvars.ContextVar[str | None] = cast(
    contextvars.ContextVar[str | None],
    contextvars.ContextVar(
        "org_id_context",
        default=None,
    ),
)


def get_current_org_id() -> str | None:
    """
    Get the current organization ID from the contextvar.

    Returns:
        The org_id if available, None otherwise.

    """
    return org_id_context_var.get()


proj_id_context_var: contextvars.ContextVar[str | None] = cast(
    contextvars.ContextVar[str | None],
    contextvars.ContextVar(
        "proj_id_context",
        default=None,
    ),
)


def get_current_proj_id() -> str | None:
    """
    Get the current project ID from the contextvar.

    Returns:
        The proj_id if available, None otherwise.

    """
    return proj_id_context_var.get()


# Context variable to hold the current user for this request
user_id_context_var: contextvars.ContextVar[str | None] = cast(
    contextvars.ContextVar[str | None],
    contextvars.ContextVar(
        "user_id_context",
        default=None,
    ),
)


def get_current_user_id() -> str | None:
    """
    Get the current user ID from the contextvar.

    Returns:
        The user_id if available, None otherwise.

    """
    return user_id_context_var.get()


MCP_SUCCESS = McpResponse(status=McpStatus.SUCCESS, message="Success")

default_mcp_id = hex(uuid.getnode())


class Params(BaseModel):
    """Model with user_id that can be overridden by MM_USER_ID env var."""

    org_id: str = Field(
        default="",
        description="Organization ID, default to 'mcp-universal' for MCP universal access.",
    )

    proj_id: str = Field(
        default="",
        description="Project ID, default to mcp-{user_id} if user_id is provided."
        "default to mcp-{mac_address} if user_id is not provided.",
    )

    user_id: str = Field(
        default="",
        description=(
            "The unique identifier of the user whose memory is being updated. "
            "This ensures the new memory is stored under the correct profile."
        ),
        examples=["user"],
    )

    def _override_with_env_if_exists(self) -> None:
        """
        Override IDs from environment variables if set.

        Override user_id if MM_USER_ID or current user is set.
        Override proj_id if MM_PROJ_ID is set or user_id is set.
        Override org_id if MM_ORG_ID is set.
        """
        env_org_id = os.getenv("MC_ORG_ID")
        if env_org_id:
            self.org_id = env_org_id
        env_proj_id = os.getenv("MC_PROJ_ID")
        if env_proj_id:
            self.proj_id = env_proj_id
        env_user_id = os.environ.get("MM_USER_ID")
        if env_user_id:
            self.user_id = env_user_id

    def _override_with_context_var_if_exists(self) -> None:
        """Override user_id, proj_id, org_id from context vars if set."""
        org_id = get_current_org_id()
        if org_id:
            self.org_id = org_id
        proj_id = get_current_proj_id()
        if proj_id:
            self.proj_id = proj_id
        current_user_id = get_current_user_id()
        if current_user_id:
            self.user_id = current_user_id

    def _set_defaults(self) -> None:
        if not self.user_id:
            self.user_id = f"user-{default_mcp_id}"
        if not self.proj_id:
            self.proj_id = f"mcp-{self.user_id}"
        if not self.org_id:
            self.org_id = "mcp-universal"

    @model_validator(mode="after")
    def _update_params(self) -> Self:
        self._override_with_env_if_exists()
        self._override_with_context_var_if_exists()
        self._set_defaults()
        return self

    def to_add_memories_spec(self, content: str) -> AddMemoriesSpec:
        """Convert to AddMemoryParam."""
        return AddMemoriesSpec(
            org_id=self.org_id,
            project_id=self.proj_id,
            messages=[
                MemoryMessage(
                    content=content,
                    producer=self.user_id,
                    produced_for="unknown",
                    timestamp=datetime.now().astimezone(),
                    role="user",
                    metadata={
                        "user_id": self.user_id,
                    },
                )
            ],
        )

    def to_search_memories_spec(self, query: str, top_k: int) -> SearchMemoriesSpec:
        """Convert to SearchMemoriesParam."""
        return SearchMemoriesSpec(
            org_id=self.org_id,
            project_id=self.proj_id,
            query=query,
            top_k=top_k,
            filter=f"user_id='{self.user_id}'",
            types=ALL_MEMORY_TYPES,
        )


class ParamsContextMiddleware:
    """
    Extract user IDs from requests and store them in a ContextVar.

    Optionally override `user_id` from header "user-id".
    """

    def __init__(
        self,
        app: StarletteWithLifespan,
        org_header_name: str = "org-id",
        proj_header_name: str = "project-id",
        user_header_name: str = "user-id",
    ) -> None:
        """Store the wrapped app and the name of the header carrying user id."""
        self.app = app
        self._org_header_name = org_header_name
        self._proj_header_name = proj_header_name
        self._user_header_name = user_header_name

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        """Extract the org_id, proj_id and user id from the request headers and stash in context."""
        org_id: str = ""
        proj_id: str = ""
        user_id: str = ""

        if scope.get("type") == "http":
            headers = {
                k.decode().lower(): v.decode() for k, v in scope.get("headers", [])
            }
            org_id = headers.get(self._org_header_name.lower(), "")
            proj_id = headers.get(self._proj_header_name.lower(), "")
            user_id = headers.get(self._user_header_name.lower(), "")

        org_token = org_id_context_var.set(org_id)
        proj_token = proj_id_context_var.set(proj_id)
        user_id_token = user_id_context_var.set(user_id)
        try:
            await self.app(scope, receive, send)
        finally:
            org_id_context_var.reset(org_token)
            proj_id_context_var.reset(proj_token)
            user_id_context_var.reset(user_id_token)

    @property
    def lifespan(self) -> Lifespan[Starlette]:
        """Expose the underlying application's lifespan handler."""
        return self.app.lifespan


class MemMachineFastMCP(FastMCP):
    """Custom FastMCP subclass for MemMachine with authentication middleware."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:  # noqa: ANN401
        """Initialize the FastMCP app with parent configuration."""
        super().__init__(*args, **kwargs)

    def get_app(self, path: str | None = None) -> ParamsContextMiddleware:
        """Override to add authentication middleware."""
        http_app = super().http_app(path=path)
        return ParamsContextMiddleware(http_app)


mcp = MemMachineFastMCP("MemMachine")
mcp_app = mcp.get_app("/")


# === Globals ===
# Global instances for memory managers, initialized during app startup.
mem_machine: MemMachine | None = None


# === Lifespan Management ===


async def initialize_resource(config_file: str) -> MemMachine:
    """
    Initialize shared resources for profile and episodic memory.

    This is a temporary solution to unify ProfileMemory and Episodic Memory
    configuration. It initializes SemanticSessionManager and EpisodicMemoryManager
    instances, and establishes necessary connections (e.g., to the database).
    These resources are cleaned up on shutdown.

    Args:
        config_file: The path to the configuration file.

    Returns:
        A tuple containing the EpisodicMemoryManager, SemanticSessionManager,
        and SessionIdManager instances.

    """
    config = load_config_yml_file(config_file)
    resource_mgr = ResourceManagerImpl(config)
    return MemMachine(config, resource_mgr)


@asynccontextmanager
async def global_memory_lifespan() -> AsyncIterator[None]:
    """
    Handle application startup and shutdown events.

    Initializes the ProfileMemory and EpisodicMemoryManager instances,
    and establishes necessary connections (e.g., to the database).
    These resources are cleaned up on shutdown.
    """
    await init_global_memory()
    yield
    await shutdown_global_memory()


async def init_global_memory() -> None:
    """Initialize global resource manager based on configuration."""
    config_file = os.getenv("MEMORY_CONFIG", "cfg.yml")
    global mem_machine
    mem_machine = await initialize_resource(config_file)
    if mem_machine is not None:
        await mem_machine.start()


async def shutdown_global_memory() -> None:
    """Shut down global resources and close connections."""
    global mem_machine
    if mem_machine is not None:
        await mem_machine.stop()


@asynccontextmanager
async def mcp_http_lifespan(application: FastAPI) -> AsyncIterator[None]:
    """
    Manage the combined lifespan of the main app and the MCP app.

    This context manager chains the `http_app_lifespan` (for main application
    resources like memory managers) and the `mcp_app.lifespan` (for
    MCP-specific resources). It ensures that all resources are initialized on
    startup and cleaned up on shutdown in the correct order.

    Args:
        application: The FastAPI application instance.

    """
    async with global_memory_lifespan(), mcp_app.lifespan(application):
        application.state.mem_machine = mem_machine
        yield


@mcp.tool(
    name="add_memory",
    description=(
        "Store important new information about the user or conversation into memory. "
        "Use this automatically whenever the user shares new facts, preferences, "
        "plans, emotions, or other details that could be useful for future context. "
        "Include the **full conversation context** in the `content` field — not just a snippet. "
        "This tool writes to both short-term (episodic) and long-term (profile) memory, "
        "so that future interactions can recall relevant background knowledge even "
        "across different sessions. "
        "\n\n**Parameters**: Supports both nested (param object) and flat (user_id, content) styles."
    ),
)
async def mcp_add_memory(
    content: str,
    org_id: str = "",
    proj_id: str = "",
    user_id: str = "",
) -> McpResponse:
    """
    Add a new memory for the specified user.

    The model should call this whenever it detects new information
    worth remembering — for example, user preferences, recurring topics,
    or summaries of recent exchanges.

    This function supports both nested and flat parameter styles:
    - Nested: pass an AddMemoryParam object to the param argument
    - Flat: pass user_id and content as separate arguments

    Args:
        org_id: The organization ID (optional, flat style).
        proj_id: The project ID (optional, flat style).
        user_id: The unique identifier of the user (flat style).
        content: The complete context or summary to store in memory (flat style).

    Returns:
        McpResponse indicating success or failure.

    """
    global mem_machine
    if mem_machine is None:
        return McpResponse(
            status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            message="MemMachine is not initialized",
        )
    try:
        param = Params(
            org_id=org_id,
            proj_id=proj_id,
            user_id=user_id,
        )
        spec = param.to_add_memories_spec(content)
        await _add_messages_to(
            target_memories=ALL_MEMORY_TYPES, spec=spec, memmachine=mem_machine
        )
    except Exception as e:
        status_code = status.HTTP_422_UNPROCESSABLE_CONTENT
        logger.exception("Failed to add memory")
        return McpResponse(status=status_code, message=str(e))
    return MCP_SUCCESS


@mcp.tool(
    name="search_memory",
    description=(
        "Retrieve relevant context, memories or profile for a user whenever "
        "context is missing or unclear. Use this whenever you need to recall "
        "what has been previously discussed, "
        "even if it was from an earlier conversation or session. "
        "This searches both profile memory (long-term user traits and facts) "
        "and episodic memory (past conversations and experiences). "
        "\n\n**Parameters**: Supports both nested (param object) and flat (user_id, query, limit) styles."
    ),
)
async def mcp_search_memory(
    query: str,
    org_id: str = "",
    proj_id: str = "",
    user_id: str = "",
    top_k: int = 20,
) -> McpResponse | SearchResult:
    """
    Search memory for the specified user.

    This function supports both nested and flat parameter styles:
    - Nested: pass a SearchMemoryParam object to the param argument
    - Flat: pass user_id, query, and optionally limit as separate arguments

    Args:
        org_id: The organization ID (optional, flat style).
        proj_id: The project ID (optional, flat style).
        user_id: The unique identifier of the user (flat style).
        query: The current user message or topic of discussion (flat style).
        top_k: The maximum number of memory entries to retrieve (flat style). Defaults to 5.

    Returns:
        McpResponse on failure, or SearchResult on success

    """
    global mem_machine
    if mem_machine is None:
        return McpResponse(
            status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            message="MemMachine is not initialized",
        )
    try:
        param = Params(
            org_id=org_id,
            proj_id=proj_id,
            user_id=user_id,
        )
        spec = param.to_search_memories_spec(query, top_k)
        return await _search_target_memories(
            target_memories=ALL_MEMORY_TYPES, spec=spec, memmachine=mem_machine
        )
    except Exception as e:
        status_code = status.HTTP_422_UNPROCESSABLE_CONTENT
        logger.exception("Failed to search memory")
        return McpResponse(status=status_code, message=str(e))
