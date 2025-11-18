"""FastAPI application for the MemMachine memory system.

This module sets up and runs a FastAPI web server that provides endpoints for
interacting with the Profile Memory and Episodic Memory components.
It includes:
- API endpoints for adding and searching memories.
- Integration with FastMCP for exposing memory functions as tools to LLMs.
- Pydantic models for request and response validation.
- Lifespan management for initializing and cleaning up resources like database
  connections and memory managers.
"""

import argparse
import asyncio
import contextvars
import copy
import logging
import os
from contextlib import asynccontextmanager
from importlib import import_module
from typing import Any, Self, cast

import uvicorn
import yaml
from dotenv import load_dotenv
from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.params import Depends
from fastapi.responses import Response
from fastmcp import FastMCP
from fastmcp.server.http import StarletteWithLifespan
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from pydantic import BaseModel, Field, model_validator
from starlette.applications import Starlette
from starlette.types import Lifespan, Receive, Scope, Send

from memmachine.common.embedder import EmbedderBuilder
from memmachine.common.language_model import LanguageModelBuilder
from memmachine.common.metrics_factory import MetricsFactoryBuilder
from memmachine.episodic_memory.data_types import ContentType
from memmachine.episodic_memory.episodic_memory import (
    AsyncEpisodicMemory,
    EpisodicMemory,
)
from memmachine.episodic_memory.episodic_memory_manager import (
    EpisodicMemoryManager,
)
from memmachine.profile_memory.profile_memory import ProfileMemory
from memmachine.profile_memory.prompt_provider import ProfilePrompt
from memmachine.profile_memory.storage.asyncpg_profile import AsyncPgProfileStorage

logger = logging.getLogger(__name__)


class AppConst:
    """Constants for app and header key names."""

    DEFAULT_GROUP_ID = "default"
    """Default value for group id when not provided."""

    DEFAULT_SESSION_ID = "default"
    """Default value for session id when not provided."""

    DEFAULT_USER_ID = "default"
    """Default value for user id when not provided."""

    DEFAULT_PRODUCER_ID = "default"
    """Default value for producer id when not provided."""

    DEFAULT_EPISODE_TYPE = "message"
    """Default value for episode type when not provided."""

    GROUP_ID_KEY = "group-id"
    """Header key for group ID."""

    SESSION_ID_KEY = "session-id"
    """Header key for session ID."""

    AGENT_ID_KEY = "agent-id"
    """Header key for agent ID."""

    USER_ID_KEY = "user-id"
    """Header key for user ID."""

    GROUP_ID_DOC = (
        "Unique identifier for a group or shared context. "
        "Used as the main filtering property. "
        "For single-user use cases, this can be the same as `user_id`. "
        "Defaults to `default` if not provided and user ids is empty. "
        "Defaults to the first user id if user ids are provided."
    )

    AGENT_ID_DOC = (
        "List of agent identifiers associated with this session. "
        "Useful if multiple AI agents participate in the same context. "
        "Defaults to `[]` if not provided."
    )

    USER_ID_DOC = (
        "List of user identifiers participating in this session. "
        "Used to isolate memories and data per user. "
        "Defaults to `['default']` if not provided."
    )

    SESSION_ID_DOC = (
        "Unique identifier for a specific session or conversation. "
        "Can represent a chat thread, Slack channel, or conversation instance. "
        "Should be unique per conversation to avoid data overlap. "
        "Defaults 'default' if not provided and user ids is empty. "
        "Defaults to the first `user_id` if user ids are provided."
    )

    PRODUCER_DOC = (
        "Identifier of the entity producing the episode. "
        "Default to the first `user_id` in the session if not provided. "
        "Default to `default` if user_id is not available."
    )

    PRODUCER_FOR_DOC = "Identifier of the entity for whom the episode is produced."

    EPISODE_CONTENT_DOC = "Content of the memory episode."

    EPISODE_TYPE_DOC = "Type of the episode content (e.g., message)."

    EPISODE_META_DOC = "Additional metadata for the episode."

    GROUP_ID_EXAMPLES = ["group-1234", "project-alpha", "team-chat"]
    AGENT_ID_EXAMPLES = ["crm", "healthcare", "sales", "agent-007"]
    USER_ID_EXAMPLES = ["user-001", "alice@example.com"]
    SESSION_ID_EXAMPLES = ["session-5678", "chat-thread-42", "conversation-abc"]
    PRODUCER_EXAMPLES = ["chatbot", "user-1234", "agent-007"]
    PRODUCER_FOR_EXAMPLES = ["user-1234", "team-alpha", "project-xyz"]
    EPISODE_CONTENT_EXAMPLES = ["Met at the coffee shop to discuss project updates."]
    EPISODE_TYPE_EXAMPLES = ["message"]
    EPISODE_META_EXAMPLES = [{"mood": "happy", "location": "office"}]


# Request session data
class SessionData(BaseModel):
    """Metadata used to organize and filter memory or conversation context.

    Each ID serves a different level of data separation:
    - `group_id`: identifies a shared context (e.g., a group chat or project).
    - `user_id`: identifies individual participants within the group.
    - `agent_id`: identifies the AI agent(s) involved in the session.
    - `session_id`: identifies a specific conversation thread or session.
    """

    group_id: str = Field(
        default="",
        description=AppConst.GROUP_ID_DOC,
        examples=AppConst.GROUP_ID_EXAMPLES,
    )

    agent_id: list[str] = Field(
        default=[],
        description=AppConst.AGENT_ID_DOC,
        examples=AppConst.AGENT_ID_EXAMPLES,
    )

    user_id: list[str] = Field(
        default=[],
        description=AppConst.USER_ID_DOC,
        examples=AppConst.USER_ID_EXAMPLES,
    )

    session_id: str = Field(
        default="",
        description=AppConst.SESSION_ID_DOC,
        examples=AppConst.SESSION_ID_EXAMPLES,
    )

    def merge(self, other: Self) -> None:
        """Merge another SessionData into this one in place.

        - Combine and deduplicate list fields.
        - Overwrite string fields if the new value is set.
        """

        def merge_lists(a: list[str], b: list[str]) -> list[str]:
            if a and b:
                ret = list(dict.fromkeys(a + b))  # preserve order & unique
            else:
                ret = a or b
            return sorted(ret)

        if other.group_id and other.group_id != AppConst.DEFAULT_GROUP_ID:
            self.group_id = other.group_id

        if other.session_id and other.session_id != AppConst.DEFAULT_SESSION_ID:
            self.session_id = other.session_id

        if other.user_id == [AppConst.DEFAULT_USER_ID]:
            other.user_id = []

        self.agent_id = merge_lists(self.agent_id, other.agent_id)
        self.user_id = merge_lists(self.user_id, other.user_id)

    def first_user_id(self) -> str:
        """Returns the first user ID if available, else default user id."""
        return self.user_id[0] if self.user_id else AppConst.DEFAULT_USER_ID

    def combined_user_ids(self) -> str:
        """format groups id to <size>#<user-id><size>#<user-id>..."""
        return "".join([f"{len(uid)}#{uid}" for uid in sorted(self.user_id)])

    def from_user_id_or(self, default_value: str) -> str:
        """returns the first user id or combined user ids as a default string."""
        size_user_id = len(self.user_id)
        if size_user_id == 0:
            return default_value
        elif size_user_id == 1:
            return self.first_user_id()
        else:
            return self.combined_user_ids()

    @model_validator(mode="after")
    def _set_default_group_id(self) -> Self:
        """Defaults group_id to default gr."""
        if not self.group_id:
            self.group_id = self.from_user_id_or(AppConst.DEFAULT_GROUP_ID)
        return self

    @model_validator(mode="after")
    def _set_default_session_id(self) -> Self:
        """Defaults session_id to 'default' if not set."""
        if not self.session_id:
            self.session_id = self.from_user_id_or(AppConst.DEFAULT_SESSION_ID)
        return self

    @model_validator(mode="after")
    def _set_default_user_id(self) -> Self:
        """Defaults user_id to ['default'] if not set."""
        if len(self.user_id) == 0 and len(self.agent_id) == 0:
            self.user_id = [AppConst.DEFAULT_USER_ID]
        else:
            self.user_id = sorted(self.user_id)
        return self

    def is_valid(self) -> bool:
        """Return False if the session data is invalid (both group_id and
        session_id are empty), True otherwise.
        """
        return (
            self.group_id != "" and self.session_id != "" and self.first_user_id() != ""
        )


class RequestWithSession(BaseModel):
    """Base class for requests that include session data."""

    session: SessionData | None = Field(
        None,
        description="Session field in the body is deprecated. "
        "Use header-based session instead.",
    )

    def log_error_with_session(self, e: HTTPException, message: str):
        sess = self.get_session()
        session_name = (
            f"{sess.group_id}-{sess.agent_id}-{sess.user_id}-{sess.session_id}"
        )
        logger.error(f"{message} for %s", session_name)
        logger.error(e)

    def get_session(self) -> SessionData:
        if self.session is None:
            return SessionData(
                group_id="",
                agent_id=[],
                user_id=[],
                session_id="",
            )
        return self.session

    def new_404_not_found_error(self, message: str):
        session = self.get_session()
        return HTTPException(
            status_code=404,
            detail=f"{message} for {session.user_id},"
            f"{session.session_id},"
            f"{session.group_id},"
            f"{session.agent_id}",
        )

    def merge_session(self, session: SessionData) -> None:
        """Merge another SessionData into this one in place.

        - Combine and deduplicate list fields.
        - Overwrite string fields if the new value is set.
        """
        if self.session is None:
            self.session = session
        else:
            self.session.merge(session)

    def validate_session(self) -> None:
        """Validate that the session data is not empty.
        Raises:
            RequestValidationError: If the session data is empty.
        """
        if self.session is None or not self.session.is_valid():
            # Raise the same type of validation error FastAPI uses
            raise RequestValidationError(
                [
                    {
                        "loc": ["header", "session"],
                        "msg": "group_id or session_id cannot be empty",
                        "type": "value_error.missing",
                    }
                ]
            )

    def merge_and_validate_session(self, other: SessionData) -> None:
        """Merge another SessionData into this one in place and validate.

        - Combine and deduplicate list fields.
        - Overwrite string fields if the new value is set.
        - Validate that the resulting session data is not empty.

        Raises:
            RequestValidationError: If the resulting session data is empty.
        """
        self.merge_session(other)
        self.validate_session()

    def update_response_session_header(self, response: Response | None) -> None:
        """Update the response headers with the session data."""
        if response is None:
            return
        sess = self.get_session()
        if sess.group_id:
            response.headers[AppConst.GROUP_ID_KEY] = sess.group_id
        if sess.session_id:
            response.headers[AppConst.SESSION_ID_KEY] = sess.session_id
        if sess.agent_id:
            response.headers[AppConst.AGENT_ID_KEY] = ",".join(sess.agent_id)
        if sess.user_id:
            response.headers[AppConst.USER_ID_KEY] = ",".join(sess.user_id)


# === Request Models ===
class NewEpisode(RequestWithSession):
    """Request model for adding a new memory episode."""

    producer: str = Field(
        default="",
        description=AppConst.PRODUCER_DOC,
        examples=AppConst.PRODUCER_EXAMPLES,
    )

    produced_for: str = Field(
        default="",
        description=AppConst.PRODUCER_FOR_DOC,
        examples=AppConst.PRODUCER_FOR_EXAMPLES,
    )

    episode_content: str | list[float] = Field(
        default="",
        description=AppConst.EPISODE_CONTENT_DOC,
        examples=AppConst.EPISODE_CONTENT_EXAMPLES,
    )

    episode_type: str = Field(
        default=AppConst.DEFAULT_EPISODE_TYPE,
        description=AppConst.EPISODE_TYPE_DOC,
        examples=AppConst.EPISODE_TYPE_EXAMPLES,
    )

    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description=AppConst.EPISODE_META_DOC,
        examples=AppConst.EPISODE_META_EXAMPLES,
    )

    @model_validator(mode="after")
    def _set_default_producer_id(self) -> Self:
        """Defaults session_id to 'default' if not set."""
        if self.producer == "":
            if self.session is not None:
                self.producer = self.session.from_user_id_or("")
        if self.producer == "":
            self.session_id = AppConst.DEFAULT_PRODUCER_ID
        return self


class SearchQuery(RequestWithSession):
    """Request model for searching memories."""

    query: str
    filter: dict[str, Any] | None = None
    limit: int | None = None


def _split_str_to_list(s: str) -> list[str]:
    return [x.strip() for x in s.split(",") if x.strip() != ""]


async def _get_session_from_header(
    request: Request,
    group_id: str = Header(
        AppConst.DEFAULT_GROUP_ID,
        alias=AppConst.GROUP_ID_KEY,
        description=AppConst.GROUP_ID_DOC,
        examples=AppConst.GROUP_ID_EXAMPLES,
    ),
    session_id: str = Header(
        AppConst.DEFAULT_SESSION_ID,
        alias=AppConst.SESSION_ID_KEY,
        description=AppConst.SESSION_ID_DOC,
        examples=AppConst.SESSION_ID_EXAMPLES,
    ),
    agent_id: str = Header(
        "",
        alias=AppConst.AGENT_ID_KEY,
        description=AppConst.AGENT_ID_DOC,
        examples=AppConst.AGENT_ID_EXAMPLES,
    ),
    user_id: str = Header(
        "",
        alias=AppConst.USER_ID_KEY,
        description=AppConst.USER_ID_DOC,
        examples=AppConst.USER_ID_EXAMPLES,
    ),
) -> SessionData:
    """Extract session data from headers and return a SessionData object."""
    group_id_keys = [AppConst.GROUP_ID_KEY, "group_id"]
    session_id_keys = [AppConst.SESSION_ID_KEY, "session_id"]
    agent_id_keys = [AppConst.AGENT_ID_KEY, "agent_id"]
    user_id_keys = [AppConst.USER_ID_KEY, "user_id"]
    headers = request.headers

    def get_with_alias(possible_keys: list[str], default: str):
        for key in possible_keys:
            for hk, hv in headers.items():
                if hk.lower() == key.lower():
                    return hv
        return default

    group_id = get_with_alias(group_id_keys, group_id)
    session_id = get_with_alias(session_id_keys, session_id)
    agent_id = get_with_alias(agent_id_keys, agent_id)
    user_id = get_with_alias(user_id_keys, user_id)
    return SessionData(
        group_id=group_id,
        session_id=session_id,
        agent_id=_split_str_to_list(agent_id),
        user_id=_split_str_to_list(user_id),
    )


# === Response Models ===
class SearchResult(BaseModel):
    """Response model for memory search results."""

    status: int = 0
    content: dict[str, Any]


class MemorySession(BaseModel):
    """Response model for session information."""

    user_ids: list[str]
    session_id: str
    group_id: str | None
    agent_ids: list[str] | None


class AllSessionsResponse(BaseModel):
    """Response model for listing all sessions."""

    sessions: list[MemorySession]


class DeleteDataRequest(RequestWithSession):
    """Request model for deleting all data for a session."""

    pass


# === Globals ===
# Global instances for memory managers, initialized during app startup.
profile_memory: ProfileMemory | None = None
episodic_memory: EpisodicMemoryManager | None = None


# === Lifespan Management ===


async def initialize_resource(
    config_file: str,
) -> tuple[EpisodicMemoryManager, ProfileMemory]:
    """
    This is a temporary solution to unify the ProfileMemory and Episodic Memory
    configuration.
    Initializes the ProfileMemory and EpisodicMemoryManager instances,
    and establishes necessary connections (e.g., to the database).
    These resources are cleaned up on shutdown.
    Args:
        config_file: The path to the configuration file.
    Returns:
        A tuple containing the EpisodicMemoryManager and ProfileMemory instances.
    """

    try:
        yaml_config = yaml.safe_load(open(config_file, encoding="utf-8"))
    except FileNotFoundError:
        raise FileNotFoundError(f"Config file {config_file} not found")
    except yaml.YAMLError:
        raise ValueError(f"Config file {config_file} is not valid YAML")
    except Exception as e:
        raise e

    def config_to_lowercase(data: Any) -> Any:
        """Recursively converts all dictionary keys in a nested structure
        to lowercase."""
        if isinstance(data, dict):
            return {k.lower(): config_to_lowercase(v) for k, v in data.items()}
        if isinstance(data, list):
            return [config_to_lowercase(i) for i in data]
        return data

    yaml_config = config_to_lowercase(yaml_config)

    # if the model is defined in the config, use it.
    profile_config = yaml_config.get("profile_memory", {})

    # create LLM model from the configuration
    model_config = yaml_config.get("model", {})

    model_name = profile_config.get("llm_model")
    if model_name is None:
        raise ValueError("Model not configured in config file for profile memory")

    model_def = model_config.get(model_name)
    if model_def is None:
        raise ValueError(f"Can not find definition of model{model_name}")

    profile_model = copy.deepcopy(model_def)
    metrics_manager = MetricsFactoryBuilder.build("prometheus", {}, {})
    profile_model["metrics_factory_id"] = "prometheus"
    metrics_injection = {}
    metrics_injection["prometheus"] = metrics_manager
    model_vendor = profile_model.pop("model_vendor")
    llm_model = LanguageModelBuilder.build(
        model_vendor, profile_model, metrics_injection
    )

    # create embedder
    embedders = yaml_config.get("embedder", {})
    embedder_id = profile_config.get("embedding_model")
    if embedder_id is None:
        raise ValueError(
            "Embedding model not configured in config file for profile memory"
        )

    embedder_def = embedders.get(embedder_id)
    if embedder_def is None:
        raise ValueError(f"Can not find definition of embedder {embedder_id}")

    embedder_config = copy.deepcopy(embedder_def["config"])
    if embedder_def["provider"] == "openai":
        embedder_config["metrics_factory_id"] = "prometheus"

    embeddings = EmbedderBuilder.build(
        embedder_def["provider"], embedder_config, metrics_injection
    )

    # Get the database configuration
    # get DB config from configuration file is available
    db_config_name = profile_config.get("database")
    if db_config_name is None:
        raise ValueError("Profile database not configured in config file")
    db_config = yaml_config.get("storage", {})
    db_config = db_config.get(db_config_name)
    if db_config is None:
        raise ValueError(f"Can not find configuration for database {db_config_name}")

    prompt_file = profile_config.get("prompt", "profile_prompt")
    prompt_module = import_module(f".prompt.{prompt_file}", __package__)
    profile_prompt = ProfilePrompt.load_from_module(prompt_module)

    profile_storage = AsyncPgProfileStorage.build_config(
        {
            "host": db_config.get("host", "localhost"),
            "port": db_config.get("port", 0),
            "user": db_config.get("user", ""),
            "password": db_config.get("password", ""),
            "database": db_config.get("database", ""),
        }
    )

    profile_memory = ProfileMemory(
        model=llm_model,
        embeddings=embeddings,
        profile_storage=profile_storage,
        prompt=profile_prompt,
    )
    episodic_memory = EpisodicMemoryManager.create_episodic_memory_manager(config_file)
    return episodic_memory, profile_memory


async def init_global_memory():
    config_file = os.getenv("MEMORY_CONFIG", "cfg.yml")

    global episodic_memory
    global profile_memory
    episodic_memory, profile_memory = await initialize_resource(config_file)
    await profile_memory.startup()


async def shutdown_global_memory():
    global episodic_memory
    global profile_memory
    if profile_memory is not None:
        await profile_memory.cleanup()
    if episodic_memory is not None:
        await episodic_memory.shut_down()


@asynccontextmanager
async def global_memory_lifespan():
    """Handles application startup and shutdown events.

    Initializes the ProfileMemory and EpisodicMemoryManager instances,
    and establishes necessary connections (e.g., to the database).
    These resources are cleaned up on shutdown.
    """
    await init_global_memory()
    yield
    await shutdown_global_memory()


# Context variable to hold the current user for this request
user_id_context_var: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "user_id_context", default=None
)


def get_current_user_id() -> str | None:
    """
    Get the current user ID from the contextvar.

    Returns:
        The user_id if available, None otherwise.
    """
    return user_id_context_var.get()


class UserIDContextMiddleware:
    """
    Middleware that extracts the user_id from the request and stores it
    in a ContextVar for easy access within MCP tools.

    Optionally override `user_id` from header "user-id".
    """

    def __init__(self, app: StarletteWithLifespan, header_name: str = "user-id"):
        self.app = app
        self.header_name = header_name

    async def __call__(self, scope: Scope, receive: Receive, send: Send):
        user_id: str | None = None

        if scope.get("type") == "http":
            headers = {
                k.decode().lower(): v.decode() for k, v in scope.get("headers", [])
            }
            user_id = headers.get(self.header_name.lower(), None)

        token = user_id_context_var.set(user_id)
        try:
            await self.app(scope, receive, send)
        finally:
            user_id_context_var.reset(token)

    @property
    def lifespan(self) -> Lifespan[Starlette]:
        return self.app.lifespan


class MemMachineFastMCP(FastMCP):
    """Custom FastMCP subclass for MemMachine with authentication middleware."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def get_app(self, path: str | None = None) -> UserIDContextMiddleware:
        """Override to add authentication middleware."""
        http_app = super().http_app(path=path)
        return UserIDContextMiddleware(http_app)


class McpStatus:
    """Status codes for MCP responses."""

    SUCCESS = 200


class McpResponse(BaseModel):
    """Error model for MCP responses."""

    status: int
    """Error status code"""
    message: str
    """Error message"""


mcpSuccess = McpResponse(status=McpStatus.SUCCESS, message="Success")


class UserIDWithEnv(BaseModel):
    """
    Model with user_id that can be overridden by MM_USER_ID env var.
    """

    user_id: str = Field(
        ...,
        description=(
            "The unique identifier of the user whose memory is being updated. "
            "This ensures the new memory is stored under the correct profile."
        ),
        examples=["user"],
    )

    @model_validator(mode="after")
    def _update_user(self):
        """is MM_USER_ID env var set? If so, override user_id"""
        env_user_id = os.environ.get("MM_USER_ID")
        if env_user_id:
            self.user_id = env_user_id
        current_user_id = get_current_user_id()
        if current_user_id:
            self.user_id = current_user_id
        return self


class AddMemoryParam(UserIDWithEnv):
    """
    Parameters for adding memory.

    This model is used by chatbots or agents to store important information
    into memory for a specific user. The content should contain the **full
    conversational or contextual summary**, not just a short fragment.

    Chatbots should call this when they learn new facts about the user,
    observe recurring behaviors, or summarize recent discussions.
    """

    content: str = Field(
        ...,
        description=(
            "The complete context or summary to store in memory. "
            "When adding memory, include **all relevant background**, "
            "such as the current user message, prior conversation context, "
            "and any inferred meaning or conclusions. "
            "This allows future recall to be more accurate and useful."
        ),
        examples=[
            (
                "User discussed plans to visit Shanghai next month. "
                "They enjoy historical architecture and local cuisine. "
                "Mentioned interest in the Yu Garden and traditional tea houses."
            )
        ],
    )

    def get_new_episode(self) -> NewEpisode:
        """Convert to NewEpisode object."""
        session = SessionData(user_id=[self.user_id])
        return NewEpisode(
            session=session,
            episode_content=self.content,
            producer=self.user_id,
            produced_for=self.user_id,
        )


class SearchMemoryParam(UserIDWithEnv):
    """
    Parameters for searching a user's memory.

    This model is used by chatbots and agents to retrieve relevant
    information from both **profile memory** (long-term traits and facts)
    and **episodic memory** (past interactions and experiences).

    Chatbots should call this search automatically when context or
    prior information about the user or topic is missing.
    """

    query: str = Field(
        ...,
        description=(
            "The current user message or topic of discussion. "
            "This will be used as the semantic query to find related memories. "
            "If the chatbot is unsure about context or past topics, use the current "
            "user message as the query to recall relevant background."
        ),
        examples=["Tell me more about our trip to New York last summer."],
    )

    limit: int = Field(
        5,
        description=(
            "The maximum number of memory entries to retrieve. "
            "Defaults to 5 for efficiency. Increase if deeper recall is needed."
        ),
        ge=1,
        le=50,
        examples=[5],
    )

    def get_search_query(self) -> SearchQuery:
        """Convert to SearchQuery object."""
        session = SessionData(user_id=[self.user_id])
        return SearchQuery(
            session=session,
            query=self.query,
            limit=self.limit,
        )


mcp = MemMachineFastMCP("MemMachine")
mcp_app = mcp.get_app("/")


@asynccontextmanager
async def mcp_http_lifespan(application: FastAPI):
    """Manages the combined lifespan of the main app and the MCP app.

    This context manager chains the `http_app_lifespan` (for main application
    resources like memory managers) and the `mcp_app.lifespan` (for
    MCP-specific resources). It ensures that all resources are initialized on
    startup and cleaned up on shutdown in the correct order.

    Args:
        application: The FastAPI application instance.
    """
    async with global_memory_lifespan():
        async with mcp_app.lifespan(application):
            yield


app = FastAPI(lifespan=mcp_http_lifespan)
app.mount("/mcp", mcp_app)


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
    param: AddMemoryParam | None = None,
    user_id: str | None = None,
    content: str | None = None,
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
        param: The memory entry containing the user ID and full context (nested style).
        user_id: The unique identifier of the user (flat style).
        content: The complete context or summary to store in memory (flat style).
    Returns:
        McpResponse indicating success or failure.
    """
    # Handle flat parameters by constructing the Pydantic model
    if param is None:
        if user_id is None or content is None:
            return McpResponse(
                status=400,
                message="Either param or both user_id and content must be provided",
            )
        param = AddMemoryParam(user_id=user_id, content=content)

    episode = param.get_new_episode()
    try:
        await _add_memory(episode)
    except HTTPException as e:
        episode.log_error_with_session(e, "Failed to add memory episode")
        return McpResponse(status=e.status_code, message=str(e.detail))
    return mcpSuccess


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
    param: SearchMemoryParam | None = None,
    user_id: str | None = None,
    query: str | None = None,
    limit: int = 5,
) -> McpResponse | SearchResult:
    """
    Search memory for the specified user.

    This function supports both nested and flat parameter styles:
    - Nested: pass a SearchMemoryParam object to the param argument
    - Flat: pass user_id, query, and optionally limit as separate arguments

    Args:
        param: The search memory parameter (nested style).
        user_id: The unique identifier of the user (flat style).
        query: The current user message or topic of discussion (flat style).
        limit: The maximum number of memory entries to retrieve (flat style). Defaults to 5.
    Returns:
        McpResponse on failure, or SearchResult on success
    """
    # Handle flat parameters by constructing the Pydantic model
    if param is None:
        if user_id is None or query is None:
            return McpResponse(
                status=400,
                message="Either param or both user_id and query must be provided",
            )
        param = SearchMemoryParam(user_id=user_id, query=query, limit=limit)

    search_result = param.get_search_query()
    try:
        return await _search_memory(search_result)
    except HTTPException as e:
        search_result.log_error_with_session(e, "Failed to search memory")
        return McpResponse(status=e.status_code, message=str(e.detail))


# === Route Handlers ===
@app.post("/v1/memories")
async def add_memory(
    episode: NewEpisode,
    response: Response,
    session: SessionData = Depends(_get_session_from_header),  # type: ignore
):
    """Adds a memory episode to both episodic and profile memory.

    This endpoint first retrieves the appropriate episodic memory instance
    based on the session context (group, agent, user, session IDs). It then
    adds the episode to the episodic memory. If successful, it also passes
    the message to the profile memory for ingestion.

    Args:
        episode: The NewEpisode object containing the memory details.
        response: The HTTP response object to update headers.
        session: The session data from headers to merge with the request.

    Raises:
        HTTPException: 404 if no matching episodic memory instance is found.
        HTTPException: 400 if the producer or produced_for IDs are invalid
                       for the given context.
    """
    episode.merge_and_validate_session(session)
    episode.update_response_session_header(response)
    await _add_memory(episode)


async def _add_memory(episode: NewEpisode):
    """Adds a memory episode to both episodic and profile memory.
    Internal function.  Shared by both REST API and MCP API

    See the docstring for add_memory() for details."""
    session = episode.get_session()
    group_id = session.group_id
    inst: EpisodicMemory | None = await cast(
        EpisodicMemoryManager, episodic_memory
    ).get_episodic_memory_instance(
        group_id=group_id if group_id is not None else "",
        agent_id=session.agent_id,
        user_id=session.user_id,
        session_id=session.session_id,
    )
    if inst is None:
        raise episode.new_404_not_found_error("unable to find episodic memory")
    async with AsyncEpisodicMemory(inst) as inst:
        success = await inst.add_memory_episode(
            producer=episode.producer,
            produced_for=episode.produced_for,
            episode_content=episode.episode_content,
            episode_type=episode.episode_type,
            content_type=ContentType.STRING,
            metadata=episode.metadata,
        )
        if not success:
            raise HTTPException(
                status_code=400,
                detail=f"""either {episode.producer} or {episode.produced_for}
                        is not in {session.user_id}
                        or {session.agent_id}""",
            )

        ctx = inst.get_memory_context()
        await cast(ProfileMemory, profile_memory).add_persona_message(
            str(episode.episode_content),
            episode.metadata if episode.metadata is not None else {},
            {
                "group_id": ctx.group_id,
                "session_id": ctx.session_id,
                "producer": episode.producer,
                "produced_for": episode.produced_for,
            },
            user_id=episode.producer,
        )


@app.post("/v1/memories/episodic")
async def add_episodic_memory(
    episode: NewEpisode,
    response: Response,
    session: SessionData = Depends(_get_session_from_header),  # type: ignore
):
    """Adds a memory episode to episodic memory only.

    This endpoint first retrieves the appropriate episodic memory instance
    based on the session context (group, agent, user, session IDs). It then
    adds the episode to the episodic memory. If successful, it also passes
    the message to the profile memory for ingestion.

    Args:
        episode: The NewEpisode object containing the memory details.
        response: The HTTP response object to update headers.
        session: The session data from headers to merge with the request.

    Raises:
        HTTPException: 404 if no matching episodic memory instance is found.
        HTTPException: 400 if the producer or produced_for IDs are invalid
                       for the given context.
    """
    episode.merge_and_validate_session(session)
    episode.update_response_session_header(response)
    await _add_episodic_memory(episode)


async def _add_episodic_memory(episode: NewEpisode):
    """Adds a memory episode to both episodic and profile memory.
    Internal function.  Shared by both REST API and MCP API

    See the docstring for add_episodic_memory() for details.
    """
    session = episode.get_session()
    group_id = session.group_id
    inst: EpisodicMemory | None = await cast(
        EpisodicMemoryManager, episodic_memory
    ).get_episodic_memory_instance(
        group_id=group_id if group_id is not None else "",
        agent_id=session.agent_id,
        user_id=session.user_id,
        session_id=session.session_id,
    )
    if inst is None:
        raise episode.new_404_not_found_error("unable to find episodic memory")
    async with AsyncEpisodicMemory(inst) as inst:
        success = await inst.add_memory_episode(
            producer=episode.producer,
            produced_for=episode.produced_for,
            episode_content=episode.episode_content,
            episode_type=episode.episode_type,
            content_type=ContentType.STRING,
            metadata=episode.metadata,
        )
        if not success:
            raise HTTPException(
                status_code=400,
                detail=f"""either {episode.producer} or {episode.produced_for}
                        is not in {session.user_id}
                        or {session.agent_id}""",
            )


@app.post("/v1/memories/profile")
async def add_profile_memory(
    episode: NewEpisode,
    response: Response,
    session: SessionData = Depends(_get_session_from_header),  # type: ignore
):
    """Adds a memory episode to both profile memory.

    This endpoint first retrieves the appropriate episodic memory instance
    based on the session context (group, agent, user, session IDs). It then
    adds the episode to the episodic memory. If successful, it also passes
    the message to the profile memory for ingestion.

    Args:
        episode: The NewEpisode object containing the memory details.
        response: The HTTP response object to update headers.
        session: The session data from headers to merge with the request.

    Raises:
        HTTPException: 404 if no matching episodic memory instance is found.
        HTTPException: 400 if the producer or produced_for IDs are invalid
                       for the given context.
    """
    episode.merge_and_validate_session(session)
    episode.update_response_session_header(response)
    await _add_profile_memory(episode)


async def _add_profile_memory(episode: NewEpisode):
    """Adds a memory episode to profile memory.
    Internal function.  Shared by both REST API and MCP API

    See the docstring for add_profile_memory() for details.
    """
    session = episode.get_session()
    group_id = session.group_id

    await cast(ProfileMemory, profile_memory).add_persona_message(
        str(episode.episode_content),
        episode.metadata if episode.metadata is not None else {},
        {
            "group_id": group_id if group_id is not None else "",
            "session_id": session.session_id,
            "producer": episode.producer,
            "produced_for": episode.produced_for,
        },
        user_id=episode.producer,
    )


@app.post("/v1/memories/search")
async def search_memory(
    q: SearchQuery,
    response: Response,
    session: SessionData = Depends(_get_session_from_header),  # type: ignore
) -> SearchResult:
    """Searches for memories across both episodic and profile memory.

    Retrieves the relevant episodic memory instance and then performs
    concurrent searches in both the episodic memory and the profile memory.
    The results are combined into a single response object.

    Args:
        q: The SearchQuery object containing the query and context.
        response: The HTTP response object to update headers.
        session: The session data from headers to merge with the request.

    Returns:
        A SearchResult object containing results from both memory types.

    Raises:
        HTTPException: 404 if no matching episodic memory instance is found.
    """
    q.merge_and_validate_session(session)
    q.update_response_session_header(response)
    return await _search_memory(q)


async def _search_memory(q: SearchQuery) -> SearchResult:
    """Searches for memories across both episodic and profile memory.
    Internal function.  Shared by both REST API and MCP API
    See the docstring for search_memory() for details."""
    session = q.get_session()
    inst: EpisodicMemory | None = await cast(
        EpisodicMemoryManager, episodic_memory
    ).get_episodic_memory_instance(
        group_id=session.group_id,
        agent_id=session.agent_id,
        user_id=session.user_id,
        session_id=session.session_id,
    )
    if inst is None:
        raise q.new_404_not_found_error("unable to find episodic memory")
    async with AsyncEpisodicMemory(inst) as inst:
        ctx = inst.get_memory_context()
        user_id = (
            session.user_id[0]
            if session.user_id is not None and len(session.user_id) > 0
            else ""
        )
        res = await asyncio.gather(
            inst.query_memory(q.query, q.limit, q.filter),
            cast(ProfileMemory, profile_memory).semantic_search(
                q.query,
                q.limit if q.limit is not None else 5,
                isolations={
                    "group_id": ctx.group_id,
                    "session_id": ctx.session_id,
                },
                user_id=user_id,
            ),
        )
        return SearchResult(
            content={"episodic_memory": res[0], "profile_memory": res[1]}
        )


@app.post("/v1/memories/episodic/search")
async def search_episodic_memory(
    q: SearchQuery,
    response: Response,
    session: SessionData = Depends(_get_session_from_header),  # type: ignore
) -> SearchResult:
    """Searches for memories across both profile memory.

    Args:
        q: The SearchQuery object containing the query and context.
        response: The HTTP response object to update headers.
        session: The session data from headers to merge with the request.

    Returns:
        A SearchResult object containing results from episodic memory.

    Raises:
        HTTPException: 404 if no matching episodic memory instance is found.
    """
    q.merge_and_validate_session(session)
    q.update_response_session_header(response)
    return await _search_episodic_memory(q)


async def _search_episodic_memory(q: SearchQuery) -> SearchResult:
    """Searches for memories across episodic memory.
    Internal function.  Shared by both REST API and MCP API
    See the docstring for search_episodic_memory() for details.
    """
    session = q.get_session()
    group_id = session.group_id if session.group_id is not None else ""
    inst: EpisodicMemory | None = await cast(
        EpisodicMemoryManager, episodic_memory
    ).get_episodic_memory_instance(
        group_id=group_id,
        agent_id=session.agent_id,
        user_id=session.user_id,
        session_id=session.session_id,
    )
    if inst is None:
        raise q.new_404_not_found_error("unable to find episodic memory")
    async with AsyncEpisodicMemory(inst) as inst:
        res = await inst.query_memory(q.query, q.limit, q.filter)
        return SearchResult(content={"episodic_memory": res})


@app.post("/v1/memories/profile/search")
async def search_profile_memory(
    q: SearchQuery,
    response: Response,
    session: SessionData = Depends(_get_session_from_header),  # type: ignore
) -> SearchResult:
    """Searches for memories across profile memory.

    Args:
        q: The SearchQuery object containing the query and context.
        response: The HTTP response object to update headers.
        session: The session data from headers to merge with the request.

    Returns:
        A SearchResult object containing results from profile memory.

    Raises:
        HTTPException: 404 if no matching episodic memory instance is found.
    """
    q.merge_and_validate_session(session)
    q.update_response_session_header(response)
    return await _search_profile_memory(q)


async def _search_profile_memory(q: SearchQuery) -> SearchResult:
    """Searches for memories across profile memory.
    Internal function.  Shared by both REST API and MCP API
    See the docstring for search_profile_memory() for details.
    """
    session = q.get_session()
    user_id = session.user_id[0] if session.user_id is not None else ""
    group_id = session.group_id if session.group_id is not None else ""

    res = await cast(ProfileMemory, profile_memory).semantic_search(
        q.query,
        q.limit if q.limit is not None else 5,
        isolations={
            "group_id": group_id,
            "session_id": session.session_id,
        },
        user_id=user_id,
    )
    return SearchResult(content={"profile_memory": res})


@app.delete("/v1/memories")
async def delete_session_data(
    delete_req: DeleteDataRequest,
    response: Response,
    session: SessionData = Depends(_get_session_from_header),  # type: ignore
):
    """
    Delete data for a particular session
    Args:
        delete_req: The DeleteDataRequest object containing the session info.
        response: The HTTP response object to update headers.
        session: The session data from headers to merge with the request.
    """
    delete_req.merge_and_validate_session(session)
    delete_req.update_response_session_header(response)
    await _delete_session_data(delete_req)


async def _delete_session_data(delete_req: DeleteDataRequest):
    """Deletes all data for a specific session.
    Internal function.  Shared by both REST API and MCP API
    See the docstring for delete_session_data() for details.
    """
    session = delete_req.get_session()
    inst: EpisodicMemory | None = await cast(
        EpisodicMemoryManager, episodic_memory
    ).get_episodic_memory_instance(
        group_id=session.group_id,
        agent_id=session.agent_id,
        user_id=session.user_id,
        session_id=session.session_id,
    )
    if inst is None:
        raise delete_req.new_404_not_found_error("unable to find episodic memory")
    async with AsyncEpisodicMemory(inst) as inst:
        await inst.delete_data()


@app.get("/metrics")
async def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.get("/v1/sessions")
async def get_all_sessions() -> AllSessionsResponse:
    """
    Get all sessions
    """
    sessions = cast(EpisodicMemoryManager, episodic_memory).get_all_sessions()
    return AllSessionsResponse(
        sessions=[
            MemorySession(
                group_id=s.group_id,
                session_id=s.session_id,
                user_ids=s.user_ids,
                agent_ids=s.agent_ids,
            )
            for s in sessions
        ]
    )


@app.get("/v1/users/{user_id}/sessions")
async def get_sessions_for_user(user_id: str) -> AllSessionsResponse:
    """
    Get all sessions for a particular user
    """
    sessions = cast(EpisodicMemoryManager, episodic_memory).get_user_sessions(user_id)
    return AllSessionsResponse(
        sessions=[
            MemorySession(
                group_id=s.group_id,
                session_id=s.session_id,
                user_ids=s.user_ids,
                agent_ids=s.agent_ids,
            )
            for s in sessions
        ]
    )


@app.get("/v1/groups/{group_id}/sessions")
async def get_sessions_for_group(group_id: str) -> AllSessionsResponse:
    """
    Get all sessions for a particular group
    """
    sessions = cast(EpisodicMemoryManager, episodic_memory).get_group_sessions(group_id)
    return AllSessionsResponse(
        sessions=[
            MemorySession(
                group_id=s.group_id,
                session_id=s.session_id,
                user_ids=s.user_ids,
                agent_ids=s.agent_ids,
            )
            for s in sessions
        ]
    )


@app.get("/v1/agents/{agent_id}/sessions")
async def get_sessions_for_agent(agent_id: str) -> AllSessionsResponse:
    """
    Get all sessions for a particular agent
    """
    sessions = cast(EpisodicMemoryManager, episodic_memory).get_agent_sessions(agent_id)
    return AllSessionsResponse(
        sessions=[
            MemorySession(
                group_id=s.group_id,
                session_id=s.session_id,
                user_ids=s.user_ids,
                agent_ids=s.agent_ids,
            )
            for s in sessions
        ]
    )


# === Health Check Endpoint ===
@app.get("/health")
async def health_check():
    """Health check endpoint for container orchestration."""
    try:
        # Check if memory managers are initialized
        if profile_memory is None or episodic_memory is None:
            raise HTTPException(
                status_code=503, detail="Memory managers not initialized"
            )

        # Basic health check - could be extended to check database connectivity
        return {
            "status": "healthy",
            "service": "memmachine",
            "version": "1.0.0",
            "memory_managers": {
                "profile_memory": profile_memory is not None,
                "episodic_memory": episodic_memory is not None,
            },
        }
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Service unhealthy: {str(e)}")


async def start():
    """Runs the FastAPI application using uvicorn server."""
    port_num = os.getenv("PORT", "8080")
    host_name = os.getenv("HOST", "0.0.0.0")

    await uvicorn.Server(
        uvicorn.Config(app, host=host_name, port=int(port_num))
    ).serve()


def main():
    """Main entry point for the application."""
    log_level = os.getenv("LOG_LEVEL", "INFO").upper()
    log_format = os.getenv("LOG_FORMAT", "%(levelname)-7s %(message)s")
    logging.basicConfig(
        level=log_level,
        format=log_format,
    )
    # Load environment variables from .env file
    load_dotenv()

    # Parse command line arguments
    parser = argparse.ArgumentParser(description="MemMachine server")
    parser.add_argument(
        "--stdio",
        action="store_true",
        help="Run in MCP stdio mode",
    )
    args = parser.parse_args()

    if args.stdio:
        # MCP stdio mode
        config_file = os.getenv("MEMORY_CONFIG", "configuration.yml")

        async def run_mcp_server():
            """Initialize resources and run MCP server in the same event loop."""
            global episodic_memory, profile_memory
            try:
                episodic_memory, profile_memory = await initialize_resource(config_file)
                await profile_memory.startup()
                await mcp.run_stdio_async()
            finally:
                # Clean up resources when server stops
                if profile_memory:
                    await profile_memory.cleanup()

        asyncio.run(run_mcp_server())
    else:
        # HTTP mode for REST API
        asyncio.run(start())


if __name__ == "__main__":
    main()
