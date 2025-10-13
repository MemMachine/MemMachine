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

import asyncio
import copy
import logging
import os
from contextlib import asynccontextmanager
from importlib import import_module
from typing import Any, cast

import uvicorn
import yaml
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from fastmcp import Context, FastMCP
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from pydantic import BaseModel

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

logger = logging.getLogger(__name__)


# Request session data
class SessionData(BaseModel):
    """Request model for session information."""

    group_id: str
    agent_id: list[str] | None
    user_id: list[str] | None
    session_id: str


# === Request Models ===
class NewEpisode(BaseModel):
    """Request model for adding a new memory episode."""

    session: SessionData
    producer: str
    produced_for: str
    episode_content: str | list[float]
    episode_type: str
    metadata: dict[str, Any] | None


class SearchQuery(BaseModel):
    """Request model for searching memories."""

    session: SessionData
    query: str
    filter: dict[str, Any] | None = None
    limit: int | None = None


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


class DeleteDataRequest(BaseModel):
    """Request model for deleting all data for a session."""

    session: SessionData


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

    api_key = os.getenv("PROFILE_API_KEY")
    if api_key is None:
        api_key = "EMPTY"

    profile_model = copy.deepcopy(model_def)
    profile_model["api_key"] = api_key
    metrics_manager = MetricsFactoryBuilder.build("prometheus", {}, {})
    profile_model["metrics_factory_id"] = "prometheus"
    metrics_injection = {}
    metrics_injection["prometheus"] = metrics_manager
    llm_model = LanguageModelBuilder.build(
        profile_model.get("model_vendor"), profile_model, metrics_injection
    )

    # create embedder
    embedders = yaml_config.get("embedder", {})
    embedder_name = profile_config.get("embedding_model")
    if embedder_name is None:
        raise ValueError(
            "Embedding model not configured in config file for profile memory"
        )

    embedder_def = embedders.get(embedder_name)
    if embedder_def is None:
        raise ValueError(f"Can not find definition of embedder {embedder_name}")

    embedder_config = copy.deepcopy(embedder_def)
    embedder_config["metrics_factory_id"] = "prometheus"
    embedder_config["api_key"] = api_key

    embeddings = EmbedderBuilder.build(
        embedder_def.get("model_vendor", "openai"), embedder_config, metrics_injection
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
    db_pass = os.getenv("PROFILE_DB_PASSWORD")

    prompt_file = profile_config.get("prompt", "profile_prompt")

    profile_memory = ProfileMemory(
        model=llm_model,
        embeddings=embeddings,
        db_config={
            "host": db_config.get("host", "localhost"),
            "port": db_config.get("port", 0),
            "user": db_config.get("user", ""),
            "password": db_pass,
            "database": db_config.get("database", ""),
        },
        prompt_module=import_module(f".prompt.{prompt_file}", __package__),
    )
    episodic_memory = EpisodicMemoryManager.create_episodic_memory_manager(config_file)
    return episodic_memory, profile_memory


@asynccontextmanager
async def http_app_lifespan(application: FastAPI):
    """Handles application startup and shutdown events.

    Initializes the ProfileMemory and EpisodicMemoryManager instances,
    and establishes necessary connections (e.g., to the database).
    These resources are cleaned up on shutdown.

    Args:
        app: The FastAPI application instance.
    """
    config_file = os.getenv("MEMORY_CONFIG", "cfg.yml")

    global episodic_memory
    global profile_memory
    episodic_memory, profile_memory = await initialize_resource(config_file)
    profile_memory.startup()
    yield
    await profile_memory.cleanup()
    await episodic_memory.shut_down()


mcp = FastMCP("MemMachine")
mcp_app = mcp.http_app("/")


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
    async with http_app_lifespan(application):
        async with mcp_app.lifespan(application):
            yield


app = FastAPI(lifespan=mcp_http_lifespan)
app.mount("/mcp", mcp_app)


@mcp.tool()
async def mcp_add_session_memory(episode: NewEpisode) -> dict[str, Any]:
    """MCP tool to add a memory episode for a specific session. It adds the
    episode to both episodic and profile memory.

    This tool does not require a pre-existing open session in the context.
    It adds a memory episode directly using the session data provided in the
    `NewEpisode` object.

    Args:
        episode: The complete new episode data, including session info.
        ctx: The MCP context (unused).

    Returns:
        Status 0 if the memory was added successfully, Status -1 otherwise
        with error message.
    """
    try:
        await add_memory(episode)
    except HTTPException as e:
        sess = episode.session
        session_name = f"""{sess.group_id}-{sess.agent_id}-
                           {sess.user_id}-{sess.session_id}"""
        logger.error("Failed to add memory episode for %s", session_name)
        logger.error(e)
        return {"status": -1, "error_msg": str(e)}
    return {"status": 0, "error_msg": ""}


@mcp.tool()
async def mcp_add_episodic_memory(episode: NewEpisode) -> dict[str, Any]:
    """MCP tool to add a memory episode for a specific session. It only
    adds the episode to the episodic memory.

    This tool does not require a pre-existing open session in the context.
    It adds a memory episode directly using the session data provided in the
    `NewEpisode` object.

    Args:
        episode: The complete new episode data, including session info.
        ctx: The MCP context (unused).

    Returns:
        Status 0 if the memory was added successfully, Status -1 otherwise
        with error message.
    """
    try:
        await add_episodic_memory(episode)
    except HTTPException as e:
        sess = episode.session
        session_name = f"""{sess.group_id}-{sess.agent_id}-
                           {sess.user_id}-{sess.session_id}"""
        logger.error("Failed to add memory episode for %s", session_name)
        logger.error(e)
        return {"status": -1, "error_msg": str(e)}
    return {"status": 0, "error_msg": ""}


@mcp.tool()
async def mcp_add_profile_memory(episode: NewEpisode) -> dict[str, Any]:
    """MCP tool to add a memory episode for a specific session. It only
    adds the episode to profile memory.

    This tool does not require a pre-existing open session in the context.
    It adds a memory episode directly using the session data provided in the
    `NewEpisode` object.

    Args:
        episode: The complete new episode data, including session info.
        ctx: The MCP context (unused).

    Returns:
        Status 0 if the memory was added successfully, Status -1 otherwise
        with error message.
    """
    try:
        await add_profile_memory(episode)
    except HTTPException as e:
        sess = episode.session
        session_name = f"""{sess.group_id}-{sess.agent_id}-
                           {sess.user_id}-{sess.session_id}"""
        logger.error("Failed to add memory episode for %s", session_name)
        logger.error(e)
        return {"status": -1, "error_msg": str(e)}
    return {"status": 0, "error_msg": ""}


@mcp.tool()
async def mcp_search_episodic_memory(q: SearchQuery) -> SearchResult:
    """MCP tool to search for episodic memories in a specific session.
    This tool does not require a pre-existing open session in the context.
    It searches only the episodic memory for the provided query.

    Args:
        q: The search query.

    Return:
        A SearchResult object if successful, None otherwise.
    """
    return await search_episodic_memory(q)


@mcp.tool()
async def mcp_search_profile_memory(q: SearchQuery) -> SearchResult:
    """MCP tool to search for profile memories in a specific session.
    This tool does not require a pre-existing open session in the context.
    It searches only the profile memory for the provided query.

    Args:
        q: The search query.

    Return:
        A SearchResult object if successful, None otherwise.
    """
    return await search_profile_memory(q)


@mcp.tool()
async def mcp_search_session_memory(q: SearchQuery) -> SearchResult:
    """MCP tool to search for memories in a specific session.

    This tool does not require a pre-existing open session in the context.
    It searches both episodic and profile memories for the provided query.

    Args:
        q: The search query.

    Return:
        A SearchResult object if successful, None otherwise.
    """
    return await search_memory(q)


@mcp.tool()
async def mcp_delete_session_data(sess: SessionData) -> dict[str, Any]:
    """MCP tool to delete all data for a specific session.

    This tool does not require a pre-existing open session in the context.
    It deletes all data associated with the provided session data.

    Args:
        sess: The session data for which to delete all memories.
        ctx: The MCP context (unused).

    Returns:
        Status 0 if deletion was successful, Status -1 otherwise
        with error message.
    """
    try:
        await delete_session_data(DeleteDataRequest(session=sess))
    except HTTPException as e:
        session_name = f"""{sess.group_id}-{sess.agent_id}-
                           {sess.user_id}-{sess.session_id}"""
        logger.error("Failed to add memory episode for %s", session_name)
        logger.error(e)
        return {"status": -1, "error_msg": str(e)}
    return {"status": 0, "error_msg": ""}


@mcp.tool()
async def mcp_delete_data(ctx: Context) -> dict[str, Any]:
    """MCP tool to delete all data for the current session.

    This tool requires an open memory session. It deletes all data associated
    with the session stored in the MCP context.

    Args:
        ctx: The MCP context.

    Returns:
        Status 0 if deletion was successful, Sttus -1 otherwise
        with error message.
    """
    try:
        sess = ctx.get_state("session_data")
        if sess is None:
            return {"status": -1, "error_msg": "No session open"}
        delete_data_req = DeleteDataRequest(session=sess)
        await delete_session_data(delete_data_req)
    except HTTPException as e:
        session_name = f"""{sess.group_id}-{sess.agent_id}-
                           {sess.user_id}-{sess.session_id}"""
        logger.error("Failed to add memory episode for %s", session_name)
        logger.error(e)
        return {"status": -1, "error_msg": str(e)}
    return {"status": 0, "error_msg": ""}


@mcp.resource("sessions://sessions")
async def mcp_get_sessions() -> AllSessionsResponse:
    """MCP resource to retrieve all memory sessions.

    Returns:
        An AllSessionsResponse containing a list of all sessions.
    """
    return await get_all_sessions()


@mcp.resource("users://{user_id}/sessions")
async def mcp_get_user_sessions(user_id: str) -> AllSessionsResponse:
    """MCP resource to retrieve all sessions for a specific user.

    Returns:
        An AllSessionsResponse containing a list of sessions for the user.
    """
    return await get_sessions_for_user(user_id)


@mcp.resource("groups://{group_id}/sessions")
async def mcp_get_group_sessions(group_id: str) -> AllSessionsResponse:
    """MCP resource to retrieve all sessions for a specific group.

    Returns:
        An AllSessionsResponse containing a list of sessions for the group.
    """
    return await get_sessions_for_group(group_id)


@mcp.resource("agents://{agent_id}/sessions")
async def mcp_get_agent_sessions(agent_id: str) -> AllSessionsResponse:
    """MCP resource to retrieve all sessions for a specific agent.

    Returns:
        An AllSessionsResponse containing a list of sessions for the agent.
    """
    return await get_sessions_for_agent(agent_id)


# === Route Handlers ===
@app.post("/v1/memories")
async def add_memory(episode: NewEpisode):
    """Adds a memory episode to both episodic and profile memory.

    This endpoint first retrieves the appropriate episodic memory instance
    based on the session context (group, agent, user, session IDs). It then
    adds the episode to the episodic memory. If successful, it also passes
    the message to the profile memory for ingestion.

    Args:
        episode: The NewEpisode object containing the memory details.

    Raises:
        HTTPException: 404 if no matching episodic memory instance is found.
        HTTPException: 400 if the producer or produced_for IDs are invalid
                       for the given context.
    """
    group_id = episode.session.group_id
    inst: EpisodicMemory | None = await cast(
        EpisodicMemoryManager, episodic_memory
    ).get_episodic_memory_instance(
        group_id=group_id if group_id is not None else "",
        agent_id=episode.session.agent_id,
        user_id=episode.session.user_id,
        session_id=episode.session.session_id,
    )
    if inst is None:
        raise HTTPException(
            status_code=404,
            detail=f"""unable to find episodic memory for
                    {episode.session.user_id},
                    {episode.session.session_id},
                    {episode.session.group_id},
                    {episode.session.agent_id}""",
        )
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
                        is not in {episode.session.user_id}
                        or {episode.session.agent_id}""",
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
async def add_episodic_memory(episode: NewEpisode):
    """Adds a memory episode to both episodic memory.

    This endpoint first retrieves the appropriate episodic memory instance
    based on the session context (group, agent, user, session IDs). It then
    adds the episode to the episodic memory. If successful, it also passes
    the message to the profile memory for ingestion.

    Args:
        episode: The NewEpisode object containing the memory details.

    Raises:
        HTTPException: 404 if no matching episodic memory instance is found.
        HTTPException: 400 if the producer or produced_for IDs are invalid
                       for the given context.
    """
    group_id = episode.session.group_id
    inst: EpisodicMemory | None = await cast(
        EpisodicMemoryManager, episodic_memory
    ).get_episodic_memory_instance(
        group_id=group_id if group_id is not None else "",
        agent_id=episode.session.agent_id,
        user_id=episode.session.user_id,
        session_id=episode.session.session_id,
    )
    if inst is None:
        raise HTTPException(
            status_code=404,
            detail=f"""unable to find episodic memory for
                    {episode.session.user_id},
                    {episode.session.session_id},
                    {episode.session.group_id},
                    {episode.session.agent_id}""",
        )
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
                        is not in {episode.session.user_id}
                        or {episode.session.agent_id}""",
            )


@app.post("/v1/memories/profile")
async def add_profile_memory(episode: NewEpisode):
    """Adds a memory episode to both profile memory.

    This endpoint first retrieves the appropriate episodic memory instance
    based on the session context (group, agent, user, session IDs). It then
    adds the episode to the episodic memory. If successful, it also passes
    the message to the profile memory for ingestion.

    Args:
        episode: The NewEpisode object containing the memory details.

    Raises:
        HTTPException: 404 if no matching episodic memory instance is found.
        HTTPException: 400 if the producer or produced_for IDs are invalid
                       for the given context.
    """
    group_id = episode.session.group_id

    await cast(ProfileMemory, profile_memory).add_persona_message(
        str(episode.episode_content),
        episode.metadata if episode.metadata is not None else {},
        {
            "group_id": group_id if group_id is not None else "",
            "session_id": episode.session.session_id,
            "producer": episode.producer,
            "produced_for": episode.produced_for,
        },
        user_id=episode.producer,
    )


@app.post("/v1/memories/search")
async def search_memory(q: SearchQuery) -> SearchResult:
    """Searches for memories across both episodic and profile memory.

    Retrieves the relevant episodic memory instance and then performs
    concurrent searches in both the episodic memory and the profile memory.
    The results are combined into a single response object.

    Args:
        q: The SearchQuery object containing the query and context.

    Returns:
        A SearchResult object containing results from both memory types.

    Raises:
        HTTPException: 404 if no matching episodic memory instance is found.
    """
    inst: EpisodicMemory | None = await cast(
        EpisodicMemoryManager, episodic_memory
    ).get_episodic_memory_instance(
        group_id=q.session.group_id,
        agent_id=q.session.agent_id,
        user_id=q.session.user_id,
        session_id=q.session.session_id,
    )
    if inst is None:
        raise HTTPException(
            status_code=404,
            detail=f"""unable to find episodic memory for
                    {q.session.user_id},
                    {q.session.session_id},
                    {q.session.group_id},
                    {q.session.agent_id}""",
        )
    async with AsyncEpisodicMemory(inst) as inst:
        ctx = inst.get_memory_context()
        user_id = (
            q.session.user_id[0]
            if q.session.user_id is not None and len(q.session.user_id) > 0
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
async def search_episodic_memory(q: SearchQuery) -> SearchResult:
    """Searches for memories across both profile memory.

    Args:
        q: The SearchQuery object containing the query and context.

    Returns:
        A SearchResult object containing results from episodic memory.

    Raises:
        HTTPException: 404 if no matching episodic memory instance is found.
    """
    group_id = q.session.group_id if q.session.group_id is not None else ""
    inst: EpisodicMemory | None = await cast(
        EpisodicMemoryManager, episodic_memory
    ).get_episodic_memory_instance(
        group_id=group_id,
        agent_id=q.session.agent_id,
        user_id=q.session.user_id,
        session_id=q.session.session_id,
    )
    if inst is None:
        raise HTTPException(
            status_code=404,
            detail=f"""unable to find episodic memory for
                    {q.session.user_id},
                    {q.session.session_id},
                    {q.session.group_id},
                    {q.session.agent_id}""",
        )
    async with AsyncEpisodicMemory(inst) as inst:
        res = await inst.query_memory(q.query, q.limit, q.filter)
        return SearchResult(content={"episodic_memory": res})


@app.post("/v1/memories/profile/search")
async def search_profile_memory(q: SearchQuery) -> SearchResult:
    """Searches for memories across profile memory.

    Args:
        q: The SearchQuery object containing the query and context.

    Returns:
        A SearchResult object containing results from profile memory.

    Raises:
        HTTPException: 404 if no matching episodic memory instance is found.
    """
    user_id = q.session.user_id[0] if q.session.user_id is not None else ""
    group_id = q.session.group_id if q.session.group_id is not None else ""

    res = await cast(ProfileMemory, profile_memory).semantic_search(
        q.query,
        q.limit if q.limit is not None else 5,
        isolations={
            "group_id": group_id,
            "session_id": q.session.session_id,
        },
        user_id=user_id,
    )
    return SearchResult(content={"profile_memory": res})


@app.delete("/v1/memories")
async def delete_session_data(delete_req: DeleteDataRequest):
    """
    Delete data for a particular session
    """
    inst: EpisodicMemory | None = await cast(
        EpisodicMemoryManager, episodic_memory
    ).get_episodic_memory_instance(
        group_id=delete_req.session.group_id,
        agent_id=delete_req.session.agent_id,
        user_id=delete_req.session.user_id,
        session_id=delete_req.session.session_id,
    )
    if inst is None:
        raise HTTPException(
            status_code=404,
            detail=f"""unable to find episodic memory for
                    {delete_req.session.user_id},
                    {delete_req.session.session_id},
                    {delete_req.session.group_id},
                    {delete_req.session.agent_id}""",
        )
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
    # Load environment variables from .env file
    load_dotenv()
    # Run the asyncio event loop
    asyncio.run(start())


if __name__ == "__main__":
    main()
