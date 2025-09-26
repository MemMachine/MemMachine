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
import logging
import os
from collections.abc import Coroutine
from contextlib import asynccontextmanager
from dataclasses import dataclass
from importlib import import_module
from typing import Any

import uvicorn
import yaml
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastmcp import Context, FastMCP
from prometheus_client import make_asgi_app
from pydantic import BaseModel

from memmachine.common.embedder.openai_embedder import OpenAIEmbedder
from memmachine.common.language_model.openai_language_model import (
    OpenAILanguageModel,
)
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

    group_id: str | None
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
    memory_selection: dict[str, Any] | None = None


class SearchQuery(BaseModel):
    """Request model for searching memories."""

    session: SessionData
    query: str
    filter: dict[str, Any] | None = None
    limit: int | None = None
    memory_selection: dict[str, Any] | None = None




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
profile_memory: ProfileMemory = None
episodic_memory: EpisodicMemoryManager = None


# === Lifespan Management ===


@dataclass
class DBConfig:
    """Database configuration model."""

    host: str | None
    port: str | None
    user: str | None
    password: str | None
    database: str | None


def get_db_config(yaml_config) -> DBConfig:
    """Helper function to retrieve database configuration."""
    db_host = os.getenv("POSTGRES_HOST")
    db_port = os.getenv("POSTGRES_PORT")
    db_user = os.getenv("POSTGRES_USER")
    db_pass = os.getenv("POSTGRES_PASS")
    db_name = os.getenv("POSTGRES_DB")
    # get DB config from configuration file is available
    profile_config = yaml_config.get("profile_memory", {})
    db_config_name = profile_config.get("database")
    if db_config_name is not None:
        db_config = yaml_config.get("storage", {})
        db_config = db_config.get(db_config_name)
        if db_config is not None:
            db_host = db_config.get("host", db_host)
            db_port = db_config.get("port", db_port)
            db_user = db_config.get("user", db_user)
            db_pass = db_config.get("password", db_pass)
            db_name = db_config.get("database", db_name)
    return DBConfig(db_host, db_port, db_user, db_pass, db_name)


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
    try:
        yaml_config = yaml.safe_load(open(config_file, encoding="utf-8"))
    except Exception as e:
        raise e

    # if the model is defined in the config, use it.
    profile_config = yaml_config.get("profile_memory", {})
    model_config = yaml_config.get("model", {})
    model_name = profile_config.get("model_name")
    api_key = os.getenv("OPENAI_API_KEY")
    model = "gpt-4.1-mini"
    if model_name is not None:
        model_def = model_config.get(model_name)
        if model_def is not None:
            api_key = model_def.get("api_key", api_key)
            model = model_def.get("model_name", model)

    # TODO switch to using builder initialization
    llm_model = OpenAILanguageModel({"api_key": api_key, "model": model})
    embeddings = OpenAIEmbedder({"api_key": api_key})

    global profile_memory
    prompt_file = yaml_config.get("prompt", {}).get(
        "profile", "profile_prompt"
    )

    db_config = get_db_config(yaml_config)
    profile_memory = ProfileMemory(
        model=llm_model,
        embeddings=embeddings,
        db_config={
            "host": db_config.host,
            "port": db_config.port,
            "user": db_config.user,
            "password": db_config.password,
            "database": db_config.database,
        },
        prompt_module=import_module(f".prompt.{prompt_file}", __package__),
    )
    global episodic_memory
    episodic_memory = EpisodicMemoryManager.create_episodic_memory_manager(
        config_file
    )
    await profile_memory.startup()
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
app.mount("/metrics", make_asgi_app())


@mcp.tool()
async def mcp_add_session_memory(episode: NewEpisode) -> dict[str, Any]:
    """MCP tool to add a memory episode for a specific session.

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
    if (
        episode.memory_selection is None
        or episode.memory_selection.get("episodic") != "disable"
    ):
        inst: EpisodicMemory | None = \
            await episodic_memory.get_episodic_memory_instance(
                group_id=episode.session.group_id,
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
                    detail=f"""either {episode.producer} or
                            {episode.produced_for}
                            is not in {episode.session.user_id}
                            or {episode.session.agent_id}""",
                )

    if (
        episode.memory_selection is None
        or episode.memory_selection.get("profile") != "disable"
    ):
        gid = episode.session.group_id \
            if episode.session.group_id is not None else ""
        await profile_memory.add_persona_message(
            str(episode.episode_content),
            episode.metadata if episode.metadata is not None else {},
            {
                "group_id": gid,
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
    tasks: List[Coroutine[Any, Any, Any]] = []
    inst: EpisodicMemory | None = None
    episodic_enabled = False
    profile_enabled = False
    if (
        q.memory_selection is None
        or q.memory_selection.get("episodic") != "disable"
    ):
        episodic_enabled = True
        inst = \
            await episodic_memory.get_episodic_memory_instance(
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
        tasks.append(inst.query_memory(q.query, q.limit, q.filter))
    if (
        q.memory_selection is None
        or q.memory_selection.get("profile") != "disable"
    ):
        profile_enabled = True
        user_id = q.session.user_id[0] if q.session.user_id is not None else ""
        tasks.append(
            profile_memory.semantic_search(
                q.query,
                q.limit if q.limit is not None else 5,
                isolations={
                    "group_id": q.session.group_id,
                    "session_id": q.session.session_id,
                },
                user_id=user_id,
            ),
        )
    res = await asyncio.gather(*tasks)
    if episodic_enabled and profile_enabled:
        return SearchResult(
            content={"episodic_memory": res[0], "profile_memory": res[1]}
        )
    if episodic_enabled:
        return SearchResult(content={"episodic_memory": res[0]})
    if profile_enabled:
        return SearchResult(content={"profile_memory": res[0]})
    return SearchResult(content={})



@app.delete("/v1/memories")
async def delete_session_data(delete_req: DeleteDataRequest):
    """
    Delete data for a particular session
    """
    inst: EpisodicMemory = await episodic_memory.get_episodic_memory_instance(
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


@app.get("/v1/sessions")
async def get_all_sessions() -> AllSessionsResponse:
    """
    Get all sessions
    """
    sessions = episodic_memory.get_all_sessions()
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
    sessions = episodic_memory.get_user_sessions(user_id)
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
    sessions = episodic_memory.get_group_sessions(group_id)
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
    sessions = episodic_memory.get_agent_sessions(agent_id)
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
        raise HTTPException(
            status_code=503, detail=f"Service unhealthy: {str(e)}"
        )


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
