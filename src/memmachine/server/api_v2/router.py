"""API v2 router for MemMachine project and memory management endpoints."""

from collections.abc import AsyncIterator
from typing import Annotated
from uuid import uuid4

from fastapi import APIRouter, Depends, FastAPI, HTTPException, Request
from pydantic import ValidationError

from memmachine.common.configuration import Configuration
from memmachine.common.configuration.episodic_config import (
    EpisodicMemoryConf,
    LongTermMemoryConf,
    ShortTermMemoryConf,
)
from memmachine.common.resource_manager.semantic_manager import SemanticResourceManager
from memmachine.common.session_manager.session_data_manager import SessionDataManager
from memmachine.episode_store.episode_model import Episode
from memmachine.episodic_memory.episodic_memory import EpisodicMemory
from memmachine.episodic_memory.episodic_memory_manager import EpisodicMemoryManager
from memmachine.semantic_memory.semantic_session_resource import (
    IsolationType,
)
from memmachine.server.api_v2.filter_parser import parse_filter
from memmachine.server.api_v2.spec import (
    AddMemoriesSpec,
    CreateProjectSpec,
    DeleteEpisodicMemorySpec,
    DeleteMemoriesSpec,
    DeleteProjectSpec,
    DeleteSemanticMemorySpec,
    GetProjectSpec,
    ListMemoriesSpec,
    SearchMemoriesSpec,
    SearchResult,
    SessionInfo,
)

router = APIRouter()


async def get_session_info(request: Request) -> SessionInfo:
    """Get session info instance."""
    try:
        body = await request.json()
        return SessionInfo.model_validate(body)
    except ValidationError as e:
        raise HTTPException(status_code=422, detail=e.errors()) from e
    except Exception as e:
        raise HTTPException(status_code=400, detail="Invalid JSON body") from e


# Placeholder dependency injection function
async def get_session_manager(request: Request) -> SessionDataManager:
    """Get session data manager instance."""
    return request.app.state.resource_manager.session_data_manager


async def get_episodic_memory_manager(request: Request) -> EpisodicMemoryManager:
    """Get episodic memory manager instance."""
    return request.app.state.resource_manager.episodic_memory_manager


async def get_semantic_memory_manager(request: Request) -> SemanticResourceManager:
    """Get semantic memory manager instance."""
    return await request.app.state.resource_manager.get_semantic_manager()


async def get_global_config(request: Request) -> Configuration:
    """Get global configuration instance."""
    return request.app.state.resource_manager.config


async def get_episodic_memory(
    conf: Annotated[Configuration, Depends(get_global_config)],
    session_manager: Annotated[SessionDataManager, Depends(get_session_manager)],
    episodic_memory_manager: Annotated[
        EpisodicMemoryManager, Depends(get_episodic_memory_manager)
    ],
    session_info: Annotated[SessionInfo, Depends(get_session_info)],
) -> AsyncIterator[EpisodicMemory]:
    """Get episodic memory instance."""
    session_key = f"{session_info.org_id}/{session_info.project_id}"
    await _create_session_if_not_exists(
        conf=conf,
        session_manager=session_manager,
        session_key=session_key,
    )
    async with episodic_memory_manager.open_episodic_memory(
        session_key=session_key
    ) as episodic_memory:
        yield episodic_memory


async def _session_exists(
    session_manager: SessionDataManager, session_key: str
) -> bool:
    """Check if a session exists."""
    try:
        await session_manager.get_session_info(session_key=session_key)
    except Exception:
        return False
    return True


async def _create_new_session(
    conf: Configuration,
    session_manager: SessionDataManager,
    session_key: str,
    description: str,
    embedder: str,
    reranker: str,
) -> None:
    """Create a new session."""
    # Get default prompts from config, with fallbacks
    short_term = conf.episodic_memory.short_term_memory
    summary_prompt_system = (
        short_term.summary_prompt_system
        if short_term and short_term.summary_prompt_system
        else "You are a helpful assistant."
    )
    summary_prompt_user = (
        short_term.summary_prompt_user
        if short_term and short_term.summary_prompt_user
        else "Based on the following episodes: {episodes}, and the previous summary: {summary}, please update the summary. Keep it under {max_length} characters."
    )

    # Get default embedder and reranker from config
    long_term = conf.episodic_memory.long_term_memory
    if embedder == "default" and long_term and long_term.embedder:
        embedder = long_term.embedder
    if reranker == "default" and long_term and long_term.reranker:
        reranker = long_term.reranker
    await session_manager.create_new_session(
        session_key=session_key,
        configuration={},
        param=EpisodicMemoryConf(
            session_key=session_key,
            long_term_memory=LongTermMemoryConf(
                session_id=session_key,
                vector_graph_store=(
                    long_term.vector_graph_store
                    if long_term and long_term.vector_graph_store
                    else "default_store"
                ),
                embedder=embedder,
                reranker=reranker,
            ),
            short_term_memory=ShortTermMemoryConf(
                session_key=session_key,
                llm_model=(
                    short_term.llm_model
                    if short_term and short_term.llm_model
                    else "gpt-4.1"
                ),
                summary_prompt_system=summary_prompt_system,
                summary_prompt_user=summary_prompt_user,
            ),
            long_term_memory_enabled=True,
            short_term_memory_enabled=True,
            enabled=True,
        ),
        description=description,
        metadata={},
    )


async def _create_session_if_not_exists(
    conf: Configuration,
    session_manager: SessionDataManager,
    session_key: str,
) -> None:
    """Create a session if it does not exist."""
    if not await _session_exists(session_manager, session_key):
        await _create_new_session(
            conf=conf,
            session_manager=session_manager,
            session_key=session_key,
            description="",
            embedder="default",
            reranker="default",
        )


@router.post("/projects")
async def create_project(
    spec: CreateProjectSpec,
    conf: Annotated[Configuration, Depends(get_global_config)],
    session_manager: Annotated[SessionDataManager, Depends(get_session_manager)],
) -> None:
    """Create a new project."""
    session_key = f"{spec.org_id}/{spec.project_id}"
    await _create_new_session(
        conf=conf,
        session_manager=session_manager,
        session_key=session_key,
        description=spec.description,
        embedder=spec.config.embedder,
        reranker=spec.config.reranker,
    )


@router.post("/projects/get")
async def get_project(
    spec: GetProjectSpec,
    session_manager: Annotated[SessionDataManager, Depends(get_session_manager)],
) -> dict:
    """Get a project."""
    session_key = f"{spec.org_id}/{spec.project_id}"
    try:
        config, description, metadata, params = await session_manager.get_session_info(
            session_key=session_key
        )
    except Exception:
        return {}
    else:
        return {
            "org_id": spec.org_id,
            "project_id": spec.project_id,
            "configuration": config,
            "description": description,
            "metadata": metadata,
            "params": params,
        }


@router.post("/projects/list")
async def list_projects(
    session_manager: Annotated[SessionDataManager, Depends(get_session_manager)],
) -> list[dict]:
    """List all projects."""
    sessions = await session_manager.get_sessions()
    return [
        {
            "org_id": org_id,
            "project_id": project_id,
        }
        for org_id, project_id in (
            session.split("/", 1) for session in sessions if "/" in session
        )
    ]


@router.post("/projects/delete")
async def delete_project(
    spec: DeleteProjectSpec,
    session_manager: Annotated[SessionDataManager, Depends(get_session_manager)],
) -> None:
    """Delete a project."""
    session_key = f"{spec.org_id}/{spec.project_id}"
    await session_manager.delete_session(session_key=session_key)


@router.post("/memories")
async def add_memories(
    spec: AddMemoriesSpec,
    episodic_memory: Annotated[EpisodicMemory, Depends(get_episodic_memory)],
    semantic_manager: Annotated[
        SemanticResourceManager, Depends(get_semantic_memory_manager)
    ],
) -> None:
    """Add memories to a project."""
    session_key = f"{spec.org_id}/{spec.project_id}"

    episodes: list[Episode] = [
        Episode(
            uid=str(uuid4()),
            content=message.content,
            session_key=session_key,
            created_at=message.timestamp,
            producer_id=message.producer,
            producer_role=message.role,
            produced_for_id=message.produced_for,
            filterable_metadata=message.metadata,
        )
        for message in spec.messages
    ]
    await episodic_memory.add_memory_episodes(episodes=episodes)

    session_id_manager = semantic_manager.simple_semantic_session_id_manager
    semantic_session_manager = await semantic_manager.get_semantic_session_manager()
    semantic_session = session_id_manager.generate_session_data(
        session_id=session_key,
    )
    await semantic_session_manager.add_message(
        episode_ids=[ep.uid for ep in episodes],
        session_data=semantic_session,
        memory_type=[IsolationType.SESSION],
    )


@router.post("/memories/episodic/add")
async def add_episodic_memories(
    spec: AddMemoriesSpec,
    episodic_memory: Annotated[EpisodicMemory, Depends(get_episodic_memory)],
) -> None:
    """Add episodic memories to a project."""
    session_key = f"{spec.org_id}/{spec.project_id}"

    episodes: list[Episode] = [
        Episode(
            uid=str(uuid4()),
            content=message.content,
            session_key=session_key,
            created_at=message.timestamp,
            producer_id=message.producer,
            producer_role=message.role,
            produced_for_id=message.produced_for,
            filterable_metadata=message.metadata,
        )
        for message in spec.messages
    ]
    await episodic_memory.add_memory_episodes(episodes=episodes)


@router.post("/memories/semantic/add")
async def add_semantic_memories(
    spec: AddMemoriesSpec,
    semantic_manager: Annotated[
        SemanticResourceManager, Depends(get_semantic_memory_manager)
    ],
) -> None:
    """Add semantic memories to a project."""
    session_key = f"{spec.org_id}/{spec.project_id}"

    session_id_manager = semantic_manager.simple_semantic_session_id_manager
    semantic_session_manager = await semantic_manager.get_semantic_session_manager()
    semantic_session = session_id_manager.generate_session_data(
        session_id=session_key,
    )
    episode_ids = [str(uuid4()) for _ in spec.messages]
    await semantic_session_manager.add_message(
        episode_ids=episode_ids,
        session_data=semantic_session,
        memory_type=[IsolationType.SESSION],
    )


@router.post("/memories/search")
async def search_memories(
    spec: SearchMemoriesSpec,
    episodic_memory: Annotated[EpisodicMemory, Depends(get_episodic_memory)],
    semantic_manager: Annotated[
        SemanticResourceManager, Depends(get_semantic_memory_manager)
    ],
) -> SearchResult:
    """Search memories in a project."""
    session_key = f"{spec.org_id}/{spec.project_id}"
    ret = SearchResult(status=0, content={"episodic_memory": [], "semantic_memory": []})
    if "episodic" in spec.types:
        episodic_result = await episodic_memory.query_memory(
            query=spec.query,
            limit=spec.top_k,
            property_filter=parse_filter(spec.filter),
        )
        ret.content["episodic_memory"] = episodic_result
    if "semantic" in spec.types:
        session_id_manager = semantic_manager.simple_semantic_session_id_manager
        semantic_session_manager = await semantic_manager.get_semantic_session_manager()
        semantic_session = session_id_manager.generate_session_data(
            session_id=session_key,
        )
        ret.content["semantic_memory"] = await semantic_session_manager.search(
            message=spec.query,
            session_data=semantic_session,
            memory_type=[IsolationType.SESSION],
            limit=spec.top_k,
        )
    return ret


@router.post("/memories/list")
async def list_memories(
    spec: ListMemoriesSpec,
    episodic_memory: Annotated[EpisodicMemory, Depends(get_episodic_memory)],
    semantic_manager: Annotated[
        SemanticResourceManager, Depends(get_semantic_memory_manager)
    ],
) -> SearchResult:
    """List memories in a project."""
    session_key = f"{spec.org_id}/{spec.project_id}"
    ret = SearchResult(status=0, content={"episodic_memory": [], "semantic_memory": []})
    if spec.type == "episodic":
        episodic_result = await episodic_memory.query_memory(
            query="",
            limit=10000,
            property_filter=parse_filter(spec.filter),
        )
        ret.content["episodic_memory"] = episodic_result
    if spec.type == "semantic":
        session_id_manager = semantic_manager.simple_semantic_session_id_manager
        semantic_session_manager = await semantic_manager.get_semantic_session_manager()
        semantic_session = session_id_manager.generate_session_data(
            session_id=session_key,
        )
        ret.content["semantic_memory"] = await semantic_session_manager.search(
            message="",
            session_data=semantic_session,
            memory_type=[IsolationType.SESSION],
            limit=10000,
        )
    return ret


@router.post("/memories/delete")
async def delete_memories(
    spec: DeleteMemoriesSpec,
    episodic_memory: Annotated[EpisodicMemory, Depends(get_episodic_memory)],
) -> None:
    """Delete memories in a project."""
    short_term, long_term, _ = await episodic_memory.query_memory(
        query="",
        property_filter=parse_filter(spec.filter),
    )
    await episodic_memory.delete_episodes(
        uids=[ep.uid for ep in short_term + long_term if ep.uid is not None]
    )


@router.post("/memories/episodic/delete")
async def delete_episodic_memory(
    spec: DeleteEpisodicMemorySpec,
    episodic_memory: Annotated[EpisodicMemory, Depends(get_episodic_memory)],
) -> None:
    """Delete episodic memories in a project."""
    await episodic_memory.delete_episodes(uids=[spec.episodic_id])


@router.post("/memories/semantic/delete")
async def delete_semantic_memory(
    spec: DeleteSemanticMemorySpec,
    episodic_memory: Annotated[EpisodicMemory, Depends(get_episodic_memory)],
) -> None:
    """Delete semantic memories in a project."""
    raise NotImplementedError("Semantic memory deletion is not implemented yet.")


def load_v2_api_router(app: FastAPI) -> APIRouter:
    """Load the API v2 router."""
    app.include_router(router, prefix="/api/v2")
    return router
