"""API v2 router for MemMachine project and memory management endpoints."""

from dataclasses import dataclass
from typing import Annotated

from fastapi import APIRouter, Depends, FastAPI, HTTPException, Request
from pydantic import ValidationError

from memmachine import MemMachine
from memmachine.episode_store.episode_model import EpisodeEntry
from memmachine.main.memmachine import ALL_MEMORY_TYPES
from memmachine.main.memmachine import MemoryType as MemoryTypeE
from memmachine.main.memmachine_errors import ConfigurationError, InvalidArgumentError
from memmachine.server.api_v2.spec import (
    AddMemoriesResponse,
    AddMemoriesSpec,
    AddMemoryResult,
    CreateProjectSpec,
    DeleteEpisodicMemorySpec,
    DeleteMemoriesSpec,
    DeleteProjectSpec,
    DeleteSemanticMemorySpec,
    GetProjectSpec,
    ListMemoriesSpec,
    MemoryType,
    ProjectConfig,
    ProjectResponse,
    SearchMemoriesSpec,
    SearchResult,
    SessionInfo,
)

router = APIRouter()

_MEMORY_TYPE_MAP: dict[MemoryType, MemoryTypeE] = {
    MemoryType.EPISODIC: MemoryTypeE.Episodic,
    MemoryType.SEMANTIC: MemoryTypeE.Semantic,
}


@dataclass
class _SessionData:
    org_id: str
    project_id: str

    @property
    def session_key(self) -> str:
        return f"{self.org_id}/{self.project_id}"

    @property
    def user_profile_id(self) -> str | None:  # pragma: no cover - simple proxy
        return None

    @property
    def role_profile_id(self) -> str | None:  # pragma: no cover - simple proxy
        return None

    @property
    def session_id(self) -> str | None:  # pragma: no cover - simple proxy
        return self.session_key


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
async def get_memmachine(request: Request) -> MemMachine:
    """Get session data manager instance."""
    return request.app.state.memmachine


@router.post("/projects")
async def create_project(
    spec: CreateProjectSpec,
    memmachine: Annotated[MemMachine, Depends(get_memmachine)],
) -> ProjectResponse:
    """Create a new project."""
    session_data = _SessionData(
        org_id=spec.org_id,
        project_id=spec.project_id,
    )
    try:
        session = await memmachine.create_session(
            session_key=session_data.session_key,
            description=spec.description,
            embedder_name=spec.config.embedder,
            reranker_name=spec.config.reranker,
        )
    except InvalidArgumentError as e:
        raise HTTPException(status_code=400, detail="invalid argument: " + str(e)) from e
    except ConfigurationError as e:
        raise HTTPException(status_code=500, detail="configuration error: " + str(e)) from e
    except ValueError as e:
        if f"Session {session_data.session_key} already exists" == str(e):
            raise HTTPException(status_code=400, detail="Project already exists") from e
        raise
    return ProjectResponse(
        org_id=spec.org_id,
        project_id=spec.project_id,
        description=spec.description,
        config=ProjectConfig(
            embedder=session.episode_memory_conf.long_term_memory.embedder,
            reranker=session.episode_memory_conf.long_term_memory.reranker,
        ),
    )


@router.post("/projects/get")
async def get_project(
    spec: GetProjectSpec,
    memmachine: Annotated[MemMachine, Depends(get_memmachine)],
) -> ProjectResponse:
    """Get a project."""
    session_data = _SessionData(
        org_id=spec.org_id,
        project_id=spec.project_id,
    )
    try:
        session_info = await memmachine.get_session(
            session_key=session_data.session_key
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail="Project does not exist") from e
    if session_info is None:
        raise HTTPException(status_code=400, detail="Project does not exist")
    return ProjectResponse(
        org_id=spec.org_id,
        project_id=spec.project_id,
        description=session_info.description,
        config=ProjectConfig(
            embedder=session_info.episode_memory_conf.long_term_memory.embedder,
            reranker=session_info.episode_memory_conf.long_term_memory.reranker,
        ),
    )


@router.post("/projects/list")
async def list_projects(
    memmachine: Annotated[MemMachine, Depends(get_memmachine)],
) -> list[ProjectResponse]:
    """List all projects."""
    sessions = await memmachine.search_sessions()
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
    memmachine: Annotated[MemMachine, Depends(get_memmachine)],
) -> None:
    """Delete a project."""
    session_data = _SessionData(
        org_id=spec.org_id,
        project_id=spec.project_id,
    )
    try:
        await memmachine.delete_session(session_key=session_data.session_key)
    except ValueError as e:
        if f"Session {session_data.session_key} does not exists" == str(e):
            raise HTTPException(status_code=400, detail="Project does not exist") from e
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail="Project does not exist") from e


async def _add_messages_to(
    target_memories: list[MemoryTypeE],
    spec: AddMemoriesSpec,
    memmachine: MemMachine,
) -> list[AddMemoryResult]:
    episodes: list[EpisodeEntry] = [
        EpisodeEntry(
            content=message.content,
            producer_id=message.producer,
            produced_for_id=message.produced_for,
            producer_role=message.role,
            created_at=message.timestamp,
            metadata=message.metadata,
        )
        for message in spec.messages
    ]

    return await memmachine.add_episodes(
        session_data=_SessionData(
            org_id=spec.org_id,
            project_id=spec.project_id,
        ),
        episode_entries=episodes,
        target_memories=target_memories,
    )


@router.post("/memories")
async def add_memories(
    spec: AddMemoriesSpec,
    memmachine: Annotated[MemMachine, Depends(get_memmachine)],
) -> AddMemoriesResponse:
    """Add memories to a project."""
    results = await _add_messages_to(
        target_memories=ALL_MEMORY_TYPES, spec=spec, memmachine=memmachine
    )
    return AddMemoriesResponse(results=results)


@router.post("/memories/episodic/add")
async def add_episodic_memories(
    spec: AddMemoriesSpec,
    memmachine: Annotated[MemMachine, Depends(get_memmachine)],
) -> None:
    """Add episodic memories to a project."""
    await _add_messages_to(
        target_memories=[MemoryTypeE.Episodic], spec=spec, memmachine=memmachine
    )


@router.post("/memories/semantic/add")
async def add_semantic_memories(
    spec: AddMemoriesSpec,
    memmachine: Annotated[MemMachine, Depends(get_memmachine)],
) -> None:
    """Add semantic memories to a project."""
    await _add_messages_to(
        target_memories=[MemoryTypeE.Semantic], spec=spec, memmachine=memmachine
    )


async def _search_target_memories(
    target_memories: list[MemoryTypeE],
    spec: SearchMemoriesSpec,
    memmachine: MemMachine,
) -> SearchResult:
    results = await memmachine.query_search(
        session_data=_SessionData(
            org_id=spec.org_id,
            project_id=spec.project_id,
        ),
        query=spec.query,
        target_memories=target_memories,
        search_filter=spec.filter,
        limit=spec.top_k,
    )
    return SearchResult(
        status=0,
        content={
            "episodic_memory": results.episodic_memory.model_dump()
            if results.episodic_memory
            else [],
            "semantic_memory": results.semantic_memory
            if results.semantic_memory
            else [],
        },
    )


@router.post("/memories/search")
async def search_memories(
    spec: SearchMemoriesSpec,
    memmachine: Annotated[MemMachine, Depends(get_memmachine)],
) -> SearchResult:
    """Search memories in a project."""
    target_memories = (
        [_MEMORY_TYPE_MAP[m] for m in spec.types] if spec.types else ALL_MEMORY_TYPES
    )
    try:
        return await _search_target_memories(
            target_memories=target_memories, spec=spec, memmachine=memmachine
        )
    except RuntimeError as e:
        if "No session info found for session" in str(e):
            raise HTTPException(status_code=400, detail="Project does not exist") from e
        raise


async def _list_target_memories(
    target_memories: list[MemoryTypeE],
    spec: ListMemoriesSpec,
    memmachine: MemMachine,
) -> SearchResult:
    results = await memmachine.list_search(
        session_data=_SessionData(
            org_id=spec.org_id,
            project_id=spec.project_id,
        ),
        target_memories=target_memories,
        search_filter=spec.filter,
        limit=spec.limit,
    )

    return SearchResult(
        status=0,
        content={
            "episodic_memory": results.episodic_memory
            if results.episodic_memory
            else [],
            "semantic_memory": results.semantic_memory
            if results.semantic_memory
            else [],
        },
    )


@router.post("/memories/list")
async def list_memories(
    spec: ListMemoriesSpec,
    memmachine: Annotated[MemMachine, Depends(get_memmachine)],
) -> SearchResult:
    """List memories in a project."""
    if spec.type == MemoryType.EPISODIC:
        target_memories = [MemoryTypeE.Episodic]
    elif spec.type == MemoryType.SEMANTIC:
        target_memories = [MemoryTypeE.Semantic]
    else:
        raise ValueError(f"Invalid memory type: {spec.type}")

    return await _list_target_memories(
        target_memories=target_memories, spec=spec, memmachine=memmachine
    )


@router.post("/memories/delete")
async def delete_memories(
    spec: DeleteMemoriesSpec,
    memmachine: Annotated[MemMachine, Depends(get_memmachine)],
) -> None:
    """Delete memories in a project."""
    await memmachine.delete_filtered(
        session_data=_SessionData(
            org_id=spec.org_id,
            project_id=spec.project_id,
        ),
        target_memories=ALL_MEMORY_TYPES,
        delete_filter=spec.filter,
    )


@router.post("/memories/episodic/delete")
async def delete_episodic_memory(
    spec: DeleteEpisodicMemorySpec,
    memmachine: Annotated[MemMachine, Depends(get_memmachine)],
) -> None:
    """Delete episodic memories in a project."""
    await memmachine.delete_episodes(
        session_data=_SessionData(
            org_id=spec.org_id,
            project_id=spec.project_id,
        ),
        episode_ids=[spec.episodic_id],
    )


@router.post("/memories/semantic/delete")
async def delete_semantic_memory(
    spec: DeleteSemanticMemorySpec,
    memmachine: Annotated[MemMachine, Depends(get_memmachine)],
) -> None:
    """Delete semantic memories in a project."""
    await memmachine.delete_features(
        feature_ids=[spec.semantic_id],
    )


def load_v2_api_router(app: FastAPI) -> APIRouter:
    """Load the API v2 router."""
    app.include_router(router, prefix="/api/v2")
    return router
