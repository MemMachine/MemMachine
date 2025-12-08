"""API v2 router for MemMachine project and memory management endpoints."""

from typing import Annotated

from fastapi import APIRouter, Depends, FastAPI, Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from memmachine import MemMachine
from memmachine.common.errors import (
    ConfigurationError,
    InvalidArgumentError,
    ResourceNotFoundError,
)
from memmachine.main.memmachine import ALL_MEMORY_TYPES, MemoryType
from memmachine.server.api_v2.doc import RouterDoc
from memmachine.server.api_v2.service import (
    _add_messages_to,
    _search_target_memories,
    _SessionData,
    get_memmachine,
)
from memmachine.server.api_v2.spec import (
    AddMemoriesResponse,
    AddMemoriesSpec,
    CreateProjectSpec,
    DeleteEpisodicMemorySpec,
    DeleteProjectSpec,
    DeleteSemanticMemorySpec,
    EpisodeCountResponse,
    GetProjectSpec,
    ListMemoriesSpec,
    ProjectConfig,
    ProjectResponse,
    RestError,
    SearchMemoriesSpec,
    SearchResult,
)

router = APIRouter()


@router.post("/projects", status_code=201, description=RouterDoc.CREATE_PROJECT)
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
        raise RestError(code=422, message="invalid argument: " + str(e)) from e
    except ConfigurationError as e:
        raise RestError(code=500, message="configuration error: " + str(e), ex=e) from e
    except ValueError as e:
        if f"Session {session_data.session_key} already exists" == str(e):
            raise RestError(code=409, message="Project already exists", ex=e) from e
        raise
    long_term = session.episode_memory_conf.long_term_memory
    return ProjectResponse(
        org_id=spec.org_id,
        project_id=spec.project_id,
        description=spec.description,
        config=ProjectConfig(
            embedder=long_term.embedder if long_term else "",
            reranker=long_term.reranker if long_term else "",
        ),
    )


@router.post("/projects/get", description=RouterDoc.GET_PROJECT)
async def get_project(
    spec: GetProjectSpec,
    memmachine: Annotated[MemMachine, Depends(get_memmachine)],
) -> ProjectResponse:
    """Retrieve a project."""
    session_data = _SessionData(
        org_id=spec.org_id,
        project_id=spec.project_id,
    )
    try:
        session_info = await memmachine.get_session(
            session_key=session_data.session_key
        )
    except Exception as e:
        raise RestError(code=500, message="Internal server error", ex=e) from e
    if session_info is None:
        raise RestError(code=404, message="Project does not exist")
    long_term = session_info.episode_memory_conf.long_term_memory
    return ProjectResponse(
        org_id=spec.org_id,
        project_id=spec.project_id,
        description=session_info.description,
        config=ProjectConfig(
            embedder=long_term.embedder if long_term else "",
            reranker=long_term.reranker if long_term else "",
        ),
    )


@router.post("/projects/episode_count/get", description=RouterDoc.GET_EPISODE_COUNT)
async def get_episode_count(
    spec: GetProjectSpec,
    memmachine: Annotated[MemMachine, Depends(get_memmachine)],
) -> EpisodeCountResponse:
    """Retrieve the episode count for a project."""
    session_data = _SessionData(
        org_id=spec.org_id,
        project_id=spec.project_id,
    )
    try:
        count = await memmachine.episodes_count(session_data=session_data)
    except Exception as e:
        raise RestError(code=500, message="Internal server error", ex=e) from e
    return EpisodeCountResponse(count=count)


@router.post("/projects/list", description=RouterDoc.LIST_PROJECTS)
async def list_projects(
    memmachine: Annotated[MemMachine, Depends(get_memmachine)],
) -> list[dict[str, str]]:
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


@router.post("/projects/delete", status_code=204, description=RouterDoc.DELETE_PROJECT)
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
        await memmachine.delete_session(session_data)
    except ValueError as e:
        if f"Session {session_data.session_key} does not exist" == str(e):
            raise RestError(code=404, message="Project does not exist", ex=e) from e
        raise
    except Exception as e:
        raise RestError(code=500, message="Unable to delete project", ex=e) from e


@router.post("/memories", description=RouterDoc.ADD_MEMORIES)
async def add_memories(
    spec: AddMemoriesSpec,
    memmachine: Annotated[MemMachine, Depends(get_memmachine)],
) -> AddMemoriesResponse:
    """Add memories to a project."""
    results = await _add_messages_to(
        target_memories=ALL_MEMORY_TYPES, spec=spec, memmachine=memmachine
    )
    return AddMemoriesResponse(results=results)


@router.post("/memories/episodic/add", description=RouterDoc.ADD_EPISODIC_MEMORIES)
async def add_episodic_memories(
    spec: AddMemoriesSpec,
    memmachine: Annotated[MemMachine, Depends(get_memmachine)],
) -> AddMemoriesResponse:
    """Add episodic memories to a project."""
    results = await _add_messages_to(
        target_memories=[MemoryType.Episodic], spec=spec, memmachine=memmachine
    )
    return AddMemoriesResponse(results=results)


@router.post("/memories/semantic/add", description=RouterDoc.ADD_SEMANTIC_MEMORIES)
async def add_semantic_memories(
    spec: AddMemoriesSpec,
    memmachine: Annotated[MemMachine, Depends(get_memmachine)],
) -> AddMemoriesResponse:
    """Add semantic memories to a project."""
    results = await _add_messages_to(
        target_memories=[MemoryType.Semantic], spec=spec, memmachine=memmachine
    )
    return AddMemoriesResponse(results=results)


@router.post("/memories/search", description=RouterDoc.SEARCH_MEMORIES)
async def search_memories(
    spec: SearchMemoriesSpec,
    memmachine: Annotated[MemMachine, Depends(get_memmachine)],
) -> SearchResult:
    """Search memories in a project."""
    target_memories = spec.types if spec.types else ALL_MEMORY_TYPES
    try:
        return await _search_target_memories(
            target_memories=target_memories, spec=spec, memmachine=memmachine
        )
    except ValueError as e:
        raise RestError(code=422, message="invalid argument", ex=e) from e
    except RuntimeError as e:
        if "No session info found for session" in str(e):
            raise RestError(code=404, message="Project does not exist", ex=e) from e
        raise


async def _list_target_memories(
    spec: ListMemoriesSpec,
    memmachine: MemMachine,
) -> SearchResult:
    results = await memmachine.list_search(
        session_data=_SessionData(
            org_id=spec.org_id,
            project_id=spec.project_id,
        ),
        target_memories=[spec.type],
        search_filter=spec.filter,
        page_size=spec.page_size,
        page_num=spec.page_num,
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


@router.post("/memories/list", description=RouterDoc.LIST_MEMORIES)
async def list_memories(
    spec: ListMemoriesSpec,
    memmachine: Annotated[MemMachine, Depends(get_memmachine)],
) -> SearchResult:
    """List memories in a project."""
    return await _list_target_memories(spec=spec, memmachine=memmachine)


@router.post(
    "/memories/episodic/delete",
    status_code=204,
    description=RouterDoc.DELETE_EPISODIC_MEMORY,
)
async def delete_episodic_memory(
    spec: DeleteEpisodicMemorySpec,
    memmachine: Annotated[MemMachine, Depends(get_memmachine)],
) -> None:
    """Delete episodic memories in a project."""
    try:
        await memmachine.delete_episodes(
            session_data=_SessionData(
                org_id=spec.org_id,
                project_id=spec.project_id,
            ),
            episode_ids=spec.get_ids(),
        )
    except ValueError as e:
        raise RestError(code=422, message="invalid argument", ex=e) from e
    except ResourceNotFoundError as e:
        raise RestError(code=404, message=str(e), ex=e) from e
    except Exception as e:
        raise RestError(
            code=500, message="Unable to delete episodic memory", ex=e
        ) from e


@router.post(
    "/memories/semantic/delete",
    status_code=204,
    description=RouterDoc.DELETE_SEMANTIC_MEMORY,
)
async def delete_semantic_memory(
    spec: DeleteSemanticMemorySpec,
    memmachine: Annotated[MemMachine, Depends(get_memmachine)],
) -> None:
    """Delete semantic memories in a project."""
    try:
        await memmachine.delete_features(
            feature_ids=spec.get_ids(),
        )
    except ValueError as e:
        raise RestError(code=422, message="invalid argument", ex=e) from e
    except ResourceNotFoundError as e:
        raise RestError(code=404, message=str(e), ex=e) from e
    except Exception as e:
        raise RestError(
            code=500, message="Unable to delete semantic memory", ex=e
        ) from e


@router.get("/metrics", description=RouterDoc.METRICS_PROMETHEUS)
async def metrics() -> Response:
    """Expose Prometheus metrics endpoint."""
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@router.get("/health", description=RouterDoc.HEALTH_CHECK)
async def health_check() -> dict[str, str]:
    """Health check endpoint for container orchestration."""
    try:
        return {
            "status": "healthy",
            "service": "memmachine",
        }
    except Exception as e:
        raise RestError(code=503, message="Service unhealthy", ex=e) from e


def load_v2_api_router(app: FastAPI) -> APIRouter:
    """Load the API v2 router."""
    app.include_router(router, prefix="/api/v2")
    return router
