"""API v2 router for MemMachine project and memory management endpoints."""

import logging
import traceback
from typing import Annotated

from fastapi import APIRouter, Depends, FastAPI, Response
from fastapi.exceptions import HTTPException, RequestValidationError
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from memmachine import MemMachine
from memmachine.common.api.doc import RouterDoc
from memmachine.common.api.spec import (
    AddMemoriesResponse,
    AddMemoriesSpec,
    CreateProjectSpec,
    DeleteEpisodicMemorySpec,
    DeleteProjectSpec,
    DeleteSemanticMemorySpec,
    EpisodeCountResponse,
    GetProjectSpec,
    InvalidNameError,
    ListMemoriesSpec,
    ProjectConfig,
    ProjectResponse,
    RestErrorModel,
    SearchMemoriesSpec,
    SearchResult,
)
from memmachine.common.configuration.episodic_config import (
    EpisodicMemoryConfPartial,
    LongTermMemoryConfPartial,
)
from memmachine.common.errors import (
    ConfigurationError,
    InvalidArgumentError,
    ResourceNotFoundError,
    SessionAlreadyExistsError,
    SessionNotFoundError,
)
from memmachine.main.memmachine import ALL_MEMORY_TYPES
from memmachine.server.api_v2.service import (
    _add_messages_to,
    _search_target_memories,
    _SessionData,
    get_memmachine,
)

logger = logging.getLogger(__name__)


class RestError(HTTPException):
    """
    Exception with a structured RestErrorModel as the 'detail'.

    Inherits from HTTPException, which dynamically resolves to:
    - FastAPI's HTTPException in server environments (when FastAPI is available)
    - A lightweight fallback Exception in client-only environments (when FastAPI is not installed)

    This design allows RestError to work in both server and client contexts without
    requiring FastAPI as a dependency for client packages. In server environments,
    RestError behaves as a standard FastAPI HTTPException and can be raised in
    FastAPI route handlers. In client environments, it provides the same interface
    but without the FastAPI dependency overhead.
    """

    def __init__(
        self,
        code: int,
        message: str,
        ex: Exception | None = None,
    ) -> None:
        """Initialize RestError with structured error details."""
        self.payload: RestErrorModel | None = None
        if ex is not None:
            if isinstance(ex, RequestValidationError):
                trace = ""
                message = self.format_validation_error_message(ex)
            elif self.is_known_error(ex):
                trace = ""
            else:
                trace = "".join(
                    traceback.format_exception(
                        type(ex),
                        ex,
                        ex.__traceback__,
                    )
                ).strip()

            self.payload = RestErrorModel(
                code=code,
                message=message,
                exception=type(ex).__name__,
                internal_error=str(ex),
                trace=trace,
            )

        # Call HTTPException with structured detail
        if self.payload is not None:
            logger.warning(
                "exception handling request, code %d, message: %s, payload: %s",
                code,
                message,
                self.payload,
            )
            super().__init__(status_code=code, detail=self.payload.model_dump())
        else:
            logger.info("error handling request, code %d, message: %s", code, message)
            super().__init__(status_code=code, detail=message)

    @staticmethod
    def is_known_error(ex: Exception) -> bool:
        known_errors = [
            SessionAlreadyExistsError,
            SessionNotFoundError,
            InvalidNameError,
            InvalidArgumentError,
        ]
        return any(isinstance(ex, err) for err in known_errors)

    @staticmethod
    def format_validation_error_message(exc: RequestValidationError) -> str:
        parts: list[str] = []

        for err in exc.errors():
            loc = ".".join(str(p) for p in err.get("loc", []) if p != "body")
            msg = err.get("msg", "Invalid value")

            if loc:
                parts.append(f"{loc}: {msg}")
            else:
                parts.append(msg)

        if not parts:
            return "Invalid request payload"

        if len(parts) == 1:
            return f"Invalid request payload: {parts[0]}"

        return "Invalid request payload:\n- " + "\n- ".join(parts)


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
        user_conf = EpisodicMemoryConfPartial(
            long_term_memory=LongTermMemoryConfPartial(
                embedder=spec.config.embedder if spec.config.embedder else None,
                reranker=spec.config.reranker if spec.config.reranker else None,
            )
        )
        session = await memmachine.create_session(
            session_key=session_data.session_key,
            description=spec.description,
            user_conf=user_conf,
        )
    except InvalidArgumentError as e:
        raise RestError(code=422, message="invalid argument: " + str(e)) from e
    except ConfigurationError as e:
        raise RestError(code=500, message="configuration error: " + str(e), ex=e) from e
    except SessionAlreadyExistsError as e:
        raise RestError(code=409, message="Project already exists", ex=e) from e
    except ValueError as e:
        raise RestError(
            code=500, message="server internal error: " + str(e), ex=e
        ) from e
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
    except SessionNotFoundError as e:
        raise RestError(code=404, message="Project does not exist", ex=e) from e
    except Exception as e:
        raise RestError(code=500, message="Unable to delete project", ex=e) from e


@router.post("/memories", description=RouterDoc.ADD_MEMORIES)
async def add_memories(
    spec: AddMemoriesSpec,
    memmachine: Annotated[MemMachine, Depends(get_memmachine)],
) -> AddMemoriesResponse:
    """Add memories to a project."""
    # Use types from spec if provided, otherwise use all memory types
    target_memories = spec.types if spec.types else ALL_MEMORY_TYPES
    results = await _add_messages_to(
        target_memories=target_memories, spec=spec, memmachine=memmachine
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
    target_memories = [spec.type] if spec.type is not None else ALL_MEMORY_TYPES
    results = await memmachine.list_search(
        session_data=_SessionData(
            org_id=spec.org_id,
            project_id=spec.project_id,
        ),
        target_memories=target_memories,
        search_filter=spec.filter,
        page_size=spec.page_size,
        page_num=spec.page_num,
    )

    content = {}
    if results.episodic_memory is not None:
        content["episodic_memory"] = results.episodic_memory
    if results.semantic_memory is not None:
        content["semantic_memory"] = results.semantic_memory

    return SearchResult(
        status=0,
        content=content,
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
