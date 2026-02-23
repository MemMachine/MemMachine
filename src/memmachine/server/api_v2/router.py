"""API v2 router for MemMachine project and memory management endpoints."""

import json
import logging
from typing import Annotated

from fastapi import (
    APIRouter,
    Depends,
    FastAPI,
    File,
    Form,
    Request,
    Response,
    UploadFile,
)
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from pydantic import ValidationError

from memmachine import MemMachine
from memmachine.common.api.doc import RouterDoc
from memmachine.common.api.spec import (
    AddFeatureResponse,
    AddFeatureSpec,
    AddMemoriesResponse,
    AddMemoriesSpec,
    AddSemanticCategoryResponse,
    AddSemanticCategorySpec,
    AddSemanticCategoryTemplateSpec,
    AddSemanticTagResponse,
    AddSemanticTagSpec,
    ConfigureEpisodicMemorySpec,
    ConfigureLongTermMemorySpec,
    ConfigureSemanticSetSpec,
    ConfigureShortTermMemorySpec,
    CreateProjectSpec,
    CreateSemanticSetTypeResponse,
    CreateSemanticSetTypeSpec,
    DeleteEpisodicMemorySpec,
    DeleteProjectSpec,
    DeleteSemanticCategorySpec,
    DeleteSemanticMemorySpec,
    DeleteSemanticSetTypeSpec,
    DeleteSemanticTagSpec,
    DisableSemanticCategorySpec,
    EpisodeCountResponse,
    EpisodicMemoryConfigEntry,
    GetEpisodicMemoryConfigSpec,
    GetFeatureSpec,
    GetLongTermMemoryConfigSpec,
    GetProjectSpec,
    GetSemanticCategorySetIdsResponse,
    GetSemanticCategorySetIdsSpec,
    GetSemanticCategorySpec,
    GetSemanticSetIdResponse,
    GetSemanticSetIdSpec,
    GetShortTermMemoryConfigSpec,
    ListMemoriesSpec,
    ListResult,
    ListSemanticCategoryTemplatesResponse,
    ListSemanticCategoryTemplatesSpec,
    ListSemanticSetIdsResponse,
    ListSemanticSetIdsSpec,
    ListSemanticSetTypesResponse,
    ListSemanticSetTypesSpec,
    LongTermMemoryConfigEntry,
    ProjectConfig,
    ProjectResponse,
    SearchMemoriesSpec,
    SearchResult,
    SemanticCategoryEntry,
    SemanticCategoryTemplateEntry,
    SemanticFeature,
    SemanticSetEntry,
    SemanticSetTypeEntry,
    ShortTermMemoryConfigEntry,
    UpdateFeatureSpec,
)
from memmachine.common.api.version import get_version
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
from memmachine.server.api_v2.config_router import config_router
from memmachine.server.api_v2.exceptions import RestError
from memmachine.server.api_v2.service import (
    _add_messages_to,
    _list_target_memories,
    _search_target_memories,
    _SessionData,
    get_memmachine,
)

logger = logging.getLogger(__name__)


router = APIRouter()


async def _parse_add_memories_request(
    request: Request,
    spec: Annotated[str | None, Form()] = None,
    image: Annotated[UploadFile | None, File()] = None,
) -> tuple[AddMemoriesSpec, UploadFile | None]:
    """Parse AddMemories request from either JSON body or multipart form-data."""
    content_type = request.headers.get("content-type", "")

    if spec is not None:
        try:
            raw = json.loads(spec)
        except json.JSONDecodeError as e:
            raise RestError(code=422, message="Invalid request payload: spec is not valid JSON", ex=e) from e
        try:
            return AddMemoriesSpec(**raw), image
        except ValidationError as e:
            raise RestError(code=422, message="Invalid request payload", ex=e) from e

    # Multipart requests must send spec explicitly
    if "multipart/form-data" in content_type:
        raise RestError(
            code=422,
            message="Invalid request payload: missing form field 'spec' for multipart request",
        )

    # Default: JSON body
    try:
        raw = await request.json()
    except Exception as e:
        raise RestError(code=422, message="Invalid request payload", ex=e) from e
    try:
        return AddMemoriesSpec(**raw), None
    except ValidationError as e:
        raise RestError(code=422, message="Invalid request payload", ex=e) from e


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
                embedder=spec.config.embedder or None,
                reranker=spec.config.reranker or None,
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
    parsed: Annotated[
        tuple[AddMemoriesSpec, UploadFile | None],
        Depends(_parse_add_memories_request),
    ],
    memmachine: Annotated[MemMachine, Depends(get_memmachine)],
) -> AddMemoriesResponse:
    """Add memories to a project."""
    spec, image = parsed

    if image is not None:
        # Ambiguity: how to attach one image to multiple messages.
        # For now, require a single message.
        if len(spec.messages) != 1:
            raise RestError(
                code=422,
                message=(
                    "Invalid request payload: image upload is only supported when messages has exactly 1 item"
                ),
            )

        image_bytes = await image.read()
        if not image_bytes:
            raise RestError(code=422, message="Invalid request payload: image file is empty")

        mime_type = image.content_type or "application/octet-stream"
        try:
            summary = await memmachine.image_summarizer.summarize_image(
                image_bytes=image_bytes,
                mime_type=mime_type,
            )
        except Exception as e:
            raise RestError(code=500, message="Unable to summarize image", ex=e) from e

        if summary:
            spec.messages[0].content = (
                f"{spec.messages[0].content}\n\n[Image Summary]\n{summary}"
            )

    # Use types from spec if provided, otherwise use all memory types
    target_memories = spec.types or ALL_MEMORY_TYPES
    results = await _add_messages_to(
        target_memories=target_memories, spec=spec, memmachine=memmachine
    )
    return AddMemoriesResponse(results=results)


@router.post(
    "/memories/search",
    description=RouterDoc.SEARCH_MEMORIES,
    response_model_exclude_none=True,
)
async def search_memories(
    spec: SearchMemoriesSpec,
    memmachine: Annotated[MemMachine, Depends(get_memmachine)],
) -> SearchResult:
    """Search memories in a project."""
    target_memories = spec.types or ALL_MEMORY_TYPES
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


@router.post(
    "/memories/list",
    description=RouterDoc.LIST_MEMORIES,
    response_model_exclude_none=True,
)
async def list_memories(
    spec: ListMemoriesSpec,
    memmachine: Annotated[MemMachine, Depends(get_memmachine)],
) -> ListResult:
    """List memories in a project."""
    target_memories = [spec.type] if spec.type is not None else ALL_MEMORY_TYPES
    return await _list_target_memories(
        target_memories=target_memories, spec=spec, memmachine=memmachine
    )


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
    except SessionNotFoundError as e:
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


@router.post(
    "/memories/semantic/feature",
    status_code=201,
    description=RouterDoc.ADD_FEATURE,
)
async def add_feature(
    spec: AddFeatureSpec,
    memmachine: Annotated[MemMachine, Depends(get_memmachine)],
) -> AddFeatureResponse:
    """Add a semantic feature to a project."""
    try:
        feature_id = await memmachine.add_feature(
            set_id=spec.set_id,
            category_name=spec.category_name,
            tag=spec.tag,
            feature=spec.feature,
            value=spec.value,
            feature_metadata=spec.feature_metadata,
            citations=spec.citations,
        )
    except ValueError as e:
        raise RestError(code=422, message="invalid argument", ex=e) from e
    except Exception as e:
        raise RestError(code=500, message="Unable to add feature", ex=e) from e
    return AddFeatureResponse(feature_id=feature_id)


@router.post(
    "/memories/semantic/feature/get",
    description=RouterDoc.GET_FEATURE,
    response_model_exclude_none=True,
)
async def get_feature(
    spec: GetFeatureSpec,
    memmachine: Annotated[MemMachine, Depends(get_memmachine)],
) -> SemanticFeature | None:
    """Get a semantic feature by ID."""
    try:
        feature = await memmachine.get_feature(
            feature_id=spec.feature_id,
            load_citations=spec.load_citations,
        )
    except ValueError as e:
        raise RestError(code=422, message="invalid argument", ex=e) from e
    except Exception as e:
        raise RestError(code=500, message="Unable to get feature", ex=e) from e
    if feature is None:
        raise RestError(code=404, message="Feature not found")
    return SemanticFeature(
        set_id=feature.set_id,
        category=feature.category,
        tag=feature.tag,
        feature_name=feature.feature_name,
        value=feature.value,
        metadata=SemanticFeature.Metadata(
            id=feature.metadata.id if feature.metadata else None,
            citations=feature.metadata.citations if feature.metadata else None,
            other=feature.metadata.other if feature.metadata else None,
        ),
    )


@router.post(
    "/memories/semantic/feature/update",
    status_code=204,
    description=RouterDoc.UPDATE_FEATURE,
)
async def update_feature(
    spec: UpdateFeatureSpec,
    memmachine: Annotated[MemMachine, Depends(get_memmachine)],
) -> None:
    """Update a semantic feature."""
    try:
        await memmachine.update_feature(
            feature_id=spec.feature_id,
            category_name=spec.category_name,
            feature=spec.feature,
            value=spec.value,
            tag=spec.tag,
            metadata=spec.metadata,
        )
    except ValueError as e:
        raise RestError(code=422, message="invalid argument", ex=e) from e
    except ResourceNotFoundError as e:
        raise RestError(code=404, message=str(e), ex=e) from e
    except Exception as e:
        raise RestError(code=500, message="Unable to update feature", ex=e) from e


@router.post(
    "/memories/semantic/set_type",
    status_code=201,
    description=RouterDoc.CREATE_SEMANTIC_SET_TYPE,
)
async def create_semantic_set_type(
    spec: CreateSemanticSetTypeSpec,
    memmachine: Annotated[MemMachine, Depends(get_memmachine)],
) -> CreateSemanticSetTypeResponse:
    """Create a new semantic set type."""
    try:
        set_type_id = await memmachine.create_semantic_set_type(
            session_data=_SessionData(
                org_id=spec.org_id,
                project_id=spec.project_id,
            ),
            is_org_level=spec.is_org_level,
            metadata_tags=spec.metadata_tags,
            name=spec.name,
            description=spec.description,
        )
    except ValueError as e:
        raise RestError(code=422, message="invalid argument", ex=e) from e
    except Exception as e:
        raise RestError(code=500, message="Unable to create set type", ex=e) from e
    return CreateSemanticSetTypeResponse(set_type_id=set_type_id)


@router.post(
    "/memories/semantic/set_type/list",
    description=RouterDoc.LIST_SEMANTIC_SET_TYPES,
)
async def list_semantic_set_types(
    spec: ListSemanticSetTypesSpec,
    memmachine: Annotated[MemMachine, Depends(get_memmachine)],
) -> ListSemanticSetTypesResponse:
    """List all semantic set types."""
    try:
        set_types = await memmachine.list_semantic_set_type(
            session_data=_SessionData(
                org_id=spec.org_id,
                project_id=spec.project_id,
            ),
        )
        entries = [
            SemanticSetTypeEntry(
                id=st.id,
                is_org_level=st.is_org_level,
                tags=st.tags,
                name=st.name,
                description=st.description,
            )
            for st in set_types
        ]
    except Exception as e:
        raise RestError(code=500, message="Unable to list set types", ex=e) from e
    return ListSemanticSetTypesResponse(set_types=entries)


@router.post(
    "/memories/semantic/set_type/delete",
    status_code=204,
    description=RouterDoc.DELETE_SEMANTIC_SET_TYPE,
)
async def delete_semantic_set_type(
    spec: DeleteSemanticSetTypeSpec,
    memmachine: Annotated[MemMachine, Depends(get_memmachine)],
) -> None:
    """Delete a semantic set type."""
    try:
        await memmachine.delete_semantic_set_type(set_type_id=spec.set_type_id)
    except ResourceNotFoundError as e:
        raise RestError(code=404, message=str(e), ex=e) from e
    except ValueError as e:
        raise RestError(code=422, message="invalid argument", ex=e) from e
    except Exception as e:
        raise RestError(code=500, message="Unable to delete set type", ex=e) from e


@router.post(
    "/memories/semantic/set_id/get",
    description=RouterDoc.GET_SEMANTIC_SET_ID,
)
async def get_semantic_set_id(
    spec: GetSemanticSetIdSpec,
    memmachine: Annotated[MemMachine, Depends(get_memmachine)],
) -> GetSemanticSetIdResponse:
    """Get or create a semantic set ID."""
    try:
        set_id = await memmachine.semantic_get_set_id(
            session_data=_SessionData(
                org_id=spec.org_id,
                project_id=spec.project_id,
            ),
            is_org_level=spec.is_org_level,
            metadata_tags=spec.metadata_tags,
            set_metadata=spec.set_metadata,
        )
    except ValueError as e:
        raise RestError(code=422, message="invalid argument", ex=e) from e
    except Exception as e:
        raise RestError(code=500, message="Unable to get set ID", ex=e) from e
    return GetSemanticSetIdResponse(set_id=set_id)


@router.post(
    "/memories/semantic/set_id/list",
    description=RouterDoc.LIST_SEMANTIC_SET_IDS,
)
async def list_semantic_set_ids(
    spec: ListSemanticSetIdsSpec,
    memmachine: Annotated[MemMachine, Depends(get_memmachine)],
) -> ListSemanticSetIdsResponse:
    """List all semantic sets."""
    try:
        sets = await memmachine.semantic_list_set_ids(
            session_data=_SessionData(
                org_id=spec.org_id,
                project_id=spec.project_id,
            ),
            set_metadata=spec.set_metadata,
        )
        entries = [
            SemanticSetEntry(
                id=s.id,
                is_org_level=s.is_org_level,
                tags=s.tags,
                name=s.name,
                description=s.description,
            )
            for s in sets
        ]
    except Exception as e:
        raise RestError(code=500, message="Unable to list sets", ex=e) from e
    return ListSemanticSetIdsResponse(sets=entries)


@router.post(
    "/memories/semantic/set/configure",
    status_code=204,
    description=RouterDoc.CONFIGURE_SEMANTIC_SET,
)
async def configure_semantic_set(
    spec: ConfigureSemanticSetSpec,
    memmachine: Annotated[MemMachine, Depends(get_memmachine)],
) -> None:
    """Configure a semantic set."""
    try:
        await memmachine.configure_semantic_set(
            set_id=spec.set_id,
            embedder_name=spec.embedder_name,
            llm_name=spec.llm_name,
        )
    except ResourceNotFoundError as e:
        raise RestError(code=404, message=str(e), ex=e) from e
    except ValueError as e:
        raise RestError(code=422, message="invalid argument", ex=e) from e
    except Exception as e:
        raise RestError(code=500, message="Unable to configure set", ex=e) from e


# --- Semantic Category Endpoints ---


@router.post(
    "/memories/semantic/category/get",
    description=RouterDoc.GET_SEMANTIC_CATEGORY,
)
async def get_semantic_category(
    spec: GetSemanticCategorySpec,
    memmachine: Annotated[MemMachine, Depends(get_memmachine)],
) -> SemanticCategoryEntry | None:
    """Get a semantic category by ID."""
    try:
        category = await memmachine.semantic_get_category(
            category_id=spec.category_id,
        )
    except ValueError as e:
        raise RestError(code=422, message="invalid argument", ex=e) from e
    except Exception as e:
        raise RestError(code=500, message="Unable to get category", ex=e) from e
    if category is None:
        raise RestError(code=404, message="Category not found")
    return SemanticCategoryEntry(
        id=category.id,
        name=category.name,
        prompt=category.prompt,
        description=category.description,
    )


@router.post(
    "/memories/semantic/category",
    status_code=201,
    description=RouterDoc.ADD_SEMANTIC_CATEGORY,
)
async def add_semantic_category(
    spec: AddSemanticCategorySpec,
    memmachine: Annotated[MemMachine, Depends(get_memmachine)],
) -> AddSemanticCategoryResponse:
    """Add a semantic category to a set."""
    try:
        category_id = await memmachine.semantic_add_category(
            set_id=spec.set_id,
            category_name=spec.category_name,
            prompt=spec.prompt,
            description=spec.description,
        )
    except ValueError as e:
        raise RestError(code=422, message="invalid argument", ex=e) from e
    except Exception as e:
        raise RestError(code=500, message="Unable to add category", ex=e) from e
    return AddSemanticCategoryResponse(category_id=category_id)


@router.post(
    "/memories/semantic/category/template",
    status_code=201,
    description=RouterDoc.ADD_SEMANTIC_CATEGORY_TEMPLATE,
)
async def add_semantic_category_template(
    spec: AddSemanticCategoryTemplateSpec,
    memmachine: Annotated[MemMachine, Depends(get_memmachine)],
) -> AddSemanticCategoryResponse:
    """Add a semantic category template to a set type."""
    try:
        category_id = await memmachine.semantic_add_category_template(
            set_type_id=spec.set_type_id,
            category_name=spec.category_name,
            prompt=spec.prompt,
            description=spec.description,
        )
    except ValueError as e:
        raise RestError(code=422, message="invalid argument", ex=e) from e
    except Exception as e:
        raise RestError(
            code=500, message="Unable to add category template", ex=e
        ) from e
    return AddSemanticCategoryResponse(category_id=category_id)


@router.post(
    "/memories/semantic/category/template/list",
    description=RouterDoc.LIST_SEMANTIC_CATEGORY_TEMPLATES,
)
async def list_semantic_category_templates(
    spec: ListSemanticCategoryTemplatesSpec,
    memmachine: Annotated[MemMachine, Depends(get_memmachine)],
) -> ListSemanticCategoryTemplatesResponse:
    """List semantic category templates for a set type."""
    try:
        categories = await memmachine.semantic_list_category_templates(
            set_type_id=spec.set_type_id,
        )
        entries = [
            SemanticCategoryTemplateEntry(
                id=cat.id,
                name=cat.name,
                origin_type=cat.origin_type,
                origin_id=cat.origin_id,
                inherited=cat.inherited,
            )
            for cat in categories
        ]
    except Exception as e:
        raise RestError(
            code=500, message="Unable to list category templates", ex=e
        ) from e
    return ListSemanticCategoryTemplatesResponse(categories=entries)


@router.post(
    "/memories/semantic/category/disable",
    status_code=204,
    description=RouterDoc.DISABLE_SEMANTIC_CATEGORY,
)
async def disable_semantic_category(
    spec: DisableSemanticCategorySpec,
    memmachine: Annotated[MemMachine, Depends(get_memmachine)],
) -> None:
    """Disable a semantic category for a set."""
    try:
        await memmachine.semantic_disable_category(
            set_id=spec.set_id,
            category_name=spec.category_name,
        )
    except ResourceNotFoundError as e:
        raise RestError(code=404, message=str(e), ex=e) from e
    except ValueError as e:
        raise RestError(code=422, message="invalid argument", ex=e) from e
    except Exception as e:
        raise RestError(code=500, message="Unable to disable category", ex=e) from e


@router.post(
    "/memories/semantic/category/set_ids/get",
    description=RouterDoc.GET_SEMANTIC_CATEGORY_SET_IDS,
)
async def get_semantic_category_set_ids(
    spec: GetSemanticCategorySetIdsSpec,
    memmachine: Annotated[MemMachine, Depends(get_memmachine)],
) -> GetSemanticCategorySetIdsResponse:
    """Get set IDs associated with a category."""
    try:
        set_ids = await memmachine.semantic_get_category_set_ids(
            category_id=spec.category_id,
        )
    except ValueError as e:
        raise RestError(code=422, message="invalid argument", ex=e) from e
    except Exception as e:
        raise RestError(code=500, message="Unable to get category set IDs", ex=e) from e
    return GetSemanticCategorySetIdsResponse(set_ids=list(set_ids))


@router.post(
    "/memories/semantic/category/delete",
    status_code=204,
    description=RouterDoc.DELETE_SEMANTIC_CATEGORY,
)
async def delete_semantic_category(
    spec: DeleteSemanticCategorySpec,
    memmachine: Annotated[MemMachine, Depends(get_memmachine)],
) -> None:
    """Delete a semantic category."""
    try:
        await memmachine.semantic_delete_category(
            category_id=spec.category_id,
        )
    except ResourceNotFoundError as e:
        raise RestError(code=404, message=str(e), ex=e) from e
    except ValueError as e:
        raise RestError(code=422, message="invalid argument", ex=e) from e
    except Exception as e:
        raise RestError(code=500, message="Unable to delete category", ex=e) from e


# --- Semantic Tag Endpoints ---


@router.post(
    "/memories/semantic/category/tag",
    status_code=201,
    description=RouterDoc.ADD_SEMANTIC_TAG,
)
async def add_semantic_tag(
    spec: AddSemanticTagSpec,
    memmachine: Annotated[MemMachine, Depends(get_memmachine)],
) -> AddSemanticTagResponse:
    """Add a tag to a semantic category."""
    try:
        tag_id = await memmachine.semantic_add_tag_to_category(
            category_id=spec.category_id,
            tag_name=spec.tag_name,
            tag_description=spec.tag_description,
        )
    except ResourceNotFoundError as e:
        raise RestError(code=404, message=str(e), ex=e) from e
    except ValueError as e:
        raise RestError(code=422, message="invalid argument", ex=e) from e
    except Exception as e:
        raise RestError(code=500, message="Unable to add tag", ex=e) from e
    return AddSemanticTagResponse(tag_id=tag_id)


@router.post(
    "/memories/semantic/category/tag/delete",
    status_code=204,
    description=RouterDoc.DELETE_SEMANTIC_TAG,
)
async def delete_semantic_tag(
    spec: DeleteSemanticTagSpec,
    memmachine: Annotated[MemMachine, Depends(get_memmachine)],
) -> None:
    """Delete a semantic tag."""
    try:
        await memmachine.semantic_delete_tag(
            tag_id=spec.tag_id,
        )
    except ResourceNotFoundError as e:
        raise RestError(code=404, message=str(e), ex=e) from e
    except ValueError as e:
        raise RestError(code=422, message="invalid argument", ex=e) from e
    except Exception as e:
        raise RestError(code=500, message="Unable to delete tag", ex=e) from e


# --- Episodic Memory Configuration Endpoints ---


@router.post(
    "/memory/episodic/config/get",
    description=RouterDoc.GET_EPISODIC_MEMORY_CONFIG,
)
async def get_episodic_memory_config(
    spec: GetEpisodicMemoryConfigSpec,
    memmachine: Annotated[MemMachine, Depends(get_memmachine)],
) -> EpisodicMemoryConfigEntry:
    """Get episodic memory configuration for a project."""
    session_key = f"{spec.org_id}/{spec.project_id}"
    try:
        session_info = await memmachine.get_session(session_key=session_key)
    except Exception as e:
        raise RestError(
            code=500, message="Unable to get episodic memory config", ex=e
        ) from e
    if session_info is None:
        raise RestError(code=404, message="Project not found")
    conf = session_info.episode_memory_conf
    return EpisodicMemoryConfigEntry(
        enabled=conf.enabled,
        long_term_memory_enabled=conf.long_term_memory_enabled,
        short_term_memory_enabled=conf.short_term_memory_enabled,
    )


@router.post(
    "/memory/episodic/config",
    status_code=204,
    description=RouterDoc.CONFIGURE_EPISODIC_MEMORY,
)
async def configure_episodic_memory(
    spec: ConfigureEpisodicMemorySpec,
    memmachine: Annotated[MemMachine, Depends(get_memmachine)],
) -> None:
    """Configure episodic memory for a project."""
    session_key = f"{spec.org_id}/{spec.project_id}"
    try:
        await memmachine.update_session_episodic_config(
            session_key=session_key,
            enabled=spec.enabled,
            long_term_memory_enabled=spec.long_term_memory_enabled,
            short_term_memory_enabled=spec.short_term_memory_enabled,
        )
    except SessionNotFoundError as e:
        raise RestError(code=404, message="Project not found", ex=e) from e
    except ValueError as e:
        raise RestError(code=422, message="invalid argument", ex=e) from e
    except Exception as e:
        raise RestError(
            code=500, message="Unable to configure episodic memory", ex=e
        ) from e


@router.post(
    "/memory/episodic/short_term/config/get",
    description=RouterDoc.GET_SHORT_TERM_MEMORY_CONFIG,
)
async def get_short_term_memory_config(
    spec: GetShortTermMemoryConfigSpec,
    memmachine: Annotated[MemMachine, Depends(get_memmachine)],
) -> ShortTermMemoryConfigEntry:
    """Get short-term memory configuration for a project."""
    session_key = f"{spec.org_id}/{spec.project_id}"
    try:
        session_info = await memmachine.get_session(session_key=session_key)
    except Exception as e:
        raise RestError(
            code=500, message="Unable to get short-term memory config", ex=e
        ) from e
    if session_info is None:
        raise RestError(code=404, message="Project not found")
    conf = session_info.episode_memory_conf
    return ShortTermMemoryConfigEntry(
        enabled=conf.short_term_memory_enabled,
    )


@router.post(
    "/memory/episodic/short_term/config",
    status_code=204,
    description=RouterDoc.CONFIGURE_SHORT_TERM_MEMORY,
)
async def configure_short_term_memory(
    spec: ConfigureShortTermMemorySpec,
    memmachine: Annotated[MemMachine, Depends(get_memmachine)],
) -> None:
    """Configure short-term memory for a project."""
    session_key = f"{spec.org_id}/{spec.project_id}"
    try:
        await memmachine.update_session_episodic_config(
            session_key=session_key,
            short_term_memory_enabled=spec.enabled,
        )
    except SessionNotFoundError as e:
        raise RestError(code=404, message="Project not found", ex=e) from e
    except ValueError as e:
        raise RestError(code=422, message="invalid argument", ex=e) from e
    except Exception as e:
        raise RestError(
            code=500, message="Unable to configure short-term memory", ex=e
        ) from e


@router.post(
    "/memory/episodic/long_term/config/get",
    description=RouterDoc.GET_LONG_TERM_MEMORY_CONFIG,
)
async def get_long_term_memory_config(
    spec: GetLongTermMemoryConfigSpec,
    memmachine: Annotated[MemMachine, Depends(get_memmachine)],
) -> LongTermMemoryConfigEntry:
    """Get long-term memory configuration for a project."""
    session_key = f"{spec.org_id}/{spec.project_id}"
    try:
        session_info = await memmachine.get_session(session_key=session_key)
    except Exception as e:
        raise RestError(
            code=500, message="Unable to get long-term memory config", ex=e
        ) from e
    if session_info is None:
        raise RestError(code=404, message="Project not found")
    conf = session_info.episode_memory_conf
    return LongTermMemoryConfigEntry(
        enabled=conf.long_term_memory_enabled,
    )


@router.post(
    "/memory/episodic/long_term/config",
    status_code=204,
    description=RouterDoc.CONFIGURE_LONG_TERM_MEMORY,
)
async def configure_long_term_memory(
    spec: ConfigureLongTermMemorySpec,
    memmachine: Annotated[MemMachine, Depends(get_memmachine)],
) -> None:
    """Configure long-term memory for a project."""
    session_key = f"{spec.org_id}/{spec.project_id}"
    try:
        await memmachine.update_session_episodic_config(
            session_key=session_key,
            long_term_memory_enabled=spec.enabled,
        )
    except SessionNotFoundError as e:
        raise RestError(code=404, message="Project not found", ex=e) from e
    except ValueError as e:
        raise RestError(code=422, message="invalid argument", ex=e) from e
    except Exception as e:
        raise RestError(
            code=500, message="Unable to configure long-term memory", ex=e
        ) from e


@router.get("/metrics", description=RouterDoc.METRICS_PROMETHEUS)
async def metrics() -> Response:
    """Expose Prometheus metrics endpoint."""
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@router.get("/health", description=RouterDoc.HEALTH_CHECK)
async def health_check() -> dict[str, str]:
    """Health check endpoint for container orchestration."""
    return {
        "status": "healthy",
        "service": "memmachine",
        "version": get_version().server_version,
    }


def load_v2_api_router(app: FastAPI, *, with_config_api: bool = False) -> APIRouter:
    """Load the API v2 router."""
    app.include_router(router, prefix="/api/v2")
    if with_config_api:
        app.include_router(config_router, prefix="/api/v2")
    return router
