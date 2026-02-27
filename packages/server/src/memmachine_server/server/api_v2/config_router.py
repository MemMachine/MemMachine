"""API v2 router for configuration management endpoints."""

import logging
from typing import Annotated

from fastapi import APIRouter, Depends, Request
from memmachine_common.api.config_spec import (
    AddEmbedderSpec,
    AddLanguageModelSpec,
    DeleteResourceResponse,
    EpisodicMemoryConfigResponse,
    GetConfigResponse,
    LongTermMemoryConfigResponse,
    ResourcesStatus,
    ResourceStatus,
    SemanticMemoryConfigResponse,
    ShortTermMemoryConfigResponse,
    UpdateEpisodicMemorySpec,
    UpdateLongTermMemorySpec,
    UpdateMemoryConfigResponse,
    UpdateMemoryConfigSpec,
    UpdateResourceResponse,
    UpdateSemanticMemorySpec,
    UpdateShortTermMemorySpec,
)
from memmachine_common.api.doc import RouterDoc

from memmachine_server.common.errors import (
    InvalidEmbedderError,
    InvalidLanguageModelError,
    InvalidRerankerError,
)
from memmachine_server.server.api_v2.config_service import ConfigService
from memmachine_server.server.api_v2.exceptions import RestError

logger = logging.getLogger(__name__)


async def get_config_service(request: Request) -> ConfigService:
    """Get config service from application state."""
    resource_manager = request.app.state.mem_machine.resource_manager
    return ConfigService(resource_manager)


config_router = APIRouter(prefix="/config", tags=["Configuration"])


@config_router.get("", description=RouterDoc.GET_CONFIG)
async def get_config(
    service: Annotated[ConfigService, Depends(get_config_service)],
) -> GetConfigResponse:
    """Get current configuration with resource status and memory configuration."""
    resources = service.get_resources_status()
    episodic_memory = service.get_episodic_memory_config()
    semantic_memory = service.get_semantic_memory_config()
    return GetConfigResponse(
        resources=resources,
        episodic_memory=episodic_memory,
        semantic_memory=semantic_memory,
    )


@config_router.get("/resources", description=RouterDoc.GET_RESOURCES)
async def get_resources(
    service: Annotated[ConfigService, Depends(get_config_service)],
) -> ResourcesStatus:
    """Get status of all configured resources."""
    return service.get_resources_status()


@config_router.put(
    "/memory",
    description=RouterDoc.UPDATE_MEMORY_CONFIG,
)
async def update_memory_config_endpoint(
    spec: UpdateMemoryConfigSpec,
    service: Annotated[ConfigService, Depends(get_config_service)],
) -> UpdateMemoryConfigResponse:
    """Update episodic and/or semantic memory configuration."""
    if spec.episodic_memory is None and spec.semantic_memory is None:
        raise RestError(
            code=400,
            message="At least one of 'episodic_memory' or 'semantic_memory' must be provided.",
        )
    try:
        message = service.update_memory_config(
            spec.episodic_memory, spec.semantic_memory
        )
        return UpdateMemoryConfigResponse(success=True, message=message)
    except Exception as e:
        raise RestError(
            code=500, message="Failed to update memory configuration", ex=e
        ) from e


@config_router.get(
    "/memory/episodic",
    description=RouterDoc.GET_EPISODIC_CONFIG,
)
async def get_episodic_memory_config(
    service: Annotated[ConfigService, Depends(get_config_service)],
) -> EpisodicMemoryConfigResponse:
    """Get current episodic memory configuration."""
    return service.get_episodic_memory_config()


@config_router.put(
    "/memory/episodic",
    description=RouterDoc.UPDATE_EPISODIC_CONFIG,
)
async def update_episodic_memory_config(
    spec: UpdateEpisodicMemorySpec,
    service: Annotated[ConfigService, Depends(get_config_service)],
) -> UpdateMemoryConfigResponse:
    """Update episodic memory configuration."""
    try:
        message = service.update_episodic_memory_config(spec)
        return UpdateMemoryConfigResponse(success=True, message=message)
    except Exception as e:
        raise RestError(
            code=500, message="Failed to update episodic memory configuration", ex=e
        ) from e


@config_router.get(
    "/memory/episodic/long_term",
    description=RouterDoc.GET_LTM_CONFIG,
)
async def get_long_term_memory_config(
    service: Annotated[ConfigService, Depends(get_config_service)],
) -> LongTermMemoryConfigResponse:
    """Get current long-term memory configuration."""
    return service.get_long_term_memory_config()


@config_router.get(
    "/memory/episodic/short_term",
    description=RouterDoc.GET_STM_CONFIG,
)
async def get_short_term_memory_config(
    service: Annotated[ConfigService, Depends(get_config_service)],
) -> ShortTermMemoryConfigResponse:
    """Get current short-term memory configuration."""
    return service.get_short_term_memory_config()


class UpdateLongTermMemoryConfigSpec(UpdateLongTermMemorySpec):
    """Specification for updating long-term memory configuration with enabled flag."""

    enabled: bool | None = None


class UpdateShortTermMemoryConfigSpec(UpdateShortTermMemorySpec):
    """Specification for updating short-term memory configuration with enabled flag."""

    enabled: bool | None = None


@config_router.put(
    "/memory/episodic/long_term",
    description=RouterDoc.UPDATE_LTM_CONFIG,
)
async def update_long_term_memory_config(
    spec: UpdateLongTermMemoryConfigSpec,
    service: Annotated[ConfigService, Depends(get_config_service)],
) -> UpdateMemoryConfigResponse:
    """Update long-term memory configuration."""
    try:
        # Create a base spec without the enabled field
        base_spec = UpdateLongTermMemorySpec(
            embedder=spec.embedder,
            reranker=spec.reranker,
            vector_graph_store=spec.vector_graph_store,
        )
        message = service.update_long_term_memory_config(base_spec, spec.enabled)
        return UpdateMemoryConfigResponse(success=True, message=message)
    except Exception as e:
        raise RestError(
            code=500, message="Failed to update long-term memory configuration", ex=e
        ) from e


@config_router.put(
    "/memory/episodic/short_term",
    description=RouterDoc.UPDATE_STM_CONFIG,
)
async def update_short_term_memory_config(
    spec: UpdateShortTermMemoryConfigSpec,
    service: Annotated[ConfigService, Depends(get_config_service)],
) -> UpdateMemoryConfigResponse:
    """Update short-term memory configuration."""
    try:
        # Create a base spec without the enabled field
        base_spec = UpdateShortTermMemorySpec(
            llm_model=spec.llm_model,
            message_capacity=spec.message_capacity,
        )
        message = service.update_short_term_memory_config(base_spec, spec.enabled)
        return UpdateMemoryConfigResponse(success=True, message=message)
    except Exception as e:
        raise RestError(
            code=500, message="Failed to update short-term memory configuration", ex=e
        ) from e


@config_router.get(
    "/memory/semantic",
    description=RouterDoc.GET_SEMANTIC_CONFIG,
)
async def get_semantic_memory_config(
    service: Annotated[ConfigService, Depends(get_config_service)],
) -> SemanticMemoryConfigResponse:
    """Get current semantic memory configuration."""
    return service.get_semantic_memory_config()


@config_router.put(
    "/memory/semantic",
    description=RouterDoc.UPDATE_SEMANTIC_CONFIG,
)
async def update_semantic_memory_config(
    spec: UpdateSemanticMemorySpec,
    service: Annotated[ConfigService, Depends(get_config_service)],
) -> UpdateMemoryConfigResponse:
    """Update semantic memory configuration."""
    try:
        message = service.update_semantic_memory_config(spec)
        return UpdateMemoryConfigResponse(success=True, message=message)
    except Exception as e:
        raise RestError(
            code=500, message="Failed to update semantic memory configuration", ex=e
        ) from e


@config_router.post(
    "/resources/embedders",
    status_code=201,
    description=RouterDoc.ADD_EMBEDDER,
)
async def add_embedder_endpoint(
    spec: AddEmbedderSpec,
    service: Annotated[ConfigService, Depends(get_config_service)],
) -> UpdateResourceResponse:
    """Add a new embedder configuration."""
    try:
        status = await service.add_embedder(spec.name, spec.provider, spec.config)
        error_msg = None
        if status == ResourceStatus.FAILED:
            error_msg = service.get_embedder_error(spec.name) or "Unknown error"
        return UpdateResourceResponse(
            success=(status == ResourceStatus.READY),
            status=status,
            error=error_msg,
        )
    except ValueError as e:
        raise RestError(code=422, message=str(e), ex=e) from e
    except Exception as e:
        raise RestError(code=500, message="Failed to add embedder", ex=e) from e


@config_router.post(
    "/resources/language_models",
    status_code=201,
    description=RouterDoc.ADD_LANGUAGE_MODEL,
)
async def add_language_model_endpoint(
    spec: AddLanguageModelSpec,
    service: Annotated[ConfigService, Depends(get_config_service)],
) -> UpdateResourceResponse:
    """Add a new language model configuration."""
    try:
        status = await service.add_language_model(spec.name, spec.provider, spec.config)
        error_msg = None
        if status == ResourceStatus.FAILED:
            error_msg = service.get_language_model_error(spec.name) or "Unknown error"
        return UpdateResourceResponse(
            success=(status == ResourceStatus.READY),
            status=status,
            error=error_msg,
        )
    except ValueError as e:
        raise RestError(code=422, message=str(e), ex=e) from e
    except Exception as e:
        raise RestError(code=500, message="Failed to add language model", ex=e) from e


@config_router.delete(
    "/resources/embedders/{name}",
    description=RouterDoc.DELETE_EMBEDDER,
)
async def delete_embedder_endpoint(
    name: str,
    service: Annotated[ConfigService, Depends(get_config_service)],
) -> DeleteResourceResponse:
    """Remove an embedder configuration."""
    removed = service.remove_embedder(name)
    if removed:
        return DeleteResourceResponse(
            success=True,
            message=f"Embedder '{name}' removed successfully.",
        )
    raise RestError(code=404, message=f"Embedder '{name}' not found.")


@config_router.delete(
    "/resources/language_models/{name}",
    description=RouterDoc.DELETE_LANGUAGE_MODEL,
)
async def delete_language_model_endpoint(
    name: str,
    service: Annotated[ConfigService, Depends(get_config_service)],
) -> DeleteResourceResponse:
    """Remove a language model configuration."""
    removed = service.remove_language_model(name)
    if removed:
        return DeleteResourceResponse(
            success=True,
            message=f"Language model '{name}' removed successfully.",
        )
    raise RestError(code=404, message=f"Language model '{name}' not found.")


@config_router.post(
    "/resources/embedders/{name}/retry",
    description=RouterDoc.RETRY_EMBEDDER,
)
async def retry_embedder_endpoint(
    name: str,
    service: Annotated[ConfigService, Depends(get_config_service)],
) -> UpdateResourceResponse:
    """Retry building a failed embedder."""
    try:
        status = await service.retry_embedder(name)
        error_msg = None
        if status == ResourceStatus.FAILED:
            error_msg = service.get_embedder_error(name) or "Unknown error"
        return UpdateResourceResponse(
            success=(status == ResourceStatus.READY),
            status=status,
            error=error_msg,
        )
    except InvalidEmbedderError as e:
        raise RestError(code=404, message=str(e), ex=e) from e
    except Exception as e:
        raise RestError(code=500, message="Failed to retry embedder", ex=e) from e


@config_router.post(
    "/resources/language_models/{name}/retry",
    description=RouterDoc.RETRY_LANGUAGE_MODEL,
)
async def retry_language_model_endpoint(
    name: str,
    service: Annotated[ConfigService, Depends(get_config_service)],
) -> UpdateResourceResponse:
    """Retry building a failed language model."""
    try:
        status = await service.retry_language_model(name)
        error_msg = None
        if status == ResourceStatus.FAILED:
            error_msg = service.get_language_model_error(name) or "Unknown error"
        return UpdateResourceResponse(
            success=(status == ResourceStatus.READY),
            status=status,
            error=error_msg,
        )
    except InvalidLanguageModelError as e:
        raise RestError(code=404, message=str(e), ex=e) from e
    except Exception as e:
        raise RestError(code=500, message="Failed to retry language model", ex=e) from e


@config_router.post(
    "/resources/rerankers/{name}/retry",
    description=RouterDoc.RETRY_RERANKER,
)
async def retry_reranker_endpoint(
    name: str,
    service: Annotated[ConfigService, Depends(get_config_service)],
) -> UpdateResourceResponse:
    """Retry building a failed reranker."""
    try:
        status = await service.retry_reranker(name)
        error_msg = None
        if status == ResourceStatus.FAILED:
            error_msg = service.get_reranker_error(name) or "Unknown error"
        return UpdateResourceResponse(
            success=(status == ResourceStatus.READY),
            status=status,
            error=error_msg,
        )
    except InvalidRerankerError as e:
        raise RestError(code=404, message=str(e), ex=e) from e
    except Exception as e:
        raise RestError(code=500, message="Failed to retry reranker", ex=e) from e
