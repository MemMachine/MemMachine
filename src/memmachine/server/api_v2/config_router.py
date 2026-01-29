"""API v2 router for configuration management endpoints."""

import logging
from typing import Annotated

from fastapi import APIRouter, Depends, Request

from memmachine.common.api.config_spec import (
    AddEmbedderSpec,
    AddLanguageModelSpec,
    DeleteResourceResponse,
    GetConfigResponse,
    ResourcesStatus,
    ResourceStatus,
    UpdateMemoryConfigResponse,
    UpdateMemoryConfigSpec,
    UpdateResourceResponse,
)
from memmachine.common.api.doc import RouterDoc
from memmachine.common.errors import (
    InvalidEmbedderError,
    InvalidLanguageModelError,
    InvalidRerankerError,
)
from memmachine.common.resource_manager.resource_manager import ResourceManagerImpl
from memmachine.server.api_v2.config_service import (
    add_embedder,
    add_language_model,
    get_resources_status,
    remove_embedder,
    remove_language_model,
    retry_embedder,
    retry_language_model,
    retry_reranker,
    update_memory_config,
)
from memmachine.server.api_v2.exceptions import RestError

logger = logging.getLogger(__name__)


async def get_resource_manager(request: Request) -> ResourceManagerImpl:
    """Get resource manager from application state."""
    return request.app.state.mem_machine.resource_manager


config_router = APIRouter(prefix="/config", tags=["Configuration"])


@config_router.get("", description=RouterDoc.GET_CONFIG)
async def get_config(
    resource_manager: Annotated[ResourceManagerImpl, Depends(get_resource_manager)],
) -> GetConfigResponse:
    """Get current configuration with resource status."""
    resources = get_resources_status(resource_manager)
    return GetConfigResponse(resources=resources)


@config_router.get("/resources", description=RouterDoc.GET_RESOURCES)
async def get_resources(
    resource_manager: Annotated[ResourceManagerImpl, Depends(get_resource_manager)],
) -> ResourcesStatus:
    """Get status of all configured resources."""
    return get_resources_status(resource_manager)


@config_router.put(
    "/memory",
    description=RouterDoc.UPDATE_MEMORY_CONFIG,
)
async def update_memory_config_endpoint(
    spec: UpdateMemoryConfigSpec,
    resource_manager: Annotated[ResourceManagerImpl, Depends(get_resource_manager)],
) -> UpdateMemoryConfigResponse:
    """Update episodic and/or semantic memory configuration."""
    if spec.episodic_memory is None and spec.semantic_memory is None:
        raise RestError(
            code=400,
            message="At least one of 'episodic_memory' or 'semantic_memory' must be provided.",
        )
    try:
        message = update_memory_config(
            resource_manager, spec.episodic_memory, spec.semantic_memory
        )
        return UpdateMemoryConfigResponse(success=True, message=message)
    except Exception as e:
        raise RestError(
            code=500, message="Failed to update memory configuration", ex=e
        ) from e


@config_router.post(
    "/resources/embedders",
    status_code=201,
    description=RouterDoc.ADD_EMBEDDER,
)
async def add_embedder_endpoint(
    spec: AddEmbedderSpec,
    resource_manager: Annotated[ResourceManagerImpl, Depends(get_resource_manager)],
) -> UpdateResourceResponse:
    """Add a new embedder configuration."""
    try:
        status = await add_embedder(
            resource_manager, spec.name, spec.provider, spec.config
        )
        error_msg = None
        if status == ResourceStatus.FAILED:
            error = resource_manager.embedder_manager.get_resource_error(spec.name)
            error_msg = str(error) if error else "Unknown error"
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
    resource_manager: Annotated[ResourceManagerImpl, Depends(get_resource_manager)],
) -> UpdateResourceResponse:
    """Add a new language model configuration."""
    try:
        status = await add_language_model(
            resource_manager, spec.name, spec.provider, spec.config
        )
        error_msg = None
        if status == ResourceStatus.FAILED:
            error = resource_manager.language_model_manager.get_resource_error(
                spec.name
            )
            error_msg = str(error) if error else "Unknown error"
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
    resource_manager: Annotated[ResourceManagerImpl, Depends(get_resource_manager)],
) -> DeleteResourceResponse:
    """Remove an embedder configuration."""
    removed = remove_embedder(resource_manager, name)
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
    resource_manager: Annotated[ResourceManagerImpl, Depends(get_resource_manager)],
) -> DeleteResourceResponse:
    """Remove a language model configuration."""
    removed = remove_language_model(resource_manager, name)
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
    resource_manager: Annotated[ResourceManagerImpl, Depends(get_resource_manager)],
) -> UpdateResourceResponse:
    """Retry building a failed embedder."""
    try:
        status = await retry_embedder(resource_manager, name)
        error_msg = None
        if status == ResourceStatus.FAILED:
            error = resource_manager.embedder_manager.get_resource_error(name)
            error_msg = str(error) if error else "Unknown error"
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
    resource_manager: Annotated[ResourceManagerImpl, Depends(get_resource_manager)],
) -> UpdateResourceResponse:
    """Retry building a failed language model."""
    try:
        status = await retry_language_model(resource_manager, name)
        error_msg = None
        if status == ResourceStatus.FAILED:
            error = resource_manager.language_model_manager.get_resource_error(name)
            error_msg = str(error) if error else "Unknown error"
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
    resource_manager: Annotated[ResourceManagerImpl, Depends(get_resource_manager)],
) -> UpdateResourceResponse:
    """Retry building a failed reranker."""
    try:
        status = await retry_reranker(resource_manager, name)
        error_msg = None
        if status == ResourceStatus.FAILED:
            error = resource_manager.reranker_manager.get_resource_error(name)
            error_msg = str(error) if error else "Unknown error"
        return UpdateResourceResponse(
            success=(status == ResourceStatus.READY),
            status=status,
            error=error_msg,
        )
    except InvalidRerankerError as e:
        raise RestError(code=404, message=str(e), ex=e) from e
    except Exception as e:
        raise RestError(code=500, message="Failed to retry reranker", ex=e) from e
