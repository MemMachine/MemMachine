from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest
from sqlalchemy.ext.asyncio import create_async_engine

from memmachine_server.common.configuration import PromptConf, SemanticMemoryConf
from memmachine_server.common.errors import ResourceNotReadyError
from memmachine_server.common.resource_manager.semantic_manager import (
    SemanticResourceManager,
)
from memmachine_server.semantic_memory.config_store.caching_semantic_config_storage import (
    CachingSemanticConfigStorage,
)
from memmachine_server.semantic_memory.config_store.config_store_sqlalchemy import (
    SemanticConfigStorageSqlAlchemy,
)


@pytest.mark.asyncio
async def test_get_semantic_config_storage_requires_config_database():
    resource_manager = MagicMock()
    manager = SemanticResourceManager(
        semantic_conf=SemanticMemoryConf(enabled=False),
        prompt_conf=PromptConf(),
        resource_manager=resource_manager,
        episode_storage=MagicMock(),
    )

    with pytest.raises(ResourceNotReadyError):
        await manager.get_semantic_config_storage()

    resource_manager.get_sql_engine.assert_not_called()


@pytest.mark.asyncio
async def test_get_semantic_config_storage_is_uncached_by_default():
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    resource_manager = MagicMock()
    resource_manager.get_sql_engine = AsyncMock(return_value=engine)
    manager = SemanticResourceManager(
        semantic_conf=SemanticMemoryConf(
            enabled=False,
            config_database="config-db",
        ),
        prompt_conf=PromptConf(),
        resource_manager=resource_manager,
        episode_storage=MagicMock(),
    )

    try:
        storage = await manager.get_semantic_config_storage()
        assert isinstance(storage, SemanticConfigStorageSqlAlchemy)
        assert not isinstance(storage, CachingSemanticConfigStorage)
    finally:
        await engine.dispose()
