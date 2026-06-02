from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from memmachine_server.common.configuration import (
    PromptConf,
    SemanticMemoryConf,
    SemanticMemoryStorageBackend,
)
from memmachine_server.common.resource_manager.semantic_manager import (
    SemanticResourceManager,
)
from memmachine_server.semantic_memory.storage.vector_store_semantic_storage import (
    VectorStoreSemanticStorage,
)


@pytest.mark.asyncio
async def test_semantic_manager_builds_vector_store_backend(sqlalchemy_sqlite_engine):
    vector_store = MagicMock()
    vector_collection = MagicMock()
    vector_store.open_or_create_collection = AsyncMock(return_value=vector_collection)

    resource_manager = MagicMock()
    resource_manager.get_sql_engine = AsyncMock(return_value=sqlalchemy_sqlite_engine)
    resource_manager.get_vector_store = AsyncMock(return_value=vector_store)

    manager = SemanticResourceManager(
        semantic_conf=SemanticMemoryConf(
            storage_backend=SemanticMemoryStorageBackend.VECTOR_STORE,
            feature_store="semantic_db",
            config_database="config_db",
            vector_collection="semantic_vectors",
            llm_model="llm",
            embedding_model="embedder",
            vector_dimensions=2,
        ),
        prompt_conf=PromptConf(),
        resource_manager=resource_manager,
        episode_storage=MagicMock(),
    )

    storage = await manager.get_semantic_storage()

    assert isinstance(storage, VectorStoreSemanticStorage)
    resource_manager.get_sql_engine.assert_awaited_once_with(
        "semantic_db", validate=True
    )
    resource_manager.get_vector_store.assert_awaited_once_with("semantic_vectors")
    vector_store.open_or_create_collection.assert_awaited_once()

    await storage.cleanup()
