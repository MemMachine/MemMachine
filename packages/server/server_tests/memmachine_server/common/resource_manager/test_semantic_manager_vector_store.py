from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, NonCallableMagicMock, create_autospec
from uuid import UUID, uuid4

import pytest
import pytest_asyncio

from memmachine_server.common.configuration import (
    PromptConf,
    SemanticMemoryConf,
    SemanticMemoryStorageBackend,
)
from memmachine_server.common.data_types import SimilarityMetric
from memmachine_server.common.partition_registry.sqlalchemy_partition_registry import (
    SQLAlchemyPartitionRegistry,
    SQLAlchemyPartitionRegistryParams,
)
from memmachine_server.common.resource_manager import semantic_manager
from memmachine_server.common.resource_manager.semantic_manager import (
    SemanticResourceManager,
)
from memmachine_server.common.vector_store import VectorStore, VectorStoreSchema
from memmachine_server.semantic_memory.storage.vector_store_semantic_storage import (
    VectorStoreSemanticStorage,
)


@pytest.mark.asyncio
async def test_semantic_manager_builds_vector_store_backend(sqlalchemy_sqlite_engine):
    vector_store = MagicMock()
    partition = MagicMock()
    vector_store.open_partition = AsyncMock(return_value=partition)
    registration = MagicMock()
    registration.partition_uuid = uuid4()
    partition_registry = MagicMock()
    partition_registry.resolve = AsyncMock(return_value=registration)

    resource_manager = MagicMock()
    resource_manager.get_sql_engine = AsyncMock(return_value=sqlalchemy_sqlite_engine)
    resource_manager.get_vector_store = AsyncMock(return_value=vector_store)
    resource_manager.get_partition_registry = AsyncMock(return_value=partition_registry)

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
    # The store is semantic memory's one store, whatever its embedder,
    # built with the keys the storage writes; the partition is the manager's
    # own, registered in a namespace of its own in the storage's database.
    resource_manager.get_vector_store.assert_awaited_once_with(
        "semantic_vectors",
        vector_store_name="semantic_memory",
        store_schema=VectorStoreSchema(
            vector_dimensions=2,
            similarity_metric=SimilarityMetric.COSINE,
            indexed_properties={},
        ),
    )
    resource_manager.get_partition_registry.assert_awaited_once_with(
        "semantic_db",
        partition_namespace="semantic_memory",
        tombstone_retention_seconds=86400,
        partitioned_stores=[vector_store],
    )
    partition_registry.resolve.assert_awaited_once_with("semantic_memory")
    vector_store.open_partition.assert_awaited_once_with(registration.partition_uuid)

    await storage.cleanup()


@pytest.fixture
def partitioned_vector_store() -> NonCallableMagicMock:
    vector_store = create_autospec(VectorStore, instance=True)
    vector_store.purge_partition.return_value = False
    return vector_store


@pytest_asyncio.fixture
async def partition_registry(
    sqlalchemy_sqlite_engine, partitioned_vector_store
) -> SQLAlchemyPartitionRegistry:
    registry = SQLAlchemyPartitionRegistry(
        SQLAlchemyPartitionRegistryParams(
            engine=sqlalchemy_sqlite_engine,
            partition_namespace="semantic_memory",
            partitioned_stores=[partitioned_vector_store],
            tombstone_retention_seconds=0,
        )
    )
    await registry.startup()
    return registry


@pytest.fixture(autouse=True)
def _no_retry_delay(monkeypatch):
    monkeypatch.setattr(semantic_manager, "_OPEN_OR_CREATE_RETRY_DELAY_SECONDS", 0)


@pytest.mark.asyncio
async def test_semantic_memory_creates_its_partition_once_and_reopens_it(
    partition_registry, partitioned_vector_store
):
    created = await SemanticResourceManager._open_or_create_partition(
        partition_registry
    )
    reopened = await SemanticResourceManager._open_or_create_partition(
        partition_registry
    )

    assert reopened.partition_uuid == created.partition_uuid
    partitioned_vector_store.create_partition.assert_awaited_once_with(
        created.partition_uuid
    )


@pytest.mark.asyncio
async def test_semantic_memory_waits_for_another_process_creating_its_partition(
    partition_registry, partitioned_vector_store, monkeypatch
):
    monkeypatch.setattr(semantic_manager, "_OPEN_OR_CREATE_RETRY_DELAY_SECONDS", 0.05)
    pending = await partition_registry._reserve("semantic_memory")

    opening = asyncio.create_task(
        SemanticResourceManager._open_or_create_partition(partition_registry)
    )
    await asyncio.sleep(0.1)
    assert not opening.done()

    await pending.confirm()
    assert (await opening).partition_uuid == pending.partition_uuid
    partitioned_vector_store.create_partition.assert_not_awaited()


@pytest.mark.asyncio
async def test_a_failed_creation_of_semantic_memorys_partition_frees_its_name(
    partition_registry, partitioned_vector_store
):
    partition_uuids: list[UUID] = []

    async def refused(partition_uuid: UUID) -> None:
        partition_uuids.append(partition_uuid)
        raise RuntimeError("the backend refused")

    partitioned_vector_store.create_partition.side_effect = refused

    with pytest.raises(RuntimeError, match="refused"):
        await SemanticResourceManager._open_or_create_partition(partition_registry)

    assert await partition_registry.resolve("semantic_memory") is None
    purged: list[UUID] = []

    async def purge(partition_uuid: UUID) -> bool:
        purged.append(partition_uuid)
        return False

    while await partition_registry._run_purge_round(purge):
        pass
    assert purged == partition_uuids
