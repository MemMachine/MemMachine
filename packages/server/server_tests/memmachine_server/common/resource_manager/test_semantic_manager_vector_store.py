from __future__ import annotations

from unittest.mock import (
    AsyncMock,
    MagicMock,
    NonCallableMagicMock,
    call,
    create_autospec,
    sentinel,
)
from uuid import UUID, uuid4

import pytest
import pytest_asyncio

from memmachine_server.common.configuration import (
    PromptConf,
    SemanticMemoryConf,
    SemanticMemoryStorageBackend,
)
from memmachine_server.common.data_types import SimilarityMetric
from memmachine_server.common.metrics_factory import MetricsFactory, OperationTracker
from memmachine_server.common.partition_registry import PartitionManager
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
    vector_store = create_autospec(VectorStore, instance=True)
    vector_store.purge_partition.return_value = False
    registration = MagicMock()
    registration.partition_uuid = uuid4()
    partition_registry = MagicMock()
    partition_registry.resolve = AsyncMock(return_value=registration)

    resource_manager = MagicMock()
    resource_manager.get_sql_engine = AsyncMock(return_value=sqlalchemy_sqlite_engine)
    resource_manager.get_vector_store = AsyncMock(return_value=vector_store)
    resource_manager.get_partition_registry = AsyncMock(return_value=partition_registry)
    metrics_factory = MagicMock(spec=MetricsFactory)
    resource_manager.get_metrics_factory = AsyncMock(return_value=metrics_factory)

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
    # own, registered in semantic memory's namespace in the storage's database.
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
        # The namespace's purge rounds run over the same store.
        partitioned_stores=[vector_store],
    )
    partition_registry.resolve.assert_awaited_once_with("semantic_memory")
    # The partition's lifecycle is timed under semantic memory's prefix.
    assert metrics_factory.get_histogram.call_args.args[0] == (
        "semantic_memory_latency_seconds"
    )
    [observation] = metrics_factory.get_histogram.return_value.observe.call_args_list
    assert observation.kwargs["labels"] == {
        "operation": "resolve_partition",
        "status": "ok",
    }
    vector_store.get_partition.assert_awaited_once_with(registration.partition_uuid)
    assert storage._vector_partition is vector_store.get_partition.return_value

    await storage.stop()


@pytest.fixture
def partitioned_vector_store() -> NonCallableMagicMock:
    vector_store = create_autospec(VectorStore, instance=True)
    vector_store.get_partition.return_value = sentinel.partition
    return vector_store


@pytest_asyncio.fixture
async def partition_registry(
    sqlalchemy_sqlite_engine, partitioned_vector_store
) -> SQLAlchemyPartitionRegistry:
    registry = SQLAlchemyPartitionRegistry(
        SQLAlchemyPartitionRegistryParams(
            engine=sqlalchemy_sqlite_engine,
            partition_namespace="semantic_memory",
            tombstone_retention_seconds=0,
        ),
    )
    await registry.prepare()
    await registry.start()
    return registry


@pytest.fixture
def partition_manager(partition_registry, partitioned_vector_store) -> PartitionManager:
    return PartitionManager(
        partition_registry,
        [partitioned_vector_store],
        tracker=OperationTracker(None, prefix="semantic_memory"),
    )


@pytest.fixture(autouse=True)
def _no_retry_delay(monkeypatch):
    monkeypatch.setattr(semantic_manager, "_GET_OR_CREATE_RETRY_DELAY_SECONDS", 0)


@pytest.mark.asyncio
async def test_semantic_memory_creates_its_partition_once_and_then_gets_it(
    partition_manager, partitioned_vector_store
):
    created = await SemanticResourceManager._get_or_create_partition(
        partition_manager, partitioned_vector_store
    )
    gotten = await SemanticResourceManager._get_or_create_partition(
        partition_manager, partitioned_vector_store
    )

    assert created is gotten is sentinel.partition
    partitioned_vector_store.create_partition.assert_awaited_once()
    [partition_uuid] = partitioned_vector_store.create_partition.await_args.args
    assert (
        partitioned_vector_store.get_partition.await_args_list
        == [call(partition_uuid)] * 2
    )


@pytest.mark.asyncio
async def test_semantic_memory_waits_for_another_process_creating_its_partition(
    partition_registry, partition_manager, partitioned_vector_store, monkeypatch
):
    pending = await partition_registry._reserve("semantic_memory")
    resolve_partition = partition_manager.resolve_partition
    lookups = 0

    async def confirmed_before_the_third_lookup(partition_name: str):
        nonlocal lookups
        lookups += 1
        if lookups == 3:
            await pending.confirm()
        return await resolve_partition(partition_name)

    monkeypatch.setattr(
        partition_manager, "resolve_partition", confirmed_before_the_third_lookup
    )

    assert (
        await SemanticResourceManager._get_or_create_partition(
            partition_manager, partitioned_vector_store
        )
        is sentinel.partition
    )
    assert lookups == 3
    partitioned_vector_store.get_partition.assert_awaited_once_with(
        pending.partition_uuid
    )
    partitioned_vector_store.create_partition.assert_not_awaited()


@pytest.mark.asyncio
async def test_a_failed_creation_of_semantic_memorys_partition_frees_its_name(
    partition_registry, partition_manager, partitioned_vector_store
):
    partition_uuids: list[UUID] = []

    async def refused(partition_uuid: UUID) -> None:
        partition_uuids.append(partition_uuid)
        raise RuntimeError("the backend refused")

    partitioned_vector_store.create_partition.side_effect = refused

    with pytest.raises(RuntimeError, match="refused"):
        await SemanticResourceManager._get_or_create_partition(
            partition_manager, partitioned_vector_store
        )

    assert await partition_registry.resolve("semantic_memory") is None
    purged: list[UUID] = []

    while (claim := await partition_registry.claim_tombstone()) is not None:
        purged.append(claim.partition_uuid)
        await claim.complete()
    assert purged == partition_uuids
