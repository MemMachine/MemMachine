"""Tests of RegistryBackedVectorStore's creation flow, on a store whose partition storage preparation the test controls."""

import asyncio
from collections.abc import Awaitable, Callable
from typing import override
from uuid import UUID

import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import create_async_engine

from memmachine_server.common.filter.filter_parser import FilterExpr
from memmachine_server.common.vector_store import (
    QueryResult,
    Record,
    VectorStorePartitionAlreadyExistsError,
    VectorStorePartitionDeletedError,
    VectorStorePartitionPendingError,
    VectorStorePartitionSchemaMismatchError,
    registry_backed_vector_store,
)
from memmachine_server.common.vector_store.partition_registry import (
    LiveRegistration,
    sqlalchemy_partition_registry,
)
from memmachine_server.common.vector_store.partition_registry.sqlalchemy_partition_registry import (
    SQLAlchemyVectorStorePartitionRegistry,
    SQLAlchemyVectorStorePartitionRegistryParams,
)
from memmachine_server.common.vector_store.registry_backed_vector_store import (
    RegistryBackedVectorStore,
    RegistryBackedVectorStoreParams,
    RegistryBackedVectorStorePartition,
)

KEY = "p"
VECTOR_DIMENSIONS = 3

type Preparation = Callable[[str, UUID], Awaitable[None]]


class _Partition(RegistryBackedVectorStorePartition):
    @override
    async def _upsert(self, records: list[Record]) -> None:
        raise NotImplementedError

    @override
    async def _query(
        self,
        query_vectors: list[list[float]],
        *,
        limit: int,
        min_cosine_similarity: float | None,
        property_filter: FilterExpr | None,
    ) -> list[QueryResult]:
        raise NotImplementedError

    @override
    async def _delete(self, record_uuids: list[UUID]) -> None:
        raise NotImplementedError


class _Store(RegistryBackedVectorStore[_Partition]):
    """A store whose partition storage preparation is the test's `prepare`."""

    def __init__(
        self,
        registry: SQLAlchemyVectorStorePartitionRegistry,
        vector_dimensions: int = VECTOR_DIMENSIONS,
    ) -> None:
        super().__init__(
            RegistryBackedVectorStoreParams(
                partition_registry=registry,
                vector_store_name="store",
                vector_dimensions=vector_dimensions,
                indexed_properties={},
            ),
            metrics_prefix="test",
        )
        self.prepare: Preparation = _prepared
        self.purged: list[UUID] = []

    @override
    async def _prepare_storage(self) -> None:
        return None

    @override
    async def _prepare_partition_storage(
        self, partition_key: str, incarnation: UUID
    ) -> None:
        await self.prepare(partition_key, incarnation)

    @override
    def _partition_handle(self, registration: LiveRegistration) -> _Partition:
        return _Partition(
            vector_store_name=self.vector_store_name,
            registration=registration,
            vector_dimensions=self.vector_dimensions,
            indexed_properties=self.indexed_properties,
            tracker=self._tracker,
        )

    @override
    async def _purge_round(self, incarnation: UUID) -> bool:
        self.purged.append(incarnation)
        return False


async def _prepared(partition_key: str, incarnation: UUID) -> None:
    return None


@pytest_asyncio.fixture
async def registry(tmp_path):
    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'registry.db'}")
    registry = SQLAlchemyVectorStorePartitionRegistry(
        SQLAlchemyVectorStorePartitionRegistryParams(
            engine=engine, vector_store_name="store", tombstone_retention_seconds=0
        )
    )
    await registry.startup()
    yield registry
    await engine.dispose()


@pytest.fixture
def store(registry):
    return _Store(registry)


async def _purged(store: _Store) -> list[UUID]:
    """Incarnations the purge rounds claim, calling the purge until it returns False."""
    while await store.purge_deleted_partitions():
        pass
    return store.purged


@pytest.mark.asyncio
async def test_a_partition_is_pending_while_its_storage_is_prepared(store):
    started = asyncio.Event()
    release = asyncio.Event()

    async def slow(partition_key, incarnation) -> None:
        started.set()
        await release.wait()

    store.prepare = slow
    creating = asyncio.create_task(store.create_partition(KEY))
    await started.wait()

    with pytest.raises(VectorStorePartitionPendingError) as pending:
        await store.get_partition(KEY)
    with pytest.raises(VectorStorePartitionPendingError) as registered:
        await store._partition_registry.resolve(KEY)
    assert pending.value.registered_at == registered.value.registered_at
    assert pending.value.schema == store._declared_schema()
    with pytest.raises(VectorStorePartitionAlreadyExistsError):
        await store.create_partition(KEY)

    release.set()
    await creating
    assert await store.get_partition(KEY) is not None


@pytest.mark.asyncio
async def test_open_or_create_waits_for_a_pending_partition(store, monkeypatch):
    monkeypatch.setattr(
        registry_backed_vector_store, "_OPEN_OR_CREATE_RETRY_DELAY_SECONDS", 0.05
    )
    started = asyncio.Event()
    release = asyncio.Event()

    async def slow(partition_key, incarnation) -> None:
        started.set()
        await release.wait()

    store.prepare = slow
    creating = asyncio.create_task(store.create_partition(KEY))
    await started.wait()
    store.prepare = _prepared
    registry = store._partition_registry
    register_partition = registry.register
    registrations = 0

    async def counted(partition_key, schema):
        nonlocal registrations
        registrations += 1
        return await register_partition(partition_key, schema)

    monkeypatch.setattr(registry, "register", counted)
    opening = asyncio.create_task(store.open_or_create_partition(KEY))
    await asyncio.sleep(0.1)
    assert not opening.done()
    # It waits on the pending partition instead of trying to register.
    assert registrations == 0

    release.set()
    await creating
    opened = await opening

    created = await store._partition_registry.resolve(KEY)
    assert created is not None
    assert opened._incarnation == created.incarnation


@pytest.mark.asyncio
async def test_open_or_create_refuses_a_pending_partition_of_another_schema(
    store, registry
):
    started = asyncio.Event()
    release = asyncio.Event()

    async def slow(partition_key, incarnation) -> None:
        started.set()
        await release.wait()

    store.prepare = slow
    creating = asyncio.create_task(store.create_partition(KEY))
    await started.wait()

    wider = _Store(registry, vector_dimensions=VECTOR_DIMENSIONS + 1)
    with pytest.raises(VectorStorePartitionSchemaMismatchError):
        await wider.open_or_create_partition(KEY)

    release.set()
    await creating


@pytest.mark.asyncio
@pytest.mark.parametrize("create", ["create_partition", "open_or_create_partition"])
async def test_a_failed_preparation_frees_the_key_and_queues_its_incarnation(
    store, create
):
    incarnations: list[UUID] = []

    async def refused(partition_key, incarnation) -> None:
        incarnations.append(incarnation)
        raise RuntimeError("the backend refused")

    store.prepare = refused
    with pytest.raises(RuntimeError, match="refused"):
        await getattr(store, create)(KEY)

    assert await store.get_partition(KEY) is None
    assert await _purged(store) == incarnations
    store.prepare = _prepared
    await store.create_partition(KEY)
    assert await store.get_partition(KEY) is not None


@pytest.mark.asyncio
async def test_a_cancelled_preparation_frees_the_key_and_queues_its_incarnation(
    store,
):
    started = asyncio.Event()
    incarnations: list[UUID] = []

    async def hangs(partition_key, incarnation) -> None:
        incarnations.append(incarnation)
        started.set()
        await asyncio.Event().wait()

    store.prepare = hangs
    creating = asyncio.create_task(store.create_partition(KEY))
    await started.wait()
    creating.cancel()
    with pytest.raises(asyncio.CancelledError):
        await creating

    assert await store.get_partition(KEY) is None
    assert await _purged(store) == incarnations


@pytest.mark.asyncio
async def test_an_unregistration_after_a_cancelled_preparation_survives_another_cancellation(
    store, monkeypatch
):
    started = asyncio.Event()
    unregistering = asyncio.Event()
    release = asyncio.Event()
    pending_registration = sqlalchemy_partition_registry._SQLAlchemyPendingRegistration
    unregister = pending_registration.unregister

    async def hangs(partition_key, incarnation) -> None:
        started.set()
        await asyncio.Event().wait()

    async def slow(pending) -> None:
        unregistering.set()
        await release.wait()
        await unregister(pending)

    store.prepare = hangs
    monkeypatch.setattr(pending_registration, "unregister", slow)
    creating = asyncio.create_task(store.create_partition(KEY))
    await started.wait()
    creating.cancel()
    await asyncio.wait_for(unregistering.wait(), 5)
    creating.cancel()
    with pytest.raises(asyncio.CancelledError):
        await creating

    release.set()
    await asyncio.wait_for(asyncio.gather(*store._unregistrations), 5)
    assert await store.get_partition(KEY) is None


@pytest.mark.asyncio
async def test_a_failed_preparation_the_registry_cannot_undo_stays_pending_until_deleted(
    store, monkeypatch
):
    monkeypatch.setattr(
        registry_backed_vector_store, "_OPEN_OR_CREATE_RETRY_DELAY_SECONDS", 0
    )

    async def refused(partition_key, incarnation) -> None:
        raise RuntimeError("the backend refused")

    async def unreachable(pending) -> None:
        raise ConnectionError("the registry is unreachable")

    store.prepare = refused
    with monkeypatch.context() as unregistration:
        unregistration.setattr(
            sqlalchemy_partition_registry._SQLAlchemyPendingRegistration,
            "unregister",
            unreachable,
        )
        with pytest.raises(RuntimeError, match="refused"):
            await store.create_partition(KEY)
    store.prepare = _prepared

    with pytest.raises(VectorStorePartitionPendingError):
        await store.get_partition(KEY)
    with pytest.raises(VectorStorePartitionAlreadyExistsError):
        await store.create_partition(KEY)
    with pytest.raises(VectorStorePartitionPendingError):
        await store.open_or_create_partition(KEY)

    await store.delete_partition(KEY)
    await store.create_partition(KEY)
    assert await store.get_partition(KEY) is not None


@pytest.mark.asyncio
async def test_a_partition_deleted_while_its_storage_is_prepared_is_not_marked_live(
    store,
):
    """The creation raises, and the partition stays deleted."""
    incarnations: list[UUID] = []

    async def deleted_meanwhile(partition_key, incarnation) -> None:
        incarnations.append(incarnation)
        await store.delete_partition(KEY)

    store.prepare = deleted_meanwhile
    with pytest.raises(VectorStorePartitionDeletedError):
        await store.create_partition(KEY)

    assert await store.get_partition(KEY) is None
    assert await _purged(store) == incarnations


@pytest.mark.asyncio
async def test_open_or_create_creates_again_after_a_deletion_during_preparation(
    store, monkeypatch
):
    monkeypatch.setattr(
        registry_backed_vector_store, "_OPEN_OR_CREATE_RETRY_DELAY_SECONDS", 0
    )
    incarnations: list[UUID] = []

    async def deleted_the_first_time(partition_key, incarnation) -> None:
        incarnations.append(incarnation)
        if len(incarnations) == 1:
            await store.delete_partition(KEY)

    store.prepare = deleted_the_first_time
    opened = await store.open_or_create_partition(KEY)

    assert len(incarnations) == 2
    assert opened._incarnation == incarnations[1]
    assert await _purged(store) == incarnations[:1]


@pytest.mark.asyncio
async def test_calling_the_purge_until_it_returns_false_drains_every_tombstone(
    store,
):
    """A round that finds nothing still ran, so the calls go on to the next
    due tombstone."""
    incarnations = []
    for partition_key in ("a", "b"):
        await store.create_partition(partition_key)
        live = await store._partition_registry.resolve(partition_key)
        assert live is not None
        incarnations.append(live.incarnation)
        await store.delete_partition(partition_key)

    assert sorted(await _purged(store)) == sorted(incarnations)
