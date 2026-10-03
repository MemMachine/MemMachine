"""Tests of RegistryBackedVectorStore's creation flow, on a store whose partition storage preparation the test controls."""

import asyncio
import logging
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
)
from memmachine_server.common.vector_store.partition_registry import (
    Registration,
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
        score_threshold: float | None,
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
    def _partition_handle(self, registration: Registration) -> _Partition:
        return _Partition(
            vector_store_name=self.vector_store_name,
            registration=registration,
            vector_dimensions=self.vector_dimensions,
            similarity_metric=self.similarity_metric,
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
async def test_a_pending_partition_of_another_schema_is_a_mismatch(store, registry):
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
        await wider.get_partition(KEY)

    release.set()
    await creating


@pytest.mark.asyncio
async def test_a_failed_preparation_frees_the_key_and_queues_its_incarnation(store):
    incarnations: list[UUID] = []

    async def refused(partition_key, incarnation) -> None:
        incarnations.append(incarnation)
        raise RuntimeError("the backend refused")

    store.prepare = refused
    with pytest.raises(RuntimeError, match="refused"):
        await store.create_partition(KEY)

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
async def test_cancelling_the_reservation_after_a_cancelled_preparation_survives_another_cancellation(
    store, monkeypatch
):
    started = asyncio.Event()
    cancelling = asyncio.Event()
    release = asyncio.Event()
    reservation_type = sqlalchemy_partition_registry._SQLAlchemyReservation
    cancel = reservation_type.cancel

    async def hangs(partition_key, incarnation) -> None:
        started.set()
        await asyncio.Event().wait()

    async def slow(reservation) -> None:
        cancelling.set()
        await release.wait()
        await cancel(reservation)

    store.prepare = hangs
    monkeypatch.setattr(reservation_type, "cancel", slow)
    creating = asyncio.create_task(store.create_partition(KEY))
    await started.wait()
    creating.cancel()
    await asyncio.wait_for(cancelling.wait(), 5)
    creating.cancel()
    with pytest.raises(asyncio.CancelledError):
        await creating

    release.set()
    await asyncio.wait_for(asyncio.gather(*store._cancellations), 5)
    assert await store.get_partition(KEY) is None


@pytest.mark.asyncio
async def test_every_lifecycle_call_is_tracked(store, monkeypatch):
    tracked: list[str] = []
    tracker = store._tracker

    def recording(operation: str):
        tracked.append(operation)
        return tracker(operation)

    monkeypatch.setattr(store, "_tracker", recording)
    await store.create_partition(KEY)
    await store.get_partition(KEY)
    await store.delete_partition(KEY)
    await store.purge_deleted_partitions()

    assert tracked == [
        "create_partition",
        "get_partition",
        "delete_partition",
        "purge_deleted_partitions",
    ]


@pytest.mark.asyncio
async def test_a_cancel_that_fails_after_its_creation_stopped_waiting_is_still_reported(
    store, monkeypatch, caplog
):
    """A creation cancelled again stops awaiting the reservation's cancel; the
    cancel's failure is still logged, naming the partition."""
    started = asyncio.Event()
    cancelling = asyncio.Event()
    release = asyncio.Event()

    async def hangs(partition_key, incarnation) -> None:
        started.set()
        await asyncio.Event().wait()

    async def fails_late(reservation) -> None:
        cancelling.set()
        await release.wait()
        raise ConnectionError("the registry is unreachable")

    store.prepare = hangs
    monkeypatch.setattr(
        sqlalchemy_partition_registry._SQLAlchemyReservation, "cancel", fails_late
    )
    creating = asyncio.create_task(store.create_partition(KEY))
    await started.wait()
    creating.cancel()
    await asyncio.wait_for(cancelling.wait(), 5)
    creating.cancel()
    with pytest.raises(asyncio.CancelledError):
        await creating

    with caplog.at_level(logging.ERROR):
        release.set()
        await asyncio.wait_for(
            asyncio.gather(*store._cancellations, return_exceptions=True), 5
        )
        await asyncio.sleep(0)
    assert [
        r
        for r in caplog.records
        if r.levelno == logging.ERROR
        and repr(KEY) in r.getMessage()
        and isinstance(r.exc_info[1], ConnectionError)
    ]
    assert not store._cancellations


@pytest.mark.asyncio
async def test_a_confirmation_that_fails_frees_the_key(store, monkeypatch):
    async def unreachable(reservation) -> Registration:
        raise ConnectionError("the registry is unreachable")

    with monkeypatch.context() as patched:
        patched.setattr(
            sqlalchemy_partition_registry._SQLAlchemyReservation,
            "confirm",
            unreachable,
        )
        with pytest.raises(ConnectionError):
            await store.create_partition(KEY)

    assert await store.get_partition(KEY) is None
    await store.create_partition(KEY)
    assert await store.get_partition(KEY) is not None


@pytest.mark.asyncio
async def test_a_cancelled_confirmation_frees_the_key(store, monkeypatch):
    confirming = asyncio.Event()

    async def hangs(reservation) -> Registration:
        confirming.set()
        await asyncio.Event().wait()
        raise AssertionError("unreachable")

    with monkeypatch.context() as patched:
        patched.setattr(
            sqlalchemy_partition_registry._SQLAlchemyReservation, "confirm", hangs
        )
        creating = asyncio.create_task(store.create_partition(KEY))
        await confirming.wait()
        creating.cancel()
        with pytest.raises(asyncio.CancelledError):
            await creating
        await asyncio.wait_for(asyncio.gather(*store._cancellations), 5)

    assert await store.get_partition(KEY) is None


@pytest.mark.asyncio
async def test_a_confirmation_that_committed_before_failing_stands(store, monkeypatch):
    """The cancel acts only on a pending partition, so a creation whose
    confirmation committed but whose reply was lost leaves it live."""
    confirm = sqlalchemy_partition_registry._SQLAlchemyReservation.confirm

    async def commits_then_fails(reservation) -> Registration:
        await confirm(reservation)
        raise ConnectionError("the reply was lost")

    with monkeypatch.context() as patched:
        patched.setattr(
            sqlalchemy_partition_registry._SQLAlchemyReservation,
            "confirm",
            commits_then_fails,
        )
        with pytest.raises(ConnectionError):
            await store.create_partition(KEY)

    assert await store.get_partition(KEY) is not None


@pytest.mark.asyncio
async def test_a_failed_preparation_the_registry_cannot_undo_stays_pending_until_deleted(
    store, monkeypatch
):
    async def refused(partition_key, incarnation) -> None:
        raise RuntimeError("the backend refused")

    async def unreachable(reservation) -> None:
        raise ConnectionError("the registry is unreachable")

    store.prepare = refused
    with monkeypatch.context() as unregistration:
        unregistration.setattr(
            sqlalchemy_partition_registry._SQLAlchemyReservation,
            "cancel",
            unreachable,
        )
        with pytest.raises(RuntimeError, match="refused"):
            await store.create_partition(KEY)
    store.prepare = _prepared

    with pytest.raises(VectorStorePartitionPendingError):
        await store.get_partition(KEY)
    with pytest.raises(VectorStorePartitionAlreadyExistsError):
        await store.create_partition(KEY)

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


@pytest.mark.asyncio
@pytest.mark.parametrize("query_vectors", [[], [[1.0, 0.0, 0.0]]])
@pytest.mark.parametrize("limit", [0, -1])
async def test_a_query_limit_that_is_not_positive_is_refused(
    store, query_vectors, limit
):
    await store.create_partition(KEY)
    partition = await store.get_partition(KEY)
    assert partition is not None

    with pytest.raises(ValueError, match="not positive"):
        await partition.query(query_vectors=query_vectors, limit=limit)
