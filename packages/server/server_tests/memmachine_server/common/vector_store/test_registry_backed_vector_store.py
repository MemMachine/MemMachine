"""Tests of RegistryBackedVectorStore's creation flow, on a store whose storage preparation the test controls."""

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
    VectorStoreAttemptsExhaustedError,
    VectorStoreCollectionAlreadyExistsError,
    VectorStoreCollectionConfig,
    VectorStoreCollectionConfigMismatchError,
    VectorStoreCollectionDeletedError,
    VectorStoreCollectionPendingError,
    registry_backed_vector_store,
)
from memmachine_server.common.vector_store.collection_registry import (
    Registration,
    sqlalchemy_collection_registry,
)
from memmachine_server.common.vector_store.collection_registry.sqlalchemy_collection_registry import (
    SQLAlchemyVectorStoreCollectionRegistry,
    SQLAlchemyVectorStoreCollectionRegistryParams,
)
from memmachine_server.common.vector_store.registry_backed_vector_store import (
    RegistryBackedVectorStore,
    RegistryBackedVectorStoreCollection,
    RegistryBackedVectorStoreParams,
)

NAMESPACE = "ns"
NAME = "c"
CONFIG = VectorStoreCollectionConfig(vector_dimensions=3)

type Preparation = Callable[[str, VectorStoreCollectionConfig, UUID], Awaitable[None]]


class _Collection(RegistryBackedVectorStoreCollection):
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


class _Store(RegistryBackedVectorStore[_Collection]):
    """A store whose storage preparation is the test's `prepare`."""

    def __init__(self, registry: SQLAlchemyVectorStoreCollectionRegistry) -> None:
        super().__init__(
            RegistryBackedVectorStoreParams(collection_registry=registry),
            metrics_prefix="test",
        )
        self.prepare: Preparation = _prepared
        self.purged: list[UUID] = []

    @override
    async def _prepare_storage(
        self, namespace: str, config: VectorStoreCollectionConfig, incarnation: UUID
    ) -> None:
        await self.prepare(namespace, config, incarnation)

    @override
    def _build_collection_handle(self, registration: Registration) -> _Collection:
        return _Collection(registration=registration, tracker=self._tracker)

    @override
    async def _purge_round(
        self, namespace: str, config: VectorStoreCollectionConfig, incarnation: UUID
    ) -> bool:
        self.purged.append(incarnation)
        return False


async def _prepared(
    namespace: str, config: VectorStoreCollectionConfig, incarnation: UUID
) -> None:
    return None


@pytest_asyncio.fixture
async def store(tmp_path):
    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'registry.db'}")
    registry = SQLAlchemyVectorStoreCollectionRegistry(
        SQLAlchemyVectorStoreCollectionRegistryParams(
            engine=engine, vector_store_name="store", tombstone_retention_seconds=0
        )
    )
    await registry.startup()
    yield _Store(registry)
    await engine.dispose()


async def _purged(store: _Store) -> list[UUID]:
    """Incarnations the purge rounds claim, calling the purge until it returns False."""
    while await store.purge_deleted_collections():
        pass
    return store.purged


@pytest.mark.asyncio
async def test_a_collection_is_pending_while_its_storage_is_prepared(store):
    started = asyncio.Event()
    release = asyncio.Event()

    async def slow(namespace, config, incarnation) -> None:
        started.set()
        await release.wait()

    store.prepare = slow
    creating = asyncio.create_task(
        store.create_collection(namespace=NAMESPACE, name=NAME, config=CONFIG)
    )
    await started.wait()

    with pytest.raises(VectorStoreCollectionPendingError) as pending:
        await store.open_collection(namespace=NAMESPACE, name=NAME)
    with pytest.raises(VectorStoreCollectionPendingError) as registered:
        await store._collection_registry.resolve(NAMESPACE, NAME)
    assert pending.value.registered_at == registered.value.registered_at
    assert pending.value.config == CONFIG
    with pytest.raises(VectorStoreCollectionAlreadyExistsError):
        await store.create_collection(namespace=NAMESPACE, name=NAME, config=CONFIG)

    release.set()
    await creating
    assert await store.open_collection(namespace=NAMESPACE, name=NAME) is not None


@pytest.mark.asyncio
async def test_open_or_create_waits_for_a_pending_collection(store, monkeypatch):
    monkeypatch.setattr(
        registry_backed_vector_store, "_OPEN_OR_CREATE_RETRY_DELAY_SECONDS", 0.05
    )
    started = asyncio.Event()
    release = asyncio.Event()

    async def slow(namespace, config, incarnation) -> None:
        started.set()
        await release.wait()

    store.prepare = slow
    creating = asyncio.create_task(
        store.create_collection(namespace=NAMESPACE, name=NAME, config=CONFIG)
    )
    await started.wait()
    store.prepare = _prepared
    registry = store._collection_registry
    reserve = registry.reserve
    reservations = 0

    async def counted(namespace, name, config):
        nonlocal reservations
        reservations += 1
        return await reserve(namespace, name, config)

    monkeypatch.setattr(registry, "reserve", counted)
    opening = asyncio.create_task(
        store.open_or_create_collection(namespace=NAMESPACE, name=NAME, config=CONFIG)
    )
    await asyncio.sleep(0.1)
    assert not opening.done()
    # It waits on the pending collection instead of trying to reserve it.
    assert reservations == 0

    release.set()
    await creating
    opened = await opening

    created = await store._collection_registry.resolve(NAMESPACE, NAME)
    assert created is not None
    assert opened._incarnation == created.incarnation


@pytest.mark.asyncio
async def test_open_or_create_gives_up_from_the_race_it_last_lost(store, monkeypatch):
    monkeypatch.setattr(
        registry_backed_vector_store, "_OPEN_OR_CREATE_RETRY_DELAY_SECONDS", 0
    )
    lost = VectorStoreCollectionAlreadyExistsError(NAMESPACE, NAME)

    async def taken(namespace, name, config):
        raise lost

    monkeypatch.setattr(store._collection_registry, "reserve", taken)

    with pytest.raises(
        VectorStoreAttemptsExhaustedError, match="no progress"
    ) as gave_up:
        await store.open_or_create_collection(
            namespace=NAMESPACE, name=NAME, config=CONFIG
        )
    assert gave_up.value.__cause__ is lost


@pytest.mark.asyncio
async def test_open_or_create_refuses_a_pending_collection_of_another_configuration(
    store,
):
    started = asyncio.Event()
    release = asyncio.Event()

    async def slow(namespace, config, incarnation) -> None:
        started.set()
        await release.wait()

    store.prepare = slow
    creating = asyncio.create_task(
        store.create_collection(namespace=NAMESPACE, name=NAME, config=CONFIG)
    )
    await started.wait()

    with pytest.raises(VectorStoreCollectionConfigMismatchError):
        await store.open_or_create_collection(
            namespace=NAMESPACE,
            name=NAME,
            config=VectorStoreCollectionConfig(vector_dimensions=4),
        )

    release.set()
    await creating


@pytest.mark.asyncio
@pytest.mark.parametrize("create", ["create_collection", "open_or_create_collection"])
async def test_a_failed_preparation_frees_the_name_and_queues_its_incarnation(
    store, create
):
    incarnations: list[UUID] = []

    async def refused(namespace, config, incarnation) -> None:
        incarnations.append(incarnation)
        raise RuntimeError("the backend refused")

    store.prepare = refused
    with pytest.raises(RuntimeError, match="refused"):
        await getattr(store, create)(namespace=NAMESPACE, name=NAME, config=CONFIG)

    assert await store.open_collection(namespace=NAMESPACE, name=NAME) is None
    assert await _purged(store) == incarnations
    store.prepare = _prepared
    await store.create_collection(namespace=NAMESPACE, name=NAME, config=CONFIG)
    assert await store.open_collection(namespace=NAMESPACE, name=NAME) is not None


@pytest.mark.asyncio
async def test_a_cancelled_preparation_frees_the_name_and_queues_its_incarnation(
    store,
):
    started = asyncio.Event()
    incarnations: list[UUID] = []

    async def hangs(namespace, config, incarnation) -> None:
        incarnations.append(incarnation)
        started.set()
        await asyncio.Event().wait()

    store.prepare = hangs
    creating = asyncio.create_task(
        store.create_collection(namespace=NAMESPACE, name=NAME, config=CONFIG)
    )
    await started.wait()
    creating.cancel()
    with pytest.raises(asyncio.CancelledError):
        await creating

    assert await store.open_collection(namespace=NAMESPACE, name=NAME) is None
    assert await _purged(store) == incarnations


@pytest.mark.asyncio
async def test_cancelling_the_reservation_after_a_cancelled_preparation_survives_another_cancellation(
    store, monkeypatch
):
    started = asyncio.Event()
    cancelling = asyncio.Event()
    release = asyncio.Event()
    reservation_type = sqlalchemy_collection_registry._SQLAlchemyReservation
    cancel = reservation_type.cancel

    async def hangs(namespace, config, incarnation) -> None:
        started.set()
        await asyncio.Event().wait()

    async def slow(reservation) -> None:
        cancelling.set()
        await release.wait()
        await cancel(reservation)

    store.prepare = hangs
    monkeypatch.setattr(reservation_type, "cancel", slow)
    creating = asyncio.create_task(
        store.create_collection(namespace=NAMESPACE, name=NAME, config=CONFIG)
    )
    await started.wait()
    creating.cancel()
    await asyncio.wait_for(cancelling.wait(), 5)
    creating.cancel()
    with pytest.raises(asyncio.CancelledError):
        await creating

    release.set()
    await asyncio.wait_for(asyncio.gather(*store._reservation_cancellations), 5)
    assert await store.open_collection(namespace=NAMESPACE, name=NAME) is None


@pytest.mark.asyncio
async def test_every_lifecycle_call_is_tracked(store, monkeypatch):
    tracked: list[str] = []
    tracker = store._tracker

    def recording(operation: str):
        tracked.append(operation)
        return tracker(operation)

    monkeypatch.setattr(store, "_tracker", recording)
    await store.create_collection(namespace=NAMESPACE, name=NAME, config=CONFIG)
    await store.open_collection(namespace=NAMESPACE, name=NAME)
    await store.open_or_create_collection(namespace=NAMESPACE, name=NAME, config=CONFIG)
    await store.delete_collection(namespace=NAMESPACE, name=NAME)
    await store.purge_deleted_collections()

    assert tracked == [
        "create_collection",
        "open_collection",
        "open_or_create_collection",
        "delete_collection",
        "purge_deleted_collections",
    ]


@pytest.mark.asyncio
async def test_a_cancel_that_fails_after_its_creation_stopped_waiting_is_still_reported(
    store, monkeypatch, caplog
):
    """A creation cancelled again stops awaiting the reservation's cancel; the
    cancel's failure is still logged, naming the collection."""
    started = asyncio.Event()
    cancelling = asyncio.Event()
    release = asyncio.Event()

    async def hangs(namespace, config, incarnation) -> None:
        started.set()
        await asyncio.Event().wait()

    async def fails_late(reservation) -> None:
        cancelling.set()
        await release.wait()
        raise ConnectionError("the registry is unreachable")

    store.prepare = hangs
    monkeypatch.setattr(
        sqlalchemy_collection_registry._SQLAlchemyReservation, "cancel", fails_late
    )
    creating = asyncio.create_task(
        store.create_collection(namespace=NAMESPACE, name=NAME, config=CONFIG)
    )
    await started.wait()
    creating.cancel()
    await asyncio.wait_for(cancelling.wait(), 5)
    creating.cancel()
    with pytest.raises(asyncio.CancelledError):
        await creating

    with caplog.at_level(logging.ERROR):
        release.set()
        await asyncio.wait_for(
            asyncio.gather(*store._reservation_cancellations, return_exceptions=True), 5
        )
        await asyncio.sleep(0)
    assert [
        r
        for r in caplog.records
        if r.levelno == logging.ERROR
        and repr(NAME) in r.getMessage()
        and isinstance(r.exc_info[1], ConnectionError)
    ]
    assert not store._reservation_cancellations


@pytest.mark.asyncio
async def test_a_confirmation_that_fails_frees_the_name(store, monkeypatch):
    async def unreachable(reservation) -> Registration:
        raise ConnectionError("the registry is unreachable")

    with monkeypatch.context() as patched:
        patched.setattr(
            sqlalchemy_collection_registry._SQLAlchemyReservation,
            "confirm",
            unreachable,
        )
        with pytest.raises(ConnectionError):
            await store.create_collection(namespace=NAMESPACE, name=NAME, config=CONFIG)

    assert await store.open_collection(namespace=NAMESPACE, name=NAME) is None
    await store.create_collection(namespace=NAMESPACE, name=NAME, config=CONFIG)
    assert await store.open_collection(namespace=NAMESPACE, name=NAME) is not None


@pytest.mark.asyncio
async def test_a_cancelled_confirmation_frees_the_name(store, monkeypatch):
    confirming = asyncio.Event()

    async def hangs(reservation) -> Registration:
        confirming.set()
        await asyncio.Event().wait()
        raise AssertionError("unreachable")

    with monkeypatch.context() as patched:
        patched.setattr(
            sqlalchemy_collection_registry._SQLAlchemyReservation, "confirm", hangs
        )
        creating = asyncio.create_task(
            store.create_collection(namespace=NAMESPACE, name=NAME, config=CONFIG)
        )
        await confirming.wait()
        creating.cancel()
        with pytest.raises(asyncio.CancelledError):
            await creating
        await asyncio.wait_for(asyncio.gather(*store._reservation_cancellations), 5)

    assert await store.open_collection(namespace=NAMESPACE, name=NAME) is None


@pytest.mark.asyncio
async def test_a_confirmation_that_committed_before_failing_stands(store, monkeypatch):
    """The cancel acts only on a pending collection, so a creation whose
    confirmation committed but whose reply was lost leaves it live."""
    confirm = sqlalchemy_collection_registry._SQLAlchemyReservation.confirm

    async def commits_then_fails(reservation) -> Registration:
        await confirm(reservation)
        raise ConnectionError("the reply was lost")

    with monkeypatch.context() as patched:
        patched.setattr(
            sqlalchemy_collection_registry._SQLAlchemyReservation,
            "confirm",
            commits_then_fails,
        )
        with pytest.raises(ConnectionError):
            await store.create_collection(namespace=NAMESPACE, name=NAME, config=CONFIG)

    assert await store.open_collection(namespace=NAMESPACE, name=NAME) is not None


@pytest.mark.asyncio
async def test_a_failed_preparation_the_registry_cannot_undo_stays_pending_until_deleted(
    store, monkeypatch
):
    monkeypatch.setattr(
        registry_backed_vector_store, "_OPEN_OR_CREATE_RETRY_DELAY_SECONDS", 0
    )

    async def refused(namespace, config, incarnation) -> None:
        raise RuntimeError("the backend refused")

    async def unreachable(reservation) -> None:
        raise ConnectionError("the registry is unreachable")

    store.prepare = refused
    with monkeypatch.context() as cancellation:
        cancellation.setattr(
            sqlalchemy_collection_registry._SQLAlchemyReservation,
            "cancel",
            unreachable,
        )
        with pytest.raises(RuntimeError, match="refused"):
            await store.create_collection(namespace=NAMESPACE, name=NAME, config=CONFIG)
    store.prepare = _prepared

    with pytest.raises(VectorStoreCollectionPendingError):
        await store.open_collection(namespace=NAMESPACE, name=NAME)
    with pytest.raises(VectorStoreCollectionAlreadyExistsError):
        await store.create_collection(namespace=NAMESPACE, name=NAME, config=CONFIG)
    with pytest.raises(VectorStoreCollectionPendingError):
        await store.open_or_create_collection(
            namespace=NAMESPACE, name=NAME, config=CONFIG
        )

    await store.delete_collection(namespace=NAMESPACE, name=NAME)
    await store.create_collection(namespace=NAMESPACE, name=NAME, config=CONFIG)
    assert await store.open_collection(namespace=NAMESPACE, name=NAME) is not None


@pytest.mark.asyncio
async def test_a_collection_deleted_while_its_storage_is_prepared_is_not_confirmed(
    store,
):
    """The creation raises, and the collection stays deleted."""
    incarnations: list[UUID] = []

    async def deleted_meanwhile(namespace, config, incarnation) -> None:
        incarnations.append(incarnation)
        await store.delete_collection(namespace=NAMESPACE, name=NAME)

    store.prepare = deleted_meanwhile
    with pytest.raises(VectorStoreCollectionDeletedError):
        await store.create_collection(namespace=NAMESPACE, name=NAME, config=CONFIG)

    assert await store.open_collection(namespace=NAMESPACE, name=NAME) is None
    assert await _purged(store) == incarnations


@pytest.mark.asyncio
async def test_open_or_create_creates_again_after_a_deletion_during_preparation(
    store, monkeypatch
):
    monkeypatch.setattr(
        registry_backed_vector_store, "_OPEN_OR_CREATE_RETRY_DELAY_SECONDS", 0
    )
    incarnations: list[UUID] = []

    async def deleted_the_first_time(namespace, config, incarnation) -> None:
        incarnations.append(incarnation)
        if len(incarnations) == 1:
            await store.delete_collection(namespace=NAMESPACE, name=NAME)

    store.prepare = deleted_the_first_time
    opened = await store.open_or_create_collection(
        namespace=NAMESPACE, name=NAME, config=CONFIG
    )

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
    for name in ("a", "b"):
        await store.create_collection(namespace=NAMESPACE, name=name, config=CONFIG)
        live = await store._collection_registry.resolve(NAMESPACE, name)
        assert live is not None
        incarnations.append(live.incarnation)
        await store.delete_collection(namespace=NAMESPACE, name=name)

    assert sorted(await _purged(store)) == sorted(incarnations)


@pytest.mark.asyncio
@pytest.mark.parametrize("query_vectors", [[], [[1.0, 0.0, 0.0]]])
@pytest.mark.parametrize("limit", [0, -1])
async def test_a_query_limit_that_is_not_positive_is_refused(
    store, query_vectors, limit
):
    await store.create_collection(namespace=NAMESPACE, name=NAME, config=CONFIG)
    collection = await store.open_collection(namespace=NAMESPACE, name=NAME)
    assert collection is not None

    with pytest.raises(ValueError, match="not positive"):
        await collection.query(query_vectors=query_vectors, limit=limit)
