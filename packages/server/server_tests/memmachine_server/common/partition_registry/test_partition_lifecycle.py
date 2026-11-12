"""Tests of the registry driving its partitions' lifecycle in its stores: creation, purge, and drop."""

import asyncio
import logging
from unittest.mock import NonCallableMagicMock, create_autospec
from uuid import UUID

import pytest
import pytest_asyncio
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import create_async_engine

from memmachine_server.common.partition_registry import (
    PartitionAlreadyExistsError,
    PartitionDeletedError,
    PartitionedStore,
    PartitionPendingError,
    Registration,
    sqlalchemy_partition_registry,
)
from memmachine_server.common.partition_registry import (
    partition_registry as partition_registry_module,
)
from memmachine_server.common.partition_registry.sqlalchemy_partition_registry import (
    PurgeQueueRow,
    SQLAlchemyPartitionRegistry,
    SQLAlchemyPartitionRegistryParams,
)
from memmachine_server.common.sql import enable_sqlite_explicit_begin

NAME = "p"


def _store() -> NonCallableMagicMock:
    store = create_autospec(PartitionedStore, instance=True)
    store.purge_partition.return_value = False
    return store


@pytest.fixture
def stores() -> list[NonCallableMagicMock]:
    return [_store(), _store()]


@pytest_asyncio.fixture
async def registry(tmp_path, stores):
    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'registry.db'}")
    enable_sqlite_explicit_begin(engine)
    registry = SQLAlchemyPartitionRegistry(
        SQLAlchemyPartitionRegistryParams(
            engine=engine,
            partition_namespace="lifecycle",
            partitioned_stores=stores,
            tombstone_retention_seconds=0,
        )
    )
    await registry.startup()
    yield registry
    await engine.dispose()


async def _purged(registry: SQLAlchemyPartitionRegistry) -> list[UUID]:
    """Partition UUIDs the purge rounds claim, running rounds that find nothing until none is due."""
    purged: list[UUID] = []

    async def purge(partition_uuid: UUID) -> bool:
        purged.append(partition_uuid)
        return False

    while await registry._run_purge_round(purge):
        pass
    return purged


async def _queued(registry: SQLAlchemyPartitionRegistry) -> int:
    async with registry._engine.connect() as connection:
        return (
            await connection.execute(select(func.count()).select_from(PurgeQueueRow))
        ).scalar_one()


@pytest.mark.asyncio
async def test_a_partition_is_created_in_every_store_while_pending_then_confirmed(
    registry, stores
):
    async def pending_while_created(partition_uuid: UUID) -> None:
        with pytest.raises(PartitionPendingError):
            await registry.resolve(NAME)

    for store in stores:
        store.create_partition.side_effect = pending_while_created

    created = await registry.create_partition(NAME)

    for store in stores:
        store.create_partition.assert_awaited_once_with(created.partition_uuid)
    resolved = await registry.resolve(NAME)
    assert resolved is not None
    assert resolved.partition_uuid == created.partition_uuid


@pytest.mark.asyncio
async def test_a_taken_name_is_refused_before_any_store_creates(registry, stores):
    await registry.create_partition(NAME)

    with pytest.raises(PartitionAlreadyExistsError):
        await registry.create_partition(NAME)
    for store in stores:
        assert store.create_partition.await_count == 1


@pytest.mark.asyncio
async def test_a_failed_store_creation_frees_the_name_and_queues_the_partition(
    registry, stores
):
    created: list[UUID] = []

    async def refused(partition_uuid: UUID) -> None:
        created.append(partition_uuid)
        raise RuntimeError("the backend refused")

    stores[0].create_partition.side_effect = refused

    with pytest.raises(RuntimeError, match="refused"):
        await registry.create_partition(NAME)

    stores[1].create_partition.assert_awaited_once_with(created[0])
    assert await registry.resolve(NAME) is None
    assert await _purged(registry) == created
    stores[0].create_partition.side_effect = None
    again = await registry.create_partition(NAME)
    assert again.partition_uuid not in created


@pytest.mark.asyncio
async def test_every_store_finishes_creating_before_the_first_error_is_raised(
    registry, stores
):
    finished = asyncio.Event()

    async def refused(partition_uuid: UUID) -> None:
        raise RuntimeError("the backend refused")

    async def slow(partition_uuid: UUID) -> None:
        await asyncio.sleep(0.05)
        finished.set()

    stores[0].create_partition.side_effect = refused
    stores[1].create_partition.side_effect = slow

    with pytest.raises(RuntimeError, match="refused"):
        await registry.create_partition(NAME)
    assert finished.is_set()


@pytest.mark.asyncio
async def test_a_partition_deleted_while_the_stores_create_it_is_not_confirmed(
    registry, stores
):
    created: list[UUID] = []

    async def deleted_meanwhile(partition_uuid: UUID) -> None:
        created.append(partition_uuid)
        await registry.unregister(NAME)

    stores[0].create_partition.side_effect = deleted_meanwhile

    with pytest.raises(PartitionDeletedError) as deleted:
        await registry.create_partition(NAME)

    assert deleted.value.partition_uuid == created[0]
    assert await registry.resolve(NAME) is None
    assert await _purged(registry) == created


@pytest.mark.asyncio
async def test_a_cancelled_creation_frees_the_name_and_queues_the_partition(
    registry, stores
):
    started = asyncio.Event()
    created: list[UUID] = []

    async def hangs(partition_uuid: UUID) -> None:
        created.append(partition_uuid)
        started.set()
        await asyncio.Event().wait()

    stores[0].create_partition.side_effect = hangs
    creating = asyncio.create_task(registry.create_partition(NAME))
    await started.wait()
    creating.cancel()
    with pytest.raises(asyncio.CancelledError):
        await creating

    assert await registry.resolve(NAME) is None
    assert await _purged(registry) == created


@pytest.mark.asyncio
async def test_cancelling_the_reservation_after_a_cancelled_creation_survives_another_cancellation(
    registry, stores, monkeypatch
):
    started = asyncio.Event()
    cancelling = asyncio.Event()
    release = asyncio.Event()
    reservation_type = sqlalchemy_partition_registry._SQLAlchemyReservation
    cancel = reservation_type.cancel

    async def hangs(partition_uuid: UUID) -> None:
        started.set()
        await asyncio.Event().wait()

    async def slow(reservation) -> None:
        cancelling.set()
        await release.wait()
        await cancel(reservation)

    stores[0].create_partition.side_effect = hangs
    monkeypatch.setattr(reservation_type, "cancel", slow)
    creating = asyncio.create_task(registry.create_partition(NAME))
    await started.wait()
    creating.cancel()
    await asyncio.wait_for(cancelling.wait(), 5)
    creating.cancel()
    with pytest.raises(asyncio.CancelledError):
        await creating

    release.set()
    await asyncio.wait_for(
        asyncio.gather(*partition_registry_module._reservation_cancellations), 5
    )
    assert await registry.resolve(NAME) is None


@pytest.mark.asyncio
async def test_a_cancel_that_fails_after_its_creation_stopped_waiting_is_still_reported(
    registry, stores, monkeypatch, caplog
):
    """A creation cancelled again stops awaiting the reservation's cancel; the
    cancel's failure is still logged, naming the partition."""
    started = asyncio.Event()
    cancelling = asyncio.Event()
    release = asyncio.Event()

    async def hangs(partition_uuid: UUID) -> None:
        started.set()
        await asyncio.Event().wait()

    async def fails_late(reservation) -> None:
        cancelling.set()
        await release.wait()
        raise ConnectionError("the registry is unreachable")

    stores[0].create_partition.side_effect = hangs
    monkeypatch.setattr(
        sqlalchemy_partition_registry._SQLAlchemyReservation, "cancel", fails_late
    )
    creating = asyncio.create_task(registry.create_partition(NAME))
    await started.wait()
    creating.cancel()
    await asyncio.wait_for(cancelling.wait(), 5)
    creating.cancel()
    with pytest.raises(asyncio.CancelledError):
        await creating

    with caplog.at_level(logging.ERROR):
        release.set()
        await asyncio.wait_for(
            asyncio.gather(
                *partition_registry_module._reservation_cancellations,
                return_exceptions=True,
            ),
            5,
        )
        await asyncio.sleep(0)
    assert [
        r
        for r in caplog.records
        if r.levelno == logging.ERROR
        and repr(NAME) in r.getMessage()
        and r.exc_info is not None
        and isinstance(r.exc_info[1], ConnectionError)
    ]
    assert not partition_registry_module._reservation_cancellations


@pytest.mark.asyncio
async def test_a_confirmation_that_fails_frees_the_name(registry, monkeypatch):
    async def unreachable(reservation) -> Registration:
        raise ConnectionError("the registry is unreachable")

    with monkeypatch.context() as patched:
        patched.setattr(
            sqlalchemy_partition_registry._SQLAlchemyReservation,
            "confirm",
            unreachable,
        )
        with pytest.raises(ConnectionError):
            await registry.create_partition(NAME)

    assert await registry.resolve(NAME) is None
    assert await registry.create_partition(NAME) is not None


@pytest.mark.asyncio
async def test_a_cancelled_confirmation_frees_the_name(registry, monkeypatch):
    confirming = asyncio.Event()

    async def hangs(reservation) -> Registration:
        confirming.set()
        await asyncio.Event().wait()
        raise AssertionError("unreachable")

    with monkeypatch.context() as patched:
        patched.setattr(
            sqlalchemy_partition_registry._SQLAlchemyReservation, "confirm", hangs
        )
        creating = asyncio.create_task(registry.create_partition(NAME))
        await confirming.wait()
        creating.cancel()
        with pytest.raises(asyncio.CancelledError):
            await creating
        await asyncio.wait_for(
            asyncio.gather(*partition_registry_module._reservation_cancellations), 5
        )

    assert await registry.resolve(NAME) is None


@pytest.mark.asyncio
async def test_a_confirmation_that_committed_before_failing_stands(
    registry, monkeypatch
):
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
            await registry.create_partition(NAME)

    assert await registry.resolve(NAME) is not None


@pytest.mark.asyncio
async def test_a_failed_creation_the_registry_cannot_undo_stays_pending_until_deleted(
    registry, stores, monkeypatch
):
    async def refused(partition_uuid: UUID) -> None:
        raise RuntimeError("the backend refused")

    async def unreachable(reservation) -> None:
        raise ConnectionError("the registry is unreachable")

    stores[0].create_partition.side_effect = refused
    with monkeypatch.context() as unregistration:
        unregistration.setattr(
            sqlalchemy_partition_registry._SQLAlchemyReservation,
            "cancel",
            unreachable,
        )
        with pytest.raises(RuntimeError, match="refused"):
            await registry.create_partition(NAME)

    stores[0].create_partition.side_effect = None
    with pytest.raises(PartitionPendingError):
        await registry.resolve(NAME)

    await registry.unregister(NAME)
    assert await registry.create_partition(NAME) is not None


@pytest.mark.asyncio
async def test_a_purge_round_purges_every_store_and_drops_once_none_finds_records(
    registry, stores
):
    created = await registry.create_partition(NAME)
    await registry.unregister(NAME)
    partition_uuid = created.partition_uuid
    stores[0].purge_partition.side_effect = [True, False]

    # A store found records: the tombstone stays due, and nothing is dropped.
    assert await registry.run_purge_round() is True
    for store in stores:
        store.purge_partition.assert_awaited_once_with(partition_uuid)
        store.drop_partition.assert_not_awaited()

    # No store finds any: every store drops the partition, and the tombstone
    # goes.
    assert await registry.run_purge_round() is True
    for store in stores:
        assert store.purge_partition.await_count == 2
        store.drop_partition.assert_awaited_once_with(partition_uuid)
    assert await _queued(registry) == 0
    assert await registry.run_purge_round() is False


@pytest.mark.asyncio
async def test_a_store_whose_purge_fails_keeps_the_tombstone_while_the_others_purge(
    registry, stores
):
    created = await registry.create_partition(NAME)
    await registry.unregister(NAME)
    stores[0].purge_partition.side_effect = ConnectionError("the store is down")

    with pytest.raises(ConnectionError):
        await registry.run_purge_round()

    stores[1].purge_partition.assert_awaited_once_with(created.partition_uuid)
    for store in stores:
        store.drop_partition.assert_not_awaited()
    assert await _queued(registry) == 1


@pytest.mark.asyncio
async def test_a_store_whose_drop_fails_keeps_the_tombstone(registry, stores):
    created = await registry.create_partition(NAME)
    await registry.unregister(NAME)
    stores[0].drop_partition.side_effect = ConnectionError("the store is down")

    with pytest.raises(ConnectionError):
        await registry.run_purge_round()

    stores[1].drop_partition.assert_awaited_once_with(created.partition_uuid)
    assert await _queued(registry) == 1


@pytest.mark.asyncio
async def test_a_round_with_nothing_due_calls_no_store(registry, stores):
    assert await registry.run_purge_round() is False
    for store in stores:
        store.purge_partition.assert_not_awaited()
        store.drop_partition.assert_not_awaited()
