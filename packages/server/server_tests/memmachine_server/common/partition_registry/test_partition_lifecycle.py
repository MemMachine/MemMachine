"""Tests of the partition manager running the lifecycle of a namespace's partitions over its stores, on a SQLite registry."""

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
    PartitionManager,
    PartitionPendingError,
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
def store() -> NonCallableMagicMock:
    return _store()


@pytest.fixture
def other_store() -> NonCallableMagicMock:
    return _store()


@pytest_asyncio.fixture
async def registry(tmp_path):
    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'registry.db'}")
    enable_sqlite_explicit_begin(engine)
    registry = SQLAlchemyPartitionRegistry(
        SQLAlchemyPartitionRegistryParams(
            engine=engine,
            partition_namespace="lifecycle",
            tombstone_retention_seconds=0,
        ),
    )
    await registry.startup()
    yield registry
    await engine.dispose()


@pytest.fixture
def manager(registry, store) -> PartitionManager:
    return PartitionManager(registry, [store])


async def _created(
    manager: PartitionManager, registry: SQLAlchemyPartitionRegistry
) -> UUID:
    """Create the partition named NAME, and answer its UUID."""
    await manager.create_partition(NAME)
    registration = await registry.resolve(NAME)
    assert registration is not None
    return registration.partition_uuid


async def _purged(registry: SQLAlchemyPartitionRegistry) -> list[UUID]:
    """Claim tombstones until none is claimable, completing each, and return their partition UUIDs."""
    purged: list[UUID] = []

    while (claim := await registry.claim_tombstone()) is not None:
        purged.append(claim.partition_uuid)
        await claim.complete()
    return purged


async def _queued(registry: SQLAlchemyPartitionRegistry) -> int:
    async with registry._engine.connect() as connection:
        return (
            await connection.execute(select(func.count()).select_from(PurgeQueueRow))
        ).scalar_one()


@pytest.mark.asyncio
async def test_a_partition_is_created_in_the_store_while_pending_then_confirmed(
    registry, manager, store
):
    async def pending_while_created(partition_uuid: UUID) -> None:
        with pytest.raises(PartitionPendingError):
            await registry.resolve(NAME)

    store.create_partition.side_effect = pending_while_created

    partition_uuid = await _created(manager, registry)

    store.create_partition.assert_awaited_once_with(partition_uuid)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "partition_name",
    ["", "nul\x00", "x" * 256, "ü" * 128],
    ids=["empty", "nul", "too_long", "too_many_bytes"],
)
async def test_an_invalid_name_is_refused_by_every_by_name_operation(
    registry, manager, store, partition_name
):
    with pytest.raises(ValueError, match="Partition name"):
        await registry.reserve(partition_name)
    with pytest.raises(ValueError, match="Partition name"):
        await registry.resolve(partition_name)
    with pytest.raises(ValueError, match="Partition name"):
        await registry.unregister(partition_name)
    with pytest.raises(ValueError, match="Partition name"):
        await manager.create_partition(partition_name)
    store.create_partition.assert_not_awaited()
    assert await _queued(registry) == 0


@pytest.mark.asyncio
async def test_a_taken_name_is_refused_before_the_store_creates(manager, store):
    await manager.create_partition(NAME)

    with pytest.raises(PartitionAlreadyExistsError):
        await manager.create_partition(NAME)
    assert store.create_partition.await_count == 1


@pytest.mark.asyncio
async def test_a_failed_store_creation_frees_the_name_and_queues_the_partition(
    registry, manager, store
):
    created: list[UUID] = []

    async def refused(partition_uuid: UUID) -> None:
        created.append(partition_uuid)
        raise RuntimeError("the backend refused")

    store.create_partition.side_effect = refused

    with pytest.raises(RuntimeError, match="refused"):
        await manager.create_partition(NAME)

    assert await registry.resolve(NAME) is None
    assert await _purged(registry) == created
    store.create_partition.side_effect = None
    assert await _created(manager, registry) not in created


@pytest.mark.asyncio
async def test_a_partition_deleted_while_the_store_creates_it_is_not_confirmed(
    registry, manager, store
):
    created: list[UUID] = []

    async def deleted_meanwhile(partition_uuid: UUID) -> None:
        created.append(partition_uuid)
        await manager.delete_partition(NAME)

    store.create_partition.side_effect = deleted_meanwhile

    with pytest.raises(PartitionDeletedError) as deleted:
        await manager.create_partition(NAME)

    assert deleted.value.partition_uuid == created[0]
    assert await registry.resolve(NAME) is None
    assert await _purged(registry) == created


@pytest.mark.asyncio
async def test_a_canceled_creation_frees_the_name_and_queues_the_partition(
    registry, manager, store
):
    started = asyncio.Event()
    created: list[UUID] = []

    async def hangs(partition_uuid: UUID) -> None:
        created.append(partition_uuid)
        started.set()
        await asyncio.Event().wait()

    store.create_partition.side_effect = hangs
    creating = asyncio.create_task(manager.create_partition(NAME))
    await started.wait()
    creating.cancel()
    with pytest.raises(asyncio.CancelledError):
        await creating

    assert await registry.resolve(NAME) is None
    assert await _purged(registry) == created


@pytest.mark.asyncio
async def test_canceling_the_reservation_after_a_canceled_creation_survives_another_cancellation(
    registry, manager, store, monkeypatch
):
    started = asyncio.Event()
    canceling = asyncio.Event()
    release = asyncio.Event()
    reservation_type = sqlalchemy_partition_registry._SQLAlchemyReservation
    cancel = reservation_type._cancel

    async def hangs(partition_uuid: UUID) -> None:
        started.set()
        await asyncio.Event().wait()

    async def slow(reservation) -> None:
        canceling.set()
        await release.wait()
        await cancel(reservation)

    store.create_partition.side_effect = hangs
    monkeypatch.setattr(reservation_type, "_cancel", slow)
    creating = asyncio.create_task(manager.create_partition(NAME))
    await started.wait()
    creating.cancel()
    await asyncio.wait_for(canceling.wait(), 5)
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
    manager, store, monkeypatch, caplog
):
    """A creation canceled again stops awaiting the reservation's cancel; the
    cancel's failure is still logged, naming the partition."""
    started = asyncio.Event()
    canceling = asyncio.Event()
    release = asyncio.Event()

    async def hangs(partition_uuid: UUID) -> None:
        started.set()
        await asyncio.Event().wait()

    async def fails_late(reservation) -> None:
        canceling.set()
        await release.wait()
        raise ConnectionError("the registry is unreachable")

    store.create_partition.side_effect = hangs
    monkeypatch.setattr(
        sqlalchemy_partition_registry._SQLAlchemyReservation, "_cancel", fails_late
    )
    creating = asyncio.create_task(manager.create_partition(NAME))
    await started.wait()
    creating.cancel()
    await asyncio.wait_for(canceling.wait(), 5)
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
async def test_a_confirmation_that_fails_frees_the_name(registry, manager, monkeypatch):
    async def unreachable(reservation) -> None:
        raise ConnectionError("the registry is unreachable")

    with monkeypatch.context() as patched:
        patched.setattr(
            sqlalchemy_partition_registry._SQLAlchemyReservation,
            "confirm",
            unreachable,
        )
        with pytest.raises(ConnectionError):
            await manager.create_partition(NAME)

    assert await registry.resolve(NAME) is None
    await _created(manager, registry)


@pytest.mark.asyncio
async def test_a_canceled_confirmation_frees_the_name(registry, manager, monkeypatch):
    confirming = asyncio.Event()

    async def hangs(reservation) -> None:
        confirming.set()
        await asyncio.Event().wait()
        raise AssertionError("unreachable")

    with monkeypatch.context() as patched:
        patched.setattr(
            sqlalchemy_partition_registry._SQLAlchemyReservation, "confirm", hangs
        )
        creating = asyncio.create_task(manager.create_partition(NAME))
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
    registry, manager, monkeypatch
):
    """The cancel acts only on a pending partition, so a creation whose
    confirmation committed but whose reply was lost leaves the partition live."""
    confirm = sqlalchemy_partition_registry._SQLAlchemyReservation.confirm

    async def commits_then_fails(reservation) -> None:
        await confirm(reservation)
        raise ConnectionError("the reply was lost")

    with monkeypatch.context() as patched:
        patched.setattr(
            sqlalchemy_partition_registry._SQLAlchemyReservation,
            "confirm",
            commits_then_fails,
        )
        with pytest.raises(ConnectionError):
            await manager.create_partition(NAME)

    assert await registry.resolve(NAME) is not None


@pytest.mark.asyncio
async def test_a_failed_creation_the_registry_cannot_undo_stays_pending_until_deleted(
    registry, manager, store, monkeypatch
):
    async def refused(partition_uuid: UUID) -> None:
        raise RuntimeError("the backend refused")

    async def unreachable(reservation) -> None:
        raise ConnectionError("the registry is unreachable")

    store.create_partition.side_effect = refused
    with monkeypatch.context() as unregistration:
        unregistration.setattr(
            sqlalchemy_partition_registry._SQLAlchemyReservation,
            "_cancel",
            unreachable,
        )
        with pytest.raises(RuntimeError, match="refused"):
            await manager.create_partition(NAME)

    store.create_partition.side_effect = None
    with pytest.raises(PartitionPendingError):
        await registry.resolve(NAME)

    await manager.delete_partition(NAME)
    await _created(manager, registry)


@pytest.mark.asyncio
async def test_a_purge_round_purges_the_store_and_drops_once_it_finds_no_records(
    registry, manager, store
):
    partition_uuid = await _created(manager, registry)
    await manager.delete_partition(NAME)
    store.purge_partition.side_effect = [True, False]

    # The store finds records: the tombstone stays due, and nothing is
    # dropped.
    assert await manager.run_purge_round() is True
    store.purge_partition.assert_awaited_once_with(partition_uuid)
    store.drop_partition.assert_not_awaited()

    # The store finds none: the round drops the partition and then deletes
    # the tombstone.
    assert await manager.run_purge_round() is True
    assert store.purge_partition.await_count == 2
    store.drop_partition.assert_awaited_once_with(partition_uuid)
    assert await _queued(registry) == 0
    assert await manager.run_purge_round() is False


@pytest.mark.asyncio
async def test_a_failed_purge_keeps_the_tombstone_and_drops_nothing(
    registry, manager, store
):
    await manager.create_partition(NAME)
    await manager.delete_partition(NAME)
    store.purge_partition.side_effect = ConnectionError("the store is down")

    with pytest.raises(ConnectionError):
        await manager.run_purge_round()

    store.drop_partition.assert_not_awaited()
    assert await _queued(registry) == 1


@pytest.mark.asyncio
async def test_a_failed_drop_keeps_the_tombstone(registry, manager, store):
    partition_uuid = await _created(manager, registry)
    await manager.delete_partition(NAME)
    store.drop_partition.side_effect = ConnectionError("the store is down")

    with pytest.raises(ConnectionError):
        await manager.run_purge_round()

    store.drop_partition.assert_awaited_once_with(partition_uuid)
    assert await _queued(registry) == 1


@pytest.mark.asyncio
async def test_a_canceled_purge_round_leaves_its_claim_to_its_lease(
    registry, manager, store
):
    await manager.create_partition(NAME)
    await manager.delete_partition(NAME)
    started = asyncio.Event()

    async def hangs(partition_uuid: UUID) -> bool:
        started.set()
        await asyncio.Event().wait()
        raise AssertionError("unreachable")

    store.purge_partition.side_effect = hangs
    purging = asyncio.create_task(manager.run_purge_round())
    await started.wait()
    purging.cancel()
    with pytest.raises(asyncio.CancelledError):
        await purging

    assert await manager.run_purge_round() is False
    assert store.purge_partition.await_count == 1
    store.drop_partition.assert_not_awaited()
    assert await _queued(registry) == 1


@pytest.mark.asyncio
async def test_a_round_with_nothing_due_calls_no_store(manager, store):
    assert await manager.run_purge_round() is False
    store.purge_partition.assert_not_awaited()
    store.drop_partition.assert_not_awaited()


@pytest.mark.asyncio
async def test_a_partition_is_created_in_every_store(registry, store, other_store):
    manager = PartitionManager(registry, [store, other_store])

    partition_uuid = await _created(manager, registry)

    store.create_partition.assert_awaited_once_with(partition_uuid)
    other_store.create_partition.assert_awaited_once_with(partition_uuid)


@pytest.mark.asyncio
async def test_every_store_finishes_creating_before_the_first_error_is_raised(
    registry, store, other_store
):
    manager = PartitionManager(registry, [store, other_store])
    finished = asyncio.Event()

    async def slow(partition_uuid: UUID) -> None:
        await asyncio.sleep(0.05)
        finished.set()

    store.create_partition.side_effect = RuntimeError("the backend refused")
    other_store.create_partition.side_effect = slow

    with pytest.raises(RuntimeError, match="refused"):
        await manager.create_partition(NAME)
    assert finished.is_set()
    assert await registry.resolve(NAME) is None


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("in_store", "in_other_store"),
    [(False, False), (True, False), (False, True), (True, True)],
)
async def test_a_purge_round_finds_records_when_any_store_does_and_drops_once_none_does(
    registry, store, other_store, in_store, in_other_store
):
    manager = PartitionManager(registry, [store, other_store])
    partition_uuid = await _created(manager, registry)
    await manager.delete_partition(NAME)
    store.purge_partition.return_value = in_store
    other_store.purge_partition.return_value = in_other_store
    found = in_store or in_other_store

    assert await manager.run_purge_round() is True
    for partitioned_store in (store, other_store):
        partitioned_store.purge_partition.assert_awaited_once_with(partition_uuid)
        if found:
            partitioned_store.drop_partition.assert_not_awaited()
        else:
            partitioned_store.drop_partition.assert_awaited_once_with(partition_uuid)
    assert await _queued(registry) == (1 if found else 0)


@pytest.mark.asyncio
async def test_every_store_purges_before_the_first_error_is_raised(
    registry, store, other_store
):
    manager = PartitionManager(registry, [store, other_store])
    partition_uuid = await _created(manager, registry)
    await manager.delete_partition(NAME)
    store.purge_partition.side_effect = ConnectionError("the store is down")

    with pytest.raises(ConnectionError):
        await manager.run_purge_round()
    other_store.purge_partition.assert_awaited_once_with(partition_uuid)
    store.drop_partition.assert_not_awaited()
    other_store.drop_partition.assert_not_awaited()


@pytest.mark.asyncio
async def test_every_store_drops_before_the_first_error_is_raised(
    registry, store, other_store
):
    manager = PartitionManager(registry, [store, other_store])
    partition_uuid = await _created(manager, registry)
    await manager.delete_partition(NAME)
    store.drop_partition.side_effect = ConnectionError("the store is down")

    with pytest.raises(ConnectionError):
        await manager.run_purge_round()
    other_store.drop_partition.assert_awaited_once_with(partition_uuid)
    assert await _queued(registry) == 1
