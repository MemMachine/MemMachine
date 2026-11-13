"""Tests of the registry driving its partitions' lifecycle in its store: creation, getting, purge, and drop."""

import asyncio
import logging
from unittest.mock import NonCallableMagicMock, call, create_autospec, sentinel
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


@pytest.fixture
def store() -> NonCallableMagicMock:
    store = create_autospec(PartitionedStore, instance=True)
    store.get_partition.return_value = sentinel.partition
    store.purge_partition.return_value = False
    return store


@pytest_asyncio.fixture
async def registry(tmp_path, store):
    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'registry.db'}")
    enable_sqlite_explicit_begin(engine)
    registry = SQLAlchemyPartitionRegistry(
        SQLAlchemyPartitionRegistryParams(
            engine=engine,
            partition_namespace="lifecycle",
            tombstone_retention_seconds=0,
        ),
        partitioned_store=store,
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
async def test_a_partition_is_created_in_the_store_while_pending_then_confirmed(
    registry, store
):
    async def pending_while_created(partition_uuid: UUID) -> None:
        with pytest.raises(PartitionPendingError):
            await registry.get_partition(NAME)

    store.create_partition.side_effect = pending_while_created

    created = await registry.create_partition(NAME)

    store.create_partition.assert_awaited_once_with(created.partition_uuid)
    gotten = await registry.get_partition(NAME)
    assert gotten is not None
    assert gotten.partition_uuid == created.partition_uuid


@pytest.mark.asyncio
async def test_a_registration_carries_the_stores_handle_on_its_partition(
    registry, store
):
    created = await registry.create_partition(NAME)
    gotten = await registry.get_partition(NAME)

    assert gotten is not None
    assert created.partition is sentinel.partition
    assert gotten.partition is sentinel.partition
    assert store.get_partition.await_args_list == [call(created.partition_uuid)] * 2


@pytest.mark.asyncio
async def test_getting_a_partition_the_store_holds_nothing_of_answers_none(
    registry, store
):
    """The store finds nothing only of a partition deleted, and dropped, since the registry's read."""
    await registry.create_partition(NAME)
    store.get_partition.return_value = None

    assert await registry.get_partition(NAME) is None


@pytest.mark.asyncio
async def test_a_creation_whose_partition_the_store_holds_nothing_of_is_deleted(
    registry, store
):
    store.get_partition.return_value = None

    with pytest.raises(PartitionDeletedError) as deleted:
        await registry.create_partition(NAME)
    assert deleted.value.partition_name == NAME


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "partition_name",
    ["", "nul\x00", "x" * 256, "ü" * 128],
    ids=["empty", "nul", "too_long", "too_many_bytes"],
)
async def test_an_invalid_name_is_refused_by_every_by_name_operation(
    registry, store, partition_name
):
    with pytest.raises(ValueError, match="Partition name"):
        await registry.create_partition(partition_name)
    with pytest.raises(ValueError, match="Partition name"):
        await registry.get_partition(partition_name)
    with pytest.raises(ValueError, match="Partition name"):
        await registry.delete_partition(partition_name)
    store.create_partition.assert_not_awaited()
    assert await _queued(registry) == 0


@pytest.mark.asyncio
async def test_a_taken_name_is_refused_before_the_store_creates(registry, store):
    await registry.create_partition(NAME)

    with pytest.raises(PartitionAlreadyExistsError):
        await registry.create_partition(NAME)
    assert store.create_partition.await_count == 1


@pytest.mark.asyncio
async def test_a_failed_store_creation_frees_the_name_and_queues_the_partition(
    registry, store
):
    created: list[UUID] = []

    async def refused(partition_uuid: UUID) -> None:
        created.append(partition_uuid)
        raise RuntimeError("the backend refused")

    store.create_partition.side_effect = refused

    with pytest.raises(RuntimeError, match="refused"):
        await registry.create_partition(NAME)

    assert await registry.get_partition(NAME) is None
    assert await _purged(registry) == created
    store.create_partition.side_effect = None
    again = await registry.create_partition(NAME)
    assert again.partition_uuid not in created


@pytest.mark.asyncio
async def test_a_partition_deleted_while_the_store_creates_it_is_not_confirmed(
    registry, store
):
    created: list[UUID] = []

    async def deleted_meanwhile(partition_uuid: UUID) -> None:
        created.append(partition_uuid)
        await registry.delete_partition(NAME)

    store.create_partition.side_effect = deleted_meanwhile

    with pytest.raises(PartitionDeletedError) as deleted:
        await registry.create_partition(NAME)

    assert deleted.value.partition_uuid == created[0]
    assert await registry.get_partition(NAME) is None
    assert await _purged(registry) == created


@pytest.mark.asyncio
async def test_a_cancelled_creation_frees_the_name_and_queues_the_partition(
    registry, store
):
    started = asyncio.Event()
    created: list[UUID] = []

    async def hangs(partition_uuid: UUID) -> None:
        created.append(partition_uuid)
        started.set()
        await asyncio.Event().wait()

    store.create_partition.side_effect = hangs
    creating = asyncio.create_task(registry.create_partition(NAME))
    await started.wait()
    creating.cancel()
    with pytest.raises(asyncio.CancelledError):
        await creating

    assert await registry.get_partition(NAME) is None
    assert await _purged(registry) == created


@pytest.mark.asyncio
async def test_cancelling_the_reservation_after_a_cancelled_creation_survives_another_cancellation(
    registry, store, monkeypatch
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

    store.create_partition.side_effect = hangs
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
    assert await registry.get_partition(NAME) is None


@pytest.mark.asyncio
async def test_a_cancel_that_fails_after_its_creation_stopped_waiting_is_still_reported(
    registry, store, monkeypatch, caplog
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

    store.create_partition.side_effect = hangs
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
    async def unreachable(reservation) -> None:
        raise ConnectionError("the registry is unreachable")

    with monkeypatch.context() as patched:
        patched.setattr(
            sqlalchemy_partition_registry._SQLAlchemyReservation,
            "confirm",
            unreachable,
        )
        with pytest.raises(ConnectionError):
            await registry.create_partition(NAME)

    assert await registry.get_partition(NAME) is None
    assert await registry.create_partition(NAME) is not None


@pytest.mark.asyncio
async def test_a_cancelled_confirmation_frees_the_name(registry, monkeypatch):
    confirming = asyncio.Event()

    async def hangs(reservation) -> None:
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

    assert await registry.get_partition(NAME) is None


@pytest.mark.asyncio
async def test_a_confirmation_that_committed_before_failing_stands(
    registry, monkeypatch
):
    """The cancel acts only on a pending partition, so a creation whose
    confirmation committed but whose reply was lost leaves it live."""
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
            await registry.create_partition(NAME)

    assert await registry.get_partition(NAME) is not None


@pytest.mark.asyncio
async def test_a_failed_creation_the_registry_cannot_undo_stays_pending_until_deleted(
    registry, store, monkeypatch
):
    async def refused(partition_uuid: UUID) -> None:
        raise RuntimeError("the backend refused")

    async def unreachable(reservation) -> None:
        raise ConnectionError("the registry is unreachable")

    store.create_partition.side_effect = refused
    with monkeypatch.context() as unregistration:
        unregistration.setattr(
            sqlalchemy_partition_registry._SQLAlchemyReservation,
            "cancel",
            unreachable,
        )
        with pytest.raises(RuntimeError, match="refused"):
            await registry.create_partition(NAME)

    store.create_partition.side_effect = None
    with pytest.raises(PartitionPendingError):
        await registry.get_partition(NAME)

    await registry.delete_partition(NAME)
    assert await registry.create_partition(NAME) is not None


@pytest.mark.asyncio
async def test_a_purge_round_purges_the_store_and_drops_once_it_finds_no_records(
    registry, store
):
    created = await registry.create_partition(NAME)
    await registry.delete_partition(NAME)
    partition_uuid = created.partition_uuid
    store.purge_partition.side_effect = [True, False]

    # The store found records: the tombstone stays due, and nothing is
    # dropped.
    assert await registry.run_purge_round() is True
    store.purge_partition.assert_awaited_once_with(partition_uuid)
    store.drop_partition.assert_not_awaited()

    # The store finds none: the round drops the partition and then deletes
    # the tombstone.
    assert await registry.run_purge_round() is True
    assert store.purge_partition.await_count == 2
    store.drop_partition.assert_awaited_once_with(partition_uuid)
    assert await _queued(registry) == 0
    assert await registry.run_purge_round() is False


@pytest.mark.asyncio
async def test_a_failed_purge_keeps_the_tombstone_and_drops_nothing(registry, store):
    await registry.create_partition(NAME)
    await registry.delete_partition(NAME)
    store.purge_partition.side_effect = ConnectionError("the store is down")

    with pytest.raises(ConnectionError):
        await registry.run_purge_round()

    store.drop_partition.assert_not_awaited()
    assert await _queued(registry) == 1


@pytest.mark.asyncio
async def test_a_failed_drop_keeps_the_tombstone(registry, store):
    created = await registry.create_partition(NAME)
    await registry.delete_partition(NAME)
    store.drop_partition.side_effect = ConnectionError("the store is down")

    with pytest.raises(ConnectionError):
        await registry.run_purge_round()

    store.drop_partition.assert_awaited_once_with(created.partition_uuid)
    assert await _queued(registry) == 1


@pytest.mark.asyncio
async def test_a_round_with_nothing_due_calls_no_store(registry, store):
    assert await registry.run_purge_round() is False
    store.purge_partition.assert_not_awaited()
    store.drop_partition.assert_not_awaited()
