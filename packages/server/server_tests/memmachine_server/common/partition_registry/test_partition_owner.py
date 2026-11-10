"""Tests of an owner's open-or-create through its registry, with a storage preparation the test controls."""

import asyncio
import logging
from uuid import UUID

import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import create_async_engine

from memmachine_server.common.partition_registry import (
    PartitionAlreadyExistsError,
    PartitionCreateAttemptsExhaustedError,
    PartitionPendingError,
    Registration,
    open_or_create_partition,
    partition_owner,
    sqlalchemy_partition_registry,
)
from memmachine_server.common.partition_registry.sqlalchemy_partition_registry import (
    SQLAlchemyPartitionRegistry,
    SQLAlchemyPartitionRegistryParams,
)
from memmachine_server.common.sql import enable_sqlite_explicit_begin

NAME = "p"


async def _prepared(incarnation: UUID) -> None:
    return None


@pytest_asyncio.fixture
async def registry(tmp_path):
    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'registry.db'}")
    enable_sqlite_explicit_begin(engine)
    registry = SQLAlchemyPartitionRegistry(
        SQLAlchemyPartitionRegistryParams(
            engine=engine, partition_namespace="owner", tombstone_retention_seconds=0
        )
    )
    await registry.startup()
    yield registry
    await engine.dispose()


@pytest.fixture(autouse=True)
def _no_retry_delay(monkeypatch):
    monkeypatch.setattr(partition_owner, "_OPEN_OR_CREATE_RETRY_DELAY_SECONDS", 0)


async def _purged(registry: SQLAlchemyPartitionRegistry) -> list[UUID]:
    """Incarnations the purge rounds claim, running rounds until none is due."""
    purged: list[UUID] = []

    async def purge(incarnation: UUID) -> bool:
        purged.append(incarnation)
        return False

    while await registry.run_purge_round(purge):
        pass
    return purged


@pytest.mark.asyncio
async def test_open_or_create_creates_once_and_opens_after(registry):
    prepared: list[UUID] = []

    async def prepare(incarnation: UUID) -> None:
        prepared.append(incarnation)

    created = await open_or_create_partition(registry, NAME, prepare)
    opened = await open_or_create_partition(registry, NAME, prepare)

    assert prepared == [created.incarnation]
    assert opened.incarnation == created.incarnation
    resolved = await registry.resolve(NAME)
    assert resolved is not None
    assert resolved.incarnation == created.incarnation


@pytest.mark.asyncio
@pytest.mark.parametrize("partition_name", ["", "nul\x00", "x" * 256])
async def test_a_name_outside_the_rule_is_refused(registry, partition_name):
    with pytest.raises(ValueError, match="Partition name"):
        await open_or_create_partition(registry, partition_name, _prepared)


@pytest.mark.asyncio
async def test_open_or_create_waits_for_a_pending_partition(registry, monkeypatch):
    monkeypatch.setattr(partition_owner, "_OPEN_OR_CREATE_RETRY_DELAY_SECONDS", 0.05)
    pending = await registry.reserve(NAME)
    reserve = registry.reserve
    reservations = 0

    async def counted(partition_name):
        nonlocal reservations
        reservations += 1
        return await reserve(partition_name)

    monkeypatch.setattr(registry, "reserve", counted)
    opening = asyncio.create_task(open_or_create_partition(registry, NAME, _prepared))
    await asyncio.sleep(0.1)
    assert not opening.done()
    # It waits on the pending partition instead of trying to reserve it.
    assert reservations == 0

    await pending.confirm()
    opened = await opening
    assert opened.incarnation == pending.incarnation


@pytest.mark.asyncio
async def test_open_or_create_reports_a_partition_that_stays_pending(registry):
    pending = await registry.reserve(NAME)

    with pytest.raises(PartitionPendingError) as still_pending:
        await open_or_create_partition(registry, NAME, _prepared)
    assert still_pending.value.partition_name == NAME
    await pending.cancel()


@pytest.mark.asyncio
async def test_open_or_create_gives_up_from_the_race_it_last_lost(
    registry, monkeypatch
):
    lost = PartitionAlreadyExistsError("owner", NAME)

    async def taken(partition_name):
        raise lost

    monkeypatch.setattr(registry, "reserve", taken)

    with pytest.raises(
        PartitionCreateAttemptsExhaustedError, match="no progress"
    ) as gave_up:
        await open_or_create_partition(registry, NAME, _prepared)
    assert gave_up.value.__cause__ is lost


@pytest.mark.asyncio
async def test_a_failed_preparation_frees_the_name_and_queues_its_incarnation(
    registry,
):
    incarnations: list[UUID] = []

    async def refused(incarnation: UUID) -> None:
        incarnations.append(incarnation)
        raise RuntimeError("the backend refused")

    with pytest.raises(RuntimeError, match="refused"):
        await open_or_create_partition(registry, NAME, refused)

    assert await registry.resolve(NAME) is None
    assert await _purged(registry) == incarnations
    created = await open_or_create_partition(registry, NAME, _prepared)
    assert created.incarnation not in incarnations


@pytest.mark.asyncio
async def test_a_cancelled_preparation_frees_the_name_and_queues_its_incarnation(
    registry,
):
    started = asyncio.Event()
    incarnations: list[UUID] = []

    async def hangs(incarnation: UUID) -> None:
        incarnations.append(incarnation)
        started.set()
        await asyncio.Event().wait()

    creating = asyncio.create_task(open_or_create_partition(registry, NAME, hangs))
    await started.wait()
    creating.cancel()
    with pytest.raises(asyncio.CancelledError):
        await creating

    assert await registry.resolve(NAME) is None
    assert await _purged(registry) == incarnations


@pytest.mark.asyncio
async def test_cancelling_the_reservation_after_a_cancelled_preparation_survives_another_cancellation(
    registry, monkeypatch
):
    started = asyncio.Event()
    cancelling = asyncio.Event()
    release = asyncio.Event()
    reservation_type = sqlalchemy_partition_registry._SQLAlchemyReservation
    cancel = reservation_type.cancel

    async def hangs(incarnation: UUID) -> None:
        started.set()
        await asyncio.Event().wait()

    async def slow(reservation) -> None:
        cancelling.set()
        await release.wait()
        await cancel(reservation)

    monkeypatch.setattr(reservation_type, "cancel", slow)
    creating = asyncio.create_task(open_or_create_partition(registry, NAME, hangs))
    await started.wait()
    creating.cancel()
    await asyncio.wait_for(cancelling.wait(), 5)
    creating.cancel()
    with pytest.raises(asyncio.CancelledError):
        await creating

    release.set()
    await asyncio.wait_for(
        asyncio.gather(*partition_owner._reservation_cancellations), 5
    )
    assert await registry.resolve(NAME) is None


@pytest.mark.asyncio
async def test_a_cancel_that_fails_after_its_creation_stopped_waiting_is_still_reported(
    registry, monkeypatch, caplog
):
    """A creation cancelled again stops awaiting the reservation's cancel; the
    cancel's failure is still logged, naming the partition."""
    started = asyncio.Event()
    cancelling = asyncio.Event()
    release = asyncio.Event()

    async def hangs(incarnation: UUID) -> None:
        started.set()
        await asyncio.Event().wait()

    async def fails_late(reservation) -> None:
        cancelling.set()
        await release.wait()
        raise ConnectionError("the registry is unreachable")

    monkeypatch.setattr(
        sqlalchemy_partition_registry._SQLAlchemyReservation, "cancel", fails_late
    )
    creating = asyncio.create_task(open_or_create_partition(registry, NAME, hangs))
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
                *partition_owner._reservation_cancellations, return_exceptions=True
            ),
            5,
        )
        await asyncio.sleep(0)
    assert [
        r
        for r in caplog.records
        if r.levelno == logging.ERROR
        and repr(NAME) in r.getMessage()
        and isinstance(r.exc_info[1], ConnectionError)
    ]
    assert not partition_owner._reservation_cancellations


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
            await open_or_create_partition(registry, NAME, _prepared)

    assert await registry.resolve(NAME) is None
    assert await open_or_create_partition(registry, NAME, _prepared) is not None


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
        creating = asyncio.create_task(
            open_or_create_partition(registry, NAME, _prepared)
        )
        await confirming.wait()
        creating.cancel()
        with pytest.raises(asyncio.CancelledError):
            await creating
        await asyncio.wait_for(
            asyncio.gather(*partition_owner._reservation_cancellations), 5
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
            await open_or_create_partition(registry, NAME, _prepared)

    assert await registry.resolve(NAME) is not None


@pytest.mark.asyncio
async def test_a_failed_preparation_the_registry_cannot_undo_stays_pending_until_deleted(
    registry, monkeypatch
):
    async def refused(incarnation: UUID) -> None:
        raise RuntimeError("the backend refused")

    async def unreachable(reservation) -> None:
        raise ConnectionError("the registry is unreachable")

    with monkeypatch.context() as unregistration:
        unregistration.setattr(
            sqlalchemy_partition_registry._SQLAlchemyReservation,
            "cancel",
            unreachable,
        )
        with pytest.raises(RuntimeError, match="refused"):
            await open_or_create_partition(registry, NAME, refused)

    with pytest.raises(PartitionPendingError):
        await open_or_create_partition(registry, NAME, _prepared)

    await registry.unregister(NAME)
    assert await open_or_create_partition(registry, NAME, _prepared) is not None


@pytest.mark.asyncio
async def test_open_or_create_creates_again_after_a_deletion_during_preparation(
    registry,
):
    incarnations: list[UUID] = []

    async def deleted_the_first_time(incarnation: UUID) -> None:
        incarnations.append(incarnation)
        if len(incarnations) == 1:
            await registry.unregister(NAME)

    opened = await open_or_create_partition(registry, NAME, deleted_the_first_time)

    assert len(incarnations) == 2
    assert opened.incarnation == incarnations[1]
    assert await _purged(registry) == incarnations[:1]
