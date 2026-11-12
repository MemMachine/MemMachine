"""The SQLAlchemy partition registry of one partition namespace: creation arbitrated by the database, deletion queued, purge claimed."""

import asyncio
import logging
import random
from collections.abc import AsyncGenerator, Callable, Iterator
from contextlib import asynccontextmanager, contextmanager
from datetime import timedelta
from uuid import UUID, uuid4

import pytest
from sqlalchemy import (
    ColumnElement,
    delete,
    event,
    func,
    insert,
    select,
    text,
    update,
)
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine
from sqlalchemy.pool import NullPool, StaticPool

from memmachine_server.common.partition_registry import (
    PartitionAlreadyExistsError,
    PartitionCreateAttemptsExhaustedError,
    PartitionDeletedError,
    PartitionPendingError,
    Registration,
    Reservation,
)
from memmachine_server.common.partition_registry.sqlalchemy_partition_registry import (
    _MAX_PURGE_ATTEMPTS_WITHOUT_PROGRESS,
    PartitionRow,
    PurgeQueueRow,
    SQLAlchemyPartitionRegistry,
    SQLAlchemyPartitionRegistryParams,
    _TombstoneClaim,
)
from memmachine_server.common.sql import enable_sqlite_explicit_begin

RETENTION = timedelta(days=1)
RETENTION_SECONDS = int(RETENTION.total_seconds())
PURGE_LEASE = timedelta(minutes=5)
# A partition name of the longest length, with characters no identifier allows.
ANY_CHARACTERS_NAME = "Session/ID-1 ü".ljust(254, "x")


@pytest.fixture
def partition_namespace() -> str:
    """A partition namespace no other test (or earlier run on a shared server) used."""
    return f"test_{uuid4().hex[:12]}"


async def _registry(
    engine: AsyncEngine,
    partition_namespace: str,
    tombstone_retention_seconds: int = RETENTION_SECONDS,
    purge_lease_seconds: int = int(PURGE_LEASE.total_seconds()),
    base_purge_retry_backoff_seconds: int = 30,
) -> SQLAlchemyPartitionRegistry:
    registry = SQLAlchemyPartitionRegistry(
        SQLAlchemyPartitionRegistryParams(
            engine=engine,
            partition_namespace=partition_namespace,
            partitioned_stores=[],
            tombstone_retention_seconds=tombstone_retention_seconds,
            purge_lease_seconds=purge_lease_seconds,
            base_purge_retry_backoff_seconds=base_purge_retry_backoff_seconds,
        )
    )
    await registry.startup()
    return registry


async def _queued(registry: SQLAlchemyPartitionRegistry) -> list[UUID]:
    """The registry's tombstones, oldest first."""
    async with registry._engine.connect() as connection:
        rows = await connection.execute(
            select(PurgeQueueRow.partition_uuid)
            .where(PurgeQueueRow.partition_namespace == registry._partition_namespace)
            .order_by(PurgeQueueRow.enqueued_at)
        )
        return list(rows.scalars())


def _database_time_ago(engine: AsyncEngine, delta: timedelta) -> ColumnElement:
    """The database clock's now less `delta`, written by the database in the form of its own stamps."""
    if engine.dialect.name == "sqlite":
        return func.datetime("now", f"-{int(delta.total_seconds())} seconds")
    return func.now() - delta


async def _age_deletion(
    registry: SQLAlchemyPartitionRegistry,
    partition_uuid: UUID,
    extra: timedelta = timedelta(0),
) -> None:
    """Move the tombstone's deletion back past the retention, by `extra` more, on the database clock."""
    async with registry._engine.begin() as connection:
        await connection.execute(
            update(PurgeQueueRow)
            .where(PurgeQueueRow.partition_uuid == partition_uuid)
            .values(
                enqueued_at=_database_time_ago(
                    registry._engine, RETENTION + timedelta(seconds=1) + extra
                )
            )
        )


async def _blocked_or_done(engine: AsyncEngine, task: asyncio.Task) -> str:
    """Wait until `task` finishes ("done") or another backend waits on a lock ("blocked").

    Decided by the database's own state (pg_stat_activity); raises TimeoutError
    if neither happens within 30 seconds.
    """
    deadline = asyncio.get_running_loop().time() + 30
    while not task.done():
        async with engine.connect() as connection:
            blocked = (
                await connection.execute(
                    text(
                        "SELECT count(*) FROM pg_stat_activity "
                        "WHERE wait_event_type = 'Lock' "
                        "AND datname = current_database() "
                        "AND pid != pg_backend_pid()"
                    )
                )
            ).scalar_one()
        if blocked:
            return "blocked"
        if asyncio.get_running_loop().time() > deadline:
            raise TimeoutError("the task neither blocked on a lock nor finished")
        await asyncio.sleep(0.01)
    return "done"


async def _attempts(registry: SQLAlchemyPartitionRegistry, partition_uuid: UUID) -> int:
    """The tombstone's count of purge attempts without progress."""
    async with registry._engine.connect() as connection:
        return (
            await connection.execute(
                select(PurgeQueueRow.attempts_without_progress).where(
                    PurgeQueueRow.partition_uuid == partition_uuid
                )
            )
        ).scalar_one()


async def _age_claimable_at(
    registry: SQLAlchemyPartitionRegistry,
    partition_uuid: UUID,
    by: timedelta = PURGE_LEASE + timedelta(minutes=1),
) -> None:
    """Move the tombstone's claimable time back by `by`, as if that much more time had passed since the claim or raise that set it, past the purge lease and the first backoff unless given."""
    if registry._engine.dialect.name == "sqlite":
        earlier = func.datetime(
            PurgeQueueRow.claimable_at, f"-{int(by.total_seconds())} seconds"
        )
    else:
        earlier = PurgeQueueRow.claimable_at - by
    async with registry._engine.begin() as connection:
        await connection.execute(
            update(PurgeQueueRow)
            .where(PurgeQueueRow.partition_uuid == partition_uuid)
            .values(claimable_at=earlier)
        )


class _HeldRound:
    """A purge round that reports its start and ends when released, with the outcome given."""

    def __init__(self) -> None:
        self.ran: list[UUID] = []
        self.running = asyncio.Event()
        self._outcome: asyncio.Future[bool] = asyncio.get_running_loop().create_future()

    async def __call__(self, partition_uuid: UUID) -> bool:
        self.ran.append(partition_uuid)
        self.running.set()
        return await self._outcome

    def end(self, any_records_found: bool) -> None:
        self._outcome.set_result(any_records_found)

    def fail(self, error: Exception) -> None:
        self._outcome.set_exception(error)


async def _start_held_round(
    registry: SQLAlchemyPartitionRegistry,
) -> tuple[_HeldRound, asyncio.Task[bool]]:
    """A purge round started on the oldest due tombstone, held until released."""
    held = _HeldRound()
    task = asyncio.create_task(registry._run_purge_round(held))
    await asyncio.wait_for(held.running.wait(), 30)
    return held, task


async def _abandon_claim(
    registry: SQLAlchemyPartitionRegistry,
    monkeypatch: pytest.MonkeyPatch,
) -> UUID:
    """Claim the oldest due tombstone and stop, as a purger that dies during its round: nothing after the claim reaches the database. The claimed partition UUID."""
    held, task = await _start_held_round(registry)

    async def lost(claim: _TombstoneClaim) -> None:
        raise ConnectionError("the purger is gone")

    with monkeypatch.context() as patch:
        patch.setattr(registry, "_end_tombstone_claim", lost)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    return held.ran[0]


class _InjectedFailureError(Exception):
    """A database statement failed on purpose."""


@contextmanager
def _failing_statement(
    engine: AsyncEngine, prefix: str, armed: Callable[[], bool] = lambda: True
) -> Iterator[None]:
    """Fail the first statement on the engine that starts with `prefix` once `armed()` holds."""
    failed = False

    def on_statement(_connection, _cursor, statement, _parameters, _context, _many):
        nonlocal failed
        if not failed and statement.startswith(prefix) and armed():
            failed = True
            raise _InjectedFailureError(statement)

    event.listen(engine.sync_engine, "before_cursor_execute", on_statement)
    try:
        yield
    finally:
        event.remove(engine.sync_engine, "before_cursor_execute", on_statement)


async def _no_round(partition_uuid: UUID) -> bool:
    raise AssertionError("a round ran where none was due")


async def _failing_round(registry: SQLAlchemyPartitionRegistry) -> UUID:
    """One purge round that raises; the partition UUID it ran on."""
    ran: list[UUID] = []

    async def refused(partition_uuid: UUID) -> bool:
        ran.append(partition_uuid)
        raise RuntimeError("the backend refused")

    with pytest.raises(RuntimeError, match="the backend refused"):
        await registry._run_purge_round(refused)
    return ran[0]


async def _round(
    registry: SQLAlchemyPartitionRegistry, any_records_found: bool
) -> UUID | None:
    """One purge round on the oldest due tombstone, reporting `any_records_found`; its partition UUID, or None."""
    ran: list[UUID] = []

    async def purge_round(partition_uuid: UUID) -> bool:
        ran.append(partition_uuid)
        return any_records_found

    if not await registry._run_purge_round(purge_round):
        return None
    return ran[0]


@pytest.mark.asyncio
async def test_a_reservation_mints_a_fresh_partition_uuid(
    sqlalchemy_engine, partition_namespace
):
    registry = await _registry(sqlalchemy_engine, partition_namespace)

    pending = await registry._reserve("c")
    live = await pending.confirm()

    resolved = await registry.resolve("c")
    assert resolved is not None
    assert resolved.partition_uuid == pending.partition_uuid
    assert live.partition_uuid == pending.partition_uuid
    assert await registry.resolve("d") is None


@pytest.mark.asyncio
async def test_a_partition_is_pending_until_its_reservation_is_confirmed(
    sqlalchemy_engine, partition_namespace
):
    registry = await _registry(sqlalchemy_engine, partition_namespace)

    pending = await registry._reserve("c")

    with pytest.raises(PartitionPendingError) as resolved:
        await registry.resolve("c")
    assert resolved.value.registered_at.tzinfo is not None
    await pending.confirm()
    live = await registry.resolve("c")
    assert live is not None
    assert live.partition_uuid == pending.partition_uuid
    with pytest.raises(PartitionDeletedError):
        await pending.confirm()


@pytest.mark.asyncio
async def test_only_the_reserved_partition_is_confirmed(
    sqlalchemy_engine, partition_namespace
):
    """A creation whose partition was deleted while its storage was
    prepared cannot confirm a partition reserved under the name since."""
    registry = await _registry(sqlalchemy_engine, partition_namespace)
    deleted = await registry._reserve("c")
    await registry.unregister("c")
    with pytest.raises(PartitionDeletedError):
        await deleted.confirm()

    reserved_since = await registry._reserve("c")

    with pytest.raises(PartitionDeletedError):
        await deleted.confirm()
    with pytest.raises(PartitionPendingError):
        await registry.resolve("c")
    live = await reserved_since.confirm()
    assert live.partition_uuid == reserved_since.partition_uuid


@pytest.mark.asyncio
async def test_cancelling_a_reservation_spares_the_name_reserved_since(
    sqlalchemy_engine, partition_namespace
):
    registry = await _registry(sqlalchemy_engine, partition_namespace)
    old = await registry._reserve("c")
    await registry.unregister("c")
    since = await registry._reserve("c")

    await old.cancel()
    with pytest.raises(PartitionPendingError):
        await registry.resolve("c")
    assert await _queued(registry) == [old.partition_uuid]

    await since.cancel()
    await since.cancel()
    assert await registry.resolve("c") is None
    assert sorted(await _queued(registry)) == sorted(
        [old.partition_uuid, since.partition_uuid]
    )


@pytest.mark.asyncio
async def test_cancelling_a_confirmed_reservation_leaves_the_partition_live(
    sqlalchemy_engine, partition_namespace
):
    """Only a deletion by name ends a live partition, so a creator that
    cancels after a confirmation it did not see succeed takes nothing back."""
    registry = await _registry(sqlalchemy_engine, partition_namespace)
    reservation = await registry._reserve("c")
    live = await reservation.confirm()

    await reservation.cancel()
    await live.require_current()
    assert await _queued(registry) == []


@pytest.mark.asyncio
async def test_a_registration_is_current_until_its_partition_is_deleted(
    sqlalchemy_engine, partition_namespace
):
    registry = await _registry(sqlalchemy_engine, partition_namespace)
    live = await (await registry._reserve("c")).confirm()
    await live.require_current()

    await registry.unregister("c")
    with pytest.raises(PartitionDeletedError):
        await live.require_current()

    await (await registry._reserve("c")).confirm()
    with pytest.raises(PartitionDeletedError):
        await live.require_current()


@pytest.mark.asyncio
async def test_a_taken_name_is_already_exists(sqlalchemy_engine, partition_namespace):
    """Taken by a pending partition, then by a live one."""
    registry = await _registry(sqlalchemy_engine, partition_namespace)
    registration = await registry._reserve("c")

    for pending in (True, False):
        with pytest.raises(PartitionAlreadyExistsError):
            await registry._reserve("c")
        if pending:
            await registration.confirm()


@pytest.mark.asyncio
async def test_a_partition_name_of_any_characters_and_the_longest_length_is_kept(
    sqlalchemy_engine, partition_namespace
):
    registry = await _registry(sqlalchemy_engine, partition_namespace)
    registration = await (await registry._reserve(ANY_CHARACTERS_NAME)).confirm()

    resolved = await registry.resolve(ANY_CHARACTERS_NAME)
    assert resolved is not None
    assert resolved.partition_uuid == registration.partition_uuid
    await registry.unregister(ANY_CHARACTERS_NAME)
    assert await registry.resolve(ANY_CHARACTERS_NAME) is None


@pytest.mark.parametrize("invalid_name", ["", "with_nul\x00", "x" * 33, "ü" * 17])
def test_a_partition_namespace_must_fit_the_registry(invalid_name, tmp_path):
    with pytest.raises(ValueError, match="Partition namespace"):
        SQLAlchemyPartitionRegistryParams(
            engine=create_async_engine(
                f"sqlite+aiosqlite:///{tmp_path / 'registry.db'}"
            ),
            partition_namespace=invalid_name,
            partitioned_stores=[],
            tombstone_retention_seconds=RETENTION_SECONDS,
        )


def test_an_engine_of_another_dialect_is_refused(monkeypatch, tmp_path):
    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'registry.db'}")
    monkeypatch.setattr(engine.dialect, "name", "mssql")
    with pytest.raises(ValueError, match="mssql"):
        SQLAlchemyPartitionRegistryParams(
            engine=engine,
            partition_namespace="store",
            partitioned_stores=[],
            tombstone_retention_seconds=RETENTION_SECONDS,
        )


def test_a_sqlite_runtime_without_returning_is_refused(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "memmachine_server.common.partition_registry.sqlalchemy_partition_registry.sqlite3.sqlite_version_info",
        (3, 34, 1),
    )
    with pytest.raises(ValueError, match="RETURNING"):
        SQLAlchemyPartitionRegistryParams(
            engine=create_async_engine(
                f"sqlite+aiosqlite:///{tmp_path / 'registry.db'}"
            ),
            partition_namespace="store",
            partitioned_stores=[],
            tombstone_retention_seconds=RETENTION_SECONDS,
        )


def test_an_engine_sharing_one_connection_is_refused(tmp_path):
    """Every session would share one connection, so concurrent operations
    would run inside one another's transactions."""
    engine = create_async_engine(
        f"sqlite+aiosqlite:///{tmp_path / 'registry.db'}", poolclass=StaticPool
    )
    with pytest.raises(ValueError, match="StaticPool"):
        SQLAlchemyPartitionRegistryParams(
            engine=engine,
            partition_namespace="store",
            partitioned_stores=[],
            tombstone_retention_seconds=RETENTION_SECONDS,
        )


@pytest.mark.parametrize("url", ["sqlite+aiosqlite://", "sqlite+aiosqlite:///:memory:"])
def test_an_in_memory_sqlite_engine_is_refused(url):
    """Each connection to in-memory SQLite gets a separate database, so the
    registry's state would not be shared, even within one process."""
    with pytest.raises(ValueError, match="in-memory"):
        SQLAlchemyPartitionRegistryParams(
            engine=create_async_engine(url, poolclass=NullPool),
            partition_namespace="store",
            partitioned_stores=[],
            tombstone_retention_seconds=RETENTION_SECONDS,
        )


@pytest.mark.asyncio
async def test_registries_of_two_namespaces_share_a_database_and_nothing_else(
    sqlalchemy_engine, partition_namespace
):
    """The same name in two namespaces' registries is two partitions: each
    resolves, deletes and purges apart from the other, and a purge round
    claims only its own registry's tombstones."""
    first = await _registry(sqlalchemy_engine, f"{partition_namespace}_a")
    second = await _registry(sqlalchemy_engine, f"{partition_namespace}_b")

    one = await (await first._reserve("c")).confirm()
    two = await (await second._reserve("c")).confirm()
    assert one.partition_uuid != two.partition_uuid
    in_first = await first.resolve("c")
    in_second = await second.resolve("c")
    assert in_first is not None
    assert in_second is not None
    assert in_first.partition_uuid == one.partition_uuid
    assert in_second.partition_uuid == two.partition_uuid

    await first.unregister("c")
    await _age_deletion(first, one.partition_uuid)
    assert await first.resolve("c") is None
    still_in_second = await second.resolve("c")
    assert still_in_second is not None
    assert still_in_second.partition_uuid == two.partition_uuid
    await two.require_current()
    with pytest.raises(PartitionDeletedError):
        await one.require_current()
    with pytest.raises(PartitionAlreadyExistsError):
        await second._reserve("c")

    assert await _round(second, any_records_found=False) is None
    assert await _round(first, any_records_found=False) == one.partition_uuid
    assert await _round(first, any_records_found=False) is None
    await two.require_current()


@pytest.mark.asyncio
async def test_concurrent_creators_get_one_winner(
    sqlalchemy_engine, partition_namespace
):
    """Two registries on one database, the two-process shape in one process."""
    first = await _registry(sqlalchemy_engine, partition_namespace)
    second = await _registry(sqlalchemy_engine, partition_namespace)

    results = await asyncio.gather(
        *(registry._reserve("c") for registry in (first, second) * 3),
        return_exceptions=True,
    )

    winners = [r for r in results if isinstance(r, Reservation)]
    losers = [r for r in results if isinstance(r, PartitionAlreadyExistsError)]
    assert len(winners) == 1
    assert len(losers) == 5
    live = await winners[0].confirm()
    resolved = await first.resolve("c")
    assert resolved is not None
    assert resolved.partition_uuid == live.partition_uuid


@pytest.mark.asyncio
async def test_delete_queues_the_partition_uuid_and_is_idempotent(
    sqlalchemy_engine, partition_namespace
):
    registry = await _registry(sqlalchemy_engine, partition_namespace)
    partition_uuid = (await registry._reserve("c")).partition_uuid

    await registry.unregister("c")

    assert await registry.resolve("c") is None
    assert await _queued(registry) == [partition_uuid]
    await registry.unregister("c")
    await registry.unregister("never")
    assert await _queued(registry) == [partition_uuid]


@pytest.mark.asyncio
async def test_a_recreated_name_gets_a_new_partition_uuid(
    sqlalchemy_engine, partition_namespace
):
    registry = await _registry(sqlalchemy_engine, partition_namespace)
    old = (await registry._reserve("c")).partition_uuid
    await registry.unregister("c")

    new = (await (await registry._reserve("c")).confirm()).partition_uuid

    assert new != old
    resolved = await registry.resolve("c")
    assert resolved is not None
    assert resolved.partition_uuid == new


@pytest.mark.asyncio
async def test_a_round_is_given_where_the_records_are(
    sqlalchemy_engine, partition_namespace
):
    """The round is given the partition UUID the deleted partition's records carry."""
    registry = await _registry(sqlalchemy_engine, partition_namespace)
    partition_uuid = (await registry._reserve("c")).partition_uuid
    await registry.unregister("c")
    await _age_deletion(registry, partition_uuid)
    given: list[UUID] = []

    async def purge_round(claimed: UUID) -> bool:
        given.append(claimed)
        return False

    assert await registry._run_purge_round(purge_round)
    assert given == [partition_uuid]


@pytest.mark.asyncio
async def test_a_tombstone_is_not_due_before_the_retention(
    sqlalchemy_engine, partition_namespace
):
    registry = await _registry(sqlalchemy_engine, partition_namespace)
    partition_uuid = (await registry._reserve("a")).partition_uuid
    await registry.unregister("a")

    assert await _round(registry, any_records_found=False) is None
    assert await _queued(registry) == [partition_uuid]


@pytest.mark.asyncio
async def test_due_tombstones_go_oldest_deletion_first_until_a_round_finds_nothing(
    sqlalchemy_engine, partition_namespace, monkeypatch
):
    registry = await _registry(sqlalchemy_engine, partition_namespace)
    minted = iter([UUID(int=1), UUID(int=2)])
    monkeypatch.setattr(
        "memmachine_server.common.partition_registry.sqlalchemy_partition_registry.uuid4",
        lambda: next(minted),
    )
    first = (await registry._reserve("a")).partition_uuid
    second = (await registry._reserve("b")).partition_uuid
    await registry.unregister("a")
    await registry.unregister("b")
    # The later deletion, with the larger partition UUID, is made the older, so
    # only the ordering by deletion can put it first.
    await _age_deletion(registry, first)
    await _age_deletion(registry, second, extra=timedelta(minutes=1))

    assert await _round(registry, any_records_found=True) == second
    # Found points: still due, and still the oldest deletion.
    assert await _round(registry, any_records_found=False) == second
    assert await _queued(registry) == [first]
    assert await _round(registry, any_records_found=False) == first
    assert await _queued(registry) == []
    assert await _round(registry, any_records_found=False) is None


@pytest.mark.asyncio
async def test_a_tombstone_queued_by_a_real_deletion_comes_due_by_the_database_clock(
    sqlalchemy_engine, partition_namespace
):
    """The claim compares the stamp the database clock wrote, by the
    database's arithmetic: with no retention the first round claims the
    tombstone, and no stamp is rewritten by the test."""
    registry = await _registry(
        sqlalchemy_engine, partition_namespace, tombstone_retention_seconds=0
    )
    partition_uuid = (await registry._reserve("a")).partition_uuid
    await registry.unregister("a")

    assert await _round(registry, any_records_found=False) == partition_uuid

    assert await _queued(registry) == []


@pytest.mark.asyncio
async def test_a_changed_retention_applies_to_tombstones_already_queued(
    sqlalchemy_engine, partition_namespace
):
    """Only the deletion's time is stored; the retention in force when a
    claim is decided applies, including to tombstones queued under another."""
    queued_under_a_day = await _registry(sqlalchemy_engine, partition_namespace)
    partition_uuid = (await queued_under_a_day._reserve("a")).partition_uuid
    await queued_under_a_day.unregister("a")
    assert await _round(queued_under_a_day, any_records_found=False) is None

    no_retention = await _registry(
        sqlalchemy_engine, partition_namespace, tombstone_retention_seconds=0
    )
    assert await _round(no_retention, any_records_found=False) == partition_uuid
    assert await _queued(no_retention) == []


@pytest.mark.asyncio
async def test_a_round_that_raises_keeps_the_tombstone_and_names_it_in_the_error(
    sqlalchemy_engine, partition_namespace
):
    registry = await _registry(sqlalchemy_engine, partition_namespace)
    partition_uuid = (await registry._reserve("a")).partition_uuid
    await registry.unregister("a")
    await _age_deletion(registry, partition_uuid)

    async def refused(claimed: UUID) -> bool:
        assert claimed == partition_uuid
        raise RuntimeError("the backend refused")

    with pytest.raises(RuntimeError) as raised:
        await registry._run_purge_round(refused)

    assert [
        note
        for note in raised.value.__notes__
        if str(partition_uuid) in note
        and f"attempt 1 of {_MAX_PURGE_ATTEMPTS_WITHOUT_PROGRESS}" in note
    ]
    assert await _queued(registry) == [partition_uuid]
    assert await _attempts(registry, partition_uuid) == 1
    # Backing off: claimed again once 30 seconds have passed since the failure.
    assert await _round(registry, any_records_found=False) is None
    await _age_claimable_at(registry, partition_uuid, timedelta(seconds=35))
    assert await _round(registry, any_records_found=False) == partition_uuid


@pytest.mark.asyncio
async def test_a_failed_tombstone_backs_off_while_the_ones_behind_it_are_claimed(
    sqlalchemy_engine, partition_namespace
):
    registry = await _registry(sqlalchemy_engine, partition_namespace)
    failing = (await registry._reserve("a")).partition_uuid
    await registry.unregister("a")
    await _age_deletion(registry, failing, extra=timedelta(hours=1))
    later = (await registry._reserve("b")).partition_uuid
    await registry.unregister("b")
    await _age_deletion(registry, later)

    assert await _failing_round(registry) == failing
    assert await _round(registry, any_records_found=False) == later


@pytest.mark.asyncio
async def test_the_backoff_doubles_with_each_failure_and_runs_from_the_last_one(
    sqlalchemy_engine, partition_namespace
):
    registry = await _registry(sqlalchemy_engine, partition_namespace)
    partition_uuid = (await registry._reserve("a")).partition_uuid
    await registry.unregister("a")
    # Due for days: the backoff still runs from the last failure, not from
    # when the tombstone came due.
    await _age_deletion(registry, partition_uuid, extra=timedelta(days=3))

    await _failing_round(registry)
    await _age_claimable_at(registry, partition_uuid, timedelta(seconds=35))
    await _failing_round(registry)
    # Two failures: a minute from the second.
    await _age_claimable_at(registry, partition_uuid, timedelta(seconds=50))
    assert await _round(registry, any_records_found=True) is None
    await _age_claimable_at(registry, partition_uuid, timedelta(seconds=20))
    assert await _round(registry, any_records_found=True) == partition_uuid


@pytest.mark.asyncio
async def test_the_backoff_stops_doubling_at_its_maximum(
    sqlalchemy_engine, partition_namespace
):
    registry = SQLAlchemyPartitionRegistry(
        SQLAlchemyPartitionRegistryParams(
            engine=sqlalchemy_engine,
            partition_namespace=partition_namespace,
            partitioned_stores=[],
            tombstone_retention_seconds=RETENTION_SECONDS,
            base_purge_retry_backoff_seconds=30,
            max_purge_retry_backoff_seconds=120,
        )
    )
    await registry.startup()
    partition_uuid = (await registry._reserve("a")).partition_uuid
    await registry.unregister("a")
    await _age_deletion(registry, partition_uuid)

    for _ in range(5):
        await _age_claimable_at(registry, partition_uuid, timedelta(days=1))
        await _failing_round(registry)
    # Five failures: 30 seconds doubled four times is 8 minutes, capped at 2.
    await _age_claimable_at(registry, partition_uuid, timedelta(seconds=110))
    assert await _round(registry, any_records_found=True) is None
    await _age_claimable_at(registry, partition_uuid, timedelta(seconds=20))
    assert await _round(registry, any_records_found=True) == partition_uuid


@pytest.mark.asyncio
async def test_a_tombstone_whose_rounds_keep_failing_is_dead_lettered_and_reported(
    sqlalchemy_engine, partition_namespace, caplog
):
    registry = await _registry(sqlalchemy_engine, partition_namespace)
    failing = (await registry._reserve("a")).partition_uuid
    await registry.unregister("a")
    await _age_deletion(registry, failing, extra=timedelta(hours=1))
    later = (await registry._reserve("b")).partition_uuid
    await registry.unregister("b")
    await _age_deletion(registry, later)

    with caplog.at_level(logging.ERROR):
        for _ in range(_MAX_PURGE_ATTEMPTS_WITHOUT_PROGRESS):
            # Past its backoff, so it is the oldest tombstone claimable.
            await _age_claimable_at(registry, failing, timedelta(days=1))
            assert await _failing_round(registry) == failing

    assert [
        r
        for r in caplog.records
        if r.levelno == logging.ERROR and str(failing) in r.getMessage()
    ]
    # Skipped from now on, but kept in the queue.
    assert await _round(registry, any_records_found=False) == later
    assert await _round(registry, any_records_found=False) is None
    assert await _queued(registry) == [failing]


@pytest.mark.asyncio
async def test_resetting_a_dead_lettered_tombstones_attempts_returns_it_to_the_purge(
    sqlalchemy_engine, partition_namespace
):
    """Setting the attempts without progress back to 0 and the claimable time
    to NULL, as the dead-letter report says, makes the tombstone claimable at
    once, its last failure just now."""
    registry = await _registry(sqlalchemy_engine, partition_namespace)
    partition_uuid = (await registry._reserve("a")).partition_uuid
    await registry.unregister("a")
    await _age_deletion(registry, partition_uuid)
    for _ in range(_MAX_PURGE_ATTEMPTS_WITHOUT_PROGRESS):
        await _age_claimable_at(registry, partition_uuid, timedelta(days=1))
        assert await _failing_round(registry) == partition_uuid
    assert await _round(registry, any_records_found=False) is None

    async with registry._engine.begin() as connection:
        await connection.execute(
            update(PurgeQueueRow)
            .where(PurgeQueueRow.partition_uuid == partition_uuid)
            .values(attempts_without_progress=0, claimable_at=None)
        )

    assert await _round(registry, any_records_found=False) == partition_uuid


@pytest.mark.asyncio
async def test_a_round_that_finds_points_resets_the_attempts_without_progress(
    sqlalchemy_engine, partition_namespace
):
    registry = await _registry(sqlalchemy_engine, partition_namespace)
    partition_uuid = (await registry._reserve("a")).partition_uuid
    await registry.unregister("a")
    await _age_deletion(registry, partition_uuid)

    for _ in range(_MAX_PURGE_ATTEMPTS_WITHOUT_PROGRESS - 1):
        await _failing_round(registry)
        await _age_claimable_at(registry, partition_uuid, timedelta(days=1))
    assert await _round(registry, any_records_found=True) == partition_uuid
    assert await _attempts(registry, partition_uuid) == 0


@pytest.mark.asyncio
async def test_a_cancelled_round_ends_its_claim_without_counting_it(
    sqlalchemy_engine, partition_namespace, caplog
):
    """A cancelled round is not a failed round, and its tombstone is
    claimable at once, without waiting out the lease."""
    registry = await _registry(sqlalchemy_engine, partition_namespace)
    partition_uuid = (await registry._reserve("a")).partition_uuid
    await registry.unregister("a")
    await _age_deletion(registry, partition_uuid)

    _, holding = await _start_held_round(registry)
    holding.cancel()
    with pytest.raises(asyncio.CancelledError):
        await holding

    assert await _attempts(registry, partition_uuid) == 0
    with caplog.at_level(logging.WARNING):
        assert await _round(registry, any_records_found=False) == partition_uuid
    assert not [r for r in caplog.records if r.levelno >= logging.WARNING]


@pytest.mark.asyncio
async def test_a_round_that_never_ended_is_retried_once_its_lease_and_the_backoff_pass(
    sqlalchemy_engine, partition_namespace, monkeypatch, caplog
):
    """A purger that dies during its round writes nothing after its claim,
    which counted the attempt. The tombstone stays claimed until the lease
    passes and then backs off; the claim after that is the next attempt, and
    is logged as a retry."""
    registry = await _registry(sqlalchemy_engine, partition_namespace)
    partition_uuid = (await registry._reserve("a")).partition_uuid
    await registry.unregister("a")
    await _age_deletion(registry, partition_uuid)

    with caplog.at_level(logging.ERROR):
        assert await _abandon_claim(registry, monkeypatch) == partition_uuid
    assert [
        r
        for r in caplog.records
        if r.levelno == logging.ERROR and str(partition_uuid) in r.getMessage()
    ], "the lost end of the claim was not reported"
    assert await _attempts(registry, partition_uuid) == 1
    assert await _round(registry, any_records_found=False) is None

    # Past the lease, but not the 30-second backoff that runs from it.
    await _age_claimable_at(
        registry, partition_uuid, PURGE_LEASE + timedelta(seconds=20)
    )
    assert await _round(registry, any_records_found=False) is None

    await _age_claimable_at(registry, partition_uuid, timedelta(seconds=20))
    caplog.clear()
    with caplog.at_level(logging.WARNING):
        assert await _round(registry, any_records_found=False) == partition_uuid
    assert [
        r
        for r in caplog.records
        if r.levelno == logging.WARNING
        and str(partition_uuid) in r.getMessage()
        and f"attempt 2 of {_MAX_PURGE_ATTEMPTS_WITHOUT_PROGRESS}" in r.getMessage()
    ]
    assert await _queued(registry) == []


@pytest.mark.asyncio
async def test_rounds_that_never_end_dead_letter_the_tombstone(
    sqlalchemy_engine, partition_namespace, monkeypatch, caplog
):
    """A tombstone whose rounds keep killing their purger is dead-lettered
    like one whose rounds keep raising, each retry logged with its attempt."""
    registry = await _registry(sqlalchemy_engine, partition_namespace)
    partition_uuid = (await registry._reserve("a")).partition_uuid
    await registry.unregister("a")
    await _age_deletion(registry, partition_uuid)

    with caplog.at_level(logging.WARNING):
        for _ in range(_MAX_PURGE_ATTEMPTS_WITHOUT_PROGRESS):
            assert await _abandon_claim(registry, monkeypatch) == partition_uuid
            await _age_claimable_at(registry, partition_uuid, timedelta(days=1))

    assert (
        await _attempts(registry, partition_uuid)
        == _MAX_PURGE_ATTEMPTS_WITHOUT_PROGRESS
    )
    assert [
        r
        for r in caplog.records
        if r.levelno == logging.WARNING
        and str(partition_uuid) in r.getMessage()
        and f"attempt {_MAX_PURGE_ATTEMPTS_WITHOUT_PROGRESS} of {_MAX_PURGE_ATTEMPTS_WITHOUT_PROGRESS}"
        in r.getMessage()
    ]
    assert await _round(registry, any_records_found=False) is None
    assert await _queued(registry) == [partition_uuid]


@pytest.mark.asyncio
async def test_a_cancelled_retry_of_a_round_that_never_ended_leaves_it_claimable(
    sqlalchemy_engine, partition_namespace, monkeypatch
):
    """A cancelled round takes back its attempt even when the attempt before
    it never ended and no round ever raised, so the tombstone is claimable at
    once."""
    registry = await _registry(sqlalchemy_engine, partition_namespace)
    partition_uuid = (await registry._reserve("a")).partition_uuid
    await registry.unregister("a")
    await _age_deletion(registry, partition_uuid)

    assert await _abandon_claim(registry, monkeypatch) == partition_uuid
    await _age_claimable_at(registry, partition_uuid, timedelta(days=1))
    _, holding = await _start_held_round(registry)
    holding.cancel()
    with pytest.raises(asyncio.CancelledError):
        await holding

    assert await _attempts(registry, partition_uuid) == 1
    assert await _round(registry, any_records_found=False) == partition_uuid


@pytest.mark.asyncio
async def test_racing_purgers_retry_a_round_that_never_ended_once(
    sqlalchemy_engine, partition_namespace, monkeypatch
):
    """Purgers on separate engines that find the same claim whose round never
    ended claim it once between them, as its next attempt."""
    engines = [create_async_engine(sqlalchemy_engine.url) for _ in range(4)]
    for engine in engines:
        enable_sqlite_explicit_begin(engine)
    ran: list[UUID] = []
    try:
        purgers = [await _registry(engine, partition_namespace) for engine in engines]
        partition_uuid = (await purgers[0]._reserve("a")).partition_uuid
        await purgers[0].unregister("a")
        await _age_deletion(purgers[0], partition_uuid)
        await _abandon_claim(purgers[0], monkeypatch)
        await _age_claimable_at(purgers[0], partition_uuid)

        # Finding nothing removes the tombstone, so a second claim would show.
        async def found_nothing(claimed: UUID) -> bool:
            ran.append(claimed)
            return False

        await asyncio.wait_for(
            asyncio.gather(
                *(purger._run_purge_round(found_nothing) for purger in purgers)
            ),
            30,
        )
    finally:
        for engine in engines:
            await engine.dispose()

    assert ran == [partition_uuid]


@pytest.mark.asyncio
async def test_a_claimed_tombstone_is_skipped_while_its_round_runs(
    sqlalchemy_engine, partition_namespace
):
    """A second purger takes the next due tombstone, and finds none once the
    rest are taken; the claimed one is not claimed again until its round ends."""
    registry = await _registry(sqlalchemy_engine, partition_namespace)
    older = (await registry._reserve("older")).partition_uuid
    newer = (await registry._reserve("newer")).partition_uuid
    await registry.unregister("older")
    await registry.unregister("newer")
    await _age_deletion(registry, older, extra=timedelta(minutes=1))
    await _age_deletion(registry, newer)

    held, holding = await _start_held_round(registry)
    try:
        assert held.ran == [older]
        assert await _round(registry, any_records_found=False) == newer
        assert await _round(registry, any_records_found=False) is None
        held.end(any_records_found=True)
        assert await asyncio.wait_for(holding, 30)
    finally:
        if not holding.done():
            holding.cancel()

    # Ended with records found: the claim is over, and the tombstone is due.
    assert await _round(registry, any_records_found=False) == older
    assert await _queued(registry) == []


@pytest.mark.asyncio
@pytest.mark.parametrize("stale_outcome", ["records found", "raised"])
async def test_a_round_that_outlasted_its_lease_leaves_the_claim_after_it(
    sqlalchemy_engine, partition_namespace, caplog, stale_outcome
):
    """Once a round's lease and the backoff have passed, the tombstone is
    claimed again as the next attempt. The first round's end then changes
    nothing: the later claim holds, and each round counts once."""
    registry = await _registry(sqlalchemy_engine, partition_namespace)
    partition_uuid = (await registry._reserve("a")).partition_uuid
    await registry.unregister("a")
    await _age_deletion(registry, partition_uuid)

    stale, stale_task = await _start_held_round(registry)
    current_task = None
    try:
        await _age_claimable_at(registry, partition_uuid)
        current, current_task = await _start_held_round(registry)
        assert current.ran == [partition_uuid]

        caplog.clear()
        with caplog.at_level(logging.WARNING):
            if stale_outcome == "raised":
                stale.fail(RuntimeError("the backend refused"))
                with pytest.raises(RuntimeError):
                    await asyncio.wait_for(stale_task, 30)
            else:
                stale.end(any_records_found=True)
                assert await asyncio.wait_for(stale_task, 30)
        assert [
            r
            for r in caplog.records
            if r.levelno == logging.WARNING and str(partition_uuid) in r.getMessage()
        ]
        assert await _attempts(registry, partition_uuid) == 2
        assert await _round(registry, any_records_found=False) is None

        current.end(any_records_found=True)
        assert await asyncio.wait_for(current_task, 30)
    finally:
        for task in (stale_task, current_task):
            if task is not None and not task.done():
                task.cancel()

    assert await _round(registry, any_records_found=False) == partition_uuid


@pytest.mark.asyncio
async def test_a_round_that_finds_nothing_removes_the_tombstone_under_any_claim(
    sqlalchemy_engine, partition_namespace
):
    """A partition UUID found empty after the retention stays empty, so a round
    that outlasted its lease still removes the tombstone it found empty."""
    registry = await _registry(sqlalchemy_engine, partition_namespace)
    partition_uuid = (await registry._reserve("a")).partition_uuid
    await registry.unregister("a")
    await _age_deletion(registry, partition_uuid)

    stale, stale_task = await _start_held_round(registry)
    current_task = None
    try:
        await _age_claimable_at(registry, partition_uuid)
        current, current_task = await _start_held_round(registry)
        stale.end(any_records_found=False)
        assert await asyncio.wait_for(stale_task, 30)
        assert await _queued(registry) == []

        current.end(any_records_found=True)
        assert await asyncio.wait_for(current_task, 30)
    finally:
        for task in (stale_task, current_task):
            if task is not None and not task.done():
                task.cancel()

    assert await _queued(registry) == []


@pytest.mark.asyncio
async def test_concurrent_purgers_claim_each_tombstone_once(
    sqlalchemy_engine, partition_namespace
):
    """Purgers on separate engines, as in separate processes, split a backlog
    between them: every tombstone is claimed once, whatever the interleaving."""
    engines = [create_async_engine(sqlalchemy_engine.url) for _ in range(3)]
    for engine in engines:
        enable_sqlite_explicit_begin(engine)
    try:
        purgers = [
            await _registry(engine, partition_namespace, tombstone_retention_seconds=0)
            for engine in engines
        ]
        partition_uuids = []
        for index in range(12):
            name = f"c{index}"
            partition_uuids.append((await purgers[0]._reserve(name)).partition_uuid)
            await purgers[0].unregister(name)
        ran: list[UUID] = []

        async def purge_round(partition_uuid: UUID) -> bool:
            ran.append(partition_uuid)
            await asyncio.sleep(0.01)
            return False

        async def drain(purger: SQLAlchemyPartitionRegistry) -> None:
            while await purger._run_purge_round(purge_round):
                pass

        await asyncio.wait_for(asyncio.gather(*(drain(p) for p in purgers)), 60)
    finally:
        for engine in engines:
            await engine.dispose()

    assert sorted(ran) == sorted(partition_uuids)


@pytest.mark.asyncio
async def test_a_partition_uuid_awaiting_purge_is_never_reminted(
    sqlalchemy_engine, partition_namespace, monkeypatch
):
    """A minted partition UUID colliding with a queued tombstone is rejected and re-minted."""
    registry = await _registry(sqlalchemy_engine, partition_namespace)
    dead = (await registry._reserve("a")).partition_uuid
    await registry.unregister("a")
    minted = iter([dead, uuid4()])
    monkeypatch.setattr(
        "memmachine_server.common.partition_registry.sqlalchemy_partition_registry.uuid4",
        lambda: next(minted),
    )

    fresh = (await registry._reserve("b")).partition_uuid

    assert fresh != dead
    assert await _queued(registry) == [dead]


@pytest.mark.asyncio
async def test_creation_that_never_mints_a_free_partition_uuid_gives_up(
    sqlalchemy_engine, partition_namespace, monkeypatch
):
    registry = await _registry(sqlalchemy_engine, partition_namespace)
    dead = (await registry._reserve("a")).partition_uuid
    await registry.unregister("a")
    monkeypatch.setattr(
        "memmachine_server.common.partition_registry.sqlalchemy_partition_registry.uuid4",
        lambda: dead,
    )

    with pytest.raises(PartitionCreateAttemptsExhaustedError):
        await registry._reserve("b")
    assert await registry.resolve("b") is None


@asynccontextmanager
async def _engine_ten_days_behind(
    engine: AsyncEngine,
) -> AsyncGenerator[AsyncEngine, None]:
    """An engine on the same PostgreSQL database whose now() runs ten days behind the real clock."""
    schema = f"skewed_{uuid4().hex[:12]}"
    async with engine.begin() as connection:
        await connection.execute(text(f"CREATE SCHEMA {schema}"))
        await connection.execute(
            text(
                f"CREATE FUNCTION {schema}.now() RETURNS timestamptz LANGUAGE sql "
                "STABLE AS $$ SELECT pg_catalog.now() - interval '10 days' $$"
            )
        )
    skewed_engine = create_async_engine(
        engine.url,
        connect_args={
            "server_settings": {"search_path": f"{schema}, public, pg_catalog"}
        },
    )
    try:
        yield skewed_engine
    finally:
        await skewed_engine.dispose()
        async with engine.begin() as connection:
            await connection.execute(text(f"DROP SCHEMA {schema} CASCADE"))


@pytest.mark.asyncio
async def test_purge_rounds_keep_time_by_the_database_clock(
    sqlalchemy_engine, partition_namespace
):
    """Every time the queue stores, and the retention's cutoff, come from the
    database's clock, never the client's.

    The database's now() is shadowed by one running ten days behind the
    real clock: the tombstone's deletion is stamped by it, and a retention
    of a day, long past by the real clock, has not passed by it.
    """
    if sqlalchemy_engine.dialect.name != "postgresql":
        pytest.skip("SQLite's clock cannot be shadowed")
    async with _engine_ten_days_behind(sqlalchemy_engine) as skewed_engine:
        registry = await _registry(skewed_engine, partition_namespace)
        partition_uuid = (await registry._reserve("a")).partition_uuid
        await registry.unregister("a")

        async with skewed_engine.connect() as connection:
            row = (
                await connection.execute(
                    select(
                        PurgeQueueRow.enqueued_at, text("pg_catalog.now() AS real_now")
                    ).where(PurgeQueueRow.partition_uuid == partition_uuid)
                )
            ).one()
        assert row.real_now - row.enqueued_at > timedelta(days=9)
        assert await _round(registry, any_records_found=False) is None
        assert await _queued(registry) == [partition_uuid]


@pytest.mark.asyncio
async def test_a_purge_lease_runs_by_the_database_clock(
    sqlalchemy_engine, partition_namespace
):
    """A claim's lease is scheduled, and measured, by the database's clock.

    By a clock ten days behind the real one, a claim taken just now still
    holds, though by the real clock its lease ended ten days ago.
    """
    if sqlalchemy_engine.dialect.name != "postgresql":
        pytest.skip("SQLite's clock cannot be shadowed")
    async with _engine_ten_days_behind(sqlalchemy_engine) as skewed_engine:
        registry = await _registry(
            skewed_engine, partition_namespace, tombstone_retention_seconds=0
        )
        partition_uuid = (await registry._reserve("a")).partition_uuid
        await registry.unregister("a")

        held, holding = await _start_held_round(registry)
        try:
            async with skewed_engine.connect() as connection:
                row = (
                    await connection.execute(
                        select(
                            PurgeQueueRow.claimable_at,
                            text("pg_catalog.now() AS real_now"),
                        ).where(PurgeQueueRow.partition_uuid == partition_uuid)
                    )
                ).one()
            assert row.real_now - row.claimable_at > timedelta(days=9)
            assert await _round(registry, any_records_found=False) is None
            held.end(any_records_found=False)
            assert await asyncio.wait_for(holding, 30)
        finally:
            if not holding.done():
                holding.cancel()


@pytest.mark.asyncio
async def test_no_transaction_is_open_while_a_round_runs(
    sqlalchemy_engine, partition_namespace
):
    """The claim commits before the round, so no backend sits idle in a
    transaction, holding a row lock or vacuum's horizon, during its remote calls."""
    if sqlalchemy_engine.dialect.name != "postgresql":
        pytest.skip("pg_stat_activity is PostgreSQL's")
    registry = await _registry(
        sqlalchemy_engine, partition_namespace, tombstone_retention_seconds=0
    )
    await registry._reserve("a")
    await registry.unregister("a")

    held, holding = await _start_held_round(registry)
    try:
        async with sqlalchemy_engine.connect() as connection:
            idle_in_transaction = (
                await connection.execute(
                    text(
                        "SELECT count(*) FROM pg_stat_activity "
                        "WHERE state LIKE 'idle in transaction%' "
                        "AND datname = current_database() "
                        "AND pid != pg_backend_pid()"
                    )
                )
            ).scalar_one()
        assert idle_in_transaction == 0
        held.end(any_records_found=False)
        assert await asyncio.wait_for(holding, 30)
    finally:
        if not holding.done():
            holding.cancel()


@pytest.mark.asyncio
async def test_a_claim_skips_a_tombstone_another_purger_holds(
    sqlalchemy_engine, partition_namespace
):
    """A purger never waits on a claim another purger is taking: it takes the next due tombstone."""
    if sqlalchemy_engine.dialect.name != "postgresql":
        pytest.skip("SQLite holds no row locks to skip")
    registry = await _registry(sqlalchemy_engine, partition_namespace)
    held = (await registry._reserve("held")).partition_uuid
    free = (await registry._reserve("free")).partition_uuid
    await registry.unregister("held")
    await registry.unregister("free")
    await _age_deletion(registry, held, extra=timedelta(minutes=1))
    await _age_deletion(registry, free)

    claim = None
    try:
        async with sqlalchemy_engine.connect() as other_purger, other_purger.begin():
            await other_purger.execute(
                select(PurgeQueueRow.partition_uuid)
                .where(PurgeQueueRow.partition_uuid == held)
                .with_for_update()
            )
            claim = asyncio.create_task(_round(registry, any_records_found=False))
            outcome = await _blocked_or_done(sqlalchemy_engine, claim)
            assert outcome == "done", (
                "the claim waited on a tombstone another purger holds"
            )
            assert await claim == free
    finally:
        if claim is not None and not claim.done():
            await asyncio.wait_for(claim, 30)


@pytest.mark.asyncio
async def test_sqlite_stays_writable_while_a_round_runs(tmp_path, partition_namespace):
    """On SQLite the claim's write commits before the round, so while the
    round runs, another connection writes at once, under a busy timeout far
    shorter than the round, and another purger claims the next tombstone."""
    url = f"sqlite+aiosqlite:///{tmp_path / 'registry.db'}"
    purger_engine = create_async_engine(url)
    impatient_engine = create_async_engine(url, connect_args={"timeout": 0.1})
    enable_sqlite_explicit_begin(purger_engine)
    enable_sqlite_explicit_begin(impatient_engine)
    try:
        purger = await _registry(purger_engine, partition_namespace)
        impatient = await _registry(impatient_engine, partition_namespace)
        older = (await purger._reserve("older")).partition_uuid
        newer = (await purger._reserve("newer")).partition_uuid
        await purger.unregister("older")
        await purger.unregister("newer")
        await _age_deletion(purger, older, extra=timedelta(minutes=1))
        await _age_deletion(purger, newer)

        held, holding = await _start_held_round(purger)
        try:
            assert held.ran == [older]
            await (await impatient._reserve("c")).confirm()
            await impatient.unregister("c")
            assert await _round(impatient, any_records_found=False) == newer
            held.end(any_records_found=False)
            assert await asyncio.wait_for(holding, 30)
        finally:
            if not holding.done():
                holding.cancel()
        assert len(await _queued(purger)) == 1
    finally:
        await purger_engine.dispose()
        await impatient_engine.dispose()


@pytest.mark.asyncio
async def test_racing_deletions_of_a_partition_queue_one_tombstone(
    sqlalchemy_engine, partition_namespace
):
    """Racing deletions serialize on the partition's row: the losers delete
    nothing, one tombstone is queued, and the name can be created again."""
    registries = [
        await _registry(sqlalchemy_engine, partition_namespace) for _ in range(4)
    ]
    for cycle in range(10):
        await registries[0]._reserve("c")
        await asyncio.wait_for(
            asyncio.gather(*(r.unregister("c") for r in registries)), 30
        )
        assert await registries[0].resolve("c") is None
        assert len(await _queued(registries[0])) == cycle + 1
    await registries[0]._reserve("c")


@pytest.mark.asyncio
async def test_a_deletion_racing_another_waits_and_queues_nothing_more(
    sqlalchemy_engine, partition_namespace
):
    """A deletion that finds another process's deletion in flight waits for
    it, then finds nothing to delete: one tombstone, and no error."""
    if sqlalchemy_engine.dialect.name != "postgresql":
        pytest.skip("SQLite serializes whole write transactions")
    registry = await _registry(sqlalchemy_engine, partition_namespace)
    partition_uuid = (await registry._reserve("c")).partition_uuid

    local = None
    try:
        async with sqlalchemy_engine.connect() as remote, remote.begin():
            # Another process's deletion, held uncommitted.
            await remote.execute(
                delete(PartitionRow).where(
                    PartitionRow.partition_uuid == partition_uuid
                )
            )
            await remote.execute(
                insert(PurgeQueueRow).values(
                    partition_uuid=partition_uuid,
                    partition_namespace=partition_namespace,
                    enqueued_at=func.now(),
                )
            )
            local = asyncio.create_task(registry.unregister("c"))
            assert await _blocked_or_done(sqlalchemy_engine, local) == "blocked"
        await asyncio.wait_for(local, 30)
    finally:
        if local is not None and not local.done():
            local.cancel()

    assert await _queued(registry) == [partition_uuid]


@pytest.mark.asyncio
async def test_a_partition_uuid_colliding_with_a_live_partitions_is_reminted(
    sqlalchemy_engine, partition_namespace, monkeypatch
):
    """A mint rejected by the partition's unique constraint, with the name
    free, is a collision to mint again, not a taken name."""
    registry = await _registry(sqlalchemy_engine, partition_namespace)
    live = (await (await registry._reserve("live")).confirm()).partition_uuid
    minted = iter([live, uuid4()])
    monkeypatch.setattr(
        "memmachine_server.common.partition_registry.sqlalchemy_partition_registry.uuid4",
        lambda: next(minted),
    )

    fresh = (await registry._reserve("fresh")).partition_uuid

    assert fresh != live
    resolved = await registry.resolve("live")
    assert resolved is not None
    assert resolved.partition_uuid == live


@pytest.mark.asyncio
async def test_a_mint_checks_the_queue_after_its_insert(
    sqlalchemy_engine, partition_namespace, monkeypatch
):
    """A mint colliding with a deletion in flight sees the deletion's tombstone.

    The deletion frees the partition's row and queues its tombstone in one
    uncommitted transaction; the colliding mint waits on it, at its insert,
    or on SQLite, where its transaction takes the write lock, at BEGIN. Only
    a queue check made after the insert sees the tombstone once the deletion
    commits: one made before reads the queue too early and registers a
    partition under a partition UUID awaiting purge.
    """
    registry = await _registry(sqlalchemy_engine, partition_namespace)
    victim = (await registry._reserve("victim")).partition_uuid
    minted = iter([victim, uuid4()])
    monkeypatch.setattr(
        "memmachine_server.common.partition_registry.sqlalchemy_partition_registry.uuid4",
        lambda: next(minted),
    )
    mint_waiting = asyncio.Event()

    def on_statement(_connection, _cursor, statement, _parameters, _context, _many):
        if statement == "BEGIN IMMEDIATE" or statement.startswith(
            f"INSERT INTO {PartitionRow.__tablename__}"
        ):
            mint_waiting.set()

    event.listen(sqlalchemy_engine.sync_engine, "before_cursor_execute", on_statement)
    creator = None
    try:
        async with sqlalchemy_engine.connect() as remote, remote.begin():
            await remote.execute(
                delete(PartitionRow).where(PartitionRow.partition_uuid == victim)
            )
            await remote.execute(
                insert(PurgeQueueRow).values(
                    partition_uuid=victim,
                    partition_namespace=partition_namespace,
                    enqueued_at=func.now(),
                )
            )
            creator = asyncio.create_task(registry._reserve("fresh"))
            await asyncio.wait_for(mint_waiting.wait(), 30)
        fresh = (await asyncio.wait_for(creator, 30)).partition_uuid
    finally:
        event.remove(
            sqlalchemy_engine.sync_engine, "before_cursor_execute", on_statement
        )
        if creator is not None and not creator.done():
            creator.cancel()

    assert fresh != victim, "a live partition took a partition UUID awaiting purge"
    assert await _queued(registry) == [victim]


@pytest.mark.asyncio
async def test_a_round_claims_one_tombstone(sqlalchemy_engine, partition_namespace):
    """A claim takes one tombstone and leaves the others to other purgers."""
    registry = await _registry(
        sqlalchemy_engine, partition_namespace, tombstone_retention_seconds=0
    )
    for name in ("a", "b", "c"):
        await registry._reserve(name)
        await registry.unregister(name)

    assert await _round(registry, any_records_found=False) is not None

    assert len(await _queued(registry)) == 2


@pytest.mark.asyncio
async def test_a_persistent_database_error_surfaces_with_its_cause(
    sqlalchemy_engine, partition_namespace, monkeypatch
):
    """A rejected insert is retried a bounded number of times, then the
    database's own error is chained to the one that gives up."""
    registry = await _registry(sqlalchemy_engine, partition_namespace)
    # A NOT NULL violation stands in for a cause that is not a collision.
    monkeypatch.setattr(
        "memmachine_server.common.partition_registry.sqlalchemy_partition_registry.uuid4",
        lambda: None,
    )

    with pytest.raises(PartitionCreateAttemptsExhaustedError) as raised:
        await asyncio.wait_for(registry._reserve("a"), 30)

    cause: BaseException | None = raised.value
    while cause is not None and not isinstance(cause, IntegrityError):
        cause = cause.__cause__
    assert isinstance(cause, IntegrityError)


@pytest.mark.asyncio
async def test_a_claim_holds_for_the_lease_it_was_taken_under(
    sqlalchemy_engine, partition_namespace
):
    """The lease is scheduled when a claim is taken: a claim taken under a
    five-minute lease holds against a one-minute lease past a minute and a
    half, until its own five minutes and the 30-second backoff pass."""
    five_minutes = await _registry(sqlalchemy_engine, partition_namespace)
    one_minute = await _registry(
        sqlalchemy_engine, partition_namespace, purge_lease_seconds=60
    )
    partition_uuid = (await five_minutes._reserve("a")).partition_uuid
    await five_minutes.unregister("a")
    await _age_deletion(five_minutes, partition_uuid)

    held, holding = await _start_held_round(five_minutes)
    try:
        await _age_claimable_at(five_minutes, partition_uuid, timedelta(seconds=100))
        assert not await one_minute._run_purge_round(_no_round)
        await _age_claimable_at(five_minutes, partition_uuid, PURGE_LEASE)
        assert await _round(one_minute, any_records_found=True) == partition_uuid
        held.end(any_records_found=True)
        assert await asyncio.wait_for(holding, 30)
    finally:
        if not holding.done():
            holding.cancel()


@pytest.mark.asyncio
async def test_the_first_purge_backoff_comes_from_its_parameter(
    sqlalchemy_engine, partition_namespace
):
    """With a ten-second first backoff, a failed tombstone is claimed again
    after ten seconds, and after twenty once it has failed twice."""
    registry = await _registry(
        sqlalchemy_engine, partition_namespace, base_purge_retry_backoff_seconds=10
    )
    partition_uuid = (await registry._reserve("a")).partition_uuid
    await registry.unregister("a")
    await _age_deletion(registry, partition_uuid)

    assert await _failing_round(registry) == partition_uuid
    await _age_claimable_at(registry, partition_uuid, timedelta(seconds=5))
    assert await _round(registry, any_records_found=True) is None
    await _age_claimable_at(registry, partition_uuid, timedelta(seconds=10))
    assert await _failing_round(registry) == partition_uuid
    await _age_claimable_at(registry, partition_uuid, timedelta(seconds=15))
    assert await _round(registry, any_records_found=True) is None
    await _age_claimable_at(registry, partition_uuid, timedelta(seconds=10))
    assert await _round(registry, any_records_found=True) == partition_uuid


@pytest.mark.asyncio
async def test_a_deletion_whose_tombstone_cannot_be_queued_changes_nothing(
    sqlalchemy_engine, partition_namespace
):
    """A deletion and its tombstone commit together: when queuing the
    tombstone fails, the partition stays registered and nothing is queued,
    so no deleted partition UUID is ever left without a tombstone."""
    registry = await _registry(sqlalchemy_engine, partition_namespace)
    live = await (await registry._reserve("c")).confirm()

    with (
        _failing_statement(
            sqlalchemy_engine, f"INSERT INTO {PurgeQueueRow.__tablename__}"
        ),
        pytest.raises(_InjectedFailureError),
    ):
        await registry.unregister("c")

    resolved = await registry.resolve("c")
    assert resolved is not None
    assert resolved.partition_uuid == live.partition_uuid
    await live.require_current()
    assert await _queued(registry) == []

    await registry.unregister("c")
    assert await registry.resolve("c") is None
    assert await _queued(registry) == [live.partition_uuid]


@pytest.mark.asyncio
async def test_a_reservation_whose_queue_check_fails_leaves_the_name_free(
    sqlalchemy_engine, partition_namespace
):
    """A reservation and its check of the purge queue commit together: when
    the check fails, no pending partition holds the name."""
    registry = await _registry(sqlalchemy_engine, partition_namespace)

    with (
        _failing_statement(sqlalchemy_engine, f"SELECT {PurgeQueueRow.__tablename__}"),
        pytest.raises(_InjectedFailureError),
    ):
        await registry._reserve("c")

    assert await registry.resolve("c") is None
    await (await registry._reserve("c")).confirm()


@pytest.mark.asyncio
async def test_a_round_whose_outcome_cannot_be_written_is_retried_once_its_lease_passes(
    sqlalchemy_engine, partition_namespace
):
    """A round whose outcome is lost is, to the registry, a round that never
    ended: its error propagates, the claim holds until the lease and the
    backoff pass, and the tombstone is then claimed as the next attempt."""
    registry = await _registry(sqlalchemy_engine, partition_namespace)
    partition_uuid = (await registry._reserve("a")).partition_uuid
    await registry.unregister("a")
    await _age_deletion(registry, partition_uuid)
    ran = False

    async def found_records(claimed: UUID) -> bool:
        nonlocal ran
        ran = True
        return True

    with (
        _failing_statement(
            sqlalchemy_engine, f"UPDATE {PurgeQueueRow.__tablename__}", lambda: ran
        ),
        pytest.raises(_InjectedFailureError),
    ):
        await registry._run_purge_round(found_records)

    assert await _attempts(registry, partition_uuid) == 1
    assert await _round(registry, any_records_found=False) is None
    await _age_claimable_at(registry, partition_uuid)
    assert await _round(registry, any_records_found=True) == partition_uuid
    assert await _attempts(registry, partition_uuid) == 0


@pytest.mark.asyncio
async def test_a_reservation_colliding_with_a_tombstone_under_purge_does_not_wait_for_the_round(
    sqlalchemy_engine, partition_namespace, monkeypatch
):
    """A purge round holds no lock while it runs, so a reservation whose
    minted partition UUID collides with the tombstone being purged finds the
    collision and mints again at once, on PostgreSQL and SQLite alike."""
    registry = await _registry(sqlalchemy_engine, partition_namespace)
    dead = (await registry._reserve("dead")).partition_uuid
    await registry.unregister("dead")
    await _age_deletion(registry, dead)

    held, holding = await _start_held_round(registry)
    try:
        assert held.ran == [dead]
        minted = iter([dead, uuid4()])
        monkeypatch.setattr(
            "memmachine_server.common.partition_registry.sqlalchemy_partition_registry.uuid4",
            lambda: next(minted),
        )
        fresh = await asyncio.wait_for(registry._reserve("b"), 10)
        assert fresh.partition_uuid != dead
        held.end(any_records_found=False)
        assert await asyncio.wait_for(holding, 30)
    finally:
        if not holding.done():
            holding.cancel()
    assert await _queued(registry) == []


class _OverlapWatch:
    """A purge round that records any round started on a tombstone whose round is still running."""

    def __init__(self) -> None:
        self.overlapping: list[UUID] = []
        self._in_flight: set[UUID] = set()

    async def __call__(self, partition_uuid: UUID) -> bool:
        if partition_uuid in self._in_flight:
            self.overlapping.append(partition_uuid)
        self._in_flight.add(partition_uuid)
        try:
            await asyncio.sleep(0.002)
            return partition_uuid.int % 2 == 0
        finally:
            self._in_flight.discard(partition_uuid)


_REFUSALS = (
    PartitionAlreadyExistsError,
    PartitionPendingError,
    PartitionDeletedError,
    PartitionDeletedError,
)


async def _churn(
    registry: SQLAlchemyPartitionRegistry,
    seed: int,
    purge_round: _OverlapWatch,
) -> None:
    """A seeded run of creations, cancellations, deletions, reads and purge rounds over a few names."""
    rng = random.Random(seed)
    for _ in range(40):
        name = rng.choice(["a", "b", "c"])
        choice = rng.random()
        try:
            if choice < 0.35:
                reservation = await registry._reserve(name)
                if rng.random() < 0.8:
                    await (await reservation.confirm()).require_current()
                else:
                    await reservation.cancel()
            elif choice < 0.6:
                await registry.unregister(name)
            elif choice < 0.8:
                registration = await registry.resolve(name)
                if registration is not None:
                    await registration.require_current()
            else:
                await registry._run_purge_round(purge_round)
        except _REFUSALS:
            pass


async def _found_nothing(partition_uuid: UUID) -> bool:
    return False


@pytest.mark.asyncio
async def test_mixed_operations_across_engines_finish_cleanly_and_strand_nothing(
    sqlalchemy_engine, partition_namespace
):
    """Creators, deleters, readers and purgers on separate engines, as in
    separate processes, race over a few shared names. Every operation ends in
    its result or a documented refusal, never a database error or a deadlock;
    no tombstone's rounds overlap; and once they stop, no partition UUID is both
    registered and queued, and the purge drains every tombstone."""
    engines = [create_async_engine(sqlalchemy_engine.url) for _ in range(4)]
    for engine in engines:
        enable_sqlite_explicit_begin(engine)
    watch = _OverlapWatch()
    try:
        registries = [
            await _registry(engine, partition_namespace, tombstone_retention_seconds=0)
            for engine in engines
        ]
        await asyncio.wait_for(
            asyncio.gather(
                *(
                    _churn(registry, seed, watch)
                    for seed, registry in enumerate(registries)
                )
            ),
            120,
        )
        assert watch.overlapping == []

        async with engines[0].connect() as connection:
            registered = set(
                (
                    await connection.execute(
                        select(PartitionRow.partition_uuid).where(
                            PartitionRow.partition_namespace == partition_namespace
                        )
                    )
                ).scalars()
            )
        assert not registered & set(await _queued(registries[0]))

        for _ in range(1000):
            if not await registries[0]._run_purge_round(_found_nothing):
                break
        assert await _queued(registries[0]) == []
    finally:
        for engine in engines:
            await engine.dispose()


class _ModelCheckedRegistry:
    """A registry driven step by step beside a model of its contract.

    The model holds which partition holds each name and whether it is live,
    which partition UUIDs are queued and whether each is backing off, and the
    reservations and registrations handed out. Each step performs one
    operation, checks its outcome against the model, and updates the model.
    """

    def __init__(
        self,
        registry: SQLAlchemyPartitionRegistry,
        rng: random.Random,
    ) -> None:
        self.registry = registry
        self.rng = rng
        self.partitions: dict[str, tuple[UUID, bool]] = {}
        self.backing_off: dict[UUID, bool] = {}
        self.reservations: list[Reservation] = []
        self.confirmed: set[UUID] = set()
        self.registrations: list[Registration] = []

    async def reserve(self, name: str) -> None:
        if name in self.partitions:
            with pytest.raises(PartitionAlreadyExistsError):
                await self.registry._reserve(name)
            return
        reservation = await self.registry._reserve(name)
        assert reservation.partition_uuid not in self.backing_off
        self.partitions[name] = (reservation.partition_uuid, False)
        self.reservations.append(reservation)

    async def confirm(self, name: str) -> None:
        unconfirmed = [
            r for r in self.reservations if r.partition_uuid not in self.confirmed
        ]
        if not unconfirmed:
            return
        reservation = self.rng.choice(unconfirmed)
        self.confirmed.add(reservation.partition_uuid)
        if self.partitions.get(reservation.partition_name) != (
            reservation.partition_uuid,
            False,
        ):
            with pytest.raises(PartitionDeletedError):
                await reservation.confirm()
            return
        self.registrations.append(await reservation.confirm())
        self.partitions[reservation.partition_name] = (reservation.partition_uuid, True)

    async def cancel(self, name: str) -> None:
        if not self.reservations:
            return
        reservation = self.rng.choice(self.reservations)
        await reservation.cancel()
        if self.partitions.get(reservation.partition_name) == (
            reservation.partition_uuid,
            False,
        ):
            del self.partitions[reservation.partition_name]
            self.backing_off[reservation.partition_uuid] = False

    async def unregister(self, name: str) -> None:
        await self.registry.unregister(name)
        if name in self.partitions:
            self.backing_off[self.partitions.pop(name)[0]] = False

    async def resolve(self, name: str) -> None:
        if name not in self.partitions:
            assert await self.registry.resolve(name) is None
            return
        partition_uuid, live = self.partitions[name]
        if not live:
            with pytest.raises(PartitionPendingError):
                await self.registry.resolve(name)
            return
        registration = await self.registry.resolve(name)
        assert registration is not None
        assert registration.partition_uuid == partition_uuid
        self.registrations.append(registration)

    async def require_current(self, name: str) -> None:
        if not self.registrations:
            return
        registration = self.rng.choice(self.registrations)
        if self.partitions.get(registration.partition_name) == (
            registration.partition_uuid,
            True,
        ):
            await registration.require_current()
            return
        with pytest.raises(PartitionDeletedError):
            await registration.require_current()

    async def purge(self, name: str) -> None:
        outcome = self.rng.choice(["nothing", "records", "raise"])
        claimed: list[UUID] = []

        async def purge_round(partition_uuid: UUID) -> bool:
            claimed.append(partition_uuid)
            if outcome == "raise":
                raise RuntimeError("the backend refused")
            return outcome == "records"

        claimable = {i for i, backs_off in self.backing_off.items() if not backs_off}
        if not claimable:
            assert not await self.registry._run_purge_round(purge_round)
            assert claimed == []
            return
        if outcome == "raise":
            with pytest.raises(RuntimeError):
                await self.registry._run_purge_round(purge_round)
        else:
            assert await self.registry._run_purge_round(purge_round)
        assert len(claimed) == 1
        assert claimed[0] in claimable
        if outcome == "nothing":
            del self.backing_off[claimed[0]]
        elif outcome == "raise":
            self.backing_off[claimed[0]] = True

    async def check_state(self) -> None:
        async with self.registry._engine.connect() as connection:
            rows = await connection.execute(
                select(
                    PartitionRow.partition_name,
                    PartitionRow.partition_uuid,
                    PartitionRow.live,
                ).where(
                    PartitionRow.partition_namespace
                    == self.registry._partition_namespace
                )
            )
            stored = {
                row.partition_name: (row.partition_uuid, row.live) for row in rows
            }
        assert stored == self.partitions
        assert set(await _queued(self.registry)) == set(self.backing_off)


@pytest.mark.asyncio
@pytest.mark.parametrize("seed", [0, 1, 2])
async def test_random_operation_sequences_agree_with_a_model(
    sqlalchemy_engine, partition_namespace, seed
):
    """Seeded sequences of every registry operation, checked step by step
    against a model of the contract: which partitions hold which names and
    whether they are live, what each reservation and registration may do, and
    which tombstones a purge round may claim and what its outcome leaves."""
    registry = await _registry(
        sqlalchemy_engine, partition_namespace, tombstone_retention_seconds=0
    )
    rng = random.Random(seed)
    model = _ModelCheckedRegistry(registry, rng)
    steps = [
        model.reserve,
        model.confirm,
        model.cancel,
        model.unregister,
        model.resolve,
        model.require_current,
        model.purge,
    ]
    for _ in range(80):
        step = rng.choice(steps)
        await step(rng.choice(["a", "b", "c"]))
        await model.check_state()
