"""The SQLAlchemy partition registry of one vector store: creation arbitrated by the database, deletion queued, purge claimed."""

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

from memmachine_server.common.data_types import SimilarityMetric
from memmachine_server.common.vector_store.data_types import (
    PartitionSchema,
    VectorStoreAttemptsExhaustedError,
    VectorStorePartitionAlreadyExistsError,
    VectorStorePartitionDeletedError,
    VectorStorePartitionHandleStaleError,
    VectorStorePartitionPendingError,
)
from memmachine_server.common.vector_store.partition_registry import (
    Registration,
    Reservation,
)
from memmachine_server.common.vector_store.partition_registry.sqlalchemy_partition_registry import (
    _MAX_PURGE_ATTEMPTS_WITHOUT_PROGRESS,
    PartitionRow,
    PurgeQueueRow,
    SQLAlchemyVectorStorePartitionRegistry,
    SQLAlchemyVectorStorePartitionRegistryParams,
    _TombstoneClaim,
)

SCHEMA = PartitionSchema(
    vector_dimensions=3,
    similarity_metric=SimilarityMetric.COSINE,
    indexed_properties={"name": "str"},
)
OTHER_SCHEMA = PartitionSchema(
    vector_dimensions=4,
    similarity_metric=SimilarityMetric.COSINE,
    indexed_properties={},
)
RETENTION = timedelta(days=1)
RETENTION_SECONDS = int(RETENTION.total_seconds())
PURGE_LEASE = timedelta(minutes=5)


@pytest.fixture
def vector_store_name() -> str:
    """A vector store name no other test (or earlier run on a shared server) used."""
    return f"test_{uuid4().hex[:12]}"


async def _registry(
    engine: AsyncEngine,
    vector_store_name: str,
    tombstone_retention_seconds: int = RETENTION_SECONDS,
    purge_lease_seconds: int = int(PURGE_LEASE.total_seconds()),
    base_purge_retry_backoff_seconds: int = 30,
) -> SQLAlchemyVectorStorePartitionRegistry:
    registry = SQLAlchemyVectorStorePartitionRegistry(
        SQLAlchemyVectorStorePartitionRegistryParams(
            engine=engine,
            vector_store_name=vector_store_name,
            tombstone_retention_seconds=tombstone_retention_seconds,
            purge_lease_seconds=purge_lease_seconds,
            base_purge_retry_backoff_seconds=base_purge_retry_backoff_seconds,
        )
    )
    await registry.startup()
    return registry


async def _queued(registry: SQLAlchemyVectorStorePartitionRegistry) -> list[UUID]:
    """The registry's tombstones, oldest first."""
    async with registry._engine.connect() as connection:
        rows = await connection.execute(
            select(PurgeQueueRow.incarnation)
            .where(PurgeQueueRow.vector_store_name == registry._vector_store_name)
            .order_by(PurgeQueueRow.enqueued_at)
        )
        return list(rows.scalars())


def _database_time_ago(engine: AsyncEngine, delta: timedelta) -> ColumnElement:
    """The database clock's now less `delta`, written by the database in the form of its own stamps."""
    if engine.dialect.name == "sqlite":
        return func.datetime("now", f"-{int(delta.total_seconds())} seconds")
    return func.now() - delta


async def _age_deletion(
    registry: SQLAlchemyVectorStorePartitionRegistry,
    incarnation: UUID,
    extra: timedelta = timedelta(0),
) -> None:
    """Move the tombstone's deletion back past the retention, by `extra` more, on the database clock."""
    async with registry._engine.begin() as connection:
        await connection.execute(
            update(PurgeQueueRow)
            .where(PurgeQueueRow.incarnation == incarnation)
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


async def _attempts(
    registry: SQLAlchemyVectorStorePartitionRegistry, incarnation: UUID
) -> int:
    """The tombstone's count of purge attempts without progress."""
    async with registry._engine.connect() as connection:
        return (
            await connection.execute(
                select(PurgeQueueRow.attempts_without_progress).where(
                    PurgeQueueRow.incarnation == incarnation
                )
            )
        ).scalar_one()


async def _age_last_failure(
    registry: SQLAlchemyVectorStorePartitionRegistry,
    incarnation: UUID,
    ago: timedelta,
) -> None:
    """Move the tombstone's last failed round back to `ago` before now, on the database clock."""
    async with registry._engine.begin() as connection:
        await connection.execute(
            update(PurgeQueueRow)
            .where(PurgeQueueRow.incarnation == incarnation)
            .values(last_failed_at=_database_time_ago(registry._engine, ago))
        )


async def _age_claim(
    registry: SQLAlchemyVectorStorePartitionRegistry,
    incarnation: UUID,
    ago: timedelta = PURGE_LEASE + timedelta(minutes=1),
) -> None:
    """Move the tombstone's claim back to `ago` before now, past the purge lease and the first backoff unless given, on the database clock."""
    async with registry._engine.begin() as connection:
        await connection.execute(
            update(PurgeQueueRow)
            .where(PurgeQueueRow.incarnation == incarnation)
            .values(claimed_at=_database_time_ago(registry._engine, ago))
        )


class _HeldRound:
    """A purge round that reports its start and ends when released, with the outcome given."""

    def __init__(self) -> None:
        self.ran: list[UUID] = []
        self.running = asyncio.Event()
        self._outcome: asyncio.Future[bool] = asyncio.get_running_loop().create_future()

    async def __call__(self, incarnation: UUID) -> bool:
        self.ran.append(incarnation)
        self.running.set()
        return await self._outcome

    def end(self, any_records_found: bool) -> None:
        self._outcome.set_result(any_records_found)

    def fail(self, error: Exception) -> None:
        self._outcome.set_exception(error)


async def _start_held_round(
    registry: SQLAlchemyVectorStorePartitionRegistry,
) -> tuple[_HeldRound, asyncio.Task[bool]]:
    """A purge round started on the oldest due tombstone, held until released."""
    held = _HeldRound()
    task = asyncio.create_task(registry.run_purge_round(held))
    await asyncio.wait_for(held.running.wait(), 30)
    return held, task


async def _abandon_claim(
    registry: SQLAlchemyVectorStorePartitionRegistry,
    monkeypatch: pytest.MonkeyPatch,
) -> UUID:
    """Claim the oldest due tombstone and stop, as a purger that dies during its round: nothing after the claim reaches the database. The claimed incarnation."""
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


async def _no_round(incarnation: UUID) -> bool:
    raise AssertionError("a round ran where none was due")


async def _failing_round(registry: SQLAlchemyVectorStorePartitionRegistry) -> UUID:
    """One purge round that raises; the incarnation it ran on."""
    ran: list[UUID] = []

    async def refused(incarnation: UUID) -> bool:
        ran.append(incarnation)
        raise RuntimeError("the backend refused")

    with pytest.raises(RuntimeError, match="the backend refused"):
        await registry.run_purge_round(refused)
    return ran[0]


async def _round(
    registry: SQLAlchemyVectorStorePartitionRegistry, any_records_found: bool
) -> UUID | None:
    """One purge round on the oldest due tombstone, reporting `any_records_found`; its incarnation, or None."""
    ran: list[UUID] = []

    async def purge_round(incarnation: UUID) -> bool:
        ran.append(incarnation)
        return any_records_found

    if not await registry.run_purge_round(purge_round):
        return None
    return ran[0]


@pytest.mark.asyncio
async def test_a_reservation_mints_a_fresh_incarnation_with_the_schema(
    sqlalchemy_engine, vector_store_name
):
    registry = await _registry(sqlalchemy_engine, vector_store_name)

    pending = await registry.reserve("c", SCHEMA)
    live = await pending.confirm()

    resolved = await registry.resolve("c")
    assert resolved is not None
    assert (resolved.incarnation, resolved.schema) == (pending.incarnation, SCHEMA)
    assert (live.incarnation, live.schema) == (pending.incarnation, SCHEMA)
    assert await registry.resolve("d") is None


@pytest.mark.asyncio
async def test_a_partition_is_pending_until_its_reservation_is_confirmed(
    sqlalchemy_engine, vector_store_name
):
    registry = await _registry(sqlalchemy_engine, vector_store_name)

    pending = await registry.reserve("c", SCHEMA)

    with pytest.raises(VectorStorePartitionPendingError) as resolved:
        await registry.resolve("c")
    assert resolved.value.schema == SCHEMA
    assert resolved.value.registered_at.tzinfo is not None
    await pending.confirm()
    live = await registry.resolve("c")
    assert live is not None
    assert live.incarnation == pending.incarnation
    with pytest.raises(VectorStorePartitionDeletedError):
        await pending.confirm()


@pytest.mark.asyncio
async def test_only_the_reserved_incarnation_is_confirmed(
    sqlalchemy_engine, vector_store_name
):
    """A creation whose partition was deleted while its storage was
    prepared cannot confirm a partition reserved under the key since."""
    registry = await _registry(sqlalchemy_engine, vector_store_name)
    deleted = await registry.reserve("c", SCHEMA)
    await registry.unregister("c")
    with pytest.raises(VectorStorePartitionDeletedError):
        await deleted.confirm()

    reserved_since = await registry.reserve("c", SCHEMA)

    with pytest.raises(VectorStorePartitionDeletedError):
        await deleted.confirm()
    with pytest.raises(VectorStorePartitionPendingError):
        await registry.resolve("c")
    live = await reserved_since.confirm()
    assert live.incarnation == reserved_since.incarnation


@pytest.mark.asyncio
async def test_cancelling_a_reservation_spares_the_key_reserved_since(
    sqlalchemy_engine, vector_store_name
):
    registry = await _registry(sqlalchemy_engine, vector_store_name)
    old = await registry.reserve("c", SCHEMA)
    await registry.unregister("c")
    since = await registry.reserve("c", SCHEMA)

    await old.cancel()
    with pytest.raises(VectorStorePartitionPendingError):
        await registry.resolve("c")
    assert await _queued(registry) == [old.incarnation]

    await since.cancel()
    await since.cancel()
    assert await registry.resolve("c") is None
    assert sorted(await _queued(registry)) == sorted(
        [old.incarnation, since.incarnation]
    )


@pytest.mark.asyncio
async def test_cancelling_a_confirmed_reservation_leaves_the_partition_live(
    sqlalchemy_engine, vector_store_name
):
    """Only a deletion by key ends a live partition, so a creator that
    cancels after a confirmation it did not see succeed takes nothing back."""
    registry = await _registry(sqlalchemy_engine, vector_store_name)
    reservation = await registry.reserve("c", SCHEMA)
    live = await reservation.confirm()

    await reservation.cancel()
    await live.require_current()
    assert await _queued(registry) == []


@pytest.mark.asyncio
async def test_a_registration_is_current_until_its_partition_is_deleted(
    sqlalchemy_engine, vector_store_name
):
    registry = await _registry(sqlalchemy_engine, vector_store_name)
    live = await (await registry.reserve("c", SCHEMA)).confirm()
    await live.require_current()

    await registry.unregister("c")
    with pytest.raises(VectorStorePartitionHandleStaleError):
        await live.require_current()

    await (await registry.reserve("c", SCHEMA)).confirm()
    with pytest.raises(VectorStorePartitionHandleStaleError):
        await live.require_current()


@pytest.mark.asyncio
async def test_a_taken_key_is_already_exists_whatever_the_schema(
    sqlalchemy_engine, vector_store_name
):
    """Taken by a pending partition, then by a live one."""
    registry = await _registry(sqlalchemy_engine, vector_store_name)
    registration = await registry.reserve("c", SCHEMA)

    for pending in (True, False):
        with pytest.raises(VectorStorePartitionAlreadyExistsError):
            await registry.reserve("c", SCHEMA)
        with pytest.raises(VectorStorePartitionAlreadyExistsError):
            await registry.reserve("c", OTHER_SCHEMA)
        if pending:
            await registration.confirm()


@pytest.mark.parametrize(
    "invalid_name", ["", "Upper", "with-hyphen", "trailing_newline\n", "x" * 33]
)
def test_a_vector_store_name_must_be_an_identifier(invalid_name, tmp_path):
    with pytest.raises(ValueError, match="Vector store name"):
        SQLAlchemyVectorStorePartitionRegistryParams(
            engine=create_async_engine(
                f"sqlite+aiosqlite:///{tmp_path / 'registry.db'}"
            ),
            vector_store_name=invalid_name,
            tombstone_retention_seconds=RETENTION_SECONDS,
        )


def test_an_engine_of_another_dialect_is_refused(monkeypatch, tmp_path):
    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'registry.db'}")
    monkeypatch.setattr(engine.dialect, "name", "mssql")
    with pytest.raises(ValueError, match="mssql"):
        SQLAlchemyVectorStorePartitionRegistryParams(
            engine=engine,
            vector_store_name="store",
            tombstone_retention_seconds=RETENTION_SECONDS,
        )


def test_a_sqlite_runtime_without_returning_is_refused(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "memmachine_server.common.vector_store.partition_registry.sqlalchemy_partition_registry.sqlite3.sqlite_version_info",
        (3, 34, 1),
    )
    with pytest.raises(ValueError, match="RETURNING"):
        SQLAlchemyVectorStorePartitionRegistryParams(
            engine=create_async_engine(
                f"sqlite+aiosqlite:///{tmp_path / 'registry.db'}"
            ),
            vector_store_name="store",
            tombstone_retention_seconds=RETENTION_SECONDS,
        )


def test_an_engine_sharing_one_connection_is_refused(tmp_path):
    """Every session would share one connection, so concurrent operations
    would run inside one another's transactions."""
    engine = create_async_engine(
        f"sqlite+aiosqlite:///{tmp_path / 'registry.db'}", poolclass=StaticPool
    )
    with pytest.raises(ValueError, match="StaticPool"):
        SQLAlchemyVectorStorePartitionRegistryParams(
            engine=engine,
            vector_store_name="store",
            tombstone_retention_seconds=RETENTION_SECONDS,
        )


@pytest.mark.parametrize("url", ["sqlite+aiosqlite://", "sqlite+aiosqlite:///:memory:"])
def test_an_in_memory_sqlite_engine_is_refused(url):
    """Each connection to in-memory SQLite gets a separate database, so the
    registry's state would not be shared, even within one process."""
    with pytest.raises(ValueError, match="in-memory"):
        SQLAlchemyVectorStorePartitionRegistryParams(
            engine=create_async_engine(url, poolclass=NullPool),
            vector_store_name="store",
            tombstone_retention_seconds=RETENTION_SECONDS,
        )


@pytest.mark.asyncio
async def test_registries_of_two_vector_stores_share_a_database_and_nothing_else(
    sqlalchemy_engine, vector_store_name
):
    """The same key in two vector stores' registries is two partitions: each
    resolves, deletes and purges apart from the other, and a purge round
    claims only its own registry's tombstones."""
    first = await _registry(sqlalchemy_engine, f"{vector_store_name}_a")
    second = await _registry(sqlalchemy_engine, f"{vector_store_name}_b")

    one = await (await first.reserve("c", SCHEMA)).confirm()
    two = await (await second.reserve("c", OTHER_SCHEMA)).confirm()
    assert one.incarnation != two.incarnation
    in_first = await first.resolve("c")
    in_second = await second.resolve("c")
    assert in_first is not None
    assert in_second is not None
    assert in_first.incarnation == one.incarnation
    assert in_second.incarnation == two.incarnation

    await first.unregister("c")
    await _age_deletion(first, one.incarnation)
    assert await first.resolve("c") is None
    still_in_second = await second.resolve("c")
    assert still_in_second is not None
    assert (still_in_second.incarnation, still_in_second.schema) == (
        two.incarnation,
        OTHER_SCHEMA,
    )
    await two.require_current()
    with pytest.raises(VectorStorePartitionHandleStaleError):
        await one.require_current()
    with pytest.raises(VectorStorePartitionAlreadyExistsError):
        await second.reserve("c", SCHEMA)

    assert await _round(second, any_records_found=False) is None
    assert await _round(first, any_records_found=False) == one.incarnation
    assert await _round(first, any_records_found=False) is None
    await two.require_current()


@pytest.mark.asyncio
async def test_concurrent_creators_get_one_winner(sqlalchemy_engine, vector_store_name):
    """Two registries on one database, the two-process shape in one process."""
    first = await _registry(sqlalchemy_engine, vector_store_name)
    second = await _registry(sqlalchemy_engine, vector_store_name)

    results = await asyncio.gather(
        *(registry.reserve("c", SCHEMA) for registry in (first, second) * 3),
        return_exceptions=True,
    )

    winners = [r for r in results if isinstance(r, Reservation)]
    losers = [
        r for r in results if isinstance(r, VectorStorePartitionAlreadyExistsError)
    ]
    assert len(winners) == 1
    assert len(losers) == 5
    live = await winners[0].confirm()
    resolved = await first.resolve("c")
    assert resolved is not None
    assert resolved.incarnation == live.incarnation


@pytest.mark.asyncio
async def test_delete_queues_the_incarnation_and_is_idempotent(
    sqlalchemy_engine, vector_store_name
):
    registry = await _registry(sqlalchemy_engine, vector_store_name)
    incarnation = (await registry.reserve("c", SCHEMA)).incarnation

    await registry.unregister("c")

    assert await registry.resolve("c") is None
    assert await _queued(registry) == [incarnation]
    await registry.unregister("c")
    await registry.unregister("never")
    assert await _queued(registry) == [incarnation]


@pytest.mark.asyncio
async def test_a_recreated_key_gets_a_new_incarnation(
    sqlalchemy_engine, vector_store_name
):
    registry = await _registry(sqlalchemy_engine, vector_store_name)
    old = (await registry.reserve("c", SCHEMA)).incarnation
    await registry.unregister("c")

    new = (await (await registry.reserve("c", SCHEMA)).confirm()).incarnation

    assert new != old
    resolved = await registry.resolve("c")
    assert resolved is not None
    assert resolved.incarnation == new


@pytest.mark.asyncio
async def test_a_round_is_given_where_the_records_are(
    sqlalchemy_engine, vector_store_name
):
    """The round is given the incarnation the deleted partition's records carry."""
    registry = await _registry(sqlalchemy_engine, vector_store_name)
    incarnation = (await registry.reserve("c", OTHER_SCHEMA)).incarnation
    await registry.unregister("c")
    await _age_deletion(registry, incarnation)
    given: list[UUID] = []

    async def purge_round(claimed: UUID) -> bool:
        given.append(claimed)
        return False

    assert await registry.run_purge_round(purge_round)
    assert given == [incarnation]


@pytest.mark.asyncio
async def test_a_tombstone_is_not_due_before_the_retention(
    sqlalchemy_engine, vector_store_name
):
    registry = await _registry(sqlalchemy_engine, vector_store_name)
    incarnation = (await registry.reserve("a", SCHEMA)).incarnation
    await registry.unregister("a")

    assert await _round(registry, any_records_found=False) is None
    assert await _queued(registry) == [incarnation]


@pytest.mark.asyncio
async def test_due_tombstones_go_oldest_deletion_first_until_a_round_finds_nothing(
    sqlalchemy_engine, vector_store_name, monkeypatch
):
    registry = await _registry(sqlalchemy_engine, vector_store_name)
    minted = iter([UUID(int=1), UUID(int=2)])
    monkeypatch.setattr(
        "memmachine_server.common.vector_store.partition_registry.sqlalchemy_partition_registry.uuid4",
        lambda: next(minted),
    )
    first = (await registry.reserve("a", SCHEMA)).incarnation
    second = (await registry.reserve("b", SCHEMA)).incarnation
    await registry.unregister("a")
    await registry.unregister("b")
    # The later deletion, with the larger incarnation, is made the older, so
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
    sqlalchemy_engine, vector_store_name
):
    """The claim compares the stamp the database clock wrote, by the
    database's arithmetic: with no retention the first round claims the
    tombstone, and no stamp is rewritten by the test."""
    registry = await _registry(
        sqlalchemy_engine, vector_store_name, tombstone_retention_seconds=0
    )
    incarnation = (await registry.reserve("a", SCHEMA)).incarnation
    await registry.unregister("a")

    assert await _round(registry, any_records_found=False) == incarnation

    assert await _queued(registry) == []


@pytest.mark.asyncio
async def test_a_changed_retention_applies_to_tombstones_already_queued(
    sqlalchemy_engine, vector_store_name
):
    """Only the deletion's time is stored; the retention in force when a
    claim is decided applies, including to tombstones queued under another."""
    queued_under_a_day = await _registry(sqlalchemy_engine, vector_store_name)
    incarnation = (await queued_under_a_day.reserve("a", SCHEMA)).incarnation
    await queued_under_a_day.unregister("a")
    assert await _round(queued_under_a_day, any_records_found=False) is None

    no_retention = await _registry(
        sqlalchemy_engine, vector_store_name, tombstone_retention_seconds=0
    )
    assert await _round(no_retention, any_records_found=False) == incarnation
    assert await _queued(no_retention) == []


@pytest.mark.asyncio
async def test_a_round_that_raises_keeps_the_tombstone_and_names_it_in_the_error(
    sqlalchemy_engine, vector_store_name
):
    registry = await _registry(sqlalchemy_engine, vector_store_name)
    incarnation = (await registry.reserve("a", SCHEMA)).incarnation
    await registry.unregister("a")
    await _age_deletion(registry, incarnation)

    async def refused(claimed: UUID) -> bool:
        assert claimed == incarnation
        raise RuntimeError("the backend refused")

    with pytest.raises(RuntimeError) as raised:
        await registry.run_purge_round(refused)

    assert [
        note
        for note in raised.value.__notes__
        if str(incarnation) in note
        and f"attempt 1 of {_MAX_PURGE_ATTEMPTS_WITHOUT_PROGRESS}" in note
    ]
    assert await _queued(registry) == [incarnation]
    assert await _attempts(registry, incarnation) == 1
    # Backing off: claimed again once 30 seconds have passed since the failure.
    assert await _round(registry, any_records_found=False) is None
    await _age_last_failure(registry, incarnation, timedelta(seconds=35))
    assert await _round(registry, any_records_found=False) == incarnation


@pytest.mark.asyncio
async def test_a_failed_tombstone_backs_off_while_the_ones_behind_it_are_claimed(
    sqlalchemy_engine, vector_store_name
):
    registry = await _registry(sqlalchemy_engine, vector_store_name)
    failing = (await registry.reserve("a", SCHEMA)).incarnation
    await registry.unregister("a")
    await _age_deletion(registry, failing, extra=timedelta(hours=1))
    later = (await registry.reserve("b", SCHEMA)).incarnation
    await registry.unregister("b")
    await _age_deletion(registry, later)

    assert await _failing_round(registry) == failing
    assert await _round(registry, any_records_found=False) == later


@pytest.mark.asyncio
async def test_the_backoff_doubles_with_each_failure_and_runs_from_the_last_one(
    sqlalchemy_engine, vector_store_name
):
    registry = await _registry(sqlalchemy_engine, vector_store_name)
    incarnation = (await registry.reserve("a", SCHEMA)).incarnation
    await registry.unregister("a")
    # Due for days: the backoff still runs from the last failure, not from
    # when the tombstone came due.
    await _age_deletion(registry, incarnation, extra=timedelta(days=3))

    await _failing_round(registry)
    await _age_last_failure(registry, incarnation, timedelta(seconds=35))
    await _failing_round(registry)
    # Two failures: a minute from the second.
    await _age_last_failure(registry, incarnation, timedelta(seconds=50))
    assert await _round(registry, any_records_found=True) is None
    await _age_last_failure(registry, incarnation, timedelta(seconds=70))
    assert await _round(registry, any_records_found=True) == incarnation


@pytest.mark.asyncio
async def test_the_backoff_stops_doubling_at_its_maximum(
    sqlalchemy_engine, vector_store_name
):
    registry = SQLAlchemyVectorStorePartitionRegistry(
        SQLAlchemyVectorStorePartitionRegistryParams(
            engine=sqlalchemy_engine,
            vector_store_name=vector_store_name,
            tombstone_retention_seconds=RETENTION_SECONDS,
            base_purge_retry_backoff_seconds=30,
            max_purge_retry_backoff_seconds=120,
        )
    )
    await registry.startup()
    incarnation = (await registry.reserve("a", SCHEMA)).incarnation
    await registry.unregister("a")
    await _age_deletion(registry, incarnation)

    for _ in range(5):
        await _failing_round(registry)
        await _age_last_failure(registry, incarnation, timedelta(days=1))
    # Five failures: 30 seconds doubled four times is 8 minutes, capped at 2.
    await _age_last_failure(registry, incarnation, timedelta(seconds=110))
    assert await _round(registry, any_records_found=True) is None
    await _age_last_failure(registry, incarnation, timedelta(seconds=130))
    assert await _round(registry, any_records_found=True) == incarnation


@pytest.mark.asyncio
async def test_a_tombstone_whose_rounds_keep_failing_is_dead_lettered_and_reported(
    sqlalchemy_engine, vector_store_name, caplog
):
    registry = await _registry(sqlalchemy_engine, vector_store_name)
    failing = (await registry.reserve("a", SCHEMA)).incarnation
    await registry.unregister("a")
    await _age_deletion(registry, failing, extra=timedelta(hours=1))
    later = (await registry.reserve("b", SCHEMA)).incarnation
    await registry.unregister("b")
    await _age_deletion(registry, later)

    with caplog.at_level(logging.ERROR):
        for _ in range(_MAX_PURGE_ATTEMPTS_WITHOUT_PROGRESS):
            # Past its backoff, so it is the oldest tombstone claimable.
            await _age_last_failure(registry, failing, timedelta(days=1))
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
    sqlalchemy_engine, vector_store_name
):
    """Setting the attempts without progress back to 0, as the dead-letter
    report says, makes the tombstone claimable at once, its last failure just
    now."""
    registry = await _registry(sqlalchemy_engine, vector_store_name)
    incarnation = (await registry.reserve("a", SCHEMA)).incarnation
    await registry.unregister("a")
    await _age_deletion(registry, incarnation)
    for _ in range(_MAX_PURGE_ATTEMPTS_WITHOUT_PROGRESS):
        await _age_last_failure(registry, incarnation, timedelta(days=1))
        assert await _failing_round(registry) == incarnation
    assert await _round(registry, any_records_found=False) is None

    async with registry._engine.begin() as connection:
        await connection.execute(
            update(PurgeQueueRow)
            .where(PurgeQueueRow.incarnation == incarnation)
            .values(attempts_without_progress=0)
        )

    assert await _round(registry, any_records_found=False) == incarnation


@pytest.mark.asyncio
async def test_a_round_that_finds_points_resets_the_attempts_without_progress(
    sqlalchemy_engine, vector_store_name
):
    registry = await _registry(sqlalchemy_engine, vector_store_name)
    incarnation = (await registry.reserve("a", SCHEMA)).incarnation
    await registry.unregister("a")
    await _age_deletion(registry, incarnation)

    for _ in range(_MAX_PURGE_ATTEMPTS_WITHOUT_PROGRESS - 1):
        await _failing_round(registry)
        await _age_last_failure(registry, incarnation, timedelta(days=1))
    assert await _round(registry, any_records_found=True) == incarnation
    assert await _attempts(registry, incarnation) == 0


@pytest.mark.asyncio
async def test_a_cancelled_round_ends_its_claim_without_counting_it(
    sqlalchemy_engine, vector_store_name, caplog
):
    """A cancelled round is not a failed round, and its tombstone is
    claimable at once, without waiting out the lease."""
    registry = await _registry(sqlalchemy_engine, vector_store_name)
    incarnation = (await registry.reserve("a", SCHEMA)).incarnation
    await registry.unregister("a")
    await _age_deletion(registry, incarnation)

    _, holding = await _start_held_round(registry)
    holding.cancel()
    with pytest.raises(asyncio.CancelledError):
        await holding

    assert await _attempts(registry, incarnation) == 0
    with caplog.at_level(logging.WARNING):
        assert await _round(registry, any_records_found=False) == incarnation
    assert not [r for r in caplog.records if r.levelno >= logging.WARNING]


@pytest.mark.asyncio
async def test_a_round_that_never_ended_is_retried_once_its_lease_and_the_backoff_pass(
    sqlalchemy_engine, vector_store_name, monkeypatch, caplog
):
    """A purger that dies during its round writes nothing after its claim,
    which counted the attempt. The tombstone stays claimed until the lease
    passes and then backs off; the claim after that is the next attempt, and
    is logged as a retry."""
    registry = await _registry(sqlalchemy_engine, vector_store_name)
    incarnation = (await registry.reserve("a", SCHEMA)).incarnation
    await registry.unregister("a")
    await _age_deletion(registry, incarnation)

    with caplog.at_level(logging.ERROR):
        assert await _abandon_claim(registry, monkeypatch) == incarnation
    assert [
        r
        for r in caplog.records
        if r.levelno == logging.ERROR and str(incarnation) in r.getMessage()
    ], "the lost end of the claim was not reported"
    assert await _attempts(registry, incarnation) == 1
    assert await _round(registry, any_records_found=False) is None

    # Past the lease, but not the 30-second backoff that runs from it.
    await _age_claim(registry, incarnation, PURGE_LEASE + timedelta(seconds=20))
    assert await _round(registry, any_records_found=False) is None

    await _age_claim(registry, incarnation, PURGE_LEASE + timedelta(seconds=40))
    caplog.clear()
    with caplog.at_level(logging.WARNING):
        assert await _round(registry, any_records_found=False) == incarnation
    assert [
        r
        for r in caplog.records
        if r.levelno == logging.WARNING
        and str(incarnation) in r.getMessage()
        and f"attempt 2 of {_MAX_PURGE_ATTEMPTS_WITHOUT_PROGRESS}" in r.getMessage()
    ]
    assert await _queued(registry) == []


@pytest.mark.asyncio
async def test_rounds_that_never_end_dead_letter_the_tombstone(
    sqlalchemy_engine, vector_store_name, monkeypatch, caplog
):
    """A tombstone whose rounds keep killing their purger is dead-lettered
    like one whose rounds keep raising, each retry logged with its attempt."""
    registry = await _registry(sqlalchemy_engine, vector_store_name)
    incarnation = (await registry.reserve("a", SCHEMA)).incarnation
    await registry.unregister("a")
    await _age_deletion(registry, incarnation)

    with caplog.at_level(logging.WARNING):
        for _ in range(_MAX_PURGE_ATTEMPTS_WITHOUT_PROGRESS):
            assert await _abandon_claim(registry, monkeypatch) == incarnation
            await _age_claim(registry, incarnation, timedelta(days=1))

    assert (
        await _attempts(registry, incarnation) == _MAX_PURGE_ATTEMPTS_WITHOUT_PROGRESS
    )
    assert [
        r
        for r in caplog.records
        if r.levelno == logging.WARNING
        and str(incarnation) in r.getMessage()
        and f"attempt {_MAX_PURGE_ATTEMPTS_WITHOUT_PROGRESS} of {_MAX_PURGE_ATTEMPTS_WITHOUT_PROGRESS}"
        in r.getMessage()
    ]
    assert await _round(registry, any_records_found=False) is None
    assert await _queued(registry) == [incarnation]


@pytest.mark.asyncio
async def test_a_cancelled_retry_of_a_round_that_never_ended_leaves_it_claimable(
    sqlalchemy_engine, vector_store_name, monkeypatch
):
    """A cancelled round takes back its attempt even when the attempt before
    it never ended and no round ever raised, so the tombstone is claimable at
    once rather than left with no time to back off from."""
    registry = await _registry(sqlalchemy_engine, vector_store_name)
    incarnation = (await registry.reserve("a", SCHEMA)).incarnation
    await registry.unregister("a")
    await _age_deletion(registry, incarnation)

    assert await _abandon_claim(registry, monkeypatch) == incarnation
    await _age_claim(registry, incarnation, timedelta(days=1))
    _, holding = await _start_held_round(registry)
    holding.cancel()
    with pytest.raises(asyncio.CancelledError):
        await holding

    assert await _attempts(registry, incarnation) == 1
    assert await _round(registry, any_records_found=False) == incarnation


@pytest.mark.asyncio
async def test_racing_purgers_retry_a_round_that_never_ended_once(
    sqlalchemy_engine, vector_store_name, monkeypatch
):
    """Purgers on separate engines that find the same claim whose round never
    ended claim it once between them, as its next attempt."""
    engines = [create_async_engine(sqlalchemy_engine.url) for _ in range(4)]
    ran: list[UUID] = []
    try:
        purgers = [await _registry(engine, vector_store_name) for engine in engines]
        incarnation = (await purgers[0].reserve("a", SCHEMA)).incarnation
        await purgers[0].unregister("a")
        await _age_deletion(purgers[0], incarnation)
        await _abandon_claim(purgers[0], monkeypatch)
        await _age_claim(purgers[0], incarnation)

        # Finding nothing removes the tombstone, so a second claim would show.
        async def found_nothing(claimed: UUID) -> bool:
            ran.append(claimed)
            return False

        await asyncio.wait_for(
            asyncio.gather(
                *(purger.run_purge_round(found_nothing) for purger in purgers)
            ),
            30,
        )
    finally:
        for engine in engines:
            await engine.dispose()

    assert ran == [incarnation]


@pytest.mark.asyncio
async def test_a_claimed_tombstone_is_skipped_while_its_round_runs(
    sqlalchemy_engine, vector_store_name
):
    """A second purger takes the next due tombstone, and finds none once the
    rest are taken; the claimed one is not claimed again until its round ends."""
    registry = await _registry(sqlalchemy_engine, vector_store_name)
    older = (await registry.reserve("older", SCHEMA)).incarnation
    newer = (await registry.reserve("newer", SCHEMA)).incarnation
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
    sqlalchemy_engine, vector_store_name, caplog, stale_outcome
):
    """Once a round's lease and the backoff have passed, the tombstone is
    claimed again as the next attempt. The first round's end then changes
    nothing: the later claim holds, and each round counts once."""
    registry = await _registry(sqlalchemy_engine, vector_store_name)
    incarnation = (await registry.reserve("a", SCHEMA)).incarnation
    await registry.unregister("a")
    await _age_deletion(registry, incarnation)

    stale, stale_task = await _start_held_round(registry)
    current_task = None
    try:
        await _age_claim(registry, incarnation)
        current, current_task = await _start_held_round(registry)
        assert current.ran == [incarnation]

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
            if r.levelno == logging.WARNING and str(incarnation) in r.getMessage()
        ]
        assert await _attempts(registry, incarnation) == 2
        assert await _round(registry, any_records_found=False) is None

        current.end(any_records_found=True)
        assert await asyncio.wait_for(current_task, 30)
    finally:
        for task in (stale_task, current_task):
            if task is not None and not task.done():
                task.cancel()

    assert await _round(registry, any_records_found=False) == incarnation


@pytest.mark.asyncio
async def test_a_round_that_finds_nothing_removes_the_tombstone_under_any_claim(
    sqlalchemy_engine, vector_store_name
):
    """An incarnation found empty after the retention stays empty, so a round
    that outlasted its lease still removes the tombstone it found empty."""
    registry = await _registry(sqlalchemy_engine, vector_store_name)
    incarnation = (await registry.reserve("a", SCHEMA)).incarnation
    await registry.unregister("a")
    await _age_deletion(registry, incarnation)

    stale, stale_task = await _start_held_round(registry)
    current_task = None
    try:
        await _age_claim(registry, incarnation)
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
    sqlalchemy_engine, vector_store_name
):
    """Purgers on separate engines, as in separate processes, split a backlog
    between them: every tombstone is claimed once, whatever the interleaving."""
    engines = [create_async_engine(sqlalchemy_engine.url) for _ in range(3)]
    try:
        purgers = [
            await _registry(engine, vector_store_name, tombstone_retention_seconds=0)
            for engine in engines
        ]
        incarnations = []
        for index in range(12):
            key = f"c{index}"
            incarnations.append((await purgers[0].reserve(key, SCHEMA)).incarnation)
            await purgers[0].unregister(key)
        ran: list[UUID] = []

        async def purge_round(incarnation: UUID) -> bool:
            ran.append(incarnation)
            await asyncio.sleep(0.01)
            return False

        async def drain(purger: SQLAlchemyVectorStorePartitionRegistry) -> None:
            while await purger.run_purge_round(purge_round):
                pass

        await asyncio.wait_for(asyncio.gather(*(drain(p) for p in purgers)), 60)
    finally:
        for engine in engines:
            await engine.dispose()

    assert sorted(ran) == sorted(incarnations)


@pytest.mark.asyncio
async def test_an_incarnation_awaiting_purge_is_never_reminted(
    sqlalchemy_engine, vector_store_name, monkeypatch
):
    """A minted incarnation colliding with a queued tombstone is rejected and re-minted."""
    registry = await _registry(sqlalchemy_engine, vector_store_name)
    dead = (await registry.reserve("a", SCHEMA)).incarnation
    await registry.unregister("a")
    minted = iter([dead, uuid4()])
    monkeypatch.setattr(
        "memmachine_server.common.vector_store.partition_registry.sqlalchemy_partition_registry.uuid4",
        lambda: next(minted),
    )

    fresh = (await registry.reserve("b", SCHEMA)).incarnation

    assert fresh != dead
    assert await _queued(registry) == [dead]


@pytest.mark.asyncio
async def test_creation_that_never_mints_a_free_incarnation_gives_up(
    sqlalchemy_engine, vector_store_name, monkeypatch
):
    registry = await _registry(sqlalchemy_engine, vector_store_name)
    dead = (await registry.reserve("a", SCHEMA)).incarnation
    await registry.unregister("a")
    monkeypatch.setattr(
        "memmachine_server.common.vector_store.partition_registry.sqlalchemy_partition_registry.uuid4",
        lambda: dead,
    )

    with pytest.raises(VectorStoreAttemptsExhaustedError):
        await registry.reserve("b", SCHEMA)
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
    sqlalchemy_engine, vector_store_name
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
        registry = await _registry(skewed_engine, vector_store_name)
        incarnation = (await registry.reserve("a", SCHEMA)).incarnation
        await registry.unregister("a")

        async with skewed_engine.connect() as connection:
            row = (
                await connection.execute(
                    select(
                        PurgeQueueRow.enqueued_at, text("pg_catalog.now() AS real_now")
                    ).where(PurgeQueueRow.incarnation == incarnation)
                )
            ).one()
        assert row.real_now - row.enqueued_at > timedelta(days=9)
        assert await _round(registry, any_records_found=False) is None
        assert await _queued(registry) == [incarnation]


@pytest.mark.asyncio
async def test_a_purge_lease_runs_by_the_database_clock(
    sqlalchemy_engine, vector_store_name
):
    """A claim is stamped, and its lease measured, by the database's clock.

    By a clock ten days behind the real one, a claim taken just now still
    holds, though by the real clock it was taken ten days ago.
    """
    if sqlalchemy_engine.dialect.name != "postgresql":
        pytest.skip("SQLite's clock cannot be shadowed")
    async with _engine_ten_days_behind(sqlalchemy_engine) as skewed_engine:
        registry = await _registry(
            skewed_engine, vector_store_name, tombstone_retention_seconds=0
        )
        incarnation = (await registry.reserve("a", SCHEMA)).incarnation
        await registry.unregister("a")

        held, holding = await _start_held_round(registry)
        try:
            async with skewed_engine.connect() as connection:
                row = (
                    await connection.execute(
                        select(
                            PurgeQueueRow.claimed_at,
                            text("pg_catalog.now() AS real_now"),
                        ).where(PurgeQueueRow.incarnation == incarnation)
                    )
                ).one()
            assert row.real_now - row.claimed_at > timedelta(days=9)
            assert await _round(registry, any_records_found=False) is None
            held.end(any_records_found=False)
            assert await asyncio.wait_for(holding, 30)
        finally:
            if not holding.done():
                holding.cancel()


@pytest.mark.asyncio
async def test_no_transaction_is_open_while_a_round_runs(
    sqlalchemy_engine, vector_store_name
):
    """The claim commits before the round, so no backend sits idle in a
    transaction, holding a row lock or vacuum's horizon, during its remote calls."""
    if sqlalchemy_engine.dialect.name != "postgresql":
        pytest.skip("pg_stat_activity is PostgreSQL's")
    registry = await _registry(
        sqlalchemy_engine, vector_store_name, tombstone_retention_seconds=0
    )
    await registry.reserve("a", SCHEMA)
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
    sqlalchemy_engine, vector_store_name
):
    """A purger never waits on a claim another purger is taking: it takes the next due tombstone."""
    if sqlalchemy_engine.dialect.name != "postgresql":
        pytest.skip("SQLite holds no row locks to skip")
    registry = await _registry(sqlalchemy_engine, vector_store_name)
    held = (await registry.reserve("held", SCHEMA)).incarnation
    free = (await registry.reserve("free", SCHEMA)).incarnation
    await registry.unregister("held")
    await registry.unregister("free")
    await _age_deletion(registry, held, extra=timedelta(minutes=1))
    await _age_deletion(registry, free)

    claim = None
    try:
        async with sqlalchemy_engine.connect() as other_purger, other_purger.begin():
            await other_purger.execute(
                select(PurgeQueueRow.incarnation)
                .where(PurgeQueueRow.incarnation == held)
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
async def test_sqlite_stays_writable_while_a_round_runs(tmp_path, vector_store_name):
    """On SQLite the claim's write commits before the round, so while the
    round runs, another connection writes at once, under a busy timeout far
    shorter than the round, and another purger claims the next tombstone."""
    url = f"sqlite+aiosqlite:///{tmp_path / 'registry.db'}"
    purger_engine = create_async_engine(url)
    impatient_engine = create_async_engine(url, connect_args={"timeout": 0.1})
    try:
        purger = await _registry(purger_engine, vector_store_name)
        impatient = await _registry(impatient_engine, vector_store_name)
        older = (await purger.reserve("older", SCHEMA)).incarnation
        newer = (await purger.reserve("newer", SCHEMA)).incarnation
        await purger.unregister("older")
        await purger.unregister("newer")
        await _age_deletion(purger, older, extra=timedelta(minutes=1))
        await _age_deletion(purger, newer)

        held, holding = await _start_held_round(purger)
        try:
            assert held.ran == [older]
            await (await impatient.reserve("c", SCHEMA)).confirm()
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
    sqlalchemy_engine, vector_store_name
):
    """Racing deletions serialize on the partition's row: the losers delete
    nothing, one tombstone is queued, and the key can be created again."""
    registries = [
        await _registry(sqlalchemy_engine, vector_store_name) for _ in range(4)
    ]
    for cycle in range(10):
        await registries[0].reserve("c", SCHEMA)
        await asyncio.wait_for(
            asyncio.gather(*(r.unregister("c") for r in registries)), 30
        )
        assert await registries[0].resolve("c") is None
        assert len(await _queued(registries[0])) == cycle + 1
    await registries[0].reserve("c", SCHEMA)


@pytest.mark.asyncio
async def test_a_deletion_racing_another_waits_and_queues_nothing_more(
    sqlalchemy_engine, vector_store_name
):
    """A deletion that finds another process's deletion in flight waits for
    it, then finds nothing to delete: one tombstone, and no error."""
    if sqlalchemy_engine.dialect.name != "postgresql":
        pytest.skip("SQLite serializes whole write transactions")
    registry = await _registry(sqlalchemy_engine, vector_store_name)
    incarnation = (await registry.reserve("c", SCHEMA)).incarnation

    local = None
    try:
        async with sqlalchemy_engine.connect() as remote, remote.begin():
            # Another process's deletion, held uncommitted.
            await remote.execute(
                delete(PartitionRow).where(PartitionRow.incarnation == incarnation)
            )
            await remote.execute(
                insert(PurgeQueueRow).values(
                    incarnation=incarnation,
                    vector_store_name=vector_store_name,
                    partition_key="c",
                    enqueued_at=func.now(),
                )
            )
            local = asyncio.create_task(registry.unregister("c"))
            assert await _blocked_or_done(sqlalchemy_engine, local) == "blocked"
        await asyncio.wait_for(local, 30)
    finally:
        if local is not None and not local.done():
            local.cancel()

    assert await _queued(registry) == [incarnation]


@pytest.mark.asyncio
async def test_an_incarnation_colliding_with_a_live_partition_is_reminted(
    sqlalchemy_engine, vector_store_name, monkeypatch
):
    """A mint rejected by the incarnation's unique constraint, with the key
    free, is a collision to mint again, not a taken key."""
    registry = await _registry(sqlalchemy_engine, vector_store_name)
    live = (await (await registry.reserve("live", SCHEMA)).confirm()).incarnation
    minted = iter([live, uuid4()])
    monkeypatch.setattr(
        "memmachine_server.common.vector_store.partition_registry.sqlalchemy_partition_registry.uuid4",
        lambda: next(minted),
    )

    fresh = (await registry.reserve("fresh", SCHEMA)).incarnation

    assert fresh != live
    resolved = await registry.resolve("live")
    assert resolved is not None
    assert resolved.incarnation == live


@pytest.mark.asyncio
async def test_a_mint_checks_the_queue_after_its_insert(
    sqlalchemy_engine, vector_store_name, monkeypatch
):
    """A mint colliding with a deletion in flight sees the deletion's tombstone.

    The deletion frees the incarnation's row and queues its tombstone in one
    uncommitted transaction; the colliding mint's insert waits on it. Only a
    queue check made after the insert sees the tombstone once the deletion
    commits: one made before reads the queue too early and registers a
    partition under an incarnation awaiting purge.
    """
    registry = await _registry(sqlalchemy_engine, vector_store_name)
    victim = (await registry.reserve("victim", SCHEMA)).incarnation
    minted = iter([victim, uuid4()])
    monkeypatch.setattr(
        "memmachine_server.common.vector_store.partition_registry.sqlalchemy_partition_registry.uuid4",
        lambda: next(minted),
    )
    insert_issued = asyncio.Event()

    def on_statement(_connection, _cursor, statement, _parameters, _context, _many):
        if statement.startswith(f"INSERT INTO {PartitionRow.__tablename__}"):
            insert_issued.set()

    event.listen(sqlalchemy_engine.sync_engine, "before_cursor_execute", on_statement)
    creator = None
    try:
        async with sqlalchemy_engine.connect() as remote, remote.begin():
            await remote.execute(
                delete(PartitionRow).where(PartitionRow.incarnation == victim)
            )
            await remote.execute(
                insert(PurgeQueueRow).values(
                    incarnation=victim,
                    vector_store_name=vector_store_name,
                    partition_key="victim",
                    enqueued_at=func.now(),
                )
            )
            creator = asyncio.create_task(registry.reserve("fresh", SCHEMA))
            await asyncio.wait_for(insert_issued.wait(), 30)
        fresh = (await asyncio.wait_for(creator, 30)).incarnation
    finally:
        event.remove(
            sqlalchemy_engine.sync_engine, "before_cursor_execute", on_statement
        )
        if creator is not None and not creator.done():
            creator.cancel()

    assert fresh != victim, "a live partition took an incarnation awaiting purge"
    assert await _queued(registry) == [victim]


@pytest.mark.asyncio
async def test_a_round_claims_one_tombstone(sqlalchemy_engine, vector_store_name):
    """A claim takes one tombstone and leaves the others to other purgers."""
    registry = await _registry(
        sqlalchemy_engine, vector_store_name, tombstone_retention_seconds=0
    )
    for name in ("a", "b", "c"):
        await registry.reserve(name, SCHEMA)
        await registry.unregister(name)

    assert await _round(registry, any_records_found=False) is not None

    assert len(await _queued(registry)) == 2


@pytest.mark.asyncio
async def test_a_persistent_database_error_surfaces_with_its_cause(
    sqlalchemy_engine, vector_store_name, monkeypatch
):
    """A rejected insert is retried a bounded number of times, then the
    database's own error is chained to the one that gives up."""
    registry = await _registry(sqlalchemy_engine, vector_store_name)
    # A NOT NULL violation stands in for a cause that is not a collision.
    monkeypatch.setattr(
        "memmachine_server.common.vector_store.partition_registry.sqlalchemy_partition_registry.uuid4",
        lambda: None,
    )

    with pytest.raises(VectorStoreAttemptsExhaustedError) as raised:
        await asyncio.wait_for(registry.reserve("a", SCHEMA), 30)

    cause: BaseException | None = raised.value
    while cause is not None and not isinstance(cause, IntegrityError):
        cause = cause.__cause__
    assert isinstance(cause, IntegrityError)


@pytest.mark.asyncio
async def test_a_changed_lease_applies_to_claims_already_held(
    sqlalchemy_engine, vector_store_name
):
    """The lease is applied when a claim is decided, from when the claim was
    taken: a claim taken under a five-minute lease holds against a one-minute
    lease, and the 30-second backoff after it, for a minute and a half and no
    longer, while it still holds against the five-minute lease."""
    five_minutes = await _registry(sqlalchemy_engine, vector_store_name)
    one_minute = await _registry(
        sqlalchemy_engine, vector_store_name, purge_lease_seconds=60
    )
    incarnation = (await five_minutes.reserve("a", SCHEMA)).incarnation
    await five_minutes.unregister("a")
    await _age_deletion(five_minutes, incarnation)

    held, holding = await _start_held_round(five_minutes)
    try:
        await _age_claim(five_minutes, incarnation, timedelta(seconds=80))
        assert not await one_minute.run_purge_round(_no_round)
        await _age_claim(five_minutes, incarnation, timedelta(seconds=100))
        assert not await five_minutes.run_purge_round(_no_round)
        assert await _round(one_minute, any_records_found=True) == incarnation
        held.end(any_records_found=True)
        assert await asyncio.wait_for(holding, 30)
    finally:
        if not holding.done():
            holding.cancel()


@pytest.mark.asyncio
async def test_the_first_purge_backoff_comes_from_its_parameter(
    sqlalchemy_engine, vector_store_name
):
    """With a ten-second first backoff, a failed tombstone is claimed again
    after ten seconds, and after twenty once it has failed twice."""
    registry = await _registry(
        sqlalchemy_engine, vector_store_name, base_purge_retry_backoff_seconds=10
    )
    incarnation = (await registry.reserve("a", SCHEMA)).incarnation
    await registry.unregister("a")
    await _age_deletion(registry, incarnation)

    assert await _failing_round(registry) == incarnation
    await _age_last_failure(registry, incarnation, timedelta(seconds=5))
    assert await _round(registry, any_records_found=True) is None
    await _age_last_failure(registry, incarnation, timedelta(seconds=15))
    assert await _failing_round(registry) == incarnation
    await _age_last_failure(registry, incarnation, timedelta(seconds=15))
    assert await _round(registry, any_records_found=True) is None
    await _age_last_failure(registry, incarnation, timedelta(seconds=25))
    assert await _round(registry, any_records_found=True) == incarnation


@pytest.mark.asyncio
async def test_a_deletion_whose_tombstone_cannot_be_queued_changes_nothing(
    sqlalchemy_engine, vector_store_name
):
    """A deletion and its tombstone commit together: when queuing the
    tombstone fails, the partition stays registered and nothing is queued,
    so no deleted incarnation is ever left without a tombstone."""
    registry = await _registry(sqlalchemy_engine, vector_store_name)
    live = await (await registry.reserve("c", SCHEMA)).confirm()

    with (
        _failing_statement(
            sqlalchemy_engine, f"INSERT INTO {PurgeQueueRow.__tablename__}"
        ),
        pytest.raises(_InjectedFailureError),
    ):
        await registry.unregister("c")

    resolved = await registry.resolve("c")
    assert resolved is not None
    assert resolved.incarnation == live.incarnation
    await live.require_current()
    assert await _queued(registry) == []

    await registry.unregister("c")
    assert await registry.resolve("c") is None
    assert await _queued(registry) == [live.incarnation]


@pytest.mark.asyncio
async def test_a_reservation_whose_queue_check_fails_leaves_the_key_free(
    sqlalchemy_engine, vector_store_name
):
    """A reservation and its check of the purge queue commit together: when
    the check fails, no pending partition holds the key."""
    registry = await _registry(sqlalchemy_engine, vector_store_name)

    with (
        _failing_statement(sqlalchemy_engine, f"SELECT {PurgeQueueRow.__tablename__}"),
        pytest.raises(_InjectedFailureError),
    ):
        await registry.reserve("c", SCHEMA)

    assert await registry.resolve("c") is None
    await (await registry.reserve("c", SCHEMA)).confirm()


@pytest.mark.asyncio
async def test_a_round_whose_outcome_cannot_be_written_is_retried_once_its_lease_passes(
    sqlalchemy_engine, vector_store_name
):
    """A round whose outcome is lost is, to the registry, a round that never
    ended: its error propagates, the claim holds until the lease and the
    backoff pass, and the tombstone is then claimed as the next attempt."""
    registry = await _registry(sqlalchemy_engine, vector_store_name)
    incarnation = (await registry.reserve("a", SCHEMA)).incarnation
    await registry.unregister("a")
    await _age_deletion(registry, incarnation)
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
        await registry.run_purge_round(found_records)

    assert await _attempts(registry, incarnation) == 1
    assert await _round(registry, any_records_found=False) is None
    await _age_claim(registry, incarnation)
    assert await _round(registry, any_records_found=True) == incarnation
    assert await _attempts(registry, incarnation) == 0


@pytest.mark.asyncio
async def test_a_reservation_colliding_with_a_tombstone_under_purge_does_not_wait_for_the_round(
    sqlalchemy_engine, vector_store_name, monkeypatch
):
    """A purge round holds no lock while it runs, so a reservation whose
    minted incarnation collides with the tombstone being purged finds the
    collision and mints again at once, on PostgreSQL and SQLite alike."""
    registry = await _registry(sqlalchemy_engine, vector_store_name)
    dead = (await registry.reserve("dead", SCHEMA)).incarnation
    await registry.unregister("dead")
    await _age_deletion(registry, dead)

    held, holding = await _start_held_round(registry)
    try:
        assert held.ran == [dead]
        minted = iter([dead, uuid4()])
        monkeypatch.setattr(
            "memmachine_server.common.vector_store.partition_registry.sqlalchemy_partition_registry.uuid4",
            lambda: next(minted),
        )
        fresh = await asyncio.wait_for(registry.reserve("b", SCHEMA), 10)
        assert fresh.incarnation != dead
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

    async def __call__(self, incarnation: UUID) -> bool:
        if incarnation in self._in_flight:
            self.overlapping.append(incarnation)
        self._in_flight.add(incarnation)
        try:
            await asyncio.sleep(0.002)
            return incarnation.int % 2 == 0
        finally:
            self._in_flight.discard(incarnation)


_REFUSALS = (
    VectorStorePartitionAlreadyExistsError,
    VectorStorePartitionPendingError,
    VectorStorePartitionDeletedError,
    VectorStorePartitionHandleStaleError,
)


async def _churn(
    registry: SQLAlchemyVectorStorePartitionRegistry,
    seed: int,
    purge_round: _OverlapWatch,
) -> None:
    """A seeded run of creations, cancellations, deletions, reads and purge rounds over a few keys."""
    rng = random.Random(seed)
    for _ in range(40):
        key = rng.choice(["a", "b", "c"])
        choice = rng.random()
        try:
            if choice < 0.35:
                reservation = await registry.reserve(key, SCHEMA)
                if rng.random() < 0.8:
                    await (await reservation.confirm()).require_current()
                else:
                    await reservation.cancel()
            elif choice < 0.6:
                await registry.unregister(key)
            elif choice < 0.8:
                registration = await registry.resolve(key)
                if registration is not None:
                    await registration.require_current()
            else:
                await registry.run_purge_round(purge_round)
        except _REFUSALS:
            pass


async def _found_nothing(incarnation: UUID) -> bool:
    return False


@pytest.mark.asyncio
async def test_mixed_operations_across_engines_finish_cleanly_and_strand_nothing(
    sqlalchemy_engine, vector_store_name
):
    """Creators, deleters, readers and purgers on separate engines, as in
    separate processes, race over a few shared keys. Every operation ends in
    its result or a documented refusal, never a database error or a deadlock;
    no tombstone's rounds overlap; and once they stop, no incarnation is both
    registered and queued, and the purge drains every tombstone."""
    engines = [create_async_engine(sqlalchemy_engine.url) for _ in range(4)]
    watch = _OverlapWatch()
    try:
        registries = [
            await _registry(engine, vector_store_name, tombstone_retention_seconds=0)
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
                        select(PartitionRow.incarnation).where(
                            PartitionRow.vector_store_name == vector_store_name
                        )
                    )
                ).scalars()
            )
        assert not registered & set(await _queued(registries[0]))

        for _ in range(1000):
            if not await registries[0].run_purge_round(_found_nothing):
                break
        assert await _queued(registries[0]) == []
    finally:
        for engine in engines:
            await engine.dispose()


class _ModelCheckedRegistry:
    """A registry driven step by step beside a model of its contract.

    The model holds which partition holds each key and whether it is live,
    which incarnations are queued and whether each is backing off, and the
    reservations and registrations handed out. Each step performs one
    operation, checks its outcome against the model, and updates the model.
    """

    def __init__(
        self,
        registry: SQLAlchemyVectorStorePartitionRegistry,
        rng: random.Random,
    ) -> None:
        self.registry = registry
        self.rng = rng
        self.partitions: dict[str, tuple[UUID, bool]] = {}
        self.backing_off: dict[UUID, bool] = {}
        self.reservations: list[Reservation] = []
        self.confirmed: set[UUID] = set()
        self.registrations: list[Registration] = []

    async def reserve(self, key: str) -> None:
        if key in self.partitions:
            with pytest.raises(VectorStorePartitionAlreadyExistsError):
                await self.registry.reserve(key, SCHEMA)
            return
        reservation = await self.registry.reserve(key, SCHEMA)
        assert reservation.incarnation not in self.backing_off
        self.partitions[key] = (reservation.incarnation, False)
        self.reservations.append(reservation)

    async def confirm(self, key: str) -> None:
        unconfirmed = [
            r for r in self.reservations if r.incarnation not in self.confirmed
        ]
        if not unconfirmed:
            return
        reservation = self.rng.choice(unconfirmed)
        self.confirmed.add(reservation.incarnation)
        if self.partitions.get(reservation.partition_key) != (
            reservation.incarnation,
            False,
        ):
            with pytest.raises(VectorStorePartitionDeletedError):
                await reservation.confirm()
            return
        self.registrations.append(await reservation.confirm())
        self.partitions[reservation.partition_key] = (reservation.incarnation, True)

    async def cancel(self, key: str) -> None:
        if not self.reservations:
            return
        reservation = self.rng.choice(self.reservations)
        await reservation.cancel()
        if self.partitions.get(reservation.partition_key) == (
            reservation.incarnation,
            False,
        ):
            del self.partitions[reservation.partition_key]
            self.backing_off[reservation.incarnation] = False

    async def unregister(self, key: str) -> None:
        await self.registry.unregister(key)
        if key in self.partitions:
            self.backing_off[self.partitions.pop(key)[0]] = False

    async def resolve(self, key: str) -> None:
        if key not in self.partitions:
            assert await self.registry.resolve(key) is None
            return
        incarnation, live = self.partitions[key]
        if not live:
            with pytest.raises(VectorStorePartitionPendingError):
                await self.registry.resolve(key)
            return
        registration = await self.registry.resolve(key)
        assert registration is not None
        assert registration.incarnation == incarnation
        self.registrations.append(registration)

    async def require_current(self, key: str) -> None:
        if not self.registrations:
            return
        registration = self.rng.choice(self.registrations)
        if self.partitions.get(registration.partition_key) == (
            registration.incarnation,
            True,
        ):
            await registration.require_current()
            return
        with pytest.raises(VectorStorePartitionHandleStaleError):
            await registration.require_current()

    async def purge(self, key: str) -> None:
        outcome = self.rng.choice(["nothing", "records", "raise"])
        claimed: list[UUID] = []

        async def purge_round(incarnation: UUID) -> bool:
            claimed.append(incarnation)
            if outcome == "raise":
                raise RuntimeError("the backend refused")
            return outcome == "records"

        claimable = {i for i, backs_off in self.backing_off.items() if not backs_off}
        if not claimable:
            assert not await self.registry.run_purge_round(purge_round)
            assert claimed == []
            return
        if outcome == "raise":
            with pytest.raises(RuntimeError):
                await self.registry.run_purge_round(purge_round)
        else:
            assert await self.registry.run_purge_round(purge_round)
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
                    PartitionRow.partition_key,
                    PartitionRow.incarnation,
                    PartitionRow.live,
                ).where(
                    PartitionRow.vector_store_name == self.registry._vector_store_name
                )
            )
            stored = {row.partition_key: (row.incarnation, row.live) for row in rows}
        assert stored == self.partitions
        assert set(await _queued(self.registry)) == set(self.backing_off)


@pytest.mark.asyncio
@pytest.mark.parametrize("seed", [0, 1, 2])
async def test_random_operation_sequences_agree_with_a_model(
    sqlalchemy_engine, vector_store_name, seed
):
    """Seeded sequences of every registry operation, checked step by step
    against a model of the contract: which partitions hold which keys and
    whether they are live, what each reservation and registration may do, and
    which tombstones a purge round may claim and what its outcome leaves."""
    registry = await _registry(
        sqlalchemy_engine, vector_store_name, tombstone_retention_seconds=0
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
