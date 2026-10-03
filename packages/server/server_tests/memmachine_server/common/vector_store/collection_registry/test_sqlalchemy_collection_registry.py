"""The SQLAlchemy collection registry: creation arbitrated by the database, deletion queued, purge claimed."""

import asyncio
import logging
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

from memmachine_server.common.vector_store.collection_registry import (
    Reservation,
)
from memmachine_server.common.vector_store.collection_registry.sqlalchemy_collection_registry import (
    _MAX_FAILED_PURGE_ROUNDS,
    CollectionRow,
    PurgeQueueRow,
    SQLAlchemyVectorStoreCollectionRegistry,
    SQLAlchemyVectorStoreCollectionRegistryParams,
)
from memmachine_server.common.vector_store.data_types import (
    VectorStoreAttemptsExhaustedError,
    VectorStoreCollectionAlreadyExistsError,
    VectorStoreCollectionConfig,
    VectorStoreCollectionDeletedError,
    VectorStoreCollectionHandleStaleError,
    VectorStoreCollectionPendingError,
)

CONFIG = VectorStoreCollectionConfig(
    vector_dimensions=3, indexed_properties_schema={"name": str}
)
OTHER_CONFIG = VectorStoreCollectionConfig(vector_dimensions=4)
RETENTION = timedelta(days=1)
RETENTION_SECONDS = int(RETENTION.total_seconds())
NAMESPACE = "ns"


@pytest.fixture
def vector_store_name() -> str:
    """A vector store name no other test (or earlier run on a shared server) used."""
    return f"test_{uuid4().hex[:12]}"


async def _registry(
    engine: AsyncEngine,
    vector_store_name: str,
    tombstone_retention_seconds: int = RETENTION_SECONDS,
) -> SQLAlchemyVectorStoreCollectionRegistry:
    registry = SQLAlchemyVectorStoreCollectionRegistry(
        SQLAlchemyVectorStoreCollectionRegistryParams(
            engine=engine,
            vector_store_name=vector_store_name,
            tombstone_retention_seconds=tombstone_retention_seconds,
        )
    )
    await registry.startup()
    return registry


async def _queued(registry: SQLAlchemyVectorStoreCollectionRegistry) -> list[UUID]:
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
    registry: SQLAlchemyVectorStoreCollectionRegistry,
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


async def _failed_rounds(
    registry: SQLAlchemyVectorStoreCollectionRegistry, incarnation: UUID
) -> int:
    """The tombstone's count of consecutive failed purge rounds."""
    async with registry._engine.connect() as connection:
        return (
            await connection.execute(
                select(PurgeQueueRow.failed_rounds).where(
                    PurgeQueueRow.incarnation == incarnation
                )
            )
        ).scalar_one()


async def _age_last_failure(
    registry: SQLAlchemyVectorStoreCollectionRegistry,
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


async def _failing_round(registry: SQLAlchemyVectorStoreCollectionRegistry) -> UUID:
    """One purge round that raises; the incarnation it ran on."""
    ran: list[UUID] = []

    async def refused(
        namespace: str, config: VectorStoreCollectionConfig, incarnation: UUID
    ) -> bool:
        ran.append(incarnation)
        raise RuntimeError("the backend refused")

    with pytest.raises(RuntimeError, match="the backend refused"):
        await registry.run_purge_round(refused)
    return ran[0]


async def _round(
    registry: SQLAlchemyVectorStoreCollectionRegistry, any_records_found: bool
) -> UUID | None:
    """One purge round on the oldest due tombstone, reporting `any_records_found`; its incarnation, or None."""
    ran: list[UUID] = []

    async def purge_round(
        namespace: str, config: VectorStoreCollectionConfig, incarnation: UUID
    ) -> bool:
        ran.append(incarnation)
        return any_records_found

    if not await registry.run_purge_round(purge_round):
        return None
    return ran[0]


@pytest.mark.asyncio
async def test_a_reservation_mints_a_fresh_incarnation_with_the_config(
    sqlalchemy_engine, vector_store_name
):
    registry = await _registry(sqlalchemy_engine, vector_store_name)

    pending = await registry.reserve(NAMESPACE, "c", CONFIG)
    live = await pending.confirm()

    resolved = await registry.resolve(NAMESPACE, "c")
    assert resolved is not None
    assert (resolved.incarnation, resolved.config) == (pending.incarnation, CONFIG)
    assert (live.incarnation, live.config) == (pending.incarnation, CONFIG)
    assert await registry.resolve(NAMESPACE, "d") is None
    assert await registry.resolve("other_ns", "c") is None


@pytest.mark.asyncio
async def test_a_collection_is_pending_until_its_reservation_is_confirmed(
    sqlalchemy_engine, vector_store_name
):
    registry = await _registry(sqlalchemy_engine, vector_store_name)

    pending = await registry.reserve(NAMESPACE, "c", CONFIG)

    with pytest.raises(VectorStoreCollectionPendingError) as resolved:
        await registry.resolve(NAMESPACE, "c")
    assert resolved.value.config == CONFIG
    assert resolved.value.registered_at.tzinfo is not None
    await pending.confirm()
    live = await registry.resolve(NAMESPACE, "c")
    assert live is not None
    assert live.incarnation == pending.incarnation
    with pytest.raises(VectorStoreCollectionDeletedError):
        await pending.confirm()


@pytest.mark.asyncio
async def test_only_the_reserved_incarnation_is_confirmed(
    sqlalchemy_engine, vector_store_name
):
    """A creation whose collection was deleted while its storage was
    prepared cannot confirm a collection reserved under the name since."""
    registry = await _registry(sqlalchemy_engine, vector_store_name)
    deleted = await registry.reserve(NAMESPACE, "c", CONFIG)
    await registry.unregister(NAMESPACE, "c")
    with pytest.raises(VectorStoreCollectionDeletedError):
        await deleted.confirm()

    reserved_since = await registry.reserve(NAMESPACE, "c", CONFIG)

    with pytest.raises(VectorStoreCollectionDeletedError):
        await deleted.confirm()
    with pytest.raises(VectorStoreCollectionPendingError):
        await registry.resolve(NAMESPACE, "c")
    live = await reserved_since.confirm()
    assert live.incarnation == reserved_since.incarnation


@pytest.mark.asyncio
async def test_cancelling_a_reservation_spares_the_name_reserved_since(
    sqlalchemy_engine, vector_store_name
):
    registry = await _registry(sqlalchemy_engine, vector_store_name)
    old = await registry.reserve(NAMESPACE, "c", CONFIG)
    await registry.unregister(NAMESPACE, "c")
    since = await registry.reserve(NAMESPACE, "c", CONFIG)

    await old.cancel()
    with pytest.raises(VectorStoreCollectionPendingError):
        await registry.resolve(NAMESPACE, "c")
    assert await _queued(registry) == [old.incarnation]

    await since.cancel()
    await since.cancel()
    assert await registry.resolve(NAMESPACE, "c") is None
    assert sorted(await _queued(registry)) == sorted(
        [old.incarnation, since.incarnation]
    )


@pytest.mark.asyncio
async def test_cancelling_a_confirmed_reservation_leaves_the_collection_live(
    sqlalchemy_engine, vector_store_name
):
    """Only a deletion by name ends a live collection, so a creator that
    cancels after a confirmation it did not see succeed takes nothing back."""
    registry = await _registry(sqlalchemy_engine, vector_store_name)
    reservation = await registry.reserve(NAMESPACE, "c", CONFIG)
    live = await reservation.confirm()

    await reservation.cancel()
    await live.require_current()
    assert await _queued(registry) == []


@pytest.mark.asyncio
async def test_a_registration_is_current_until_its_collection_is_deleted(
    sqlalchemy_engine, vector_store_name
):
    registry = await _registry(sqlalchemy_engine, vector_store_name)
    live = await (await registry.reserve(NAMESPACE, "c", CONFIG)).confirm()
    await live.require_current()

    await registry.unregister(NAMESPACE, "c")
    with pytest.raises(VectorStoreCollectionHandleStaleError):
        await live.require_current()

    await (await registry.reserve(NAMESPACE, "c", CONFIG)).confirm()
    with pytest.raises(VectorStoreCollectionHandleStaleError):
        await live.require_current()


@pytest.mark.asyncio
async def test_a_taken_name_is_already_exists_whatever_the_config(
    sqlalchemy_engine, vector_store_name
):
    """Taken by a pending collection, then by a live one."""
    registry = await _registry(sqlalchemy_engine, vector_store_name)
    reservation = await registry.reserve(NAMESPACE, "c", CONFIG)

    for pending in (True, False):
        with pytest.raises(VectorStoreCollectionAlreadyExistsError):
            await registry.reserve(NAMESPACE, "c", CONFIG)
        with pytest.raises(VectorStoreCollectionAlreadyExistsError):
            await registry.reserve(NAMESPACE, "c", OTHER_CONFIG)
        if pending:
            await reservation.confirm()


@pytest.mark.asyncio
async def test_any_string_of_at_most_255_characters_names_a_registry(
    sqlalchemy_engine, vector_store_name
):
    name = f"Store-{vector_store_name} ".ljust(255, "x")
    registry = await _registry(sqlalchemy_engine, name)
    pending = await registry.reserve(NAMESPACE, "c", CONFIG)
    await pending.confirm()

    resolved = await registry.resolve(NAMESPACE, "c")
    assert resolved is not None
    assert resolved.incarnation == pending.incarnation


def test_an_engine_of_another_dialect_is_refused(monkeypatch):
    engine = create_async_engine("sqlite+aiosqlite://")
    monkeypatch.setattr(engine.dialect, "name", "mssql")
    with pytest.raises(ValueError, match="mssql"):
        SQLAlchemyVectorStoreCollectionRegistryParams(
            engine=engine,
            vector_store_name="store",
            tombstone_retention_seconds=RETENTION_SECONDS,
        )


def test_a_sqlite_runtime_without_returning_is_refused(monkeypatch):
    monkeypatch.setattr(
        "memmachine_server.common.vector_store.collection_registry.sqlalchemy_collection_registry.sqlite3.sqlite_version_info",
        (3, 34, 1),
    )
    with pytest.raises(ValueError, match="RETURNING"):
        SQLAlchemyVectorStoreCollectionRegistryParams(
            engine=create_async_engine("sqlite+aiosqlite://"),
            vector_store_name="store",
            tombstone_retention_seconds=RETENTION_SECONDS,
        )


@pytest.mark.asyncio
async def test_registries_of_two_vector_stores_share_a_database_and_nothing_else(
    sqlalchemy_engine, vector_store_name
):
    """Each registry has its own collections, and a purge round claims
    only its own tombstones."""
    first = await _registry(sqlalchemy_engine, f"{vector_store_name}_a")
    second = await _registry(sqlalchemy_engine, f"{vector_store_name}_b")

    one = (await (await first.reserve(NAMESPACE, "c", CONFIG)).confirm()).incarnation
    two = (
        await (await second.reserve(NAMESPACE, "c", OTHER_CONFIG)).confirm()
    ).incarnation
    assert one != two
    in_first = await first.resolve(NAMESPACE, "c")
    in_second = await second.resolve(NAMESPACE, "c")
    assert in_first is not None
    assert in_second is not None
    assert in_first.incarnation == one
    assert in_second.incarnation == two

    await first.unregister(NAMESPACE, "c")
    await _age_deletion(first, one)
    still_in_second = await second.resolve(NAMESPACE, "c")
    assert still_in_second is not None
    assert still_in_second.incarnation == two
    assert await _round(second, any_records_found=False) is None
    assert await _round(first, any_records_found=False) == one


@pytest.mark.asyncio
async def test_concurrent_creators_get_one_winner(sqlalchemy_engine, vector_store_name):
    """Two registries on one database, the two-process shape in one process."""
    first = await _registry(sqlalchemy_engine, vector_store_name)
    second = await _registry(sqlalchemy_engine, vector_store_name)

    results = await asyncio.gather(
        *(registry.reserve(NAMESPACE, "c", CONFIG) for registry in (first, second) * 3),
        return_exceptions=True,
    )

    winners = [r for r in results if isinstance(r, Reservation)]
    losers = [
        r for r in results if isinstance(r, VectorStoreCollectionAlreadyExistsError)
    ]
    assert len(winners) == 1
    assert len(losers) == 5
    live = await winners[0].confirm()
    resolved = await first.resolve(NAMESPACE, "c")
    assert resolved is not None
    assert resolved.incarnation == live.incarnation


@pytest.mark.asyncio
async def test_delete_queues_the_incarnation_and_is_idempotent(
    sqlalchemy_engine, vector_store_name
):
    registry = await _registry(sqlalchemy_engine, vector_store_name)
    incarnation = (await registry.reserve(NAMESPACE, "c", CONFIG)).incarnation

    await registry.unregister(NAMESPACE, "c")

    assert await registry.resolve(NAMESPACE, "c") is None
    assert await _queued(registry) == [incarnation]
    await registry.unregister(NAMESPACE, "c")
    await registry.unregister(NAMESPACE, "never")
    assert await _queued(registry) == [incarnation]


@pytest.mark.asyncio
async def test_a_recreated_name_gets_a_new_incarnation(
    sqlalchemy_engine, vector_store_name
):
    registry = await _registry(sqlalchemy_engine, vector_store_name)
    old = (await registry.reserve(NAMESPACE, "c", CONFIG)).incarnation
    await registry.unregister(NAMESPACE, "c")

    new = (await (await registry.reserve(NAMESPACE, "c", CONFIG)).confirm()).incarnation

    assert new != old
    resolved = await registry.resolve(NAMESPACE, "c")
    assert resolved is not None
    assert resolved.incarnation == new


@pytest.mark.asyncio
async def test_a_round_is_given_where_the_records_are(
    sqlalchemy_engine, vector_store_name
):
    """The round is given what it needs to find the records: namespace, configuration, incarnation."""
    registry = await _registry(sqlalchemy_engine, vector_store_name)
    incarnation = (await registry.reserve(NAMESPACE, "c", OTHER_CONFIG)).incarnation
    await registry.unregister(NAMESPACE, "c")
    await _age_deletion(registry, incarnation)
    given: list[tuple[str, VectorStoreCollectionConfig, UUID]] = []

    async def purge_round(
        namespace: str, config: VectorStoreCollectionConfig, incarnation: UUID
    ) -> bool:
        given.append((namespace, config, incarnation))
        return False

    assert await registry.run_purge_round(purge_round)
    assert given == [(NAMESPACE, OTHER_CONFIG, incarnation)]


@pytest.mark.asyncio
async def test_a_tombstone_is_not_due_before_the_retention(
    sqlalchemy_engine, vector_store_name
):
    registry = await _registry(sqlalchemy_engine, vector_store_name)
    incarnation = (await registry.reserve(NAMESPACE, "a", CONFIG)).incarnation
    await registry.unregister(NAMESPACE, "a")

    assert await _round(registry, any_records_found=False) is None
    assert await _queued(registry) == [incarnation]


@pytest.mark.asyncio
async def test_due_tombstones_go_oldest_deletion_first_until_a_round_finds_nothing(
    sqlalchemy_engine, vector_store_name, monkeypatch
):
    registry = await _registry(sqlalchemy_engine, vector_store_name)
    minted = iter([UUID(int=1), UUID(int=2)])
    monkeypatch.setattr(
        "memmachine_server.common.vector_store.collection_registry.sqlalchemy_collection_registry.uuid4",
        lambda: next(minted),
    )
    first = (await registry.reserve(NAMESPACE, "a", CONFIG)).incarnation
    second = (await registry.reserve(NAMESPACE, "b", CONFIG)).incarnation
    await registry.unregister(NAMESPACE, "a")
    await registry.unregister(NAMESPACE, "b")
    # The later deletion, with the larger incarnation, is made the older, so
    # only the ordering by deletion can put it first.
    await _age_deletion(registry, first)
    await _age_deletion(registry, second, extra=timedelta(minutes=1))

    assert await _round(registry, any_records_found=True) == second
    # Found records: still due, and still the oldest deletion.
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
    incarnation = (await registry.reserve(NAMESPACE, "a", CONFIG)).incarnation
    await registry.unregister(NAMESPACE, "a")

    assert await _round(registry, any_records_found=False) == incarnation

    assert await _queued(registry) == []


@pytest.mark.asyncio
async def test_a_changed_retention_applies_to_tombstones_already_queued(
    sqlalchemy_engine, vector_store_name
):
    """Only the deletion's time is stored; the retention in force when a
    claim is decided applies, including to tombstones queued under another."""
    queued_under_a_day = await _registry(sqlalchemy_engine, vector_store_name)
    incarnation = (await queued_under_a_day.reserve(NAMESPACE, "a", CONFIG)).incarnation
    await queued_under_a_day.unregister(NAMESPACE, "a")
    assert await _round(queued_under_a_day, any_records_found=False) is None

    no_retention = await _registry(
        sqlalchemy_engine, vector_store_name, tombstone_retention_seconds=0
    )
    assert await _round(no_retention, any_records_found=False) == incarnation
    assert await _queued(no_retention) == []


@pytest.mark.asyncio
async def test_a_round_that_raises_keeps_the_tombstone_and_counts_the_failure(
    sqlalchemy_engine, vector_store_name
):
    registry = await _registry(sqlalchemy_engine, vector_store_name)
    incarnation = (await registry.reserve(NAMESPACE, "a", CONFIG)).incarnation
    await registry.unregister(NAMESPACE, "a")
    await _age_deletion(registry, incarnation)

    async def refused(
        namespace: str, config: VectorStoreCollectionConfig, claimed: UUID
    ) -> bool:
        assert claimed == incarnation
        raise RuntimeError("the backend refused")

    with pytest.raises(RuntimeError):
        await registry.run_purge_round(refused)

    assert await _queued(registry) == [incarnation]
    assert await _failed_rounds(registry, incarnation) == 1
    # Backing off: claimed again once 30 seconds have passed since the failure.
    assert await _round(registry, any_records_found=False) is None
    await _age_last_failure(registry, incarnation, timedelta(seconds=35))
    assert await _round(registry, any_records_found=False) == incarnation


@pytest.mark.asyncio
async def test_a_failed_tombstone_backs_off_while_the_ones_behind_it_are_claimed(
    sqlalchemy_engine, vector_store_name
):
    registry = await _registry(sqlalchemy_engine, vector_store_name)
    failing = (await registry.reserve(NAMESPACE, "a", CONFIG)).incarnation
    await registry.unregister(NAMESPACE, "a")
    await _age_deletion(registry, failing, extra=timedelta(hours=1))
    later = (await registry.reserve(NAMESPACE, "b", CONFIG)).incarnation
    await registry.unregister(NAMESPACE, "b")
    await _age_deletion(registry, later)

    assert await _failing_round(registry) == failing
    assert await _round(registry, any_records_found=False) == later


@pytest.mark.asyncio
async def test_the_backoff_doubles_with_each_failure_and_runs_from_the_last_one(
    sqlalchemy_engine, vector_store_name
):
    registry = await _registry(sqlalchemy_engine, vector_store_name)
    incarnation = (await registry.reserve(NAMESPACE, "a", CONFIG)).incarnation
    await registry.unregister(NAMESPACE, "a")
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
    registry = SQLAlchemyVectorStoreCollectionRegistry(
        SQLAlchemyVectorStoreCollectionRegistryParams(
            engine=sqlalchemy_engine,
            vector_store_name=vector_store_name,
            tombstone_retention_seconds=RETENTION_SECONDS,
            purge_retry_backoff_seconds=30,
            max_purge_retry_backoff_seconds=120,
        )
    )
    await registry.startup()
    incarnation = (await registry.reserve(NAMESPACE, "a", CONFIG)).incarnation
    await registry.unregister(NAMESPACE, "a")
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
    failing = (await registry.reserve(NAMESPACE, "a", CONFIG)).incarnation
    await registry.unregister(NAMESPACE, "a")
    await _age_deletion(registry, failing, extra=timedelta(hours=1))
    later = (await registry.reserve(NAMESPACE, "b", CONFIG)).incarnation
    await registry.unregister(NAMESPACE, "b")
    await _age_deletion(registry, later)

    with caplog.at_level(logging.ERROR):
        for _ in range(_MAX_FAILED_PURGE_ROUNDS):
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
async def test_a_failure_counted_past_the_dead_letter_bound_is_reported(
    sqlalchemy_engine, vector_store_name, caplog
):
    """A count already at the bound, as when two purgers failed one tombstone at
    once, still reports the dead-lettering."""
    registry = await _registry(sqlalchemy_engine, vector_store_name)
    incarnation = (await registry.reserve(NAMESPACE, "a", CONFIG)).incarnation
    await registry.unregister(NAMESPACE, "a")
    async with registry._engine.begin() as connection:
        await connection.execute(
            update(PurgeQueueRow)
            .where(PurgeQueueRow.incarnation == incarnation)
            .values(failed_rounds=_MAX_FAILED_PURGE_ROUNDS)
        )

    with caplog.at_level(logging.ERROR):
        await registry._count_failed_round(incarnation, RuntimeError("refused"))

    assert await _failed_rounds(registry, incarnation) == _MAX_FAILED_PURGE_ROUNDS + 1
    assert [
        r
        for r in caplog.records
        if r.levelno == logging.ERROR and str(incarnation) in r.getMessage()
    ]


@pytest.mark.asyncio
async def test_a_round_that_finds_points_clears_the_failed_rounds(
    sqlalchemy_engine, vector_store_name
):
    registry = await _registry(sqlalchemy_engine, vector_store_name)
    incarnation = (await registry.reserve(NAMESPACE, "a", CONFIG)).incarnation
    await registry.unregister(NAMESPACE, "a")
    await _age_deletion(registry, incarnation)

    for _ in range(_MAX_FAILED_PURGE_ROUNDS - 1):
        await _failing_round(registry)
        await _age_last_failure(registry, incarnation, timedelta(days=1))
    assert await _round(registry, any_records_found=True) == incarnation
    assert await _failed_rounds(registry, incarnation) == 0


@pytest.mark.asyncio
async def test_a_stored_configuration_that_no_longer_validates_counts_as_a_failed_round(
    sqlalchemy_engine, vector_store_name
):
    registry = await _registry(sqlalchemy_engine, vector_store_name)
    incarnation = (await registry.reserve(NAMESPACE, "a", CONFIG)).incarnation
    await registry.unregister(NAMESPACE, "a")
    await _age_deletion(registry, incarnation)
    async with registry._engine.begin() as connection:
        await connection.execute(
            update(PurgeQueueRow)
            .where(PurgeQueueRow.incarnation == incarnation)
            .values(config={"vector_dimensions": "not a number"})
        )

    async def never_run(
        namespace: str, config: VectorStoreCollectionConfig, claimed: UUID
    ) -> bool:
        raise AssertionError("the round ran on a configuration that does not validate")

    with pytest.raises(ValueError, match="vector_dimensions"):
        await registry.run_purge_round(never_run)
    assert await _failed_rounds(registry, incarnation) == 1


@pytest.mark.asyncio
async def test_a_cancelled_round_is_not_a_failed_round(
    sqlalchemy_engine, vector_store_name
):
    registry = await _registry(sqlalchemy_engine, vector_store_name)
    incarnation = (await registry.reserve(NAMESPACE, "a", CONFIG)).incarnation
    await registry.unregister(NAMESPACE, "a")
    await _age_deletion(registry, incarnation)

    async def cancelled(
        namespace: str, config: VectorStoreCollectionConfig, claimed: UUID
    ) -> bool:
        raise asyncio.CancelledError

    with pytest.raises(asyncio.CancelledError):
        await registry.run_purge_round(cancelled)
    assert await _failed_rounds(registry, incarnation) == 0


@pytest.mark.asyncio
async def test_an_incarnation_awaiting_purge_is_never_reminted(
    sqlalchemy_engine, vector_store_name, monkeypatch
):
    """A minted incarnation colliding with a queued tombstone is rejected and re-minted."""
    registry = await _registry(sqlalchemy_engine, vector_store_name)
    dead = (await registry.reserve(NAMESPACE, "a", CONFIG)).incarnation
    await registry.unregister(NAMESPACE, "a")
    minted = iter([dead, uuid4()])
    monkeypatch.setattr(
        "memmachine_server.common.vector_store.collection_registry.sqlalchemy_collection_registry.uuid4",
        lambda: next(minted),
    )

    fresh = (await registry.reserve(NAMESPACE, "b", CONFIG)).incarnation

    assert fresh != dead
    assert await _queued(registry) == [dead]


@pytest.mark.asyncio
async def test_creation_that_never_mints_a_free_incarnation_gives_up(
    sqlalchemy_engine, vector_store_name, monkeypatch
):
    registry = await _registry(sqlalchemy_engine, vector_store_name)
    dead = (await registry.reserve(NAMESPACE, "a", CONFIG)).incarnation
    await registry.unregister(NAMESPACE, "a")
    monkeypatch.setattr(
        "memmachine_server.common.vector_store.collection_registry.sqlalchemy_collection_registry.uuid4",
        lambda: dead,
    )

    with pytest.raises(VectorStoreAttemptsExhaustedError):
        await registry.reserve(NAMESPACE, "b", CONFIG)
    assert await registry.resolve(NAMESPACE, "b") is None


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
    schema = f"skewed_{uuid4().hex[:12]}"
    async with sqlalchemy_engine.begin() as connection:
        await connection.execute(text(f"CREATE SCHEMA {schema}"))
        await connection.execute(
            text(
                f"CREATE FUNCTION {schema}.now() RETURNS timestamptz LANGUAGE sql "
                "STABLE AS $$ SELECT pg_catalog.now() - interval '10 days' $$"
            )
        )
    skewed_engine = create_async_engine(
        sqlalchemy_engine.url,
        connect_args={
            "server_settings": {"search_path": f"{schema}, public, pg_catalog"}
        },
    )
    try:
        registry = await _registry(skewed_engine, vector_store_name)
        incarnation = (await registry.reserve(NAMESPACE, "a", CONFIG)).incarnation
        await registry.unregister(NAMESPACE, "a")

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
    finally:
        await skewed_engine.dispose()
        async with sqlalchemy_engine.begin() as connection:
            await connection.execute(text(f"DROP SCHEMA {schema} CASCADE"))


@pytest.mark.asyncio
async def test_a_claim_skips_a_tombstone_another_purger_holds(
    sqlalchemy_engine, vector_store_name
):
    """A purger never waits on another purger's claim: it takes the next due tombstone."""
    if sqlalchemy_engine.dialect.name != "postgresql":
        pytest.skip("SQLite holds no row locks to skip")
    registry = await _registry(sqlalchemy_engine, vector_store_name)
    held = (await registry.reserve(NAMESPACE, "held", CONFIG)).incarnation
    free = (await registry.reserve(NAMESPACE, "free", CONFIG)).incarnation
    await registry.unregister(NAMESPACE, "held")
    await registry.unregister(NAMESPACE, "free")
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
async def test_sqlite_purge_rounds_run_one_at_a_time(tmp_path, vector_store_name):
    """On SQLite the claim is a write: a second purger waits for the round under
    way, then claims the next tombstone, never the same one."""
    url = f"sqlite+aiosqlite:///{tmp_path / 'registry.db'}"
    # A busy timeout far past the held round, so the waiting purger never
    # gives up on the lock, however slow the machine.
    engines = [create_async_engine(url, connect_args={"timeout": 60}) for _ in range(2)]
    try:
        first, second = [
            await _registry(engine, vector_store_name) for engine in engines
        ]
        older = (await first.reserve(NAMESPACE, "older", CONFIG)).incarnation
        newer = (await first.reserve(NAMESPACE, "newer", CONFIG)).incarnation
        await first.unregister(NAMESPACE, "older")
        await first.unregister(NAMESPACE, "newer")
        await _age_deletion(first, older, extra=timedelta(minutes=1))
        await _age_deletion(first, newer)
        running = asyncio.Event()
        release = asyncio.Event()
        ran: list[UUID] = []

        async def held_round(
            namespace: str, config: VectorStoreCollectionConfig, incarnation: UUID
        ) -> bool:
            ran.append(incarnation)
            running.set()
            await release.wait()
            return False

        async def quick_round(
            namespace: str, config: VectorStoreCollectionConfig, incarnation: UUID
        ) -> bool:
            ran.append(incarnation)
            return False

        holding = asyncio.create_task(first.run_purge_round(held_round))
        await running.wait()
        waiting = asyncio.create_task(second.run_purge_round(quick_round))
        await asyncio.sleep(0.5)
        assert not waiting.done(), "the second purger claimed during the first's round"

        release.set()
        assert await asyncio.wait_for(holding, 30)
        assert await asyncio.wait_for(waiting, 30)
        assert ran == [older, newer]
        assert await _queued(first) == []
    finally:
        for engine in engines:
            await engine.dispose()


@pytest.mark.asyncio
async def test_racing_deletions_of_a_collection_queue_one_tombstone(
    sqlalchemy_engine, vector_store_name
):
    """Racing deletions serialize on the collection's row: the losers delete
    nothing, one tombstone is queued, and the name can be created again."""
    registries = [
        await _registry(sqlalchemy_engine, vector_store_name) for _ in range(4)
    ]
    for cycle in range(10):
        await registries[0].reserve(NAMESPACE, "c", CONFIG)
        await asyncio.wait_for(
            asyncio.gather(*(r.unregister(NAMESPACE, "c") for r in registries)), 30
        )
        assert await registries[0].resolve(NAMESPACE, "c") is None
        assert len(await _queued(registries[0])) == cycle + 1
    await registries[0].reserve(NAMESPACE, "c", CONFIG)


@pytest.mark.asyncio
async def test_a_deletion_racing_another_waits_and_queues_nothing_more(
    sqlalchemy_engine, vector_store_name
):
    """A deletion that finds another process's deletion in flight waits for
    it, then finds nothing to delete: one tombstone, and no error."""
    if sqlalchemy_engine.dialect.name != "postgresql":
        pytest.skip("SQLite serializes whole write transactions")
    registry = await _registry(sqlalchemy_engine, vector_store_name)
    incarnation = (await registry.reserve(NAMESPACE, "c", CONFIG)).incarnation

    local = None
    try:
        async with sqlalchemy_engine.connect() as remote, remote.begin():
            # Another process's deletion, held uncommitted.
            await remote.execute(
                delete(CollectionRow).where(CollectionRow.incarnation == incarnation)
            )
            await remote.execute(
                insert(PurgeQueueRow).values(
                    incarnation=incarnation,
                    vector_store_name=vector_store_name,
                    namespace=NAMESPACE,
                    name="c",
                    config=CONFIG.model_dump(mode="json"),
                    enqueued_at=func.now(),
                )
            )
            local = asyncio.create_task(registry.unregister(NAMESPACE, "c"))
            assert await _blocked_or_done(sqlalchemy_engine, local) == "blocked"
        await asyncio.wait_for(local, 30)
    finally:
        if local is not None and not local.done():
            local.cancel()

    assert await _queued(registry) == [incarnation]


@pytest.mark.asyncio
async def test_an_incarnation_colliding_with_a_live_collection_is_reminted(
    sqlalchemy_engine, vector_store_name, monkeypatch
):
    """A mint rejected by the incarnation's unique constraint, with the name
    free, is a collision to mint again, not a taken name."""
    registry = await _registry(sqlalchemy_engine, vector_store_name)
    live = (
        await (await registry.reserve(NAMESPACE, "live", CONFIG)).confirm()
    ).incarnation
    minted = iter([live, uuid4()])
    monkeypatch.setattr(
        "memmachine_server.common.vector_store.collection_registry.sqlalchemy_collection_registry.uuid4",
        lambda: next(minted),
    )

    fresh = (await registry.reserve(NAMESPACE, "fresh", CONFIG)).incarnation

    assert fresh != live
    resolved = await registry.resolve(NAMESPACE, "live")
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
    collection under an incarnation awaiting purge.
    """
    registry = await _registry(sqlalchemy_engine, vector_store_name)
    victim = (await registry.reserve(NAMESPACE, "victim", CONFIG)).incarnation
    minted = iter([victim, uuid4()])
    monkeypatch.setattr(
        "memmachine_server.common.vector_store.collection_registry.sqlalchemy_collection_registry.uuid4",
        lambda: next(minted),
    )
    insert_issued = asyncio.Event()

    def on_statement(_connection, _cursor, statement, _parameters, _context, _many):
        if statement.startswith(f"INSERT INTO {CollectionRow.__tablename__}"):
            insert_issued.set()

    event.listen(sqlalchemy_engine.sync_engine, "before_cursor_execute", on_statement)
    creator = None
    try:
        async with sqlalchemy_engine.connect() as remote, remote.begin():
            await remote.execute(
                delete(CollectionRow).where(CollectionRow.incarnation == victim)
            )
            await remote.execute(
                insert(PurgeQueueRow).values(
                    incarnation=victim,
                    vector_store_name=vector_store_name,
                    namespace=NAMESPACE,
                    name="victim",
                    config=CONFIG.model_dump(mode="json"),
                    enqueued_at=func.now(),
                )
            )
            creator = asyncio.create_task(registry.reserve(NAMESPACE, "fresh", CONFIG))
            await asyncio.wait_for(insert_issued.wait(), 30)
        fresh = (await asyncio.wait_for(creator, 30)).incarnation
    finally:
        event.remove(
            sqlalchemy_engine.sync_engine, "before_cursor_execute", on_statement
        )
        if creator is not None and not creator.done():
            creator.cancel()

    assert fresh != victim, "a live collection took an incarnation awaiting purge"
    assert await _queued(registry) == [victim]


@pytest.mark.asyncio
async def test_a_round_claims_one_tombstone(sqlalchemy_engine, vector_store_name):
    """A claim takes one tombstone and leaves the others to other purgers."""
    registry = await _registry(
        sqlalchemy_engine, vector_store_name, tombstone_retention_seconds=0
    )
    for name in ("a", "b", "c"):
        await registry.reserve(NAMESPACE, name, CONFIG)
        await registry.unregister(NAMESPACE, name)

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
        "memmachine_server.common.vector_store.collection_registry.sqlalchemy_collection_registry.uuid4",
        lambda: None,
    )

    with pytest.raises(VectorStoreAttemptsExhaustedError) as raised:
        await asyncio.wait_for(registry.reserve(NAMESPACE, "a", CONFIG), 30)

    cause: BaseException | None = raised.value
    while cause is not None and not isinstance(cause, IntegrityError):
        cause = cause.__cause__
    assert isinstance(cause, IntegrityError)
