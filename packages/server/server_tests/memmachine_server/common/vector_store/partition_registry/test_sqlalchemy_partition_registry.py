"""The SQLAlchemy partition registry of one vector store: creation arbitrated by the database, deletion queued, purge claimed."""

import asyncio
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

from memmachine_server.common.vector_store.data_types import (
    PartitionSchema,
    VectorStoreAttemptsExhaustedError,
    VectorStorePartitionAlreadyExistsError,
)
from memmachine_server.common.vector_store.partition_registry.sqlalchemy_partition_registry import (
    SQLAlchemyVectorStorePartitionRegistry,
)

SCHEMA = PartitionSchema(
    vector_dimensions=3,
    indexed_properties={"name": "str"},
)
OTHER_SCHEMA = PartitionSchema(
    vector_dimensions=4,
    indexed_properties={},
)
RETENTION = timedelta(days=1)


@pytest.fixture
def vector_store_name() -> str:
    """A vector store name no other test (or earlier run on a shared server) used."""
    return f"test_{uuid4().hex[:12]}"


async def _registry(
    engine: AsyncEngine,
    vector_store_name: str,
    tombstone_retention: timedelta = RETENTION,
) -> SQLAlchemyVectorStorePartitionRegistry:
    registry = SQLAlchemyVectorStorePartitionRegistry(
        engine=engine,
        vector_store_name=vector_store_name,
        tombstone_retention=tombstone_retention,
    )
    await registry.provision()
    return registry


async def _queued(registry: SQLAlchemyVectorStorePartitionRegistry) -> list[UUID]:
    """The registry's tombstones, oldest first."""
    async with registry._engine.connect() as connection:
        rows = await connection.execute(
            select(registry._purge_queue.c.incarnation).order_by(
                registry._purge_queue.c.enqueued_at
            )
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
            update(registry._purge_queue)
            .where(registry._purge_queue.c.incarnation == incarnation)
            .values(
                enqueued_at=_database_time_ago(
                    registry._engine, RETENTION + timedelta(seconds=1) + extra
                )
            )
        )


async def _blocked_or_done(engine: AsyncEngine, task: asyncio.Task) -> str:
    """Wait until `task` finishes ("done") or another backend waits on a lock ("blocked").

    Decided by the database's own state (pg_stat_activity), not elapsed time.
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


async def _round(
    registry: SQLAlchemyVectorStorePartitionRegistry, found: bool
) -> UUID | None:
    """One purge round on the oldest due tombstone, reporting `found`; its incarnation, or None."""
    async with registry.claim_due() as claim:
        if claim is None:
            return None
        claim.found = found
        return claim.incarnation


@pytest.mark.asyncio
async def test_create_registers_a_fresh_incarnation_with_the_schema(
    sqlalchemy_engine, vector_store_name
):
    registry = await _registry(sqlalchemy_engine, vector_store_name)

    incarnation = await registry.register("c", SCHEMA)

    registered = await registry.get("c")
    assert registered is not None
    assert registered.incarnation == incarnation
    assert registered.schema == SCHEMA
    assert await registry.is_live(incarnation)
    assert await registry.get("d") is None


@pytest.mark.asyncio
async def test_a_taken_key_is_already_exists_whatever_the_schema(
    sqlalchemy_engine, vector_store_name
):
    registry = await _registry(sqlalchemy_engine, vector_store_name)
    await registry.register("c", SCHEMA)

    with pytest.raises(VectorStorePartitionAlreadyExistsError):
        await registry.register("c", SCHEMA)
    with pytest.raises(VectorStorePartitionAlreadyExistsError):
        await registry.register("c", OTHER_SCHEMA)


@pytest.mark.parametrize(
    "invalid_name", ["", "Upper", "with-hyphen", "trailing_newline\n", "x" * 33]
)
def test_a_vector_store_name_must_be_an_identifier(invalid_name):
    with pytest.raises(ValueError, match="Vector store name"):
        SQLAlchemyVectorStorePartitionRegistry(
            engine=create_async_engine("sqlite+aiosqlite://"),
            vector_store_name=invalid_name,
            tombstone_retention=RETENTION,
        )


def test_an_engine_of_another_dialect_is_refused(monkeypatch):
    engine = create_async_engine("sqlite+aiosqlite://")
    monkeypatch.setattr(engine.dialect, "name", "mssql")
    with pytest.raises(ValueError, match="mssql"):
        SQLAlchemyVectorStorePartitionRegistry(
            engine=engine, vector_store_name="store", tombstone_retention=RETENTION
        )


@pytest.mark.asyncio
async def test_registries_of_two_vector_stores_share_a_database_and_nothing_else(
    sqlalchemy_engine, vector_store_name
):
    """Each registry has its own partitions, and a purge round claims
    only its own tombstones."""
    first = await _registry(sqlalchemy_engine, f"{vector_store_name}_a")
    second = await _registry(sqlalchemy_engine, f"{vector_store_name}_b")

    one = await first.register("c", SCHEMA)
    two = await second.register("c", OTHER_SCHEMA)
    assert one != two
    in_first = await first.get("c")
    in_second = await second.get("c")
    assert in_first is not None
    assert in_second is not None
    assert in_first.incarnation == one
    assert in_second.incarnation == two

    await first.unregister("c")
    await _age_deletion(first, one)
    assert await second.is_live(two)
    assert await _round(second, found=False) is None
    assert await _round(first, found=False) == one


@pytest.mark.asyncio
async def test_concurrent_creators_get_one_winner(sqlalchemy_engine, vector_store_name):
    """Two registries on one database, the two-process shape in one process."""
    first = await _registry(sqlalchemy_engine, vector_store_name)
    second = await _registry(sqlalchemy_engine, vector_store_name)

    results = await asyncio.gather(
        *(registry.register("c", SCHEMA) for registry in (first, second) * 3),
        return_exceptions=True,
    )

    winners = [r for r in results if isinstance(r, UUID)]
    losers = [
        r for r in results if isinstance(r, VectorStorePartitionAlreadyExistsError)
    ]
    assert len(winners) == 1
    assert len(losers) == 5
    assert await first.is_live(winners[0])


@pytest.mark.asyncio
async def test_delete_queues_the_incarnation_and_is_idempotent(
    sqlalchemy_engine, vector_store_name
):
    registry = await _registry(sqlalchemy_engine, vector_store_name)
    incarnation = await registry.register("c", SCHEMA)

    await registry.unregister("c")

    assert await registry.get("c") is None
    assert not await registry.is_live(incarnation)
    assert await _queued(registry) == [incarnation]
    await registry.unregister("c")
    await registry.unregister("never")
    assert await _queued(registry) == [incarnation]


@pytest.mark.asyncio
async def test_a_recreated_key_gets_a_new_incarnation(
    sqlalchemy_engine, vector_store_name
):
    registry = await _registry(sqlalchemy_engine, vector_store_name)
    old = await registry.register("c", SCHEMA)
    await registry.unregister("c")

    new = await registry.register("c", SCHEMA)

    assert new != old
    assert await registry.is_live(new)
    assert not await registry.is_live(old)


@pytest.mark.asyncio
async def test_a_claim_names_where_the_points_are(sqlalchemy_engine, vector_store_name):
    """The tombstone names the incarnation to purge and the key it was under."""
    registry = await _registry(sqlalchemy_engine, vector_store_name)
    incarnation = await registry.register("c", OTHER_SCHEMA)
    await registry.unregister("c")
    await _age_deletion(registry, incarnation)

    async with registry.claim_due() as claim:
        assert claim is not None
        assert claim.incarnation == incarnation
        claim.found = False


@pytest.mark.asyncio
async def test_a_tombstone_is_not_due_before_the_retention(
    sqlalchemy_engine, vector_store_name
):
    registry = await _registry(sqlalchemy_engine, vector_store_name)
    incarnation = await registry.register("a", SCHEMA)
    await registry.unregister("a")

    assert await _round(registry, found=False) is None
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
    first = await registry.register("a", SCHEMA)
    second = await registry.register("b", SCHEMA)
    await registry.unregister("a")
    await registry.unregister("b")
    # The later deletion, with the larger incarnation, is made the older, so
    # only the ordering by deletion can put it first.
    await _age_deletion(registry, first)
    await _age_deletion(registry, second, extra=timedelta(minutes=1))

    assert await _round(registry, found=True) == second
    # Found points: still due, and still the oldest deletion.
    assert await _round(registry, found=False) == second
    assert await _queued(registry) == [first]
    assert await _round(registry, found=False) == first
    assert await _queued(registry) == []
    assert await _round(registry, found=False) is None


@pytest.mark.asyncio
async def test_a_tombstone_queued_by_a_real_deletion_comes_due_by_the_database_clock(
    sqlalchemy_engine, vector_store_name
):
    """The claim compares the stamp the database clock wrote, by the
    database's arithmetic: with no retention the first round claims the
    tombstone, and no stamp is rewritten by the test."""
    registry = await _registry(
        sqlalchemy_engine, vector_store_name, tombstone_retention=timedelta(0)
    )
    incarnation = await registry.register("a", SCHEMA)
    await registry.unregister("a")

    assert await _round(registry, found=False) == incarnation

    assert await _queued(registry) == []


@pytest.mark.asyncio
async def test_a_changed_retention_applies_to_tombstones_already_queued(
    sqlalchemy_engine, vector_store_name
):
    """Only the deletion's time is stored; the retention in force when a
    claim is decided applies, including to tombstones queued under another."""
    queued_under_a_day = await _registry(sqlalchemy_engine, vector_store_name)
    incarnation = await queued_under_a_day.register("a", SCHEMA)
    await queued_under_a_day.unregister("a")
    assert await _round(queued_under_a_day, found=False) is None

    no_retention = await _registry(
        sqlalchemy_engine, vector_store_name, tombstone_retention=timedelta(0)
    )
    assert await _round(no_retention, found=False) == incarnation
    assert await _queued(no_retention) == []


@pytest.mark.asyncio
async def test_a_round_whose_body_raises_keeps_the_tombstone_as_it_was(
    sqlalchemy_engine, vector_store_name
):
    registry = await _registry(sqlalchemy_engine, vector_store_name)
    incarnation = await registry.register("a", SCHEMA)
    await registry.unregister("a")
    await _age_deletion(registry, incarnation)

    async def refused_reclamation() -> None:
        async with registry.claim_due() as claim:
            assert claim is not None
            assert claim.incarnation == incarnation
            claim.found = False
            raise RuntimeError("the backend refused")

    with pytest.raises(RuntimeError):
        await refused_reclamation()

    assert await _queued(registry) == [incarnation]


@pytest.mark.asyncio
async def test_a_round_that_reports_nothing_is_an_error(
    sqlalchemy_engine, vector_store_name
):
    registry = await _registry(sqlalchemy_engine, vector_store_name)
    incarnation = await registry.register("a", SCHEMA)
    await registry.unregister("a")
    await _age_deletion(registry, incarnation)

    with pytest.raises(RuntimeError, match="without reporting"):
        async with registry.claim_due():
            pass
    assert await _queued(registry) == [incarnation]


@pytest.mark.asyncio
async def test_an_incarnation_awaiting_purge_is_never_reminted(
    sqlalchemy_engine, vector_store_name, monkeypatch
):
    """A minted incarnation colliding with queued garbage is rejected and re-minted."""
    registry = await _registry(sqlalchemy_engine, vector_store_name)
    dead = await registry.register("a", SCHEMA)
    await registry.unregister("a")
    minted = iter([dead, uuid4()])
    monkeypatch.setattr(
        "memmachine_server.common.vector_store.partition_registry.sqlalchemy_partition_registry.uuid4",
        lambda: next(minted),
    )

    fresh = await registry.register("b", SCHEMA)

    assert fresh != dead
    assert await _queued(registry) == [dead]


@pytest.mark.asyncio
async def test_creation_that_never_mints_a_free_incarnation_gives_up(
    sqlalchemy_engine, vector_store_name, monkeypatch
):
    registry = await _registry(sqlalchemy_engine, vector_store_name)
    dead = await registry.register("a", SCHEMA)
    await registry.unregister("a")
    monkeypatch.setattr(
        "memmachine_server.common.vector_store.partition_registry.sqlalchemy_partition_registry.uuid4",
        lambda: dead,
    )

    with pytest.raises(VectorStoreAttemptsExhaustedError):
        await registry.register("b", SCHEMA)
    assert await registry.get("b") is None


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
        incarnation = await registry.register("a", SCHEMA)
        await registry.unregister("a")

        queue = registry._purge_queue
        async with skewed_engine.connect() as connection:
            row = (
                await connection.execute(
                    select(queue.c.enqueued_at, text("pg_catalog.now() AS real_now"))
                )
            ).one()
        assert row.real_now - row.enqueued_at > timedelta(days=9)
        assert await _round(registry, found=False) is None
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
    held = await registry.register("held", SCHEMA)
    free = await registry.register("free", SCHEMA)
    await registry.unregister("held")
    await registry.unregister("free")
    await _age_deletion(registry, held, extra=timedelta(minutes=1))
    await _age_deletion(registry, free)

    queue = registry._purge_queue
    claim = None
    try:
        async with sqlalchemy_engine.connect() as other_purger, other_purger.begin():
            await other_purger.execute(
                select(queue.c.incarnation)
                .where(queue.c.incarnation == held)
                .with_for_update()
            )
            claim = asyncio.create_task(_round(registry, found=False))
            outcome = await _blocked_or_done(sqlalchemy_engine, claim)
            assert outcome == "done", (
                "the claim waited on a tombstone another purger holds"
            )
            assert await claim == free
    finally:
        if claim is not None and not claim.done():
            await asyncio.wait_for(claim, 30)


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
        await registries[0].register("c", SCHEMA)
        await asyncio.wait_for(
            asyncio.gather(*(r.unregister("c") for r in registries)), 30
        )
        assert await registries[0].get("c") is None
        assert len(await _queued(registries[0])) == cycle + 1
    await registries[0].register("c", SCHEMA)


@pytest.mark.asyncio
async def test_a_deletion_racing_another_waits_and_queues_nothing_more(
    sqlalchemy_engine, vector_store_name
):
    """A deletion that finds another process's deletion in flight waits for
    it, then finds nothing to delete: one tombstone, and no error."""
    if sqlalchemy_engine.dialect.name != "postgresql":
        pytest.skip("SQLite serializes whole write transactions")
    registry = await _registry(sqlalchemy_engine, vector_store_name)
    incarnation = await registry.register("c", SCHEMA)
    partitions, queue = registry._partitions, registry._purge_queue

    local = None
    try:
        async with sqlalchemy_engine.connect() as remote, remote.begin():
            # Another process's deletion, held uncommitted.
            await remote.execute(
                delete(partitions).where(partitions.c.partition_key == "c")
            )
            await remote.execute(
                insert(queue).values(
                    incarnation=incarnation,
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
    live = await registry.register("live", SCHEMA)
    minted = iter([live, uuid4()])
    monkeypatch.setattr(
        "memmachine_server.common.vector_store.partition_registry.sqlalchemy_partition_registry.uuid4",
        lambda: next(minted),
    )

    fresh = await registry.register("fresh", SCHEMA)

    assert fresh != live
    registered = await registry.get("live")
    assert registered is not None
    assert registered.incarnation == live


@pytest.mark.asyncio
async def test_a_mint_checks_the_queue_after_its_insert(
    sqlalchemy_engine, vector_store_name, monkeypatch
):
    """A mint colliding with a deletion in flight sees the deletion's tombstone.

    The deletion frees the incarnation's row and queues its tombstone in one
    uncommitted transaction; the colliding mint's insert waits on it. Only a
    queue check made after the insert sees the tombstone once the deletion
    commits: one made before reads the queue too early and registers a live
    partition under an incarnation awaiting purge.
    """
    registry = await _registry(sqlalchemy_engine, vector_store_name)
    victim = await registry.register("victim", SCHEMA)
    minted = iter([victim, uuid4()])
    monkeypatch.setattr(
        "memmachine_server.common.vector_store.partition_registry.sqlalchemy_partition_registry.uuid4",
        lambda: next(minted),
    )
    partitions, queue = registry._partitions, registry._purge_queue
    insert_issued = asyncio.Event()

    def on_statement(_connection, _cursor, statement, _parameters, _context, _many):
        if statement.startswith(f"INSERT INTO {partitions.name}"):
            insert_issued.set()

    event.listen(sqlalchemy_engine.sync_engine, "before_cursor_execute", on_statement)
    creator = None
    try:
        async with sqlalchemy_engine.connect() as remote, remote.begin():
            await remote.execute(
                delete(partitions).where(partitions.c.incarnation == victim)
            )
            await remote.execute(
                insert(queue).values(
                    incarnation=victim,
                    partition_key="victim",
                    enqueued_at=func.now(),
                )
            )
            creator = asyncio.create_task(registry.register("fresh", SCHEMA))
            await asyncio.wait_for(insert_issued.wait(), 30)
        fresh = await asyncio.wait_for(creator, 30)
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
        sqlalchemy_engine, vector_store_name, tombstone_retention=timedelta(0)
    )
    for name in ("a", "b", "c"):
        await registry.register(name, SCHEMA)
        await registry.unregister(name)

    assert await _round(registry, found=False) is not None

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
        await asyncio.wait_for(registry.register("a", SCHEMA), 30)

    cause: BaseException | None = raised.value
    while cause is not None and not isinstance(cause, IntegrityError):
        cause = cause.__cause__
    assert isinstance(cause, IntegrityError)
