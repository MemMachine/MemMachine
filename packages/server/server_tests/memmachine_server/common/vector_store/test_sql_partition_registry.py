"""The SQL partition registry: creation arbitrated by the database, deletion queued, purge claimed."""

import asyncio
from datetime import timedelta
from uuid import UUID, uuid4

import pytest
from sqlalchemy import DateTime, func, select, update
from sqlalchemy.ext.asyncio import AsyncEngine

from memmachine_server.common.vector_store.data_types import (
    PartitionSchema,
    VectorStoreAttemptsExhaustedError,
    VectorStorePartitionAlreadyExistsError,
)
from memmachine_server.common.vector_store.sql_partition_registry import (
    SqlPartitionRegistry,
)

SCHEMA = PartitionSchema(vector_dimensions=3, indexed_properties={"name": "str"})
OTHER_SCHEMA = PartitionSchema(vector_dimensions=4, indexed_properties={})
PREFIX = "vector_store_test"
RETENTION = timedelta(days=1)


@pytest.fixture
def collection() -> str:
    """A collection name no other test (or earlier run on a shared server) used."""
    return f"c_{uuid4().hex}"


async def _registry(engine: AsyncEngine, collection: str) -> SqlPartitionRegistry:
    registry = SqlPartitionRegistry(
        engine=engine,
        table_prefix=PREFIX,
        collection=collection,
        tombstone_retention=RETENTION,
    )
    await registry.provision()
    return registry


async def _queued(registry: SqlPartitionRegistry) -> list[UUID]:
    """The registry's collection's tombstones, oldest first."""
    async with registry._engine.connect() as connection:
        rows = await connection.execute(
            select(registry._purge_queue.c.incarnation)
            .where(registry._purge_queue.c.collection == registry._collection)
            .order_by(registry._purge_queue.c.enqueued_at)
        )
        return list(rows.scalars())


async def _clean_rounds(registry: SqlPartitionRegistry) -> dict[UUID, bool]:
    """Each tombstone of the registry's collection, and whether a round has found it clean."""
    async with registry._engine.connect() as connection:
        rows = await connection.execute(
            select(
                registry._purge_queue.c.incarnation, registry._purge_queue.c.clean_at
            ).where(registry._purge_queue.c.collection == registry._collection)
        )
        return {row.incarnation: row.clean_at is not None for row in rows}


async def _age_clean_round(registry: SqlPartitionRegistry, incarnation: UUID) -> None:
    """Move the tombstone's clean round back past the retention, on the database clock."""
    async with registry._engine.begin() as connection:
        database_now = (
            await connection.execute(select(func.now(type_=DateTime(timezone=True))))
        ).scalar_one()
        await connection.execute(
            update(registry._purge_queue)
            .where(registry._purge_queue.c.incarnation == incarnation)
            .values(clean_at=database_now - RETENTION - timedelta(seconds=1))
        )


async def _round(registry: SqlPartitionRegistry, found: bool) -> UUID | None:
    """One purge round on the oldest due tombstone, reporting `found`; its incarnation, or None."""
    async with registry.claim_oldest() as claim:
        if claim is None:
            return None
        claim.found = found
        return claim.incarnation


@pytest.mark.asyncio
async def test_create_registers_a_fresh_incarnation_with_the_schema(
    sqlalchemy_engine, collection
):
    registry = await _registry(sqlalchemy_engine, collection)

    incarnation = await registry.create("p", SCHEMA)

    registered = await registry.get("p")
    assert registered is not None
    assert registered.incarnation == incarnation
    assert registered.schema == SCHEMA
    assert await registry.is_live(incarnation)
    assert await registry.get("q") is None


@pytest.mark.asyncio
async def test_a_taken_key_is_already_exists_whatever_the_schema(
    sqlalchemy_engine, collection
):
    registry = await _registry(sqlalchemy_engine, collection)
    await registry.create("p", SCHEMA)

    with pytest.raises(VectorStorePartitionAlreadyExistsError):
        await registry.create("p", SCHEMA)
    with pytest.raises(VectorStorePartitionAlreadyExistsError):
        await registry.create("p", OTHER_SCHEMA)


@pytest.mark.asyncio
async def test_collections_keep_apart_in_one_table(sqlalchemy_engine, collection):
    first = await _registry(sqlalchemy_engine, f"{collection}_1")
    second = await _registry(sqlalchemy_engine, f"{collection}_2")

    one = await first.create("p", SCHEMA)
    two = await second.create("p", OTHER_SCHEMA)

    assert one != two
    assert await first.is_live(one)
    assert await second.is_live(two)
    await first.delete("p")
    assert await first.get("p") is None
    assert not await first.is_live(one)
    assert await second.is_live(two)


@pytest.mark.asyncio
async def test_concurrent_creators_get_one_winner(sqlalchemy_engine, collection):
    """Two registries on one database, the two-process shape in one process."""
    first = await _registry(sqlalchemy_engine, collection)
    second = await _registry(sqlalchemy_engine, collection)

    results = await asyncio.gather(
        *(registry.create("p", SCHEMA) for registry in (first, second) * 3),
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
    sqlalchemy_engine, collection
):
    registry = await _registry(sqlalchemy_engine, collection)
    incarnation = await registry.create("p", SCHEMA)

    await registry.delete("p")

    assert await registry.get("p") is None
    assert not await registry.is_live(incarnation)
    assert await _queued(registry) == [incarnation]
    await registry.delete("p")
    await registry.delete("never")
    assert await _queued(registry) == [incarnation]


@pytest.mark.asyncio
async def test_a_recreated_key_gets_a_new_incarnation(sqlalchemy_engine, collection):
    registry = await _registry(sqlalchemy_engine, collection)
    old = await registry.create("p", SCHEMA)
    await registry.delete("p")

    new = await registry.create("p", SCHEMA)

    assert new != old
    assert await registry.is_live(new)
    assert not await registry.is_live(old)


@pytest.mark.asyncio
async def test_rounds_go_oldest_first_and_a_clean_round_stamps_the_tombstone(
    sqlalchemy_engine, collection
):
    registry = await _registry(sqlalchemy_engine, collection)
    first = await registry.create("a", SCHEMA)
    second = await registry.create("b", SCHEMA)
    await registry.delete("a")
    await registry.delete("b")

    assert await _round(registry, found=True) == first
    # Found points: the tombstone stays due, and it is still the oldest.
    assert await _round(registry, found=False) == first
    assert await _round(registry, found=False) == second
    # Both had a clean round less than the retention ago: nothing is due.
    assert await _round(registry, found=False) is None
    assert await _queued(registry) == [first, second]
    assert await _clean_rounds(registry) == {first: True, second: True}


@pytest.mark.asyncio
async def test_a_tombstone_is_removed_by_a_clean_round_after_the_retention(
    sqlalchemy_engine, collection
):
    registry = await _registry(sqlalchemy_engine, collection)
    incarnation = await registry.create("a", SCHEMA)
    await registry.delete("a")
    assert await _round(registry, found=False) == incarnation
    await _age_clean_round(registry, incarnation)

    assert await _round(registry, found=False) == incarnation

    assert await _queued(registry) == []


@pytest.mark.asyncio
async def test_a_late_write_found_after_the_retention_restarts_the_rounds(
    sqlalchemy_engine, collection
):
    registry = await _registry(sqlalchemy_engine, collection)
    incarnation = await registry.create("a", SCHEMA)
    await registry.delete("a")
    assert await _round(registry, found=False) == incarnation
    await _age_clean_round(registry, incarnation)

    assert await _round(registry, found=True) == incarnation

    assert await _clean_rounds(registry) == {incarnation: False}
    assert await _round(registry, found=False) == incarnation
    assert await _clean_rounds(registry) == {incarnation: True}
    assert await _round(registry, found=False) is None
    assert await _queued(registry) == [incarnation]


@pytest.mark.asyncio
async def test_a_round_whose_body_raises_keeps_the_tombstone_as_it_was(
    sqlalchemy_engine, collection
):
    registry = await _registry(sqlalchemy_engine, collection)
    incarnation = await registry.create("a", SCHEMA)
    await registry.delete("a")

    async def refused_reclamation() -> None:
        async with registry.claim_oldest() as claim:
            assert claim is not None
            assert claim.incarnation == incarnation
            claim.found = True
            raise RuntimeError("the backend refused")

    with pytest.raises(RuntimeError):
        await refused_reclamation()

    assert await _clean_rounds(registry) == {incarnation: False}
    assert await _round(registry, found=False) == incarnation
    assert await _clean_rounds(registry) == {incarnation: True}


@pytest.mark.asyncio
async def test_a_round_that_reports_nothing_is_an_error(sqlalchemy_engine, collection):
    registry = await _registry(sqlalchemy_engine, collection)
    await registry.create("a", SCHEMA)
    await registry.delete("a")

    with pytest.raises(RuntimeError, match="without reporting"):
        async with registry.claim_oldest():
            pass


@pytest.mark.asyncio
async def test_a_claim_is_scoped_to_its_collection(sqlalchemy_engine, collection):
    first = await _registry(sqlalchemy_engine, f"{collection}_1")
    second = await _registry(sqlalchemy_engine, f"{collection}_2")
    dead = await second.create("p", SCHEMA)
    await second.delete("p")

    assert await _round(first, found=False) is None
    assert await _round(second, found=False) == dead


@pytest.mark.asyncio
async def test_an_incarnation_awaiting_purge_is_never_reminted(
    sqlalchemy_engine, collection, monkeypatch
):
    """A minted incarnation colliding with queued garbage is rejected and re-minted."""
    registry = await _registry(sqlalchemy_engine, collection)
    dead = await registry.create("a", SCHEMA)
    await registry.delete("a")
    minted = iter([dead, uuid4()])
    monkeypatch.setattr(
        "memmachine_server.common.vector_store.sql_partition_registry.uuid4",
        lambda: next(minted),
    )

    fresh = await registry.create("b", SCHEMA)

    assert fresh != dead
    assert await _queued(registry) == [dead]


@pytest.mark.asyncio
async def test_creation_that_never_mints_a_free_incarnation_gives_up(
    sqlalchemy_engine, collection, monkeypatch
):
    registry = await _registry(sqlalchemy_engine, collection)
    dead = await registry.create("a", SCHEMA)
    await registry.delete("a")
    monkeypatch.setattr(
        "memmachine_server.common.vector_store.sql_partition_registry.uuid4",
        lambda: dead,
    )

    with pytest.raises(VectorStoreAttemptsExhaustedError):
        await registry.create("b", SCHEMA)
    assert await registry.get("b") is None
