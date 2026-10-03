"""
The partition lifecycle contract the registry-backed vector stores satisfy.

A store's test module mixes `PartitionLifecycleContract` into a test class
and supplies a `store` fixture and three hooks: `count_stored` and
`stored_uuids` read the backend directly, and `settle` returns once the
store's reads reflect every write so far. The backend may persist across
tests, so each test first deletes the partitions it uses, and the purge
tests count against a drained baseline.
"""

import asyncio
import math
import random
from uuid import UUID, uuid4

import pytest

from memmachine_server.common.vector_store import (
    Record,
    VectorStoreAttemptsExhaustedError,
    VectorStorePartitionAlreadyExistsError,
    VectorStorePartitionDeletedError,
    VectorStorePartitionHandleStaleError,
    VectorStorePartitionPendingError,
    VectorStorePartitionSchemaMismatchError,
    registry_backed_vector_store,
)

LIFECYCLE_KEY = "lifecycle"


def _unit(vector: list[float]) -> list[float]:
    magnitude = math.sqrt(sum(x * x for x in vector))
    return [x / magnitude for x in vector]


def _records(count: int) -> list[Record]:
    return [
        Record(uuid=uuid4(), vector=_unit([1.0, 0.01 * index, 0.0]))
        for index in range(count)
    ]


async def _fresh(store, key: str):
    """A new partition under `key`, whatever a persisted backend held for it."""
    await store.delete_partition(key)
    await store.create_partition(key)
    partition = await store.get_partition(key)
    assert partition is not None
    return partition


class PartitionLifecycleContract:
    """Mixed into a store's test class, which supplies a `store` fixture and the `count_stored`, `stored_uuids` and `settle` hooks."""

    @staticmethod
    async def count_stored(store) -> int:
        """Records the backend holds for the store, deleted partitions' included."""
        raise NotImplementedError

    @staticmethod
    async def stored_uuids(partition) -> set[UUID]:
        """Record UUIDs the backend holds under the handle's incarnation."""
        raise NotImplementedError

    @staticmethod
    async def settle(partition) -> None:
        """Return once the store's reads reflect every write made so far."""
        raise NotImplementedError

    async def _drained_count(self, store) -> int:
        """`count_stored` once nothing deleted is left to reclaim."""
        while await store.purge_deleted_partitions():
            pass
        return await self.count_stored(store)

    @pytest.mark.asyncio
    async def test_a_handle_is_stale_once_its_partition_is_deleted(self, store):
        collection = await _fresh(store, LIFECYCLE_KEY)
        record = _records(1)[0]
        await collection.upsert(records=[record])

        await store.delete_partition(LIFECYCLE_KEY)

        assert await store.get_partition(LIFECYCLE_KEY) is None
        with pytest.raises(VectorStorePartitionHandleStaleError, match=LIFECYCLE_KEY):
            await collection.upsert(records=[record])
        with pytest.raises(VectorStorePartitionHandleStaleError, match=LIFECYCLE_KEY):
            await collection.query(query_vectors=[record.vector], limit=5)
        with pytest.raises(VectorStorePartitionHandleStaleError, match=LIFECYCLE_KEY):
            await collection.delete(record_uuids=[record.uuid])
        # So does an operation with nothing to send to the backend.
        with pytest.raises(VectorStorePartitionHandleStaleError, match=LIFECYCLE_KEY):
            await collection.upsert(records=[])
        with pytest.raises(VectorStorePartitionHandleStaleError, match=LIFECYCLE_KEY):
            await collection.query(query_vectors=[], limit=5)
        with pytest.raises(VectorStorePartitionHandleStaleError, match=LIFECYCLE_KEY):
            await collection.delete(record_uuids=[])

    @pytest.mark.asyncio
    async def test_a_recreated_partition_starts_empty_and_the_old_handle_stays_stale(
        self, store
    ):
        old = await _fresh(store, LIFECYCLE_KEY)
        old_record, new_record = _records(2)
        await old.upsert(records=[old_record])

        await store.delete_partition(LIFECYCLE_KEY)
        new = await _fresh(store, LIFECYCLE_KEY)

        [before] = await new.query(query_vectors=[old_record.vector], limit=5)
        assert before.matches == []

        await new.upsert(records=[new_record])
        assert await self.stored_uuids(new) == {new_record.uuid}

        # The old life's handle cannot reach the new life's records.
        with pytest.raises(VectorStorePartitionHandleStaleError):
            await old.query(query_vectors=[new_record.vector], limit=5)
        with pytest.raises(VectorStorePartitionHandleStaleError):
            await old.delete(record_uuids=[new_record.uuid])
        assert await self.stored_uuids(new) == {new_record.uuid}

        await store.delete_partition(LIFECYCLE_KEY)

    @pytest.mark.asyncio
    async def test_open_or_create_adopts_the_live_incarnation(self, store):
        """Opening an existing collection binds to its life; creating one mints a new life."""
        await store.delete_partition(LIFECYCLE_KEY)
        first = await store.open_or_create_partition(LIFECYCLE_KEY)
        second = await store.open_or_create_partition(LIFECYCLE_KEY)
        record = _records(1)[0]
        await first.upsert(records=[record])
        assert await self.stored_uuids(second) == {record.uuid}

        await store.delete_partition(LIFECYCLE_KEY)
        third = await store.open_or_create_partition(LIFECYCLE_KEY)
        [empty] = await third.query(query_vectors=[record.vector], limit=5)
        assert empty.matches == []
        with pytest.raises(VectorStorePartitionHandleStaleError):
            await first.query(query_vectors=[record.vector], limit=5)
        await store.delete_partition(LIFECYCLE_KEY)

    @pytest.mark.asyncio
    async def test_deleting_twice_and_deleting_nothing_are_no_ops(self, store):
        await _fresh(store, LIFECYCLE_KEY)
        await store.delete_partition(LIFECYCLE_KEY)
        await store.delete_partition(LIFECYCLE_KEY)
        assert await store.get_partition(LIFECYCLE_KEY) is None
        await store.delete_partition(f"{LIFECYCLE_KEY}_never_created")

    @pytest.mark.asyncio
    async def test_purge_reclaims_what_deletion_deferred(self, store):
        collection = await _fresh(store, LIFECYCLE_KEY)
        baseline = await self._drained_count(store)
        records = _records(5)
        await collection.upsert(records=records)
        await self.settle(collection)
        assert await self.count_stored(store) == baseline + 5

        await store.delete_partition(LIFECYCLE_KEY)
        # Unreachable at once, and the records are left for the purge:
        # deletion touches the registry alone, whatever the partition holds.
        assert await store.get_partition(LIFECYCLE_KEY) is None
        assert await self.count_stored(store) == baseline + 5

        assert await self._drained_count(store) == baseline
        # Nothing left to claim.
        assert await store.purge_deleted_partitions() is False

    @pytest.mark.asyncio
    async def test_purge_leaves_a_live_partition_alone(self, store):
        live_name = f"{LIFECYCLE_KEY}_live"
        live = await _fresh(store, live_name)
        dead = await _fresh(store, LIFECYCLE_KEY)
        baseline = await self._drained_count(store)
        kept = _records(3)
        gone = _records(3)
        await live.upsert(records=kept)
        await dead.upsert(records=gone)
        await self.settle(live)
        await self.settle(dead)

        await store.delete_partition(LIFECYCLE_KEY)

        assert await self._drained_count(store) == baseline + 3
        assert await self.stored_uuids(live) == {record.uuid for record in kept}
        await store.delete_partition(live_name)
        assert await self._drained_count(store) == baseline

    @pytest.mark.asyncio
    async def test_a_write_landing_under_a_dead_incarnation_raises_and_is_reclaimed(
        self, store, monkeypatch
    ):
        collection = await _fresh(store, LIFECYCLE_KEY)
        baseline = await self._drained_count(store)
        records = _records(2)

        # The collection dies between the handle's check and its write.
        registration_type = type(collection._registration)
        require_current = registration_type.require_current

        async def deleted_once_checked(registration) -> None:
            await require_current(registration)
            await store.delete_partition(registration.partition_key)

        monkeypatch.setattr(registration_type, "require_current", deleted_once_checked)

        with pytest.raises(VectorStorePartitionHandleStaleError, match=LIFECYCLE_KEY):
            await collection.upsert(records=records)
        await self.settle(collection)

        # The write landed, under an incarnation nothing can reach...
        assert await self.count_stored(store) == baseline + 2
        # ...and the tombstone's next round reclaims it.
        assert await store.purge_deleted_partitions() is True
        assert await self._drained_count(store) == baseline

    @pytest.mark.asyncio
    async def test_a_trailing_newline_is_not_part_of_a_valid_partition_key(self, store):
        with pytest.raises(ValueError, match="must match"):
            await store.create_partition(f"{LIFECYCLE_KEY}\n")

    @pytest.mark.asyncio
    async def test_an_upsert_checks_the_registry_twice_and_a_query_or_delete_once(
        self, store, monkeypatch
    ):
        partition = await _fresh(store, LIFECYCLE_KEY)
        record = _records(1)[0]
        checks = 0
        registration_type = type(partition._registration)
        require_current = registration_type.require_current

        async def counted(registration) -> None:
            nonlocal checks
            checks += 1
            await require_current(registration)

        monkeypatch.setattr(registration_type, "require_current", counted)

        await partition.upsert(records=[record])
        assert checks == 2
        checks = 0
        await partition.query(query_vectors=[record.vector], limit=1)
        assert checks == 1
        checks = 0
        await partition.delete(record_uuids=[record.uuid])
        assert checks == 1

    @pytest.mark.asyncio
    async def test_open_or_create_gives_up_after_losing_every_race(
        self, store, monkeypatch
    ):
        """A creator that keeps losing to a winner that keeps vanishing gives
        up after a bounded number of attempts instead of looping."""
        registry = store._partition_registry

        async def lost(partition_key, schema):
            # Yields as a real round trip would, so an unbounded loop fails
            # the timeout below instead of starving the event loop.
            await asyncio.sleep(0)
            raise VectorStorePartitionAlreadyExistsError(
                store.vector_store_name, partition_key
            )

        async def vanished(partition_key) -> None:
            return None

        monkeypatch.setattr(registry, "reserve", lost)
        monkeypatch.setattr(registry, "resolve", vanished)
        monkeypatch.setattr(
            registry_backed_vector_store, "_OPEN_OR_CREATE_RETRY_DELAY_SECONDS", 0
        )

        with pytest.raises(VectorStoreAttemptsExhaustedError):
            await asyncio.wait_for(store.open_or_create_partition(LIFECYCLE_KEY), 30)

    @pytest.mark.asyncio
    @pytest.mark.parametrize("create", ["create_partition", "open_or_create_partition"])
    async def test_a_failed_partition_storage_preparation_frees_the_key(
        self, store, monkeypatch, create
    ):
        """A creation whose storage preparation fails unregisters its
        pending partition: nothing opens, and the key is free again."""
        await store.delete_partition(LIFECYCLE_KEY)

        async def refused(partition_key, incarnation) -> None:
            raise RuntimeError("the backend refused")

        monkeypatch.setattr(store, "_prepare_partition_storage", refused)
        with pytest.raises(RuntimeError, match="refused"):
            await getattr(store, create)(LIFECYCLE_KEY)
        monkeypatch.undo()

        assert await store.get_partition(LIFECYCLE_KEY) is None
        await store.create_partition(LIFECYCLE_KEY)
        await store.delete_partition(LIFECYCLE_KEY)

    @pytest.mark.asyncio
    async def test_open_or_create_opens_the_winner_after_losing_the_create(
        self, store, monkeypatch
    ):
        """A creator whose reservation loses to another process's opens
        the winner's partition, bound to the winner's life."""
        await store.delete_partition(LIFECYCLE_KEY)
        registry = store._partition_registry
        reserve = registry.reserve
        lost_reservations = 0

        async def another_process_wins(partition_key, schema):
            # The other process creates the partition between this caller's
            # lookup and its own reservation.
            nonlocal lost_reservations
            lost_reservations += 1
            reservation = await reserve(partition_key, schema)
            await store._prepare_partition_storage(
                partition_key, reservation.incarnation
            )
            await reservation.confirm()
            raise VectorStorePartitionAlreadyExistsError(
                store.vector_store_name, partition_key
            )

        monkeypatch.setattr(registry, "reserve", another_process_wins)
        partition = await store.open_or_create_partition(LIFECYCLE_KEY)
        monkeypatch.undo()

        # It lost once, then found the winner instead of registering again.
        assert lost_reservations == 1

        winner = await registry.resolve(LIFECYCLE_KEY)
        assert winner is not None
        assert partition._incarnation == winner.incarnation
        record = _records(1)[0]
        await partition.upsert(records=[record])
        opened = await store.get_partition(LIFECYCLE_KEY)
        assert opened is not None
        assert await self.stored_uuids(opened) == {record.uuid}
        await store.delete_partition(LIFECYCLE_KEY)

    @pytest.mark.asyncio
    async def test_open_or_create_refuses_a_winner_of_another_schema(
        self, store, monkeypatch
    ):
        """A creator that loses to a winner registered with another schema
        gets the mismatch, not a handle to the winner's partition."""
        await store.delete_partition(LIFECYCLE_KEY)
        registry = store._partition_registry
        reserve = registry.reserve
        lost_reservations = 0

        async def another_process_wins(partition_key, schema):
            nonlocal lost_reservations
            lost_reservations += 1
            reservation = await reserve(
                partition_key,
                schema.model_copy(
                    update={"vector_dimensions": schema.vector_dimensions + 1}
                ),
            )
            await reservation.confirm()
            raise VectorStorePartitionAlreadyExistsError(
                store.vector_store_name, partition_key
            )

        monkeypatch.setattr(registry, "reserve", another_process_wins)
        with pytest.raises(VectorStorePartitionSchemaMismatchError):
            await store.open_or_create_partition(LIFECYCLE_KEY)
        monkeypatch.undo()
        assert lost_reservations == 1
        await store.delete_partition(LIFECYCLE_KEY)

    @pytest.mark.asyncio
    async def test_open_or_create_creates_again_when_the_winner_is_gone(
        self, store, monkeypatch
    ):
        """Losing the create to a winner that is deleted before it can be
        opened is not an error: open-or-create creates the partition again."""
        await store.delete_partition(LIFECYCLE_KEY)
        registry = store._partition_registry
        reserve = registry.reserve
        lost = False

        async def lose_once(partition_key, schema):
            nonlocal lost
            if not lost:
                lost = True
                raise VectorStorePartitionAlreadyExistsError(
                    store.vector_store_name, partition_key
                )
            return await reserve(partition_key, schema)

        monkeypatch.setattr(registry, "reserve", lose_once)

        partition = await store.open_or_create_partition(LIFECYCLE_KEY)

        assert lost
        record = _records(1)[0]
        await partition.upsert(records=[record])
        assert await self.stored_uuids(partition) == {record.uuid}

    @pytest.mark.asyncio
    async def test_lifecycle_churn_raises_only_domain_errors(self, store):
        """Concurrent create, open-or-create, get and delete of a few keys
        raise nothing but the domain's own outcomes."""
        keys = [f"{LIFECYCLE_KEY}_{index}" for index in range(4)]

        async def worker(seed: int) -> None:
            rng = random.Random(seed)
            for _ in range(30):
                key = rng.choice(keys)
                operation = rng.randrange(4)
                try:
                    if operation == 0:
                        await store.create_partition(key)
                    elif operation == 1:
                        await store.open_or_create_partition(key)
                    elif operation == 2:
                        await store.get_partition(key)
                    else:
                        await store.delete_partition(key)
                except (
                    VectorStorePartitionAlreadyExistsError,
                    VectorStorePartitionDeletedError,
                    VectorStorePartitionPendingError,
                ):
                    pass

        await asyncio.wait_for(
            asyncio.gather(*(worker(seed) for seed in range(6))), 120
        )
