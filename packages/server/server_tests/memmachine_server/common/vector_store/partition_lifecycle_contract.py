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
)

LIFECYCLE_KEY = "lifecycle"

# Purge rounds a drain runs before it fails the test: far more than the few
# deleted partitions a test leaves need, so only a round that keeps finding
# records reaches it.
_MAX_DRAIN_ROUNDS = 1000


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
        for _ in range(_MAX_DRAIN_ROUNDS):
            if not await store.purge_deleted_partitions():
                return await self.count_stored(store)
        pytest.fail(
            f"a deleted partition was still due for purge after "
            f"{_MAX_DRAIN_ROUNDS} rounds"
        )

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

        # Once the store's reads reflect the old life's write, the new life
        # still does not hold it.
        await self.settle(new)
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
    async def test_a_stale_upsert_writes_nothing(self, store):
        """An upsert through a handle whose partition is already deleted
        raises before it writes: the dead life keeps exactly what it held."""
        partition = await _fresh(store, LIFECYCLE_KEY)
        kept, refused = _records(2)
        await partition.upsert(records=[kept])

        await store.delete_partition(LIFECYCLE_KEY)

        with pytest.raises(VectorStorePartitionHandleStaleError, match=LIFECYCLE_KEY):
            await partition.upsert(records=[refused])
        assert await self.stored_uuids(partition) == {kept.uuid}

    @pytest.mark.asyncio
    async def test_a_failed_partition_storage_preparation_frees_the_key(
        self, store, monkeypatch
    ):
        """A creation whose storage preparation fails unregisters its
        pending partition: nothing opens, and the key is free again."""
        await store.delete_partition(LIFECYCLE_KEY)

        async def refused(partition_key, incarnation) -> None:
            raise RuntimeError("the backend refused")

        monkeypatch.setattr(store, "_prepare_partition_storage", refused)
        with pytest.raises(RuntimeError, match="refused"):
            await store.create_partition(LIFECYCLE_KEY)
        monkeypatch.undo()

        assert await store.get_partition(LIFECYCLE_KEY) is None
        await store.create_partition(LIFECYCLE_KEY)
        await store.delete_partition(LIFECYCLE_KEY)

    @pytest.mark.asyncio
    async def test_lifecycle_churn_raises_only_domain_errors(self, store):
        """Concurrent create, get and delete of a few keys
        raise nothing but the domain's own outcomes."""
        keys = [f"{LIFECYCLE_KEY}_{index}" for index in range(4)]

        async def worker(seed: int) -> None:
            rng = random.Random(seed)
            for _ in range(30):
                key = rng.choice(keys)
                operation = rng.randrange(3)
                try:
                    if operation == 0:
                        await store.create_partition(key)
                    elif operation == 1:
                        await store.get_partition(key)
                    else:
                        await store.delete_partition(key)
                except (
                    VectorStoreAttemptsExhaustedError,
                    VectorStorePartitionAlreadyExistsError,
                    VectorStorePartitionDeletedError,
                    VectorStorePartitionPendingError,
                ):
                    pass

        await asyncio.wait_for(
            asyncio.gather(*(worker(seed) for seed in range(6))), 120
        )
