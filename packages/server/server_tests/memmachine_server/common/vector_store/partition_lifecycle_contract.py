"""
The partition lifecycle contract the registry-backed vector stores satisfy.

Each store's test module mixes `PartitionLifecycleContract` into a test
class and supplies the `store` fixture, a provisioned and started store, and
`count_stored(store)`, the number of records the backend physically holds
for the whole collection, which the purge tests read past the store's own
API. The backend may persist across tests (a shared Qdrant or Milvus
server): every test starts by deleting the partitions it uses, and the
purge tests count relative to a drained baseline.

A partition is identified to callers by its key and inside the store by an
incarnation minted per life of the key. The contract: a handle is bound to
one incarnation and raises once that incarnation is deleted; a partition
re-created under a deleted key starts empty; deletion is a registry write
and `purge_deleted_partitions` reclaims the records afterward; a write can
land under an incarnation that died while it was in flight, and the
operation then raises instead of reporting success, with the incarnation's
tombstone having the records reclaimed by a later purge round.
"""

import asyncio
import math
import random
from uuid import uuid4

import pytest

from memmachine_server.common.vector_store import (
    Record,
    VectorStoreAttemptsExhaustedError,
    VectorStorePartitionAlreadyExistsError,
    VectorStorePartitionHandleStaleError,
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


def _uuids(result) -> set:
    return {match.record_uuid for match in result.matches}


class PartitionLifecycleContract:
    """Mixed into a store's test class, with its `store` fixture and `count_stored`."""

    @staticmethod
    async def count_stored(store) -> int:
        """Records the backend physically holds for the collection, live or dead."""
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
        [after] = await new.query(query_vectors=[new_record.vector], limit=5)
        assert _uuids(after) == {new_record.uuid}

        # The old life's handle cannot reach the new life's records.
        with pytest.raises(VectorStorePartitionHandleStaleError):
            await old.query(query_vectors=[new_record.vector], limit=5)
        with pytest.raises(VectorStorePartitionHandleStaleError):
            await old.delete(record_uuids=[new_record.uuid])
        [still] = await new.query(query_vectors=[new_record.vector], limit=5)
        assert _uuids(still) == {new_record.uuid}

        await store.delete_partition(LIFECYCLE_KEY)

    @pytest.mark.asyncio
    async def test_open_or_create_adopts_the_live_incarnation(self, store):
        """Opening an existing collection binds to its life; creating one mints a new life."""
        await store.delete_partition(LIFECYCLE_KEY)
        first = await store.open_or_create_partition(LIFECYCLE_KEY)
        second = await store.open_or_create_partition(LIFECYCLE_KEY)
        record = _records(1)[0]
        await first.upsert(records=[record])
        [seen] = await second.query(query_vectors=[record.vector], limit=5)
        assert _uuids(seen) == {record.uuid}

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

        await store.delete_partition(LIFECYCLE_KEY)

        assert await self._drained_count(store) == baseline + 3
        [result] = await live.query(query_vectors=[kept[0].vector], limit=10)
        assert _uuids(result) == {record.uuid for record in kept}
        await store.delete_partition(live_name)
        assert await self._drained_count(store) == baseline

    @pytest.mark.asyncio
    async def test_a_write_landing_under_a_dead_incarnation_raises_and_is_reclaimed(
        self, store
    ):
        collection = await _fresh(store, LIFECYCLE_KEY)
        baseline = await self._drained_count(store)
        records = _records(2)

        # The collection dies between the handle's check and its write.
        is_live = collection._is_live

        async def deleted_once_checked(incarnation) -> bool:
            live = await is_live(incarnation)
            if live:
                await store.delete_partition(LIFECYCLE_KEY)
            return live

        collection._is_live = deleted_once_checked

        with pytest.raises(VectorStorePartitionHandleStaleError, match=LIFECYCLE_KEY):
            await collection.upsert(records=records)

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
    async def test_a_read_checks_the_registry_once_and_a_write_twice(self, store):
        """A read costs one registry round trip, before it; a write two,
        around it, so a write under an incarnation that died meanwhile raises."""
        partition = await _fresh(store, LIFECYCLE_KEY)
        record = _records(1)[0]
        checks = 0
        is_live = partition._is_live

        async def counted_is_live(incarnation) -> bool:
            nonlocal checks
            checks += 1
            return await is_live(incarnation)

        partition._is_live = counted_is_live

        await partition.upsert(records=[record])
        assert checks == 2
        checks = 0
        await partition.query(query_vectors=[record.vector], limit=1)
        assert checks == 1
        checks = 0
        await partition.delete(record_uuids=[record.uuid])
        assert checks == 2

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

        monkeypatch.setattr(registry, "register", lost)
        monkeypatch.setattr(registry, "get", vanished)

        with pytest.raises(VectorStoreAttemptsExhaustedError):
            await asyncio.wait_for(store.open_or_create_partition(LIFECYCLE_KEY), 30)

    @pytest.mark.asyncio
    async def test_open_or_create_creates_again_when_the_winner_is_gone(
        self, store, monkeypatch
    ):
        """Losing the create to a winner that is deleted before it can be
        opened is not an error: open-or-create creates the partition again."""
        await store.delete_partition(LIFECYCLE_KEY)
        registry = store._partition_registry
        register = registry.register
        lost = False

        async def lose_once(partition_key, schema):
            nonlocal lost
            if not lost:
                lost = True
                raise VectorStorePartitionAlreadyExistsError(
                    store.vector_store_name, partition_key
                )
            return await register(partition_key, schema)

        monkeypatch.setattr(registry, "register", lose_once)

        partition = await store.open_or_create_partition(LIFECYCLE_KEY)

        assert lost
        record = _records(1)[0]
        await partition.upsert(records=[record])
        [result] = await partition.query(query_vectors=[record.vector], limit=1)
        assert _uuids(result) == {record.uuid}

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
                except VectorStorePartitionAlreadyExistsError:
                    pass

        await asyncio.wait_for(
            asyncio.gather(*(worker(seed) for seed in range(6))), 120
        )
