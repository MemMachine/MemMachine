"""
The partition lifecycle contract every vector store satisfies.

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
and `purge_deleted_partitions` reclaims the records afterward.

`RemotePartitionLifecycleContract` adds what holds for a store whose
records live outside the registry's database: a write can land under an
incarnation that died while it was in flight; the operation then raises
instead of reporting success, and the incarnation's tombstone has the
records reclaimed by a later purge round.
"""

import math
from uuid import uuid4

import pytest

from memmachine_server.common.vector_store import (
    Record,
    VectorStorePartitionHandleStaleError,
)

LIFECYCLE_KEY = "lifecycle"


def _unit(vector: list[float]) -> list[float]:
    magnitude = math.sqrt(sum(x * x for x in vector))
    return [x / magnitude for x in vector]


def _records(dimensions: int, count: int) -> list[Record]:
    return [
        Record(
            uuid=uuid4(),
            vector=_unit([1.0] + [0.01 * index] + [0.0] * (dimensions - 2)),
        )
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
        partition = await _fresh(store, LIFECYCLE_KEY)
        record = _records(store.vector_dimensions, 1)[0]
        await partition.upsert(records=[record])

        await store.delete_partition(LIFECYCLE_KEY)

        assert await store.get_partition(LIFECYCLE_KEY) is None
        with pytest.raises(VectorStorePartitionHandleStaleError, match=LIFECYCLE_KEY):
            await partition.upsert(records=[record])
        with pytest.raises(VectorStorePartitionHandleStaleError, match=LIFECYCLE_KEY):
            await partition.query(query_vectors=[record.vector], limit=5)
        with pytest.raises(VectorStorePartitionHandleStaleError, match=LIFECYCLE_KEY):
            await partition.delete(record_uuids=[record.uuid])

    @pytest.mark.asyncio
    async def test_a_recreated_partition_starts_empty_and_the_old_handle_stays_stale(
        self, store
    ):
        old = await _fresh(store, LIFECYCLE_KEY)
        old_record, new_record = _records(store.vector_dimensions, 2)
        await old.upsert(records=[old_record])

        await store.delete_partition(LIFECYCLE_KEY)
        new = await _fresh(store, LIFECYCLE_KEY)

        [before] = await new.query(query_vectors=[old_record.vector], limit=5)
        assert before.matches == []

        await new.upsert(records=[new_record])
        [after] = await new.query(query_vectors=[new_record.vector], limit=5)
        assert {match.record_uuid for match in after.matches} == {new_record.uuid}

        # The old life's handle cannot reach the new life's records.
        with pytest.raises(VectorStorePartitionHandleStaleError):
            await old.query(query_vectors=[new_record.vector], limit=5)
        with pytest.raises(VectorStorePartitionHandleStaleError):
            await old.delete(record_uuids=[new_record.uuid])
        [still] = await new.query(query_vectors=[new_record.vector], limit=5)
        assert {match.record_uuid for match in still.matches} == {new_record.uuid}

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
        partition = await _fresh(store, LIFECYCLE_KEY)
        baseline = await self._drained_count(store)
        records = _records(store.vector_dimensions, 5)
        await partition.upsert(records=records)
        assert await self.count_stored(store) == baseline + 5

        await store.delete_partition(LIFECYCLE_KEY)
        # Unreachable at once; the records may still be held.
        assert await store.get_partition(LIFECYCLE_KEY) is None

        assert await self._drained_count(store) == baseline
        # Nothing left to claim.
        assert await store.purge_deleted_partitions() is False

    @pytest.mark.asyncio
    async def test_purge_leaves_a_live_partition_alone(self, store):
        live_key = f"{LIFECYCLE_KEY}_live"
        live = await _fresh(store, live_key)
        dead = await _fresh(store, LIFECYCLE_KEY)
        baseline = await self._drained_count(store)
        kept = _records(store.vector_dimensions, 3)
        gone = _records(store.vector_dimensions, 3)
        await live.upsert(records=kept)
        await dead.upsert(records=gone)

        await store.delete_partition(LIFECYCLE_KEY)

        assert await self._drained_count(store) == baseline + 3
        [result] = await live.query(query_vectors=[kept[0].vector], limit=10)
        assert {match.record_uuid for match in result.matches} == {
            record.uuid for record in kept
        }
        await store.delete_partition(live_key)
        assert await self._drained_count(store) == baseline


class RemotePartitionLifecycleContract(PartitionLifecycleContract):
    """The lifecycle contract plus the late-write clause, for stores with a remote backend."""

    @pytest.mark.asyncio
    async def test_a_write_landing_under_a_dead_incarnation_raises_and_is_reclaimed(
        self, store
    ):
        partition = await _fresh(store, LIFECYCLE_KEY)
        baseline = await self._drained_count(store)
        records = _records(store.vector_dimensions, 2)

        # The partition dies between the handle's check and its write.
        is_live = partition._is_live

        async def deleted_once_checked(incarnation) -> bool:
            live = await is_live(incarnation)
            if live:
                await store.delete_partition(LIFECYCLE_KEY)
            return live

        partition._is_live = deleted_once_checked

        with pytest.raises(VectorStorePartitionHandleStaleError, match=LIFECYCLE_KEY):
            await partition.upsert(records=records)

        # The write landed, under an incarnation nothing can reach...
        assert await self.count_stored(store) == baseline + 2
        # ...and the tombstone's next round reclaims it.
        assert await store.purge_deleted_partitions() is True
        assert await self._drained_count(store) == baseline
