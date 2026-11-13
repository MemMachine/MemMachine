"""
The partition lifecycle contract the remote vector stores satisfy.

A store's test module mixes `PartitionLifecycleContract` into a test class
and supplies a `store` fixture and three hooks: `count_stored` and
`stored_uuids` read the backend directly, and `settle` returns once the
store's reads reflect every write so far. The backend may persist across
tests, so each test uses fresh partition UUIDs and counts against the records
the store held when it began.
"""

import math
from uuid import UUID, uuid4

import pytest

from memmachine_server.common.vector_store import Record

# Purge calls a drain makes before it fails the test: far more than the
# records a test writes need, so only a purge that keeps finding records
# reaches it.
_MAX_PURGE_CALLS = 1000


def _unit(vector: list[float]) -> list[float]:
    magnitude = math.sqrt(sum(x * x for x in vector))
    return [x / magnitude for x in vector]


def _records(count: int) -> list[Record]:
    return [
        Record(uuid=uuid4(), vector=_unit([1.0, 0.01 * index, 0.0]))
        for index in range(count)
    ]


async def _created(store, partition_uuid: UUID):
    """A handle on a freshly created partition."""
    await store.create_partition(partition_uuid)
    return await store.get_partition(partition_uuid)


async def _purge(store, partition_uuid: UUID) -> int:
    """Purge a partition until the store finds nothing of it; the calls that found something."""
    for calls in range(_MAX_PURGE_CALLS):
        if not await store.purge_partition(partition_uuid):
            return calls
    pytest.fail(
        f"partition UUID {partition_uuid} still held records after "
        f"{_MAX_PURGE_CALLS} purge calls"
    )


class PartitionLifecycleContract:
    """Mixed into a store's test class, which supplies a `store` fixture and the `count_stored`, `stored_uuids` and `settle` hooks."""

    @staticmethod
    async def count_stored(store) -> int:
        """Records the backend holds for the store, purged partition UUIDs' leftovers included."""
        raise NotImplementedError

    @staticmethod
    async def stored_uuids(partition) -> set[UUID]:
        """Record UUIDs the backend holds under the handle's partition UUID."""
        raise NotImplementedError

    @staticmethod
    async def settle(partition) -> None:
        """Return once the store's reads reflect every write made so far."""
        raise NotImplementedError

    @pytest.mark.asyncio
    async def test_a_created_partition_holds_its_records(self, store):
        partition = await _created(store, uuid4())
        records = _records(3)

        await partition.upsert(records=records)
        await self.settle(partition)

        assert await self.stored_uuids(partition) == {record.uuid for record in records}

    @pytest.mark.asyncio
    async def test_creating_a_partition_again_changes_nothing(self, store):
        partition_uuid = uuid4()
        partition = await _created(store, partition_uuid)
        records = _records(2)
        await partition.upsert(records=records)

        await store.create_partition(partition_uuid)
        await self.settle(partition)

        assert await self.stored_uuids(partition) == {record.uuid for record in records}

    @pytest.mark.asyncio
    async def test_partitions_are_isolated_from_one_another(self, store):
        first = await _created(store, uuid4())
        second = await _created(store, uuid4())
        shared = _records(1)[0]
        only_first = _records(1)[0]

        await first.upsert(records=[shared, only_first])
        await second.upsert(
            records=[Record(uuid=shared.uuid, vector=_unit([0.0, 1.0, 0.0]))]
        )
        await self.settle(first)
        await self.settle(second)

        assert await self.stored_uuids(first) == {shared.uuid, only_first.uuid}
        assert await self.stored_uuids(second) == {shared.uuid}
        [result] = await second.query(query_vectors=[shared.vector], limit=10)
        assert [match.record_uuid for match in result.matches] == [shared.uuid]

    @pytest.mark.asyncio
    async def test_purge_removes_a_partitions_records(self, store):
        partition_uuid = uuid4()
        partition = await _created(store, partition_uuid)
        baseline = await self.count_stored(store)
        await partition.upsert(records=_records(5))
        await self.settle(partition)
        assert await self.count_stored(store) == baseline + 5

        assert await _purge(store, partition_uuid) >= 1

        assert await self.count_stored(store) == baseline
        assert await store.purge_partition(partition_uuid) is False

    @pytest.mark.asyncio
    async def test_purge_leaves_other_partitions_alone(self, store):
        kept_partition_uuid = uuid4()
        purged_partition_uuid = uuid4()
        kept = await _created(store, kept_partition_uuid)
        purged = await _created(store, purged_partition_uuid)
        kept_records = _records(3)
        await kept.upsert(records=kept_records)
        await purged.upsert(records=_records(3))
        await self.settle(kept)
        await self.settle(purged)

        await _purge(store, purged_partition_uuid)

        assert await self.stored_uuids(kept) == {record.uuid for record in kept_records}

    @pytest.mark.asyncio
    async def test_purging_an_unknown_partition_finds_nothing(self, store):
        assert await store.purge_partition(uuid4()) is False

    @pytest.mark.asyncio
    async def test_a_purged_partition_drops_and_holds_nothing(self, store):
        partition_uuid = uuid4()
        partition = await _created(store, partition_uuid)
        baseline = await self.count_stored(store)
        await partition.upsert(records=_records(3))
        await self.settle(partition)
        await _purge(store, partition_uuid)

        await store.drop_partition(partition_uuid)
        await store.drop_partition(partition_uuid)

        assert await self.count_stored(store) == baseline
        assert await store.purge_partition(partition_uuid) is False

    @pytest.mark.asyncio
    async def test_dropping_a_partition_leaves_other_partitions_alone(self, store):
        kept = await _created(store, uuid4())
        dropped_partition_uuid = uuid4()
        await _created(store, dropped_partition_uuid)
        kept_records = _records(2)
        await kept.upsert(records=kept_records)
        await self.settle(kept)

        await _purge(store, dropped_partition_uuid)
        await store.drop_partition(dropped_partition_uuid)

        assert await self.stored_uuids(kept) == {record.uuid for record in kept_records}

    @pytest.mark.asyncio
    async def test_a_write_landing_after_a_purge_is_reclaimed_by_the_next(self, store):
        partition_uuid = uuid4()
        partition = await _created(store, partition_uuid)
        baseline = await self.count_stored(store)
        await _purge(store, partition_uuid)

        await partition.upsert(records=_records(2))
        await self.settle(partition)
        assert await self.count_stored(store) == baseline + 2

        assert await _purge(store, partition_uuid) >= 1
        assert await self.count_stored(store) == baseline
