"""
The partition incarnation lifecycle contract the remote vector stores satisfy.

A store's test module mixes `PartitionLifecycleContract` into a test class
and supplies a `store` fixture and three hooks: `count_stored` and
`stored_uuids` read the backend directly, and `settle` returns once the
store's reads reflect every write so far. The backend may persist across
tests, so each test uses fresh incarnations and counts against the records
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


async def _prepared(store, incarnation: UUID):
    """A handle on a freshly prepared partition incarnation."""
    await store.prepare_partition_incarnation(incarnation)
    return await store.open_partition_incarnation(incarnation)


async def _purge(store, incarnation: UUID) -> int:
    """Purge an incarnation until the store finds nothing of it; the calls that found something."""
    for calls in range(_MAX_PURGE_CALLS):
        if not await store.purge_partition_incarnation(incarnation):
            return calls
    pytest.fail(
        f"incarnation {incarnation} still held records after "
        f"{_MAX_PURGE_CALLS} purge calls"
    )


class PartitionLifecycleContract:
    """Mixed into a store's test class, which supplies a `store` fixture and the `count_stored`, `stored_uuids` and `settle` hooks."""

    @staticmethod
    async def count_stored(store) -> int:
        """Records the backend holds for the store, purged incarnations' leftovers included."""
        raise NotImplementedError

    @staticmethod
    async def stored_uuids(partition) -> set[UUID]:
        """Record UUIDs the backend holds under the handle's incarnation."""
        raise NotImplementedError

    @staticmethod
    async def settle(partition) -> None:
        """Return once the store's reads reflect every write made so far."""
        raise NotImplementedError

    @pytest.mark.asyncio
    async def test_a_prepared_incarnation_holds_its_records(self, store):
        partition = await _prepared(store, uuid4())
        records = _records(3)

        await partition.upsert(records=records)
        await self.settle(partition)

        assert await self.stored_uuids(partition) == {record.uuid for record in records}

    @pytest.mark.asyncio
    async def test_preparing_an_incarnation_again_changes_nothing(self, store):
        incarnation = uuid4()
        partition = await _prepared(store, incarnation)
        records = _records(2)
        await partition.upsert(records=records)

        await store.prepare_partition_incarnation(incarnation)
        await self.settle(partition)

        assert await self.stored_uuids(partition) == {record.uuid for record in records}

    @pytest.mark.asyncio
    async def test_incarnations_are_isolated_from_one_another(self, store):
        first = await _prepared(store, uuid4())
        second = await _prepared(store, uuid4())
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
    async def test_purge_removes_an_incarnations_records(self, store):
        incarnation = uuid4()
        partition = await _prepared(store, incarnation)
        baseline = await self.count_stored(store)
        await partition.upsert(records=_records(5))
        await self.settle(partition)
        assert await self.count_stored(store) == baseline + 5

        assert await _purge(store, incarnation) >= 1

        assert await self.count_stored(store) == baseline
        assert await store.purge_partition_incarnation(incarnation) is False

    @pytest.mark.asyncio
    async def test_purge_leaves_other_incarnations_alone(self, store):
        kept_incarnation = uuid4()
        purged_incarnation = uuid4()
        kept = await _prepared(store, kept_incarnation)
        purged = await _prepared(store, purged_incarnation)
        kept_records = _records(3)
        await kept.upsert(records=kept_records)
        await purged.upsert(records=_records(3))
        await self.settle(kept)
        await self.settle(purged)

        await _purge(store, purged_incarnation)

        assert await self.stored_uuids(kept) == {record.uuid for record in kept_records}

    @pytest.mark.asyncio
    async def test_purging_an_unknown_incarnation_finds_nothing(self, store):
        assert await store.purge_partition_incarnation(uuid4()) is False

    @pytest.mark.asyncio
    async def test_a_write_landing_after_a_purge_is_reclaimed_by_the_next(self, store):
        incarnation = uuid4()
        partition = await _prepared(store, incarnation)
        baseline = await self.count_stored(store)
        await _purge(store, incarnation)

        await partition.upsert(records=_records(2))
        await self.settle(partition)
        assert await self.count_stored(store) == baseline + 2

        assert await _purge(store, incarnation) >= 1
        assert await self.count_stored(store) == baseline
