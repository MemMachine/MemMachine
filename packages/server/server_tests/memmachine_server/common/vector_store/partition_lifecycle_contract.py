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

from memmachine_server.common.partition_registry import (
    PartitionAlreadyExistsError,
    PartitionCreateAttemptsExhaustedError,
    PartitionIncarnationEndedError,
    PartitionPendingError,
)
from memmachine_server.common.vector_store import (
    Record,
    registry_backed_vector_store,
)

LIFECYCLE_NAME = "lifecycle"
# A partition name of the longest length, with characters no identifier allows.
ANY_CHARACTERS_NAME = "Session/ID-1 ü".ljust(254, "x")

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
    partition = await store.open_partition(key)
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
    @pytest.mark.parametrize(
        "partition_name", ["", "k" * 256, "ü" * 128, "with\x00nul"]
    )
    async def test_a_partition_name_outside_the_rule_is_refused(
        self, store, partition_name
    ):
        with pytest.raises(ValueError, match="Partition name"):
            await store.create_partition(partition_name)

    @pytest.mark.asyncio
    async def test_a_partition_name_of_any_characters_and_the_longest_length_works(
        self, store
    ):
        partition = await _fresh(store, ANY_CHARACTERS_NAME)
        record = _records(1)[0]
        await partition.upsert(records=[record])
        assert await self.stored_uuids(partition) == {record.uuid}

        await store.delete_partition(ANY_CHARACTERS_NAME)
        assert await store.open_partition(ANY_CHARACTERS_NAME) is None

    @pytest.mark.asyncio
    async def test_a_handle_is_stale_once_its_partition_is_deleted(self, store):
        partition = await _fresh(store, LIFECYCLE_NAME)
        record = _records(1)[0]
        await partition.upsert(records=[record])

        await store.delete_partition(LIFECYCLE_NAME)

        assert await store.open_partition(LIFECYCLE_NAME) is None
        with pytest.raises(PartitionIncarnationEndedError, match=LIFECYCLE_NAME):
            await partition.upsert(records=[record])
        with pytest.raises(PartitionIncarnationEndedError, match=LIFECYCLE_NAME):
            await partition.query(query_vectors=[record.vector], limit=5)
        with pytest.raises(PartitionIncarnationEndedError, match=LIFECYCLE_NAME):
            await partition.delete(record_uuids=[record.uuid])
        # So does an operation with nothing to send to the backend.
        with pytest.raises(PartitionIncarnationEndedError, match=LIFECYCLE_NAME):
            await partition.upsert(records=[])
        with pytest.raises(PartitionIncarnationEndedError, match=LIFECYCLE_NAME):
            await partition.query(query_vectors=[], limit=5)
        with pytest.raises(PartitionIncarnationEndedError, match=LIFECYCLE_NAME):
            await partition.delete(record_uuids=[])

    @pytest.mark.asyncio
    async def test_a_recreated_partition_starts_empty_and_the_old_handle_stays_stale(
        self, store
    ):
        old = await _fresh(store, LIFECYCLE_NAME)
        old_record, new_record = _records(2)
        await old.upsert(records=[old_record])

        await store.delete_partition(LIFECYCLE_NAME)
        new = await _fresh(store, LIFECYCLE_NAME)

        # Once the store's reads reflect the old life's write, the new life
        # still does not hold it.
        await self.settle(new)
        [before] = await new.query(query_vectors=[old_record.vector], limit=5)
        assert before.matches == []

        await new.upsert(records=[new_record])
        assert await self.stored_uuids(new) == {new_record.uuid}

        # The old life's handle cannot reach the new life's records.
        with pytest.raises(PartitionIncarnationEndedError):
            await old.query(query_vectors=[new_record.vector], limit=5)
        with pytest.raises(PartitionIncarnationEndedError):
            await old.delete(record_uuids=[new_record.uuid])
        assert await self.stored_uuids(new) == {new_record.uuid}

        await store.delete_partition(LIFECYCLE_NAME)

    @pytest.mark.asyncio
    async def test_open_or_create_adopts_the_live_incarnation(self, store):
        """Opening an existing partition binds to its life; creating one mints a new life."""
        await store.delete_partition(LIFECYCLE_NAME)
        first = await store.open_or_create_partition(LIFECYCLE_NAME)
        second = await store.open_or_create_partition(LIFECYCLE_NAME)
        record = _records(1)[0]
        await first.upsert(records=[record])
        assert await self.stored_uuids(second) == {record.uuid}

        await store.delete_partition(LIFECYCLE_NAME)
        third = await store.open_or_create_partition(LIFECYCLE_NAME)
        await self.settle(third)
        [empty] = await third.query(query_vectors=[record.vector], limit=5)
        assert empty.matches == []
        with pytest.raises(PartitionIncarnationEndedError):
            await first.query(query_vectors=[record.vector], limit=5)
        await store.delete_partition(LIFECYCLE_NAME)

    @pytest.mark.asyncio
    async def test_deleting_twice_and_deleting_nothing_are_no_ops(self, store):
        await _fresh(store, LIFECYCLE_NAME)
        await store.delete_partition(LIFECYCLE_NAME)
        await store.delete_partition(LIFECYCLE_NAME)
        assert await store.open_partition(LIFECYCLE_NAME) is None
        await store.delete_partition(f"{LIFECYCLE_NAME}_never_created")

    @pytest.mark.asyncio
    async def test_purge_reclaims_what_deletion_deferred(self, store):
        partition = await _fresh(store, LIFECYCLE_NAME)
        baseline = await self._drained_count(store)
        records = _records(5)
        await partition.upsert(records=records)
        await self.settle(partition)
        assert await self.count_stored(store) == baseline + 5

        await store.delete_partition(LIFECYCLE_NAME)
        # Unreachable at once, and the records are left for the purge:
        # deletion touches the registry alone, whatever the partition holds.
        assert await store.open_partition(LIFECYCLE_NAME) is None
        assert await self.count_stored(store) == baseline + 5

        assert await self._drained_count(store) == baseline
        # Nothing left to claim.
        assert await store.purge_deleted_partitions() is False

    @pytest.mark.asyncio
    async def test_purge_leaves_a_live_partition_alone(self, store):
        live_name = f"{LIFECYCLE_NAME}_live"
        live = await _fresh(store, live_name)
        dead = await _fresh(store, LIFECYCLE_NAME)
        baseline = await self._drained_count(store)
        kept = _records(3)
        gone = _records(3)
        await live.upsert(records=kept)
        await dead.upsert(records=gone)
        await self.settle(live)
        await self.settle(dead)

        await store.delete_partition(LIFECYCLE_NAME)

        assert await self._drained_count(store) == baseline + 3
        assert await self.stored_uuids(live) == {record.uuid for record in kept}
        await store.delete_partition(live_name)
        assert await self._drained_count(store) == baseline

    @pytest.mark.asyncio
    async def test_a_write_landing_under_a_dead_incarnation_raises_and_is_reclaimed(
        self, store, monkeypatch
    ):
        partition = await _fresh(store, LIFECYCLE_NAME)
        baseline = await self._drained_count(store)
        records = _records(2)

        # The partition dies between the handle's check and its write.
        registration_type = type(partition._registration)
        require_current = registration_type.require_current

        async def deleted_once_checked(registration) -> None:
            await require_current(registration)
            await store.delete_partition(registration.partition_name)

        monkeypatch.setattr(registration_type, "require_current", deleted_once_checked)

        with pytest.raises(PartitionIncarnationEndedError, match=LIFECYCLE_NAME):
            await partition.upsert(records=records)
        await self.settle(partition)

        # The write landed, under an incarnation nothing can reach...
        assert await self.count_stored(store) == baseline + 2
        # ...and the tombstone's next round reclaims it.
        assert await store.purge_deleted_partitions() is True
        assert await self._drained_count(store) == baseline

    @pytest.mark.asyncio
    async def test_a_stale_upsert_writes_nothing(self, store):
        """An upsert through a handle whose partition is already deleted
        raises before it writes: the dead life keeps exactly what it held."""
        partition = await _fresh(store, LIFECYCLE_NAME)
        kept, refused = _records(2)
        await partition.upsert(records=[kept])

        await store.delete_partition(LIFECYCLE_NAME)

        with pytest.raises(PartitionIncarnationEndedError, match=LIFECYCLE_NAME):
            await partition.upsert(records=[refused])
        assert await self.stored_uuids(partition) == {kept.uuid}

    @pytest.mark.asyncio
    async def test_open_or_create_gives_up_after_losing_every_race(
        self, store, monkeypatch
    ):
        """A creator that keeps losing to a winner that keeps vanishing gives
        up after a bounded number of attempts instead of looping."""
        registry = store._partition_registry

        async def lost(partition_name):
            # Yields as a real round trip would, so an unbounded loop fails
            # the timeout below instead of starving the event loop.
            await asyncio.sleep(0)
            raise PartitionAlreadyExistsError(store._vector_store_name, partition_name)

        async def vanished(partition_name) -> None:
            return None

        monkeypatch.setattr(registry, "reserve", lost)
        monkeypatch.setattr(registry, "resolve", vanished)
        monkeypatch.setattr(
            registry_backed_vector_store, "_OPEN_OR_CREATE_RETRY_DELAY_SECONDS", 0
        )

        with pytest.raises(PartitionCreateAttemptsExhaustedError):
            await asyncio.wait_for(store.open_or_create_partition(LIFECYCLE_NAME), 30)

    @pytest.mark.asyncio
    @pytest.mark.parametrize("create", ["create_partition", "open_or_create_partition"])
    async def test_a_failed_partition_storage_preparation_frees_the_key(
        self, store, monkeypatch, create
    ):
        """A creation whose storage preparation fails unregisters its
        pending partition: nothing opens, and the key is free again."""
        await store.delete_partition(LIFECYCLE_NAME)

        async def refused(partition_name, incarnation) -> None:
            raise RuntimeError("the backend refused")

        monkeypatch.setattr(store, "_prepare_partition_storage", refused)
        with pytest.raises(RuntimeError, match="refused"):
            await getattr(store, create)(LIFECYCLE_NAME)
        monkeypatch.undo()

        assert await store.open_partition(LIFECYCLE_NAME) is None
        await store.create_partition(LIFECYCLE_NAME)
        await store.delete_partition(LIFECYCLE_NAME)

    @pytest.mark.asyncio
    async def test_open_or_create_opens_the_winner_after_losing_the_create(
        self, store, monkeypatch
    ):
        """A creator whose reservation loses to another process's opens
        the winner's partition, bound to the winner's life."""
        await store.delete_partition(LIFECYCLE_NAME)
        registry = store._partition_registry
        reserve = registry.reserve
        lost_reservations = 0

        async def another_process_wins(partition_name):
            # The other process creates the partition between this caller's
            # lookup and its own reservation.
            nonlocal lost_reservations
            lost_reservations += 1
            reservation = await reserve(partition_name)
            await store._prepare_partition_storage(
                partition_name, reservation.incarnation
            )
            await reservation.confirm()
            raise PartitionAlreadyExistsError(store._vector_store_name, partition_name)

        monkeypatch.setattr(registry, "reserve", another_process_wins)
        partition = await store.open_or_create_partition(LIFECYCLE_NAME)
        monkeypatch.undo()

        # It lost once, then found the winner instead of registering again.
        assert lost_reservations == 1

        winner = await registry.resolve(LIFECYCLE_NAME)
        assert winner is not None
        assert partition._incarnation == winner.incarnation
        record = _records(1)[0]
        await partition.upsert(records=[record])
        opened = await store.open_partition(LIFECYCLE_NAME)
        assert opened is not None
        assert await self.stored_uuids(opened) == {record.uuid}
        await store.delete_partition(LIFECYCLE_NAME)

    @pytest.mark.asyncio
    async def test_open_or_create_creates_again_when_the_winner_is_gone(
        self, store, monkeypatch
    ):
        """Losing the create to a winner that is deleted before it can be
        opened is not an error: open-or-create creates the partition again."""
        await store.delete_partition(LIFECYCLE_NAME)
        registry = store._partition_registry
        reserve = registry.reserve
        lost = False

        async def lose_once(partition_name):
            nonlocal lost
            if not lost:
                lost = True
                raise PartitionAlreadyExistsError(
                    store._vector_store_name, partition_name
                )
            return await reserve(partition_name)

        monkeypatch.setattr(registry, "reserve", lose_once)

        partition = await store.open_or_create_partition(LIFECYCLE_NAME)

        assert lost
        record = _records(1)[0]
        await partition.upsert(records=[record])
        assert await self.stored_uuids(partition) == {record.uuid}

    @pytest.mark.asyncio
    async def test_lifecycle_churn_raises_only_domain_errors(self, store):
        """Concurrent create, open-or-create, get and delete of a few keys
        raise nothing but the domain's own outcomes."""
        keys = [f"{LIFECYCLE_NAME}_{index}" for index in range(4)]

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
                        await store.open_partition(key)
                    else:
                        await store.delete_partition(key)
                except (
                    PartitionCreateAttemptsExhaustedError,
                    PartitionAlreadyExistsError,
                    PartitionIncarnationEndedError,
                    PartitionPendingError,
                ):
                    pass

        await asyncio.wait_for(
            asyncio.gather(*(worker(seed) for seed in range(6))), 120
        )
