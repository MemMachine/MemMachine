"""
The collection lifecycle contract the registry-backed vector stores satisfy.

A store's test module mixes `CollectionLifecycleContract` into a test class
and supplies a `store` fixture and three hooks: `count_stored` and
`stored_uuids` read the backend directly, and `settle` returns once the
store's reads reflect every write so far. The backend may persist across
tests, so each test first deletes the collections it uses, and the purge
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
    VectorStoreCollectionAlreadyExistsError,
    VectorStoreCollectionConfig,
    VectorStoreCollectionConfigMismatchError,
    VectorStoreCollectionDeletedError,
    VectorStoreCollectionHandleStaleError,
    VectorStoreCollectionPendingError,
    registry_backed_vector_store,
)

LIFECYCLE_NAMESPACE = "lifecycle_ns"
LIFECYCLE_NAME = "lifecycle"
LIFECYCLE_CONFIG = VectorStoreCollectionConfig(vector_dimensions=3)


def _unit(vector: list[float]) -> list[float]:
    magnitude = math.sqrt(sum(x * x for x in vector))
    return [x / magnitude for x in vector]


def _records(count: int) -> list[Record]:
    return [
        Record(uuid=uuid4(), vector=_unit([1.0, 0.01 * index, 0.0]))
        for index in range(count)
    ]


async def _fresh(store, name: str):
    """A new collection under `name`, whatever a persisted backend held for it."""
    await store.delete_collection(namespace=LIFECYCLE_NAMESPACE, name=name)
    await store.create_collection(
        namespace=LIFECYCLE_NAMESPACE, name=name, config=LIFECYCLE_CONFIG
    )
    collection = await store.open_collection(namespace=LIFECYCLE_NAMESPACE, name=name)
    assert collection is not None
    return collection


class CollectionLifecycleContract:
    """Mixed into a store's test class, which supplies a `store` fixture and the `count_stored`, `stored_uuids` and `settle` hooks."""

    @staticmethod
    async def count_stored(store, namespace: str, config) -> int:
        """Records the backend holds in the native collection, deleted collections' included."""
        raise NotImplementedError

    @staticmethod
    async def stored_uuids(collection) -> set[UUID]:
        """Record UUIDs the backend holds under the handle's incarnation."""
        raise NotImplementedError

    @staticmethod
    async def settle(collection) -> None:
        """Return once the store's reads reflect every write made so far."""
        raise NotImplementedError

    async def _drained_count(self, store) -> int:
        """`count_stored` once nothing deleted is left to reclaim."""
        while await store.purge_deleted_collections():
            pass
        return await self.count_stored(store, LIFECYCLE_NAMESPACE, LIFECYCLE_CONFIG)

    @pytest.mark.asyncio
    async def test_a_handle_is_stale_once_its_collection_is_deleted(self, store):
        collection = await _fresh(store, LIFECYCLE_NAME)
        record = _records(1)[0]
        await collection.upsert(records=[record])

        await store.delete_collection(
            namespace=LIFECYCLE_NAMESPACE, name=LIFECYCLE_NAME
        )

        assert (
            await store.open_collection(
                namespace=LIFECYCLE_NAMESPACE, name=LIFECYCLE_NAME
            )
            is None
        )
        with pytest.raises(VectorStoreCollectionHandleStaleError, match=LIFECYCLE_NAME):
            await collection.upsert(records=[record])
        with pytest.raises(VectorStoreCollectionHandleStaleError, match=LIFECYCLE_NAME):
            await collection.query(query_vectors=[record.vector], limit=5)
        with pytest.raises(VectorStoreCollectionHandleStaleError, match=LIFECYCLE_NAME):
            await collection.delete(record_uuids=[record.uuid])
        # So does an operation with nothing to send to the backend.
        with pytest.raises(VectorStoreCollectionHandleStaleError, match=LIFECYCLE_NAME):
            await collection.upsert(records=[])
        with pytest.raises(VectorStoreCollectionHandleStaleError, match=LIFECYCLE_NAME):
            await collection.query(query_vectors=[], limit=5)
        with pytest.raises(VectorStoreCollectionHandleStaleError, match=LIFECYCLE_NAME):
            await collection.delete(record_uuids=[])

    @pytest.mark.asyncio
    async def test_a_recreated_collection_starts_empty_and_the_old_handle_stays_stale(
        self, store
    ):
        old = await _fresh(store, LIFECYCLE_NAME)
        old_record, new_record = _records(2)
        await old.upsert(records=[old_record])

        await store.delete_collection(
            namespace=LIFECYCLE_NAMESPACE, name=LIFECYCLE_NAME
        )
        new = await _fresh(store, LIFECYCLE_NAME)

        [before] = await new.query(query_vectors=[old_record.vector], limit=5)
        assert before.matches == []

        await new.upsert(records=[new_record])
        assert await self.stored_uuids(new) == {new_record.uuid}

        # The old life's handle cannot reach the new life's records.
        with pytest.raises(VectorStoreCollectionHandleStaleError):
            await old.query(query_vectors=[new_record.vector], limit=5)
        with pytest.raises(VectorStoreCollectionHandleStaleError):
            await old.delete(record_uuids=[new_record.uuid])
        assert await self.stored_uuids(new) == {new_record.uuid}

        await store.delete_collection(
            namespace=LIFECYCLE_NAMESPACE, name=LIFECYCLE_NAME
        )

    @pytest.mark.asyncio
    async def test_open_or_create_adopts_the_live_incarnation(self, store):
        """Opening an existing collection binds to its life; creating one mints a new life."""
        await store.delete_collection(
            namespace=LIFECYCLE_NAMESPACE, name=LIFECYCLE_NAME
        )
        first = await store.open_or_create_collection(
            namespace=LIFECYCLE_NAMESPACE, name=LIFECYCLE_NAME, config=LIFECYCLE_CONFIG
        )
        second = await store.open_or_create_collection(
            namespace=LIFECYCLE_NAMESPACE, name=LIFECYCLE_NAME, config=LIFECYCLE_CONFIG
        )
        record = _records(1)[0]
        await first.upsert(records=[record])
        assert await self.stored_uuids(second) == {record.uuid}

        await store.delete_collection(
            namespace=LIFECYCLE_NAMESPACE, name=LIFECYCLE_NAME
        )
        third = await store.open_or_create_collection(
            namespace=LIFECYCLE_NAMESPACE, name=LIFECYCLE_NAME, config=LIFECYCLE_CONFIG
        )
        [empty] = await third.query(query_vectors=[record.vector], limit=5)
        assert empty.matches == []
        with pytest.raises(VectorStoreCollectionHandleStaleError):
            await first.query(query_vectors=[record.vector], limit=5)
        await store.delete_collection(
            namespace=LIFECYCLE_NAMESPACE, name=LIFECYCLE_NAME
        )

    @pytest.mark.asyncio
    async def test_deleting_twice_and_deleting_nothing_are_no_ops(self, store):
        await _fresh(store, LIFECYCLE_NAME)
        await store.delete_collection(
            namespace=LIFECYCLE_NAMESPACE, name=LIFECYCLE_NAME
        )
        await store.delete_collection(
            namespace=LIFECYCLE_NAMESPACE, name=LIFECYCLE_NAME
        )
        assert (
            await store.open_collection(
                namespace=LIFECYCLE_NAMESPACE, name=LIFECYCLE_NAME
            )
            is None
        )
        await store.delete_collection(
            namespace=LIFECYCLE_NAMESPACE, name=f"{LIFECYCLE_NAME}_never_created"
        )

    @pytest.mark.asyncio
    async def test_purge_reclaims_what_deletion_deferred(self, store):
        collection = await _fresh(store, LIFECYCLE_NAME)
        baseline = await self._drained_count(store)
        records = _records(5)
        await collection.upsert(records=records)
        await self.settle(collection)
        assert (
            await self.count_stored(store, LIFECYCLE_NAMESPACE, LIFECYCLE_CONFIG)
            == baseline + 5
        )

        await store.delete_collection(
            namespace=LIFECYCLE_NAMESPACE, name=LIFECYCLE_NAME
        )
        # Unreachable at once, and the records are left for the purge:
        # deletion touches the registry alone, whatever the collection holds.
        assert (
            await store.open_collection(
                namespace=LIFECYCLE_NAMESPACE, name=LIFECYCLE_NAME
            )
            is None
        )
        assert (
            await self.count_stored(store, LIFECYCLE_NAMESPACE, LIFECYCLE_CONFIG)
            == baseline + 5
        )

        assert await self._drained_count(store) == baseline
        # Nothing left to claim.
        assert await store.purge_deleted_collections() is False

    @pytest.mark.asyncio
    async def test_purge_leaves_a_live_collection_alone(self, store):
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

        await store.delete_collection(
            namespace=LIFECYCLE_NAMESPACE, name=LIFECYCLE_NAME
        )

        assert await self._drained_count(store) == baseline + 3
        assert await self.stored_uuids(live) == {record.uuid for record in kept}
        await store.delete_collection(namespace=LIFECYCLE_NAMESPACE, name=live_name)
        assert await self._drained_count(store) == baseline

    @pytest.mark.asyncio
    async def test_a_write_landing_under_a_dead_incarnation_raises_and_is_reclaimed(
        self, store, monkeypatch
    ):
        collection = await _fresh(store, LIFECYCLE_NAME)
        baseline = await self._drained_count(store)
        records = _records(2)

        # The collection dies between the handle's check and its write.
        registration_type = type(collection._registration)
        require_current = registration_type.require_current

        async def deleted_once_checked(registration) -> None:
            await require_current(registration)
            await store.delete_collection(
                namespace=registration.namespace, name=registration.name
            )

        monkeypatch.setattr(registration_type, "require_current", deleted_once_checked)

        with pytest.raises(VectorStoreCollectionHandleStaleError, match=LIFECYCLE_NAME):
            await collection.upsert(records=records)
        await self.settle(collection)

        # The write landed, under an incarnation nothing can reach...
        assert (
            await self.count_stored(store, LIFECYCLE_NAMESPACE, LIFECYCLE_CONFIG)
            == baseline + 2
        )
        # ...and the tombstone's next round reclaims it.
        assert await store.purge_deleted_collections() is True
        assert await self._drained_count(store) == baseline

    @pytest.mark.asyncio
    async def test_a_trailing_newline_is_not_part_of_a_valid_identifier(self, store):
        for namespace, name in (
            (LIFECYCLE_NAMESPACE, f"{LIFECYCLE_NAME}\n"),
            (f"{LIFECYCLE_NAMESPACE}\n", LIFECYCLE_NAME),
        ):
            with pytest.raises(ValueError, match="must match"):
                await store.create_collection(
                    namespace=namespace, name=name, config=LIFECYCLE_CONFIG
                )

    @pytest.mark.asyncio
    async def test_an_upsert_checks_the_registry_twice_and_a_query_or_delete_once(
        self, store, monkeypatch
    ):
        collection = await _fresh(store, LIFECYCLE_NAME)
        record = _records(1)[0]
        checks = 0
        registration_type = type(collection._registration)
        require_current = registration_type.require_current

        async def counted(registration) -> None:
            nonlocal checks
            checks += 1
            await require_current(registration)

        monkeypatch.setattr(registration_type, "require_current", counted)

        await collection.upsert(records=[record])
        assert checks == 2
        checks = 0
        await collection.query(query_vectors=[record.vector], limit=1)
        assert checks == 1
        checks = 0
        await collection.delete(record_uuids=[record.uuid])
        assert checks == 1

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "create", ["create_collection", "open_or_create_collection"]
    )
    async def test_a_failed_storage_preparation_frees_the_name(
        self, store, monkeypatch, create
    ):
        """A creation whose storage preparation fails unregisters its
        pending collection: nothing opens, and the name is free again."""
        await store.delete_collection(
            namespace=LIFECYCLE_NAMESPACE, name=LIFECYCLE_NAME
        )

        async def refused(namespace, config, incarnation) -> None:
            raise RuntimeError("the backend refused")

        monkeypatch.setattr(store, "_prepare_storage", refused)
        with pytest.raises(RuntimeError, match="refused"):
            await getattr(store, create)(
                namespace=LIFECYCLE_NAMESPACE,
                name=LIFECYCLE_NAME,
                config=LIFECYCLE_CONFIG,
            )
        monkeypatch.undo()

        assert (
            await store.open_collection(
                namespace=LIFECYCLE_NAMESPACE, name=LIFECYCLE_NAME
            )
            is None
        )
        await store.create_collection(
            namespace=LIFECYCLE_NAMESPACE, name=LIFECYCLE_NAME, config=LIFECYCLE_CONFIG
        )
        await store.delete_collection(
            namespace=LIFECYCLE_NAMESPACE, name=LIFECYCLE_NAME
        )

    @pytest.mark.asyncio
    async def test_open_or_create_gives_up_after_losing_every_race(
        self, store, monkeypatch
    ):
        """A creator that keeps losing to a winner that keeps vanishing gives
        up after a bounded number of attempts instead of looping."""
        registry = store._collection_registry

        async def lost(namespace, name, config):
            # Yields as a real round trip would, so an unbounded loop fails
            # the timeout below instead of starving the event loop.
            await asyncio.sleep(0)
            raise VectorStoreCollectionAlreadyExistsError(namespace, name)

        async def vanished(namespace, name) -> None:
            return None

        monkeypatch.setattr(registry, "reserve", lost)
        monkeypatch.setattr(registry, "resolve", vanished)
        monkeypatch.setattr(
            registry_backed_vector_store, "_OPEN_OR_CREATE_RETRY_DELAY_SECONDS", 0
        )

        with pytest.raises(VectorStoreAttemptsExhaustedError):
            await asyncio.wait_for(
                store.open_or_create_collection(
                    namespace=LIFECYCLE_NAMESPACE,
                    name=LIFECYCLE_NAME,
                    config=LIFECYCLE_CONFIG,
                ),
                30,
            )

    @pytest.mark.asyncio
    async def test_open_or_create_opens_the_winner_after_losing_the_create(
        self, store, monkeypatch
    ):
        """A creator whose reservation loses to another process's opens
        the winner's collection, bound to the winner's life."""
        await store.delete_collection(
            namespace=LIFECYCLE_NAMESPACE, name=LIFECYCLE_NAME
        )
        registry = store._collection_registry
        reserve = registry.reserve
        lost_reservations = 0

        async def another_process_wins(namespace, name, config):
            # The other process creates the collection between this caller's
            # lookup and its own reservation.
            nonlocal lost_reservations
            lost_reservations += 1
            reservation = await reserve(namespace, name, config)
            await store._prepare_storage(namespace, config, reservation.incarnation)
            await reservation.confirm()
            raise VectorStoreCollectionAlreadyExistsError(namespace, name)

        monkeypatch.setattr(registry, "reserve", another_process_wins)
        collection = await store.open_or_create_collection(
            namespace=LIFECYCLE_NAMESPACE, name=LIFECYCLE_NAME, config=LIFECYCLE_CONFIG
        )
        monkeypatch.undo()

        # It lost once, then found the winner instead of registering again.
        assert lost_reservations == 1

        winner = await registry.resolve(LIFECYCLE_NAMESPACE, LIFECYCLE_NAME)
        assert winner is not None
        assert collection._incarnation == winner.incarnation
        record = _records(1)[0]
        await collection.upsert(records=[record])
        opened = await store.open_collection(
            namespace=LIFECYCLE_NAMESPACE, name=LIFECYCLE_NAME
        )
        assert opened is not None
        assert await self.stored_uuids(opened) == {record.uuid}
        await store.delete_collection(
            namespace=LIFECYCLE_NAMESPACE, name=LIFECYCLE_NAME
        )

    @pytest.mark.asyncio
    async def test_open_or_create_refuses_a_winner_of_another_configuration(
        self, store, monkeypatch
    ):
        """A creator that loses to a winner registered with another
        configuration gets the mismatch, not a handle to the winner's."""
        await store.delete_collection(
            namespace=LIFECYCLE_NAMESPACE, name=LIFECYCLE_NAME
        )
        registry = store._collection_registry
        reserve = registry.reserve
        other_config = VectorStoreCollectionConfig(vector_dimensions=4)
        lost_reservations = 0

        async def another_process_wins(namespace, name, config):
            nonlocal lost_reservations
            lost_reservations += 1
            reservation = await reserve(namespace, name, other_config)
            await reservation.confirm()
            raise VectorStoreCollectionAlreadyExistsError(namespace, name)

        monkeypatch.setattr(registry, "reserve", another_process_wins)
        with pytest.raises(VectorStoreCollectionConfigMismatchError):
            await store.open_or_create_collection(
                namespace=LIFECYCLE_NAMESPACE,
                name=LIFECYCLE_NAME,
                config=LIFECYCLE_CONFIG,
            )
        monkeypatch.undo()
        assert lost_reservations == 1
        await store.delete_collection(
            namespace=LIFECYCLE_NAMESPACE, name=LIFECYCLE_NAME
        )

    @pytest.mark.asyncio
    async def test_open_or_create_creates_again_when_the_winner_is_gone(
        self, store, monkeypatch
    ):
        """Losing the create to a winner that is deleted before it can be
        opened is not an error: open-or-create creates the collection again."""
        await store.delete_collection(
            namespace=LIFECYCLE_NAMESPACE, name=LIFECYCLE_NAME
        )
        registry = store._collection_registry
        reserve = registry.reserve
        lost = False

        async def lose_once(namespace, name, config):
            nonlocal lost
            if not lost:
                lost = True
                raise VectorStoreCollectionAlreadyExistsError(namespace, name)
            return await reserve(namespace, name, config)

        monkeypatch.setattr(registry, "reserve", lose_once)

        collection = await store.open_or_create_collection(
            namespace=LIFECYCLE_NAMESPACE, name=LIFECYCLE_NAME, config=LIFECYCLE_CONFIG
        )

        assert lost
        record = _records(1)[0]
        await collection.upsert(records=[record])
        assert await self.stored_uuids(collection) == {record.uuid}

    @pytest.mark.asyncio
    async def test_lifecycle_churn_raises_only_domain_errors(self, store):
        """Concurrent create, open-or-create, open and delete of a few names
        raise nothing but the domain's own outcomes."""
        names = [f"{LIFECYCLE_NAME}_{index}" for index in range(4)]

        async def worker(seed: int) -> None:
            rng = random.Random(seed)
            for _ in range(30):
                name = rng.choice(names)
                operation = rng.randrange(4)
                try:
                    if operation == 0:
                        await store.create_collection(
                            namespace=LIFECYCLE_NAMESPACE,
                            name=name,
                            config=LIFECYCLE_CONFIG,
                        )
                    elif operation == 1:
                        await store.open_or_create_collection(
                            namespace=LIFECYCLE_NAMESPACE,
                            name=name,
                            config=LIFECYCLE_CONFIG,
                        )
                    elif operation == 2:
                        await store.open_collection(
                            namespace=LIFECYCLE_NAMESPACE, name=name
                        )
                    else:
                        await store.delete_collection(
                            namespace=LIFECYCLE_NAMESPACE, name=name
                        )
                except (
                    VectorStoreCollectionAlreadyExistsError,
                    VectorStoreCollectionConfigMismatchError,
                    VectorStoreCollectionDeletedError,
                    VectorStoreCollectionPendingError,
                ):
                    pass

        await asyncio.wait_for(
            asyncio.gather(*(worker(seed) for seed in range(6))), 120
        )
