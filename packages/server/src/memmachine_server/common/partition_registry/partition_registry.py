"""
Abstract base classes for a partition registry, its reservations, and registrations.

The catalog of one partition namespace: which partitions exist under which
names, and which deleted partitions await purge. A partition has a name, which
its owner chooses and may reuse, and a UUID, which the registry mints for it
alone. The registry drives each partition's lifecycle in the stores that hold
its records: it creates the partition in every store before confirming it,
and once the partition is deleted, its purge rounds purge it from every store
and then drop it from each. Its calls are arbitrated across every process
sharing it: a reservation mints a UUID no registered or queued partition
carries, unregistration makes the partition unreachable when it returns, and a
purge round runs on a due tombstone, possibly on two purgers at once.

The registry is addressed by partition name. Creating a partition, or
resolving a name, answers a `Registration`, through which a handle checks that
the partition has not been deleted. It acts on one partition alone, and a
deletion by name voids it.
"""

import asyncio
import contextlib
import logging
from abc import ABC, abstractmethod
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from uuid import UUID

from .partitioned_store import PartitionedStore

logger = logging.getLogger(__name__)

# Reservations cancelled after a failed creation, held until done so the
# garbage collector cannot drop one whose creation was cancelled.
_reservation_cancellations: set[asyncio.Task[None]] = set()


@dataclass(frozen=True)
class _RegistryEntry:
    """
    A registry's entry for one partition, from its reservation to its deletion.

    Its fields never change: `partition_uuid` is the value the partition's
    records carry.
    """

    partition_name: str
    partition_uuid: UUID


@dataclass(frozen=True)
class Reservation(_RegistryEntry, ABC):
    """
    A partition's hold on its name, kept while the registry creates the partition in every store.

    The partition is pending until the reservation is confirmed. A registry
    implementation supplies the methods, each of which writes the registry.
    """

    @abstractmethod
    async def confirm(self) -> "Registration":
        """
        Confirm this reservation, marking the pending partition live.

        Called once, when every store has created the partition. A concurrent
        deletion of the partition either follows the confirmation or makes it
        raise.

        Returns:
            Registration: The same partition, live.

        Raises:
            PartitionDeletedError:
                If the partition is no longer pending: it was deleted.
        """
        raise NotImplementedError

    @abstractmethod
    async def cancel(self) -> None:
        """
        Cancel this reservation, unregistering its partition while it is pending and queuing the partition's UUID for purge.

        The pending partition is unreachable when this returns, and purge
        rounds reclaim its records later. Once the reservation is confirmed,
        cancelling it does nothing: only a deletion by name ends a live
        partition. A partition reserved since under the same name is another
        partition and stays. Idempotent.
        """
        raise NotImplementedError


@dataclass(frozen=True)
class Registration(_RegistryEntry, ABC):
    """
    A live partition's registration.

    A registry implementation supplies the method, which reads the registry.
    """

    @abstractmethod
    async def require_current(self) -> None:
        """
        Raise unless this partition is still the one registered under its name.

        One read of the registry: a deletion committed before the read makes
        it raise, and one committed after does not.

        Raises:
            PartitionDeletedError:
                If the partition was deleted, whether or not another was
                registered under the name since.
        """
        raise NotImplementedError


type PurgeRound = Callable[[UUID], Awaitable[bool]]
"""A purge round: purges the stores of the deleted partition whose UUID it is given, and returns whether any found records."""


class PartitionRegistry(ABC):
    """
    The partition registry of one partition namespace, over the stores that hold its partitions.

    A deleted partition's UUID waits on a queue as a tombstone. A write
    checked as live before the deletion can land in a store after it, so a
    tombstone's purge starts once a retention, longer than any write can be in
    flight, has passed since the deletion. The tombstone is removed when a
    purge round finds nothing in any store and has dropped the partition from
    each, and its UUID is not minted again before then.

    A registry implementation supplies the catalog: `_reserve`, `resolve`,
    `unregister`, and `_run_purge_round`. `create_partition` and
    `run_purge_round` call the stores on the catalog's steps, in order.
    """

    def __init__(self, partitioned_stores: Sequence[PartitionedStore]) -> None:
        """Initialize with the stores that hold the namespace's partitions."""
        self._partitioned_stores = tuple(partitioned_stores)

    @property
    @abstractmethod
    def partition_namespace(self) -> str:
        """The partition namespace the registry serves."""
        raise NotImplementedError

    @abstractmethod
    async def startup(self) -> None:
        """
        Make the registry ready for use, creating its durable resources idempotently.

        Its owner calls it before the first use.
        """
        raise NotImplementedError

    async def create_partition(self, partition_name: str) -> Registration:
        """
        Create a partition under a name: reserve the name, create the partition in every store, and confirm the reservation.

        The partition is pending, holding its name, until it is confirmed. A
        creation that raises or is cancelled cancels its reservation, shielded,
        so the name is freed and the partition's purge rounds reclaim whatever
        the stores created. The cancel acts only on a pending partition, so a
        confirmation that committed before its failure or cancellation was
        observed stands. A partition whose reservation the registry cannot
        cancel stays pending until it is deleted.

        Args:
            partition_name (str): The name of the partition.

        Returns:
            Registration: The new partition's registration.

        Raises:
            PartitionAlreadyExistsError:
                If the name is taken, by a live or a pending partition.
            PartitionCreateAttemptsExhaustedError:
                If the registry gave up after repeated attempts to reserve
                the free name failed.
            PartitionDeletedError:
                If the partition was deleted while the stores created it.
            Exception:
                The first error a store's `create_partition` raised, once
                every store's call has finished.
        """
        reservation = await self._reserve(partition_name)
        try:
            await _on_every_store(
                "create",
                reservation.partition_uuid,
                [
                    store.create_partition(reservation.partition_uuid)
                    for store in self._partitioned_stores
                ],
            )
            return await reservation.confirm()
        except BaseException:
            # Shielded, so a cancelled creation still frees the name. The task
            # reports its own failure, since a creation cancelled again stops
            # awaiting it before it ends.
            cancellation = asyncio.create_task(reservation.cancel())
            _reservation_cancellations.add(cancellation)

            def finish(task: asyncio.Task[None]) -> None:
                _reservation_cancellations.discard(task)
                if not task.cancelled() and task.exception() is not None:
                    logger.exception(
                        "Could not cancel the reservation of partition %r after "
                        "its creation failed or was cancelled; it stays pending "
                        "until deleted",
                        reservation.partition_name,
                        exc_info=task.exception(),
                    )

            cancellation.add_done_callback(finish)
            with contextlib.suppress(Exception):
                await asyncio.shield(cancellation)
            raise

    @abstractmethod
    async def _reserve(self, partition_name: str) -> Reservation:
        """
        Reserve a partition name for a new pending partition under a freshly minted UUID.

        The partition name is arbitrated across processes, and the UUID is
        one no registered or queued partition carries, so the new partition
        starts empty and no purge reclaims its records. The partition is
        pending until its reservation is confirmed, and holds its name
        meanwhile.

        Args:
            partition_name (str): The name of the partition.

        Returns:
            Reservation: The new partition's reservation.

        Raises:
            PartitionAlreadyExistsError:
                The partition name is taken, by a live or a pending partition.
            PartitionCreateAttemptsExhaustedError:
                The registry gave up after repeated attempts to reserve
                the free partition name failed.
        """
        raise NotImplementedError

    @abstractmethod
    async def resolve(self, partition_name: str) -> Registration | None:
        """
        Resolve a partition name to the live partition registered under it.

        One read of the registry decides the outcome.

        Args:
            partition_name (str): The name of the partition.

        Returns:
            Registration | None:
                The live partition's registration, or None when no partition
                holds the name.

        Raises:
            PartitionPendingError:
                If the partition registered under the name is pending.
        """
        raise NotImplementedError

    @abstractmethod
    async def unregister(self, partition_name: str) -> None:
        """
        Unregister the partition under a name and queue its UUID for purge.

        The partition, pending or live, is unreachable when this returns,
        which voids its reservation or registration, and purge rounds reclaim
        its records later. Idempotent.

        Args:
            partition_name (str): The name of the partition.
        """
        raise NotImplementedError

    async def run_purge_round(self) -> bool:
        """
        Run one purge round on the tombstone that came due first, and return whether one was due.

        The round purges the partition from every store and, once no store's
        purge finds anything, drops it from every store, which removes the
        tombstone. Each step finishes in every store before the round goes on
        or raises the first store's error, so the stores that answer keep
        purging while another is down.

        Returns:
            bool:
                Whether a tombstone was due.
        """
        return await self._run_purge_round(self._purge_partition)

    @abstractmethod
    async def _run_purge_round(self, purge_round: PurgeRound) -> bool:
        """
        Run `purge_round` on the tombstone that came due first, and return whether one was due.

        A tombstone is due once the retention has passed since its deletion.
        The registry claims it, calls `purge_round` with its partition UUID, and
        records the outcome under the claim: a round that returns False found
        no records, which removes the tombstone and frees its UUID; one that
        returns True keeps the tombstone due. A round that raises is a failed
        round and its error propagates: the tombstone stays for a later call,
        and a registry may delay a failed tombstone's next round, or stop
        running one whose rounds keep failing, leaving its records in place. A
        round must be safe to repeat, since a registry may run one tombstone's
        round on two purgers at once.

        Args:
            purge_round (PurgeRound):
                Purges the stores of the partition whose UUID it is given, and
                returns whether any found records.

        Returns:
            bool:
                Whether a tombstone was due.
        """
        raise NotImplementedError

    async def _purge_partition(self, partition_uuid: UUID) -> bool:
        """Purge a deleted partition from every store and, once none finds anything, drop it from each; return whether any found records."""
        found = await _on_every_store(
            "purge",
            partition_uuid,
            [
                store.purge_partition(partition_uuid)
                for store in self._partitioned_stores
            ],
        )
        if any(found):
            return True
        await _on_every_store(
            "drop",
            partition_uuid,
            [
                store.drop_partition(partition_uuid)
                for store in self._partitioned_stores
            ],
        )
        return False


async def _on_every_store[T](
    step: str, partition_uuid: UUID, calls: Sequence[Awaitable[T]]
) -> list[T]:
    """Run one step of a partition's lifecycle on every store at once, and raise the first error once every store has finished."""
    results = await asyncio.gather(*calls, return_exceptions=True)
    errors = [result for result in results if isinstance(result, BaseException)]
    for error in errors:
        logger.error(
            "A store failed to %s partition %s",
            step,
            partition_uuid,
            exc_info=error,
        )
    if errors:
        raise errors[0]
    return [result for result in results if not isinstance(result, BaseException)]
