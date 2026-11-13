"""
Abstract base classes for a partition registry, its reservations, and registrations.

A partition registry names the partitions of one partition namespace and
drives each partition's lifecycle in the namespace's store. A partition has a
name, which its owner chooses and may reuse, and a UUID, which the registry
mints for it alone; the store holds the partition's records under the UUID.
The registry's methods address partitions by name, as the store's address them
by UUID. Its calls are arbitrated across every process sharing it: a
reservation mints a UUID no registered or queued partition carries, a deletion
makes the partition unreachable when it returns, and a purge round runs on a
due tombstone, possibly on two purgers at once.

Creating a partition, or getting one by name, answers a `Registration`, which
carries the store's handle on the partition and through which an owner checks
that the partition has not been deleted. A registration acts on one partition
alone, and a deletion by name voids it.
"""

import asyncio
import contextlib
import logging
from abc import ABC, abstractmethod
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from uuid import UUID

from .data_types import PartitionDeletedError, validate_partition_name
from .partitioned_store import PartitionedStore

logger = logging.getLogger(__name__)

# Reservations cancelled after a failed creation, held until done so the
# garbage collector cannot drop one whose creation was cancelled.
_reservation_cancellations: set[asyncio.Task[None]] = set()


@dataclass(frozen=True)
class Reservation(ABC):
    """
    A pending partition's hold on its name, kept while the registry creates the partition in its store.

    The partition is pending until the reservation is confirmed. A registry
    implementation supplies the methods, each of which writes the registry.
    """

    partition_name: str
    partition_uuid: UUID

    @abstractmethod
    async def confirm(self) -> None:
        """
        Confirm this reservation, marking the pending partition live.

        Called once, when the store has created the partition. A concurrent
        deletion of the partition either follows the confirmation or makes it
        raise.

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
class Registration[PartitionT](ABC):
    """
    A live partition's registration, carrying the store's handle on the partition.

    Its fields never change: `partition_uuid` is the value the partition's
    records carry, and `partition` the store's handle on them. A registry
    implementation supplies the method, which reads the registry.
    """

    partition_name: str
    partition_uuid: UUID
    partition: PartitionT

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
"""A purge round: purges the store of the deleted partition whose UUID it is given, and returns whether it found records."""


class PartitionRegistry[PartitionT](ABC):
    """
    The partition registry of one partition namespace, over the store that holds its partitions.

    A deleted partition's UUID waits on a queue as a tombstone. A write
    checked as live before the deletion can land in the store after it, so a
    tombstone's purge starts once a retention, longer than any write can be in
    flight, has passed since the deletion. The tombstone is removed when a
    purge round finds nothing in the store and has dropped the partition from
    it, and its UUID is not minted again before then.

    A registry implementation supplies the catalog: `_reserve`, `_resolve`,
    `_registration`, `_unregister`, and `_run_purge_round`. The registry's
    public methods call the store on the catalog's steps, in order.
    """

    def __init__(self, partitioned_store: PartitionedStore[PartitionT]) -> None:
        """Initialize with the store that holds the namespace's partitions."""
        self._partitioned_store = partitioned_store

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

    async def create_partition(self, partition_name: str) -> Registration[PartitionT]:
        """
        Create a partition under a name: reserve the name, create the partition in the store, and confirm the reservation.

        The partition is pending, holding its name, until it is confirmed. A
        creation that raises or is cancelled cancels its reservation, shielded,
        so the name is freed and the partition's purge rounds reclaim whatever
        the store created. The cancel acts only on a pending partition, so a
        confirmation that committed before its failure or cancellation was
        observed stands. A partition whose reservation the registry cannot
        cancel stays pending until it is deleted.

        Args:
            partition_name (str): The name of the partition.

        Returns:
            Registration[PartitionT]: The new partition's registration.

        Raises:
            ValueError:
                If the partition name is empty, too long, or contains a NUL
                character.
            PartitionAlreadyExistsError:
                If the name is taken, by a live or a pending partition.
            PartitionCreateAttemptsExhaustedError:
                If the registry gave up after repeated attempts to reserve
                the free name failed.
            PartitionDeletedError:
                If the partition was deleted while the store created it.
            Exception:
                Whatever the store's `create_partition` raised.
        """
        validate_partition_name(partition_name)
        reservation = await self._reserve(partition_name)
        try:
            await self._partitioned_store.create_partition(reservation.partition_uuid)
            await reservation.confirm()
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
        return await self._get_registration(partition_name, reservation.partition_uuid)

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

    async def get_partition(
        self, partition_name: str
    ) -> Registration[PartitionT] | None:
        """
        Get the live partition registered under a name.

        One read of the registry decides the outcome; the store then answers
        its handle on the partition.

        Args:
            partition_name (str): The name of the partition.

        Returns:
            Registration[PartitionT] | None:
                The live partition's registration, or None when no partition
                holds the name.

        Raises:
            ValueError:
                If the partition name is empty, too long, or contains a NUL
                character.
            PartitionPendingError:
                If the partition registered under the name is pending.
            PartitionDeletedError:
                If the store holds nothing of the partition: it was deleted
                and its purge has dropped it.
        """
        validate_partition_name(partition_name)
        partition_uuid = await self._resolve(partition_name)
        if partition_uuid is None:
            return None
        return await self._get_registration(partition_name, partition_uuid)

    @abstractmethod
    async def _resolve(self, partition_name: str) -> UUID | None:
        """
        Resolve a partition name to the UUID of the live partition registered under it, or None when no partition holds the name.

        One read of the registry decides the outcome.

        Raises:
            PartitionPendingError:
                If the partition registered under the name is pending.
        """
        raise NotImplementedError

    async def _get_registration(
        self, partition_name: str, partition_uuid: UUID
    ) -> Registration[PartitionT]:
        """Get the store's handle on a live partition and answer its registration."""
        partition = await self._partitioned_store.get_partition(partition_uuid)
        if partition is None:
            raise PartitionDeletedError(
                self.partition_namespace, partition_name, partition_uuid
            )
        return self._registration(partition_name, partition_uuid, partition)

    @abstractmethod
    def _registration(
        self, partition_name: str, partition_uuid: UUID, partition: PartitionT
    ) -> Registration[PartitionT]:
        """Build the registration of a live partition, carrying the store's handle on it."""
        raise NotImplementedError

    async def delete_partition(self, partition_name: str) -> None:
        """
        Delete the partition registered under a name, pending or live, and queue its UUID for purge.

        The partition is unreachable when this returns, which voids its
        reservation or registration, and purge rounds reclaim its records
        later. Idempotent.

        Args:
            partition_name (str): The name of the partition.

        Raises:
            ValueError:
                If the partition name is empty, too long, or contains a NUL
                character.
        """
        validate_partition_name(partition_name)
        await self._unregister(partition_name)

    @abstractmethod
    async def _unregister(self, partition_name: str) -> None:
        """Unregister the partition under a name, pending or live, and queue its UUID for purge, in one transaction. Idempotent."""
        raise NotImplementedError

    async def run_purge_round(self) -> bool:
        """
        Run one purge round on the tombstone that came due first, and return whether one was due.

        The round purges the partition from the store and, once the store's
        purge finds nothing, drops it from the store, which removes the
        tombstone.

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
                Purges the store of the partition whose UUID it is given, and
                returns whether it found records.

        Returns:
            bool:
                Whether a tombstone was due.
        """
        raise NotImplementedError

    async def _purge_partition(self, partition_uuid: UUID) -> bool:
        """Purge a deleted partition from the store and, once the store finds nothing, drop it; return whether the store found records."""
        if await self._partitioned_store.purge_partition(partition_uuid):
            return True
        await self._partitioned_store.drop_partition(partition_uuid)
        return False
