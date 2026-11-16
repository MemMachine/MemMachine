"""
Abstract base classes for a partition registry, its reservations, and registrations.

A partition registry names the partitions of one partition namespace and
orders each partition's lifecycle, running the steps the namespace's owner
supplies on the owner's stores at their places in it. A partition has a name,
which its owner chooses and may reuse, and a UUID, which the registry mints
for it alone; the owner's stores hold the partition's records under the UUID.
The registry's methods address partitions by name, as the stores' address
them by UUID. Its calls are arbitrated across every process sharing it: a
reservation mints a UUID no registered or queued partition carries, a deletion
makes the partition unreachable when it returns, and a purge round runs on a
due tombstone, possibly on two purgers at once.

Creating a partition, or getting one by name, answers a `Registration`, which
carries the partition's UUID and through which an owner checks that the
partition has not been deleted. A registration acts on one partition alone,
and a deletion by name voids it.
"""

import asyncio
import contextlib
import logging
from abc import ABC, abstractmethod
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from uuid import UUID

from .data_types import validate_partition_name
from .partitioned_store import PartitionedStore

logger = logging.getLogger(__name__)

# The tasks cancelling failed creations' reservations, held until done so the
# garbage collector cannot drop one whose creation stopped awaiting it.
_reservation_cancellations: set[asyncio.Task[None]] = set()


@dataclass(frozen=True)
class Reservation(ABC):
    """
    A pending partition's hold on its name, kept while the partition's storage is created.

    The partition is pending until the reservation is confirmed. A registry
    implementation supplies the methods, each of which writes the registry.
    """

    partition_name: str
    partition_uuid: UUID

    @abstractmethod
    async def confirm(self) -> None:
        """
        Confirm this reservation, marking the pending partition live.

        Called once, when the partition's storage is created. A concurrent
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

        Called by the registry's creation, in a task of its own, when
        creating the partition's storage or confirming it raised or was
        cancelled; it may follow a confirmation that committed before its
        failure was observed, and race a deletion by name. The pending
        partition is unreachable when this returns, and purge rounds reclaim
        its records later. Once the reservation is confirmed, cancelling it
        does nothing: only a deletion by name ends a live partition. A
        partition reserved since under the same name is another partition
        and stays. Idempotent.

        A call that raises leaves the partition pending until it is deleted
        by name.
        """
        raise NotImplementedError


@dataclass(frozen=True)
class Registration(ABC):
    """
    A live partition's registration.

    Its fields never change: `partition_uuid` is the value the partition's
    records carry. A registry implementation supplies the method, which reads
    the registry.
    """

    partition_name: str
    partition_uuid: UUID

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
"""
A purge round of a deleted partition, given its UUID: purges the partition from its owner's stores and, once they find none of its records, drops it from them.

Returns whether the stores found records: False once none of the
partition's records remains and the partition is dropped, True when a later
round may find some. Safe to repeat, and to run on several processes at once.
"""


class PartitionRegistry(ABC):
    """
    The partition registry of one partition namespace.

    A deleted partition's UUID waits on a queue as a tombstone. A write
    checked as live before the deletion can land in a store after it, so a
    tombstone's purge starts once a retention, longer than any write can be in
    flight, has passed since the deletion. The registry deletes the tombstone
    once a purge round finds none of the partition's records and has dropped
    the partition, and its UUID is not minted again before then.

    A registry implementation supplies the catalog: `_reserve`, `_resolve`,
    `_registration`, `_unregister`, and `run_purge_round`. The registry's
    public methods run the owner's steps between the catalog's, in order.
    """

    @property
    @abstractmethod
    def partition_namespace(self) -> str:
        """The partition namespace the registry serves."""
        raise NotImplementedError

    @abstractmethod
    async def startup(self) -> None:
        """
        Make the registry ready for use, creating its durable resources idempotently.

        Whoever builds the registry calls it before the first use.
        """
        raise NotImplementedError

    async def create_partition(
        self,
        partition_name: str,
        create_partition_storage: Callable[[UUID], Awaitable[None]],
    ) -> Registration:
        """
        Create a partition under a name: reserve the name, create the partition's storage, and confirm the reservation.

        The partition is pending, holding its name, until it is confirmed. A
        creation that raises or is cancelled cancels its reservation, shielded,
        so the name is freed and the partition's purge rounds reclaim whatever
        storage was created. The cancel acts only on a pending partition, so a
        confirmation that committed before its failure or cancellation was
        observed stands. A partition whose reservation the registry cannot
        cancel stays pending until it is deleted.

        Args:
            partition_name (str): The name of the partition.
            create_partition_storage (Callable[[UUID], Awaitable[None]]):
                Creates the storage the partition keeps for itself in the
                owner's stores, given the partition's UUID. Called once,
                before the reservation is confirmed.

        Returns:
            Registration: The new partition's registration.

        Raises:
            ValueError:
                If the partition name is invalid.
            PartitionAlreadyExistsError:
                If the name is taken, by a live or a pending partition.
            PartitionCreateAttemptsExhaustedError:
                If the registry gave up after repeated attempts to reserve
                the free name failed.
            PartitionDeletedError:
                If the partition was deleted while its storage was created.
            Exception:
                Whatever `create_partition_storage` raised.
        """
        validate_partition_name(partition_name)
        reservation = await self._reserve(partition_name)
        try:
            await create_partition_storage(reservation.partition_uuid)
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
        return self._registration(partition_name, reservation.partition_uuid)

    @abstractmethod
    async def _reserve(self, partition_name: str) -> Reservation:
        """
        Reserve a partition name for a new pending partition under a freshly minted UUID.

        Called by `create_partition` with a valid name, concurrently with
        any other call on any process. The partition name is arbitrated
        across processes, and the UUID is one no registered or queued
        partition carries, so the new partition starts empty and no purge
        reclaims its records. The partition is pending until its reservation
        is confirmed, and holds its name meanwhile.

        Args:
            partition_name (str): The name of the partition.

        Returns:
            Reservation: The new partition's reservation.

        Raises:
            PartitionAlreadyExistsError:
                If the partition name is taken, by a live or a pending
                partition.
            PartitionCreateAttemptsExhaustedError:
                If the registry gave up after repeated attempts to reserve
                the free partition name failed.
        """
        raise NotImplementedError

    async def get_partition(self, partition_name: str) -> Registration | None:
        """
        Get the live partition registered under a name.

        One read of the registry decides the outcome.

        Args:
            partition_name (str): The name of the partition.

        Returns:
            Registration | None:
                The live partition's registration, or None when no partition
                holds the name.

        Raises:
            ValueError:
                If the partition name is invalid.
            PartitionPendingError:
                If the partition registered under the name is pending.
        """
        validate_partition_name(partition_name)
        partition_uuid = await self._resolve(partition_name)
        if partition_uuid is None:
            return None
        return self._registration(partition_name, partition_uuid)

    @abstractmethod
    async def _resolve(self, partition_name: str) -> UUID | None:
        """
        Resolve a partition name to the UUID of the live partition registered under it, or None when no partition holds the name.

        Called by `get_partition` with a valid name, concurrently with any
        other call on any process. One read of the registry decides the
        outcome: a creation, confirmation, or deletion committed before the
        read is reflected, and one committed after is not. It only reads, so
        a call that raises changes nothing.

        Args:
            partition_name (str): The name of the partition.

        Returns:
            UUID | None:
                The live partition's UUID, or None when no partition holds
                the name.

        Raises:
            PartitionPendingError:
                If the partition registered under the name is pending.
        """
        raise NotImplementedError

    @abstractmethod
    def _registration(self, partition_name: str, partition_uuid: UUID) -> Registration:
        """
        Build the registration of a live partition.

        Called by `create_partition` once the partition is confirmed, and by
        `get_partition` once `_resolve` answered it. It builds the object
        alone, reading and writing nothing, so a partition deleted since is
        reported by the registration's `require_current`.

        Args:
            partition_name (str): The name the partition is registered under.
            partition_uuid (UUID): The partition's UUID.

        Returns:
            Registration: The partition's registration.
        """
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
                If the partition name is invalid.
        """
        validate_partition_name(partition_name)
        await self._unregister(partition_name)

    @abstractmethod
    async def _unregister(self, partition_name: str) -> None:
        """
        Unregister the partition under a name, pending or live, and queue its UUID for purge, in one transaction.

        Called by `delete_partition` with a valid name, concurrently with
        any other call on any process. Once it returns, the partition's
        reservation cannot be confirmed, its registrations' `require_current`
        raises, and its tombstone is queued as of the deletion, so its purge
        starts once the retention has passed. Idempotent: racing deletions
        of one partition queue one tombstone, and a name no partition holds
        is left as it is. A call that raises either committed both writes or
        neither, so the caller may call again.

        Args:
            partition_name (str): The name of the partition.
        """
        raise NotImplementedError

    @abstractmethod
    async def run_purge_round(self, purge_round: PurgeRound) -> bool:
        """
        Run `purge_round` on the tombstone that came due first, and return whether it ran one.

        A tombstone is due once the retention has passed since its deletion.
        The registry claims it, calls `purge_round` with its partition UUID,
        and records the outcome under the claim: a round that returns False
        found none of the partition's records and dropped the partition, so
        the registry deletes the tombstone and frees its UUID; one that
        returns True keeps the tombstone due. A round that raises is a failed
        round and its error propagates: the tombstone stays for a later call,
        and a registry may delay a failed tombstone's next round, or stop
        running one whose rounds keep failing, leaving its records in place.
        A registry may run one tombstone's round on two purgers at once.

        Args:
            purge_round (PurgeRound):
                The purge round of the namespace's partitions in its owner's
                stores.

        Returns:
            bool:
                Whether a round ran: False also when every due tombstone is
                claimed by another round, delayed after failed rounds, or no
                longer run.
        """
        raise NotImplementedError


def single_store_purge_round[PartitionT](
    partitioned_store: PartitionedStore[PartitionT],
) -> PurgeRound:
    """The purge round of a namespace whose partitions one store holds: purge the store and, once its purge finds nothing, drop the partition from it."""

    async def purge_round(partition_uuid: UUID) -> bool:
        if await partitioned_store.purge_partition(partition_uuid):
            return True
        await partitioned_store.drop_partition(partition_uuid)
        return False

    return purge_round
