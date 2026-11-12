"""
Abstract base classes for a partition registry, its reservations, and registrations.

The catalog of one partition namespace, for stores whose backends cannot
arbitrate one: which partitions exist under which names, and which deleted
partitions await purge. A partition has a name, which its owner chooses and
may reuse, and a UUID, which the registry mints for it alone. Its calls are
arbitrated across every process sharing it: a reservation mints a UUID no
registered or queued partition carries, unregistration makes the partition unreachable
when it returns, and a purge round runs on a due tombstone, possibly on two
purgers at once.

The registry is addressed by partition name. Reserving a name answers a
`Reservation`, which the partition's creator confirms once the partition's
storage is created, or cancels. Confirming it, or resolving a name, answers a
`Registration`, through which a handle checks that the partition has not been
deleted. Each acts on one partition alone, and a deletion by name voids
either.
"""

from abc import ABC, abstractmethod
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from uuid import UUID


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
    A partition's hold on its name, kept by its creator while it creates the partition's storage.

    The partition is pending until the reservation is confirmed. A registry
    implementation supplies the methods, each of which writes the registry.
    """

    @abstractmethod
    async def confirm(self) -> "Registration":
        """
        Confirm this reservation, marking the pending partition live.

        Called once, when the partition's storage is created. A concurrent
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
"""A purge round: deletes the records of the deleted partition whose UUID it is given, and returns whether it found any."""


class PartitionRegistry(ABC):
    """
    The partition registry of one partition namespace.

    A deleted partition's UUID waits on a queue as a tombstone. A
    write checked as live before the deletion can land in the backend after
    it, so a tombstone's purge starts once a retention, longer than any
    write can be in flight, has passed since the deletion. The tombstone is
    removed when a purge round finds nothing, and its UUID is not minted
    again before then.
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

        Its owner calls it before the first use.
        """
        raise NotImplementedError

    @abstractmethod
    async def reserve(self, partition_name: str) -> Reservation:
        """
        Reserve a partition name for a new pending partition under a freshly minted UUID.

        The partition name is arbitrated across processes, and the UUID is
        one no registered or queued partition carries, so the new partition
        starts empty and no purge reclaims its records. The
        partition is pending until its reservation is confirmed, and holds its
        name meanwhile.

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

    @abstractmethod
    async def run_purge_round(self, purge_round: PurgeRound) -> bool:
        """
        Run one purge round on the tombstone that came due first, and return whether one was due.

        A tombstone is due once the retention has passed since its deletion.
        The registry claims it, calls `purge_round` with its partition UUID, and
        records the outcome under the claim: a round that returns False found
        no records, which removes the tombstone and frees its UUID; one
        that returns True keeps the tombstone due. A round that raises is a
        failed round and its error propagates: the tombstone stays for a later
        call, and a registry may delay a failed tombstone's next round, or stop
        running one whose rounds keep failing, leaving its records in place. A
        round must be safe to repeat, since a registry may run one tombstone's
        round on two purgers at once.

        Args:
            purge_round (PurgeRound):
                Deletes the records of the partition whose UUID it is given,
                and returns whether it found any.

        Returns:
            bool:
                Whether a tombstone was due.
        """
        raise NotImplementedError
