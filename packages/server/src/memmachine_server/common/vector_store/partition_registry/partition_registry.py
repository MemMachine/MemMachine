"""
Abstract base classes for a partition registry, its reservations and registrations.

The catalog of a vector store whose backend cannot arbitrate one: which
partitions exist, under which incarnation and schema, and which deleted
incarnations await purge. A registry belongs to one vector store. Its calls
are arbitrated across every process sharing it: a reservation mints an
incarnation no registered or queued partition carries, unregistration makes
the partition unreachable when it returns, and a purge claim hands out a due
tombstone, possibly to two purgers at once.

The registry is addressed by partition key. Reserving a key answers a
`Reservation`, which the partition's creator confirms once the partition's
storage is prepared, or cancels. Confirming it, or resolving a key, answers a
`Registration`, through which a handle checks that the partition has not been
deleted. Each acts on one life of the partition alone, and a deletion by key
voids either.
"""

from abc import ABC, abstractmethod
from contextlib import AbstractAsyncContextManager
from dataclasses import dataclass
from uuid import UUID

from memmachine_server.common.vector_store.data_types import PartitionSchema


@dataclass(frozen=True)
class _RegistryEntry:
    """
    A registry's entry for one life of a partition, from its reservation to its deletion.

    Its fields belong to this life and never change: `incarnation` is the
    value its records carry, and `schema` what it was created under.
    """

    partition_key: str
    schema: PartitionSchema
    incarnation: UUID


@dataclass(frozen=True)
class Reservation(_RegistryEntry, ABC):
    """
    A partition's hold on its key, kept by its creator while it prepares the partition's storage.

    The partition is pending until the reservation is confirmed. A registry
    implementation supplies the methods, each of which writes the registry.
    """

    @abstractmethod
    async def confirm(self) -> "Registration":
        """
        Confirm this reservation, marking the pending partition live.

        Called once, when the partition's storage is prepared. A concurrent
        deletion of the partition either follows the confirmation or makes it
        raise.

        Returns:
            Registration: The same life of the partition, live.

        Raises:
            VectorStorePartitionDeletedError:
                If the partition is no longer pending: it was deleted.
        """
        raise NotImplementedError

    @abstractmethod
    async def cancel(self) -> None:
        """
        Cancel this reservation, unregistering this life of the partition while it is pending and queuing its incarnation for purge.

        The pending partition is unreachable when this returns, and purge
        rounds reclaim its records later. Once the reservation is confirmed,
        cancelling it does nothing: only a deletion by key ends a live
        partition. A partition reserved since under the same key is another
        life and stays. Idempotent.
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
        Raise unless this life is still the partition registered under its key.

        One read of the registry: a deletion committed before the read makes
        it raise, and one committed after does not.

        Raises:
            VectorStorePartitionHandleStaleError:
                If the partition was deleted, whether or not another was
                registered under the key since.
        """
        raise NotImplementedError


@dataclass
class PurgeClaim:
    """
    One purge round's claim on a tombstone.

    The registry fills in `incarnation`, the value the deleted partition's
    records carry in the store. The round sets `any_records_found` before the
    claim ends: whether it found records under the incarnation.
    """

    incarnation: UUID
    any_records_found: bool | None = None


class VectorStorePartitionRegistry(ABC):
    """
    The partition registry of one vector store, identified by the store's name.

    A store name identifies one store among all the stores whose registries
    are kept together, so stores that must stay apart have distinct names.

    A deleted partition's incarnation waits on a queue as a tombstone. A
    write checked as live before the deletion can land in the backend after
    it, so a tombstone's purge starts once a retention, longer than any
    write can be in flight, has passed since the deletion. The tombstone is
    removed when a purge round finds nothing, and its incarnation is not
    minted again before then.
    """

    @abstractmethod
    async def startup(self) -> None:
        """
        Make the registry ready for use, creating its durable resources idempotently.

        Its owner calls it before the first use.
        """
        raise NotImplementedError

    @abstractmethod
    async def reserve(self, partition_key: str, schema: PartitionSchema) -> Reservation:
        """
        Reserve a partition key for a new pending partition under a freshly minted incarnation.

        The partition key is arbitrated across processes, and the
        incarnation is one no registered or queued partition carries, so the
        new partition starts empty and no purge reclaims its records. The
        partition is pending until its reservation is confirmed, and holds its
        key meanwhile.

        Args:
            partition_key (str): The key of the partition.
            schema (PartitionSchema):
                What the partition is created under: its store's
                dimensions and declared schema.

        Returns:
            Reservation: The new partition's reservation.

        Raises:
            VectorStorePartitionAlreadyExistsError:
                The partition key is taken, by a live or a pending partition.
            VectorStoreAttemptsExhaustedError:
                The registry gave up after repeated attempts to reserve
                the free partition key failed.
        """
        raise NotImplementedError

    @abstractmethod
    async def resolve(self, partition_key: str) -> Registration | None:
        """
        Resolve a partition key to the live partition registered under it.

        One read of the registry decides the outcome.

        Args:
            partition_key (str): The key of the partition.

        Returns:
            Registration | None:
                The live partition's registration, or None when no partition
                holds the key.

        Raises:
            VectorStorePartitionPendingError:
                If the partition registered under the key is pending.
        """
        raise NotImplementedError

    @abstractmethod
    async def unregister(self, partition_key: str) -> None:
        """
        Unregister the partition under a key and queue its incarnation for purge.

        The partition, pending or live, is unreachable when this returns,
        which voids its reservation or registration, and purge rounds reclaim
        its records later. Idempotent.

        Args:
            partition_key (str): The key of the partition.
        """
        raise NotImplementedError

    @abstractmethod
    def claim_purgeable_incarnation(
        self,
    ) -> AbstractAsyncContextManager[PurgeClaim | None]:
        """
        Claim a due tombstone for one purge round, run in the body of the context.

        A tombstone is due once the retention has passed since its
        deletion. In the body, the caller deletes records under
        `claim.incarnation` in the store and sets `claim.any_records_found`.
        A round that found no records removes the tombstone and frees its
        incarnation. A body that raises is a failed round, and the tombstone
        stays for a later claim; a registry may delay a failed tombstone's
        next claim, or stop claiming one whose rounds keep failing, leaving
        its records in place. A round must be safe to repeat, since a
        registry may hand one tombstone to two purgers.

        Returns:
            AbstractAsyncContextManager[PurgeClaim | None]:
                The claim, or None when no tombstone is due.
        """
        raise NotImplementedError
