"""
Abstract base classes for a partition registry and its registrations.

The catalog of a vector store whose backend cannot arbitrate one: which
partitions exist, under which incarnation and schema, and which deleted
incarnations await purge. A registry belongs to one vector store. Its calls
are arbitrated across every process sharing it: registration mints an
incarnation no registered or queued partition carries, unregistration makes
the partition unreachable when it returns, and a purge claim hands out a due
tombstone, possibly to two purgers at once.

The registry is addressed by partition key. Registering a partition answers
a `PendingRegistration`, through which its creator marks it live or abandons
it; resolving a key answers a `LiveRegistration`, through which a handle
checks that the partition has not been deleted. Each acts on one life of the
partition alone.
"""

from abc import ABC, abstractmethod
from contextlib import AbstractAsyncContextManager
from dataclasses import dataclass
from uuid import UUID

from memmachine_server.common.vector_store.data_types import PartitionSchema


@dataclass(frozen=True)
class Registration:
    """
    One life of a registered partition, from its registration to its deletion.

    Its fields belong to this life and never change: `incarnation` is the
    value its records carry, and `schema` what it was created under.
    """

    partition_key: str
    schema: PartitionSchema
    incarnation: UUID


@dataclass(frozen=True)
class PendingRegistration(Registration, ABC):
    """
    A partition's registration while its creator prepares its storage.

    A registry implementation supplies the methods, each of which writes the
    registry.
    """

    @abstractmethod
    async def mark_live(self) -> "LiveRegistration":
        """
        Mark this pending partition as live.

        Called once, when the partition's storage is prepared. A concurrent
        deletion of the partition either follows the mark or makes it raise.

        Returns:
            LiveRegistration: The same life of the partition, live.

        Raises:
            VectorStorePartitionDeletedError:
                If this life is no longer pending: the partition was
                deleted.
        """
        raise NotImplementedError

    @abstractmethod
    async def unregister(self) -> None:
        """
        Unregister this life of the partition and queue its incarnation for purge.

        The partition is unreachable when this returns, and purge rounds
        reclaim its records later. A partition registered since under the
        same key is another life and stays. Idempotent.
        """
        raise NotImplementedError


@dataclass(frozen=True)
class LiveRegistration(Registration, ABC):
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
    async def register(
        self, partition_key: str, schema: PartitionSchema
    ) -> PendingRegistration:
        """
        Register a new pending partition under a freshly minted incarnation.

        The partition key is arbitrated across processes, and the
        incarnation is one no registered or queued partition carries, so the
        new partition starts empty and no purge reclaims its records. The
        partition is pending until its registration's `mark_live`, and holds
        its key meanwhile.

        Args:
            partition_key (str): The key of the partition.
            schema (PartitionSchema):
                What the partition is created under: its store's
                dimensions and declared schema.

        Returns:
            PendingRegistration: The new partition's registration.

        Raises:
            VectorStorePartitionAlreadyExistsError:
                The partition key is taken, by a live or a pending partition.
            VectorStoreAttemptsExhaustedError:
                The registry gave up after repeated attempts to register
                the free partition key failed.
        """
        raise NotImplementedError

    @abstractmethod
    async def resolve(self, partition_key: str) -> LiveRegistration | None:
        """
        Resolve a partition key to the live partition registered under it.

        One read of the registry decides the outcome.

        Args:
            partition_key (str): The key of the partition.

        Returns:
            LiveRegistration | None:
                The live partition's registration, or None when no partition
                is registered under the key.

        Raises:
            VectorStorePartitionPendingError:
                If the partition registered under the key is pending.
        """
        raise NotImplementedError

    @abstractmethod
    async def unregister(self, partition_key: str) -> None:
        """
        Unregister the partition under a key and queue its incarnation for purge.

        The partition, pending or live, is unreachable when this returns, and
        purge rounds reclaim its records later. Idempotent.

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
