"""
Abstract base classes for a collection registry, its reservations, and registrations.

The catalog of a vector store whose backend cannot arbitrate one: which
logical collections exist, under which incarnation and configuration, and
which deleted incarnations await purge. Its calls are arbitrated across
every process sharing it: a reservation mints an incarnation no registered
or queued collection carries, unregistration makes the collection
unreachable when it returns, and a purge round runs on a due tombstone,
possibly on two purgers at once.

The registry is addressed by (namespace, name). Reserving a name answers a
`Reservation`, which the collection's creator confirms once the collection's
storage is prepared, or cancels. Confirming it, or resolving a name, answers
a `Registration`, through which a handle checks that the collection has not
been deleted. Each acts on one life of the collection alone, and a deletion
by name voids either.
"""

from abc import ABC, abstractmethod
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from uuid import UUID

from memmachine_server.common.vector_store.data_types import (
    VectorStoreCollectionConfig,
)


@dataclass(frozen=True)
class _RegistryEntry:
    """
    A registry's entry for one life of a collection, from its reservation to its deletion.

    Its fields belong to this life and never change: `incarnation` is the
    value its records carry, and `config` the configuration it was created
    with.
    """

    namespace: str
    name: str
    config: VectorStoreCollectionConfig
    incarnation: UUID


@dataclass(frozen=True)
class Reservation(_RegistryEntry, ABC):
    """
    A collection's hold on its (namespace, name), kept by its creator while it prepares the collection's storage.

    The collection is pending until the reservation is confirmed. A registry
    implementation supplies the methods, each of which writes the registry.
    """

    @abstractmethod
    async def confirm(self) -> "Registration":
        """
        Confirm this reservation, marking the pending collection live.

        Called once, when the collection's storage is prepared. A concurrent
        deletion of the collection either follows the confirmation or makes
        it raise.

        Returns:
            Registration: The same life of the collection, live.

        Raises:
            VectorStoreCollectionDeletedError:
                If the collection is no longer pending: it was deleted.
        """
        raise NotImplementedError

    @abstractmethod
    async def cancel(self) -> None:
        """
        Cancel this reservation, unregistering this life of the collection while it is pending and queuing its incarnation for purge.

        The pending collection is unreachable when this returns, and purge
        rounds reclaim its records later. Once the reservation is confirmed,
        cancelling it does nothing: only a deletion by (namespace, name) ends
        a live collection. A collection reserved since under the same
        (namespace, name) is another life and stays. Idempotent.
        """
        raise NotImplementedError


@dataclass(frozen=True)
class Registration(_RegistryEntry, ABC):
    """
    A live collection's registration.

    A registry implementation supplies the method, which reads the registry.
    """

    @abstractmethod
    async def require_current(self) -> None:
        """
        Raise unless this life is still the collection registered under its (namespace, name).

        One read of the registry: a deletion committed before the read makes
        it raise, and one committed after does not.

        Raises:
            VectorStoreCollectionHandleStaleError:
                If the collection was deleted, whether or not another was
                registered under the (namespace, name) since.
        """
        raise NotImplementedError


type PurgeRound = Callable[[str, VectorStoreCollectionConfig, UUID], Awaitable[bool]]
"""A purge round: deletes the records under an incarnation, which the namespace and configuration locate, and returns whether it found any."""


class VectorStoreCollectionRegistry(ABC):
    """
    The collection registry of one vector store.

    A deleted collection's incarnation waits on a queue as a tombstone. A
    write checked as live before the deletion can land in the backend after
    it, so a tombstone's purge starts once a retention, longer than any
    write can be in flight, has passed since the deletion. The tombstone is
    removed when a purge round finds nothing, and its incarnation is not
    minted again before then.
    """

    @abstractmethod
    async def startup(self) -> None:
        """Make the registry ready for use; its owner calls it before the first use."""
        raise NotImplementedError

    @abstractmethod
    async def reserve(
        self, namespace: str, name: str, config: VectorStoreCollectionConfig
    ) -> Reservation:
        """
        Reserve a (namespace, name) for a new pending collection under a freshly minted incarnation.

        The (namespace, name) is arbitrated across processes, and the
        incarnation is one no registered or queued collection carries, so the
        new collection starts empty and no purge reclaims its records. The
        collection is pending until its reservation is confirmed, and holds
        its (namespace, name) meanwhile.

        Args:
            namespace (str): Namespace of the collection.
            name (str): Name of the collection within the namespace.
            config (VectorStoreCollectionConfig):
                The configuration the collection is created with.

        Returns:
            Reservation: The new collection's reservation.

        Raises:
            VectorStoreCollectionAlreadyExistsError:
                The (namespace, name) is taken, by a live or a pending
                collection.
            VectorStoreAttemptsExhaustedError:
                The registry gave up after repeated attempts to reserve
                the free (namespace, name) failed.
        """
        raise NotImplementedError

    @abstractmethod
    async def resolve(self, namespace: str, name: str) -> Registration | None:
        """
        Resolve a (namespace, name) to the live collection registered under it.

        One read of the registry decides the outcome.

        Args:
            namespace (str): Namespace of the collection.
            name (str): Name of the collection within the namespace.

        Returns:
            Registration | None:
                The live collection's registration, or None when no
                collection holds the (namespace, name).

        Raises:
            VectorStoreCollectionPendingError:
                If the collection registered under the (namespace, name) is
                pending.
        """
        raise NotImplementedError

    @abstractmethod
    async def unregister(self, namespace: str, name: str) -> None:
        """
        Unregister the collection under a (namespace, name) and queue its incarnation for purge.

        The collection, pending or live, is unreachable when this returns,
        which voids its reservation or registration, and purge rounds reclaim
        its records later. Idempotent.

        Args:
            namespace (str): Namespace of the collection.
            name (str): Name of the collection within the namespace.
        """
        raise NotImplementedError

    @abstractmethod
    async def run_purge_round(self, purge_round: PurgeRound) -> bool:
        """
        Run one purge round on the tombstone that came due first, and return whether one was due.

        A tombstone is due once the retention has passed since its deletion.
        The registry claims it, calls `purge_round` with its namespace,
        configuration, and incarnation, and records the outcome under the claim:
        a round that returns False found no records, which removes the tombstone
        and frees its incarnation; one that returns True keeps the tombstone due.
        A round that raises is a failed round and its error propagates: the
        tombstone stays for a later call, and a registry may delay a failed
        tombstone's next round, or stop running one whose rounds keep failing,
        leaving its records in place. A round must be safe to repeat, since a
        registry may run one tombstone's round on two purgers at once.

        Args:
            purge_round (PurgeRound):
                Deletes the records under the incarnation it is given, which the
                namespace and configuration locate, and returns whether it found
                any.

        Returns:
            bool:
                Whether a tombstone was due.
        """
        raise NotImplementedError
