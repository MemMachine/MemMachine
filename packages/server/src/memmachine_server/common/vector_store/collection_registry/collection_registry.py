"""
Abstract base class for a collection registry.

The catalog of a vector store whose backend holds only points: which
logical collections exist, under which incarnation and configuration, and
which dead incarnations await purge. A registry keys its collections by
namespace and name. Its calls are arbitrated across every process sharing
it: registration mints an incarnation no live or queued collection
carries, unregistration makes the collection unreachable when it returns, and a purge
claim is handed to one purger at a time.

A registry belongs to one vector deployment: every store whose client
reaches that deployment uses it, and stores on other deployments use
other registries, since a store reclaims its registry's tombstones
through its own client.
"""

from abc import ABC, abstractmethod
from contextlib import AbstractAsyncContextManager
from dataclasses import dataclass
from uuid import UUID

from memmachine_server.common.vector_store.data_types import (
    VectorStoreCollectionConfig,
)


@dataclass(frozen=True)
class RegisteredCollection:
    """A live collection: the incarnation its points carry and the configuration it was created with."""

    incarnation: UUID
    config: VectorStoreCollectionConfig


@dataclass
class PurgeClaim:
    """
    One purge round's hold on a tombstone: what the round needs, and what it found.

    The registry fills in what the round needs to find the dead
    incarnation's points: `incarnation`, the value they carry, and
    `namespace` and `config`, which name the native collection they are
    in. The round sets `found` before the claim ends: True when points
    remained under the incarnation, False when none did. The registry
    records that outcome when the claim ends.
    """

    incarnation: UUID
    namespace: str
    config: VectorStoreCollectionConfig
    found: bool | None = None


class VectorStoreCollectionRegistry(ABC):
    """
    The collection registry of one vector store.

    A queue entry is a dead incarnation's tombstone. The backend holds the
    points, and a write the registry read as live can land there after the
    deletion, so purging starts only once a retention has passed since the
    deletion, longer than any write can be in flight: nothing more lands
    under the incarnation after that. The entry stays through purge rounds
    until one finds nothing, and is then removed; until then the
    incarnation is never re-minted.
    """

    @abstractmethod
    async def startup(self) -> None:
        """Startup."""
        raise NotImplementedError

    @abstractmethod
    async def register(
        self, namespace: str, name: str, config: VectorStoreCollectionConfig
    ) -> UUID:
        """
        Register a new collection under a freshly minted incarnation.

        The (namespace, name) is arbitrated across processes, and the
        incarnation is one no live or queued collection carries, so no
        points can be adopted by, or reclaimed out from under, the new
        collection.

        Args:
            namespace (str): Namespace of the collection.
            name (str): Name of the collection within the namespace.
            config (VectorStoreCollectionConfig):
                The configuration the collection is created with.

        Returns:
            UUID: The incarnation the collection's points carry.

        Raises:
            VectorStoreCollectionAlreadyExistsError: The (namespace, name) is taken.
            VectorStoreAttemptsExhaustedError:
                Every minted incarnation was rejected for another reason.
        """
        raise NotImplementedError

    @abstractmethod
    async def get(self, namespace: str, name: str) -> RegisteredCollection | None:
        """
        Look up the live collection under a (namespace, name).

        Args:
            namespace (str): Namespace of the collection.
            name (str): Name of the collection within the namespace.

        Returns:
            RegisteredCollection | None:
                The live collection, or None when there is none.
        """
        raise NotImplementedError

    @abstractmethod
    async def is_live(self, incarnation: UUID) -> bool:
        """
        Whether a collection is still registered under an incarnation.

        Args:
            incarnation (UUID): The incarnation a handle is bound to.

        Returns:
            bool: Whether the incarnation's collection is live.
        """
        raise NotImplementedError

    @abstractmethod
    async def unregister(self, namespace: str, name: str) -> None:
        """
        Unregister a collection and queue its incarnation for purge.

        The collection is unreachable when this returns, and its points
        are reclaimed by the purge rounds that claim its tombstone. It is
        idempotent: no collection under the key is the no-op case.

        Args:
            namespace (str): Namespace of the collection.
            name (str): Name of the collection within the namespace.
        """
        raise NotImplementedError

    @abstractmethod
    def claim_purgeable_incarnation(
        self,
    ) -> AbstractAsyncContextManager[PurgeClaim | None]:
        """
        Claim a dead incarnation whose tombstone is due, for one purge round held for the body of the context.

        A tombstone is due once the retention has passed since its
        deletion. The caller runs one round in the body: it looks for
        points under `claim.incarnation` in the native collection that
        `claim.namespace` and `claim.config` name, deletes any it finds,
        and sets `claim.found`. When the body ends, the registry records
        the outcome: a round that found points leaves the tombstone due; a
        round that found none removes the tombstone and frees its
        incarnation. A body that raises leaves the tombstone as it was.

        A registry that can hold a claim hands the tombstone to no other
        purger for the body's duration; one that cannot lets a doubly
        claimed tombstone cost a repeated, idempotent round and never a
        missed one.

        Returns:
            AbstractAsyncContextManager[PurgeClaim | None]:
                The claim, or None when no tombstone is due.
        """
        raise NotImplementedError
