"""
Abstract base class for a partition registry.

The catalog of a vector store whose backend holds only points: which
partitions exist, under which incarnation and schema, and which dead
incarnations await purge. A registry belongs to one vector store and keys
its partitions by their key. Its calls are arbitrated across every process
sharing it: registration mints an incarnation no live or queued
partition carries, unregistration makes the partition unreachable when it
returns, and a purge claim is handed to one purger at a time.
"""

from abc import ABC, abstractmethod
from contextlib import AbstractAsyncContextManager
from dataclasses import dataclass
from uuid import UUID

from memmachine_server.common.vector_store.data_types import PartitionSchema


@dataclass(frozen=True)
class RegisteredPartition:
    """A live partition: the incarnation its points carry and the schema it was created under."""

    incarnation: UUID
    schema: PartitionSchema


@dataclass
class PurgeClaim:
    """
    One purge round's hold on a tombstone: what the round needs, and what it found.

    The registry fills in what the round needs to find the dead
    incarnation's points: `incarnation`, the value they carry in the
    store's collection. The round sets `found` before the claim ends: True
    when points remained under the incarnation, False when none did. The
    registry records that outcome when the claim ends.
    """

    incarnation: UUID
    found: bool | None = None


class VectorStorePartitionRegistry(ABC):
    """
    The partition registry of one vector store.

    A queue entry is a dead incarnation's tombstone. The backend holds the
    points, and a write the registry read as live can land there after the
    deletion, so purging starts only once a retention has passed since the
    deletion, longer than any write can be in flight: nothing more lands
    under the incarnation after that. The entry stays through purge rounds
    until one finds nothing, and is then removed; until then the
    incarnation is never re-minted.
    """

    @abstractmethod
    async def provision(self) -> None:
        """Create the registry's durable resources, idempotently."""
        raise NotImplementedError

    @abstractmethod
    async def register(self, partition_key: str, schema: PartitionSchema) -> UUID:
        """
        Register a new partition under a freshly minted incarnation.

        The partition key is arbitrated across processes, and the
        incarnation is one no live or queued partition carries, so no
        points can be adopted by, or reclaimed out from under, the new
        partition.

        Args:
            partition_key (str): The key of the partition.
            schema (PartitionSchema):
                What the partition is created under: its store's
                dimensions, metric and declared schema.

        Returns:
            UUID: The incarnation the partition's points carry.

        Raises:
            VectorStorePartitionAlreadyExistsError: The partition key is taken.
            VectorStoreAttemptsExhaustedError:
                Every minted incarnation was rejected for another reason.
        """
        raise NotImplementedError

    @abstractmethod
    async def get(self, partition_key: str) -> RegisteredPartition | None:
        """
        Look up the live partition under a key.

        Args:
            partition_key (str): The key of the partition.

        Returns:
            RegisteredPartition | None:
                The live partition, or None when there is none.
        """
        raise NotImplementedError

    @abstractmethod
    async def is_live(self, incarnation: UUID) -> bool:
        """
        Whether a partition is still registered under an incarnation.

        Args:
            incarnation (UUID): The incarnation a handle is bound to.

        Returns:
            bool: Whether the incarnation's partition is live.
        """
        raise NotImplementedError

    @abstractmethod
    async def unregister(self, partition_key: str) -> None:
        """
        Unregister a partition and queue its incarnation for purge.

        The partition is unreachable when this returns, and its points
        are reclaimed by the purge rounds that claim its tombstone. It is
        idempotent: no partition under the key is the no-op case.

        Args:
            partition_key (str): The key of the partition.
        """
        raise NotImplementedError

    @abstractmethod
    def claim_due(self) -> AbstractAsyncContextManager[PurgeClaim | None]:
        """
        Claim a tombstone that is due for a purge round, held for the body of the context.

        A tombstone is due once the retention has passed since its
        deletion. The caller runs one round in the body: it looks for
        points under `claim.incarnation` in the store's collection,
        deletes any it finds, and sets `claim.found`. When the body ends,
        the registry records the outcome: a round that found points leaves
        the tombstone due; a round that found none removes the tombstone
        and frees its incarnation. A body that raises leaves the tombstone
        as it was.

        A registry that can hold a claim hands the tombstone to no other
        purger for the body's duration; one that cannot lets a doubly
        claimed tombstone cost a repeated, idempotent round and never a
        missed one.

        Returns:
            AbstractAsyncContextManager[PurgeClaim | None]:
                The claim, or None when no tombstone is due.
        """
        raise NotImplementedError
