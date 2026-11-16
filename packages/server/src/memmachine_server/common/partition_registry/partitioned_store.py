"""
The interface of a store whose records are divided into partitions by the UUIDs a partition registry mints, and the purge round of an owner with one such store.

A store is one kind of resource an owner partitions by its registry's UUIDs.
"""

from abc import ABC, abstractmethod
from uuid import UUID

from .partition_registry import PurgeRound


class PartitionedStore[PartitionT](ABC):
    """
    A store whose records are divided into partitions, each identified by the UUID its owner's partition registry mints.

    The store holds no partition names and decides nothing about which
    partition is live. Its owner calls these methods, for each partition:
    `create_partition` once, as it provisions the partition before the
    registry confirms it, and `get_partition` whenever the owner answers the
    partition; then, once the partition is deleted or its creation cancelled
    and the retention has passed, in the purge rounds the registry runs,
    `purge_partition` until it returns False, then `drop_partition`.
    Partitions are isolated from one another, including partitions created
    under the same name: records written to one are never read from another.
    """

    @abstractmethod
    async def create_partition(self, partition_uuid: UUID) -> None:
        """
        Create the storage a partition keeps for itself.

        Called once for each partition, before the registry confirms it, with
        a UUID no other partition carries. A store whose partitions share all
        their storage has nothing to create. Safe to repeat, and to run on
        several processes at once.

        A call that raises fails the partition's creation: the registry
        cancels it, and the partition's purge rounds reclaim whatever the call
        created.

        Args:
            partition_uuid (UUID):
                The partition's UUID, minted by the registry.
        """
        raise NotImplementedError

    @abstractmethod
    async def get_partition(self, partition_uuid: UUID) -> PartitionT | None:
        """
        Get a handle on a created partition.

        The handle reads and writes the partition's records alone, whether
        or not the partition is still live.

        Args:
            partition_uuid (UUID):
                The partition's UUID.

        Returns:
            PartitionT | None:
                A handle bound to the partition, or None when the store finds
                nothing under the UUID: the partition was never created, or was
                dropped. A store that cannot tell answers a handle.
        """
        raise NotImplementedError

    @abstractmethod
    async def purge_partition(self, partition_uuid: UUID) -> bool:
        """
        Delete some or all of a deleted partition's records that its store keeps in storage shared with other partitions.

        Called once the partition is deleted or its creation cancelled and the
        retention has passed, until it returns False. A store that keeps a
        partition's records only in storage of the partition's own returns
        False, since `drop_partition` removes them with that storage. Safe to
        repeat, and to run on several processes at once.

        A call that raises leaves the partition's tombstone queued, so a
        later round purges again.

        Args:
            partition_uuid (UUID):
                The deleted partition's UUID.

        Returns:
            bool:
                False when none of the partition's records remains in the
                shared storage; True when a later call may find some.
        """
        raise NotImplementedError

    @abstractmethod
    async def drop_partition(self, partition_uuid: UUID) -> None:
        """
        Remove the storage a deleted partition keeps for itself, with any records still in it.

        Called once `purge_partition` has returned False for the partition.
        Idempotent, and safe to run on several processes at once.

        A call that raises leaves the partition's tombstone queued, so a
        later round purges and drops again.

        Args:
            partition_uuid (UUID):
                The deleted partition's UUID.
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
