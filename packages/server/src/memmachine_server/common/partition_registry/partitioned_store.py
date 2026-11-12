"""
The interface of a store whose records a partition registry divides into partitions.

Imports nothing from this package; the registry builds on it.
"""

from abc import ABC, abstractmethod
from uuid import UUID


class PartitionedStore(ABC):
    """
    A store whose records are divided into partitions, each identified by the UUID its owner's partition registry mints.

    The store holds no partition names and decides nothing about which
    partition is live. The registry calls these methods, for each partition:
    `create_partition` once, before it confirms the partition; then, once the
    partition is deleted or its creation cancelled and the retention has
    passed, `purge_partition` until every store's call returns False, then
    `drop_partition`. Partitions are isolated from one another, including
    partitions created under the same name: records written to one are never
    read from another.
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
    async def purge_partition(self, partition_uuid: UUID) -> bool:
        """
        Delete a batch of a deleted partition's records that its store keeps in storage shared with other partitions.

        Called once the partition is deleted or its creation cancelled and the
        retention has passed, until it returns False. The store sizes each
        call for its backend, so a call does not noticeably degrade concurrent
        request serving. A store that keeps a partition's records only in
        storage of the partition's own returns False, since `drop_partition`
        removes them with that storage. Safe to repeat, and to run on several
        processes at once.

        A call that raises leaves the partition's tombstone queued, so a
        later round purges again.

        Args:
            partition_uuid (UUID):
                The deleted partition's UUID.

        Returns:
            bool:
                Whether the call found any of the partition's records. A call
                that returns False proves that none remains in the shared
                storage, by a search from the partition's first record.
        """
        raise NotImplementedError

    @abstractmethod
    async def drop_partition(self, partition_uuid: UUID) -> None:
        """
        Remove the storage a deleted partition keeps for itself, with any records still in it.

        Called once every store's `purge_partition` has returned False for the
        partition. Idempotent, and safe to run on several processes at once.

        A call that raises leaves the partition's tombstone queued, so a
        later round purges and drops again.

        Args:
            partition_uuid (UUID):
                The deleted partition's UUID.
        """
        raise NotImplementedError
