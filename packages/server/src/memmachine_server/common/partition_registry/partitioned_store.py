"""
The interface of a store whose records are divided into partitions by UUID.

Imports nothing from this package.
"""

from abc import ABC, abstractmethod
from uuid import UUID


class PartitionedStore(ABC):
    """
    A store whose records are divided into partitions, each identified by a UUID.

    A partition manager calls these methods, for each partition:
    `create_partition` once, before the registry confirms the partition; then,
    once the partition is deleted or its creation canceled and the retention
    has passed, `purge_partition` in each purge round, and `drop_partition` in
    a round where no store's purge finds records. Rounds repeat until one drops
    the partition and ends, and two may run at once, so either call may come
    again after either. Partitions are isolated from one another, including
    partitions created under the same name: a record written to one
    partition is visible in that partition alone.
    """

    @abstractmethod
    async def create_partition(self, partition_uuid: UUID) -> None:
        """
        Create the storage a partition keeps for itself.

        Called once for each partition, before the registry confirms it, with
        a UUID no other partition carries. A store whose partitions share all
        their storage has nothing to create. Safe to repeat, and to run on
        several processes at once.

        A call that raises fails the partition's creation, and the
        partition's purge rounds reclaim whatever the call created.

        Args:
            partition_uuid (UUID):
                The partition's UUID.
        """
        raise NotImplementedError

    @abstractmethod
    async def purge_partition(self, partition_uuid: UUID) -> bool:
        """
        Delete some or all of a deleted partition's records from storage shared with other partitions.

        Called in each of the partition's purge rounds, once it is deleted or
        its creation canceled and the retention has passed, also after it
        returned False or after `drop_partition`. A store that keeps a
        partition's records only in the partition's own storage returns
        False, since `drop_partition` removes them with that storage. Safe to
        repeat, and to run on several processes at once.

        A call that raises fails the round, and a later round purges again.

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

        Called in a purge round once no store's `purge_partition` finds
        records of the partition. Idempotent, and safe to run on several
        processes at once.

        A call that raises fails the round, and a later round purges and
        drops again.

        Args:
            partition_uuid (UUID):
                The deleted partition's UUID.
        """
        raise NotImplementedError
