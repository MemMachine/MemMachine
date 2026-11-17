"""
The interface of a store whose records are divided into partitions by UUID.

Imports nothing from this package.
"""

from abc import ABC, abstractmethod
from uuid import UUID


class PartitionedStore(ABC):
    """
    A store whose records are divided into partitions, each identified by a UUID.

    For each partition, `create_partition` is called once, before the
    partition is used. Once the partition is deleted, or its creation
    abandoned, and no write to it can still be in flight, `purge_partition` is
    called repeatedly, and `drop_partition` after a `purge_partition` call that
    returned False; either call may come again after either, also on several
    processes at once. Partitions are isolated from one another: a record
    written to one partition is visible in that partition alone.
    """

    @abstractmethod
    async def create_partition(self, partition_uuid: UUID) -> None:
        """
        Create the storage a partition keeps for itself.

        Called once for each partition, before the partition is used, with a
        UUID no other partition carries. A store whose partitions share all
        their storage has nothing to create. Safe to repeat, and to run on
        several processes at once.

        A call that raises fails the partition's creation, and purging the
        partition later reclaims whatever the call created.

        Args:
            partition_uuid (UUID):
                The partition's UUID.
        """
        raise NotImplementedError

    @abstractmethod
    async def purge_partition(self, partition_uuid: UUID) -> bool:
        """
        Delete some or all of a deleted partition's records from storage shared with other partitions.

        Called repeatedly once the partition is deleted, or its creation
        abandoned, and no write to it can still be in flight, also after it
        returned False or after `drop_partition`. A store that keeps a
        partition's records only in the partition's own storage returns
        False, since `drop_partition` removes them with that storage. Safe to
        repeat, and to run on several processes at once.

        A call that raises is retried later.

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

        Called after a `purge_partition` call that returned False for the
        partition. Idempotent, and safe to run on several processes at once.

        A call that raises is retried later, after another `purge_partition`
        call.

        Args:
            partition_uuid (UUID):
                The deleted partition's UUID.
        """
        raise NotImplementedError
