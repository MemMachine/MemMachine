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
    partition is used; a call that raises fails the partition's creation, and
    purging reclaims whatever the call created. Once the partition is deleted,
    or its creation abandoned, and no write to it can still be in flight,
    `purge_partition` is called repeatedly, and `drop_partition` after a
    `purge_partition` call that returned False. Purges and drops may repeat in
    any order, and one that raises is retried later. Each method is safe to
    repeat and to run on several processes at once. A store that keeps a
    partition's records only in the partition's own storage has nothing to
    purge, and its drop removes them. Partitions are isolated from one
    another: a record written to one partition is visible in that partition
    alone.
    """

    @abstractmethod
    async def create_partition(self, partition_uuid: UUID) -> None:
        """
        Create the storage a partition keeps for itself.

        The UUID is one no other partition carries. A store whose partitions
        share all their storage has nothing to create.

        Args:
            partition_uuid (UUID):
                The partition's UUID.
        """
        raise NotImplementedError

    @abstractmethod
    async def purge_partition(self, partition_uuid: UUID) -> bool:
        """
        Delete some or all of a deleted partition's records from storage shared with other partitions.

        Args:
            partition_uuid (UUID):
                The deleted partition's UUID.

        Returns:
            bool:
                False only when the call searched all the shared storage for
                the partition's records and found none; True otherwise.
        """
        raise NotImplementedError

    @abstractmethod
    async def drop_partition(self, partition_uuid: UUID) -> None:
        """
        Remove the storage a deleted partition keeps for itself, with any records still in it.

        Args:
            partition_uuid (UUID):
                The deleted partition's UUID.
        """
        raise NotImplementedError
