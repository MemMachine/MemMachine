"""
Partitioned stores over the components that hold a MemMachine partition's data.

Each component holds a partition's data under one key, its session key.
MemMachine gives every component the string of the partition's UUID as that
key, so each store here addresses a partition's data in its component by
`str(partition_uuid)`.
"""

import contextlib
from typing import Final, override
from uuid import UUID

from memmachine_server.common.configuration.episodic_config import (
    DeclarativeLongTermMemoryConf,
)
from memmachine_server.common.errors import SessionNotFoundError
from memmachine_server.common.filter.filter_parser import (
    Comparison as FilterComparison,
)
from memmachine_server.common.partition_registry import PartitionedStore
from memmachine_server.common.resource_manager.resource_manager import (
    ResourceManagerImpl,
)
from memmachine_server.episodic_memory.long_term_memory.service_locator import (
    create_event_memory_storage,
    delete_event_memory_storage,
    delete_long_term_memory_storage,
)

EPISODE_DELETE_BATCH_SIZE: Final[int] = 1000


class SessionDataPartitionedStore(PartitionedStore):
    """
    The session data manager as a partitioned store.

    A partition's record, with its short-term memory summary, is held under
    the partition's key. The record is the partition's own storage: whoever
    creates the partition writes it, since the record holds the partition's
    settings, and the drop deletes it, after every store's purge has read
    what it needs from it.
    """

    def __init__(self, resource_manager: ResourceManagerImpl) -> None:
        """Initialize with the resource manager that provides the session data manager."""
        self._resource_manager = resource_manager

    @override
    async def create_partition(self, partition_uuid: UUID) -> None:
        return

    @override
    async def purge_partition(self, partition_uuid: UUID) -> bool:
        return False

    @override
    async def drop_partition(self, partition_uuid: UUID) -> None:
        session_data_manager = await self._resource_manager.get_session_data_manager()
        # A partition whose creation stopped before its record was written,
        # or whose record an earlier drop deleted, has none.
        with contextlib.suppress(SessionNotFoundError):
            await session_data_manager.delete_session(str(partition_uuid))


class EpisodePartitionedStore(PartitionedStore):
    """
    The episode store as a partitioned store.

    A partition's episodes are those stored under the partition's key. A
    purge deletes one batch of them, and first their history and citations
    in semantic memory, which records them by episode UUID, when the
    deployment configures semantic storage.
    """

    def __init__(self, resource_manager: ResourceManagerImpl) -> None:
        """Initialize with the resource manager that provides the episode store and semantic storage."""
        self._resource_manager = resource_manager

    @override
    async def create_partition(self, partition_uuid: UUID) -> None:
        return

    @override
    async def purge_partition(self, partition_uuid: UUID) -> bool:
        episode_storage = await self._resource_manager.get_episode_storage()
        episode_ids = await episode_storage.get_episode_ids(
            filter_expr=FilterComparison(
                field="session_key", op="=", value=str(partition_uuid)
            ),
            page_size=EPISODE_DELETE_BATCH_SIZE,
        )
        if not episode_ids:
            return False
        if self._resource_manager.config.semantic_memory.storage_configured():
            semantic_storage = await self._resource_manager.get_semantic_storage()
            # First, so a failure leaves the episodes in place for a retry.
            await semantic_storage.delete_history(episode_ids)
        await episode_storage.delete_episodes(episode_ids)
        return True

    @override
    async def drop_partition(self, partition_uuid: UUID) -> None:
        return


class LongTermMemoryPartitionedStore(PartitionedStore):
    """
    Long-term memory as a partitioned store.

    A partition's event memory partition is named by the partition's key,
    and so are its declarative memory collections. Event memory keeps its
    own registry: deleting its partition by name is one transaction there,
    and event memory's own purge removes the records. A partition's record
    says whether the partition uses declarative memory, and where.
    """

    def __init__(self, resource_manager: ResourceManagerImpl) -> None:
        """Initialize with the resource manager that provides the long-term memory backends."""
        self._resource_manager = resource_manager

    @override
    async def create_partition(self, partition_uuid: UUID) -> None:
        await create_event_memory_storage(str(partition_uuid), self._resource_manager)

    @override
    async def purge_partition(self, partition_uuid: UUID) -> bool:
        session_key = str(partition_uuid)
        await delete_event_memory_storage(session_key, self._resource_manager)
        session_data_manager = await self._resource_manager.get_session_data_manager()
        session_info = await session_data_manager.get_session_info(session_key)
        if session_info is not None and isinstance(
            session_info.episode_memory_conf.long_term_memory,
            DeclarativeLongTermMemoryConf,
        ):
            await delete_long_term_memory_storage(
                session_info.episode_memory_conf.long_term_memory,
                self._resource_manager,
            )
        return False

    @override
    async def drop_partition(self, partition_uuid: UUID) -> None:
        return
