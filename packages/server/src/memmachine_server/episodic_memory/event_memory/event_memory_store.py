"""Event memory's storage: its segment store and vector store, as one partitioned store."""

import asyncio
import logging
from collections.abc import Awaitable, Sequence
from dataclasses import dataclass
from typing import override
from uuid import UUID

from memmachine_server.common.partition_registry import PartitionedStore
from memmachine_server.common.vector_store import VectorStore, VectorStorePartition

from .segment_store import SegmentStore, SegmentStorePartition

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class EventMemoryStorePartition:
    """A partition's handles in each of event memory's stores."""

    segment_store_partition: SegmentStorePartition
    vector_store_partition: VectorStorePartition


class EventMemoryStore(PartitionedStore[EventMemoryStorePartition]):
    """
    Event memory's stores, as one partitioned store.

    Creating, purging, and dropping a partition run on every store at once:
    every store finishes, each error is logged, and the first failing
    store's error is raised, so the stores that answer keep their progress
    while another is down. A purge finds records when any store's purge
    does. Getting a partition answers None when the segment store does, and
    gets the vector store's handle only otherwise, since a vector store
    answers a handle for any UUID.
    """

    def __init__(self, segment_store: SegmentStore, vector_store: VectorStore) -> None:
        """Initialize with the segment store and the vector store."""
        self._segment_store = segment_store
        self._vector_store = vector_store

    @property
    def vector_store(self) -> VectorStore:
        """The vector store holding the derivatives' vectors."""
        return self._vector_store

    @override
    async def create_partition(self, partition_uuid: UUID) -> None:
        await _on_every_store(
            "create",
            partition_uuid,
            [
                self._segment_store.create_partition(partition_uuid),
                self._vector_store.create_partition(partition_uuid),
            ],
        )

    @override
    async def get_partition(
        self, partition_uuid: UUID
    ) -> EventMemoryStorePartition | None:
        segment_store_partition = await self._segment_store.get_partition(
            partition_uuid
        )
        if segment_store_partition is None:
            return None
        return EventMemoryStorePartition(
            segment_store_partition=segment_store_partition,
            vector_store_partition=await self._vector_store.get_partition(
                partition_uuid
            ),
        )

    @override
    async def purge_partition(self, partition_uuid: UUID) -> bool:
        found = await _on_every_store(
            "purge",
            partition_uuid,
            [
                self._segment_store.purge_partition(partition_uuid),
                self._vector_store.purge_partition(partition_uuid),
            ],
        )
        return any(found)

    @override
    async def drop_partition(self, partition_uuid: UUID) -> None:
        await _on_every_store(
            "drop",
            partition_uuid,
            [
                self._segment_store.drop_partition(partition_uuid),
                self._vector_store.drop_partition(partition_uuid),
            ],
        )


async def _on_every_store[T](
    step: str, partition_uuid: UUID, calls: Sequence[Awaitable[T]]
) -> list[T]:
    """Run one step of a partition's lifecycle on every store at once, and raise the first error once every store has finished."""
    results = await asyncio.gather(*calls, return_exceptions=True)
    errors = [result for result in results if isinstance(result, BaseException)]
    for error in errors:
        logger.error(
            "A store failed to %s partition %s",
            step,
            partition_uuid,
            exc_info=error,
        )
    if errors:
        raise errors[0]
    return [result for result in results if not isinstance(result, BaseException)]
