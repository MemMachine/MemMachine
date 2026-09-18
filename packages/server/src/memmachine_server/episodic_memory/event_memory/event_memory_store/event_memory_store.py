"""
Abstract base class for a event memory store.

Defines an interface for adding, retrieving, and deleting the segments of
events.
"""

from abc import ABC, abstractmethod
from collections.abc import Iterable, Mapping
from contextlib import AbstractAsyncContextManager
from datetime import datetime
from uuid import UUID

from memmachine_server.common.filter.filter_parser import FilterExpr
from memmachine_server.episodic_memory.event_memory.data_types import (
    Neighborhood,
    Segment,
)
from memmachine_server.episodic_memory.event_memory.event_memory_store.data_types import (
    EventMemoryStorePartitionConfig,
)


class EventMemoryStorePartitionWriter(ABC):
    """One write transaction on a partition, open inside `write()`.

    Everything done through the writer commits when the `write()` block
    exits normally and is rolled back when it exits by exception, so a
    caller can make a write conditional on work of its own, such as a
    write to another store, by doing that work inside the block. The
    partition stays live for the whole block: its deletion waits.
    """

    @abstractmethod
    async def add_events(
        self,
        events: Mapping[UUID, Mapping[Segment, Iterable[UUID]]],
    ) -> None:
        """
        Add events, each with its segments and their derivative UUIDs.

        The partition holds an event at most once. A batch naming an
        event the partition already holds is rejected whole, before
        anything is stored, so a repeated encoding never stores a second
        copy; the event must be deleted first.

        Args:
            events (Mapping[UUID, Mapping[Segment, Iterable[UUID]]]):
                A mapping from each event's UUID to its segments, and
                from each segment to the UUIDs of its derivatives.

        Raises:
            EventMemoryStoreEventAlreadyStoredError:
                If the partition already holds any of the events; it
                names them all.
            ValueError:
                If a segment names an event other than the one it is
                listed under.
        """
        raise NotImplementedError

    @abstractmethod
    async def delete_derivatives(
        self,
        derivative_uuids: Iterable[UUID],
    ) -> None:
        """
        Delete derivative links by derivative UUID, leaving their segments.

        The segments and their events stay held; a derivative UUID the
        partition does not hold is ignored. A caller that unlinks a
        derivative because another one displaces it adds the displacing
        links in this same transaction, and deletes the displaced
        records from the vector store only once the block has committed:
        a record no link names is reclaimed when a search returns it,
        while a link naming a record that is gone is not.

        Args:
            derivative_uuids (Iterable[UUID]):
                The UUIDs of the derivatives to unlink.
        """
        raise NotImplementedError

    @abstractmethod
    async def get_segment_uuids_by_derivative_uuids(
        self,
        derivative_uuids: Iterable[UUID],
    ) -> dict[UUID, UUID]:
        """
        Get the segment each of the given derivatives belongs to.

        The same read as the partition's, inside this transaction: with
        `write(exclusive=True)`, it sees every write that was in flight
        when the block was entered, committed or rolled back.

        Args:
            derivative_uuids (Iterable[UUID]):
                The UUIDs of the derivatives whose owning segments to look up.

        Returns:
            dict[UUID, UUID]:
                A mapping from each derivative UUID to its segment's UUID.
        """
        raise NotImplementedError


class EventMemoryStorePartition(ABC):
    """Partition-scoped handle for a event memory store.

    A handle is bound to the partition incarnation it was opened on:
    deleting the partition permanently invalidates the handle, and its
    data operations raise `EventMemoryStorePartitionHandleStaleError` from
    then on, even if a partition is later created under the same key.
    A call with empty input may do no work and return without checking
    the handle.

    Events and their segments are immutable, and the partition holds an
    event at most once: `add_events` rejects an event that is already
    stored rather than replacing it, and no operation that edits a
    stored segment may be added to this contract; a changed event is
    deleted and added again.

    Segments within a partition are in one total order,
    `(timestamp, event_uuid, index, offset)`.
    """

    @property
    @abstractmethod
    def config(self) -> EventMemoryStorePartitionConfig:
        """The configuration for this partition."""
        raise NotImplementedError

    @abstractmethod
    def write(
        self, *, exclusive: bool = False
    ) -> AbstractAsyncContextManager[EventMemoryStorePartitionWriter]:
        """
        Open a write transaction on the partition.

        Entering the block checks the handle and pins the partition
        against deletion until the block exits. With `exclusive`, entry
        also waits for every write in flight on the partition to commit
        or roll back, and no write can start until the block exits; a
        reader that must see the partition settled uses it, so keep such
        a block short.

        Args:
            exclusive (bool):
                Whether to wait for in-flight writes and exclude new ones
                (default: False).

        Returns:
            AbstractAsyncContextManager[EventMemoryStorePartitionWriter]:
                The transaction, as a context manager; its writer is
                usable only inside the block.

        Raises:
            EventMemoryStorePartitionHandleStaleError: On entry, if the handle is stale.
        """
        raise NotImplementedError

    @abstractmethod
    async def get_segments(
        self,
        segment_uuids: Iterable[UUID],
        *,
        since: datetime | None = None,
        until: datetime | None = None,
        session_ids: Iterable[str] | None = None,
        source_ids: Iterable[str] | None = None,
        block_kinds: Iterable[str] | None = None,
        property_filter: FilterExpr | None = None,
    ) -> dict[UUID, Segment]:
        """
        Get segments by UUID.

        A UUID the partition does not hold, or whose segment fails a
        filter, is excluded from the result.

        Args:
            segment_uuids (Iterable[UUID]):
                The UUIDs of the segments to get.
            since (datetime | None):
                Inclusive lower bound on the segments' timestamps, timezone-aware
                (default: None).
            until (datetime | None):
                Exclusive upper bound on the segments' timestamps, timezone-aware,
                so ranges meet without overlap (default: None).
            session_ids (Iterable[str] | None):
                Keep only segments whose session id is one of these; an
                empty list keeps none, and None keeps every session
                (default: None).
            source_ids (Iterable[str] | None):
                Keep only segments whose source id is one of these; an
                empty list keeps none, and None keeps every source
                (default: None).
            block_kinds (Iterable[str] | None):
                Keep only segments whose block is of one of these kinds;
                an empty list keeps none, and None keeps every kind
                (default: None).
            property_filter (FilterExpr | None):
                A filter expression over segment properties (default: None).

        Returns:
            dict[UUID, Segment]:
                A mapping from each UUID found and admitted to its segment.

        Raises:
            ValueError: If `since` or `until` is naive.
        """
        raise NotImplementedError

    @abstractmethod
    async def get_segment_neighborhoods(
        self,
        seed_uuids: Iterable[UUID],
        *,
        before: int = 0,
        after: int = 0,
        since: datetime | None = None,
        until: datetime | None = None,
        source_ids: Iterable[str] | None = None,
        block_kinds: Iterable[str] | None = None,
        property_filter: FilterExpr | None = None,
    ) -> dict[UUID, Neighborhood]:
        """
        Get the segments around each seed segment, excluding the seed itself.

        A walk outward from the seed in the partition's total order,
        within the seed's session. The seed is an address: it is located
        whether or not it passes any filter, and the filters select the
        neighbors.

        Args:
            seed_uuids (Iterable[UUID]):
                The UUIDs of the segments to gather neighbors around.
            before (int):
                The maximum number of neighbors before each seed, nonnegative
                (default: 0).
            after (int):
                The maximum number of neighbors after each seed, nonnegative
                (default: 0).
            since (datetime | None):
                Inclusive lower bound on the neighbors' timestamps, timezone-aware
                (default: None).
            until (datetime | None):
                Exclusive upper bound on the neighbors' timestamps, timezone-aware,
                so ranges meet without overlap (default: None).
            source_ids (Iterable[str] | None):
                Keep only neighbors whose source id is one of these; an
                empty list keeps none, and None keeps every source
                (default: None).
            block_kinds (Iterable[str] | None):
                Keep only neighbors whose block is of one of these kinds;
                an empty list keeps none, and None keeps every kind
                (default: None).
            property_filter (FilterExpr | None):
                A filter expression over the neighbors' properties
                (default: None).

        Returns:
            dict[UUID, Neighborhood]:
                A mapping from each known seed to its neighbors: `before`
                in order ending just before the seed, `after` in order
                starting just after it. A seed with no neighbors to show
                maps to two empty lists; an unknown seed is excluded from
                the result.

        Raises:
            ValueError:
                If `before` or `after` is negative, or `since` or `until`
                is naive.
        """
        raise NotImplementedError

    @abstractmethod
    async def get_derivative_uuids_by_event_uuids(
        self,
        event_uuids: Iterable[UUID],
    ) -> dict[UUID, list[UUID]]:
        """
        Get the derivative UUIDs of the events given by their UUIDs.

        Args:
            event_uuids (Iterable[UUID]):
                The UUIDs of the events whose derivatives to look up.

        Returns:
            dict[UUID, list[UUID]]:
                A mapping from each event UUID the partition holds to the
                UUIDs of the derivatives of its segments; an event the
                partition does not hold is omitted.
        """
        raise NotImplementedError

    @abstractmethod
    async def get_segment_uuids_by_derivative_uuids(
        self,
        derivative_uuids: Iterable[UUID],
    ) -> dict[UUID, UUID]:
        """
        Get the segment each of the given derivatives belongs to.

        A derivative belongs to exactly one segment, so this answers one
        UUID rather than a list. UUIDs the partition does not hold are
        omitted.

        Args:
            derivative_uuids (Iterable[UUID]):
                The UUIDs of the derivatives whose owning segments to look up.

        Returns:
            dict[UUID, UUID]:
                A mapping from each derivative UUID to its segment's UUID.
        """
        raise NotImplementedError

    @abstractmethod
    async def delete_events(
        self,
        event_uuids: Iterable[UUID],
    ) -> None:
        """
        Delete events, with their segments and derivatives. Idempotent.

        Args:
            event_uuids (Iterable[UUID]):
                The UUIDs of the events to delete.
        """
        raise NotImplementedError

    @abstractmethod
    async def delete_segments(
        self,
        segment_uuids: Iterable[UUID],
    ) -> None:
        """
        Delete segments and their derivatives. Idempotent.

        The segments' events stay held: a deleted segment is not added
        back by adding its event again.

        Args:
            segment_uuids (Iterable[UUID]):
                The UUIDs of the segments to delete.
        """
        raise NotImplementedError


class EventMemoryStore(ABC):
    """
    Abstract base class for a event memory store.

    Manages partition-scoped handles.

    Partition keys must match `[a-z0-9_]+`
    (lowercase alphanumeric and underscores only)
    and be at most 32 bytes.
    """

    @abstractmethod
    async def startup(self) -> None:
        """Startup."""
        raise NotImplementedError

    @abstractmethod
    async def shutdown(self) -> None:
        """Shutdown."""
        raise NotImplementedError

    @abstractmethod
    async def create_partition(
        self,
        partition_key: str,
        config: EventMemoryStorePartitionConfig,
    ) -> None:
        """
        Create a new partition.

        Args:
            partition_key (str):
                The key of the partition.
            config (EventMemoryStorePartitionConfig):
                Configuration for the partition.

        Raises:
            EventMemoryStorePartitionAlreadyExistsError: If the partition already exists.
            EventMemoryStoreAttemptsExhaustedError:
                If creation exhausted its internal attempts on a
                failure that should not recur; an immediate retry is
                unlikely to succeed -- diagnose the chained cause.
        """
        raise NotImplementedError

    @abstractmethod
    async def open_partition(
        self, partition_key: str
    ) -> EventMemoryStorePartition | None:
        """
        Open a partition-scoped handle for an existing partition.

        Args:
            partition_key (str):
                The key of the partition.

        Returns:
            EventMemoryStorePartition | None:
                A partition-scoped handle, or None if the partition does not exist.
        """
        raise NotImplementedError

    @abstractmethod
    async def open_or_create_partition(
        self,
        partition_key: str,
        config: EventMemoryStorePartitionConfig,
    ) -> EventMemoryStorePartition:
        """
        Open the partition if it exists, or create it if it does not.

        Args:
            partition_key (str):
                The key of the partition.
            config (EventMemoryStorePartitionConfig):
                Configuration for the partition.

        Returns:
            EventMemoryStorePartition:
                A partition-scoped handle.

        Raises:
            EventMemoryStorePartitionConfigMismatchError:
                If the partition already exists with a different configuration.
            EventMemoryStoreAttemptsExhaustedError:
                If creation exhausted its internal attempts on a
                failure that should not recur; an immediate retry is
                unlikely to succeed -- diagnose the chained cause.
        """
        raise NotImplementedError

    @abstractmethod
    async def close_partition(
        self, event_memory_store_partition: EventMemoryStorePartition
    ) -> None:
        """
        Close a partition-scoped handle.

        Args:
            event_memory_store_partition (EventMemoryStorePartition):
                The partition-scoped handle to close.
        """
        raise NotImplementedError

    @abstractmethod
    async def delete_partition(self, partition_key: str) -> None:
        """
        Delete a partition.

        The partition becomes unreachable immediately: it can no longer be
        opened, and handles opened on it raise from then on.
        Implementations may defer physically reclaiming its storage to
        `purge_deleted_partitions`. Idempotent.

        Args:
            partition_key (str):
                The key of the partition to delete.
        """
        raise NotImplementedError

    @abstractmethod
    async def purge_deleted_partitions(self) -> bool:
        """
        Physically reclaim storage for deleted partitions, bounded per call.

        The sweeper: reclaims what `delete_partition` deferred, for every
        partition, oldest deletion first at the database clock's
        resolution. Each call does a bounded amount
        of work, sized so it does not noticeably degrade concurrent
        request serving, commits what it did or nothing, and is safe to
        repeat, including after a failure on backend contention with
        another writer, and to run concurrently from any process. The
        store never schedules
        it; a deployment must run it somewhere (the server's resource
        manager runs it in the background). Implementations that reclaim
        physically in
        `delete_partition` may return False without doing anything.

        Returns:
            bool:
                True if another call may reclaim more. False if this call
                found nothing to claim; entries a concurrent purger holds
                are that purger's to finish, so the caller may back off.
        """
        raise NotImplementedError
