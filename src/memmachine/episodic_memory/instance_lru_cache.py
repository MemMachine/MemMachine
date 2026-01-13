"""LRU cache implementation for managing episodic memory instances."""

import asyncio
import logging
from collections.abc import Coroutine
from datetime import UTC, datetime
from typing import cast

from memmachine.common import rw_locks
from memmachine.episodic_memory.episodic_memory import EpisodicMemory

logger = logging.getLogger(__name__)


class Node:
    """
    Represent a node for the doubly linked list.

    Each node stores a key-value pair.
    """

    def __init__(self, key: str | None, value: EpisodicMemory | None) -> None:
        """Initialize a node with an optional key/value pair."""
        self.key = key
        self.value = value
        self.ref_count = 1
        self.last_access = datetime.now(tz=UTC)
        self.prev: Node = self
        self.next: Node = self


class MemoryInstanceCache:
    """
    Implement an LRU cache that manages memory instances.

    Attributes:
        capacity (int): The maximum number of items the cache can hold.
        cache (dict): A dictionary mapping keys to Node objects for O(1) lookups.
        head (Node): A sentinel head node for the doubly linked list.
        tail (Node): A sentinel tail node for the doubly linked list.

    """

    def __init__(self, capacity: int, max_lifetime: int) -> None:
        """Initialize the cache with capacity and lifetime limits."""
        if capacity <= 0:
            raise ValueError("Capacity must be a positive integer")
        self.capacity = capacity
        self.max_lifetime = max_lifetime
        self.cache: dict[str, Node] = {}  # Stores key -> Node
        self._lock = rw_locks.AsyncRWLock()

        # Initialize sentinel head and tail nodes for the doubly linked list.
        # head.next points to the most recently used item.
        # tail.prev points to the least recently used item.
        self.head = Node(None, None)
        self.tail = Node(None, None)
        self.head.next = self.tail
        self.tail.prev = self.head

        # The Disposal Queue
        self._disposal_queue = asyncio.Queue()
        self._janitor_task = asyncio.create_task(self._janitor_routine())

    async def _janitor_routine(self) -> None:
        """Background task that closes discarded memory instances."""
        while True:
            node = await self._disposal_queue.get()
            try:
                if node.value is not None:
                    await node.value.close()
            except Exception:
                logger.exception("error closing memory instance")
            finally:
                self._disposal_queue.task_done()

    @staticmethod
    def _remove_node(node: Node) -> None:
        """Remove a node from the doubly linked list."""
        if node.prev and node.next:
            prev_node = node.prev
            next_node = node.next
            prev_node.next = next_node
            next_node.prev = prev_node

    def _add_to_front(self, node: Node) -> None:
        """Add a node to the front of the doubly linked list (right after head)."""
        node.prev = self.head
        node.next = self.head.next
        if self.head.next:
            self.head.next.prev = node
        self.head.next = node

    async def clear(self) -> None:
        """Remove all items from the cache."""
        async with self._lock.write_lock():
            self.cache.clear()
            self.head.next = self.tail
            self.tail.prev = self.head

    async def keys(self) -> list[str]:
        """Return a list of all keys in the cache."""
        async with self._lock.read_lock():
            return list(self.cache.keys())

    async def erase(self, key: str | None) -> None:
        """
        Remove an item from the cache.

        If the key is ``None``, this method performs no operation.
        """
        async with self._lock.write_lock():
            if key in self.cache:
                node = self.cache[key]
                if node.ref_count > 0:
                    raise RuntimeError(f"Key {key} is still in use {node.ref_count}")
                self._remove_node(node)
                del self.cache[key]

    async def get(self, key: str | None) -> EpisodicMemory | None:
        """
        Retrieve an item from the cache.

        Returns the value if the key exists, otherwise -1 (or None/raise KeyError).
        Moves the accessed item to the front (most recently used).
        """
        ret: EpisodicMemory | None = None
        async with self._lock.write_lock():
            if key in self.cache:
                node = self.cache[key]
                ret = node.value
                await self._pop_node(node)
        return ret

    async def _pop_node(self, node: Node) -> None:
        """Pop a specific node from the cache."""
        self._remove_node(node)
        self._add_to_front(node)
        node.ref_count += 1
        node.last_access = datetime.now(tz=UTC)

    async def get_ref_count(self, key: str | None) -> int:
        """
        Retrieve the reference count of an item in the cache.

        Returns the reference count if the key exists, otherwise -1.
        """
        async with self._lock.read_lock():
            if key in self.cache:
                return self.cache[key].ref_count
        return -1

    async def _clear_cache(self) -> None:
        """Clear all items from the cache."""
        async with self._lock.write_lock():
            close_memory_coroutines: list[Coroutine] = []
            close_memory_coroutines.extend(
                cast(EpisodicMemory, node.value).close() for node in self.cache.values()
            )
            await asyncio.gather(*close_memory_coroutines)
            self.cache.clear()
            self.head.next = self.tail
            self.tail.prev = self.head

    async def add(self, key: str | None, value: EpisodicMemory) -> None:
        """Add a new item to the cache."""
        if key is None:
            return
        async with self._lock.write_lock():
            if key in self.cache:
                raise ValueError(f"Key {key} already exists")

            # Add new key
            lru_node = self.tail.prev
            while len(self.cache) >= self.capacity and lru_node != self.head:
                if lru_node.ref_count > 0:
                    lru_node = cast("Node", lru_node.prev)
                    continue
                tmp = lru_node.prev
                await self._remove_and_close_node(lru_node)
                lru_node = tmp

            new_node = Node(key, value)
            self.cache[key] = new_node
            self._add_to_front(new_node)

    async def _remove_and_close_node(self, node: Node) -> None:
        """Remove a node from the cache and close its value."""
        self._remove_node(node)
        del self.cache[cast("str", node.key)]
        self._disposal_queue.put_nowait(node)

    async def release_ref(self, key: str | None) -> None:
        """Release the object reference."""
        if key is None:
            return
        async with self._lock.write_lock():
            if key in self.cache:
                # Update existing key's value and move it to the front
                node = self.cache[key]
                assert node.ref_count > 0
                node.ref_count -= 1
            else:
                raise ValueError(f"Key {key} does not exist")

    async def clean_old_instance(self) -> None:
        """Remove unused instance with long lifetime."""
        async with self._lock.write_lock():
            now = datetime.now(tz=UTC)
            lru_node = self.tail.prev
            while lru_node != self.head:
                if lru_node.ref_count > 0:
                    lru_node = cast("Node", lru_node.prev)
                    continue
                tmp = lru_node.prev
                if (now - lru_node.last_access).total_seconds() > self.max_lifetime:
                    await self._remove_and_close_node(lru_node)
                    lru_node = self.tail.prev
                lru_node = tmp

    async def close(self) -> None:
        """Close the cache and all its resources."""
        await self._clear_cache()
        if self._disposal_queue.qsize() > 0:
            logger.debug(
                "Waiting for janitor to close %d remaining items...",
                self._disposal_queue.qsize(),
            )
            await self._disposal_queue.join()
        self._janitor_task.cancel()
        try:
            await self._janitor_task
        except asyncio.CancelledError as e:
            logger.warning("failed to cancel the janitor task: %s", e)
        finally:
            logger.debug("closing the janitor task")
