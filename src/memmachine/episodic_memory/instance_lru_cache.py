"""LRU cache implementation for managing episodic memory instances."""

import asyncio
from collections.abc import Coroutine
from datetime import UTC, datetime
from typing import cast

from memmachine.episodic_memory.episodic_memory import EpisodicMemory


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

        # Initialize sentinel head and tail nodes for the doubly linked list.
        # head.next points to the most recently used item.
        # tail.prev points to the least recently used item.
        self.head = Node(None, None)
        self.tail = Node(None, None)
        self.head.next = self.tail
        self.tail.prev = self.head

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

    def clear(self) -> None:
        """Remove all items from the cache."""
        self.cache.clear()
        self.head.next = self.tail
        self.tail.prev = self.head

    def keys(self) -> list[str]:
        """Return a list of all keys in the cache."""
        return list(self.cache.keys())

    def erase(self, key: str) -> None:
        """Remove an item from the cache."""
        if key in self.cache:
            node = self.cache[key]
            if node.ref_count > 0:
                raise RuntimeError(f"Key {key} is still in use {node.ref_count}")
            self._remove_node(node)
            del self.cache[key]

    def get(self, key: str) -> EpisodicMemory | None:
        """
        Retrieve an item from the cache.

        Returns the value if the key exists, otherwise -1 (or None/raise KeyError).
        Moves the accessed item to the front (most recently used).
        """
        if key in self.cache:
            node = self.cache[key]
            node.ref_count += 1
            # Move accessed node to the front
            self._remove_node(node)
            self._add_to_front(node)
            node.last_access = datetime.now(tz=UTC)
            return node.value
        return None

    def get_ref_count(self, key: str) -> int:
        """
        Retrieve the reference count of an item in the cache.

        Returns the reference count if the key exists, otherwise -1.
        """
        if key in self.cache:
            return self.cache[key].ref_count
        return -1

    async def clear_cache(self) -> None:
        """Clear all items from the cache."""
        close_memory_coroutines: list[Coroutine] = []
        close_memory_coroutines.extend(
            cast(EpisodicMemory, node.value).close() for node in self.cache.values()
        )
        await asyncio.gather(*close_memory_coroutines)
        self.cache.clear()
        self.head.next = self.tail
        self.tail.prev = self.head

    async def add(self, key: str, value: EpisodicMemory) -> None:
        """Add a new item to the cache."""
        if key in self.cache:
            raise ValueError(f"Key {key} already exists")

        # Add new key
        lru_node = self.tail.prev
        while len(self.cache) >= self.capacity and lru_node != self.head:
            if lru_node.ref_count > 0:
                lru_node = cast("Node", lru_node.prev)
                continue
            tmp = lru_node.prev
            self._remove_node(lru_node)
            if lru_node.value is not None:
                await lru_node.value.close()
            del self.cache[cast("str", lru_node.key)]
            lru_node = tmp

        new_node = Node(key, value)
        self.cache[key] = new_node
        self._add_to_front(new_node)

    def put(self, key: str) -> None:
        """Release the object reference."""
        if key in self.cache:
            # Update existing key's value and move it to the front
            node = self.cache[key]
            assert node.ref_count > 0
            node.ref_count -= 1
        else:
            raise ValueError(f"Key {key} does not exist")

    async def clean_old_instance(self) -> None:
        """Remove unused instance with long lifetime."""
        now = datetime.now(tz=UTC)
        lru_node = self.tail.prev
        while lru_node != self.head:
            if lru_node.ref_count > 0:
                lru_node = cast("Node", lru_node.prev)
                continue
            tmp = lru_node.prev
            if (now - lru_node.last_access).total_seconds() > self.max_lifetime:
                self._remove_node(lru_node)
                if lru_node.value is not None:
                    await lru_node.value.close()
                del self.cache[cast("str", lru_node.key)]
                lru_node = self.tail.prev
            lru_node = tmp
