from typing import Any


class Node:
    """
    Node for the doubly linked list.
    Each node stores a key-value pair.
    """

    def __init__(self, key: Any, value: Any):
        self.key = key
        self.value = value
        self.prev: Node | None = None
        self.next: Node | None = None


class LRUCache:
    """
    A Least Recently Used (LRU) Cache implementation.

    Attributes:
        capacity (int): The maximum number of items the cache can hold.
        cache (dict): A dictionary mapping keys to Node objects for O(1) lookups.
        head (Node): A sentinel head node for the doubly linked list.
        tail (Node): A sentinel tail node for the doubly linked list.
    """

    def __init__(self, capacity: int):
        if capacity <= 0:
            raise ValueError("Capacity must be a positive integer")
        self.capacity = capacity
        self.cache: dict[Any, Any] = {}  # Stores key -> Node

        # Initialize sentinel head and tail nodes for the doubly linked list.
        # head.next points to the most recently used item.
        # tail.prev points to the least recently used item.
        self.head = Node(None, None)
        self.tail = Node(None, None)
        self.head.next = self.tail
        self.tail.prev = self.head

    def _remove_node(self, node: Node) -> None:
        """Removes a node from the doubly linked list."""
        if node.prev and node.next:
            prev_node = node.prev
            next_node = node.next
            prev_node.next = next_node
            next_node.prev = prev_node

    def _add_to_front(self, node: Node) -> None:
        """Adds a node to the front of the doubly linked list (right after head)."""
        node.prev = self.head
        node.next = self.head.next
        if self.head.next:
            self.head.next.prev = node
        self.head.next = node

    def clean(self) -> None:
        self.cache = {}
        self.head = Node(None, None)
        self.tail = Node(None, None)
        self.head.next = self.tail
        self.tail.prev = self.head

    def erase(self, key: Any) -> None:
        """
        Removes an item from the cache.
        """
        if key in self.cache:
            node = self.cache[key]
            self._remove_node(node)
            del self.cache[key]

    def get(self, key: Any) -> Any:
        """
        Retrieves an item from the cache.
        Returns the value if the key exists, otherwise -1 (or None/raise KeyError).
        Moves the accessed item to the front (most recently used).
        """
        if key in self.cache:
            node = self.cache[key]
            # TODO: Add TTL invalidation to nodes
            # Move accessed node to the front
            self._remove_node(node)
            self._add_to_front(node)
            return node.value
        return None

    def put(self, key: Any, value: Any) -> None:
        """
        Adds or updates an item in the cache.
        If the key exists, its value is updated and it's moved to the front.
        If the key doesn't exist, it's added.
        If the cache is full, the least recently used item is evicted.
        """
        if key in self.cache:
            # Update existing key's value and move it to the front
            node = self.cache[key]
            node.value = value
            self._remove_node(node)
            self._add_to_front(node)
        else:
            # Add new key
            if len(self.cache) >= self.capacity:
                # Cache is full, evict the least recently used item (from tail.prev)
                if (
                    self.tail.prev and self.tail.prev != self.head
                ):  # Ensure there's an item to evict
                    lru_node = self.tail.prev
                    self._remove_node(lru_node)
                    del self.cache[lru_node.key]

            new_node = Node(key, value)
            self.cache[key] = new_node
            self._add_to_front(new_node)
