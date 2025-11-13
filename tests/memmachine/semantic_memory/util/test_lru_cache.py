"""Unit tests for the LRU Cache utility."""

import pytest

from memmachine.semantic_memory.util.lru_cache import LRUCache, Node


class TestNode:
    """Tests for the Node class."""

    def test_node_initialization(self):
        node = Node("key1", "value1")
        assert node.key == "key1"
        assert node.value == "value1"
        assert node.prev is None
        assert node.next is None


class TestLRUCache:
    """Tests for the LRUCache class."""

    def test_initialization_valid_capacity(self):
        cache = LRUCache(5)
        assert cache.capacity == 5
        assert len(cache.cache) == 0

    def test_initialization_invalid_capacity_zero(self):
        with pytest.raises(ValueError, match="Capacity must be a positive integer"):
            LRUCache(0)

    def test_initialization_invalid_capacity_negative(self):
        with pytest.raises(ValueError, match="Capacity must be a positive integer"):
            LRUCache(-1)

    def test_put_and_get_single_item(self):
        cache = LRUCache(2)
        cache.put("key1", "value1")
        assert cache.get("key1") == "value1"

    def test_get_nonexistent_key_returns_none(self):
        cache = LRUCache(2)
        assert cache.get("missing") is None

    def test_put_updates_existing_key(self):
        cache = LRUCache(2)
        cache.put("key1", "value1")
        cache.put("key1", "updated_value")
        assert cache.get("key1") == "updated_value"
        assert len(cache.cache) == 1

    def test_eviction_when_capacity_exceeded(self):
        cache = LRUCache(2)
        cache.put("key1", "value1")
        cache.put("key2", "value2")
        cache.put("key3", "value3")  # Should evict key1

        assert cache.get("key1") is None
        assert cache.get("key2") == "value2"
        assert cache.get("key3") == "value3"

    def test_lru_ordering_on_get(self):
        cache = LRUCache(2)
        cache.put("key1", "value1")
        cache.put("key2", "value2")

        # Access key1 to make it most recently used
        cache.get("key1")

        # Adding key3 should evict key2, not key1
        cache.put("key3", "value3")

        assert cache.get("key1") == "value1"
        assert cache.get("key2") is None
        assert cache.get("key3") == "value3"

    def test_lru_ordering_on_update(self):
        cache = LRUCache(2)
        cache.put("key1", "value1")
        cache.put("key2", "value2")

        # Update key1 to make it most recently used
        cache.put("key1", "updated")

        # Adding key3 should evict key2, not key1
        cache.put("key3", "value3")

        assert cache.get("key1") == "updated"
        assert cache.get("key2") is None
        assert cache.get("key3") == "value3"

    def test_erase_existing_key(self):
        cache = LRUCache(3)
        cache.put("key1", "value1")
        cache.put("key2", "value2")

        cache.erase("key1")

        assert cache.get("key1") is None
        assert cache.get("key2") == "value2"
        assert len(cache.cache) == 1

    def test_erase_nonexistent_key(self):
        cache = LRUCache(2)
        cache.put("key1", "value1")

        # Should not raise error
        cache.erase("missing")

        assert cache.get("key1") == "value1"
        assert len(cache.cache) == 1

    def test_clean_empties_cache(self):
        cache = LRUCache(3)
        cache.put("key1", "value1")
        cache.put("key2", "value2")
        cache.put("key3", "value3")

        cache.clear()

        assert len(cache.cache) == 0
        assert cache.get("key1") is None
        assert cache.get("key2") is None
        assert cache.get("key3") is None

    def test_clean_allows_reuse(self):
        cache = LRUCache(2)
        cache.put("key1", "value1")
        cache.clear()

        cache.put("key2", "value2")
        cache.put("key3", "value3")

        assert cache.get("key2") == "value2"
        assert cache.get("key3") == "value3"

    def test_capacity_one_cache(self):
        cache = LRUCache(1)
        cache.put("key1", "value1")
        assert cache.get("key1") == "value1"

        cache.put("key2", "value2")
        assert cache.get("key1") is None
        assert cache.get("key2") == "value2"

    def test_multiple_gets_maintain_order(self):
        cache = LRUCache(2)
        cache.put("key1", "value1")
        cache.put("key2", "value2")

        # Access key1 multiple times
        cache.get("key1")
        cache.get("key1")
        cache.get("key1")

        # key2 should still be evicted when adding key3
        cache.put("key3", "value3")

        assert cache.get("key1") == "value1"
        assert cache.get("key2") is None
        assert cache.get("key3") == "value3"

    def test_various_data_types_as_values(self):
        cache = LRUCache(5)
        cache.put("int", 42)
        cache.put("list", [1, 2, 3])
        cache.put("dict", {"a": 1})
        cache.put("none", None)
        cache.put("bool", True)

        assert cache.get("int") == 42
        assert cache.get("list") == [1, 2, 3]
        assert cache.get("dict") == {"a": 1}
        assert cache.get("none") is None
        assert cache.get("bool") is True

    def test_erase_affects_eviction_order(self):
        cache = LRUCache(2)
        cache.put("key1", "value1")
        cache.put("key2", "value2")

        # Erase key1
        cache.erase("key1")

        # Add two new keys
        cache.put("key3", "value3")
        cache.put("key4", "value4")

        # key2 should be evicted
        assert cache.get("key2") is None
        assert cache.get("key3") == "value3"
        assert cache.get("key4") == "value4"
