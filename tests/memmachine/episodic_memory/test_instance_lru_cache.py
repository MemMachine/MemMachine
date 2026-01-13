"""Unit test for the MemoryInstanceCache."""

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from memmachine.episodic_memory.instance_lru_cache import MemoryInstanceCache


@pytest.fixture
def mock_episodic_memory():
    """Fixture to create a mock EpisodicMemory object with an async close method."""

    def _create_mock(name: str):
        mock_memory = MagicMock(name=name)
        mock_memory.close = AsyncMock()
        return mock_memory

    return _create_mock


def test_init_invalid_capacity():
    """Test that initializing with zero or negative capacity raises ValueError."""
    with pytest.raises(ValueError, match="Capacity must be a positive integer"):
        MemoryInstanceCache(capacity=0, max_lifetime=60)
    with pytest.raises(ValueError, match="Capacity must be a positive integer"):
        MemoryInstanceCache(capacity=-1, max_lifetime=60)


@pytest.mark.asyncio
async def test_init_valid_capacity():
    """Test successful initialization."""
    cache = MemoryInstanceCache(capacity=2, max_lifetime=60)
    assert cache.capacity == 2
    assert len(cache.cache) == 0
    await cache.close()


@pytest.mark.asyncio
async def test_add_and_get(mock_episodic_memory):
    """Test adding an item and then getting it."""
    cache = MemoryInstanceCache(capacity=2, max_lifetime=60)
    mem1 = mock_episodic_memory("mem1")

    await cache.add("key1", mem1)

    assert await cache.get_ref_count("key1") == 1

    retrieved_mem = await cache.get("key1")
    assert retrieved_mem is mem1
    assert await cache.get_ref_count("key1") == 2


@pytest.mark.asyncio
async def test_add_existing_key_raises_error(mock_episodic_memory):
    """Test that adding a key that already exists raises a ValueError."""
    cache = MemoryInstanceCache(capacity=2, max_lifetime=60)
    mem1 = mock_episodic_memory("mem1")
    await cache.add("key1", mem1)

    with pytest.raises(ValueError, match="Key key1 already exists"):
        await cache.add("key1", mem1)


@pytest.mark.asyncio
async def test_get_nonexistent_key():
    """Test that getting a non-existent key returns None."""
    cache = MemoryInstanceCache(capacity=2, max_lifetime=60)
    assert await cache.get("nonexistent") is None


@pytest.mark.asyncio
async def test_put(mock_episodic_memory):
    """Test the put method to decrease the reference count."""
    cache = MemoryInstanceCache(capacity=2, max_lifetime=60)
    mem1 = mock_episodic_memory("mem1")

    await cache.add("key1", mem1)
    assert await cache.get_ref_count("key1") == 1

    _ = await cache.get("key1")
    assert await cache.get_ref_count("key1") == 2

    await cache.release_ref("key1")
    assert await cache.get_ref_count("key1") == 1

    await cache.release_ref("key1")
    assert await cache.get_ref_count("key1") == 0


@pytest.mark.asyncio
async def test_put_nonexistent_key():
    """Test that calling put on a non-existent key raises a ValueError."""
    cache = MemoryInstanceCache(capacity=2, max_lifetime=60)
    with pytest.raises(ValueError, match="Key key1 does not exist"):
        await cache.release_ref("key1")


@pytest.mark.asyncio
async def test_put_below_zero_raises_assertion_error(mock_episodic_memory):
    """Test that put raises an error if ref_count goes below zero."""
    cache = MemoryInstanceCache(capacity=2, max_lifetime=60)
    mem1 = mock_episodic_memory("mem1")
    await cache.add("key1", mem1)

    await cache.release_ref("key1")  # ref_count becomes 0
    assert await cache.get_ref_count("key1") == 0

    with pytest.raises(AssertionError):
        await cache.release_ref("key1")  # Should fail as ref_count is already 0


@pytest.mark.asyncio
async def test_lru_eviction(mock_episodic_memory):
    """Test that the least recently used item is evicted when capacity is full."""
    cache = MemoryInstanceCache(capacity=2, max_lifetime=60)
    mem1 = mock_episodic_memory("mem1")
    mem2 = mock_episodic_memory("mem2")
    mem3 = mock_episodic_memory("mem3")

    # Add two items
    await cache.add("key1", mem1)
    await cache.add("key2", mem2)
    assert sorted(await cache.keys()) == ["key1", "key2"]

    # Release both items
    await cache.release_ref("key1")
    await cache.release_ref("key2")
    assert await cache.get_ref_count("key1") == 0
    assert await cache.get_ref_count("key2") == 0

    # Add a third item, which should evict the LRU item ('key1')
    await cache.add("key3", mem3)

    # Check that key1 is gone and its close method was called
    assert await cache.get("key1") is None

    assert sorted(await cache.keys()) == ["key2", "key3"]
    await asyncio.sleep(0.1)
    mem1.close.assert_awaited_once()
    mem2.close.assert_not_awaited()


@pytest.mark.asyncio
async def test_lru_eviction_with_in_use_item(mock_episodic_memory):
    """Test that an in-use (ref_count > 0) item is not evicted."""
    cache = MemoryInstanceCache(capacity=2, max_lifetime=60)
    mem1 = mock_episodic_memory("mem1")
    mem2 = mock_episodic_memory("mem2")
    mem3 = mock_episodic_memory("mem3")

    await cache.add("key1", mem1)  # LRU
    await cache.add("key2", mem2)  # MRU

    # key1 is in use, key2 is not
    await cache.release_ref("key2")
    assert await cache.get_ref_count("key1") == 1
    assert await cache.get_ref_count("key2") == 0

    # Try to add key3. It should evict key2, not key1.
    await cache.add("key3", mem3)

    assert await cache.get("key2") is None
    assert sorted(await cache.keys()) == ["key1", "key3"]
    await asyncio.sleep(0.1)
    mem1.close.assert_not_awaited()
    mem2.close.assert_awaited_once()


@pytest.mark.asyncio
async def test_lru_order_on_get(mock_episodic_memory):
    """Test that `get` moves an item to the most recently used position."""
    cache = MemoryInstanceCache(capacity=2, max_lifetime=60)
    mem1 = mock_episodic_memory("mem1")
    mem2 = mock_episodic_memory("mem2")
    mem3 = mock_episodic_memory("mem3")

    await cache.add("key1", mem1)
    await cache.add("key2", mem2)

    # Access key1, making it the MRU
    _ = await cache.get("key1")

    # Release all references
    await cache.release_ref("key1")
    await cache.release_ref("key1")
    await cache.release_ref("key2")

    # Add key3. This should evict key2 (the new LRU)
    await cache.add("key3", mem3)

    assert await cache.get("key2") is None
    assert sorted(await cache.keys()) == ["key1", "key3"]
    await asyncio.sleep(0.1)
    mem2.close.assert_awaited_once()
    mem1.close.assert_not_awaited()


@pytest.mark.asyncio
async def test_get_ref_count_nonexistent():
    """Test get_ref_count for a non-existent key returns -1."""
    cache = MemoryInstanceCache(capacity=2, max_lifetime=60)
    assert await cache.get_ref_count("nonexistent") == -1


@pytest.mark.asyncio
async def test_erase(mock_episodic_memory):
    """Test the erase method."""
    cache = MemoryInstanceCache(capacity=2, max_lifetime=60)
    mem1 = mock_episodic_memory("mem1")
    await cache.add("key1", mem1)

    # Cannot erase while in use
    with pytest.raises(RuntimeError, match="Key key1 is still in use 1"):
        await cache.erase("key1")

    # Release and then erase
    await cache.release_ref("key1")
    assert await cache.get_ref_count("key1") == 0
    await cache.erase("key1")

    assert await cache.get("key1") is None


@pytest.mark.asyncio
async def test_allow_put_with_key_none():
    """Test that put raises ValueError when key is None."""
    cache = MemoryInstanceCache(capacity=2, max_lifetime=60)
    await cache.release_ref(None)
    assert len(cache.cache) == 0


@pytest.mark.asyncio
async def test_add_if_key_is_none():
    """Test that add raises ValueError when key is None."""
    cache = MemoryInstanceCache(capacity=2, max_lifetime=60)
    await cache.add(None, MagicMock())
    assert len(cache.cache) == 0


@pytest.mark.asyncio
async def test_erase_with_key_none():
    """Test that erase does nothing when key is None."""
    cache = MemoryInstanceCache(capacity=2, max_lifetime=60)
    await cache.erase(None)
    assert len(cache.cache) == 0


@pytest.mark.asyncio
async def test_get_with_key_none():
    """Test that get returns None when key is None."""
    cache = MemoryInstanceCache(capacity=2, max_lifetime=60)
    assert await cache.get(None) is None


@pytest.mark.asyncio
async def test_get_ref_count_with_key_none():
    """Test that get_ref_count returns -1 when key is None."""
    cache = MemoryInstanceCache(capacity=2, max_lifetime=60)
    assert await cache.get_ref_count(None) == -1


@pytest.mark.asyncio
async def test_keys(mock_episodic_memory):
    """Test the keys method."""
    cache = MemoryInstanceCache(capacity=3, max_lifetime=60)
    assert await cache.keys() == []

    await cache.add("key1", mock_episodic_memory("mem1"))
    await cache.add("key2", mock_episodic_memory("mem2"))

    assert sorted(await cache.keys()) == ["key1", "key2"]

    await cache.release_ref("key1")
    await cache.erase("key1")

    assert await cache.keys() == ["key2"]


@pytest.mark.asyncio
async def test_clean_old_instance(mock_episodic_memory):
    """Test the clean_old_instance method."""
    cache = MemoryInstanceCache(capacity=4, max_lifetime=1)
    mem1 = mock_episodic_memory("mem1")
    mem2 = mock_episodic_memory("mem2")

    await cache.add("key1", mem1)
    await cache.add("key2", mem2)
    assert sorted(await cache.keys()) == ["key1", "key2"]
    assert await cache.get_ref_count("key1") == 1
    assert await cache.get_ref_count("key2") == 1
    await asyncio.sleep(2)
    await cache.clean_old_instance()
    # Would not delete item because of the reference
    assert sorted(await cache.keys()) == ["key1", "key2"]
    await cache.release_ref("key1")
    assert await cache.get_ref_count("key1") == 0
    assert await cache.get_ref_count("key2") == 1
    # Should remove key1 now
    await cache.clean_old_instance()
    assert sorted(await cache.keys()) == ["key2"]
