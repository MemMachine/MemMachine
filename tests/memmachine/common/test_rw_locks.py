import asyncio

import pytest

from src.memmachine.common.rw_locks import AsyncRWLock, AsyncRWLockPool


@pytest.mark.asyncio
async def test_acquire_and_release_read():
    lock = AsyncRWLock()
    await lock.acquire_read()
    # does not block when acquiring read lock again
    await lock.acquire_read()
    await lock.release_read()
    await lock.release_read()


@pytest.mark.asyncio
async def test_acquire_and_release_write():
    lock = AsyncRWLock()
    await lock.acquire_write()
    lock.release_write()


@pytest.mark.asyncio
async def test_readers_blocked_by_writer():
    lock = AsyncRWLock()
    await lock.acquire_write()
    read_acquired = False

    async def try_read():
        nonlocal read_acquired
        await lock.acquire_read()
        read_acquired = True
        await lock.release_read()

    task = asyncio.create_task(try_read())
    await asyncio.sleep(0.1)
    assert not read_acquired
    lock.release_write()
    await asyncio.sleep(0.1)
    assert read_acquired
    task.cancel()


@pytest.mark.asyncio
async def test_writer_blocked_by_reader():
    lock = AsyncRWLock()
    await lock.acquire_read()
    write_acquired = False

    async def try_write():
        nonlocal write_acquired
        await lock.acquire_write()
        write_acquired = True
        lock.release_write()

    task = asyncio.create_task(try_write())
    await asyncio.sleep(0.1)
    assert not write_acquired
    await lock.release_read()
    await asyncio.sleep(0.1)
    assert write_acquired
    task.cancel()


@pytest.mark.asyncio
async def test_read_lock_allows_concurrent_reads():
    lock = AsyncRWLock()
    results = []

    async def reader(idx):
        async with lock.read_lock():
            results.append(f"reader{idx}_acquired")
            await asyncio.sleep(0.1)
            results.append(f"reader{idx}_released")

    await asyncio.gather(reader(1), reader(2))

    assert results == [
        "reader1_acquired",
        "reader2_acquired",
        "reader1_released",
        "reader2_released",
    ] or results == [
        "reader2_acquired",
        "reader1_acquired",
        "reader2_released",
        "reader1_released",
    ]


@pytest.mark.asyncio
async def test_write_lock_excludes_others():
    lock = AsyncRWLock()
    order = []

    async def writer():
        async with lock.write_lock():
            order.append("writer_acquired")
            await asyncio.sleep(0.01)
            order.append("writer_released")

    async def reader():
        async with lock.read_lock():
            order.append("reader_acquired")
            await asyncio.sleep(0.01)
            order.append("reader_released")

    t1 = asyncio.create_task(writer())
    t2 = asyncio.create_task(reader())
    await asyncio.gather(t1, t2)
    assert order in (
        [
            "writer_acquired",
            "writer_released",
            "reader_acquired",
            "reader_released",
        ],
        [
            "reader_acquired",
            "reader_released",
            "writer_acquired",
            "writer_released",
        ],
    ), f"order={order}"


@pytest.mark.asyncio
async def test_readers_blocked_by_writer_with():
    lock = AsyncRWLock()
    events = []

    async def writer():
        async with lock.write_lock():
            events.append("writer_acquired")
            await asyncio.sleep(0.1)
            events.append("writer_released")

    async def reader():
        async with lock.read_lock():
            events.append("reader_acquired")
            await asyncio.sleep(0.1)
            events.append("reader_released")

    t1 = asyncio.create_task(writer())
    await asyncio.sleep(0.01)
    t2 = asyncio.create_task(reader())

    await asyncio.gather(t1, t2)
    assert events[0] == "writer_acquired"
    assert events[1] == "writer_released"
    assert events[2] == "reader_acquired"
    assert events[3] == "reader_released"


@pytest.mark.asyncio
async def test_lock_cleanup():
    """Verify that locks are removed from the dictionary after use."""
    manager = AsyncRWLockPool(8)
    key = "resource_1"

    async with manager.read_lock(key):
        assert len(manager._locks) == 8

    await manager.close()
    # After close, dictionary should be empty
    assert len(manager._locks) == 0


@pytest.mark.asyncio
async def test_concurrent_readers():
    """Verify multiple readers can access the same key simultaneously."""
    manager = AsyncRWLockPool()
    key = "shared_key"
    results = []

    async def read_task(task_id):
        async with manager.read_lock(key):
            results.append(f"start_{task_id}")
            await asyncio.sleep(0.1)
            results.append(f"end_{task_id}")

    await asyncio.gather(read_task(1), read_task(2))

    assert "start_1" in results
    assert "start_2" in results
    assert results[:2] == [
        "start_1",
        "start_2",
    ] or results[:2] == [
        "start_2",
        "start_1",
    ]


@pytest.mark.asyncio
async def test_writer_exclusion():
    """Verify a writer blocks other writers and readers."""
    manager = AsyncRWLockPool()
    key = "exclusive_key"
    status = []

    async def writer():
        async with manager.write_lock(key):
            status.append("writing")
            await asyncio.sleep(0.1)
            status.append("done_writing")

    async def reader():
        await asyncio.sleep(0.1)
        async with manager.read_lock(key):
            status.append("reading")

    await asyncio.gather(writer(), reader())
    # Reader must wait until writer is done
    assert status == ["writing", "done_writing", "reading"]


@pytest.mark.asyncio
async def test_multiple_keys_independent():
    """Verify locks for different keys do not block each other."""
    manager = AsyncRWLockPool()

    # This should complete instantly if they don't block
    async with manager.write_lock("a"), manager.write_lock("b"):
        pass


@pytest.mark.asyncio
async def test_writer_priority_blocks_new_readers():
    """
    Scenario:
    1. Reader 1 starts and holds the lock.
    2. Writer 1 tries to acquire (blocked by Reader 1).
    3. Reader 2 tries to acquire (should be blocked by Writer 1's presence).

    Result: Reader 1 must finish, then Writer 1 MUST go before Reader 2.
    """
    manager = AsyncRWLockPool()
    key = "priority_key"
    execution_order = []

    async def reader_1():
        async with manager.read_lock(key):
            execution_order.append("reader_1_start")
            await asyncio.sleep(0.2)  # Hold the lock long enough for others to arrive
            execution_order.append("reader_1_end")

    async def writer_1():
        await asyncio.sleep(0.05)  # Arrive after reader_1
        async with manager.write_lock(key):
            execution_order.append("writer_1_start")
            await asyncio.sleep(0.1)
            execution_order.append("writer_1_end")

    async def reader_2():
        await asyncio.sleep(0.1)  # Arrive after writer_1 is already waiting
        async with manager.read_lock(key):
            execution_order.append("reader_2_start")

    # Run concurrently
    await asyncio.gather(reader_1(), writer_1(), reader_2())

    expected = [
        "reader_1_start",
        "reader_1_end",  # Reader 1 finishes
        "writer_1_start",  # Writer 1 gets it next (priority!)
        "writer_1_end",  # Writer 1 finishes
        "reader_2_start",  # Reader 2 finally gets in
    ]
    assert execution_order == expected


@pytest.mark.asyncio
async def test_concurrent_write_exclusion():
    """Verify that two writers cannot access the same key at once."""
    manager = AsyncRWLockPool()
    key = "exclusive"
    counter = 0

    async def writer_task():
        nonlocal counter
        async with manager.write_lock(key):
            # If exclusion fails, counter might be modified by two tasks at once
            current = counter
            await asyncio.sleep(0.05)
            counter = current + 1

    await asyncio.gather(writer_task(), writer_task(), writer_task())
    assert counter == 3


@pytest.mark.asyncio
async def test_multiple_independent_keys():
    """Verify that locking 'key_a' does not block 'key_b'."""
    manager = AsyncRWLockPool()

    events = []

    async def lock_a():
        async with manager.write_lock("a"):
            events.append("a_locked")
            await asyncio.sleep(0.1)
            events.append("a_unlocked")

    async def lock_b():
        async with manager.write_lock("b"):
            events.append("b_locked")
            await asyncio.sleep(0.1)
            events.append("b_unlocked")

    await asyncio.gather(lock_a(), lock_b())
    assert events[:2] == [
        "a_locked",
        "b_locked",
    ] or events[:2] == [
        "b_locked",
        "a_locked",
    ]
    assert "a_locked" in events
    assert "b_locked" in events
    assert "a_unlocked" in events
    assert "b_unlocked" in events


@pytest.mark.asyncio
async def test_lock_reuse():
    """Verify that a lock released by one key is reused by another."""
    manager = AsyncRWLockPool(pool_size=1)

    # 1. Use and release a lock for 'key1'
    async with manager.write_lock("key1"):
        pass

    assert len(manager._locks) == 1
    original_lock_id = id(manager._locks[0])

    # 2. Immediately request a lock for 'key2'
    async with manager.write_lock("key2"):
        # The manager should have pulled the lock from the pool
        assert len(manager._locks) == 1
        # Check that the actual object is the same
        # We access the internal _locks for testing purposes
        current_lock = id(manager._locks[0])
        assert current_lock == original_lock_id

    await manager.close()


@pytest.mark.asyncio
async def test_hash_collision_in_lock_pool_deterministic():
    """Verify that different keys resulting in the same hash will wait using event logs"""
    # Use a pool size of 1 to force every different key to collide on the same slot
    manager = AsyncRWLockPool(pool_size=1)
    events = []

    async def use_lock(key, task_id):
        async with manager.write_lock(key):
            events.append(f"acquire_{task_id}")
            await asyncio.sleep(0.05)  # Short sleep just to yield control
            events.append(f"release_{task_id}")

    # Gather tasks. Because pool_size is 1, they MUST run sequentially.
    await asyncio.gather(
        use_lock("a", 1),
        use_lock("b", 2),
    )

    # Verify the sequence: acquire_1 -> release_1 -> acquire_2 -> release_2
    # If the locks didn't collide/wait, we would see acquire_1 -> acquire_2
    assert events[0] == "acquire_1"
    assert events[1] == "release_1"
    assert events[2] == "acquire_2"
    assert events[3] == "release_2"

    assert events in (
        [
            "acquire_1",
            "release_1",
            "acquire_2",
            "release_2",
        ],
        [
            "acquire_2",
            "release_2",
            "acquire_1",
            "release_1",
        ],
    ), f"events={events}"

    assert len(manager._locks) == 1
    await manager.close()
