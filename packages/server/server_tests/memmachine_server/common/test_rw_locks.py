import asyncio

import pytest

from memmachine_server.common.rw_locks import AsyncRWLock


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
async def test_writer_priority_blocks_new_readers():
    """
    Scenario:
    1. Reader 1 starts and holds the lock.
    2. Writer 1 tries to acquire (blocked by Reader 1).
    3. Reader 2 tries to acquire (should be blocked by Writer 1's presence).

    Result: Reader 1 must finish, then Writer 1 MUST go before Reader 2.
    """
    lock = AsyncRWLock()
    execution_order = []

    async def reader_1():
        async with lock.read_lock():
            execution_order.append("reader_1_start")
            await asyncio.sleep(0.2)
            execution_order.append("reader_1_end")

    async def writer_1():
        await asyncio.sleep(0.05)
        async with lock.write_lock():
            execution_order.append("writer_1_start")
            await asyncio.sleep(0.1)
            execution_order.append("writer_1_end")

    async def reader_2():
        await asyncio.sleep(0.1)
        async with lock.read_lock():
            execution_order.append("reader_2_start")

    await asyncio.gather(reader_1(), writer_1(), reader_2())

    expected = [
        "reader_1_start",
        "reader_1_end",
        "writer_1_start",
        "writer_1_end",
        "reader_2_start",
    ]
    assert execution_order == expected


@pytest.mark.asyncio
async def test_concurrent_write_exclusion():
    """Verify that two writers cannot access the same lock at once."""
    lock = AsyncRWLock()
    counter = 0

    async def writer_task():
        nonlocal counter
        async with lock.write_lock():
            current = counter
            await asyncio.sleep(0.05)
            counter = current + 1

    await asyncio.gather(writer_task(), writer_task(), writer_task())
    assert counter == 3
