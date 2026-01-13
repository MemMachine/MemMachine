"""Asynchronous Read-Write Lock Implementation."""

import asyncio
import logging
from collections.abc import AsyncGenerator, AsyncIterator
from contextlib import asynccontextmanager

logger = logging.getLogger(__name__)


class AsyncRWLock:
    """
    An asynchronous read-write lock for use with asyncio.

    This lock allows multiple concurrent readers, but only one writer at a time.
    When a writer holds the lock, all readers and other writers are blocked
    until the writer releases the lock.

    Usage:
        lock = AsyncRWLock()

        # Acquire a read lock
        async with lock.read_lock():
            # perform read operations

        # Acquire a write lock
        async with lock.write_lock():
            # perform write operations
    """

    def __init__(self) -> None:
        """Initialize the AsyncRWLock."""
        self._readers = 0
        self._readers_lock = asyncio.Lock()  # Protects the reader count
        self._writer_lock = asyncio.Lock()  # The actual exclusion lock
        self._read_gate = (
            asyncio.Lock()
        )  # Prevents new readers when a writer is waiting

    @asynccontextmanager
    async def read_lock(self) -> AsyncGenerator[None, None]:
        """Acquire a read lock."""
        # 1. Wait at the gate. If a writer is waiting or active, this blocks.
        # We acquire and immediately release to just 'pass through' the gate.
        async with self._read_gate, self._readers_lock:
            self._readers += 1
            if self._readers == 1:
                # First reader acquires the writer lock to block writers
                await self._writer_lock.acquire()
        try:
            yield
        finally:
            async with self._readers_lock:
                self._readers -= 1
                if self._readers == 0:
                    # Last reader releases the writer lock
                    self._writer_lock.release()

    @asynccontextmanager
    async def write_lock(self) -> AsyncGenerator[None, None]:
        """Acquire a write lock."""
        # 1. Close the gate so no NEW readers can enter.
        # 2. Acquire the writer lock to wait for EXISTING readers to finish.
        async with self._read_gate, self._writer_lock:
            yield

    async def acquire_read(self) -> None:
        """Acquire a read lock."""
        async with self._read_gate, self._readers_lock:
            self._readers += 1
            if self._readers == 1:
                await self._writer_lock.acquire()

    async def release_read(self) -> None:
        """Release a read lock."""
        async with self._readers_lock:
            self._readers -= 1
            if self._readers == 0:
                self._writer_lock.release()

    async def acquire_write(self) -> None:
        """Acquire a write lock."""
        # Writers must acquire the gate first to stop new readers
        await self._read_gate.acquire()
        await self._writer_lock.acquire()

    def release_write(self) -> None:
        """Release a write lock."""
        self._writer_lock.release()
        self._read_gate.release()


class AsyncRWLockPool:
    """
    Manages a fixed-size pool of AsyncRWLocks using hash-based striping.

    This implementation eliminates global lock contention and the need for
    background cleanup tasks. Multiple keys may map to the same lock (collision),
    but with a large enough pool size, this is rare.
    """

    def __init__(self, pool_size: int = 128) -> None:
        """
        Initialize the manager with a fixed number of locks.

        Args:
            pool_size: The number of underlying locks. Higher values reduce
                       the chance of collisions between different keys.

        """
        if pool_size <= 0:
            raise ValueError("pool_size must be a positive integer")

        self._pool_size = pool_size
        # Pre-allocate all locks
        self._locks = [AsyncRWLock() for _ in range(pool_size)]

    def _get_lock(self, key: str) -> AsyncRWLock:
        """Calculate the hash-based index and return the associated lock."""
        # hash() is stable within a single process run
        index = hash(key) % self._pool_size
        return self._locks[index]

    @asynccontextmanager
    async def read_lock(self, key: str) -> AsyncIterator[None]:
        """Acquire the read lock for the bucket associated with the key."""
        lock = self._get_lock(key)
        async with lock.read_lock():
            yield

    @asynccontextmanager
    async def write_lock(self, key: str) -> AsyncIterator[None]:
        """Acquire the write lock for the bucket associated with the key."""
        lock = self._get_lock(key)
        async with lock.write_lock():
            yield

    async def close(self) -> None:
        """
        Clear the pool.

        Since there's no background task,
        this just helps with garbage collection.
        """
        self._locks.clear()
