"""Asynchronous Read-Write Lock Implementation."""

import asyncio
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager


class AsyncRWLock:
    """
    An asynchronous read-write lock for use with asyncio.

    This lock allows multiple concurrent readers, but only one writer at a time.
    A waiting writer blocks new readers from entering, so writers cannot be
    starved by a steady stream of readers.

    Usage:
        lock = AsyncRWLock()

        async with lock.read_lock():
            ...
        async with lock.write_lock():
            ...

        # Or explicitly:
        await lock.acquire_read()
        try:
            ...
        finally:
            lock.release_read()
    """

    def __init__(self) -> None:
        """Initialize the AsyncRWLock."""
        self._readers = 0
        # Held by an active writer, or by the cohort of concurrent readers
        # (acquired by the first reader, released by the last). Provides
        # mutual exclusion between readers and writers.
        self._writer_lock = asyncio.Lock()
        # Held by a writer (waiting for or holding _writer_lock) to block
        # new readers, giving writer priority over arriving readers.
        # Readers pass through it to enforce the same FIFO queueing.
        self._read_gate = asyncio.Lock()

    @asynccontextmanager
    async def read_lock(self) -> AsyncGenerator[None, None]:
        """Acquire a read lock as an async context manager."""
        await self.acquire_read()
        try:
            yield
        finally:
            self.release_read()

    @asynccontextmanager
    async def write_lock(self) -> AsyncGenerator[None, None]:
        """Acquire a write lock as an async context manager."""
        await self.acquire_write()
        try:
            yield
        finally:
            self.release_write()

    async def acquire_read(self) -> None:
        """Acquire a read lock; multiple readers may hold it simultaneously."""
        await self._read_gate.acquire()
        try:
            # Increment the counter only after _writer_lock is acquired,
            # so cancellation during the await leaves _readers unchanged.
            if self._readers == 0:
                await self._writer_lock.acquire()
            self._readers += 1
        finally:
            self._read_gate.release()

    def release_read(self) -> None:
        """Release a read lock previously acquired by acquire_read()."""
        self._readers -= 1
        if self._readers == 0:
            self._writer_lock.release()

    async def acquire_write(self) -> None:
        """Acquire a write lock; excludes all readers and other writers."""
        await self._read_gate.acquire()
        try:
            await self._writer_lock.acquire()
        except BaseException:
            # Cancellation or error while waiting for _writer_lock:
            # release _read_gate so the lock isn't permanently jammed.
            self._read_gate.release()
            raise

    def release_write(self) -> None:
        """Release a write lock previously acquired by acquire_write()."""
        self._writer_lock.release()
        self._read_gate.release()
