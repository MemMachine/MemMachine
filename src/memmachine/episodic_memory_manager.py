
import asyncio
from contextlib import asynccontextmanager
from typing import AsyncIterator

from memmachine.configuration.episodic_config import (
    EpisodicMemoryManagerParam,
    EpisodicMemoryParams,
)
from memmachine.episodic_memory.episodic_memory import EpisodicMemory

from .instance_lru_cache import MemoryInstanceCache


class EpisodicMemoryManager:
    """
    Manages the lifecycle and access of semantic memory instances.

    This class is responsible for creating, retrieving, and closing
    `SemanticMemory` instances based on a session key. It uses a
    reference counting mechanism to manage the lifecycle of each memory
    instance, ensuring that resources are properly released when no
    longer needed.
    """

    def __init__(self, param: EpisodicMemoryManagerParam):
        """
        Initializes the SemanticMemoryManager.

        Args:
            config: The overall MemMachine configuration.
        """
        self._instance_cache: MemoryInstanceCache = MemoryInstanceCache(
            param.instance_cache_size, param.max_life_time
        )
        self._session_storage = param.session_storage
        self._lock = asyncio.Lock()
        self._closed = False
        self._check_instance_task = asyncio.create_task(self._check_instance_life_time())


    async def _check_instance_life_time(self):
        while not self._closed:
            await asyncio.sleep(2)
            async with self._lock:
                await self._instance_cache.clean_old_instance()

    @asynccontextmanager
    async def open_episodic_memory(self, session_key: str) -> AsyncIterator[EpisodicMemory]:
        """
        Asynchronously provides a SemanticMemory instance for a given session key.

        This is an asynchronous context manager. It will create a new
        `SemanticMemory` instance if one doesn't exist for the given session key,
        or return an existing one. It manages a reference count for each instance.

        Args:
            session_key: The unique identifier for the session.

        Yields:
            A SemanticMemory instance.

        Raises:
            ValueError: If semantic memory is not enabled in the configuration.
        """
        instance: EpisodicMemory | None = None
        async with self._lock:
            if self._closed:
                raise RuntimeError(f"Memory is closed {session_key}")

            # Check if the instance is in the cache and in use
            instance = self._instance_cache.get(session_key)
            if instance is None:
                # load from the database
                _, _, _, param = await self._session_storage.get_session_info(session_key)
                # TODO: callback to instantiate the param
                instance = await EpisodicMemory.create(param)
                await self._instance_cache.add(session_key, instance)
        try:
            yield instance
        finally:
            if instance is not None:
                async with self._lock:
                    self._instance_cache.put(session_key)

    @asynccontextmanager
    async def create_episodic_memory(self, session_key: str, param: EpisodicMemoryParams, description: str, metadata: dict, config: dict | None = None) -> AsyncIterator[EpisodicMemory]:
        """
        Creates a new episodic memory instance and stores its configuration.

        Args:
            session_key: The unique identifier for the session.
            param: The parameters for configuring the episodic memory.
            description: A brief description of the session.
            metadata: User-defined metadata for the session.

        Raises:
            ValueError: If a session with the given session_key already exists.
        """
        instance: EpisodicMemory | None = None
        if config is None:
            config = {}
        async with self._lock:
            if self._closed:
                raise RuntimeError(f"Memory is closed {session_key}")

            await self._session_storage.create_new_session(session_key, config, param, description, metadata)
            instance = await EpisodicMemory.create(param)
            await self._instance_cache.add(session_key, instance)
        try:
            yield instance
        finally:
            if instance is not None:
                async with self._lock:
                    self._instance_cache.put(session_key)

    async def delete_episodic_session(self, session_key: str):
        """
        Deletes an episodic memory instance and its associated data.

        Args:
            session_key: The unique identifier of the session to delete.
        """
        instance: EpisodicMemory | None = None
        async with self._lock:
            if self._closed:
                raise RuntimeError(f"Memory is closed {session_key}")
            # Check if the instance is in the cache and in use
            ref_count = self._instance_cache.get_ref_count(session_key)
            instance = self._instance_cache.get(session_key)
            if instance and ref_count > 0:
                raise RuntimeError(f"Session {session_key} is still in use {ref_count}")
            if instance:
                self._instance_cache.put(session_key)
            self._instance_cache.erase(session_key)
            if instance is None:
                # Open it
                _, _, _, param = await self._session_storage.get_session_info(session_key)
                instance = await EpisodicMemory.create(param)
            await instance.delete_data()
            await instance.close()
            await self._session_storage.delete_session(session_key)


    async def get_episodic_memory_keys(self, filter: dict[str, str] | None) -> list[str]:
        """
        Retrieves a list of all available episodic memory session keys.

        Returns:
            A list of session keys.
        """
        return await self._session_storage.get_sessions(filter)

    async def get_session_configuration(self, session_key: str) -> tuple[dict, str, dict, EpisodicMemoryParams]:
        """
        Retrieves the configuration, description, and metadata for a given session.
        """
        return await self._session_storage.get_session_info(session_key)

    async def close_session(self, session_key: str):
        """
        Closes an idle episodic memory instance and its associated data.

        Args:
            session_key: The unique identifier of the session to close.
        """
        async with self._lock:
            if self._closed:
                raise RuntimeError(f"Memory is closed {session_key}")
            ref_count = self._instance_cache.get_ref_count(session_key)
            if ref_count < 0:
                return
            if ref_count > 0:
                raise RuntimeError(f"Session {session_key} is busy")
            instance = self._instance_cache.get(session_key)
            if instance is not None:
                await instance.close()
                self._instance_cache.put(session_key)
            self._instance_cache.erase(session_key)

    async def close(self):
        """
        Closes all open episodic memory instances and the session storage.
        """
        tasks = []
        async with self._lock:
            if self._closed:
                return
            for key in self._instance_cache.keys():
                tasks.append(self._instance_cache.get(key).close())
            await asyncio.gather(*tasks)
            await self._session_storage.close()
            self._instance_cache.clear()
            self._closed = True

        if hasattr(self, "_check_instance_task"):
            await self._check_instance_task
