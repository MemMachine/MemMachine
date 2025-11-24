"""Factory and manager for per-session episodic memory instances."""

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from pydantic import BaseModel, Field, InstanceOf

from memmachine.common.configuration.episodic_config import EpisodicMemoryConf
from memmachine.common.resource_manager import CommonResourceManager
from memmachine.common.session_manager.session_data_manager import SessionDataManager
from memmachine.episodic_memory.episodic_memory import EpisodicMemory
from memmachine.episodic_memory.service_locator import (
    episodic_memory_params_from_config,
)

from .instance_lru_cache import MemoryInstanceCache


class EpisodicMemoryManagerParams(BaseModel):
    """
    Parameters for configuring the EpisodicMemoryManager.

    Attributes:
        instance_cache_size (int): The maximum number of instances to cache.
        max_life_time (int): The maximum idle lifetime of an instance in seconds.
        resource_manager (ResourceManager): The resource manager.
        session_data_manager (SessionDataManager): The session data manager.

    """

    instance_cache_size: int = Field(
        default=100,
        gt=0,
        description="The maximum number of instances to cache",
    )
    max_life_time: int = Field(
        default=600,
        gt=0,
        description="The maximum idle lifetime of an instance in seconds",
    )
    resource_manager: InstanceOf[CommonResourceManager] = Field(
        ...,
        description="Resource manager",
    )
    session_data_manager: InstanceOf[SessionDataManager] = Field(
        ...,
        description="Session data manager",
    )


class EpisodicMemoryManager:
    """
    Manage the lifecycle and access of episodic memory instances.

    This class is responsible for creating, retrieving, and closing
    `SemanticMemory` instances based on a session key. It uses a
    reference counting mechanism to manage the lifecycle of each memory
    instance, ensuring that resources are properly released when no
    longer needed.
    """

    def __init__(self, params: EpisodicMemoryManagerParams) -> None:
        """
        Initialize the SemanticMemoryManager.

        Args:
            params: The configuration parameters for the manager.

        """
        self._instance_cache: MemoryInstanceCache = MemoryInstanceCache(
            params.instance_cache_size,
            params.max_life_time,
        )
        self._resource_manager = params.resource_manager
        self._session_data_manager = params.session_data_manager

        self._lock = asyncio.Lock()
        self._closed = False
        self._check_instance_task = asyncio.create_task(
            self._check_instance_life_time(),
        )

    async def _check_instance_life_time(self) -> None:
        while not self._closed:
            await asyncio.sleep(2)
            async with self._lock:
                await self._instance_cache.clean_old_instance()

    @asynccontextmanager
    async def open_episodic_memory(
        self,
        session_key: str,
    ) -> AsyncIterator[EpisodicMemory]:
        """
        Provide a SemanticMemory instance for a given session key.

        This is an asynchronous context manager. It will create a new
        `SemanticMemory` instance if one doesn't exist for the given session key,
        or return an existing one. It manages a reference count for each instance.

        Args:
            session_key: The unique identifier for the session.

        Yields:
            A SemanticMemory instance.

        Raises:
            ValueError: If episodic memory is not enabled in the configuration.

        """
        instance: EpisodicMemory | None = None
        async with self._lock:
            if self._closed:
                raise RuntimeError(f"Memory is closed {session_key}")

            # Check if the instance is in the cache and in use
            instance = self._instance_cache.get(session_key)
            if instance is None:
                # load from the database
                session_info = await self._session_data_manager.get_session_info(
                    session_key
                )
                if session_info is None:
                    raise RuntimeError(
                        f"No session info found for session {session_key}"
                    )

                episodic_memory_params = await episodic_memory_params_from_config(
                    session_info.episode_memory_conf,
                    self._resource_manager,
                )
                instance = EpisodicMemory(episodic_memory_params)
                await self._instance_cache.add(session_key, instance)
        try:
            yield instance
        finally:
            if instance is not None:
                async with self._lock:
                    self._instance_cache.put(session_key)

    @asynccontextmanager
    async def create_episodic_memory(
        self,
        session_key: str,
        episodic_memory_config: EpisodicMemoryConf,
        description: str,
        metadata: dict,
        config: dict | None = None,
    ) -> AsyncIterator[EpisodicMemory]:
        """
        Create a new episodic memory instance and store its configuration.

        Args:
            session_key: The unique identifier for the session.
            episodic_memory_config: Parameters for configuring the episodic memory.
            description: A brief description of the session.
            metadata: User-defined metadata for the session.
            config: Additional configuration values for the session metadata.

        Raises:
            ValueError: If a session with the given session_key already exists.

        """
        instance: EpisodicMemory | None = None
        if config is None:
            config = {}
        async with self._lock:
            if self._closed:
                raise RuntimeError(f"Memory is closed {session_key}")

            await self._session_data_manager.create_new_session(
                session_key,
                config,
                episodic_memory_config,
                description,
                metadata,
            )
            episodic_memory_params = await episodic_memory_params_from_config(
                episodic_memory_config,
                self._resource_manager,
            )
            instance = EpisodicMemory(episodic_memory_params)
            await self._instance_cache.add(session_key, instance)
        try:
            yield instance
        finally:
            if instance is not None:
                async with self._lock:
                    self._instance_cache.put(session_key)

    @asynccontextmanager
    async def open_or_create_episodic_memory(
        self,
        session_key: str,
        episodic_memory_config: EpisodicMemoryConf,
        description: str,
        metadata: dict,
        config: dict | None = None,
    ) -> AsyncIterator[EpisodicMemory]:
        """
        Create a new episodic memory instance and store its configuration if it doesn't exist. If the session already exists, it will be opened and returned.

        Args:
            session_key: The unique identifier for the session.
            episodic_memory_config: Parameters for configuring the episodic memory.
            description: A brief description of the session.
            metadata: User-defined metadata for the session.
            config: Additional configuration values for the session metadata.

        """
        instance: EpisodicMemory | None = None
        if config is None:
            config = {}
        async with self._lock:
            if self._closed:
                raise RuntimeError(f"Memory is closed {session_key}")

            # Check if the instance is in the cache
            instance = self._instance_cache.get(session_key)
            if instance is None:
                # try to load from the database
                session_info = await self._session_data_manager.get_session_info(
                    session_key
                )
                if session_info is not None:
                    episodic_memory_params = await episodic_memory_params_from_config(
                        session_info.episode_memory_conf,
                        self._resource_manager,
                    )
                    instance = EpisodicMemory(episodic_memory_params)
                    await self._instance_cache.add(session_key, instance)

            if instance is None:
                # session does not exist, create it
                await self._session_data_manager.create_new_session(
                    session_key,
                    config,
                    episodic_memory_config,
                    description,
                    metadata,
                )
                episodic_memory_params = await episodic_memory_params_from_config(
                    episodic_memory_config,
                    self._resource_manager,
                )
                instance = EpisodicMemory(episodic_memory_params)
                await self._instance_cache.add(session_key, instance)
        try:
            yield instance
        finally:
            if instance is not None:
                async with self._lock:
                    self._instance_cache.put(session_key)

    async def delete_episodic_session(self, session_key: str) -> None:
        """
        Delete an episodic memory instance and its associated data.

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
                session_info = await self._session_data_manager.get_session_info(
                    session_key
                )
                if session_info is None:
                    raise RuntimeError(f"Session {session_key} is None")

                params = await episodic_memory_params_from_config(
                    session_info.episode_memory_conf,
                    self._resource_manager,
                )
                instance = EpisodicMemory(params)
            await instance.delete_session_episodes()
            await instance.close()
            await self._session_data_manager.delete_session(session_key)

    async def get_episodic_memory_keys(
        self,
        filters: dict[str, object] | None,
    ) -> list[str]:
        """
        Retrieve a list of all available episodic memory session keys.

        Args:
            filters: Optional metadata filters for narrowing sessions.

        Returns:
            A list of session keys.

        """
        return await self._session_data_manager.get_sessions(filters)

    async def get_session_configuration(
        self,
        session_key: str,
    ) -> SessionDataManager.SessionInfo | None:
        """Retrieve the configuration, description, and metadata for a given session."""
        return await self._session_data_manager.get_session_info(session_key)

    async def close_session(self, session_key: str) -> None:
        """
        Close an idle episodic memory instance and its associated data.

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

    async def close(self) -> None:
        """Close all open episodic memory instances and the session storage."""
        async with self._lock:
            if self._closed:
                return
            await self._instance_cache.clear_cache()
            await self._session_data_manager.close()
            self._instance_cache.clear()
            self._closed = True

        if hasattr(self, "_check_instance_task"):
            await self._check_instance_task
