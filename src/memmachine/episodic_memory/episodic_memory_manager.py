"""
Manages the lifecycle of EpisodicMemory instances.

This module provides the `EpisodicMemoryMemoryManager`, a singleton class that
acts as a central factory and registry for `EpisodicMemory` objects. It
is responsible for:

- Loading and merging configurations from files.
- Creating, retrieving, and managing session-specific memory instances based
  on group and session IDs.
- Ensuring that each unique conversational session has a dedicated memory
  instance.
- Interacting with a `SessionManager` to persist and retrieve session
  information.
"""

import asyncio
import logging
from contextlib import asynccontextmanager
from typing import Any

import yaml

from .data_types import MemoryContext
from .episodic_memory import EpisodicMemory
from .session_manager.data_types import SessionInfo
from .session_manager.session_manager import SessionManager

logger = logging.getLogger(__name__)


class EpisodicMemoryManager:
    """
    Manages the creation and lifecycle of EpisodicMemory instances.

    This class acts as a factory and a central registry for all
    session-specific memories (EpisodicMemory). It ensures that each
    unique session (defined by group and session IDs) has its
    own dedicated EpisodicMemory.

    It follows a singleton pattern, ensuring only one manager exists. It
    handles loading configurations from environment variables and provides
    a way to safely create, access, and close these memory instances.
    """

    _instance = None

    def __new__(cls, config: dict[str, Any] | None = None) -> Self:
        # Enforce singleton pattern
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            return cls._instance

        if config is not None and config is not cls._instance._config:
            raise RuntimeError(
                "EpisodicMemoryManager is a singleton; cannot reinitialize with new config."
            )

        return cls._instance

    def __init__(self, config: dict[str, Any]):
        """
        Initializes the EpisodicMemoryManager.

        Args:
            config: A configuration dictionary containing all necessary
                    settings for models, storage, and memory parameters.
        """
        if self._instance is not None:
            return

        self._config = config

        self._resources = config.get("resources", {})

        self._episodic_memories: dict[MemoryContext, EpisodicMemory] = {}
        self._episodic_memories_lock = asyncio.Lock()

        self._base_episodic_memory_config = config.get(
            "base_episodic_memory_config", {}
        )

        session_manager_config = config.get("session_manager_config", {})
        self._session_manager = SessionManager(session_manager_config)

    def _merge_episodic_memory_configs(
        self,
        base_episodic_memory_config: dict[str, Any] = {},
        override_episodic_memory_config: dict[str, Any] = {},
    ) -> dict[str, Any]:
        """Recursively merges two dictionaries. `override_config` values take
        precedence."""
        result = base_episodic_memory_config.copy()
        for k, v in override_episodic_memory_config.items():
            if k in result and isinstance(result[k], dict) and isinstance(v, dict):
                result[k] = self._merge_episodic_memory_configs(result[k], v)
            else:
                result[k] = v
        return result

    async def close_episodic_memory_instance(
        self,
        group_id: str,
        session_id: str,
    ) -> bool:
        """
        Closes an EpisodicMemory instance for a specific context.

        Args:
            group_id: The identifier for the group
            agent_id: The identifier for the list of agent context
            user_id: The identifier for the list of user context
            session_id: The identifier for the session context

        Returns:
            True if the instance was successfully closed, False otherwise.
        """
        # Validate that the context is sufficiently defined.
        if group_id is None or len(group_id) < 1:
            raise ValueError("Invalid group id")
        if session_id is None or len(session_id) < 1:
            raise ValueError("Invalid session id")

        inst = None
        context = MemoryContext(
            group_id=group_id, agent_id=set(), user_id=set(), session_id=session_id
        )
        if context in self._context_memory:
            inst = self._context_memory[context]
        if inst is None:
            return False
        await inst.close()
        return True

    async def create_group(
        self, group_id: str, agent_ids: list[str] | None, user_ids: list[str] | None
    ):
        """
        Creates a new group.
        Args:
            group_id: The ID of the group
            agent_ids: A list of agent IDs of the group
            user_ids: A lit of user IDs of the group
        """
        if len(group_id) < 1:
            raise ValueError("Invalid group ID")
        agent_ids = [] if agent_ids is None else agent_ids
        user_ids = [] if user_ids is None else user_ids
        if len(agent_ids) < 1 and len(user_ids) < 1:
            raise ValueError("The group must have at least one user ID or agent ID")
        async with self._lock:
            self._session_manager.create_new_group(group_id, agent_ids, user_ids)

    async def create_episodic_memory_instance(
        self, group_id: str, session_id: str, configuration: dict | None = None
    ) -> EpisodicMemory:
        """
        Creates EpisodicMemory for a new session.
        If the group does not exist, this function fails.
        If the session already exists, this function fails.

        Args:
            group_id (str): The ID of the group for this session.
            session_id (str): The unique identifier for the session.
            configuration (dict | None): A dictionary for session
            configuration.

        Returns:
            New EpisodicMemory instance.
        """
        async with self._lock:
            group = self._session_manager.retrieve_group(group_id)
            if group is None:
                raise ValueError(f"""Failed to get the group {group_id}""")
            configuration = {} if configuration is None else configuration
            configuration = self._merge_configs(self._memory_config, configuration)
            session = self._session_manager.create_session(
                group_id, session_id, configuration
            )
            context = MemoryContext(
                group_id=group_id,
                agent_id=set(group.agent_list),
                user_id=set(group.user_list),
                session_id=session_id,
            )
            final_config = self._merge_configs(
                self._memory_config, session.configuration or {}
            )
            memory_instance = EpisodicMemory(self, final_config, context)
            self._context_memory[context] = memory_instance
            await memory_instance.reference()
            return memory_instance

    async def open_episodic_memory_instance(
        self, group_id: str, session_id: str
    ) -> EpisodicMemory:
        """
        Opens an EpisodicMemory instance for a specific group and session.
        Args:
            group_id: The identifier for the group context.
            session_id: The identifier for the session context.
        Returns:
            The EpisodicMemory instance for the specified group and session.
        """
        context = MemoryContext(
            group_id=group_id, agent_id=set(), user_id=set(), session_id=session_id
        )
        async with self._lock:
            if context in self._context_memory:
                inst = self._context_memory[context]
                await inst.reference()
                return inst

            session_info = self._session_manager.open_session(group_id, session_id)
            memory_instance = EpisodicMemory(self, session_info.configuration, context)
            self._context_memory[context] = memory_instance
            await memory_instance.reference()
            return memory_instance

    @asynccontextmanager
    async def async_open_episodic_memory_instance(
        self,
        group_id: str,
        session_id: str,
    ):
        """
        Retrieves an AsyncEpisodicMemory instance for a specific
        context.
        Args:
            group_id: The identifier for the group context.
            session_id: The identifier for the session context.
        """
        inst = await self.open_episodic_memory_instance(group_id, session_id)
        yield inst
        if inst is not None:
            await inst.close()

    @asynccontextmanager
    async def async_create_episodic_memory_instance(
        self, group_id: str, session_id: str, configuration: dict | None = None
    ):
        """
        Creates an AsyncEpisodicMemory instance for a specific
        context.
        Args:
            group_id: The identifier for the group context.
            session_id: The identifier for the session context.
            configuration: The session specific configuration
        """
        inst = await self.create_episodic_memory_instance(
            group_id, session_id, configuration
        )
        yield inst
        if inst is not None:
            await inst.close()

    async def get_episodic_memory_instance(
        self,
        group_id: str,
        agent_id: list[str] | None = None,
        user_id: list[str] | None = None,
        session_id: str = "",
        configuration: dict | None = None,
    ) -> EpisodicMemory | None:
        """
        Retrieves or creates a EpisodicMemory instance for a specific context.

        This method ensures that only one EpisodicMemory object exists for
        each unique combination of group, agent, user, and session IDs. It is
        thread-safe.

        Args:
            group_id: The identifier for the group context.
            session_id: The identifier for the session context.
            configuration: session specific configuration.

        Returns:
            The EpisodicMemory instance for the specified context.
        """
        episodic_memory_config = {}
        if config_path is not None:
            with open(config_path) as config_file:
                episodic_memory_config = yaml.safe_load(config_file)

        # Validate that the context is sufficiently defined.
        if group_id is None and len(user_ids) < 1 and len(agent_ids) < 1:
            raise ValueError("Invalid context")
        if session_id is None:
            raise ValueError("Invalid session id")
        if len(user_ids) < 1:
            user_ids = agent_ids

        # If group_id is not provided, create a composite ID from user IDs.
        if group_id is None or len(group_id) < 1:
            group_id = EpisodicMemoryManager._group_id_from_users_and_agents(
                user_ids, agent_ids
            )

        # Create the unique memory context object.
        memory_context = MemoryContext(
            group_id=group_id,
            agent_id=set(agent_ids),
            user_id=set(user_ids),
            session_id=session_id,
        )

        async with self._episodic_memories_lock:
            # If an instance for this context already exists, increment its
            # reference count and return it.
            if memory_context in self._episodic_memories:
                print("Use existing session")
                instance = self._episodic_memories[memory_context]
                get_it = await instance.reference()
                if get_it:
                    return instance
                # The instance was closed between checking and referencing.
                logger.error("Failed get instance reference")
                return None
            print("Create new session")
            # If no instance exists, create a new one.
            info = self._session_manager.create_session_if_not_exist(
                group_id,
                agent_ids,
                user_ids,
                session_id,
                episodic_memory_config,
            )

            # Merge the base config with the session-specific config.
            final_config = self._merge_episodic_memory_configs(
                self._base_episodic_memory_config, info.configuration or {}
            )

            # Create and store the new memory instance.
            memory_instance = EpisodicMemory(
                self, memory_context, final_config, self._resources
            )

            self._episodic_memories[memory_context] = memory_instance

            await memory_instance.reference()
            return memory_instance

    async def delete_episodic_memory(self, context: MemoryContext):
        """
        Removes a specific EpisodicMemory instance from the manager's registry.

        This method should be only called when the EpisodicMemory instance is
        closed and there are no more active references.

        Args:
            context: The memory context of the instance to delete.
        """
        async with self._episodic_memories_lock:
            if context in self._episodic_memories:
                logger.info("Deleting context memory %s\n", context)
                del self._episodic_memories[context]
            else:
                logger.info("Context memory %s does not exist\n", context)

    def shut_down(self):
        """
        Close all sessions and clean up resources.
        """
        for episodic_memory in self._episodic_memories.values():
            episodic_memory.close()
        del self._session_manager
        self._session_manager = None

    def get_all_sessions(self) -> list[SessionInfo]:
        """
        Retrieves all sessions from the session manager.

        Returns:
            A list of SessionInfo objects for all stored sessions.
        """
        return self._session_manager.get_all_sessions()

    def get_user_sessions(self, user_id: str) -> list[SessionInfo]:
        """
        Retrieves all sessions associated with a specific user ID.

        Args:
            user_id: The ID of the user.

        Returns:
            A list of SessionInfo objects for the given user.
        """
        return self._session_manager.get_session_by_user(user_id)

    def get_agent_sessions(self, agent_id: str) -> list[SessionInfo]:
        """
        Retrieves all sessions associated with a specific agent ID.

        Args:
            agent_id: The ID of the agent.

        Returns:
            A list of SessionInfo objects for the given agent.
        """
        return self._session_manager.get_session_by_agent(agent_id)

    def get_group_sessions(self, group_id: str) -> list[SessionInfo]:
        """
        Retrieves all sessions associated with a specific group ID.

        Args:
            group_id: The ID of the group.

        Returns:
            A list of SessionInfo objects for the given group.
        """
        return self._session_manager.get_session_by_group(group_id)

    def get_group_configuration(self, group_id: str) -> GroupConfiguration | None:
        """
        Retrieve one group information
        Args:
            group_id: The ID of the group
        Return:
            The group information
        """
        return self._session_manager.retrieve_group(group_id)
