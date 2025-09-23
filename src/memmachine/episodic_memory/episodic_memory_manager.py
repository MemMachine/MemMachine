"""
Manages the lifecycle and configuration of memory instances.

This module provides the `EpisodicMemoryMemoryManager`, a singleton class that
acts as a central factory and registry for `EpisodicMemory` objects. It
is responsible for:

- Loading and merging configurations from files.
- Creating, retrieving, and managing context-specific memory instances based
 on group, agent, user, and session IDs.
- Ensuring that each unique conversational context has a dedicated memory
  instance.
- Interacting with a `SessionManager` to persist and retrieve session
  information.
"""

import asyncio
import logging
from typing import Any

import yaml

from .data_types import MemoryContext, SessionInfo
from .episodic_memory import EpisodicMemory
from .prompt.summary_prompt import (
    episode_summary_system_prompt,
    episode_summary_user_prompt,
)
from .session_manager.session_manager import SessionManager
import os
logger = logging.getLogger(__name__)
from memmachine.tracing import get_tracer, init_tracer
    

class EpisodicMemoryManager:
    """
    Manages the creation and lifecycle of ContextMemory instances.

    This class acts as a factory and a central registry for all
    context-specific memories (EpisodicMemory). It ensures that each
    unique context (defined by group, agent, user, and session IDs) has its
    own dedicated EpisodicMemory.

    It follows a singleton pattern, ensuring only one manager exists. It
    handles loading configurations from environment variables and provides
    a way to safely create, access, and close these memory instances.
    """

    _instance = None
    _tracer = None

    def __init__(self, config: dict):
        """
        Initializes the MemoryManager.

        Note: This constructor should not be called directly. Use the factory
        method `create_memory_manager` instead.

        Args:
            config: A configuration dictionary containing all necessary
                    settings for models, storage, and memory parameters.
        """
        self._memory_config = config
        # A dictionary to hold active memory instances, keyed by their context
        # hash string.
        self._context_memory: dict[str, EpisodicMemory] = {}
        # A lock to ensure thread-safe access to the _context_memory
        # dictionary.
        self._lock = asyncio.Lock()
        # Initialize the session manager for handling session data persistence.
        jaeger_host = os.getenv("JAEGER_HOST", "jaeger")   
        _tracer = init_tracer(jaeger_host=jaeger_host)
        sessiondb = config.get("sessiondb", {})
        self._session_manager = SessionManager(sessiondb)

    @classmethod
    def create_episodic_memory_manager(cls, config_path: str):
        """
        Factory class method to create a MemoryManager instance from
        environment variables.

        This method implements the singleton pattern. It reads all necessary
        configurations for models, storage, and prompts, sets up logging,
        and creates a new instance of MemoryManager if one does not already
        exist.
        Args:
            config_path: The path to the configuration file.

        Returns:
            A new instance of MemoryManager.
        """

        if cls._instance is not None:
            return cls._instance

        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)

        def config_to_lowercase(data: Any) -> Any:
            """Recursively converts all dictionary keys in a nested structure
            to lowercase."""
            if isinstance(data, dict):
                return {
                    k.lower(): config_to_lowercase(v) for k, v in data.items()
                }
            if isinstance(data, list):
                return [config_to_lowercase(i) for i in data]
            return data

        config = config_to_lowercase(config)

        # Configure logging based on the loaded configuration.
        log_path = config.get("logging", {}).get("path", "/tmp/MemMachine.log")
        log_level = config.get("logging", {}).get("level", "info")
        logging.basicConfig(filename=log_path, level=log_level.upper())

        # Load custom prompts from files if specified in the config,
        # overriding defaults.
        def load_prompt(
            prompt_config: dict, key: str, default_value: str
        ) -> str:
            """
            Helper function to load a prompt from a file path specified in the
            config.

            Args:
                prompt_config: The dictionary containing prompt file paths.
                key: The key for the specific prompt to load.
                default_value: The default prompt content to use if the file
                is not specified or found.

            Returns:
                The content of the prompt.
            """
            custom_prompt_path = prompt_config.get(key)
            if custom_prompt_path:
                with open(custom_prompt_path, "r", encoding="utf-8") as f:
                    return f.read()
            return default_value

        prompt_config = config.get("prompts", {})

        prompt_config["episode_summary_prompt_system"] = load_prompt(
            prompt_config,
            "episode_summary_prompt_system",
            episode_summary_system_prompt,
        )
        prompt_config["episode_summary_prompt_user"] = load_prompt(
            prompt_config,
            "episode_summary_prompt_user",
            episode_summary_user_prompt,
        )

        config["prompts"] = prompt_config

        manager = cls(config)
        cls._instance = manager
        return manager

    def _merge_configs(self, base_config: dict, override_config: dict) -> dict:
        """Recursively merges two dictionaries. `override_config` values take
        precedence."""
        result = base_config.copy()
        for k, v in override_config.items():
            if (
                k in result
                and isinstance(result[k], dict)
                and isinstance(v, dict)
            ):
                result[k] = self._merge_configs(result[k], v)
            else:
                result[k] = v
        return result

    async def close_episodic_memory_instance(
        self,
        group_id: str | None = None,
        agent_id: list[str] | None = None,
        user_id: list[str] | None = None,
        session_id: str = "",
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
        inst = await self.get_episodic_memory_instance(
            group_id, agent_id, user_id, session_id
        )
        if inst is None:
            return False
        await inst.close()
        await inst.close()
        return True

    async def get_episodic_memory_instance(
        self,
        group_id: str | None = None,
        agent_id: list[str] | None = None,
        user_id: list[str] | None = None,
        session_id: str = "",
        config_path: str | None = None,
        parent_span: None = None,
    ) -> EpisodicMemory | None:
        # pylint: disable=too-many-arguments
        # pylint: disable=too-many-positional-arguments
        """
        Retrieves or creates a EpisodicMemory instance for a specific context.

        This method ensures that only one EpisodicMemory object exists for
        each unique combination of group, agent, user, and session IDs. It is
        thread-safe.

        Args:
            group_id: The identifier for the group context.
            agent_id: The identifier for the list of agent context.
            user_id: The identifier for the list of user context.
            session_id: The identifier for the session context.
            config_path: Optional path to a session-specific config file to
            override defaults.

        Returns:
            The EpisodicMemory instance for the specified context.
        """
        tracer = get_tracer()
        with tracer.start_active_span('get_episodic_memory_instance', child_of=parent_span) as scope:
            span = scope.span
            span.set_tag("group_id", group_id)
            span.set_tag("agent_id", ",".join(agent_id) if agent_id else "")
            span.set_tag("user_id", ",".join(user_id) if user_id else "")
            span.set_tag("session_id", session_id)
            span.log_event("get_episodic_memory_instance.START")
            config = {}
            if config_path is not None:
                with open(config_path, "r", encoding="utf-8") as f:
                    config = yaml.safe_load(f)

            # Validate that the context is sufficiently defined.
            if user_id is None:
                user_id = []
            if agent_id is None:
                agent_id = []
            if group_id is None and len(user_id) < 1 and len(agent_id) < 1:
                raise ValueError("Invalid context")
            if session_id is None:
                raise ValueError("Invalid session id")
            if len(user_id) < 1:
                user_id = agent_id

            # If group_id is not provided, create a composite ID from user IDs.
            if group_id is None or len(group_id) < 1:
                group_id = "".join(user_id).join(agent_id)

            hash_str = (
                group_id
                + "_".join(agent_id)
                + "_".join(user_id)
                + "_"
                + session_id
            )

            # Create the unique memory context object.
            memory_context = MemoryContext(
                group_id=group_id,
                agent_id=set(agent_id),
                user_id=set(user_id),
                session_id=session_id,
                hash_str=hash_str,
            )

            async with self._lock:
                # If an instance for this context already exists, increment its
                # reference count and return it.
                if hash_str in self._context_memory:
                    print("Use existing session")
                    instance = self._context_memory[hash_str]
                    get_it = await instance.reference()
                    if get_it:
                        return instance
                    # The instance was closed between checking and referencing.
                    logger.error("Failed get instance reference")
                    return None
                print("Create new session")
                # If no instance exists, create a new one.
                info = self._session_manager.create_session_if_not_exist(
                    group_id, agent_id, user_id, session_id, config
                )

                # Merge the base config with the session-specific config.
                final_config = self._merge_configs(
                    self._memory_config, info.configuration or {}
                )

                # Create and store the new memory instance.
                memory_instance = EpisodicMemory(
                    self, final_config, memory_context
                )

                self._context_memory[hash_str] = memory_instance

                await memory_instance.reference()
                return memory_instance

    async def delete_context_memory(self, context: MemoryContext):
        """
        Removes a specific EpisodicMemory instance from the manager's registry.

        This method should be only called when the EpisodicMemory instance is
        closed and there are no more active references.

        Args:
            context: The memory context of the instance to delete.
        """

        async with self._lock:
            if context.hash_str in self._context_memory:
                logger.info("Deleting context memory %s\n", context.hash_str)
                del self._context_memory[context.hash_str]
            else:
                logger.info(
                    "Context memory %s does not exist\n", context.hash_str
                )

    def shut_down(self):
        """
        Close all sessions and clean up resources.
        """
        for keys in self._context_memory.keys():
            self._context_memory[keys].close()
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
