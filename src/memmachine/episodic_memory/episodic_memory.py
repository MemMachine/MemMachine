"""
Defines the core memory instance for a specific conversational context.

This module provides the `EpisodicMemory` class, which acts as the primary
orchestrator for an individual memory session. It integrates short-term
(session) and long-term (declarative) memory stores to provide a unified
interface for adding and retrieving conversational data.

Key responsibilities include:
- Managing the lifecycle of the memory instance through reference counting.
- Adding new conversational `Episode` objects to both session and declarative
  memory.
- Retrieving relevant context for a query by searching both memory types.
- Interacting with a language model for memory-related tasks.
- Each instance is uniquely identified by a `MemoryContext` and managed by the
  `EpisodicMemoryManager`.
"""

import asyncio
import copy
import logging
import uuid
from datetime import datetime
from typing import cast

from memmachine.common.language_model.language_model_builder import (
    LanguageModelBuilder,
)
from memmachine.common.metrics_factory.metrics_factory_builder import (
    MetricsFactoryBuilder,
)

from .data_types import ContentType, Episode, MemoryContext
from .long_term_memory.long_term_memory import LongTermMemory
from .short_term_memory.session_memory import SessionMemory

logger = logging.getLogger(__name__)


class EpisodicMemory:
    # pylint: disable=too-many-instance-attributes
    """
    Represents a single, isolated memory instance for a specific context.

    This class orchestrates the interaction between short-term (session)
    memory and long-term (declarative) memory. It manages the lifecycle of
    the memory, handles adding new information (episodes), and provides
    methods to retrieve contextual information for queries.

    Each instance is tied to a unique `MemoryContext` (defined by group, agent,
    user, and session IDs) and is managed by a central
    `EpisodicMemoryManager`.
    """

    def __init__(self, manager, config: dict, memory_context: MemoryContext):
        # pylint: disable=too-many-instance-attributes
        """
        Initializes a EpisodicMemory instance.

        Args:
            manager: The EpisodicMemoryManager that created this instance.
            config: A dictionary containing the configuration for this memory
                    instance.
            memory_context: The unique context for this memory instance.
        """
        self._memory_context = memory_context
        self._manager = manager  # The manager that created this instance
        self._lock = asyncio.Lock()  # Lock for thread-safe operations

        model_config = config.get("model")
        short_config = config.get("sessionmemory", {})
        long_term_config = config.get("long_term_memory", {})

        self._ref_count = 1  # For reference counting to manage lifecycle
        self._session_memory: SessionMemory | None = None
        self._long_term_memory: LongTermMemory | None = None
        metrics_manager = MetricsFactoryBuilder.build("prometheus", {}, {})

        if len(short_config) > 0 and short_config.get("enabled") != "false":
            model_name = short_config.get("model_name")
            if model_name is None or len(model_name) < 1:
                raise ValueError("Invalid model name")

            if model_config is None or model_config.get(model_name) is None:
                raise ValueError("Invalid model configuration")

            model_config = copy.deepcopy(model_config.get(model_name))
            """
            only support prometheus now.
            TODO: support different metrics and make it configurable
            """
            model_config["metrics_factory_id"] = "prometheus"
            model_vendor = model_config.pop("model_vendor")
            metrics_injection = {}
            metrics_injection["prometheus"] = metrics_manager

            llm_model = LanguageModelBuilder.build(
                model_vendor,
                model_config,
                metrics_injection,
            )

            # Initialize short-term session memory
            self._session_memory = SessionMemory(
                llm_model,
                config.get("prompts", {}).get("episode_summary_prompt_system"),
                config.get("prompts", {}).get("episode_summary_prompt_user"),
                short_config.get("message_capacity", 1000),
                short_config.get("max_message_length", 128000),
                short_config.get("max_token_num", 65536),
                self._memory_context,
            )

        if len(long_term_config) > 0 and long_term_config.get("enabled") != "false":
            # Initialize long-term declarative memory
            self._long_term_memory = LongTermMemory(config, self._memory_context)
        if self._session_memory is None and self._long_term_memory is None:
            raise ValueError("No memory is configured")

        # Initialize metrics
        self._ingestion_latency_summary = metrics_manager.get_summary(
            "Ingestion_latency", "Latency of Episode ingestion in milliseconds"
        )
        self._query_latency_summary = metrics_manager.get_summary(
            "query_latency", "Latency of query processing in milliseconds"
        )
        self._ingestion_counter = metrics_manager.get_counter(
            "Ingestion_count", "Count of Episode ingestion"
        )
        self._query_counter = metrics_manager.get_counter(
            "query_count", "Count of query processing"
        )

    @property
    def short_term_memory(self) -> SessionMemory | None:
        """
        Get the short-term memory of the episodic memory instance
        Returns:
            The short-term memory of the episodic memory instance.
        """
        return self._session_memory

    @short_term_memory.setter
    def short_term_memory(self, value: SessionMemory | None):
        """
        Set the short-term memory of the episodic memory instance
        This makes the short term memory can be injected
        Args:
            value: The new short-term memory of the episodic memory instance.
        """
        self._session_memory = value

    @property
    def long_term_memory(self) -> LongTermMemory | None:
        """
        Get the long-term memory of the episodic memory instance
        Returns:
            The long-term memory of the episodic memory instance.
        """
        return self._long_term_memory

    @long_term_memory.setter
    def long_term_memory(self, value: LongTermMemory | None):
        """
        Set the long-term memory of the episodic memory instance
        This makes the long term memory can be injected
        Args:
            value: The new long-term memory of the episodic memory instance.
        """
        self._long_term_memory = value

    def get_memory_context(self) -> MemoryContext:
        """
        Get the memory context of the episodic memory instance
        Returns:
            The memory context of the episodic memory instance.
        """
        return self._memory_context

    def get_reference_count(self) -> int:
        """
        Get the reference count of the episodic memory instance
        Returns:
            The reference count of the episodic memory instance.
        """
        return self._ref_count

    async def reference(self) -> bool:
        """
        Increments the reference count for this instance.

        Used by the manager to track how many clients are actively using this
        memory instance.

        Returns:
            True if the reference was successfully added, False if the instance
            is already closed.
        """
        async with self._lock:
            if self._ref_count <= 0:
                return False
            self._ref_count += 1
            return True

    async def add_memory_episode(
        self,
        producer: str,
        produced_for: str,
        episode_content: str | list[float],
        episode_type: str,
        content_type: ContentType,
        timestamp: datetime | None = None,
        metadata: dict | None = None,
    ):
        # pylint: disable=too-many-arguments
        # pylint: disable=too-many-positional-arguments
        """
        Adds a new memory episode to both session and declarative memory.

        Validates that the producer and recipient of the episode are part of
        the current memory context.

        Args:
            producer: The ID of the user or agent that created the episode.
            produced_for: The ID of the intended recipient.
            episode_content: The content of the episode (string or vector).
            episode_type: The type of the episode (e.g., 'message', 'thought').
            content_type: The type of the content (e.g., STRING).
            timestamp: The timestamp of the episode. Defaults to now().
            metadata: Optional dictionary of user-defined metadata.

        Returns:
            True if the episode was added successfully, False otherwise.
        """
        # Validate that the producer and recipient are part of this memory
        # context
        if (
            producer not in self._memory_context.user_id
            and producer not in self._memory_context.agent_id
        ):
            logger.error("The producer %s does not belong to the session", producer)
            raise ValueError(f"The producer {producer} does not belong to the session")

        if (
            produced_for not in self._memory_context.user_id
            and produced_for not in self._memory_context.agent_id
        ):
            logger.error(
                "The produced_for %s does not belong to the session",
                produced_for,
            )
            raise ValueError(
                f"""The produced_for {produced_for} does not belong to
                 the session"""
            )

        start_time = datetime.now()

        # Create a new Episode object
        episode = Episode(
            uuid=uuid.uuid4(),
            episode_type=episode_type,
            content_type=content_type,
            content=episode_content,
            timestamp=timestamp if timestamp else datetime.now(),
            group_id=self._memory_context.group_id,
            session_id=self._memory_context.session_id,
            producer_id=producer,
            produced_for_id=produced_for,
            user_metadata=metadata,
        )

        # Add the episode to both memory stores concurrently
        tasks = []
        if self._session_memory:
            tasks.append(self._session_memory.add_episode(episode))
        if self._long_term_memory:
            tasks.append(self._long_term_memory.add_episode(episode))
        await asyncio.gather(
            *tasks,
        )
        end_time = datetime.now()
        delta = end_time - start_time
        self._ingestion_latency_summary.observe(
            delta.total_seconds() * 1000 + delta.microseconds / 1000
        )
        self._ingestion_counter.increment()
        return True

    async def close(self):
        """
        Decrements the reference count and closes the instance if it reaches
        zero.

        When the reference count is zero, it closes the underlying memory
        stores and notifies the manager to remove this instance from its
        registry.
        """
        async with self._lock:
            self._ref_count -= 1
            if self._ref_count > 0:
                return

            # If no more references, proceed with closing
            logger.info("Closing context memory: %s", str(self._memory_context))
            tasks = []
            if self._session_memory:
                tasks.append(self._session_memory.close())
            await asyncio.gather(
                *tasks,
            )
            await self._manager.delete_context_memory(self._memory_context)
            return

    async def delete_data(self):
        """
        Deletes all data from both session and declarative memory for this
        context.
        This is a destructive operation.
        """
        async with self._lock:
            tasks = []
            if self._session_memory:
                tasks.append(self._session_memory.clear_memory())
            if self._long_term_memory:
                tasks.append(self._long_term_memory.forget_session())
            await asyncio.gather(*tasks)
            return

    async def query_memory(
        self,
        query: str,
        limit: int | None = None,
        property_filter: dict | None = None,
    ) -> tuple[list[Episode], list[Episode], list[str]]:
        """
        Retrieves relevant context for a given query from all memory stores.

        It fetches episodes from both short-term (session) and long-term
        (declarative) memory, deduplicates them, and returns them along with
        any available summary.

        Args:
            query: The query string to find context for.
            limit: The maximum number of episodes to return. The limit is
                   applied to both short and long term memories. The default
                   value is 20.
            filter: A dictionary of properties to filter the search in
                    declarative memory.

        Returns:
            A tuple containing a list of short term memory Episode objects,
            a list of long term memory Episode objects, and a
            list of summary strings.
        """

        start_time = datetime.now()
        search_limit = limit if limit is not None else 20
        if property_filter is None:
            property_filter = {}
        # By default, always allow cross session search
        property_filter["group_id"] = self._memory_context.group_id

        async with self._lock:
            if self._session_memory is None:
                short_episode: list[Episode] = []
                short_summary = ""
                long_episode = await cast(
                    LongTermMemory, self._long_term_memory
                ).search(
                    query,
                    search_limit,
                    property_filter,
                )
            elif self._long_term_memory is None:
                session_result = await self._session_memory.get_session_memory_context(
                    query, limit=search_limit
                )
                long_episode = []
                short_episode, short_summary = session_result
            else:
                # Concurrently search both memory stores
                session_result, long_episode = await asyncio.gather(
                    self._session_memory.get_session_memory_context(
                        query, limit=search_limit
                    ),
                    self._long_term_memory.search(query, search_limit, property_filter),
                )
                short_episode, short_summary = session_result

        # Deduplicate episodes from both memory stores, prioritizing
        # short-term memory
        uuid_set = {episode.uuid for episode in short_episode}

        unique_long_episodes = []
        for episode in long_episode:
            if episode.uuid not in uuid_set:
                uuid_set.add(episode.uuid)
                unique_long_episodes.append(episode)

        end_time = datetime.now()
        delta = end_time - start_time
        self._query_latency_summary.observe(
            delta.total_seconds() * 1000 + delta.microseconds / 1000
        )
        self._query_counter.increment()
        return short_episode, unique_long_episodes, [short_summary]

    async def formalize_query_with_context(
        self,
        query: str,
        limit: int | None = None,
        property_filter: dict | None = None,
    ) -> str:
        """
        Constructs a finalized query string that includes context from memory.

        The context (summary and recent episodes) is prepended to the original
        query, formatted with XML-like tags for the language model to parse.

        Args:
            query: The original query string.
            limit: The maximum number of episodes to include in the context.
            filter: A dictionary of properties to filter the search.

        Returns:
            A new query string enriched with context.
        """
        short_memory, long_memory, summary = await self.query_memory(
            query, limit, property_filter
        )
        episodes = sorted(short_memory + long_memory, key=lambda x: x.timestamp)

        finalized_query = ""
        # Add summary if it exists
        if summary and len(summary) > 0:
            total_summary = ""
            for summ in summary:
                total_summary = total_summary + summ + "\n"
            finalized_query += "<Summary>\n"
            finalized_query += total_summary
            finalized_query += "\n</Summary>\n"

        # Add episodes if they exist
        if episodes and len(episodes) > 0:
            finalized_query += "<Episodes>\n"
            for episode in episodes:
                # Ensure content is a string before concatenating
                if isinstance(episode.content, str):
                    finalized_query += episode.content
                    finalized_query += "\n"
            finalized_query += "</Episodes>\n"

        # Append the original query
        finalized_query += f"<Query>\n{query}\n</Query>"

        return finalized_query


class AsyncEpisodicMemory:
    """
    Asynchronous context manager for EpisodicMemory instances.

    This class provides an `async with` interface for `EpisodicMemory` objects,
    ensuring that `reference()` is called upon entry and `close()` is called
    upon exit, handling the lifecycle management automatically.
    """

    def __init__(self, episodic_memory_instance: EpisodicMemory):
        """
        Initializes the AsyncEpisodicMemory context manager.

        Args:
            episodic_memory_instance: The EpisodicMemory instance to manage.
        """
        self.episodic_memory_instance = episodic_memory_instance

    async def __aenter__(self) -> EpisodicMemory:
        """
        Enters the asynchronous context.

        Returns:
            The EpisodicMemory instance.

        """
        return self.episodic_memory_instance

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """
        Exits the asynchronous context.

        Decrements the reference count of the managed EpisodicMemory instance,
        triggering its closure if the count reaches zero.
        """
        await self.episodic_memory_instance.close()
