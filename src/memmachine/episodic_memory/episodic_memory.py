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
import logging
import uuid
from datetime import datetime
import time
from typing import cast

from memmachine.configuration.episodic_config import EpisodicMemoryParams


from .data_types import ContentType, Episode, EpisodeType
from .long_term_memory.long_term_memory import LongTermMemory
from .short_term_memory.short_term_memory import ShortTermMemory

logger = logging.getLogger(__name__)


class EpisodicMemory:
    # pylint: disable=too-many-instance-attributes
    """
    Represents a single, isolated memory instance for a specific session.

    This class orchestrates the interaction between short-term (session)
    memory and long-term (declarative) memory. It manages the lifecycle of
    the memory, handles adding new information (episodes), and provides
    methods to retrieve contextual information for queries.

    Each instance is tied to a unique session key
    """

    def __init__(
            self,
            param: EpisodicMemoryParams,
            session_memory: ShortTermMemory | None = None,
            long_term_memory: LongTermMemory | None = None
        ):
        # pylint: disable=too-many-instance-attributes
        """
        Initializes a EpisodicMemory instance.

        Args:
            manager: The EpisodicMemoryManager that created this instance.
            param: The paraters required to initialize the episodic memory
        """
        self._session_key = param.session_key
        self._closed = False
        self._short_term_memory: ShortTermMemory | None = session_memory
        self._long_term_memory: LongTermMemory | None = long_term_memory
        metrics_manager = param.metrics_factory
        self._enabled = param.enabled
        if not self._enabled:
            return
        if self._short_term_memory is None and self._long_term_memory is None:
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

    @classmethod
    async def create(cls, param: EpisodicMemoryParams) -> 'EpisodicMemory':
        session_memory: ShortTermMemory | None = None
        if param.short_term_memory and param.short_term_memory.enabled:
            session_memory = await ShortTermMemory.create(param.short_term_memory)
        long_term_memory: LongTermMemory | None = None
        if param.long_term_memory and param.long_term_memory.enabled:
            long_term_memory = LongTermMemory(param.long_term_memory)
        return EpisodicMemory(param, session_memory, long_term_memory)

    @property
    def short_term_memory(self) -> ShortTermMemory | None:
        """
        Get the short-term memory of the episodic memory instance
        Returns:
            The short-term memory of the episodic memory instance.
        """
        return self._short_term_memory

    @short_term_memory.setter
    def short_term_memory(self, value: ShortTermMemory | None):
        """
        Set the short-term memory of the episodic memory instance
        This makes the short term memory can be injected
        Args:
            value: The new short-term memory of the episodic memory instance.
        """
        self._short_term_memory = value

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

    @property
    def session_key(self) -> str:
        """
        Get the session key of the episodic memory instance
        Returns:
            The session key of the episodic memory instance.
        """
        return self._session_key

    async def add_memory_episode(
        self,
        episode: Episode
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
        if not self._enabled:
            return
        start_time = time.monotonic_ns()

        if self._closed:
            raise RuntimeError(f"Memory is closed {self._session_key}")
        # Add the episode to both memory stores concurrently
        tasks = []
        if self._short_term_memory:
            tasks.append(self._short_term_memory.add_episode(episode))
        if self._long_term_memory:
            tasks.append(self._long_term_memory.add_episode(episode))
        await asyncio.gather(
            *tasks,
        )
        end_time = time.monotonic_ns()
        delta = (end_time - start_time) / 1000000
        self._ingestion_latency_summary.observe(delta)
        self._ingestion_counter.increment()

    async def close(self):
        """
        Decrements the reference count and closes the instance if it reaches
        zero.

        When the reference count is zero, it closes the underlying memory
        stores and notifies the manager to remove this instance from its
        registry.
        """
        self._closed = True
        if not self._enabled:
            return
        tasks = []
        if self._short_term_memory:
            tasks.append(self._short_term_memory.close())
        if self._long_term_memory:
            tasks.append(self._long_term_memory.close())
        await asyncio.gather(*tasks)
        return



    async def delete_episode(self, uuid: uuid.UUID):
        """Delete one episode by uuid"""
        if not self._enabled:
            return
        tasks = []
        if self._short_term_memory:
            tasks.append(self._short_term_memory.delete_episode(uuid))
        if self._long_term_memory:
            tasks.append(self._long_term_memory.delete_episode(uuid))
        await asyncio.gather(*tasks)
        return

    async def delete_data(self):
        """
        Deletes all data from both session and declarative memory for this
        context.
        This is a destructive operation.
        """
        if not self._enabled:
            return
        tasks = []
        if self._short_term_memory:
            tasks.append(self._short_term_memory.clear_memory())
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
        if not self._enabled:
            return [], [], []
        start_time = time.monotonic_ns()
        search_limit = limit if limit is not None else 20
        if property_filter is None:
            property_filter = {}

        if self._short_term_memory is None:
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
            session_result = await self._short_term_memory.get_session_memory_context(
                query, limit=search_limit, filter=property_filter
            )
            long_episode = []
            short_episode, short_summary = session_result
        else:
            # Concurrently search both memory stores
            session_result, long_episode = await asyncio.gather(
                self._short_term_memory.get_session_memory_context(
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

        end_time = time.monotonic_ns()
        delta = (end_time - start_time) / 1000000
        self._query_latency_summary.observe(delta)
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
