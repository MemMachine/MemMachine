"""
Defines the core memory instance for a specific conversational context.

This module provides the `EpisodicMemory` class, which acts as the primary
orchestrator for an individual memory session. It integrates short-term
(session) and long-term (declarative) memory stores to provide a unified
interface for adding and retrieving conversational data.

Key responsibilities include:
- Adding new conversational `Episode` objects to both session and declarative
  memory.
- Retrieving relevant context for a query by searching both memory types.
- Each instance is uniquely identified by a `MemoryContext` and managed by the
  `EpisodicMemoryManager`.
"""

import asyncio
import logging
import uuid
from datetime import datetime
from typing import Any

from memmachine.common.metrics_factory import MetricsFactory
from memmachine.common.resource_initializer import ResourceInitializer

from .data_types import ContentType, Episode, MemoryContext
from .episodic_memory_manager import EpisodicMemoryManager

logger = logging.getLogger(__name__)


class EpisodicMemory:
    """
    Represents a single, isolated memory instance for a specific context.

    This class orchestrates the interaction between short-term
    memory and long-term memory. It manages the lifecycle of
    the memory, handles adding new information (episodes), and provides
    methods to retrieve contextual information for queries.

    Each instance is tied to a unique `MemoryContext` (defined by group, agent,
    user, and session IDs) and is managed by a central
    `EpisodicMemoryManager`.
    """

    def __init__(
        self,
        manager: EpisodicMemoryManager,
        memory_context: MemoryContext,
        config: dict[str, Any],
        resources: dict[str, Any],
    ):
        """
        Initializes a EpisodicMemory instance.

        Args:
            manager: The EpisodicMemoryManager that created this instance.
            memory_context: The unique context for this memory instance.
            config: A dictionary containing the configuration for this memory
                    instance.
            resources: A dictionary of pre-initialized dependencies to be
                        injected into memory components.
        """
        self._manager = manager  # The manager that created this instance.
        self._ref_count = 1  # For reference counting to manage lifecycle

        self._memory_context = memory_context

        self._lock = asyncio.Lock()

        long_term_memory_config = config.get("long_term_memory", {})
        session_memory_config = config.get("session_memory", {})

        vector_graph_store_id = long_term_memory_config.get(
            "vector_graph_store_id"
        )
        embedder_id = long_term_memory_config.get("embedder_id")
        reranker_id = long_term_memory_config.get("reranker_id")

        # Configure derivative deriver.
        derivative_deriver_name = long_term_memory_config.get(
            "derivative_deriver", "sentence"
        )

        # Configure metadata derivative mutator and episode metadata template.
        metadata_prefix = long_term_memory_config.get(
            "metadata_prefix",
            "[$timestamp] $producer_id: ",
        )
        if not isinstance(metadata_prefix, str):
            raise TypeError("Metadata prefix must be a string")

        derivative_metadata_template = f"{metadata_prefix}$content"
        episode_metadata_template = f"{metadata_prefix}$content"

        derivation_workflow_definition = {
            "related_episode_postulator_id": (
                "_null_related_episode_postulator"
            ),
            "derivative_derivation_workflows": [
                {
                    "derivative_deriver_id": ("_episode_derivative_deriver"),
                    "derivative_mutation_workflows": [
                        {
                            "derivative_mutator_id": (
                                "_metadata_derivative_mutator"
                            ),
                        },
                    ],
                },
            ],
        }

        long_term_memory_resource_definitions = {
            "_long_term_memory": {
                "type": "long_term_memory",
                "name": "long_term_memory",
                "config": {
                    "declarative_memory_id": "_declarative_memory",
                },
            },
            "_declarative_memory": {
                "type": "declarative_memory",
                "name": "declarative_memory",
                "config": {
                    "vector_graph_store_id": vector_graph_store_id,
                    "embedder_id": embedder_id,
                    "reranker_id": reranker_id,
                    "query_derivative_deriver_id": "_query_derivative_deriver",
                    "related_episode_postulator_ids": [
                        "_previous_related_episode_postulator"
                    ],
                    "derivation_workflows": {
                        "default": [derivation_workflow_definition],
                    },
                    "episode_metadata_template": episode_metadata_template,
                },
            },
            "_previous_related_episode_postulator": {
                "type": "related_episode_postulator",
                "name": "previous",
                "config": {
                    "vector_graph_store_id": vector_graph_store_id,
                    "filterable_property_keys": [
                        "group_id",
                        "session_id",
                    ],
                },
            },
            "_query_derivative_deriver": {
                "type": "derivative_deriver",
                "name": "identity",
                "config": {},
            },
            "_metadata_derivative_mutator": {
                "type": "derivative_mutator",
                "name": "metadata",
                "config": {
                    "template": derivative_metadata_template,
                },
            },
            "_episode_derivative_deriver": {
                "type": "derivative_deriver",
                "name": derivative_deriver_name,
                "config": {},
            },
            "_null_related_episode_postulator": {
                "type": "related_episode_postulator",
                "name": "null",
                "config": {},
            },
        }

        session_memory_resource_definitions = {
            "_session_memory": {
                "type": "session_memory",
                "name": "session_memory",
                "config": session_memory_config,
            }
        }

        resources = ResourceInitializer.initialize(
            long_term_memory_resource_definitions
            | session_memory_resource_definitions,
            resources,
        )

        self._long_term_memory = resources.get("_long_term_memory")
        self._session_memory = resources.get("_session_memory")

        metrics_factory = config.get("metrics_factory")
        if metrics_factory is not None and not isinstance(
            metrics_factory, MetricsFactory
        ):
            raise TypeError(
                "Metrics factory must be an instance of MetricsFactory"
            )

        self._collect_metrics = False
        if metrics_factory is not None:
            self._collect_metrics = True
            self._user_metrics_labels = config.get("user_metrics_labels", {})
            label_names = self._user_metrics_labels.keys()

            self._ingestion_latency_summary = metrics_factory.get_summary(
                "ingestion_latency",
                "Latency of Episode ingestion in milliseconds",
                label_names=label_names,
            )
            self._query_latency_summary = metrics_factory.get_summary(
                "query_latency",
                "Latency of query processing in milliseconds",
                label_names=label_names,
            )
            self._ingestion_counter = metrics_factory.get_counter(
                "ingestion_count",
                "Count of Episode ingestion",
                label_names=label_names,
            )
            self._query_counter = metrics_factory.get_counter(
                "query_count",
                "Count of query processing",
                label_names=label_names,
            )

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
    ) -> bool:
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
            logger.error(
                "The producer %s does not belong to the session", producer
            )
            return False

        if (
            produced_for not in self._memory_context.user_id
            and produced_for not in self._memory_context.agent_id
        ):
            logger.error(
                "The produced_for %s does not belong to the session",
                produced_for,
            )
            return False
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
        await asyncio.gather(
            self._session_memory.add_episode(episode),
            self._long_term_memory.add_episode(episode),
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
            logger.info(
                "Closing context memory: group %s, session %s",
                self._memory_context.group_id,
                self._memory_context.session_id,
            )
            await self._session_memory.close()
            await self._manager.delete_episodic_memory(self._memory_context)

    async def delete_data(self):
        """
        Deletes all data from both session and declarative memory for this
        context.
        This is a destructive operation.
        """
        async with self._lock:
            await asyncio.gather(
                self._session_memory.clear_memory(),
                self._long_term_memory.forget_session(),
            )
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
            # Concurrently search both memory stores
            session_result, long_episode = await asyncio.gather(
                self._session_memory.get_session_memory_context(
                    query, limit=search_limit
                ),
                self._long_term_memory.search(
                    query, search_limit, property_filter
                ),
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
        episodes, summary = await self.query_memory(query, limit)

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
