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
- Each instance is managed by the `EpisodicMemoryManager`.
"""

import asyncio
import datetime
import json
import logging
import time
from collections.abc import Coroutine, Iterable
from typing import cast, get_args

from pydantic import BaseModel, Field, InstanceOf, model_validator

from memmachine.common.data_types import FilterablePropertyValue
from memmachine.common.episode_store import (
    Episode,
    EpisodeResponse,
    EpisodeType,
)
from memmachine.common.filter.filter_parser import (
    FilterExpr,
)
from memmachine.common.metrics_factory import MetricsFactory
from memmachine.episodic_memory.long_term_memory.long_term_memory import LongTermMemory
from memmachine.episodic_memory.short_term_memory.short_term_memory import (
    ShortTermMemory,
)

logger = logging.getLogger(__name__)


class EpisodicMemoryParams(BaseModel):
    """
    Parameters for configuring the EpisodicMemory.

    Attributes:
        session_key (str): The unique identifier for the session.
        metrics_factory (MetricsFactory): The metrics factory.
        long_term_memory (LongTermMemory): The long-term memory.
        short_term_memory (ShortTermMemory): The short-term memory.
        enabled (bool): Whether the episodic memory is enabled.

    """

    session_key: str = Field(
        ...,
        min_length=1,
        description="The unique identifier for the session",
    )
    metrics_factory: InstanceOf[MetricsFactory] = Field(
        ...,
        description="The metrics factory",
    )
    long_term_memory: InstanceOf[LongTermMemory] | None = Field(
        default=None,
        description="The long-term memory",
    )
    short_term_memory: InstanceOf[ShortTermMemory] | None = Field(
        default=None,
        description="The short-term memory",
    )
    enabled: bool = Field(
        default=True,
        description="Whether the episodic memory is enabled",
    )

    @model_validator(mode="after")
    def validate_memory_params(self) -> "EpisodicMemoryParams":
        if not self.enabled:
            return self
        if self.short_term_memory is None and self.long_term_memory is None:
            raise ValueError(
                "At least one of short_term_memory or long_term_memory must be provided.",
            )
        return self


class EpisodicMemory:
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
        params: EpisodicMemoryParams,
    ) -> None:
        """
        Initialize a EpisodicMemory instance.

        Args:
            params (EpisodicMemoryParams): Parameters for the EpisodicMemory.

        """
        self._closed = False

        self._session_key = params.session_key

        self._short_term_memory: ShortTermMemory | None = params.short_term_memory
        self._long_term_memory: LongTermMemory | None = params.long_term_memory

        self._enabled = params.enabled
        if not self._enabled:
            return
        if self._short_term_memory is None and self._long_term_memory is None:
            raise ValueError("No memory is configured")

        metrics_manager = params.metrics_factory
        # Initialize metrics
        self._ingestion_latency_summary = metrics_manager.get_summary(
            "Ingestion_latency",
            "Latency of Episode ingestion in milliseconds",
        )
        self._query_latency_summary = metrics_manager.get_summary(
            "query_latency",
            "Latency of query processing in milliseconds",
        )
        self._ingestion_counter = metrics_manager.get_counter(
            "Ingestion_count",
            "Count of Episode ingestion",
        )
        self._query_counter = metrics_manager.get_counter(
            "query_count",
            "Count of query processing",
        )

    @property
    def short_term_memory(self) -> ShortTermMemory | None:
        """
        Get the short-term memory of the episodic memory instance.

        Returns:
            The short-term memory of the episodic memory instance.

        """
        return self._short_term_memory

    @short_term_memory.setter
    def short_term_memory(self, value: ShortTermMemory | None) -> None:
        """
        Set the short-term memory of the episodic memory instance.

        This makes the short term memory can be injected.

        Args:
            value: The new short-term memory of the episodic memory instance.

        """
        self._short_term_memory = value

    @property
    def long_term_memory(self) -> LongTermMemory | None:
        """
        Get the long-term memory of the episodic memory instance.

        Returns:
            The long-term memory of the episodic memory instance.

        """
        return self._long_term_memory

    @long_term_memory.setter
    def long_term_memory(self, value: LongTermMemory | None) -> None:
        """
        Set the long-term memory of the episodic memory instance.

        This makes the long term memory can be injected.

        Args:
            value: The new long-term memory of the episodic memory instance.

        """
        self._long_term_memory = value

    @property
    def session_key(self) -> str:
        """
        Get the session key of the episodic memory instance.

        Returns:
            The session key of the episodic memory instance.

        """
        return self._session_key

    async def add_memory_episodes(self, episodes: list[Episode]) -> None:
        """
        Add a new memory episode to both session and declarative memory.

        Args:
            episodes: Episode instances to ingest.

        """
        if not self._enabled:
            return
        start_time = time.monotonic_ns()

        if self._closed:
            raise RuntimeError(f"Memory is closed {self._session_key}")
        # Create filterable property
        for episode in episodes:
            if episode.metadata is not None and episode.filterable_metadata is None:
                episode.filterable_metadata = {}
                for key, value in episode.metadata.items():
                    if isinstance(value, get_args(FilterablePropertyValue)):
                        episode.filterable_metadata[key] = value

        # Add the episode to both memory stores concurrently
        tasks: list[Coroutine] = []
        if self._short_term_memory:
            tasks.append(self._short_term_memory.add_episodes(episodes))
        if self._long_term_memory:
            tasks.append(self._long_term_memory.add_episodes(episodes))
        await asyncio.gather(
            *tasks,
        )
        end_time = time.monotonic_ns()
        delta = (end_time - start_time) / 1000000
        self._ingestion_latency_summary.observe(delta)
        self._ingestion_counter.increment()

    async def close(self) -> None:
        """
        Decrement the reference count and close the underlying memory stores.

        When the reference count is zero, it closes the memory stores and
        notifies the manager to remove this instance from its registry.
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

    async def delete_episodes(self, uids: Iterable[str]) -> None:
        """Delete episodes by UID."""
        if not self._enabled:
            return

        uids = list(uids)

        delete_episodes_coroutines: list[Coroutine] = []
        if self._short_term_memory:
            delete_episodes_coroutines.extend(
                self._short_term_memory.delete_episode(uid) for uid in uids
            )
        if self._long_term_memory:
            delete_episodes_coroutines.append(
                self._long_term_memory.delete_episodes(uids)
            )
        await asyncio.gather(*delete_episodes_coroutines)

    async def delete_session_episodes(self) -> None:
        """Delete all data from both session and declarative memory for this context."""
        if not self._enabled:
            return
        tasks = []
        if self._short_term_memory:
            tasks.append(self._short_term_memory.clear_memory())
        if self._long_term_memory:
            tasks.append(self._long_term_memory.delete_matching_episodes())
        await asyncio.gather(*tasks)

    class QueryResponse(BaseModel):
        """Aggregated search results from both long- and short-term memory."""

        class ShortTermMemoryResponse(BaseModel):
            """Aggregated search results from short-term memory."""

            episodes: list[EpisodeResponse]
            episode_summary: list[str]

        class LongTermMemoryResponse(BaseModel):
            """Aggregated search results from long-term memory."""

            episodes: list[EpisodeResponse]

        long_term_memory: LongTermMemoryResponse
        short_term_memory: ShortTermMemoryResponse

    async def query_memory(
        self,
        query: str,
        limit: int | None = None,
        property_filter: FilterExpr | None = None,
    ) -> QueryResponse | None:
        """
        Retrieve relevant context for a given query from all memory stores.

        It fetches episodes from both short-term (session) and long-term
        (declarative) memory, deduplicates them, and returns them along with
        any available summary.

        Args:
            query: The query string to find context for.
            limit: The maximum number of episodes to return. The limit is
                   applied to both short and long term memories. The default
                   value is 20.
            property_filter: Properties to filter declarative memory searches.

        Returns:
            A tuple containing a list of short term memory Episode objects,
            a list of long term memory Episode objects, and a
            list of summary strings.

        """
        if not self._enabled:
            return None
        start_time = time.monotonic_ns()
        search_limit = limit if limit is not None else 20

        if self._short_term_memory is None:
            short_episode: list[Episode] = []
            short_summary = ""
            long_episode = await cast("LongTermMemory", self._long_term_memory).search(
                query,
                num_episodes_limit=search_limit,
                property_filter=property_filter,
            )
        elif self._long_term_memory is None:
            session_result = (
                await self._short_term_memory.get_short_term_memory_context(
                    query,
                    limit=search_limit,
                    filters=property_filter,
                )
            )
            long_episode = []
            short_episode, short_summary = session_result
        else:
            # Concurrently search both memory stores
            session_result, long_episode = await asyncio.gather(
                self._short_term_memory.get_short_term_memory_context(
                    query,
                    limit=search_limit,
                    filters=property_filter,
                ),
                self._long_term_memory.search(
                    query,
                    num_episodes_limit=search_limit,
                    property_filter=property_filter,
                ),
            )
            short_episode, short_summary = session_result

        # Deduplicate episodes from both memory stores, prioritizing
        # short-term memory
        episode_uid_set = {episode.uid for episode in short_episode}

        unique_long_episodes = []
        for episode in long_episode:
            if episode.uid not in episode_uid_set:
                episode_uid_set.add(episode.uid)
                unique_long_episodes.append(episode)

        end_time = time.monotonic_ns()
        delta = (end_time - start_time) / 1000000
        self._query_latency_summary.observe(delta)
        self._query_counter.increment()

        return EpisodicMemory.QueryResponse(
            short_term_memory=EpisodicMemory.QueryResponse.ShortTermMemoryResponse(
                episodes=[
                    EpisodeResponse(**episode.model_dump()) for episode in short_episode
                ],
                episode_summary=[short_summary],
            ),
            long_term_memory=EpisodicMemory.QueryResponse.LongTermMemoryResponse(
                episodes=[
                    EpisodeResponse(**episode.model_dump())
                    for episode in unique_long_episodes
                ],
            ),
        )

    async def formalize_query_with_context(
        self,
        query: str,
        limit: int | None = None,
        property_filter: FilterExpr | None = None,
    ) -> str:
        """
        Construct a finalized query string that includes context from memory.

        The context (summary and recent episodes) is prepended to the original
        query, formatted with XML-like tags for the language model to parse.

        Args:
            query: The original query string.
            limit: The maximum number of episodes to include in the context.
            property_filter: Properties to filter the search.

        Returns:
            A new query string enriched with context.

        """
        query_result = await self.query_memory(
            query,
            limit,
            property_filter,
        )
        if query_result is None:
            logger.warning("Query result is None in formalize_query_with_context")
            return query

        episodes = sorted(
            query_result.short_term_memory.episodes
            + query_result.long_term_memory.episodes,
            key=lambda x: cast(datetime.datetime, x.created_at),
        )

        finalized_query = ""

        # Add summary if it exists
        if (
            query_result.short_term_memory.episode_summary
            and len(query_result.short_term_memory.episode_summary) > 0
        ):
            total_summary = ""
            for summ in query_result.short_term_memory.episode_summary:
                if not summ:
                    continue
                total_summary = total_summary + summ + "\n"
            total_summary = total_summary.strip()
            if total_summary:
                finalized_query += "<Summary>\n"
                finalized_query += total_summary
                finalized_query += "\n</Summary>\n"

        # Add episodes if they exist
        if episodes and len(episodes) > 0:
            finalized_query += "<Episodes>\n"
            finalized_query += EpisodicMemory.string_from_episode_response_context(
                episodes,
            )
            finalized_query += "</Episodes>\n"

        # Append the original query
        finalized_query += f"<Query>\n{query}\n</Query>"

        return finalized_query

    @staticmethod
    def string_from_episode_response_context(
        episode_response_context: Iterable[EpisodeResponse],
    ) -> str:
        """Format episode response context as a string."""
        context_string = ""

        for episode_response in episode_response_context:
            match episode_response.episode_type:
                case EpisodeType.MESSAGE:
                    context_date = (
                        EpisodicMemory._format_date(
                            episode_response.created_at.date(),
                        )
                        if episode_response.created_at
                        else "Unknown Date"
                    )
                    context_time = (
                        EpisodicMemory._format_time(
                            episode_response.created_at.time(),
                        )
                        if episode_response.created_at
                        else "Unknown Time"
                    )
                    context_string += f"[{context_date} at {context_time}] {episode_response.producer_id}: {json.dumps(episode_response.content)}\n"
                case _:
                    context_string += json.dumps(episode_response.content) + "\n"

        return context_string

    @staticmethod
    def _format_date(date: datetime.date) -> str:
        """Format the date as a string."""
        return date.strftime("%A, %B %d, %Y")

    @staticmethod
    def _format_time(time: datetime.time) -> str:
        """Format the time as a string."""
        return time.strftime("%I:%M %p")
