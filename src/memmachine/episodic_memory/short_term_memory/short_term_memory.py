"""
Manages short-term memory for a conversational session.

This module provides the `SessionMemory` class, which is responsible for
storing and managing a sequence of conversational turns (episodes) within a
single session. It uses a deque with a fixed capacity and evicts older
episodes when memory limits (message length) are reached. Evicted episodes
are summarized asynchronously to maintain context over a longer conversation.
"""

import asyncio
import contextlib
import logging
import string
from collections import deque
from datetime import datetime
from typing import Self, cast, get_args

from pydantic import BaseModel, Field, InstanceOf, field_validator

from memmachine.common import rw_locks
from memmachine.common.data_types import (
    ExternalServiceAPIError,
    FilterablePropertyValue,
)
from memmachine.common.episode_store import Episode
from memmachine.common.episode_store.episode_model import episodes_to_string
from memmachine.common.errors import ShortTermMemoryClosedError
from memmachine.common.filter.filter_parser import And, Comparison, FilterExpr, Or
from memmachine.common.language_model import LanguageModel
from memmachine.common.session_manager.session_data_manager import SessionDataManager

logger = logging.getLogger(__name__)


class ShortTermMemoryParams(BaseModel):
    """
    Parameters for configuring the short-term memory.

    Attributes:
        session_key (str): The unique identifier for the session.
        llm_model (LanguageModel): The language model to use for summarization.
        data_manager (SessionDataManager): The session data manager.
        summary_prompt_system (str): The system prompt for the summarization.
        summary_prompt_user (str): The user prompt for the summarization.
        message_capacity (int): The maximum number of messages to summarize.

    """

    session_key: str = Field(..., description="Session identifier", min_length=1)
    llm_model: InstanceOf[LanguageModel] = Field(
        ...,
        description="The language model to use for summarization",
    )
    data_manager: InstanceOf[SessionDataManager] | None = Field(
        default=None,
        description="The session data manager",
    )
    summary_prompt_system: str = Field(
        ...,
        min_length=1,
        description="The system prompt for the summarization",
    )
    summary_prompt_user: str = Field(
        ...,
        min_length=1,
        description="The user prompt for the summarization",
    )
    message_capacity: int = Field(
        default=64000,
        gt=0,
        description="The maximum length of short-term memory",
    )

    @field_validator("summary_prompt_user")
    @classmethod
    def validate_summary_user_prompt(cls, v: str) -> str:
        """Validate the user prompt for the summarization."""
        fields = [fname for _, fname, _, _ in string.Formatter().parse(v) if fname]
        if len(fields) != 3:
            raise ValueError(f"Expect 3 fields in {v}")
        if "episodes" not in fields:
            raise ValueError(f"Expect 'episodes' in {v}")
        if "summary" not in fields:
            raise ValueError(f"Expect 'summary' in {v}")
        if "max_length" not in fields:
            raise ValueError(f"Expect 'max_length' in {v}")
        return v


class ShortTermMemory:
    # pylint: disable=too-many-instance-attributes
    """
    Manages the short-term memory of conversion context.

    This class stores a sequence of recent events (episodes) in a deque with a
    fixed capacity. When the memory becomes full (based on the total message length),
    older events are evicted and summarized.
    """

    def __init__(
        self,
        param: ShortTermMemoryParams,
        summary: str = "",
        episodes: list[Episode] | None = None,
    ) -> None:
        # pylint: disable=too-many-arguments
        # pylint: disable=too-many-positional-arguments
        """Initialize the ShortTermMemory instance."""
        self._memory: deque[Episode] = deque()
        self._current_episode_count = 0
        self._max_message_len = param.message_capacity
        self._current_message_len = 0
        self._session_key = param.session_key
        self._closed = False
        self._lock = rw_locks.AsyncRWLock()
        params = ShortTermMemoryConsolidator.Params(
            summary_user_prompt=param.summary_prompt_user,
            summary_system_prompt=param.summary_prompt_system,
            max_summary_length_words=self.max_summary_length_words,
            session_key=self._session_key,
            model=param.llm_model,
            data_manager=param.data_manager,
            summary=summary,
        )
        self._consolidator = ShortTermMemoryConsolidator(params)
        if episodes is not None:
            self._memory.extend(episodes)
            self._current_episode_count = len(episodes)
            for e in episodes:
                self._current_message_len += len(e.content)

    @property
    def max_summary_length_words(self) -> int:
        """Get the maximum summary length in words."""
        approximate_characters_per_word = 8
        max_summary_length_words = int(
            self._max_message_len / 2 / approximate_characters_per_word
        )
        # round the length to nearest 100
        max_summary_length_words = (max_summary_length_words + 99) // 100 * 100
        return max_summary_length_words

    @classmethod
    async def create(cls, params: ShortTermMemoryParams) -> Self:
        """Create a new ShortTermMemory instance."""
        if params.data_manager is not None:
            with contextlib.suppress(ValueError):
                await params.data_manager.create_tables()
            try:
                (
                    summary,
                    _,
                    _,
                ) = await params.data_manager.get_short_term_memory(params.session_key)
                # ToDo: Retrieve the episodes from raw data storage
                return cls(params, summary)
            except ValueError:
                pass
        return cls(params)

    async def _is_full(self) -> bool:
        """
        Check if the short-term memory has reached its capacity.

        Memory is considered full if total message
        length exceeds its respective maximums.

        Returns:
            True if the memory is full, False otherwise.

        """
        return await self._get_total_message_len() > self._max_message_len

    async def add_episodes(self, episodes: list[Episode]) -> bool:
        """
        Add new episodes to the short-term memory.

        Args:
            episodes: The episodes to add.

        Returns:
            True if the memory is full after adding the event, False
            otherwise.

        """
        async with self._lock.write_lock():
            if self._closed:
                raise ShortTermMemoryClosedError(self._session_key)
            self._memory.extend(episodes)

            self._current_episode_count += len(episodes)
            self._current_message_len += sum(len(e.content) for e in episodes)
            full = await self._is_full()
            if full:
                await self._do_evict()
            return full

    async def _get_total_message_len(self) -> int:
        """Get the total message length in short-term memory."""
        return self._current_message_len + await self.get_summary_length()

    async def _wait_for_summary_to_finish(self) -> None:
        """Wait for any ongoing summarization to complete."""
        async with self._lock.read_lock():
            if self._closed:
                raise ShortTermMemoryClosedError(self._session_key)
            await self._consolidator.wait_until_done()

    async def _do_evict(self) -> None:
        """
        Evict episodes to make space while building a summary asynchronously.

        asynchronously. It clears the stats. It keeps as many episode
        as possible for current capacity.
        """
        # Remove old messages that have been summarized
        while len(self._memory) > self._current_episode_count and await self._is_full():
            self._current_message_len -= len(self._memory[0].content)
            self._memory.popleft()

        if len(self._memory) == 0 or not await self._is_full():
            return

        result = list(self._memory)
        # Reset the count so it will only count new episodes
        self._current_episode_count = 0
        await self._consolidator.summarize(result)

    async def close(self) -> None:
        """
        Clear all events and the summary from the short-term memory.

        Resets the message length to zero.
        """
        async with self._lock.write_lock():
            await self._do_reset()
            self._closed = True

    async def _do_reset(self) -> None:
        """Reset the status of the short-term memory."""
        if self._closed:
            return
        await self._consolidator.wait_until_done()
        self._memory.clear()
        self._current_episode_count = 0
        self._current_message_len = 0

    async def clear_memory(self) -> None:
        """Clear all events and summary. Reset the message length to zero."""
        async with self._lock.write_lock():
            await self._do_reset()

    async def delete_episode(self, uid: str) -> bool:
        """Delete one episode by UID."""
        async with self._lock.write_lock():
            for index, episode in enumerate(self._memory):
                if episode.uid == uid:
                    if index >= len(self._memory) - self._current_episode_count:
                        # only update the count if it's in the current episode set
                        self._current_episode_count -= 1
                    self._current_message_len -= len(episode.content)
                    self._memory.remove(episode)
                    return True
            return False

    @staticmethod
    def _safe_compare(
        a: FilterablePropertyValue,
        b: FilterablePropertyValue | list[FilterablePropertyValue],
        op: str,
    ) -> bool:
        """Safely compare two filterable property values."""
        if a is None or b is None or isinstance(b, list):
            return False

        comparisons = (
            ((int, float), lambda x, y: (float(x), float(y))),
            (str, lambda x, y: (x, y)),
            (datetime, lambda x, y: (x, y)),
        )
        for expected_type, formatter in comparisons:
            if isinstance(a, expected_type) and isinstance(b, expected_type):
                left, right = formatter(a, b)
                break
        else:
            logger.warning(
                "Unsupported operator: %s, %s, %s",
                op,
                type(a).__name__,
                type(b).__name__,
            )
            return False

        match op:
            case ">=":
                return left >= right
            case "<=":
                return left <= right
            case ">":
                return left > right
            case "<":
                return left < right
            case _:
                logger.warning("Unsupported operator: %s", op)
                return False

    def _do_comparison(
        self, comp: Comparison, value: FilterablePropertyValue | None
    ) -> bool:
        """Do comparison for a single comparison expression."""
        match comp.op:
            case "==" | "=":
                return value == comp.value
            case "!=":
                return value != comp.value
            case "in":
                if not isinstance(comp.value, list):
                    raise TypeError("IN operator requires a list value")
                return value in comp.value
            case "not_in":
                if not isinstance(comp.value, list):
                    raise TypeError("NOT IN operator requires a list value")
                return value not in comp.value
            case "is_null":
                return value is None
        if value is None or comp.value is None:
            return False
        return self._safe_compare(value, comp.value, comp.op)

    def _do_logical_check(self, episode: Episode, filters: FilterExpr) -> bool:
        """Do logical check for AND/OR expressions."""
        if isinstance(filters, And):
            if not self._check_filter(episode, filters.left):
                return False
            return self._check_filter(episode, filters.right)

        if isinstance(filters, Or):
            or_filter = cast(Or, filters)
            if self._check_filter(episode, or_filter.left):
                return True
            return self._check_filter(episode, or_filter.right)
        logger.warning("Unsupported logical filter: %s", type(filters).__name__)
        return False

    def _check_filter(self, episode: Episode, filters: FilterExpr | None) -> bool:
        """Check if an episode matches the given filters."""
        if filters is None:
            return True

        if isinstance(filters, Comparison):
            match filters.field:
                case "producer_id":
                    return self._do_comparison(filters, episode.producer_id)
                case "produced_for_id":
                    return self._do_comparison(filters, episode.produced_for_id)
                case "producer_role":
                    return self._do_comparison(filters, episode.producer_role)
            if filters.field.startswith(("m.", "metadata.")):
                key = (
                    filters.field[9:]
                    if filters.field.startswith("metadata.")
                    else filters.field[2:]
                )
                if episode.metadata is None or not isinstance(episode.metadata, dict):
                    return False
                if key not in episode.metadata:
                    return False
                if not isinstance(
                    episode.metadata[key], get_args(FilterablePropertyValue)
                ):
                    return False
                return self._do_comparison(
                    filters, cast(FilterablePropertyValue, episode.metadata[key])
                )
            logger.warning("Unsupported filter field: %s", filters.field)
            return False

        return self._do_logical_check(episode, filters)

    async def get_summary(self) -> str:
        """Get the current summary."""
        await self._wait_for_summary_to_finish()
        return await self._consolidator.summary

    async def get_summary_length(self) -> int:
        """Get the current summary length."""
        return await self._consolidator.summary_len

    async def get_short_term_memory_context(
        self,
        query: str,
        limit: int = 0,
        max_message_length: int = 0,
        filters: FilterExpr | None = None,
    ) -> tuple[list[Episode], str]:
        """
        Retrieve context from short-term memory for a given query.

        This includes the current summary and as many recent episodes as can
        fit within a specified message length limit.

        Args:
            query: The user's query string.
            limit: Maximum number of episodes to include.
            max_message_length: The maximum length of messages for the context. If 0,
            no limit is applied.
            filters: Optional property filters for episodes.

        Returns:
            A tuple containing a list of episodes and the current summary.

        """
        logger.debug("Get session for %s", query)
        async with self._lock.read_lock():
            if self._closed:
                raise ShortTermMemoryClosedError(self._session_key)
            await self._consolidator.wait_until_done()
            length = await self.get_summary_length()
            episodes: deque[Episode] = deque()

            for e in reversed(self._memory):
                if length >= max_message_length > 0:
                    break
                if len(episodes) >= limit > 0:
                    break
                # check if should filter the message
                if not self._check_filter(e, filters):
                    continue

                msg_len = self._compute_episode_length(e)
                if length + msg_len > max_message_length > 0:
                    break
                episodes.appendleft(e)
                length += msg_len
            return list(episodes), await self.get_summary()

    @staticmethod
    def _compute_episode_length(episode: Episode) -> int:
        """Compute the message length in an episode."""
        result = 0
        if episode.content is None:
            return 0
        if isinstance(episode.content, str):
            result += len(episode.content)
        else:
            result += len(repr(episode.content))
        if episode.metadata is None:
            return result
        if isinstance(episode.metadata, str):
            result += len(episode.metadata)
        elif isinstance(episode.metadata, dict):
            for v in episode.metadata.values():
                if isinstance(v, str):
                    result += len(v)
                else:
                    result += len(repr(v))
        return result


class ShortTermMemoryConsolidator:
    """
    Async consolidator that handles summarization of episodic memory.

    This class decouples the ingestion of new memory episodes from the high-latency
    process of LLM-based summarization. When new episodes are provided via
    `summarize()`, they are added to an internal buffer. A single background worker
    processes these episodes sequentially, ensuring that even if episodes arrive
    rapidly, the system does not "explode" with concurrent API calls.

    Design Pattern:
        Background Worker with Dynamic Batching. This ensures sequential
        integrity (summaries are never updated out of order) and eventual
        consistency (the summary will eventually reflect all ingested episodes).
    """

    class Params(BaseModel):
        """Parameters for ShortTermMemoryConsolidator."""

        summary_user_prompt: str
        summary_system_prompt: str
        max_summary_length_words: int
        session_key: str
        model: InstanceOf[LanguageModel]
        data_manager: InstanceOf[SessionDataManager] | None = None
        summary: str = ""

    def __init__(self, params: Params) -> None:
        """Create a ShortTermMemoryConsolidator instance."""
        self._summary_user_prompt = params.summary_user_prompt
        self._summary_system_prompt = params.summary_system_prompt
        self._max_summary_length_words = params.max_summary_length_words
        self._session_key = params.session_key
        self._data_manager = params.data_manager
        self._model = params.model
        self._summary = params.summary

        # Batching state
        self._pending_episodes: list[Episode] = []
        self._worker_task: asyncio.Task | None = None
        self._lock = asyncio.Lock()
        self._summary_lock = (
            rw_locks.AsyncRWLock()
        )  # Protects the pending list and task state

    @property
    def is_running(self) -> bool:
        """Check if the background summarization worker is running."""
        return self._worker_task is not None and not self._worker_task.done()

    async def summarize(self, episodes: list[Episode]) -> None:
        """Non-blocking call to add episodes to the summarization queue."""
        async with self._lock:
            self._pending_episodes.extend(episodes)

            # Start the worker if it's not already running
            if not self.is_running:
                self._worker_task = asyncio.create_task(self._run_summary_loop())

    async def _run_summary_loop(self) -> None:
        """Background worker that drains the pending episodes."""
        while True:
            # 1. Grab the current batch and clear the buffer
            async with self._lock:
                if not self._pending_episodes:
                    # No more episodes to process, exit the worker
                    break

                batch_to_process = self._pending_episodes[:]
                self._pending_episodes.clear()

            # 2. Perform the slow async summary (Outside the lock)
            try:
                # We use the accumulated batch for this run
                new_summary = await self._create_summary(
                    await self.summary, batch_to_process
                )

                await self.set_summary(new_summary)
            except Exception:
                # Log the error, but don't crash the loop
                # or we lose the background worker.
                logger.exception("Summarization failed in helper.")

            # Loop continues to check if more episodes arrived
            # while we were 'awaiting' the summary.

    @property
    async def summary(self) -> str:
        """Get the current summary."""
        async with self._summary_lock.read_lock():
            return self._summary

    async def set_summary(self, summary: str) -> None:
        """Set the current summary if not empty."""
        if summary:
            async with self._summary_lock.write_lock():
                self._summary = summary

    @property
    async def summary_len(self) -> int:
        """Get the length of the current summary."""
        async with self._summary_lock.read_lock():
            return len(self._summary) if self._summary else 0

    async def wait_until_done(self) -> None:
        """Wait for the background summarization to catch up."""
        task = self._worker_task
        if task is not None and not task.done():
            await task

    @staticmethod
    def _is_exceed_context_window_error(e: Exception) -> bool:
        """Check if the exception is due to exceeding context window."""
        error_msg = str(e).lower()
        keywords = [
            "context length",
            "context window",
            "maximum input size",
            "input length",
            "token limit",
            "too long",
            "exceeds the maximum",
        ]
        return any(keyword in error_msg for keyword in keywords)

    async def _create_summary(self, summary: str, episodes: list[Episode]) -> str:
        """
        Generate a summary recursively.

        splitting the batch if it exceeds the context window or encounters an error.
        """
        # Base Case: Nothing to process
        if not episodes:
            return summary

        try:
            # Attempt to summarize the current batch
            episode_content = episodes_to_string(episodes)
            msg = self._summary_user_prompt.format(
                episodes=episode_content,
                summary=summary,
                max_length=self._max_summary_length_words,
            )

            result = await self._model.generate_response(
                system_prompt=self._summary_system_prompt,
                user_prompt=msg,
            )

            new_summary = result[0]

            # Save progress for this successful chunk
            if self._data_manager:
                await self._data_manager.save_short_term_memory(
                    self._session_key,
                    new_summary,
                    episodes[-1].sequence_num,
                    len(episodes),
                )
        except (ExternalServiceAPIError, ValueError, RuntimeError) as e:
            if self._is_exceed_context_window_error(e):
                # If a single episode fails, drop it and return the current summary
                if len(episodes) == 1:
                    logger.exception("Dropping failed episode %s", episodes[0].uid)
                    return summary

                # Otherwise, split and recurse
                mid = len(episodes) // 2
                logger.warning(
                    "Batch failed. Splitting %d episodes into halves.", len(episodes)
                )

                # 1. Summarize the first half
                summary_after_first_half = await self._create_summary(
                    summary, episodes[:mid]
                )

                # 2. Use that result as the 'base' to summarize the second half
                return await self._create_summary(
                    summary_after_first_half, episodes[mid:]
                )
            # For other errors, log and ignore
            logger.exception("Summarization failed due to unexpected error")
        else:
            return new_summary
        # return old summary if failed
        return summary
