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
from collections.abc import Mapping

from pydantic import BaseModel, Field, InstanceOf, field_validator

from memmachine.common.data_types import (
    ExternalServiceAPIError,
    FilterablePropertyValue,
)
from memmachine.common.language_model import LanguageModel
from memmachine.common.session_manager.session_data_manager import SessionDataManager
from memmachine.episode_store.episode_model import Episode

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
        self._model: LanguageModel = param.llm_model
        self._data_manager: SessionDataManager | None = param.data_manager
        self._summary_user_prompt = param.summary_prompt_user
        self._summary_system_prompt = param.summary_prompt_system
        self._memory: deque[Episode] = deque()
        self._current_episode_count = 0
        self._max_message_len = param.message_capacity
        self._current_message_len = 0
        self._summary = summary
        self._session_key = param.session_key
        self._summary_task: asyncio.Task | None = None
        self._closed = False
        self._lock = asyncio.Lock()
        if episodes is not None:
            self._memory.extend(episodes)
            self._current_episode_count = len(episodes)
            for e in episodes:
                self._current_message_len += len(e.content)

    @classmethod
    async def create(cls, params: ShortTermMemoryParams) -> "ShortTermMemory":
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
                # ToDo: Retreive the episodes from raw data storage
                return ShortTermMemory(params, summary)
            except ValueError:
                pass
        return ShortTermMemory(params)

    def _is_full(self) -> bool:
        """
        Check if the short-term memory has reached its capacity.

        Memory is considered full if total message
        length exceeds its respective maximums.

        Returns:
            True if the memory is full, False otherwise.

        """
        result = self._current_message_len + len(self._summary) > self._max_message_len
        return result

    async def add_episodes(self, episodes: list[Episode]) -> bool:
        """
        Add new episodes to the short-term memory.

        Args:
            episodes: The episodes to add.

        Returns:
            True if the memory is full after adding the event, False
            otherwise.

        """
        async with self._lock:
            if self._closed:
                raise RuntimeError(f"Memory is closed {self._session_key}")
            self._memory.extend(episodes)

            self._current_episode_count += len(episodes)
            self._current_message_len += sum(len(e.content) for e in episodes)
            full = self._is_full()
            if full:
                await self._do_evict()
            return full

    async def _do_evict(self) -> None:
        """
        Evict episodes to make space while building a summary asynchronously.

        asynchronously. It clears the stats. It keeps as many episode
        as possible for current capacity.
        """
        result = []
        # Remove old messages that have been summarized
        while (
            len(self._memory) > self._current_episode_count
            and self._current_message_len + len(self._summary) > self._max_message_len
        ):
            self._current_message_len -= len(self._memory[0].content)
            self._memory.popleft()

        if (
            len(self._memory) == 0
            or self._current_message_len + len(self._summary) <= self._max_message_len
        ):
            return

        result = list(self._memory)
        # Reset the count so it will only count new episodes
        self._current_episode_count = 0
        # if previous summary task is still running, wait for it
        if self._summary_task is not None:
            await self._summary_task
        self._summary_task = asyncio.create_task(self._create_summary(result))

    async def close(self) -> None:
        """
        Clear all events and the summary from the short-term memory.

        Resets the message length to zero.
        """
        async with self._lock:
            if self._closed:
                return
            self._closed = True
            if self._summary_task is not None:
                await self._summary_task
            self._summary_task = None
            self._memory.clear()
            self._current_episode_count = 0
            self._current_message_len = 0
            self._summary = ""

    async def clear_memory(self) -> None:
        """Clear all events and summary. Reset the message length to zero."""
        async with self._lock:
            if self._closed:
                return
            if self._summary_task is not None:
                await self._summary_task
            self._summary_task = None
            self._memory.clear()
            self._current_episode_count = 0
            self._current_message_len = 0
            self._summary = ""

    async def delete_episode(self, uid: str) -> bool:
        """Delete one episode by UID."""
        async with self._lock:
            for e in self._memory:
                if e.uid == uid:
                    self._current_episode_count -= 1
                    self._current_message_len -= len(e.content)
                    self._memory.remove(e)
                    return True
            return False

    async def _create_summary(self, episodes: list[Episode]) -> None:
        """
        Generate a new summary of the events currently in memory.

        If no summary exists, it creates a new one. If a summary already
        exists, it creates a "rolling" summary that incorporates the previous
        summary and the new episodes. It uses the configured language model
        and prompts to generate the summary.
        """
        try:
            episode_content = ""
            for entry in episodes:
                meta = ""
                if entry.metadata is None:
                    pass
                elif isinstance(entry.metadata, str):
                    meta = entry.metadata
                elif isinstance(entry.metadata, dict):
                    for k, v in entry.metadata.items():
                        meta += f"[{k}: {v}] "
                else:
                    meta = repr(entry.metadata)
                episode_content += f"[{entry.uid!s} : {meta} : {entry.content}]"
            msg = self._summary_user_prompt.format(
                episodes=episode_content,
                summary=self._summary,
                max_length=self._max_message_len / 2,
            )
            result = await self._model.generate_response(
                system_prompt=self._summary_system_prompt,
                user_prompt=msg,
            )
            self._summary = result[0]
            if self._data_manager is not None:
                await self._data_manager.save_short_term_memory(
                    self._session_key,
                    self._summary,
                    episodes[-1].sequence_num,
                    len(episodes),
                )

            logger.debug("Summary: %s\n", self._summary)
        except ExternalServiceAPIError:
            logger.info("External API error when creating summary")
        except ValueError:
            logger.info("Value error when creating summary")
        except RuntimeError:
            logger.info("Runtime error when creating summary")

    async def get_short_term_memory_context(  # noqa: C901
        self,
        query: str,
        limit: int = 0,
        max_message_length: int = 0,
        filters: Mapping[str, FilterablePropertyValue] | None = None,
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
        async with self._lock:
            if self._closed:
                raise RuntimeError(f"Memory is closed {self._session_key}")
            if self._summary_task is not None:
                await self._summary_task
                self._summary_task = None
            length = 0 if self._summary is None else len(self._summary)
            episodes: deque[Episode] = deque()

            for e in reversed(self._memory):
                if length >= max_message_length > 0:
                    break
                if len(episodes) >= limit > 0:
                    break
                # check if should filter the message
                matched = True
                if filters is not None:
                    for key, value in filters.items():
                        if key == "producer_id" and e.producer_id != value:
                            matched = False
                            break
                        if key == "producer_role" and e.producer_role != value:
                            matched = False
                            break
                        if key == "produced_for_id" and e.produced_for_id != value:
                            matched = False
                            break
                        if key.startswith(("m.", "metadata.")):
                            if e.filterable_metadata is None:
                                matched = False
                                break
                            if key.startswith("m."):
                                key = key[len("m.") :]
                            elif key.startswith("metadata."):
                                key = key[len("metadata.") :]
                            if key not in e.filterable_metadata:
                                matched = False
                                break
                            if e.filterable_metadata[key] != value:
                                matched = False
                                break
                        else:
                            logger.warning("Unsupported filter key: %s", key)

                    if not matched:
                        continue

                msg_len = self._compute_episode_length(e)
                if length + msg_len > max_message_length > 0:
                    break
                episodes.appendleft(e)
                length += msg_len
            return list(episodes), self._summary

    def _compute_episode_length(self, episode: Episode) -> int:
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
