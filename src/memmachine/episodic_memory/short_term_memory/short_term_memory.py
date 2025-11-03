"""
Manages short-term memory for a conversational session.

This module provides the `SessionMemory` class, which is responsible for
storing and managing a sequence of conversational turns (episodes) within a
single session. It uses a deque with a fixed capacity and evicts older
episodes when memory limits (message length) are reached. Evicted episodes
are summarized asynchronously to maintain context over a longer conversation.
"""

import asyncio
import logging
import uuid
from collections import deque

from memmachine.common.data_types import ExternalServiceAPIError
from memmachine.common.language_model import LanguageModel
from memmachine.configuration.episodic_config import ShortTermMemoryParams
from memmachine.session_manager_interface import SessionDataManager
from ..data_types import Episode

logger = logging.getLogger(__name__)


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
    ):
        # pylint: disable=too-many-arguments
        # pylint: disable=too-many-positional-arguments
        """
        Initializes the ShortTermMemory instance.
        """
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
        self._summary_task = None
        self._closed = False
        self._lock = asyncio.Lock()
        if episodes is not None:
            self._memory.extend(episodes)
            self._current_episode_count = len(episodes)
            for e in episodes:
                self._current_message_len += len(e.content)

    @classmethod
    async def create(cls, param: ShortTermMemoryParams) -> "ShortTermMemory":
        """
        Creates a new ShortTermMemory instance.
        """
        if param.data_manager is not None:
            try:
                await param.data_manager.create_tables()
            except ValueError:
                pass
            try:
                summary, episodes, _, _ = await param.data_manager.get_short_term_memory(
                    param.session_key
                )
                return ShortTermMemory(param, summary, episodes)
            except ValueError:
                pass
        return ShortTermMemory(param)

    def _is_full(self) -> bool:
        """
        Checks if the short-term memory has reached its capacity.

        Memory is considered full if total message
        length exceeds its respective maximums.

        Returns:
            True if the memory is full, False otherwise.
        """
        result = self._current_message_len + len(self._summary) > self._max_message_len
        return result

    async def add_episode(self, episode: Episode) -> bool:
        """
        Adds a new episode to the short-term memory.

        Args:
            episode: The episode to add.

        Returns:
            True if the memory is full after adding the event, False
            otherwise.
        """
        async with self._lock:
            if self._closed:
                raise RuntimeError(f"Memory is closed {self._session_key}")
            self._memory.append(episode)

            self._current_episode_count += 1
            self._current_message_len += len(episode.content)
            full = self._is_full()
            if full:
                await self._do_evict()
            return full

    async def _do_evict(self):
        """
        The eviction make a copy of the episode to create summary
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

        for e in self._memory:
            result.append(e)
        # Reset the count so it will only count new episodes
        self._current_episode_count = 0
        # if previous summary task is still running, wait for it
        if self._summary_task is not None:
            await self._summary_task
        self._summary_task = asyncio.create_task(self._create_summary(result))

    async def close(self):
        """
        Clears all events and the summary from the short-term memory.

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

    async def clear_memory(self):
        """
        Clear all events and summary. Reset the message length to zero.
        """
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

    async def delete_episode(self, uuid: uuid.UUID):
        """Delete one episode by uuid"""
        async with self._lock:
            for e in self._memory:
                if e.uuid == uuid:
                    self._current_episode_count -= 1
                    self._current_message_len -= len(e.content)
                    self._memory.remove(e)
                    return True
            return False

    async def _create_summary(self, episodes: list[Episode]):
        """
        Generates a new summary of the events currently in memory.

        If no summary exists, it creates a new one. If a summary already
        exists, it creates a "rolling" summary that incorporates the previous
        summary and the new episodes. It uses the configured language model
        and prompts to generate the summary.
        """
        try:
            episode_content = ""
            for entry in episodes:
                meta = ""
                if entry.user_metadata is None:
                    pass
                elif isinstance(entry.user_metadata, str):
                    meta = entry.user_metadata
                elif isinstance(entry.user_metadata, dict):
                    for k, v in entry.user_metadata.items():
                        meta += f"[{k}: {v}] "
                else:
                    meta = repr(entry.user_metadata)
                episode_content += f"[{str(entry.uuid)} : {meta} : {entry.content}]"
            msg = self._summary_user_prompt.format(
                episodes=episode_content,
                summary=self._summary,
                max_length=self._max_message_len / 2,
            )
            result = await self._model.generate_response(
                system_prompt=self._summary_system_prompt, user_prompt=msg
            )
            self._summary = result[0]
            if self._data_manager is not None:
                await self._data_manager.save_short_term_memory(
                    self._session_key, self._summary, episodes, episodes[-1].sequence_num, len(episodes)
                )

            logger.debug("Summary: %s\n", self._summary)
        except ExternalServiceAPIError:
            logger.info("External API error when creating summary")
        except ValueError:
            logger.info("Value error when creating summary")
        except RuntimeError:
            logger.info("Runtime error when creating summary")

    async def get_session_memory_context(
        self, query, limit: int = 0, max_message_length: int = 0, filter: dict[str, str] | None = None
    ) -> tuple[list[Episode], str]:
        """
        Retrieves context from short-term memory for a given query.

        This includes the current summary and as many recent episodes as can
        fit within a specified message length limit.

        Args:
            query: The user's query string.
            max_message_length: The maximum length of messages for the context. If 0,
            no limit is applied.

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
                #check if should filter the message
                if filter is not None:
                    if "producer" in filter and filter["producer"] != e.producer_id:
                        continue

                    matched = True
                    for key, value in filter.items():
                        if e.user_metadata.get(key) != value:
                            matched = False
                            break
                    if not matched:
                        continue

                msg_len = self._compute_episode_length(e)
                if length + msg_len > max_message_length > 0:
                    break
                episodes.appendleft(e)
                length += msg_len
            return list(episodes), self._summary

    def _compute_episode_length(self, episode: Episode) -> int:
        """
        Computes the message length in an episodes.
        """
        result = 0
        if episode.content is None:
            return 0
        if isinstance(episode.content, str):
            result += len(episode.content)
        else:
            result += len(repr(episode.content))
        if episode.user_metadata is None:
            return result
        if isinstance(episode.user_metadata, str):
            result += len(episode.user_metadata)
        elif isinstance(episode.user_metadata, dict):
            for _, v in episode.user_metadata.items():
                if isinstance(v, str):
                    result += len(v)
                else:
                    result += len(repr(v))
        return result
