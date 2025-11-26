"""
Manages short-term memory for a conversational session.

This module provides the `SessionMemory` class, which is responsible for
storing and managing a sequence of conversational turns (episodes) within a
single session. It uses a deque with a fixed capacity and evicts older
episodes when memory limits (number of episodes, message length, or token
count) are reached. Evicted episodes are summarized asynchronously to maintain
context over a longer conversation.
"""

import asyncio
import logging
from collections import deque

from memmachine.common.data_types import ExternalServiceAPIError

from ..data_types import Episode, MemoryContext

logger = logging.getLogger(__name__)


class SessionMemory:
    # pylint: disable=too-many-instance-attributes
    """
    Manages the short-term memory of conversion context.

    This class stores a sequence of recent events (episodes) in a deque with a
    fixed capacity. When the memory becomes full (based on the number of
    events, total message length, or total token count), older events are
    evicted and summarized.
    """

    def __init__(
        self,
        model,
        summary_system_prompt: str,
        summary_user_prompt: str,
        capacity: int,
        max_message_len: int,
        max_token_num: int,
        memory_context: MemoryContext,
    ):
        # pylint: disable=too-many-arguments
        # pylint: disable=too-many-positional-arguments
        """
        Initializes the ShortTermMemory instance.

        Args:
            model: The language model API for generating summaries.
            storage: The memory storage API.
            summary_system_prompt: The system prompt for creating the initial
                                   summary.
            summary_user_prompt: The user prompt for creating the initial
                                 summary.
            capacity: The maximum number of episodes to store.
            max_message_len: The maximum total length of all messages in
                             characters.
            max_token_num: The maximum total number of tokens for all
                           messages.
            memory_context: The context (group, agent, user, session) for the
                            memory.
        """
        self._model = model
        self._summary_user_prompt = summary_user_prompt
        self._summary_system_prompt = summary_system_prompt
        self._memory: deque[Episode] = deque(maxlen=capacity)
        self._capacity = capacity
        self._current_episode_count = 0
        self._max_message_len = max_message_len
        self._max_token_num = max_token_num
        self._current_message_len = 0
        self._current_token_num = 0
        self._summary = ""
        self._memory_context = memory_context
        self._summary_task = None
        self._lock = asyncio.Lock()

    def _is_full(self) -> bool:
        """
        Checks if the short-term memory has reached its capacity.

        Memory is considered full if the number of events, total message
        length, or total token count exceeds their respective maximums.

        Returns:
            True if the memory is full, False otherwise.
        """
        result = (
            self._current_episode_count >= self._capacity
            or self._current_message_len >= self._max_message_len
            or self._current_token_num >= self._max_token_num
        )
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
            self._memory.append(episode)

            self._current_episode_count += 1
            self._current_message_len += len(episode.content)
            self._current_token_num += self._compute_token_num(self._memory[-1])
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
        # do not clear the episode memory here so rolling episode can be
        #  used as context
        # just remove the episode that left over from previous evition.
        while len(self._memory) > self._current_episode_count:
            self._memory.popleft()

        for e in self._memory:
            result.append(e)
        self._current_episode_count = 0
        self._current_message_len = 0
        self._current_token_num = 0
        # if previous summary task is still running, wait for it
        if self._summary_task is not None:
            await self._summary_task
        self._summary_task = asyncio.create_task(self._create_summary(result))

    async def clear_memory(self):
        """
        Clears all events and the summary from the short-term memory.

        Resets the capacity, message length, and token count to zero.
        """
        async with self._lock:
            if self._summary_task is not None:
                self._summary_task.cancel()
            self._memory.clear()
            self._current_episode_count = 0
            self._current_message_len = 0
            self._current_token_num = 0
            self._summary = ""

    async def close(self):
        """Closes the memory, which currently just involves clearing it."""
        await self.clear_memory()

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
                episodes=episode_content, summary=self._summary
            )
            result = await self._model.generate_response(
                system_prompt=self._summary_system_prompt, user_prompt=msg
            )
            self._summary = result[0]
            logger.debug("Summary: %s\n", self._summary)
        except ExternalServiceAPIError:
            logger.info("External API error when creating summary")
        except ValueError:
            logger.info("Value error when creating summary")
        except RuntimeError:
            logger.info("Runtime error when creating summary")

    async def get_session_memory_context(
        self, query, limit: int = 0, max_token_num: int = 0
    ) -> tuple[list[Episode], str]:
        """
        Retrieves context from short-term memory for a given query.

        This includes the current summary and as many recent episodes as can
        fit within a specified token limit.

        Args:
            query: The user's query string.
            max_token_num: The maximum number of tokens for the context. If 0,
            no limit is applied.
        """
        logger.debug("Get session for %s", query)
        async with self._lock:
            if self._summary_task is not None:
                await self._summary_task
                self._summary_task = None
            length = (
                self._compute_token_num(self._summary)
                if self._summary is not None
                else 0
            )
            episodes: deque[Episode] = deque()
            for e in reversed(self._memory):
                if length >= max_token_num > 0:
                    break
                if len(episodes) >= limit > 0:
                    break
                token_num = self._compute_token_num(e)
                if length + token_num > max_token_num > 0:
                    break
                episodes.appendleft(e)
                length += token_num
            return list(episodes), self._summary

    def _compute_token_num(self, episode: Episode | str) -> int:
        """
        Computes the total number of tokens in an episodes.
        """
        result = 0
        if isinstance(episode, str):
            return int(len(episode) / 4)  # 4 character per token
        if episode.content is None:
            return 0
        if isinstance(episode.content, str):
            result += len(episode.content)
        else:
            result += len(repr(episode.content))
        if episode.user_metadata is None:
            return int(result / 4)  # 4 character per token
        if isinstance(episode.user_metadata, str):
            result += len(episode.user_metadata)
        elif isinstance(episode.user_metadata, dict):
            for _, v in episode.user_metadata.items():
                if isinstance(v, str):
                    result += len(v)
                else:
                    result += len(repr(v))
        return int(result / 4)  # 4 character per token
