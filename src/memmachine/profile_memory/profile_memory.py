"""Core module for the Profile Memory engine.

This module contains the `ProfileMemory` class, which is the central component
for creating, managing, and searching user profiles based on their
conversation history. It integrates with language models for intelligent
information extraction and a vector database for semantic search capabilities.
"""

import asyncio
import datetime
import json
import logging
from itertools import accumulate, groupby, tee
from typing import Any

import numpy as np
from pydantic import BaseModel

from memmachine.common.data_types import ExternalServiceAPIError
from memmachine.common.embedder.embedder import Embedder
from memmachine.common.language_model.language_model import LanguageModel

from .prompt_provider import ProfilePrompt
from .storage.storage_base import ProfileStorageBase
from .util.lru_cache import LRUCache

logger = logging.getLogger(__name__)


class ProfileUpdateTracker:
    """Tracks profile update activity for a user.
    When a user sends messages, this class keeps track of how many
    messages have been sent and when the first message was sent.
    This is used to determine when to trigger profile updates based
    on message count and time intervals.
    """

    def __init__(self, user: str, message_limit: int, time_limit_sec: float):
        self._user = user
        self._message_limit: int = message_limit
        self._time_limit: float = time_limit_sec
        self._message_count: int = 0
        self._first_updated: datetime.datetime | None = None

    def mark_update(self):
        """Marks that a new message has been sent by the user.
        Increments the message count and sets the first updated time
        if this is the first message.
        """
        self._message_count += 1
        if self._first_updated is None:
            self._first_updated = datetime.datetime.now()

    def _seconds_from_first_update(self) -> float | None:
        """Returns the number of seconds since the first message was sent.
        If no messages have been sent, returns None.
        """
        if self._first_updated is None:
            return None
        delta = datetime.datetime.now() - self._first_updated
        return delta.total_seconds()

    def reset(self):
        """Resets the tracker state.
        Clears the message count and first updated time.
        """
        self._message_count = 0
        self._first_updated = None

    def should_update(self) -> bool:
        """Determines if a profile update should be triggered.
        A profile update is triggered if either the message count
        exceeds the limit or the time since the first message exceeds
        the time limit.

        Returns:
            bool: True if a profile update should be triggered, False otherwise.
        """
        if self._message_count == 0:
            return False
        elapsed = self._seconds_from_first_update()
        exceed_time_limit = elapsed is not None and elapsed >= self._time_limit
        exceed_msg_limit = self._message_count >= self._message_limit
        return exceed_time_limit or exceed_msg_limit


class ProfileUpdateTrackerManager:
    """Manages ProfileUpdateTracker instances for multiple users."""

    def __init__(self, message_limit: int, time_limit_sec: float):
        self._trackers: dict[str, ProfileUpdateTracker] = {}
        self._trackers_lock = asyncio.Lock()
        self._message_limit = message_limit
        self._time_limit_sec = time_limit_sec

    def _new_tracker(self, user: str) -> ProfileUpdateTracker:
        return ProfileUpdateTracker(
            user=user,
            message_limit=self._message_limit,
            time_limit_sec=self._time_limit_sec,
        )

    async def mark_update(self, user: str):
        """Marks that a new message has been sent by the user.
        Creates a new tracker if one does not exist for the user.
        """
        async with self._trackers_lock:
            if user not in self._trackers:
                self._trackers[user] = self._new_tracker(user)
            self._trackers[user].mark_update()

    async def get_users_to_update(self) -> list[str]:
        """Returns a list of users whose profiles need to be updated.
        A profile update is needed if the user's tracker indicates
        that an update should be triggered.
        """
        async with self._trackers_lock:
            ret = []
            for user, tracker in self._trackers.items():
                if tracker.should_update():
                    ret.append(user)
                    tracker.reset()
            return ret


class ProfileMemory:
    # pylint: disable=too-many-instance-attributes
    """Manages and maintains user profiles based on conversation history.

    This class uses a language model to intelligently extract, update, and
    consolidate user profile information from conversations. It stores structured
    profile data (features, values, tags) along with their vector embeddings in a
    persistent database, allowing for efficient semantic search.

    Key functionalities include:
    - Ingesting conversation messages to update profiles.
    - Consolidating and deduplicating profile entries to maintain accuracy and
      conciseness.
    - Providing CRUD operations for profile data.
    - Performing semantic searches on user profiles.
    - Caching frequently accessed profiles to improve performance.

    The process is largely asynchronous, designed to work within an async
    application.

    Args:
        model (LanguageModel): The language model for profile extraction.
        embeddings (Embedder): The model for generating vector embeddings.
        profile_storage (ProfileStorageBase): Connection to the profile database.
        prompt (ProfilePrompt): The system prompts to be used.
        max_cache_size (int, optional): Max size for the profile LRU cache.
            Defaults to 1000.
    """

    PROFILE_UPDATE_INTERVAL_SEC = 2
    """ Interval in seconds for profile updates. This controls how often the
    background task checks for dirty users and processes their
    conversation history to update profiles.
    """

    PROFILE_UPDATE_MESSAGE_LIMIT = 5
    """ Number of messages after which a profile update is triggered.
    If a user sends this many messages, their profile will be updated.
    """

    PROFILE_UPDATE_TIME_LIMIT_SEC = 120.0
    """ Time in seconds after which a profile update is triggered.
    If a user has sent messages and this much time has passed since
    the first message, their profile will be updated.
    """

    def __init__(
        self,
        *,
        model: LanguageModel,
        embeddings: Embedder,
        prompt: ProfilePrompt,
        max_cache_size=1000,
        profile_storage: ProfileStorageBase,
    ):
        if model is None:
            raise ValueError("model must be provided")
        if embeddings is None:
            raise ValueError("embeddings must be provided")
        if prompt is None:
            raise ValueError("prompt must be provided")
        if profile_storage is None:
            raise ValueError("profile_storage must be provided")

        self._model = model
        self._embeddings = embeddings
        self._profile_storage = profile_storage

        self._max_cache_size = max_cache_size

        self._update_prompt = prompt.update_prompt
        self._consolidation_prompt = prompt.consolidation_prompt

        self._update_interval = 1
        self._dirty_users: ProfileUpdateTrackerManager = ProfileUpdateTrackerManager(
            message_limit=self.PROFILE_UPDATE_MESSAGE_LIMIT,
            time_limit_sec=self.PROFILE_UPDATE_TIME_LIMIT_SEC,
        )
        self._ingestion_task = asyncio.create_task(self._background_ingestion_task())
        self._is_shutting_down = False
        self._profile_cache = LRUCache(self._max_cache_size)

    async def startup(self):
        """Initializes resources, such as the database connection pool."""
        await self._profile_storage.startup()

    async def cleanup(self):
        """Releases resources, such as the database connection pool."""
        self._is_shutting_down = True
        await self._ingestion_task
        await self._profile_storage.cleanup()

    # === CRUD ===

    async def get_user_profile(
        self,
        user_id: str,
        isolations: dict[str, bool | int | float | str] | None = None,
    ):
        """Retrieves a user's profile, using a cache for performance.

        Args:
            user_id: The ID of the user.
            isolations: A dictionary for data isolation.

        Returns:
            The user's profile data.
        """
        if isolations is None:
            isolations = {}
        profile = self._profile_cache.get((user_id, json.dumps(isolations)))
        if profile is not None:
            return profile
        profile = await self._profile_storage.get_profile(user_id, isolations)
        self._profile_cache.put((user_id, json.dumps(isolations)), profile)
        return profile

    async def delete_all(self):
        """Deletes all user profiles from the database and clears the cache."""
        self._profile_cache = LRUCache(self._max_cache_size)
        await self._profile_storage.delete_all()

    async def delete_user_profile(
        self,
        user_id: str,
        isolations: dict[str, bool | int | float | str] | None = None,
    ):
        """Deletes a specific user's profile.

        Args:
            user_id: The ID of the user whose profile will be deleted.
            isolations: A dictionary for data isolation.
        """
        if isolations is None:
            isolations = {}
        self._profile_cache.erase((user_id, json.dumps(isolations)))
        await self._profile_storage.delete_profile(user_id, isolations)

    async def add_new_profile(
        self,
        user_id: str,
        feature: str,
        value: str,
        tag: str,
        metadata: dict[str, str] | None = None,
        isolations: dict[str, bool | int | float | str] | None = None,
        citations: list[int] | None = None,
    ):
        # pylint: disable=too-many-arguments
        # pylint: disable=too-many-positional-arguments
        """Adds a new feature to a user's profile.

        This invalidates the cache for the user's profile.

        Args:
            user_id: The ID of the user.
            feature: The profile feature (e.g., "likes").
            value: The value for the feature (e.g., "dogs").
            tag: A category or tag for the feature.
            metadata: Additional metadata for the profile entry.
            isolations: A dictionary for data isolation.
            citations: A list of message IDs that are sources for this feature.
        """
        if isolations is None:
            isolations = {}
        if metadata is None:
            metadata = {}
        if citations is None:
            citations = []
        self._profile_cache.erase((user_id, json.dumps(isolations)))
        emb = (await self._embeddings.ingest_embed([value]))[0]
        await self._profile_storage.add_profile_feature(
            user_id,
            feature,
            value,
            tag,
            np.array(emb),
            metadata=metadata,
            isolations=isolations,
            citations=citations,
        )

    async def delete_user_profile_feature(
        self,
        user_id: str,
        feature: str,
        tag: str,
        value: str | None = None,
        isolations: dict[str, bool | int | float | str] | None = None,
    ):
        """Deletes a specific feature from a user's profile.

        This invalidates the cache for the user's profile.

        Args:
            user_id: The ID of the user.
            feature: The profile feature to delete.
            tag: The tag of the feature to delete.
            value: The specific value to delete. If None, all values for the
                feature and tag are deleted.
            isolations: A dictionary for data isolation.
        """
        if isolations is None:
            isolations = {}
        self._profile_cache.erase((user_id, json.dumps(isolations)))
        await self._profile_storage.delete_profile_feature(
            user_id, feature, tag, value, isolations
        )

    def range_filter(
        self, arr: list[tuple[float, Any]], max_range: float, max_std: float
    ) -> list[Any]:
        """
        Filters a list of semantically searched entries based on similarity.

        Finds the longest prefix of the list of entries returned by
        semantic_search such that:
         - The difference between max and min similarity is at most
           `max_range`.
         - The standard deviation of similarity scores is at most `max_std`.

        Args:
            arr: A list of tuples, where each tuple contains a similarity
                score and the corresponding entry.
            max_range: The maximum allowed range between the highest and lowest
                similarity scores.
            max_std: The maximum allowed standard deviation of similarity
                     scores.

        Returns:
            A filtered list of entries.
        """
        if len(arr) == 0:
            return []
        new_min = arr[0][0] - max_range
        k, v = zip(*arr)
        k1, k2, _, k4 = tee(k, 4)
        sums = accumulate(k1)
        square_sums = accumulate(i * i for i in k2)
        divs = range(1, len(arr) + 1)
        take = max(
            (d if ((sq - s * s / d) / d) ** 0.5 < max_std else -1)
            for (s, sq, d) in zip(sums, square_sums, divs)
        )
        return [val for (f, val, _) in zip(k4, v, range(take)) if f > new_min]

    async def semantic_search(
        self,
        query: str,
        k: int = 1_000_000,
        min_cos: float = -1.0,
        max_range: float = 2.0,
        max_std: float = 1.0,
        isolations: dict[str, bool | int | float | str] | None = None,
        user_id: str = "",
    ) -> list[Any]:
        """Performs a semantic search on a user's profile.

        Args:
            user_id: The ID of the user.
            query: The search query string.
            k: The maximum number of results to retrieve from the database.
            min_cos: The minimum cosine similarity for results.
            max_range: The maximum range for the `range_filter`.
            max_std: The maximum standard deviation for the `range_filter`.
            isolations: A dictionary for data isolation.

        Returns:
            A list of matching profile entries, filtered by similarity scores.
        """
        # TODO: cache this # pylint: disable=fixme
        if isolations is None:
            isolations = {}
        qemb = (await self._embeddings.search_embed([query]))[0]
        candidates = await self._profile_storage.semantic_search(
            user_id, np.array(qemb), k, min_cos, isolations
        )
        formatted = [(i["metadata"]["similarity_score"], i) for i in candidates]
        return self.range_filter(formatted, max_range, max_std)

    async def get_large_profile_sections(
        self,
        user_id: str,
        thresh: int = 5,
        isolations: dict[str, bool | int | float | str] | None = None,
    ) -> list[list[dict[str, Any]]]:
        """Retrieves profile sections with a large number of entries.

        A "section" is a group of profile entries with the same feature and
        tag. This is used to find sections that may need consolidation.

        Args:
            user_id: The ID of the user.
            thresh: The minimum number of entries for a section to be
                considered "large".
            isolations: A dictionary for data isolation.

        Returns:
            A list of large profile sections, where each section is a list of
            profile entries.
        """
        # TODO: useless wrapper. delete? # pylint: disable=fixme
        if isolations is None:
            isolations = {}
        return await self._profile_storage.get_large_profile_sections(
            user_id, thresh, isolations
        )

    # === Profile Ingestion ===
    async def add_persona_message(
        self,
        content: str,
        metadata: dict[str, str] | None = None,
        isolations: dict[str, bool | int | float | str] | None = None,
        user_id: str = "",  # TODO fully deprecate user_id parameter
    ):
        """Adds a message to the history and may trigger a profile update.

        After a certain number of messages (`_update_interval`), this method
        will trigger a profile update and consolidation process.

        Args:
            user_id: The ID of the user.
            content: The content of the message.
            metadata: Metadata associated with the message, such as the
                     speaker.
            isolations: A dictionary for data isolation.
            wait_consolidate: If true, wait for consolidation to finish
                     before returning.

        Returns:
            A boolean indicating whether the consolidation process was awaited.
        """
        # TODO: add or adopt system for more general modifications of
        # pylint: disable=fixme
        # the message
        if metadata is None:
            metadata = {}
        if isolations is None:
            isolations = {}

        if "speaker" in metadata:
            content = f"{metadata['speaker']} sends '{content}'"

        await self._profile_storage.add_history(user_id, content, metadata, isolations)

        await self._dirty_users.mark_update(user_id)

    async def uningested_message_count(self):
        return await self._profile_storage.get_uningested_history_messages_count()

    async def _background_ingestion_task(self):
        while not self._is_shutting_down:
            dirty_users = await self._dirty_users.get_users_to_update()

            if len(dirty_users) == 0:
                await asyncio.sleep(self.PROFILE_UPDATE_INTERVAL_SEC)
                continue

            await asyncio.gather(
                *[self._process_uningested_memories(user_id) for user_id in dirty_users]
            )

    async def _get_isolation_grouped_memories(self, user_id: str):
        rows = await self._profile_storage.get_history_messages_by_ingestion_status(
            user_id=user_id,
            k=100,
            is_ingested=False,
        )

        def key_fn(r):
            # normalize JSONB dict to a stable string key
            return json.dumps(r["isolations"], sort_keys=True)

        rows = sorted(rows, key=key_fn)
        return [list(group) for _, group in groupby(rows, key_fn)]

    async def _process_uningested_memories(
        self,
        user_id: str,
    ):
        message_isolation_groups = await self._get_isolation_grouped_memories(user_id)

        async def process_messages(messages):
            if len(messages) == 0:
                return

            mark_tasks = []

            for i in range(0, len(messages) - 1):
                message = messages[i]
                await self._update_user_profile_think(message)
                mark_tasks.append(
                    self._profile_storage.mark_messages_ingested([message["id"]])
                )

            await self._update_user_profile_think(messages[-1], wait_consolidate=True)
            mark_tasks.append(
                self._profile_storage.mark_messages_ingested([messages[-1]["id"]])
            )
            await asyncio.gather(*mark_tasks)

        tasks = []
        for isolation_messages in message_isolation_groups:
            tasks.append(process_messages(isolation_messages))

        await asyncio.gather(*tasks)

    async def _update_user_profile_think(
        self,
        record: Any,
        wait_consolidate: bool = False,
    ):
        """
        update user profile based on json output, after doing a chain
        of thought.
        """
        # TODO: These really should not be raw data structures.
        citation_id = record["id"]  # Think this is an int
        user_id = record["user_id"]
        isolations = json.loads(record["isolations"])
        # metadata = json.loads(record["metadata"])

        profile = await self.get_user_profile(user_id, isolations)
        memory_content = record["content"]

        user_prompt = (
            "The old profile is provided below:\n"
            "<OLD_PROFILE>\n"
            "{profile}\n"
            "</OLD_PROFILE>\n"
            "\n"
            "The history is provided below:\n"
            "<HISTORY>\n"
            "{memory_content}\n"
            "</HISTORY>\n"
        ).format(
            profile=str(profile),
            memory_content=memory_content,
        )
        # Use chain-of-thought to get entity profile update commands.
        try:
            response_text, _ = await self._model.generate_response(
                system_prompt=self._update_prompt, user_prompt=user_prompt
            )
        except (ExternalServiceAPIError, ValueError, RuntimeError) as e:
            logger.error("Eror when update profile: %s", str(e))
            return

        # Get thinking and JSON from language model response.
        thinking, _, response_json = response_text.removeprefix("<think>").rpartition(
            "</think>"
        )
        thinking = thinking.strip()

        # TODO: These really should not be raw data structures.
        try:
            profile_update_commands = json.loads(response_json)
        except ValueError as e:
            logger.warning(
                "Unable to load language model output '%s' as JSON, Error %s: "
                "Proceeding with no profile update commands",
                str(response_json),
                str(e),
            )
            profile_update_commands = {}
            return
        finally:
            logger.info(
                "PROFILE MEMORY INGESTOR",
                extra={
                    "queries_to_ingest": memory_content,
                    "thoughts": thinking,
                    "outputs": profile_update_commands,
                },
            )

        # This should probably just be a list of commands
        # instead of a dictionary mapping
        # from integers in strings (not even bare ints!)
        # to commands.
        # TODO: Consider improving this design in a breaking change.
        if not isinstance(profile_update_commands, dict):
            logger.warning(
                "AI response format incorrect: expected dict, got %s %s",
                type(profile_update_commands).__name__,
                profile_update_commands,
            )
            return

        commands = profile_update_commands.values()

        valid_commands = []
        for command in commands:
            if not isinstance(command, dict):
                logger.warning(
                    "AI response format incorrect: "
                    "expected profile update command to be dict, got %s %s",
                    type(command).__name__,
                    command,
                )
                continue

            if "command" not in command:
                logger.warning(
                    "AI response format incorrect: missing 'command' key in %s",
                    command,
                )
                continue

            if command["command"] not in ("add", "delete"):
                logger.warning(
                    "AI response format incorrect: "
                    "expected 'command' value in profile update command "
                    "to be 'add' or 'delete', got '%s'",
                    command["command"],
                )
                continue

            if "feature" not in command:
                logger.warning(
                    "AI response format incorrect: missing 'feature' key in %s",
                    command,
                )
                continue

            if "tag" not in command:
                logger.warning(
                    "AI response format incorrect: missing 'tag' key in %s",
                    command,
                )
                continue

            if command["command"] == "add" and "value" not in command:
                logger.warning(
                    "AI response format incorrect: missing 'value' key in %s",
                    command,
                )
                continue

            valid_commands.append(command)

        for command in valid_commands:
            if command["command"] == "add":
                await self.add_new_profile(
                    user_id,
                    command["feature"],
                    command["value"],
                    command["tag"],
                    citations=[citation_id],
                    isolations=isolations,
                    # metadata=metadata
                )
            elif command["command"] == "delete":
                value = command["value"] if "value" in command else None
                await self.delete_user_profile_feature(
                    user_id,
                    command["feature"],
                    command["tag"],
                    value=value,
                    isolations=isolations,
                )
            else:
                logger.error("Command with unknown action: %s", command["command"])
                raise ValueError(
                    "Command with unknown action: " + str(command["command"])
                )

        if wait_consolidate:
            s = await self.get_large_profile_sections(
                user_id, thresh=5, isolations=isolations
            )
            await asyncio.gather(
                *[self._deduplicate_profile(user_id, section) for section in s]
            )

    async def _deduplicate_profile(
        self,
        user_id: str,
        memories: list[dict[str, Any]],
    ):
        """
        sends a list of features to an llm to consolidated
        """
        try:
            response_text, _ = await self._model.generate_response(
                system_prompt=self._consolidation_prompt,
                user_prompt=json.dumps(memories),
            )
        except (ExternalServiceAPIError, ValueError, RuntimeError) as e:
            logger.error("Model Error when deduplicate profile: %s", str(e))
            return

        # Get thinking and JSON from language model response.
        thinking, _, response_json = response_text.removeprefix("<think>").rpartition(
            "</think>"
        )
        thinking = thinking.strip()

        try:
            updated_profile_entries = json.loads(response_json)
        except ValueError as e:
            logger.warning(
                "Unable to load language model output '%s' as JSON, Error %s",
                str(response_json),
                str(e),
            )
            updated_profile_entries = {}
            return
        finally:
            logger.info(
                "PROFILE MEMORY CONSOLIDATOR",
                extra={
                    "receives": memories,
                    "thoughts": thinking,
                    "outputs": updated_profile_entries,
                },
            )

        if not isinstance(updated_profile_entries, dict):
            logger.warning(
                "AI response format incorrect: expected dict, got %s %s",
                type(updated_profile_entries).__name__,
                updated_profile_entries,
            )
            return

        if "consolidate_memories" not in updated_profile_entries:
            logger.warning(
                "AI response format incorrect: "
                "missing 'consolidate_memories' key, got %s",
                updated_profile_entries,
            )
            updated_profile_entries["consolidate_memories"] = []

        keep_all_memories = False

        if "keep_memories" not in updated_profile_entries:
            logger.warning(
                "AI response format incorrect: missing 'keep_memories' key, got %s",
                updated_profile_entries,
            )
            updated_profile_entries["keep_memories"] = []
            keep_all_memories = True

        consolidate_memories = updated_profile_entries["consolidate_memories"]
        keep_memories = updated_profile_entries["keep_memories"]

        if not isinstance(consolidate_memories, list):
            logger.warning(
                "AI response format incorrect: "
                "'consolidate_memories' value is not a list, got %s %s",
                type(consolidate_memories).__name__,
                consolidate_memories,
            )
            consolidate_memories = []
            keep_all_memories = True

        if not isinstance(keep_memories, list):
            logger.warning(
                "AI response format incorrect: "
                "'keep_memories' value is not a list, got %s %s",
                type(keep_memories).__name__,
                keep_memories,
            )
            keep_memories = []
            keep_all_memories = True

        if not keep_all_memories:
            valid_keep_memories = []
            for memory_id in keep_memories:
                if not isinstance(memory_id, int):
                    logger.warning(
                        "AI response format incorrect: "
                        "expected int memory id in 'keep_memories', got %s %s",
                        type(memory_id).__name__,
                        memory_id,
                    )
                    continue

                valid_keep_memories.append(memory_id)

            for memory in memories:
                if memory["metadata"]["id"] not in valid_keep_memories:
                    self._profile_cache.erase(user_id)
                    await self._profile_storage.delete_profile_feature_by_id(
                        memory["metadata"]["id"]
                    )

        class ConsolidateMemoryMetadata(BaseModel):
            citations: list[int]

        class ConsolidateMemory(BaseModel):
            tag: str
            feature: str
            value: str
            metadata: ConsolidateMemoryMetadata

        for memory in consolidate_memories:
            try:
                consolidate_memory = ConsolidateMemory(**memory)
            except Exception as e:
                logger.warning(
                    "AI response format incorrect: unable to parse memory %s, error %s",
                    memory,
                    str(e),
                )
                continue

            associations = await self._profile_storage.get_all_citations_for_ids(
                consolidate_memory.metadata.citations
            )

            new_citations = [i[0] for i in associations]

            # a derivative shall contain all routing information of its
            # components that do not mutually conflict.
            new_isolations: dict[str, bool | int | float | str] = {}
            bad = set()
            for i in associations:
                for k, v in i[1].items():
                    old_val = new_isolations.get(k)
                    if old_val is None:
                        new_isolations[k] = v
                    elif old_val != v:
                        bad.add(k)
            for k in bad:
                del new_isolations[k]
            logger.debug(
                "CITATION_CHECK",
                extra={
                    "content_citations": new_citations,
                    "profile_citations": consolidate_memory.metadata.citations,
                    "think": thinking,
                },
            )
            await self.add_new_profile(
                user_id,
                consolidate_memory.feature,
                consolidate_memory.value,
                consolidate_memory.tag,
                citations=new_citations,
                isolations=new_isolations,
            )
