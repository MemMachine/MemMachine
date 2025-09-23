"""Core module for the Profile Memory engine.

This module contains the `ProfileMemory` class, which is the central component
for creating, managing, and searching user profiles based on their
conversation history. It integrates with language models for intelligent
information extraction and a vector database for semantic search capabilities.
"""

import asyncio
import json
import logging
from itertools import accumulate, tee
from types import ModuleType
from typing import Any

import numpy as np

from memmachine.common.embedder.embedder import Embedder
from memmachine.common.language_model.language_model import LanguageModel

from memmachine.tracing import get_tracer, init_tracer

from .storage.asyncpg_profile import AsyncPgProfileStorage
from .util.lru_cache import LRUCache

logger = logging.getLogger(__name__)
# logger.addHandler(MultiLineStreamHandler())


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
        db_config (dict[str, Any]): Configuration for the database connection.
        prompt_module (ModuleType): A Python module containing system prompts
            ('UPDATE_PROMPT', 'CONSOLIDATION_PROMPT').
        max_cache_size (int, optional): Max size for the profile LRU cache.
            Defaults to 1000.
    """

    def __init__(
        self,
        model: LanguageModel,
        embeddings: Embedder,
        db_config: dict[str, Any],
        prompt_module: ModuleType,
        max_cache_size=1000,
    ):
        # pylint: disable=too-many-arguments
        # pylint: disable=too-many-positional-arguments
        # add model initialization
        self._model = model
        self._embeddings = embeddings

        self._max_cache_size = max_cache_size
        self._update_prompt = getattr(prompt_module, "UPDATE_PROMPT", "")
        self._consolidation_prompt = getattr(
            prompt_module, "CONSOLIDATION_PROMPT", ""
        )
        self._update_prompt_data = """The old version persona profile is
            {profile}.\nThe conversation history is {session}.\n"""
        self._profile_storage = AsyncPgProfileStorage(db_config)
        self._update_interval = 1
        self._msg_count: dict[str, int] = {}
        self._profile_cache = LRUCache(self._max_cache_size)

    async def startup(self):
        """Initializes resources, such as the database connection pool."""
        await self._profile_storage.startup()

    async def cleanup(self):
        """Releases resources, such as the database connection pool."""
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
        formatted = [
            (i["metadata"]["similarity_score"], i) for i in candidates
        ]
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
        parent_span: None = None,  # type: ignore
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

        Returns:
            A boolean indicating whether the consolidation process was awaited.
        """
        tracer = get_tracer()
        with tracer.start_active_span('add_persona_message', child_of=parent_span) as scope:
            span = scope.span
        
            # TODO: add or adopt system for more general modifications of
            # pylint: disable=fixme
            # the message
            if metadata is None:
                metadata = {}
            if isolations is None:
                isolations = {}
            if "speaker" in metadata:
                content = f"{metadata['speaker']} sends '{content}'"
            message = await self._profile_storage.add_history(
                user_id, content, metadata, isolations
            )
            self._msg_count[user_id] = self._msg_count.get(user_id, 0) + 1
            wait_consolidate = False
            if self._msg_count[user_id] >= self._update_interval:
                self._msg_count[user_id] = 0
                wait_consolidate = True
            fut = asyncio.create_task(
                self._update_user_profile_think(
                    message, wait_consolidate=wait_consolidate
                )
            )
            if wait_consolidate:
                await fut
            return wait_consolidate

    async def _reconsolidate_memory(
        self,
        user_id: str,
        isolations: dict[str, bool | int | float | str] | None = None,
    ):
        """request re-ingestion of session data to profile"""
        if isolations is None:
            isolations = {}

        messages = await self._profile_storage.get_last_history_messages(
            user_id=user_id, k=1_000_000, isolations=isolations
        )
        for i in range(0, len(messages), 10):
            chunk = messages[i : i + 10]
            await asyncio.gather(
                *[self._update_user_profile_think(msg) for msg in chunk]
            )

    async def _update_user_profile_think(
        self,
        record: Any,
        wait_consolidate: bool = False,
    ):
        """
        update user profile based on json output, after doing a chain
        of thought.
        """
        citation_id = record["id"]
        user_id = record["user_id"]
        isolations = json.loads(record["isolations"])
        # metadata = json.loads(record["metadata"])
        profile = await self.get_user_profile(user_id, isolations)
        userp = self._update_prompt_data.format(
            profile=str(profile), session=record["content"]
        )

        raw = await self._model.generate_response(
            system_prompt=self._update_prompt, user_prompt=userp
        )

        def get_suffix(a: str):
            idx = a.find("</think>")
            if idx == -1:
                return None, a
            return (a[8:idx], a[(idx + 8) :])

        pfx, sfx = get_suffix(raw[0])
        result = {}
        try:
            result = json.loads(sfx)
        except ValueError as e:
            logger.error(
                "unable to format ai output %s, Error %s", str(sfx), str(e)
            )
            result = {}
        logger.info(
            "PROFILE MEMORY INGESTOR",
            extra={
                "queries_to_ingest": record["content"],
                "thoughts": pfx,
                "outputs": result,
            },
        )
        for command in result.values():
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
                v = command["value"] if "value" in command else None
                await self.delete_user_profile_feature(
                    user_id,
                    command["feature"],
                    command["tag"],
                    value=v,
                    isolations=isolations,
                )
            else:
                logger.error("AI tool call formatting incorrect")
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

        raw = await self._model.generate_response(
            system_prompt=self._consolidation_prompt,
            user_prompt=json.dumps(memories),
        )

        def get_suffix(a: str):
            idx = a.find("</think>")
            if idx == -1:
                return None, a
            return (a[8:idx], a[(idx + 8) :])

        pfx, sfx = get_suffix(raw[0])
        result = {}
        try:
            result = json.loads(sfx)
        except ValueError as e:
            logger.error(
                "unable to format ai output %s. Error %s", str(sfx), str(e)
            )
            result = {}
        logger.info(
            "PROFILE MEMORY CONSOLIDATOR",
            extra={"receives": memories, "thoughts": pfx, "outputs": result},
        )
        for memory in result["consolidate_memories"]:
            associations = (
                await self._profile_storage.get_all_citations_for_ids(
                    memory["metadata"]["citations"]
                )
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
                    "profile_citations": memory["metadata"]["citations"],
                    "think": pfx,
                },
            )
            await self.add_new_profile(
                user_id,
                memory["feature"],
                memory["value"],
                memory["tag"],
                citations=new_citations,
                isolations=new_isolations,
            )
        for memory in memories:
            if memory["metadata"]["id"] not in result["keep_memories"]:
                self._profile_cache.erase(user_id)
                await self._profile_storage.delete_profile_feature_by_id(
                    memory["metadata"]["id"]
                )
