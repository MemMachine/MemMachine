"""Core module for the Semantic Memory engine.

This module contains the `SemanticMemoryManager` class, which is the central component
for creating, managing, and searching feature sets based on their
conversation history. It integrates with language models for intelligent
information extraction and a vector database for semantic search capabilities.
"""

import asyncio
import logging
from typing import Any, Sequence

import numpy as np
from pydantic import BaseModel, InstanceOf, validate_call

from memmachine.common.embedder.embedder import Embedder
from memmachine.common.language_model.language_model import LanguageModel

from .semantic_ingestion import (
    SemanticCommand,
    SemanticFeature,
    llm_consolidate_features,
    llm_feature_update,
)
from .semantic_prompt import SemanticPrompt
from .semantic_tracker import SemanticUpdateTrackerManager
from .storage.storage_base import SemanticStorageBase
from .util.lru_cache import LRUCache

logger = logging.getLogger(__name__)


class SemanticMemoryManagerParams(BaseModel):
    model: InstanceOf[LanguageModel]
    embeddings: InstanceOf[Embedder]
    prompt: SemanticPrompt
    semantic_storage: InstanceOf[SemanticStorageBase]
    max_cache_size: int = 1000
    consolidation_threshold: int = 20

    feature_update_interval_sec: float = 2.0
    """ Interval in seconds for feature updates. This controls how often the
    background task checks for dirty sets and processes their
    conversation history to update features.
    """

    feature_update_message_limit: int = 5
    """ Number of messages after which a feature update is triggered.
    If a set sends this many messages, their features will be updated.
    """

    feature_update_time_limit_sec: float = 120.0
    """ Time in seconds after which a feature update is triggered.
    If a set has sent messages and this much time has passed since
    the first message, their features will be updated.
    """


class SemanticMemoryManager:
    # pylint: disable=too-many-instance-attributes
    """Manages and maintains semantic feature sets based on conversation history.

    This class uses a language model to intelligently extract, update, and
    consolidate information from conversations. It stores structured
    semantic data (features, values, tags) along with their vector embeddings in a
    persistent database, allowing for efficient semantic search.

    Key functionalities include:
    - Ingesting conversation messages to update semantic features.
    - Consolidating and deduplicating semantic entries to maintain accuracy and
      conciseness.
    - Providing CRUD operations for semantic data.
    - Performing semantic searches on features.
    - Caching frequently accessed features to improve performance.

    The process is largely asynchronous, designed to work within an async
    application.
    """

    def __init__(
        self,
        params: SemanticMemoryManagerParams,
    ):
        self._model = params.model
        self._embeddings = params.embeddings
        self._semantic_storage = params.semantic_storage

        self._prompt = params.prompt

        self._dirty_sets: SemanticUpdateTrackerManager = SemanticUpdateTrackerManager(
            message_limit=params.feature_update_message_limit,
            time_limit_sec=params.feature_update_time_limit_sec,
        )

        self._background_ingestion_interval_sec = params.feature_update_interval_sec

        self._ingestion_task = asyncio.create_task(self._background_ingestion_task())
        self._is_shutting_down = False

        self._semantic_cache = LRUCache(params.max_cache_size)

        self._consolidation_threshold = params.consolidation_threshold

    async def stop(self):
        """Releases resources, such as the database connection pool."""
        self._is_shutting_down = True
        await self._ingestion_task

    # === CRUD ===

    async def get_set_features(
        self,
        set_id: str,
    ):
        """Retrieves all features from a feature set, using a cache for performance.

        Args:
            set_id: The ID of the feature set.

        Returns:
            The feature data.
        """
        features = self._semantic_cache.get(set_id)
        if features is not None:
            return features
        features = await self._semantic_storage.get_set_features(set_id=set_id)
        self._semantic_cache.put(set_id, features)
        return features

    async def delete_all(self):
        """Deletes all set features from the database and clears the cache."""
        await self._semantic_storage.delete_all()
        self._semantic_cache.clean()

    async def delete_set_features(
        self,
        set_id: str,
    ):
        """Deletes a specific feature set.

        Args:
            set_id: The ID of the feature set whose features will be deleted.
        """
        self._semantic_cache.erase(set_id)
        await self._semantic_storage.delete_feature_set(set_id=set_id)

    async def add_new_feature(
        self,
        *,
        set_id: str,
        feature: str,
        value: str,
        tag: str,
        metadata: dict[str, str] | None = None,
        citations: list[int] | None = None,
    ):
        """Adds a new feature to a feature set.

        This invalidates the cache for the feature set.

        Args:
            set_id: The ID of the feature set to add the feature to.
            feature: The feature (e.g., "likes").
            value: The value for the feature (e.g., "dogs").
            tag: A category or tag for the feature.
            metadata: Additional metadata for the feature entry.
            citations: A list of message IDs that are sources for this feature.
        """
        if metadata is None:
            metadata = {}
        if citations is None:
            citations = []
        self._semantic_cache.erase(set_id)
        emb = (await self._embeddings.ingest_embed([value]))[0]
        await self._semantic_storage.add_feature(
            set_id=set_id,
            semantic_type_id="default",
            feature=feature,
            value=value,
            tag=tag,
            embedding=np.array(emb),
            metadata=metadata,
            citations=citations,
        )

    async def delete_set_feature(
        self,
        *,
        set_id: str,
        feature: str,
        tag: str,
        value: str | None = None,
    ):
        """Deletes a specific feature from a feature set.

        This invalidates the cache for the feature set.

        Args:
            set_id: The ID of the feature set to delete the feature from.
            feature: The feature to delete.
            tag: The tag of the feature to delete.
            value: The specific value to delete. If None, all values for the
                feature and tag are deleted.
        """
        self._semantic_cache.erase(set_id)
        await self._semantic_storage.delete_feature_with_filter(
            set_id=set_id,
            semantic_type_id="default",
            feature=feature,
            tag=tag,
        )

    @validate_call
    async def semantic_search(
        self,
        *,
        set_id: str,
        query: str,
        k: int = 1_000_000,
        min_cos: float = -1.0,
        max_range: float = 2.0,
        max_std: float = 1.0,
    ) -> list[Any]:
        """Performs a semantic search on a feature set based on a query string.

        Args:
            set_id: The ID of the feature set.
            query: The search query string.
            k: The maximum number of results to retrieve from the database.
            min_cos: The minimum cosine similarity for results.
            max_range: The maximum range for the `range_filter`.
            max_std: The maximum standard deviation for the `range_filter`.

        Returns:
            A list of matching feature entries, filtered by similarity scores.
        """
        qemb = (await self._embeddings.search_embed([query]))[0]
        candidates = await self._semantic_storage.semantic_search(
            set_id=set_id, qemb=np.array(qemb), k=k, min_cos=min_cos
        )
        formatted = [(i["metadata"]["similarity_score"], i) for i in candidates]

        def range_filter(
            arr: Sequence[tuple[float, object]], max_range: float, max_std: float
        ) -> list[object]:
            if not arr:
                return []

            first_f = arr[0][0]
            new_min = first_f - max_range

            s = 0.0
            sq = 0.0
            take = -1
            for d, (f, _) in enumerate(arr, start=1):
                s += f
                sq += f * f
                std = (
                    (sq - (s * s) / d) / d
                ) ** 0.5  # population stddev of prefix [0..d-1]
                if std < max_std:
                    take = d

            if take <= 0:
                return []
            return [val for (f, val) in arr[:take] if f > new_min]

        return range_filter(formatted, max_range, max_std)

    async def add_persona_message(
        self,
        *,
        set_id: str,
        content: str,
        metadata: dict[str, str] | None = None,
    ):
        """Adds a message to the history which will be used to update the feature set specified.

        After a certain number of messages (`_update_interval`), this method
        will trigger a feature update and consolidation process.

        Args:
            set_id: The ID of the feature set to update.
            content: The content of the message.
            metadata: Metadata associated with the message, such as the
                     speaker.

        Returns:
            A boolean indicating whether the consolidation process was awaited.
        """
        if metadata is None:
            metadata = {}

        if "speaker" in metadata:
            content = f"{metadata['speaker']} sends '{content}'"

        await self._semantic_storage.add_history(set_id, content, metadata)

        await self._dirty_sets.mark_update(set_id)

    async def uningested_message_count(self):
        return await self._semantic_storage.get_uningested_history_messages_count()

    async def _background_ingestion_task(self):
        while not self._is_shutting_down:
            dirty_sets = await self._dirty_sets.get_sets_to_update()

            if len(dirty_sets) == 0:
                await asyncio.sleep(self._background_ingestion_interval_sec)
                continue

            await asyncio.gather(
                *[self._process_uningested_memories(set_id) for set_id in dirty_sets]
            )

    async def _get_set_uningested_memories(self, set_id: str):
        rows = await self._semantic_storage.get_history_messages_by_ingestion_status(
            set_id=set_id,
            k=100,
            is_ingested=False,
        )
        return rows

    async def _process_uningested_memories(
        self,
        set_id: str,
    ):
        messages = await self._get_set_uningested_memories(set_id)

        if len(messages) == 0:
            return

        mark_messages = []

        for message in messages:
            await self._update_set_features_think(set_id, message)
            mark_messages.append(message["id"])

        await asyncio.gather(
            self._consolidate_memories_if_applicable(set_id=set_id),
            self._semantic_storage.mark_messages_ingested(ids=mark_messages),
        )

    async def _update_set_features_think(
        self,
        set_id: str,
        record: Any,
    ):
        """
        update set features based on json output, after doing a chain
        of thought.
        """
        # TODO: These really should not be raw data structures.
        citation_id = record["id"]  # Think this is an int

        features = await self.get_set_features(set_id)
        message_content = record["content"]

        try:
            commands = await llm_feature_update(
                features=features,
                message_content=message_content,
                model=self._model,
                update_prompt=self._prompt.update_prompt,
            )
        except (ValueError, TypeError) as e:
            logger.error("Failed to update features while calling LLM", e)
            return

        await self._apply_commands(
            commands=commands,
            set_id=set_id,
            citation_id=citation_id,
        )

    async def _apply_commands(
        self,
        *,
        commands: list[SemanticCommand],
        set_id: str,
        citation_id: int,
    ):
        for command in commands:
            match command.command:
                case "add":
                    await self.add_new_feature(
                        set_id=set_id,
                        feature=command.feature,
                        value=command.value,
                        tag=command.tag,
                        citations=[citation_id],
                    )
                case "delete":
                    await self.delete_set_feature(
                        set_id=set_id,
                        feature=command.feature,
                        tag=command.tag,
                        value=command.value,
                    )
                case _:
                    logger.error("Command with unknown action: %s", command.command)


    async def _store_consolidated_memory(self, *, memory: SemanticFeature, set_id: str):
        # TODO: Validate that this association citation logic is correct.

        if memory.metadata.citations is None:
            logger.error("No citations passed for store_consolidated_memory")
            new_citations = []
        else:
            associations = await self._semantic_storage.get_all_citations_for_ids(
                memory.metadata.citations
            )

            new_citations = [i for i in associations]

        logger.debug(
            "CITATION_CHECK",
            extra={
                "content_citations": new_citations,
                "feature_citations": memory.metadata.citations,
            },
        )

        await self.add_new_feature(
            set_id=set_id,
            feature=memory.feature,
            value=memory.value,
            tag=memory.tag,
            citations=new_citations,
        )

    async def _consolidate_memories_if_applicable(self, *, set_id: str):
        s = await self._semantic_storage.get_large_feature_sections(
            set_id=set_id, thresh=self._consolidation_threshold
        )
        await asyncio.gather(
            *[
                self._deduplicate_features(
                    set_id, [SemanticFeature(**memories) for memories in section]
                )
                for section in s
            ]
        )

    async def _deduplicate_features(
        self,
        set_id: str,
        memories: list[SemanticFeature],
    ):
        """
        sends a list of features to a llm to consolidate
        """
        try:
            consolidate_resp = await llm_consolidate_features(
                memories=memories,
                model=self._model,
                consolidate_prompt=self._prompt.consolidation_prompt,
            )
        except (ValueError, TypeError) as e:
            logger.error("Failed to update features while calling LLM", e)
            return

        if consolidate_resp is None or consolidate_resp.keep_memories is None:
            logger.warning("Failed to consolidate features")
            return

        memory_ids_to_delete = [
            m.metadata.id
            for m in memories
            if m.metadata.id is not None
            and m.metadata.id not in consolidate_resp.keep_memories
        ]
        await self._semantic_storage.delete_features(memory_ids_to_delete)
        self._semantic_cache.erase(set_id)
