"""Ingestion pipeline for converting episodes into semantic features."""

import asyncio
import itertools
import logging
from collections.abc import AsyncIterator, Mapping, MutableMapping, Sequence
from itertools import chain

import numpy as np
from pydantic import BaseModel, Field, InstanceOf, TypeAdapter

from memmachine.common.embedder import Embedder
from memmachine.common.episode_store import Episode, EpisodeIdT, EpisodeStorage
from memmachine.common.filter.filter_parser import And, Comparison, Or
from memmachine.semantic_memory.cluster_manager import (
    ClusterManager,
    ClusterParams,
    ClusterState,
)
from memmachine.semantic_memory.cluster_store.cluster_store import ClusterStateStorage
from memmachine.semantic_memory.cluster_store.in_memory_cluster_store import (
    InMemoryClusterStateStorage,
)
from memmachine.semantic_memory.semantic_llm import (
    LLMReducedFeature,
    llm_consolidate_features,
    llm_feature_update,
)
from memmachine.semantic_memory.semantic_model import (
    ResourceRetrieverT,
    Resources,
    SemanticCategory,
    SemanticCommand,
    SemanticCommandType,
    SemanticFeature,
    SetIdT,
)
from memmachine.semantic_memory.storage.storage_base import SemanticStorage

logger = logging.getLogger(__name__)
_CLUSTER_METADATA_KEY = "cluster_id"


class IngestionService:
    """
    Processes un-ingested history for each set_id and updates semantic features.

    The service pulls pending messages, invokes the LLM to generate mutation commands,
    applies the resulting changes, and optionally consolidates redundant memories.
    """

    class Params(BaseModel):
        """Dependencies and tuning knobs for the ingestion workflow."""

        semantic_storage: InstanceOf[SemanticStorage]
        history_store: InstanceOf[EpisodeStorage]
        resource_retriever: ResourceRetrieverT
        consolidated_threshold: int = 20
        debug_fail_loudly: bool = False
        cluster_state_storage: InstanceOf[ClusterStateStorage] = Field(
            default_factory=InMemoryClusterStateStorage,
        )
        cluster_params: ClusterParams = Field(default_factory=ClusterParams)

    def __init__(self, params: Params) -> None:
        """Initialize the ingestion service with storage backends and helpers."""
        self._semantic_storage = params.semantic_storage
        self._history_store = params.history_store
        self._resource_retriever = params.resource_retriever
        self._consolidation_threshold = params.consolidated_threshold
        self._debug_fail_loudly = params.debug_fail_loudly
        self._cluster_state_storage = params.cluster_state_storage
        self._cluster_manager = ClusterManager(params.cluster_params)

    async def process_set_ids(
        self, set_ids: Sequence[SetIdT] | AsyncIterator[SetIdT]
    ) -> None:
        if isinstance(set_ids, Sequence):
            logger.info("Starting ingestion processing for set ids: %s", set_ids)
        else:
            logger.info("Starting ingestion processing for streamed set ids")

        tasks = [
            asyncio.create_task(self._run_set_id(set_id))
            async for set_id in self._iter_set_ids(set_ids)
        ]

        if not tasks:
            return

        results = await asyncio.gather(*tasks, return_exceptions=True)

        errors = [r for r in results if isinstance(r, Exception)]
        if len(errors) > 0:
            raise ExceptionGroup("Failed to process set ids", errors)

    async def _run_set_id(self, set_id: SetIdT) -> None:
        try:
            await self._process_single_set(set_id)
        except Exception:
            logger.exception("Failed to process set_id %s", set_id)
            raise

    @staticmethod
    async def _iter_set_ids(
        set_ids: Sequence[SetIdT] | AsyncIterator[SetIdT],
    ) -> AsyncIterator[SetIdT]:
        if isinstance(set_ids, AsyncIterator):
            async for set_id in set_ids:
                yield set_id
        else:
            for set_id in set_ids:
                yield set_id

    async def _process_single_set(self, set_id: str) -> None:  # noqa: C901
        resources = await self._resource_retriever(set_id)

        history_ids = [
            history_id
            async for history_id in self._semantic_storage.get_history_messages(
                set_ids=[set_id],
                limit=50,
                is_ingested=False,
            )
        ]

        if len(resources.semantic_categories) == 0:
            logger.info(
                "No semantic categories configured for set %s, skipping ingestion",
                set_id,
            )

            await self._semantic_storage.mark_messages_ingested(
                set_id=set_id,
                history_ids=history_ids,
            )

        if len(history_ids) == 0:
            return

        async with asyncio.TaskGroup() as tg:
            tasks = {
                h_id: tg.create_task(self._history_store.get_episode(h_id))
                for h_id in history_ids
            }

        raw_messages = [tasks[h_id].result() for h_id in history_ids]
        none_h_ids = [h_id for h_id, task in tasks.items() if task.result() is None]

        if len(none_h_ids) != 0:
            raise ValueError(
                "Failed to retrieve messages. Invalid episode_ids exist for set_id "
                f"{set_id}: {none_h_ids}"
            )

        raw_messages = [m for m in raw_messages if m is not None]
        messages = TypeAdapter(list[Episode]).validate_python(raw_messages)

        logger.info("Processing %d messages for set %s", len(messages), set_id)

        clustered_messages, cluster_state = await self._cluster_messages(
            set_id=set_id,
            messages=messages,
            embedder=resources.embedder,
        )

        async def process_semantic_type(
            semantic_category: InstanceOf[SemanticCategory],
        ) -> None:
            for cluster_id, cluster_messages in clustered_messages:
                filter_expr = self._cluster_scoped_filter(
                    set_id=set_id,
                    category_name=semantic_category.name,
                    cluster_id=cluster_id,
                )

                features = [
                    feature
                    async for feature in self._semantic_storage.get_feature_set(
                        filter_expr=filter_expr,
                    )
                ]

                message_content = self._format_cluster_messages(cluster_messages)

                try:
                    commands = await llm_feature_update(
                        features=features,
                        message_content=message_content,
                        model=resources.language_model,
                        update_prompt=semantic_category.prompt.update_prompt,
                    )
                except Exception:
                    logger.exception(
                        "Failed to process cluster %s for semantic type %s",
                        cluster_id,
                        semantic_category.name,
                    )
                    if self._debug_fail_loudly:
                        raise

                    continue

                citation_ids = [m.uid for m in cluster_messages if m.uid is not None]

                await self._apply_commands(
                    commands=commands,
                    set_id=set_id,
                    category_name=semantic_category.name,
                    citation_ids=citation_ids,
                    embedder=resources.embedder,
                    cluster_id=cluster_id,
                )

                mark_messages.update(citation_ids)

        mark_messages: set[EpisodeIdT] = set()
        semantic_category_runners = []
        for t in resources.semantic_categories:
            task = process_semantic_type(t)
            semantic_category_runners.append(task)

        await asyncio.gather(*semantic_category_runners)

        logger.info(
            "Finished processing %d messages out of %d for set %s",
            len(mark_messages),
            len(messages),
            set_id,
        )

        if cluster_state is not None:
            await self._cluster_state_storage.save_state(
                set_id=set_id,
                state=cluster_state,
            )

        if len(mark_messages) == 0:
            return

        await self._semantic_storage.mark_messages_ingested(
            set_id=set_id,
            history_ids=list(mark_messages),
        )

        await self._consolidate_set_memories_if_applicable(
            set_id=set_id,
            resources=resources,
        )

    async def _apply_commands(
        self,
        *,
        commands: Sequence[SemanticCommand],
        set_id: SetIdT,
        category_name: str,
        citation_ids: Sequence[EpisodeIdT] | None,
        embedder: InstanceOf[Embedder],
        cluster_id: str | None,
    ) -> None:
        for command in commands:
            match command.command:
                case SemanticCommandType.ADD:
                    value_embedding = (await embedder.ingest_embed([command.value]))[0]

                    metadata = self._metadata_for_cluster(cluster_id)

                    f_id = await self._semantic_storage.add_feature(
                        set_id=set_id,
                        category_name=category_name,
                        feature=command.feature,
                        value=command.value,
                        tag=command.tag,
                        embedding=np.array(value_embedding),
                        metadata=metadata,
                    )

                    if citation_ids:
                        await self._semantic_storage.add_citations(f_id, citation_ids)

                case SemanticCommandType.DELETE:
                    filter_expr = And(
                        left=self._cluster_scoped_filter(
                            set_id=set_id,
                            category_name=category_name,
                            cluster_id=cluster_id,
                        ),
                        right=And(
                            left=Comparison(
                                field="feature", op="=", value=command.feature
                            ),
                            right=Comparison(field="tag", op="=", value=command.tag),
                        ),
                    )

                    await self._semantic_storage.delete_feature_set(
                        filter_expr=filter_expr
                    )

                case _:
                    logger.error("Command with unknown action: %s", command.command)

    async def _consolidate_set_memories_if_applicable(
        self,
        *,
        set_id: SetIdT,
        resources: InstanceOf[Resources],
    ) -> None:
        async def _consolidate_type(
            semantic_category: InstanceOf[SemanticCategory],
        ) -> None:
            from memmachine.common.filter.filter_parser import And, Comparison

            filter_expr = And(
                left=Comparison(field="set_id", op="=", value=set_id),
                right=Comparison(
                    field="category_name", op="=", value=semantic_category.name
                ),
            )

            features = [
                feature
                async for feature in self._semantic_storage.get_feature_set(
                    filter_expr=filter_expr,
                    tag_threshold=self._consolidation_threshold,
                    load_citations=True,
                )
            ]

            consolidation_sections: list[Sequence[SemanticFeature]] = list(
                self._group_features_by_tag_and_cluster(features).values(),
            )

            if self._consolidation_threshold > 0:
                consolidation_sections = [
                    section
                    for section in consolidation_sections
                    if len(section) >= self._consolidation_threshold
                ]

            await asyncio.gather(
                *[
                    self._deduplicate_features(
                        set_id=set_id,
                        memories=section_features,
                        resources=resources,
                        semantic_category=semantic_category,
                        cluster_id=self._extract_cluster_id(section_features),
                    )
                    for section_features in consolidation_sections
                ],
            )

        category_tasks = []
        for t in resources.semantic_categories:
            task = _consolidate_type(t)
            category_tasks.append(task)

        await asyncio.gather(*category_tasks)

    async def _deduplicate_features(
        self,
        *,
        set_id: str,
        memories: Sequence[SemanticFeature],
        semantic_category: InstanceOf[SemanticCategory],
        resources: InstanceOf[Resources],
        cluster_id: str | None,
    ) -> None:
        try:
            consolidate_resp = await llm_consolidate_features(
                features=memories,
                model=resources.language_model,
                consolidate_prompt=semantic_category.prompt.consolidation_prompt,
            )
        except (ValueError, TypeError):
            logger.exception("Failed to update features while calling LLM")
            if self._debug_fail_loudly:
                raise
            return

        if consolidate_resp is None or consolidate_resp.keep_memories is None:
            logger.warning("Failed to consolidate features")
            if self._debug_fail_loudly:
                raise ValueError("Failed to consolidate features")
            return

        memories_to_delete = [
            m
            for m in memories
            if m.metadata.id is not None
            and m.metadata.id not in consolidate_resp.keep_memories
        ]
        await self._semantic_storage.delete_features(
            [m.metadata.id for m in memories_to_delete if m.metadata.id is not None],
        )

        merged_citations: chain[EpisodeIdT] = itertools.chain.from_iterable(
            [
                m.metadata.citations
                for m in memories_to_delete
                if m.metadata.citations is not None
            ],
        )
        citation_ids = TypeAdapter(list[EpisodeIdT]).validate_python(
            list(merged_citations),
        )

        async def _add_feature(f: LLMReducedFeature) -> None:
            value_embedding = (await resources.embedder.ingest_embed([f.value]))[0]

            metadata = self._metadata_for_cluster(cluster_id)

            f_id = await self._semantic_storage.add_feature(
                set_id=set_id,
                category_name=semantic_category.name,
                tag=f.tag,
                feature=f.feature,
                value=f.value,
                embedding=np.array(value_embedding),
                metadata=metadata,
            )

            await self._semantic_storage.add_citations(f_id, citation_ids)

        await asyncio.gather(
            *[
                _add_feature(feature)
                for feature in consolidate_resp.consolidated_memories
            ],
        )

    async def _cluster_messages(
        self,
        *,
        set_id: SetIdT,
        messages: Sequence[Episode],
        embedder: InstanceOf[Embedder],
    ) -> tuple[Sequence[tuple[str, Sequence[Episode]]], ClusterState]:
        if not messages:
            return [], ClusterState()

        cluster_state = await self._cluster_state_storage.get_state(set_id=set_id)
        if cluster_state is None:
            cluster_state = ClusterState()

        ordered_messages = sorted(messages, key=lambda m: (m.created_at, m.uid))
        embeddings = await embedder.ingest_embed(
            [m.content for m in ordered_messages],
        )

        clusters: MutableMapping[str, list[Episode]] = {}

        for message, embedding in zip(ordered_messages, embeddings, strict=True):
            if message.uid is None:
                logger.error("Message ID is None for message %s", message.model_dump())
                raise ValueError(
                    f"Message ID is None for message {message.model_dump()}"
                )

            assignment, cluster_state = self._cluster_manager.assign(
                event_id=message.uid,
                embedding=embedding,
                timestamp=message.created_at,
                state=cluster_state,
            )
            clusters.setdefault(assignment.cluster_id, []).append(message)

        def _cluster_sort_key(item: tuple[str, Sequence[Episode]]) -> tuple:
            cluster_messages = item[1]
            earliest = min(m.created_at for m in cluster_messages)
            return (earliest, item[0])

        ordered_clusters = sorted(clusters.items(), key=_cluster_sort_key)

        return ordered_clusters, cluster_state

    @staticmethod
    def _format_cluster_messages(messages: Sequence[Episode]) -> str:
        return "\n\n".join(m.content for m in messages)

    @staticmethod
    def _metadata_for_cluster(cluster_id: str | None) -> Mapping[str, str] | None:
        if cluster_id is None:
            return None
        return {_CLUSTER_METADATA_KEY: cluster_id}

    @staticmethod
    def _cluster_scoped_filter(
        *,
        set_id: SetIdT,
        category_name: str,
        cluster_id: str | None,
    ) -> And:
        base_filter = And(
            left=Comparison(field="set_id", op="=", value=set_id),
            right=Comparison(field="category_name", op="=", value=category_name),
        )
        if cluster_id is None:
            return base_filter

        cluster_filter = Or(
            left=Comparison(
                field=f"metadata.{_CLUSTER_METADATA_KEY}",
                op="=",
                value=cluster_id,
            ),
            right=Comparison(
                field=f"metadata.{_CLUSTER_METADATA_KEY}",
                op="is_null",
                value="",
            ),
        )

        return And(
            left=base_filter,
            right=cluster_filter,
        )

    @staticmethod
    def _group_features_by_tag_and_cluster(
        features: Sequence[SemanticFeature],
    ) -> Mapping[tuple[str, str | None], Sequence[SemanticFeature]]:
        grouped: MutableMapping[tuple[str, str | None], list[SemanticFeature]] = {}
        for feature in features:
            cluster_id = None
            if feature.metadata.other:
                cluster_id = feature.metadata.other.get(_CLUSTER_METADATA_KEY)
            key = (feature.tag, cluster_id)
            grouped.setdefault(key, []).append(feature)
        return grouped

    @staticmethod
    def _extract_cluster_id(memories: Sequence[SemanticFeature]) -> str | None:
        for memory in memories:
            if memory.metadata.other and _CLUSTER_METADATA_KEY in memory.metadata.other:
                return memory.metadata.other[_CLUSTER_METADATA_KEY]
        return None
