"""Ingestion pipeline for converting episodes into semantic features."""

import asyncio
import itertools
import logging
from collections.abc import Sequence
from itertools import chain

import numpy as np
from pydantic import BaseModel, Field, InstanceOf, TypeAdapter

from memmachine_server.common.embedder import Embedder
from memmachine_server.common.episode_store import Episode, EpisodeIdT, EpisodeStorage
from memmachine_server.common.filter.filter_parser import And, Comparison
from memmachine_server.semantic_memory.semantic_llm import (
    LLMReducedFeature,
    llm_consolidate_features,
    llm_feature_update,
)
from memmachine_server.semantic_memory.semantic_model import (
    ResourceRetrieverT,
    Resources,
    SemanticCategory,
    SemanticCommand,
    SemanticCommandType,
    SemanticFeature,
    SetIdT,
)
from memmachine_server.semantic_memory.storage.storage_base import SemanticStorage

logger = logging.getLogger(__name__)


def _is_context_length_exceeded_error(error: Exception) -> bool:
    seen: set[int] = set()
    current: BaseException | None = error
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        code = getattr(current, "code", None)
        if isinstance(code, str) and code.lower() == "context_length_exceeded":
            return True

        message = str(current).lower()
        if (
            "context_length_exceeded" in message
            or "exceeds the context window" in message
        ):
            return True

        current = current.__cause__ or current.__context__

    return False


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
        max_features_per_update: int = Field(
            50,
            description=(
                "Maximum number of existing features passed to the LLM per update "
                "call. Limiting this prevents the LLM output from overflowing its "
                "token budget when the profile has grown very large."
            ),
            gt=0,
        )

    def __init__(self, params: Params) -> None:
        """Initialize the ingestion service with storage backends and helpers."""
        self._semantic_storage = params.semantic_storage
        self._history_store = params.history_store
        self._resource_retriever = params.resource_retriever
        self._consolidation_threshold = params.consolidated_threshold
        self._debug_fail_loudly = params.debug_fail_loudly
        self._max_features_per_update = params.max_features_per_update

    async def process_set_ids(self, set_ids: list[SetIdT]) -> None:
        async def _run(set_id: SetIdT) -> None:
            try:
                await self._process_single_set(set_id)
            except Exception:
                logger.exception("Failed to process set_id %s", set_id)
                raise

        logger.debug("Starting ingestion processing for set ids: %s", set_ids)

        results = await asyncio.gather(
            *[_run(set_id) for set_id in set_ids],
            return_exceptions=True,
        )

        errors = [r for r in results if isinstance(r, Exception)]
        if len(errors) > 0:
            raise ExceptionGroup("Failed to process set ids", errors)

    async def _process_single_set(self, set_id: str) -> None:  # noqa: C901
        resources = await self._resource_retriever(set_id)

        history_ids = [
            h
            async for h in self._semantic_storage.get_history_messages(
                set_ids=[set_id],
                limit=5,
                is_ingested=False,
            )
        ]

        if len(resources.semantic_categories) == 0:
            logger.debug(
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

        raw_messages = [
            r for h_id in history_ids if (r := tasks[h_id].result()) is not None
        ]
        none_h_ids = [h_id for h_id, task in tasks.items() if task.result() is None]

        if len(none_h_ids) != 0:
            logger.warning(
                "Failed to retrieve messages. Invalid episode_ids exist for set_id %s; delisting the following messages as recovery: %s",
                set_id,
                none_h_ids,
            )
            if self._debug_fail_loudly:
                raise ValueError(
                    f"Failed to retrieve messages for set_id {set_id} due to invalid episode_ids: {none_h_ids}"
                )

            try:
                await self._semantic_storage.delete_history(
                    history_ids=none_h_ids,
                )
            except Exception:
                logger.exception(
                    "Failed to delete messages with invalid episode_ids for set_id %s",
                    set_id,
                )
                if self._debug_fail_loudly:
                    raise

        messages = TypeAdapter(list[Episode]).validate_python(raw_messages)

        logger.debug("Processing %d messages for set %s", len(messages), set_id)

        async def process_semantic_type(
            semantic_category: InstanceOf[SemanticCategory],
        ) -> None:
            for message in messages:
                if message.uid is None:
                    logger.error(
                        "Message ID is None for message %s", message.model_dump()
                    )

                    raise ValueError(
                        f"Message ID is None for message {message.model_dump()}"
                    )

                filter_expr = And(
                    left=Comparison(field="set_id", op="=", value=set_id),
                    right=Comparison(
                        field="category", op="=", value=semantic_category.name
                    ),
                )

                features = [
                    f
                    async for f in self._semantic_storage.get_feature_set(
                        filter_expr=filter_expr,
                        page_size=self._max_features_per_update,
                    )
                ]

                try:
                    commands = await llm_feature_update(
                        features=features,
                        message_content=message.content,
                        model=resources.language_model,
                        update_prompt=semantic_category.prompt.update_prompt,
                    )
                except Exception as err:
                    if _is_context_length_exceeded_error(err):
                        logger.warning(
                            "Skipping message %s for semantic type %s due to non-retryable context length error",
                            message.uid,
                            semantic_category.name,
                        )
                        if message.uid not in mark_messages:
                            mark_messages.append(message.uid)
                        continue

                    logger.exception(
                        "Failed to process message %s for semantic type %s",
                        message.uid,
                        semantic_category.name,
                    )
                    if self._debug_fail_loudly:
                        raise

                    continue

                await self._apply_commands(
                    commands=commands,
                    set_id=set_id,
                    category_name=semantic_category.name,
                    citation_id=message.uid,
                    embedder=resources.embedder,
                )

                mark_messages.append(message.uid)

        mark_messages: list[EpisodeIdT] = []
        semantic_category_runners = []
        for t in resources.semantic_categories:
            task = process_semantic_type(t)
            semantic_category_runners.append(task)

        await asyncio.gather(*semantic_category_runners)

        logger.debug(
            "Finished processing %d messages out of %d for set %s",
            len(mark_messages),
            len(messages),
            set_id,
        )

        if len(mark_messages) == 0:
            return

        await self._semantic_storage.mark_messages_ingested(
            set_id=set_id,
            history_ids=mark_messages,
        )

        await self._consolidate_set_memories_if_applicable(
            set_id=set_id,
            resources=resources,
        )

    async def _apply_commands(
        self,
        *,
        commands: list[SemanticCommand],
        set_id: SetIdT,
        category_name: str,
        citation_id: EpisodeIdT | None,
        embedder: InstanceOf[Embedder],
    ) -> None:
        for command in commands:
            match command.command:
                case SemanticCommandType.ADD:
                    value_embedding = (await embedder.ingest_embed([command.value]))[0]

                    f_id = await self._semantic_storage.add_feature(
                        set_id=set_id,
                        category_name=category_name,
                        feature=command.feature,
                        value=command.value,
                        tag=command.tag,
                        embedding=np.array(value_embedding),
                    )

                    if citation_id is not None:
                        await self._semantic_storage.add_citations(f_id, [citation_id])

                case SemanticCommandType.DELETE:
                    filter_expr = And(
                        left=And(
                            left=Comparison(field="set_id", op="=", value=set_id),
                            right=Comparison(
                                field="category_name", op="=", value=category_name
                            ),
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
            from memmachine_server.common.filter.filter_parser import And, Comparison

            filter_expr = And(
                left=Comparison(field="set_id", op="=", value=set_id),
                right=Comparison(
                    field="category_name", op="=", value=semantic_category.name
                ),
            )

            features = [
                f
                async for f in self._semantic_storage.get_feature_set(
                    filter_expr=filter_expr,
                    tag_threshold=self._consolidation_threshold,
                    load_citations=True,
                )
            ]

            consolidation_sections: list[Sequence[SemanticFeature]] = list(
                SemanticFeature.group_features_by_tag(features).values(),
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
    ) -> None:
        try:
            consolidate_resp = await llm_consolidate_features(
                features=list(memories),
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
            list(set(merged_citations)),
        )

        original_tag = memories[0].tag if memories else None

        async def _add_feature(f: LLMReducedFeature) -> None:
            tag = original_tag if original_tag is not None else f.tag
            if tag != f.tag:
                logger.warning(
                    "Consolidation LLM changed tag from %r to %r; "
                    "reverting to original tag",
                    tag,
                    f.tag,
                )

            value_embedding = (await resources.embedder.ingest_embed([f.value]))[0]

            f_id = await self._semantic_storage.add_feature(
                set_id=set_id,
                category_name=semantic_category.name,
                tag=tag,
                feature=f.feature,
                value=f.value,
                embedding=np.array(value_embedding),
            )

            await self._semantic_storage.add_citations(f_id, citation_ids)

        await asyncio.gather(
            *[
                _add_feature(feature)
                for feature in consolidate_resp.consolidated_memories
            ],
        )
