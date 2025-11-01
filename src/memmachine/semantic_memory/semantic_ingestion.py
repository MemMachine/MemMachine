import asyncio
import itertools
import logging

import numpy as np
from pydantic import BaseModel, InstanceOf

from memmachine.common.embedder import Embedder
from memmachine.semantic_memory.semantic_llm import (
    llm_consolidate_features,
    llm_feature_update,
)
from memmachine.semantic_memory.semantic_model import (
    ResourceRetriever,
    SemanticCommand,
    SemanticFeature,
    SemanticType,
)
from memmachine.semantic_memory.storage.storage_base import SemanticStorageBase

logger = logging.getLogger(__name__)


class IngestionService:
    class Params(BaseModel):
        semantic_storage: InstanceOf[SemanticStorageBase]
        resource_retriever: InstanceOf[ResourceRetriever]
        consolidated_threshold: int = 20

    def __init__(self, params: Params):
        self._semantic_storage = params.semantic_storage
        self._resource_retriever = params.resource_retriever
        self._consolidation_threshold = params.consolidated_threshold

    async def process_set_ids(self, set_ids: list[str]) -> None:
        results = await asyncio.gather(
            *[self._process_single_set(set_id) for set_id in set_ids],
            return_exceptions=True,
        )

        errors = [r for r in results if isinstance(r, Exception)]
        if len(errors) > 0:
            for e in errors:
                logger.error("Failed to process set", e)

            raise errors[0]

    async def _process_single_set(self, set_id: str):
        resources = self._resource_retriever.get_resources(set_id)
        messages = await self._semantic_storage.get_history_messages(
            set_id=set_id,
            k=50,
            is_ingested=False,
        )

        if len(messages) == 0:
            return

        mark_messages = []

        async def process_semantic_type(semantic_type: InstanceOf[SemanticType]):
            for message in messages:
                features = await self._semantic_storage.get_feature_set(
                    set_ids=[set_id],
                    type_names=[semantic_type.name],
                )

                try:
                    commands = await llm_feature_update(
                        features=features,
                        message_content=message.content,
                        model=resources.language_model,
                        update_prompt=semantic_type.prompt.update_prompt,
                    )
                except (ValueError, TypeError) as e:
                    logger.error(
                        f"Failed to process message {message.id} for semantic type {semantic_type.name}",
                        e,
                    )
                    continue

                await self._apply_commands(
                    commands=commands,
                    set_id=set_id,
                    type_name=semantic_type.name,
                    citation_id=message.metadata.id,
                    embedder=resources.embedder,
                )

                mark_messages.append(message["id"])

        semantic_type_runners = []
        for t in resources.semantic_types:
            task = process_semantic_type(t)
            semantic_type_runners.append(task)

        await asyncio.gather(*semantic_type_runners)

        await self._semantic_storage.mark_messages_ingested(
            set_id=set_id,
            ids=mark_messages,
        )

        await self._consolidate_set_memories_if_applicable(
            set_id=set_id, resources=resources
        )

    async def _apply_commands(
        self,
        *,
        commands: list[SemanticCommand],
        set_id: str,
        type_name: str,
        citation_id: int,
        embedder: InstanceOf[Embedder],
    ):
        for command in commands:
            match command.command:
                case "add":
                    value_embedding = (await embedder.ingest_embed([command.value]))[0]

                    f_id = await self._semantic_storage.add_feature(
                        set_id=set_id,
                        type_name=type_name,
                        feature=command.feature,
                        value=command.value,
                        tag=command.tag,
                        embedding=np.array(value_embedding),
                    )

                    await self._semantic_storage.add_citations(f_id, [citation_id])

                case "delete":
                    await self._semantic_storage.delete_feature_set(
                        set_ids=[set_id],
                        type_names=[type_name],
                        feature_names=[command.feature],
                        tags=[command.tag],
                    )

                case _:
                    logger.error("Command with unknown action: %s", command.command)

    async def _consolidate_set_memories_if_applicable(
        self,
        *,
        set_id: str,
        resources: InstanceOf[ResourceRetriever.Resources],
    ):
        async def _consolidate_type(semantic_type: InstanceOf[SemanticType]):
            features = await self._semantic_storage.get_feature_set(
                set_ids=[set_id],
                type_names=[semantic_type.name],
                tag_threshold=self._consolidation_threshold,
                load_citations=True,
            )

            consolidation_sections: list[list[SemanticFeature]] = list(
                SemanticFeature.group_features_by_tag(features).values()
            )

            await asyncio.gather(
                *[
                    self._deduplicate_features(
                        set_id=set_id,
                        memories=section_features,
                        resources=resources,
                        semantic_type=semantic_type,
                    )
                    for section_features in consolidation_sections
                ]
            )

        type_tasks = []
        for t in resources.semantic_types:
            task = _consolidate_type(t)
            type_tasks.append(task)

        await asyncio.gather(*type_tasks)

    async def _deduplicate_features(
        self,
        *,
        set_id: str,
        memories: list[SemanticFeature],
        semantic_type: InstanceOf[SemanticType],
        resources: InstanceOf[ResourceRetriever.Resources],
    ):
        try:
            consolidate_resp = await llm_consolidate_features(
                features=memories,
                model=resources.language_model,
                consolidate_prompt=semantic_type.prompt.consolidation_prompt,
            )
        except (ValueError, TypeError) as e:
            logger.error("Failed to update features while calling LLM", e)
            return

        if consolidate_resp is None or consolidate_resp.keep_memories is None:
            logger.warning("Failed to consolidate features")
            return

        memories_to_delete = [
            m
            for m in memories
            if m.metadata.id is not None
            and m.metadata.id not in consolidate_resp.keep_memories
        ]
        await self._semantic_storage.delete_features(
            [m.metadata.id for m in memories_to_delete]
        )

        merged_citations = itertools.chain.from_iterable(
            [m.metadata.citations for m in memories_to_delete]
        )

        async def _add_feature(f: SemanticFeature):
            value_embedding = (await resources.embedder.ingest_embed([f.value]))[0]

            f_id = await self._semantic_storage.add_feature(
                set_id=set_id,
                type_name=semantic_type.name,
                tag=f.tag,
                feature=f.feature,
                value=f.value,
                embedding=np.array(value_embedding),
            )
            await self._semantic_storage.add_citations(f_id, merged_citations)

        await asyncio.gather(
            *[
                _add_feature(feature)
                for feature in consolidate_resp.consolidated_memories
            ],
        )
