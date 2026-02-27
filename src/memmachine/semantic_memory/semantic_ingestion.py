"""Ingestion pipeline for converting episodes into semantic features."""

import asyncio
import itertools
import logging
from dataclasses import dataclass, field
from itertools import chain
from typing import Any

import numpy as np
from pydantic import BaseModel, InstanceOf, TypeAdapter

from memmachine.common.embedder import Embedder
from memmachine.common.episode_store import Episode, EpisodeIdT, EpisodeStorage
from memmachine.common.filter.filter_parser import And, Comparison
from memmachine.common.language_model import LanguageModel
from memmachine.semantic_memory.semantic_llm import (
    LLMReducedFeature,
    llm_classify_relationship,
    llm_consolidate_features,
    llm_feature_update,
)
from memmachine.semantic_memory.semantic_model import (
    FeatureIdT,
    ResourceRetrieverT,
    Resources,
    SemanticCategory,
    SemanticCommand,
    SemanticCommandType,
    SemanticFeature,
    SetIdT,
)
from memmachine.semantic_memory.storage.feature_relationship_types import (
    FeatureRelationshipType,
)
from memmachine.semantic_memory.storage.semantic_relationship_storage import (
    SemanticRelationshipStorage,
)
from memmachine.semantic_memory.storage.storage_base import SemanticStorage

logger = logging.getLogger(__name__)


def _cosine_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    """Compute cosine similarity between two embedding vectors."""
    a = np.array(vec_a, dtype=float)
    b = np.array(vec_b, dtype=float)
    norm_a = float(np.linalg.norm(a))
    norm_b = float(np.linalg.norm(b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


@dataclass
class _AppliedCommand:
    """Record of a command applied during ingestion, used for relationship detection."""

    command_type: SemanticCommandType
    tag: str
    feature_name: str
    value: str
    feature_id: FeatureIdT | None = None
    embedding: list[float] | None = None


@dataclass
class _BatchCommands:
    """Accumulates applied commands across a set_id ingestion batch."""

    commands: list[_AppliedCommand] = field(default_factory=list)


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
        related_to_threshold: float = 0.70
        max_relationship_llm_calls: int = 10

    def __init__(self, params: Params) -> None:
        """Initialize the ingestion service with storage backends and helpers."""
        self._semantic_storage = params.semantic_storage
        self._history_store = params.history_store
        self._resource_retriever = params.resource_retriever
        self._consolidation_threshold = params.consolidated_threshold
        self._debug_fail_loudly = params.debug_fail_loudly
        self._related_to_threshold = params.related_to_threshold
        self._max_relationship_llm_calls = params.max_relationship_llm_calls

    async def process_set_ids(self, set_ids: list[SetIdT]) -> None:
        async def _run(set_id: SetIdT) -> None:
            try:
                await self._process_single_set(set_id)
            except Exception:
                logger.exception("Failed to process set_id %s", set_id)
                raise

        logger.info("Starting ingestion processing for set ids: %s", set_ids)

        results = await asyncio.gather(
            *[_run(set_id) for set_id in set_ids],
            return_exceptions=True,
        )

        errors = [r for r in results if isinstance(r, Exception)]
        if len(errors) > 0:
            raise ExceptionGroup("Failed to process set ids", errors)

    async def _process_single_set(self, set_id: str) -> None:  # noqa: C901
        resources = await self._resource_retriever(set_id)

        history_ids = await self._semantic_storage.get_history_messages(
            set_ids=[set_id],
            limit=5,
            is_ingested=False,
        )

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

        batch = _BatchCommands()

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

                features = await self._semantic_storage.get_feature_set(
                    filter_expr=filter_expr,
                )

                try:
                    commands = await llm_feature_update(
                        features=features,
                        message_content=message.content,
                        model=resources.language_model,
                        update_prompt=semantic_category.prompt.update_prompt,
                    )
                except Exception:
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
                    batch=batch,
                )

                mark_messages.append(message.uid)

        mark_messages: list[EpisodeIdT] = []
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

        if len(mark_messages) == 0:
            return

        # Detect feature relationships after all commands are applied.
        await self._detect_feature_relationships(
            set_id=set_id,
            batch=batch,
            embedder=resources.embedder,
            language_model=resources.language_model,
        )

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
        batch: _BatchCommands | None = None,
    ) -> None:
        for command in commands:
            match command.command:
                case SemanticCommandType.ADD:
                    value_embedding = (await embedder.ingest_embed([command.value]))[0]

                    metadata: dict[str, Any] | None = None
                    if command.entity_type is not None:
                        metadata = {"entity_type": command.entity_type}

                    f_id = await self._semantic_storage.add_feature(
                        set_id=set_id,
                        category_name=category_name,
                        feature=command.feature,
                        value=command.value,
                        tag=command.tag,
                        embedding=np.array(value_embedding),
                        metadata=metadata,
                    )

                    if citation_id is not None:
                        await self._semantic_storage.add_citations(f_id, [citation_id])

                    if batch is not None:
                        batch.commands.append(
                            _AppliedCommand(
                                command_type=SemanticCommandType.ADD,
                                tag=command.tag,
                                feature_name=command.feature,
                                value=command.value,
                                feature_id=f_id,
                                embedding=value_embedding,
                            )
                        )

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

                    if batch is not None:
                        batch.commands.append(
                            _AppliedCommand(
                                command_type=SemanticCommandType.DELETE,
                                tag=command.tag,
                                feature_name=command.feature,
                                value=command.value,
                            )
                        )

                case _:
                    logger.error("Command with unknown action: %s", command.command)

    async def _detect_feature_relationships(
        self,
        *,
        set_id: SetIdT,
        batch: _BatchCommands,
        embedder: InstanceOf[Embedder],
        language_model: InstanceOf[LanguageModel],
    ) -> None:
        """Detect and create feature relationships from the current ingestion batch.

        This runs after all ``_apply_commands()`` calls for a set_id.  Two kinds
        of relationships are detected:

        1. **RELATED_TO** — pairwise cosine similarity between newly added
           features and all existing features in the set.  Pairs above
           ``self._related_to_threshold`` get a RELATED_TO edge.
        2. **CONTRADICTS / SUPERSEDES** — when a DELETE+ADD pair targets the
           same ``(tag, feature_name)``, the LLM classifies the relationship.
           Capped at ``self._max_relationship_llm_calls`` LLM invocations.
        """
        if not isinstance(self._semantic_storage, SemanticRelationshipStorage):
            return

        storage: SemanticRelationshipStorage = self._semantic_storage

        # Collect new ADD commands that have both an ID and embedding.
        added = [
            cmd
            for cmd in batch.commands
            if cmd.command_type == SemanticCommandType.ADD
            and cmd.feature_id is not None
            and cmd.embedding is not None
        ]

        if not added:
            return

        await self._detect_related_to(
            set_id=set_id,
            added=added,
            embedder=embedder,
            storage=storage,
        )

        await self._detect_contradicts_supersedes(
            batch=batch,
            language_model=language_model,
        )

    async def _detect_related_to(
        self,
        *,
        set_id: SetIdT,
        added: list[_AppliedCommand],
        embedder: InstanceOf[Embedder],
        storage: SemanticRelationshipStorage,
    ) -> None:
        """Create RELATED_TO edges for new features similar to existing ones.

        Compares newly added features against:
        1. Existing features in the **same** set (intra-set relationships).
        2. Existing features in **all other** sets (cross-set relationships).

        Cross-set comparison is essential for discovering semantic connections
        between features that belong to different entities (e.g. a person's
        TensorFlow expertise and a project's TensorFlow dependency).
        """
        # --- Intra-set candidates ---
        same_set_features = await self._semantic_storage.get_feature_set(
            filter_expr=Comparison(field="set_id", op="=", value=set_id),
        )
        same_set_with_id = [f for f in same_set_features if f.metadata.id is not None]

        # --- Cross-set candidates ---
        # Retrieve all features and exclude the current set to avoid
        # duplicating the intra-set comparison.
        all_features = await self._semantic_storage.get_feature_set()
        cross_set_with_id = [
            f for f in all_features if f.metadata.id is not None and f.set_id != set_id
        ]

        # Combine both candidate pools.
        candidates = same_set_with_id + cross_set_with_id
        if not candidates:
            return

        candidate_values = [f.value for f in candidates]
        candidate_embeddings = await embedder.ingest_embed(candidate_values)
        new_ids = {cmd.feature_id for cmd in added}

        for cmd in added:
            new_emb = np.array(cmd.embedding, dtype=float)
            new_norm = float(np.linalg.norm(new_emb))
            if new_norm == 0:
                continue

            for feat, existing_emb_list in zip(
                candidates, candidate_embeddings, strict=True
            ):
                if feat.metadata.id in new_ids and feat.metadata.id == cmd.feature_id:
                    continue

                sim = _cosine_similarity(cmd.embedding, existing_emb_list)  # type: ignore[arg-type]
                if sim >= self._related_to_threshold:
                    # Only set ``similarity`` on edges that represent a
                    # *meaningful* semantic bridge — cross-feature-name
                    # connections where the values are not near-duplicates.
                    #
                    # Same-name edges (``role → role``, ``name → name``)
                    # are trivial positional matches and get similarity=NULL.
                    #
                    # Near-duplicate values (sim >= 0.99) are the same fact
                    # extracted twice into differently-named features (e.g.
                    # during re-ingestion or paraphrase extraction).  Treating
                    # them as semantic bridges would create high-quality paths
                    # between unrelated episodes that share a common producer,
                    # boosting irrelevant results.  They get similarity=NULL
                    # so traversal treats them as quality 0.0.
                    is_cross_name = cmd.feature_name != feat.feature_name
                    is_near_duplicate = sim >= 0.99
                    is_semantic_bridge = is_cross_name and not is_near_duplicate
                    await storage.add_feature_relationship(
                        source_id=cmd.feature_id,  # type: ignore[arg-type]
                        target_id=feat.metadata.id,
                        relationship_type=FeatureRelationshipType.RELATED_TO,
                        confidence=sim,
                        source="rule",
                        similarity=sim if is_semantic_bridge else None,
                    )

    async def _detect_contradicts_supersedes(
        self,
        *,
        batch: _BatchCommands,
        language_model: InstanceOf[LanguageModel],
    ) -> None:
        """Classify DELETE+ADD pairs via LLM for CONTRADICTS / SUPERSEDES."""
        groups: dict[tuple[str, str], list[_AppliedCommand]] = {}
        for cmd in batch.commands:
            key = (cmd.tag, cmd.feature_name)
            groups.setdefault(key, []).append(cmd)

        llm_calls_remaining = self._max_relationship_llm_calls

        for (tag, feature_name), cmds in groups.items():
            if llm_calls_remaining <= 0:
                break

            deletes = [c for c in cmds if c.command_type == SemanticCommandType.DELETE]
            adds = [
                c
                for c in cmds
                if c.command_type == SemanticCommandType.ADD
                and c.feature_id is not None
            ]
            if not deletes or not adds:
                continue

            llm_calls_remaining = await self._classify_delete_add_pairs(
                deletes=deletes,
                adds=adds,
                tag=tag,
                feature_name=feature_name,
                language_model=language_model,
                llm_calls_remaining=llm_calls_remaining,
            )

    async def _classify_delete_add_pairs(
        self,
        *,
        deletes: list[_AppliedCommand],
        adds: list[_AppliedCommand],
        tag: str,
        feature_name: str,
        language_model: InstanceOf[LanguageModel],
        llm_calls_remaining: int,
    ) -> int:
        """Classify each delete-add pair via LLM. Returns remaining call budget."""
        for delete_cmd in deletes:
            for add_cmd in adds:
                if llm_calls_remaining <= 0:
                    return llm_calls_remaining

                try:
                    classification = await llm_classify_relationship(
                        deleted_value=delete_cmd.value,
                        added_value=add_cmd.value,
                        model=language_model,
                    )
                except Exception:
                    logger.exception(
                        "Failed to classify relationship between "
                        "deleted '%s' and added '%s'",
                        delete_cmd.value,
                        add_cmd.value,
                    )
                    continue
                finally:
                    llm_calls_remaining -= 1

                if (
                    classification is None
                    or classification.classification == "UNRELATED"
                ):
                    continue

                # The deleted feature no longer exists in storage, so we
                # cannot create a graph edge.  Log the detection for
                # observability — a future iteration could pre-capture
                # deleted feature IDs to enable edge creation.
                logger.info(
                    "Relationship %s detected (confidence=%.2f) for "
                    "tag=%s feature=%s (deleted feature has no stored ID)",
                    classification.classification,
                    classification.confidence,
                    tag,
                    feature_name,
                )
        return llm_calls_remaining

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

            features = await self._semantic_storage.get_feature_set(
                filter_expr=filter_expr,
                tag_threshold=self._consolidation_threshold,
                load_citations=True,
            )

            consolidation_sections: list[list[SemanticFeature]] = list(
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
        memories: list[SemanticFeature],
        semantic_category: InstanceOf[SemanticCategory],
        resources: InstanceOf[Resources],
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
            list(set(merged_citations)),
        )

        # Determine consensus entity type from deleted memories:
        # same type -> keep; conflicting types -> None
        source_entity_types = {
            m.entity_type for m in memories_to_delete if m.entity_type is not None
        }
        consensus_entity_type: str | None = None
        if len(source_entity_types) == 1:
            consensus_entity_type = source_entity_types.pop()

        async def _add_feature(f: LLMReducedFeature) -> None:
            value_embedding = (await resources.embedder.ingest_embed([f.value]))[0]

            # Use LLM-returned entity_type if present, else fall back to
            # consensus from the source features being consolidated.
            resolved_entity_type = f.entity_type or consensus_entity_type
            metadata: dict[str, Any] | None = None
            if resolved_entity_type is not None:
                metadata = {"entity_type": resolved_entity_type}

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
