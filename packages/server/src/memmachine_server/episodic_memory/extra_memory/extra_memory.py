"""Extra memory system for storing and retrieving episodic memory."""

import asyncio
import datetime
import json
import logging
from collections.abc import Iterable, Mapping
from uuid import UUID, uuid4

from langchain_text_splitters import RecursiveCharacterTextSplitter
from pydantic import BaseModel, Field, InstanceOf

from memmachine_server.common.data_types import SimilarityMetric
from memmachine_server.common.embedder import Embedder
from memmachine_server.common.filter.filter_parser import (
    FilterExpr,
)
from memmachine_server.common.reranker import Reranker
from memmachine_server.common.utils import (
    compute_similarity,
    extract_sentences,
    similarity_gt,
    unflatten_like,
)
from memmachine_server.common.vector_store import (
    Collection,
    Record,
)
from memmachine_server.common.vector_store.data_types import (
    QueryResult as VectorStoreQueryResult,
)

from .data_types import (
    Block,
    CitationContext,
    Content,
    Context,
    Derivative,
    Episode,
    FileRef,
    MessageContext,
    QueryResult,
    ReadFile,
    Segment,
    Text,
)
from .segment_linker import DerivativeNotActiveError, SegmentLinkerPartition

logger = logging.getLogger(__name__)


class ExtraMemoryParams(BaseModel):
    """
    Parameters for ExtraMemory.

    Attributes:
        collection (Collection):
            Collection instance in a vector store.
        segment_linker_partition (SegmentLinkerPartition):
            Segment linker partition handle for managing segments.
        embedder (Embedder):
            Embedder instance for creating embeddings.
        reranker (Reranker):
            Reranker instance for reranking search results.
        derive_sentences (bool):
            Whether to derive sentence-level derivatives from content (default: False).
        max_text_chunk_length (int):
            Max code-point length for text chunking in segment creation (default: 2000).
        derivative_consolidation_threshold (float):
            Threshold for consolidating derivatives (default: 0.0, range: 0.0 to 1.0).
        purge_interval (float | None):
            Seconds between purge cycles. None disables periodic purging (default: None).
    """

    collection: InstanceOf[Collection] = Field(
        ...,
        description="Collection instance in a vector store",
    )
    segment_linker_partition: InstanceOf[SegmentLinkerPartition] = Field(
        ...,
        description="Segment linker partition handle for managing segments",
    )
    embedder: InstanceOf[Embedder] = Field(
        ...,
        description="Embedder instance for creating embeddings",
    )
    reranker: InstanceOf[Reranker] = Field(
        ...,
        description="Reranker instance for reranking search results",
    )
    derive_sentences: bool = Field(
        False,
        description="Whether to derive sentence-level derivatives from content",
    )
    max_text_chunk_length: int = Field(
        2000,
        description="Max code-point length for text chunking in segment creation",
    )
    derivative_consolidation_threshold: float | None = Field(
        None,
        description="Threshold for consolidating derivatives",
    )
    purge_interval: float | None = Field(
        None,
        description="Seconds between purge cycles. None disables periodic purging.",
    )


class ExtraMemory:
    """Extra memory system."""

    def __init__(self, params: ExtraMemoryParams) -> None:
        """
        Initialize an ExtraMemory with the provided parameters.

        Args:
            params (ExtraMemoryParams):
                Parameters for the ExtraMemory.

        """
        self._collection = params.collection
        self._segment_linker_partition = params.segment_linker_partition

        self._embedder = params.embedder
        self._reranker = params.reranker

        self._derive_sentences = params.derive_sentences
        self._derivative_consolidation_threshold = (
            params.derivative_consolidation_threshold
        )

        self._purge_interval = params.purge_interval
        self._purge_task: asyncio.Task[None] | None = None
        self._shutdown_event = asyncio.Event()

        self._text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=params.max_text_chunk_length,
            chunk_overlap=0,
            separators=[
                "\n\n",
                "],\n",
                "},\n",
                "),\n",
                "]\n",
                "}\n",
                ")\n",
                ",\n",
                "\uff1f\n",  # Fullwidth question mark
                "?\n",
                "\uff01\n",  # Fullwidth exclamation mark
                "!\n",
                "\u3002\n",  # Ideographic full stop
                ".\n",
                "\uff1f",  # Fullwidth question mark
                "? ",
                "\uff01",  # Fullwidth exclamation mark
                "! ",
                "\u3002",  # Ideographic full stop
                ". ",
                "; ",
                ": ",
                "—",
                "--",
                "\uff0c",  # Fullwidth comma
                "\u3001",  # Ideographic comma
                ", ",
                "\u200b",  # Zero-width space
                " ",
                "",
            ],
            keep_separator="end",
        )

    async def startup(self) -> None:
        """Start the periodic purge loop if purge_interval is configured."""
        if self._purge_interval is not None:
            self._shutdown_event.clear()
            self._purge_task = asyncio.create_task(self._purge_loop())

    async def shutdown(self) -> None:
        """Stop the periodic purge loop."""
        if self._purge_task is not None:
            self._shutdown_event.set()
            await self._purge_task
            self._purge_task = None

    async def _purge_loop(self) -> None:
        """Periodically purge orphaned derivatives."""
        assert self._purge_interval is not None
        while True:
            try:
                await asyncio.wait_for(
                    self._shutdown_event.wait(), timeout=self._purge_interval
                )
            except TimeoutError:
                pass
            else:
                # Shutdown event was set, so exit the loop.
                return

            try:
                await self._purge_orphaned_derivatives()
            except Exception:
                logger.exception("Error during derivative purge cycle")

    async def _purge_orphaned_derivatives(self) -> None:
        """Run a single purge cycle: identify, mark, delete from collection, then remove."""
        orphan_uuids = list(
            await self._segment_linker_partition.get_orphaned_derivatives()
        )
        if not orphan_uuids:
            return

        marked_uuids = list(
            await self._segment_linker_partition.mark_orphaned_derivatives_for_purging(
                orphan_uuids
            )
        )
        if not marked_uuids:
            return

        await self._collection.delete(record_uuids=marked_uuids)
        await self._segment_linker_partition.purge_derivatives(marked_uuids)

    async def encode_episodes(
        self,
        episodes: Iterable[Episode],
    ) -> None:
        """
        Encode episodes.

        Args:
            episodes (Iterable[Episode]): The episodes to encode.

        """
        episodes = sorted(
            episodes,
            key=lambda episode: (episode.timestamp, episode.uuid),
        )

        episodes_segments = await asyncio.gather(
            *[self._create_segments(episode) for episode in episodes]
        )

        segments = [
            segment
            for episode_segments in episodes_segments
            for segment in episode_segments
        ]

        segments_derivatives = await asyncio.gather(
            *[self._derive_derivatives(segment) for segment in segments]
        )

        derivatives = [
            derivative
            for segment_derivatives in segments_derivatives
            for derivative in segment_derivatives
        ]
        derivative_uuids_set = {d.uuid for d in derivatives}

        derivative_embeddings = await self._embedder.ingest_embed(
            [derivative.text for derivative in derivatives],
        )

        if self._derivative_consolidation_threshold is None:
            await self._segment_linker_partition.register_segments(
                links={
                    segment: [derivative.uuid for derivative in segment_derivatives]
                    for segment, segment_derivatives in zip(
                        segments,
                        segments_derivatives,
                        strict=True,
                    )
                },
                active=None,
            )

            derivative_records = [
                Record(
                    uuid=derivative.uuid,
                    vector=derivative_embedding,
                )
                for derivative, derivative_embedding in zip(
                    derivatives,
                    derivative_embeddings,
                    strict=True,
                )
            ]

        else:
            derivative_query_results = await self._collection.query(
                query_vectors=derivative_embeddings,
                score_threshold=self._derivative_consolidation_threshold,
                limit=1,
                return_vector=False,
                return_properties=False,
            )

            # Retry on stale derivative matches.
            excluded_uuids: set[UUID] = set()
            while True:
                consolidated_uuids = ExtraMemory._consolidate_derivatives(
                    derivatives=derivatives,
                    derivative_embeddings=derivative_embeddings,
                    derivative_query_results=derivative_query_results,
                    excluded_uuids=excluded_uuids,
                    score_threshold=self._derivative_consolidation_threshold,
                    similarity_metric=self._embedder.similarity_metric,
                )
                active_uuids = set(consolidated_uuids) - derivative_uuids_set

                consolidated_uuids_per_segment = unflatten_like(
                    consolidated_uuids, segments_derivatives
                )
                links = dict(zip(segments, consolidated_uuids_per_segment, strict=True))

                try:
                    await self._segment_linker_partition.register_segments(
                        links=links,
                        active=active_uuids,
                    )
                    break
                except DerivativeNotActiveError as e:
                    excluded_uuids |= e.not_active

            derivative_records = [
                Record(
                    uuid=derivative.uuid,
                    vector=derivative_embedding,
                )
                for consolidated_uuid, derivative, derivative_embedding in zip(
                    consolidated_uuids,
                    derivatives,
                    derivative_embeddings,
                    strict=True,
                )
                if consolidated_uuid == derivative.uuid
            ]

        if derivative_records:
            await self._collection.upsert(records=derivative_records)

    @staticmethod
    def _consolidate_derivatives(
        derivatives: Iterable[Derivative],
        derivative_embeddings: Iterable[list[float]],
        derivative_query_results: Iterable[VectorStoreQueryResult],
        excluded_uuids: Iterable[UUID],
        score_threshold: float,
        similarity_metric: SimilarityMetric,
    ) -> list[UUID]:
        """
        Consolidate derivatives by deduplicating within-batch and against the DB.

        Args:
            derivatives (Iterable[Derivative]): The derivatives to consolidate.
            derivative_embeddings (Iterable[list[float]]): The embeddings of the derivatives, in the same order.
            derivative_query_results (Iterable[VectorStoreQueryResult]): Pre-fetched DB query results for the derivatives, in the same order.
            excluded_uuids (Iterable[UUID]): DB derivative UUIDs to exclude from consolidation.
            score_threshold (float): Score threshold for consolidation.
            similarity_metric (SimilarityMetric): Metric to use for comparing similarity scores.

        Returns:
            list[UUID]: The consolidated UUID for each derivative, in the same order.

        """
        consolidated_uuids: list[UUID] = []
        representatives: dict[UUID, list[float]] = {}

        for derivative, derivative_embedding, derivative_query_result in zip(
            derivatives,
            derivative_embeddings,
            derivative_query_results,
            strict=True,
        ):
            derivative_matches = derivative_query_result.matches

            best_db_uuid: UUID | None = None
            best_db_score: float | None = None
            for match in derivative_matches:
                if match.record.uuid not in excluded_uuids:
                    best_db_uuid = match.record.uuid
                    best_db_score = match.score
                    break

            consolidated_uuid = ExtraMemory._find_best_consolidation_match(
                derivative_uuid=derivative.uuid,
                derivative_embedding=derivative_embedding,
                representatives=representatives,
                best_db_uuid=best_db_uuid,
                best_db_score=best_db_score,
                score_threshold=score_threshold,
                similarity_metric=similarity_metric,
            )
            consolidated_uuids.append(consolidated_uuid)

            if consolidated_uuid == derivative.uuid:
                representatives[consolidated_uuid] = derivative_embedding

        return consolidated_uuids

    @staticmethod
    def _find_best_consolidation_match(
        derivative_uuid: UUID,
        derivative_embedding: list[float],
        representatives: Mapping[UUID, list[float]],
        best_db_uuid: UUID | None,
        best_db_score: float | None,
        score_threshold: float,
        similarity_metric: SimilarityMetric,
    ) -> UUID:
        """
        Find the best consolidation match for a derivative embedding.

        Args:
            derivative_uuid (UUID): The UUID of the derivative being consolidated.
            derivative_embedding (list[float]): The embedding of the derivative being consolidated.
            representatives (Mapping[UUID, list[float]]): Representative derivative UUIDs and their embeddings for within-batch consolidation.
            best_db_score (float | None): The pre-computed best similarity score against the DB for this derivative, or None if there are no matches.
            best_db_uuid (UUID | None): The UUID of the best DB match for this derivative, or None if there are no matches.
            score_threshold (float): The similarity score threshold for consolidation.
            similarity_metric (SimilarityMetric): The similarity metric to use for comparing scores.

        Returns:
            UUID:
                The UUID of the best match for consolidation.
                If no match exceeds the threshold, returns the derivative's own UUID.

        """
        gt = similarity_gt(similarity_metric)

        best_uuid: UUID = derivative_uuid
        best_score: float | None = None

        # Check within-batch representatives.
        representative_uuids = list(representatives.keys())
        batch_scores = compute_similarity(
            query_embedding=derivative_embedding,
            candidate_embeddings=list(representatives.values()),
            similarity_metric=similarity_metric,
        )

        # We do not use max() because gt is similarity-metric dependent.
        for uuid, score in zip(representative_uuids, batch_scores, strict=True):
            if gt(score, score_threshold) and (
                best_score is None or gt(score, best_score)
            ):
                best_score = score
                best_uuid = uuid

        # Check DB match.
        if (
            best_db_uuid is not None
            and best_db_score is not None
            and gt(best_db_score, score_threshold)
            and (best_score is None or gt(best_db_score, best_score))
        ):
            best_uuid = best_db_uuid

        return best_uuid

    async def _create_segments(
        self,
        episode: Episode,
    ) -> list[Segment]:
        """
        Create segments from an episode.

        Args:
            episode (Episode):
                The episode from which to create segments.

        Returns:
            list[Segment]: A list of created segments.

        """
        match episode.body:
            case Content(context=context, items=primitives):
                return self._segment_episode_content_items(
                    episode=episode,
                    items=primitives,
                    context=context,
                )
            case ReadFile(file=file_ref):
                return [
                    Segment(
                        uuid=uuid4(),
                        episode_uuid=episode.uuid,
                        index=0,
                        offset=0,
                        timestamp=episode.timestamp,
                        block=file_ref,
                        properties=episode.properties,
                    )
                ]
            case _:
                logger.warning("Unsupported body type: %s", type(episode.body))
                return []

    def _segment_episode_content_items(
        self,
        episode: Episode,
        items: Iterable[Block],
        context: Context | None,
    ) -> list[Segment]:
        """Split content items into single-block segments, propagating context."""
        segments: list[Segment] = []
        for index, item in enumerate(items):
            match item:
                case Text(text=text):
                    chunks = self._text_splitter.split_text(text)
                    segments.extend(
                        Segment(
                            uuid=uuid4(),
                            episode_uuid=episode.uuid,
                            index=index,
                            offset=offset,
                            timestamp=episode.timestamp,
                            block=Text(text=chunk),
                            context=context,
                            properties=episode.properties,
                        )
                        for offset, chunk in enumerate(chunks)
                    )
                case _:
                    segments.append(
                        Segment(
                            uuid=uuid4(),
                            episode_uuid=episode.uuid,
                            index=index,
                            offset=0,
                            timestamp=episode.timestamp,
                            block=item,
                            context=context,
                            properties=episode.properties,
                        )
                    )
        return segments

    async def _derive_derivatives(
        self,
        segment: Segment,
    ) -> list[Derivative]:
        """
        Derive derivatives from a segment.

        Args:
            segment (Segment):
                The segment from which to derive derivatives.

        Returns:
            list[Derivative]: A list of derived derivatives.

        """
        match segment.block:
            case Text(text=text):
                return self._derive_from_text(text, segment.context)
            case FileRef():
                return []
            case _:
                logger.warning("Non-text primitive derivatives are not yet supported")
                return []

    @staticmethod
    def _format_with_context(text: str, context: Context | None) -> str:
        """Format text within its context."""
        match context:
            case MessageContext(source=source):
                return f"{source}: {text}"
            case CitationContext(source=source):
                return f"From '{source}': {text}"
            case _:
                return text

    def _derive_from_text(self, text: str, context: Context | None) -> list[Derivative]:
        """Derive derivatives from a text string."""
        if not self._derive_sentences:
            return [
                Derivative(
                    uuid=uuid4(), text=ExtraMemory._format_with_context(text, context)
                )
            ]
        sentences = extract_sentences(text)
        return [
            Derivative(
                uuid=uuid4(), text=ExtraMemory._format_with_context(sentence, context)
            )
            for sentence in sentences
        ]

    async def query(
        self,
        query: str,
        *,
        max_num_segments: int = 20,
        expand_context: int = 0,
        property_filter: FilterExpr | None = None,
    ) -> QueryResult:
        """
        Query extra memory for segments relevant to the query.

        Args:
            query (str):
                The search query.
            max_num_segments (int):
                The maximum number of segments to return
                (default: 20).
            expand_context (int):
                The number of additional segments to include
                around each matched segment for additional context
                (default: 0).
            property_filter (FilterExpr | None):
                Attribute keys and values
                to use for filtering segments
                (default: None).

        Returns:
            list[QueryResult]:
                A list of query results containing segments relevant to the query, ordered chronologically.

        """
        query_embedding = (
            await self._embedder.search_embed(
                [query],
            )
        )[0]

        # Search derivative collection for matches.
        query_result = next(
            iter(
                await self._collection.query(
                    query_vectors=[query_embedding],
                    limit=min(5 * max_num_segments, 200),
                    return_vector=False,
                    return_properties=False,
                )
            )
        )

        matched_derivative_uuids = [match.record.uuid for match in query_result.matches]

        segments_by_derivatives = (
            await self._segment_linker_partition.get_segments_by_derivatives(
                derivative_uuids=matched_derivative_uuids,
                property_filter=property_filter,
            )
        )

        # Preserve vector search similarity ordering.
        seed_segments = [
            segment
            for derivative_uuid in matched_derivative_uuids
            if derivative_uuid in segments_by_derivatives
            for segment in segments_by_derivatives[derivative_uuid]
        ]

        expand_context = min(max(0, expand_context), max_num_segments - 1)
        max_backward_segments = expand_context // 3
        max_forward_segments = expand_context - max_backward_segments

        segment_contexts_by_seed = (
            await self._segment_linker_partition.get_segment_contexts(
                seed_segment_uuids=[segment.uuid for segment in seed_segments],
                max_backward_segments=max_backward_segments,
                max_forward_segments=max_forward_segments,
                property_filter=property_filter,
            )
        )

        # Build aligned lists, preserving similarity ordering from seed_segments.
        # Deduplicate by UUID (multiple derivatives can map to the same segment).
        kept_seed_segments = list(
            dict.fromkeys(
                seed_segment
                for seed_segment in seed_segments
                if seed_segment.uuid in segment_contexts_by_seed
            )
        )
        segment_contexts: list[list[Segment]] = [
            list(segment_contexts_by_seed[seed_segment.uuid])
            for seed_segment in kept_seed_segments
        ]

        # Rerank segment contexts.
        segment_context_scores = await self._score_segment_contexts(
            query,
            segment_contexts,
        )

        reranked_anchored_segment_contexts = [
            (seed_segment, segment_context)
            for _, seed_segment, segment_context in sorted(
                zip(
                    segment_context_scores,
                    kept_seed_segments,
                    segment_contexts,
                    strict=True,
                ),
                key=lambda triple: triple[0],
                reverse=True,
            )
        ]

        # Unify segment contexts.
        unified_segment_context = ExtraMemory._unify_anchored_segment_contexts(
            reranked_anchored_segment_contexts,
            max_num_segments=max_num_segments,
        )

        unified_segment_context_string = ExtraMemory.string_from_segment_context(
            unified_segment_context
        )

        return QueryResult(
            unified_segment_context=unified_segment_context,
            unified_segment_context_string=unified_segment_context_string,
        )

    async def _score_segment_contexts(
        self,
        query: str,
        segment_contexts: Iterable[Iterable[Segment]],
    ) -> list[float]:
        """Score segment contexts based on their relevance to the query."""
        context_strings = []
        for segment_context in segment_contexts:
            context_string = ExtraMemory.string_from_segment_context(segment_context)
            context_strings.append(context_string)

        segment_context_scores = await self._reranker.score(query, context_strings)
        return segment_context_scores

    @staticmethod
    def string_from_segment_context(segment_context: Iterable[Segment]) -> str:
        """Format segment context as a string."""
        context_string = ""
        last_segment: Segment | None = None
        accumulated_text = ""
        first = True

        for segment in segment_context:
            is_continuation = (
                last_segment is not None
                and segment.episode_uuid == last_segment.episode_uuid
                and segment.index == last_segment.index
            )

            if not is_continuation:
                if not first:
                    context_string += json.dumps(accumulated_text) + "\n"
                first = False
                accumulated_text = ""

                context_date = ExtraMemory._format_date(
                    segment.timestamp.date(),
                )
                context_time = ExtraMemory._format_time(
                    segment.timestamp.time(),
                )
                timestamp = f"[{context_date} at {context_time}]"

                match segment.context:
                    case MessageContext(source=source):
                        context_string += f"{timestamp} {source}: "
                    case CitationContext(source=source):
                        context_string += f"{timestamp} From '{source}': "
                    case _:
                        context_string += f"{timestamp} "

            text = ExtraMemory._extract_text(segment.block)
            if text is not None:
                accumulated_text += text
            elif not is_continuation:
                context_string += f"[{segment.block.type}]\n"

            last_segment = segment

        if not first:
            context_string += json.dumps(accumulated_text) + "\n"

        return context_string.strip()

    @staticmethod
    def _extract_text(block: Block) -> str | None:
        """Extract text from a block, if it contains text."""
        match block:
            case Text(text=text):
                return text
            case _:
                return None

    @staticmethod
    def _format_date(date: datetime.date) -> str:
        """Format the date as a string."""
        return date.strftime("%A, %B %d, %Y")

    @staticmethod
    def _format_time(time: datetime.time) -> str:
        """Format the time as a string."""
        return time.strftime("%I:%M %p")

    async def forget_episodes(self, episode_uuids: Iterable[UUID]) -> None:
        """Forget episodes by their UUIDs."""
        await self._segment_linker_partition.delete_segments_by_episodes(
            episode_uuids=episode_uuids,
        )

    async def forget_all_episodes(self) -> None:
        """Forget all episodes in this partition."""
        await self._segment_linker_partition.delete_all_segments()

    @staticmethod
    def _unify_anchored_segment_contexts(
        anchored_segment_contexts: Iterable[tuple[Segment, Iterable[Segment]]],
        max_num_segments: int,
    ) -> list[Segment]:
        """Unify anchored segment contexts into a single list within the limit."""
        unified_segment_context_set: set[Segment] = set()

        for seed_segment, context in anchored_segment_contexts:
            context = list(context)

            if len(unified_segment_context_set) >= max_num_segments:
                break
            if (len(unified_segment_context_set) + len(context)) <= max_num_segments:
                # It is impossible that the context exceeds the limit.
                unified_segment_context_set.update(context)
            else:
                # It is possible that the context exceeds the limit.
                # Prioritize segments near the seed segment.

                # Sort chronological segments by weighted index-proximity to the seed segment.
                seed_index = context.index(seed_segment)

                seed_context = sorted(
                    context,
                    key=lambda segment: ExtraMemory._weighted_index_proximity(
                        segment=segment,
                        context=context,
                        seed_index=seed_index,
                    ),
                )

                # Add segments to unified context until limit is reached,
                # or until the context is exhausted.
                for segment in seed_context:
                    if len(unified_segment_context_set) >= max_num_segments:
                        break
                    unified_segment_context_set.add(segment)

        unified_segment_context = sorted(
            unified_segment_context_set,
            key=lambda segment: (
                segment.timestamp,
                segment.episode_uuid,
                segment.index,
                segment.offset,
            ),
        )

        return unified_segment_context

    @staticmethod
    def _weighted_index_proximity(
        segment: Segment,
        context: list[Segment],
        seed_index: int,
    ) -> float:
        proximity = context.index(segment) - seed_index
        if proximity >= 0:
            # Forward recall is better than backward recall.
            return (proximity - 0.5) / 2
        return -proximity
