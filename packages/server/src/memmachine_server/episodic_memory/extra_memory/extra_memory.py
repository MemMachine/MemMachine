"""Extra memory system for storing and retrieving episodic memory."""

import asyncio
import datetime
import json
import logging
from collections.abc import Iterable
from uuid import UUID, uuid4, uuid5

from langchain_text_splitters import RecursiveCharacterTextSplitter
from pydantic import BaseModel, Field, InstanceOf

from memmachine_server.common.data_types import PropertyValue
from memmachine_server.common.embedder import Embedder
from memmachine_server.common.filter.filter_parser import (
    FilterExpr,
)
from memmachine_server.common.reranker import Reranker
from memmachine_server.common.utils import (
    extract_sentences,
)
from memmachine_server.common.vector_store import (
    Collection,
    Record,
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
from .segment_linker import SegmentLinkerPartition

logger = logging.getLogger(__name__)

_DERIVATIVE_UUID_NAMESPACE = UUID("0af4a33f-57b4-4a38-a412-c03f9a9929bc")


class ExtraMemoryParams(BaseModel):
    """
    Parameters for ExtraMemory.

    Attributes:
        partition_key (str):
            Partition key for scoping the memory.
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
        deduplicate_derivatives (bool):
            Whether to deduplicate derivatives by content using uuid5 (default: False).
        max_text_chunk_length (int):
            Max code-point length for text chunking in segment creation (default: 2000).
        purge_interval (float | None):
            Seconds between purge cycles. None disables periodic purging (default: None).
    """

    partition_key: str = Field(
        ...,
        description="Partition key for scoping the memory",
    )
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
    deduplicate_derivatives: bool = Field(
        False,
        description="Whether to deduplicate derivatives by content using uuid5.",
    )
    max_text_chunk_length: int = Field(
        2000,
        description="Max code-point length for text chunking in segment creation",
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
        self._partition_key = params.partition_key

        self._collection = params.collection
        self._segment_linker_partition = params.segment_linker_partition

        self._embedder = params.embedder
        self._reranker = params.reranker

        self._derive_sentences = params.derive_sentences
        self._deduplicate_derivatives = params.deduplicate_derivatives

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
        """Run a single purge cycle."""
        await self._segment_linker_partition.mark_orphaned_derivatives_for_purging()

        pending_uuids = (
            await self._segment_linker_partition.get_derivatives_pending_purge()
        )
        if not pending_uuids:
            return

        await self._collection.delete(record_uuids=pending_uuids)
        await self._segment_linker_partition.purge_derivatives(pending_uuids)

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

        derivative_embeddings = await self._embedder.ingest_embed(
            [derivative.text for derivative in derivatives],
        )

        links = {
            segment: [derivative.uuid for derivative in segment_derivatives]
            for segment, segment_derivatives in zip(
                segments,
                segments_derivatives,
                strict=True,
            )
        }

        await self._segment_linker_partition.register_segments(links=links)

        # In non-deduplication mode, attach segment properties to vector records.
        derivative_properties: dict[UUID, dict[str, PropertyValue]] | None = None
        if not self._deduplicate_derivatives:
            derivative_properties = {}
            for segment, segment_derivatives in zip(
                segments, segments_derivatives, strict=True
            ):
                if segment.properties:
                    for derivative in segment_derivatives:
                        derivative_properties[derivative.uuid] = segment.properties

        derivative_records = [
            Record(
                uuid=derivative.uuid,
                vector=derivative_embedding,
                properties=(
                    derivative_properties.get(derivative.uuid)
                    if derivative_properties is not None
                    else None
                ),
            )
            for derivative, derivative_embedding in zip(
                derivatives,
                derivative_embeddings,
                strict=True,
            )
        ]

        if derivative_records:
            await self._collection.upsert(records=derivative_records)

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
            formatted = ExtraMemory._format_with_context(text, context)
            return [
                Derivative(
                    uuid=self._derivative_uuid(formatted),
                    text=formatted,
                )
            ]
        sentences = extract_sentences(text)
        derivatives = []
        for sentence in sentences:
            formatted = ExtraMemory._format_with_context(sentence, context)
            derivatives.append(
                Derivative(
                    uuid=self._derivative_uuid(formatted),
                    text=formatted,
                )
            )
        return derivatives

    def _derivative_uuid(self, text: str) -> UUID:
        if self._deduplicate_derivatives:
            return uuid5(
                _DERIVATIVE_UUID_NAMESPACE,
                json.dumps([self._partition_key, text]),
            )
        return uuid4()

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

        # Filter at the vector DB level in non-dedup mode.
        vector_filter = property_filter if not self._deduplicate_derivatives else None

        # Search derivative collection for matches.
        [query_result] = await self._collection.query(
            query_vectors=[query_embedding],
            limit=min(5 * max_num_segments, 200),
            property_filter=vector_filter,
            return_vector=False,
            return_properties=False,
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
            segment_contexts_by_seed[seed_segment.uuid]
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
