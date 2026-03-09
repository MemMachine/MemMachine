"""Extra memory system for storing and retrieving episodic memory."""

import asyncio
import datetime
import json
import logging
from collections.abc import Iterable
from uuid import UUID, uuid4

from langchain_text_splitters import RecursiveCharacterTextSplitter
from pydantic import BaseModel, Field, InstanceOf

from memmachine_server.common.embedder import Embedder
from memmachine_server.common.filter.filter_parser import (
    FilterExpr,
)
from memmachine_server.common.reranker import Reranker
from memmachine_server.common.utils import extract_sentences
from memmachine_server.common.vector_store import (
    Collection,
    Record,
)

from .data_types import (
    Derivative,
    Episode,
    ImageContent,
    MessageContent,
    QueryResult,
    Segment,
    TextContent,
)
from .segment_linker import DerivativeNotActiveError, SegmentLinker

logger = logging.getLogger(__name__)


class ExtraMemoryParams(BaseModel):
    """
    Parameters for ExtraMemory.

    Attributes:
        session_key (str):
            Session key.
        collection (Collection):
            Collection instance in a vector store.
        segment_linker (SegmentLinker):
            Segment linker instance for managing segments.
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
    """

    session_key: str = Field(
        ...,
        description="Session key",
    )
    collection: InstanceOf[Collection] = Field(
        ...,
        description="Collection instance in a vector store",
    )
    segment_linker: InstanceOf[SegmentLinker] = Field(
        ...,
        description="Segment linker instance for managing segments",
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


class ExtraMemory:
    """Extra memory system."""

    def __init__(self, params: ExtraMemoryParams) -> None:
        """
        Initialize an ExtraMemory with the provided parameters.

        Args:
            params (ExtraMemoryParams):
                Parameters for the ExtraMemory.

        """
        self._session_key = params.session_key
        self._collection = params.collection
        self._segment_linker = params.segment_linker

        self._embedder = params.embedder
        self._reranker = params.reranker

        self._derive_sentences = params.derive_sentences
        self._derivative_consolidation_threshold = (
            params.derivative_consolidation_threshold
        )

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
            [derivative.content for derivative in derivatives],
        )

        if self._derivative_consolidation_threshold is not None:
            # TODO: Logic to discover existing derivatives.
            pass

        # TODO: Loop condition
        while True:
            try:
                await self._segment_linker.register_segments(
                    session_key=self._session_key,
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
                break
            except DerivativeNotActiveError:
                # TODO: Logic to use new derivatives instead of existing ones.
                break

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
        segments: list[Segment] = []
        for block, content in enumerate(episode.content):
            match content:
                case MessageContent(source=source, text=text):
                    chunks = self._text_splitter.split_text(text)
                    segments.extend(
                        Segment(
                            uuid=uuid4(),
                            episode_uuid=episode.uuid,
                            block=block,
                            index=index,
                            timestamp=episode.timestamp,
                            content=MessageContent(source=source, text=chunk),
                            properties=episode.properties,
                        )
                        for index, chunk in enumerate(chunks)
                    )
                case TextContent(text=text):
                    chunks = self._text_splitter.split_text(text)
                    segments.extend(
                        Segment(
                            uuid=uuid4(),
                            episode_uuid=episode.uuid,
                            block=block,
                            index=index,
                            timestamp=episode.timestamp,
                            content=TextContent(text=chunk),
                            properties=episode.properties,
                        )
                        for index, chunk in enumerate(chunks)
                    )
                case ImageContent():
                    segments.append(
                        Segment(
                            uuid=uuid4(),
                            episode_uuid=episode.uuid,
                            block=block,
                            index=0,
                            timestamp=episode.timestamp,
                            content=content,
                            properties=episode.properties,
                        )
                    )
                case _:
                    logger.warning("Unsupported content type: %s", type(content))
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
        match segment.content:
            case MessageContent(source=source, text=text):
                if not self._derive_sentences:
                    return [
                        Derivative(
                            uuid=uuid4(),
                            content=f"{source}: {text}",
                        ),
                    ]
                sentences = extract_sentences(text)
                return [
                    Derivative(
                        uuid=uuid4(),
                        content=f"{source}: {sentence}",
                    )
                    for sentence in sentences
                ]
            case TextContent(text=text):
                if not self._derive_sentences:
                    return [
                        Derivative(
                            uuid=uuid4(),
                            content=text,
                        ),
                    ]
                sentences = extract_sentences(text)
                return [
                    Derivative(
                        uuid=uuid4(),
                        content=sentence,
                    )
                    for sentence in sentences
                ]
            case ImageContent():
                # TODO: Generate image description.
                logger.warning("Image content derivatives are not yet supported")
                return []
            case _:
                logger.warning(
                    "Unsupported content type for derivatives: %s",
                    type(segment.content),
                )
                return []

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
        query_results = next(
            iter(
                await self._collection.query(
                    query_vectors=[query_embedding],
                    limit=min(5 * max_num_segments, 200),
                    return_vector=False,
                    return_properties=False,
                )
            )
        )

        matched_derivative_uuids = [
            query_result.record.uuid for query_result in query_results
        ]

        segments_by_derivatives = (
            await self._segment_linker.get_segments_by_derivatives(
                session_key=self._session_key,
                derivative_uuids=matched_derivative_uuids,
                property_filter=property_filter,
            )
        )

        seed_segments = [
            segment
            for linked_segments in segments_by_derivatives.values()
            for segment in linked_segments
        ]

        expand_context = min(max(0, expand_context), max_num_segments - 1)
        max_backward_segments = expand_context // 3
        max_forward_segments = expand_context - max_backward_segments

        segment_contexts = [
            list(context)
            for context in (
                await self._segment_linker.get_segment_contexts(
                    session_key=self._session_key,
                    seed_segment_uuids=[segment.uuid for segment in seed_segments],
                    max_backward_segments=max_backward_segments,
                    max_forward_segments=max_forward_segments,
                    property_filter=property_filter,
                )
            ).values()
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
                    seed_segments,
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

        for segment in segment_context:
            is_continuation = (
                last_segment is not None
                and segment.episode_uuid == last_segment.episode_uuid
                and segment.block == last_segment.block
            )

            if is_continuation:
                match segment.content:
                    case MessageContent(text=text) | TextContent(text=text):
                        accumulated_text += text
                continue

            context_date = ExtraMemory._format_date(
                segment.timestamp.date(),
            )
            context_time = ExtraMemory._format_time(
                segment.timestamp.time(),
            )

            match segment.content:
                case MessageContent(source=source, text=text):
                    if accumulated_text:
                        context_string += json.dumps(accumulated_text) + "\n"
                    context_string += f"[{context_date} at {context_time}] {source}: "
                    accumulated_text = text
                case TextContent(text=text):
                    if accumulated_text:
                        context_string += json.dumps(accumulated_text) + "\n"
                    context_string += f"[{context_date} at {context_time}] "
                    accumulated_text = text
                case _:
                    if accumulated_text:
                        context_string += json.dumps(accumulated_text) + "\n"
                    context_string += (
                        f"[{context_date} at {context_time}] [{segment.content.type}]\n"
                    )
                    accumulated_text = ""

            last_segment = segment

        if accumulated_text:
            context_string += json.dumps(accumulated_text) + "\n"

        return context_string.strip()

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
        await self._segment_linker.delete_segments_by_episodes(
            session_key=self._session_key,
            episode_uuids=episode_uuids,
        )
        # TODO: Add background job to delete records from collection based on reference counting. May belong to class instead of instance.

    async def forget_all_episodes(self) -> None:
        """Forget all episodes in this session."""
        await self._segment_linker.delete_all_segments(
            session_key=self._session_key,
        )
        # TODO: Add background job to delete records from collection based on reference counting. May belong to class instead of instance.

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
                segment.block,
                segment.index,
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
