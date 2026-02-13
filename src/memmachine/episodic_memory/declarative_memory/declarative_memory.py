"""Declarative memory system for storing and retrieving episodic memory."""

import asyncio
import datetime
import json
import logging
from collections.abc import Iterable
from uuid import UUID, uuid4

from pydantic import BaseModel, Field, InstanceOf

from memmachine.common.embedder.embedder import Embedder
from memmachine.common.filter.filter_parser import (
    FilterExpr,
)
from memmachine.common.reranker.reranker import Reranker
from memmachine.common.utils import extract_sentences
from memmachine.common.vector_store import Collection, Record

from .data_types import (
    ContentType,
    Derivative,
    Episode,
    Segment,
)
from .segment_store import SegmentStore

logger = logging.getLogger(__name__)


class DeclarativeMemoryParams(BaseModel):
    """
    Parameters for DeclarativeMemory.

    Attributes:
        session_key (str):
            Session key.
        collection (Collection):
            Collection instance in a vector store.
        segment_store (SegmentStore):
            Segment store instance for managing segments.
        embedder (Embedder):
            Embedder instance for creating embeddings.
        reranker (Reranker):
            Reranker instance for reranking search results.

    """

    session_key: str = Field(
        ...,
        description="Session key",
    )
    collection: InstanceOf[Collection] = Field(
        ...,
        description="Collection instance in a vector store",
    )
    segment_store: InstanceOf[SegmentStore] = Field(
        ...,
        description="Segment store instance for managing segments",
    )
    embedder: InstanceOf[Embedder] = Field(
        ...,
        description="Embedder instance for creating embeddings",
    )
    reranker: InstanceOf[Reranker] = Field(
        ...,
        description="Reranker instance for reranking search results",
    )
    message_sentence_chunking: bool = Field(
        False,
        description="Whether to chunk message episodes into sentences for embedding",
    )


class DeclarativeMemory:
    """Declarative memory system."""

    def __init__(self, params: DeclarativeMemoryParams) -> None:
        """
        Initialize a DeclarativeMemory with the provided parameters.

        Args:
            params (DeclarativeMemoryParams):
                Parameters for the DeclarativeMemory.

        """
        self._session_key = params.session_key
        self._collection = params.collection
        self._segment_store = params.segment_store

        self._embedder = params.embedder
        self._reranker = params.reranker

        self._message_sentence_chunking = params.message_sentence_chunking

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

        derivative_records = [
            Record(
                uuid=derivative.uuid,
                vector=derivative_embedding,
                properties={
                    "_segment_uuid": str(derivative.segment_uuid),
                    "_episode_uuid": str(derivative.episode_uuid),
                    "_timestamp": derivative.timestamp,
                    "_context": derivative.context,
                    "_content_type": derivative.content_type.value,
                    "_content": derivative.content,
                    **(derivative.attributes or {}),
                },
            )
            for derivative, derivative_embedding in zip(
                derivatives,
                derivative_embeddings,
                strict=True,
            )
        ]

        await self._segment_store.add_segments(
            session_key=self._session_key,
            segments={
                segment: [derivative.uuid for derivative in segment_derivatives]
                for segment, segment_derivatives in zip(
                    segments,
                    segments_derivatives,
                    strict=True,
                )
            },
        )

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
        return [
            Segment(
                uuid=uuid4(),
                episode_uuid=episode.uuid,
                index=index,
                timestamp=episode.timestamp,
                context=episode.context,
                content_type=episode.content_type,
                content=episode.content,
                attributes=episode.attributes,
            )
            for index in range(1)
        ]

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
        match segment.content_type:
            case ContentType.MESSAGE:
                if not self._message_sentence_chunking:
                    return [
                        Derivative(
                            uuid=uuid4(),
                            segment_uuid=segment.uuid,
                            episode_uuid=segment.episode_uuid,
                            timestamp=segment.timestamp,
                            context=segment.context,
                            content_type=ContentType.MESSAGE,
                            content=f"{segment.context}: {segment.content}",
                            attributes=segment.attributes,
                        ),
                    ]

                sentences = extract_sentences(segment.content)

                return [
                    Derivative(
                        uuid=uuid4(),
                        segment_uuid=segment.uuid,
                        episode_uuid=segment.episode_uuid,
                        timestamp=segment.timestamp,
                        context=segment.context,
                        content_type=ContentType.MESSAGE,
                        content=f"{segment.context}: {sentence}",
                        attributes=segment.attributes,
                    )
                    for sentence in sentences
                ]
            case ContentType.TEXT:
                return [
                    Derivative(
                        uuid=uuid4(),
                        segment_uuid=segment.uuid,
                        episode_uuid=segment.episode_uuid,
                        timestamp=segment.timestamp,
                        context=segment.context,
                        content_type=ContentType.TEXT,
                        content=segment.content,
                        attributes=segment.attributes,
                    ),
                ]
            case _:
                logger.warning(
                    "Unsupported content type for derivative derivation: %s",
                    segment.content_type,
                )
                return []

    async def search(
        self,
        query: str,
        *,
        max_num_segments: int = 20,
        expand_context: int = 0,
        property_filter: FilterExpr | None = None,
    ) -> list[Segment]:
        """
        Search declarative memory for segments relevant to the query.

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
            list[Segment]:
                A list of segments relevant to the query, ordered chronologically.

        """
        scored_segments = await self.search_scored(
            query,
            max_num_segments=max_num_segments,
            expand_context=expand_context,
            property_filter=property_filter,
        )
        return [segment for _, segment in scored_segments]

    async def search_scored(
        self,
        query: str,
        *,
        max_num_segments: int = 20,
        expand_context: int = 0,
        property_filter: FilterExpr | None = None,
    ) -> list[tuple[float, Segment]]:
        """
        Search declarative memory for segments relevant to the query, returning scored segments.

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
            list[tuple[float, Segment]]:
                A list of scored segments relevant to the query, ordered chronologically.

        """
        query_embedding = (
            await self._embedder.search_embed(
                [query],
            )
        )[0]

        # Search derivative collection for matches.
        query_results = await self._collection.query(
            query_vector=query_embedding,
            limit=max(5 * max_num_segments, 200),
            property_filter=property_filter,
            return_vector=False,
            return_properties=True,
        )

        matched_derivative_records = [
            query_result.record for query_result in query_results
        ]

        # Get origin segments of matched derivatives.
        derivative_origin_segment_uuids = dict.fromkeys(
            UUID(matched_derivative_record.properties["_segment_uuid"])
            for matched_derivative_record in matched_derivative_records
        )

        seed_segment_uuids = list(derivative_origin_segment_uuids.keys())

        expand_context = min(max(0, expand_context), max_num_segments - 1)
        max_backward_segments = expand_context // 3
        max_forward_segments = expand_context - max_backward_segments

        segment_contexts = [
            list(context)
            for context in await self._segment_store.get_segment_contexts(
                session_key=self._session_key,
                seed_segment_uuids=seed_segment_uuids,
                max_backward_segments=max_backward_segments,
                max_forward_segments=max_forward_segments,
                property_filter=property_filter,
            )
        ]

        # Rerank segment contexts.
        segment_context_scores = await self._score_segment_contexts(
            query,
            segment_contexts,
        )

        seed_segments = [
            next(s for s in context if s.uuid == seed_uuid)
            for seed_uuid, context in zip(
                seed_segment_uuids,
                segment_contexts,
                strict=True,
            )
        ]

        reranked_scored_anchored_segment_contexts = [
            (segment_context_score, seed_segment, segment_context)
            for segment_context_score, seed_segment, segment_context in sorted(
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
        unified_scored_segment_context = (
            DeclarativeMemory._unify_scored_anchored_segment_contexts(
                reranked_scored_anchored_segment_contexts,
                max_num_segments=max_num_segments,
            )
        )
        return unified_scored_segment_context

    async def _score_segment_contexts(
        self,
        query: str,
        segment_contexts: Iterable[Iterable[Segment]],
    ) -> list[float]:
        """Score segment contexts based on their relevance to the query."""
        context_strings = []
        for segment_context in segment_contexts:
            context_string = DeclarativeMemory.string_from_segment_context(
                segment_context
            )
            context_strings.append(context_string)

        segment_context_scores = await self._reranker.score(query, context_strings)
        return segment_context_scores

    @staticmethod
    def string_from_segment_context(segment_context: Iterable[Segment]) -> str:
        """Format segment context as a string."""
        context_string = ""

        for segment in segment_context:
            context_date = DeclarativeMemory._format_date(
                segment.timestamp.date(),
            )
            context_time = DeclarativeMemory._format_time(
                segment.timestamp.time(),
            )
            context_string += f"[{context_date} at {context_time}] {segment.context}: {json.dumps(segment.content)}\n"

        return context_string

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
        record_uuids = await self._segment_store.delete_episodes_segments(
            session_key=self._session_key,
            episode_uuids=episode_uuids,
        )
        await self._collection.delete(record_uuids=record_uuids)

    @staticmethod
    def _unify_scored_anchored_segment_contexts(
        scored_anchored_segment_contexts: Iterable[
            tuple[float, Segment, Iterable[Segment]]
        ],
        max_num_segments: int,
    ) -> list[tuple[float, Segment]]:
        """Unify anchored segment contexts into a single list within the limit."""
        segment_scores: dict[Segment, float] = {}

        for score, seed_segment, context in scored_anchored_segment_contexts:
            context = list(context)

            if len(segment_scores) >= max_num_segments:
                break
            if (len(segment_scores) + len(context)) <= max_num_segments:
                # It is impossible that the context exceeds the limit.
                segment_scores.update(
                    {
                        segment: score
                        for segment in context
                        if segment not in segment_scores
                    }
                )
            else:
                # It is possible that the context exceeds the limit.
                # Prioritize segments near the seed segment.

                # Sort chronological segments by weighted index-proximity to the seed segment.
                seed_index = context.index(seed_segment)

                seed_context = sorted(
                    context,
                    key=lambda segment: DeclarativeMemory._weighted_index_proximity(
                        segment=segment,
                        context=context,
                        seed_index=seed_index,
                    ),
                )

                # Add segments to unified context until limit is reached,
                # or until the context is exhausted.
                for segment in seed_context:
                    if len(segment_scores) >= max_num_segments:
                        break
                    segment_scores.setdefault(segment, score)

        unified_segment_context = sorted(
            [(score, segment) for segment, score in segment_scores.items()],
            key=lambda scored_segment: (
                scored_segment[1].timestamp,
                scored_segment[1].uuid,
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
