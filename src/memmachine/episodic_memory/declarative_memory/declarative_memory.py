"""Declarative memory system for storing and retrieving episodic memory."""

import asyncio
import datetime
import json
import logging
from collections.abc import Iterable
from uuid import UUID, uuid4

from nltk import sent_tokenize
from pydantic import BaseModel, Field, InstanceOf

from memmachine.common.embedder.embedder import Embedder
from memmachine.common.filter.filter_parser import (
    And as FilterAnd,
)
from memmachine.common.filter.filter_parser import (
    Comparison as FilterComparison,
)
from memmachine.common.filter.filter_parser import (
    FilterExpr,
)
from memmachine.common.filter.filter_parser import (
    Or as FilterOr,
)
from memmachine.common.reranker.reranker import Reranker
from memmachine.common.vector_store import Collection, Record

from .data_types import (
    ContentType,
    Derivative,
    Episode,
    Snapshot,
)
from .snapshot_store import SnapshotStore

logger = logging.getLogger(__name__)


class DeclarativeMemoryParams(BaseModel):
    """
    Parameters for DeclarativeMemory.

    Attributes:
        session_key (str):
            Session key.
        collection (Collection):
            Collection instance in a vector store.
        snapshot_store (SnapshotStore):
            Snapshot store instance for managing snapshots.
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
    snapshot_store: InstanceOf[SnapshotStore] = Field(
        ...,
        description="Snapshot store instance for managing snapshots",
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
        self._collection = params.collection
        self._snapshot_store = params.snapshot_store

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

        episodes_snapshots = await asyncio.gather(
            *[self._create_snapshots(episode) for episode in episodes]
        )

        snapshots = [
            snapshot
            for episode_snapshots in episodes_snapshots
            for snapshot in episode_snapshots
        ]

        snapshots_derivatives = await asyncio.gather(
            *[self._derive_derivatives(snapshot) for snapshot in snapshots]
        )

        derivatives = [
            derivative
            for snapshot_derivatives in snapshots_derivatives
            for derivative in snapshot_derivatives
        ]

        derivative_embeddings = await self._embedder.ingest_embed(
            [derivative.content for derivative in derivatives],
        )

        derivative_records = [
            Record(
                uuid=derivative.uuid,
                vector=derivative_embedding,
                properties={None: None},
            )
            for derivative, derivative_embedding in zip(
                derivatives,
                derivative_embeddings,
                strict=True,
            )
        ]

        await self._snapshot_store.add_snapshots(
            session_key=self._session_key,
            snapshots=snapshots,
        )

        await self._collection.upsert(derivative_records)

        await self._snapshot_store.add_snapshot_derivative_uuids(
            session_key=self._session_key,
            snapshot_derivative_uuids={
                snapshot.uuid: [derivative.uuid for derivative in snapshot_derivatives]
                for snapshot, snapshot_derivatives in zip(
                    snapshots,
                    snapshots_derivatives,
                    strict=True,
                )
            },
        )

    async def _create_snapshots(
        self,
        episode: Episode,
    ) -> list[Snapshot]:
        """
        Create snapshots from an episode.

        Args:
            episode (Episode):
                The episode from which to create snapshots.

        Returns:
            list[Snapshot]: A list of created snapshots.

        """
        return [
            Snapshot(
                uuid=uuid4(),
                episode_uuid=episode.uuid,
                index=index,
                timestamp=episode.timestamp,
                context=episode.context,
                content_type=episode.content_type,
                content=episode.content,
                filterable_properties=episode.filterable_properties,
            )
            for index in range(1)
        ]

    async def _derive_derivatives(
        self,
        snapshot: Snapshot,
    ) -> list[Derivative]:
        """
        Derive derivatives from a snapshot.

        Args:
            snapshot (Snapshot):
                The snapshot from which to derive derivatives.

        Returns:
            list[Derivative]: A list of derived derivatives.

        """
        match snapshot.content_type:
            case ContentType.MESSAGE:
                if not self._message_sentence_chunking:
                    return [
                        Derivative(
                            uuid=uuid4(),
                            snapshot_uuid=snapshot.uuid,
                            episode_uuid=snapshot.episode_uuid,
                            timestamp=snapshot.timestamp,
                            context=snapshot.context,
                            content_type=ContentType.MESSAGE,
                            content=f"{snapshot.context}: {snapshot.content}",
                            filterable_properties=snapshot.filterable_properties,
                        ),
                    ]

                sentences = {
                    sentence
                    for line in snapshot.content.strip().splitlines()
                    for sentence in sent_tokenize(line.strip())
                }

                return [
                    Derivative(
                        uuid=uuid4(),
                        snapshot_uuid=snapshot.uuid,
                        episode_uuid=snapshot.episode_uuid,
                        timestamp=snapshot.timestamp,
                        context=snapshot.context,
                        content_type=ContentType.MESSAGE,
                        content=f"{snapshot.context}: {sentence}",
                        filterable_properties=snapshot.filterable_properties,
                    )
                    for sentence in sentences
                ]
            case ContentType.TEXT:
                return [
                    Derivative(
                        uuid=uuid4(),
                        snapshot_uuid=snapshot.uuid,
                        episode_uuid=snapshot.episode_uuid,
                        timestamp=snapshot.timestamp,
                        context=snapshot.context,
                        content_type=ContentType.TEXT,
                        content=snapshot.content,
                        filterable_properties=snapshot.filterable_properties,
                    ),
                ]
            case _:
                logger.warning(
                    "Unsupported content type for derivative derivation: %s",
                    snapshot.content_type,
                )
                return []

    async def search(
        self,
        query: str,
        *,
        max_num_snapshots: int = 20,
        expand_context: int = 0,
        property_filter: FilterExpr | None = None,
    ) -> list[Snapshot]:
        """
        Search declarative memory for snapshots relevant to the query.

        Args:
            query (str):
                The search query.
            max_num_snapshots (int):
                The maximum number of snapshots to return
                (default: 20).
            expand_context (int):
                The number of additional snapshots to include
                around each matched snapshot for additional context
                (default: 0).
            property_filter (FilterExpr | None):
                Filterable property keys and values
                to use for filtering snapshots
                (default: None).

        Returns:
            list[Snapshot]:
                A list of snapshots relevant to the query, ordered chronologically.

        """
        scored_snapshots = await self.search_scored(
            query,
            max_num_snapshots=max_num_snapshots,
            expand_context=expand_context,
            property_filter=property_filter,
        )
        return [snapshot for _, snapshot in scored_snapshots]

    async def search_scored(
        self,
        query: str,
        *,
        max_num_snapshots: int = 20,
        expand_context: int = 0,
        property_filter: FilterExpr | None = None,
    ) -> list[tuple[float, Snapshot]]:
        """
        Search declarative memory for snapshots relevant to the query, returning scored snapshots.

        Args:
            query (str):
                The search query.
            max_num_snapshots (int):
                The maximum number of snapshots to return
                (default: 20).
            expand_context (int):
                The number of additional snapshots to include
                around each matched snapshot for additional context
                (default: 0).
            property_filter (FilterExpr | None):
                Filterable property keys and values
                to use for filtering snapshots
                (default: None).

        Returns:
            list[tuple[float, Snapshot]]:
                A list of scored snapshots relevant to the query, ordered chronologically.

        """
        mangled_property_filter = DeclarativeMemory._mangle_property_filter(
            property_filter,
        )

        query_embedding = (
            await self._embedder.search_embed(
                [query],
            )
        )[0]

        # Search derivative collection for matches.
        query_results = await self._collection.query(
            query_vector=query_embedding,
            limit=max(5 * max_num_snapshots, 200),
            property_filter=mangled_property_filter,
            return_vector=False,
            return_properties=True,
        )

        matched_derivative_records = [
            query_result.record for query_result in query_results
        ]

        # Get origin snapshots of matched derivatives.
        derivative_origin_snapshot_uuids = dict.fromkeys(
            UUID(matched_derivative_record.properties["_snapshot_uuid"])
            for matched_derivative_record in matched_derivative_records
        )

        seed_snapshot_uuids = list(derivative_origin_snapshot_uuids.keys())

        expand_context = min(max(0, expand_context), max_num_snapshots - 1)
        max_backward_snapshots = expand_context // 3
        max_forward_snapshots = expand_context - max_backward_snapshots

        snapshot_contexts = await self._snapshot_store.get_snapshot_contexts(
            session_key=self._session_key,
            seed_snapshot_uuids=seed_snapshot_uuids,
            max_backward_snapshots=max_backward_snapshots,
            max_forward_snapshots=max_forward_snapshots,
            property_filter=mangled_property_filter,
        )

        # Rerank snapshot contexts.
        snapshot_context_scores = await self._score_snapshot_contexts(
            query,
            snapshot_contexts,
        )

        reranked_scored_anchored_snapshot_contexts = [
            (snapshot_context_score, seed_snapshot, snapshot_context)
            for snapshot_context_score, seed_snapshot, snapshot_context in sorted(
                zip(
                    snapshot_context_scores,
                    seed_snapshot_uuids,
                    snapshot_contexts,
                    strict=True,
                ),
                key=lambda triple: triple[0],
                reverse=True,
            )
        ]

        # Unify episode contexts.
        unified_scored_snapshot_context = (
            DeclarativeMemory._unify_scored_anchored_snapshot_contexts(
                reranked_scored_anchored_snapshot_contexts,
                max_num_snapshots=max_num_snapshots,
            )
        )
        return unified_scored_snapshot_context

    async def _score_snapshot_contexts(
        self,
        query: str,
        snapshot_contexts: Iterable[Iterable[Snapshot]],
    ) -> list[float]:
        """Score snapshot contexts based on their relevance to the query."""
        context_strings = []
        for snapshot_context in snapshot_contexts:
            context_string = DeclarativeMemory.string_from_snapshot_context(
                snapshot_context
            )
            context_strings.append(context_string)

        snapshot_context_scores = await self._reranker.score(query, context_strings)
        return snapshot_context_scores

    @staticmethod
    def string_from_snapshot_context(snapshot_context: Iterable[Snapshot]) -> str:
        """Format snapshot context as a string."""
        context_string = ""

        for snapshot in snapshot_context:
            context_date = DeclarativeMemory._format_date(
                snapshot.timestamp.date(),
            )
            context_time = DeclarativeMemory._format_time(
                snapshot.timestamp.time(),
            )
            context_string += f"[{context_date} at {context_time}] {snapshot.source}: {json.dumps(snapshot.content)}\n"

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
        uuids = list(episode_uuids)

        episode_snapshots_uuids = [
            snapshot.uuid
            for snapshot in await self._snapshot_store.get_episodes_snapshots(
                session_key=self._session_key,
                episode_uuids=uuids,
            )
        ]
        episode_derivative_uuids = [
            derivative_uuid
            for derivative_uuid in await self._snapshot_store.get_snapshot_derivative_uuids(
                session_key=self._session_key,
                snapshot_uuids=episode_snapshots_uuids,
            )
        ]

        deletion_tasks = [
            self._collection.delete(
                episode_derivative_uuids,
            ),
            self._snapshot_store.delete_snapshots(
                session_key=self._session_key,
                snapshot_uuids=episode_snapshots_uuids,
            ),
        ]

        await asyncio.gather(*deletion_tasks)

    @staticmethod
    def _unify_scored_anchored_snapshot_contexts(
        scored_anchored_snapshot_contexts: Iterable[
            tuple[float, Snapshot, Iterable[Snapshot]]
        ],
        max_num_snapshots: int,
    ) -> list[tuple[float, Snapshot]]:
        """Unify anchored episode contexts into a single list within the limit."""
        snapshot_scores: dict[Snapshot, float] = {}

        for score, seed_snapshot, context in scored_anchored_snapshot_contexts:
            context = list(context)

            if len(snapshot_scores) >= max_num_snapshots:
                break
            if (len(snapshot_scores) + len(context)) <= max_num_snapshots:
                # It is impossible that the context exceeds the limit.
                snapshot_scores.update(
                    {
                        snapshot: score
                        for snapshot in context
                        if snapshot not in snapshot_scores
                    }
                )
            else:
                # It is possible that the context exceeds the limit.
                # Prioritize snapshots near the seed snapshot.

                # Sort chronological episodes by weighted index-proximity to the seed snapshot.
                seed_index = context.index(seed_snapshot)

                seed_context = sorted(
                    context,
                    key=lambda snapshot: DeclarativeMemory._weighted_index_proximity(
                        snapshot=snapshot,
                        context=context,
                        seed_index=seed_index,
                    ),
                )

                # Add snapshots to unified context until limit is reached,
                # or until the context is exhausted.
                for snapshot in seed_context:
                    if len(snapshot_scores) >= max_num_snapshots:
                        break
                    snapshot_scores.setdefault(snapshot, score)

        unified_snapshot_context = sorted(
            [(score, snapshot) for snapshot, score in snapshot_scores.items()],
            key=lambda scored_snapshot: (
                scored_snapshot[1].timestamp,
                scored_snapshot[1].uid,
            ),
        )

        return unified_snapshot_context

    @staticmethod
    def _weighted_index_proximity(
        snapshot: Snapshot,
        context: list[Snapshot],
        nuclear_index: int,
    ) -> float:
        proximity = context.index(snapshot) - nuclear_index
        if proximity >= 0:
            # Forward recall is better than backward recall.
            return (proximity - 0.5) / 2
        return -proximity

    @staticmethod
    def _mangle_property_filter(
        property_filter: FilterExpr | None,
    ) -> FilterExpr | None:
        if property_filter is None:
            return None

        return DeclarativeMemory._mangle_filter_expr(property_filter)

    @staticmethod
    def _mangle_filter_expr(expr: FilterExpr | None) -> FilterExpr | None:
        if expr is None:
            return None

        if isinstance(expr, FilterComparison):
            return FilterComparison(
                field=mangle_filterable_property_key(expr.field),
                op=expr.op,
                value=expr.value,
            )
        if isinstance(expr, FilterAnd):
            return FilterAnd(
                left=DeclarativeMemory._mangle_filter_expr(expr.left),
                right=DeclarativeMemory._mangle_filter_expr(expr.right),
            )
        if isinstance(expr, FilterOr):
            return FilterOr(
                left=DeclarativeMemory._mangle_filter_expr(expr.left),
                right=DeclarativeMemory._mangle_filter_expr(expr.right),
            )
        raise TypeError(f"Unsupported filter expression type: {type(expr)!r}")
