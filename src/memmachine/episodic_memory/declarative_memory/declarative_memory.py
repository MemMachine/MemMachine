"""Declarative memory system for storing and retrieving episodic memory."""

import asyncio
import datetime
import json
import logging
from collections.abc import Iterable
from typing import ClassVar, cast
from uuid import uuid4

from pydantic import BaseModel, Field, InstanceOf

from memmachine.common.data_types import PropertyValue
from memmachine.common.embedder.embedder import Embedder
from memmachine.common.filter.filter_parser import (
    FilterExpr,
    map_filter_fields,
)
from memmachine.common.reranker.reranker import Reranker
from memmachine.common.utils import extract_sentences
from memmachine.common.vector_graph_store import Edge, Node, VectorGraphStore
from memmachine.common.vector_graph_store.data_types import MultiHopResult
from memmachine.common.vector_graph_store.graph_traversal_store import (
    GraphTraversalStore,
)

from .data_types import (
    ContentType,
    Derivative,
    Episode,
    demangle_property_key,
    is_mangled_property_key,
    mangle_property_key,
)

logger = logging.getLogger(__name__)


class DeclarativeMemoryParams(BaseModel):
    """
    Parameters for DeclarativeMemory.

    Attributes:
        session_id (str):
            Session identifier.
        vector_graph_store (VectorGraphStore):
            VectorGraphStore instance
            for storing and retrieving memories.
        embedder (Embedder):
            Embedder instance for creating embeddings.
        reranker (Reranker):
            Reranker instance for reranking search results.

    """

    session_id: str = Field(
        ...,
        description="Session identifier",
    )
    vector_graph_store: InstanceOf[VectorGraphStore] = Field(
        ...,
        description="VectorGraphStore instance for storing and retrieving memories",
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
    pagerank_blend_alpha: float = Field(
        0.8,
        ge=0.0,
        le=1.0,
        description=(
            "Blending factor for PageRank re-ranking. "
            "final_score = alpha * similarity + (1 - alpha) * normalized_pagerank. "
            "Higher values favour vector similarity; lower values favour graph centrality."
        ),
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
        session_id = params.session_id

        self._vector_graph_store = params.vector_graph_store
        self._embedder = params.embedder
        self._reranker = params.reranker

        self._message_sentence_chunking = params.message_sentence_chunking
        self._pagerank_blend_alpha = params.pagerank_blend_alpha

        self._episode_collection = f"Episode_{session_id}"
        self._derivative_collection = f"Derivative_{session_id}"
        self._producer_collection = "Producer"

        self._derived_from_relation = f"DERIVED_FROM_{session_id}"
        self._produced_by_relation = "PRODUCED_BY"

    async def add_episodes(
        self,
        episodes: Iterable[Episode],
    ) -> None:
        """
        Add episodes.

        Episodes are sorted by timestamp.
        Episodes with the same timestamp are sorted by UID.

        Args:
            episodes (Iterable[Episode]): The episodes to add.

        """
        episodes = sorted(
            episodes,
            key=lambda episode: (episode.timestamp, episode.uid),
        )
        episode_nodes = [
            Node(
                uid=episode.uid,
                properties={
                    "uid": str(episode.uid),
                    "timestamp": episode.timestamp,
                    "source": episode.source,
                    "content_type": episode.content_type.value,
                    "content": episode.content,
                    "user_metadata": json.dumps(episode.user_metadata),
                }
                | {
                    mangle_property_key(key): value
                    for key, value in episode.filterable_properties.items()
                },
            )
            for episode in episodes
        ]

        derive_derivatives_tasks = [
            self._derive_derivatives(episode) for episode in episodes
        ]

        episodes_derivatives = await asyncio.gather(*derive_derivatives_tasks)

        derivatives = [
            derivative
            for episode_derivatives in episodes_derivatives
            for derivative in episode_derivatives
        ]

        derivative_embeddings = await self._embedder.ingest_embed(
            [derivative.content for derivative in derivatives],
        )

        # Build a lookup of entity types from episode user_metadata so that
        # derivative nodes inherit them. Per design D2, no LLM call is made
        # in the episodic path — the entity type is taken directly from
        # user_metadata when present.
        episode_entity_types: dict[str, list[str]] = {}
        for episode in episodes:
            if isinstance(episode.user_metadata, dict):
                et = episode.user_metadata.get("entity_type")
                if isinstance(et, str) and et:
                    episode_entity_types[episode.uid] = [et]

        # Map each derivative UID to its source episode UID so we can look
        # up the entity types later.
        derivative_to_episode: dict[str, str] = {}
        for episode, episode_derivatives in zip(
            episodes, episodes_derivatives, strict=True
        ):
            for derivative in episode_derivatives:
                derivative_to_episode[derivative.uid] = episode.uid

        derivative_nodes = [
            Node(
                uid=derivative.uid,
                properties={
                    "uid": derivative.uid,
                    "timestamp": derivative.timestamp,
                    "source": derivative.source,
                    "content_type": derivative.content_type.value,
                    "content": derivative.content,
                }
                | {
                    mangle_property_key(key): value
                    for key, value in derivative.filterable_properties.items()
                },
                embeddings={
                    DeclarativeMemory._embedding_name(
                        self._embedder.model_id,
                        self._embedder.dimensions,
                    ): (embedding, self._embedder.similarity_metric),
                },
                entity_types=episode_entity_types.get(
                    derivative_to_episode.get(derivative.uid, ""), []
                ),
            )
            for derivative, embedding in zip(
                derivatives,
                derivative_embeddings,
                strict=True,
            )
        ]

        derivative_episode_edges = [
            Edge(
                uid=str(uuid4()),
                source_uid=derivative.uid,
                target_uid=episode.uid,
            )
            for episode, episode_derivatives in zip(
                episodes,
                episodes_derivatives,
                strict=True,
            )
            for derivative in episode_derivatives
        ]

        add_nodes_tasks = [
            self._vector_graph_store.add_nodes(
                collection=self._episode_collection,
                nodes=episode_nodes,
            ),
            self._vector_graph_store.add_nodes(
                collection=self._derivative_collection,
                nodes=derivative_nodes,
            ),
        ]
        await asyncio.gather(*add_nodes_tasks)

        await self._vector_graph_store.add_edges(
            relation=self._derived_from_relation,
            source_collection=self._derivative_collection,
            target_collection=self._episode_collection,
            edges=derivative_episode_edges,
        )

        # Create PRODUCED_BY relationships: Episode -> Producer
        # Collect unique producer_ids and create Producer anchor nodes
        producer_ids = {episode.source for episode in episodes if episode.source}
        if producer_ids:
            producer_nodes = [
                Node(
                    uid=pid,
                    properties={"producer_id": pid},
                )
                for pid in producer_ids
            ]
            await self._vector_graph_store.add_nodes(
                collection=self._producer_collection,
                nodes=producer_nodes,
            )

            produced_by_edges = [
                Edge(
                    uid=str(uuid4()),
                    source_uid=episode.uid,
                    target_uid=episode.source,
                )
                for episode in episodes
                if episode.source
            ]
            await self._vector_graph_store.add_edges(
                relation=self._produced_by_relation,
                source_collection=self._episode_collection,
                target_collection=self._producer_collection,
                edges=produced_by_edges,
            )

    async def _derive_derivatives(
        self,
        episode: Episode,
    ) -> list[Derivative]:
        """
        Derive derivatives from an episode.

        Args:
            episode (Episode):
                The episode from which to derive derivatives.

        Returns:
            list[Derivative]: A list of derived derivatives.

        """
        match episode.content_type:
            case ContentType.MESSAGE:
                if not self._message_sentence_chunking:
                    return [
                        Derivative(
                            uid=str(uuid4()),
                            timestamp=episode.timestamp,
                            source=episode.source,
                            content_type=ContentType.MESSAGE,
                            content=f"{episode.source}: {episode.content}",
                            filterable_properties=episode.filterable_properties,
                        ),
                    ]

                sentences = extract_sentences(episode.content)

                return [
                    Derivative(
                        uid=str(uuid4()),
                        timestamp=episode.timestamp,
                        source=episode.source,
                        content_type=ContentType.MESSAGE,
                        content=f"{episode.source}: {sentence}",
                        filterable_properties=episode.filterable_properties,
                    )
                    for sentence in sentences
                ]
            case ContentType.TEXT:
                text_content = episode.content
                return [
                    Derivative(
                        uid=str(uuid4()),
                        timestamp=episode.timestamp,
                        source=episode.source,
                        content_type=ContentType.TEXT,
                        content=text_content,
                        filterable_properties=episode.filterable_properties,
                    ),
                ]
            case _:
                logger.warning(
                    "Unsupported content type for derivative derivation: %s",
                    episode.content_type,
                )
                return []

    async def search(
        self,
        query: str,
        *,
        max_num_episodes: int = 20,
        expand_context: int = 0,
        property_filter: FilterExpr | None = None,
        entity_types: list[str] | None = None,
    ) -> list[Episode]:
        """
        Search declarative memory for episodes relevant to the query.

        Args:
            query (str):
                The search query.
            max_num_episodes (int):
                The maximum number of episodes to return
                (default: 20).
            expand_context (int):
                The number of additional episodes to include
                around each matched episode for additional context
                (default: 0).
            property_filter (FilterExpr | None):
                Filterable property keys and values
                to use for filtering episodes
                (default: None).
            entity_types (list[str] | None):
                Optional entity type filter. When provided, only
                derivative nodes whose ``entity_types`` contain at
                least one of the specified types are included in
                search results (default: None — no filtering).

        Returns:
            list[Episode]:
                A list of episodes relevant to the query, ordered chronologically.

        """
        scored_episodes = await self.search_scored(
            query,
            max_num_episodes=max_num_episodes,
            expand_context=expand_context,
            property_filter=property_filter,
            entity_types=entity_types,
        )
        return [episode for _, episode in scored_episodes]

    async def search_scored(
        self,
        query: str,
        *,
        max_num_episodes: int = 20,
        expand_context: int = 0,
        property_filter: FilterExpr | None = None,
        entity_types: list[str] | None = None,
    ) -> list[tuple[float, Episode]]:
        """
        Search declarative memory for episodes relevant to the query, returning scored episodes.

        Args:
            query (str):
                The search query.
            max_num_episodes (int):
                The maximum number of episodes to return
                (default: 20).
            expand_context (int):
                The number of additional episodes to include
                around each matched episode for additional context
                (default: 0).
            property_filter (FilterExpr | None):
                Filterable property keys and values
                to use for filtering episodes
                (default: None).
            entity_types (list[str] | None):
                Optional entity type filter. When provided, only
                derivative nodes whose ``entity_types`` contain at
                least one of the specified types are included in
                search results (default: None — no filtering).

        Returns:
            list[tuple[float, Episode]]:
                A list of scored episodes relevant to the query, ordered by score (highest first).

        """
        mangled_property_filter = DeclarativeMemory._mangle_property_filter(
            property_filter,
        )

        query_embedding = (
            await self._embedder.search_embed(
                [query],
            )
        )[0]

        # Search graph store for vector matches.
        matched_derivative_nodes = await self._vector_graph_store.search_similar_nodes(
            collection=self._derivative_collection,
            embedding_name=(
                DeclarativeMemory._embedding_name(
                    self._embedder.model_id,
                    self._embedder.dimensions,
                )
            ),
            query_embedding=query_embedding,
            similarity_metric=self._embedder.similarity_metric,
            limit=min(5 * max_num_episodes, 200),
            property_filter=mangled_property_filter,
        )

        # Filter by entity types when requested. A derivative node matches
        # if its entity_types list shares at least one type with the filter.
        if entity_types:
            entity_type_set = set(entity_types)
            matched_derivative_nodes = [
                node
                for node in matched_derivative_nodes
                if entity_type_set.intersection(node.entity_types)
            ]

        # --- Graph-enhanced retrieval (multi-hop + PageRank) ---
        (
            matched_derivative_nodes,
            graph_discovery_scores,
        ) = await self._graph_enhance_results(
            matched_derivative_nodes,
            mangled_property_filter=mangled_property_filter,
        )

        # Get source episodes of matched derivatives.
        # Track which episode UIDs came from graph-discovered derivatives
        # so we can boost their reranker scores later.
        search_derivatives_source_episode_nodes_tasks = [
            self._vector_graph_store.search_related_nodes(
                relation=self._derived_from_relation,
                other_collection=self._episode_collection,
                this_collection=self._derivative_collection,
                this_node_uid=matched_derivative_node.uid,
                find_sources=False,
                find_targets=True,
                node_property_filter=mangled_property_filter,
            )
            for matched_derivative_node in matched_derivative_nodes
        ]

        # Build a derivative_uid → graph_score lookup, and map episodes
        # back to their derivative's graph score.
        graph_discovered_derivative_uids = set(graph_discovery_scores.keys())

        all_episode_node_lists = await asyncio.gather(
            *search_derivatives_source_episode_nodes_tasks,
        )

        # Map episode_uid → best graph_discovery_score from its derivative.
        episode_graph_boost: dict[str, float] = {}
        for deriv_node, ep_nodes in zip(
            matched_derivative_nodes,
            all_episode_node_lists,
            strict=True,
        ):
            if deriv_node.uid in graph_discovered_derivative_uids:
                gscore = graph_discovery_scores[deriv_node.uid]
                for ep_node in ep_nodes:
                    cur = episode_graph_boost.get(ep_node.uid, 0.0)
                    episode_graph_boost[ep_node.uid] = max(cur, gscore)

        if episode_graph_boost:
            logger.info(
                "Episode graph boost mapping: %s",
                {uid: f"{s:.3f}" for uid, s in episode_graph_boost.items()},
            )

        # Use a dict instead of a set to preserve order.
        source_episode_nodes = dict.fromkeys(
            episode_node
            for episode_nodes in all_episode_node_lists
            for episode_node in episode_nodes
        )

        # Use source episodes as nuclei for contextualization.
        nuclear_episodes = [
            DeclarativeMemory._episode_from_episode_node(source_episode_node)
            for source_episode_node in source_episode_nodes
        ]

        expand_context = min(max(0, expand_context), max_num_episodes - 1)
        max_backward_episodes = expand_context // 3
        max_forward_episodes = expand_context - max_backward_episodes

        contextualize_episode_tasks = [
            self._contextualize_episode(
                nuclear_episode,
                max_backward_episodes=max_backward_episodes,
                max_forward_episodes=max_forward_episodes,
                mangled_property_filter=mangled_property_filter,
            )
            for nuclear_episode in nuclear_episodes
        ]

        episode_contexts = await asyncio.gather(*contextualize_episode_tasks)

        # Rerank episode contexts.
        episode_context_scores = await self._score_episode_contexts(
            query,
            episode_contexts,
        )

        # Boost reranker scores for graph-connected episodes.
        # The graph confirms these episodes are relevant through structural
        # relationships (e.g. Bob → TensorFlow → Atlas) even when the
        # text-based reranker scores them low.
        #
        # Strategy: scale graph-connected episodes relative to the TOP
        # reranker score, weighted by path quality:
        #
        #   boosted = max(score, top_score * graph_score)
        #
        # At 5 hops with decay 0.85 and path quality 1.0 (exact feature
        # match) graph_score ≈ 0.44, so Bob lands at ~44% of the best
        # result — solidly in the upper half regardless of top_k.
        #
        # Anchoring to top_score (position 0) rather than the bottom-of-
        # window inclusion threshold makes the boost top_k-independent:
        # the threshold score shrinks as top_k grows, which was causing
        # Bob to vanish at larger top_k values.
        if episode_graph_boost:
            sorted_scores = sorted(episode_context_scores, reverse=True)
            top_score = sorted_scores[0] if sorted_scores else 0.0
            logger.info(
                "Graph boost: top_score=%.4f (%d total scores)",
                top_score,
                len(sorted_scores),
            )

            boosted_scores: list[float] = []
            for score, nuclear_ep in zip(
                episode_context_scores,
                nuclear_episodes,
                strict=True,
            ):
                graph_boost = episode_graph_boost.get(nuclear_ep.uid, 0.0)
                if graph_boost > 0.0:
                    # Place at graph_score fraction of the best result.
                    boosted = max(score, top_score * graph_boost)
                    logger.info(
                        "Boosting episode %s: reranker=%.4f → boosted=%.4f "
                        "(graph_score=%.3f, top_score=%.4f)",
                        nuclear_ep.uid,
                        score,
                        boosted,
                        graph_boost,
                        top_score,
                    )
                    boosted_scores.append(boosted)
                else:
                    boosted_scores.append(score)
            episode_context_scores = boosted_scores

        reranked_scored_anchored_episode_contexts = [
            (episode_context_score, nuclear_episode, episode_context)
            for episode_context_score, nuclear_episode, episode_context in sorted(
                zip(
                    episode_context_scores,
                    nuclear_episodes,
                    episode_contexts,
                    strict=True,
                ),
                key=lambda triple: triple[0],
                reverse=True,
            )
        ]

        # Unify episode contexts.
        unified_scored_episode_context = (
            DeclarativeMemory._unify_scored_anchored_episode_contexts(
                reranked_scored_anchored_episode_contexts,
                max_num_episodes=max_num_episodes,
            )
        )
        return unified_scored_episode_context

    async def _graph_enhance_results(
        self,
        matched_nodes: list[Node],
        *,
        mangled_property_filter: FilterExpr | None,
    ) -> tuple[list[Node], dict[str, float]]:
        """Expand search results via multi-hop traversal and re-rank.

        When the underlying store implements :class:`GraphTraversalStore`,
        the top-k initial results are used as anchors for a 4-hop traversal
        that bridges the episodic and semantic layers through Feature nodes::

            Derivative → Episode ← Feature → Episode ← Derivative

        New nodes discovered via traversal are merged into the result set
        (deduplicated by UID, keeping the higher score).  Their graph
        traversal scores are preserved and used in the reranking step so
        that graph-discovered nodes receive a meaningful similarity estimate
        instead of zero.

        If PageRank scores are present on nodes (written by a prior
        ``compute_pagerank()`` call), the final scores are blended:
        ``final_score = alpha * similarity + (1 - alpha) * norm_pagerank``.

        When the store does **not** implement the protocol, the input list
        is returned unchanged (graceful fallback).

        Returns:
            A tuple of ``(nodes, graph_discovery_scores)`` where
            *graph_discovery_scores* maps derivative UIDs that were found
            via multi-hop traversal (not in the initial vector results)
            to their best graph score.  This dict is empty when no graph
            discoveries were made.  Callers use it to boost the reranker
            scores of graph-discovered episodes.
        """
        if not isinstance(self._vector_graph_store, GraphTraversalStore):
            logger.info(
                "Graph enhancement skipped: store does not implement GraphTraversalStore"
            )
            return matched_nodes, {}

        if not matched_nodes:
            return matched_nodes, {}

        logger.info(
            "Graph enhancement: expanding %d anchors via multi-hop traversal",
            min(10, len(matched_nodes)),
        )
        enhanced, graph_scores = await self._expand_multi_hop(
            matched_nodes,
            store=self._vector_graph_store,
            mangled_property_filter=mangled_property_filter,
        )
        if graph_scores:
            logger.info(
                "Graph discoveries: %d nodes with graph scores (scores: %s)",
                len(graph_scores),
                {uid[:12]: f"{s:.3f}" for uid, s in graph_scores.items()},
            )
        else:
            logger.info("Graph enhancement: no graph-connected nodes found")

        reranked = self._rerank_with_graph_signals(
            enhanced,
            matched_nodes,
            graph_scores,
        )
        return reranked, graph_scores

    # Relationship types used by the episodic layer (will be sanitized by
    # the store implementation).
    _EPISODIC_RELATION_TYPES: ClassVar[list[str]] = [
        "DERIVED_FROM_universal/universal",
    ]

    # Relationship types used by the semantic layer that are already stored
    # with their plain Neo4j names (NOT sanitized by the vector-graph store).
    _SEMANTIC_RAW_RELATION_TYPES: ClassVar[list[str]] = [
        "EXTRACTED_FROM",
        "RELATED_TO",
    ]

    async def _expand_multi_hop(
        self,
        matched_nodes: list[Node],
        *,
        store: GraphTraversalStore,
        mangled_property_filter: FilterExpr | None,
    ) -> tuple[list[Node], dict[str, float]]:
        """Use top-k results as anchors for multi-hop traversal.

        The traversal path that bridges episodic and semantic layers is::

            Derivative -(DERIVED_FROM)-> Episode <-(EXTRACTED_FROM)- Feature₁
                       -(RELATED_TO)-> Feature₂ -(EXTRACTED_FROM)-> Episode
                       <-(DERIVED_FROM)- Derivative

        Each Feature connects to exactly one Episode, so the bridge
        between episodes requires a ``RELATED_TO`` hop between Features.
        This makes the full path **5 hops**.

        We intentionally exclude hub relationships (``PRODUCED_BY``,
        ``BELONGS_TO``, ``HAS_HISTORY``, ``REFERENCES_EPISODE``) that
        would connect everything through producer/set nodes and destroy
        selectivity.

        Returns:
            A tuple of ``(nodes, graph_scores)`` where *graph_scores* maps
            discovered UIDs to their best multi-hop graph score
            (``score_decay ** hop_distance``).  Nodes already in the initial
            results do not appear in *graph_scores*.
        """
        anchor_limit = min(10, len(matched_nodes))
        anchors = matched_nodes[:anchor_limit]

        logger.debug(
            "Multi-hop anchors (%d): %s",
            len(anchors),
            [a.uid[:12] for a in anchors],
        )

        multi_hop_tasks = [
            store.search_multi_hop_nodes(
                collection=self._derivative_collection,
                this_node_uid=anchor.uid,
                max_hops=5,
                relation_types=self._EPISODIC_RELATION_TYPES,
                raw_relation_types=self._SEMANTIC_RAW_RELATION_TYPES,
                score_decay=0.85,
                node_property_filter=mangled_property_filter,
                target_collections=[self._derivative_collection],
            )
            for anchor in anchors
        ]

        multi_hop_results_per_anchor: list[list[MultiHopResult]] = await asyncio.gather(
            *multi_hop_tasks
        )

        # Build a UID → Node map from the initial results (preserve order).
        uid_to_node: dict[str, Node] = {n.uid: n for n in matched_nodes}

        # UIDs of the anchor nodes used to start traversal — we don't want
        # to boost anchors themselves (they already ranked high).
        anchor_uids = {a.uid for a in matched_nodes[:anchor_limit]}

        # Track graph scores for nodes reachable via traversal.  This
        # includes nodes already in the initial vector results — they
        # should still receive a graph boost because the graph confirms
        # their relevance through structural connections.  Anchor nodes
        # are excluded because they're the starting points.
        graph_scores: dict[str, float] = {}

        for hop_results in multi_hop_results_per_anchor:
            for hop_result in hop_results:
                uid = hop_result.node.uid
                if uid in anchor_uids:
                    # Don't boost the anchor itself.
                    continue
                if uid not in graph_scores or hop_result.score > graph_scores[uid]:
                    graph_scores[uid] = hop_result.score
                if uid not in uid_to_node:
                    uid_to_node[uid] = hop_result.node

        return list(uid_to_node.values()), graph_scores

    def _rerank_with_graph_signals(
        self,
        enhanced_nodes: list[Node],
        initial_nodes: list[Node],
        graph_scores: dict[str, float],
    ) -> list[Node]:
        """Blend similarity, graph traversal scores, and PageRank.

        Scoring formula for each node::

            final = alpha * similarity + (1 - alpha) * norm_pagerank

        Where *similarity* is:

        - For initial (vector-matched) nodes: position-based score in
          ``[0, 1]`` derived from their rank in the initial results.
        - For graph-discovered nodes: the multi-hop graph score
          (``score_decay ** hops``), scaled to be competitive with
          initial results.  The lowest initial similarity is used as
          the ceiling so graph discoveries slot in naturally below
          confirmed vector matches but above zero.

        This ensures graph-discovered nodes participate in the ranking
        instead of being silently buried with score zero.
        """
        alpha = self._pagerank_blend_alpha

        # Position-based similarity scores for the initial result set.
        sim_scores: dict[str, float] = {}
        n_initial = len(initial_nodes)
        for idx, node in enumerate(initial_nodes):
            sim_scores[node.uid] = 1.0 - (idx / max(n_initial, 1))

        # Scale graph scores so the best graph discovery gets a similarity
        # score just below the worst initial result.  This keeps graph
        # discoveries competitive without outranking confirmed vector matches.
        if graph_scores and sim_scores:
            min_initial_sim = min(sim_scores.values())
            max_graph = max(graph_scores.values()) if graph_scores else 1.0
            if max_graph > 0:
                for uid, gscore in graph_scores.items():
                    sim_scores[uid] = min_initial_sim * (gscore / max_graph)

        # Check for PageRank values and blend if available.
        pagerank_values: list[float | None] = [
            n.properties.get("pagerank_score")  # type: ignore[assignment]
            for n in enhanced_nodes
        ]
        has_pagerank = any(v is not None for v in pagerank_values)

        if not has_pagerank and not graph_scores:
            # No graph signals at all — return as-is.
            return enhanced_nodes

        # Normalise PageRank to [0, 1].
        raw_pr = [float(v) if v is not None else 0.0 for v in pagerank_values]
        max_pr = max(raw_pr) if raw_pr else 0.0
        norm_pr = [v / max_pr if max_pr > 0 else 0.0 for v in raw_pr]

        scored_nodes: list[tuple[float, Node]] = []
        for node, npr in zip(enhanced_nodes, norm_pr, strict=True):
            sim = sim_scores.get(node.uid, 0.0)
            # When PageRank is available, blend similarity and PageRank.
            # Otherwise use pure similarity (which now includes graph
            # scores for discovered nodes).
            final = alpha * sim + (1.0 - alpha) * npr if has_pagerank else sim
            scored_nodes.append((final, node))

        scored_nodes.sort(key=lambda t: t[0], reverse=True)
        return [node for _, node in scored_nodes]

    async def _contextualize_episode(
        self,
        nuclear_episode: Episode,
        max_backward_episodes: int = 0,
        max_forward_episodes: int = 0,
        mangled_property_filter: FilterExpr | None = None,
    ) -> list[Episode]:
        previous_episode_nodes = []
        next_episode_nodes = []

        if max_backward_episodes > 0:
            previous_episode_nodes = (
                await self._vector_graph_store.search_directional_nodes(
                    collection=self._episode_collection,
                    by_properties=("timestamp", "uid"),
                    starting_at=(
                        nuclear_episode.timestamp,
                        str(nuclear_episode.uid),
                    ),
                    order_ascending=(False, False),
                    include_equal_start=False,
                    limit=max_backward_episodes,
                    property_filter=mangled_property_filter,
                )
            )

        if max_forward_episodes > 0:
            next_episode_nodes = (
                await self._vector_graph_store.search_directional_nodes(
                    collection=self._episode_collection,
                    by_properties=("timestamp", "uid"),
                    starting_at=(
                        nuclear_episode.timestamp,
                        str(nuclear_episode.uid),
                    ),
                    order_ascending=(True, True),
                    include_equal_start=False,
                    limit=max_forward_episodes,
                    property_filter=mangled_property_filter,
                )
            )

        context = (
            [
                DeclarativeMemory._episode_from_episode_node(episode_node)
                for episode_node in reversed(previous_episode_nodes)
            ]
            + [nuclear_episode]
            + [
                DeclarativeMemory._episode_from_episode_node(episode_node)
                for episode_node in next_episode_nodes
            ]
        )

        return context

    async def _score_episode_contexts(
        self,
        query: str,
        episode_contexts: Iterable[Iterable[Episode]],
    ) -> list[float]:
        """Score episode contexts based on their relevance to the query."""
        context_strings = []
        for episode_context in episode_contexts:
            context_string = DeclarativeMemory.string_from_episode_context(
                episode_context
            )
            context_strings.append(context_string)

        episode_context_scores = await self._reranker.score(query, context_strings)

        return episode_context_scores

    @staticmethod
    def string_from_episode_context(episode_context: Iterable[Episode]) -> str:
        """Format episode context as a string."""
        context_string = ""

        for episode in episode_context:
            context_date = DeclarativeMemory._format_date(
                episode.timestamp.date(),
            )
            context_time = DeclarativeMemory._format_time(
                episode.timestamp.time(),
            )
            context_string += f"[{context_date} at {context_time}] {episode.source}: {json.dumps(episode.content)}\n"

        return context_string

    @staticmethod
    def _format_date(date: datetime.date) -> str:
        """Format the date as a string."""
        return date.strftime("%A, %B %d, %Y")

    @staticmethod
    def _format_time(time: datetime.time) -> str:
        """Format the time as a string."""
        return time.strftime("%I:%M %p")

    async def get_episodes(self, uids: Iterable[str]) -> list[Episode]:
        """Get episodes by their UIDs."""
        episode_nodes = await self._vector_graph_store.get_nodes(
            collection=self._episode_collection,
            node_uids=uids,
        )

        episodes = [
            DeclarativeMemory._episode_from_episode_node(episode_node)
            for episode_node in episode_nodes
        ]

        return episodes

    async def get_matching_episodes(
        self,
        property_filter: FilterExpr | None = None,
    ) -> list[Episode]:
        """Filter episodes by their properties."""
        mangled_property_filter = DeclarativeMemory._mangle_property_filter(
            property_filter,
        )

        matching_episode_nodes = await self._vector_graph_store.search_matching_nodes(
            collection=self._episode_collection,
            property_filter=mangled_property_filter,
        )

        matching_episodes = [
            DeclarativeMemory._episode_from_episode_node(matching_episode_node)
            for matching_episode_node in matching_episode_nodes
        ]

        return matching_episodes

    async def delete_episodes(self, uids: Iterable[str]) -> None:
        """Delete episodes by their UIDs."""
        uids = list(uids)

        search_derived_derivative_nodes_tasks = [
            self._vector_graph_store.search_related_nodes(
                relation=self._derived_from_relation,
                other_collection=self._derivative_collection,
                this_collection=self._episode_collection,
                this_node_uid=episode_uid,
                find_sources=True,
                find_targets=False,
            )
            for episode_uid in uids
        ]

        derived_derivative_nodes = [
            derivative_node
            for derivative_nodes in await asyncio.gather(
                *search_derived_derivative_nodes_tasks,
            )
            for derivative_node in derivative_nodes
        ]

        delete_nodes_tasks = [
            self._vector_graph_store.delete_nodes(
                collection=self._episode_collection,
                node_uids=uids,
            ),
            self._vector_graph_store.delete_nodes(
                collection=self._derivative_collection,
                node_uids=[
                    derivative_node.uid for derivative_node in derived_derivative_nodes
                ],
            ),
        ]

        await asyncio.gather(*delete_nodes_tasks)

    @staticmethod
    def _unify_scored_anchored_episode_contexts(
        scored_anchored_episode_contexts: Iterable[
            tuple[float, Episode, Iterable[Episode]]
        ],
        max_num_episodes: int,
    ) -> list[tuple[float, Episode]]:
        """Unify anchored episode contexts into a single list within the limit."""
        episode_scores: dict[Episode, float] = {}

        for score, nuclear_episode, context in scored_anchored_episode_contexts:
            context = list(context)

            if len(episode_scores) >= max_num_episodes:
                break
            if (len(episode_scores) + len(context)) <= max_num_episodes:
                # It is impossible that the context exceeds the limit.
                episode_scores.update(
                    {
                        episode: score
                        for episode in context
                        if episode not in episode_scores
                    }
                )
            else:
                # It is possible that the context exceeds the limit.
                # Prioritize episodes near the nuclear episode.

                # Sort chronological episodes by weighted index-proximity to the nuclear episode.
                nuclear_index = context.index(nuclear_episode)

                nuclear_context = sorted(
                    context,
                    key=lambda episode: DeclarativeMemory._weighted_index_proximity(
                        episode=episode,
                        context=context,
                        nuclear_index=nuclear_index,
                    ),
                )

                # Add episodes to unified context until limit is reached,
                # or until the context is exhausted.
                for episode in nuclear_context:
                    if len(episode_scores) >= max_num_episodes:
                        break
                    episode_scores.setdefault(episode, score)

        unified_episode_context = sorted(
            [(score, episode) for episode, score in episode_scores.items()],
            key=lambda scored_episode: scored_episode[0],
            reverse=True,
        )

        return unified_episode_context

    @staticmethod
    def _weighted_index_proximity(
        episode: Episode,
        context: list[Episode],
        nuclear_index: int,
    ) -> float:
        proximity = context.index(episode) - nuclear_index
        if proximity >= 0:
            # Forward recall is better than backward recall.
            return (proximity - 0.5) / 2
        return -proximity

    @staticmethod
    def _episode_from_episode_node(episode_node: Node) -> Episode:
        return Episode(
            uid=cast("str", episode_node.properties["uid"]),
            timestamp=cast("datetime.datetime", episode_node.properties["timestamp"]),
            source=cast("str", episode_node.properties["source"]),
            content_type=ContentType(episode_node.properties["content_type"]),
            content=episode_node.properties["content"],
            filterable_properties={
                demangle_property_key(key): cast(
                    "PropertyValue",
                    value,
                )
                for key, value in episode_node.properties.items()
                if is_mangled_property_key(key)
            },
            user_metadata=json.loads(
                cast("str", episode_node.properties["user_metadata"]),
            ),
        )

    @staticmethod
    def _embedding_name(model_id: str, dimensions: int) -> str:
        """
        Generate a standardized property name for embeddings based on the model ID and embedding dimensions.

        Args:
            model_id (str): The identifier of the embedding model.
            dimensions (int): The dimensionality of the embedding.

        Returns:
            str: A standardized property name for the embedding.

        """
        return f"embedding_{model_id}_{dimensions}d"

    @staticmethod
    def _mangle_property_filter(
        property_filter: FilterExpr | None,
    ) -> FilterExpr | None:
        if property_filter is None:
            return None

        return map_filter_fields(property_filter, mangle_property_key)
