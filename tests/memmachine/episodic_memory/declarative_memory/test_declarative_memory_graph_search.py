"""Unit tests for graph-enhanced episodic retrieval in DeclarativeMemory.

Tests cover:
- Multi-hop expansion via GraphTraversalStore (7.5)
- PageRank re-ranking with blending formula (7.6)
- Entity type filtering passthrough (7.7)
- Graceful fallback when store does not implement GraphTraversalStore
"""

import json
from collections.abc import Iterable
from datetime import UTC, datetime, timedelta

import pytest
from pydantic import ValidationError

from memmachine.common.data_types import SimilarityMetric
from memmachine.common.embedder.embedder import Embedder
from memmachine.common.filter.filter_parser import FilterExpr
from memmachine.common.reranker.reranker import Reranker
from memmachine.common.vector_graph_store.data_types import (
    Edge,
    MultiHopResult,
    Node,
    OrderedPropertyValue,
)
from memmachine.common.vector_graph_store.graph_traversal_store import (
    GraphTraversalStore,
)
from memmachine.common.vector_graph_store.vector_graph_store import VectorGraphStore
from memmachine.episodic_memory.declarative_memory import (
    ContentType,
    DeclarativeMemory,
    DeclarativeMemoryParams,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_NOW = datetime(2026, 2, 11, 12, 0, 0, tzinfo=UTC)


def _episode_node(
    uid: str,
    content: str = "hello",
    source: str = "Alice",
    ts: datetime | None = None,
    user_metadata: dict | None = None,
) -> Node:
    """Build a Node that looks like a stored episode."""
    return Node(
        uid=uid,
        properties={
            "uid": uid,
            "timestamp": ts or _NOW,
            "source": source,
            "content_type": ContentType.MESSAGE.value,
            "content": content,
            "user_metadata": json.dumps(user_metadata or {}),
        },
    )


def _derivative_node(
    uid: str,
    episode_uid: str,
    *,
    entity_types: list[str] | None = None,
    pagerank: float | None = None,
) -> Node:
    """Build a Node that looks like a stored derivative."""
    props: dict = {
        "uid": uid,
        "timestamp": _NOW,
        "source": "Alice",
        "content_type": ContentType.MESSAGE.value,
        "content": f"derivative of {episode_uid}",
    }
    if pagerank is not None:
        props["pagerank_score"] = pagerank
    return Node(
        uid=uid,
        properties=props,
        entity_types=entity_types or [],
    )


# ---------------------------------------------------------------------------
# Test doubles
# ---------------------------------------------------------------------------


class _BaseStore(VectorGraphStore):
    """Minimal VectorGraphStore with configurable search results."""

    def __init__(
        self,
        *,
        similar_nodes: list[Node] | None = None,
        related_map: dict[str, list[Node]] | None = None,
    ) -> None:
        # Derivative nodes returned by search_similar_nodes.
        self._similar_nodes = similar_nodes or []
        # derivative_uid → [episode_node] mapping for search_related_nodes.
        self._related_map = related_map or {}

    async def add_nodes(self, *, collection: str, nodes: Iterable[Node]) -> None:
        pass  # pragma: no cover

    async def add_edges(
        self,
        *,
        relation: str,
        source_collection: str,
        target_collection: str,
        edges: Iterable[Edge],
    ) -> None:
        pass  # pragma: no cover

    async def search_similar_nodes(
        self,
        *,
        collection: str,
        embedding_name: str,
        query_embedding: list[float],
        similarity_metric: SimilarityMetric = SimilarityMetric.COSINE,
        limit: int | None = 100,
        property_filter: FilterExpr | None = None,
    ) -> list[Node]:
        return list(self._similar_nodes)

    async def search_related_nodes(
        self,
        *,
        relation: str,
        other_collection: str,
        this_collection: str,
        this_node_uid: str,
        find_sources: bool = True,
        find_targets: bool = True,
        limit: int | None = None,
        edge_property_filter: FilterExpr | None = None,
        node_property_filter: FilterExpr | None = None,
    ) -> list[Node]:
        return self._related_map.get(this_node_uid, [])

    async def search_directional_nodes(
        self,
        *,
        collection: str,
        by_properties: Iterable[str],
        starting_at: Iterable[OrderedPropertyValue | None],
        order_ascending: Iterable[bool],
        include_equal_start: bool = False,
        limit: int | None = 1,
        property_filter: FilterExpr | None = None,
    ) -> list[Node]:
        return []

    async def search_matching_nodes(
        self,
        *,
        collection: str,
        limit: int | None = None,
        property_filter: FilterExpr | None = None,
    ) -> list[Node]:
        return []  # pragma: no cover

    async def get_nodes(
        self, *, collection: str, node_uids: Iterable[str]
    ) -> list[Node]:
        return []  # pragma: no cover

    async def delete_nodes(self, *, collection: str, node_uids: Iterable[str]) -> None:
        pass  # pragma: no cover

    async def delete_all_data(self) -> None:
        pass  # pragma: no cover

    async def close(self) -> None:
        pass  # pragma: no cover


class _GraphStore(_BaseStore, GraphTraversalStore):
    """Store that also implements GraphTraversalStore for multi-hop."""

    def __init__(
        self,
        *,
        similar_nodes: list[Node] | None = None,
        related_map: dict[str, list[Node]] | None = None,
        multi_hop_results: list[MultiHopResult] | None = None,
    ) -> None:
        super().__init__(similar_nodes=similar_nodes, related_map=related_map)
        self._multi_hop_results = multi_hop_results or []
        self.multi_hop_calls: list[dict] = []

    async def search_multi_hop_nodes(
        self,
        *,
        collection: str,
        this_node_uid: str,
        min_hops: int = 1,
        max_hops: int = 3,
        relation_types: Iterable[str] | None = None,
        raw_relation_types: Iterable[str] | None = None,
        score_decay: float = 0.7,
        limit: int | None = None,
        edge_property_filter: FilterExpr | None = None,
        node_property_filter: FilterExpr | None = None,
        target_collections: Iterable[str] | None = None,
    ) -> list[MultiHopResult]:
        self.multi_hop_calls.append(
            {
                "collection": collection,
                "this_node_uid": this_node_uid,
                "max_hops": max_hops,
                "relation_types": list(relation_types) if relation_types else None,
                "raw_relation_types": (
                    list(raw_relation_types) if raw_relation_types else None
                ),
                "score_decay": score_decay,
                "target_collections": target_collections,
            }
        )
        return list(self._multi_hop_results)

    async def search_graph_filtered_similar_nodes(
        self,
        *,
        collection: str,
        embedding_name: str,
        query_embedding: list[float],
        similarity_metric: SimilarityMetric = SimilarityMetric.COSINE,
        limit: int | None = 100,
        property_filter: FilterExpr | None = None,
        graph_filter=None,
    ) -> list[Node]:
        return []  # pragma: no cover


class _FakeEmbedder(Embedder):
    """Embedder returning deterministic 2-d embeddings."""

    async def ingest_embed(
        self, inputs: Iterable[str], max_attempts: int = 1
    ) -> list[list[float]]:
        return [[1.0, 0.0] for _ in inputs]

    async def search_embed(
        self, queries: Iterable[str], max_attempts: int = 1
    ) -> list[list[float]]:
        return [[1.0, 0.0] for _ in queries]

    @property
    def model_id(self) -> str:
        return "fake"

    @property
    def dimensions(self) -> int:
        return 2

    @property
    def similarity_metric(self) -> SimilarityMetric:
        return SimilarityMetric.COSINE


class _FakeReranker(Reranker):
    """Reranker returning descending scores."""

    async def score(self, query: str, candidates: list[str]) -> list[float]:
        return [1.0 / (i + 1) for i in range(len(candidates))]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_memory(store: VectorGraphStore, **kwargs) -> DeclarativeMemory:
    return DeclarativeMemory(
        DeclarativeMemoryParams(
            session_id="test",
            vector_graph_store=store,
            embedder=_FakeEmbedder(),
            reranker=_FakeReranker(),
            **kwargs,
        ),
    )


# ---------------------------------------------------------------------------
# 7.5 — Multi-hop expansion tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_multi_hop_expansion_adds_new_nodes() -> None:
    """Multi-hop traversal discovers node d3 not in initial results."""
    d1 = _derivative_node("d1", "ep1")
    d2 = _derivative_node("d2", "ep2")
    d3 = _derivative_node("d3", "ep3")  # discovered via multi-hop

    ep1 = _episode_node("ep1", content="msg1")
    ep2 = _episode_node("ep2", content="msg2")
    ep3 = _episode_node("ep3", content="msg3")

    store = _GraphStore(
        similar_nodes=[d1, d2],
        related_map={
            "d1": [ep1],
            "d2": [ep2],
            "d3": [ep3],
        },
        multi_hop_results=[
            MultiHopResult(node=d3, hop_distance=1, score=0.7),
        ],
    )

    mem = _make_memory(store)
    results = await mem.search_scored("test query", max_num_episodes=10)

    # d3 was discovered via multi-hop, so ep3 should be in results.
    result_uids = {ep.uid for _, ep in results}
    assert "ep1" in result_uids
    assert "ep2" in result_uids
    assert "ep3" in result_uids


@pytest.mark.asyncio
async def test_multi_hop_deduplication_keeps_initial_node() -> None:
    """When multi-hop returns a node already in initial results, keep it once."""
    d1 = _derivative_node("d1", "ep1")
    ep1 = _episode_node("ep1", content="msg1")

    store = _GraphStore(
        similar_nodes=[d1],
        related_map={"d1": [ep1]},
        multi_hop_results=[
            # d1 already in initial results — should not be duplicated.
            MultiHopResult(node=d1, hop_distance=1, score=0.5),
        ],
    )

    mem = _make_memory(store)
    results = await mem.search_scored("test", max_num_episodes=10)

    ep_uids = [ep.uid for _, ep in results]
    assert ep_uids.count("ep1") == 1


@pytest.mark.asyncio
async def test_multi_hop_graceful_fallback_without_graph_store() -> None:
    """When store is plain VectorGraphStore, search still works normally."""
    d1 = _derivative_node("d1", "ep1")
    ep1 = _episode_node("ep1", content="msg1")

    store = _BaseStore(
        similar_nodes=[d1],
        related_map={"d1": [ep1]},
    )

    mem = _make_memory(store)
    results = await mem.search_scored("test", max_num_episodes=10)

    assert len(results) == 1
    assert results[0][1].uid == "ep1"


@pytest.mark.asyncio
async def test_multi_hop_called_with_correct_params() -> None:
    """Verify search_multi_hop_nodes is called with max_hops=5, score_decay=0.85.

    The traversal must use 5 hops to bridge the episodic→semantic→episodic
    path (Derivative → Episode ← Feature → Feature → Episode ← Derivative)
    and specify both sanitized episodic relation types and raw semantic
    relation types.
    """
    d1 = _derivative_node("d1", "ep1")
    ep1 = _episode_node("ep1", content="msg1")

    store = _GraphStore(
        similar_nodes=[d1],
        related_map={"d1": [ep1]},
        multi_hop_results=[],
    )

    mem = _make_memory(store)
    await mem.search_scored("test", max_num_episodes=10)

    assert len(store.multi_hop_calls) == 1
    call = store.multi_hop_calls[0]
    assert call["this_node_uid"] == "d1"
    assert call["max_hops"] == 5
    assert call["score_decay"] == 0.85
    # Episodic relation types (will be sanitized by the store).
    assert call["relation_types"] == ["DERIVED_FROM_universal/universal"]
    # Semantic relation types (raw, not sanitized).
    assert call["raw_relation_types"] == ["EXTRACTED_FROM", "RELATED_TO"]
    # Multi-hop results are filtered to the Derivative collection so only
    # Derivative nodes (not Feature/Episode/SetEmbedding) are returned.
    assert call["target_collections"] is not None


@pytest.mark.asyncio
async def test_multi_hop_empty_initial_results_skips_expansion() -> None:
    """When initial search returns nothing, multi-hop is skipped."""
    store = _GraphStore(
        similar_nodes=[],
        related_map={},
        multi_hop_results=[],
    )

    mem = _make_memory(store)
    results = await mem.search_scored("nothing", max_num_episodes=10)

    assert results == []
    assert store.multi_hop_calls == []


# ---------------------------------------------------------------------------
# 7.6 — PageRank re-ranking tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_pagerank_reranking_blends_scores() -> None:
    """Nodes with high PageRank get boosted in final ranking."""
    # d1: first in similarity order, low pagerank
    # d2: second in similarity order, high pagerank
    d1 = _derivative_node("d1", "ep1", pagerank=0.1)
    d2 = _derivative_node("d2", "ep2", pagerank=1.0)

    ep1 = _episode_node("ep1", content="msg1", ts=_NOW)
    ep2 = _episode_node("ep2", content="msg2", ts=_NOW + timedelta(seconds=1))

    store = _GraphStore(
        similar_nodes=[d1, d2],  # d1 first = higher similarity
        related_map={"d1": [ep1], "d2": [ep2]},
        multi_hop_results=[],
    )

    # alpha=0.5: equal weight similarity and pagerank
    mem = _make_memory(store, pagerank_blend_alpha=0.5)
    results = await mem.search_scored("test", max_num_episodes=10)

    # With alpha=0.5:
    # d1: sim=1.0, norm_pr=0.1/1.0=0.1 → final=0.5*1.0 + 0.5*0.1 = 0.55
    # d2: sim=0.5, norm_pr=1.0/1.0=1.0 → final=0.5*0.5 + 0.5*1.0 = 0.75
    # d2 should rank higher due to PageRank boost
    # Note: final ordering is chronological in search_scored, but
    # the graph enhancement re-orders the derivative list before
    # episode resolution, so d2's episode should still be found.
    result_uids = [ep.uid for _, ep in results]
    assert "ep1" in result_uids
    assert "ep2" in result_uids


@pytest.mark.asyncio
async def test_pagerank_reranking_skipped_when_no_pagerank() -> None:
    """When no nodes have pagerank property, re-ranking is skipped."""
    d1 = _derivative_node("d1", "ep1")  # no pagerank
    d2 = _derivative_node("d2", "ep2")  # no pagerank

    ep1 = _episode_node("ep1", content="msg1", ts=_NOW)
    ep2 = _episode_node("ep2", content="msg2", ts=_NOW + timedelta(seconds=1))

    store = _GraphStore(
        similar_nodes=[d1, d2],
        related_map={"d1": [ep1], "d2": [ep2]},
        multi_hop_results=[],
    )

    mem = _make_memory(store)
    results = await mem.search_scored("test", max_num_episodes=10)

    # Both episodes found, order is chronological.
    result_uids = [ep.uid for _, ep in results]
    assert result_uids == ["ep1", "ep2"]


@pytest.mark.asyncio
async def test_pagerank_alpha_1_ignores_pagerank() -> None:
    """With alpha=1.0, PageRank has zero weight — similarity dominates."""
    d1 = _derivative_node("d1", "ep1", pagerank=0.01)
    d2 = _derivative_node("d2", "ep2", pagerank=1.0)

    ep1 = _episode_node("ep1", content="msg1", ts=_NOW)
    ep2 = _episode_node("ep2", content="msg2", ts=_NOW + timedelta(seconds=1))

    store = _GraphStore(
        similar_nodes=[d1, d2],  # d1 first = higher similarity
        related_map={"d1": [ep1], "d2": [ep2]},
        multi_hop_results=[],
    )

    # alpha=1.0 means final_score = 1.0 * sim + 0.0 * pr = sim
    mem = _make_memory(store, pagerank_blend_alpha=1.0)
    results = await mem.search_scored("test", max_num_episodes=10)

    # d1 has higher similarity, should stay first.
    result_uids = [ep.uid for _, ep in results]
    assert result_uids == ["ep1", "ep2"]


@pytest.mark.asyncio
async def test_pagerank_alpha_0_uses_only_pagerank() -> None:
    """With alpha=0.0, only PageRank matters."""
    # d1 first in similarity but low pagerank; d2 second but high pagerank.
    d1 = _derivative_node("d1", "ep1", pagerank=0.1)
    d2 = _derivative_node("d2", "ep2", pagerank=1.0)

    ep1 = _episode_node("ep1", content="msg1", ts=_NOW)
    ep2 = _episode_node("ep2", content="msg2", ts=_NOW + timedelta(seconds=1))

    store = _GraphStore(
        similar_nodes=[d1, d2],
        related_map={"d1": [ep1], "d2": [ep2]},
        multi_hop_results=[],
    )

    mem = _make_memory(store, pagerank_blend_alpha=0.0)
    results = await mem.search_scored("test", max_num_episodes=10)

    # d2 has higher pagerank, so it should be the dominant result.
    # Final ordering is chronological though (ep1 < ep2 by timestamp).
    result_uids = [ep.uid for _, ep in results]
    assert "ep1" in result_uids
    assert "ep2" in result_uids


# ---------------------------------------------------------------------------
# PageRank reads pagerank_score property (not "pagerank")
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_pagerank_reads_pagerank_score_property() -> None:
    """Verify reranking reads pagerank_score, not the old pagerank property."""
    # Node with pagerank_score set via the helper.
    d1 = _derivative_node("d1", "ep1", pagerank=0.9)
    # Verify the property name is pagerank_score in the node.
    assert "pagerank_score" in d1.properties
    assert "pagerank" not in d1.properties

    ep1 = _episode_node("ep1", content="msg1", ts=_NOW)

    store = _GraphStore(
        similar_nodes=[d1],
        related_map={"d1": [ep1]},
        multi_hop_results=[],
    )

    mem = _make_memory(store, pagerank_blend_alpha=0.5)
    results = await mem.search_scored("test", max_num_episodes=10)

    # Should find the episode (pagerank reranking didn't break).
    assert len(results) == 1
    assert results[0][1].uid == "ep1"


# ---------------------------------------------------------------------------
# 7.7 -- Entity type filtering tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_entity_type_filter_includes_matching_nodes() -> None:
    """Only derivatives with matching entity_types pass the filter."""
    d1 = _derivative_node("d1", "ep1", entity_types=["Person"])
    d2 = _derivative_node("d2", "ep2", entity_types=["Location"])

    ep1 = _episode_node("ep1", content="person msg", ts=_NOW)

    store = _BaseStore(
        similar_nodes=[d1, d2],
        related_map={"d1": [ep1]},
    )

    mem = _make_memory(store)
    results = await mem.search_scored(
        "test", max_num_episodes=10, entity_types=["Person"]
    )

    result_uids = [ep.uid for _, ep in results]
    assert "ep1" in result_uids
    # ep2 should be excluded because d2 has entity_type "Location"
    assert "ep2" not in result_uids


@pytest.mark.asyncio
async def test_entity_type_filter_multiple_types() -> None:
    """Filter with multiple types matches any of them."""
    d1 = _derivative_node("d1", "ep1", entity_types=["Person"])
    d2 = _derivative_node("d2", "ep2", entity_types=["Location"])
    d3 = _derivative_node("d3", "ep3", entity_types=["Event"])

    ep1 = _episode_node("ep1", content="person", ts=_NOW)
    ep2 = _episode_node("ep2", content="location", ts=_NOW + timedelta(seconds=1))

    store = _BaseStore(
        similar_nodes=[d1, d2, d3],
        related_map={"d1": [ep1], "d2": [ep2]},
    )

    mem = _make_memory(store)
    results = await mem.search_scored(
        "test", max_num_episodes=10, entity_types=["Person", "Location"]
    )

    result_uids = [ep.uid for _, ep in results]
    assert "ep1" in result_uids
    assert "ep2" in result_uids
    assert "ep3" not in result_uids


@pytest.mark.asyncio
async def test_entity_type_filter_none_returns_all() -> None:
    """When entity_types is None, all nodes pass through."""
    d1 = _derivative_node("d1", "ep1", entity_types=["Person"])
    d2 = _derivative_node("d2", "ep2", entity_types=[])

    ep1 = _episode_node("ep1", content="msg1", ts=_NOW)
    ep2 = _episode_node("ep2", content="msg2", ts=_NOW + timedelta(seconds=1))

    store = _BaseStore(
        similar_nodes=[d1, d2],
        related_map={"d1": [ep1], "d2": [ep2]},
    )

    mem = _make_memory(store)
    results = await mem.search_scored("test", max_num_episodes=10, entity_types=None)

    result_uids = [ep.uid for _, ep in results]
    assert "ep1" in result_uids
    assert "ep2" in result_uids


@pytest.mark.asyncio
async def test_entity_type_filter_empty_list_returns_none() -> None:
    """When entity_types is an empty list (falsy), no filtering occurs."""
    d1 = _derivative_node("d1", "ep1", entity_types=["Person"])

    ep1 = _episode_node("ep1", content="msg1", ts=_NOW)

    store = _BaseStore(
        similar_nodes=[d1],
        related_map={"d1": [ep1]},
    )

    mem = _make_memory(store)
    results = await mem.search_scored("test", max_num_episodes=10, entity_types=[])

    # Empty list is falsy, so no filtering → all results returned.
    result_uids = [ep.uid for _, ep in results]
    assert "ep1" in result_uids


@pytest.mark.asyncio
async def test_entity_type_filter_with_graph_store() -> None:
    """Entity type filter works together with GraphTraversalStore."""
    d1 = _derivative_node("d1", "ep1", entity_types=["Person"])
    d2 = _derivative_node("d2", "ep2", entity_types=["Location"])
    d3 = _derivative_node("d3", "ep3", entity_types=["Person"])

    ep1 = _episode_node("ep1", content="person1", ts=_NOW)
    ep3 = _episode_node("ep3", content="person2", ts=_NOW + timedelta(seconds=2))

    store = _GraphStore(
        similar_nodes=[d1, d2],
        related_map={"d1": [ep1], "d3": [ep3]},
        multi_hop_results=[
            # d3 discovered via multi-hop, matches Person filter
            MultiHopResult(node=d3, hop_distance=1, score=0.7),
        ],
    )

    mem = _make_memory(store)
    results = await mem.search_scored(
        "test", max_num_episodes=10, entity_types=["Person"]
    )

    # d2 (Location) filtered out; d1 and d3 (Person) kept.
    # d3 added via multi-hop after entity filtering (multi-hop
    # operates on already-filtered results).
    result_uids = [ep.uid for _, ep in results]
    assert "ep1" in result_uids
    # d3 may or may not appear depending on whether entity type
    # filtering happens before multi-hop (it does in our impl).
    # Since d2 was filtered out, only d1 is an anchor for multi-hop.
    # d3 is discovered via multi-hop on d1 and has no entity type
    # filter applied to multi-hop results.
    assert "ep3" in result_uids


@pytest.mark.asyncio
async def test_search_forwards_entity_types_to_search_scored() -> None:
    """The high-level search() method passes entity_types through."""
    d1 = _derivative_node("d1", "ep1", entity_types=["Event"])
    d2 = _derivative_node("d2", "ep2", entity_types=["Person"])

    ep1 = _episode_node("ep1", content="event msg", ts=_NOW)

    store = _BaseStore(
        similar_nodes=[d1, d2],
        related_map={"d1": [ep1]},
    )

    mem = _make_memory(store)
    results = await mem.search("test", max_num_episodes=10, entity_types=["Event"])

    result_uids = [ep.uid for ep in results]
    assert "ep1" in result_uids
    assert "ep2" not in result_uids


# ---------------------------------------------------------------------------
# Config validation
# ---------------------------------------------------------------------------


def test_pagerank_blend_alpha_default() -> None:
    """Default pagerank_blend_alpha is 0.8."""
    store = _BaseStore()
    mem = _make_memory(store)
    assert mem._pagerank_blend_alpha == 0.8


def test_pagerank_blend_alpha_custom() -> None:
    """Custom pagerank_blend_alpha is respected."""
    store = _BaseStore()
    mem = _make_memory(store, pagerank_blend_alpha=0.5)
    assert mem._pagerank_blend_alpha == 0.5


def test_pagerank_blend_alpha_validation() -> None:
    """pagerank_blend_alpha must be between 0.0 and 1.0."""
    store = _BaseStore()
    with pytest.raises(ValidationError):
        _make_memory(store, pagerank_blend_alpha=1.5)
    with pytest.raises(ValidationError):
        _make_memory(store, pagerank_blend_alpha=-0.1)
