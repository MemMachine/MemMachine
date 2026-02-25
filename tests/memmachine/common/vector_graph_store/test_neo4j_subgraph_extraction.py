"""Unit tests for extract_subgraph() method on Neo4jVectorGraphStore."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from memmachine.common.vector_graph_store.neo4j_vector_graph_store import (
    Neo4jVectorGraphStore,
    Neo4jVectorGraphStoreParams,
    SubgraphResult,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_driver() -> AsyncMock:
    from neo4j import AsyncDriver

    driver = AsyncMock(spec=AsyncDriver)
    driver.execute_query = AsyncMock(return_value=([], MagicMock(), MagicMock()))
    return driver


@pytest.fixture
def store(mock_driver: AsyncMock) -> Neo4jVectorGraphStore:
    return Neo4jVectorGraphStore(
        Neo4jVectorGraphStoreParams(
            driver=mock_driver,
            force_exact_similarity_search=True,
            dedup_trigger_threshold=0,
        )
    )


def _make_node_rec(uid: str, **props) -> MagicMock:
    """Create a mock record with a Neo4j node."""
    data = {"uid": uid, **props}
    node = MagicMock()
    node.get = lambda k, default="", _d=data: _d.get(k, default)
    node.__iter__ = lambda self, _d=data: iter(_d)
    node.items = lambda _d=data: _d.items()
    node.__getitem__ = lambda self, k, _d=data: _d[k]
    node.keys = lambda _d=data: _d.keys()
    node.values = lambda _d=data: _d.values()
    node.__len__ = lambda self, _d=data: len(_d)

    rec = MagicMock()
    rec.__getitem__ = lambda self, k, _n=node: _n if k == "n" else None
    return rec


def _make_edge_rec(
    src: str, tgt: str, rel_type: str, props: dict | None = None
) -> MagicMock:
    """Create a mock record for an edge query result."""
    data = {"src": src, "tgt": tgt, "rel_type": rel_type, "props": props or {}}
    rec = MagicMock()
    rec.__getitem__ = lambda self, k, _d=data: _d[k]
    return rec


def _result(records: list) -> tuple:
    return (records, MagicMock(), MagicMock())


def _empty() -> tuple:
    return ([], MagicMock(), MagicMock())


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestExtractSubgraph:
    # -------------------------------------------------------------------
    # 3.1 — Returns correct nodes and edges
    # -------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_returns_correct_nodes_and_edges(
        self, store: Neo4jVectorGraphStore, mock_driver: AsyncMock
    ) -> None:
        mock_driver.execute_query.side_effect = [
            # Query 1: nodes
            _result(
                [
                    _make_node_rec("alice", name="Alice"),
                    _make_node_rec("bob", name="Bob"),
                    _make_node_rec("charlie", name="Charlie"),
                ]
            ),
            # Query 2: edges
            _result(
                [
                    _make_edge_rec("alice", "bob", "KNOWS"),
                    _make_edge_rec("bob", "charlie", "WORKS_WITH"),
                ]
            ),
        ]

        result = await store.extract_subgraph(collection="test", anchor_uid="alice")

        assert isinstance(result, SubgraphResult)
        assert len(result.nodes) == 3
        assert {n.uid for n in result.nodes} == {"alice", "bob", "charlie"}
        assert len(result.edges) == 2
        assert result.edges[0].type == "KNOWS"
        assert result.edges[1].type == "WORKS_WITH"

    # -------------------------------------------------------------------
    # 3.2 — Anchor node only when no neighbors
    # -------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_returns_only_anchor_when_no_neighbors(
        self, store: Neo4jVectorGraphStore, mock_driver: AsyncMock
    ) -> None:
        mock_driver.execute_query.side_effect = [
            _result([_make_node_rec("lonely")]),
            # No second query because < 2 nodes
        ]

        result = await store.extract_subgraph(collection="test", anchor_uid="lonely")

        assert len(result.nodes) == 1
        assert result.nodes[0].uid == "lonely"
        assert result.edges == []

    # -------------------------------------------------------------------
    # 3.3 — Relationship type filter
    # -------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_passes_relation_type_filter(
        self, store: Neo4jVectorGraphStore, mock_driver: AsyncMock
    ) -> None:
        mock_driver.execute_query.side_effect = [
            _result([_make_node_rec("a"), _make_node_rec("b")]),
            _empty(),
        ]

        await store.extract_subgraph(
            collection="test",
            anchor_uid="a",
            relation_types=["KNOWS"],
        )

        # Both queries should reference the relation type
        first_call = mock_driver.execute_query.call_args_list[0]
        assert "`KNOWS`" in first_call[0][0].text
        second_call = mock_driver.execute_query.call_args_list[1]
        assert "`KNOWS`" in second_call[0][0].text

    # -------------------------------------------------------------------
    # 3.4 — node_limit and max_depth in Cypher
    # -------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_respects_node_limit_and_max_depth(
        self, store: Neo4jVectorGraphStore, mock_driver: AsyncMock
    ) -> None:
        mock_driver.execute_query.side_effect = [
            _result([_make_node_rec("a")]),
        ]

        await store.extract_subgraph(
            collection="test",
            anchor_uid="a",
            max_depth=3,
            node_limit=5,
        )

        call_args = mock_driver.execute_query.call_args_list[0]
        query_text = call_args[0][0].text
        assert "*0..3" in query_text
        # node_limit passed as parameter
        assert call_args[1]["node_limit"] == 5
