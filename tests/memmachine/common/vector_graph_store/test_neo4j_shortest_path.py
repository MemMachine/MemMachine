"""Unit tests for shortest_path() method on Neo4jVectorGraphStore."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from memmachine.common.vector_graph_store.neo4j_vector_graph_store import (
    Neo4jVectorGraphStore,
    Neo4jVectorGraphStoreParams,
    ShortestPathResult,
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


def _make_neo4j_node(uid: str, **props) -> MagicMock:
    """Create a mock Neo4j path node with uid and properties."""
    node = MagicMock()
    data = {"uid": uid, **props}
    node.get = lambda k, default="", _d=data: _d.get(k, default)
    node.__iter__ = lambda self, _d=data: iter(_d)
    node.items = lambda _d=data: _d.items()
    node.__getitem__ = lambda self, k, _d=data: _d[k]
    node.keys = lambda _d=data: _d.keys()
    node.values = lambda _d=data: _d.values()
    node.__len__ = lambda self, _d=data: len(_d)
    return node


def _make_neo4j_rel(
    start_node: MagicMock, end_node: MagicMock, rel_type: str, **props
) -> MagicMock:
    """Create a mock Neo4j path relationship."""
    rel = MagicMock()
    rel.start_node = start_node
    rel.end_node = end_node
    rel.type = rel_type
    data = {**props}
    rel.__iter__ = lambda self, _d=data: iter(_d)
    rel.items = lambda _d=data: _d.items()
    rel.__getitem__ = lambda self, k, _d=data: _d[k]
    rel.keys = lambda _d=data: _d.keys()
    rel.values = lambda _d=data: _d.values()
    rel.__len__ = lambda self, _d=data: len(_d)
    return rel


def _path_result(nodes: list, relationships: list) -> tuple:
    """Create a mock Cypher result containing a path."""
    path = MagicMock()
    path.nodes = nodes
    path.relationships = relationships
    rec = MagicMock()
    rec.__getitem__ = lambda self, k: path if k == "p" else None
    return ([rec], MagicMock(), MagicMock())


def _empty() -> tuple:
    return ([], MagicMock(), MagicMock())


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestShortestPath:
    # -------------------------------------------------------------------
    # 3.1 — Returns correct path with nodes and edges
    # -------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_returns_correct_path(
        self, store: Neo4jVectorGraphStore, mock_driver: AsyncMock
    ) -> None:
        node_a = _make_neo4j_node("alice", name="Alice")
        node_b = _make_neo4j_node("bob", name="Bob")
        node_c = _make_neo4j_node("charlie", name="Charlie")

        rel_ab = _make_neo4j_rel(node_a, node_b, "KNOWS")
        rel_bc = _make_neo4j_rel(node_b, node_c, "WORKS_WITH")

        mock_driver.execute_query.return_value = _path_result(
            [node_a, node_b, node_c], [rel_ab, rel_bc]
        )

        result = await store.shortest_path(
            collection="test", source_uid="alice", target_uid="charlie"
        )

        assert isinstance(result, ShortestPathResult)
        assert result.path_length == 2
        assert len(result.nodes) == 3
        assert result.nodes[0].uid == "alice"
        assert result.nodes[1].uid == "bob"
        assert result.nodes[2].uid == "charlie"
        assert len(result.edges) == 2
        assert result.edges[0].type == "KNOWS"
        assert result.edges[1].type == "WORKS_WITH"

    # -------------------------------------------------------------------
    # 3.2 — Returns empty result when no path found
    # -------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_returns_empty_when_no_path(
        self, store: Neo4jVectorGraphStore, mock_driver: AsyncMock
    ) -> None:
        mock_driver.execute_query.return_value = _empty()

        result = await store.shortest_path(
            collection="test", source_uid="alice", target_uid="disconnected"
        )

        assert result.path_length == 0
        assert result.nodes == []
        assert result.edges == []

    # -------------------------------------------------------------------
    # 3.3 — Relationship type filter in Cypher query
    # -------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_passes_relation_type_filter(
        self, store: Neo4jVectorGraphStore, mock_driver: AsyncMock
    ) -> None:
        mock_driver.execute_query.return_value = _empty()

        await store.shortest_path(
            collection="test",
            source_uid="a",
            target_uid="b",
            relation_types=["KNOWS", "WORKS_WITH"],
        )

        call_args = mock_driver.execute_query.call_args
        query_text = call_args[0][0].text
        assert "`KNOWS`" in query_text
        assert "`WORKS_WITH`" in query_text

    # -------------------------------------------------------------------
    # 3.4 — Max depth parameter
    # -------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_respects_max_depth(
        self, store: Neo4jVectorGraphStore, mock_driver: AsyncMock
    ) -> None:
        mock_driver.execute_query.return_value = _empty()

        await store.shortest_path(
            collection="test",
            source_uid="a",
            target_uid="b",
            max_depth=3,
        )

        call_args = mock_driver.execute_query.call_args
        query_text = call_args[0][0].text
        assert "*..3" in query_text
