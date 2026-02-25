"""Unit tests for cross-collection multi-hop traversal.

Tests the target_collections parameter and the removal of the end-node
collection constraint in search_multi_hop_nodes().
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from memmachine.common.vector_graph_store.neo4j_vector_graph_store import (
    Neo4jVectorGraphStore,
    Neo4jVectorGraphStoreParams,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_driver() -> AsyncMock:
    """Create a mock Neo4j AsyncDriver with spec to satisfy Pydantic InstanceOf."""
    from neo4j import AsyncDriver

    driver = AsyncMock(spec=AsyncDriver)
    driver.execute_query = AsyncMock(return_value=([], MagicMock(), MagicMock()))
    return driver


@pytest.fixture
def store(mock_driver: AsyncMock) -> Neo4jVectorGraphStore:
    """Create a Neo4jVectorGraphStore with mocked driver."""
    return Neo4jVectorGraphStore(
        Neo4jVectorGraphStoreParams(
            driver=mock_driver,
            force_exact_similarity_search=True,
            dedup_trigger_threshold=0,
        )
    )


def _get_query_text(mock_driver: AsyncMock) -> str:
    """Extract the Cypher query string from the last execute_query call."""
    return str(mock_driver.execute_query.call_args[0][0])


# ---------------------------------------------------------------------------
# 5.1 - No target_collections omits end-node collection label
# ---------------------------------------------------------------------------


class TestCrossCollectionTraversal:
    @pytest.mark.asyncio
    async def test_no_target_collections_omits_end_node_label(
        self, store: Neo4jVectorGraphStore, mock_driver: AsyncMock
    ) -> None:
        """Without target_collections, end_node should have no label constraint."""
        await store.search_multi_hop_nodes(
            collection="test",
            this_node_uid="n1",
            max_hops=2,
        )

        query = _get_query_text(mock_driver)
        sanitized_col = Neo4jVectorGraphStore._sanitize_name("test")

        # Start node should still have collection label.
        assert f"(start:{sanitized_col}" in query
        # End node should NOT have a collection label -- just (end_node).
        assert "(end_node)\n" in query
        # Should NOT contain end_node:{sanitized_col}.
        assert f"end_node:{sanitized_col}" not in query

    # -------------------------------------------------------------------
    # 5.2 - Single target_collection generates WHERE clause
    # -------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_single_target_collection_generates_where(
        self, store: Neo4jVectorGraphStore, mock_driver: AsyncMock
    ) -> None:
        """With target_collections=["Episode"], Cypher should have a label WHERE."""
        await store.search_multi_hop_nodes(
            collection="test",
            this_node_uid="n1",
            max_hops=2,
            target_collections=["Episode"],
        )

        query = _get_query_text(mock_driver)
        sanitized_episode = Neo4jVectorGraphStore._sanitize_name("Episode")

        assert f"end_node:{sanitized_episode}" in query
        assert "AND (" in query

    # -------------------------------------------------------------------
    # 5.3 - Multiple target_collections generates OR-combined clause
    # -------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_multiple_target_collections_generates_or(
        self, store: Neo4jVectorGraphStore, mock_driver: AsyncMock
    ) -> None:
        """With multiple target_collections, Cypher should OR the labels."""
        await store.search_multi_hop_nodes(
            collection="test",
            this_node_uid="n1",
            max_hops=2,
            target_collections=["Episode", "Feature"],
        )

        query = _get_query_text(mock_driver)
        sanitized_episode = Neo4jVectorGraphStore._sanitize_name("Episode")
        sanitized_feature = Neo4jVectorGraphStore._sanitize_name("Feature")

        assert f"end_node:{sanitized_episode}" in query
        assert f"end_node:{sanitized_feature}" in query
        assert " OR " in query

    # -------------------------------------------------------------------
    # 5.4 - Empty target_collections returns no results
    # -------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_empty_target_collections_returns_nothing(
        self, store: Neo4jVectorGraphStore, mock_driver: AsyncMock
    ) -> None:
        """With target_collections=[] (empty), Cypher should use AND false."""
        await store.search_multi_hop_nodes(
            collection="test",
            this_node_uid="n1",
            max_hops=2,
            target_collections=[],
        )

        query = _get_query_text(mock_driver)
        assert "AND false" in query

    # -------------------------------------------------------------------
    # Backward compat - None default works like before (minus label)
    # -------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_target_collections_none_is_default(
        self, store: Neo4jVectorGraphStore, mock_driver: AsyncMock
    ) -> None:
        """Default target_collections=None should not add any label clause."""
        await store.search_multi_hop_nodes(
            collection="test",
            this_node_uid="n1",
            max_hops=2,
        )

        query = _get_query_text(mock_driver)
        # Should not contain any target collection WHERE clause.
        # The only AND clauses should be for edge/node filters (none here).
        assert "AND (end_node:" not in query
        assert "AND false" not in query

    # -------------------------------------------------------------------
    # Start node still constrained to its collection
    # -------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_start_node_still_constrained_to_collection(
        self, store: Neo4jVectorGraphStore, mock_driver: AsyncMock
    ) -> None:
        """Start node should always be constrained to the given collection."""
        await store.search_multi_hop_nodes(
            collection="Derivative_mydb",
            this_node_uid="n1",
            max_hops=2,
        )

        query = _get_query_text(mock_driver)
        sanitized_col = Neo4jVectorGraphStore._sanitize_name("Derivative_mydb")
        assert f"(start:{sanitized_col}" in query
