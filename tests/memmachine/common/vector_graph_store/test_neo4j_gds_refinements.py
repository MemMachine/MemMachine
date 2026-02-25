"""Unit tests for GDS ranking refinements.

Tests the new relation_types, write_property, and projection-size metrics
parameters added to compute_pagerank(), detect_communities(), and
_create_scoped_projection().
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from memmachine.common.vector_graph_store.neo4j_vector_graph_store import (
    Neo4jVectorGraphStore,
    Neo4jVectorGraphStoreParams,
    ProjectionInfo,
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


def _graph_list_record(node_count: int = 10, rel_count: int = 25) -> MagicMock:
    rec = MagicMock()
    rec.__getitem__ = lambda self, k, d={
        "nodeCount": node_count,
        "relationshipCount": rel_count,
    }: d[k]
    return rec


def _empty_result() -> tuple:
    return ([], MagicMock(), MagicMock())


# ---------------------------------------------------------------------------
# 6.1 - Projection without relation_types
# ---------------------------------------------------------------------------


class TestCreateScopedProjection:
    @pytest.mark.asyncio
    async def test_no_relation_types_omits_where_clause(
        self, store: Neo4jVectorGraphStore, mock_driver: AsyncMock
    ) -> None:
        """When relation_types is None, the relationship Cypher has no WHERE."""
        store._gds_enabled = True
        store._gds_available = True

        mock_driver.execute_query.side_effect = [
            _empty_result(),  # gds.graph.project.cypher
            ([_graph_list_record()], MagicMock(), MagicMock()),  # gds.graph.list
        ]

        result = await store._create_scoped_projection(
            collection="test",
            projection_name="proj_1",
        )

        assert isinstance(result, ProjectionInfo)
        assert result.name == "proj_1"

        # Verify the projection call does NOT contain WHERE type(r)
        project_call = mock_driver.execute_query.call_args_list[0]
        query_text = str(project_call[0][0])
        assert "WHERE type(r)" not in query_text

    # -------------------------------------------------------------------
    # 6.2 - Projection with single relation_type
    # -------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_single_relation_type_generates_where_clause(
        self, store: Neo4jVectorGraphStore, mock_driver: AsyncMock
    ) -> None:
        """When relation_types has one element, WHERE clause filters to it."""
        store._gds_enabled = True
        store._gds_available = True

        mock_driver.execute_query.side_effect = [
            _empty_result(),  # gds.graph.project.cypher
            ([_graph_list_record()], MagicMock(), MagicMock()),  # gds.graph.list
        ]

        await store._create_scoped_projection(
            collection="test",
            projection_name="proj_2",
            relation_types=["RELATES_TO"],
        )

        project_call = mock_driver.execute_query.call_args_list[0]
        query_text = str(project_call[0][0])
        assert "WHERE type(r) IN ['RELATES_TO']" in query_text

    # -------------------------------------------------------------------
    # 6.3 - Projection returns ProjectionInfo with counts
    # -------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_returns_projection_info_with_counts(
        self, store: Neo4jVectorGraphStore, mock_driver: AsyncMock
    ) -> None:
        """ProjectionInfo contains node_count and relationship_count from gds.graph.list()."""
        store._gds_enabled = True
        store._gds_available = True

        mock_driver.execute_query.side_effect = [
            _empty_result(),  # gds.graph.project.cypher
            ([_graph_list_record(42, 99)], MagicMock(), MagicMock()),  # gds.graph.list
        ]

        info = await store._create_scoped_projection(
            collection="test",
            projection_name="proj_3",
        )

        assert info.node_count == 42
        assert info.relationship_count == 99

    @pytest.mark.asyncio
    async def test_multiple_relation_types_generates_or_clause(
        self, store: Neo4jVectorGraphStore, mock_driver: AsyncMock
    ) -> None:
        """Multiple relation types are comma-separated in the IN clause."""
        store._gds_enabled = True
        store._gds_available = True

        mock_driver.execute_query.side_effect = [
            _empty_result(),
            ([_graph_list_record()], MagicMock(), MagicMock()),
        ]

        await store._create_scoped_projection(
            collection="test",
            projection_name="proj_multi",
            relation_types=["RELATES_TO", "EXTRACTED_FROM"],
        )

        project_call = mock_driver.execute_query.call_args_list[0]
        query_text = str(project_call[0][0])
        assert "WHERE type(r) IN ['RELATES_TO', 'EXTRACTED_FROM']" in query_text


# ---------------------------------------------------------------------------
# 7.1-7.3 - compute_pagerank() parameter extensions
# ---------------------------------------------------------------------------


class TestComputePagerankRefinements:
    @pytest.mark.asyncio
    async def test_passes_relation_types_to_projection(
        self, store: Neo4jVectorGraphStore, mock_driver: AsyncMock
    ) -> None:
        """relation_types is forwarded to _create_scoped_projection()."""
        store._gds_enabled = True
        store._gds_available = True

        score_rec = MagicMock()
        score_rec.__getitem__ = lambda self, k, d={"uid": "n1", "score": 0.5}: d[k]

        mock_driver.execute_query.side_effect = [
            _empty_result(),  # gds.graph.project.cypher
            ([_graph_list_record()], MagicMock(), MagicMock()),  # gds.graph.list
            ([score_rec], MagicMock(), MagicMock()),  # gds.pageRank.stream
            _empty_result(),  # _drop_projection
        ]

        await store.compute_pagerank(
            collection="test",
            relation_types=["RELATES_TO"],
        )

        # Verify the projection Cypher contains the relation_types filter
        project_call = mock_driver.execute_query.call_args_list[0]
        query_text = str(project_call[0][0])
        assert "WHERE type(r) IN ['RELATES_TO']" in query_text

    @pytest.mark.asyncio
    async def test_custom_write_property(
        self, store: Neo4jVectorGraphStore, mock_driver: AsyncMock
    ) -> None:
        """write_property='importance_score' uses that property in gds.pageRank.write()."""
        store._gds_enabled = True
        store._gds_available = True

        score_rec = MagicMock()
        score_rec.__getitem__ = lambda self, k, d={"uid": "n1", "score": 0.5}: d[k]

        mock_driver.execute_query.side_effect = [
            _empty_result(),  # gds.graph.project.cypher
            ([_graph_list_record()], MagicMock(), MagicMock()),  # gds.graph.list
            _empty_result(),  # gds.pageRank.write
            ([score_rec], MagicMock(), MagicMock()),  # gds.pageRank.stream
            _empty_result(),  # _drop_projection
        ]

        await store.compute_pagerank(
            collection="test",
            write_back=True,
            write_property="importance_score",
        )

        # The write call is the 3rd execute_query call (index 2)
        write_call = mock_driver.execute_query.call_args_list[2]
        # Check the write_prop kwarg
        assert write_call.kwargs.get("write_prop") == "importance_score"

    @pytest.mark.asyncio
    async def test_default_write_property(
        self, store: Neo4jVectorGraphStore, mock_driver: AsyncMock
    ) -> None:
        """write_property=None defaults to 'pagerank_score'."""
        store._gds_enabled = True
        store._gds_available = True

        score_rec = MagicMock()
        score_rec.__getitem__ = lambda self, k, d={"uid": "n1", "score": 0.5}: d[k]

        mock_driver.execute_query.side_effect = [
            _empty_result(),  # gds.graph.project.cypher
            ([_graph_list_record()], MagicMock(), MagicMock()),  # gds.graph.list
            _empty_result(),  # gds.pageRank.write
            ([score_rec], MagicMock(), MagicMock()),  # gds.pageRank.stream
            _empty_result(),  # _drop_projection
        ]

        await store.compute_pagerank(
            collection="test",
            write_back=True,
            write_property=None,
        )

        write_call = mock_driver.execute_query.call_args_list[2]
        assert write_call.kwargs.get("write_prop") == "pagerank_score"


# ---------------------------------------------------------------------------
# 7.4-7.6 - detect_communities() parameter extensions
# ---------------------------------------------------------------------------


class TestDetectCommunitiesRefinements:
    @pytest.mark.asyncio
    async def test_passes_relation_types_to_projection(
        self, store: Neo4jVectorGraphStore, mock_driver: AsyncMock
    ) -> None:
        """relation_types is forwarded to _create_scoped_projection()."""
        store._gds_enabled = True
        store._gds_available = True

        rec = MagicMock()
        rec.__getitem__ = lambda self, k, d={"uid": "n1", "communityId": 0}: d[k]

        mock_driver.execute_query.side_effect = [
            _empty_result(),  # gds.graph.project.cypher
            ([_graph_list_record()], MagicMock(), MagicMock()),  # gds.graph.list
            ([rec], MagicMock(), MagicMock()),  # gds.louvain.stream
            _empty_result(),  # _drop_projection
        ]

        await store.detect_communities(
            collection="test",
            relation_types=["RELATES_TO", "EXTRACTED_FROM"],
        )

        project_call = mock_driver.execute_query.call_args_list[0]
        query_text = str(project_call[0][0])
        assert "WHERE type(r) IN ['RELATES_TO', 'EXTRACTED_FROM']" in query_text

    @pytest.mark.asyncio
    async def test_custom_write_property(
        self, store: Neo4jVectorGraphStore, mock_driver: AsyncMock
    ) -> None:
        """write_property='topic_cluster' uses that property in gds.louvain.write()."""
        store._gds_enabled = True
        store._gds_available = True

        rec = MagicMock()
        rec.__getitem__ = lambda self, k, d={"uid": "n1", "communityId": 0}: d[k]

        mock_driver.execute_query.side_effect = [
            _empty_result(),  # gds.graph.project.cypher
            ([_graph_list_record()], MagicMock(), MagicMock()),  # gds.graph.list
            _empty_result(),  # gds.louvain.write
            ([rec], MagicMock(), MagicMock()),  # gds.louvain.stream
            _empty_result(),  # _drop_projection
        ]

        await store.detect_communities(
            collection="test",
            write_back=True,
            write_property="topic_cluster",
        )

        write_call = mock_driver.execute_query.call_args_list[2]
        assert write_call.kwargs.get("write_prop") == "topic_cluster"

    @pytest.mark.asyncio
    async def test_default_write_property(
        self, store: Neo4jVectorGraphStore, mock_driver: AsyncMock
    ) -> None:
        """write_property=None defaults to 'community_id'."""
        store._gds_enabled = True
        store._gds_available = True

        rec = MagicMock()
        rec.__getitem__ = lambda self, k, d={"uid": "n1", "communityId": 0}: d[k]

        mock_driver.execute_query.side_effect = [
            _empty_result(),  # gds.graph.project.cypher
            ([_graph_list_record()], MagicMock(), MagicMock()),  # gds.graph.list
            _empty_result(),  # gds.louvain.write
            ([rec], MagicMock(), MagicMock()),  # gds.louvain.stream
            _empty_result(),  # _drop_projection
        ]

        await store.detect_communities(
            collection="test",
            write_back=True,
            write_property=None,
        )

        write_call = mock_driver.execute_query.call_args_list[2]
        assert write_call.kwargs.get("write_prop") == "community_id"
