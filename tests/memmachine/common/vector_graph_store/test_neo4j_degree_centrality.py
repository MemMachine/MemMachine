"""Unit tests for degree_centrality() method on Neo4jVectorGraphStore."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from memmachine.common.vector_graph_store.neo4j_vector_graph_store import (
    DegreeCentralityResult,
    Neo4jVectorGraphStore,
    Neo4jVectorGraphStoreParams,
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


def _rec(data: dict) -> MagicMock:
    rec = MagicMock()
    rec.__getitem__ = lambda self, k, d=data: d[k]
    return rec


def _result(records: list[dict]) -> tuple:
    return ([_rec(r) for r in records], MagicMock(), MagicMock())


def _empty() -> tuple:
    return ([], MagicMock(), MagicMock())


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestDegreeCentrality:
    # -------------------------------------------------------------------
    # 3.1 — Returns correct degree metrics
    # -------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_returns_correct_degree_metrics(
        self, store: Neo4jVectorGraphStore, mock_driver: AsyncMock
    ) -> None:
        mock_driver.execute_query.return_value = _result(
            [
                {"uid": "hub", "in_deg": 10, "out_deg": 5, "total_deg": 15},
                {"uid": "leaf", "in_deg": 1, "out_deg": 0, "total_deg": 1},
            ]
        )

        results = await store.degree_centrality(collection="test")

        assert len(results) == 2
        assert isinstance(results[0], DegreeCentralityResult)
        assert results[0].uid == "hub"
        assert results[0].in_degree == 10
        assert results[0].out_degree == 5
        assert results[0].total_degree == 15
        assert results[1].uid == "leaf"

    # -------------------------------------------------------------------
    # 3.2 — Results ordered by total_degree descending
    # -------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_results_ordered_by_total_degree(
        self, store: Neo4jVectorGraphStore, mock_driver: AsyncMock
    ) -> None:
        mock_driver.execute_query.return_value = _result(
            [
                {"uid": "top", "in_deg": 20, "out_deg": 10, "total_deg": 30},
                {"uid": "mid", "in_deg": 5, "out_deg": 5, "total_deg": 10},
                {"uid": "low", "in_deg": 1, "out_deg": 0, "total_deg": 1},
            ]
        )

        results = await store.degree_centrality(collection="test")

        degrees = [r.total_degree for r in results]
        assert degrees == [30, 10, 1]

    # -------------------------------------------------------------------
    # 3.3 — Empty collection returns empty list
    # -------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_returns_empty_for_empty_collection(
        self, store: Neo4jVectorGraphStore, mock_driver: AsyncMock
    ) -> None:
        mock_driver.execute_query.return_value = _empty()

        results = await store.degree_centrality(collection="test")

        assert results == []

    # -------------------------------------------------------------------
    # 3.4 — Relationship type filter in Cypher query
    # -------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_passes_relation_type_filter(
        self, store: Neo4jVectorGraphStore, mock_driver: AsyncMock
    ) -> None:
        mock_driver.execute_query.return_value = _empty()

        await store.degree_centrality(
            collection="test",
            relation_types=["RELATES_TO", "KNOWS"],
        )

        call_args = mock_driver.execute_query.call_args
        query_text = call_args[0][0].text
        assert "'RELATES_TO'" in query_text
        assert "'KNOWS'" in query_text
