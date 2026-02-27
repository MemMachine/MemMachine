"""Unit tests for graph_stats() method on Neo4jVectorGraphStore."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from memmachine.common.vector_graph_store.neo4j_vector_graph_store import (
    GraphStatsResult,
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
# 3.1 — node_count and edge_count
# ---------------------------------------------------------------------------


class TestGraphStats:
    @pytest.mark.asyncio
    async def test_returns_correct_counts(
        self, store: Neo4jVectorGraphStore, mock_driver: AsyncMock
    ) -> None:
        mock_driver.execute_query.side_effect = [
            _result([{"node_count": 100, "edge_count": 250}]),  # counts
            _result(
                [
                    {"rel_type": "RELATES_TO", "cnt": 200},
                    {"rel_type": "EXTRACTED_FROM", "cnt": 50},
                ]
            ),  # rel dist
            _result(
                [
                    {"label": "ENTITY_TYPE_Person", "cnt": 60},
                    {"label": "ENTITY_TYPE_Organization", "cnt": 40},
                ]
            ),  # entity dist
        ]

        result = await store.graph_stats(collection="test")

        assert isinstance(result, GraphStatsResult)
        assert result.node_count == 100
        assert result.edge_count == 250

    # -------------------------------------------------------------------
    # 3.2 — relationship type distribution
    # -------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_returns_relationship_type_distribution(
        self, store: Neo4jVectorGraphStore, mock_driver: AsyncMock
    ) -> None:
        mock_driver.execute_query.side_effect = [
            _result([{"node_count": 10, "edge_count": 20}]),
            _result(
                [
                    {"rel_type": "RELATES_TO", "cnt": 15},
                    {"rel_type": "BELONGS_TO", "cnt": 5},
                ]
            ),
            _empty(),  # no entity types
        ]

        result = await store.graph_stats(collection="test")

        assert result.relationship_type_distribution == {
            "RELATES_TO": 15,
            "BELONGS_TO": 5,
        }

    # -------------------------------------------------------------------
    # 3.3 — entity type distribution (desanitized)
    # -------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_returns_desanitized_entity_type_distribution(
        self, store: Neo4jVectorGraphStore, mock_driver: AsyncMock
    ) -> None:
        mock_driver.execute_query.side_effect = [
            _result([{"node_count": 10, "edge_count": 5}]),
            _empty(),  # no relationships
            _result(
                [
                    {"label": "ENTITY_TYPE_Person", "cnt": 7},
                    {"label": "ENTITY_TYPE_Organization", "cnt": 3},
                ]
            ),
        ]

        result = await store.graph_stats(collection="test")

        assert result.entity_type_distribution == {
            "Person": 7,
            "Organization": 3,
        }

    # -------------------------------------------------------------------
    # 3.4 — avg_degree = 0.0 when empty
    # -------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_returns_zero_avg_degree_when_empty(
        self, store: Neo4jVectorGraphStore, mock_driver: AsyncMock
    ) -> None:
        mock_driver.execute_query.side_effect = [
            _result([{"node_count": 0, "edge_count": 0}]),
            _empty(),
            _empty(),
        ]

        result = await store.graph_stats(collection="test")

        assert result.node_count == 0
        assert result.edge_count == 0
        assert result.avg_degree == 0.0
        assert result.relationship_type_distribution == {}
        assert result.entity_type_distribution == {}

    @pytest.mark.asyncio
    async def test_avg_degree_computed_correctly(
        self, store: Neo4jVectorGraphStore, mock_driver: AsyncMock
    ) -> None:
        mock_driver.execute_query.side_effect = [
            _result([{"node_count": 50, "edge_count": 125}]),
            _empty(),
            _empty(),
        ]

        result = await store.graph_stats(collection="test")

        assert result.avg_degree == pytest.approx(2.5)
