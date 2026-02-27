"""Unit tests for betweenness_centrality() method on Neo4jVectorGraphStore."""

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
    from neo4j import AsyncDriver

    driver = AsyncMock(spec=AsyncDriver)
    driver.execute_query = AsyncMock(return_value=([], MagicMock(), MagicMock()))
    return driver


@pytest.fixture
def store(mock_driver: AsyncMock) -> Neo4jVectorGraphStore:
    s = Neo4jVectorGraphStore(
        Neo4jVectorGraphStoreParams(
            driver=mock_driver,
            force_exact_similarity_search=True,
            dedup_trigger_threshold=0,
            gds_enabled=True,
        )
    )
    return s


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


class TestBetweennessCentrality:
    # -------------------------------------------------------------------
    # 3.1 — Returns correct (uid, score) pairs
    # -------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_returns_correct_scores(
        self, store: Neo4jVectorGraphStore, mock_driver: AsyncMock
    ) -> None:
        mock_driver.execute_query.side_effect = [
            _empty(),  # gds.version()
            _empty(),  # project.cypher
            _result([{"nodeCount": 10, "relationshipCount": 20}]),  # gds.graph.list
            _result(
                [  # gds.betweenness.stream
                    {"uid": "bridge", "score": 15.0},
                    {"uid": "leaf", "score": 0.0},
                ]
            ),
            _empty(),  # gds.graph.drop
        ]

        results = await store.betweenness_centrality(collection="test")

        assert len(results) == 2
        assert results[0] == ("bridge", 15.0)
        assert results[1] == ("leaf", 0.0)

    # -------------------------------------------------------------------
    # 3.2 — Calls write when write_back is True
    # -------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_calls_write_when_write_back(
        self, store: Neo4jVectorGraphStore, mock_driver: AsyncMock
    ) -> None:
        mock_driver.execute_query.side_effect = [
            _empty(),  # gds.version()
            _empty(),  # project.cypher
            _result([{"nodeCount": 5, "relationshipCount": 10}]),  # gds.graph.list
            _empty(),  # gds.betweenness.write
            _result([{"uid": "n1", "score": 5.0}]),  # gds.betweenness.stream
            _empty(),  # gds.graph.drop
        ]

        await store.betweenness_centrality(
            collection="test", write_back=True, write_property="my_score"
        )

        # Find the write call
        write_call = mock_driver.execute_query.call_args_list[3]
        assert "gds.betweenness.write" in write_call[0][0].text
        assert write_call[1]["write_prop"] == "my_score"

    # -------------------------------------------------------------------
    # 3.3 — Passes sampling_size to GDS
    # -------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_passes_sampling_size(
        self, store: Neo4jVectorGraphStore, mock_driver: AsyncMock
    ) -> None:
        mock_driver.execute_query.side_effect = [
            _empty(),  # gds.version()
            _empty(),  # project.cypher
            _result([{"nodeCount": 5, "relationshipCount": 10}]),  # gds.graph.list
            _result([{"uid": "n1", "score": 3.0}]),  # gds.betweenness.stream
            _empty(),  # gds.graph.drop
        ]

        await store.betweenness_centrality(collection="test", sampling_size=50)

        # Stream call should include sampling_size
        stream_call = mock_driver.execute_query.call_args_list[3]
        assert "samplingSize" in stream_call[0][0].text
        assert stream_call[1]["sampling_size"] == 50

    # -------------------------------------------------------------------
    # 3.4 — Raises RuntimeError when GDS not available
    # -------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_raises_when_gds_not_available(self, mock_driver: AsyncMock) -> None:
        store = Neo4jVectorGraphStore(
            Neo4jVectorGraphStoreParams(
                driver=mock_driver,
                force_exact_similarity_search=True,
                dedup_trigger_threshold=0,
                gds_enabled=False,
            )
        )

        with pytest.raises(RuntimeError, match="GDS"):
            await store.betweenness_centrality(collection="test")
