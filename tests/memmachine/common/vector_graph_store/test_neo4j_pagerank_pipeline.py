"""Unit tests for the background PageRank pipeline.

Tests the _maybe_trigger_pagerank() and _run_pagerank_for_collection() methods
on Neo4jVectorGraphStore using mocked Neo4j drivers so they run without Docker.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

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


def _make_store(
    mock_driver: AsyncMock,
    *,
    pagerank_auto_enabled: bool = True,
    pagerank_trigger_threshold: int = 5,
    gds_enabled: bool = True,
) -> Neo4jVectorGraphStore:
    """Create a Neo4jVectorGraphStore with configurable PageRank settings."""
    return Neo4jVectorGraphStore(
        Neo4jVectorGraphStoreParams(
            driver=mock_driver,
            force_exact_similarity_search=True,
            dedup_trigger_threshold=0,  # Disable auto-dedup
            pagerank_auto_enabled=pagerank_auto_enabled,
            pagerank_trigger_threshold=pagerank_trigger_threshold,
            gds_enabled=gds_enabled,
        )
    )


@pytest.fixture
def store(mock_driver: AsyncMock) -> Neo4jVectorGraphStore:
    """Store with PageRank enabled, GDS enabled, threshold=5."""
    return _make_store(mock_driver)


# ---------------------------------------------------------------------------
# 2.5 - Trigger fires when all conditions met
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_maybe_trigger_pagerank_schedules_task_when_conditions_met(
    mock_driver: AsyncMock,
) -> None:
    """When all conditions are met, a background task should be scheduled."""
    store = _make_store(mock_driver, pagerank_trigger_threshold=5)
    collection = "Derivative_test"
    store._collection_node_counts[collection] = 5

    with patch.object(store, "_track_task") as track_mock:
        store._maybe_trigger_pagerank(collection)

    track_mock.assert_called_once()
    assert collection in store._pagerank_pending_collections

    # Clean up the created task to avoid "task was destroyed" warnings.
    task_arg = track_mock.call_args[0][0]
    task_arg.cancel()


# ---------------------------------------------------------------------------
# 2.6 - No task when pagerank_auto_enabled=False
# ---------------------------------------------------------------------------


def test_maybe_trigger_pagerank_skipped_when_disabled(
    mock_driver: AsyncMock,
) -> None:
    """When pagerank_auto_enabled=False, no task should be scheduled."""
    store = _make_store(
        mock_driver, pagerank_auto_enabled=False, pagerank_trigger_threshold=5
    )
    collection = "Derivative_test"
    store._collection_node_counts[collection] = 100

    with patch.object(store, "_track_task") as track_mock:
        store._maybe_trigger_pagerank(collection)

    track_mock.assert_not_called()
    assert collection not in store._pagerank_pending_collections


# ---------------------------------------------------------------------------
# 2.7 - No task when GDS unavailable
# ---------------------------------------------------------------------------


def test_maybe_trigger_pagerank_skipped_when_gds_unavailable(
    mock_driver: AsyncMock,
) -> None:
    """When gds_enabled=False, no task should be scheduled."""
    store = _make_store(mock_driver, gds_enabled=False, pagerank_trigger_threshold=5)
    collection = "Derivative_test"
    store._collection_node_counts[collection] = 100

    with patch.object(store, "_track_task") as track_mock:
        store._maybe_trigger_pagerank(collection)

    track_mock.assert_not_called()
    assert collection not in store._pagerank_pending_collections


# ---------------------------------------------------------------------------
# 2.8 - No task for non-Derivative collections
# ---------------------------------------------------------------------------


def test_maybe_trigger_pagerank_skipped_for_non_derivative_collection(
    mock_driver: AsyncMock,
) -> None:
    """Only Derivative_ collections should trigger PageRank."""
    store = _make_store(mock_driver, pagerank_trigger_threshold=5)
    collection = "Entity"
    store._collection_node_counts[collection] = 100

    with patch.object(store, "_track_task") as track_mock:
        store._maybe_trigger_pagerank(collection)

    track_mock.assert_not_called()
    assert collection not in store._pagerank_pending_collections


# ---------------------------------------------------------------------------
# 2.9 - No task below threshold
# ---------------------------------------------------------------------------


def test_maybe_trigger_pagerank_skipped_below_threshold(
    mock_driver: AsyncMock,
) -> None:
    """When node count is below threshold, no task should be scheduled."""
    store = _make_store(mock_driver, pagerank_trigger_threshold=10)
    collection = "Derivative_test"
    store._collection_node_counts[collection] = 9  # Below threshold of 10

    with patch.object(store, "_track_task") as track_mock:
        store._maybe_trigger_pagerank(collection)

    track_mock.assert_not_called()
    assert collection not in store._pagerank_pending_collections


# ---------------------------------------------------------------------------
# 2.10 - No duplicate task when collection already pending
# ---------------------------------------------------------------------------


def test_maybe_trigger_pagerank_skipped_when_already_pending(
    mock_driver: AsyncMock,
) -> None:
    """When a collection is already in the pending set, don't schedule again."""
    store = _make_store(mock_driver, pagerank_trigger_threshold=5)
    collection = "Derivative_test"
    store._collection_node_counts[collection] = 100
    store._pagerank_pending_collections.add(collection)

    with patch.object(store, "_track_task") as track_mock:
        store._maybe_trigger_pagerank(collection)

    track_mock.assert_not_called()
    # Should still be pending (was already there).
    assert collection in store._pagerank_pending_collections


# ---------------------------------------------------------------------------
# 3.2 - Dedup re-triggers PageRank
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_dedup_retriggers_pagerank(
    mock_driver: AsyncMock,
) -> None:
    """After dedup completes, PageRank should be re-triggered for derivative
    collections that meet the threshold."""
    store = _make_store(mock_driver, pagerank_trigger_threshold=5, gds_enabled=True)
    collection = "Derivative_test"
    store._collection_node_counts[collection] = 50

    # Pre-add collection to pagerank pending so it can be discarded and
    # re-triggered.
    store._pagerank_pending_collections.add(collection)
    store._dedup_pending_collections.add(collection)

    # Mock _detect_duplicates to return no proposals (dedup succeeds
    # but finds nothing).
    with patch.object(
        store, "_detect_duplicates", new_callable=AsyncMock, return_value=[]
    ):
        await store._run_dedup_for_collection(collection)

    # After dedup, collection should have been removed from pending and
    # re-added via _maybe_trigger_pagerank().
    assert collection in store._pagerank_pending_collections

    # Clean up any created background tasks.
    for task in store._background_tasks:
        task.cancel()
    if store._background_tasks:
        await asyncio.gather(*store._background_tasks, return_exceptions=True)


# ---------------------------------------------------------------------------
# 4.1 - Exception handling in _run_pagerank_for_collection
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_run_pagerank_logs_warning_on_failure(
    mock_driver: AsyncMock,
) -> None:
    """When compute_pagerank() raises, the method should log a warning and
    remove the collection from the pending set."""
    store = _make_store(mock_driver, pagerank_trigger_threshold=5)
    collection = "Derivative_test"
    store._pagerank_pending_collections.add(collection)

    with (
        patch.object(
            store,
            "compute_pagerank",
            new_callable=AsyncMock,
            side_effect=RuntimeError("GDS unavailable"),
        ) as compute_mock,
        patch(
            "memmachine.common.vector_graph_store.neo4j_vector_graph_store.logger"
        ) as logger_mock,
    ):
        await store._run_pagerank_for_collection(collection)

    compute_mock.assert_awaited_once()
    logger_mock.warning.assert_called_once()
    assert "Derivative_test" in logger_mock.warning.call_args[0][1]
    # Collection should be cleaned up from pending set.
    assert collection not in store._pagerank_pending_collections


# ---------------------------------------------------------------------------
# 4.2 - Application continues after PageRank failure
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_app_continues_after_pagerank_failure(
    mock_driver: AsyncMock,
) -> None:
    """After a PageRank failure, the store should remain usable — subsequent
    _maybe_trigger_pagerank calls should work."""
    store = _make_store(mock_driver, pagerank_trigger_threshold=5)
    collection = "Derivative_test"
    store._pagerank_pending_collections.add(collection)

    # Simulate failure.
    with patch.object(
        store,
        "compute_pagerank",
        new_callable=AsyncMock,
        side_effect=RuntimeError("boom"),
    ):
        await store._run_pagerank_for_collection(collection)

    # Collection cleaned from pending.
    assert collection not in store._pagerank_pending_collections

    # Store should be able to re-trigger.
    store._collection_node_counts[collection] = 100

    with patch.object(store, "_track_task") as track_mock:
        store._maybe_trigger_pagerank(collection)

    track_mock.assert_called_once()
    assert collection in store._pagerank_pending_collections

    # Clean up.
    task_arg = track_mock.call_args[0][0]
    task_arg.cancel()


# ---------------------------------------------------------------------------
# Successful run cleans up pending set
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_run_pagerank_cleans_up_pending_on_success(
    mock_driver: AsyncMock,
) -> None:
    """After a successful PageRank run, the collection should be removed from
    the pending set."""
    store = _make_store(mock_driver, pagerank_trigger_threshold=5)
    collection = "Derivative_test"
    store._pagerank_pending_collections.add(collection)

    with patch.object(
        store, "compute_pagerank", new_callable=AsyncMock
    ) as compute_mock:
        await store._run_pagerank_for_collection(collection)

    compute_mock.assert_awaited_once_with(
        collection=collection,
        damping_factor=0.85,
        max_iterations=20,
        write_back=True,
    )
    assert collection not in store._pagerank_pending_collections


# ---------------------------------------------------------------------------
# Multiple collections tracked independently
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_maybe_trigger_pagerank_independent_collections(
    mock_driver: AsyncMock,
) -> None:
    """Different derivative collections should be tracked independently."""
    store = _make_store(mock_driver, pagerank_trigger_threshold=5)
    col_a = "Derivative_a"
    col_b = "Derivative_b"
    col_c = "Derivative_c"

    store._collection_node_counts[col_a] = 10
    store._collection_node_counts[col_b] = 3  # Below threshold
    store._collection_node_counts[col_c] = 10

    with patch.object(store, "_track_task") as track_mock:
        store._maybe_trigger_pagerank(col_a)
        store._maybe_trigger_pagerank(col_b)
        store._maybe_trigger_pagerank(col_c)

    assert track_mock.call_count == 2
    assert col_a in store._pagerank_pending_collections
    assert col_b not in store._pagerank_pending_collections
    assert col_c in store._pagerank_pending_collections

    # Clean up created tasks.
    for call in track_mock.call_args_list:
        call[0][0].cancel()
