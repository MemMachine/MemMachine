"""Unit tests for `LongTermMemory.initialize_fts`.

`initialize_fts` creates the Neo4j Full-Text Search index up front at session
creation so the first ingest is immediately searchable via hybrid search.

These tests do NOT require a live Neo4j instance. The `VectorGraphStore` is
mocked, so we verify only the dispatch logic — whether the FTS index-creation
call is made or skipped — not the real Cypher.

Cases:
- declarative + fts_enabled + Neo4j store -> creates the index
- declarative + fts_enabled=False -> skips (no call)
- declarative + non-Neo4j (e.g. Nebula) store -> skips with a warning
- event backend -> skips (FTS is declarative-only)
"""

from unittest.mock import AsyncMock, MagicMock

import pytest

from memmachine_server.common.vector_graph_store.neo4j_vector_graph_store import (
    Neo4jVectorGraphStore,
)
from memmachine_server.episodic_memory.long_term_memory import LongTermMemory


def _declarative_ltm(*, fts_enabled: bool, vector_graph_store) -> LongTermMemory:
    """Build a declarative-backed LongTermMemory without running __init__.

    `initialize_fts` only touches `self._fts_enabled`, `self._backend`,
    `self._declarative_memory` (and the declarative memory's
    `_vector_graph_store` / `_derivative_collection`), so we set those
    attributes directly instead of constructing the full dependency graph.
    """
    ltm = LongTermMemory.__new__(LongTermMemory)
    ltm._backend = "declarative"
    ltm._fts_enabled = fts_enabled

    declarative_memory = MagicMock()
    declarative_memory._vector_graph_store = vector_graph_store
    declarative_memory._derivative_collection = "Derivative_test-session"
    ltm._declarative_memory = declarative_memory
    return ltm


@pytest.mark.asyncio
async def test_initialize_fts_creates_index_on_neo4j():
    """With FTS enabled and a Neo4j store, the index is created up front."""
    store = MagicMock(spec=Neo4jVectorGraphStore)
    store._sanitize_name = MagicMock(return_value="Derivative_test_session")
    store._create_fts_index_if_not_exists = AsyncMock()

    ltm = _declarative_ltm(fts_enabled=True, vector_graph_store=store)

    await ltm.initialize_fts()

    store._create_fts_index_if_not_exists.assert_awaited_once_with(
        sanitized_collection="Derivative_test_session",
    )


@pytest.mark.asyncio
async def test_initialize_fts_skips_when_disabled():
    """When `fts_enabled=False`, no FTS index is created."""
    store = MagicMock(spec=Neo4jVectorGraphStore)
    store._create_fts_index_if_not_exists = AsyncMock()

    ltm = _declarative_ltm(fts_enabled=False, vector_graph_store=store)

    await ltm.initialize_fts()

    store._create_fts_index_if_not_exists.assert_not_awaited()


@pytest.mark.asyncio
async def test_initialize_fts_skips_non_neo4j_store(caplog):
    """A non-Neo4j store (e.g. Nebula) is skipped with a warning, not an error.

    FTS initialization is opt-in and must not break session creation on
    backends without FTS, so `initialize_fts` returns silently instead of
    raising `NotImplementedError` (unlike `_search_fts`, which raises because
    the caller explicitly asked for FTS).
    """
    # A plain MagicMock is not a Neo4jVectorGraphStore, so the isinstance
    # guard treats it as an unsupported backend.
    store = MagicMock()
    store._create_fts_index_if_not_exists = AsyncMock()

    ltm = _declarative_ltm(fts_enabled=True, vector_graph_store=store)

    with caplog.at_level(
        "WARNING",
        logger="memmachine_server.episodic_memory.long_term_memory.long_term_memory",
    ):
        await ltm.initialize_fts()

    store._create_fts_index_if_not_exists.assert_not_awaited()
    assert any("not Neo4j" in record.message for record in caplog.records)


@pytest.mark.asyncio
async def test_initialize_fts_skips_on_event_backend():
    """The event backend has no FTS, so initialization is a no-op.

    The event backend sets `_fts_enabled=False` in __init__; we replicate
    that state here to confirm `initialize_fts` short-circuits before
    touching any store.
    """
    ltm = LongTermMemory.__new__(LongTermMemory)
    ltm._backend = "event"
    ltm._fts_enabled = False
    ltm._declarative_memory = None  # event backend has no declarative memory

    # Must not raise (no assert on _declarative_memory fires) and must not
    # touch any store.
    await ltm.initialize_fts()
