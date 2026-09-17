"""Unit tests for `LongTermMemory.create_fts_index` and the EpisodicMemory proxy.

`create_fts_index` is the explicit, user-triggered command that creates the
Neo4j Full-Text Search index on demand (e.g. for a project whose FTS was
disabled at creation time, or an existing Neo4j-backed project that should gain
keyword search without re-ingesting data). Unlike `initialize_fts` (a silent
no-op on unsupported backends), this command reports whether the backend
supports FTS by returning a `(status, index_name)` tuple.

These tests do NOT require a live Neo4j instance. The `VectorGraphStore` is
mocked, so we verify only the dispatch logic — whether the FTS index-creation
call is made or skipped — not the real Cypher.

Cases (LongTermMemory.create_fts_index):
- declarative + Neo4j store -> ("created", fts_<collection>_content)
- declarative + non-Neo4j (e.g. Nebula) store -> ("unsupported", None)
- event backend -> ("unsupported", None)

Cases (EpisodicMemory.create_fts_index proxy):
- long_term_memory is None -> ("unsupported", None)
- long_term_memory present -> delegates through
"""

from unittest.mock import AsyncMock, MagicMock

import pytest

from memmachine_server.common.vector_graph_store.neo4j_vector_graph_store import (
    Neo4jVectorGraphStore,
)
from memmachine_server.episodic_memory.episodic_memory import EpisodicMemory
from memmachine_server.episodic_memory.long_term_memory import LongTermMemory


def _declarative_ltm(*, vector_graph_store) -> LongTermMemory:
    """Build a declarative-backed LongTermMemory without running __init__.

    `create_fts_index` only touches `self._backend`, `self._declarative_memory`
    (and the declarative memory's `_vector_graph_store` /
    `_derivative_collection`), so we set those attributes directly instead of
    constructing the full dependency graph.
    """
    ltm = LongTermMemory.__new__(LongTermMemory)
    ltm._backend = "declarative"

    declarative_memory = MagicMock()
    declarative_memory._vector_graph_store = vector_graph_store
    declarative_memory._derivative_collection = "Derivative_test-session"
    ltm._declarative_memory = declarative_memory
    return ltm


@pytest.mark.asyncio
async def test_create_fts_index_creates_index_on_neo4j():
    """On a Neo4j store, the index is created and its name returned.

    `create_fts_index` runs regardless of `fts_enabled` (it is the explicit
    user command), so unlike `initialize_fts` it does not consult
    `self._fts_enabled`.
    """
    store = MagicMock(spec=Neo4jVectorGraphStore)
    store._sanitize_name = MagicMock(return_value="Derivative_test_session")
    store._create_fts_index_if_not_exists = AsyncMock()

    ltm = _declarative_ltm(vector_graph_store=store)

    status, index_name = await ltm.create_fts_index()

    store._create_fts_index_if_not_exists.assert_awaited_once_with(
        sanitized_collection="Derivative_test_session",
    )
    assert status == "created"
    assert index_name == "fts_Derivative_test_session_content"


@pytest.mark.asyncio
async def test_create_fts_index_unsupported_on_non_neo4j_store():
    """A non-Neo4j store (e.g. Nebula) returns ``("unsupported", None)``.

    Unlike `initialize_fts` (which skips silently), the explicit command
    reports the unsupported backend so the API layer can surface a
    consistent response without try/except branching.
    """
    # A plain MagicMock is not a Neo4jVectorGraphStore, so the isinstance
    # guard treats it as an unsupported backend.
    store = MagicMock()
    store._create_fts_index_if_not_exists = AsyncMock()

    ltm = _declarative_ltm(vector_graph_store=store)

    status, index_name = await ltm.create_fts_index()

    store._create_fts_index_if_not_exists.assert_not_awaited()
    assert status == "unsupported"
    assert index_name is None


@pytest.mark.asyncio
async def test_create_fts_index_unsupported_on_event_backend():
    """The event backend has no FTS, so the command reports ``unsupported``."""
    ltm = LongTermMemory.__new__(LongTermMemory)
    ltm._backend = "event"
    ltm._declarative_memory = None  # event backend has no declarative memory

    status, index_name = await ltm.create_fts_index()

    assert status == "unsupported"
    assert index_name is None


@pytest.mark.asyncio
async def test_episodic_memory_proxy_returns_unsupported_when_ltm_is_none():
    """The EpisodicMemory proxy returns unsupported when LTM is not wired.

    This covers the path where `EpisodicMemory.create_fts_index` short-
    circuits before delegating to `LongTermMemory.create_fts_index`.
    """
    episodic_memory = EpisodicMemory.__new__(EpisodicMemory)
    episodic_memory._long_term_memory = None

    status, index_name = await episodic_memory.create_fts_index()

    assert status == "unsupported"
    assert index_name is None


@pytest.mark.asyncio
async def test_episodic_memory_proxy_delegates_to_ltm():
    """The EpisodicMemory proxy delegates to its LongTermMemory when present."""
    ltm = MagicMock()
    ltm.create_fts_index = AsyncMock(return_value=("created", "fts_xyz_content"))

    episodic_memory = EpisodicMemory.__new__(EpisodicMemory)
    episodic_memory._long_term_memory = ltm

    status, index_name = await episodic_memory.create_fts_index()

    ltm.create_fts_index.assert_awaited_once()
    assert status == "created"
    assert index_name == "fts_xyz_content"
