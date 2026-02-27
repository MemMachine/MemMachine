"""
Tests for memmachine_server.server.api_v2.service.

Verifies that service-layer helpers (_search_target_memories, _list_target_memories)
correctly wire spec fields — in particular set_metadata — through to the MemMachine
core methods (query_search / list_search).
"""

from unittest.mock import AsyncMock, MagicMock

import pytest
from memmachine_common.api import MemoryType
from memmachine_common.api.spec import (
    ListMemoriesSpec,
    SearchMemoriesSpec,
)

from memmachine_server.server.api_v2.service import (
    _list_target_memories,
    _search_target_memories,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_empty_memmachine() -> AsyncMock:
    """Return a memmachine mock whose search/list methods return empty results."""
    memmachine = AsyncMock()

    empty = MagicMock()
    empty.episodic_memory = None
    empty.semantic_memory = None

    memmachine.query_search.return_value = empty
    memmachine.list_search.return_value = empty
    return memmachine


# ---------------------------------------------------------------------------
# _search_target_memories
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_search_target_memories_passes_set_metadata():
    """set_metadata from SearchMemoriesSpec is forwarded to query_search."""
    spec = SearchMemoriesSpec.model_validate(
        {
            "query": "test query",
            "set_metadata": {"user_id": "user123", "tenant": "acme"},
        }
    )
    memmachine = _make_empty_memmachine()

    await _search_target_memories(
        target_memories=[MemoryType.Semantic],
        spec=spec,
        memmachine=memmachine,
    )

    memmachine.query_search.assert_awaited_once()
    call_kwargs = memmachine.query_search.call_args[1]
    assert call_kwargs["set_metadata"] == {"user_id": "user123", "tenant": "acme"}


@pytest.mark.asyncio
async def test_search_target_memories_set_metadata_none():
    """When set_metadata is omitted, None is forwarded to query_search."""
    spec = SearchMemoriesSpec.model_validate({"query": "test query"})
    assert spec.set_metadata is None

    memmachine = _make_empty_memmachine()

    await _search_target_memories(
        target_memories=[MemoryType.Semantic],
        spec=spec,
        memmachine=memmachine,
    )

    call_kwargs = memmachine.query_search.call_args[1]
    assert call_kwargs["set_metadata"] is None


@pytest.mark.asyncio
async def test_search_target_memories_set_metadata_mixed_value_types():
    """set_metadata supports non-string JSON values (int, bool, null)."""
    spec = SearchMemoriesSpec.model_validate(
        {
            "query": "test",
            "set_metadata": {"score": 42, "active": True, "tag": None},
        }
    )
    memmachine = _make_empty_memmachine()

    await _search_target_memories(
        target_memories=[MemoryType.Semantic],
        spec=spec,
        memmachine=memmachine,
    )

    call_kwargs = memmachine.query_search.call_args[1]
    assert call_kwargs["set_metadata"] == {"score": 42, "active": True, "tag": None}


@pytest.mark.asyncio
async def test_search_target_memories_other_fields_still_passed():
    """Existing query_search fields (query, limit, etc.) are still forwarded correctly."""
    spec = SearchMemoriesSpec.model_validate(
        {
            "query": "my query",
            "top_k": 5,
            "set_metadata": {"user_id": "u1"},
        }
    )
    memmachine = _make_empty_memmachine()

    await _search_target_memories(
        target_memories=[MemoryType.Semantic],
        spec=spec,
        memmachine=memmachine,
    )

    call_kwargs = memmachine.query_search.call_args[1]
    assert call_kwargs["query"] == "my query"
    assert call_kwargs["limit"] == 5
    assert call_kwargs["set_metadata"] == {"user_id": "u1"}


# ---------------------------------------------------------------------------
# _list_target_memories
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_list_target_memories_passes_set_metadata():
    """set_metadata from ListMemoriesSpec is forwarded to list_search."""
    spec = ListMemoriesSpec.model_validate(
        {
            "set_metadata": {"user_id": "user456", "region": "us-east"},
        }
    )
    memmachine = _make_empty_memmachine()

    await _list_target_memories(
        target_memories=[MemoryType.Semantic],
        spec=spec,
        memmachine=memmachine,
    )

    memmachine.list_search.assert_awaited_once()
    call_kwargs = memmachine.list_search.call_args[1]
    assert call_kwargs["set_metadata"] == {"user_id": "user456", "region": "us-east"}


@pytest.mark.asyncio
async def test_list_target_memories_set_metadata_none():
    """When set_metadata is omitted, None is forwarded to list_search."""
    spec = ListMemoriesSpec.model_validate({})
    assert spec.set_metadata is None

    memmachine = _make_empty_memmachine()

    await _list_target_memories(
        target_memories=[MemoryType.Semantic],
        spec=spec,
        memmachine=memmachine,
    )

    call_kwargs = memmachine.list_search.call_args[1]
    assert call_kwargs["set_metadata"] is None


@pytest.mark.asyncio
async def test_list_target_memories_set_metadata_mixed_value_types():
    """set_metadata in list supports non-string JSON values (int, bool, null)."""
    spec = ListMemoriesSpec.model_validate(
        {
            "set_metadata": {"priority": 1, "verified": False, "label": None},
        }
    )
    memmachine = _make_empty_memmachine()

    await _list_target_memories(
        target_memories=[MemoryType.Semantic],
        spec=spec,
        memmachine=memmachine,
    )

    call_kwargs = memmachine.list_search.call_args[1]
    assert call_kwargs["set_metadata"] == {
        "priority": 1,
        "verified": False,
        "label": None,
    }


@pytest.mark.asyncio
async def test_list_target_memories_other_fields_still_passed():
    """Existing list_search fields (page_size, page_num, etc.) are still forwarded."""
    spec = ListMemoriesSpec.model_validate(
        {
            "page_size": 25,
            "page_num": 2,
            "set_metadata": {"user_id": "u2"},
        }
    )
    memmachine = _make_empty_memmachine()

    await _list_target_memories(
        target_memories=[MemoryType.Semantic],
        spec=spec,
        memmachine=memmachine,
    )

    call_kwargs = memmachine.list_search.call_args[1]
    assert call_kwargs["page_size"] == 25
    assert call_kwargs["page_num"] == 2
    assert call_kwargs["set_metadata"] == {"user_id": "u2"}
