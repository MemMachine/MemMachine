from unittest.mock import AsyncMock, MagicMock

import pytest

from memmachine.common.api import MemoryType as MemoryTypeE
from memmachine.common.api.spec import (
    AddMemoriesSpec,
    ListMemoriesSpec,
    MemoryMessage,
    SearchMemoriesSpec,
)
from memmachine.server.api_v2.service import (
    _add_messages_to,
    _list_target_memories,
    _search_target_memories,
    _SessionData,
)


def test_session_data_profile_ids():
    session = _SessionData(org_id="org", project_id="proj")
    assert session.session_key == "org/proj"
    assert session.user_profile_id is None
    assert session.role_profile_id is None

    session = _SessionData(org_id="org", project_id="proj", user_id="user_1")
    assert session.user_profile_id == "org/proj/user_1"
    assert session.role_profile_id is None

    session = _SessionData(org_id="org", project_id="proj", user_role="admin")
    assert session.user_profile_id is None
    assert session.role_profile_id == "org/proj/admin"


@pytest.mark.asyncio
async def test_add_messages_to_forwards_user_id_and_role_per_message():
    memmachine = AsyncMock()
    memmachine.add_episodes.side_effect = [["ep-1"], ["ep-2"]]

    messages = [
        MemoryMessage(
            role="user",
            content="hello",
            metadata={"user_id": "user_1", "user_role": "admin"},
        ),
        MemoryMessage(
            role="user",
            content="world",
            metadata={"user_id": "user_2"},
        ),
    ]
    spec = AddMemoriesSpec(org_id="org", project_id="proj", messages=messages)

    results = await _add_messages_to(
        target_memories=[MemoryTypeE.Semantic],
        spec=spec,
        memmachine=memmachine,
    )

    assert [r.uid for r in results] == ["ep-1", "ep-2"]
    assert memmachine.add_episodes.await_count == 2

    first_kwargs = memmachine.add_episodes.call_args_list[0].kwargs
    assert first_kwargs["session_data"].user_id == "user_1"
    assert first_kwargs["session_data"].user_role == "admin"
    assert len(first_kwargs["episode_entries"]) == 1
    assert first_kwargs["episode_entries"][0].metadata["user_id"] == "user_1"
    assert first_kwargs["episode_entries"][0].metadata["user_role"] == "admin"

    second_kwargs = memmachine.add_episodes.call_args_list[1].kwargs
    assert second_kwargs["session_data"].user_id == "user_2"
    assert second_kwargs["session_data"].user_role is None
    assert len(second_kwargs["episode_entries"]) == 1
    assert second_kwargs["episode_entries"][0].metadata["user_id"] == "user_2"
    assert "user_role" not in second_kwargs["episode_entries"][0].metadata


@pytest.mark.asyncio
async def test_search_target_memories_forwards_user_id_and_role():
    memmachine = AsyncMock()
    memmachine.query_search.return_value = MagicMock(
        episodic_memory=None, semantic_memory=None
    )

    spec = SearchMemoriesSpec(
        org_id="org",
        project_id="proj",
        query="hello",
        user_id="user_1",
        user_role="admin",
    )

    result = await _search_target_memories(
        target_memories=[MemoryTypeE.Semantic],
        spec=spec,
        memmachine=memmachine,
    )

    memmachine.query_search.assert_awaited_once()
    call_kwargs = memmachine.query_search.call_args.kwargs
    assert call_kwargs["session_data"].session_key == "org/proj"
    assert call_kwargs["session_data"].user_id == "user_1"
    assert call_kwargs["session_data"].user_role == "admin"
    assert call_kwargs["query"] == "hello"

    assert result.status == 0
    assert result.content.episodic_memory is None
    assert result.content.semantic_memory is None


@pytest.mark.asyncio
async def test_list_target_memories_forwards_user_id_and_role():
    memmachine = AsyncMock()
    memmachine.list_search.return_value = MagicMock(
        episodic_memory=None, semantic_memory=None
    )

    spec = ListMemoriesSpec(
        org_id="org",
        project_id="proj",
        user_id="user_1",
        user_role="admin",
    )

    result = await _list_target_memories(
        target_memories=[MemoryTypeE.Semantic],
        spec=spec,
        memmachine=memmachine,
    )

    memmachine.list_search.assert_awaited_once()
    call_kwargs = memmachine.list_search.call_args.kwargs
    assert call_kwargs["session_data"].session_key == "org/proj"
    assert call_kwargs["session_data"].user_id == "user_1"
    assert call_kwargs["session_data"].user_role == "admin"

    assert result.status == 0
    assert result.content.episodic_memory is None
    assert result.content.semantic_memory is None
