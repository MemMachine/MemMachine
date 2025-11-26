from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

import numpy as np
import pytest

from memmachine import MemMachine
from memmachine.common.episode_store import EpisodeEntry
from memmachine.main.memmachine import MemoryType


@pytest.mark.asyncio
@pytest.mark.integration
async def test_memmachine_get_empty(memmachine: MemMachine, session_data):
    res = await memmachine.list_search(session_data=session_data)

    assert res.semantic_memory == []
    assert res.episodic_memory == []


@pytest.mark.asyncio
@pytest.mark.integration
async def test_memmachine_list_search_paginates_episodic(
    memmachine: MemMachine,
    session_data,
):
    base_time = datetime.now(tz=UTC)
    episodes = [
        EpisodeEntry(
            content=f"episode-{idx}",
            producer_id="producer",
            producer_role="user",
            created_at=base_time + timedelta(minutes=idx),
        )
        for idx in range(5)
    ]

    episode_ids = await memmachine.add_episodes(session_data, episodes)

    try:
        first_page = await memmachine.list_search(
            session_data=session_data,
            target_memories=[MemoryType.Episodic],
            page_size=2,
            page_num=0,
        )
        second_page = await memmachine.list_search(
            session_data=session_data,
            target_memories=[MemoryType.Episodic],
            page_size=2,
            page_num=1,
        )
        final_page = await memmachine.list_search(
            session_data=session_data,
            target_memories=[MemoryType.Episodic],
            page_size=2,
            page_num=2,
        )

        assert [episode.content for episode in first_page.episodic_memory] == [
            "episode-0",
            "episode-1",
        ]
        assert [episode.content for episode in second_page.episodic_memory] == [
            "episode-2",
            "episode-3",
        ]
        assert [episode.content for episode in final_page.episodic_memory] == [
            "episode-4",
        ]
    finally:
        episode_storage = await memmachine._resources.get_episode_storage()
        await episode_storage.delete_episodes(episode_ids)


@dataclass
class _TempSession:
    user_profile_id: str | None
    session_id: str | None
    role_profile_id: str | None
    session_key: str


@pytest.mark.asyncio
@pytest.mark.integration
async def test_memmachine_list_search_paginates_semantic(memmachine: MemMachine):
    session_info = _TempSession(
        user_profile_id="pagination-user",
        session_id="pagination-session",
        role_profile_id=None,
        session_key="pagination-session",
    )
    await memmachine.create_session(session_info.session_key)

    semantic_service = await memmachine._resources.get_semantic_service()
    semantic_storage = semantic_service._semantic_storage

    user_set_id = f"mem_user_{session_info.user_profile_id}"
    feature_ids = [
        await semantic_storage.add_feature(
            set_id=user_set_id,
            category_name="profile",
            feature="topic",
            value=f"semantic-{idx}",
            tag="facts",
            embedding=np.array([float(idx), 1.0], dtype=float),
        )
        for idx in range(5)
    ]

    try:
        first_page = await memmachine.list_search(
            session_data=session_info,
            target_memories=[MemoryType.Semantic],
            page_size=2,
            page_num=0,
        )
        second_page = await memmachine.list_search(
            session_data=session_info,
            target_memories=[MemoryType.Semantic],
            page_size=2,
            page_num=1,
        )
        final_page = await memmachine.list_search(
            session_data=session_info,
            target_memories=[MemoryType.Semantic],
            page_size=2,
            page_num=2,
        )

        assert [feature.value for feature in first_page.semantic_memory] == [
            "semantic-0",
            "semantic-1",
        ]
        assert [feature.value for feature in second_page.semantic_memory] == [
            "semantic-2",
            "semantic-3",
        ]
        assert [feature.value for feature in final_page.semantic_memory] == [
            "semantic-4",
        ]
    finally:
        await semantic_storage.delete_features(feature_ids)
        await memmachine.delete_session(session_info)
