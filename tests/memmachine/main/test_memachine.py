from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from uuid import uuid4

import numpy as np
import pytest

from memmachine import MemMachine
from memmachine.common.episode_store import EpisodeEntry
from memmachine.common.filter.filter_parser import parse_filter
from memmachine.main.memmachine import MemoryType

# TODO (@o-love): Blanket mark all tests in this file as integration tests for now
pytestmark = pytest.mark.integration


@pytest.mark.asyncio
async def test_memmachine_get_empty(memmachine: MemMachine, session_data):
    res = await memmachine.list_search(session_data=session_data)

    assert res.semantic_memory == []
    assert res.episodic_memory == []


@pytest.mark.asyncio
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


@pytest.mark.asyncio
async def test_memmachine_create_get_and_delete_session(memmachine: MemMachine):
    session_key = f"session-{uuid4()}"
    delete_handle = _TempSession(
        user_profile_id=session_key,
        session_id=session_key,
        role_profile_id=None,
        session_key=session_key,
    )
    deleted = False

    try:
        session_info = await memmachine.create_session(
            session_key,
            description="integration-session",
        )

        assert session_info.description == "integration-session"
        assert session_info.episode_memory_conf.session_key == session_key

        fetched = await memmachine.get_session(session_key)
        assert fetched is not None
        assert fetched.description == "integration-session"

        await memmachine.delete_session(delete_handle)
        deleted = True
        assert await memmachine.get_session(session_key) is None
    finally:
        if not deleted:
            remaining = await memmachine.get_session(session_key)
            if remaining is not None:
                await memmachine.delete_session(delete_handle)


@pytest.mark.asyncio
async def test_memmachine_search_sessions_filters_metadata(memmachine: MemMachine):
    session_manager = await memmachine._resources.get_session_data_manager()
    created_sessions: list[str] = []

    try:
        for topic in ("alpha", "beta"):
            new_session_key = f"metadata-session-{uuid4()}"
            created_sessions.append(new_session_key)
            await session_manager.create_new_session(
                session_key=new_session_key,
                configuration={"scope": "integration"},
                param=memmachine._with_default_episodic_memory_conf(
                    session_key=new_session_key
                ),
                description=f"session-{topic}",
                metadata={"topic": topic},
            )

        all_sessions = await memmachine.search_sessions()
        assert created_sessions[0] in all_sessions
        assert created_sessions[1] in all_sessions

        filter_expr = parse_filter("topic = 'alpha'")
        filtered = await memmachine.search_sessions(search_filter=filter_expr)
        assert set(filtered) == {created_sessions[0]}
    finally:
        for key in created_sessions:
            cleanup_session = _TempSession(
                user_profile_id=key,
                session_id=key,
                role_profile_id=None,
                session_key=key,
            )
            remaining = await memmachine.get_session(key)
            if remaining is not None:
                await memmachine.delete_session(cleanup_session)


@pytest.mark.asyncio
async def test_memmachine_count_episodes_totals_all(
    memmachine: MemMachine,
    session_data,
):
    entries = [
        EpisodeEntry(
            content="count-1",
            producer_id="user",
            producer_role="assistant",
        ),
        EpisodeEntry(
            content="count-2",
            producer_id="user",
            producer_role="assistant",
        ),
    ]
    episode_ids = await memmachine.add_episodes(
        session_data, entries, target_memories=[]
    )

    try:
        total = await memmachine.episodes_count(session_data=session_data)
        assert total == len(entries)
    finally:
        await memmachine.delete_episodes(episode_ids, session_data=session_data)


@pytest.mark.asyncio
async def test_memmachine_list_search_filters_metadata(
    memmachine: MemMachine,
    session_data,
):
    episode_ids = await memmachine.add_episodes(
        session_data,
        [
            EpisodeEntry(
                content="hello there",
                producer_id="user",
                producer_role="assistant",
                metadata={"topic": "greeting"},
            ),
            EpisodeEntry(
                content="status update",
                producer_id="user",
                producer_role="assistant",
                metadata={"topic": "status"},
            ),
        ],
        target_memories=[],
    )

    try:
        filtered = await memmachine.list_search(
            session_data=session_data,
            target_memories=[MemoryType.Episodic],
            search_filter="metadata.topic = 'greeting'",
        )

        assert filtered.episodic_memory is not None
        assert [episode.content for episode in filtered.episodic_memory] == [
            "hello there"
        ]
    finally:
        await memmachine.delete_episodes(episode_ids, session_data=session_data)


@pytest.mark.asyncio
async def test_memmachine_count_episodes_respects_filters(
    memmachine: MemMachine,
    session_data,
):
    entries = [
        EpisodeEntry(
            content="alpha-1",
            producer_id="user",
            producer_role="assistant",
            metadata={"topic": "alpha"},
        ),
        EpisodeEntry(
            content="beta-1",
            producer_id="user",
            producer_role="assistant",
            metadata={"topic": "beta"},
        ),
        EpisodeEntry(
            content="alpha-2",
            producer_id="user",
            producer_role="assistant",
            metadata={"topic": "alpha"},
        ),
    ]
    episode_ids = await memmachine.add_episodes(
        session_data, entries, target_memories=[]
    )

    try:
        filtered = await memmachine.episodes_count(
            session_data=session_data,
            search_filter="metadata.topic = 'alpha'",
        )
        total = await memmachine.episodes_count(session_data=session_data)

        assert filtered == 2
        assert total == len(entries)
    finally:
        await memmachine.delete_episodes(episode_ids, session_data=session_data)


@pytest.mark.asyncio
async def test_memmachine_delete_episodes_removes_history(
    memmachine: MemMachine,
    session_data,
):
    episode_ids = await memmachine.add_episodes(
        session_data,
        [
            EpisodeEntry(
                content="first",
                producer_id="user",
                producer_role="assistant",
            ),
            EpisodeEntry(
                content="second",
                producer_id="user",
                producer_role="assistant",
            ),
        ],
        target_memories=[],
    )
    deleted = False

    try:
        before_delete = await memmachine.list_search(
            session_data=session_data,
            target_memories=[MemoryType.Episodic],
        )
        assert before_delete.episodic_memory is not None
        assert len(before_delete.episodic_memory) == 2

        await memmachine.delete_episodes(episode_ids, session_data=session_data)
        deleted = True

        after_delete = await memmachine.list_search(
            session_data=session_data,
            target_memories=[MemoryType.Episodic],
        )
        assert after_delete.episodic_memory == []
    finally:
        if not deleted:
            await memmachine.delete_episodes(episode_ids, session_data=session_data)


@pytest.mark.asyncio
async def test_memmachine_delete_features_removes_semantic_entries(
    memmachine: MemMachine,
    session_data,
):
    semantic_service = await memmachine._resources.get_semantic_service()
    semantic_storage = semantic_service._semantic_storage
    user_set_id = f"mem_user_{session_data.user_profile_id}"

    feature_id = await semantic_storage.add_feature(
        set_id=user_set_id,
        category_name="profile",
        feature="alias",
        value="integration alias",
        tag="facts",
        embedding=np.array([0.5, 0.5], dtype=float),
    )

    try:
        assert await semantic_storage.get_feature(feature_id) is not None
        await memmachine.delete_features([feature_id])
        assert await semantic_storage.get_feature(feature_id) is None
    finally:
        leftover = await semantic_storage.get_feature(feature_id)
        if leftover is not None:
            await memmachine.delete_features([feature_id])
