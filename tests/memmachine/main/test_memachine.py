import asyncio
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from uuid import uuid4

import pytest
from pydantic import JsonValue

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
class _Session:
    session_key: str
    org_id: str
    project_id: str
    metadata: dict[str, JsonValue] | None


@pytest.mark.asyncio
async def test_memmachine_list_search_paginates_semantic(memmachine: MemMachine):
    session_info = _Session(
        session_key="pagination-session",
        org_id="org_pagination",
        project_id="project_pagination",
        metadata={
            "work_type": "pagination",
        },
    )
    await memmachine.create_session(session_info.session_key)
    await memmachine.create_semantic_set_type(
        session_data=session_info,
        is_org_level=False,
        metadata_tags=list(session_info.metadata.keys()),
    )

    set_id = await memmachine.semantic_get_set_id(
        session_data=session_info,
        is_org_level=False,
        metadata_tags=[],
    )

    feature_ids = [
        await memmachine.add_feature(
            set_id=set_id,
            category_name="profile",
            feature="topic",
            value=f"semantic-{idx}",
            tag="facts",
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
        await memmachine.delete_features(feature_ids)
        await memmachine.delete_session(session_info)


@pytest.mark.asyncio
async def test_memmachine_create_get_and_delete_session(memmachine: MemMachine):
    session_key = f"session-{uuid4()}"
    delete_handle = _Session(
        session_key=session_key,
        org_id=f"org-{session_key}",
        project_id=f"project-{session_key}",
        metadata={
            "work_type": "integration",
            "session": f"test-{session_key}",
        },
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
            cleanup_session = _Session(
                session_key=key,
                org_id=f"org-{key}",
                project_id=f"project-{key}",
                metadata={},
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
    set_id = await memmachine.semantic_get_set_id(
        session_data=session_data,
        is_org_level=False,
        metadata_tags=[],
    )

    feature_id = await memmachine.add_feature(
        set_id=set_id,
        category_name="profile",
        feature="alias",
        value="integration alias",
        tag="facts",
    )

    try:
        assert await memmachine.get_feature(feature_id) is not None
        await memmachine.delete_features([feature_id])
        assert await memmachine.get_feature(feature_id) is None
    finally:
        leftover = await memmachine.get_feature(feature_id)
        if leftover is not None:
            await memmachine.delete_features([feature_id])


@pytest.mark.asyncio
async def test_add_then_get_feature(memmachine: MemMachine, session_data):
    set_id = await memmachine.semantic_get_set_id(
        session_data=session_data,
        is_org_level=False,
        metadata_tags=[],
    )

    feature_id = await memmachine.add_feature(
        set_id=set_id,
        category_name="profile",
        feature="alias",
        value="integration alias",
        tag="facts",
    )

    feature = await memmachine.get_feature(feature_id)

    assert feature.value == "integration alias"


@pytest.mark.asyncio
async def test_add_update_get_feature(memmachine: MemMachine, session_data):
    set_id = await memmachine.semantic_get_set_id(
        session_data=session_data,
        is_org_level=False,
        metadata_tags=[],
    )

    feature_id = await memmachine.add_feature(
        set_id=set_id,
        category_name="profile",
        feature="alias",
        value="integration alias",
        tag="facts",
    )

    feature = await memmachine.get_feature(feature_id)
    assert feature.value == "integration alias"

    await memmachine.update_feature(
        feature_id=feature_id,
        category_name="profile",
        feature="alias",
        value="integration alias updated",
    )

    updated_feature = await memmachine.get_feature(feature_id)
    assert updated_feature.value == "integration alias updated"


@pytest.mark.asyncio
async def test_create_and_delete_semantic_set_type(
    memmachine: MemMachine, session_data
):
    session_data.metadata = {
        "work_type": "integration",
        "user_id": 123,
    }

    semantic_set_types = await memmachine.list_semantic_set_type(
        session_data=session_data
    )
    assert semantic_set_types == []

    await memmachine.create_semantic_set_type(
        session_data=session_data,
        is_org_level=False,
        metadata_tags=["user_id"],
    )

    semantic_set_types = await memmachine.list_semantic_set_type(
        session_data=session_data
    )
    assert len(semantic_set_types) == 1
    assert semantic_set_types[0].tags == ["user_id"]
    assert semantic_set_types[0].is_org_level is False

    await memmachine.delete_semantic_set_type(semantic_set_types[0].id)

    semantic_set_types = await memmachine.list_semantic_set_type(
        session_data=session_data
    )
    assert semantic_set_types == []


@pytest.mark.asyncio
async def test_custom_semantic_set_type_ingestion(memmachine: MemMachine, session_data):
    session_data.metadata = {
        "work_type": "integration",
        "user_id": "123",
    }

    await memmachine.create_semantic_set_type(
        session_data=session_data,
        is_org_level=False,
        metadata_tags=["user_id"],
    )

    set_id = await memmachine.semantic_get_set_id(
        session_data=session_data,
        is_org_level=False,
        metadata_tags=["user_id"],
    )

    category_id = await memmachine.semantic_add_category(
        set_id=set_id,
        category_name="profile",
        prompt="profile category",
        description="Category for user profile",
    )

    await memmachine.semantic_add_tag_to_category(
        category_id=category_id,
        tag_name="alias",
        tag_description="profile alias",
    )

    await memmachine.add_episodes(
        session_data,
        [
            EpisodeEntry(
                content="User alias is Tom",
                producer_id="producer",
                producer_role="user",
                metadata={"user_id": "123"},
            ),
        ],
    )

    async def is_user_id_data_ingested():
        while (  # noqa: ASYNC110
            len(
                [
                    feat
                    for feat in (
                        await memmachine.list_search(
                            session_data=session_data,
                            target_memories=[MemoryType.Semantic],
                        )
                    ).semantic_memory
                    if "user_id" in feat.set_id
                ]
            )
            == 0
        ):
            await asyncio.sleep(0.1)

    await asyncio.wait_for(is_user_id_data_ingested(), timeout=30.0)


@pytest.mark.asyncio
async def test_memmachine_semantic_add_and_get_category(
    memmachine: MemMachine, session_data
):
    session_data.metadata = {
        "work_type": "integration",
        "user_id": "123",
    }

    await memmachine.create_semantic_set_type(
        session_data=session_data,
        is_org_level=False,
        metadata_tags=["user_id"],
    )

    set_id = await memmachine.semantic_get_set_id(
        session_data=session_data,
        is_org_level=False,
        metadata_tags=["user_id"],
    )

    category_id = await memmachine.semantic_add_category(
        set_id=set_id,
        category_name="profile",
        prompt="profile category",
        description="Category for user profile",
    )

    category = await memmachine.semantic_get_category(
        category_id=category_id,
    )

    assert category is not None
    assert category.id == category_id
    assert category.name == "profile"
    assert category.prompt == "profile category"
    assert category.description == "Category for user profile"
