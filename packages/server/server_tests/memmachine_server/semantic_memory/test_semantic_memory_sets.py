"""Unit tests for how SemanticService resolves and manages the sets of a session."""

from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock
from uuid import UUID

import pytest

from memmachine_server.common.episode_store import Episode, EpisodeEntry, EpisodeStorage
from memmachine_server.common.errors import (
    InvalidSetIdConfigurationError,
    ResourceNotFoundError,
)
from memmachine_server.common.filter.filter_parser import parse_filter
from memmachine_server.semantic_memory.config_store.config_store import (
    SemanticConfigStorage,
)
from memmachine_server.semantic_memory.legacy_set_ids import legacy_set_id
from memmachine_server.semantic_memory.semantic_memory import SemanticService
from server_tests.memmachine_server.semantic_memory.semantic_test_utils import (
    SessionData,
    SpyEmbedder,
    get_set_id,
)
from server_tests.memmachine_server.semantic_memory.storage.in_memory_semantic_storage import (
    SemanticStorage,
)

pytestmark = pytest.mark.asyncio


async def _collect_async(iterator):
    return [item async for item in iterator]


async def _collect_feature_set(storage: SemanticStorage, **kwargs):
    return [item async for item in storage.get_feature_set(**kwargs)]


async def _collect_history_messages(storage: SemanticStorage, **kwargs):
    return [item async for item in storage.get_history_messages(**kwargs)]


async def test_add_message_records_history_and_uningested_counts(
    semantic_service: SemanticService,
    semantic_storage: SemanticStorage,
    episode_storage: EpisodeStorage,
    session_data,
):
    # Given a session with both session and profile identifiers
    episodes = await episode_storage.add_episodes(
        session_key="session_id",
        episodes=[
            EpisodeEntry(
                content="Alpha memory",
                producer_id="profile_id",
                producer_role="dev",
            )
        ],
    )
    await semantic_service.add_message(session_data=session_data, episodes=episodes)

    profile_id = await get_set_id(semantic_service, session_data, is_org_level=True)
    session_id = await get_set_id(semantic_service, session_data)

    profile_messages = await _collect_history_messages(
        semantic_storage,
        set_ids=[profile_id],
        is_ingested=False,
    )
    session_messages = await _collect_history_messages(
        semantic_storage,
        set_ids=[session_id],
        is_ingested=False,
    )

    episode_ids = [episode.uid for episode in episodes]

    # Then the history is recorded for both set ids and marked as uningested
    assert isinstance(episodes[0].uid, UUID)
    assert list(profile_messages) == episode_ids
    assert list(session_messages) == episode_ids
    for set_id in (profile_id, session_id):
        assert (
            await semantic_storage.get_history_messages_count(
                set_ids=[set_id], is_ingested=False
            )
            == 1
        )


async def test_add_message_orders_equal_timestamps_by_uuid(
    semantic_service: SemanticService,
    semantic_storage: SemanticStorage,
    episode_storage: EpisodeStorage,
    session_data,
):
    await semantic_service.stop()
    event_time = datetime(2025, 1, 1, tzinfo=UTC)
    episodes = await episode_storage.add_episodes(
        "session_id",
        [
            EpisodeEntry(
                uid=uid,
                content=str(position),
                producer_id="user",
                producer_role="user",
                created_at=event_time,
            )
            for position, uid in enumerate(
                [
                    "f0000000-0000-4000-8000-000000000000",
                    "00000000-0000-4000-8000-000000000000",
                ]
            )
        ],
    )
    await semantic_service.add_message(session_data=session_data, episodes=episodes)

    profile_id = await get_set_id(semantic_service, session_data, is_org_level=True)
    assert await _collect_history_messages(
        semantic_storage, set_ids=[profile_id], is_ingested=False
    ) == sorted(episode.uid for episode in episodes)


async def test_add_message_preserves_episode_time_at_storage(
    semantic_service: SemanticService,
    semantic_storage: SemanticStorage,
    session_data,
    monkeypatch: pytest.MonkeyPatch,
):
    await semantic_service.stop()
    created_at = datetime(2025, 1, 1, 12, 34, 56, 123456, tzinfo=UTC)
    episode = Episode(
        uid="f0000000-0000-4000-8000-000000000000",
        content="My favorite color is blue",
        producer_id="profile_id",
        producer_role="user",
        session_key="session_id",
        created_at=created_at,
    )
    add_history = AsyncMock(wraps=semantic_storage.add_history_to_set)
    monkeypatch.setattr(semantic_storage, "add_history_to_set", add_history)

    await semantic_service.add_message(episodes=[episode], session_data=session_data)

    assert add_history.await_count == 3
    for call in add_history.await_args_list:
        assert call.kwargs["history_id"] == episode.uid
        assert call.kwargs.get("created_at") == created_at
    assert [
        sid
        async for sid in semantic_storage.get_history_set_ids(
            older_than=created_at + timedelta(microseconds=1)
        )
    ] == []
    assert (
        len(
            [
                sid
                async for sid in semantic_storage.get_history_set_ids(
                    older_than=datetime.now(UTC)
                )
            ]
        )
        == 3
    )


async def test_search_returns_relevant_features(
    semantic_service: SemanticService,
    spy_embedder: SpyEmbedder,
    session_data,
):
    profile_id = await get_set_id(semantic_service, session_data, is_org_level=True)
    session_id = await get_set_id(semantic_service, session_data)

    await semantic_service.add_feature(
        session_data=session_data,
        set_id=profile_id,
        category_name="Profile",
        feature="alpha_fact",
        value="Alpha enjoys calm chats",
        tag="facts",
    )
    await semantic_service.add_feature(
        session_data=session_data,
        set_id=session_id,
        category_name="Profile",
        feature="beta_fact",
        value="Beta prefers debates",
        tag="facts",
    )

    # When searching with an alpha-focused query
    matches = await _collect_async(
        semantic_service.search(
            message="Why does alpha stay calm?",
            session_data=session_data,
            min_distance=0.5,
        )
    )

    # Then only the alpha feature is returned and embedder search was invoked
    assert spy_embedder.search_calls == [["Why does alpha stay calm?"]]
    assert len(matches) == 1
    assert matches[0].feature_name == "alpha_fact"
    assert matches[0].set_id in {profile_id, session_id}


async def test_add_feature_isolation(
    semantic_service: SemanticService,
    semantic_storage: SemanticStorage,
    spy_embedder: SpyEmbedder,
    session_data,
):
    set_id_a = await get_set_id(semantic_service, session_data, topic="a")
    set_id_b = await get_set_id(semantic_service, session_data, topic="b")

    # Given an add feature to set_id_a
    feature_id = await semantic_service.add_feature(
        session_data=session_data,
        set_id=set_id_a,
        category_name="Profile",
        feature="tone",
        value="Alpha casual",
        tag="writing_style",
    )

    # When retrieving features for each set id
    a_features = await _collect_feature_set(
        semantic_storage,
        filter_expr=parse_filter(f"set_id IN ('{set_id_a}')"),
    )
    b_features = await _collect_feature_set(
        semantic_storage,
        filter_expr=parse_filter(f"set_id IN ('{set_id_b}')"),
    )

    # Then only the profile receives the new feature and embeddings were generated
    assert len(a_features) == 1
    assert len(b_features) == 0

    assert feature_id == a_features[0].metadata.id
    assert a_features[0].feature_name == "tone"
    assert a_features[0].value == "Alpha casual"

    assert spy_embedder.ingest_calls == [["Alpha casual"]]


async def test_delete_feature_set_removes_filtered_entries(
    semantic_service: SemanticService,
    semantic_storage: SemanticStorage,
    session_data,
):
    # Given profile and session features with distinct tags
    set_type_id = await get_set_id(semantic_service, session_data, is_org_level=True)
    project_set_id = await get_set_id(semantic_service, session_data)

    await semantic_service.add_feature(
        session_data=session_data,
        set_id=set_type_id,
        category_name="Profile",
        feature="favorite_color",
        value="Blue",
        tag="profile_tag",
    )
    await semantic_service.add_feature(
        session_data=session_data,
        set_id=project_set_id,
        category_name="Profile",
        feature="session_note",
        value="Remember to ask about projects",
        tag="session_tag",
    )

    # When deleting only the profile-tagged features
    property_filter = parse_filter("tag IN ('profile_tag')")
    await semantic_service.delete_feature_set(
        session_data=session_data,
        property_filter=property_filter,
    )

    # Then profile features are cleared while session-scoped entries remain
    org_features = await _collect_feature_set(
        semantic_storage,
        filter_expr=parse_filter(f"set_id IN ('{set_type_id}')"),
    )
    project_features = await _collect_feature_set(
        semantic_storage,
        filter_expr=parse_filter(f"set_id IN ('{project_set_id}')"),
    )

    assert org_features == []
    assert len(project_features) == 1
    assert project_features[0].feature_name == "session_note"


async def test_delete_features_deletes_features_in_sets_the_session_addresses(
    semantic_service: SemanticService,
    session_data,
):
    # Given features in the org's sets and in the project's sets
    set_ids = [
        await get_set_id(semantic_service, session_data, is_org_level=True),
        await get_set_id(
            semantic_service, session_data, is_org_level=True, producer_id="alice"
        ),
        await get_set_id(semantic_service, session_data),
        await get_set_id(semantic_service, session_data, topic="travel"),
    ]
    feature_ids = [
        await semantic_service.add_feature(
            session_data=session_data,
            set_id=set_id,
            category_name="Profile",
            feature="note",
            value="Likes trains",
            tag="facts",
        )
        for set_id in set_ids
    ]

    # When deleting them through the session
    await semantic_service.delete_features(
        session_data=session_data,
        feature_ids=feature_ids,
    )

    # Then every one is deleted
    for feature_id in feature_ids:
        assert await semantic_service.get_feature(session_data, feature_id) is None


@pytest.mark.parametrize(
    ("org_id", "project_id"),
    [
        ("test_org", "other_proj"),
        ("test_org", "test_proj_1"),
        ("other_org", "test_proj"),
        ("other_org", None),
        ("test_org_1", None),
    ],
)
async def test_delete_features_skips_features_outside_the_sets_the_session_addresses(
    semantic_service: SemanticService,
    session_data,
    org_id: str,
    project_id: str | None,
):
    # Given a feature in the project's set and one in another project's or org's set
    other_session = SessionData(org_id=org_id, project_id=project_id or "unused")
    own_feature_id = await semantic_service.add_feature(
        session_data=session_data,
        set_id=await get_set_id(semantic_service, session_data),
        category_name="Profile",
        feature="note",
        value="Likes trains",
        tag="facts",
    )
    other_feature_id = await semantic_service.add_feature(
        session_data=other_session,
        set_id=await get_set_id(
            semantic_service,
            other_session,
            is_org_level=project_id is None,
        ),
        category_name="Profile",
        feature="note",
        value="Likes boats",
        tag="facts",
    )

    # When deleting both through the session
    await semantic_service.delete_features(
        session_data=session_data,
        feature_ids=[own_feature_id, other_feature_id],
    )

    # Then only the session's own feature is deleted
    assert await semantic_service.get_feature(session_data, own_feature_id) is None
    assert (
        await semantic_service.get_feature(other_session, other_feature_id) is not None
    )


async def test_add_message_uses_all_isolations(
    semantic_service: SemanticService,
    semantic_storage: SemanticStorage,
    session_data,
):
    history_id = UUID("550e8400-e29b-41d4-a716-446655440abc")
    created_at = datetime.now(tz=UTC)
    await semantic_service.add_message(
        session_data=session_data,
        episodes=[
            Episode(
                uid=history_id,
                content="Alpha memory",
                producer_id="profile_id",
                producer_role="dev",
                session_key="session_id",
                created_at=created_at,
            ),
        ],
    )

    set_ids = {
        await get_set_id(semantic_service, session_data),
        await get_set_id(semantic_service, session_data, is_org_level=True),
        await get_set_id(
            semantic_service, session_data, is_org_level=True, producer_id="profile_id"
        ),
    }
    assert {
        set_id async for set_id in semantic_storage.get_history_set_ids()
    } == set_ids
    for set_id in set_ids:
        assert await _collect_history_messages(semantic_storage, set_ids=[set_id]) == [
            history_id
        ]


async def test_list_set_ids(
    semantic_service: SemanticService,
    session_data,
):
    # A new project has no sets
    assert list(await semantic_service.list_set_ids(session_data=session_data)) == []

    # Once the org and project sets exist, both are listed
    await get_set_id(semantic_service, session_data, is_org_level=True)
    await get_set_id(semantic_service, session_data)
    set_ids = list(await semantic_service.list_set_ids(session_data=session_data))
    assert len(set_ids) == 2

    expanded_session = SessionData(
        org_id="test_org",
        project_id="test_proj",
    )
    set_metadata = {
        "user_id": "test_user",
    }

    # Add user level isolation
    await semantic_service.create_set_type(
        session_data=session_data,
        is_org_level=True,
        metadata_tags=["user_id"],
    )

    # Since session contains user_id its set appears on the list once it exists
    await get_set_id(
        semantic_service, expanded_session, is_org_level=True, user_id="test_user"
    )
    set_ids = list(
        await semantic_service.list_set_ids(
            session_data=expanded_session,
            set_metadata=set_metadata,
        )
    )
    assert len(set_ids) == 3

    # While if we create an isolation level that isn't covered by session_data it won't be included.
    await semantic_service.create_set_type(
        session_data=session_data,
        is_org_level=True,
        metadata_tags=["other_id"],
    )
    set_ids = list(
        await semantic_service.list_set_ids(
            session_data=expanded_session,
            set_metadata=set_metadata,
        )
    )
    assert len(set_ids) == 3


async def test_list_set_ids_returns_set_details(
    semantic_service: SemanticService,
    session_data,
):
    """Test that list_set_ids returns is_org_level, tags, name, and description."""
    expanded_session = SessionData(
        org_id="test_org",
        project_id="test_proj",
    )
    set_metadata = {
        "user_id": "test_user",
        "repo_id": "test_repo",
    }

    # Create set types with name and description
    await semantic_service.create_set_type(
        session_data=session_data,
        is_org_level=True,
        metadata_tags=["user_id"],
        name="User Settings",
        description="User-specific configuration",
    )

    await semantic_service.create_set_type(
        session_data=session_data,
        is_org_level=False,
        metadata_tags=["repo_id"],
        name="Repository Data",
        description="Repository-specific information",
    )

    await get_set_id(semantic_service, expanded_session, is_org_level=True)
    await get_set_id(semantic_service, expanded_session)
    await get_set_id(
        semantic_service, expanded_session, is_org_level=True, user_id="test_user"
    )
    await get_set_id(semantic_service, expanded_session, repo_id="test_repo")

    # Get all set IDs
    sets = list(
        await semantic_service.list_set_ids(
            session_data=expanded_session,
            set_metadata=set_metadata,
        )
    )

    # Should have 4 sets: 2 default (org + project) + 2 custom
    assert len(sets) == 4

    # Find the user set
    user_set = next((s for s in sets if "user_id" in s.tags), None)
    assert user_set is not None
    assert user_set.is_org_level is True
    assert user_set.tags == ["user_id"]
    assert user_set.name == "User Settings"
    assert user_set.description == "User-specific configuration"

    # Find the repo set
    repo_set = next((s for s in sets if "repo_id" in s.tags), None)
    assert repo_set is not None
    assert repo_set.is_org_level is False
    assert repo_set.tags == ["repo_id"]
    assert repo_set.name == "Repository Data"
    assert repo_set.description == "Repository-specific information"

    # Verify default sets have no tags
    default_sets = [s for s in sets if len(s.tags) == 0]
    assert len(default_sets) == 2

    # One should be org level, one should be project level
    org_level_default = next((s for s in default_sets if s.is_org_level), None)
    project_level_default = next((s for s in default_sets if not s.is_org_level), None)
    assert org_level_default is not None
    assert project_level_default is not None


async def test_default_sets_are_configurable(
    semantic_service: SemanticService,
    semantic_config_storage: SemanticConfigStorage,
    session_data,
):
    await get_set_id(semantic_service, session_data, is_org_level=True)
    await get_set_id(semantic_service, session_data)
    sets = list(await semantic_service.list_set_ids(session_data=session_data))

    assert len(sets) == 2
    org_set = next(s for s in sets if s.is_org_level)
    project_set = next(s for s in sets if not s.is_org_level)

    await semantic_service.configure_set(
        session_data=session_data,
        set_id=org_set.id,
        llm_name="llm-org",
    )
    await semantic_service.configure_set(
        session_data=session_data,
        set_id=project_set.id,
        llm_name="llm-project",
    )

    org_config = await semantic_config_storage.get_setid_config(set_id=org_set.id)
    project_config = await semantic_config_storage.get_setid_config(
        set_id=project_set.id
    )
    assert org_config is not None
    assert project_config is not None

    assert org_config.llm_name == "llm-org"
    assert project_config.llm_name == "llm-project"


async def test_user_default_set_requires_producer_metadata_and_is_configurable(
    semantic_service: SemanticService,
    semantic_config_storage: SemanticConfigStorage,
    session_data,
):
    await get_set_id(semantic_service, session_data, is_org_level=True)
    await get_set_id(semantic_service, session_data)
    base_sets = list(await semantic_service.list_set_ids(session_data=session_data))
    assert len(base_sets) == 2

    user_session = SessionData(
        org_id=session_data.org_id,
        project_id=session_data.project_id,
    )
    set_metadata = {"producer_id": "user-123"}
    expected_user_id = await get_set_id(
        semantic_service, user_session, is_org_level=True, producer_id="user-123"
    )

    sets_with_user = list(
        await semantic_service.list_set_ids(
            session_data=user_session,
            set_metadata=set_metadata,
        )
    )

    unique_sets = {s.id: s for s in sets_with_user}
    assert len(unique_sets) == 3

    user_set = next(s for s in unique_sets.values() if s.tags == ["producer_id"])
    assert user_set.is_org_level is True
    assert user_set.id == expected_user_id

    await semantic_service.configure_set(
        session_data=session_data,
        set_id=user_set.id,
        llm_name="llm-user",
    )

    user_config = await semantic_config_storage.get_setid_config(set_id=user_set.id)
    assert user_config is not None
    assert user_config.llm_name == "llm-user"


async def test_list_sets_deduplicates_default_overrides(
    semantic_service: SemanticService,
    session_data,
):
    base_set_ids = {
        await get_set_id(semantic_service, session_data, is_org_level=True),
        await get_set_id(semantic_service, session_data),
    }

    base_sets = list(await semantic_service.list_set_ids(session_data=session_data))
    assert {s.id for s in base_sets} == base_set_ids

    await semantic_service.create_set_type(
        session_data=session_data,
        is_org_level=True,
        metadata_tags=[],
        name="Custom Org",
        description="Overrides default org set",
    )

    await semantic_service.create_set_type(
        session_data=session_data,
        is_org_level=False,
        metadata_tags=[],
        name="Custom Project",
        description="Overrides default project set",
    )

    deduped_sets = list(await semantic_service.list_set_ids(session_data=session_data))
    assert len(deduped_sets) == 2
    assert {s.id for s in deduped_sets} == base_set_ids

    org_set = next(s for s in deduped_sets if s.is_org_level)
    project_set = next(s for s in deduped_sets if not s.is_org_level)

    assert org_set.tags == []
    assert project_set.tags == []


async def test_configure_set(
    semantic_service: SemanticService,
    semantic_config_storage: SemanticConfigStorage,
    session_data,
):
    set_id = await get_set_id(semantic_service, session_data)

    await semantic_service.configure_set(
        session_data=session_data,
        set_id=set_id,
        llm_name="test_llm",
        embedder_name="test_embedder",
    )

    config = await semantic_config_storage.get_setid_config(set_id=set_id)
    assert config is not None

    assert config.llm_name == "test_llm"
    assert config.embedder_name == "test_embedder"


async def test_invalid_embedder_change_with_dirty_set(
    semantic_service: SemanticService,
    session_data,
):
    set_id = await get_set_id(semantic_service, session_data)
    await semantic_service.configure_set(
        session_data=session_data,
        set_id=set_id,
        llm_name="test_llm",
        embedder_name="test_embedder",
    )

    await semantic_service.add_feature(
        session_data=session_data,
        set_id=set_id,
        category_name="Profile",
        feature="test_feature",
        value="test_value",
        tag="test_tag",
    )

    with pytest.raises(InvalidSetIdConfigurationError):
        await semantic_service.configure_set(
            session_data=session_data,
            set_id=set_id,
            llm_name="test_llm_2",
            embedder_name="test_embedder_2",
        )
    with pytest.raises(InvalidSetIdConfigurationError):
        await semantic_service.configure_set(
            session_data=session_data,
            set_id=set_id,
            llm_name="test_llm_2",
            embedder_name=None,
        )


async def test_category_templates(semantic_service: SemanticService, session_data):
    set_type_id = await semantic_service.create_set_type(
        session_data=session_data,
        is_org_level=False,
        metadata_tags=["repo"],
    )
    c_id = await semantic_service.add_category_template(
        session_data=session_data,
        set_type_id=set_type_id,
        category_name="test_category",
        description="test_description",
        prompt="test_prompt",
    )

    await semantic_service.add_tag(
        session_data=session_data,
        category_id=c_id,
        tag_name="test_tag",
        tag_description="test_tag_description",
    )

    categories = list(
        await semantic_service.list_category_templates(
            session_data=session_data, set_type_id=set_type_id
        )
    )
    assert len(categories) == 1

    assert categories[0].id == c_id
    assert categories[0].name == "test_category"


async def test_category_templates_are_visible_to_children(
    semantic_service: SemanticService,
    semantic_config_storage: SemanticConfigStorage,
    session_data,
):
    expanded_session = SessionData(
        org_id="test_org",
        project_id="test_proj",
    )
    set_metadata = {
        "user_id": "test_user",
    }

    set_type_id = await semantic_service.create_set_type(
        session_data=session_data,
        is_org_level=False,
        metadata_tags=["user_id"],
    )
    set_id = await semantic_service.get_set_id(
        session_data=expanded_session,
        is_org_level=False,
        set_metadata_keys=["user_id"],
        set_metadata=set_metadata,
    )

    c_id = await semantic_service.add_category_template(
        session_data=session_data,
        set_type_id=set_type_id,
        category_name="test_category",
        description="test_description",
        prompt="test_prompt",
    )

    await semantic_service.add_tag(
        session_data=session_data,
        category_id=c_id,
        tag_name="test_tag",
        tag_description="test_tag_description",
    )

    config = await semantic_config_storage.get_setid_config(set_id=set_id)
    assert config is not None

    categories = config.categories

    assert len(categories) == 1
    assert categories[0].id == c_id


async def test_search_with_time_filter_str(
    semantic_service: SemanticService,
    session_data,
):
    filter_str = "created_at < date('2026-01-19T01:56:41.513342Z')"
    _ = await _collect_async(
        semantic_service.search(
            message="Find alpha info",
            session_data=session_data,
            search_filter=parse_filter(filter_str),
        )
    )

    # Add features
    set_id = await semantic_service.get_set_id(
        session_data=session_data,
        set_metadata_keys=[],
    )

    await semantic_service.add_feature(
        session_data=session_data,
        set_id=set_id,
        category_name="Profile",
        feature="alpha_fact",
        tag="alpha_tag",
        value="alpha_value",
    )

    _ = await _collect_async(
        semantic_service.search(
            message="Find alpha info",
            session_data=session_data,
            search_filter=parse_filter(filter_str),
        )
    )


async def test_sets_whose_legacy_ids_collide_are_separate(
    semantic_service: SemanticService,
):
    # Given an org named like another org's project, whose set ids used to collide
    web = SessionData(org_id="acme", project_id="web")
    other_org = SessionData(org_id="acme_project_web", project_id="any")
    assert legacy_set_id(
        org_id="acme", project_id="web", set_metadata={"producer_id": "alice"}
    ) == legacy_set_id(
        org_id="acme_project_web",
        project_id=None,
        set_metadata={"producer_id": "alice"},
    )
    await semantic_service.create_set_type(
        session_data=web, is_org_level=False, metadata_tags=["producer_id"]
    )

    # When each gets its set for the same producer
    web_set_id = await get_set_id(semantic_service, web, producer_id="alice")
    other_set_id = await get_set_id(
        semantic_service, other_org, is_org_level=True, producer_id="alice"
    )
    await semantic_service.add_feature(
        session_data=web,
        set_id=web_set_id,
        category_name="Profile",
        feature="alpha_fact",
        value="Alpha secret",
        tag="facts",
    )

    # Then the sets differ and the other org finds none of the features
    assert web_set_id != other_set_id
    assert (
        await _collect_async(
            semantic_service.search(
                message="alpha",
                session_data=other_org,
                set_metadata={"producer_id": "alice"},
            )
        )
        == []
    )


@pytest.mark.parametrize(
    "other_session",
    [
        SessionData(org_id="test_org", project_id="other_proj"),
        SessionData(org_id="other_org", project_id="test_proj"),
    ],
    ids=["other_project", "other_org"],
)
async def test_ids_outside_the_session_behave_as_missing(
    semantic_service: SemanticService,
    semantic_config_storage: SemanticConfigStorage,
    session_data,
    other_session: SessionData,
):
    # Given a set, feature, category and tag of another project or org
    other_set_id = await get_set_id(semantic_service, other_session)
    category_id = await semantic_service.add_new_category(
        session_data=other_session,
        set_id=other_set_id,
        category_name="hobbies",
        prompt="Hobbies",
        description=None,
    )
    tag_id = await semantic_service.add_tag(
        session_data=other_session,
        category_id=category_id,
        tag_name="sport",
        tag_description="Sports",
    )
    feature_id = await semantic_service.add_feature(
        session_data=other_session,
        set_id=other_set_id,
        category_name="hobbies",
        feature="sport",
        value="Plays chess",
        tag="sport",
    )

    # Then reads through the session find nothing
    assert await semantic_service.get_feature(session_data, feature_id) is None
    assert (
        await semantic_service.get_category(
            session_data=session_data, category_id=category_id
        )
        is None
    )
    assert (
        await semantic_service.get_category_set_ids(
            session_data=session_data, category_id=category_id
        )
        == []
    )

    # And writes into them answer not found
    with pytest.raises(ResourceNotFoundError):
        await semantic_service.update_feature(session_data, feature_id, value="Chess")
    with pytest.raises(ResourceNotFoundError):
        await semantic_service.add_feature(
            session_data=session_data,
            set_id=other_set_id,
            category_name="Profile",
            feature="note",
            value="Likes trains",
            tag="facts",
        )
    with pytest.raises(ResourceNotFoundError):
        await semantic_service.configure_set(
            session_data=session_data, set_id=other_set_id, llm_name="llm"
        )
    with pytest.raises(ResourceNotFoundError):
        await semantic_service.add_new_category(
            session_data=session_data,
            set_id=other_set_id,
            category_name="other",
            prompt="Other",
            description=None,
        )
    with pytest.raises(ResourceNotFoundError):
        await semantic_service.disable_category(
            session_data=session_data, set_id=other_set_id, category_name="Profile"
        )
    with pytest.raises(ResourceNotFoundError):
        await semantic_service.add_tag(
            session_data=session_data,
            category_id=category_id,
            tag_name="music",
            tag_description="Music",
        )

    # And deletes leave them in place
    await semantic_service.delete_features(session_data, [feature_id])
    await semantic_service.delete_tag(session_data=session_data, tag_id=tag_id)
    await semantic_service.delete_category(
        session_data=session_data, category_id=category_id
    )

    assert await semantic_service.get_feature(other_session, feature_id) is not None
    assert await semantic_config_storage.get_tag(tag_id=tag_id) is not None
    assert (
        await semantic_service.get_category(
            session_data=other_session, category_id=category_id
        )
        is not None
    )
    config = await semantic_config_storage.get_setid_config(set_id=other_set_id)
    assert config.llm_name is None
    assert not config.disabled_categories
    assert [c.name for c in config.categories] == ["hobbies"]


async def test_set_types_of_another_org_behave_as_missing(
    semantic_service: SemanticService,
    session_data,
):
    other_org = SessionData(org_id="other_org", project_id=session_data.project_id)
    set_type_id = await semantic_service.create_set_type(
        session_data=other_org, metadata_tags=["repo"]
    )
    category_id = await semantic_service.add_category_template(
        session_data=other_org,
        set_type_id=set_type_id,
        category_name="code",
        prompt="Code",
        description=None,
    )

    assert (
        await semantic_service.get_category(
            session_data=session_data, category_id=category_id
        )
        is None
    )
    assert (
        await semantic_service.list_category_templates(
            session_data=session_data, set_type_id=set_type_id
        )
        == []
    )
    with pytest.raises(ResourceNotFoundError):
        await semantic_service.add_category_template(
            session_data=session_data,
            set_type_id=set_type_id,
            category_name="other",
            prompt="Other",
            description=None,
        )
    await semantic_service.delete_set_type(
        session_data=session_data, set_type_id=set_type_id
    )

    assert [
        set_type.id
        for set_type in await semantic_service.list_set_types(session_data=other_org)
    ] == [set_type_id]


async def test_org_sets_are_shared_by_the_projects_of_the_org(
    semantic_service: SemanticService,
    session_data,
):
    other_project = SessionData(org_id=session_data.org_id, project_id="other_proj")
    org_set_id = await get_set_id(semantic_service, other_project, is_org_level=True)
    feature_id = await semantic_service.add_feature(
        session_data=other_project,
        set_id=org_set_id,
        category_name="Profile",
        feature="note",
        value="Alpha note",
        tag="facts",
    )

    assert await get_set_id(semantic_service, session_data, is_org_level=True) == (
        org_set_id
    )
    assert await semantic_service.get_feature(session_data, feature_id) is not None

    await semantic_service.delete_features(session_data, [feature_id])

    assert await semantic_service.get_feature(other_project, feature_id) is None


async def test_reads_add_no_sets(
    semantic_service: SemanticService,
    semantic_config_storage: SemanticConfigStorage,
    session_data,
):
    metadata = {"producer_id": "alice"}

    await _collect_async(
        semantic_service.search(
            message="alpha", session_data=session_data, set_metadata=metadata
        )
    )
    await _collect_async(
        semantic_service.get_set_features(
            session_data=session_data, set_metadata=metadata
        )
    )
    assert (
        list(
            await semantic_service.list_set_ids(
                session_data=session_data, set_metadata=metadata
            )
        )
        == []
    )

    for project_id in (None, session_data.project_id):
        assert (
            await semantic_config_storage.find_sets(
                org_id=session_data.org_id, project_id=project_id
            )
            == []
        )
