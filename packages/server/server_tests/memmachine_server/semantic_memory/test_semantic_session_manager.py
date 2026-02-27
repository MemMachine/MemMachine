"""Unit tests for the SemanticSessionManager using in-memory storage."""

from dataclasses import dataclass
from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock

import pytest
import pytest_asyncio

from memmachine_server.common.episode_store import Episode, EpisodeEntry, EpisodeStorage
from memmachine_server.common.errors import InvalidSetIdConfigurationError
from memmachine_server.common.filter.filter_parser import parse_filter
from memmachine_server.semantic_memory.config_store.config_store import (
    SemanticConfigStorage,
)
from memmachine_server.semantic_memory.semantic_memory import SemanticService
from memmachine_server.semantic_memory.semantic_session_manager import (
    SemanticSessionManager,
)
from server_tests.memmachine_server.semantic_memory.semantic_test_utils import (
    SpyEmbedder,
)
from server_tests.memmachine_server.semantic_memory.storage.in_memory_semantic_storage import (
    SemanticStorage,
)

pytestmark = pytest.mark.asyncio


@dataclass
class _SessionData:
    org_id: str
    project_id: str


@pytest.fixture
def session_data():
    return _SessionData(
        org_id="test_org",
        project_id="test_proj",
    )


@pytest_asyncio.fixture
async def session_manager(
    semantic_service: SemanticService,
    semantic_config_storage: SemanticConfigStorage,
) -> SemanticSessionManager:
    return SemanticSessionManager(
        semantic_service=semantic_service,
        semantic_config_storage=semantic_config_storage,
    )


@pytest.fixture
def mock_semantic_service() -> MagicMock:
    service = MagicMock(spec=SemanticService)
    service.add_message_to_sets = AsyncMock()
    service.search = AsyncMock(return_value=[])
    service.number_of_uningested = AsyncMock(return_value=0)
    service.add_new_feature = AsyncMock(return_value=101)
    service.get_feature = AsyncMock(return_value="feature")
    service.update_feature = AsyncMock()
    service.get_set_features = AsyncMock(return_value=["features"])
    return service


@pytest_asyncio.fixture
async def mock_session_manager(
    mock_semantic_service: MagicMock,
    semantic_storage: SemanticStorage,
    semantic_config_storage: SemanticConfigStorage,
) -> SemanticSessionManager:
    return SemanticSessionManager(
        semantic_service=mock_semantic_service,
        semantic_config_storage=semantic_config_storage,
    )


async def test_add_message_records_history_and_uningested_counts(
    session_manager: SemanticSessionManager,
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
    await session_manager.add_message(session_data=session_data, episodes=episodes)

    profile_id = session_manager._generate_set_id(
        org_id=session_data.org_id,
        metadata={},
    )
    session_id = session_manager._generate_set_id(
        org_id=session_data.org_id,
        project_id=session_data.project_id,
        metadata={},
    )

    profile_messages = await semantic_storage.get_history_messages(
        set_ids=[profile_id],
        is_ingested=False,
    )
    session_messages = await semantic_storage.get_history_messages(
        set_ids=[session_id],
        is_ingested=False,
    )

    episode_ids = [episode.uid for episode in episodes]

    # Then the history is recorded for both set ids and marked as uningested
    assert len(episodes[0].uid) > 0
    assert list(profile_messages) == episode_ids
    assert list(session_messages) == episode_ids
    assert await semantic_service.number_of_uningested([profile_id]) == 1
    assert await semantic_service.number_of_uningested([session_id]) == 1


async def test_search_returns_relevant_features(
    session_manager: SemanticSessionManager,
    semantic_service: SemanticService,
    spy_embedder: SpyEmbedder,
    session_data,
):
    profile_id = session_manager._generate_set_id(
        org_id=session_data.org_id,
        metadata={},
    )
    session_id = session_manager._generate_set_id(
        org_id=session_data.org_id,
        project_id=session_data.project_id,
        metadata={},
    )

    await semantic_service.add_new_feature(
        set_id=profile_id,
        category_name="Profile",
        feature="alpha_fact",
        value="Alpha enjoys calm chats",
        tag="facts",
    )
    await semantic_service.add_new_feature(
        set_id=session_id,
        category_name="Profile",
        feature="beta_fact",
        value="Beta prefers debates",
        tag="facts",
    )

    # When searching with an alpha-focused query
    matches = list(
        await session_manager.search(
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
    session_manager: SemanticSessionManager,
    semantic_storage: SemanticStorage,
    spy_embedder: SpyEmbedder,
):
    set_id_a = "set_id_a"
    set_id_b = "set_id_b"

    # Given an add feature to set_id_a
    feature_id = await session_manager.add_feature(
        set_id=set_id_a,
        category_name="Profile",
        feature="tone",
        value="Alpha casual",
        tag="writing_style",
    )

    # When retrieving features for each set id
    a_features = await semantic_storage.get_feature_set(
        filter_expr=parse_filter(f"set_id IN ('{set_id_a}')")
    )
    b_features = await semantic_storage.get_feature_set(
        filter_expr=parse_filter(f"set_id IN ('{set_id_b}')")
    )

    # Then only the profile receives the new feature and embeddings were generated
    assert len(a_features) == 1
    assert len(b_features) == 0

    assert feature_id == a_features[0].metadata.id
    assert a_features[0].feature_name == "tone"
    assert a_features[0].value == "Alpha casual"

    assert spy_embedder.ingest_calls == [["Alpha casual"]]


async def test_delete_feature_set_removes_filtered_entries(
    session_manager: SemanticSessionManager,
    semantic_service: SemanticService,
    semantic_storage: SemanticStorage,
    session_data,
):
    # Given profile and session features with distinct tags
    set_type_id = session_manager._generate_set_id(
        org_id=session_data.org_id,
        metadata={},
    )
    project_set_id = session_manager._generate_set_id(
        org_id=session_data.org_id,
        project_id=session_data.project_id,
        metadata={},
    )

    await semantic_service.add_new_feature(
        set_id=set_type_id,
        category_name="Profile",
        feature="favorite_color",
        value="Blue",
        tag="profile_tag",
    )
    await semantic_service.add_new_feature(
        set_id=project_set_id,
        category_name="Profile",
        feature="session_note",
        value="Remember to ask about projects",
        tag="session_tag",
    )

    # When deleting only the profile-tagged features
    property_filter = parse_filter("tag IN ('profile_tag')")
    await session_manager.delete_feature_set(
        session_data=session_data,
        property_filter=property_filter,
    )

    # Then profile features are cleared while session-scoped entries remain
    org_features = await semantic_storage.get_feature_set(
        filter_expr=parse_filter(f"set_id IN ('{set_type_id}')")
    )
    project_features = await semantic_storage.get_feature_set(
        filter_expr=parse_filter(f"set_id IN ('{project_set_id}')")
    )

    assert org_features == []
    assert len(project_features) == 1
    assert project_features[0].feature_name == "session_note"


async def test_add_message_uses_all_isolations(
    mock_session_manager: SemanticSessionManager,
    mock_semantic_service: MagicMock,
    session_data,
):
    history_id = "abc"
    await mock_session_manager.add_message(
        session_data=session_data,
        episodes=[
            Episode(
                uid=history_id,
                content="Alpha memory",
                producer_id="profile_id",
                producer_role="dev",
                session_key="session_id",
                created_at=datetime.now(tz=UTC),
            ),
        ],
    )

    session_id = mock_session_manager._generate_set_id(
        org_id=session_data.org_id,
        project_id=session_data.project_id,
        metadata={},
    )

    profile_id = mock_session_manager._generate_set_id(
        org_id=session_data.org_id,
        metadata={},
    )

    user_set_id = mock_session_manager._generate_set_id(
        org_id=session_data.org_id,
        project_id=None,
        metadata={"producer_id": "profile_id"},
    )

    mock_semantic_service.add_message_to_sets.assert_awaited_once()
    args, kwargs = mock_semantic_service.add_message_to_sets.await_args
    assert kwargs == {}

    assert args[0] == history_id
    assert set(args[1]) == {profile_id, session_id, user_set_id}


async def test_add_message_with_session_only_isolation(
    mock_session_manager: SemanticSessionManager,
    mock_semantic_service: MagicMock,
    session_data,
):
    await mock_session_manager.add_message(
        episodes=[
            Episode(
                uid="abc",
                content="Alpha memory",
                producer_id="profile_id",
                producer_role="dev",
                session_key="session_id",
                created_at=datetime.now(tz=UTC),
            ),
        ],
        session_data=session_data,
    )

    mock_semantic_service.add_message_to_sets.assert_awaited_once()
    args, kwargs = mock_semantic_service.add_message_to_sets.await_args
    assert kwargs == {}

    project_id = mock_session_manager._generate_set_id(
        org_id=session_data.org_id,
        project_id=session_data.project_id,
        metadata={},
    )
    set_type_id = mock_session_manager._generate_set_id(
        org_id=session_data.org_id,
        metadata={},
    )

    user_set_id = mock_session_manager._generate_set_id(
        org_id=session_data.org_id,
        project_id=None,
        metadata={"producer_id": "profile_id"},
    )
    assert sorted(args[1]) == sorted([set_type_id, project_id, user_set_id])


async def test_search_passes_set_ids_and_filters(
    mock_session_manager: SemanticSessionManager,
    mock_semantic_service: MagicMock,
    session_data,
):
    mock_semantic_service.search.return_value = ["result"]

    filter_str = "tag IN ('facts') AND feature_name IN ('alpha_fact')"
    result = await mock_session_manager.search(
        message="Find alpha info",
        session_data=session_data,
        search_filter=parse_filter(filter_str),
        limit=5,
        load_citations=True,
    )

    mock_semantic_service.search.assert_awaited_once()
    kwargs = mock_semantic_service.search.await_args.kwargs

    set_type_id = mock_session_manager._generate_set_id(
        org_id=session_data.org_id,
        metadata={},
    )
    session_set_id = mock_session_manager._generate_set_id(
        org_id=session_data.org_id,
        project_id=session_data.project_id,
        metadata={},
    )

    assert sorted(kwargs["set_ids"]) == sorted([set_type_id, session_set_id])
    assert kwargs["limit"] == 5
    assert kwargs["load_citations"] is True
    assert result == ["result"]


async def test_number_of_uningested_messages_delegates(
    mock_session_manager: SemanticSessionManager,
    mock_semantic_service: MagicMock,
    session_data,
):
    mock_semantic_service.number_of_uningested.return_value = 7

    count = await mock_session_manager.number_of_uningested_messages(
        session_data=session_data,
    )

    project_set_id = mock_session_manager._generate_set_id(
        org_id=session_data.org_id,
        project_id=session_data.project_id,
        metadata={},
    )
    set_type_id = mock_session_manager._generate_set_id(
        org_id=session_data.org_id,
        metadata={},
    )

    mock_semantic_service.number_of_uningested.assert_awaited_once()
    _, kwargs = mock_semantic_service.number_of_uningested.await_args

    assert sorted(kwargs["set_ids"]) == sorted([set_type_id, project_set_id])
    assert count == 7


async def test_add_feature_translates_to_single_set(
    mock_session_manager: SemanticSessionManager,
    mock_semantic_service: MagicMock,
):
    set_id = "test_set_id"

    feature_id = await mock_session_manager.add_feature(
        set_id=set_id,
        category_name="Profile",
        feature="tone",
        value="Alpha calm",
        tag="writing_style",
        feature_metadata={"source": "test"},
        citations=["1", "2"],
    )

    mock_semantic_service.add_new_feature.assert_awaited_once()
    _, kwargs = mock_semantic_service.add_new_feature.await_args
    assert kwargs == {
        "set_id": set_id,
        "category_name": "Profile",
        "feature": "tone",
        "value": "Alpha calm",
        "tag": "writing_style",
        "metadata": {"source": "test"},
        "citations": ["1", "2"],
    }
    assert feature_id == 101


async def test_get_feature_proxies_call(
    mock_session_manager: SemanticSessionManager,
    mock_semantic_service: MagicMock,
):
    result = await mock_session_manager.get_feature("42", load_citations=True)

    mock_semantic_service.get_feature.assert_awaited_once()
    args, kwargs = mock_semantic_service.get_feature.await_args
    assert args == ("42",)
    assert kwargs == {"load_citations": True}
    assert result == "feature"


async def test_update_feature_forwards_arguments(
    mock_session_manager: SemanticSessionManager,
    mock_semantic_service: MagicMock,
):
    await mock_session_manager.update_feature(
        "17",
        category_name="Profile",
        feature="tone",
        value="calm",
        tag="writing_style",
        metadata={"updated": "true"},
    )

    mock_semantic_service.update_feature.assert_awaited_once()
    args, kwargs = mock_semantic_service.update_feature.await_args
    assert args == ("17",)
    assert kwargs == {
        "category_name": "Profile",
        "feature": "tone",
        "value": "calm",
        "tag": "writing_style",
        "metadata": {"updated": "true"},
    }


async def test_get_set_features_wraps_opts(
    mock_session_manager: SemanticSessionManager,
    mock_semantic_service: MagicMock,
    session_data,
):
    filter_str = "tag IN ('facts') AND feature_name IN ('alpha_fact')"
    result = await mock_session_manager.get_set_features(
        session_data=session_data,
        search_filter=parse_filter(filter_str),
        page_size=7,
        load_citations=True,
    )

    mock_semantic_service.get_set_features.assert_awaited_once()
    kwargs = mock_semantic_service.get_set_features.await_args.kwargs
    assert kwargs["page_size"] == 7
    assert kwargs["with_citations"] is True
    assert result == ["features"]


async def test_list_set_ids(
    session_manager: SemanticSessionManager,
    session_data,
):
    # Default org has 2 ids. Org level and project level set.
    set_ids = list(await session_manager.list_sets(session_data=session_data))
    assert len(set_ids) == 2

    expanded_session = _SessionData(
        org_id="test_org",
        project_id="test_proj",
    )
    set_metadata = {
        "user_id": "test_user",
    }

    # Add user level isolation
    await session_manager.create_set_type(
        session_data=session_data,
        is_org_level=True,
        metadata_tags=["user_id"],
    )

    # Since session contains user_id it should apear on the list
    set_ids = list(
        await session_manager.list_sets(
            session_data=expanded_session,
            set_metadata=set_metadata,
        )
    )
    assert len(set_ids) == 3

    # While if we create an isolation level that isn't covered by session_data it won't be included.
    await session_manager.create_set_type(
        session_data=session_data,
        is_org_level=True,
        metadata_tags=["other_id"],
    )
    set_ids = list(
        await session_manager.list_sets(
            session_data=expanded_session,
            set_metadata=set_metadata,
        )
    )
    assert len(set_ids) == 3


async def test_list_set_ids_returns_set_details(
    session_manager: SemanticSessionManager,
    session_data,
):
    """Test that list_set_ids returns is_org_level, tags, name, and description."""
    expanded_session = _SessionData(
        org_id="test_org",
        project_id="test_proj",
    )
    set_metadata = {
        "user_id": "test_user",
        "repo_id": "test_repo",
    }

    # Create set types with name and description
    await session_manager.create_set_type(
        session_data=session_data,
        is_org_level=True,
        metadata_tags=["user_id"],
        name="User Settings",
        description="User-specific configuration",
    )

    await session_manager.create_set_type(
        session_data=session_data,
        is_org_level=False,
        metadata_tags=["repo_id"],
        name="Repository Data",
        description="Repository-specific information",
    )

    # Get all set IDs
    sets = list(
        await session_manager.list_sets(
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
    session_manager: SemanticSessionManager, session_data
):
    sets = list(await session_manager.list_sets(session_data=session_data))

    assert len(sets) == 2
    org_set = next(s for s in sets if s.is_org_level)
    project_set = next(s for s in sets if not s.is_org_level)

    await session_manager.configure_set(
        set_id=org_set.id,
        llm_name="llm-org",
    )
    await session_manager.configure_set(
        set_id=project_set.id,
        llm_name="llm-project",
    )

    org_config = await session_manager.get_set_id_config(set_id=org_set.id)
    project_config = await session_manager.get_set_id_config(set_id=project_set.id)
    assert org_config is not None
    assert project_config is not None

    assert org_config.llm_name == "llm-org"
    assert project_config.llm_name == "llm-project"


async def test_user_default_set_requires_producer_metadata_and_is_configurable(
    session_manager: SemanticSessionManager,
    session_data,
):
    base_sets = list(await session_manager.list_sets(session_data=session_data))
    assert len(base_sets) == 2

    user_session = _SessionData(
        org_id=session_data.org_id,
        project_id=session_data.project_id,
    )
    set_metadata = {"producer_id": "user-123"}

    sets_with_user = list(
        await session_manager.list_sets(
            session_data=user_session,
            set_metadata=set_metadata,
        )
    )

    unique_sets = {s.id: s for s in sets_with_user}
    assert len(unique_sets) == 3

    user_set = next(s for s in unique_sets.values() if s.tags == ["producer_id"])
    assert user_set.is_org_level is True

    expected_user_id = SemanticSessionManager.generate_user_set_id(
        org_id=user_session.org_id,
        producer_id="user-123",
    )
    assert user_set.id == expected_user_id

    await session_manager.configure_set(
        set_id=user_set.id,
        llm_name="llm-user",
    )

    user_config = await session_manager.get_set_id_config(set_id=user_set.id)
    assert user_config is not None
    assert user_config.llm_name == "llm-user"


async def test_list_sets_deduplicates_default_overrides(
    session_manager: SemanticSessionManager,
    session_data,
):
    base_set_ids = {
        SemanticSessionManager._generate_set_id(
            org_id=session_data.org_id,
            project_id=None,
            metadata={},
        ),
        SemanticSessionManager._generate_set_id(
            org_id=session_data.org_id,
            project_id=session_data.project_id,
            metadata={},
        ),
    }

    base_sets = list(await session_manager.list_sets(session_data=session_data))
    assert {s.id for s in base_sets} == base_set_ids

    await session_manager.create_set_type(
        session_data=session_data,
        is_org_level=True,
        metadata_tags=[],
        name="Custom Org",
        description="Overrides default org set",
    )

    await session_manager.create_set_type(
        session_data=session_data,
        is_org_level=False,
        metadata_tags=[],
        name="Custom Project",
        description="Overrides default project set",
    )

    deduped_sets = list(await session_manager.list_sets(session_data=session_data))
    assert len(deduped_sets) == 2
    assert {s.id for s in deduped_sets} == base_set_ids

    org_set = next(s for s in deduped_sets if s.is_org_level)
    project_set = next(s for s in deduped_sets if not s.is_org_level)

    assert org_set.tags == []
    assert project_set.tags == []


async def test_configure_set(
    session_manager: SemanticSessionManager,
):
    set_id = "test_set"

    await session_manager.configure_set(
        set_id=set_id,
        llm_name="test_llm",
        embedder_name="test_embedder",
    )

    config = await session_manager.get_set_id_config(set_id=set_id)
    assert config is not None

    assert config.llm_name == "test_llm"
    assert config.embedder_name == "test_embedder"


async def test_invalid_embedder_change_with_dirty_set(
    session_manager: SemanticSessionManager,
):
    set_id = "test_set"
    await session_manager.configure_set(
        set_id=set_id,
        llm_name="test_llm",
        embedder_name="test_embedder",
    )

    await session_manager.add_feature(
        set_id=set_id,
        category_name="Profile",
        feature="test_feature",
        value="test_value",
        tag="test_tag",
    )

    with pytest.raises(InvalidSetIdConfigurationError):
        await session_manager.configure_set(
            set_id=set_id,
            llm_name="test_llm_2",
            embedder_name="test_embedder_2",
        )
    with pytest.raises(InvalidSetIdConfigurationError):
        await session_manager.configure_set(
            set_id=set_id,
            llm_name="test_llm_2",
            embedder_name=None,
        )


async def test_category_templates(
    session_manager: SemanticSessionManager, session_data
):
    set_type_id = await session_manager.create_set_type(
        session_data=session_data,
        is_org_level=False,
        metadata_tags=["repo"],
    )
    c_id = await session_manager.add_category_template(
        set_type_id=set_type_id,
        category_name="test_category",
        description="test_description",
        prompt="test_prompt",
    )

    await session_manager.add_tag(
        category_id=c_id,
        tag_name="test_tag",
        tag_description="test_tag_description",
    )

    categories = list(
        await session_manager.list_category_templates(set_type_id=set_type_id)
    )
    assert len(categories) == 1

    assert categories[0].id == c_id
    assert categories[0].name == "test_category"


async def test_category_templates_are_visible_to_children(
    session_manager: SemanticSessionManager,
    session_data,
):
    expanded_session = _SessionData(
        org_id="test_org",
        project_id="test_proj",
    )
    set_metadata = {
        "user_id": "test_user",
    }

    set_type_id = await session_manager.create_set_type(
        session_data=session_data,
        is_org_level=False,
        metadata_tags=["user_id"],
    )
    set_id = await session_manager.get_set_id(
        session_data=expanded_session,
        is_org_level=False,
        set_metadata_keys=["user_id"],
        set_metadata=set_metadata,
    )

    c_id = await session_manager.add_category_template(
        set_type_id=set_type_id,
        category_name="test_category",
        description="test_description",
        prompt="test_prompt",
    )

    await session_manager.add_tag(
        category_id=c_id,
        tag_name="test_tag",
        tag_description="test_tag_description",
    )

    config = await session_manager.get_set_id_config(set_id=set_id)
    assert config is not None

    categories = config.categories

    assert len(categories) == 1
    assert categories[0].id == c_id


async def test_search_with_time_filter_str(
    session_manager: SemanticSessionManager,
    session_data,
):
    filter_str = "created_at < date('2026-01-19T01:56:41.513342Z')"
    _ = await session_manager.search(
        message="Find alpha info",
        session_data=session_data,
        search_filter=parse_filter(filter_str),
    )

    # Add features
    set_id = await session_manager.get_set_id(
        session_data=session_data,
        set_metadata_keys=[],
    )

    cat_iter = await session_manager.get_set_id_category_names(set_id=set_id)
    cat_name = next(iter(cat_iter), "test")

    await session_manager.add_feature(
        set_id=set_id,
        category_name=cat_name,
        feature="alpha_fact",
        tag="alpha_tag",
        value="alpha_value",
    )

    _ = await session_manager.search(
        message="Find alpha info",
        session_data=session_data,
        search_filter=parse_filter(filter_str),
    )
