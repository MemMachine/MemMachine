"""Unit tests for the SemanticSessionManager using in-memory storage."""

from dataclasses import dataclass
from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock

import pytest
import pytest_asyncio
from pydantic import JsonValue

from memmachine.common.episode_store import Episode, EpisodeEntry, EpisodeStorage
from memmachine.common.filter.filter_parser import parse_filter
from memmachine.semantic_memory.config_store.config_store import SemanticConfigStorage
from memmachine.semantic_memory.semantic_memory import SemanticService
from memmachine.semantic_memory.semantic_session_manager import (
    SemanticSessionManager,
)
from tests.memmachine.semantic_memory.semantic_test_utils import SpyEmbedder
from tests.memmachine.semantic_memory.storage.in_memory_semantic_storage import (
    SemanticStorage,
)

pytestmark = pytest.mark.asyncio


@pytest.fixture
def session_data():
    @dataclass
    class _SessionData:
        org_id: str
        project_id: str
        metadata: dict[str, JsonValue] | None = None

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
    matches = await session_manager.search(
        message="Why does alpha stay calm?",
        session_data=session_data,
        min_distance=0.5,
    )

    # Then only the alpha feature is returned and embedder search was invoked
    assert spy_embedder.search_calls == [["Why does alpha stay calm?"]]
    assert len(matches) == 1
    assert matches[0].feature_name == "alpha_fact"
    assert matches[0].set_id in {profile_id, session_id}


async def test_add_feature_applies_requested_isolation(
    session_manager: SemanticSessionManager,
    semantic_storage: SemanticStorage,
    spy_embedder: SpyEmbedder,
    session_data,
):
    # Given a profile-scoped feature request
    feature_id = await session_manager.add_feature(
        session_data=session_data,
        category_name="Profile",
        feature="tone",
        value="Alpha casual",
        tag="writing_style",
        set_metadata_keys=[],
    )

    # When retrieving features for each set id
    org_set_id = session_manager._generate_set_id(
        org_id=session_data.org_id,
        metadata={},
    )
    project_set_id = session_manager._generate_set_id(
        org_id=session_data.org_id,
        project_id=session_data.project_id,
        metadata={},
    )

    org_features = await semantic_storage.get_feature_set(
        filter_expr=parse_filter(f"set_id IN ('{org_set_id}')")
    )
    project_features = await semantic_storage.get_feature_set(
        filter_expr=parse_filter(f"set_id IN ('{project_set_id}')")
    )

    # Then only the profile receives the new feature and embeddings were generated
    assert len(org_features) == 0
    assert len(project_features) == 1

    assert feature_id == project_features[0].metadata.id
    assert project_features[0].feature_name == "tone"
    assert project_features[0].value == "Alpha casual"

    assert spy_embedder.ingest_calls == [["Alpha casual"]]


async def test_delete_feature_set_removes_filtered_entries(
    session_manager: SemanticSessionManager,
    semantic_service: SemanticService,
    semantic_storage: SemanticStorage,
    session_data,
):
    # Given profile and session features with distinct tags
    org_set_id = session_manager._generate_set_id(
        org_id=session_data.org_id,
        metadata={},
    )
    project_set_id = session_manager._generate_set_id(
        org_id=session_data.org_id,
        project_id=session_data.project_id,
        metadata={},
    )

    await semantic_service.add_new_feature(
        set_id=org_set_id,
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
        filter_expr=parse_filter(f"set_id IN ('{org_set_id}')")
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

    mock_semantic_service.add_message_to_sets.assert_awaited_once()
    args, kwargs = mock_semantic_service.add_message_to_sets.await_args
    assert kwargs == {}

    assert args[0] == history_id
    assert set(args[1]) == {profile_id, session_id}


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
    org_set_id = mock_session_manager._generate_set_id(
        org_id=session_data.org_id,
        metadata={},
    )

    assert sorted(args[1]) == sorted([org_set_id, project_id])


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

    org_set_id = mock_session_manager._generate_set_id(
        org_id=session_data.org_id,
        metadata={},
    )
    session_set_id = mock_session_manager._generate_set_id(
        org_id=session_data.org_id,
        project_id=session_data.project_id,
        metadata={},
    )

    assert sorted(kwargs["set_ids"]) == sorted([org_set_id, session_set_id])
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
    org_set_id = mock_session_manager._generate_set_id(
        org_id=session_data.org_id,
        metadata={},
    )

    mock_semantic_service.number_of_uningested.assert_awaited_once()
    _, kwargs = mock_semantic_service.number_of_uningested.await_args

    assert sorted(kwargs["set_ids"]) == sorted([org_set_id, project_set_id])
    assert count == 7


async def test_add_feature_translates_to_single_set(
    mock_session_manager: SemanticSessionManager,
    mock_semantic_service: MagicMock,
    session_data,
):
    feature_id = await mock_session_manager.add_feature(
        session_data=session_data,
        category_name="Profile",
        feature="tone",
        value="Alpha calm",
        tag="writing_style",
        feature_metadata={"source": "test"},
        citations=["1", "2"],
        set_metadata_keys=[],
    )

    project_set_id = mock_session_manager._generate_set_id(
        org_id=session_data.org_id,
        project_id=session_data.project_id,
        metadata={},
    )

    mock_semantic_service.add_new_feature.assert_awaited_once()
    _, kwargs = mock_semantic_service.add_new_feature.await_args
    assert kwargs == {
        "set_id": project_set_id,
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
