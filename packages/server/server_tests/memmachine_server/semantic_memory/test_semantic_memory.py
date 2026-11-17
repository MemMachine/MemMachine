"""Unit tests for the SemanticService using an in-memory storage backend."""

from collections.abc import Sequence

import pytest

from memmachine_server.common.errors import (
    CategoryNotFoundError,
    InvalidSetIdConfigurationError,
)
from memmachine_server.common.filter.filter_parser import parse_filter
from memmachine_server.semantic_memory.config_store.config_store import (
    SemanticConfigStorage,
)
from memmachine_server.semantic_memory.semantic_memory import SemanticService
from memmachine_server.semantic_memory.semantic_model import SetIdT
from memmachine_server.semantic_memory.storage.storage_base import SemanticStorage
from server_tests.memmachine_server.semantic_memory.semantic_test_utils import (
    SessionData,
    SpyEmbedder,
    get_set_id,
)

pytestmark = pytest.mark.asyncio


async def _collect_async(iterator):
    return [item async for item in iterator]


async def _set_features(semantic_storage: SemanticStorage, set_ids: Sequence[SetIdT]):
    quoted = ", ".join(f"'{set_id}'" for set_id in set_ids)
    return await _collect_async(
        semantic_storage.get_feature_set(
            filter_expr=parse_filter(f"set_id IN ({quoted})")
        )
    )


async def test_store_custom_embedder(
    semantic_service: SemanticService,
    semantic_config_storage: SemanticConfigStorage,
    session_data: SessionData,
):
    set_id = await get_set_id(semantic_service, session_data)

    await semantic_service.configure_set(
        session_data=session_data, set_id=set_id, embedder_name="embed-custom"
    )

    conf = await semantic_config_storage.get_setid_config(set_id=set_id)

    assert conf.embedder_name == "embed-custom"
    assert conf.llm_name is None
    assert conf.disabled_categories == []


async def test_use_length_embedder(
    semantic_service: SemanticService,
    semantic_config_storage: SemanticConfigStorage,
    session_data: SessionData,
):
    set_id = await get_set_id(semantic_service, session_data)

    await semantic_service.configure_set(
        session_data=session_data, set_id=set_id, embedder_name="1234"
    )

    conf = await semantic_config_storage.get_setid_config(set_id=set_id)
    assert conf.embedder_name == "1234"

    await semantic_service.add_feature(
        session_data=session_data,
        set_id=set_id,
        category_name="Profile",
        feature="tone",
        value="Alpha friendly",
        tag="writing_style",
    )

    response = await _collect_async(
        semantic_service.search(
            message="Why does alpha prefer quiet chats?",
            session_data=session_data,
        )
    )

    assert len(response) == 1
    assert response[0].feature_name == "tone"


async def test_use_custom_embedder(
    semantic_service: SemanticService,
    session_data: SessionData,
):
    set_id = await get_set_id(semantic_service, session_data)

    await semantic_service.configure_set(
        session_data=session_data, set_id=set_id, embedder_name="1"
    )

    await semantic_service.add_feature(
        session_data=session_data,
        set_id=set_id,
        category_name="Profile",
        feature="tone",
        value="Alpha friendly",
        tag="writing_style",
    )


async def test_can_not_change_embedder(
    semantic_service: SemanticService,
    session_data: SessionData,
):
    set_id = await get_set_id(semantic_service, session_data)

    await semantic_service.configure_set(
        session_data=session_data, set_id=set_id, embedder_name="5"
    )

    with pytest.raises(InvalidSetIdConfigurationError):
        await semantic_service.configure_set(
            session_data=session_data, set_id=set_id, embedder_name="1234"
        )


async def test_add_new_category_to_set_id_config(
    semantic_service: SemanticService,
    semantic_config_storage: SemanticConfigStorage,
    session_data: SessionData,
):
    set_id = await get_set_id(semantic_service, session_data)
    prompt = "Try to identify interests the user may have."
    c_id = await semantic_service.add_new_category(
        session_data=session_data,
        set_id=set_id,
        category_name="interests",
        prompt=prompt,
        description="Prompt to extract user interests",
    )
    assert c_id is not None

    conf = await semantic_config_storage.get_setid_config(set_id=set_id)

    assert conf.categories[0].name == "interests"
    assert conf.categories[0].id == c_id
    assert prompt in conf.categories[0].prompt.update_prompt

    category = await semantic_service.get_category(
        session_data=session_data, category_id=c_id
    )

    assert category is not None
    assert category.id == c_id
    assert category.name == "interests"
    assert category.description == "Prompt to extract user interests"
    assert category.prompt == prompt


async def test_add_tags_to_category(
    semantic_service: SemanticService,
    semantic_config_storage: SemanticConfigStorage,
    session_data: SessionData,
):
    set_id = await get_set_id(semantic_service, session_data)
    description = "Try to identify interests the user may have."
    c_id = await semantic_service.add_new_category(
        session_data=session_data,
        set_id=set_id,
        category_name="interests",
        prompt=description,
        description="",
    )

    tag_a_name = "Music"
    tag_b_name = "Technology"

    tag_a_id = await semantic_service.add_tag(
        session_data=session_data,
        category_id=c_id,
        tag_name=tag_a_name,
        tag_description="",
    )
    tag_b_id = await semantic_service.add_tag(
        session_data=session_data,
        category_id=c_id,
        tag_name=tag_b_name,
        tag_description="",
    )

    assert tag_a_id is not None
    assert tag_b_id is not None

    conf = await semantic_config_storage.get_setid_config(set_id=set_id)
    assert tag_a_name in conf.categories[0].prompt.update_prompt
    assert tag_b_name in conf.categories[0].prompt.update_prompt


async def test_add_feature_stores_entry(
    semantic_service: SemanticService,
    spy_embedder: SpyEmbedder,
    session_data: SessionData,
):
    set_id = await get_set_id(semantic_service, session_data)

    feature_id = await semantic_service.add_feature(
        session_data=session_data,
        set_id=set_id,
        category_name="Profile",
        feature="tone",
        value="Alpha voice",
        tag="writing_style",
        feature_metadata={"source": "test"},
    )

    features = await _collect_async(
        semantic_service.get_set_features(session_data=session_data)
    )

    assert spy_embedder.ingest_calls == [["Alpha voice"]]
    assert len(features) == 1
    feature = features[0]
    assert feature.metadata.id == feature_id
    assert feature.set_id == set_id
    assert feature.feature_name == "tone"
    assert feature.value == "Alpha voice"
    assert feature.tag == "writing_style"


async def test_get_set_features_filters_by_tag(
    semantic_service: SemanticService,
    session_data: SessionData,
):
    set_id = await get_set_id(semantic_service, session_data)
    await semantic_service.add_feature(
        session_data=session_data,
        set_id=set_id,
        category_name="Profile",
        feature="tone",
        value="Alpha friendly",
        tag="writing_style",
    )
    await semantic_service.add_feature(
        session_data=session_data,
        set_id=set_id,
        category_name="Profile",
        feature="favorite_color",
        value="Blue",
        tag="personal_info",
    )

    filtered = await _collect_async(
        semantic_service.get_set_features(
            session_data=session_data,
            search_filter=parse_filter("tag IN ('writing_style')"),
        )
    )

    assert len(filtered) == 1
    assert filtered[0].feature_name == "tone"
    assert filtered[0].tag == "writing_style"


async def test_update_feature_changes_value_and_embedding(
    semantic_service: SemanticService,
    spy_embedder: SpyEmbedder,
    session_data: SessionData,
):
    set_id = await get_set_id(semantic_service, session_data)
    feature_id = await semantic_service.add_feature(
        session_data=session_data,
        set_id=set_id,
        category_name="Profile",
        feature="tone",
        value="Alpha calm",
        tag="writing_style",
    )

    await semantic_service.update_feature(
        session_data,
        feature_id,
        category_name="Profile",
        value="Alpha energetic",
        tag="writing_style",
    )

    feature = await semantic_service.get_feature(session_data, feature_id)
    assert feature is not None
    assert feature.value == "Alpha energetic"
    assert spy_embedder.ingest_calls == [["Alpha calm"], ["Alpha energetic"]]


async def test_delete_features_removes_selected_entries(
    semantic_service: SemanticService,
    session_data: SessionData,
):
    set_id = await get_set_id(semantic_service, session_data)
    to_remove = await semantic_service.add_feature(
        session_data=session_data,
        set_id=set_id,
        category_name="Profile",
        feature="tone",
        value="Alpha calm",
        tag="writing_style",
    )
    to_keep = await semantic_service.add_feature(
        session_data=session_data,
        set_id=set_id,
        category_name="Profile",
        feature="hobby",
        value="Gardening",
        tag="personal_info",
    )

    await semantic_service.delete_features(session_data, [to_remove])

    assert await semantic_service.get_feature(session_data, to_remove) is None
    assert await semantic_service.get_feature(session_data, to_keep) is not None


async def test_search_returns_matching_features(
    semantic_service: SemanticService,
    spy_embedder: SpyEmbedder,
    session_data: SessionData,
):
    set_id = await get_set_id(semantic_service, session_data)
    await semantic_service.add_feature(
        session_data=session_data,
        set_id=set_id,
        category_name="Profile",
        feature="alpha_fact",
        value="Alpha prefers calm conversations.",
        tag="facts",
    )
    await semantic_service.add_feature(
        session_data=session_data,
        set_id=set_id,
        category_name="Profile",
        feature="beta_fact",
        value="Beta enjoys debates.",
        tag="facts",
    )

    results = await _collect_async(
        semantic_service.search(
            message="Why does alpha prefer quiet chats?",
            session_data=session_data,
            min_distance=0.5,
        )
    )

    assert spy_embedder.search_calls
    assert all(
        call == ["Why does alpha prefer quiet chats?"]
        for call in spy_embedder.search_calls
    )
    assert len(results) == 1
    assert results[0].feature_name == "alpha_fact"


async def test_delete_category_deletes_underlying_features(
    semantic_service: SemanticService,
    session_data: SessionData,
):
    """Test that deleting a set_id category also deletes all features in that category."""
    set_id = await get_set_id(semantic_service, session_data)

    category_id = await semantic_service.add_new_category(
        session_data=session_data,
        set_id=set_id,
        category_name="TestCategory",
        prompt="Test category prompt",
        description="Test description",
    )

    await semantic_service.add_feature(
        session_data=session_data,
        set_id=set_id,
        category_name="TestCategory",
        feature="feature1",
        value="value1",
        tag="tag1",
    )
    await semantic_service.add_feature(
        session_data=session_data,
        set_id=set_id,
        category_name="TestCategory",
        feature="feature2",
        value="value2",
        tag="tag2",
    )

    # Add a feature in a different category (should not be deleted)
    await semantic_service.add_new_category(
        session_data=session_data,
        set_id=set_id,
        category_name="OtherCategory",
        prompt="Other category prompt",
        description="Other description",
    )
    await semantic_service.add_feature(
        session_data=session_data,
        set_id=set_id,
        category_name="OtherCategory",
        feature="other_feature",
        value="other_value",
        tag="other_tag",
    )

    features_before = await _collect_async(
        semantic_service.get_set_features(session_data=session_data)
    )
    assert len(features_before) == 3
    test_category_features = [
        f for f in features_before if f.category == "TestCategory"
    ]
    other_category_features = [
        f for f in features_before if f.category == "OtherCategory"
    ]
    assert len(test_category_features) == 2
    assert len(other_category_features) == 1

    await semantic_service.delete_category(
        session_data=session_data, category_id=category_id
    )

    category = await semantic_service.get_category(
        session_data=session_data, category_id=category_id
    )
    assert category is None

    features_after = await _collect_async(
        semantic_service.get_set_features(session_data=session_data)
    )
    assert len(features_after) == 1
    assert features_after[0].category == "OtherCategory"
    assert features_after[0].feature_name == "other_feature"


async def test_delete_set_type_category_deletes_inherited_features(
    semantic_service: SemanticService,
    semantic_storage: SemanticStorage,
    session_data: SessionData,
):
    """Test that deleting a set_type category deletes features from all inheriting set_ids."""
    set_type_id = await semantic_service.create_set_type(
        session_data=session_data,
        is_org_level=False,
        metadata_tags=["repo"],
    )

    category_id = await semantic_service.add_category_template(
        session_data=session_data,
        set_type_id=set_type_id,
        category_name="SharedCategory",
        prompt="Shared category prompt",
        description="Shared description",
    )

    set_ids = [
        await get_set_id(semantic_service, session_data, repo=repo)
        for repo in ("one", "two", "three")
    ]
    for index, set_id in enumerate(set_ids):
        await semantic_service.add_feature(
            session_data=session_data,
            set_id=set_id,
            category_name="SharedCategory",
            feature=f"feature{index}",
            value=f"value{index}",
            tag=f"tag{index}",
        )

    all_features = await _set_features(semantic_storage, set_ids)
    assert len(all_features) == 3
    assert all(f.category == "SharedCategory" for f in all_features)

    await semantic_service.delete_category(
        session_data=session_data, category_id=category_id
    )

    assert await _set_features(semantic_storage, set_ids) == []


async def test_delete_set_type_category_only_deletes_non_overridden_features(
    semantic_service: SemanticService,
    semantic_storage: SemanticStorage,
    session_data: SessionData,
):
    """Test that deleting a set_type category only deletes features from non-overriding set_ids."""
    set_type_id = await semantic_service.create_set_type(
        session_data=session_data,
        is_org_level=False,
        metadata_tags=["repo"],
    )

    set_type_category_id = await semantic_service.add_category_template(
        session_data=session_data,
        set_type_id=set_type_id,
        category_name="OverridableCategory",
        prompt="Original org prompt",
        description="Original description",
    )

    set_id_inherit = await get_set_id(semantic_service, session_data, repo="inherit")
    set_id_override = await get_set_id(semantic_service, session_data, repo="override")

    await semantic_service.add_feature(
        session_data=session_data,
        set_id=set_id_inherit,
        category_name="OverridableCategory",
        feature="inherited_feature",
        value="inherited value",
        tag="tag1",
    )
    await semantic_service.add_feature(
        session_data=session_data,
        set_id=set_id_override,
        category_name="OverridableCategory",
        feature="override_feature",
        value="override value",
        tag="tag2",
    )

    # Override the category in set_id_override
    local_category_id = await semantic_service.add_new_category(
        session_data=session_data,
        set_id=set_id_override,
        category_name="OverridableCategory",
        prompt="Local override prompt",
        description="Local override description",
    )

    set_ids = [set_id_inherit, set_id_override]
    assert len(await _set_features(semantic_storage, set_ids)) == 2

    await semantic_service.delete_category(
        session_data=session_data, category_id=set_type_category_id
    )

    remaining_features = await _set_features(semantic_storage, set_ids)
    assert len(remaining_features) == 1
    assert remaining_features[0].set_id == set_id_override
    assert remaining_features[0].feature_name == "override_feature"

    local_category = await semantic_service.get_category(
        session_data=session_data, category_id=local_category_id
    )
    assert local_category is not None
    assert local_category.name == "OverridableCategory"


async def test_add_feature_validates_category_exists(
    semantic_service: SemanticService,
    session_data: SessionData,
):
    """Test that adding a feature to a non-existent category raises CategoryNotFoundError."""
    set_id = await get_set_id(semantic_service, session_data)

    with pytest.raises(CategoryNotFoundError) as exc_info:
        await semantic_service.add_feature(
            session_data=session_data,
            set_id=set_id,
            category_name="NonExistentCategory",
            feature="test_feature",
            value="test value",
            tag="test_tag",
        )

    assert exc_info.value.set_id == set_id
    assert exc_info.value.category_name == "NonExistentCategory"
    assert "does not exist" in str(exc_info.value)


async def test_add_feature_succeeds_with_valid_category(
    semantic_service: SemanticService,
    session_data: SessionData,
):
    """Test that adding a feature to a valid category succeeds."""
    set_id = await get_set_id(semantic_service, session_data)

    await semantic_service.add_new_category(
        session_data=session_data,
        set_id=set_id,
        category_name="ValidCategory",
        prompt="Test category prompt",
        description="Test description",
    )

    feature_id = await semantic_service.add_feature(
        session_data=session_data,
        set_id=set_id,
        category_name="ValidCategory",
        feature="test_feature",
        value="test value",
        tag="test_tag",
    )

    assert feature_id is not None

    features = await _collect_async(
        semantic_service.get_set_features(session_data=session_data)
    )
    assert len(features) == 1
    assert features[0].category == "ValidCategory"


async def test_update_feature_validates_category_exists(
    semantic_service: SemanticService,
    session_data: SessionData,
):
    """Test that updating a feature to a non-existent category raises CategoryNotFoundError."""
    set_id = await get_set_id(semantic_service, session_data)

    await semantic_service.add_new_category(
        session_data=session_data,
        set_id=set_id,
        category_name="OriginalCategory",
        prompt="Original prompt",
        description="Original description",
    )

    feature_id = await semantic_service.add_feature(
        session_data=session_data,
        set_id=set_id,
        category_name="OriginalCategory",
        feature="test_feature",
        value="test value",
        tag="test_tag",
    )

    with pytest.raises(CategoryNotFoundError) as exc_info:
        await semantic_service.update_feature(
            session_data,
            feature_id=feature_id,
            category_name="NonExistentCategory",
        )

    assert exc_info.value.set_id == set_id
    assert exc_info.value.category_name == "NonExistentCategory"


async def test_update_feature_succeeds_with_valid_category(
    semantic_service: SemanticService,
    session_data: SessionData,
):
    """Test that updating a feature to a valid category succeeds."""
    set_id = await get_set_id(semantic_service, session_data)

    await semantic_service.add_new_category(
        session_data=session_data,
        set_id=set_id,
        category_name="Category1",
        prompt="Category 1 prompt",
        description="Category 1 description",
    )
    await semantic_service.add_new_category(
        session_data=session_data,
        set_id=set_id,
        category_name="Category2",
        prompt="Category 2 prompt",
        description="Category 2 description",
    )

    feature_id = await semantic_service.add_feature(
        session_data=session_data,
        set_id=set_id,
        category_name="Category1",
        feature="test_feature",
        value="test value",
        tag="test_tag",
    )

    await semantic_service.update_feature(
        session_data,
        feature_id=feature_id,
        category_name="Category2",
    )

    feature = await semantic_service.get_feature(session_data, feature_id)
    assert feature is not None
    assert feature.category == "Category2"
