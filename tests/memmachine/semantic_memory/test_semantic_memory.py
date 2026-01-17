"""Unit tests for the SemanticService using an in-memory storage backend."""

import pytest

from memmachine.common.episode_store import EpisodeStorage
from memmachine.common.errors import InvalidSetIdConfigurationError
from memmachine.common.filter.filter_parser import parse_filter
from memmachine.semantic_memory.semantic_memory import SemanticService
from memmachine.semantic_memory.storage.storage_base import SemanticStorage
from tests.memmachine.semantic_memory.semantic_test_utils import (
    SpyEmbedder,
    add_history,
)

pytestmark = pytest.mark.asyncio


async def test_store_custom_embedder(
    semantic_service: SemanticService,
):
    await semantic_service.set_set_id_config(
        set_id="user-custom",
        embedder_name="embed-custom",
    )

    conf = await semantic_service.get_set_id_config(
        set_id="user-custom",
    )

    assert conf.embedder_name == "embed-custom"
    assert conf.llm_name is None
    assert conf.disabled_categories == []


async def test_use_length_embedder(
    semantic_service: SemanticService,
):
    await semantic_service.set_set_id_config(
        set_id="user-length-1234",
        embedder_name="1234",
    )

    conf = await semantic_service.get_set_id_config(
        set_id="user-length-1234",
    )
    assert conf.embedder_name == "1234"

    await semantic_service.add_new_feature(
        set_id="user-length-1234",
        category_name="Profile",
        feature="tone",
        value="Alpha friendly",
        tag="writing_style",
    )

    response = await semantic_service.search(
        set_ids=["user-length-1234"],
        query="Why does alpha prefer quiet chats?",
    )

    assert len(response) == 1
    assert response[0].feature_name == "tone"


async def test_use_custom_embedder(
    semantic_service: SemanticService,
):
    await semantic_service.set_set_id_config(
        set_id="user-custom",
        embedder_name="1",
    )

    await semantic_service.add_new_feature(
        set_id="user-custom",
        category_name="Profile",
        feature="tone",
        value="Alpha friendly",
        tag="writing_style",
    )


async def test_can_not_change_embedder(
    semantic_service: SemanticService,
):
    await semantic_service.set_set_id_config(
        set_id="user",
        embedder_name="5",
    )

    with pytest.raises(InvalidSetIdConfigurationError):
        await semantic_service.set_set_id_config(
            set_id="user",
            embedder_name="1234",
        )


async def test_add_new_category_to_set_id_config(
    semantic_service: SemanticService,
):
    prompt = "Try to identify interests the user may have."
    c_id = await semantic_service.add_new_category_to_set_id(
        set_id="user-123",
        category_name="interests",
        prompt=prompt,
        description="Prompt to extract user interests",
    )
    assert c_id is not None

    conf = await semantic_service.get_set_id_config(
        set_id="user-123",
    )

    assert conf.categories[0].name == "interests"
    assert conf.categories[0].id == c_id
    assert prompt in conf.categories[0].prompt.update_prompt

    category = await semantic_service.get_category(category_id=c_id)

    assert category is not None
    assert category.id == c_id
    assert category.name == "interests"
    assert category.description == "Prompt to extract user interests"
    assert category.prompt == prompt


async def test_add_tags_to_category(
    semantic_service: SemanticService,
):
    description = "Try to identify interests the user may have."
    c_id = await semantic_service.add_new_category_to_set_id(
        set_id="user-123",
        category_name="interests",
        prompt=description,
        description="",
    )

    tag_a_name = "Music"
    tag_b_name = "Technology"

    tag_a_id = await semantic_service.add_tag(
        category_id=c_id, tag_name=tag_a_name, tag_description=""
    )
    tag_b_id = await semantic_service.add_tag(
        category_id=c_id,
        tag_name=tag_b_name,
        tag_description="",
    )

    assert tag_a_id is not None
    assert tag_b_id is not None

    conf = await semantic_service.get_set_id_config(
        set_id="user-123",
    )
    assert tag_a_name in conf.categories[0].prompt.update_prompt
    assert tag_b_name in conf.categories[0].prompt.update_prompt


async def test_add_new_feature_stores_entry(
    semantic_service: SemanticService,
    spy_embedder: SpyEmbedder,
):
    # Given a fresh semantic service
    feature_id = await semantic_service.add_new_feature(
        set_id="user-123",
        category_name="Profile",
        feature="tone",
        value="Alpha voice",
        tag="writing_style",
        metadata={"source": "test"},
    )

    # When retrieving the stored features
    features = await semantic_service.get_set_features(
        set_ids=["user-123"],
    )

    # Then the feature is persisted with embeddings recorded
    assert spy_embedder.ingest_calls == [["Alpha voice"]]
    assert len(features) == 1
    feature = features[0]
    assert feature.metadata.id == feature_id
    assert feature.set_id == "user-123"
    assert feature.feature_name == "tone"
    assert feature.value == "Alpha voice"
    assert feature.tag == "writing_style"


async def test_get_set_features_filters_by_tag(
    semantic_service: SemanticService,
):
    # Given multiple features under a single set
    await semantic_service.add_new_feature(
        set_id="user-42",
        category_name="Profile",
        feature="tone",
        value="Alpha friendly",
        tag="writing_style",
    )
    await semantic_service.add_new_feature(
        set_id="user-42",
        category_name="Profile",
        feature="favorite_color",
        value="Blue",
        tag="personal_info",
    )

    # When filtering on a specific tag
    filtered = await semantic_service.get_set_features(
        set_ids=["user-42"],
        filter_expr=parse_filter("tag IN ('writing_style')"),
    )

    # Then only matching features are returned
    assert len(filtered) == 1
    assert filtered[0].feature_name == "tone"
    assert filtered[0].tag == "writing_style"


async def test_update_feature_changes_value_and_embedding(
    semantic_service: SemanticService,
    spy_embedder: SpyEmbedder,
):
    # Given an existing feature
    feature_id = await semantic_service.add_new_feature(
        set_id="user-7",
        category_name="Profile",
        feature="tone",
        value="Alpha calm",
        tag="writing_style",
    )

    # When updating the value
    await semantic_service.update_feature(
        feature_id,
        set_id="user-7",
        category_name="Profile",
        value="Alpha energetic",
        tag="writing_style",
    )

    # Then the feature reflects the new value and re-embeds
    feature = await semantic_service.get_feature(feature_id, load_citations=False)
    assert feature is not None
    assert feature.value == "Alpha energetic"
    assert spy_embedder.ingest_calls == [["Alpha calm"], ["Alpha energetic"]]


async def test_delete_features_removes_selected_entries(
    semantic_service: SemanticService,
):
    # Given two stored features
    to_remove = await semantic_service.add_new_feature(
        set_id="user-55",
        category_name="Profile",
        feature="tone",
        value="Alpha calm",
        tag="writing_style",
    )
    to_keep = await semantic_service.add_new_feature(
        set_id="user-55",
        category_name="Profile",
        feature="hobby",
        value="Gardening",
        tag="personal_info",
    )

    # When deleting one feature by id
    await semantic_service.delete_features([to_remove])

    # Then the targeted feature is gone and the other remains
    assert await semantic_service.get_feature(to_remove, load_citations=False) is None
    assert await semantic_service.get_feature(to_keep, load_citations=False) is not None


async def test_delete_feature_set_applies_filters(
    semantic_service: SemanticService,
):
    # Given two features with different tags
    await semantic_service.add_new_feature(
        set_id="user-88",
        category_name="Profile",
        feature="tone",
        value="Alpha calm",
        tag="writing_style",
    )
    await semantic_service.add_new_feature(
        set_id="user-88",
        category_name="Profile",
        feature="favorite_color",
        value="Blue",
        tag="personal_info",
    )

    # When deleting by tag filter
    filter_str = "tag in ('writing_style')"
    filter_expr = parse_filter(filter_str)
    assert filter_expr is not None

    await semantic_service.delete_feature_set(
        set_ids=["user-88"],
        filter_expr=filter_expr,
    )

    # Then only the non-matching feature remains
    remaining = await semantic_service.get_set_features(
        set_ids=["user-88"],
    )
    assert len(remaining) == 1
    assert remaining[0].feature_name == "favorite_color"


async def test_add_messages_tracks_uningested_counts(
    semantic_service: SemanticService,
    semantic_storage: SemanticStorage,
    episode_storage: EpisodeStorage,
):
    # Given a stored history message
    history_id = await add_history(
        history_storage=episode_storage,
        content="Alpha memory",
    )

    # When associating the message to a set
    await semantic_service.add_messages(set_id="user-21", history_ids=[history_id])

    # Then the set reports one uningested message
    assert await semantic_service.number_of_uningested(["user-21"]) == 1

    # When the message is marked ingested
    await semantic_storage.mark_messages_ingested(
        set_id="user-21",
        history_ids=[history_id],
    )

    # Then the uningested count drops to zero
    assert await semantic_service.number_of_uningested(["user-21"]) == 0


async def test_add_message_to_sets_supports_multiple_targets(
    semantic_service: SemanticService,
    semantic_storage: SemanticStorage,
    episode_storage: EpisodeStorage,
):
    # Given a history entry
    history_id = await add_history(
        history_storage=episode_storage,
        content="Alpha shared memory",
    )

    # When linking the message to multiple sets
    await semantic_service.add_message_to_sets(
        history_id=history_id,
        set_ids=["user-a", "user-b"],
    )

    # Then all sets report the pending ingestion
    assert await semantic_service.number_of_uningested(["user-a"]) == 1
    assert await semantic_service.number_of_uningested(["user-b"]) == 1


async def test_search_returns_matching_features(
    semantic_service: SemanticService,
    spy_embedder: SpyEmbedder,
):
    # Given a set with two features
    await semantic_service.add_new_feature(
        set_id="user-search",
        category_name="Profile",
        feature="alpha_fact",
        value="Alpha prefers calm conversations.",
        tag="facts",
    )
    await semantic_service.add_new_feature(
        set_id="user-search",
        category_name="Profile",
        feature="beta_fact",
        value="Beta enjoys debates.",
        tag="facts",
    )

    # When searching with an alpha-focused query
    results = await semantic_service.search(
        set_ids=["user-search"],
        query="Why does alpha prefer quiet chats?",
        min_distance=0.5,
    )

    # Then only the matching feature is returned using the query embedding
    assert spy_embedder.search_calls == [["Why does alpha prefer quiet chats?"]]
    assert len(results) == 1
    assert results[0].feature_name == "alpha_fact"


async def test_delete_category_deletes_underlying_features(
    semantic_service: SemanticService,
):
    """Test that deleting a set_id category also deletes all features in that category."""
    set_id = "user-delete-cat"

    # Create a category
    category_id = await semantic_service.add_new_category_to_set_id(
        set_id=set_id,
        category_name="TestCategory",
        prompt="Test category prompt",
        description="Test description",
    )

    # Add features to this category
    await semantic_service.add_new_feature(
        set_id=set_id,
        category_name="TestCategory",
        feature="feature1",
        value="value1",
        tag="tag1",
    )
    await semantic_service.add_new_feature(
        set_id=set_id,
        category_name="TestCategory",
        feature="feature2",
        value="value2",
        tag="tag2",
    )

    # Add a feature in a different category (should not be deleted)
    await semantic_service.add_new_category_to_set_id(
        set_id=set_id,
        category_name="OtherCategory",
        prompt="Other category prompt",
        description="Other description",
    )
    await semantic_service.add_new_feature(
        set_id=set_id,
        category_name="OtherCategory",
        feature="other_feature",
        value="other_value",
        tag="other_tag",
    )

    # Verify features exist
    features_before = await semantic_service.get_set_features(
        set_ids=[set_id],
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

    # Delete the category
    await semantic_service.delete_category(category_id=category_id)

    # Verify the category config is deleted
    category = await semantic_service.get_category(category_id=category_id)
    assert category is None

    # Verify only TestCategory features are deleted
    features_after = await semantic_service.get_set_features(
        set_ids=[set_id],
    )
    assert len(features_after) == 1
    assert features_after[0].category == "OtherCategory"
    assert features_after[0].feature_name == "other_feature"


async def test_delete_org_set_category_deletes_inherited_features(
    semantic_service: SemanticService,
):
    """Test that deleting an org_set category deletes features from all inheriting set_ids."""
    # Create an org set
    org_set_id = await semantic_service._semantic_config_storage.add_org_set_id(
        org_id="org-delete-test",
        org_level_set=False,
        metadata_tags=["repo"],
    )

    # Create a category in the org set
    category_id = await semantic_service.add_new_category_to_org_set(
        org_set_id=org_set_id,
        category_name="SharedCategory",
        prompt="Shared category prompt",
        description="Shared description",
    )

    # Register three set_ids to this org set
    set_id_1 = "set-inherit-del-1"
    set_id_2 = "set-inherit-del-2"
    set_id_3 = "set-inherit-del-3"

    for sid in [set_id_1, set_id_2, set_id_3]:
        await semantic_service._semantic_config_storage.register_set_id_org_set(
            set_id=sid,
            org_set_id=org_set_id,
        )

    # Add features to each set_id with the shared category
    await semantic_service.add_new_feature(
        set_id=set_id_1,
        category_name="SharedCategory",
        feature="feature1",
        value="value1 for set1",
        tag="tag1",
    )
    await semantic_service.add_new_feature(
        set_id=set_id_2,
        category_name="SharedCategory",
        feature="feature2",
        value="value2 for set2",
        tag="tag2",
    )
    await semantic_service.add_new_feature(
        set_id=set_id_3,
        category_name="SharedCategory",
        feature="feature3",
        value="value3 for set3",
        tag="tag3",
    )

    # Verify features exist in all sets
    all_features = await semantic_service.get_set_features(
        set_ids=[set_id_1, set_id_2, set_id_3],
    )
    assert len(all_features) == 3
    assert all(f.category == "SharedCategory" for f in all_features)

    # Delete the category
    await semantic_service.delete_category(category_id=category_id)

    # Verify all features are deleted from all set_ids
    remaining_features = await semantic_service.get_set_features(
        set_ids=[set_id_1, set_id_2, set_id_3],
    )
    assert len(remaining_features) == 0


async def test_delete_org_set_category_only_deletes_non_overridden_features(
    semantic_service: SemanticService,
):
    """Test that deleting an org_set category only deletes features from non-overriding set_ids."""
    # Create an org set
    org_set_id = await semantic_service._semantic_config_storage.add_org_set_id(
        org_id="org-delete-override",
        org_level_set=False,
        metadata_tags=["repo"],
    )

    # Create a category in the org set
    org_category_id = await semantic_service.add_new_category_to_org_set(
        org_set_id=org_set_id,
        category_name="OverridableCategory",
        prompt="Original org prompt",
        description="Original description",
    )

    # Register two set_ids
    set_id_inherit = "set-inherit-no-override"
    set_id_override = "set-override-local"

    for sid in [set_id_inherit, set_id_override]:
        await semantic_service._semantic_config_storage.register_set_id_org_set(
            set_id=sid,
            org_set_id=org_set_id,
        )

    # Add features to both sets with the shared category
    await semantic_service.add_new_feature(
        set_id=set_id_inherit,
        category_name="OverridableCategory",
        feature="inherited_feature",
        value="inherited value",
        tag="tag1",
    )
    await semantic_service.add_new_feature(
        set_id=set_id_override,
        category_name="OverridableCategory",
        feature="override_feature",
        value="override value",
        tag="tag2",
    )

    # Override the category in set_id_override
    local_category_id = await semantic_service.add_new_category_to_set_id(
        set_id=set_id_override,
        category_name="OverridableCategory",
        prompt="Local override prompt",
        description="Local override description",
    )

    # Verify both features exist
    all_features_before = await semantic_service.get_set_features(
        set_ids=[set_id_inherit, set_id_override],
    )
    assert len(all_features_before) == 2

    # Delete the org category
    await semantic_service.delete_category(category_id=org_category_id)

    # Verify only the inherited feature is deleted
    remaining_features = await semantic_service.get_set_features(
        set_ids=[set_id_inherit, set_id_override],
    )
    assert len(remaining_features) == 1
    assert remaining_features[0].set_id == set_id_override
    assert remaining_features[0].feature_name == "override_feature"

    # Verify the local category still exists
    local_category = await semantic_service.get_category(category_id=local_category_id)
    assert local_category is not None
    assert local_category.name == "OverridableCategory"


async def test_add_new_feature_validates_category_exists(
    semantic_service: SemanticService,
):
    """Test that adding a feature to a non-existent category raises CategoryNotFoundError."""
    from memmachine.common.errors import CategoryNotFoundError

    set_id = "user-validate-category"

    # Try to add a feature to a category that doesn't exist
    with pytest.raises(CategoryNotFoundError) as exc_info:
        await semantic_service.add_new_feature(
            set_id=set_id,
            category_name="NonExistentCategory",
            feature="test_feature",
            value="test value",
            tag="test_tag",
        )

    # Verify the error message contains the correct information
    assert exc_info.value.set_id == set_id
    assert exc_info.value.category_name == "NonExistentCategory"
    assert "does not exist" in str(exc_info.value)


async def test_add_new_feature_succeeds_with_valid_category(
    semantic_service: SemanticService,
):
    """Test that adding a feature to a valid category succeeds."""
    set_id = "user-valid-category"

    # Create a category
    await semantic_service.add_new_category_to_set_id(
        set_id=set_id,
        category_name="ValidCategory",
        prompt="Test category prompt",
        description="Test description",
    )

    # Add a feature to the valid category - should succeed
    feature_id = await semantic_service.add_new_feature(
        set_id=set_id,
        category_name="ValidCategory",
        feature="test_feature",
        value="test value",
        tag="test_tag",
    )

    assert feature_id is not None

    # Verify the feature was created
    features = await semantic_service.get_set_features(set_ids=[set_id])
    assert len(features) == 1
    assert features[0].category == "ValidCategory"


async def test_update_feature_validates_category_exists(
    semantic_service: SemanticService,
):
    """Test that updating a feature to a non-existent category raises CategoryNotFoundError."""
    from memmachine.common.errors import CategoryNotFoundError

    set_id = "user-update-validate"

    # Create a category and feature
    await semantic_service.add_new_category_to_set_id(
        set_id=set_id,
        category_name="OriginalCategory",
        prompt="Original prompt",
        description="Original description",
    )

    feature_id = await semantic_service.add_new_feature(
        set_id=set_id,
        category_name="OriginalCategory",
        feature="test_feature",
        value="test value",
        tag="test_tag",
    )

    # Try to update the feature to a non-existent category
    with pytest.raises(CategoryNotFoundError) as exc_info:
        await semantic_service.update_feature(
            feature_id=feature_id,
            category_name="NonExistentCategory",
        )

    # Verify the error message contains the correct information
    assert exc_info.value.set_id == set_id
    assert exc_info.value.category_name == "NonExistentCategory"


async def test_update_feature_succeeds_with_valid_category(
    semantic_service: SemanticService,
):
    """Test that updating a feature to a valid category succeeds."""
    set_id = "user-update-valid"

    # Create two categories
    await semantic_service.add_new_category_to_set_id(
        set_id=set_id,
        category_name="Category1",
        prompt="Category 1 prompt",
        description="Category 1 description",
    )
    await semantic_service.add_new_category_to_set_id(
        set_id=set_id,
        category_name="Category2",
        prompt="Category 2 prompt",
        description="Category 2 description",
    )

    # Create a feature in Category1
    feature_id = await semantic_service.add_new_feature(
        set_id=set_id,
        category_name="Category1",
        feature="test_feature",
        value="test value",
        tag="test_tag",
    )

    # Update the feature to Category2 - should succeed
    await semantic_service.update_feature(
        feature_id=feature_id,
        category_name="Category2",
    )

    # Verify the feature was updated
    feature = await semantic_service.get_feature(feature_id, load_citations=False)
    assert feature is not None
    assert feature.category == "Category2"
