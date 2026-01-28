"""Unit tests for semantic model classes and their methods."""

from datetime import UTC

import pytest

from memmachine.common.episode_store import Episode
from memmachine.semantic_memory.semantic_model import (
    SemanticFeature,
    StructuredSemanticPrompt,
)


class TestSemanticFeatureGrouping:
    """Tests for SemanticFeature grouping methods."""

    @pytest.fixture
    def sample_features(self):
        return [
            SemanticFeature(
                category="Profile",
                tag="food",
                feature_name="favorite_pizza",
                value="pepperoni",
            ),
            SemanticFeature(
                category="Profile",
                tag="food",
                feature_name="favorite_pizza",
                value="margherita",
            ),
            SemanticFeature(
                category="Profile",
                tag="food",
                feature_name="favorite_drink",
                value="water",
            ),
            SemanticFeature(
                category="Preferences",
                tag="food",
                feature_name="favorite_pizza",
                value="hawaiian",
            ),
            SemanticFeature(
                category="Profile",
                tag="music",
                feature_name="favorite_genre",
                value="jazz",
            ),
        ]

    def test_group_features_by_type_tag_feature(self, sample_features):
        grouped = SemanticFeature.group_features(sample_features)

        # Should have 4 unique groups: (Profile, food, favorite_pizza), (Profile, food, favorite_drink),
        # (Preferences, food, favorite_pizza), (Profile, music, favorite_genre)
        assert len(grouped) == 4

        # Check (Profile, food, favorite_pizza) group has 2 items
        profile_food_pizza_key = ("Profile", "food", "favorite_pizza")
        assert profile_food_pizza_key in grouped
        assert len(grouped[profile_food_pizza_key]) == 2
        assert grouped[profile_food_pizza_key][0].value == "pepperoni"
        assert grouped[profile_food_pizza_key][1].value == "margherita"

        # Check (Profile, food, favorite_drink) group has 1 item
        profile_food_drink_key = ("Profile", "food", "favorite_drink")
        assert profile_food_drink_key in grouped
        assert len(grouped[profile_food_drink_key]) == 1
        assert grouped[profile_food_drink_key][0].value == "water"

        # Check (Preferences, food, favorite_pizza) group has 1 item
        preferences_food_pizza_key = ("Preferences", "food", "favorite_pizza")
        assert preferences_food_pizza_key in grouped
        assert len(grouped[preferences_food_pizza_key]) == 1
        assert grouped[preferences_food_pizza_key][0].value == "hawaiian"

        # Check (Profile, music, favorite_genre) group has 1 item
        profile_music_genre_key = ("Profile", "music", "favorite_genre")
        assert profile_music_genre_key in grouped
        assert len(grouped[profile_music_genre_key]) == 1
        assert grouped[profile_music_genre_key][0].value == "jazz"

    def test_group_features_empty_list(self):
        grouped = SemanticFeature.group_features([])
        assert grouped == {}

    def test_group_features_single_item(self):
        features = [
            SemanticFeature(
                category="Profile",
                tag="hobby",
                feature_name="activity",
                value="reading",
            ),
        ]
        grouped = SemanticFeature.group_features(features)

        assert len(grouped) == 1
        key = ("Profile", "hobby", "activity")
        assert key in grouped
        assert len(grouped[key]) == 1
        assert grouped[key][0].value == "reading"

    def test_group_features_by_tag(self, sample_features):
        grouped = SemanticFeature.group_features_by_tag(sample_features)

        # Should have 2 unique groups: food and music
        assert len(grouped) == 2

        # Check food group - should include all food-related entries
        food_key = "food"
        assert food_key in grouped
        assert len(grouped[food_key]) == 4
        values = {f.value for f in grouped[food_key]}
        assert values == {"pepperoni", "margherita", "hawaiian", "water"}

        # Check music group
        music_key = "music"
        assert music_key in grouped
        assert len(grouped[music_key]) == 1
        assert grouped[music_key][0].value == "jazz"

    def test_group_features_by_tag_empty_list(self):
        grouped = SemanticFeature.group_features_by_tag([])
        assert grouped == {}

    def test_group_features_by_tag_single_item(self):
        features = [
            SemanticFeature(
                category="Profile",
                tag="color",
                feature_name="favorite",
                value="blue",
            ),
        ]
        grouped = SemanticFeature.group_features_by_tag(features)

        assert len(grouped) == 1
        key = "color"
        assert key in grouped
        assert len(grouped[key]) == 1
        assert grouped[key][0].value == "blue"


class TestHistoryMessage:
    """Tests for HistoryMessage model."""

    def test_history_message_with_minimal_fields(self):
        from datetime import datetime

        now = datetime.now(UTC)
        msg = Episode(
            uid="123",
            content="Test message",
            created_at=now,
            session_key="session_key",
            producer_id="profile_id",
            producer_role="user_role",
        )

        assert msg.content == "Test message"
        assert msg.created_at == now
        assert msg.uid == "123"
        assert msg.metadata is None

    def test_history_message_with_metadata(self):
        from datetime import datetime

        now = datetime.now(UTC)
        msg = Episode(
            content="Test message",
            created_at=now,
            uid="123",
            metadata={"source": "test", "priority": "high"},
            session_key="session_key",
            producer_id="profile_id",
            producer_role="user_role",
        )

        assert msg.content == "Test message"
        assert msg.created_at == now
        assert msg.uid == "123"
        assert msg.metadata == {"source": "test", "priority": "high"}


class TestSemanticFeature:
    """Tests for SemanticFeature model."""

    def test_semantic_feature_with_minimal_fields(self):
        feature = SemanticFeature(
            category="Profile",
            tag="food",
            feature_name="favorite_meal",
            value="pasta",
        )

        assert feature.category == "Profile"
        assert feature.tag == "food"
        assert feature.feature_name == "favorite_meal"
        assert feature.value == "pasta"
        assert feature.set_id is None
        assert feature.metadata.id is None
        assert feature.metadata.citations is None
        assert feature.metadata.other is None

    def test_semantic_feature_with_all_fields(self):
        from datetime import datetime

        now = datetime.now(UTC)
        citation = Episode(
            content="I love pasta",
            created_at=now,
            uid="456aw3w",
            session_key="session_key",
            producer_id="profile_id",
            producer_role="user_role",
        )

        feature = SemanticFeature(
            set_id="user-123",
            category="Profile",
            tag="food",
            feature_name="favorite_meal",
            value="pasta",
            metadata=SemanticFeature.Metadata(
                id="a789",
                citations=[citation.uid],
                other={"confidence": 0.95},
            ),
        )

        assert feature.set_id == "user-123"
        assert feature.category == "Profile"
        assert feature.tag == "food"
        assert feature.feature_name == "favorite_meal"
        assert feature.value == "pasta"
        assert feature.metadata.id == "a789"
        assert len(feature.metadata.citations) == 1
        assert feature.metadata.citations[0] == "456aw3w"
        assert feature.metadata.other == {"confidence": 0.95}


class TestStructuredSemanticPrompt:
    """Tests for StructuredSemanticPrompt prompt construction."""

    def test_update_prompt_includes_tags_and_description(self):
        prompt = StructuredSemanticPrompt(
            tags={
                "Profile": "Details about the user",
                "Preferences": "User likes",
            },
            description="Extra context",
        )

        built_prompt = prompt.update_prompt

        assert "Profile" in built_prompt
        assert "Details about the user" in built_prompt
        assert "Preferences" in built_prompt
        assert "User likes" in built_prompt
        assert "Extra context" in built_prompt
        assert "command" in built_prompt  # sanity check on template body

    def test_update_prompt_with_empty_description(self):
        prompt = StructuredSemanticPrompt(
            tags={"Preferences": "User likes"},
            description="",
        )

        built_prompt = prompt.update_prompt

        assert "Preferences" in built_prompt
        assert "User likes" in built_prompt
        # empty description should not break template and should still include command schema
        assert "command" in built_prompt

    def test_consolidation_prompt_matches_template(self):
        prompt = StructuredSemanticPrompt(tags={"Tag": "Description"})

        built_prompt = prompt.consolidation_prompt

        assert "memory consolidation" in built_prompt
        assert "keep_memories" in built_prompt
        assert "consolidate_memories" in built_prompt
