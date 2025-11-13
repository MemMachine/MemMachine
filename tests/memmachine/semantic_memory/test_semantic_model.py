"""Unit tests for semantic model classes and their methods."""

from datetime import UTC
from types import ModuleType

import pytest

from memmachine.history_store.history_model import HistoryIdT
from memmachine.semantic_memory.semantic_model import (
    HistoryMessage,
    RawSemanticPrompt,
    SemanticFeature,
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
            )
        ]
        grouped = SemanticFeature.group_features(features)

        assert len(grouped) == 1
        key = ("Profile", "hobby", "activity")
        assert key in grouped
        assert len(grouped[key]) == 1
        assert grouped[key][0].value == "reading"

    def test_group_features_by_tag(self, sample_features):
        grouped = SemanticFeature.group_features_by_tag(sample_features)

        # Should have 3 unique groups: (food, favorite_pizza), (food, favorite_drink), (music, favorite_genre)
        assert len(grouped) == 3

        # Check (food, favorite_pizza) group - should include all types
        food_pizza_key = ("food", "favorite_pizza")
        assert food_pizza_key in grouped
        assert len(grouped[food_pizza_key]) == 3  # pepperoni, margherita, hawaiian
        values = {f.value for f in grouped[food_pizza_key]}
        assert values == {"pepperoni", "margherita", "hawaiian"}

        # Check (food, favorite_drink) group
        food_drink_key = ("food", "favorite_drink")
        assert food_drink_key in grouped
        assert len(grouped[food_drink_key]) == 1
        assert grouped[food_drink_key][0].value == "water"

        # Check (music, favorite_genre) group
        music_genre_key = ("music", "favorite_genre")
        assert music_genre_key in grouped
        assert len(grouped[music_genre_key]) == 1
        assert grouped[music_genre_key][0].value == "jazz"

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
            )
        ]
        grouped = SemanticFeature.group_features_by_tag(features)

        assert len(grouped) == 1
        key = ("color", "favorite")
        assert key in grouped
        assert len(grouped[key]) == 1
        assert grouped[key][0].value == "blue"


class TestRawSemanticPrompt:
    """Tests for RawSemanticPrompt class."""

    def test_load_from_module_with_both_prompts(self):
        # Create a mock module with both prompts
        mock_module = ModuleType("mock_prompts")
        mock_module.UPDATE_PROMPT = "Update the profile"
        mock_module.CONSOLIDATION_PROMPT = "Consolidate memories"

        prompt = RawSemanticPrompt.load_from_module(mock_module)

        assert prompt.update_prompt == "Update the profile"
        assert prompt.consolidation_prompt == "Consolidate memories"

    def test_load_from_module_missing_update_prompt(self):
        mock_module = ModuleType("mock_prompts")
        mock_module.CONSOLIDATION_PROMPT = "Consolidate memories"

        prompt = RawSemanticPrompt.load_from_module(mock_module)

        assert prompt.update_prompt == ""
        assert prompt.consolidation_prompt == "Consolidate memories"

    def test_load_from_module_missing_consolidation_prompt(self):
        mock_module = ModuleType("mock_prompts")
        mock_module.UPDATE_PROMPT = "Update the profile"

        prompt = RawSemanticPrompt.load_from_module(mock_module)

        assert prompt.update_prompt == "Update the profile"
        assert prompt.consolidation_prompt == ""

    def test_load_from_module_missing_both_prompts(self):
        mock_module = ModuleType("mock_prompts")

        prompt = RawSemanticPrompt.load_from_module(mock_module)

        assert prompt.update_prompt == ""
        assert prompt.consolidation_prompt == ""

    def test_load_from_module_with_other_attributes(self):
        mock_module = ModuleType("mock_prompts")
        mock_module.UPDATE_PROMPT = "Update prompt"
        mock_module.CONSOLIDATION_PROMPT = "Consolidation prompt"
        mock_module.SOME_OTHER_ATTRIBUTE = "ignored"
        mock_module.ANOTHER_THING = 42

        prompt = RawSemanticPrompt.load_from_module(mock_module)

        assert prompt.update_prompt == "Update prompt"
        assert prompt.consolidation_prompt == "Consolidation prompt"


class TestHistoryMessage:
    """Tests for HistoryMessage model."""

    def test_history_message_with_minimal_fields(self):
        from datetime import datetime

        now = datetime.now(UTC)
        msg = HistoryMessage(
            content="Test message",
            created_at=now,
        )

        assert msg.content == "Test message"
        assert msg.created_at == now
        assert msg.metadata.id is None
        assert msg.metadata.other is None

    def test_history_message_with_metadata(self):
        from datetime import datetime

        now = datetime.now(UTC)
        msg = HistoryMessage(
            content="Test message",
            created_at=now,
            metadata=HistoryMessage.Metadata(
                id=HistoryIdT(123),
                other={"source": "test", "priority": "high"},
            ),
        )

        assert msg.content == "Test message"
        assert msg.created_at == now
        assert msg.metadata.id == "123"
        assert msg.metadata.other == {"source": "test", "priority": "high"}


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
        citation = HistoryMessage(
            content="I love pasta",
            created_at=now,
            metadata=HistoryMessage.Metadata(id="456aw3w"),
        )

        feature = SemanticFeature(
            set_id="user-123",
            category="Profile",
            tag="food",
            feature_name="favorite_meal",
            value="pasta",
            metadata=SemanticFeature.Metadata(
                id="a789",
                citations=[citation.metadata.id],
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
