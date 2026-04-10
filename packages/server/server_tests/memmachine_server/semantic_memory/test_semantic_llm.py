from unittest.mock import AsyncMock, MagicMock

import pytest

from memmachine_server.common.language_model import LanguageModel
from memmachine_server.semantic_memory.semantic_llm import (
    _features_to_consolidation_format,
    _features_to_llm_format,
    llm_consolidate_features,
    llm_feature_update,
)
from memmachine_server.semantic_memory.semantic_model import (
    SemanticCommand,
    SemanticCommandType,
    SemanticFeature,
)


@pytest.fixture
def magic_mock_llm_model() -> MagicMock:
    mock = MagicMock(spec=LanguageModel)
    mock.generate_parsed_response = AsyncMock()
    return mock


@pytest.fixture
def basic_features():
    return [
        SemanticFeature(
            category="Profile",
            tag="food",
            feature_name="favorite_pizza",
            value="peperoni pizza",
        ),
        SemanticFeature(
            category="Profile",
            tag="food",
            feature_name="favorite_bread",
            value="whole grain",
        ),
    ]


@pytest.mark.asyncio
async def test_empty_update_response(
    magic_mock_llm_model: MagicMock,
    basic_features: list[SemanticFeature],
):
    # Given an empty LLM response from the prompt
    magic_mock_llm_model.generate_parsed_response.return_value = {"commands": []}

    commands = await llm_feature_update(
        features=basic_features,
        message_content="I like blue cars",
        model=magic_mock_llm_model,
        update_prompt="Update features",
    )

    # Expect no commands to be returned
    assert commands == []


@pytest.mark.asyncio
async def test_single_command_update_response(
    magic_mock_llm_model: MagicMock,
    basic_features: list[SemanticFeature],
):
    # Given a single LLM response from the prompt
    magic_mock_llm_model.generate_parsed_response.return_value = {
        "commands": [
            {
                "command": "add",
                "tag": "car",
                "feature": "favorite_car_color",
                "value": "blue",
            },
        ],
    }

    commands = await llm_feature_update(
        features=basic_features,
        message_content="I like blue cars",
        model=magic_mock_llm_model,
        update_prompt="Update features",
    )

    assert commands == [
        SemanticCommand(
            command=SemanticCommandType.ADD,
            tag="car",
            feature="favorite_car_color",
            value="blue",
        ),
    ]


@pytest.mark.asyncio
async def test_multiple_commands_update_response(
    magic_mock_llm_model: MagicMock,
    basic_features: list[SemanticFeature],
):
    magic_mock_llm_model.generate_parsed_response.return_value = {
        "commands": [
            {
                "command": "add",
                "tag": "car",
                "feature": "favorite_car_color",
                "value": "blue",
            },
            {
                "command": "add",
                "tag": "car",
                "feature": "favorite_car",
                "value": "Tesla",
            },
        ],
    }

    commands = await llm_feature_update(
        features=basic_features,
        message_content="I like blue Tesla cars",
        model=magic_mock_llm_model,
        update_prompt="Update features",
    )

    assert len(commands) == 2
    assert commands[0].command == SemanticCommandType.ADD
    assert commands[0].feature == "favorite_car_color"
    assert commands[1].command == SemanticCommandType.ADD
    assert commands[1].feature == "favorite_car"


@pytest.mark.asyncio
async def test_empty_consolidate_response(
    magic_mock_llm_model: MagicMock,
    basic_features: list[SemanticFeature],
):
    magic_mock_llm_model.generate_parsed_response.return_value = {
        "consolidated_memories": [],
        "keep_memories": None,
    }

    new_feature_resp = await llm_consolidate_features(
        features=basic_features,
        model=magic_mock_llm_model,
        consolidate_prompt="Consolidate features",
    )

    assert new_feature_resp is not None
    assert new_feature_resp.consolidated_memories == []
    assert new_feature_resp.keep_memories is None


@pytest.mark.asyncio
async def test_no_action_consolidate_response(
    magic_mock_llm_model: MagicMock,
    basic_features: list[SemanticFeature],
):
    magic_mock_llm_model.generate_parsed_response.return_value = {
        "keep_memories": [],
        "consolidated_memories": [],
    }

    new_feature_resp = await llm_consolidate_features(
        features=basic_features,
        model=magic_mock_llm_model,
        consolidate_prompt="Consolidate features",
    )

    assert new_feature_resp is not None
    assert new_feature_resp.keep_memories == []
    assert new_feature_resp.consolidated_memories == []


@pytest.mark.asyncio
async def test_consolidate_with_valid_memories(
    magic_mock_llm_model: MagicMock,
    basic_features: list[SemanticFeature],
):
    magic_mock_llm_model.generate_parsed_response.return_value = {
        "keep_memories": [1, 2],
        "consolidated_memories": [
            {
                "tag": "food",
                "feature": "favorite_pizza",
                "value": "pepperoni",
            },
            {
                "tag": "food",
                "feature": "favorite_drink",
                "value": "water",
            },
        ],
    }

    result = await llm_consolidate_features(
        features=basic_features,
        model=magic_mock_llm_model,
        consolidate_prompt="Consolidate features",
    )

    assert result is not None
    assert result.keep_memories == ["1", "2"]
    assert len(result.consolidated_memories) == 2
    assert result.consolidated_memories[0].feature == "favorite_pizza"
    assert result.consolidated_memories[1].feature == "favorite_drink"


@pytest.mark.asyncio
async def test_llm_feature_update_handles_model_api_error(
    magic_mock_llm_model: MagicMock,
    basic_features: list[SemanticFeature],
):
    from memmachine_server.common.data_types import ExternalServiceAPIError

    # Given an LLM that raises API error
    magic_mock_llm_model.generate_parsed_response.side_effect = ExternalServiceAPIError(
        "API timeout",
    )

    with pytest.raises(ExternalServiceAPIError):
        await llm_feature_update(
            features=basic_features,
            message_content="I like blue cars",
            model=magic_mock_llm_model,
            update_prompt="Update features",
        )


@pytest.mark.asyncio
async def test_llm_feature_update_with_delete_command(
    magic_mock_llm_model: MagicMock,
    basic_features: list[SemanticFeature],
):
    magic_mock_llm_model.generate_parsed_response.return_value = {
        "commands": [
            {
                "command": "delete",
                "tag": "food",
                "feature": "favorite_pizza",
                "value": "",
            },
        ],
    }

    commands = await llm_feature_update(
        features=basic_features,
        message_content="I don't like pizza anymore",
        model=magic_mock_llm_model,
        update_prompt="Update features",
    )

    assert len(commands) == 1
    assert commands[0].command == SemanticCommandType.DELETE
    assert commands[0].feature == "favorite_pizza"


class TestConsolidationSerialization:
    """Consolidation needs a separate serializer that includes feature IDs
    so the LLM can reference them in ``keep_memories``."""

    @pytest.fixture
    def features_with_ids(self):
        return [
            SemanticFeature(
                category="CodeKnowledge",
                tag="bugfix",
                feature_name="observer_fix",
                value="Fixed observer subagent bug",
                metadata=SemanticFeature.Metadata(id="42"),
            ),
            SemanticFeature(
                category="CodeKnowledge",
                tag="progress",
                feature_name="more_agents",
                value="User added more agents",
                metadata=SemanticFeature.Metadata(id="43"),
            ),
        ]

    def test_update_format_omits_ids(self, features_with_ids):
        """The update serializer intentionally omits IDs — updates don't
        need them."""
        import json

        formatted = _features_to_llm_format(features_with_ids)
        serialized = json.dumps(formatted)

        assert "42" not in serialized
        assert "43" not in serialized

    def test_consolidation_format_includes_ids(self, features_with_ids):
        """The consolidation serializer must include ``metadata.id`` so
        the LLM can return them in ``keep_memories``."""
        import json

        formatted = _features_to_consolidation_format(features_with_ids)
        serialized = json.dumps(formatted)

        assert "42" in serialized
        assert "43" in serialized
        assert "metadata" in serialized

    def test_consolidation_format_preserves_all_fields(self, features_with_ids):
        """Each entry in the consolidation format should have tag, feature,
        value, and metadata.id."""
        formatted = _features_to_consolidation_format(features_with_ids)

        assert len(formatted) == 2
        entry = formatted[0]
        assert entry["tag"] == "bugfix"
        assert entry["feature"] == "observer_fix"
        assert entry["value"] == "Fixed observer subagent bug"
        assert entry["metadata"] == {"id": "42"}
