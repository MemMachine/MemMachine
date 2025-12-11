from unittest.mock import AsyncMock, MagicMock

import pytest

from memmachine.common.language_model import LanguageModel
from memmachine.semantic_memory.semantic_llm import (
    llm_consolidate_features,
    llm_feature_update,
)
from memmachine.semantic_memory.semantic_model import (
    SemanticCommand,
    SemanticCommandType,
    SemanticFeature,
)


@pytest.fixture
def magic_mock_llm_model():
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
    magic_mock_llm_model: LanguageModel,
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
    magic_mock_llm_model: LanguageModel,
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
            command="add",
            tag="car",
            feature="favorite_car_color",
            value="blue",
        ),
    ]


@pytest.mark.asyncio
async def test_multiple_commands_update_response(
    magic_mock_llm_model: LanguageModel,
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
    magic_mock_llm_model: LanguageModel,
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
    magic_mock_llm_model: LanguageModel,
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
    magic_mock_llm_model: LanguageModel,
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
    magic_mock_llm_model: LanguageModel,
    basic_features: list[SemanticFeature],
):
    from memmachine.common.data_types import ExternalServiceAPIError

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
    magic_mock_llm_model: LanguageModel,
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
