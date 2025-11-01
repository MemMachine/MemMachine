from unittest.mock import MagicMock

import pytest

from memmachine.common.language_model import LanguageModel
from memmachine.semantic_memory.semantic_llm import (
    llm_consolidate_features,
    llm_feature_update,
)
from memmachine.semantic_memory.semantic_model import SemanticCommand, SemanticFeature


@pytest.fixture
def magic_mock_llm_model():
    yield MagicMock(spec=LanguageModel)


@pytest.fixture
def basic_features():
    return [
        SemanticFeature(
            type="Profile",
            tag="food",
            feature="favorite_pizza",
            value="peperoni pizza",
        ),
        SemanticFeature(
            type="Profile",
            tag="food",
            feature="favorite_bread",
            value="whole grain",
        ),
    ]


@pytest.mark.asyncio
async def test_empty_update_response(
    magic_mock_llm_model: LanguageModel,
    basic_features: list[SemanticFeature],
):
    # Given an empty LLM response from the prompt
    magic_mock_llm_model.generate_response.return_value = ("{}", None)

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
    magic_mock_llm_model.generate_response.return_value = (
        """
        {
            "0": {
                "command": "add",
                "tag": "car",
                "feature": "favorite_car_color",
                "value": "blue"
            }
        }
        """,
        None,
    )

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
        )
    ]


@pytest.mark.asyncio
async def test_single_command_with_think_update_response(
    magic_mock_llm_model: LanguageModel,
    basic_features: list[SemanticFeature],
):
    magic_mock_llm_model.generate_response.return_value = (
        """
        <think>
        I am thinking
        </think>
        {
            "0": {
                "command": "add",
                "tag": "car",
                "feature": "favorite_car_color",
                "value": "blue"
            }
        }
        """,
        None,
    )

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
        )
    ]


@pytest.mark.asyncio
async def test_empty_consolidate_response(
    magic_mock_llm_model: LanguageModel,
    basic_features: list[SemanticFeature],
):
    magic_mock_llm_model.generate_response.return_value = ("{}", None)

    new_feature_resp = await llm_consolidate_features(
        features=basic_features,
        model=magic_mock_llm_model,
        consolidate_prompt="Consolidate features",
    )

    assert new_feature_resp is None


@pytest.mark.asyncio
async def test_no_action_consolidate_response(
    magic_mock_llm_model: LanguageModel,
    basic_features: list[SemanticFeature],
):
    magic_mock_llm_model.generate_response.return_value = (
        """
        {
            "keep_memories": [],
            "consolidated_memories": []
        }
        """,
        None,
    )

    new_feature_resp = await llm_consolidate_features(
        features=basic_features,
        model=magic_mock_llm_model,
        consolidate_prompt="Consolidate features",
    )

    assert new_feature_resp is not None
    assert new_feature_resp.keep_memories == []
    assert new_feature_resp.consolidated_memories == []
