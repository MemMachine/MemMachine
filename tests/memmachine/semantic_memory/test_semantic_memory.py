"""Unit tests for the profile_memory module."""

import asyncio

import pytest
import pytest_asyncio

from memmachine.common.embedder import Embedder
from memmachine.common.language_model import LanguageModel
from memmachine.semantic_memory.semantic_memory import (
    SemanticMemoryManager,
    SemanticMemoryManagerParams,
)
from memmachine.semantic_memory.semantic_prompt import SemanticPrompt
from memmachine.semantic_memory.storage.storage_base import SemanticStorageBase
from tests.memmachine.common.reranker.fake_embedder import FakeEmbedder
from tests.memmachine.semantic_memory.storage.in_memory_semantic_storage import (
    InMemorySemanticStorage,
)

# Mark all tests in this file as asyncio
pytestmark = pytest.mark.asyncio


@pytest.fixture
def llm_embedder(mock_llm_embedder):
    return mock_llm_embedder


@pytest.fixture
def llm_model(mock_llm_model):
    return mock_llm_model


@pytest.fixture
def mock_prompt():
    prompt = SemanticPrompt(
        update_prompt="mock_update_prompt",
        consolidation_prompt="mock_consolidation_prompt",
    )
    yield prompt


@pytest.fixture(
    params=[pytest.param("postgres", marks=pytest.mark.integration), "inmemory"]
)
def storage():
    storage = InMemorySemanticStorage()
    yield storage


@pytest_asyncio.fixture
async def semantic_memory(
        llm_embedder: Embedder,
        llm_model: LanguageModel,
    mock_prompt: SemanticPrompt,
    storage: SemanticStorageBase,
):
    params = SemanticMemoryManagerParams(
        model=llm_model,
        embeddings=llm_embedder,
        prompt=mock_prompt,
        semantic_storage=storage,
        feature_update_interval_sec=0.1,
        feature_update_message_limit=1,
        feature_update_time_limit_sec=0.1,
    )
    pm = SemanticMemoryManager(params=params)
    yield pm
    await pm.delete_all()
    await pm.stop()


@pytest_asyncio.fixture
async def single_feature_profile_response(semantic_memory):
    await semantic_memory.add_new_feature(
        set_id="test_user",
        feature="test_feature",
        value="test_value",
        tag="test_tag",
        metadata={"test_metadata": "test_metadata_value"},
    )

    yield {
        "test_tag": {
            "test_feature": {
                "metadata": {"test_metadata": "test_metadata_value"},
                "value": "test_value",
            }
        }
    }

    await semantic_memory.delete_set_feature(
        set_id="test_user",
        feature="test_feature",
        tag="test_tag",
    )


async def test_store_and_get_profile(
    semantic_memory: SemanticMemoryManager, single_feature_profile_response
):
    # Given a profile with a single user
    # When we retrieve the profile
    profile = await semantic_memory.get_set_features(
        set_id="test_user",
    )

    # Expect the profile to contain the feature
    assert profile == single_feature_profile_response


@pytest_asyncio.fixture
async def multiple_feature_profile_response(semantic_memory: SemanticMemoryManager):
    await semantic_memory.add_new_feature(
        set_id="test_user",
        feature="test_feature_a",
        value="test_value_a",
        tag="test_tag_a",
    )
    await semantic_memory.add_new_feature(
        set_id="test_user",
        feature="test_feature_b",
        value="test_value_b",
        tag="test_tag_b",
    )

    yield {
        "test_tag_a": {
            "test_feature_a": {
                "value": "test_value_a",
            }
        },
        "test_tag_b": {
            "test_feature_b": {
                "value": "test_value_b",
            }
        },
    }

    await semantic_memory.delete_set_feature(
        set_id="test_user",
        feature="test_feature_a",
        tag="test_tag_a",
    )
    await semantic_memory.delete_set_feature(
        set_id="test_user",
        feature="test_feature_b",
        tag="test_tag_b",
    )


async def test_multiple_features(
    semantic_memory: SemanticMemoryManager, multiple_feature_profile_response
):
    # Given a profile with two features
    # When we retrieve the profile
    profile = await semantic_memory.get_set_features(
        set_id="test_user",
    )

    # Expect the profile to contain both features
    assert profile == multiple_feature_profile_response


async def test_delete_feature(
    semantic_memory: SemanticMemoryManager, multiple_feature_profile_response
):
    # Given a user profile with feature 'a' and 'b'
    profile = await semantic_memory.get_set_features(
        set_id="test_user",
    )
    assert profile == multiple_feature_profile_response
    assert "test_tag_a" in profile
    assert "test_tag_b" in profile

    # When deleting feature 'a'
    await semantic_memory.delete_set_feature(
        set_id="test_user",
        feature="test_feature_a",
        tag="test_tag_a",
    )

    # Expect feature 'a' to no longer exist. While feature 'b' still exists.
    profile = await semantic_memory.get_set_features(
        set_id="test_user",
    )
    assert "test_tag_a" not in profile
    assert "test_tag_b" in profile

    del multiple_feature_profile_response["test_tag_a"]
    assert profile == multiple_feature_profile_response


async def test_delete_profile(semantic_memory, single_feature_profile_response):
    # Given a profile
    profile = await semantic_memory.get_set_features(
        set_id="test_user",
    )
    assert profile == single_feature_profile_response

    # When we delete the profile
    await semantic_memory.delete_set_features(
        set_id="test_user",
    )

    # Then the profile should no longer exist
    profile = await semantic_memory.get_set_features(
        set_id="test_user",
    )
    assert profile == {}


async def test_add_persona_message_with_speaker_metadata(semantic_memory):
    """Ensure persona messages store speaker metadata and trigger updates."""
    await semantic_memory.add_persona_message(
        content="My dog is pretty",
        set_id="test_user",
        metadata={"speaker": "User"},
    )

    history = await semantic_memory._semantic_storage.get_ingested_history_messages(
        set_id="test_user",
        k=1,
    )

    assert history
    assert history[0]["content"] == "User sends 'My dog is pretty'"


@pytest_asyncio.fixture
async def mock_persona_think_response(llm_model, semantic_memory: SemanticMemoryManager):
    llm_model.generate_response.return_value = (
        """{
      "1": {
        "command": "add",
        "feature": "tone",
        "value": "casual and friendly, with conversational elements",
        "tag": "writing_style_general",
        "author": null
      },
      "2": {
        "command": "add",
        "feature": "register",
        "value": "casual, suitable for informal conversation among peers",
        "tag": "writing_style_general",
        "author": null
      },
      "3": {
        "command": "add",
        "feature": "voice",
        "value": "personal and approachable, with a relatable perspective",
        "tag": "writing_style_general",
        "author": null
      },
      "4": {
        "command": "add",
        "feature": "sentence_structure",
        "value": "simple and compound sentences, with an informal structure",
        "tag": "writing_style_general",
        "author": null
      },
      "5": {
        "command": "add",
        "feature": "clarity",
        "value": "direct and easy to understand, with clear references to experiences",
        "tag": "writing_style_general",
        "author": null
      },
      "6": {
        "command": "add",
        "feature": "self_reference",
        "value": "frequent use of first-person references and personal experiences",
        "tag": "writing_style_general",
        "author": null
      }
    }""",
        [],
    )

    await semantic_memory.add_persona_message(
        content="test_content",
        set_id="test_user",
        metadata={"speaker": "User"},
    )

    yield {
        "writing_style_general": {
            "tone": {
                "value": "casual and friendly, with conversational elements",
            },
            "register": {
                "value": "casual, suitable for informal conversation among peers",
            },
            "voice": {
                "value": "personal and approachable, with a relatable perspective",
            },
            "sentence_structure": {
                "value": "simple and compound sentences, with an informal structure",
            },
            "clarity": {
                "value": "direct and easy to understand, with clear references to experiences",
            },
            "self_reference": {
                "value": "frequent use of first-person references and personal experiences",
            },
        }
    }


async def test_persona_think_updates_features(
    semantic_memory, mock_persona_think_response
):
    count = -1
    for i in range(10):
        count = await semantic_memory.uningested_message_count()
        if count == 0:
            break
        await asyncio.sleep(0.1)
    if count != 0:
        pytest.fail("Messages are not ingested")

    features = await semantic_memory.get_set_features(
        set_id="test_user",
    )

    assert features == mock_persona_think_response
