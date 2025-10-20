"""Unit tests for the profile_memory module."""

import asyncio
import time
from unittest.mock import create_autospec

import pytest
import pytest_asyncio

from memmachine.common.embedder import Embedder
from memmachine.common.language_model import LanguageModel
from memmachine.semantic_memory.prompt_provider import SemanticPrompt
from memmachine.semantic_memory.semantic_memory import (
    SemanticMemory,
    SemanticMemoryParams,
    SemanticUpdateTracker,
    SemanticUpdateTrackerManager,
)
from memmachine.semantic_memory.storage.storage_base import SemanticStorageBase
from tests.memmachine.common.reranker.test_embedder_reranker import FakeEmbedder
from tests.memmachine.semantic_memory.storage.in_memory_profile_storage import (
    InMemorySemanticStorage,
)

# Mark all tests in this file as asyncio
pytestmark = pytest.mark.asyncio


@pytest.fixture
def profile_tracker():
    return SemanticUpdateTracker("a", message_limit=2, time_limit_sec=0.1)


def test_profile_tracker_expires(profile_tracker):
    assert not profile_tracker.should_update()
    profile_tracker.mark_update()
    assert not profile_tracker.should_update()
    time.sleep(0.15)
    assert profile_tracker.should_update()


def test_profile_tracker_message_limit(profile_tracker):
    assert not profile_tracker.should_update()
    profile_tracker.mark_update()
    assert not profile_tracker.should_update()
    profile_tracker.mark_update()
    assert profile_tracker.should_update()
    profile_tracker.reset()
    assert not profile_tracker.should_update()


@pytest.fixture
def profile_update_tracker_manager():
    return SemanticUpdateTrackerManager(message_limit=2, time_limit_sec=0.1)


async def test_profile_update_tracker_manager_with_message_limit(
    profile_update_tracker_manager,
):
    users = await profile_update_tracker_manager.get_users_to_update()
    assert users == []

    for user in ["a", "b", "a", "a"]:
        await profile_update_tracker_manager.mark_update(user)

    users = await profile_update_tracker_manager.get_users_to_update()
    assert set(users) == {"a"}

    for user in ["b", "a"]:
        await profile_update_tracker_manager.mark_update(user)
    users = await profile_update_tracker_manager.get_users_to_update()
    assert set(users) == {"b"}


async def test_profile_update_tracker_manager_with_time_limit(
    profile_update_tracker_manager,
):
    users = await profile_update_tracker_manager.get_users_to_update()
    assert users == []

    await profile_update_tracker_manager.mark_update("a")
    await profile_update_tracker_manager.mark_update("b")
    users = await profile_update_tracker_manager.get_users_to_update()
    assert users == []

    time.sleep(0.15)
    users = await profile_update_tracker_manager.get_users_to_update()
    assert set(users) == {"a", "b"}


@pytest.fixture
def mock_embedder():
    embedder = FakeEmbedder()
    yield embedder


@pytest.fixture
def mock_llm():
    languange_model = create_autospec(LanguageModel, instance=True)
    yield languange_model


@pytest.fixture
def mock_prompt():
    prompt = SemanticPrompt(
        update_prompt="mock_update_prompt",
        consolidation_prompt="mock_consolidation_prompt",
    )
    yield prompt


@pytest.fixture
def mock_storage():
    storage = InMemorySemanticStorage()
    yield storage


@pytest_asyncio.fixture
async def semantic_memory(
    mock_embedder: Embedder,
    mock_llm: LanguageModel,
    mock_prompt: SemanticPrompt,
    mock_storage: SemanticStorageBase,
):
    params = SemanticMemoryParams(
        model=mock_llm,
        embeddings=mock_embedder,
        prompt=mock_prompt,
        semantic_storage=mock_storage,
        feature_update_interval_sec=0.1,
        feature_update_message_limit=1,
        feature_update_time_limit_sec=0.1,
    )
    pm = SemanticMemory(params=params)
    await pm.startup()
    yield pm
    await pm.delete_all()
    await pm.cleanup()


@pytest_asyncio.fixture
async def single_feature_profile_response(semantic_memory):
    await semantic_memory.add_new_profile(
        user_id="test_user",
        feature="test_feature",
        value="test_value",
        tag="test_tag",
        metadata={"test_metadata": "test_metadata_value"},
    )

    yield {
        "test_tag": {
            "test_feature": {
                "value": "test_value",
            }
        }
    }

    await semantic_memory.delete_user_profile_feature(
        user_id="test_user",
        feature="test_feature",
        tag="test_tag",
    )


async def test_store_and_get_profile(
    semantic_memory: SemanticMemory, single_feature_profile_response
):
    # Given a profile with a single user
    # When we retrieve the profile
    profile = await semantic_memory.get_user_profile(
        user_id="test_user",
    )

    # Expect the profile to contain the feature
    assert profile == single_feature_profile_response


@pytest_asyncio.fixture
async def multiple_feature_profile_response(semantic_memory: SemanticMemory):
    await semantic_memory.add_new_profile(
        user_id="test_user",
        feature="test_feature_a",
        value="test_value_a",
        tag="test_tag_a",
    )
    await semantic_memory.add_new_profile(
        user_id="test_user",
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

    await semantic_memory.delete_user_profile_feature(
        user_id="test_user",
        feature="test_feature_a",
        tag="test_tag_a",
    )
    await semantic_memory.delete_user_profile_feature(
        user_id="test_user",
        feature="test_feature_b",
        tag="test_tag_b",
    )


async def test_multiple_features(
    semantic_memory: SemanticMemory, multiple_feature_profile_response
):
    # Given a profile with two features
    # When we retrieve the profile
    profile = await semantic_memory.get_user_profile(
        user_id="test_user",
    )

    # Expect the profile to contain both features
    assert profile == multiple_feature_profile_response


async def test_delete_feature(
    semantic_memory: SemanticMemory, multiple_feature_profile_response
):
    # Given a user profile with feature 'a' and 'b'
    profile = await semantic_memory.get_user_profile(
        user_id="test_user",
    )
    assert profile == multiple_feature_profile_response
    assert "test_tag_a" in profile
    assert "test_tag_b" in profile

    # When deleting feature 'a'
    await semantic_memory.delete_user_profile_feature(
        user_id="test_user",
        feature="test_feature_a",
        tag="test_tag_a",
    )

    # Expect feature 'a' to no longer exist. While feature 'b' still exists.
    profile = await semantic_memory.get_user_profile(
        user_id="test_user",
    )
    assert "test_tag_a" not in profile
    assert "test_tag_b" in profile

    del multiple_feature_profile_response["test_tag_a"]
    assert profile == multiple_feature_profile_response


async def test_delete_profile(semantic_memory, single_feature_profile_response):
    # Given a profile
    profile = await semantic_memory.get_user_profile(
        user_id="test_user",
    )
    assert profile == single_feature_profile_response

    # When we delete the profile
    await semantic_memory.delete_user_profile(
        user_id="test_user",
    )

    # Then the profile should no longer exist
    profile = await semantic_memory.get_user_profile(
        user_id="test_user",
    )
    assert profile == {}


async def test_add_persona_message_with_speaker_metadata(semantic_memory):
    """Ensure persona messages store speaker metadata and trigger updates."""
    await semantic_memory.add_persona_message(
        content="My dog is pretty",
        user_id="test_user",
        metadata={"speaker": "User"},
    )

    history = await semantic_memory._semantic_storage.get_ingested_history_messages(
        user_id="test_user",
        k=1,
    )

    assert history
    assert history[0]["content"] == "User sends 'My dog is pretty'"


@pytest_asyncio.fixture
async def mock_persona_think_response(mock_llm, semantic_memory: SemanticMemory):
    mock_llm.generate_response.return_value = (
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
        user_id="test_user",
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


async def test_persona_think_updates_profile(
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

    profile = await semantic_memory.get_user_profile(
        user_id="test_user",
    )

    assert profile == mock_persona_think_response
