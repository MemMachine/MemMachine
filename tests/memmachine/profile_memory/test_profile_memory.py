"""Unit tests for the profile_memory module."""

import asyncio
import time
from unittest.mock import create_autospec

import pytest
import pytest_asyncio

from memmachine.common.embedder import Embedder
from memmachine.common.language_model import LanguageModel
from memmachine.profile_memory.profile_memory import (
    ProfileMemory,
    ProfileUpdateTracker,
    ProfileUpdateTrackerManager,
)
from memmachine.profile_memory.prompt_provider import ProfilePrompt
from memmachine.profile_memory.storage.storage_base import ProfileStorageBase
from tests.memmachine.common.reranker.test_embedder_reranker import FakeEmbedder
from tests.memmachine.profile_memory.storage.in_memory_profile_storage import (
    InMemoryProfileStorage,
)

# Mark all tests in this file as asyncio
pytestmark = pytest.mark.asyncio


@pytest.fixture
def profile_tracker():
    return ProfileUpdateTracker("a", message_limit=2, time_limit_sec=0.1)


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
    return ProfileUpdateTrackerManager(message_limit=2, time_limit_sec=0.1)


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
    prompt = ProfilePrompt(
        update_prompt="mock_update_prompt",
        consolidation_prompt="mock_consolidation_prompt",
    )
    yield prompt


@pytest.fixture
def mock_storage():
    storage = InMemoryProfileStorage()
    yield storage


@pytest_asyncio.fixture
async def profile_memory(
    mock_embedder: Embedder,
    mock_llm: LanguageModel,
    mock_prompt: ProfilePrompt,
    mock_storage: ProfileStorageBase,
):
    ProfileMemory.PROFILE_UPDATE_INTERVAL_SEC = 0.1
    ProfileMemory.PROFILE_UPDATE_TRACKER_INTERVAL_SEC = 0.01
    ProfileMemory.PROFILE_UPDATE_MESSAGE_LIMIT = 1
    pm = ProfileMemory(
        model=mock_llm,
        embeddings=mock_embedder,
        prompt=mock_prompt,
        profile_storage=mock_storage,
    )
    await pm.startup()
    yield pm
    await pm.delete_all()
    await pm.cleanup()


@pytest_asyncio.fixture
async def single_feature_profile_response(profile_memory):
    await profile_memory.add_new_profile(
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

    await profile_memory.delete_user_profile_feature(
        user_id="test_user",
        feature="test_feature",
        tag="test_tag",
    )


async def test_store_and_get_profile(
    profile_memory: ProfileMemory, single_feature_profile_response
):
    # Given a profile with a single user
    # When we retrieve the profile
    profile = await profile_memory.get_user_profile(
        user_id="test_user",
    )

    # Expect the profile to contain the feature
    assert profile == single_feature_profile_response


@pytest_asyncio.fixture
async def multiple_feature_profile_response(profile_memory: ProfileMemory):
    await profile_memory.add_new_profile(
        user_id="test_user",
        feature="test_feature_a",
        value="test_value_a",
        tag="test_tag_a",
    )
    await profile_memory.add_new_profile(
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

    await profile_memory.delete_user_profile_feature(
        user_id="test_user",
        feature="test_feature_a",
        tag="test_tag_a",
    )
    await profile_memory.delete_user_profile_feature(
        user_id="test_user",
        feature="test_feature_b",
        tag="test_tag_b",
    )


async def test_multiple_features(
    profile_memory: ProfileMemory, multiple_feature_profile_response
):
    # Given a profile with two features
    # When we retrieve the profile
    profile = await profile_memory.get_user_profile(
        user_id="test_user",
    )

    # Expect the profile to contain both features
    assert profile == multiple_feature_profile_response


async def test_delete_feature(
    profile_memory: ProfileMemory, multiple_feature_profile_response
):
    # Given a user profile with feature 'a' and 'b'
    profile = await profile_memory.get_user_profile(
        user_id="test_user",
    )
    assert profile == multiple_feature_profile_response
    assert "test_tag_a" in profile
    assert "test_tag_b" in profile

    # When deleting feature 'a'
    await profile_memory.delete_user_profile_feature(
        user_id="test_user",
        feature="test_feature_a",
        tag="test_tag_a",
    )

    # Expect feature 'a' to no longer exist. While feature 'b' still exists.
    profile = await profile_memory.get_user_profile(
        user_id="test_user",
    )
    assert "test_tag_a" not in profile
    assert "test_tag_b" in profile

    del multiple_feature_profile_response["test_tag_a"]
    assert profile == multiple_feature_profile_response


async def test_delete_profile(profile_memory, single_feature_profile_response):
    # Given a profile
    profile = await profile_memory.get_user_profile(
        user_id="test_user",
    )
    assert profile == single_feature_profile_response

    # When we delete the profile
    await profile_memory.delete_user_profile(
        user_id="test_user",
    )

    # Then the profile should no longer exist
    profile = await profile_memory.get_user_profile(
        user_id="test_user",
    )
    assert profile == {}


async def test_add_persona_message_with_speaker_metadata(profile_memory):
    """Ensure persona messages store speaker metadata and trigger updates."""
    await profile_memory.add_persona_message(
        content="My dog is pretty",
        user_id="test_user",
        metadata={"speaker": "User"},
    )

    history = await profile_memory._profile_storage.get_ingested_history_messages(
        user_id="test_user",
        k=1,
    )

    assert history
    assert history[0]["content"] == "User sends 'My dog is pretty'"


@pytest_asyncio.fixture
async def mock_persona_think_response(mock_llm, profile_memory: ProfileMemory):
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

    await profile_memory.add_persona_message(
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
    profile_memory, mock_persona_think_response
):
    count = -1
    for i in range(10):
        count = await profile_memory.uningested_message_count()
        if count == 0:
            break
        await asyncio.sleep(0.1)
    if count != 0:
        pytest.fail("Messages are not ingested")

    profile = await profile_memory.get_user_profile(
        user_id="test_user",
    )

    assert profile == mock_persona_think_response
