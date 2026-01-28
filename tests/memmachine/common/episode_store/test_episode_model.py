"""Test for the Episode models."""

from datetime import UTC, datetime

import pytest

from memmachine.common.api import EpisodeType
from memmachine.common.episode_store import Episode
from memmachine.common.episode_store.episode_model import (
    EpisodeResponse,
    episodes_to_string,
)


@pytest.fixture
def base_episode_data():
    """Provides common data for creating Episode instances."""
    return {
        "uid": "msg_123",
        "content": "Hello world",
        "session_key": "session_abc",
        "created_at": datetime(2026, 1, 14, 13, 30, tzinfo=UTC),  # Wednesday
        "producer_id": "user_1",
        "producer_role": "user",
        "episode_type": EpisodeType.MESSAGE,
    }


def test_episodes_to_string_empty():
    """Verify that an empty list returns an empty string."""
    assert episodes_to_string([]) == ""


def test_episodes_to_string_multiple_mixed(base_episode_data):
    """Verify that multiple episodes are concatenated with newlines."""
    ep1 = Episode(**base_episode_data)

    base_episode_data["uid"] = "msg_456"
    base_episode_data["episode_type"] = EpisodeType.MESSAGE
    base_episode_data["content"] = "Brief summary"
    ep2 = Episode(**base_episode_data)

    result = episodes_to_string([ep1, ep2])

    lines = result.strip().split("\n")
    assert len(lines) == 2
    line0 = '[Wednesday, January 14, 2026 at 01:30 PM] user_1: "Hello world"'
    line1 = '[Wednesday, January 14, 2026 at 01:30 PM] user_1: "Brief summary"'
    assert lines[0] == line0
    assert lines[1] == line1


def test_episodes_to_string_with_episode_response(base_episode_data):
    """Verify it works with EpisodeResponse instances (score included)."""
    # Since EpisodeResponse inherits from EpisodeEntry/Episode, we mock it similarly
    er = EpisodeResponse(**base_episode_data, score=0.95)
    result = episodes_to_string([er])

    lines = result.strip().split("\n")
    assert len(lines) == 1
    line0 = '[Wednesday, January 14, 2026 at 01:30 PM] user_1: "Hello world"'
    assert lines[0] == line0
