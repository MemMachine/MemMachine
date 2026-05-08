"""Test for the Episode models."""

import json
from datetime import UTC, datetime

import pytest
from memmachine_common.api import EpisodeType

from memmachine_server.common.episode_store import Episode
from memmachine_server.common.episode_store.episode_model import (
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


def test_episodes_to_string_message_preserves_non_ascii(base_episode_data):
    """Non-ASCII content must appear literally in the LLM context, not as
    ``\\uXXXX`` escapes — escapes inflate the prompt token count and
    obscure semantic content."""
    base_episode_data["content"] = "寿司 café 🍕 Привет"
    ep = Episode(**base_episode_data)
    result = episodes_to_string([ep])

    assert "寿司" in result
    assert "café" in result
    assert "🍕" in result
    assert "Привет" in result
    assert "\\u" not in result

    # The JSON-quoted content must round-trip back to the original string.
    line = result.rstrip("\n")
    json_part = line.split(": ", 1)[1]
    assert json.loads(json_part) == "寿司 café 🍕 Привет"


def test_episodes_to_string_non_message_preserves_non_ascii(base_episode_data):
    """The ``case _:`` fallback (e.g. an EpisodeResponse with no episode
    type) must also preserve Unicode literally."""
    fallback_data = {k: v for k, v in base_episode_data.items() if k != "session_key"}
    fallback_data["episode_type"] = None
    fallback_data["content"] = "要約: ☕ résumé"
    er = EpisodeResponse(**fallback_data)
    result = episodes_to_string([er])

    assert "要約" in result
    assert "☕" in result
    assert "résumé" in result
    assert "\\u" not in result
    assert json.loads(result.rstrip("\n")) == "要約: ☕ résumé"


def test_episodes_to_string_output_is_utf8_encodable(base_episode_data):
    """The formatted string is the exact text fed to LanguageModel prompts;
    it must be losslessly UTF-8 encodable (no surrogate pairs from broken
    escaping)."""
    base_episode_data["content"] = "ASCII + 中文 + 🚀 + emoji modifier 👨‍👩‍👧‍👦"
    ep = Episode(**base_episode_data)
    result = episodes_to_string([ep])

    encoded = result.encode("utf-8")
    assert encoded.decode("utf-8") == result
