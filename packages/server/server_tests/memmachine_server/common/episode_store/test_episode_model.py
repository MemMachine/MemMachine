"""Test for the Episode models."""

import json
import math
from datetime import UTC, datetime, timedelta, timezone
from uuid import UUID

import pytest
from memmachine_common.api import EpisodeType
from pydantic import ValidationError

from memmachine_server.common.episode_store import Episode, EpisodeEntry
from memmachine_server.common.episode_store.episode_model import (
    EpisodeResponse,
    episodes_to_string,
)


@pytest.fixture
def base_episode_data():
    """Provides common data for creating Episode instances."""
    return {
        "uid": UUID("550e8400-e29b-41d4-a716-446655440123"),
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


def test_episode_entry_assigns_distinct_uuid4_ids_before_storage():
    first = EpisodeEntry(content="first", producer_id="user", producer_role="user")
    second = EpisodeEntry(content="second", producer_id="user", producer_role="user")

    first_uuid = first.uid
    second_uuid = second.uid

    assert first_uuid.version == 4
    assert second_uuid.version == 4
    assert isinstance(first_uuid, UUID)
    assert isinstance(second_uuid, UUID)
    assert first.uid != second.uid


def test_episode_entry_preserves_explicit_internal_id():
    supplied_id = "550e8400-e29b-41d4-a716-446655440000"
    entry = EpisodeEntry(
        uid=supplied_id,
        content="message",
        producer_id="user",
        producer_role="user",
    )

    assert entry.uid == UUID(supplied_id)


def test_episodes_to_string_multiple_mixed(base_episode_data):
    """Verify that multiple episodes are concatenated with newlines."""
    ep1 = Episode(**base_episode_data)

    base_episode_data["uid"] = UUID("550e8400-e29b-41d4-a716-446655440456")
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


_ENTRY_FIELDS = {"content": "hello", "producer_id": "user", "producer_role": "user"}


@pytest.mark.parametrize(
    "fields",
    [
        pytest.param({"metadata": {"key": math.nan}}, id="metadata-nan"),
        pytest.param({"metadata": {"key": -math.inf}}, id="metadata-negative-inf"),
        pytest.param({"metadata": {"key": 2**63}}, id="metadata-int-above-int64"),
        pytest.param({"metadata": {"key": "a\x00b"}}, id="metadata-value-nul"),
        pytest.param(
            {"metadata": {"key": "a\ud800b"}}, id="metadata-value-lone-surrogate"
        ),
        pytest.param({"metadata": {"a\x00b": "value"}}, id="metadata-key-nul"),
        pytest.param({"producer_id": "a\x00b"}, id="producer-id-nul"),
        pytest.param({"producer_role": "a\x00b"}, id="producer-role-nul"),
        pytest.param({"produced_for_id": "a\x00b"}, id="produced-for-id-nul"),
        pytest.param(
            {"created_at": datetime(1, 1, 1, tzinfo=timezone(timedelta(hours=5)))},
            id="created-at-before-year-1-in-utc",
        ),
        pytest.param(
            {
                "created_at": datetime(
                    2026,
                    1,
                    1,
                    tzinfo=timezone(timedelta(hours=5, minutes=30, seconds=45)),
                )
            },
            id="created-at-offset-seconds",
        ),
    ],
)
def test_episode_entry_refuses_property_values_outside_the_domain(fields):
    with pytest.raises(ValidationError, match="A property"):
        EpisodeEntry.model_validate({**_ENTRY_FIELDS, **fields})


def test_episode_entry_keeps_metadata_values_that_are_not_property_values():
    metadata = {"tags": ["a", "b"], "nested": {"count": 2**64}, "missing": None}

    assert EpisodeEntry(**_ENTRY_FIELDS, metadata=metadata).metadata == metadata
