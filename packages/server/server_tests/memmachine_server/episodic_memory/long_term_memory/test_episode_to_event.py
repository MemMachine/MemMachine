"""Unit tests for the Episode → Event translation in LongTermMemory."""

from datetime import UTC, datetime
from uuid import NAMESPACE_URL, uuid5

import pytest

from memmachine_server.common.episode_store import (
    ContentType,
    Episode,
    EpisodeType,
)
from memmachine_server.common.property_keys import reserved_property_key
from memmachine_server.episodic_memory.event_memory.data_types import (
    ProducerContext,
    TextBlock,
)
from memmachine_server.episodic_memory.long_term_memory.long_term_memory import (
    _EVENT_UUID_NAMESPACE,
    LongTermMemory,
)


def _episode(
    *,
    uid: str = "ep1",
    content: str = "hello",
    producer_id: str = "alice",
    producer_role: str = "user",
    produced_for_id: str | None = None,
    episode_type: EpisodeType = EpisodeType.MESSAGE,
    filterable_metadata: dict | None = None,
    metadata: dict | None = None,
    session_key: str = "session-1",
) -> Episode:
    return Episode(
        uid=uuid5(NAMESPACE_URL, uid),
        content=content,
        session_key=session_key,
        created_at=datetime(2026, 1, 15, 12, 0, tzinfo=UTC),
        producer_id=producer_id,
        producer_role=producer_role,
        produced_for_id=produced_for_id,
        episode_type=episode_type,
        content_type=ContentType.STRING,
        filterable_metadata=filterable_metadata,
        metadata=metadata,
    )


def test_event_uuid_is_deterministic_uuid5_of_episode_uid():
    episode = _episode(uid="abc-123")
    event = LongTermMemory._episode_to_event(episode)
    assert event.uuid == uuid5(_EVENT_UUID_NAMESPACE, str(episode.uid))
    # Stable across calls.
    assert LongTermMemory._episode_to_event(episode).uuid == event.uuid


def test_producer_is_the_source():
    """The producer id is the event's source.

    A message carries a `ProducerContext`, as before.
    """
    episode = _episode(producer_id="alice")
    event = LongTermMemory._episode_to_event(episode)
    assert event.source_id == "alice"
    assert event.context == ProducerContext(producer="alice")


def test_event_has_single_text_block_with_episode_content():
    episode = _episode(content="hello world")
    event = LongTermMemory._episode_to_event(episode)
    assert len(event.blocks) == 1
    assert isinstance(event.blocks[0], TextBlock)
    assert event.blocks[0].text == "hello world"


def _system_key(field: str) -> str:
    return reserved_property_key("ltm", field)


def test_system_fields_are_under_reserved_keys():
    episode = _episode(
        uid="ep-1",
        producer_id="alice",
        producer_role="user",
        produced_for_id="bob",
        session_key="sess-X",
    )
    event = LongTermMemory._episode_to_event(episode)
    assert event.properties == {
        _system_key("episode_uid"): str(episode.uid),
        _system_key("session_key"): "sess-X",
        _system_key("producer_id"): "alice",
        _system_key("producer_role"): "user",
        _system_key("produced_for_id"): "bob",
        _system_key("episode_type"): "message",
        _system_key("content_type"): "string",
        _system_key("created_at"): episode.created_at,
    }


def test_produced_for_id_omitted_when_none():
    episode = _episode(produced_for_id=None)
    event = LongTermMemory._episode_to_event(episode)
    assert _system_key("produced_for_id") not in event.properties
    assert _system_key("producer_id") in event.properties


def test_user_filterable_metadata_keys_are_bare():
    episode = _episode(
        filterable_metadata={"my_field": "value", "count": 7},
    )
    event = LongTermMemory._episode_to_event(episode)
    p = event.properties
    # User-defined keys land as given, so the v2 client filter API
    # `m.my_field` resolves to property `my_field`.
    assert p["my_field"] == "value"
    assert p["count"] == 7


def test_episode_to_event_rejects_reserved_user_keys():
    """A user key in the reserved namespace would claim a system field's identity.

    Without the check a client could send another producer's or session's
    system field through filterable_metadata, get its content indexed under
    that spoofed identity, and impersonate it through search_scored's
    property_filter.
    """
    episode = _episode(
        filterable_metadata={
            "safe_key": "ok",
            _system_key("producer_id"): "victim",
            _system_key("session_key"): "other-session",
        }
    )
    with pytest.raises(ValueError, match="reserved"):
        LongTermMemory._episode_to_event(episode)


def test_an_underscore_user_key_is_a_user_key():
    """A leading underscore reserves nothing; the system fields stay the adapter's."""
    episode = _episode(
        producer_id="alice", filterable_metadata={"_producer_id": "victim"}
    )
    event = LongTermMemory._episode_to_event(episode)
    assert event.properties["_producer_id"] == "victim"
    assert event.properties[_system_key("producer_id")] == "alice"


@pytest.mark.parametrize("key", ["metadata.foo", "userId"])
def test_episode_to_event_rejects_an_illegal_user_key(key: str):
    episode = _episode(filterable_metadata={key: "bar"})
    with pytest.raises(ValueError, match=r"\[a-z0-9_\]"):
        LongTermMemory._episode_to_event(episode)
