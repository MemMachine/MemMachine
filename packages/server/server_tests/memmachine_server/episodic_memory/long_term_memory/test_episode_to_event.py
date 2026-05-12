"""Unit tests for the Episode → Event translation in LongTermMemory."""

from datetime import UTC, datetime
from uuid import uuid5

import pytest

from memmachine_server.common.episode_store import (
    ContentType,
    Episode,
    EpisodeType,
)
from memmachine_server.episodic_memory.event_memory.data_types import (
    NullContext,
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
    sequence_num: int = 0,
    episode_type: EpisodeType = EpisodeType.MESSAGE,
    filterable_metadata: dict | None = None,
    metadata: dict | None = None,
    session_key: str = "session-1",
) -> Episode:
    return Episode(
        uid=uid,
        content=content,
        session_key=session_key,
        created_at=datetime(2026, 1, 15, 12, 0, tzinfo=UTC),
        producer_id=producer_id,
        producer_role=producer_role,
        produced_for_id=produced_for_id,
        sequence_num=sequence_num,
        episode_type=episode_type,
        content_type=ContentType.STRING,
        filterable_metadata=filterable_metadata,
        metadata=metadata,
    )


def test_event_uuid_is_deterministic_uuid5_of_episode_uid():
    episode = _episode(uid="abc-123")
    event = LongTermMemory._episode_to_event(episode)
    assert event.uuid == uuid5(_EVENT_UUID_NAMESPACE, "abc-123")
    # Stable across calls.
    assert LongTermMemory._episode_to_event(episode).uuid == event.uuid


def test_message_episode_uses_producer_context():
    episode = _episode(producer_id="alice", episode_type=EpisodeType.MESSAGE)
    event = LongTermMemory._episode_to_event(episode)
    assert isinstance(event.context, ProducerContext)
    assert event.context.producer == "alice"


def test_non_message_episode_uses_null_context():
    # Use any non-MESSAGE EpisodeType. The enum is defined in
    # memmachine_common.api; pick the first non-MESSAGE entry dynamically so
    # this test doesn't break when the enum gains members.
    non_message = next(
        (t for t in EpisodeType if t is not EpisodeType.MESSAGE),
        None,
    )
    if non_message is None:
        # Only MESSAGE exists today; this branch will start exercising once
        # additional Episode types are introduced. Skip rather than assert
        # invariant we can't yet exercise.
        import pytest

        pytest.skip("Only MESSAGE EpisodeType exists; nothing else to verify yet")
    episode = _episode(episode_type=non_message)
    event = LongTermMemory._episode_to_event(episode)
    assert isinstance(event.context, NullContext)


def test_event_has_single_text_block_with_episode_content():
    episode = _episode(content="hello world")
    event = LongTermMemory._episode_to_event(episode)
    assert len(event.blocks) == 1
    assert isinstance(event.blocks[0], TextBlock)
    assert event.blocks[0].text == "hello world"


def test_system_fields_are_underscore_prefixed():
    episode = _episode(
        uid="ep-1",
        producer_id="alice",
        producer_role="user",
        produced_for_id="bob",
        sequence_num=42,
        session_key="sess-X",
    )
    event = LongTermMemory._episode_to_event(episode)
    p = event.properties
    assert p["_episode_uid"] == "ep-1"
    assert p["_session_key"] == "sess-X"
    assert p["_producer_id"] == "alice"
    assert p["_producer_role"] == "user"
    assert p["_produced_for_id"] == "bob"
    assert p["_sequence_num"] == 42
    assert p["_episode_type"] == "message"
    assert p["_content_type"] == "string"
    assert p["_created_at"] == episode.created_at


def test_produced_for_id_omitted_when_none():
    episode = _episode(produced_for_id=None)
    event = LongTermMemory._episode_to_event(episode)
    assert "_produced_for_id" not in event.properties


def test_user_filterable_metadata_keys_are_bare():
    episode = _episode(
        filterable_metadata={"my_field": "value", "count": 7},
    )
    event = LongTermMemory._episode_to_event(episode)
    p = event.properties
    # User-defined keys land bare (no `_` prefix), so the v2 client filter API
    # `m.my_field` resolves to vector-record property `my_field`.
    assert p["my_field"] == "value"
    assert p["count"] == 7


def test_episode_to_event_rejects_underscore_prefixed_user_keys():
    """Reserved `_`-prefixed user keys must be rejected at the event-backend
    translation layer. Without this check a client could send
    {"_producer_id": "victim", "_session_key": "other-session"} via
    filterable_metadata, get its content indexed under those spoofed
    identities, and impersonate the victim through search_scored's
    property_filter API. We raise loudly so callers see the misuse instead
    of silently dropping. (Event-backend only — the declarative backend
    mangles user keys with a `metadata.` prefix and is unaffected.)
    """
    episode = _episode(
        filterable_metadata={
            "safe_key": "ok",
            "_producer_id": "victim",
            "_session_key": "other-session",
            "_episode_uid": "spoofed-uid",
        }
    )
    with pytest.raises(ValueError, match=r"reserved.*`_`-prefixed"):
        LongTermMemory._episode_to_event(episode)


def test_episode_to_event_allows_metadata_dot_prefixed_keys():
    """The declarative backend mangles user keys with a `metadata.` prefix.
    On the event backend, a literal `metadata.foo` key is just a regular
    user property and must not be rejected.
    """
    episode = _episode(filterable_metadata={"metadata.foo": "bar"})
    event = LongTermMemory._episode_to_event(episode)
    assert event.properties["metadata.foo"] == "bar"
