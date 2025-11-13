from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta

import pytest
import pytest_asyncio

from memmachine.episodic_memory.data_types import EpisodeType
from memmachine.history_store.history_model import HistoryIdT, HistoryMessage
from memmachine.history_store.history_storage import HistoryStorage


DEFAULT_HISTORY_ARGS = {
    "session_key": "session-default",
    "producer_id": "producer-default",
    "producer_role": "user",
}


async def create_history_entry(
    history_storage: HistoryStorage,
    *,
    content: str = "content",
    session_key: str | None = None,
    producer_id: str | None = None,
    producer_role: str | None = None,
    produced_for_id: str | None = None,
    metadata: dict[str, str] | None = None,
    created_at: datetime | None = None,
    episode_type: EpisodeType | None = None,
) -> HistoryIdT:
    params = {
        "session_key": session_key or DEFAULT_HISTORY_ARGS["session_key"],
        "producer_id": producer_id or DEFAULT_HISTORY_ARGS["producer_id"],
        "producer_role": producer_role or DEFAULT_HISTORY_ARGS["producer_role"],
    }

    return await history_storage.add_history(
        content=content,
        episode_type=episode_type,
        produced_for_id=produced_for_id,
        metadata=metadata,
        created_at=created_at,
        **params,
    )


@pytest_asyncio.fixture
async def timestamped_history(history_storage: HistoryStorage):
    created_at = datetime.now(tz=UTC) - timedelta(days=1)
    before = created_at - timedelta(minutes=1)
    after = created_at + timedelta(minutes=1)

    history_id = await create_history_entry(
        history_storage,
        content="first",
        metadata={"source": "chat"},
        created_at=created_at,
    )

    message = await history_storage.get_history(history_id)

    yield (message, before, after)

    await history_storage.delete_history([history_id])


@pytest.mark.asyncio
async def test_add_and_get_history(history_storage: HistoryStorage):
    history_id = await create_history_entry(
        history_storage,
        content="hello",
        metadata={"role": "user"},
        session_key="chat-session",
        producer_id="user-123",
        producer_role="assistant",
        produced_for_id="agent-456",
        episode_type=EpisodeType.MESSAGE,
    )

    assert type(history_id) is HistoryIdT

    history = await history_storage.get_history(history_id)
    assert history.metadata.id == history_id
    assert history.metadata.other == {"role": "user"}
    assert history.content == "hello"
    assert history.session_key == "chat-session"
    assert history.producer_id == "user-123"
    assert history.producer_role == "assistant"
    assert history.produced_for_id == "agent-456"
    assert history.episode_type == EpisodeType.MESSAGE


@pytest.mark.asyncio
async def test_history_identity_filters(history_storage: HistoryStorage):
    user_message = await create_history_entry(
        history_storage,
        content="user message",
        session_key="session-user",
        producer_id="user-id",
        producer_role="user",
        produced_for_id="agent-id",
        episode_type=EpisodeType.MESSAGE,
    )
    assistant_message = await create_history_entry(
        history_storage,
        content="assistant message",
        session_key="session-assistant",
        producer_id="assistant-id",
        producer_role="assistant",
        produced_for_id="user-id",
        episode_type=EpisodeType.ACTION,
    )
    system_message = await create_history_entry(
        history_storage,
        content="system message",
        session_key="session-system",
        producer_id="system-id",
        producer_role="system",
        produced_for_id="group-id",
        episode_type=EpisodeType.THOUGHT,
    )

    try:
        by_session = await history_storage.get_history_messages(
            session_keys=["session-assistant"]
        )
        assert [m.metadata.id for m in by_session] == [assistant_message]

        by_producer_id = await history_storage.get_history_messages(
            producer_ids=["system-id"]
        )
        assert [m.metadata.id for m in by_producer_id] == [system_message]

        by_producer_role = await history_storage.get_history_messages(
            producer_roles=["user"]
        )
        assert [m.metadata.id for m in by_producer_role] == [user_message]

        by_produced_for = await history_storage.get_history_messages(
            produced_for_ids=["user-id"]
        )
        assert [m.metadata.id for m in by_produced_for] == [assistant_message]

        by_episode_type = await history_storage.get_history_messages(
            episode_types=[EpisodeType.THOUGHT]
        )
        assert [m.metadata.id for m in by_episode_type] == [system_message]
    finally:
        await history_storage.delete_history(
            [user_message, assistant_message, system_message]
        )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("start_key", "end_key", "expected_count"),
    [
        ("before", None, 1),
        (None, "before", 0),
        ("after", None, 0),
        (None, "after", 1),
        ("before", "after", 1),
        ("at", None, 1),
        (None, "at", 1),
        ("at", "at", 1),
    ],
)
async def test_history_time_filters(
    history_storage: HistoryStorage,
    timestamped_history,
    start_key,
    end_key,
    expected_count,
):
    message, before, after = timestamped_history
    created_at = message.created_at
    reference = {
        "before": before,
        "after": after,
        "at": created_at,
        None: None,
    }

    window = await history_storage.get_history_messages(
        start_time=reference[start_key],
        end_time=reference[end_key],
    )

    assert len(window) == expected_count
    if expected_count:
        assert window[0].metadata.id == message.metadata.id


@pytest.mark.asyncio
async def test_history_metadata_filter(history_storage: HistoryStorage):
    first = await create_history_entry(
        history_storage,
        content="alpha",
        metadata={"scope": "a"},
    )
    second = await create_history_entry(
        history_storage,
        content="beta",
        metadata={"scope": "b"},
    )

    results = await history_storage.get_history_messages(metadata={"scope": "b"})
    assert [entry.metadata.id for entry in results] == [second]

    await history_storage.delete_history([first, second])


@pytest.mark.asyncio
async def test_delete_history(history_storage: HistoryStorage):
    history_id = await create_history_entry(history_storage, content="to delete")
    await history_storage.delete_history([history_id])

    history = await history_storage.get_history(history_id)

    assert history is None


@pytest.mark.asyncio
async def test_delete_history_messages_by_range(history_storage: HistoryStorage):
    _ = await create_history_entry(
        history_storage,
        content="old",
        created_at=datetime.now(UTC) - timedelta(days=2),
    )
    newer = await create_history_entry(history_storage, content="new")

    cutoff = datetime.now(UTC) - timedelta(days=1)
    await history_storage.delete_history_messages(end_time=cutoff)

    remaining = await history_storage.get_history_messages()
    assert [entry.metadata.id for entry in remaining] == [newer]

    await history_storage.delete_history([newer])


@pytest.mark.asyncio
async def test_delete_history_messages_with_identity_filters(
    history_storage: HistoryStorage,
):
    keep_history = await create_history_entry(
        history_storage,
        content="keep",
        producer_role="user",
    )
    drop_history = await create_history_entry(
        history_storage,
        content="drop",
        producer_role="assistant",
    )

    await history_storage.delete_history_messages(producer_roles=["assistant"])

    remaining = await history_storage.get_history_messages()
    assert [entry.metadata.id for entry in remaining] == [keep_history]

    await history_storage.delete_history([keep_history])


@pytest.mark.asyncio
async def test_history_time_window_workflow(history_storage: HistoryStorage):
    first = await create_history_entry(
        history_storage,
        content="first",
        metadata={"rank": "low"},
    )
    await asyncio.sleep(0)
    second = await create_history_entry(
        history_storage,
        content="second",
        metadata={"rank": "mid"},
    )
    cutoff = datetime.now(UTC)
    await asyncio.sleep(1)
    third = await create_history_entry(
        history_storage,
        content="third",
        metadata={"rank": "high"},
    )

    before_third = await history_storage.get_history_messages(end_time=cutoff)
    assert [m.metadata.id for m in before_third] == [first, second]

    await history_storage.delete_history_messages(end_time=cutoff)
    remaining = await history_storage.get_history_messages()
    assert [m.metadata.id for m in remaining] == [third]

    await history_storage.delete_history_messages()
    assert await history_storage.get_history_messages() == []
