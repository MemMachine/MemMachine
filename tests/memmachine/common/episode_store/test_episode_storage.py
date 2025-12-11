from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta

import pytest
import pytest_asyncio

from memmachine.common.episode_store import (
    EpisodeEntry,
    EpisodeIdT,
    EpisodeStorage,
    EpisodeType,
)
from memmachine.common.errors import InvalidArgumentError
from memmachine.common.filter.filter_parser import FilterExpr, parse_filter

DEFAULT_HISTORY_ARGS = {
    "session_key": "session-default",
    "producer_id": "producer-default",
    "producer_role": "user",
}


async def create_history_entry(
    episode_storage: EpisodeStorage,
    *,
    content: str = "content",
    session_key: str | None = None,
    producer_id: str | None = None,
    producer_role: str | None = None,
    produced_for_id: str | None = None,
    metadata: dict[str, str] | None = None,
    created_at: datetime | None = None,
    episode_type: EpisodeType | None = None,
) -> EpisodeIdT:
    params = {
        "producer_id": producer_id or DEFAULT_HISTORY_ARGS["producer_id"],
        "producer_role": producer_role or DEFAULT_HISTORY_ARGS["producer_role"],
    }

    episode = await episode_storage.add_episodes(
        episodes=[
            EpisodeEntry(
                content=content,
                episode_type=episode_type,
                produced_for_id=produced_for_id,
                metadata=metadata,
                created_at=created_at,
                **params,
            ),
        ],
        session_key=session_key or DEFAULT_HISTORY_ARGS["session_key"],
    )

    return episode[0].uid


@pytest_asyncio.fixture
async def timestamped_history(episode_storage: EpisodeStorage):
    created_at = datetime.now(tz=UTC) - timedelta(days=1)
    before = created_at - timedelta(minutes=1)
    after = created_at + timedelta(minutes=1)

    history_id = await create_history_entry(
        episode_storage,
        content="first",
        metadata={"source": "chat"},
        created_at=created_at,
    )

    message = await episode_storage.get_episode(history_id)

    yield (message, before, after)

    await episode_storage.delete_episodes([history_id])


@pytest.mark.asyncio
async def test_add_and_get_history(episode_storage: EpisodeStorage):
    history_id = await create_history_entry(
        episode_storage,
        content="hello",
        metadata={"role": "user"},
        session_key="chat-session",
        producer_id="user-123",
        producer_role="assistant",
        produced_for_id="agent-456",
        episode_type=EpisodeType.MESSAGE,
    )

    assert type(history_id) is EpisodeIdT

    history = await episode_storage.get_episode(history_id)
    assert history.uid == history_id
    assert history.metadata == {"role": "user"}
    assert history.content == "hello"
    assert history.session_key == "chat-session"
    assert history.producer_id == "user-123"
    assert history.producer_role == "assistant"
    assert history.produced_for_id == "agent-456"
    assert history.episode_type == EpisodeType.MESSAGE


@pytest.mark.asyncio
async def test_add_multiple_episodes_returns_models(
    episode_storage: EpisodeStorage,
):
    created_at = datetime.now(tz=UTC)
    entries = [
        EpisodeEntry(
            content="first",
            producer_id="p-1",
            producer_role="role-1",
            produced_for_id="consumer-1",
            metadata={"key": "value"},
            created_at=created_at,
        ),
        EpisodeEntry(
            content="second",
            producer_id="p-2",
            producer_role="role-2",
            episode_type=EpisodeType.MESSAGE,
        ),
    ]

    episodes = await episode_storage.add_episodes("batch-session", entries)

    try:
        assert [e.content for e in episodes] == ["first", "second"]
        assert all(e.session_key == "batch-session" for e in episodes)
        assert episodes[0].created_at == created_at
        assert episodes[0].metadata == {"key": "value"}
        assert episodes[0].produced_for_id == "consumer-1"
        assert episodes[1].episode_type == EpisodeType.MESSAGE
    finally:
        await episode_storage.delete_episodes([e.uid for e in episodes])

    assert await episode_storage.get_episode_messages() == []


@pytest.mark.asyncio
async def test_history_identity_filters(episode_storage: EpisodeStorage):
    user_message = await create_history_entry(
        episode_storage,
        content="user message",
        session_key="session-user",
        producer_id="user-id",
        producer_role="user",
        produced_for_id="agent-id",
        episode_type=EpisodeType.MESSAGE,
    )
    assistant_message = await create_history_entry(
        episode_storage,
        content="assistant message",
        session_key="session-assistant",
        producer_id="assistant-id",
        producer_role="assistant",
        produced_for_id="user-id",
        episode_type=EpisodeType.MESSAGE,
    )
    system_message = await create_history_entry(
        episode_storage,
        content="system message",
        session_key="session-system",
        producer_id="system-id",
        producer_role="system",
        produced_for_id="group-id",
        episode_type=EpisodeType.MESSAGE,
    )

    try:
        by_session = await episode_storage.get_episode_messages(
            filter_expr=_filter("session_key = 'session-assistant'"),
        )
        assert [m.uid for m in by_session] == [assistant_message]

        by_producer_id = await episode_storage.get_episode_messages(
            filter_expr=_filter("producer_id = 'system-id'"),
        )
        assert [m.uid for m in by_producer_id] == [system_message]

        by_producer_role = await episode_storage.get_episode_messages(
            filter_expr=_filter("producer_role = 'user'"),
        )
        assert [m.uid for m in by_producer_role] == [user_message]

        by_produced_for = await episode_storage.get_episode_messages(
            filter_expr=_filter("produced_for_id = 'user-id'"),
        )
        assert [m.uid for m in by_produced_for] == [assistant_message]

    finally:
        await episode_storage.delete_episodes(
            [user_message, assistant_message, system_message],
        )


@pytest.mark.asyncio
async def test_history_comparison_filters(episode_storage: EpisodeStorage):
    first = await create_history_entry(episode_storage, content="first")
    second = await create_history_entry(episode_storage, content="second")
    third = await create_history_entry(episode_storage, content="third")

    try:
        greater_than_first = await episode_storage.get_episode_messages(
            filter_expr=_filter(f"id > {first}"),
        )
        assert {entry.uid for entry in greater_than_first} == {second, third}

        up_to_second = await episode_storage.get_episode_messages(
            filter_expr=_filter(f"id <= {second}"),
        )
        assert {entry.uid for entry in up_to_second} == {first, second}
    finally:
        await episode_storage.delete_episodes([first, second, third])


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
    episode_storage: EpisodeStorage,
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

    window = await episode_storage.get_episode_messages(
        start_time=reference[start_key],
        end_time=reference[end_key],
    )

    assert len(window) == expected_count
    if expected_count:
        assert window[0].uid == message.uid


@pytest.mark.asyncio
async def test_history_metadata_filter(episode_storage: EpisodeStorage):
    first = await create_history_entry(
        episode_storage,
        content="alpha",
        metadata={"scope": "a"},
    )
    second = await create_history_entry(
        episode_storage,
        content="beta",
        metadata={"scope": "b"},
    )

    results = await episode_storage.get_episode_messages(
        filter_expr=_filter("metadata.scope = 'b'"),
    )
    assert [entry.uid for entry in results] == [second]

    await episode_storage.delete_episodes([first, second])


@pytest.mark.asyncio
async def test_history_pagination_with_page_offset(
    episode_storage: EpisodeStorage,
):
    base_time = datetime.now(tz=UTC)
    episode_ids = []

    for idx in range(5):
        created_at = base_time + timedelta(minutes=idx)
        episode_ids.append(
            await create_history_entry(
                episode_storage,
                content=f"message-{idx}",
                created_at=created_at,
            )
        )

    try:
        first_page = await episode_storage.get_episode_messages(page_size=2, page_num=0)
        second_page = await episode_storage.get_episode_messages(
            page_size=2, page_num=1
        )
        third_page = await episode_storage.get_episode_messages(page_size=2, page_num=2)

        assert [entry.uid for entry in first_page] == episode_ids[:2]
        assert [entry.uid for entry in second_page] == episode_ids[2:4]
        assert [entry.uid for entry in third_page] == episode_ids[4:5]
    finally:
        await episode_storage.delete_episodes(episode_ids)


@pytest.mark.asyncio
async def test_history_pagination_offset_without_limit_raises(
    episode_storage: EpisodeStorage,
):
    with pytest.raises(InvalidArgumentError):
        await episode_storage.get_episode_messages(page_num=1)


@pytest.mark.asyncio
async def test_delete_history(episode_storage: EpisodeStorage):
    history_id = await create_history_entry(episode_storage, content="to delete")
    await episode_storage.delete_episodes([history_id])

    history = await episode_storage.get_episode(history_id)

    assert history is None


@pytest.mark.asyncio
async def test_delete_history_messages_by_range(episode_storage: EpisodeStorage):
    _ = await create_history_entry(
        episode_storage,
        content="old",
        created_at=datetime.now(UTC) - timedelta(days=2),
    )
    newer = await create_history_entry(episode_storage, content="new")

    cutoff = datetime.now(UTC) - timedelta(days=1)
    await episode_storage.delete_episode_messages(end_time=cutoff)

    remaining = await episode_storage.get_episode_messages()
    assert [entry.uid for entry in remaining] == [newer]

    await episode_storage.delete_episodes([newer])


@pytest.mark.asyncio
async def test_delete_history_messages_with_identity_filters(
    episode_storage: EpisodeStorage,
):
    keep_history = await create_history_entry(
        episode_storage,
        content="keep",
        producer_role="user",
    )
    await create_history_entry(
        episode_storage,
        content="drop",
        producer_role="assistant",
    )

    await episode_storage.delete_episode_messages(
        filter_expr=_filter("producer_role = 'assistant'"),
    )

    remaining = await episode_storage.get_episode_messages()
    assert [entry.uid for entry in remaining] == [keep_history]

    await episode_storage.delete_episodes([keep_history])


@pytest.mark.asyncio
async def test_history_time_window_workflow(episode_storage: EpisodeStorage):
    first = await create_history_entry(
        episode_storage,
        content="first",
        metadata={"rank": "low"},
    )
    await asyncio.sleep(0)
    second = await create_history_entry(
        episode_storage,
        content="second",
        metadata={"rank": "mid"},
    )
    cutoff = datetime.now(UTC)
    await asyncio.sleep(1)
    third = await create_history_entry(
        episode_storage,
        content="third",
        metadata={"rank": "high"},
    )

    before_third = await episode_storage.get_episode_messages(end_time=cutoff)
    assert [m.uid for m in before_third] == [first, second]

    await episode_storage.delete_episode_messages(end_time=cutoff)
    remaining = await episode_storage.get_episode_messages()
    assert [m.uid for m in remaining] == [third]

    await episode_storage.delete_episode_messages()
    assert await episode_storage.get_episode_messages() == []


@pytest.mark.asyncio
async def test_number_of_messages_in_empty(episode_storage: EpisodeStorage):
    total_count = await episode_storage.get_episode_messages_count()
    assert total_count == 0


@pytest.mark.asyncio
async def test_number_no_filter_returns_all(episode_storage: EpisodeStorage):
    await create_history_entry(episode_storage, session_key="first", content="first")
    await create_history_entry(episode_storage, session_key="second", content="second")

    first_count = await episode_storage.get_episode_messages_count(
        filter_expr=_filter("session_key = 'first'"),
    )
    assert first_count == 1

    second_count = await episode_storage.get_episode_messages_count(
        filter_expr=_filter("session_key = 'second'"),
    )
    assert second_count == 1

    total_count = await episode_storage.get_episode_messages_count()
    assert total_count == 2


def _filter(spec: str) -> FilterExpr:
    expr = parse_filter(spec)
    assert expr is not None
    return expr
