"""Tests for episode content-hash deduplication."""

from __future__ import annotations

import pytest

from memmachine.common.episode_store.episode_model import EpisodeEntry
from memmachine.common.episode_store.episode_sqlalchemy_store import (
    compute_content_hash,
)

# ---------------------------------------------------------------------------
# Unit tests for compute_content_hash
# ---------------------------------------------------------------------------


class TestComputeContentHash:
    """Verify the SHA-256 content hash utility."""

    def test_deterministic(self):
        h1 = compute_content_hash("s", "p", "hello")
        h2 = compute_content_hash("s", "p", "hello")
        assert h1 == h2

    def test_different_content_different_hash(self):
        h1 = compute_content_hash("s", "p", "hello")
        h2 = compute_content_hash("s", "p", "world")
        assert h1 != h2

    def test_different_producer_different_hash(self):
        h1 = compute_content_hash("s", "user", "hello")
        h2 = compute_content_hash("s", "assistant", "hello")
        assert h1 != h2

    def test_different_session_different_hash(self):
        h1 = compute_content_hash("org/proj1", "p", "hello")
        h2 = compute_content_hash("org/proj2", "p", "hello")
        assert h1 != h2

    def test_null_byte_separator_prevents_ambiguity(self):
        """Ensure 'ab' + 'cd' != 'a' + 'bcd'."""
        h1 = compute_content_hash("ab", "cd", "msg")
        h2 = compute_content_hash("a", "bcd", "msg")
        assert h1 != h2

    def test_returns_64_char_hex(self):
        h = compute_content_hash("s", "p", "content")
        assert len(h) == 64
        assert all(c in "0123456789abcdef" for c in h)


# ---------------------------------------------------------------------------
# Integration tests for dedup in add_episodes (SQLite)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_first_insert_returns_is_new_true(sql_db_episode_storage):
    entry = EpisodeEntry(content="hello", producer_id="user", producer_role="user")
    result = await sql_db_episode_storage.add_episodes("session-a", [entry])

    assert len(result) == 1
    assert result[0].is_new is True
    assert result[0].content == "hello"


@pytest.mark.asyncio
async def test_duplicate_returns_is_new_false(sql_db_episode_storage):
    entry = EpisodeEntry(content="hello", producer_id="user", producer_role="user")

    first = await sql_db_episode_storage.add_episodes("session-a", [entry])
    second = await sql_db_episode_storage.add_episodes("session-a", [entry])

    assert first[0].is_new is True
    assert second[0].is_new is False
    # Same episode ID returned.
    assert first[0].uid == second[0].uid


@pytest.mark.asyncio
async def test_duplicate_does_not_create_extra_row(sql_db_episode_storage):
    entry = EpisodeEntry(content="hello", producer_id="user", producer_role="user")

    await sql_db_episode_storage.add_episodes("session-a", [entry])
    await sql_db_episode_storage.add_episodes("session-a", [entry])

    count = await sql_db_episode_storage.get_episode_messages_count()
    assert count == 1


@pytest.mark.asyncio
async def test_different_producer_not_deduped(sql_db_episode_storage):
    e1 = EpisodeEntry(content="hello", producer_id="user", producer_role="user")
    e2 = EpisodeEntry(
        content="hello", producer_id="assistant", producer_role="assistant"
    )

    r1 = await sql_db_episode_storage.add_episodes("session-a", [e1])
    r2 = await sql_db_episode_storage.add_episodes("session-a", [e2])

    assert r1[0].is_new is True
    assert r2[0].is_new is True
    assert r1[0].uid != r2[0].uid


@pytest.mark.asyncio
async def test_different_session_not_deduped(sql_db_episode_storage):
    entry = EpisodeEntry(content="hello", producer_id="user", producer_role="user")

    r1 = await sql_db_episode_storage.add_episodes("session-a", [entry])
    r2 = await sql_db_episode_storage.add_episodes("session-b", [entry])

    assert r1[0].is_new is True
    assert r2[0].is_new is True
    assert r1[0].uid != r2[0].uid


@pytest.mark.asyncio
async def test_mixed_batch_new_and_duplicate(sql_db_episode_storage):
    """Batch with 1 existing + 2 new: all returned, correct is_new flags."""
    existing = EpisodeEntry(
        content="existing", producer_id="user", producer_role="user"
    )
    await sql_db_episode_storage.add_episodes("session-a", [existing])

    batch = [
        EpisodeEntry(content="existing", producer_id="user", producer_role="user"),
        EpisodeEntry(content="new-one", producer_id="user", producer_role="user"),
        EpisodeEntry(content="new-two", producer_id="user", producer_role="user"),
    ]
    result = await sql_db_episode_storage.add_episodes("session-a", batch)

    assert len(result) == 3
    assert result[0].is_new is False  # duplicate of "existing"
    assert result[1].is_new is True
    assert result[2].is_new is True

    count = await sql_db_episode_storage.get_episode_messages_count()
    assert count == 3  # 1 existing + 2 new
