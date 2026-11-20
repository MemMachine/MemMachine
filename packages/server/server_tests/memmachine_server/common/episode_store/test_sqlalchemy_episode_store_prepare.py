"""Tests for SqlAlchemyEpisodeStore's prepare and startup."""

from __future__ import annotations

import pytest
from sqlalchemy import Uuid

from memmachine_server.common.episode_store.episode_model import (
    EpisodeEntry,
    EpisodeType,
)
from memmachine_server.common.episode_store.episode_sqlalchemy_store import (
    Episode,
    SqlAlchemyEpisodeStore,
)


def test_episode_primary_key_is_uuid_and_not_autoincremented():
    uid_column = Episode.__table__.c.uid

    assert isinstance(uid_column.type, Uuid)
    assert uid_column.autoincrement is False


@pytest.mark.asyncio
async def test_store_serves_after_prepare(sqlalchemy_engine):
    store = SqlAlchemyEpisodeStore(sqlalchemy_engine)
    await store.prepare()
    await store.startup()

    episodes = await store.add_episodes(
        session_key="test-session",
        episodes=[
            EpisodeEntry(
                content="hello",
                producer_id="test",
                producer_role="user",
                episode_type=EpisodeType.MESSAGE,
            )
        ],
    )

    assert len(episodes) == 1
    assert episodes[0].content == "hello"
