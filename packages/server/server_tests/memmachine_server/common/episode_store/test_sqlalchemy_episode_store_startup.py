"""Tests for SqlAlchemyEpisodeStore.startup()."""

from __future__ import annotations

import socket

import pytest
from sqlalchemy import Uuid
from sqlalchemy.exc import OperationalError

from memmachine_server.common.episode_store.episode_model import (
    EpisodeEntry,
    EpisodeType,
)
from memmachine_server.common.episode_store.episode_sqlalchemy_store import (
    Episode,
    SqlAlchemyEpisodeStore,
)
from memmachine_server.common.errors import ConfigurationError, SchemaMismatchError


class _FailingContext:
    def __init__(self, exc: Exception):
        self._exc = exc

    async def __aenter__(self):
        raise self._exc

    async def __aexit__(self, exc_type, exc, tb):
        return False


class _FakeAsyncEngine:
    def __init__(self, exc: Exception):
        self._exc = exc

    def begin(self):
        return _FailingContext(self._exc)

    def connect(self):
        return _FailingContext(self._exc)


@pytest.mark.asyncio
async def test_startup_wraps_operational_error():
    engine = _FakeAsyncEngine(OperationalError("select 1", {}, Exception("db down")))
    store = SqlAlchemyEpisodeStore(engine)  # type: ignore[arg-type]  # ty: ignore[invalid-argument-type]

    with pytest.raises(ConfigurationError) as exc_info:
        await store.startup()

    assert isinstance(exc_info.value.__cause__, OperationalError)


@pytest.mark.asyncio
async def test_startup_wraps_socket_gaierror():
    engine = _FakeAsyncEngine(socket.gaierror(8, "dns lookup failed"))
    store = SqlAlchemyEpisodeStore(engine)  # type: ignore[arg-type]  # ty: ignore[invalid-argument-type]

    with pytest.raises(ConfigurationError) as exc_info:
        await store.startup()

    assert isinstance(exc_info.value.__cause__, socket.gaierror)


def test_episode_primary_key_is_uuid_and_not_autoincremented():
    uid_column = Episode.__table__.c.uid

    assert isinstance(uid_column.type, Uuid)
    assert uid_column.autoincrement is False


@pytest.mark.asyncio
async def test_startup_refuses_an_unmigrated_database(sqlalchemy_engine):
    store = SqlAlchemyEpisodeStore(sqlalchemy_engine)

    with pytest.raises(SchemaMismatchError, match="memmachine-server migrate"):
        await store.startup()


@pytest.mark.asyncio
async def test_store_serves_after_migrate(sqlalchemy_engine):
    store = SqlAlchemyEpisodeStore(sqlalchemy_engine)
    await store.migrate()
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
