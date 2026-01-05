import socket

import pytest
from sqlalchemy.exc import OperationalError

from memmachine.common.episode_store.episode_sqlalchemy_store import (
    SqlAlchemyEpisodeStore,
)
from memmachine.common.errors import ConfigurationError


class _FailingBeginContext:
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
        return _FailingBeginContext(self._exc)


@pytest.mark.asyncio
async def test_startup_wraps_operational_error():
    engine = _FakeAsyncEngine(OperationalError("select 1", {}, Exception("db down")))
    store = SqlAlchemyEpisodeStore(engine)  # type: ignore[arg-type]

    with pytest.raises(ConfigurationError) as exc_info:
        await store.startup()

    assert isinstance(exc_info.value.__cause__, OperationalError)


@pytest.mark.asyncio
async def test_startup_wraps_socket_gaierror():
    engine = _FakeAsyncEngine(socket.gaierror(8, "dns lookup failed"))
    store = SqlAlchemyEpisodeStore(engine)  # type: ignore[arg-type]

    with pytest.raises(ConfigurationError) as exc_info:
        await store.startup()

    assert isinstance(exc_info.value.__cause__, socket.gaierror)
