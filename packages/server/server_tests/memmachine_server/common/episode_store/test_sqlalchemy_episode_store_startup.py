"""Tests for SqlAlchemyEpisodeStore.startup().

Covers:
- Connection-failure wrapping (OperationalError, socket.gaierror → ConfigurationError)
- Idempotent episode_type enum creation on PostgreSQL (GH-1174):
  - Unit tests verify savepoint usage, dialect guard, and checkfirst flag.
  - Integration tests (marked ``integration``) run against a real PostgreSQL
    instance and prove the fix handles the TOCTOU enum-creation race.
"""

from __future__ import annotations

import socket
from unittest.mock import MagicMock, patch

import pytest
from sqlalchemy import text
from sqlalchemy.exc import IntegrityError, OperationalError, ProgrammingError
from sqlalchemy.ext.asyncio import AsyncEngine

from memmachine_server.common.episode_store.episode_model import (
    EpisodeEntry,
    EpisodeType,
)
from memmachine_server.common.episode_store.episode_sqlalchemy_store import (
    _EPISODE_PG_ENUM,
    BaseEpisodeStore,
    SqlAlchemyEpisodeStore,
)
from memmachine_server.common.errors import ConfigurationError

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


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


class _AsyncCtxMgr:
    """Minimal async context manager for faking begin() / begin_nested()."""

    def __init__(self, value: object = None) -> None:
        self._value = value

    async def __aenter__(self):
        return self._value

    async def __aexit__(self, exc_type, exc, tb):
        return False


def _make_pg_engine() -> tuple[MagicMock, MagicMock, MagicMock]:
    """Build a fake async engine that mimics the PostgreSQL dialect."""
    engine = MagicMock()

    conn = MagicMock()
    conn.dialect = MagicMock()
    conn.dialect.name = "postgresql"

    sync_conn = MagicMock()
    sync_conn.dialect = conn.dialect

    async def _run_sync(fn):
        return fn(sync_conn)

    conn.run_sync = _run_sync
    conn.begin_nested.return_value = _AsyncCtxMgr()
    engine.begin.return_value = _AsyncCtxMgr(conn)

    return engine, conn, sync_conn


async def _cleanup(engine: AsyncEngine) -> None:
    """Drop test artefacts so integration tests are independent."""
    async with engine.begin() as conn:
        await conn.execute(text("DROP TABLE IF EXISTS episodestore CASCADE"))
        await conn.execute(text("DROP TYPE IF EXISTS episode_type"))


# ---------------------------------------------------------------------------
# Connection-failure tests
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Enum idempotency unit tests (no database required)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_startup_creates_enum_with_checkfirst_on_postgresql():
    """On PostgreSQL, startup() must explicitly create the episode_type enum
    with checkfirst=True before calling metadata.create_all."""
    engine, _conn, _sync_conn = _make_pg_engine()

    with (
        patch.object(
            _EPISODE_PG_ENUM, "create", wraps=lambda *a, **kw: None
        ) as mock_enum_create,
        patch.object(
            BaseEpisodeStore.metadata, "create_all", wraps=lambda *a, **kw: None
        ),
    ):
        store = SqlAlchemyEpisodeStore(engine)
        await store.startup()

    mock_enum_create.assert_called_once()
    _, kwargs = mock_enum_create.call_args
    assert kwargs.get("checkfirst") is True


@pytest.mark.asyncio
async def test_startup_handles_integrity_error_from_concurrent_enum_creation():
    """When a concurrent worker already created the enum type (TOCTOU race),
    the IntegrityError must be caught and startup must still complete,
    including the create_all step for tables."""
    engine, _conn, _sync_conn = _make_pg_engine()

    create_all_called = False

    def fake_create_all(bind, **kw):
        nonlocal create_all_called
        create_all_called = True

    def fake_enum_create(bind, checkfirst=True):
        raise IntegrityError(
            "CREATE TYPE episode_type",
            {},
            Exception("duplicate key value violates unique constraint"),
        )

    with (
        patch.object(_EPISODE_PG_ENUM, "create", side_effect=fake_enum_create),
        patch.object(
            BaseEpisodeStore.metadata, "create_all", side_effect=fake_create_all
        ),
    ):
        store = SqlAlchemyEpisodeStore(engine)
        await store.startup()

    assert create_all_called, "create_all must still run after IntegrityError on enum"


@pytest.mark.asyncio
async def test_startup_handles_programming_error_from_concurrent_enum_creation():
    """asyncpg maps DuplicateObjectError to ProgrammingError rather than
    IntegrityError.  Startup must handle both."""
    engine, _conn, _sync_conn = _make_pg_engine()

    create_all_called = False

    def fake_create_all(bind, **kw):
        nonlocal create_all_called
        create_all_called = True

    def fake_enum_create(bind, checkfirst=True):
        raise ProgrammingError(
            "CREATE TYPE episode_type",
            {},
            Exception('type "episode_type" already exists'),
        )

    with (
        patch.object(_EPISODE_PG_ENUM, "create", side_effect=fake_enum_create),
        patch.object(
            BaseEpisodeStore.metadata, "create_all", side_effect=fake_create_all
        ),
    ):
        store = SqlAlchemyEpisodeStore(engine)
        await store.startup()

    assert create_all_called, "create_all must still run after ProgrammingError on enum"


@pytest.mark.asyncio
async def test_startup_skips_enum_creation_on_non_postgresql():
    """On non-PostgreSQL dialects (e.g. SQLite), the explicit enum creation
    block must be skipped entirely."""
    engine = MagicMock()

    conn = MagicMock()
    conn.dialect = MagicMock()
    conn.dialect.name = "sqlite"

    sync_conn = MagicMock()
    sync_conn.dialect = conn.dialect

    async def _run_sync(fn):
        return fn(sync_conn)

    conn.run_sync = _run_sync
    engine.begin.return_value = _AsyncCtxMgr(conn)

    with (
        patch.object(_EPISODE_PG_ENUM, "create") as mock_enum_create,
        patch.object(BaseEpisodeStore.metadata, "create_all"),
    ):
        store = SqlAlchemyEpisodeStore(engine)
        await store.startup()

    mock_enum_create.assert_not_called()


@pytest.mark.asyncio
async def test_startup_uses_savepoint_for_enum_creation():
    """The enum creation must be wrapped in begin_nested() (SAVEPOINT) so that
    an IntegrityError only rolls back the savepoint, not the outer transaction."""
    engine, conn, _sync_conn = _make_pg_engine()

    with (
        patch.object(_EPISODE_PG_ENUM, "create"),
        patch.object(BaseEpisodeStore.metadata, "create_all"),
    ):
        store = SqlAlchemyEpisodeStore(engine)
        await store.startup()

    conn.begin_nested.assert_called_once()


# ---------------------------------------------------------------------------
# Integration tests (require Docker / PostgreSQL)
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.asyncio
async def test_duplicate_enum_creation_raises(sqlalchemy_pg_engine):
    """Prove the underlying race: creating the episode_type enum a second time
    (without checkfirst, simulating the TOCTOU window) raises ProgrammingError.

    SQLAlchemy's create_all uses checkfirst=True which masks this in sequential
    calls, but under concurrency two workers can both pass the check and then
    collide on CREATE TYPE.  This test bypasses checkfirst to demonstrate
    the collision directly.
    """
    from sqlalchemy.dialects.postgresql import ENUM as PG_ENUM

    pg_enum = PG_ENUM(
        *(member.name for member in EpisodeType),
        name="episode_type",
    )
    engine = sqlalchemy_pg_engine
    try:
        await _cleanup(engine)

        async with engine.begin() as conn:
            await conn.run_sync(lambda c: pg_enum.create(c, checkfirst=False))

        with pytest.raises((ProgrammingError, IntegrityError), match="episode_type"):
            async with engine.begin() as conn:
                await conn.run_sync(lambda c: pg_enum.create(c, checkfirst=False))
    finally:
        await _cleanup(engine)


@pytest.mark.integration
@pytest.mark.asyncio
async def test_startup_idempotent_on_postgresql(sqlalchemy_pg_engine):
    """The fixed startup() must succeed when called twice against the same
    database — the second call encounters the existing episode_type enum and
    episodestore table and handles both gracefully."""
    engine = sqlalchemy_pg_engine
    store = SqlAlchemyEpisodeStore(engine)

    try:
        await store.startup()
        await store.startup()
    finally:
        await _cleanup(engine)


@pytest.mark.integration
@pytest.mark.asyncio
async def test_startup_functional_after_double_init(sqlalchemy_pg_engine):
    """After two startup() calls the store must be fully functional — able to
    insert and query episodes."""
    engine = sqlalchemy_pg_engine
    store = SqlAlchemyEpisodeStore(engine)

    try:
        await store.startup()
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
    finally:
        await store.delete_all()
        await _cleanup(engine)
