"""Tests for the shared SQL engine setup and transactions."""

import contextlib

import pytest
import pytest_asyncio
from sqlalchemy import event, text
from sqlalchemy.exc import OperationalError
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from memmachine_server.common.sql import (
    enable_sqlite_explicit_begin,
    write_connection,
    write_session,
)


@pytest_asyncio.fixture
async def database(tmp_path):
    """A database file with one table, and an engine set up for explicit BEGIN."""
    path = tmp_path / "test.db"
    engine = create_async_engine(f"sqlite+aiosqlite:///{path}")
    enable_sqlite_explicit_begin(engine)
    async with engine.begin() as connection:
        await connection.execute(text("CREATE TABLE t (k INTEGER PRIMARY KEY)"))
    yield path, engine
    await engine.dispose()


@pytest_asyncio.fixture
async def other_writer(database):
    """Another connection's engine, whose blocked writes give up after 0.2 s instead of the default 5 s."""
    path, _ = database
    engine = create_async_engine(
        f"sqlite+aiosqlite:///{path}", connect_args={"timeout": 0.2}
    )
    yield engine
    await engine.dispose()


async def _keys(engine) -> list[int]:
    async with engine.connect() as connection:
        return list(
            (await connection.execute(text("SELECT k FROM t ORDER BY k"))).scalars()
        )


@pytest.mark.asyncio
async def test_a_write_session_can_still_write_after_it_has_read(
    database, other_writer
):
    """
    A write transaction's reads and writes are one atomic unit.

    Under a deferred BEGIN, a transaction that reads first leaves a gap in
    which another writer can take the lock, and the reader then loses: its
    write cannot upgrade, and SQLite reports that at once rather than waiting
    out the busy timeout, since waiting could only deadlock.
    """
    _, engine = database
    async with write_session(async_sessionmaker(engine)) as session:
        await session.execute(text("SELECT count(*) FROM t"))

        # Another writer takes what it can in the gap and holds it across the
        # write below.
        async with other_writer.connect() as blocked:
            with contextlib.suppress(OperationalError):
                await blocked.execute(text("INSERT INTO t VALUES (1)"))
            await session.execute(text("INSERT INTO t VALUES (2)"))
            await blocked.rollback()

    assert await _keys(engine) == [2]


@pytest.mark.asyncio
async def test_a_read_transaction_leaves_the_write_lock_free(database, other_writer):
    _, engine = database
    async with async_sessionmaker(engine)() as session:
        await session.execute(text("SELECT count(*) FROM t"))

        async with other_writer.connect() as writer:
            await writer.execute(text("INSERT INTO t VALUES (1)"))
            await writer.rollback()


@pytest.mark.asyncio
async def test_a_write_session_commits_when_its_block_exits(database):
    _, engine = database
    async with write_session(async_sessionmaker(engine)) as session:
        await session.execute(text("INSERT INTO t VALUES (1)"))

    assert await _keys(engine) == [1]


@pytest.mark.asyncio
async def test_a_write_session_whose_commit_fails_leaves_nothing_open(
    database, other_writer
):
    """
    A write transaction whose commit fails is rolled back, and its connection is reusable.

    SQLite keeps a transaction whose COMMIT failed open, holding the write
    lock, until it is rolled back.
    """
    _, engine = database

    def fail(_connection) -> None:
        raise RuntimeError("injected commit failure")

    event.listen(engine.sync_engine, "commit", fail)
    try:
        with pytest.raises(RuntimeError, match="injected commit failure"):
            async with write_session(async_sessionmaker(engine)) as session:
                await session.execute(text("INSERT INTO t VALUES (1)"))
    finally:
        event.remove(engine.sync_engine, "commit", fail)

    async with other_writer.connect() as connection:
        await connection.execute(text("INSERT INTO t VALUES (2)"))
        await connection.commit()
    async with write_session(async_sessionmaker(engine)) as session:
        await session.execute(text("INSERT INTO t VALUES (3)"))

    assert await _keys(engine) == [2, 3]


@pytest.mark.asyncio
async def test_a_write_session_that_raises_rolls_back_its_ddl_too(database):
    _, engine = database

    async def write_then_fail() -> None:
        async with write_session(async_sessionmaker(engine)) as session:
            await session.execute(text("CREATE TABLE u (k INTEGER)"))
            await session.execute(text("INSERT INTO t VALUES (1)"))
            raise RuntimeError

    with pytest.raises(RuntimeError):
        await write_then_fail()

    async with engine.connect() as connection:
        tables = (
            await connection.execute(
                text("SELECT name FROM sqlite_master WHERE type = 'table'")
            )
        ).scalars()
        assert list(tables) == ["t"]
    assert await _keys(engine) == []


@pytest.mark.asyncio
async def test_a_write_connection_can_still_write_after_it_has_read(
    database, other_writer
):
    """A write connection's reads and writes are one atomic unit, as a write transaction's are."""
    _, engine = database
    async with write_connection(engine) as connection:
        await connection.execute(text("SELECT count(*) FROM t"))

        # Another writer takes what it can in the gap and holds it across the
        # write below.
        async with other_writer.connect() as blocked:
            with contextlib.suppress(OperationalError):
                await blocked.execute(text("INSERT INTO t VALUES (1)"))
            await connection.execute(text("INSERT INTO t VALUES (2)"))
            await blocked.rollback()

    assert await _keys(engine) == [2]


@pytest.mark.asyncio
async def test_a_write_connection_commits_when_its_block_exits(database):
    _, engine = database
    async with write_connection(engine) as connection:
        await connection.execute(text("INSERT INTO t VALUES (1)"))

    assert await _keys(engine) == [1]


@pytest.mark.asyncio
async def test_a_write_connection_whose_commit_fails_leaves_nothing_open(
    database, other_writer
):
    _, engine = database

    def fail(_connection) -> None:
        raise RuntimeError("injected commit failure")

    event.listen(engine.sync_engine, "commit", fail)
    try:
        with pytest.raises(RuntimeError, match="injected commit failure"):
            async with write_connection(engine) as connection:
                await connection.execute(text("INSERT INTO t VALUES (1)"))
    finally:
        event.remove(engine.sync_engine, "commit", fail)

    async with other_writer.connect() as connection:
        await connection.execute(text("INSERT INTO t VALUES (2)"))
        await connection.commit()
    async with write_connection(engine) as connection:
        await connection.execute(text("INSERT INTO t VALUES (3)"))

    assert await _keys(engine) == [2, 3]


@pytest.mark.asyncio
async def test_a_write_connection_that_raises_rolls_back_its_ddl_too(database):
    _, engine = database

    async def write_then_fail() -> None:
        async with write_connection(engine) as connection:
            await connection.execute(text("CREATE TABLE u (k INTEGER)"))
            await connection.execute(text("INSERT INTO t VALUES (1)"))
            raise RuntimeError

    with pytest.raises(RuntimeError):
        await write_then_fail()

    async with engine.connect() as connection:
        tables = (
            await connection.execute(
                text("SELECT name FROM sqlite_master WHERE type = 'table'")
            )
        ).scalars()
        assert list(tables) == ["t"]
    assert await _keys(engine) == []
