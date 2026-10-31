"""SQL engine setup and write transactions."""

import contextlib
from collections.abc import AsyncGenerator

from sqlalchemy import event
from sqlalchemy.engine import Connection
from sqlalchemy.engine.interfaces import DBAPIConnection
from sqlalchemy.ext.asyncio import (
    AsyncConnection,
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
)
from sqlalchemy.pool import ConnectionPoolEntry, StaticPool

_BEGIN_IMMEDIATE_OPTION = "memmachine_sqlite_begin_immediate"
"""Execution option asking the begin hook for `BEGIN IMMEDIATE`."""


def enable_sqlite_foreign_keys(engine: AsyncEngine) -> None:
    """Enforce foreign keys on every connection of a SQLite engine.

    SQLite defaults foreign_keys to OFF, per connection. Registering the
    pragma at engine creation, before any connection is pooled, is the
    only placement that covers every connection: a listener added later
    misses already-pooled connections, and mutating a live pool's
    listeners races its event dispatch. No-op for other dialects.
    """
    if engine.dialect.name != "sqlite":
        return

    @event.listens_for(engine.sync_engine, "connect")
    def _enable_foreign_keys(
        dbapi_connection: DBAPIConnection,
        _connection_record: ConnectionPoolEntry,
    ) -> None:
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()


def enable_sqlite_explicit_begin(engine: AsyncEngine) -> None:
    """
    Begin every transaction of a SQLite engine explicitly, with `BEGIN IMMEDIATE` for a write transaction.

    The driver emits no `BEGIN` before a SELECT and a deferred one before
    the first data-modifying statement, so a transaction's earlier reads
    and DDL run outside it. A transaction `write_session` or
    `write_connection` opens begins with `BEGIN IMMEDIATE`, which takes the
    write lock; any other begins with a deferred `BEGIN`, so reads never take
    the write lock.

    Registered at engine creation, before any connection is pooled: a
    listener added later misses the connections already pooled. Every
    user of the engine then opens its write transactions with
    `write_session` or `write_connection`. No-op for other dialects.
    """
    if engine.dialect.name != "sqlite":
        return
    event.listen(engine.sync_engine, "connect", _disable_implicit_begin)
    event.listen(engine.sync_engine, "begin", _begin)


def require_separate_connections_to_one_database(engine: AsyncEngine) -> None:
    """
    Raise ValueError unless every session of the engine gets its own connection to one database.

    A pool that hands every session one connection (`StaticPool`) cannot
    keep their transactions apart, and separate connections to in-memory
    SQLite each open their own empty database.
    """
    shares_one_connection = isinstance(engine.pool, StaticPool)
    if shares_one_connection:
        raise ValueError(
            "Engine uses StaticPool, which shares one connection across "
            "sessions. Use a multi-connection pool instead."
        )
    database = engine.url.database
    if engine.dialect.name == "sqlite" and database in (None, "", ":memory:"):
        raise ValueError(
            "Engine uses in-memory SQLite, where each connection gets a "
            "separate database. Use a file path instead."
        )


@contextlib.asynccontextmanager
async def write_session(
    create_session: async_sessionmaker[AsyncSession],
) -> AsyncGenerator[AsyncSession, None]:
    """
    A session whose transaction holds SQLite's write lock from `BEGIN`, committed when the block exits.

    A deferred `BEGIN` takes the lock at the first write, so another writer
    can commit between an earlier read and that write; taking it at `BEGIN`
    makes read-then-write atomic. A block that raises, or a commit that
    fails, rolls the transaction back. Requires an engine set up by
    `enable_sqlite_explicit_begin`. On other dialects, an ordinary
    transaction.
    """
    # The transaction's context also rolls back a commit that fails: SQLite
    # keeps such a transaction open, and closing the session alone would
    # return its connection to the pool with the transaction still open.
    async with create_session() as session, session.begin():
        # Set before the transaction begins, where the begin hook reads it.
        await session.connection(execution_options={_BEGIN_IMMEDIATE_OPTION: True})
        yield session


@contextlib.asynccontextmanager
async def write_connection(
    engine: AsyncEngine,
) -> AsyncGenerator[AsyncConnection, None]:
    """
    A connection whose transaction holds SQLite's write lock from `BEGIN`, committed when the block exits.

    The counterpart of `write_session` for work on a connection: a block
    that raises, or a commit that fails, rolls the transaction back.
    Requires an engine set up by `enable_sqlite_explicit_begin`. On other
    dialects, an ordinary transaction.
    """
    async with engine.connect() as connection:
        # Set before the transaction begins, where the begin hook reads it.
        await connection.execution_options(**{_BEGIN_IMMEDIATE_OPTION: True})
        async with connection.begin():
            yield connection


def _disable_implicit_begin(
    dbapi_connection: DBAPIConnection, _record: ConnectionPoolEntry
) -> None:
    # In autocommit, the driver emits no BEGIN of its own, so the begin hook
    # chooses the mode, per transaction.
    dbapi_connection.isolation_level = None


def _begin(connection: Connection) -> None:
    if connection.get_execution_options().get(_BEGIN_IMMEDIATE_OPTION):
        connection.exec_driver_sql("BEGIN IMMEDIATE")
    else:
        connection.exec_driver_sql("BEGIN")
